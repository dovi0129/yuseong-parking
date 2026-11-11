"""Deterministic reception support; no LLM, no external messages or adjudication."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .domain import ROOT, ROUTES, Plan, PlanConfig, day_type, load_planning_data
from .network import build_matrix
from .planner import solve

PLAN_COLUMNS = ["단속 차량", "순번", "지점 ID", "위치", "위도", "경도", "도착 예정", "기대 신고 수", "위험 가중치"]
RECEPTION_COLUMNS = ["접수 일시", "위치", "지점 ID", "지점 등급", "금일 순번", "묶음 번호", "조치 구분", "시설 개선 검토", "도착 예정", "회신 내용"]
SPOT_COLUMNS = ["지점 ID", "위치", "동", "위도", "경도", "학습 기간 신고 수", "단속 수", "보호구역 지정시설 300m 이내", "위험 가중치", "지점 등급"]
PROCESSING_RULES_VERSION="monthly-patrol-class-daily-serial-v3"
# '조치 구분' values (the screen shows the column as '순회 구분'). A report is classified by its spot, not by its time:
# the patrol is a monthly route, not a dispatch to each report.
IN_ROUTE, NEXT_MONTH, ORDINARY = "이달 순회 지점", "다음 달 순회 검토", "일반"


def grade(count: float, repeat_threshold=10, habitual_threshold=20):
    """Input train report count and thresholds; output 일반/반복/상습 string."""
    if not 1 <= repeat_threshold < habitual_threshold:
        raise ValueError("등급 기준은 1 ≤ 반복 < 상습이어야 합니다.")
    return "상습" if count >= habitual_threshold else "반복" if count >= repeat_threshold else "일반"


def load_catalog(directory=None, *, mock=False):
    """Read spot_id,label,dong,lat,lon,n_reports_train,n_enf_2023,child_zone_300m,risk_weight; return those columns."""
    directory = Path(directory or ROOT / "data/processed")
    prefix = "mock_" if mock else ""
    # Held-out counts are never loaded for grading/processing.
    columns = ["spot_id", "label", "dong", "lat", "lon", "n_reports_train", "n_enf_2023", "child_zone_300m", "risk_weight"]
    frame = pd.read_csv(directory / f"{prefix}spots.csv", usecols=columns, dtype={"spot_id":str})
    if frame.spot_id.duplicated().any():
        raise ValueError("지점 ID가 중복되었습니다.")
    return frame


def plan_table(plan: Plan, planning_date):
    """Input Plan/date; return PLAN_COLUMNS; 단속 차량=1호차/2호차/3호차, 순번 within vehicle."""
    day = pd.Timestamp(planning_date).normalize()
    rows = []
    sequence = {}
    for visit in plan.visits:
        sequence[visit.vehicle] = sequence.get(visit.vehicle, 0)+1
        rows.append({"단속 차량":f"{visit.vehicle+1}호차", "순번":sequence[visit.vehicle], "지점 ID":visit.spot_id,
            "위치":visit.label, "위도":visit.lat, "경도":visit.lon,
            "도착 예정":(day+pd.Timedelta(seconds=visit.arrival_s)).strftime("%Y-%m-%d %H:%M:%S"),
            "기대 신고 수":visit.score, "위험 가중치":visit.risk_weight})
    return pd.DataFrame(rows,columns=PLAN_COLUMNS)


def spot_table(catalog: pd.DataFrame, repeat_threshold=10, habitual_threshold=20):
    """Input load_catalog columns/thresholds; return SPOT_COLUMNS for train counts >= repeat threshold."""
    grade(0,repeat_threshold,habitual_threshold)
    out = catalog.rename(columns={"spot_id":"지점 ID", "label":"위치", "dong":"동", "lat":"위도", "lon":"경도",
        "n_reports_train":"학습 기간 신고 수", "n_enf_2023":"단속 수", "child_zone_300m":"보호구역 지정시설 300m 이내", "risk_weight":"위험 가중치"}).copy()
    out["지점 등급"] = out["학습 기간 신고 수"].map(lambda n:grade(n,repeat_threshold,habitual_threshold))
    return out.loc[out["학습 기간 신고 수"]>=repeat_threshold,SPOT_COLUMNS].sort_values(["학습 기간 신고 수","지점 ID"],ascending=[False,True]).reset_index(drop=True)


def reception_table(reports: pd.DataFrame, catalog: pd.DataFrame, plan: Plan,
                    selected_date, *, through=None, limit=None, repeat_threshold=10, habitual_threshold=20,
                    surge_spots=None):
    """Input report_id,ts,spot_id,raw_loc plus catalog/Plan; return RECEPTION_COLUMNS.

    through/limit restrict the prefix before today's order/grouping. 조치 구분 (shown as 순회 구분):
    IN_ROUTE when the spot is on the plan's route (the month's plan for the date's day type), otherwise NEXT_MONTH
    when the spot is 반복·상습 by train counts or in surge_spots (급증 spots of the month before), otherwise ORDINARY.
    도착 예정 is the planned arrival at a route spot, for information only. 시설 개선 검토 uses train counts and
    the child_zone_300m proxy.
    """
    grade(0,repeat_threshold,habitual_threshold)
    selected_date = pd.Timestamp(selected_date).date()
    frame = reports.copy()
    frame["ts"] = pd.to_datetime(frame.ts)
    frame = frame.loc[frame.ts.dt.date.eq(selected_date)].sort_values(["ts","report_id"],kind="stable")
    if through is not None:
        frame = frame.loc[frame.ts <= pd.Timestamp(through)]
    if limit is not None:
        frame = frame.head(limit)
    # Only the displayed prefix is used for within-day order and grouping.
    frame = frame.copy()
    frame["spot_id"] = frame.spot_id.astype(str)
    frame["today_order"] = frame.groupby("spot_id",sort=False).cumcount()+1
    frame["group_number"] = frame.groupby("spot_id",sort=False).ngroup()+1
    indexed = catalog.set_index("spot_id")
    scheduled = {v.spot_id:v for v in plan.visits}
    surging = {str(s) for s in (surge_spots or ())}
    midnight = pd.Timestamp(selected_date)
    rows = []
    for r in frame.itertuples():
        sid = str(r.spot_id)
        known = sid in indexed.index
        count = float(indexed.loc[sid,"n_reports_train"]) if known else 0.
        category = grade(count,repeat_threshold,habitual_threshold)
        # child_zone_300m is empty only for spots without coordinates (pipeline/features.py); nearness to a
        # child zone cannot be judged there, so such a spot is not treated as near one (same rule as spot_table).
        zone = indexed.loc[sid,"child_zone_300m"] if known else float("nan")
        protected = bool(pd.notna(zone)) and int(zone)==1
        visit = scheduled.get(sid)
        arrival = midnight+pd.Timedelta(seconds=visit.arrival_s) if visit else None
        if visit is not None:
            action = IN_ROUTE
        elif category in ("반복","상습") or sid in surging:
            action = NEXT_MONTH
        else:
            action = ORDINARY
        facility = "해당" if category=="상습" or (category=="반복" and protected) else "비해당"
        when = arrival.strftime("%Y-%m-%d %H:%M:%S") if arrival is not None else ""
        reply = f"접수하신 {r.raw_loc} 불법주정차 신고는 오늘 이 지점에서 {r.today_order}번째로 접수되었습니다."
        if category in ("반복","상습"):
            reply += " 이 지점은 신고가 반복되는 지점으로 관리하고 있으며, 이동형 단속 차량 순회에 포함하여 단속을 강화하겠습니다."
        rows.append({"접수 일시":r.ts.strftime("%Y-%m-%d %H:%M:%S"), "위치":str(r.raw_loc), "지점 ID":sid,
            "지점 등급":category, "금일 순번":int(r.today_order),
            "묶음 번호":f"{selected_date:%Y%m%d}-{r.group_number:03d}", "조치 구분":action, "시설 개선 검토":facility,
            "도착 예정":when, "회신 내용":reply})
    return pd.DataFrame(rows,columns=RECEPTION_COLUMNS)


def csv_bytes(frame: pd.DataFrame):
    """Input Korean-column DataFrame; output utf-8-sig CSV bytes with identical columns, no index."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def save_csv(frame: pd.DataFrame, path: Path):
    """Input DataFrame/path; save identical columns, read back; return path/rows/columns metadata."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(csv_bytes(frame))
    checked = pd.read_csv(path,encoding="utf-8-sig")
    if len(checked)!=len(frame) or checked.columns.tolist()!=frame.columns.tolist():
        raise AssertionError("내보낸 CSV의 행 수 또는 열 이름이 다릅니다.")
    return {"path":str(path),"rows":len(checked),"columns":checked.columns.tolist()}


def main():
    import json
    parser=argparse.ArgumentParser()
    parser.add_argument("--date",required=True)
    parser.add_argument("--start",default="17:00")
    parser.add_argument("--end",default="21:00")
    parser.add_argument("--vehicles",type=int,default=1)
    parser.add_argument("--service-min",type=float,default=5.)
    parser.add_argument("--speed-kmh",type=float,default=25.)
    parser.add_argument("--repeat-threshold",type=int,default=10)
    parser.add_argument("--habitual-threshold",type=int,default=20)
    parser.add_argument("--mock",action="store_true")
    parser.add_argument("--method",choices=("optimized","b3","b1","b2"),default=None)
    args=parser.parse_args()
    def minute(text):
        h,m=map(int,text.split(":"));return h*60+m
    directory=ROUTES if args.mock else ROOT/"data/processed"
    from .api import load_context,generate_plan
    context=load_context(directory=directory,mock=args.mock,speed_kmh=args.speed_kmh)
    catalog=load_catalog(directory,mock=args.mock)
    visits=generate_plan(args.date,args.start,args.end,vehicles=args.vehicles,service_minutes=args.service_min,
        method=args.method,context=context)
    plan=Plan.from_dict(visits.attrs["plan"])
    if plan.status!="ok":
        raise RuntimeError("경로 계산 실패로 내보내기를 중단합니다.")
    reports=pd.read_parquet(directory/("mock_reports.parquet" if args.mock else "reports.parquet"))
    tag=f"{'mock_' if args.mock else ''}{args.date}_{args.start.replace(':','')}-{args.end.replace(':','')}_K{args.vehicles}"
    outputs={"순회계획":plan_table(plan,args.date),
        "접수처리":reception_table(reports,catalog,plan,args.date,repeat_threshold=args.repeat_threshold,habitual_threshold=args.habitual_threshold),
        "반복지점":spot_table(catalog,args.repeat_threshold,args.habitual_threshold)}
    checked=[save_csv(frame,ROOT/"artifacts/exports"/f"{tag}_{name}.csv") for name,frame in outputs.items()]
    print(json.dumps(checked,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
