"""UI-independent public API. Import this module from ui/; no Streamlit calls."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import time
from pathlib import Path
import hashlib
import json
import os
import fcntl

import numpy as np
import pandas as pd

from .domain import ROOT, ROUTES, Plan, PlanConfig, PlanningData, Visit, day_type, load_planning_data
from .network import TravelNetwork, build_matrix
from .planner import solve
from .processing import (PLAN_COLUMNS, RECEPTION_COLUMNS, SPOT_COLUMNS, load_catalog,
    plan_table, reception_table, spot_table, csv_bytes, save_csv)


@dataclass
class Context:
    """Planning inputs and road cache; reports/test records are not loaded here."""
    data: PlanningData
    network: TravelNetwork | None
    catalog: pd.DataFrame
    directory: Path
    mock: bool = False
    speed_kmh: float = 25.
    cache_token: str = ""


PLANNER_CACHE_VERSION="b3-warmstart-weighted-30s-v1"
OVERVIEW_GRADE_VERSION="period-grade-default-v2"


def default_planner_method():
    """Return frozen inner-validation planner choice (b3/optimized), no test data."""
    path=ROUTES/"planner_choice.json"
    if not path.exists():return "optimized"
    chosen=json.loads(path.read_text())["chosen_method"]
    if chosen not in ("b3","optimized"):raise ValueError("기본 순회 방식 선택값이 잘못되었습니다.")
    return chosen


def _context_token(context):
    if not getattr(context,"cache_token",""):
        points=context.data.spots[["spot_id","dong","lat","lon","n_reports_train","risk_weight"]].copy()
        for col in ("lat","lon","n_reports_train","risk_weight"):points[col]=points[col].astype(float)
        priority=context.data.priority[["spot_id","day_type","hour_block","score","method"]].sort_values(["spot_id","day_type","hour_block","method"])
        digest=hashlib.sha256()
        digest.update(points.to_csv(index=False,float_format="%.12g").encode())
        digest.update(priority.to_csv(index=False,float_format="%.12g").encode())
        digest.update(str(context.mock).encode())
        graph=ROUTES/"yuseong_drive.graphml"
        if not context.mock and graph.exists():
            digest.update(f"{graph.stat().st_size}:{graph.stat().st_mtime_ns}".encode())
        context.cache_token=digest.hexdigest()
    return context.cache_token


def _plan_cache_path(context,config,method,score_method):
    from dataclasses import asdict
    obj={"model":PLANNER_CACHE_VERSION,"inputs":_context_token(context),"config":asdict(config),
         "method":method,"score_method":score_method,"speed_kmh":context.speed_kmh}
    key=hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
    folder=ROUTES/"plan_cache"/PLANNER_CACHE_VERSION
    folder.mkdir(parents=True,exist_ok=True)
    return folder/f"{key}.json",key


def _store_plan(path,key,plan):
    temporary=path.with_suffix(f".{os.getpid()}.writing")
    temporary.write_text(json.dumps({"key":key,"plan":plan.to_dict()},ensure_ascii=False))
    temporary.replace(path)


def cache_existing_plan(plan, *, context):
    """Persist a previously validated Plan under the same date-independent key.

    Input Plan has trained score_method/config and relative arrival seconds;
    output is a JSON cache path. Used to seed four official evaluation plans
    without solving them again. No report counts enter the key or plan.
    """
    path,key=_plan_cache_path(context,plan.config,plan.method,plan.score_method)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _store_plan(path,key,plan)
    return path


def plan_for_day_type(day_type_value,start="17:00",end="21:00",vehicles=1,risk_weight=True,
                      *,service_minutes=5.,method=None,score_method=None,solver_seconds=30.,context=None):
    """Return a cached Plan for weekday class, shift, vehicles and risk settings.

    Input day_type_value is 평일/주말. Key additionally includes service time,
    speed, score_method, routing method, solve budget, model/input versions.
    Output Plan has arrival seconds without a date; generate_plan stamps dates
    afterwards. Persisted JSON files are under routes/plan_cache/. A file lock
    prevents duplicate solves for the same combination across processes.
    """
    context=context or load_context()
    method=default_planner_method() if method in (None,"default") else method
    chosen=score_method or context.data.chosen_method
    data=context.data if chosen==context.data.chosen_method else replace(context.data,chosen_method=chosen)
    if chosen not in set(data.priority.method):raise ValueError("우선순위 방식이 없습니다.")
    config=PlanConfig(_minute(start),_minute(end),int(vehicles),float(service_minutes),day_type_value,bool(risk_weight),solver_seconds)
    path,key=_plan_cache_path(context,config,method,chosen)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            cached=json.loads(path.read_text())
            if cached.get("key")!=key:raise ValueError("계획 캐시 키가 맞지 않습니다.")
            return Plan.from_dict(cached["plan"])
        plan=solve(data,_network(context),config,method)
        if plan.status=="ok":_store_plan(path,key,plan)
        return plan


def load_context(*, mock=False, max_spots=None, speed_kmh=25., directory=None, with_network=False):
    """Return Context; the caller may cache this by input-file modification times.

    Input files: spots.csv, priority_train.csv, priority_choice.json (mock_ prefix
    only when mock=True). Coordinates must be WGS84; candidate selection uses
    n_reports_train>0 and coordinates only. No n_reports_test column is loaded
    into the planning table. Output is a Context, not a DataFrame.
    """
    directory=Path(directory or (ROUTES if mock else ROOT/"data/processed"))
    data=load_planning_data(directory,mock=mock,max_spots=max_spots)
    return Context(data,build_matrix(data,speed_kmh,mock=mock) if with_network else None,
                   load_catalog(directory,mock=mock),directory,mock,speed_kmh)


def _network(context):
    if context.network is None:
        context.network=build_matrix(context.data,context.speed_kmh,mock=context.mock)
    return context.network


def _minute(value):
    if isinstance(value,time):return value.hour*60+value.minute
    if isinstance(value,str):
        h,m=map(int,value.split(":"))
        if not 0<=m<60:raise ValueError("분은 0~59여야 합니다.")
        return h*60+m
    return int(value)


def generate_plan(planning_date, start="17:00", end="21:00", staff=1, risk_weight=True,
                  *, service_minutes=5., method=None, score_method=None,
                  solver_seconds=30., context: Context | None=None, vehicles=None):
    """Return inspection visit DataFrame for date/shift/staff/risk settings.

    Inputs: planning_date (2024 ISO date/date), start/end (HH:MM, datetime.time,
    or minutes since midnight), vehicles (1..3; staff is retained as an alias), risk_weight (bool), service_minutes,
    method (None uses planner_choice.json; explicit optimized/b1/b2/b3), optional
    score_method (default frozen priority chosen_method). Standard combinations
    are read from a date-independent file cache; only date stamps are rebuilt.
    Context supplies spots/priority matrices; no report records enter planning.
    Output columns, in order: 단속 차량 (1호차/2호차/3호차), 순번, 지점 ID,
    위치, 위도, 경도, 도착 예정 (YYYY-MM-DD HH:MM:SS), 기대 신고 수, 위험 가중치.
    attrs['plan'] holds a JSON-compatible frozen Plan; attrs['summary'] contains
    status, method, score_method, expected_reports, expected_weighted, travel_min,
    elapsed_s, source, candidate_spots and excluded_spots. An empty frame retains
    every output column. status=no_solution is not silently replaced by b1.
    """
    context=context or load_context()
    staff=staff if vehicles is None else vehicles
    data=context.data if score_method is None else replace(context.data,chosen_method=score_method)
    plan=plan_for_day_type(day_type(planning_date),start,end,staff,risk_weight,service_minutes=service_minutes,
        method=method,score_method=data.chosen_method,solver_seconds=solver_seconds,context=context)
    result=plan_table(plan,planning_date)
    result.attrs["plan"]=plan.to_dict()
    result.attrs["date"]=pd.Timestamp(planning_date).date().isoformat()
    result.attrs["summary"]={"status":plan.status,"method":plan.method,"score_method":plan.score_method,
        "expected_reports":plan.expected_reports,"expected_weighted":plan.expected_weighted,"travel_min":plan.travel_s/60,
        "elapsed_s":plan.elapsed_s,"source":data.source,"candidate_spots":len(data.spots),"excluded_spots":data.excluded_spots}
    result.attrs["summary"].update({"seed_expected_weighted":plan.seed_expected_weighted,
        "seed_expected_reports":plan.seed_expected_reports,"initial_assignment_used":plan.initial_assignment_used,
        "incumbent_source":plan.incumbent_source})
    return result


def _plan_from_table(frame, selected_date, start, end, staff):
    if frame.attrs.get("date") not in (None,pd.Timestamp(selected_date).date().isoformat()):
        raise ValueError("점검 계획 날짜와 접수 날짜가 다릅니다.")
    if "plan" in frame.attrs:return Plan.from_dict(frame.attrs["plan"])
    if set(PLAN_COLUMNS)-set(frame):raise ValueError("점검 계획의 필수 열이 없습니다.")
    config=PlanConfig(_minute(start),_minute(end),int(staff),day_type=day_type(selected_date))
    visits=[]
    midnight=pd.Timestamp(selected_date).normalize()
    for row in frame.to_dict("records"):
        stamp=pd.Timestamp(row["도착 예정"])
        if stamp.normalize()!=midnight:raise ValueError("점검 계획 날짜와 접수 날짜가 다릅니다.")
        visits.append(Visit(str(row["지점 ID"]),int((stamp-midnight).total_seconds()),int(str(row["단속 차량"]).removesuffix("호차"))-1,
            float(row["기대 신고 수"]),float(row["위험 가중치"]),str(row["위치"]),float(row["위도"]),float(row["경도"])))
    return Plan("imported_table",config,visits)


def classify_reports(selected_date, repeat_threshold=10, habitual_threshold=20, *,
                     plan_df=None, through=None, limit=None, start="17:00", end="21:00", staff=1,
                     context: Context | None=None, reports: pd.DataFrame | None=None, surge_spots=None):
    """Return reception processing DataFrame; grades depend only on train counts.

    Inputs: ISO date/date; 1 <= repeat_threshold < habitual_threshold; optional
    plan_df from generate_plan/CSV with PLAN_COLUMNS; through timestamp and/or
    limit for the visible prefix. Optional reports columns: report_id, ts,
    spot_id, raw_loc. With no reports argument, read test split from the contract
    parquet. No records after through/limit affect today's counts or grouping.
    Output columns: 접수 일시, 위치, 지점 ID, 지점 등급, 금일 순번, 묶음 번호,
    조치 구분, 시설 개선 검토, 도착 예정, 회신 내용. 조치 구분 is 이달 순회 지점 (spot on the plan's route),
    다음 달 순회 검토 (반복·상습, or in surge_spots) or 일반; it does not compare report and arrival times.
    The last column is an unsent deterministic reply draft, not a completed administrative action.
    """
    context=context or load_context()
    if plan_df is None:plan_df=generate_plan(selected_date,start,end,staff,context=context)
    plan=_plan_from_table(plan_df,selected_date,start,end,staff)
    if reports is None:
        reports=pd.read_parquet(context.directory/("mock_reports.parquet" if context.mock else "reports.parquet"))
        reports=reports.loc[reports.split.eq("test")]
    required={"report_id","ts","spot_id","raw_loc"}
    if missing:=required-set(reports):raise ValueError(f"접수 입력 열 누락: {sorted(missing)}")
    return reception_table(reports,context.catalog,plan,selected_date,through=through,limit=limit,
        repeat_threshold=repeat_threshold,habitual_threshold=habitual_threshold,surge_spots=surge_spots)


def repeat_spots(repeat_threshold=10, habitual_threshold=20, *, context: Context | None=None):
    """Return all catalog spots meeting repeat_threshold, including missing coords.

    Inputs: repeat_threshold>=1, habitual_threshold>repeat_threshold and Context
    catalog columns spot_id,label,dong,lat,lon,n_reports_train,n_enf_2023,
    child_zone_300m,risk_weight. Output columns: 지점 ID, 위치, 동, 위도, 경도,
    학습 기간 신고 수, 단속 수, 보호구역 지정시설 300m 이내, 위험 가중치, 지점 등급.
    child_zone_300m is a facility-coordinate buffer proxy, not a legal polygon.
    """
    context=context or load_context()
    return spot_table(context.catalog,repeat_threshold,habitual_threshold)


def export_csv(frame: pd.DataFrame, path: str | Path | None=None):
    """Preserve Korean columns and row order; return BOM bytes or saved-file info.

    Input: DataFrame from generate_plan/classify_reports/repeat_spots. Output
    CSV has identical columns (PLAN_COLUMNS/RECEPTION_COLUMNS/SPOT_COLUMNS),
    utf-8-sig encoding, no index. If path=None return bytes for download_button;
    otherwise save and read back to verify row count/headers, return a dict with
    path, rows, columns. Default CLI paths live under artifacts/exports/.
    """
    return csv_bytes(frame) if path is None else save_csv(frame,Path(path))


def route_polylines(plan_df, *, context: Context | None=None):
    """Input PLAN_COLUMNS or generated plan; output columns 단속 차량, 경로 좌표.

    경로 좌표 is a list of (lat,lon) pairs following directed OSM shortest paths,
    including depot departure/return. The caller may draw it directly in Folium.
    Empty inspection tours retain columns and return no line for that vehicle.
    """
    context=context or load_context()
    rows=[]
    network=_network(context)
    lookup={sid:i for i,sid in enumerate(network.ids)}
    for vehicle,frame in plan_df.groupby("단속 차량",sort=True):
        seq=[0]+[lookup[str(s)] for s in frame.sort_values("순번")["지점 ID"]]+[0]
        points=[]
        for a,b in zip(seq,seq[1:]):points.extend(network.path(a,b))
        rows.append({"단속 차량":str(vehicle),"경로 좌표":points})
    return pd.DataFrame(rows,columns=["단속 차량","경로 좌표"])


def overview(*, context: Context | None=None, reports: pd.DataFrame | None=None):
    """Return dict of summary/map/heatmap/concentration DataFrames, train only.

    Input reports: split, ts, spot_id (optional, otherwise read contract parquet).
    Outputs: summary columns 학습 신고 (건), 당일 반복 신고 (건), 경로 후보 (곳);
    spots = Context.data.spots (no n_reports_test); heatmap index 요일 0..6,
    columns hour 0..23, values counts; concentration columns 지점 비율 (%),
    누적 신고 비율 (%). All calculations are independent of Streamlit.
    """
    context=context or load_context()
    if reports is None:reports=pd.read_parquet(context.directory/("mock_reports.parquet" if context.mock else "reports.parquet"))
    train=reports.loc[reports.split.eq("train")].copy()
    train["ts"]=pd.to_datetime(train.ts)
    counts=train.groupby("spot_id").size().sort_values(ascending=False)
    repeated=train.groupby([train.ts.dt.date,"spot_id"]).size()
    heat=train.groupby([train.ts.dt.day_of_week,train.ts.dt.hour]).size().unstack(fill_value=0).reindex(index=range(7),columns=range(24),fill_value=0)
    cdf=pd.DataFrame({"지점 비율 (%)":np.arange(1,len(counts)+1)/max(1,len(counts))*100,
                      "누적 신고 비율 (%)":counts.cumsum().to_numpy()/max(1,counts.sum())*100})
    return {"summary":pd.DataFrame([{"학습 신고 (건)":len(train),"당일 반복 신고 (건)":int(repeated[repeated>=2].sum()),"경로 후보 (곳)":len(context.data.spots)}]),
            "spots":context.data.spots.copy(),"heatmap":heat,"concentration":cdf}


def reception_groups(received: pd.DataFrame):
    """Input RECEPTION_COLUMNS; output 묶음 번호, 지점 ID, 위치, 신고 수, 최초 접수, 최종 접수.

    Only rows in received enter this aggregation, so a replay prefix cannot see
    future reports. One report remains a distinct fine record even when grouped.
    """
    columns=["묶음 번호","지점 ID","위치","신고 수","최초 접수","최종 접수"]
    if received.empty:return pd.DataFrame(columns=columns)
    return (received.groupby(["묶음 번호","지점 ID","위치"],sort=False)
        .agg(**{"신고 수":("금일 순번","size"),"최초 접수":("접수 일시","min"),"최종 접수":("접수 일시","max")})
        .reset_index()[columns])


def overview_filtered(start_date, end_date, dong=None, grades=("상습","반복","일반"),
                      repeat_threshold=10, habitual_threshold=20, grade_basis="period", *,
                      context: Context | None=None, reports: pd.DataFrame | None=None):
    """Return date/dong/grade-filtered descriptive tables for any 2024 dates.

    Input: inclusive start_date/end_date within 2024; dong=None, a string or list;
    grades drawn from 상습/반복/일반; grade_basis='period' uses queried counts,
    'train' uses fixed Jan-Sep n_reports_train. Planning/reception stay train-based.
    Optional reports columns ts,spot_id; otherwise read contract reports.parquet.
    Output dict: summary one row with 신고 건수, 반복 지점 수, 상습 지점 수,
    상위 100개 지점 비중 (%); spots includes every spot with >=1 queried report,
    sorted by 신고 건수 descending, with columns 지점 ID, 위치, 동, 위도, 경도,
    신고 건수, 학습 기간 신고 수, 단속 수, 보호구역 지정시설 300m 이내,
    위험 가중치, 지점 등급; heatmap is day-of-week 0..6 by hour 0..23 counts.
    Summary and heatmap use the same post-filter records. 반복/상습 counts are
    disjoint grade categories. This descriptive function never changes planning
    candidates or operational reception grades and never calls solve.
    """
    from .processing import grade
    grade(0,repeat_threshold,habitual_threshold)
    if grade_basis not in ("period","train"):raise ValueError("grade_basis는 period 또는 train이어야 합니다.")
    start=pd.Timestamp(start_date).normalize();end=pd.Timestamp(end_date).normalize()
    if start.year!=2024 or end.year!=2024 or start>end:
        raise ValueError("조회 기간은 2024년 안에서 시작일 ≤ 종료일이어야 합니다.")
    if not set(grades).issubset({"상습","반복","일반"}):raise ValueError("지점 등급이 잘못되었습니다.")
    context=context or load_context()
    if reports is None:reports=pd.read_parquet(context.directory/("mock_reports.parquet" if context.mock else "reports.parquet"))
    queried=reports.copy();queried["ts"]=pd.to_datetime(queried.ts)
    queried=queried.loc[(queried.ts>=start)&(queried.ts<end+pd.Timedelta(days=1))].copy()
    queried["spot_id"]=queried.spot_id.astype(str)
    counts=queried.groupby("spot_id").size().rename("신고 건수")
    spots=context.catalog.copy()
    spots=spots.join(counts,on="spot_id")
    spots=spots.loc[spots["신고 건수"].fillna(0)>0]
    basis=spots["신고 건수"] if grade_basis=="period" else spots.n_reports_train
    spots["지점 등급"]=basis.map(lambda n:grade(n,repeat_threshold,habitual_threshold))
    if dong is not None:
        allowed=[dong] if isinstance(dong,str) else list(dong)
        spots=spots.loc[spots.dong.isin(allowed)]
    spots=spots.loc[spots["지점 등급"].isin(grades)]
    queried=queried.loc[queried.spot_id.isin(spots.spot_id)]
    spots=spots.rename(columns={"spot_id":"지점 ID","label":"위치","dong":"동","lat":"위도","lon":"경도",
        "n_reports_train":"학습 기간 신고 수","n_enf_2023":"단속 수","child_zone_300m":"보호구역 지정시설 300m 이내","risk_weight":"위험 가중치"})
    columns=["지점 ID","위치","동","위도","경도","신고 건수","학습 기간 신고 수","단속 수","보호구역 지정시설 300m 이내","위험 가중치","지점 등급"]
    spots=spots[columns].sort_values(["신고 건수","지점 ID"],ascending=[False,True]).reset_index(drop=True)
    spots["신고 건수"]=spots["신고 건수"].astype(int)
    top_share=float(spots["신고 건수"].head(100).sum()/max(1,len(queried))*100)
    summary=pd.DataFrame([{"신고 건수":len(queried),"반복 지점 수":int(spots["지점 등급"].eq("반복").sum()),
        "상습 지점 수":int(spots["지점 등급"].eq("상습").sum()),"상위 100개 지점 비중 (%)":top_share}])
    heat=(queried.groupby([queried.ts.dt.day_of_week,queried.ts.dt.hour]).size()
        .unstack(fill_value=0).reindex(index=range(7),columns=range(24),fill_value=0))
    return {"summary":summary,"spots":spots,"heatmap":heat}
