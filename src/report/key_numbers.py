"""Build the proposal's scalar number bank; never refit or rerun evaluations.

Run: .venv/bin/python -m src.report.key_numbers
Output columns: 항목, 값, 단위, 정의, 기간·조건, 근거 파일.
Rates are percentages; rate differences are percentage points. Paired route
differences always mean optimized minus baseline. Saved bootstrap intervals
are reused and checked against the corresponding daily point estimates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.processing import processing_metrics
from src.photo.evaluate import summary as photo_summary

ROOT=Path(__file__).resolve().parents[2]
COLUMNS=["항목","값","단위","정의","기간·조건","근거 파일"]
ROUTE_METRICS={"caught":("포착 신고 수","건/일"),
    "risk_caught":("위험 가중 포착 수","가중 건/일"),
    "travel_min":("이동시간","분/일")}
SCENARIOS=("10-14_K1","10-14_K2","17-21_K1","17-21_K2")
METHODS=("b1","b2","b3","optimized")


def ratio(numerator,denominator):
    """Return a ratio of totals; a zero denominator is undefined, not zero."""
    if denominator<=0:raise ValueError("비율의 분모가 0 이하입니다.")
    return float(numerator/denominator)


def concentration_point(frame: pd.DataFrame):
    """Input one persisted concentration series; return ceil-top10 count/share."""
    n=int(frame.n_spots.iloc[0])
    if len(frame)!=n or frame.n_spots.nunique()!=1:raise ValueError("집중도 지점 수가 맞지 않습니다.")
    ordered=frame.sort_values("spot_fraction")
    n_top=math.ceil(.1*n)
    point=ordered.iloc[n_top-1]
    if not np.isclose(point.spot_fraction,n_top/n):raise ValueError("상위 10% 지점 순위가 맞지 않습니다.")
    return n_top,float(point.cumulative_fraction)


def repetition_counts(frame: pd.DataFrame):
    """Count ALL records in spot-days with >=2 reports, not just excess records."""
    if not np.array_equal(frame.n_reports,frame.reports_per_spot_day*frame.n_spot_days):
        raise ValueError("반복 신고 분포의 건수와 묶음 수가 맞지 않습니다.")
    records=int(frame.n_reports.sum())
    repeated=int(frame.loc[frame.reports_per_spot_day>=2,"n_reports"].sum())
    return records,repeated,ratio(repeated,records)


def map_review_counts(frame: pd.DataFrame):
    """Return sample/completed counts; map_checked is completion, not accuracy."""
    if frame.spot_id.duplicated().any():raise ValueError("지도 확인 표본 지점이 중복되었습니다.")
    return len(frame),int(frame.review_status.eq("map_checked").sum())


def trimmed_location_count(values: pd.Series):
    """Count location strings after edge-whitespace removal; preserve inner spaces."""
    return int(values.astype("string").str.strip().nunique())


def photo_decision_counts(frame: pd.DataFrame):
    """Reuse the photo evaluator's rates and count normal/reject denominators."""
    allowed={"요건 충족","보완 필요"}
    if not frame.expected_verdict.isin(allowed).all() or not frame.verdict.isin(allowed).all():
        raise ValueError("알 수 없는 사진 판정값이 있습니다.")
    normal=frame.expected_verdict.eq("요건 충족")
    positive=frame.verdict.eq("요건 충족")
    return {**photo_summary(frame),"n_normal":int(normal.sum()),"n_reject":int((~normal).sum()),
        "normal_accepted":int((normal&positive).sum()),"reject_detected":int((~normal&~positive).sum()),
        "n_correct":int(frame.verdict.eq(frame.expected_verdict).sum())}


def real_photo_counts(frame: pd.DataFrame):
    """Count manually confirmed real photos; a wrong reading is not unreadable."""
    confirmed=frame.loc[frame.truth_status.eq("수기 확인")].copy()
    if confirmed.plate_true.isna().any():raise ValueError("수기 확인 실차 사진에 정답 번호가 없습니다.")
    predicted=confirmed.plate.fillna("").astype(str).str.strip()
    exact=predicted.eq(confirmed.plate_true.astype(str))
    return len(confirmed),int(exact.sum()),int(predicted.eq("").sum())


def priority_rate(daily: pd.DataFrame):
    """Return ratio of summed captured/total records, not mean daily ratios."""
    return ratio(daily.captured.sum(),daily.total.sum())


def paired_difference(daily: pd.DataFrame,scenario,baseline,metric):
    """Return signed paired daily mean for one scenario/metric; reject missing days."""
    sub=daily.loc[(daily.scenario==scenario)&daily.method.isin(["optimized",baseline])]
    if sub.duplicated(["date","method"]).any():raise ValueError("순회 날짜·방식이 중복되었습니다.")
    pivot=sub.pivot(index="date",columns="method",values=metric).sort_index()
    if set(pivot)!={"optimized",baseline} or pivot.isna().any().any():
        raise ValueError("순회 방식 사이의 평가 날짜가 맞지 않습니다.")
    return float((pivot.optimized-pivot[baseline]).mean()),len(pivot)


class Bank:
    def __init__(self,root):
        self.root=Path(root)
        self.rows=[]
        self._tables={}
        self._objects={}

    def csv(self,name):
        if name not in self._tables:self._tables[name]=pd.read_csv(self.root/name,encoding="utf-8-sig")
        return self._tables[name]

    def obj(self,name):
        if name not in self._objects:self._objects[name]=json.loads((self.root/name).read_text())
        return self._objects[name]

    def add(self,item,value,unit,definition,condition,*sources):
        for source in sources:
            if not (self.root/source).is_file():raise FileNotFoundError(source)
        if isinstance(value,(float,np.floating)) and not np.isfinite(value):
            raise ValueError(f"유효하지 않은 수치: {item}")
        self.rows.append(dict(zip(COLUMNS,[item,value,unit,definition,condition,"; ".join(sources)])))


def date_period(frame):
    return f"{pd.Timestamp(frame.ts.min()).date()}~{pd.Timestamp(frame.ts.max()).date()}"


def add_descriptive(bank,reports,enforcement,spots):
    rp="data/processed/reports.parquet";ep="data/processed/enforcement.parquet"
    sp="data/processed/spots.csv";qp="data/processed/geocode_quality.json"
    train=reports.loc[reports.split.eq("train")];test=reports.loc[reports.split.eq("test")]
    period=date_period(reports);train_period=date_period(train);test_period=date_period(test)
    for label,frame,path,definition in [
        ("신고",reports,rp,"과태료 처리 상태가 기록된 불법주정차 신고"),
        ("단속",enforcement,ep,"불법주정차 단속 기록")]:
        condition=date_period(frame)
        bank.add(f"자료 | {label} 건수",len(frame),"건",definition,condition,path)
        bank.add(f"자료 | {label} 기간 시작",pd.Timestamp(frame.ts.min()).date().isoformat(),"일","기록 일시의 최솟값에 해당하는 날짜",condition,path)
        bank.add(f"자료 | {label} 기간 종료",pd.Timestamp(frame.ts.max()).date().isoformat(),"일","기록 일시의 최댓값에 해당하는 날짜",condition,path)
        bank.add(f"지점 | {label} 원문 위치 종류 수",int(frame.raw_loc.nunique()),"종류","정규화 전 위치 문자열의 고유값 수",condition,path)
    for label,frame in [("학습",train),("평가",test)]:
        bank.add(f"자료 | {label} 신고 건수",len(frame),"건","split으로 구분한 과태료 처리 신고",date_period(frame),rp)
    concentration=bank.csv("data/processed/analysis_tables/concentration.csv")
    annotations=bank.csv("data/processed/analysis_tables/concentration_annotations.csv")
    for series,frame in concentration.groupby("series",sort=False):
        name="신고" if series.startswith("신고") else "단속"
        data=reports if name=="신고" else enforcement
        path=rp if name=="신고" else ep
        if frame.n_records.nunique()!=1 or int(frame.n_records.iloc[0])!=len(data):
            raise ValueError("집중도 표와 처리 기록의 모집단이 다릅니다.")
        top,share=concentration_point(frame)
        saved=annotations.loc[annotations.series.eq(series)].iloc[0]
        if int(saved.n_top10_spots)!=top or not np.isclose(saved.cumulative_fraction,share):
            raise ValueError("집중도 그림 주석과 원표가 다릅니다.")
        sources=("data/processed/analysis_tables/concentration.csv",path)
        bank.add(f"지점 | {name} 정규화 지점 수",int(frame.n_spots.iloc[0]),"곳","고정 위치 정규화 규칙으로 묶은 지점; 단속은 신고와 미대응인 장소도 포함한다.",date_period(data),*sources)
        bank.add(f"집중도 | {name} 상위 10% 지점 수",top,"곳","ceil(정규화 지점 수×0.1); 빈도 내림차순",date_period(data),*sources)
        bank.add(f"집중도 | {name} 상위 10% 지점 건수 비중",share*100,"%","상위 ceil(지점 수×0.1) 지점의 누적 건수 / 전체 기록 수 ×100",date_period(data),*sources)
    quality=bank.obj(qp)
    accepted=spots.lat.notna()&spots.lon.notna()
    if reports.spot_id.nunique()!=len(spots) or int(accepted.sum())!=quality["n_successful_locations"]:
        raise ValueError("좌표 품질 표와 지점 목록이 다릅니다.")
    bank.add("좌표 | 연결된 정규화 지점 수",int(accepted.sum()),"곳","위도·경도가 모두 있는 지점 수",period,sp,qp)
    bank.add("좌표 | 지점 단위 연결률",ratio(int(accepted.sum()),len(spots))*100,"%","연결된 지점 수 / 정규화 지점 수 ×100; 신고 가중률과 구별한다.",period,sp)
    for label,columns,key,cond in [
        ("전체",["n_reports_train","n_reports_test"],"all",period),
        ("학습",["n_reports_train"],"train",train_period),
        ("평가",["n_reports_test"],"test",test_period)]:
        weight=spots[columns].sum(axis=1)
        value=ratio(weight[accepted].sum(),weight.sum())
        if not np.isclose(value,quality[f"report_weighted_success_{key}"],atol=1e-12):
            raise ValueError("신고 가중 좌표 연결률이 품질 파일과 다릅니다.")
        bank.add(f"좌표 | {label} 신고 가중 연결률",value*100,"%","연결된 지점의 신고 수 합 / 해당 기간 전체 신고 수 ×100",cond,sp,qp)
    rep_path="data/processed/analysis_tables/same_day_repetition.csv"
    total,repeated,rate=repetition_counts(bank.csv(rep_path))
    if total!=len(reports):raise ValueError("반복 신고 표의 분모가 전체 신고와 다릅니다.")
    bank.add("반복 | 같은 지점·같은 날 반복 신고 건수",repeated,"건","같은 지점·같은 날 2건 이상인 묶음에 속한 신고를 모두 센다.",period,rep_path,rp)
    bank.add("반복 | 같은 지점·같은 날 반복 신고 비율",rate*100,"%","2건 이상인 지점×날짜 묶음의 전체 신고 수 / 전체 신고 수 ×100; 첫 신고 이후 초과 건수만 세는 비율이 아니다.",period,rep_path,rp)
    process_path="data/processed/processing_metrics.csv"
    saved=bank.csv(process_path)
    computed=processing_metrics(reports)
    for metric in ("train_threshold_coverage","spot_date_groups","spot_date_hour_block_groups"):
        left=saved.loc[saved.metric.eq(metric)].sort_values("threshold").reset_index(drop=True)
        right=computed.loc[computed.metric.eq(metric)].sort_values("threshold").reset_index(drop=True)
        for column in ("n_spots_train","n_reports_test","n_test_total","n_units_after","coverage_rate","reduction_rate"):
            if not np.allclose(left[column].astype(float),right[column].astype(float),equal_nan=True,atol=1e-12):
                raise ValueError(f"기존 처리 지표와 원함수 계산이 다릅니다: {metric}/{column}")
    for row in computed.itertuples():
        if row.metric=="train_threshold_coverage":
            threshold=int(row.threshold);condition=f"기준 {train_period}; 평가 {test_period}; 학습 신고 {threshold}건 이상"
            for suffix,value,unit,definition in [
                ("지점 수",int(row.n_spots_train),"곳",f"학습 기간 신고가 {threshold}건 이상인 고정 지점 수; 누적 기준"),
                ("평가 신고 건수",int(row.n_reports_test),"건","위 지점 집합에서 발생한 평가 신고 수"),
                ("평가 신고 비율",float(row.coverage_rate)*100,"%","위 지점 집합의 평가 신고 수 / 평가 전체 신고 수 ×100")]:
                bank.add(f"운영 기준 | {threshold}건 이상 {suffix}",value,unit,definition,condition,process_path,rp)
        else:
            grouping="지점×날짜" if row.metric=="spot_date_groups" else "지점×날짜×hour_block"
            bank.add(f"묶음 | {grouping} 묶음 수",int(row.n_units_after),"개",row.definition,test_period,process_path,rp)
            bank.add(f"묶음 | {grouping} 감소율",float(row.reduction_rate)*100,"%","(1-묶음 수/신고 건수)×100; 현장 확인·회신 단위의 감소율이며 처리시간 감소율이 아니다.",test_period,process_path,rp)
    frequency=train.groupby("spot_id").size()
    exclusive=set(frequency[(frequency>=10)&(frequency<20)].index)
    bank.add("운영 등급 | 반복(10~19건) 지점 수",len(exclusive),"곳","누적 10건 이상 집합에서 상습(20건 이상) 집합을 제외한 지점 수",train_period,rp)
    bank.add("운영 등급 | 반복(10~19건) 지점의 평가 신고 비율",ratio(int(test.spot_id.isin(exclusive).sum()),len(test))*100,"%","반복 등급 지점의 평가 신고 / 평가 전체 신고 ×100",f"운영 {train_period}; 평가 {test_period}",rp)


def add_priority(bank):
    source="data/processed/evaluation_priority.csv";daily_source="data/processed/evaluation_priority_daily.csv"
    saved=bank.csv(source);daily=bank.csv(daily_source);protocol=bank.obj("data/processed/evaluation_protocol.json")
    if set(saved.K)!={20,50,100} or set(saved.method)!={"freq_baseline","shrinkage"} or len(saved)!=6:
        raise ValueError("우선순위 평가의 K×방식 구성이 다릅니다.")
    for row in saved.sort_values(["K","method"]).itertuples():
        sub=daily.loc[daily.method.eq(row.method)&daily.K.eq(row.K)]
        if not np.isclose(priority_rate(sub),row.capture_rate,atol=1e-12):raise ValueError("우선순위 포착률 정의가 일별 표와 다릅니다.")
        if int(sub.captured.sum())!=row.captured or int(sub.total.sum())!=row.total:
            raise ValueError("우선순위 포착 건수/분모가 다릅니다.")
        cond=f"{protocol['period']}; {int(row.n_calendar_days)}개 달력일; K={int(row.K)}; {row.method}; {int(row.bootstrap_replicates)}회 날짜 부트스트랩"
        prefix=f"우선순위 | K={int(row.K)} | {row.method}"
        bank.add(f"{prefix} | 포착 건수",int(row.captured),"건","지점·day_type·hour_block별 상위K에 해당하는 평가 신고 수",cond,source,daily_source)
        for suffix,column in [("포착률","capture_rate"),("포착률 95% 하한","ci_low"),("포착률 95% 상한","ci_high")]:
            bank.add(f"{prefix} | {suffix}",float(getattr(row,column))*100,"%","포착 건수 합 / 전체 평가 신고 수 합 ×100; 날짜별 비율의 단순평균이 아니다.",cond,source,daily_source)


def route_condition(bank,scenario,days):
    frozen=bank.obj("data/processed/routes/plans_frozen.json")
    config=frozen["plans"][f"{scenario}_평일_optimized"]["config"]
    date_range=bank.obj("data/processed/evaluation_protocol.json")["period"]
    return (f"{date_range}; 평일 {days}일(공휴일 제외); "
        f"{config['start_min']//60:02}:{config['start_min']%60:02}~{config['end_min']//60:02}:{config['end_min']%60:02}; "
        f"차량 {config['vehicles']}대; 지점당 {config['service_min']:g}분; "
        f"{frozen['network']['speed_kmh']:g}km/h; 학습 후보 {len(frozen['candidate_spots'])}곳; {frozen['chosen_method']}")


def add_routes(bank):
    daily_path="data/processed/routes/evaluation_with_b3_daily.csv";daily=bank.csv(daily_path)
    summary_path="data/processed/routes/evaluation_summary.csv";summary=bank.csv(summary_path)
    if set(daily.scenario)!=set(SCENARIOS) or set(daily.method)!=set(METHODS):raise ValueError("최종 순회 평가의 조건/방식 구성이 다릅니다.")
    if daily.duplicated(["scenario","method","date"]).any():raise ValueError("순회 일별 표가 중복되었습니다.")
    for (scenario,method),frame in daily.groupby(["scenario","method"],sort=True):
        cond=route_condition(bank,scenario,frame.date.nunique())
        for metric,(label,unit) in ROUTE_METRICS.items():
            value=float(frame[metric].mean())
            if method!="b3":
                row=summary[(summary.scenario==scenario)&(summary.method==method)&(summary.metric==metric)].iloc[0]
                if not np.isclose(value,row['mean'],atol=1e-12):raise ValueError("최종 순회 평균 표와 일별 표가 다릅니다.")
            bank.add(f"순회 | {scenario} | {method} | 일평균 {label}",value,unit,
                "날짜별 지표의 산술평균; 포착은 방문 지점과 도착 hour_block 일치, 위험 가중은 고정 risk_weight 합, 이동시간은 전 차량의 주행시간 합이다.",
                cond,daily_path)
    for source in ["data/processed/routes/evaluation_paired.csv","data/processed/routes/evaluation_b3_paired.csv"]:
        paired=bank.csv(source)
        for row in paired.loc[paired.metric.isin(ROUTE_METRICS)].itertuples():
            observed,days=paired_difference(daily,row.scenario,row.baseline,row.metric)
            if days!=row.days or not np.isclose(observed,row.mean_delta,atol=1e-12):raise ValueError("짝지은 차이의 날짜/부호/평균이 일별 표와 다릅니다.")
            if not np.isfinite([row.ci_low,row.ci_high]).all() or row.ci_low>row.ci_high:raise ValueError("잘못된 저장 신뢰구간입니다.")
            label,unit=ROUTE_METRICS[row.metric];cond=route_condition(bank,row.scenario,days)
            notes=bank.obj("data/processed/routes/figure_notes.json")["ci"]
            cond+=f"; {notes}"
            prefix=f"순회 차이 | {row.scenario} | optimized-{row.baseline} | {label}"
            for suffix,value in [("평균",observed),("95% 하한",float(row.ci_low)),("95% 상한",float(row.ci_high))]:
                bank.add(f"{prefix} | {suffix}",value,unit,"같은 날짜의 optimized 지표 - 기준선 지표; 음수도 그대로 보존한다.",cond,source,daily_path)


def add_choice_and_emerging(bank,reports):
    source="data/processed/routes/planner_choice.json";choice=bank.obj(source)
    summary_source="data/processed/routes/planner_inner_summary.csv";summary=bank.csv(summary_source)
    if choice["method_selection_uses_test"] is not False:raise ValueError("기본 방식 선택에 테스트를 사용했습니다.")
    cond=f"적합 {choice['rule']['fit']}; 검증 {choice['rule']['validation']}; 평일 {choice['validation_days']}일; alpha={choice['alpha']:g}; 후보 {choice['inner_candidate_count']}곳"
    for method in ("b3","optimized"):
        rows=summary[summary.method.eq(method)]
        value=float(rows.mean_caught.sum())
        if not np.isclose(value,choice['sums'][method],atol=1e-12):raise ValueError("기본 방식 선택 합계와 검증 표가 다릅니다.")
        bank.add(f"기본 방식 선택 | {method} 4조건 일평균 포착 합",value,"건/일·4조건 합","미리 정한 네 조건의 검증 일평균 caught를 합한다.",cond,source,summary_source)
        for row in rows.itertuples():
            bank.add(f"기본 방식 선택 | {row.scenario} | {method} 일평균 포착",float(row.mean_caught),"건/일","1~7월 점수로 고정한 계획의 8~9월 평일 일평균 caught",cond,summary_source)
    bank.add("기본 방식 선택 | optimized-b3 4조건 합 차이",float(choice['difference_optimized_minus_b3']),"건/일·4조건 합","optimized의 네 조건 평균 합 - b3의 네 조건 평균 합",cond,source)
    bank.add("기본 방식 선택 | 단순 방식 우선 차이 기준",float(choice['threshold']),"건/일·4조건 합","차이가 기준보다 작으면 b3, optimized가 기준 이상 앞서면 optimized를 선택한다.",cond,source)
    bank.add("기본 방식 선택 | 채택 방식",choice['chosen_method'],"방식",choice['reason'],cond,source)
    bank.add("기본 방식 선택 | 검증 평일 수",int(choice['validation_days']),"일","8~9월 검증 달력의 평일 수; 공휴일은 제외한다.",cond,source)
    path="data/processed/emerging_spots.csv";emerging=bank.csv(path)
    counts=reports.groupby("spot_id").size();train=reports.loc[reports.split.eq("train")].groupby("spot_id").size()
    expected=set(counts[(counts>=20)&(train.reindex(counts.index,fill_value=0)<10)].index)
    if set(emerging["지점 ID"])!=expected or emerging["지점 ID"].duplicated().any():raise ValueError("신규 상습 지점 목록과 정의가 다릅니다.")
    bank.add("신규 상습 | 연간 상습·운영 일반 지점 수",len(emerging),"곳","연간 신고 20건 이상이면서 1~9월 신고 10건 미만인 지점 수",f"연간 {date_period(reports)}; 운영 {date_period(reports.loc[reports.split.eq('train')])}",path,"data/processed/reports.parquet")


def add_reference_and_review(bank,reports,enforcement):
    school_path="data/processed/reference/schools.csv"
    school_meta_path="data/processed/reference/schools_source.json"
    schools=bank.csv(school_path);school_meta=bank.obj(school_meta_path)
    if schools["학교ID"].duplicated().any() or len(schools)!=school_meta["local_rows"]:
        raise ValueError("유성구 학교 목록과 출처 메타데이터의 학교 수가 다릅니다.")
    school_dates=", ".join(sorted(schools["데이터기준일자"].dropna().astype(str).unique()))
    bank.add("보조자료 | 유성구 운영 학교 수",int(schools["학교ID"].nunique()),"곳",
        "Worker_2가 추출한 관내 운영 학교 목록의 고유 학교ID 수",f"시설자료 기준일: {school_dates}",school_path,school_meta_path)
    child_path="data/processed/reference/child_zones.csv"
    child_meta_path="data/processed/reference/child_zones_source.json"
    child=bank.csv(child_path);child_meta=bank.obj(child_meta_path)
    if len(child)!=child_meta["local_rows"]:raise ValueError("보호구역 시설 자료와 출처 메타데이터의 행 수가 다릅니다.")
    child_dates=", ".join(sorted(child["데이터기준일자"].dropna().astype(str).unique()))
    bank.add("보조자료 | 어린이보호구역 지정시설 자료 행 수",len(child),"행",
        "명시적 해제만 제외한 보호구역 표준데이터 대상시설 행 수; 지정여부 공란 행을 포함하며 법적 구역 수나 중복 제거 시설 수가 아니다.",
        f"시설자료 기준일: {child_dates}",child_path,child_meta_path)
    accident_path="data/processed/accident_coordinate_quality.json"
    accident=bank.obj(accident_path)
    if not 0<=accident["inside_current_osm_boundary"]<=accident["n_records"]:
        raise ValueError("교통사고 경계 내부 건수가 전체 건수를 벗어납니다.")
    accident_period="~".join(str(pd.Timestamp(value).date()) for value in accident["date_range"])
    bank.add("보조자료 | 교통사고 전체 건수",int(accident["n_records"]),"건","좌표 변환 품질 산출물에 기록된 전체 교통사고 수",accident_period,accident_path)
    bank.add("보조자료 | 교통사고 유성구 경계 내부 건수",int(accident["inside_current_osm_boundary"]),"건",
        "EPSG:5179 원좌표를 WGS84로 변환한 뒤 현재 OSM 유성구 경계 안에 속한 사고 수",accident_period,accident_path)
    review_path="data/processed/geocode_review_30.csv";quality_path="data/processed/geocode_quality.json"
    review=bank.csv(review_path);quality=bank.obj(quality_path)
    sample,completed=map_review_counts(review)
    if completed!=quality["manual_map_review_n"] or bool(completed==sample)!=quality["manual_map_review_done"]:
        raise ValueError("지도 확인 목록과 좌표 품질 요약의 검토 결과가 다릅니다.")
    condition=f"좌표 연결 성공 지점의 무작위 표본; {quality['manual_review_scope']}"
    bank.add("좌표 검토 | 무작위 지도 확인 대상 지점 수",sample,"곳","지도 확인 목록의 중복 없는 표본 지점 수",condition,review_path,quality_path)
    bank.add("좌표 검토 | 무작위 지도 확인 완료 지점 수",completed,"곳","review_status가 map_checked인 지점 수; 좌표 측량 정확도 판정 수가 아니다.",condition,review_path,quality_path)
    bank.add("좌표 검토 | 무작위 지도 확인 완료율",ratio(completed,sample)*100,"%","지도 확인 완료 지점 수 / 표본 지점 수 ×100; 위치 정확률과 구별한다.",condition,review_path,quality_path)
    for label,frame,path in [("신고위치",reports,"data/processed/reports.parquet"),("단속장소",enforcement,"data/processed/enforcement.parquet")]:
        bank.add(f"위치 문자열 | {label} 종류 수(앞뒤 공백 제거)",trimmed_location_count(frame.raw_loc),"종류",
            "원문 위치 문자열의 앞뒤 공백만 제거한 고유값 수; 내부 공백과 동·지번·부근 등의 표현은 보존하며 정규화 지점 수와 구별한다.",date_period(frame),path)


def add_photo_metrics(bank):
    results_path="data/photo_demo/evaluation_results.csv"
    if not (bank.root/results_path).exists():
        bank.add("사진 평가 | 추후 추가",None,"","Worker_2의 사진 평가 산출물이 확정되면 추가한다. 빈 값은 0을 뜻하지 않는다.","사진 평가 자료 확정 후 기재")
        return
    frozen_path="data/photo_demo/evaluation_frozen.json";summary_path="data/photo_demo/evaluation_summary.json"
    cases_path="data/photo_demo/evaluation_by_case.csv"
    results=bank.csv(results_path);frozen=bank.obj(frozen_path);saved=bank.obj(summary_path)
    if results.pair_id.duplicated().any() or len(results)!=frozen["n_pairs"]:
        raise ValueError("사진 고정 평가의 쌍 수 또는 ID가 맞지 않습니다.")
    included=results.evaluation_include.astype(str).str.lower().eq("true")
    if not included.all():raise ValueError("평가 예측에 통계 제외 시연 쌍이 들어 있습니다.")
    if set(frozen["development_source_ids"])&set(frozen["evaluation_source_ids"]):
        raise ValueError("사진 개발/평가 원천이 겹칩니다.")
    if set(results.source_id)!=set(frozen["evaluation_source_ids"]):
        raise ValueError("사진 평가 예측과 고정 원천 목록이 다릅니다.")
    computed=photo_decision_counts(results)
    for key in ("n_pairs","normal_recall","rejection_recall","balanced_accuracy","verdict_accuracy"):
        if not np.isclose(computed[key],saved[key],atol=1e-12):raise ValueError(f"사진 요약 수치와 예측 집계가 다릅니다: {key}")
    condition=(f"서로 다른 AI 바탕 사진 {results.source_id.nunique()}장에서 파생한 합성 번호판·합성 EXIF 쌍; "
        "최종 고정 평가; 통계 제외 시연은 포함하지 않음")
    for suffix,value,unit,definition in [
        ("평가 쌍 수",computed["n_pairs"],"쌍","고정 평가에 포함된 서로 다른 pair_id 수"),
        ("정상 쌍 수",computed["n_normal"],"쌍","expected_verdict가 요건 충족인 평가 쌍 수"),
        ("거부 쌍 수",computed["n_reject"],"쌍","expected_verdict가 보완 필요인 평가 쌍 수"),
        ("정상 쌍 중 요건 충족 판정 수",computed["normal_accepted"],"쌍","실제 정상 쌍 중 요건 충족으로 판정한 수"),
        ("거부 쌍 중 보완 필요 판정 수",computed["reject_detected"],"쌍","실제 거부 쌍 중 보완 필요로 판정한 수"),
        ("판정 일치 쌍 수",computed["n_correct"],"쌍","verdict와 expected_verdict가 같은 평가 쌍 수"),
        ("정상 수용률",computed["normal_recall"]*100,"%","정상 쌍 중 요건 충족 판정 수 / 정상 쌍 수 ×100"),
        ("거부 검출률",computed["rejection_recall"]*100,"%","거부 쌍 중 보완 필요 판정 수 / 거부 쌍 수 ×100"),
        ("균형 정확도",computed["balanced_accuracy"]*100,"%","(정상 수용률 + 거부 검출률) / 2; 두 클래스에 같은 가중치를 준다."),
        ("전체 정확도",computed["verdict_accuracy"]*100,"%","판정 일치 쌍 수 / 전체 평가 쌍 수 ×100")]:
        bank.add(f"사진 평가 | {suffix}",value,unit,definition,condition,results_path,summary_path,frozen_path)
    cases=bank.csv(cases_path)
    for case,frame in results.groupby("case",sort=False):
        row=cases.loc[cases.case.eq(case)]
        if len(row)!=1 or int(row.iloc[0].n_pairs)!=len(frame):raise ValueError("사진 유형별 쌍 수가 요약과 다릅니다.")
        verdicts=sorted(frame.expected_verdict.unique())
        bank.add(f"사진 구성 | {case} 쌍 수",len(frame),"쌍","case별 고정 평가 쌍 수",f"{condition}; 정답 판정: {', '.join(verdicts)}",results_path,cases_path)
    ai_path="data/photo_demo/ai_original_results.csv";ai=bank.csv(ai_path)
    if ai.source_id.duplicated().any() or ai.raw_expected_plate.isna().any():raise ValueError("AI 원본 판독 정답/원천이 유효하지 않습니다.")
    exact=ai.plate.fillna("").eq(ai.raw_expected_plate)
    recorded=ai.exact_match.astype(str).str.lower().map({"true":True,"false":False})
    if recorded.isna().any() or not exact.eq(recorded).all():raise ValueError("AI 원본 번호 완전 일치 결과가 저장 판정과 다릅니다.")
    ai_condition=f"번호판을 덮지 않은 AI 원본; 정답 번호 고유값 {ai.raw_expected_plate.nunique()}개; 합성 평가 쌍과 별도 집계"
    bank.add("AI 원본 판독 | 평가 사진 수",len(ai),"장","가리지 않은 AI 원본 번호판의 판독 결과 행 수",ai_condition,ai_path)
    bank.add("AI 원본 판독 | 번호 완전 일치 수",int(exact.sum()),"장","저장된 판독 번호와 원본 정답 번호가 완전히 같은 사진 수",ai_condition,ai_path)
    bank.add("AI 원본 판독 | 번호 완전 일치율",ratio(int(exact.sum()),len(ai))*100,"%","번호 완전 일치 수 / AI 원본 판독 사진 수 ×100",ai_condition,ai_path)
    real_path="data/photo_demo/real_plate_results.csv";real=bank.csv(real_path)
    n_real,n_match,n_unreadable=real_photo_counts(real)
    real_condition="공개 사용권 실차 사진 중 truth_status가 수기 확인인 정성 확인 대상; 가림·수기 미확정은 제외"
    bank.add("실차 정성 확인 | 수기 정답 확인 사진 수",n_real,"장","수기 정답을 확정한 공개 실차 사진의 수",real_condition,real_path)
    bank.add("실차 정성 확인 | 번호 완전 일치 수",n_match,"장","수기 정답과 저장된 판독 번호가 완전히 같은 사진 수",real_condition,real_path)
    bank.add("실차 정성 확인 | 번호 판독 불가 수",n_unreadable,"장","수기 정답 확인 대상 중 저장된 판독 번호가 없는 사진 수; 오독과 구별한다.",real_condition,real_path)
    thresholds=frozen["thresholds"]
    for key,label,unit,definition in [
        ("min_gap_s","최소 촬영 간격","초","두 사진의 촬영 시각 차이가 이 값 이상이어야 한다."),
        ("max_move_m","최대 사진 간 거리","m","두 사진의 GPS 사이 거리가 이 값 이하여야 한다."),
        ("max_report_m","최대 신고 지점 거리","m","두 사진 각각과 신고 지점의 거리 중 큰 값이 이 값 이하여야 한다.")]:
        bank.add(f"사진 판정 기준 | {label}",thresholds[key],unit,definition,"최종 사진 평가 전 고정한 판정 기준",frozen_path)


def build(root=ROOT):
    bank=Bank(root)
    reports=pd.read_parquet(bank.root/"data/processed/reports.parquet")
    enforcement=pd.read_parquet(bank.root/"data/processed/enforcement.parquet")
    reports["ts"]=pd.to_datetime(reports.ts);enforcement["ts"]=pd.to_datetime(enforcement.ts)
    spots=bank.csv("data/processed/spots.csv")
    add_descriptive(bank,reports,enforcement,spots)
    add_priority(bank)
    add_routes(bank)
    add_choice_and_emerging(bank,reports)
    add_reference_and_review(bank,reports,enforcement)
    add_photo_metrics(bank)
    result=pd.DataFrame(bank.rows,columns=COLUMNS)
    if result.duplicated(["항목","기간·조건"]).any():raise ValueError("핵심 수치 항목·조건이 중복되었습니다.")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"reports/key_numbers.csv")
    args=parser.parse_args()
    frame=build();args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False,encoding="utf-8-sig")
    checked=pd.read_csv(args.output,encoding="utf-8-sig",keep_default_na=False)
    if len(checked)!=len(frame) or checked.columns.tolist()!=COLUMNS:raise AssertionError("출력 CSV 행 수·열 이름 불일치")
    print(f"{args.output}: {len(frame)}행; 사진 평가 {int(frame['값'].isna().sum())}행은 빈 값으로 예약")


if __name__=="__main__":main()
