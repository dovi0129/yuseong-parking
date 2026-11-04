"""Freeze planner choice using Jan-Jul scores and Aug-Sep outcomes only."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd

from src.pipeline.priority import fit_scores
from .domain import ROOT,ROUTES,Plan,PlanConfig,PlanningData,day_type,load_planning_data
from .network import TravelNetwork,build_matrix
from .planner import solve
from .evaluate import captured_reports

INNER_DATA=None
INNER_NETWORK=None


def choose_planner(summary: pd.DataFrame,margin=.05):
    """Input scenario,method,mean_caught (4 x 2 rows); return rule-based choice.

    Sum the four weekday-condition means for each method; choose optimized only
    when its sum exceeds b3 by at least 0.05. Within 0.05 choose simpler b3.
    Output dict: chosen_method, sums, difference_optimized_minus_b3, threshold.
    """
    expected={f"{window}_K{k}" for window in ("10-14","17-21") for k in (1,2)}
    if set(summary.scenario)!=expected or set(summary.method)!={"b3","optimized"} or len(summary)!=8:
        raise ValueError("사전 고정한 평일 4조건×2방식이 필요합니다.")
    sums=summary.groupby("method").mean_caught.sum()
    delta=float(sums.optimized-sums.b3)
    selected="optimized" if delta>margin or abs(delta-margin)<1e-12 else "b3"
    return {"chosen_method":selected,"sums":{m:float(v) for m,v in sums.items()},
            "difference_optimized_minus_b3":delta,"threshold":margin}


def make_inner_plan(config):
    optimized=solve(INNER_DATA,INNER_NETWORK,config,"optimized")
    b3=solve(INNER_DATA,INNER_NETWORK,config,"b3")
    if optimized.expected_weighted+1e-10<b3.expected_weighted:
        raise AssertionError("내부 검증 계획의 학습 목적함수 하한 실패")
    name=f"{config.start_min//60:02}-{config.end_min//60:02}_K{config.vehicles}"
    return name,{"optimized":optimized.to_dict(),"b3":b3.to_dict()}


def main():
    global INNER_DATA,INNER_NETWORK
    freeze={"rule":"sum of weekday mean caught over four fixed conditions; optimized requires >=0.05 advantage",
        "fit":"2024-01-01 <= ts < 2024-08-01","validation":"2024-08-01 <= ts < 2024-10-01",
        "margin":.05,"conditions":["10-14_K1","10-14_K2","17-21_K1","17-21_K2"],"test_used":False}
    (ROUTES/"planner_choice_rule.json").write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
    reports=pd.read_parquet(ROOT/"data/processed/reports.parquet",filters=[("ts","<",pd.Timestamp("2024-10-01"))])
    reports["ts"]=pd.to_datetime(reports.ts)
    if not (reports.ts<pd.Timestamp("2024-10-01")).all():raise AssertionError("내부 선택에 테스트 기록이 들어왔습니다.")
    priority_choice=json.loads((ROOT/"data/processed/priority_choice.json").read_text())
    alpha=float(priority_choice["alpha"]);score_method=priority_choice["chosen_method"]
    fitted=fit_scores(reports,pd.Timestamp("2024-01-01"),pd.Timestamp("2024-08-01"),alpha)
    fitted.to_csv(ROUTES/"inner_priority_train.csv",index=False)
    full=load_planning_data()
    network=build_matrix(full)
    counts=reports.loc[reports.ts<pd.Timestamp("2024-08-01")].groupby("spot_id").size()
    spots=full.spots.copy()
    spots["n_reports_train"]=spots.spot_id.map(counts).fillna(0)
    spots=spots.loc[spots.n_reports_train>0].sort_values(["n_reports_train","spot_id"],ascending=[False,True]).reset_index(drop=True)
    INNER_DATA=PlanningData(spots,fitted[fitted.spot_id.isin(spots.spot_id)],score_method,"inner_train")
    ids=["DEPOT"]+spots.spot_id.tolist();indexes=[network.ids.index(s) for s in ids]
    INNER_NETWORK=TravelNetwork(ids,network.seconds[np.ix_(indexes,indexes)],
        [network.coordinates[i] for i in indexes],network.provenance)
    configs=[PlanConfig(start,end,vehicles,5.,"평일",True,30.)
             for start,end in ((600,840),(1020,1260)) for vehicles in (1,2)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        plans=dict(pool.map(make_inner_plan,configs))
    (ROUTES/"planner_inner_plans.json").write_text(json.dumps({"plans":plans,"fit":freeze["fit"],
        "candidate_spots":spots.spot_id.tolist(),"alpha":alpha,"score_method":score_method},ensure_ascii=False,indent=2))
    validation=reports.loc[reports.ts>=pd.Timestamp("2024-08-01")].copy()
    validation["date"]=validation.ts.dt.date.astype(str)
    dates=[d for d in pd.date_range("2024-08-01","2024-09-30") if day_type(d)=="평일"]
    risk=dict(zip(spots.spot_id,spots.risk_weight));rows=[]
    for current in dates:
        daily=validation.loc[validation.date.eq(current.date().isoformat())]
        for scenario,methods in plans.items():
            for method,obj in methods.items():
                count,weighted=captured_reports(Plan.from_dict(obj),daily,risk)
                rows.append({"date":current.date().isoformat(),"scenario":scenario,"method":method,"caught":count,"risk_caught":weighted})
    daily=pd.DataFrame(rows);daily.to_csv(ROUTES/"planner_inner_daily.csv",index=False)
    summary=(daily.groupby(["scenario","method"]).agg(mean_caught=("caught","mean"),
        total_caught=("caught","sum"),days=("date","nunique")).reset_index())
    summary.to_csv(ROUTES/"planner_inner_summary.csv",index=False)
    choice={**choose_planner(summary),"rule":freeze,"alpha":alpha,"score_method":score_method,
        "validation_days":len(dates),"inner_candidate_count":len(spots),"method_selection_uses_test":False,
        "validation_summary":summary.to_dict("records"),
        "reason":"1~7월 점수로 고정 계획을 만들고 8~9월 평일 4조건의 일평균 caught 합으로 선택함"}
    (ROUTES/"planner_choice.json").write_text(json.dumps(choice,ensure_ascii=False,indent=2))
    print(json.dumps(choice,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
