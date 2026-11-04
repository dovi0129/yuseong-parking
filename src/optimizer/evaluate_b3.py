"""Add the information-matched b3 baseline without rewriting the frozen results."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .domain import ROOT,ROUTES,Plan,PlanningData
from .evaluate import captured_reports
from .network import TravelNetwork,build_matrix
from .planner import solve


def paired_b3(daily,repetitions=2000):
    rng=np.random.default_rng(20241001)
    rows=[]
    for scenario,frame in daily.groupby("scenario",sort=True):
        for metric in ("caught","caught_per_person_hour","risk_caught","travel_min"):
            pivot=frame.pivot(index="date",columns="method",values=metric).sort_index()
            delta=(pivot.optimized-pivot.b3).to_numpy(float)
            boot=delta[rng.integers(len(delta),size=(repetitions,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(scenario=scenario,baseline="b3",metric=metric,
                optimized_mean=float(pivot.optimized.mean()),baseline_mean=float(pivot.b3.mean()),
                mean_delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),days=len(delta)))
    return pd.DataFrame(rows)


def main():
    original_names=("plans_frozen.json","evaluation_daily.csv","evaluation_summary.csv","evaluation_paired.csv")
    before={n:hashlib.sha256((ROUTES/n).read_bytes()).hexdigest() for n in original_names}
    frozen=json.loads((ROUTES/"plans_frozen.json").read_text())
    data=PlanningData(pd.DataFrame(frozen["planning_spots"]),pd.DataFrame(frozen["planning_priority"]),
        frozen["chosen_method"],frozen["source"],frozen["excluded_spots"])
    network=build_matrix(data,speed_kmh=frozen["network"]["speed_kmh"])
    original=pd.read_csv(ROUTES/"evaluation_daily.csv")
    reports=pd.read_parquet(ROOT/"data/processed/reports.parquet")
    reports["ts"]=pd.to_datetime(reports.ts)
    test=reports.loc[reports.split.eq("test")].copy()
    test["date"]=test.ts.dt.date.astype(str)
    test["spot_id"]=test.spot_id.astype(str)
    risk=dict(zip(data.spots.spot_id,data.spots.risk_weight))
    plans={};new_rows=[]
    for key,obj in frozen["plans"].items():
        if not key.endswith("_optimized"):continue
        base_plan=Plan.from_dict(obj)
        plans[key.replace("optimized","b3")]=solve(data,network,base_plan.config,"b3").to_dict()
    for old in original.loc[original.method.eq("optimized")].to_dict("records"):
        daily=test.loc[test.date.eq(old["date"])]
        plan=Plan.from_dict(plans[f"{old['scenario']}_{old['day_type']}_b3"])
        count,weighted=captured_reports(plan,daily,risk)
        person_hours=plan.config.vehicles*(plan.config.end_min-plan.config.start_min)/60
        new_rows.append({**old,"method":"b3","caught":count,"risk_caught":weighted,
            "caught_per_person_hour":count/person_hours,"travel_min":plan.travel_s/60,
            "expected":plan.expected_reports,"visits":len(plan.visits),"plan_seconds":plan.elapsed_s})
    combined=pd.concat([original,pd.DataFrame(new_rows)],ignore_index=True)
    combined.to_csv(ROUTES/"evaluation_with_b3_daily.csv",index=False)
    comparison=paired_b3(combined)
    comparison.to_csv(ROUTES/"evaluation_b3_paired.csv",index=False)
    (ROUTES/"plans_b3.json").write_text(json.dumps({"reference_sha256":before,"plans":plans,
        "definition":"rank by overlapping hour_block score * risk_weight; capacity identical to b1; nearest-neighbor order",
        "chosen_method":data.chosen_method},ensure_ascii=False,indent=2))
    # Targeted counterfactual: remove newly geocoded Doryong 3-1 from the top
    # frequency pool; the original 61-report Gwanpyeong 947 then enters it.
    added="S4fccbab6a78782e6";displaced="Sdd27d8bb9736532e"
    if added in set(data.spots.spot_id) and displaced in set(data.spots.spot_id):
        old_spots=data.spots.loc[~data.spots.spot_id.eq(added)].head(48).copy().reset_index(drop=True)
        assert displaced in set(old_spots.spot_id)
        ids=["DEPOT"]+old_spots.spot_id.tolist()
        indexes=[network.ids.index(s) for s in ids]
        old_network=TravelNetwork(ids,network.seconds[np.ix_(indexes,indexes)],
            [network.coordinates[i] for i in indexes],network.provenance)
        old_data=PlanningData(old_spots,data.priority,data.chosen_method,data.source)
        config=Plan.from_dict(frozen["plans"]["17-21_K1_평일_b1"]).config
        old_plan=solve(old_data,old_network,config,"b1")
        dates=original[(original.scenario=="17-21_K1")&(original.method=="b1")].date.tolist()
        counts=[captured_reports(old_plan,test.loc[test.date.eq(d)],risk)[0] for d in dates]
        current=original[(original.scenario=="17-21_K1")&(original.method=="b1")]
        diagnostic={"changed_factor":"eligible coordinate candidate pool, not sorting rule or common-point road matrix",
            "added":added,"displaced":displaced,"old_pool_visits":len(old_plan.visits),
            "old_pool_caught_mean_same_final_matrix":float(np.mean(counts)),
            "final_pool_caught_mean":float(current.caught.mean()),
            "old_pool_travel_min_same_final_matrix":old_plan.travel_s/60,
            "final_pool_travel_min":float(current.travel_min.iloc[0]),
            "old_pool_ids":ids[1:],"old_pool_plan":old_plan.to_dict()}
        (ROUTES/"b1_scope_diagnostic.json").write_text(json.dumps(diagnostic,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in diagnostic.items() if k not in ("old_pool_ids","old_pool_plan")},ensure_ascii=False))
    after={n:hashlib.sha256((ROUTES/n).read_bytes()).hexdigest() for n in original_names}
    if before!=after:raise AssertionError("기존 평가 산출물이 변경되었습니다.")
    print(comparison.loc[comparison.metric.eq("caught")].to_string(index=False))


if __name__=="__main__":main()
