"""Prepare every UI-standard plan using training inputs only, four CPU workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import time

import pandas as pd

from .api import load_context,plan_for_day_type,cache_existing_plan,_plan_cache_path,default_planner_method
from .domain import ROUTES,Plan

CONTEXT=None
SHIFTS=(("10:00","14:00"),("17:00","21:00"),("07:00","10:00"),("14:00","17:00"))


def init_worker():
    global CONTEXT
    CONTEXT=load_context()


def compute_one(task):
    dt,start,end,vehicles,risk=task
    started=time.monotonic()
    plan=plan_for_day_type(dt,start,end,vehicles,risk,context=CONTEXT)
    path,_=_plan_cache_path(CONTEXT,plan.config,plan.method,plan.score_method)
    if plan.method=="optimized" and (not plan.initial_assignment_used or plan.expected_weighted+1e-10<plan.seed_expected_weighted):
        raise AssertionError("사전계산 계획의 b3 목적함수 하한 실패")
    return {"day_type":dt,"start":start,"end":end,"vehicles":vehicles,"risk_weight":risk,
        "chosen_method":plan.score_method,"planner_method":plan.method,"path":str(path),"seconds":time.monotonic()-started,
        "expected":plan.expected_reports,"expected_weighted":plan.expected_weighted,
        "seed_expected_weighted":plan.seed_expected_weighted,"incumbent_source":plan.incumbent_source}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--priority-shifts-only",action="store_true")
    args=parser.parse_args()
    if not 1<=args.workers<=4:raise ValueError("사전계산 프로세스는 최대 4개입니다.")
    context=load_context()
    frozen=json.loads((ROUTES/"plans_frozen.json").read_text())
    if frozen["chosen_method"]!=context.data.chosen_method or frozen["candidate_spots"]!=context.data.spots.spot_id.tolist():
        raise ValueError("평가와 현재 학습 입력의 선택 방식/후보 버전이 다릅니다.")
    for obj in frozen["plans"].values():
        plan=Plan.from_dict(obj)
        if plan.method=="optimized" and plan.initial_assignment_used:
            cache_existing_plan(plan,context=context)
    shifts=SHIFTS[:2] if args.priority_shifts_only else SHIFTS
    tasks=[(dt,start,end,vehicles,risk) for start,end in shifts for dt in ("평일","주말")
           for vehicles in (1,2,3) for risk in (True,False)]
    manifest=[];status=ROUTES/"plan_cache/precompute_status.json"
    status.write_text(json.dumps({"state":"running","total":len(tasks),"complete":0}))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        futures={pool.submit(compute_one,task):task for task in tasks}
        for future in as_completed(futures):
            result=future.result();manifest.append(result)
            pd.DataFrame(manifest).to_csv(ROUTES/"plan_cache/precomputed_manifest.csv",index=False)
            status.write_text(json.dumps({"state":"running","total":len(tasks),"complete":len(manifest)}))
            print(json.dumps({"complete":len(manifest),"total":len(tasks),**result},ensure_ascii=False),flush=True)
    status.write_text(json.dumps({"state":"complete","total":len(tasks),"complete":len(manifest)}))
    (ROUTES/"plan_cache/precompute.done").touch()


if __name__=="__main__":main()
