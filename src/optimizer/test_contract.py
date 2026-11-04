import json
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from .domain import BLOCKS, Plan, PlanConfig, PlanningData, ROUTES, Visit, day_type, hour_block, load_planning_data
from .evaluate import captured_reports, summarize
from .network import TravelNetwork
from .planner import solve
from .processing import grade,plan_table,spot_table,reception_table,save_csv,PLAN_COLUMNS,RECEPTION_COLUMNS,SPOT_COLUMNS


def fixture(score_a=1., score_b=1., risk_b=1., journey=60):
    spots = pd.DataFrame([dict(spot_id="a", label="A", dong="가동",lat=36.36,lon=127.35,
        n_reports_train=100.,risk_weight=1.,child_zone_300m=0),
        dict(spot_id="b",label="B",dong="나동",lat=36.37,lon=127.36,
        n_reports_train=50.,risk_weight=risk_b,child_zone_300m=1)])
    p = pd.DataFrame([dict(spot_id=s, day_type=d, hour_block=b, score=score, method="fixed_train")
        for s,score in [("a",score_a),("b",score_b)] for d in ("평일","주말") for _,_,b in BLOCKS])
    data = PlanningData(spots,p,"fixed_train","unit")
    matrix = np.array([[0,journey,journey],[journey,0,journey],[journey,journey,0]])
    return data, TravelNetwork(["DEPOT","a","b"],matrix,[(36.36,127.35)]*3,{"source":"known answer"})


class ContractTests(unittest.TestCase):
    def test_half_open_blocks_and_holidays(self):
        self.assertEqual(hour_block(7*3600-1),"00-07")
        self.assertEqual(hour_block(7*3600),"07-10")
        self.assertEqual(hour_block(14*3600),"14-17")
        self.assertEqual(day_type("2024-10-01"),"주말")
        self.assertEqual(day_type("2024-10-02"),"평일")

    def test_risk_changes_choice_under_binding_budget(self):
        data,network=fixture(score_a=10.,score_b=5.,risk_b=3.)
        config=PlanConfig(600,603,1,1.,solver_seconds=.2)
        risk=solve(data,network,config)
        plain=solve(data,network,replace(config,use_risk=False))
        self.assertEqual([v.spot_id for v in risk.visits],["b"])
        self.assertEqual([v.spot_id for v in plain.visits],["a"])

    def test_visit_copy_waits_for_correct_prize_block_once(self):
        data,network=fixture(journey=10)
        data.priority.loc[data.priority.spot_id.eq("b"),"score"]=0.
        data.priority.loc[data.priority.spot_id.eq("a") & data.priority.hour_block.eq("14-17"),"score"]=50.
        config=PlanConfig(13*60+59,14*60+10,2,1.,solver_seconds=.3)
        plan=solve(data,network,config)
        self.assertEqual(len(plan.visits),1)
        self.assertEqual(plan.visits[0].block,"14-17")
        self.assertEqual(plan.expected_reports,50.)

    def test_depot_return_excludes_infeasible_stop(self):
        data,network=fixture(journey=120)
        plan=solve(data,network,PlanConfig(600,603,1,1.,solver_seconds=.2))
        self.assertEqual(len(plan.visits),0)
        self.assertEqual(plan.travel_s,0)

    def test_information_matched_b3_uses_block_score_and_risk(self):
        data,network=fixture(score_a=10.,score_b=5.,risk_b=3.,journey=10)
        config=PlanConfig(600,601,1,2/3,solver_seconds=.2)
        b1=solve(data,network,config,"b1")
        b3=solve(data,network,config,"b3")
        self.assertEqual([v.spot_id for v in b1.visits],["a"])
        self.assertEqual([v.spot_id for v in b3.visits],["b"])

    def test_warm_started_training_objective_guard_all_default_conditions(self):
        data,network=fixture(score_a=10.,score_b=5.,risk_b=3.,journey=10)
        for start,end in [(600,840),(1020,1260)]:
            for vehicles in (1,2):
                config=PlanConfig(start,end,vehicles,5.,solver_seconds=.2)
                optimized=solve(data,network,config)
                b3=solve(data,network,config,"b3")
                self.assertTrue(optimized.initial_assignment_used)
                self.assertGreaterEqual(optimized.expected_weighted+1e-10,b3.expected_weighted)

    def test_matching_counts_records_once_in_arrival_block(self):
        config=PlanConfig()
        plan=Plan("unit",config,[Visit("a",17*3600,0,1.,2.,"A",36.36,127.35)])
        daily=pd.DataFrame({"spot_id":["a","a","a","b"],"hour_block":["17-21","17-21","14-17","17-21"]})
        self.assertEqual(captured_reports(plan,daily,{"a":2.,"b":5.}),(2,4.))

    def test_zero_reports_and_zero_matches_are_numeric_zero(self):
        plan=Plan("unit",PlanConfig(),[])
        empty=pd.DataFrame({"spot_id":pd.Series([],dtype="str"),"hour_block":pd.Series([],dtype="str")})
        self.assertEqual(captured_reports(plan,empty,{}),(0,0.))

    def test_heldout_spot_counts_cannot_select_candidates(self):
        directory=ROUTES / "test_inputs"
        directory.mkdir(parents=True,exist_ok=True)
        data,_=fixture()
        spots=data.spots.copy()
        spots["n_reports_test"]=[0,999999]
        spots.to_csv(directory/"spots.csv",index=False)
        data.priority.to_csv(directory/"priority_train.csv",index=False)
        (directory/"priority_choice.json").write_text(json.dumps({"chosen_method":"fixed_train"}))
        result=load_planning_data(directory,max_spots=1)
        self.assertEqual(result.spots.spot_id.tolist(),["a"])
        self.assertNotIn("n_reports_test",result.spots)

    def test_paired_bootstrap_preserves_zero_difference(self):
        frame=pd.DataFrame([dict(scenario="test",date=d,method=m,caught=v,caught_per_person_hour=v/4,
            risk_caught=v*2,travel_min=20.) for d,v in [("2024-10-02",0),("2024-10-04",4)]
            for m in ("b1","b2","optimized")])
        _,paired=summarize(frame,repetitions=100)
        self.assertTrue((paired[["mean_delta","ci_low","ci_high"]]==0).all().all())

    def test_grade_threshold_boundaries(self):
        self.assertEqual([grade(x) for x in (0,9,10,19,20)],["일반","일반","반복","반복","상습"])
        self.assertEqual(grade(9,5,10),"반복")
        with self.assertRaises(ValueError):grade(20,20,10)

    def test_reception_rules_and_prefix_groups(self):
        catalog=pd.DataFrame([dict(spot_id=s,n_reports_train=n,child_zone_300m=z)
                              for s,n,z in [("a",20,0),("b",10,0),("c",20,0),("d",9,1)]])
        plan=Plan("unit",PlanConfig(),[Visit("a",17*3600+30*60,0,1.,1.,"A",36.36,127.35)])
        reports=pd.DataFrame([dict(report_id=str(i),ts=pd.Timestamp(t),spot_id=s,raw_loc=s)
                             for i,(s,t) in enumerate([("a","2024-10-02 17:10"),("b","2024-10-02 17:20"),
                                 ("c","2024-10-02 17:25"),("d","2024-10-02 17:26"),("a","2024-10-02 17:40")])])
        result=reception_table(reports,catalog,plan,"2024-10-02")
        with self.subTest("post-arrival recurrence"):
            self.assertEqual(result["조치 구분"].tolist(),["금일 순회 포함","순회 경로 반영 검토","순회 경로 반영 검토","일반 처리","순회 경로 반영 검토"])
        with self.subTest("ordinary protected spot is excluded"):
            self.assertEqual(result["시설 개선 검토"].tolist(),["해당","비해당","해당","비해당","해당"])
        self.assertEqual(result.iloc[0]["묶음 번호"],result.iloc[4]["묶음 번호"])
        with self.subTest("daily serial group numbers"):
            self.assertEqual(result.iloc[0]["묶음 번호"],"20241002-001")
            self.assertEqual(result.iloc[1]["묶음 번호"],"20241002-002")
        self.assertEqual(result.iloc[4]["금일 순번"],2)
        prefix=reception_table(reports,catalog,plan,"2024-10-02",limit=1)
        self.assertEqual(prefix.iloc[0]["금일 순번"],1)
        self.assertIn("1번째",prefix.iloc[0]["회신 내용"])

    def test_all_exports_round_trip_korean_columns_and_bom(self):
        data,_=fixture()
        catalog=data.spots.copy();catalog["n_enf_2023"]=2
        plan=Plan("unit",PlanConfig(),[Visit("a",17*3600,0,1.,1.,"A",36.36,127.35)])
        empty=pd.DataFrame(columns=["ts","report_id","spot_id","raw_loc"])
        tables=[(plan_table(plan,"2024-10-02"),PLAN_COLUMNS),
                (reception_table(empty,catalog,plan,"2024-10-02"),RECEPTION_COLUMNS),
                (spot_table(catalog),SPOT_COLUMNS)]
        for i,(frame,columns) in enumerate(tables):
            path=ROUTES/"test_exports"/f"export_{i}.csv"
            result=save_csv(frame,path)
            self.assertEqual(result["rows"],len(frame));self.assertEqual(result["columns"],columns)
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
