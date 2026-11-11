from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

import pandas as pd

from .api import Context,generate_plan,classify_reports,repeat_spots,export_csv,route_polylines,overview,overview_filtered,reception_groups
from .domain import ROUTES
from .processing import PLAN_COLUMNS,RECEPTION_COLUMNS,SPOT_COLUMNS
from .test_contract import fixture


class PublicAPITests(unittest.TestCase):
    def context(self):
        data,network=fixture(score_a=10,score_b=5,risk_b=3)
        catalog=data.spots.copy();catalog["n_enf_2023"]=[7,3]
        return Context(data,network,catalog,ROUTES,True)

    def test_plan_dataframe_known_risk_selection(self):
        context=self.context()
        frame=generate_plan("2024-10-02","10:00","10:03",1,True,service_minutes=1,solver_seconds=.2,context=context,method="optimized")
        self.assertEqual(frame.columns.tolist(),PLAN_COLUMNS)
        self.assertEqual(frame.iloc[0]["단속 차량"],"1호차")
        self.assertEqual(frame["지점 ID"].tolist(),["b"])
        self.assertEqual(frame.iloc[0]["도착 예정"],"2024-10-02 10:01:00")
        self.assertEqual(frame.attrs["summary"]["expected_weighted"],15.)
        lines=route_polylines(frame,context=context)
        self.assertEqual(lines.columns.tolist(),["단속 차량","경로 좌표"])
        self.assertEqual(len(lines),1)

    def test_receipt_from_dataframe_and_export(self):
        context=self.context()
        plan=generate_plan("2024-10-02","10:00","10:03",1,True,service_minutes=1,solver_seconds=.2,context=context)
        reports=pd.DataFrame([dict(report_id=str(i),ts=pd.Timestamp("2024-10-02 10:00:20")+pd.Timedelta(seconds=i),spot_id="b",raw_loc="B") for i in range(2)])
        result=classify_reports("2024-10-02",plan_df=plan,reports=reports,limit=1,context=context)
        self.assertEqual(result.columns.tolist(),RECEPTION_COLUMNS)
        self.assertEqual(result.iloc[0]["조치 구분"],"이달 순회 지점")
        self.assertEqual(result.iloc[0]["시설 개선 검토"],"해당")
        self.assertEqual(result.iloc[0]["금일 순번"],1)
        reply=result.iloc[0]["회신 내용"]
        self.assertIn("이동형 단속 차량 순회",reply)
        for forbidden in ("학습 기간","조치 구분","10:01","17-21","점검 검토 시간대"):
            self.assertNotIn(forbidden,reply)
        self.assertTrue(export_csv(result).startswith(b"\xef\xbb\xbf"))
        info=export_csv(result,ROUTES/"test_exports"/"api_receipt.csv")
        self.assertEqual(info["rows"],1)
        self.assertEqual(repeat_spots(context=context).columns.tolist(),SPOT_COLUMNS)

    def test_overview_ignores_heldout_counts(self):
        context=self.context()
        reports=pd.DataFrame([dict(split=s,ts=pd.Timestamp(t),spot_id="a") for s,t in
            [("train","2024-09-02 17:10"),("train","2024-09-02 17:20"),("test","2024-10-02 17:10")]])
        out=overview(context=context,reports=reports)
        self.assertEqual(out["summary"].iloc[0]["학습 신고 (건)"],2)
        self.assertEqual(out["summary"].iloc[0]["당일 반복 신고 (건)"],2)
        self.assertEqual(out["heatmap"].to_numpy().sum(),2)
        self.assertEqual(out["concentration"].iloc[-1]["누적 신고 비율 (%)"],100.)

    def test_filtered_overview_queries_counts_but_freezes_train_grades(self):
        context=self.context()
        context.catalog.loc[context.catalog.spot_id.eq("a"),"n_reports_train"]=10
        context.catalog.loc[context.catalog.spot_id.eq("b"),"n_reports_train"]=0
        reports=pd.DataFrame([dict(ts=pd.Timestamp(t),spot_id=s) for s,t in
            [("a","2024-10-02 17:10"),("a","2024-10-02 18:20"),("b","2024-10-02 19:00"),("b","2024-10-03 10:00")]])
        out=overview_filtered("2024-10-02","2024-10-02",context=context,reports=reports,grade_basis="train")
        self.assertEqual(out["summary"].iloc[0]["신고 건수"],3)
        self.assertEqual(out["summary"].iloc[0]["반복 지점 수"],1)
        self.assertEqual(out["spots"]["지점 등급"].tolist(),["반복","일반"])
        self.assertEqual(out["heatmap"].to_numpy().sum(),3)
        restricted=overview_filtered("2024-10-02","2024-10-02",dong="나동",grades=("일반",),context=context,reports=reports,grade_basis="train")
        self.assertEqual(restricted["spots"]["지점 ID"].tolist(),["b"])
        self.assertEqual(restricted["summary"].iloc[0]["신고 건수"],1)
        self.assertEqual(restricted["summary"].iloc[0]["상위 100개 지점 비중 (%)"],100.)
        empty=overview_filtered("2024-12-01","2024-12-31",context=context,reports=reports)
        self.assertEqual(empty["summary"].iloc[0]["신고 건수"],0)
        self.assertEqual(empty["heatmap"].shape,(7,24))

    def test_plan_cache_reuses_weekday_and_restamps_date(self):
        from . import api
        context=self.context();context.cache_token="unit_date_cache_"+uuid.uuid4().hex
        with patch.object(api,"solve",wraps=api.solve) as counted:
            first=generate_plan("2024-10-02","10:00","10:03",vehicles=1,service_minutes=1,solver_seconds=.2,context=context)
            second=generate_plan("2024-10-04","10:00","10:03",vehicles=1,service_minutes=1,solver_seconds=.2,context=context)
            self.assertEqual(counted.call_count,1)
        self.assertEqual(first["지점 ID"].tolist(),second["지점 ID"].tolist())
        self.assertTrue(second["도착 예정"].str.startswith("2024-10-04").all())
        self.assertTrue(first["도착 예정"].str.startswith("2024-10-02").all())

    def test_inner_choice_margin_prefers_simple_planner_near_tie(self):
        from .select_planner import choose_planner
        scenarios=["10-14_K1","10-14_K2","17-21_K1","17-21_K2"]
        def summary(delta):
            return pd.DataFrame([{"scenario":s,"method":m,"mean_caught":1.+(delta/4 if m=="optimized" else 0)}
                                 for s in scenarios for m in ("b3","optimized")])
        self.assertEqual(choose_planner(summary(.04))["chosen_method"],"b3")
        self.assertEqual(choose_planner(summary(.05))["chosen_method"],"optimized")
        self.assertEqual(choose_planner(summary(-1.))["chosen_method"],"b3")

    def test_overview_period_grade_matches_displayed_count(self):
        context=self.context()
        context.catalog.loc[context.catalog.spot_id.eq("a"),"n_reports_train"]=0
        reports=pd.DataFrame({"ts":pd.date_range("2024-10-02 10:00",periods=20,freq="min"),"spot_id":["a"]*20})
        period=overview_filtered("2024-10-02","2024-10-02",context=context,reports=reports)
        train=overview_filtered("2024-10-02","2024-10-02",grade_basis="train",context=context,reports=reports)
        self.assertEqual(period["spots"].iloc[0]["신고 건수"],20)
        self.assertEqual(period["spots"].iloc[0]["지점 등급"],"상습")
        self.assertEqual(train["spots"].iloc[0]["지점 등급"],"일반")


if __name__=="__main__":unittest.main()
