"""Smoke test: every menu and tab renders without exceptions (streamlit AppTest)."""
import json
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import photo

MAIN = Path(__file__).with_name("main.py")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimizer import api, monthly, surge  # noqa: E402
from src.optimizer.domain import Plan  # noqa: E402

KWANPYEONG_1287 = "Sa470d4c77e7de3e0"


def html_text(at):
    return " ".join(e.proto.body for e in at.get("html"))



class ScreenTest(unittest.TestCase):
    def open(self, menu, tab=None):
        at = AppTest.from_file(str(MAIN), default_timeout=300)
        at.run()
        if menu != "현황":
            at.radio(key="menu").set_value(menu).run()
        if tab:
            page = {"현황": "overview", "순회 계획": "plan", "민원 접수": "reception"}[menu]
            at.radio(key=f"{page}_tab").set_value(tab).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def submit(self, at):
        next(b for b in at.button if b.label == "조회").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])

    def test_overview(self):
        at = self.open("현황", "명단")
        self.assertEqual(list(at.dataframe[0].value.columns), ["위치", "동", "지점 등급", "신고 건수", "단속 수", "교통안전 가중치",
                                                               "직전 한 달 (12월)", "앞 석 달 월평균 (9~11월)"])
        at.multiselect(key="ov_grades").set_value(["상습"])
        self.submit(at)
        self.assertEqual(set(at.dataframe[0].value["지점 등급"]), {"상습"})
        self.assertEqual(len(at.get("download_button")), 1)
        import backend as be
        plan = api.generate_plan(date(2024, 10, 2), "17:00", "21:00", 1, True, context=monthly.context("2024-10"))
        header = be.csv(plan).decode("utf-8-sig").splitlines()[0].split(",")
        self.assertIn("계획 도착 시각", header)
        self.assertIn("과거 평균 신고 수", header)
        self.assertIn("교통안전 가중치", header)
        self.assertNotIn("도착 예정", header)

    def test_plan(self):
        at = self.open("순회 계획", "방문표")
        table = at.dataframe[0].value
        self.assertGreater(len(table), 0)
        self.assertEqual(list(table.columns), ["단속 차량", "순번", "위치", "계획 도착 시각", "과거 평균 신고 수", "교통안전 가중치"])
        self.assertIn("교통안전 우선", [s.label for s in at.selectbox])
        at.radio(key="plan_tab").set_value("지도").run()
        self.assertFalse(at.exception)
        next(b for b in at.button if b.label == "로그아웃").click().run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.radio(key="menu").value, "현황")
        self.assertEqual(at.radio(key="overview_tab").value, "지도")

    def test_reception(self):
        at = self.open("민원 접수")
        table = at.dataframe[0].value
        self.assertGreater(len(table), 0)
        self.assertEqual(list(table.columns), ["접수 일시", "위치", "지점 등급", "금일 순번", "묶음 번호", "순회 구분", "시설 개선 검토", "사진 검토"])
        self.assertEqual(len(at.text_area), 1)
        self.assertEqual(table["사진 검토"].iloc[0], "")
        at.radio(key="reception_tab").set_value("사진 검토").run()
        demo = next(s for s in at.selectbox if s.label == "시연 사진")
        demo.set_value(photo.demo_pairs()["pair_id"].iloc[0])
        self.submit(at)
        at.radio(key="reception_tab").set_value("접수 목록").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertIn(at.dataframe[0].value["사진 검토"].iloc[0].split(":")[0], ("요건 충족", "보완 필요"))

    def test_overview_surge(self):
        at = self.open("현황", "명단")
        self.assertIn("23,508", html_text(at))   # all grades + 급증: same summary as before 급증 existed
        at.date_input(key="ov_period").set_value((date(2024, 1, 1), date(2024, 10, 31)))
        at.multiselect(key="ov_grades").set_value(["급증"])
        self.submit(at)
        table = at.dataframe[0].value
        self.assertIn("직전 한 달 (10월)", table.columns)
        self.assertIn("앞 석 달 월평균 (7~9월)", table.columns)
        # 51 = 2024-10 급증 spots (MonthlyBasisTest.test_surge_rule checks 관평동 1287 is one of them)
        expected = int(surge.surge_table(monthly.read_reports(), pd.Period("2024-10", "M")).surge.sum())
        self.assertIn(f'총 <span class="n">{expected:,}</span>건', html_text(at))
        recent, mean = table["직전 한 달 (10월)"], table["앞 석 달 월평균 (7~9월)"]
        self.assertTrue(((recent >= 5) & (recent >= 2 * mean - 1e-9)).all())
        at.radio(key="overview_tab").set_value("지도").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])

    def test_monthly_plan_screen(self):
        at = self.open("순회 계획", "방문표")
        self.assertIn("기준 자료 2024-01-01 ~ 2024-09-30", html_text(at))
        next(d for d in at.date_input if d.label == "날짜").set_value(date(2024, 11, 6))
        self.submit(at)
        self.assertIn("기준 자료 2024-01-01 ~ 2024-10-31", html_text(at))
        self.assertIn("계획 2024-11 평일", html_text(at))
        shown = at.dataframe[0].value
        plan = api.generate_plan(date(2024, 11, 6), "17:00", "21:00", 1, True, context=monthly.context("2024-11"))
        self.assertEqual(list(shown["위치"]), list(plan["위치"].head(len(shown))))
        at.radio(key="plan_tab").set_value("지도").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])


class MonthlyBasisTest(unittest.TestCase):
    """Information timing of the monthly plans; no screen."""

    @classmethod
    def setUpClass(cls):
        cls.reports = monthly.read_reports()

    def test_inputs_use_reports_before_the_month_only(self):
        for month in monthly.MONTHS:
            start, last, end = monthly.basis(month)
            folder = monthly.folder(month)
            meta = json.loads((folder / "priority_choice.json").read_text())
            self.assertLess(pd.Timestamp(meta["input_max_ts"]), end)
            self.assertEqual(meta["basis_last_day"], str(last.date()))
            # Rebuilding from reports with everything on/after the month start removed gives the same files.
            spots, scores, _ = monthly.build_inputs(month, self.reports.loc[self.reports.ts < end])
            disk_spots = pd.read_csv(folder / "spots.csv", dtype={"spot_id": str})
            disk_scores = pd.read_csv(folder / "priority_train.csv", dtype={"spot_id": str})
            pd.testing.assert_frame_equal(spots.reset_index(drop=True), disk_spots, check_dtype=False)
            pd.testing.assert_frame_equal(scores.reset_index(drop=True), disk_scores, check_dtype=False, rtol=1e-12)
            counts = self.reports.loc[self.reports.ts < end].groupby(self.reports.spot_id.astype(str)).size()
            self.assertTrue((disk_spots.n_reports_train == disk_spots.spot_id.map(counts).fillna(0)).all())

    def test_october_equals_frozen_plans(self):
        frozen = json.loads((ROOT / "data/processed/routes/plans_frozen.json").read_text())["plans"]
        ctx = monthly.context("2024-10")
        self.assertEqual(api._context_token(ctx), api._context_token(api.load_context()))
        for k in (1, 2):
            plan = api.plan_for_day_type("평일", "17:00", "21:00", k, True, context=ctx)
            fixed = Plan.from_dict(frozen[f"17-21_K{k}_평일_optimized"])
            self.assertEqual([(v.spot_id, v.arrival_s) for v in plan.visits], [(v.spot_id, v.arrival_s) for v in fixed.visits])

    def test_every_screen_combination_is_cached(self):
        import backend as be
        for month in monthly.MONTHS:
            # the screen's own context (backend.month_context) must hit the precomputed plans: the cache key includes speed_kmh
            for ctx in (monthly.context(month), be.month_context(be.version(), month)):
                missing = [c for c in monthly.combinations() if not monthly.cache_path(ctx, c).exists()]
                self.assertEqual(missing, [], month)

    def test_reception_uses_the_month_plan(self):
        """민원 접수 (backend.reception, the screen's path) classifies with the plan of the date's month."""
        import backend as be
        day, through = date(2024, 11, 7), pd.Timestamp("2024-11-07 23:50")
        ver = be.version()
        shown = be.reception(ver, day, "17:00~21:00", 1, True, through.time())
        test = self.reports.loc[self.reports.split.eq("test")]

        def classify(ctx):
            plan = api.generate_plan(day, "17:00", "21:00", 1, True, context=ctx)
            return api.classify_reports(day, plan_df=plan, through=through, staff=1, context=be.context(ver), reports=test,
                                        surge_spots=be.month_surge(ver, "2024-11"))

        pd.testing.assert_frame_equal(shown.reset_index(drop=True), classify(monthly.context("2024-11")).reset_index(drop=True))
        self.assertFalse(shown.reset_index(drop=True).equals(classify(api.load_context()).reset_index(drop=True)))
        self.assertTrue(shown["조치 구분"].eq("이달 순회 지점").any())
        # 순회 구분 depends on the spot only: every report at a route spot is 이달 순회 지점, whatever its time
        route = set(api.generate_plan(day, "17:00", "21:00", 1, True, context=monthly.context("2024-11"))["지점 ID"])
        self.assertTrue((shown["조치 구분"].eq("이달 순회 지점") == shown["지점 ID"].isin(route)).all())
        off = shown.loc[~shown["지점 ID"].isin(route)]
        hot = off["지점 등급"].isin(["반복", "상습"]) | off["지점 ID"].isin(be.month_surge(ver, "2024-11"))
        self.assertTrue((off["조치 구분"].eq("다음 달 순회 검토") == hot).all())

    def test_surge_rule(self):
        frame = pd.DataFrame({"ts": pd.to_datetime(["2024-03-05", "2024-04-02", "2024-04-03"] + ["2024-06-01"] * 5
                                                   + ["2024-06-02"] * 4 + ["2024-03-01"] * 9 + ["2024-06-03"] * 6),
                              "spot_id": ["A"] * 8 + ["B"] * 4 + ["C"] * 15})
        table = surge.surge_table(frame, pd.Period("2024-06", "M")).set_index("spot_id")
        self.assertEqual(table.surge.to_dict(), {"A": True, "B": False, "C": True})   # C: 6 = 2 x 3 is included
        self.assertEqual(str(surge.counted_month("2024-10-31")), "2024-10")
        self.assertEqual(str(surge.counted_month("2024-11-15")), "2024-10")
        self.assertIsNone(surge.counted_month("2024-03-31"))
        real = surge.surge_table(self.reports, pd.Period("2024-10", "M")).set_index("spot_id")
        self.assertTrue(real.loc[KWANPYEONG_1287, "surge"])


if __name__ == "__main__":
    unittest.main()
