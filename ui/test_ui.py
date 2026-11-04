"""Smoke test: every menu and tab renders without exceptions (streamlit AppTest)."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import photo

MAIN = Path(__file__).with_name("main.py")



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
        self.assertEqual(list(at.dataframe[0].value.columns), ["위치", "동", "지점 등급", "신고 건수", "단속 수", "위험 가중치"])
        at.multiselect(key="ov_grades").set_value(["상습"])
        self.submit(at)
        self.assertEqual(set(at.dataframe[0].value["지점 등급"]), {"상습"})
        self.assertEqual(len(at.get("download_button")), 1)

    def test_plan(self):
        at = self.open("순회 계획", "방문표")
        table = at.dataframe[0].value
        self.assertGreater(len(table), 0)
        self.assertEqual(list(table.columns), ["단속 차량", "순번", "위치", "도착 예정", "기대 신고 수", "위험 가중치"])
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
        self.assertEqual(list(table.columns), ["접수 일시", "위치", "지점 등급", "금일 순번", "묶음 번호", "조치 구분", "시설 개선 검토", "사진 검토"])
        self.assertEqual(len(at.text_area), 1)
        self.assertEqual(table["사진 검토"].iloc[0], "")
        at.radio(key="reception_tab").set_value("사진 검토").run()
        demo = next(s for s in at.selectbox if s.label == "시연 사진")
        demo.set_value(photo.demo_pairs()["pair_id"].iloc[0])
        self.submit(at)
        at.radio(key="reception_tab").set_value("접수 목록").run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertIn(at.dataframe[0].value["사진 검토"].iloc[0].split(":")[0], ("요건 충족", "보완 필요"))


if __name__ == "__main__":
    unittest.main()
