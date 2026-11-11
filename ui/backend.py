"""Cached adapters between the screens and src.optimizer.api.

Planning, classification, aggregation and CSV encoding stay in src.optimizer.api;
this module only caches those calls by input-file modification times.
"""
from __future__ import annotations

import inspect
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimizer import api, monthly, processing, surge  # noqa: E402
from src.optimizer.domain import day_type  # noqa: E402
from src.optimizer.network import get_depot  # noqa: E402

DATA = ROOT / "data/processed"
INPUTS = ("spots.csv", "priority_train.csv", "priority_choice.json", "reports.parquet")
REPEAT, HABITUAL = 10, 20
GRADES = ("상습", "반복", "일반")
SURGE = "급증"
CATEGORIES = (*GRADES, SURGE)
RECENT, PRIOR = "직전 한 달", "앞 석 달 월평균"
# Screen and CSV names of api columns (src keeps its own names; scripts outside ui/ read those).
ARRIVAL = "계획 도착 시각"        # 도착 예정: planned arrival, 5 min per stop at 25 km/h
EXPECTED = "과거 평균 신고 수"    # 기대 신고 수: mean reports per day of that spot and time block before the plan month
WEIGHT = "교통안전 가중치"       # 위험 가중치: 1 + 0.5 school 300 m + 0.5 child-zone facility 300 m + 0.5 min(accidents 200 m / 10, 1)
RISK_LABEL = "교통안전 우선"      # 위험 가중 (적용/미적용)
PATROL_CLASS = "순회 구분"        # 조치 구분: 이달 순회 지점 / 다음 달 순회 검토 / 일반 (processing.reception_table)
SHOWN = {"도착 예정": ARRIVAL, "기대 신고 수": EXPECTED, "위험 가중치": WEIGHT, "조치 구분": PATROL_CLASS}
YEAR_START, YEAR_END = date(2024, 1, 1), date(2024, 12, 31)
EVAL_START, EVAL_END = date(2024, 10, 1), date(2024, 12, 31)
SHIFTS = {"07:00~10:00": ("07:00", "10:00"), "10:00~14:00": ("10:00", "14:00"),
          "14:00~17:00": ("14:00", "17:00"), "17:00~21:00": ("17:00", "21:00")}


def version():
    files = tuple((name, (DATA / name).stat().st_mtime_ns) for name in INPUTS)
    months = tuple((m, (monthly.folder(m) / "priority_choice.json").stat().st_mtime_ns) for m in monthly.MONTHS
                   if (monthly.folder(m) / "priority_choice.json").exists())
    return files + months + (("planner", getattr(api, "PLANNER_CACHE_VERSION", None)),
                             ("processing", getattr(processing, "PROCESSING_RULES_VERSION", None)),
                             ("monthly", monthly.VERSION))


@st.cache_resource(show_spinner=False)
def context(ver):
    return api.load_context(speed_kmh=25.0)   # same float as the precomputed plans and matrices (cache keys)


@st.cache_resource(show_spinner=False)
def month_context(ver, month):
    """Planning inputs of reports up to the last day of the month before (src/optimizer/monthly.py)."""
    return monthly.context(month, speed_kmh=25.0)


def plan_basis(day):
    return monthly.basis_label(monthly.month_of(day))


def plan_unit(day):
    """A plan is made per month and day type: every 평일 of 2024-11 shares one route."""
    return f"{monthly.month_of(day)} {day_type(day)}"


@st.cache_data(show_spinner=False, max_entries=8)
def month_surge(ver, month):
    """급증 spots used for the month's 민원 접수: counted on the month before it (same information time as its plan)."""
    counted = pd.Period(month, "M") - 1
    if counted - 3 < surge.DATA_START:
        return frozenset()
    table = surge.surge_table(reports(ver), counted)
    return frozenset(table.loc[table.surge, "spot_id"])


@st.cache_data(show_spinner=False)
def reports(ver):
    frame = pd.read_parquet(DATA / "reports.parquet")
    frame["ts"] = pd.to_datetime(frame.ts)
    return frame


def dongs(ver):
    return sorted(context(ver).catalog.dong.dropna().unique())


def _overview(ver, start, end, dong, grades, frame):
    # 현황 grades follow the queried period (main decision 2026-10-30); planning/reception keep the 1~9월 basis.
    basis = {"grade_basis": "period"} if "grade_basis" in inspect.signature(api.overview_filtered).parameters else {}
    return api.overview_filtered(start, end, dong=dong, grades=tuple(grades), repeat_threshold=REPEAT,
                                 habitual_threshold=HABITUAL, context=context(ver), reports=frame, **basis)


@st.cache_data(show_spinner=False, max_entries=64)
def overview(ver, start, end, dong, categories):
    """Grades as before; 급증 adds the surge spots of the last complete month up to `end` (src/optimizer/surge.py).

    Shown spots = selected grades or (급증 selected and surge). The summary is the original overview_filtered over
    exactly those spots (their reports only), so with all grades or without 급증 it equals the earlier screen.
    """
    grades = tuple(g for g in GRADES if g in categories)
    month = surge.counted_month(end)
    table = surge.surge_table(reports(ver), month) if month is not None else None
    flagged = set(table.loc[table.surge, "spot_id"]) if table is not None else set()
    if SURGE not in categories or set(GRADES) <= set(categories) or not flagged:
        data = _overview(ver, start, end, dong, grades, reports(ver))
    else:
        every = _overview(ver, start, end, dong, GRADES, reports(ver))["spots"]
        keep = set(every.loc[every["지점 등급"].isin(grades) | every["지점 ID"].isin(flagged), "지점 ID"])
        frame = reports(ver)
        data = _overview(ver, start, end, dong, GRADES, frame.loc[frame.spot_id.astype(str).isin(keep)])
    spots = data["spots"]
    if table is None:
        spots[RECENT], spots[PRIOR], spots[SURGE] = float("nan"), float("nan"), False
    else:
        lookup = table.set_index("spot_id")
        spots[RECENT] = spots["지점 ID"].map(lookup.recent).fillna(0).astype(int)
        spots[PRIOR] = spots["지점 ID"].map(lookup.prior_mean).fillna(0.0)
        spots[SURGE] = spots["지점 ID"].isin(flagged)
    data["surge_month"] = None if month is None else f"{month.month}월"           # data cover 2024 only
    data["surge_prior"] = None if month is None else f"{(month - 3).month}~{(month - 1).month}월"
    return data


@st.cache_data(show_spinner=False, max_entries=48)
def plan(ver, day, shift, vehicles, risk):
    """Plan of the month of `day` (reports up to the end of the month before); 순회 계획 and 민원 접수 share it."""
    start, end = SHIFTS[shift]
    return api.generate_plan(day, start, end, risk_weight=risk, vehicles=vehicles,
                             context=month_context(ver, monthly.month_of(day)))


@st.cache_data(show_spinner=False, max_entries=48)
def routes(ver, day, shift, vehicles, risk):
    return api.route_polylines(plan(ver, day, shift, vehicles, risk), context=month_context(ver, monthly.month_of(day)))


@st.cache_data(show_spinner=False, max_entries=96)
def reception(ver, day, shift, vehicles, risk, through):
    frame = reports(ver)
    start, end = SHIFTS[shift]
    return api.classify_reports(day, REPEAT, HABITUAL, plan_df=plan(ver, day, shift, vehicles, risk),
                                through=datetime.combine(day, through), start=start, end=end, staff=vehicles,
                                context=context(ver), reports=frame.loc[frame.split.eq("test")],
                                surge_spots=month_surge(ver, monthly.month_of(day)))


def spot_point(ver, spot_id):
    catalog = context(ver).catalog
    row = catalog.loc[catalog.spot_id.eq(str(spot_id))]
    if row.empty or row.lat.isna().iloc[0] or row.lon.isna().iloc[0]:
        return None
    return float(row.lat.iloc[0]), float(row.lon.iloc[0])


def depot():
    return get_depot()


def shown(frame):
    return frame.rename(columns=SHOWN)


def csv(frame):
    return api.export_csv(shown(frame))
