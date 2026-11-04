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

from src.optimizer import api, processing  # noqa: E402
from src.optimizer.network import get_depot  # noqa: E402

DATA = ROOT / "data/processed"
INPUTS = ("spots.csv", "priority_train.csv", "priority_choice.json", "reports.parquet")
REPEAT, HABITUAL = 10, 20
GRADES = ("상습", "반복", "일반")
YEAR_START, YEAR_END = date(2024, 1, 1), date(2024, 12, 31)
EVAL_START, EVAL_END = date(2024, 10, 1), date(2024, 12, 31)
SHIFTS = {"07:00~10:00": ("07:00", "10:00"), "10:00~14:00": ("10:00", "14:00"),
          "14:00~17:00": ("14:00", "17:00"), "17:00~21:00": ("17:00", "21:00")}


def version():
    files = tuple((name, (DATA / name).stat().st_mtime_ns) for name in INPUTS)
    return files + (("planner", getattr(api, "PLANNER_CACHE_VERSION", None)),
                    ("processing", getattr(processing, "PROCESSING_RULES_VERSION", None)))


@st.cache_resource(show_spinner=False)
def context(ver):
    return api.load_context(speed_kmh=25)


@st.cache_data(show_spinner=False)
def reports(ver):
    frame = pd.read_parquet(DATA / "reports.parquet")
    frame["ts"] = pd.to_datetime(frame.ts)
    return frame


def dongs(ver):
    return sorted(context(ver).catalog.dong.dropna().unique())


@st.cache_data(show_spinner=False, max_entries=64)
def overview(ver, start, end, dong, grades):
    # 현황 grades follow the queried period (main decision 2026-10-30); planning/reception keep the 1~9월 basis.
    basis = {"grade_basis": "period"} if "grade_basis" in inspect.signature(api.overview_filtered).parameters else {}
    return api.overview_filtered(start, end, dong=dong, grades=tuple(grades), repeat_threshold=REPEAT,
                                 habitual_threshold=HABITUAL, context=context(ver), reports=reports(ver), **basis)


@st.cache_data(show_spinner=False, max_entries=48)
def plan(ver, day, shift, vehicles, risk):
    start, end = SHIFTS[shift]
    return api.generate_plan(day, start, end, risk_weight=risk, vehicles=vehicles, context=context(ver))


@st.cache_data(show_spinner=False, max_entries=48)
def routes(ver, day, shift, vehicles, risk):
    return api.route_polylines(plan(ver, day, shift, vehicles, risk), context=context(ver))


@st.cache_data(show_spinner=False, max_entries=96)
def reception(ver, day, shift, vehicles, risk, through):
    frame = reports(ver)
    start, end = SHIFTS[shift]
    return api.classify_reports(day, REPEAT, HABITUAL, plan_df=plan(ver, day, shift, vehicles, risk),
                                through=datetime.combine(day, through), start=start, end=end, staff=vehicles,
                                context=context(ver), reports=frame.loc[frame.split.eq("test")])


def spot_point(ver, spot_id):
    catalog = context(ver).catalog
    row = catalog.loc[catalog.spot_id.eq(str(spot_id))]
    if row.empty or row.lat.isna().iloc[0] or row.lon.isna().iloc[0]:
        return None
    return float(row.lat.iloc[0]), float(row.lon.iloc[0])


def depot():
    return get_depot()


def csv(frame):
    return api.export_csv(frame)
