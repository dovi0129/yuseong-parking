"""Monthly plans: a date in month m uses a plan made from reports up to the last day of month m-1.

Each month uses the same scoring (fit_scores: shrinkage with the fixed alpha), the same candidate rule
(load_planning_data: coordinates and at least one report before the month) and the same solver; only the input
window changes.

Per month a folder plan_cache/monthly/<YYYYMM>/ holds spots.csv, priority_train.csv and priority_choice.json; plans
for every screen combination are kept in the usual plan cache (the cache key hashes those inputs).
October equals the original Jan~Sep inputs, so its plans are the existing ones.

Run (precompute): python -m src.optimizer.monthly --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import pandas as pd

from .api import _minute, _plan_cache_path, cache_existing_plan, default_planner_method, load_context, plan_for_day_type
from .domain import ROOT, ROUTES, Plan, PlanConfig

DATA = ROOT / "data/processed"
FOLDER = ROUTES / "plan_cache/monthly"
BASIS_START = pd.Timestamp("2024-01-01")
MONTHS = ("2024-10", "2024-11", "2024-12")          # months the screens can choose (EVAL_START..EVAL_END)
SHIFTS = (("07:00", "10:00"), ("10:00", "14:00"), ("14:00", "17:00"), ("17:00", "21:00"))
ROLLING_PLANS = DATA / "added/add2_plans.json"     # optional: monthly plans solved earlier (17~21시 평일, 1·2대, 위험 가중 적용)
VERSION = "rolling-m3-v1"


def month_of(day) -> str:
    return pd.Timestamp(day).strftime("%Y-%m")


def basis(month: str):
    """Input window [BASIS_START, month start) as (start, last day inclusive, exclusive end)."""
    end = pd.Timestamp(month + "-01")
    return BASIS_START, end - pd.Timedelta(days=1), end


def folder(month: str):
    return FOLDER / month.replace("-", "")


def read_reports():
    reports = pd.read_parquet(DATA / "reports.parquet")
    reports["ts"] = pd.to_datetime(reports.ts)
    return reports


def build_inputs(month: str, reports: pd.DataFrame):
    """spots/priority/choice for a month from reports before the month only (original fit_scores)."""
    from src.pipeline.priority import fit_scores
    start, last, end = basis(month)
    window = reports[(reports.ts >= start) & (reports.ts < end)]
    choice = json.loads((DATA / "priority_choice.json").read_text())
    scores = fit_scores(reports, start, end, choice["alpha"])
    spots = pd.read_csv(DATA / "spots.csv", dtype={"spot_id": str})
    spots = spots.drop(columns=[c for c in ("n_reports_test",) if c in spots])
    spots["n_reports_train"] = spots.spot_id.map(window.groupby(window.spot_id.astype(str)).size()).fillna(0).astype(int)
    meta = {**choice, "final_fit_end_exclusive": str(end.date()), "basis_start": str(start.date()),
            "basis_last_day": str(last.date()), "input_max_ts": str(window.ts.max()), "input_reports": int(len(window)),
            "monthly_version": VERSION}
    return spots, scores, meta


def prepare(month: str, reports: pd.DataFrame | None = None):
    """Write the month's input folder once; an existing folder is left as it is (its inputs are checked in tests)."""
    target = folder(month)
    if (target / "priority_choice.json").exists():
        return target
    spots, scores, meta = build_inputs(month, read_reports() if reports is None else reports)
    target.mkdir(parents=True, exist_ok=True)
    spots.to_csv(target / "spots.csv", index=False)
    scores.to_csv(target / "priority_train.csv", index=False)
    (target / "priority_choice.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return target


def context(month: str, **kwargs):
    target = folder(month)
    if not (target / "priority_choice.json").exists():
        raise FileNotFoundError(f"{month} 월별 계획 입력이 없습니다: python -m src.optimizer.monthly")
    return load_context(directory=target, **kwargs)


def basis_label(month: str):
    start, last, _ = basis(month)
    return f"{start:%Y-%m-%d} ~ {last:%Y-%m-%d}"


def combinations():
    return [(dt, s, e, k, r) for s, e in SHIFTS for dt in ("평일", "주말") for k in (1, 2, 3) for r in (True, False)]


def cache_path(ctx, combo):
    dt, s, e, k, r = combo
    config = PlanConfig(_minute(s), _minute(e), k, 5.0, dt, r, 30.0)
    return _plan_cache_path(ctx, config, default_planner_method(), ctx.data.chosen_method)[0]


def seed_rolling(month, ctx):
    """Store plans of this month solved earlier under the screen's cache key when the inputs and settings are the same."""
    if not ROLLING_PLANS.exists():
        return []
    plans = json.loads(ROLLING_PLANS.read_text())
    stored = []
    for vehicles in (1, 2):
        obj = plans.get(f"{month}|{vehicles}|M3")
        if obj is None:
            continue
        plan = Plan.from_dict(obj)
        expected = PlanConfig(17 * 60, 21 * 60, vehicles, 5.0, "평일", True, 30.0)
        if asdict(plan.config) != asdict(expected) or plan.method != default_planner_method() \
                or plan.score_method != ctx.data.chosen_method:
            continue
        ids = {v.spot_id for v in plan.visits}
        if not ids <= set(ctx.data.spots.spot_id):
            raise ValueError(f"{month} 미리 푼 계획의 방문 지점이 후보에 없습니다.")
        path = cache_path(ctx, ("평일", "17:00", "21:00", vehicles, True))
        if not path.exists():
            from .planner import validate_plan
            validate_plan(plan, ctx.network)
            cache_existing_plan(plan, context=ctx)
            stored.append(str(path.relative_to(ROOT)))
    return stored


CONTEXTS = {}


def _solve(task):
    month, combo = task
    started = time.monotonic()
    plan = plan_for_day_type(*combo, context=CONTEXTS[month])
    if plan.status != "ok":
        raise RuntimeError(f"풀이 실패 {task}")
    return {"month": month, "day_type": combo[0], "start": combo[1], "end": combo[2], "vehicles": combo[3],
            "risk_weight": combo[4], "seconds": round(time.monotonic() - started, 1), "visits": len(plan.visits),
            "expected_weighted": plan.expected_weighted, "incumbent_source": plan.incumbent_source,
            "path": str(cache_path(CONTEXTS[month], combo).relative_to(ROOT))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    reports = read_reports()
    status = {"version": VERSION, "months": {}}
    tasks = []
    for month in MONTHS:
        prepare(month, reports)
        ctx = context(month, with_network=True)   # builds (or loads) the travel matrix once, before workers fork
        CONTEXTS[month] = ctx
        seeded = [] if args.count_only else seed_rolling(month, ctx)
        missing = [c for c in combinations() if not cache_path(ctx, c).exists()]
        status["months"][month] = {"basis": basis_label(month), "candidates": len(ctx.data.spots),
                                   "cached_before": len(combinations()) - len(missing), "seeded_from_add2": seeded,
                                   "to_solve": len(missing)}
        tasks += [(month, c) for c in missing]
        print(month, json.dumps(status["months"][month], ensure_ascii=False), flush=True)
    print("plans to solve:", len(tasks), flush=True)
    if args.count_only:
        return
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(_solve, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            print(json.dumps({"done": len(rows), "total": len(tasks), **rows[-1]}, ensure_ascii=False), flush=True)
    for month in MONTHS:
        ctx = CONTEXTS[month]
        status["months"][month]["cached_after"] = sum(cache_path(ctx, c).exists() for c in combinations())
    status["solved"] = len(rows)
    status["elapsed_s"] = round(time.monotonic() - started, 1)
    FOLDER.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(FOLDER / "solved_manifest.csv", index=False)
    (FOLDER / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=1))
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
