"""Freeze training-only plans before loading held-out reports; paired-day CIs."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .domain import BLOCKS, ROOT, ROUTES, Plan, PlanConfig, day_type, load_planning_data
from .network import build_matrix
from .planner import solve


def captured_reports(plan: Plan, daily: pd.DataFrame, risk_by_spot: dict):
    """Count each held-out record once by (spot_id, arrival hour_block).

    A visit matches its entire coarse hour_block, not a five-minute detection
    window. This is a retrospective coverage metric, not avoided violations.
    """
    keys = {(v.spot_id, v.block) for v in plan.visits}
    mask = [(str(s), str(b)) in keys for s, b in zip(daily.spot_id, daily.hour_block)]
    caught = daily.loc[mask]
    if caught.empty:
        return 0, 0.
    weights = pd.to_numeric(caught.spot_id.astype(str).map(risk_by_spot), errors="raise").fillna(1.)
    return len(caught), float(weights.sum())


def summarize(daily: pd.DataFrame, repetitions=2000):
    rng = np.random.default_rng(20241001)
    metric_columns = ["caught", "caught_per_person_hour", "risk_caught", "travel_min"]
    summaries, paired = [], []
    for scenario, frame in daily.groupby("scenario", sort=True):
        for method, subset in frame.groupby("method", sort=True):
            for metric in metric_columns:
                values = subset.sort_values("date")[metric].to_numpy(float)
                boot = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                summaries.append(dict(scenario=scenario, method=method, metric=metric,
                    mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi), days=len(values)))
        for comparison in ("b1", "b2"):
            for metric in metric_columns:
                pivot = frame.pivot(index="date", columns="method", values=metric).sort_index()
                if pivot[["optimized", comparison]].isna().any().any():
                    raise ValueError("방식 사이의 평가 날짜가 맞지 않습니다.")
                delta = (pivot.optimized-pivot[comparison]).to_numpy(float)
                boot = delta[rng.integers(len(delta), size=(repetitions, len(delta)))].mean(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                paired.append(dict(scenario=scenario, baseline=comparison, metric=metric,
                    mean_delta=float(delta.mean()), ci_low=float(lo), ci_high=float(hi), days=len(delta)))
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def evaluate(*, mock=False, max_spots=None, solver_seconds=30., weekdays_only=True):
    directory = ROUTES if mock else ROOT / "data/processed"
    prefix = "mock_" if mock else ""
    data = load_planning_data(directory, mock=mock, max_spots=max_spots)
    network = build_matrix(data, mock=mock)
    plans = {}
    scenarios = []
    for begin, end in ((10*60, 14*60), (17*60, 21*60)):
        for vehicles in (1, 2):
            scenarios.append((f"{begin//60:02}-{end//60:02}_K{vehicles}", begin, end, vehicles))
    day_types = ("평일",) if weekdays_only else ("평일", "주말")
    # All four planning scenarios are fixed before opening the test parquet.
    for name, begin, end, vehicles in scenarios:
        for dt in day_types:
            for method in ("b1", "b2", "optimized"):
                key = f"{name}_{dt}_{method}"
                config = PlanConfig(begin, end, vehicles, 5., dt, True, solver_seconds)
                plan = solve(data, network, config, method)
                if plan.status != "ok":
                    raise RuntimeError(f"최적화 실패: {key}; 평가 결과를 내지 않습니다.")
                plans[key] = plan.to_dict()
                print(key, "visits", len(plan.visits), "seconds", round(plan.elapsed_s, 3), flush=True)
    snapshot = dict(source=data.source, chosen_method=data.chosen_method,
        candidate_spots=data.spots.spot_id.tolist(), excluded_spots=data.excluded_spots,
        planning_spots=data.spots.to_dict(orient="records"),
        planning_priority=data.priority.to_dict(orient="records"),
        network=network.provenance, plans=plans,
        evaluation_definition="unique report records matching visited spot and arrival hour_block",
        work_hour_denominator="vehicles * shift duration in hours",
        candidate_definition="all valid geocoded training spots" if max_spots is None else f"top {max_spots} valid geocoded training spots",
        weekdays_only=weekdays_only, holiday_calendar_year=2024)
    (ROUTES / f"{prefix}plans_frozen.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    # The guard uses only trained scores and a calendar, before opening test data.
    gate_rows=[]
    for calendar_date in pd.date_range("2024-10-01","2024-12-31",freq="D"):
        dt=day_type(calendar_date)
        if dt not in day_types:continue
        for name,_,_,_ in scenarios:
            plan=Plan.from_dict(plans[f"{name}_{dt}_optimized"])
            passed=plan.initial_assignment_used and plan.expected_weighted+1e-10>=plan.seed_expected_weighted
            if not passed:raise AssertionError("학습 목적함수 b3 하한 검사 실패")
            gate_rows.append({"date":calendar_date.date().isoformat(),"scenario":name,
                "optimized_expected_weighted":plan.expected_weighted,"b3_expected_weighted":plan.seed_expected_weighted,
                "optimized_expected":plan.expected_reports,"b3_expected":plan.seed_expected_reports,"passed":passed,
                "incumbent_source":plan.incumbent_source})
    pd.DataFrame(gate_rows).to_csv(ROUTES/f"{prefix}training_objective_gate.csv",index=False)
    reports = pd.read_parquet(directory / f"{prefix}reports.parquet")
    reports["ts"] = pd.to_datetime(reports.ts)
    test = reports.loc[reports.split.eq("test")].copy()
    if test.empty or not test.ts.between("2024-10-01", "2025-01-01", inclusive="left").all():
        raise ValueError("평가 split은 2024-10-01 이상 2025-01-01 미만이어야 합니다.")
    if (reports.loc[reports.split.eq("train"), "ts"] >= pd.Timestamp("2024-10-01")).any():
        raise ValueError("학습 split에 평가 시점 기록이 있습니다.")
    test["date"] = test.ts.dt.date.astype(str)
    test["spot_id"] = test.spot_id.astype(str)
    for lo, hi, block in BLOCKS:
        at_block = test.ts.dt.hour.between(lo, hi-1)
        if not test.loc[at_block, "hour_block"].eq(block).all():
            raise ValueError("신고 시각과 hour_block의 정렬이 맞지 않습니다.")
    if not test.day_type.eq(test.ts.map(day_type)).all():
        raise ValueError("평가 기록의 요일 구분과 공휴일 목록이 맞지 않습니다.")
    risk = dict(zip(data.spots.spot_id, data.spots.risk_weight))
    rows = []
    # Include days with zero reports: derive dates from the calendar, not groups.
    calendar = pd.date_range("2024-10-01", "2024-12-31", freq="D")
    if mock:
        calendar = pd.DatetimeIndex([pd.Timestamp("2024-10-02")])
    for date_value in calendar:
        dt = day_type(date_value)
        if dt not in day_types:
            continue
        date_string = date_value.date().isoformat()
        daily = test.loc[test.date.eq(date_string)]
        for scenario, begin, end, vehicles in scenarios:
            available = daily.loc[daily.ts.dt.hour.between(begin//60, end//60-1)]
            geocoded = available.spot_id.isin(data.spots.spot_id)
            for method in ("b1", "b2", "optimized"):
                plan = Plan.from_dict(plans[f"{scenario}_{dt}_{method}"])
                count, weighted = captured_reports(plan, daily, risk)
                person_hours = vehicles*(end-begin)/60
                rows.append(dict(date=date_string, day_type=dt, scenario=scenario, method=method,
                    caught=count, caught_per_person_hour=count/person_hours, risk_caught=weighted,
                    travel_min=plan.travel_s/60, expected=plan.expected_reports,expected_weighted=plan.expected_weighted,
                    visits=len(plan.visits), shift_reports=len(available), candidate_reports=int(geocoded.sum()),
                    plan_seconds=plan.elapsed_s))
    daily = pd.DataFrame(rows)
    summary, paired = summarize(daily)
    daily.to_csv(ROUTES / f"{prefix}evaluation_daily.csv", index=False)
    summary.to_csv(ROUTES / f"{prefix}evaluation_summary.csv", index=False)
    paired.to_csv(ROUTES / f"{prefix}evaluation_paired.csv", index=False)
    print(paired[paired.metric.eq("caught")].to_string(index=False))
    return snapshot, daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-spots", type=int, default=0, help="0: all training spots with valid coordinates")
    parser.add_argument("--solver-seconds", type=float, default=30.)
    parser.add_argument("--include-weekends", action="store_true")
    args = parser.parse_args()
    evaluate(mock=args.mock, max_spots=args.max_spots or None, solver_seconds=args.solver_seconds,
             weekdays_only=not args.include_weekends)


if __name__ == "__main__":
    main()
