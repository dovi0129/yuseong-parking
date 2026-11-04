"""Train-only method selection and paired calendar-day bootstrap coverage."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import BLOCKS, DAY_TYPES, END, INNER_END, OUT, TEST_START, TRAIN_START, calendar, write_frame, write_json

KS = (20, 50, 100)
ALPHAS = (5.0, 20.0, 50.0)


def fit_scores(reports: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, alpha: float = 20) -> pd.DataFrame:
    """Expected reports per calendar day in a day_type/hour_block, including zero days.

    freq_baseline: n(s,d,b)/days(d).
    shrinkage: n(s,d)/days(d) * [n(s,d,b)+alpha*p(d,b)]/[n(s,d)+alpha].
    All spot identities and p(d,b) are fitted exclusively on timestamps in [start,end).
    """
    train = reports[(reports.ts >= start) & (reports.ts < end)]
    ids = sorted(train.spot_id.unique())
    index = pd.MultiIndex.from_product([ids, DAY_TYPES, BLOCKS], names=["spot_id", "day_type", "hour_block"])
    grid = train.groupby(["spot_id", "day_type", "hour_block"]).size().reindex(index, fill_value=0).rename("n").reset_index()
    exposures = calendar(start, end).groupby("day_type").size()
    grid["days"] = grid.day_type.map(exposures)
    total = grid.groupby(["spot_id", "day_type"])["n"].transform("sum")
    global_counts = grid.groupby(["day_type", "hour_block"])["n"].sum()
    global_days = grid.groupby("day_type")["n"].sum()
    prob = global_counts / global_counts.index.get_level_values("day_type").map(global_days)
    grid["p"] = [prob.get((d, b), 0.0) for d, b in zip(grid.day_type, grid.hour_block)]
    base = grid[["spot_id", "day_type", "hour_block"]].copy()
    base["score"] = grid.n / grid.days
    base["method"] = "freq_baseline"
    shrink = base.copy()
    shrink["score"] = (total / grid.days) * ((grid.n + alpha * grid.p) / (total + alpha))
    shrink["method"] = "shrinkage"
    return pd.concat([base, shrink], ignore_index=True)


def daily_capture(scores: pd.DataFrame, observations: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Numerator: reports in selected top K for their own day_type/hour_block.

    Denominator: ALL reports on each calendar day, including unseen and ungeocoded spots.
    Ties use spot_id ascending and zero-score entries are not selected.
    """
    obs = observations[(observations.ts >= start) & (observations.ts < end)]
    days = calendar(start, end)["date"]
    totals = obs.groupby("date").size().reindex(days, fill_value=0)
    rows = []
    for method, frame in scores.groupby("method", sort=True):
        ranked = frame[frame.score > 0].sort_values(["score", "spot_id"], ascending=[False, True])
        for k in KS:
            chosen = ranked.groupby(["day_type", "hour_block"], sort=False).head(k)
            selected = set(zip(chosen.spot_id, chosen.day_type, chosen.hour_block))
            mask = [(s, d, b) in selected for s, d, b in zip(obs.spot_id, obs.day_type, obs.hour_block)]
            hits = obs.loc[mask].groupby("date").size().reindex(days, fill_value=0)
            for date, numerator, denominator in zip(days, hits, totals):
                rows.append({"date": date, "method": method, "K": k, "captured": int(numerator), "total": int(denominator)})
    return pd.DataFrame(rows)


def capture_summary(daily: pd.DataFrame, bootstrap: int = 2000, seed: int = 20261030) -> pd.DataFrame:
    """Ratio-of-sums coverage; paired resamples of whole days for every method and K."""
    days = sorted(daily.date.unique())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(days), size=(bootstrap, len(days)))
    results = []
    base_draws = {}
    for k in KS:
        base = daily[(daily.method == "freq_baseline") & (daily.K == k)].set_index("date").reindex(days)
        base_draws[k] = base.captured.to_numpy()[indices].sum(axis=1) / np.maximum(base.total.to_numpy()[indices].sum(axis=1), 1)
    for (method, k), rows in daily.groupby(["method", "K"]):
        rows = rows.set_index("date").reindex(days)
        numerator, denominator = rows.captured.to_numpy(), rows.total.to_numpy()
        draws = numerator[indices].sum(axis=1) / np.maximum(denominator[indices].sum(axis=1), 1)
        base = daily[(daily.method == "freq_baseline") & (daily.K == k)]
        coverage = numerator.sum() / max(denominator.sum(), 1)
        delta = draws - base_draws[k]
        results.append({"method": method, "K": int(k), "captured": int(numerator.sum()),
                        "total": int(denominator.sum()), "capture_rate": float(coverage),
                        "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                        "delta_vs_baseline": float(coverage - base.captured.sum() / max(base.total.sum(), 1)),
                        "delta_ci_low": float(np.quantile(delta, .025)), "delta_ci_high": float(np.quantile(delta, .975)),
                        "n_calendar_days": len(days), "bootstrap_replicates": bootstrap})
    return pd.DataFrame(results)


def train_priority(reports: pd.DataFrame) -> pd.DataFrame:
    """Choose on Jan-Jul -> Aug-Sep; refit Jan-Sep; never inspect Oct-Dec here."""
    train = reports[reports.ts < TEST_START].copy()
    evaluations = []
    for alpha in ALPHAS:
        score = fit_scores(train, TRAIN_START, INNER_END, alpha)
        daily = daily_capture(score, train, INNER_END, TEST_START)
        sums = daily.groupby(["method", "K"])[["captured", "total"]].sum()
        rates = (sums.captured / sums.total.clip(lower=1)).groupby("method").mean()
        evaluations.append({"alpha": alpha, "freq_baseline": float(rates["freq_baseline"]), "shrinkage": float(rates["shrinkage"])})
    best = max(evaluations, key=lambda r: (r["shrinkage"], -r["alpha"]))
    chosen = "shrinkage" if best["shrinkage"] > best["freq_baseline"] else "freq_baseline"
    scores = fit_scores(train, TRAIN_START, TEST_START, best["alpha"])
    write_frame(scores, OUT / "priority_train.csv")
    write_json({"chosen_method": chosen,
                "선택 근거": "1~7월 적합, 8~9월의 K=20/50/100 전체 구간 신고 포착률 산술평균으로 선택. 동률은 빈도 기준선. 10~12월 미사용.",
                "alpha": best["alpha"], "inner_validation": evaluations,
                "fit_start": "2024-01-01", "inner_fit_end_exclusive": "2024-08-01",
                "validation_end_exclusive": "2024-10-01", "final_fit_end_exclusive": "2024-10-01",
                "candidate_rule": "학습 기간 신고 1건 이상; 모든 학습 지점으로 점수 적합; 좌표 실패 지점은 경로에서 제외",
                "score_unit": "해당 day_type의 하루 중 해당 hour_block의 기대 신고 수",
                "normalization": "고정 지번 문자열 규칙; 좌표 근접 병합 없음", "method_selection_uses_test": False}, OUT / "priority_choice.json")
    return scores


def evaluate_once(reports: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Persist final test result once; repeat render calls reuse it rather than retune."""
    path = OUT / "evaluation_priority.csv"
    if path.exists():
        return pd.read_csv(path)
    if not (OUT / "priority_choice.json").exists():
        raise RuntimeError("Freeze priority_choice.json before final evaluation")
    daily = daily_capture(scores, reports, TEST_START, END)
    summary = capture_summary(daily)
    write_frame(daily, OUT / "evaluation_priority_daily.csv")
    write_frame(summary, path)
    choice = json.loads((OUT / "priority_choice.json").read_text())
    write_json({"period": "2024-10-01~2024-12-31", "chosen_method": choice["chosen_method"],
                "alpha": choice["alpha"], "denominator": "전체 평가 기간 신고; 새 지점·좌표 실패 포함",
                "bootstrap": "92개 달력일을 짝지어 2,000회 복원추출; 95% percentile CI",
                "interpretation": "선정된 지점·구간에서 발생한 과태료 처리 신고의 사후 포착률; 현장 조치 효과가 아님"}, OUT / "evaluation_protocol.json")
    return summary
