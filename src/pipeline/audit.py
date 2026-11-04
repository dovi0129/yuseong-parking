"""Check final artifact contracts and figures without rerunning final evaluation."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .common import FIG, OUT, ROOT, write_json
from .features import SPOT_COLUMNS
from .geocode import load_key


def main():
    reports = pd.read_parquet(OUT / "reports.parquet")
    enforcement = pd.read_parquet(OUT / "enforcement.parquet")
    spots = pd.read_csv(OUT / "spots.csv")
    scores = pd.read_csv(OUT / "priority_train.csv")
    choice = json.loads((OUT / "priority_choice.json").read_text())
    checks = []

    def check(condition, description):
        if not bool(condition):
            raise AssertionError(description)
        checks.append(description)

    check(list(reports.columns) == ["report_id", "ts", "date", "day_type", "hour", "hour_block", "raw_loc", "spot_id", "fine_status", "split"], "reports.parquet contract columns")
    check(list(enforcement.columns) == ["enf_id", "ts", "hour_block", "raw_loc", "spot_id"], "enforcement.parquet contract columns")
    check(list(spots.columns) == SPOT_COLUMNS, "spots.csv amended contract columns")
    check(list(scores.columns) == ["spot_id", "day_type", "hour_block", "score", "method"], "priority_train.csv contract columns")
    check(len(reports) == 23508 and len(enforcement) == 121091, "approved record counts preserved")
    check(reports.report_id.is_unique and spots.spot_id.is_unique and enforcement.enf_id.is_unique, "stable record/spot identifiers unique")
    train = reports[reports.split == "train"]
    test = reports[reports.split == "test"]
    check(train.ts.max() < pd.Timestamp("2024-10-01") <= test.ts.min(), "chronological final split")
    check(set(scores.spot_id) == set(train.spot_id), "priority candidate universe is training only")
    check(set(scores.method) == {"freq_baseline", "shrinkage"}, "both methods retained")
    check(not scores[["spot_id", "day_type", "hour_block", "method"]].duplicated().any(), "score keys unique")
    check(np.isfinite(scores.score).all() and scores.score.ge(0).all(), "expected counts finite and nonnegative")
    check(choice["chosen_method"] in set(scores.method) and choice["method_selection_uses_test"] is False, "method chosen before final test")
    for split in ("train", "test"):
        expected = reports[reports.split == split].groupby("spot_id").size()
        actual = spots.set_index("spot_id")[f"n_reports_{split}"]
        check(actual.sum() == len(reports[reports.split == split]), f"{split} spot counts conserve reports")
        np.testing.assert_array_equal(actual.reindex(expected.index), expected)
    valid = spots.lat.notna() & spots.lon.notna()
    check(spots.loc[valid, "child_zone_300m"].isin([0, 1]).all(), "valid coordinate risk proxy is binary")
    check(spots.risk_weight.between(1, 2.5).all(), "fixed risk weights in [1,2.5]")
    check(set(spots.geocode_method).issubset({"kakao_address", "kakao_keyword", "manual", "failed", "pending"}), "geocode lifecycle values valid")
    check((spots.geocode_method == "pending").sum() == 0, "no unprocessed geocode locations")
    quality = json.loads((OUT / "geocode_quality.json").read_text())
    check(quality["report_weighted_success_all"] >= .95 and quality["n_boundary_outside_raw"] == 0, "final weighted geocode target and boundary checks")
    check(quality["manual_map_review_done"] and quality["manual_map_review_n"] == 30, "random 30-point street-map review recorded")
    final = pd.read_csv(OUT / "evaluation_priority.csv")
    check(final.total.eq(len(test)).all() and final.n_calendar_days.eq(92).all(), "all held-out reports and all calendar days remain in evaluation")
    repeat = pd.read_csv(OUT / "analysis_tables/same_day_repetition.csv")
    check(repeat.n_reports.sum() == len(reports), "repetition plot conserves source reports")
    concentration = pd.read_csv(OUT / "analysis_tables/concentration.csv")
    for _, frame in concentration.groupby("series"):
        check(np.isclose(frame.cumulative_fraction.iloc[-1], 1), "concentration curve ends at 100 percent")
    processing = pd.read_csv(OUT / "processing_metrics.csv")
    check(len(processing) == 5 and processing.n_test_total.eq(len(test)).all(), "processing metrics include three training thresholds and two grouping units")
    figures = []
    for name in ["concentration", "weekday_hour", "same_day_repetition", "repeat_map", "share_difference_map", "priority_capture"]:
        path = FIG / f"analysis_{name}.png"
        with Image.open(path) as image:
            size = image.size
            dpi = image.info.get("dpi", (0, 0))
            image.verify()
        check(size[0] > 500 and size[1] > 500 and min(dpi) >= 199.8, f"{path.name}: valid PNG at brief-required 200dpi")
        figures.append({"file": str(path.relative_to(ROOT)), "size_px": list(size), "dpi": list(dpi)})
    key = load_key()
    owned = [ROOT / "src/pipeline", OUT]
    leaks = 0
    if key:
        for directory in owned:
            for path in directory.rglob("*"):
                if directory == OUT and "routes" in path.relative_to(OUT).parts:
                    continue
                if path.is_file() and path.suffix in {".py", ".csv", ".json", ".log", ".md"}:
                    leaks += key.encode() in path.read_bytes()
    check(leaks == 0, "credential value absent from owned code, logs, caches, and text artifacts")
    for path in (ROOT / "src/pipeline").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    checks.append("all pipeline Python modules parse")
    write_json({"checks": checks, "n_checks": len(checks), "figures": figures,
                "png_audit_scope": "User brief explicitly requires 200dpi, overriding the figure-audit default of 300dpi. Overlap/glyph readability inspected visually; raster output has no embedded font-size measurements.",
                "statistical_validation": "17 known-answer tests incl. no-final-period leakage, paired null arm, unseen-location denominator, EPSG axis order, processing groups, cache override dtype; captured totals also cross-checked by key joins",
                "reused": "approved raw record counts; frozen priority choice and final evaluation; geographic source caches"}, OUT / "pipeline_checks.json")
    print(f"PASS: {len(checks)} final artifact checks; six analysis PNGs; no key value found in owned text artifacts")


if __name__ == "__main__":
    main()
