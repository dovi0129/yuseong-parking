"""Reproducible stages. Run: .venv/bin/python -m src.pipeline.run --only all"""
from __future__ import annotations

import argparse
import json
import subprocess

import pandas as pd

from .common import OUT, ROOT, write_json
from .features import build_spots
from .geocode import CACHE_COLUMNS, geocode, load_key
from .prepare import prepare
from .priority import evaluate_once, train_priority
from .reference import fetch_all


def checkpoint(locations, cache):
    spots = build_spots(locations, cache)
    ready = int(((spots.n_reports_train > 0) & spots.lat.notna()).sum())
    path = ROOT / "data/processed/spots.csv"
    chosen = json.loads((OUT / "priority_choice.json").read_text())["chosen_method"]
    message = f"[Worker_2] spots 1차판 준비: {path} / 학습 빈도순 상위권 {ready}개 좌표 성공(실패 지점 제외), priority_choice={chosen} 고정, 전체 지점 계속 처리 중. 경로 후보는 n_reports_train>0 및 유효 좌표만 사용."
    for target in ("wC:p4", "wC:p1"):
        result = subprocess.run(["herdr", "agent", "prompt", target, message], capture_output=True, text=True, timeout=20)
        print(f"checkpoint notification {target}: returncode={result.returncode}", flush=True)
    write_json({"geocoded_train_spots": ready, "spots_path": str(path), "selection": "학습 빈도순; 좌표 실패 제외; 테스트 빈도 미사용"}, OUT / "geocode_first300.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["prepare", "references", "geocode", "features", "evaluate", "processing", "figures", "all"], default="all")
    parser.add_argument("--geocode-limit", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only in ("prepare", "all"):
        reports, _, locations = prepare()
        if not (OUT / "priority_choice.json").exists():
            train_priority(reports)
        print(f"prepared reports={len(reports)}, locations={len(locations)}", flush=True)
    if args.only in ("references", "all"):
        fetch_all()
    if args.only in ("geocode", "all"):
        if not load_key():
            print("KAKAO_KEY_MISSING: no request sent; run geocode after key is available", flush=True)
        locations = pd.read_csv(OUT / "spot_locations.csv", keep_default_na=False)
        cache = geocode(locations, limit=args.geocode_limit, callback=lambda c: checkpoint(locations, c))
        build_spots(locations, cache)
    if args.only == "features":
        locations = pd.read_csv(OUT / "spot_locations.csv", keep_default_na=False)
        path = OUT / "geocode_cache.csv"
        cache = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=CACHE_COLUMNS)
        build_spots(locations, cache)
    if args.only in ("evaluate", "all"):
        reports = pd.read_parquet(OUT / "reports.parquet")
        scores = pd.read_csv(OUT / "priority_train.csv")
        summary = evaluate_once(reports, scores)
        print(summary.to_string(index=False), flush=True)
    if args.only in ("processing", "all"):
        from .processing import save_processing_metrics
        save_processing_metrics()
    if args.only in ("figures", "all"):
        from .figures import render_all
        render_all()
        print("analysis figures saved", flush=True)
    if args.only == "all":
        quality = json.loads((OUT / "geocode_quality.json").read_text())
        write_json({"execution_finished": True, "geocode_key_present": bool(load_key()),
                    "artifacts_complete": quality["n_pending_locations"] == 0 and quality["report_weighted_success_all"] >= .95 and quality["manual_map_review_done"],
                    "manual_review_required": not quality["manual_map_review_done"]}, OUT / "pipeline_run_finished.json")


if __name__ == "__main__":
    main()
