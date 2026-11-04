"""Descriptive processing-unit counts; not time savings or merged fines."""
from __future__ import annotations

import pandas as pd

from .common import OUT, write_frame


def processing_metrics(reports: pd.DataFrame) -> pd.DataFrame:
    train = reports[reports.split == "train"]
    test = reports[reports.split == "test"]
    frequency = train.groupby("spot_id").size()
    n = len(test)
    rows = []
    for threshold in (5, 10, 20):
        ids = set(frequency[frequency >= threshold].index)
        count = int(test.spot_id.isin(ids).sum())
        rows.append({"metric": "train_threshold_coverage", "threshold": threshold,
                     "n_spots_train": len(ids), "n_reports_test": count, "n_test_total": n,
                     "n_units_after": None, "coverage_rate": count / max(n, 1), "reduction_rate": None,
                     "definition": f"2024년 1~9월 정규화 지점별 신고 {threshold}건 이상인 지점을 고정한 후 10~12월 신고 비율",
                     "unit": "지점 수·신고 비율; 지점 등급은 학습 신고 수로만 결정"})
    for metric, keys, definition in [
        ("spot_date_groups", ["spot_id", "date"], "10~12월 신고를 정규화 지점×날짜로 묶음"),
        ("spot_date_hour_block_groups", ["spot_id", "date", "hour_block"], "10~12월 신고를 정규화 지점×날짜×hour_block으로 묶음"),
    ]:
        count = test.groupby(keys).ngroups
        rows.append({"metric": metric, "threshold": None, "n_spots_train": None,
                     "n_reports_test": n, "n_test_total": n, "n_units_after": count,
                     "coverage_rate": None, "reduction_rate": 1 - count / max(n, 1),
                     "definition": definition + "; 감소율 = 1 - 묶음 수 / 신고 건수",
                     "unit": "현장 확인·회신 단위; 차량별 과태료와 처리 시간의 감소를 뜻하지 않음"})
    return pd.DataFrame(rows)


def save_processing_metrics():
    reports = pd.read_parquet(OUT / "reports.parquet")
    frame = processing_metrics(reports)
    write_frame(frame, OUT / "processing_metrics.csv")
    print(frame.drop(columns=["definition", "unit"]).to_string(index=False))
