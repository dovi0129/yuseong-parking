"""Shared data contract, deterministic calendar, and atomic writers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed"
REF = OUT / "reference"
FIG = ROOT / "artifacts/figures"
TRAIN_START = pd.Timestamp("2024-01-01")
INNER_END = pd.Timestamp("2024-08-01")
TEST_START = pd.Timestamp("2024-10-01")
END = pd.Timestamp("2025-01-01")
BLOCKS = ("00-07", "07-10", "10-14", "14-17", "17-21", "21-24")
DAY_TYPES = ("평일", "주말")
HOLIDAYS = frozenset(
    # KASI 2024 calendar; 2024-10-01 temporary holiday: MPM announcement.
    "2024-01-01 2024-02-09 2024-02-10 2024-02-11 2024-02-12 "
    "2024-03-01 2024-04-10 2024-05-05 2024-05-06 2024-05-15 "
    "2024-06-06 2024-08-15 2024-09-16 2024-09-17 2024-09-18 "
    "2024-10-01 2024-10-03 2024-10-09 2024-12-25".split()
)


def day_type(ts: pd.Timestamp) -> str:
    return "주말" if ts.weekday() >= 5 or ts.strftime("%Y-%m-%d") in HOLIDAYS else "평일"


def hour_block(hour: int) -> str:
    if not 0 <= hour < 24:
        raise ValueError("hour must be in [0, 24)")
    return next(b for b in BLOCKS if int(b[:2]) <= hour < int(b[3:]))


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported encoding: {path.name}")


def write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    if path.suffix == ".parquet":
        df.to_parquet(temp, index=False)
    else:
        df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, inclusive="left", freq="D")
    return pd.DataFrame({"date": dates, "day_type": [day_type(d) for d in dates]})
