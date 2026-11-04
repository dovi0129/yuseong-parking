"""Normalize strings without learning from the held-out period."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from .common import END, OUT, ROOT, TEST_START, TRAIN_START, day_type, hour_block, read_csv, write_frame

PARCEL = re.compile(r"(?P<dong>[가-힣A-Za-z0-9]+(?:동|리))\s*(?P<mountain>산\s*)?(?P<main>\d+)(?:\s*[-－]\s*(?P<sub>\d+))?\s*(?:번지)?")


def normalize_location(value: str) -> tuple[str, str, str]:
    """Return normalized label, dong, and address/keyword class; never merge by proximity."""
    value = unicodedata.normalize("NFKC", str(value)).strip()
    value = re.sub(r"대전(?:광역)?시?\s*유성구\s*", "", value)
    value = re.sub(r"^유성구\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    parcel = PARCEL.match(value)
    if parcel:
        # Parcel-identical suffixes (building/front/near) share one parcel anchor.
        main = str(int(parcel["main"]))
        sub = "-" + str(int(parcel["sub"])) if parcel["sub"] else ""
        mountain = "산 " if parcel["mountain"] else ""
        return f"{parcel['dong']} {mountain}{main}{sub}", parcel["dong"], "address"
    value = re.sub(r"\s*(?:부근|주변|인근)\s*$", "", value).strip(" ,")
    value = re.sub(r"\s+", " ", value)
    dong = re.match(r"([가-힣0-9]+동)\b", value)
    return value, dong[1] if dong else "", "keyword"


def spot_id(label: str) -> str:
    return "S" + hashlib.sha256(label.encode()).hexdigest()[:16]


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_csv(ROOT / "data/raw/15154474.csv")
    reports = pd.DataFrame({
        "report_id": [f"R{i:06d}" for i in range(1, len(raw) + 1)],
        "ts": pd.to_datetime(raw["신고일시"], errors="raise"),
        "raw_loc": raw["신고위치"], "fine_status": raw["과태료부과여부"],
    })
    if not reports.ts.between(TRAIN_START, END, inclusive="left").all():
        raise ValueError("Report timestamps outside the approved 2024 population")
    reports["date"] = reports.ts.dt.normalize()
    reports["day_type"] = reports.ts.map(day_type)
    reports["hour"] = reports.ts.dt.hour
    reports["hour_block"] = reports.hour.map(hour_block)
    labels = reports.raw_loc.map(normalize_location)
    if any(not x[0] for x in labels):
        raise ValueError("Empty normalized report location")
    reports["spot_id"] = labels.map(lambda x: spot_id(x[0]))
    reports["split"] = np.where(reports.ts < TEST_START, "train", "test")
    loc = pd.DataFrame(labels.tolist(), columns=["label", "dong", "location_type"])
    loc["spot_id"] = reports.spot_id
    spots = loc.drop_duplicates("spot_id").set_index("spot_id")
    for split in ("train", "test"):
        spots[f"n_reports_{split}"] = reports[reports.split == split].groupby("spot_id").size()
        spots[f"n_reports_{split}"] = spots[f"n_reports_{split}"].fillna(0).astype(int)
    raw_enf = read_csv(ROOT / "data/raw/15125439.csv")
    enforcement = pd.DataFrame({
        "enf_id": [f"E{i:06d}" for i in range(1, len(raw_enf) + 1)],
        "ts": pd.to_datetime(raw_enf["단속일시"], errors="raise"),
        "raw_loc": raw_enf["단속장소"],
    })
    enforcement["hour_block"] = enforcement.ts.dt.hour.map(hour_block)
    normalized_enf = enforcement.raw_loc.map(lambda x: normalize_location(x)[0])
    # Exact normalized labels only. Never use nearest-neighbor fuzzy matches as facts.
    enforcement["spot_id"] = normalized_enf.map(lambda s: spot_id(s) if spot_id(s) in spots.index else None)
    spots["n_enf_2023"] = enforcement.dropna(subset="spot_id").groupby("spot_id").size()
    spots["n_enf_2023"] = spots.n_enf_2023.fillna(0).astype(int)
    spots = spots.reset_index().sort_values(["n_reports_train", "spot_id"], ascending=[False, True])
    columns = ["report_id", "ts", "date", "day_type", "hour", "hour_block", "raw_loc", "spot_id", "fine_status", "split"]
    reports = reports[columns]
    enforcement = enforcement[["enf_id", "ts", "hour_block", "raw_loc", "spot_id"]]
    write_frame(reports, OUT / "reports.parquet")
    write_frame(enforcement, OUT / "enforcement.parquet")
    write_frame(spots, OUT / "spot_locations.csv")
    return reports, enforcement, spots
