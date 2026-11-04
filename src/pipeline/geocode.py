"""Resume-safe Kakao geocoding. Credentials live only in request headers."""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

from .common import OUT, REF, ROOT, write_frame, write_json
from .prepare import normalize_location

CACHE_COLUMNS = ["spot_id", "label", "query", "lat", "lon", "geocode_method", "matched_address", "matched_name", "quality", "attempted_at"]


def load_key() -> str:
    """No credential value is logged or serialized, including on failure."""
    key = os.environ.get("KAKAO_REST_API_KEY", "").strip()
    if key:
        return key
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith("KAKAO_REST_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def load_boundary():
    return gpd.read_file(REF / "yuseong_boundary.geojson").geometry.union_all()


def effective_cache(cache: pd.DataFrame) -> pd.DataFrame:
    """Apply source-backed coordinate QA corrections without repeating API calls."""
    path = OUT / "geocode_overrides.csv"
    if not path.exists():
        return cache
    corrections = pd.read_csv(path, keep_default_na=False).set_index("spot_id")
    result = apply_geocode_overrides(cache, corrections)
    write_frame(result, OUT / "geocode_effective.csv")
    return result


def apply_geocode_overrides(cache: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    result = cache.copy().set_index("spot_id")
    for column in ("lat", "lon"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for sid, row in corrections.iterrows():
        if sid not in result.index:
            raise ValueError("A geocode override must refer to an attempted cache entry")
        for column in ("lat", "lon", "geocode_method", "quality"):
            result.loc[sid, column] = row[column]
    result = result.reset_index()
    return result


class KakaoGeocoder:
    def __init__(self, key: str, boundary, interval: float = .12):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": "KakaoAK " + key})
        self.boundary = boundary
        self.interval = interval

    def search(self, label: str, kind: str) -> dict:
        query = "대전광역시 유성구 " + label
        endpoint = "address" if kind == "address" else "keyword"
        params = {"query": query, "size": 10}
        if endpoint == "keyword":
            params["rect"] = ",".join(str(v) for v in self.boundary.bounds)
        response = None
        for retry in range(3):
            try:
                response = self.session.get(f"https://dapi.kakao.com/v2/local/search/{endpoint}.json", params=params, timeout=15)
            except requests.RequestException:
                if retry == 2:
                    raise RuntimeError("Kakao transport failed; no failed cache entry was committed") from None
                time.sleep(1 + retry)
                continue
            if response.status_code in (401, 403):
                raise RuntimeError("Kakao authorization rejected; verify REST key and Local API access")
            if response.status_code == 429 or response.status_code >= 500:
                if retry == 2:
                    raise RuntimeError(f"Kakao service HTTP {response.status_code}; resume from successful cache")
                time.sleep(2 + retry)
                continue
            response.raise_for_status()
            break
        if response is None:
            raise RuntimeError("Kakao produced no response")
        documents = response.json()["documents"]
        time.sleep(self.interval)
        candidates = []
        for document in documents:
            address = document.get("address_name", "")
            if not ("대전" in address and "유성구" in address):
                continue
            lon, lat = float(document["x"]), float(document["y"])
            if endpoint == "address" and normalize_location(address)[0] != label:
                continue
            candidates.append({"lat": lat, "lon": lon, "matched_address": address,
                               "matched_name": document.get("place_name", ""),
                               "quality": "ok" if self.boundary.covers(Point(lon, lat)) else "outside_boundary"})
        if not candidates:
            return {"query": query, "lat": np.nan, "lon": np.nan, "geocode_method": "failed",
                    "matched_address": "", "matched_name": "", "quality": "no_local_match"}
        # API relevance order, restricted to district; named-site matches are reviewed separately.
        chosen = candidates[0]
        return {**chosen, "query": query,
                "geocode_method": f"kakao_{endpoint}" if chosen["quality"] == "ok" else "failed"}


def geocode(spots: pd.DataFrame, limit: int | None = None, callback=None) -> pd.DataFrame:
    path = OUT / "geocode_cache.csv"
    cache = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=CACHE_COLUMNS)
    # Never request cached successes OR cached genuine no-match responses again.
    attempted = set(cache.spot_id)
    missing = spots[~spots.spot_id.isin(attempted)].sort_values(["n_reports_train", "spot_id"], ascending=[False, True])
    if limit is not None:
        missing = missing.head(limit)
    key = load_key()
    if not key or missing.empty:
        return cache
    geocoder = KakaoGeocoder(key, load_boundary())
    records = cache.to_dict("records")
    checkpoint_sent = (OUT / "geocode_first300.json").exists()
    for count, row in enumerate(missing.itertuples(index=False), start=1):
        record = {"spot_id": row.spot_id, "label": row.label,
                  **geocoder.search(row.label, row.location_type),
                  "attempted_at": datetime.now(timezone.utc).isoformat()}
        records.append(record)
        # Persist every billable response, so a crash does not repeat up to 24 calls.
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8" if exists else "utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(record)
            handle.flush()
            os.fsync(handle.fileno())
        if count % 25 == 0 or count == len(missing):
            cache = pd.DataFrame(records, columns=CACHE_COLUMNS)
            write_frame(cache, path)
            print(f"geocoded {len(cache)}/{len(spots)} locations; ok={int((cache.quality == 'ok').sum())}", flush=True)
            train_ids = set(spots.loc[spots.n_reports_train > 0, "spot_id"])
            good = set(cache.loc[cache.quality == "ok", "spot_id"])
            if callback and not checkpoint_sent and len(good & train_ids) >= 300:
                callback(cache)
                checkpoint_sent = True
    return pd.DataFrame(records, columns=CACHE_COLUMNS)


def quality_metrics(spots: pd.DataFrame, cache: pd.DataFrame) -> dict:
    accepted = spots.lat.notna() & spots.lon.notna()
    raw_valid = pd.to_numeric(cache.get("lat", pd.Series(dtype=float)), errors="coerce").notna()
    outside = int((cache.get("quality", pd.Series(dtype=str)) == "outside_boundary").sum())
    result = {"n_locations": len(spots), "n_successful_locations": int(accepted.sum()),
              "n_attempted_locations": len(cache), "n_boundary_outside_raw": outside,
              "n_pending_locations": int((spots.geocode_method == "pending").sum()),
              "n_failed_locations": int((spots.geocode_method == "failed").sum()),
              "boundary_outside_raw_fraction": outside / max(int(raw_valid.sum()), 1),
              "boundary_source": "OpenStreetMap relation 3890428, cached current polygon",
              "manual_map_review_done": False,
              "target_report_weighted_success": .95}
    for split in ("train", "test"):
        weight = spots[f"n_reports_{split}"]
        result[f"report_weighted_success_{split}"] = float(weight[accepted].sum() / max(weight.sum(), 1))
    result["report_weighted_success_all"] = float((spots.n_reports_train + spots.n_reports_test)[accepted].sum() / max((spots.n_reports_train + spots.n_reports_test).sum(), 1))
    all_weight = spots.n_reports_train + spots.n_reports_test
    result["report_weighted_failure_all"] = float(all_weight[spots.geocode_method == "failed"].sum() / max(all_weight.sum(), 1))
    result["report_weighted_pending_all"] = float(all_weight[spots.geocode_method == "pending"].sum() / max(all_weight.sum(), 1))
    review_path = OUT / "geocode_review_30.csv"
    if review_path.exists():
        review = pd.read_csv(review_path, keep_default_na=False)
        result["manual_map_review_n"] = int((review.review_status == "map_checked").sum())
        result["manual_map_review_done"] = bool(len(review) == 30 and (review.review_status == "map_checked").all())
        result["manual_review_scope"] = "무작위 좌표 30곳의 공개 도로지도상 배치·주변 도로/시설과 캐시 주소 대조; 개별 지번 경계의 측량 검증은 아님"
    write_json(result, OUT / "geocode_quality.json")
    return result


def write_review_sample(spots: pd.DataFrame, cache: pd.DataFrame) -> None:
    path = OUT / "geocode_review_30.csv"
    population = int(spots.lat.notna().sum())
    if population < 30:
        return
    if path.exists():
        previous = pd.read_csv(path, keep_default_na=False)
        if (previous.review_status != "pending").any():
            return
    sample = spots[spots.lat.notna()].sample(n=30, random_state=20261030)
    sample = sample[["spot_id", "label", "lat", "lon", "geocode_method"]].merge(
        cache[["spot_id", "matched_address", "matched_name"]], on="spot_id", how="left")
    sample["map_url"] = [f"https://map.kakao.com/link/map/{label},{lat},{lon}" for label, lat, lon in zip(sample.label, sample.lat, sample.lon)]
    sample["review_status"] = "pending"
    sample["review_note"] = ""
    sample["sample_population_size"] = population
    write_frame(sample, path)
