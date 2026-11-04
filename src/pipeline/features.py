"""Metric-distance auxiliary features and an explicit, bounded policy weight."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point

from .common import OUT, REF, ROOT, read_csv, write_frame, write_json
from .geocode import effective_cache, load_boundary, quality_metrics, write_review_sample

SPOT_COLUMNS = ["spot_id", "label", "dong", "lat", "lon", "geocode_method", "n_reports_train", "n_reports_test",
                "n_enf_2023", "dist_school_m", "child_zone_300m", "accidents_200m", "risk_weight"]


def xy(lon, lat):
    transformer = Transformer.from_crs(4326, 5179, always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return np.column_stack([x, y])


def accident_coordinates() -> pd.DataFrame:
    path = OUT / "accidents.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = read_csv(ROOT / "data/raw/15108925.csv")
    transformer = Transformer.from_crs(5179, 4326, always_xy=True)
    # Despite the column labels, 위도 is easting (~984k) and 경도 northing (~1819k).
    lon, lat = transformer.transform(pd.to_numeric(raw["위도"]), pd.to_numeric(raw["경도"]))
    boundary = load_boundary()
    accidents = pd.DataFrame({"ts": pd.to_datetime(raw["발생일시"]), "lon": lon, "lat": lat,
                              "casualties": pd.to_numeric(raw["사상자수"]), "accident_type": raw["사고유형"]})
    accidents["in_yuseong"] = [boundary.covers(Point(x, y)) for x, y in zip(lon, lat)]
    write_frame(accidents, path)
    write_json({"source_crs": "EPSG:5179", "target_crs": "EPSG:4326", "always_xy": True,
                "input_x_column": "위도", "input_y_column": "경도", "n_records": len(accidents),
                "inside_current_osm_boundary": int(accidents.in_yuseong.sum()),
                "outside_current_osm_boundary": int((~accidents.in_yuseong).sum()),
                "outside_handling": "안팎 여부를 보존; 위험 가중치 사고 수에는 유성구 안 자료만 사용",
                "date_range": [str(accidents.ts.min()), str(accidents.ts.max())]}, OUT / "accident_coordinate_quality.json")
    return accidents


def build_spots(locations: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    cache = effective_cache(cache)
    columns = ["spot_id", "lat", "lon", "geocode_method"]
    good = cache[cache.quality == "ok"][columns].copy() if len(cache) else pd.DataFrame(columns=columns)
    for col in ("lat", "lon"):
        good[col] = pd.to_numeric(good[col], errors="coerce")
    spots = locations.merge(good, on="spot_id", how="left", validate="one_to_one")
    attempted_methods = cache.set_index("spot_id").geocode_method.to_dict() if len(cache) else {}
    spots["geocode_method"] = spots.spot_id.map(attempted_methods).fillna("pending")
    spots["dist_school_m"] = np.nan
    spots["child_zone_300m"] = pd.array([pd.NA] * len(spots), dtype="Int64")
    spots["accidents_200m"] = pd.array([pd.NA] * len(spots), dtype="Int64")
    spots["risk_weight"] = 1.0
    mask = spots.lat.notna() & spots.lon.notna()
    if mask.any():
        target = xy(spots.loc[mask, "lon"], spots.loc[mask, "lat"])
        schools = pd.read_csv(REF / "schools.csv")
        schools = schools.dropna(subset=["위도", "경도"])
        zones = pd.read_csv(REF / "child_zones.csv").dropna(subset=["위도", "경도"])
        school_dist = cKDTree(xy(schools["경도"], schools["위도"])).query(target)[0]
        zone_dist = cKDTree(xy(zones["경도"], zones["위도"])).query(target)[0]
        accidents = accident_coordinates()
        accidents = accidents[accidents.in_yuseong]
        nearby = cKDTree(xy(accidents.lon, accidents.lat)).query_ball_point(target, r=200, return_length=True)
        spots.loc[mask, "dist_school_m"] = school_dist
        spots.loc[mask, "child_zone_300m"] = (zone_dist <= 300).astype(int)
        spots.loc[mask, "accidents_200m"] = nearby
        spots.loc[mask, "risk_weight"] = 1 + .5 * (school_dist <= 300) + .5 * (zone_dist <= 300) + .5 * np.minimum(nearby / 10, 1)
    if REF.joinpath("yuseong_boundary.geojson").exists():
        accident_coordinates()
    write_frame(spots[SPOT_COLUMNS], OUT / "spots.csv")
    quality_metrics(spots, cache)
    write_review_sample(spots, cache)
    write_json({"formula": "risk_weight = 1 + 0.5 I(dist_school_m<=300) + 0.5 child_zone_300m + 0.5 min(accidents_200m/10,1)",
                "range": [1, 2.5], "child_zone_300m": "어린이보호구역 지정시설 점좌표에서 300m 이내; 법적 구역 내부 아님",
                "distance": "EPSG:5179 평면 직선거리(m)", "accidents": "2017~2021 유성구 내부 사고 건수; 반경 200m",
                "temporal_use": "학교·보호구역은 현재 공개 시설; 사고는 2021년까지. 사전 고정한 정책 가중치이며 최종 평가로 조정하지 않음",
                "missing_coordinates": "거리·보호구역·사고수는 결측, risk_weight=1; 경로 후보에서 제외",
                "normalization": "지번형은 동+산 여부+본번+부번; 같은 지번의 건물·부근 꼬리 병합; 장소명은 접두·부근/주변/인근 꼬리만 제거. 좌표 근접 병합 없음"}, OUT / "feature_definitions.json")
    return spots[SPOT_COLUMNS]
