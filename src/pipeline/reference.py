"""Cache public boundary and nationwide standard data, retaining source provenance."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import requests

from .common import REF, write_frame, write_json

STANDARD = {"schools": "15021148", "child_zones": "15012891"}


def fetch_standard(name: str) -> pd.DataFrame:
    path = REF / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    pk = STANDARD[name]
    base = "https://www.data.go.kr"
    session = requests.Session()
    session.get(f"{base}/data/{pk}/standard.do", timeout=30).raise_for_status()
    response = session.get(f"{base}/download/columList.json", params={"pk": pk, "ext": "CSV"}, timeout=30)
    response.raise_for_status()
    header = response.json()
    table = header["tableVO"]
    params = {"publicDataPk": pk, "colNmList": table["colNmList"],
              "totalCount": header["totalCount"], "svcTableNm": table["svcTableNm"], "perPage": 10000}
    columns = {x["columCode"]: x["columNm"] for x in header["columList"]}
    local_download = REF / f"{name}_download.csv"
    if local_download.exists():
        local = pd.read_csv(local_download, dtype=str, keep_default_na=False)
        national_rows = header["totalCount"]
    else:
        records = []
        for page in range(1, math.ceil(header["totalCount"] / 10000) + 1):
            response = session.get(f"{base}/download/standard.json", params={**params, "page": page}, timeout=60)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("Unexpected public standard data response")
            records.extend(data)
        full = pd.DataFrame(records).rename(columns=columns)
        national_rows = len(full)
        addresses = full.get("소재지지번주소", pd.Series("", index=full.index)).fillna("") + " " + full.get("소재지도로명주소", pd.Series("", index=full.index)).fillna("")
        local = full[addresses.str.contains("유성구") & addresses.str.contains("대전")].copy()
        write_frame(local, local_download)
    if name == "schools" and "운영상태" in local:
        local = local[local["운영상태"] == "운영"]
    if name == "child_zones" and "어린이보호구역지정여부" in local:
        # The current Yuseong records leave this newly added field blank. Inclusion
        # in the designated-facility standard is retained; explicit rescissions go.
        local = local[~local["어린이보호구역지정여부"].fillna("").str.upper().isin(["N", "아니오", "해제", "0"])]
    if len(local) == 0:
        raise ValueError(f"No Yuseong {name} records")
    write_frame(local, path)
    write_json({"source": f"{base}/data/{pk}/standard.do", "download_endpoint": f"{base}/download/standard.json",
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "national_rows": national_rows, "local_rows": len(local),
                "filter": "주소에 대전·유성구 포함; 학교는 운영; 보호구역은 명시적 해제만 제외(현재 유성구 지정여부 필드는 모두 공란)",
                "reference_dates": sorted(local["데이터기준일자"].dropna().unique().tolist()),
                "spatial_semantics": "시설 점좌표; 법적 구역 경계 폴리곤 아님"}, REF / f"{name}_source.json")
    return local


def fetch_boundary() -> None:
    path = REF / "yuseong_boundary.geojson"
    if path.exists():
        return
    response = requests.get("https://nominatim.openstreetmap.org/search",
                            params={"q": "Yuseong-gu, Daejeon, South Korea", "format": "jsonv2", "polygon_geojson": 1, "limit": 1},
                            headers={"User-Agent": "YuseongParkingResearch/1.0 (local academic prototype)"}, timeout=45)
    response.raise_for_status()
    result = response.json()[0]
    if result["type"] != "administrative" or result["geojson"]["type"] not in ("Polygon", "MultiPolygon"):
        raise ValueError("Expected Yuseong administrative polygon")
    write_json({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": result["display_name"], "osm_id": result["osm_id"]}, "geometry": result["geojson"]}]}, path)
    write_json({"source": "https://www.openstreetmap.org/relation/3890428", "download_endpoint": "https://nominatim.openstreetmap.org/search",
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "license": "OpenStreetMap ODbL; © OpenStreetMap contributors",
                "usage": "유성구 안팎 좌표 품질 확인; 현재 OSM 경계, 행정기관 인증 경계 아님"}, REF / "boundary_source.json")


def fetch_all() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    fetch_boundary()
    for name in STANDARD:
        frame = fetch_standard(name)
        print(f"reference {name}: {len(frame)} local records", flush=True)
