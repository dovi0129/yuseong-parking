"""사진 검토 영역: src.photo.review_photo_pair 결과를 화면에 보여 준다.

결과 dict 키: plate_1/2, box_1/2, taken_1/2, gps_1/2, gap_s, move_m, report_m, verdict, reasons.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Final manifest first, development pairs as a fallback.
MANIFESTS = (ROOT / "data/photo_demo/manifest.csv", ROOT / "data/photo_demo/dev_manifest.csv")
SOURCES = ROOT / "data/photo_demo/sources.csv"
FONT = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
MIN_GAP_S, MAX_MOVE_M, MAX_REPORT_M = 60, 15, 50
BOX = "#1b3561"

# Interim demo pairs (no photo files yet); keys match review_photo_pair.
_INTERIM = {
    "정상": dict(plate_1="123가4567", plate_2="123가4567", gap_s=74.0, move_m=6.2, report_m=18.4, reasons=[]),
    "간격 미달": dict(plate_1="58누2041", plate_2="58누2041", gap_s=31.0, move_m=3.8, report_m=22.0, reasons=["촬영 간격 미달"]),
    "다른 번호": dict(plate_1="231머7715", plate_2="231머7716", gap_s=88.0, move_m=4.1, report_m=12.7, reasons=["번호 불일치"]),
    "위치 정보 없음": dict(plate_1="47서9032", plate_2="47서9032", gap_s=65.0, move_m=None, report_m=None, reasons=["위치 정보 없음"]),
}


def review_function():
    try:
        from src.photo import review_photo_pair
    except ImportError:
        return None
    return review_photo_pair


def ai_photos():
    """True when the demo photos shown come from AI-generated sources."""
    manifest = next((path for path in MANIFESTS if path.exists()), None)
    if manifest is None or review_function() is None or not SOURCES.exists():
        return False
    sources = pd.read_csv(SOURCES, encoding="utf-8-sig")
    pairs = pd.read_csv(manifest, encoding="utf-8-sig")
    if "source_id" in pairs and "source_id" in sources:
        sources = sources.loc[sources.source_id.isin(pairs.source_id)]
    return bool(sources.astype(str).apply(lambda column: column.str.contains("AI 생성", regex=False)).to_numpy().any())


def demo_pairs():
    """pair_id, path1, path2, case, reported_lat, reported_lon (manifest or interim rows)."""
    manifest = next((path for path in MANIFESTS if path.exists()), None)
    if manifest is not None and review_function() is not None:
        frame = pd.read_csv(manifest, encoding="utf-8-sig")
        if "scene" not in frame and SOURCES.exists():
            scenes = pd.read_csv(SOURCES, encoding="utf-8-sig")
            if {"source_id", "scene"} <= set(scenes) and "source_id" in frame:
                frame = frame.merge(scenes[["source_id", "scene"]], on="source_id", how="left")
        width = len(str(len(frame)))
        place = frame["scene"].fillna("").astype(str) if "scene" in frame else pd.Series("", index=frame.index)
        frame["label"] = [" · ".join(x for x in (f"사례 {i:0{width}d}", p, c) if x)
                          for i, (p, c) in enumerate(zip(place, frame["case"].astype(str)), 1)]
        return frame[["pair_id", "path1", "path2", "case", "reported_lat", "reported_lon", "label"]]
    return pd.DataFrame([{"pair_id": f"P{i:03d}", "path1": None, "path2": None, "case": case, "reported_lat": None,
                          "reported_lon": None, "label": f"사례 {i:03d} · {case}"} for i, case in enumerate(_INTERIM, 1)])


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


@st.cache_data(show_spinner=False, max_entries=256)
def review_demo(pair_id, path1, path2, case, reported):
    fn = review_function()
    if fn is None or path1 is None:
        return _interim(case, pair_id)
    lat, lon = reported if reported else (None, None)
    return fn(str(_resolve(path1)), str(_resolve(path2)), reported_lat=lat, reported_lon=lon,
              min_gap_s=MIN_GAP_S, max_move_m=MAX_MOVE_M, max_report_m=MAX_REPORT_M)


@st.cache_data(show_spinner=False, max_entries=64)
def review_upload(data1: bytes, name1: str, data2: bytes, name2: str, reported):
    fn = review_function()
    if fn is None:
        return _interim(None, hashlib.sha1(data1 + data2).hexdigest()[:6])
    lat, lon = reported if reported else (None, None)
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = Path(tmp) / f"1{Path(name1).suffix}", Path(tmp) / f"2{Path(name2).suffix}"
        p1.write_bytes(data1)
        p2.write_bytes(data2)
        return fn(str(p1), str(p2), reported_lat=lat, reported_lon=lon,
                  min_gap_s=MIN_GAP_S, max_move_m=MAX_MOVE_M, max_report_m=MAX_REPORT_M)


def _interim(case, seed):
    base = dict(_INTERIM.get(case, dict(plate_1=None, plate_2=None, gap_s=None, move_m=None, report_m=None,
                                         reasons=["번호 판독 불가", "촬영 시각 정보 없음", "위치 정보 없음"])))
    t0 = pd.Timestamp("2024-10-02 18:03:12")
    gap = base["gap_s"]
    base.update(taken_1=str(t0) if gap is not None else None,
                taken_2=str(t0 + pd.Timedelta(seconds=gap)) if gap is not None else None,
                gps_1=None if base["move_m"] is None else (36.3567, 127.3485),
                gps_2=None if base["move_m"] is None else (36.35674, 127.34856),
                box_1=(236, 318, 404, 356) if base["plate_1"] else None,
                box_2=(250, 322, 418, 360) if base["plate_2"] else None,
                verdict="보완 필요" if base["reasons"] else "요건 충족", interim=True, seed=seed)
    return base


def interim_image(plate, shift=0):
    """Placeholder frame drawn until data/photo_demo images exist."""
    img = Image.new("RGB", (640, 480), "#e4e8ee")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 400, 640, 480), fill="#cfd5dd")
    d.rounded_rectangle((120 + shift, 170, 520 + shift, 410), radius=36, fill="#9aa5b1")
    d.rounded_rectangle((170 + shift, 120, 470 + shift, 230), radius=28, fill="#b3bcc6")
    d.rectangle((236 + shift, 318, 404 + shift, 356), fill="#ffffff", outline="#3d4a59", width=2)
    if plate:
        d.text((320 + shift, 337), plate, fill="#111111", font=ImageFont.truetype(FONT, 24), anchor="mm")
    return img


def with_box(img, box):
    """Navy frame around the recognised plate, thick enough to survive the ~1/4 display scale."""
    out = img.convert("RGB").copy()
    if box:
        x1, y1, x2, y2 = box
        line = max(5, round(max(out.size) / 110))
        pad = line + 2
        draw = ImageDraw.Draw(out)
        draw.rectangle((x1 - pad - 2, y1 - pad - 2, x2 + pad + 2, y2 + pad + 2), outline="#ffffff", width=line + 4)
        draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), outline=BOX, width=line)
    return out


def summary(result):
    """One cell for the 사진 검토 column of the reception list and CSV."""
    if not result:
        return ""
    return result["verdict"] if not result["reasons"] else f"{result['verdict']}: {', '.join(result['reasons'])}"


def check_rows(result):
    """검토 항목 table rows: (항목, 내용, 결과), decided only by the returned reasons."""
    reasons = set(result["reasons"])
    p1, p2 = result["plate_1"] or "판독 불가", result["plate_2"] or "판독 불가"
    gap = result["gap_s"]
    taken = (f"{gap:.0f}초 ({pd.Timestamp(result['taken_1']):%H:%M:%S} → {pd.Timestamp(result['taken_2']):%H:%M:%S})"
             if gap is not None and result["taken_1"] and result["taken_2"] else "-")
    move = "-" if result["move_m"] is None else f"두 사진 간 {result['move_m']:.0f} m"
    report = "-" if result["report_m"] is None else f"신고 위치에서 {result['report_m']:.0f} m"

    def outcome(missing, failed, failed_label):
        return "미확인" if missing & reasons else failed_label if failed in reasons else "적합"

    return [("차량 번호", f"{p1} / {p2}", "판독 불가" if "번호 판독 불가" in reasons else
             "불일치" if "번호 불일치" in reasons else "일치"),
            ("촬영 간격", taken, outcome({"촬영 시각 정보 없음"}, "촬영 간격 미달", "미달")),
            ("촬영 위치", move, outcome({"위치 정보 없음"}, "촬영 위치 불일치", "불일치")),  # 위치 정보 없음 → both rows 미확인
            ("신고 위치", report, outcome({"위치 정보 없음", "신고 위치 정보 없음"}, "신고 위치 불일치", "불일치"))]
