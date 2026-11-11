"""Read image pixels and EXIF only; never consult filenames or demo labels."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
import math
import os
import re
import threading
import unicodedata

import numpy as np
from PIL import Image, ImageOps
from pyproj import Geod

from .common import DATA, LETTERS, configure_runtime

PLATE = re.compile(r"[0-9]{2,3}[" + LETTERS + r"][0-9]{4}")
GEOD = Geod(ellps="WGS84")
LOCK = threading.RLock()


def normalize_plate(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^0-9가-힣]", "", text)
    return text if PLATE.fullmatch(text) else None


@lru_cache(maxsize=1)
def get_reader():
    configure_runtime()
    import torch
    import cv2
    import easyocr
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    # Use the GPU if present, otherwise the CPU. GPU names listed in OCR_SKIP_GPU
    # (comma-separated, e.g. "3090") are skipped in favour of the CPU.
    skip = [s.strip() for s in os.environ.get("OCR_SKIP_GPU", "").split(",") if s.strip()]
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if name and not any(s in name for s in skip):
        gpu = "cuda:0"
    else:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        name, gpu = "CPU", False
    print(f"photo_ocr_device: {name}; CPU threads={torch.get_num_threads()}", flush=True)
    directory = DATA / "models/easyocr"
    directory.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(["ko", "en"], gpu=gpu, model_storage_directory=str(directory),
                          user_network_directory=str(DATA / "models/user_network"), verbose=False)


@contextmanager
def unpinned(reader):
    """easyocr's recognizer always builds DataLoader(pin_memory=True). Pinning host memory opens a CUDA context on
    GPU 0 even for a CPU reader (about 500 MiB, and it fails when GPU 0 is full). For a CPU reader the DataLoader is
    made with pin_memory=False while the reader runs; callers hold LOCK, and a GPU reader is left unchanged."""
    if reader.device != "cpu":
        yield
        return
    import torch.utils.data as data
    original = data.DataLoader

    class Unpinned(original):
        def __init__(self, *args, **kwargs):
            kwargs["pin_memory"] = False
            super().__init__(*args, **kwargs)

    data.DataLoader = Unpinned
    try:
        yield
    finally:
        data.DataLoader = original


def read_exif(image):
    """DateTimeOriginal and GPS are metadata, not authenticated capture evidence."""
    import piexif
    result = {"taken": None, "datetime": None, "gps": None}
    data = image.info.get("exif")
    if not data:
        return result
    try:
        exif = piexif.load(data)
    except (ValueError, TypeError, piexif.InvalidImageDataError):
        result["metadata_error"] = "invalid_exif"
        return result
    stamp = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
    if stamp:
        try:
            stamp = stamp.decode("ascii") if isinstance(stamp, bytes) else stamp
            dt = datetime.strptime(stamp.strip("\x00"), "%Y:%m:%d %H:%M:%S")
            result.update(taken=dt.strftime("%Y-%m-%d %H:%M:%S"), datetime=dt)
        except (ValueError, TypeError, UnicodeError):
            result["time_error"] = "invalid_datetime_original"
    gps = exif.get("GPS", {})
    try:
        def coordinate(key, ref_key):
            raw = gps[key]
            values = [float(a) / float(b) for a, b in raw]
            angle = values[0] + values[1] / 60 + values[2] / 3600
            ref = gps.get(ref_key, b"")
            if isinstance(ref, bytes):
                ref = ref.decode("ascii")
            valid_refs=("N","S") if ref_key==piexif.GPSIFD.GPSLatitudeRef else ("E","W")
            if ref not in valid_refs:
                raise ValueError("Missing or invalid GPS hemisphere reference")
            return -angle if ref in ("S", "W") else angle
        if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
            lat = coordinate(piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLatitudeRef)
            lon = coordinate(piexif.GPSIFD.GPSLongitude, piexif.GPSIFD.GPSLongitudeRef)
            if math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
                result["gps"] = (lat, lon)
    except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError, UnicodeError):
        result["gps_error"] = "invalid_gps"
    return result


def distance_m(a, b):
    return float(GEOD.inv(a[1], a[0], b[1], b[0])[2])


def box_to_raw(box, orientation, width, height):
    if box is None or orientation == 1:
        return box
    inverse = {
        2: lambda u,v:(width-u,v), 3: lambda u,v:(width-u,height-v),
        4: lambda u,v:(u,height-v), 5: lambda u,v:(v,u),
        6: lambda u,v:(v,height-u), 7: lambda u,v:(width-v,height-u),
        8: lambda u,v:(width-v,u),
    }.get(orientation,lambda u,v:(u,v))
    x1,y1,x2,y2=box
    corners=np.asarray([inverse(u,v) for u,v in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]])
    return [int(corners[:,0].min()),int(corners[:,1].min()),int(corners[:,0].max()),int(corners[:,1].max())]


def plate_regions(rgb):
    """Generic bright, elongated quadrilaterals; no demo ROI or ground truth access."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contours = []
    for threshold in (180, 220):
        mask = cv2.inRange(rgb, np.array([threshold]*3), np.array([255]*3))
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    h, w = gray.shape
    candidates = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) < 12 or max(rw, rh) < 65:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        area = rw * rh
        if not (2.5 <= ratio <= 8 and .0005 * h * w <= area <= .13 * h * w):
            continue
        polygon=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        points=polygon.reshape(4,2).astype(np.float32) if len(polygon)==4 and cv2.isContourConvex(polygon) else cv2.boxPoints(rect).astype(np.float32)
        sums=points.sum(axis=1);differences=points[:,1]-points[:,0]
        indices=[int(sums.argmin()),int(differences.argmin()),int(sums.argmax()),int(differences.argmax())]
        if len(set(indices))!=4:
            continue
        quad=points[indices]
        width = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        height = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        if height < 8 or width < height * 2:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        candidates.append((area, quad, [max(0, x), max(0, y), min(w, x+bw), min(h, y+bh)]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    accepted = []
    for item in candidates:
        box = item[2]
        if any(abs(box[0]-x[2][0]) + abs(box[1]-x[2][1]) < 20 for x in accepted):
            continue
        accepted.append(item)
        if len(accepted) == 12:
            break
    return accepted


def read_plate(image):
    import cv2
    with LOCK:
        reader = get_reader()
    rgb = np.asarray(image.convert("RGB"))
    original_h, original_w = rgb.shape[:2]
    scale = min(1., 1600 / max(original_h, original_w))
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    candidates = []
    allow = "0123456789" + LETTERS
    with LOCK, unpinned(reader):
        for _, quad, box in plate_regions(rgb):
            width,height = 520,110
            target = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
            crop = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(quad, target), (width, height), borderValue=(255,255,255))
            crop = cv2.copyMakeBorder(crop, 5, 5, 8, 8, cv2.BORDER_CONSTANT, value=(255,255,255))
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            result = reader.recognize(gray, horizontal_list=[[0, gray.shape[1], 0, gray.shape[0]]], free_list=[],
                                      detail=1, paragraph=False, allowlist=allow, batch_size=1, workers=0)
            for _, text, confidence in result:
                plate = normalize_plate(text)
                if plate and confidence >= .25:
                    candidates.append((float(confidence), plate, [int(v / scale) for v in box]))
        if not candidates:
            results = reader.readtext(rgb, detail=1, paragraph=False, allowlist=allow, workers=0, batch_size=1,
                                      min_size=8, canvas_size=1600, mag_ratio=1.3)
            for points, text, confidence in results:
                plate = normalize_plate(text)
                if plate and confidence >= .25:
                    points = np.asarray(points) / scale
                    box = [int(points[:,0].min()), int(points[:,1].min()), int(points[:,0].max()), int(points[:,1].max())]
                    candidates.append((float(confidence), plate, box))
    if not candidates:
        return {"plate": None, "box": None, "confidence": None}
    confidence, plate, box = max(candidates, key=lambda x: x[0])
    return {"plate": plate, "box": box, "confidence": confidence}


def decide_pair(first, second, reported_lat=None, reported_lon=None, min_gap_s=60, max_move_m=15, max_report_m=50):
    """Pure fixed-threshold rules, also used by known-answer tests."""
    if not all(math.isfinite(x) and x >= 0 for x in (min_gap_s,max_move_m,max_report_m)):
        raise ValueError("Photo-review thresholds must be finite and nonnegative")
    result = {"plate_1": first.get("plate"), "plate_2": second.get("plate"),
              "box_1": first.get("box"), "box_2": second.get("box"),
              "taken_1": first.get("taken"), "taken_2": second.get("taken"),
              "gps_1": first.get("gps"), "gps_2": second.get("gps"),
              "gap_s": None, "move_m": None, "report_m": None, "verdict": "보완 필요", "reasons": []}
    reasons = result["reasons"]
    if not result["plate_1"] or not result["plate_2"]:
        reasons.append("번호 판독 불가")
    elif result["plate_1"] != result["plate_2"]:
        reasons.append("번호 불일치")
    t1, t2 = first.get("datetime"), second.get("datetime")
    if t1 is None or t2 is None:
        reasons.append("촬영 시각 정보 없음")
    else:
        result["gap_s"] = abs((t2 - t1).total_seconds())
        if result["gap_s"] < min_gap_s:
            reasons.append("촬영 간격 미달")
    gps1, gps2 = first.get("gps"), second.get("gps")
    if gps1 is None or gps2 is None:
        reasons.append("위치 정보 없음")
    else:
        result["move_m"] = distance_m(gps1, gps2)
        if result["move_m"] > max_move_m + 1e-6:
            reasons.append("촬영 위치 불일치")
    if reported_lat is None or reported_lon is None:
        reasons.append("신고 위치 정보 없음")
    else:
        point = (float(reported_lat), float(reported_lon))
        if not all(math.isfinite(v) for v in point) or not (-90 <= point[0] <= 90 and -180 <= point[1] <= 180):
            raise ValueError("Reported GPS coordinates outside their valid range")
        if gps1 is not None and gps2 is not None:
            result["report_m"] = max(distance_m(gps1, point), distance_m(gps2, point))
            if result["report_m"] > max_report_m + 1e-6:
                reasons.append("신고 위치 불일치")
    if not reasons:
        result["verdict"] = "요건 충족"
    return result


def review_photo_pair(path1, path2, reported_lat=None, reported_lon=None, min_gap_s=60, max_move_m=15, max_report_m=50):
    """Return JSON-serializable OCR/EXIF evidence and the verdict used by the UI.

    Recognition reads image pixels only. Neither filename nor manifest ground truth
    is an input to read_plate. EXIF presence does not establish its authenticity.
    """
    photos = []
    for path in (path1, path2):
        with Image.open(path) as raw:
            metadata = read_exif(raw)
            orientation=raw.getexif().get(274,1)
            width,height=raw.size
            image = ImageOps.exif_transpose(raw).convert("RGB")
            recognized=read_plate(image)
            recognized["box"]=box_to_raw(recognized["box"],orientation,width,height)
            photos.append({**metadata, **recognized})
    return decide_pair(*photos, reported_lat=reported_lat, reported_lon=reported_lon,
                       min_gap_s=min_gap_s, max_move_m=max_move_m, max_report_m=max_report_m)
