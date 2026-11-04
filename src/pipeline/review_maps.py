"""Build cached street-map contact sheets for the frozen random 30-point review.

This creates review material, not a claim that the review has been performed.
"""
from __future__ import annotations

import io
import math
import time

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

from .common import FIG, OUT, REF


def tile_position(lat, lon, zoom=17):
    n = 2 ** zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def street_crop(lat, lon, session):
    zoom = 17
    x, y = tile_position(lat, lon, zoom)
    left, top = math.floor(x * 256 - 256), math.floor(y * 256 - 180)
    first_x, first_y = left // 256, top // 256
    last_x, last_y = (left + 511) // 256, (top + 359) // 256
    canvas = Image.new("RGB", ((last_x - first_x + 1) * 256, (last_y - first_y + 1) * 256), "white")
    for tx in range(first_x, last_x + 1):
        for ty in range(first_y, last_y + 1):
            path = REF / "osm_tiles" / str(zoom) / str(tx) / f"{ty}.png"
            if path.exists():
                data = path.read_bytes()
            else:
                response = session.get(f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png", timeout=20)
                response.raise_for_status()
                data = response.content
                Image.open(io.BytesIO(data)).verify()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                time.sleep(.12)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            canvas.paste(image, ((tx - first_x) * 256, (ty - first_y) * 256))
    ox, oy = left - first_x * 256, top - first_y * 256
    crop = canvas.crop((ox, oy, ox + 512, oy + 360))
    drawing = ImageDraw.Draw(crop)
    drawing.ellipse((249, 173, 263, 187), fill="#D55E00", outline="white", width=2)
    drawing.line((256, 162, 256, 198), fill="#D55E00", width=2)
    drawing.line((238, 180, 274, 180), fill="#D55E00", width=2)
    return crop


def main():
    frame = pd.read_csv(OUT / "geocode_review_30.csv", keep_default_na=False)
    session = requests.Session()
    session.headers["User-Agent"] = "YuseongParkingV1/1.0 (local research map review; caching enabled)"
    font_path = findfont(FontProperties(family="NanumGothic"))
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)
    for page in range(3):
        path = FIG / f"analysis_geocode_review_{page + 1:02d}.png"
        canvas = Image.new("RGB", (1120, 2310), "white")
        draw = ImageDraw.Draw(canvas)
        for local, row in enumerate(frame.iloc[page * 10:(page + 1) * 10].itertuples(index=False)):
            x, y = (local % 2) * 560 + 14, (local // 2) * 455 + 10
            draw.text((x, y), f"{page * 10 + local + 1:02d}. {row.label}", font=font, fill="black")
            draw.text((x, y + 28), f"{row.lat:.6f}, {row.lon:.6f}  {row.geocode_method}", font=small, fill="#333333")
            canvas.paste(street_crop(row.lat, row.lon, session), (x, y + 53))
            draw.text((x, y + 417), str(row.matched_address)[-40:], font=small, fill="black")
        draw.text((14, 2285), "지도: © OpenStreetMap contributors (ODbL). 표본 시각 검토용; 경계·지번의 측량 검증 자료가 아님.", font=small, fill="black")
        canvas.save(path, dpi=(200, 200))
        print(f"review map saved: {path.name}", flush=True)


if __name__ == "__main__":
    main()
