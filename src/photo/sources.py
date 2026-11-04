"""License-checked Wikimedia download; AI inputs are ingested by the generator."""
from __future__ import annotations

import html
import io
import json
import re
import time

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

from .common import DATA

SELECTION = [99597207, 134022527, 199386154, 154775355, 135544162, 135544164,
             136651582, 109681829, 174447438, 146353915, 133742075, 108205201,109681849]


def clean(value):
    return html.unescape(re.sub("<[^>]*>", "", value or "")).strip()


def download():
    pages = {}
    for path in DATA.glob("commons_*candidates.json"):
        pages.update({int(k): v for k, v in json.loads(path.read_text()).get("query", {}).get("pages", {}).items()})
    session = requests.Session()
    session.headers["User-Agent"] = "YuseongPhotoDemoResearch/1.0 (academic prototype; license-checked images)"
    rows = []
    for index, pid in enumerate(SELECTION, start=1):
        page = pages[pid]
        info = page["imageinfo"][0]
        meta = info["extmetadata"]
        license_name = clean(meta.get("LicenseShortName", {}).get("value"))
        if not license_name.startswith(("CC BY", "CC0", "Public domain")):
            raise ValueError("Unapproved source license")
        target = DATA / "bases" / f"wm_{pid}.jpg"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            response = session.get(info.get("thumburl", info["url"]), timeout=45)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.thumbnail((1280, 1280))
            image.save(target, quality=95)
            time.sleep(.3)
        rows.append({"source_id": f"wm_{pid}", "source_type": "위키미디어 공용", "path": str(target.relative_to(DATA.parents[1])),
                     "source_url": info["descriptionurl"], "image_url": info["url"],
                     "author": clean(meta.get("Artist", {}).get("value")), "license": license_name,
                     "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                     "purpose":"background",
                     "adaptation": "해상도 조정; 평가 쌍은 별도 합성 번호판·밝기·원근 변환·시연용 EXIF 적용",
                     "title": page["title"]})
        print(f"downloaded {index}/{len(SELECTION)} licensed base images", flush=True)
    frame = pd.DataFrame(rows)
    if (DATA / "sources.csv").exists():
        previous = pd.read_csv(DATA / "sources.csv", keep_default_na=False)
        preserve=~previous.source_id.isin([f"wm_{pid}" for pid in SELECTION])
        frame = pd.concat([frame, previous[preserve]], ignore_index=True)
    frame.to_csv(DATA / "sources.csv", index=False, encoding="utf-8-sig")
    contact_sheet(frame[frame.source_type.eq("위키미디어 공용")])


def contact_sheet(frame):
    font = ImageFont.truetype(findfont(FontProperties(family="NanumGothic")), 18)
    width, height = 460, 355
    canvas = Image.new("RGB", (width * 3, height * ((len(frame)+2)//3)), "white")
    drawing = ImageDraw.Draw(canvas)
    for index, row in enumerate(frame.itertuples(index=False)):
        image = Image.open(DATA.parents[1] / row.path).convert("RGB")
        image.thumbnail((width - 10, height - 45))
        x, y = index % 3 * width, index // 3 * height
        canvas.paste(image, (x + 5, y + 30))
        drawing.text((x + 5, y + 4), f"{row.source_id}  {Image.open(DATA.parents[1] / row.path).size}", font=font, fill="black")
    canvas.save(DATA / "source_contact_sheet.jpg", quality=95)


if __name__ == "__main__":
    download()
