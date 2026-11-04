"""Site settings (ui/site.toml), SVG word logo and footer lines."""
from __future__ import annotations

import base64
import html
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAVY, BLUE = "#1b2b4b", "#0369d1"
BOLD_FONT = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"
LATIN_FONT = "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf"
FONT = "'Noto Sans KR','Pretendard','Noto Sans CJK KR','NanumBarunGothic',sans-serif"
SOURCE_LINE = ("자료: 공공데이터포털 대전광역시 유성구 불법주정차 신고 현황(2024), 불법주정차 단속 현황(2023), "
               "교통사고 현황(2017~2021) · 지도 © OpenStreetMap contributors")
PHOTO_LINE = {True: "사진 검토 화면의 차량 사진은 AI로 생성한 시연용 사진입니다.",
              False: "사진 검토 화면의 차량 사진은 합성한 시연용 사진입니다."}
CONTEST_LINE = "2026학년도 전국 대학(원)생 빅데이터 분석 경진대회 출품 시제품"


def settings():
    return tomllib.loads((HERE / "site.toml").read_text(encoding="utf-8"))


def _text_path(text, size, font=BOLD_FONT):
    """Outline of text as one SVG path (y down), so the logo renders without the font installed."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    path = TextPath((0, 0), text, size=size, prop=FontProperties(fname=font))
    codes = {1: "M", 2: "L", 3: "Q", 4: "C", 79: "Z"}
    parts, i, verts = [], 0, path.vertices
    while i < len(path.codes):
        code = int(path.codes[i])
        step = {1: 1, 2: 1, 3: 2, 4: 3, 79: 1}[code]
        if code == 79:
            parts.append("Z")
        else:
            pts = " ".join(f"{x:.1f},{-y:.1f}" for x, y in verts[i:i + step])
            parts.append(f"{codes[code]}{pts}")
        i += step
    extent = path.get_extents()
    return " ".join(parts), extent.x1


def logo_svg(name, height=34):
    """Map pin with a parking 'P' followed by the system name in outlined glyphs."""
    fs = height * 0.6
    d, text_w = _text_path(name, fs)
    gap = height * 0.32
    pin_w = height * 0.75
    width = pin_w + gap + text_w + 2
    k = height / 32
    p, _ = _text_path("P", 11.5, LATIN_FONT)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height}" viewBox="0 0 {width:.1f} {height}">'
            f'<g transform="scale({k:.4f})"><path d="M12 0C5.4 0 0 5.2 0 11.7 0 20.5 12 32 12 32s12-11.5 12-20.3C24 5.2 18.6 0 12 0z" '
            f'fill="{BLUE}"/><circle cx="12" cy="11.6" r="7.4" fill="#fff"/>'
            f'<path transform="translate(8.6 15.7)" d="{p}" fill="{BLUE}"/></g>'
            f'<path transform="translate({pin_w + gap:.1f} {height * 0.73:.1f})" d="{d}" fill="{NAVY}"/></svg>')


def logo_img(name, height=34):
    svg = logo_svg(name, height)
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="logo" src="data:image/svg+xml;base64,{data}" height="{height}" alt="{html.escape(name)}">'


def icon_img(svg, cls=""):
    data = base64.b64encode(svg.encode()).decode()
    return f'<img class="{cls}" src="data:image/svg+xml;base64,{data}" alt="">'


def footer_html(name, team, ai_photos):
    lines = "".join(f"<p>{html.escape(t)}</p>" for t in (SOURCE_LINE, PHOTO_LINE[bool(ai_photos)], CONTEST_LINE))
    return (f'<div class="footer"><div class="f-row"><div class="f-logo">{logo_img(name, 30)}</div>'
            f'<div class="f-info">{lines}</div></div><div class="f-sep"></div>'
            f'<div class="f-copy">{html.escape(" ".join(x for x in ("© 2026", (team or "").strip()) if x))}</div></div>')
