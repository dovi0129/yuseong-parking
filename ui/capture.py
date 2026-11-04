"""Capture every screen state at 1600x1000, check map tiles, and list the visible UI strings.

Usage: .venv/bin/python ui/capture.py [--url http://127.0.0.1:8502] [--width 1600] [--scale 1] [--out artifacts/screens] [--zoom 100]   (without --url a server is started on 8765)
Outputs: artifacts/screens/ui_<state>.png (+ _full.png), artifacts/logs/ui_tile_check.json, ui/visible_strings.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "artifacts/screens"
LOGS = ROOT / "artifacts/logs"
# Known placeholder tiles: CARTO "API KEY REQUIRED" (2,049 B) and OSM "Access blocked" (6,933 B), saved 2026-10-30.
PLACEHOLDER_FILES = {"carto_api_key_required": "carto_api_key_required.png", "osm_access_blocked": "osm_access_blocked.png"}
PLACEHOLDER_SHA256 = {}
STATES = [  # name, query, action
    ("overview", "menu=overview", None),
    ("overview_list", "menu=overview&tab=list", None),
    ("overview_focus", "menu=overview&tab=list", "pick_row"),
    ("plan", "menu=plan", None),
    ("plan_list", "menu=plan&tab=list", None),
    ("plan_focus", "menu=plan&tab=list", "pick_row"),
    ("reception", "menu=reception", None),
    ("reception_photo", "menu=reception&tab=photo", "photo"),
]
DATA_SELECTORS = ('[data-testid="stDataFrame"], textarea, .kv .v, .chk td.v, .kpi .v, [data-testid="stImage"], '
                  '[data-testid="stIconMaterial"], [data-testid="stDateInput"] input, [data-testid="stSelectbox"] input, '
                  '[data-testid="stMultiSelect"] [data-baseweb="tag"] span, [data-testid="stSliderThumbValue"], '
                  '[data-testid="stSliderTickBar"], .total .n')


def wait_health(url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Streamlit server did not become healthy")


def load_placeholders():
    folder = HERE / "reference_tiles"
    for name, file in PLACEHOLDER_FILES.items():
        path = folder / file
        if path.exists():
            PLACEHOLDER_SHA256[hashlib.sha256(path.read_bytes()).hexdigest()] = name


def settle(page):
    page.locator(".footer").wait_for(timeout=600_000)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stSpinner\"]')", timeout=600_000)
    frames = page.locator('iframe[title="streamlit_folium.st_folium"]')
    for i in range(frames.count()):
        page.frame_locator('iframe[title="streamlit_folium.st_folium"]').nth(i).locator(".leaflet-tile-loaded").first.wait_for(timeout=60_000)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(2500)
    errors = page.locator('[data-testid="stException"]').all_text_contents()
    if errors:
        raise RuntimeError(str(errors))


def pick_row(page):
    box = page.locator('[data-testid="stDataFrame"]').first.bounding_box()
    page.mouse.click(box["x"] + 16, box["y"] + 35 + 35 * 2 + 17)   # third data row checkbox
    page.wait_for_timeout(1500)


def photo(page):
    page.get_by_role("combobox", name="시연 사진").click()
    page.locator('[role="option"]').first.click()
    page.get_by_role("button", name="조회").click()
    page.locator('[data-testid="stImage"]').nth(1).wait_for(timeout=180_000)


def visible_strings(page):
    page.evaluate(f"() => document.querySelectorAll('{DATA_SELECTORS}').forEach(e => e.remove())")
    lines = [line.strip() for line in page.locator('[data-testid="stMain"]').inner_text().splitlines()]
    frames = page.frame_locator('iframe[title="streamlit_folium.st_folium"]')
    for i in range(page.locator('iframe[title="streamlit_folium.st_folium"]').count()):
        frame = frames.nth(i)
        for selector in (".map-legend", ".leaflet-control-zoom", ".leaflet-control-attribution"):
            if frame.locator(selector).count():
                lines += [x.strip() for x in frame.locator(selector).first.inner_text().splitlines()]
        button = frame.locator(".leaflet-control-fullscreen a, a.leaflet-control-zoom-fullscreen")
        if button.count():
            lines.append(f"[지도 단추 제목] {button.first.get_attribute('title')}")
    return [x for x in dict.fromkeys(lines) if x and not re.fullmatch(r"[\d,.\s%:-]+", x)]


def capture(url, only=None, width=1600, scale=1, out=OUT, zoom=100):
    out.mkdir(parents=True, exist_ok=True)
    tiles, strings = [], {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                     args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": width, "height": 1000}, device_scale_factor=scale, locale="ko-KR")
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)))

        def on_response(response):
            if "/tiles/" not in response.url and "tile.openstreetmap.org" not in response.url \
                    and "basemaps.cartocdn.com" not in response.url:
                return
            try:
                body = response.body()
            except Exception as error:  # noqa: BLE001 - recorded as a failed tile
                tiles.append({"url": response.url, "status": response.status, "error": str(error)})
                return
            digest = hashlib.sha256(body).hexdigest()
            tiles.append({"url": response.url, "status": response.status, "bytes": len(body), "sha256": digest,
                          "placeholder": PLACEHOLDER_SHA256.get(digest), "source": response.headers.get("x-tile-source")})
        page.on("response", on_response)
        for name, query, action in STATES:
            if only and name not in only:
                continue
            page.set_viewport_size({"width": width, "height": 1000})
            page.goto(f"{url}/?{query}" + (f"&zoom={zoom}" if zoom != 100 else ""), wait_until="domcontentloaded")
            settle(page)
            if action:
                {"pick_row": pick_row, "photo": photo}[action](page)
                settle(page)
            page.mouse.move(5, 5)
            page.screenshot(path=str(out / f"ui_{name}.png"))
            height = page.locator('[data-testid="stMain"]').evaluate("e => e.scrollHeight")
            page.set_viewport_size({"width": width, "height": min(6000, max(1000, height + 10))})
            page.wait_for_timeout(1500)
            page.screenshot(path=str(out / f"ui_{name}_full.png"))
            strings[name] = visible_strings(page)
            print("captured", name, height, flush=True)
        browser.close()
    if failures:
        raise RuntimeError(str(failures))
    return tiles, strings


def code_values():
    """Values that appear inside tables, badges and option lists rather than in the static page text."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    import backend as be
    import photo as photo_module
    from src.optimizer import processing
    import brand
    pairs = photo_module.demo_pairs()
    site = brand.settings()
    return {
        "시스템명(머리·바닥글 로고 그림, 브라우저 탭 제목)": [site["system_name"]],
        "근무 구간 선택지": list(be.SHIFTS),
        "차량 수 선택지": ["1대", "2대", "3대"],
        "위험 가중 선택지": ["적용", "미적용"],
        "지점 등급 값": list(be.GRADES),
        "동 선택지": ["전체", *be.dongs(be.version())],
        "조치 구분 값": ["금일 순회 포함", "순회 경로 반영 검토", "일반 처리"],
        "시설 개선 검토 값": ["해당", "비해당"],
        "단속 차량 값": ["1호차", "2호차", "3호차"],
        "사진 검토 열 값": ["요건 충족", "보완 필요: <사유 목록>"],
        "사진 검토 결과 값": ["일치", "불일치", "판독 불가", "적합", "미달", "미확인"],
        "사진 검토 사유": ["번호 불일치", "번호 판독 불가", "촬영 간격 미달", "촬영 위치 불일치", "신고 위치 불일치",
                      "위치 정보 없음", "신고 위치 정보 없음", "촬영 시각 정보 없음"],
        "시연 사진 선택지": list(pairs["label"]),
        "빈 결과": ["조회 결과가 없습니다."],
        "숫자 칸·결과 형식": ["<N>건", "<N>곳", "<N.N>%", "총 <N>건"],
        "화면 크기 값": [f"{n}%" for n in (80, 90, 100, 110, 120, 130)],
        "지도 마우스 올림 표시": ["<위치> · <지점 등급> · <N>건", "<N>호차 <순번>번 · <위치> · <HH:MM>", "유성구청"],
        "파일 선택 단추(CSS 글자)": ["파일 선택"],
        "표 열 이름(현황 명단)": ["위치", "동", "지점 등급", "신고 건수", "단속 수", "위험 가중치"],
        "표 열 이름(순회 계획 방문표)": ["단속 차량", "순번", "위치", "도착 예정", "기대 신고 수", "위험 가중치"],
        "표 열 이름(민원 접수)": ["접수 일시", "위치", "지점 등급", "금일 순번", "묶음 번호", "조치 구분", "시설 개선 검토", "사진 검토"],
        "CSV 열 이름(현황)": ["지점 ID", "위치", "동", "위도", "경도", "신고 건수", "학습 기간 신고 수", "단속 수",
                          "보호구역 지정시설 300m 이내", "위험 가중치", "지점 등급"],
        "CSV 열 이름(순회 계획)": list(processing.PLAN_COLUMNS),
        "CSV 열 이름(민원 접수)": [*processing.RECEPTION_COLUMNS, "사진 검토"],
    }


def write_strings(strings):
    out = ["# 화면에 보이는 글자 (자동 추출, 지명·숫자·회신 본문 같은 데이터 값은 뺐다). 상태마다 한 묶음이다."]
    for name, lines in strings.items():
        out += ["", f"## {name}", *lines]
    out += ["", "## 표·선택지·배지 값 (코드 기준)"]
    for group, values in code_values().items():
        out += [f"### {group}", *values]
    (HERE / "visible_strings.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--only", action="append")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--zoom", type=int, default=100, help="app 화면 크기 (%%), same state as the +/- buttons")
    args = parser.parse_args()
    load_placeholders()
    server = None
    url = args.url or f"http://127.0.0.1:{args.port}"
    if not args.url:
        log = open(LOGS / "ui_capture_server.log", "w", encoding="utf-8")
        env = {**os.environ, "TILE_PORT": str(args.port + 1), "PORT": str(args.port)}
        server = subprocess.Popen(["bash", str(HERE / "run.sh")], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_health(url)
        out = args.out if args.out.is_absolute() else ROOT / args.out
        tiles, strings = capture(url, args.only, args.width, args.scale, out, args.zoom)
    finally:
        if server:
            server.terminate()
            server.wait(timeout=30)
    bad = [t for t in tiles if t.get("status") != 200 or t.get("placeholder") or "error" in t]
    sizes = sorted(t["bytes"] for t in tiles if "bytes" in t)
    summary = {"url": url, "tiles": len(tiles), "status_200": sum(t.get("status") == 200 for t in tiles),
               "placeholder_hits": sum(bool(t.get("placeholder")) for t in tiles),
               "relay_sources": {s: sum(t.get("source") == s for t in tiles) for s in ("cache", "fetched", "gray")},
               "hosts": sorted({re.sub(r"^https?://([^/]+)/.*", r"\1", t["url"]) for t in tiles}),
               "bytes_min": sizes[0] if sizes else None, "bytes_median": sizes[len(sizes) // 2] if sizes else None,
               "bytes_max": sizes[-1] if sizes else None, "known_placeholders": sorted(PLACEHOLDER_SHA256.values()),
               "bad": bad[:20]}
    default_out = out.resolve() == OUT.resolve()
    tile_log = LOGS / ("ui_tile_check.json" if default_out else f"ui_tile_check_{out.name}.json")
    tile_log.write_text(json.dumps({"summary": summary, "tiles": tiles}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if not args.only and default_out:
        write_strings(strings)


if __name__ == "__main__":
    main()
