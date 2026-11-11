"""Folium map builders for the three screens (drawing only)."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import folium
import pandas as pd
from branca.element import MacroElement
from folium.plugins import Fullscreen
from jinja2 import Template

# OSM standard tiles through the local caching relay (ui/tile_proxy.py), rendered as a light-gray base.
# (CARTO light_all answers 200 with an "API KEY REQUIRED" placeholder; OSM direct intermittently with "Access blocked".)
TILE_PORT = int(os.environ.get("TILE_PROXY_PORT", "8503"))
ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
CENTER = (36.375, 127.345)
BOUNDARY_FILE = Path(__file__).resolve().parents[1] / "data/processed/routes/yuseong_boundary.geojson"
BOUNDARY = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8")) if BOUNDARY_FILE.exists() else None


def _boundary_bounds():
    if BOUNDARY is None:
        return None
    coords = [pt for f in BOUNDARY["features"] for ring in f["geometry"]["coordinates"] for pt in ring]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    return [(min(lats), min(lons)), (max(lats), max(lons))]


DISTRICT = _boundary_bounds()
FOCUS_ZOOM = 16
GRADE_COLORS = {"상습": "#d03b3b", "반복": "#ec835a", "일반": "#9aa5b1"}
GRADE_RANK = {"일반": 0, "반복": 1, "상습": 2}
SURGE_COLOR = "#7a4fb5"
TEAM_COLORS = ("#1b3561", "#2f6fb3", "#6f93c4")
TEAM_DASH = (None, "10 6", "2 6")
NAVY = "#1b3561"
FONT = "'Noto Sans KR','Pretendard','Noto Sans CJK KR','NanumBarunGothic',sans-serif"
HEAD = f"""<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap">
<style>
.leaflet-tile-pane{{filter:grayscale(1) contrast(.55) brightness(1.24)}}
.leaflet-container,.leaflet-tooltip,.map-legend{{font-family:{FONT}}}
.leaflet-container{{background:#f2f3f5}}
.leaflet-control-attribution{{font-size:10.5px;color:#5d6b7a}}
.leaflet-tooltip{{font-size:12.5px;color:#1e2a38;border:1px solid #d9dfe7;border-radius:3px;box-shadow:0 1px 3px rgba(0,0,0,.12)}}
.map-legend{{position:absolute;left:10px;bottom:10px;z-index:1000;background:#fff;border:1px solid #d9dfe7;border-radius:3px;
  padding:6px 10px;font-size:12px;color:#1e2a38;line-height:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.map-legend .lg{{display:flex;align-items:center;gap:7px;white-space:nowrap}}
.map-legend svg{{flex:none;display:block}}
.map-legend .muted{{color:#5d6b7a}}
.seq{{width:20px;height:20px;border-radius:50%;color:#fff;font:500 10.5px/17px {FONT};text-align:center;
  border:1.5px solid #fff;box-shadow:0 0 0 1px rgba(27,53,97,.35);box-sizing:border-box}}
.seq.on{{width:28px;height:28px;font-size:13px;line-height:24px;border-width:2px;box-shadow:0 0 0 3px {NAVY}}}
.depot{{display:inline-block;background:#132849;color:#fff;font:500 11px/19px {FONT};padding:0 6px;border-radius:2px;
  white-space:nowrap;border:1.5px solid #fff;box-shadow:0 1px 2px rgba(0,0,0,.3)}}
</style>"""


class _NoPrefix(MacroElement):
    """Drop Leaflet's own prefix; the OpenStreetMap credit stays."""
    _template = Template("{% macro script(this, kwargs) %}{{ this._parent.get_name() }}.attributionControl.setPrefix(false);"
                         "{% endmacro %}")


class _RelayTiles(MacroElement):
    """Tile layer served by the relay on the page's own host, so local and network viewers both work."""
    _template = Template("{% macro script(this, kwargs) %}L.tileLayer(location.protocol + '//' + location.hostname + "
                         "':{{ this.port }}/tiles/{z}/{x}/{y}.png', {maxZoom: 19, attribution: {{ this.attr|tojson }}})"
                         ".addTo({{ this._parent.get_name() }});{% endmacro %}")

    def __init__(self, port, attr):
        super().__init__()
        self.port, self.attr = port, attr


class _Legend(MacroElement):
    """Legend inside the map container, so it stays visible in full-screen mode."""
    _template = Template("{% macro script(this, kwargs) %}(function(){var d=document.createElement('div');"
                         "d.className='map-legend';d.innerHTML={{ this.html|tojson }};"
                         "{{ this._parent.get_name() }}.getContainer().appendChild(d);})();{% endmacro %}")

    def __init__(self, html):
        super().__init__()
        self.html = html


def _base(center=CENTER, zoom=12):
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, prefer_canvas=True, zoom_control=True, scrollWheelZoom=True,
                   zoom_snap=0.25)
    m.get_root().header.add_child(folium.Element(HEAD))
    _RelayTiles(TILE_PORT, ATTR).add_to(m)
    _NoPrefix().add_to(m)
    Fullscreen(position="topleft", title="전체 화면", title_cancel="닫기", force_separate_button=True).add_to(m)
    return m


def _dot(color):
    return (f'<svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="{color}" stroke="#fff" stroke-width="1"/>'
            '</svg>')


def _line(color, dash):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<svg width="26" height="8"><line x1="1" y1="4" x2="25" y2="4" stroke="{color}" stroke-width="3"{extra}/></svg>'


def _legend(m, rows):
    _Legend("".join(rows)).add_to(m)


def _outline(m):
    """Fade everything outside 유성구 and draw its boundary as a light dashed line."""
    if BOUNDARY is None:
        return
    holes = [f["geometry"]["coordinates"][0] for f in BOUNDARY["features"] if f["geometry"]["type"] == "Polygon"]
    mask = {"type": "Polygon", "coordinates": [[[120, 30], [135, 30], [135, 42], [120, 42], [120, 30]], *holes]}
    folium.GeoJson(mask, style_function=lambda _: {"stroke": False, "fillColor": "#ffffff", "fillOpacity": 0.55},
                   interactive=False, control=False).add_to(m)
    folium.GeoJson(BOUNDARY, style_function=lambda _: {"color": NAVY, "weight": 1.5, "opacity": 0.6, "dashArray": "6 5",
                                                      "fill": False}, interactive=False, control=False).add_to(m)


def _fit(m, points):
    if points:
        lats, lons = [p[0] for p in points], [p[1] for p in points]
        m.fit_bounds([(min(lats), min(lons)), (max(lats), max(lons))], padding=(18, 18))


def _radius(count):
    return 2.5 + 0.75 * math.sqrt(count)


def spots_map(spots: pd.DataFrame, focus=None, surge_ids=None):
    """Spots sized by 신고 건수 and colored by 지점 등급 (급증 spots in their own color, drawn on top);
    focus = 지점 ID to center and ring."""
    m = _base()
    _outline(m)
    surge_ids = set(surge_ids or ())
    frame = spots.dropna(subset=["위도", "경도"]).copy()
    frame["_surge"] = frame["지점 ID"].isin(surge_ids)
    frame["_rank"] = frame["지점 등급"].map(GRADE_RANK) + 3 * frame["_surge"]
    frame = frame.sort_values(["_rank", "신고 건수"], ascending=[True, False])
    for lat, lon, count, grade, label, hot in frame[["위도", "경도", "신고 건수", "지점 등급", "위치", "_surge"]].itertuples(index=False):
        color = SURGE_COLOR if hot else GRADE_COLORS[grade]
        folium.CircleMarker((lat, lon), radius=_radius(count), color="#ffffff", weight=1,
                            fill=True, fill_color=color, fill_opacity=0.62 if grade == "일반" and not hot else 0.85,
                            tooltip=f"{label} · {grade}{' · 급증' if hot else ''} · {int(count):,}건").add_to(m)
    target = frame.loc[frame["지점 ID"].eq(focus)] if focus is not None else frame.iloc[:0]
    if target.empty:
        if DISTRICT:
            m.fit_bounds(DISTRICT, padding=(4, 4))
        else:
            _fit(m, list(zip(frame["위도"], frame["경도"])))
    else:
        t = target.iloc[0]
        point = (float(t["위도"]), float(t["경도"]))
        folium.CircleMarker(point, radius=_radius(t["신고 건수"]) + 6, color=NAVY, weight=3, fill=False,
                            tooltip=f"{t['위치']} · {t['지점 등급']} · {int(t['신고 건수']):,}건").add_to(m)
        m.fit_bounds([point, point], max_zoom=FOCUS_ZOOM)
    rows = [f'<div class="lg">{_dot(SURGE_COLOR)}급증</div>'] if frame["_surge"].any() else []
    rows += [f'<div class="lg">{_dot(GRADE_COLORS[g])}{g}</div>' for g in ("상습", "반복", "일반")
             if g in set(frame.loc[~frame["_surge"], "지점 등급"])]
    rows.append('<div class="lg muted">원 크기: 신고 건수</div>')
    _legend(m, rows)
    return m


def _vehicle(value):
    """Style slot for a 단속 차량 value such as '2호차'."""
    digits = re.sub(r"\D", "", str(value))
    return (int(digits) - 1 if digits else 0) % len(TEAM_COLORS)


def plan_map(visits: pd.DataFrame, lines: pd.DataFrame, depot, focus=None):
    """Routes per 단속 차량 with numbered stops; focus = (단속 차량, 순번) to center and enlarge."""
    m = _base(depot, 13)
    _outline(m)
    points = [depot]
    for vehicle, path in lines[["단속 차량", "경로 좌표"]].itertuples(index=False):
        if len(path) >= 2:
            k = _vehicle(vehicle)
            folium.PolyLine(path, color=TEAM_COLORS[k], weight=4, opacity=0.85, dash_array=TEAM_DASH[k]).add_to(m)
            points.extend(path)
    target = None
    for vehicle, seq, label, arrival, lat, lon in visits[["단속 차량", "순번", "위치", "도착 예정", "위도", "경도"]].itertuples(index=False):
        k = _vehicle(vehicle)
        on = focus is not None and (vehicle, int(seq)) == tuple(focus)
        size = 28 if on else 20
        html = f'<div class="seq{" on" if on else ""}" style="background:{TEAM_COLORS[k]}">{int(seq)}</div>'
        marker = folium.Marker((lat, lon), icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size // 2, size // 2)),
                               tooltip=f"{vehicle} {int(seq)}번 · {label} · {pd.Timestamp(arrival):%H:%M}",
                               z_index_offset=1000 if on else 0)
        marker.add_to(m)
        if on:
            target = (float(lat), float(lon))
    folium.Marker(depot, icon=folium.DivIcon(html='<div class="depot">구청</div>', icon_size=(36, 20), icon_anchor=(18, 10)),
                  tooltip="유성구청").add_to(m)
    if target is None:
        _fit(m, points) if len(points) > 1 or not DISTRICT else m.fit_bounds(DISTRICT, padding=(10, 10))
    else:
        m.fit_bounds([target, target], max_zoom=FOCUS_ZOOM)
    vehicles = sorted(set(visits["단속 차량"]) | set(lines["단속 차량"]), key=_vehicle)
    rows = [f'<div class="lg">{_line(TEAM_COLORS[_vehicle(v)], TEAM_DASH[_vehicle(v)])}{v}</div>' for v in vehicles]
    rows.append('<div class="lg"><span class="depot" style="font-size:10px;line-height:15px">구청</span>출발·복귀</div>')
    _legend(m, rows)
    return m


def detail_map(point, grade):
    if point is None:
        return _base()
    m = _base(point, FOCUS_ZOOM)
    folium.Circle(point, radius=60, color=NAVY, weight=1.5, fill=True, fill_opacity=0.08).add_to(m)
    folium.CircleMarker(point, radius=9, color="#ffffff", weight=2.5, fill=True, fill_color=NAVY, fill_opacity=1,
                        tooltip=grade).add_to(m)
    return m
