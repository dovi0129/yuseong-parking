"""Streamlit entry point for the parking-complaint support screens. Launch with ui/run.sh."""
from __future__ import annotations

import html
import io
import math
from datetime import date, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

import brand

SITE = brand.settings()
SYSTEM = SITE["system_name"]
st.set_page_config(page_title=SYSTEM, layout="wide", initial_sidebar_state="collapsed")

import backend as be  # noqa: E402
import maps  # noqa: E402
import photo  # noqa: E402

HERE = Path(__file__).resolve().parent
MENUS = {"overview": "현황", "plan": "순회 계획", "reception": "민원 접수"}
TABS = {"overview": ("지도", "명단"), "plan": ("지도", "방문표"), "reception": ("접수 목록", "사진 검토")}
TAB_QUERY = {"map": 0, "list": 1, "photo": 1}
SCALES = (80, 90, 100, 110, 120, 130, 140)
GRADE_BADGE = {"상습": "g-h", "반복": "g-r", "일반": "g-n"}
GRADE_FILL = {"상습": "#fbe9e9", "반복": "#fdeee6"}
EMPTY = "조회 결과가 없습니다."
ROW_PX = 33
DEFAULT_DAY = date(2024, 10, 2)
STEP_MIN = 10
DAY_END = time(23, 50)
SPOT_VIEW = ["위치", "동", "지점 등급", "신고 건수", "단속 수", "위험 가중치"]
PLAN_VIEW = ["단속 차량", "순번", "위치", "도착 예정", "기대 신고 수", "위험 가중치"]
RECEPTION_VIEW = ["접수 일시", "위치", "지점 등급", "금일 순번", "묶음 번호", "조치 구분", "시설 개선 검토", "사진 검토"]
PAGE_SIZE = {"overview": 15, "plan": 15, "reception": 10}
COND = [1, 2.4, 1, 2.4]
HOME = ('<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><path d="M12 3 2 11.5h3V21h5.5v-6h3v6H19v-9.5h3z" '
        'fill="#555"/></svg>')
CARET = '<svg xmlns="http://www.w3.org/2000/svg" width="11" height="8" viewBox="0 0 11 8"><path d="M0 0h11L5.5 7z" fill="#555"/></svg>'


# common parts ------------------------------------------------------------------

def _zoom(step):
    i = SCALES.index(st.session_state.get("scale", 100))
    st.session_state.scale = SCALES[min(max(i + step, 0), len(SCALES) - 1)]


PERSON = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4" '
          'fill="none" stroke="#555" stroke-width="2"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7" fill="none" stroke="#555" '
          'stroke-width="2" stroke-linecap="round"/></svg>')


def _logout():
    """No authentication: just return to the first 현황 state."""
    for key in list(st.session_state):
        del st.session_state[key]
    st.query_params.clear()


def style():
    scale = st.session_state.get("scale", 100)
    css = (HERE / "theme.css").read_text(encoding="utf-8")
    st.html(f"<style>{css}\n[data-testid='stMainBlockContainer']{{zoom:{scale / 100}; --zoom:{scale / 100}}}</style>")


def header():
    scale = st.session_state.get("scale", 100)
    with st.container(key="head", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.html(brand.logo_img(SYSTEM, 34), width="content")
        with st.container(key="util", horizontal=True, vertical_alignment="center", width="content"):
            with st.container(key="account", horizontal=True, vertical_alignment="center", width="content"):
                st.html(f'<span class="util-label user">{brand.icon_img(PERSON, "person")}{html.escape(SITE["user_label"])}</span>',
                        width="content")
                st.html('<span class="util-sep">|</span>', width="content")
                st.button("로그아웃", key="logout", on_click=_logout, type="tertiary")
            with st.container(key="zoom", horizontal=True, vertical_alignment="center", width="content"):
                st.html('<span class="util-label">화면 크기</span>', width="content")
                st.button("+", key="zoom_in", on_click=_zoom, args=(1,), disabled=scale == SCALES[-1])
                st.html(f'<span class="util-value">{scale}%</span>', width="content")
                st.button("−", key="zoom_out", on_click=_zoom, args=(-1,), disabled=scale == SCALES[0])


def gnb():
    labels = list(MENUS.values())
    if "menu" not in st.session_state:
        st.session_state.menu = MENUS.get(st.query_params.get("menu", "overview"), labels[0])
        st.session_state.start_tab = st.query_params.get("tab", "")
    with st.container(key="gnb"):
        label = st.radio("메뉴", labels, key="menu", horizontal=True, label_visibility="collapsed")
    page = list(MENUS)[labels.index(label)]
    if st.query_params.get("menu") != page:
        st.query_params["menu"] = page
    return page


def current_tab(page):
    key = f"{page}_tab"
    if key not in st.session_state:
        st.session_state[key] = TABS[page][TAB_QUERY.get(st.session_state.pop("start_tab", ""), 0)]
    return st.session_state[key]


def split_tabs(page):
    with st.container(key=f"tabs_{page}"):
        st.radio("화면", TABS[page], key=f"{page}_tab", horizontal=True, label_visibility="collapsed")


def crumb(page, tab):
    home, caret = brand.icon_img(HOME, "home"), brand.icon_img(CARET, "caret")
    st.html(f'<div class="crumb"><span>{home}</span><span>{MENUS[page]}{caret}</span><span>{tab}{caret}</span></div>')


def title(text):
    st.html(f'<div class="page-title">{text}</div>')


def label(col, text, required=False):
    col.html(f'<div class="cl">{text}{"<span class=req></span>" if required else ""}</div>')


def search_button(page, on_click=None):
    with st.container(key=f"search_{page}", horizontal=True, horizontal_alignment="center"):
        return st.form_submit_button("조회", icon=":material/search:", type="primary", on_click=on_click)


def stat_tiles(items):
    cells = "".join(f'<div class="kpi"><div class="l">{name}</div><div class="v">{value}<span class="u">{unit}</span></div></div>'
                    for name, value, unit in items)
    st.html(f'<div class="kpis">{cells}</div>')


def result_head(name, n, data, file_name, controls=None):
    with st.container(key=f"reshead_{name}", horizontal=True, horizontal_alignment="distribute", vertical_alignment="bottom"):
        st.html(f'<div class="total">총 <span class="n">{n:,}</span>건</div>', width="content")
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", width="content", gap="small"):
            if controls:
                controls()
            st.download_button("엑셀 내려받기", data, file_name=file_name, mime="text/csv", on_click="ignore",
                               icon=":material/download:", key=f"csv_{name}")


def empty():
    st.html(f'<div class="empty">{EMPTY}</div>')


def table_height(rows):
    return ROW_PX * (rows + 1) + 3


def grade_style(frame, formats=None):
    styler = frame.style.map(lambda g: f"background-color:{GRADE_FILL[g]}" if g in GRADE_FILL else "", subset=["지점 등급"])
    return styler.format(formats or {})


def _goto(key, page):
    st.session_state[key] = page


def page_slice(frame, key, size):
    pages = max(1, math.ceil(len(frame) / size))
    page = min(max(1, st.session_state.get(key, 1)), pages)
    st.session_state[key] = page
    start = (page - 1) * size
    return frame.iloc[start:start + size], start, page, pages


def pager(name, key, page, pages):
    if pages <= 1:
        return
    first = (page - 1) // 10 * 10 + 1
    last = min(first + 9, pages)
    with st.container(horizontal=True, horizontal_alignment="center", gap="small", key=f"pgbar_{name}"):
        st.button("«", key=f"pg_{name}_first", on_click=_goto, args=(key, 1), disabled=page == 1, type="tertiary")
        st.button("‹", key=f"pg_{name}_prev", on_click=_goto, args=(key, first - 1), disabled=first == 1, type="tertiary")
        for n in range(first, last + 1):
            st.button(str(n), key=f"pg_{name}_{n}", on_click=_goto, args=(key, n), type="primary" if n == page else "tertiary")
        st.button("›", key=f"pg_{name}_next", on_click=_goto, args=(key, last + 1), disabled=last == pages, type="tertiary")
        st.button("»", key=f"pg_{name}_last", on_click=_goto, args=(key, pages), disabled=page == pages, type="tertiary")


def _pick(table_key, values, focus_key, tab_key):
    """Row selection in a list tab: remember the row and switch to the map tab."""
    rows = st.session_state[table_key].selection.rows
    if rows:
        st.session_state[focus_key] = values[rows[0]]
        st.session_state[tab_key] = "지도"


def plan_settings():
    return st.session_state.setdefault("plan_settings", {"day": DEFAULT_DAY, "shift": "17:00~21:00", "vehicles": 1, "risk": True})


# 현황 -------------------------------------------------------------------------

def page_overview(ver, tab):
    with st.container(key="cond_overview"):
        with st.form("ov_form", border=False, enter_to_submit=False):
            c = st.columns(COND, gap=None, vertical_alignment="center")
            label(c[0], "기간", True)
            period = c[1].date_input("기간", (be.YEAR_START, be.YEAR_END), min_value=be.YEAR_START, max_value=be.YEAR_END,
                                     format="YYYY-MM-DD", key="ov_period", label_visibility="collapsed")
            label(c[2], "동")
            dong = c[3].selectbox("동", ["전체", *be.dongs(ver)], key="ov_dong", label_visibility="collapsed")
            c = st.columns([1, 5.8], gap=None, vertical_alignment="center")
            label(c[0], "지점 등급")
            grades = c[1].multiselect("지점 등급", be.GRADES, default=list(be.GRADES), key="ov_grades", placeholder="",
                                      label_visibility="collapsed")
            if search_button("overview"):
                st.session_state.pop("ov_focus", None)
    start, end = (period[0], period[-1]) if len(period) else (be.YEAR_START, be.YEAR_END)
    data = be.overview(ver, start, end, None if dong == "전체" else dong, tuple(grades))
    s = data["summary"].iloc[0]
    stat_tiles([("신고 건수", f"{int(s['신고 건수']):,}", "건"),
                ("반복 지점 수", f"{int(s['반복 지점 수']):,}", "곳"),
                ("상습 지점 수", f"{int(s['상습 지점 수']):,}", "곳"),
                ("상위 100개 지점 비중", f"{float(s['상위 100개 지점 비중 (%)']):.1f}", "%")])
    spots = data["spots"]
    if tab == "지도":
        with st.container(key="mapwrap_overview"):
            st_folium(maps.spots_map(spots, st.session_state.get("ov_focus")), height=720, use_container_width=True,
                      returned_objects=[], key="ov_map")
        return
    result_head("overview", len(spots), be.csv(spots), f"지점목록_{start:%Y%m%d}-{end:%Y%m%d}.csv")
    if spots.empty:
        empty()
        return
    key = f"ov_page:{start}:{end}:{dong}:{','.join(grades)}"
    part, _, page, pages = page_slice(spots, key, PAGE_SIZE["overview"])
    table_key = f"ov_table_{page}"
    ids = list(part["지점 ID"])
    st.dataframe(grade_style(part[SPOT_VIEW], {"위험 가중치": "{:.1f}"}), height=table_height(len(part)), hide_index=True,
                 row_height=ROW_PX, on_select=lambda: _pick(table_key, ids, "ov_focus", "overview_tab"), selection_mode="single-row",
                 key=table_key, column_config={"신고 건수": st.column_config.NumberColumn(format="%d"),
                                               "단속 수": st.column_config.NumberColumn(format="%d")})
    pager("overview", key, page, pages)


# 순회 계획 ----------------------------------------------------------------------

def page_plan(ver, tab):
    cur = plan_settings()
    with st.container(key="cond_plan"):
        with st.form("plan_form", border=False, enter_to_submit=False):
            c = st.columns(COND, gap=None, vertical_alignment="center")
            label(c[0], "날짜", True)
            day = c[1].date_input("날짜", cur["day"], min_value=be.EVAL_START, max_value=be.EVAL_END, format="YYYY-MM-DD",
                                  label_visibility="collapsed")
            label(c[2], "근무 구간", True)
            shift = c[3].selectbox("근무 구간", list(be.SHIFTS), index=list(be.SHIFTS).index(cur["shift"]),
                                   label_visibility="collapsed")
            c = st.columns(COND, gap=None, vertical_alignment="center")
            label(c[0], "차량 수", True)
            vehicles = c[1].selectbox("차량 수", (1, 2, 3), index=cur["vehicles"] - 1, format_func=lambda n: f"{n}대",
                                      label_visibility="collapsed")
            label(c[2], "위험 가중")
            risk = c[3].selectbox("위험 가중", ("적용", "미적용"), index=0 if cur["risk"] else 1, label_visibility="collapsed")
            if search_button("plan"):
                cur.update(day=day, shift=shift, vehicles=vehicles, risk=risk == "적용")
                st.session_state.pop("plan_focus", None)
    args = (ver, cur["day"], cur["shift"], cur["vehicles"], cur["risk"])
    with st.spinner(""):
        visits = be.plan(*args)
        lines = be.routes(*args)
    if visits.attrs["summary"]["status"] != "ok":
        visits, lines = visits.iloc[:0], lines.iloc[:0]
    if tab == "지도":
        with st.container(key="mapwrap_plan"):
            st_folium(maps.plan_map(visits, lines, be.depot(), st.session_state.get("plan_focus")), height=640,
                      use_container_width=True, returned_objects=[], key="plan_map")
        return
    result_head("plan", len(visits), be.csv(visits),
                f"순회계획_{cur['day']:%Y%m%d}_{cur['shift'].replace(':', '').replace('~', '-')}_{cur['vehicles']}대.csv")
    if visits.empty:
        empty()
        return
    key = f"plan_page:{cur['day']}:{cur['shift']}:{cur['vehicles']}:{cur['risk']}"
    part, _, page, pages = page_slice(visits, key, PAGE_SIZE["plan"])
    table = part.assign(**{"도착 예정": pd.to_datetime(part["도착 예정"]).dt.strftime("%H:%M")})
    table_key = f"plan_table_{page}"
    stops = [(v, int(n)) for v, n in zip(part["단속 차량"], part["순번"])]
    st.dataframe(table[PLAN_VIEW], height=table_height(len(table)), hide_index=True, row_height=ROW_PX,
                 on_select=lambda: _pick(table_key, stops, "plan_focus", "plan_tab"), selection_mode="single-row",
                 key=table_key, column_config={"기대 신고 수": st.column_config.NumberColumn(format="%.2f"),
                                               "위험 가중치": st.column_config.NumberColumn(format="%.1f")})
    pager("plan", key, page, pages)


# 민원 접수 ----------------------------------------------------------------------

def _minutes(value: time):
    return value.hour * 60 + value.minute


def _clock(minutes):
    return time(minutes // 60, minutes % 60)


def _play():
    if _minutes(st.session_state.rc_time) >= _minutes(DAY_END):
        st.session_state.rc_time = time(0, 0)
    st.session_state.rc_play = True
    st.session_state.rc_sel = None
    st.session_state.rc_full_rerun = True


def _stop():
    st.session_state.rc_play = False
    st.session_state.rc_full_rerun = True


def _apply_reception():
    st.session_state.rc_day = st.session_state.rc_day_in
    st.session_state.rc_time = st.session_state.rc_time_in


def playback():
    playing = st.session_state.rc_play
    st.button("재생", key="rc_play_btn", on_click=_play, disabled=playing, icon=":material/play_arrow:")
    st.button("정지", key="rc_stop_btn", on_click=_stop, disabled=not playing, icon=":material/stop:")


def received_rows(ver, cur):
    day = st.session_state.rc_day
    received = be.reception(ver, day, cur["shift"], cur["vehicles"], cur["risk"], st.session_state.rc_time).reset_index(drop=True)
    keys = [f"reply_{day:%Y%m%d}_{i}" for i in range(len(received))]
    reviews = st.session_state.setdefault("photo_reviews", {})
    edited = received.assign(**{"회신 내용": [st.session_state.get(k, v) for k, v in zip(keys, received["회신 내용"])],
                                "사진 검토": [photo.summary(reviews.get(k)) for k in keys]})
    return day, received, edited, keys


def reception_body(ver, cur, tab):
    if st.session_state.pop("rc_full_rerun", False):
        st.rerun()
    if st.session_state.rc_play:
        now = _minutes(st.session_state.rc_time)
        if now >= _minutes(DAY_END):
            st.session_state.rc_play = False
            st.rerun()
        st.session_state.rc_time = _clock(min(now + STEP_MIN, _minutes(DAY_END)))
    if tab == "접수 목록":
        # rc_day/rc_time are plain state (they survive the 사진 검토 tab); the form widgets mirror them.
        st.session_state.rc_day_in, st.session_state.rc_time_in = st.session_state.rc_day, st.session_state.rc_time
        with st.container(key="cond_reception"):
            with st.form("rc_form", border=False, enter_to_submit=False):
                c = st.columns(COND, gap=None, vertical_alignment="center")
                label(c[0], "날짜", True)
                c[1].date_input("날짜", min_value=be.EVAL_START, max_value=be.EVAL_END, format="YYYY-MM-DD", key="rc_day_in",
                                label_visibility="collapsed")
                label(c[2], "기준 시각")
                c[3].slider("기준 시각", min_value=time(0, 0), max_value=DAY_END, step=timedelta(minutes=STEP_MIN),
                            format="HH:mm", key="rc_time_in", label_visibility="collapsed")
                search_button("reception", _apply_reception)
    day, received, edited, keys = received_rows(ver, cur)
    if st.session_state.get("rc_last_day") != day:
        st.session_state.rc_last_day, st.session_state.rc_sel = day, None
    sel = st.session_state.get("rc_sel")
    if sel is not None and sel >= len(edited):
        sel = None
    if tab == "접수 목록":
        reception_list(ver, day, received, edited, keys, sel)
    else:
        reception_photo(ver, edited, keys, sel)


def reception_list(ver, day, received, edited, keys, sel):
    result_head("reception", len(edited), be.csv(edited), f"민원접수_{day:%Y%m%d}_{st.session_state.rc_time:%H%M}.csv", playback)
    if edited.empty:
        empty()
        return
    view = edited.iloc[::-1][RECEPTION_VIEW].copy()
    view["접수 일시"] = pd.to_datetime(view["접수 일시"]).dt.strftime("%Y-%m-%d %H:%M")
    page_key = f"rc_page:{day}"
    part, offset, page, pages = page_slice(view, page_key, PAGE_SIZE["reception"])
    shown = None if sel is None else len(edited) - 1 - sel - offset
    default = {"selection": {"rows": [shown], "columns": []}} if shown is not None and 0 <= shown < len(part) else None
    event = st.dataframe(grade_style(part), height=table_height(len(part)), hide_index=True, row_height=ROW_PX, on_select="rerun",
                         selection_mode="single-row", key=f"rc_table_{day:%Y%m%d}_{len(edited)}_{page}",
                         selection_default=default)
    if event.selection.rows:
        sel = len(edited) - 1 - (offset + event.selection.rows[0])
        st.session_state.rc_sel = sel
    pager("reception", page_key, page, pages)
    idx = len(edited) - 1 if sel is None else sel
    row = edited.iloc[idx]
    with st.container(key="detail_reception"):
        left, right = st.columns([1, 1.35], gap="medium")
        with left:
            st_folium(maps.detail_map(be.spot_point(ver, row["지점 ID"]), row["지점 등급"]), height=318,
                      use_container_width=True, returned_objects=[], key="rc_map")
        with right:
            arrival = pd.Timestamp(row["도착 예정"]).strftime("%H:%M") if isinstance(row["도착 예정"], str) and row["도착 예정"] else "-"
            fields = [("접수 일시", pd.Timestamp(row["접수 일시"]).strftime("%Y-%m-%d %H:%M")), ("금일 순번", f"{int(row['금일 순번'])}"),
                      ("지점 등급", grade_badge(row["지점 등급"])), ("도착 예정", arrival),
                      ("조치 구분", html.escape(str(row["조치 구분"]))), ("시설 개선 검토", html.escape(str(row["시설 개선 검토"])))]
            cells = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in fields)
            st.html(f'<div class="kv"><div class="k">위치</div><div class="v wide">{html.escape(str(row["위치"]))}</div>{cells}'
                    f'<div class="k">묶음 번호</div><div class="v wide">{html.escape(str(row["묶음 번호"]))}</div></div>')
            st.text_area("회신 내용", value=received.iloc[idx]["회신 내용"], key=keys[idx], height=96)


def grade_badge(grade):
    return f'<span class="badge {GRADE_BADGE.get(grade, "g-n")}">{html.escape(str(grade))}</span>'


def reception_photo(ver, edited, keys, sel):
    if edited.empty:
        empty()
        return
    idx = len(edited) - 1 if sel is None else sel
    row, key = edited.iloc[idx], keys[idx]
    st.html(f'<div class="kv"><div class="k">접수 일시</div><div class="v">{pd.Timestamp(row["접수 일시"]):%Y-%m-%d %H:%M}</div>'
            f'<div class="k">지점 등급</div><div class="v">{grade_badge(row["지점 등급"])}</div>'
            f'<div class="k">위치</div><div class="v">{html.escape(str(row["위치"]))}</div>'
            f'<div class="k">묶음 번호</div><div class="v">{html.escape(str(row["묶음 번호"]))}</div></div>')
    pairs = photo.demo_pairs()
    labels = dict(zip(pairs["pair_id"], pairs["label"]))
    with st.container(key="cond_photo"):
        with st.form(f"photo_form_{key}", border=False, enter_to_submit=False):
            c = st.columns(COND, gap=None, vertical_alignment="center")
            label(c[0], "시연 사진")
            pair_id = c[1].selectbox("시연 사진", list(labels), index=None, format_func=labels.get, placeholder="",
                                     key=f"pp_{key}", label_visibility="collapsed")
            label(c[2], "사진 파일")
            files = c[3].file_uploader("사진 파일", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"pu_{key}",
                                       label_visibility="collapsed")
            submitted = search_button("photo")
    reviews = st.session_state.photo_reviews
    if submitted:
        spot = be.spot_point(ver, row["지점 ID"])
        if files and len(files) == 2:
            data = [f.getvalue() for f in files]
            result = photo.review_upload(data[0], files[0].name, data[1], files[1].name, spot)
            result = {**result, "_images": ("upload", tuple(data))}
        elif pair_id is not None:
            pair = pairs.set_index("pair_id").loc[pair_id]
            has_files = isinstance(pair.path1, str) and isinstance(pair.path2, str)
            # Demo pairs carry their own report location; a missing one goes to the review as None (Worker_2 contract).
            reported = ((float(pair.reported_lat), float(pair.reported_lon))
                        if pd.notna(pair.reported_lat) and pd.notna(pair.reported_lon) else None)
            result = photo.review_demo(pair_id, pair.path1 if has_files else None, pair.path2 if has_files else None,
                                       pair.case, reported)
            result = {**result, "_images": ("demo", (pair.path1, pair.path2) if has_files else None)}
        else:
            result = None
        if result is None:
            reviews.pop(key, None)
        else:
            reviews[key] = result
    result = reviews.get(key)
    if result is None:
        return
    kind, source = result["_images"]
    if kind == "upload":
        images = [Image.open(io.BytesIO(d)) for d in source]
    elif source:
        images = [Image.open(photo._resolve(path)) for path in source]
    else:
        images = [photo.interim_image(result["plate_1"]), photo.interim_image(result["plate_2"], 14)]
    with st.container(key="detail_photo"):
        a, b, c = st.columns([1, 1, 1.3], gap="medium")
        a.image(photo.with_box(images[0], result["box_1"]), width="stretch")
        b.image(photo.with_box(images[1], result["box_2"]), width="stretch")
        rows = "".join(f'<tr><td class="k">{k}</td><td class="v">{html.escape(v)}</td>'
                       f'<td class="{"ok" if r in ("일치", "적합") else "ng"}">{r}</td></tr>'
                       for k, v, r in photo.check_rows(result))
        verdict = "ok" if result["verdict"] == "요건 충족" else "ng"
        reasons = ", ".join(result["reasons"]) or "-"
        c.html(f'<table class="chk"><thead><tr><th>검토 항목</th><th>내용</th><th>결과</th></tr></thead><tbody>{rows}</tbody></table>'
               f'<div class="kv two"><div class="k">판정</div><div><span class="verdict {verdict}">{result["verdict"]}</span></div>'
               f'<div class="k">사유</div><div class="v">{html.escape(reasons)}</div></div>')


def page_reception(ver, tab):
    cur = plan_settings()
    st.session_state.setdefault("rc_day", DEFAULT_DAY)
    st.session_state.setdefault("rc_time", DAY_END)
    st.session_state.setdefault("rc_play", False)
    st.session_state.setdefault("rc_sel", None)
    with st.spinner(""):
        be.plan(ver, st.session_state.rc_day, cur["shift"], cur["vehicles"], cur["risk"])
    st.fragment(run_every=1.0 if st.session_state.rc_play else None)(reception_body)(ver, cur, tab)


def initial_zoom():
    """'zoom' in the address sets the same 화면 크기 state as the +/- buttons, once per session."""
    if "scale" not in st.session_state:
        value = st.query_params.get("zoom", "")
        if value.isdigit() and int(value) in SCALES:
            st.session_state.scale = int(value)


def main():
    initial_zoom()
    style()
    header()
    page = gnb()
    tab = current_tab(page)
    crumb(page, tab)
    title(MENUS[page])
    split_tabs(page)
    ver = be.version()
    {"overview": page_overview, "plan": page_plan, "reception": page_reception}[page](ver, tab)
    st.html(brand.footer_html(SYSTEM, SITE["team_name"], photo.ai_photos()))


main()
