"""Render 5.5-inch/200dpi analysis figures exclusively from saved tables."""
from __future__ import annotations

import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import paper_figstyle as ps
from .common import FIG, OUT, REF, write_frame, write_json
from .prepare import normalize_location
from .basemap import WEB, gray_backdrop

TABLE = OUT / "analysis_tables"


def save(fig, name: str, note: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(FIG / f"analysis_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    captions_path = OUT / "analysis_figure_captions.json"
    captions = json.loads(captions_path.read_text()) if captions_path.exists() else {}
    captions[f"analysis_{name}.png"] = {"caption": note, "width_in": 5.5, "dpi": 200,
                                        "source_table": str((TABLE / f"{name}.csv").relative_to(OUT.parent.parent))}
    write_json(captions, captions_path)


def concentration(reports, enforcement):
    rows = []
    for label, keys in [("신고 2024", reports.spot_id), ("단속 2023", enforcement.raw_loc.map(lambda s: normalize_location(s)[0]))]:
        counts = keys.value_counts().sort_values(ascending=False)
        for i, n in enumerate(counts.cumsum(), start=1):
            rows.append({"series": label, "spot_fraction": i / len(counts), "cumulative_fraction": n / counts.sum(), "n_spots": len(counts), "n_records": int(counts.sum())})
    path = TABLE / "concentration.csv"
    write_frame(pd.DataFrame(rows), path)
    ps.apply()
    fig, ax = plt.subplots()
    annotations = []
    ax.axvline(10, color="#888888", ls=":", lw=.8, zorder=0)
    for (series, frame), color, linestyle in zip(pd.read_csv(path).groupby("series", sort=False), ["#0072B2", "#D55E00"], ["-", "--"]):
        ax.plot(frame.spot_fraction * 100, frame.cumulative_fraction * 100, label=series, color=color, ls=linestyle)
        n_top = int(np.ceil(.1 * int(frame.n_spots.iloc[0])))
        point = frame.iloc[n_top - 1]
        x, y = point.spot_fraction * 100, point.cumulative_fraction * 100
        ax.plot(x, y, marker="o", color=color, ms=5, zorder=4)
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(12, 9 if series == "단속 2023" else -17),
                    textcoords="offset points", color=color, fontsize=10, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.5})
        annotations.append({"series": series, "n_top10_spots": n_top, "n_spots": int(point.n_spots),
                            "spot_fraction": point.spot_fraction, "cumulative_fraction": point.cumulative_fraction})
    ax.set(xlabel="상위 지점 비율 (%)", ylabel="누적 건수 비율 (%)", xlim=(0, 100), ylim=(0, 100))
    ax.set_xticks([0, 10, 20, 40, 60, 80, 100])
    ax.legend(loc="lower right")
    write_frame(pd.DataFrame(annotations), TABLE / "concentration_annotations.csv")
    save(fig, "concentration", "정규화한 지점별 빈도를 내림차순으로 누적. 상위 10%는 지점 수의 10%를 올림한 개수이며 점과 숫자로 표시한다. 단속 2023년, 신고 2024년은 서로 다른 기간이다.")


def heatmap(reports):
    frame = reports.assign(weekday=reports.ts.dt.dayofweek).groupby(["weekday", "hour"]).size().unstack(fill_value=0).reindex(index=range(7), columns=range(24), fill_value=0)
    write_frame(frame.rename_axis("weekday").reset_index(), TABLE / "weekday_hour.csv")
    values = pd.read_csv(TABLE / "weekday_hour.csv").set_index("weekday").to_numpy()
    ps.apply(.7)
    fig, ax = plt.subplots()
    im = ax.imshow(values, aspect="auto", cmap="YlOrRd", origin="upper")
    ax.set(xticks=range(0, 24, 2), yticks=range(7), yticklabels=list("월화수목금토일"), xlabel="신고 시각 (시)", ylabel="요일")
    fig.colorbar(im, ax=ax, label="과태료 처리 신고 (건)", shrink=.85)
    save(fig, "weekday_hour", f"2024년 과태료 처리 신고 {len(reports):,}건의 실제 요일과 시간. 공휴일 재분류는 우선순위 점수에만 적용한다.")


def repetition(reports):
    groups = reports.groupby(["spot_id", "date"]).size()
    frame = groups.value_counts().sort_index().rename_axis("reports_per_spot_day").rename("n_spot_days").reset_index()
    frame["n_reports"] = frame.reports_per_spot_day * frame.n_spot_days
    write_frame(frame, TABLE / "same_day_repetition.csv")
    frame = pd.read_csv(TABLE / "same_day_repetition.csv")
    ps.apply()
    fig, ax = plt.subplots()
    ax.bar(frame.reports_per_spot_day, frame.n_spot_days, color="#0072B2", width=.85)
    ax.set(xlabel="같은 지점·같은 날 신고 수 (건)", ylabel="지점·날짜 조합 수 (개)", yscale="log")
    save(fig, "same_day_repetition", "고정 위치 정규화 후 지점·날짜별 신고 건수의 분포. 세로축은 로그 눈금이다.")


def base_map(spots):
    ps.apply(1.12)
    fig, ax = plt.subplots()
    boundary = gpd.read_file(REF / "yuseong_boundary.geojson")
    west, south, east, north = boundary.total_bounds
    bounds = (west - .008, south - .005, east + .012, north + .005)
    background, extent = gray_backdrop(bounds)
    ax.imshow(background, extent=extent, zorder=0)
    boundary.to_crs(3857).boundary.plot(ax=ax, color="#6f777d", linewidth=.8, zorder=1)
    x0, y0 = WEB.transform(bounds[0], bounds[1])
    x1, y1 = WEB.transform(bounds[2], bounds[3])
    ax.set(xlim=(x0, x1), ylim=(y0, y1))
    ax.set_axis_off()
    ax.text(.01, .007, "© OpenStreetMap contributors", transform=ax.transAxes, fontsize=7,
            color="#555555", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    weights = spots.assign(n_reports=spots.n_reports_train + spots.n_reports_test)
    leading = weights.groupby("dong").n_reports.sum().nlargest(5).index
    offsets = {"봉명동": (2200, -1700), "관평동": (1500, 1500), "노은동": (-2600, -1000),
               "지족동": (-2200, 1500), "신성동": (1600, 1300)}
    for dong in leading:
        rows = weights[weights.dong == dong]
        lon = np.average(rows.lon, weights=rows.n_reports)
        lat = np.average(rows.lat, weights=rows.n_reports)
        x, y = WEB.transform(lon, lat)
        dx, dy = offsets.get(dong, (1500, 1200))
        ax.annotate(dong, xy=(x, y), xytext=(x + dx, y + dy), fontsize=9, fontweight="bold",
                    color="#222222", ha="center", va="center", zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "alpha": .96, "boxstyle": "round,pad=.25", "lw": .5},
                    arrowprops={"arrowstyle": "-", "color": "#555555", "lw": .7})
    return fig, ax


def maps(reports, spots):
    valid = spots[spots.lat.notna() & spots.lon.notna()].copy()
    if valid.empty:
        return
    repeated = reports.groupby(["spot_id", "date"]).size().reset_index(name="n")
    repeated = repeated[repeated.n >= 2].groupby("spot_id").n.sum()
    valid["repeated_reports"] = valid.spot_id.map(repeated).fillna(0)
    write_frame(valid[["spot_id", "label", "lat", "lon", "repeated_reports"]], TABLE / "repeat_map.csv")
    data = pd.read_csv(TABLE / "repeat_map.csv")
    fig, ax = base_map(valid)
    positive = data[data.repeated_reports > 0]
    bins = np.digitize(positive.repeated_reports, [10, 20, 50, 100])
    colors = np.array(["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"])
    sizes = np.array([7, 14, 25, 39, 57])
    x, y = WEB.transform(positive.lon.to_numpy(), positive.lat.to_numpy())
    ax.scatter(x, y, s=sizes[bins], c=colors[bins], linewidths=.2, edgecolors="white", alpha=.88, zorder=3)
    labels = ["2~9건", "10~19건", "20~49건", "50~99건", "100건 이상"]
    handles = [Line2D([], [], ls="", marker="o", markerfacecolor=c, markeredgecolor="white", ms=6, label=label) for c, label in zip(colors, labels)]
    ax.legend(handles=handles, title="반복 신고 수", loc="upper right", fontsize=7.5, title_fontsize=8.5,
              frameon=True, facecolor="white", edgecolor="#dddddd", framealpha=.95)
    save(fig, "repeat_map", "같은 지점·날짜에 2건 이상인 묶음에 속한 신고 수를 실제 건수 구간으로 표시한다. 밝은 회색 OSM 바탕지도에 좌표 확인 지점을 표시하며 상위 5개 동은 신고 수 가중 중심에 이름을 붙인다.")
    total_reports = spots.n_reports_train.sum() + spots.n_reports_test.sum()
    valid["report_share"] = (valid.n_reports_train + valid.n_reports_test) / max(total_reports, 1)
    # Enforcement denominator includes unmatched locations, avoiding renormalization to matches.
    enforcement_total = len(pd.read_parquet(OUT / "enforcement.parquet"))
    valid["enforcement_share"] = valid.n_enf_2023 / enforcement_total
    valid["share_difference_pp"] = 100 * (valid.report_share - valid.enforcement_share)
    write_frame(valid[["spot_id", "label", "lat", "lon", "report_share", "enforcement_share", "share_difference_pp"]], TABLE / "share_difference_map.csv")
    data = pd.read_csv(TABLE / "share_difference_map.csv")
    fig, ax = base_map(valid)
    bins = np.digitize(data.share_difference_pp, [-.5, -.1, .1, .5])
    colors = np.array(["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"])
    x, y = WEB.transform(data.lon.to_numpy(), data.lat.to_numpy())
    ax.scatter(x, y, c=colors[bins], s=6 + np.minimum(55, 32 * data.share_difference_pp.abs()),
               alpha=.86, linewidths=.22, edgecolors="#888888", zorder=3)
    labels = ["-0.5 미만", "-0.5~-0.1", "-0.1~+0.1", "+0.1~+0.5", "+0.5 이상"]
    handles = [Line2D([], [], ls="", marker="o", markerfacecolor=c, markeredgecolor="#888888", ms=5.5, label=label) for c, label in zip(colors, labels)]
    ax.legend(handles=handles, title="신고 2024 - 단속 2023\n비중 차이 (%p)", loc="upper right", fontsize=7.5,
              title_fontsize=8, frameon=True, facecolor="white", edgecolor="#dddddd", framealpha=.95)
    save(fig, "share_difference_map", "단속 2023년·신고 2024년. 각 전체 건수를 분모로 사용. 문자열 연결 실패도 있어 단속 사각지대나 정책 효과로 해석하지 않는다.")


def evaluation_plot():
    source = OUT / "evaluation_priority.csv"
    if not source.exists():
        return
    summary = pd.read_csv(source)
    write_frame(summary, TABLE / "priority_capture.csv")
    summary = pd.read_csv(TABLE / "priority_capture.csv")
    ps.apply()
    fig, ax = plt.subplots()
    for method, rows in summary.groupby("method"):
        rows = rows.sort_values("K")
        ax.errorbar(rows.K, rows.capture_rate * 100,
                    yerr=np.vstack([(rows.capture_rate - rows.ci_low) * 100, (rows.ci_high - rows.capture_rate) * 100]),
                    capsize=3, **ps.style(method))
    ax.set(xlabel="구간별 우선 점검 지점 수 K (개)", ylabel="평가 기간 신고 포착률 (%)", xticks=[20, 50, 100])
    ax.legend(loc="upper left")
    save(fig, "priority_capture", "2024년 10~12월의 모든 과태료 처리 신고가 분모이다. 오차막대는 92개 달력일을 2,000회 재표집한 95% 구간이며 두 방식에 같은 날짜 표본을 사용한다.")


def render_all():
    TABLE.mkdir(parents=True, exist_ok=True)
    reports = pd.read_parquet(OUT / "reports.parquet")
    enforcement = pd.read_parquet(OUT / "enforcement.parquet")
    spots = pd.read_csv(OUT / "spots.csv")
    concentration(reports, enforcement)
    heatmap(reports)
    repetition(reports)
    maps(reports, spots)
    evaluation_plot()
