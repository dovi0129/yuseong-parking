"""PNG figures at 7.16 inches, 200 dpi; values read from persisted CSV/JSON."""
from __future__ import annotations

import argparse
import json
import sys
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .domain import ROOT, ROUTES, Plan, PlanningData, load_planning_data
from .network import build_matrix
from .planner import METHOD_LABELS

COLORS = {"optimized": "#0072B2", "b1": "#E69F00", "b2": "#009E73"}
ROUTE_FILE = re.compile(r"^route_(?P<mock>mock_)?(?P<scenario>\d{2}-\d{2}_K[123])_(?P<representative>median|p90|worst)_(?P<date>\d{4}-\d{2}-\d{2})\.png$")


def condition_label(scenario):
    """Convert stored condition IDs to judge-facing Korean without changing data."""
    matched=re.fullmatch(r"(\d{2})-(\d{2})_K([123])",scenario)
    if matched is None:raise ValueError(f"알 수 없는 순회 조건: {scenario}")
    start,end,vehicles=map(int,matched.groups())
    return f"{start}~{end}시·{vehicles}대"


def save_checked(fig,path,audits):
    # Optional layout check: set FIGURE_AUDIT_DIR to a folder containing bbox_overlap_check.py.
    audit_dir=os.environ.get("FIGURE_AUDIT_DIR")
    if audit_dir:sys.path.insert(0,audit_dir)
    try:
        from bbox_overlap_check import audit_fig
    except ImportError:
        audit_fig=lambda fig,verbose=False:[]
    if path.name.startswith("route_mock_"):
        fig.text(.995,.004,"예시 자료",ha="right",va="bottom",fontsize=8,color=".4")
    findings=audit_fig(fig,verbose=False)
    from matplotlib.text import Text
    labels=sorted({text.get_text() for text in fig.findobj(Text) if text.get_visible() and text.get_text()})
    if any(re.search(r"[A-Za-z]",label) for label in labels):
        raise ValueError(f"그림에 영문 내부 표기가 남았습니다: {path}: {labels}")
    audits.append({"file":str(path),"labels":labels,"findings":[dict(severity=f.severity,category=f.category,note=f.note) for f in findings]})
    if any(f.severity.lower()=="critical" for f in findings):
        raise RuntimeError(f"그림 배치 검사 실패: {path}")
    temporary=path.with_name(f".{path.stem}.{os.getpid()}.redraw.png")
    fig.savefig(temporary,dpi=200)
    temporary.replace(path)


def style():
    plt.rcParams.update({"font.family": "NanumBarunGothic", "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})


def frozen_network(frozen,mock=False):
    if "planning_spots" in frozen:
        data=PlanningData(pd.DataFrame(frozen["planning_spots"]),pd.DataFrame(frozen["planning_priority"]),
            frozen["chosen_method"],frozen["source"],frozen["excluded_spots"])
    else:
        data=load_planning_data(ROUTES if mock else None,mock=mock,max_spots=len(frozen["candidate_spots"]))
    return build_matrix(data,mock=mock)


def draw_route_map(frozen,network,scenario,day_type_value,path,audits):
    """Draw stored tours; labels change, routing assignments and dates do not."""
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter
    methods=("b1","b2","optimized")
    fig,axes=plt.subplots(1,3,figsize=(7.16,3.35),layout="constrained",sharex=True,sharey=True)
    all_x,all_y=[],[]
    lookup={sid:i for i,sid in enumerate(network.ids)}
    for ax,method in zip(axes,methods):
        plan=Plan.from_dict(frozen["plans"][f"{scenario}_{day_type_value}_{method}"])
        for vehicle in range(plan.config.vehicles):
            visits=[v for v in plan.visits if v.vehicle==vehicle]
            sequence=[0]+[lookup[v.spot_id] for v in visits]+[0]
            for a,b in zip(sequence,sequence[1:]):
                points=network.path(a,b);ys,xs=zip(*points)
                all_x.extend(xs);all_y.extend(ys)
                ax.plot(xs,ys,color=COLORS[method],linewidth=.65,linestyle="-" if vehicle==0 else "--",alpha=.85)
            ax.scatter([v.lon for v in visits],[v.lat for v in visits],s=5,color=COLORS[method])
        ax.scatter(network.coordinates[0][1],network.coordinates[0][0],marker="*",color="black",s=40)
        ax.text(.02,1.02,METHOD_LABELS[method],transform=ax.transAxes,fontsize=9)
        ax.set_xlabel("경도 (°)")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_aspect(1/np.cos(np.deg2rad(network.coordinates[0][0])))
    axes[0].set_ylabel("위도 (°)")
    if all_x:
        pad=.002;axes[0].set_xlim(min(all_x)-pad,max(all_x)+pad);axes[0].set_ylim(min(all_y)-pad,max(all_y)+pad)
    fig.supxlabel(condition_label(scenario),fontsize=9)
    save_checked(fig,path,audits)
    plt.close(fig)


def make_figures(mock=False,existing_paths=()):
    style()
    prefix = "mock_" if mock else ""
    out = ROOT / "artifacts/figures"
    out.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(ROUTES / f"{prefix}evaluation_daily.csv")
    summary = pd.read_csv(ROUTES / f"{prefix}evaluation_summary.csv")
    frozen = json.loads((ROUTES / f"{prefix}plans_frozen.json").read_text())
    network=frozen_network(frozen,mock=mock)
    scenarios = sorted(daily.scenario.unique())
    audits=[]
    methods = ("b1", "b2", "optimized")
    for metric, axis in (("caught_per_person_hour", "기록상 포착 신고 수 (건/차량·시간)"),
                         ("risk_caught", "위험 가중 포착 신고 수 (가중 건/일)"),
                         ("travel_min", "총 이동시간 (분/일)")):
        fig, ax = plt.subplots(figsize=(7.16, 3.6), layout="constrained")
        for j, method in enumerate(methods):
            sub = summary[(summary.metric == metric) & (summary.method == method)].set_index("scenario").loc[scenarios]
            ax.bar(np.arange(len(scenarios))+(j-1)*.24, sub["mean"], .23,
                color=COLORS[method], label=METHOD_LABELS[method], hatch=["//", "..", ""][j],
                yerr=np.array([sub["mean"]-sub.ci_low, sub.ci_high-sub["mean"]]), capsize=2)
        ax.set_xticks(np.arange(len(scenarios)), [condition_label(s) for s in scenarios])
        ax.set_xlabel("근무 구간·단속 차량 수")
        ax.set_ylabel(axis)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.23), ncol=3, frameon=False)
        save_checked(fig,out / f"route_{prefix}comparison_{metric}.png",audits)
        plt.close(fig)
    b3_path=ROUTES/"evaluation_b3_paired.csv"
    if not mock and b3_path.exists():
        paired=pd.read_csv(b3_path)
        sub=paired.loc[paired.metric.eq("caught")].set_index("scenario").loc[scenarios]
        fig,ax=plt.subplots(figsize=(7.16,3.6),layout="constrained")
        ax.axhline(0,color=".35",linewidth=.8)
        ax.errorbar(np.arange(len(scenarios)),sub.mean_delta,
            yerr=np.array([sub.mean_delta-sub.ci_low,sub.ci_high-sub.mean_delta]),
            fmt="o",color="#0072B2",capsize=4)
        ax.set_xticks(np.arange(len(scenarios)),[condition_label(s) for s in scenarios])
        ax.set_xlabel("근무 구간·단속 차량 수")
        ax.set_ylabel(f"{METHOD_LABELS['optimized']} - {METHOD_LABELS['b3']} 포착 차이 (건/일)")
        save_checked(fig,out/"route_b3_paired_caught.png",audits)
        plt.close(fig)
    representatives = []
    for scenario, frame in daily.groupby("scenario"):
        pivot = frame.pivot(index="date", columns="method", values="caught")
        delta = pivot.optimized-pivot.b1
        targets = {"median": delta.quantile(.5), "p90": delta.quantile(.9), "worst": delta.min()}
        for label, target in targets.items():
            selected = (delta-target).abs().sort_values(kind="stable").index[0]
            dt = frame.loc[frame.date == selected, "day_type"].iloc[0]
            representatives.append(dict(scenario=scenario, representative=label, date=selected,
                optimized_minus_b1=float(delta.loc[selected]), reference="caught; b1; signed difference"))
            draw_route_map(frozen,network,scenario,dt,out/f"route_{prefix}{scenario}_{label}_{selected}.png",audits)
    # Preserve pre-existing filenames from older representative selections.
    # Their original frozen plans, rather than newly selected dates, are used.
    rendered={Path(entry["file"]).name for entry in audits}
    older_frozen=None;older_network=None
    for path in existing_paths:
        match=ROUTE_FILE.fullmatch(path.name)
        if not match or bool(match["mock"])!=mock or path.name in rendered:continue
        if mock:raise ValueError(f"예시 그림의 원본 버전을 확인할 수 없습니다: {path.name}")
        previous=ROUTES/"previous_before_warmstart"
        previous_days=pd.read_csv(previous/"representative_days.csv")
        row=previous_days.loc[previous_days.scenario.eq(match["scenario"])&
            previous_days.representative.eq(match["representative"])&previous_days.date.eq(match["date"])]
        if len(row)!=1:raise ValueError(f"이전 그림의 원본 버전을 확인할 수 없습니다: {path.name}")
        if older_frozen is None:
            older_frozen=json.loads((previous/"plans_frozen.json").read_text())
            older_network=frozen_network(older_frozen)
        from .domain import day_type
        draw_route_map(older_frozen,older_network,match["scenario"],day_type(match["date"]),path,audits)
    pd.DataFrame(representatives).to_csv(ROUTES / f"{prefix}representative_days.csv", index=False)
    (ROUTES / f"{prefix}figure_notes.json").write_text(json.dumps({
        "width_in": 7.16, "dpi": 200, "ci": "date bootstrap 2000 resamples, percentile 95%",
        "representatives": "closest to median / p90 / minimum optimized-minus-b1 daily caught count, each scenario",
        "mock": mock, "metric": "spot and arrival hour_block matching; not causal reduction",
    }, ensure_ascii=False, indent=2))
    (ROUTES / f"{prefix}figure_audit.json").write_text(json.dumps(audits,ensure_ascii=False,indent=2))
    print("figures ready", out, "mock", mock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--all-existing",action="store_true")
    args=parser.parse_args()
    if args.all_existing:
        existing=tuple(sorted((ROOT/"artifacts/figures").glob("route_*.png")))
        make_figures(False,existing)
        make_figures(True,existing)
        audit_files=[ROUTES/"figure_audit.json",ROUTES/"mock_figure_audit.json"]
        entries=[item for p in audit_files for item in json.loads(p.read_text())]
        rendered={Path(item["file"]).name for item in entries}
        missing={p.name for p in existing}-rendered
        if missing:raise ValueError(f"다시 그리지 않은 순회 그림: {sorted(missing)}")
        print("all original filenames redrawn",len(existing))
    else:
        make_figures(args.mock)
