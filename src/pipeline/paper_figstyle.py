"""Project figure style, adapted from figure-make's paper_figstyle template."""
from pathlib import Path
import sys

import matplotlib as mpl

sys.path.insert(0, str(Path.home() / ".agents/skills/figure-make/scripts"))
import figstyle as _style

METHODS = {
    "freq_baseline": dict(label="빈도 기준선", color="#0072B2", ls="--", marker="s"),
    "shrinkage": dict(label="축소 추정", color="#D55E00", ls="-", marker="o"),
}


def apply(aspect=.65):
    _style.apply(width="text", aspect=aspect, korean=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["NanumBarunGothic", "NanumGothic"],
                         "axes.unicode_minus": False, "savefig.dpi": 200})


def style(method):
    return dict(METHODS[method])
