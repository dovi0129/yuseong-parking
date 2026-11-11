"""Photo-demo paths and fixed pre-evaluation criteria."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/photo_demo"
FIG = ROOT / "artifacts/figures"
LETTERS = "가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주"
LIMITS = {"min_gap_s": 60, "max_move_m": 15, "max_report_m": 50}


def configure_runtime():
    # Use only the first visible GPU. An explicitly empty CUDA_VISIBLE_DEVICES (every GPU hidden by the operator)
    # is kept, so the photo reader then runs on the CPU without opening any GPU.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + ".writing")
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(target, path)
