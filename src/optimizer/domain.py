from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / "data/processed/routes"
BLOCKS = ((0, 7, "00-07"), (7, 10, "07-10"), (10, 14, "10-14"),
          (14, 17, "14-17"), (17, 21, "17-21"), (21, 24, "21-24"))
# Republic of Korea national/substitute holidays, including the 2024 election
# and the temporary Armed Forces Day holiday. A Saturday/Sunday is also weekend.
HOLIDAYS_2024 = frozenset("2024-01-01 2024-02-09 2024-02-10 2024-02-11 "
    "2024-02-12 2024-03-01 2024-04-10 2024-05-05 2024-05-06 2024-05-15 "
    "2024-06-06 2024-08-15 2024-09-16 2024-09-17 2024-09-18 2024-10-01 "
    "2024-10-03 2024-10-09 2024-12-25".split())


def day_type(value: date | datetime | str) -> str:
    value = pd.Timestamp(value)
    if value.year != 2024:
        raise ValueError("2024년 이외 날짜는 해당 연도 공휴일 목록을 추가해야 합니다.")
    return "주말" if value.weekday() >= 5 or value.date().isoformat() in HOLIDAYS_2024 else "평일"


def hour_block(seconds: int) -> str:
    if not 0 <= seconds < 86400:
        raise ValueError("방문 시각은 같은 날의 0시 이상 24시 미만이어야 합니다.")
    return next(label for lo, hi, label in BLOCKS if lo * 3600 <= seconds < hi * 3600)


@dataclass(frozen=True)
class PlanConfig:
    start_min: int = 17 * 60
    end_min: int = 21 * 60
    vehicles: int = 1
    service_min: float = 5.0
    day_type: str = "평일"
    use_risk: bool = True
    solver_seconds: float = 30.0

    def __post_init__(self):
        if not 0 <= self.start_min < self.end_min <= 1440:
            raise ValueError("근무 시작·종료 시각은 같은 날 안에서 지정하세요.")
        if self.vehicles not in (1, 2, 3) or self.service_min <= 0:
            raise ValueError("점검 인원은 1~3명, 점검시간은 0분 초과여야 합니다.")
        if self.day_type not in ("평일", "주말") or not 0 < self.solver_seconds <= 30:
            raise ValueError("요일 구분이나 계산시간이 잘못되었습니다.")


@dataclass
class Visit:
    spot_id: str
    arrival_s: int
    vehicle: int
    score: float
    risk_weight: float
    label: str
    lat: float
    lon: float

    @property
    def block(self) -> str:
        return hour_block(self.arrival_s)


@dataclass
class Plan:
    method: str
    config: PlanConfig
    visits: list[Visit] = field(default_factory=list)
    vehicle_returns_s: list[int] = field(default_factory=list)
    travel_s: float = 0
    elapsed_s: float = 0
    status: str = "ok"
    score_method: str = ""
    seed_expected_weighted: float | None = None
    seed_expected_reports: float | None = None
    initial_assignment_used: bool = False
    incumbent_source: str = ""

    @property
    def expected_reports(self):
        return sum(v.score for v in self.visits)

    @property
    def expected_weighted(self):
        return sum(v.score * v.risk_weight for v in self.visits)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, obj):
        obj = dict(obj)
        obj["config"] = PlanConfig(**obj["config"])
        obj["visits"] = [Visit(**v) for v in obj["visits"]]
        return cls(**obj)


@dataclass
class PlanningData:
    spots: pd.DataFrame
    priority: pd.DataFrame
    chosen_method: str
    source: str
    excluded_spots: int = 0

    def scores(self, config: PlanConfig):
        part = self.priority[(self.priority.method == self.chosen_method) &
                             (self.priority.day_type == config.day_type)]
        return {(str(r.spot_id), str(r.hour_block)): float(r.score)
                for r in part.itertuples()}


def load_planning_data(directory: Path | None = None, *, mock: bool = False,
                       max_spots: int | None = None, method: str | None = None) -> PlanningData:
    directory = Path(directory or ROOT / "data/processed")
    prefix = "mock_" if mock else ""
    spots = pd.read_csv(directory / f"{prefix}spots.csv", dtype={"spot_id": str})
    priority = pd.read_csv(directory / f"{prefix}priority_train.csv", dtype={"spot_id": str})
    choice = json.loads((directory / f"{prefix}priority_choice.json").read_text())
    chosen = method or choice["chosen_method"]
    required = {"spot_id", "label", "dong", "lat", "lon", "n_reports_train", "risk_weight", "child_zone_300m"}
    if missing := required - set(spots):
        raise ValueError(f"spots.csv 필수 열 누락: {sorted(missing)}")
    if set(("spot_id", "day_type", "hour_block", "score", "method")) - set(priority):
        raise ValueError("priority_train.csv 데이터 계약이 맞지 않습니다.")
    if chosen not in set(priority.method):
        raise ValueError(f"선택된 점수 방식이 없습니다: {chosen}")
    if spots.spot_id.duplicated().any() or priority.duplicated(["spot_id", "day_type", "hour_block", "method"]).any():
        raise ValueError("중복 지점 또는 중복 우선순위 행이 있습니다.")
    if not set(priority.day_type).issubset({"평일", "주말"}) or not set(priority.hour_block).issubset({b[2] for b in BLOCKS}):
        raise ValueError("요일/시간대 코드가 데이터 계약과 다릅니다.")
    for column in ("score",):
        priority[column] = pd.to_numeric(priority[column], errors="raise")
        if not np.isfinite(priority[column]).all() or (priority[column] < 0).any():
            raise ValueError("우선순위 점수는 유한한 0 이상 값이어야 합니다.")
    for column in ("lat", "lon", "n_reports_train", "risk_weight"):
        spots[column] = pd.to_numeric(spots[column], errors="raise")
    if not np.isfinite(spots[["n_reports_train", "risk_weight"]]).all().all() or (spots.risk_weight < 1).any() or (spots.n_reports_train < 0).any():
        raise ValueError("학습 신고 건수·위험 가중치가 유효하지 않습니다.")
    valid = (np.isfinite(spots.lat) & np.isfinite(spots.lon) &
             spots.lat.between(36.20, 36.65) & spots.lon.between(127.10, 127.60) &
             (spots.n_reports_train > 0))
    eligible = spots.loc[valid].sort_values(["n_reports_train", "spot_id"], ascending=[False, True])
    if max_spots is not None and max_spots<1:
        raise ValueError("max_spots는 양수 또는 전체 지점을 뜻하는 None이어야 합니다.")
    selected = eligible.copy() if max_spots is None else eligible.head(max_spots).copy()
    if selected.empty:
        raise ValueError("학습 기록과 유효한 좌표를 가진 지점이 없습니다.")
    # Deliberately remove held-out totals: no route code can use n_reports_test.
    columns = [c for c in selected if c != "n_reports_test"]
    selected = selected[columns].reset_index(drop=True)
    return PlanningData(selected, priority[priority.spot_id.isin(selected.spot_id)].copy(),
                        chosen, "mock" if mock else "real", len(spots) - len(selected))
