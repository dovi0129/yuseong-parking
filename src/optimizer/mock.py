"""Write clearly named mock inputs exclusively inside the routes directory."""
import json

import numpy as np
import pandas as pd

from .domain import BLOCKS, ROUTES


def main():
    ROUTES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(27)
    spots = []
    priority = []
    for i in range(24):
        sid = f"MOCK_{i:03}"
        spots.append(dict(spot_id=sid, label=f"예시 점검지 {i+1}", dong=["예시 봉명동", "예시 궁동", "예시 어은동"][i % 3],
            lat=36.3622+rng.uniform(-.014, .014), lon=127.3561+rng.uniform(-.018, .012),
            geocode_method="manual", n_reports_train=240-i*7, n_reports_test=99999,
            n_enf_2023=0, dist_school_m=100 if i % 5 == 0 else 700,
            child_zone_300m=int(i % 5 == 0), accidents_200m=i % 4,
            risk_weight=1.5 if i % 5 == 0 else 1.0))
        for dt in ("평일", "주말"):
            for lo, hi, block in BLOCKS:
                score = (24-i)/70 * (1.8 if block == "17-21" else 1.0)
                for method in ("freq_baseline", "mock_shrinkage"):
                    priority.append(dict(spot_id=sid, day_type=dt, hour_block=block,
                        score=score, method=method))
    pd.DataFrame(spots).to_csv(ROUTES / "mock_spots.csv", index=False)
    pd.DataFrame(priority).to_csv(ROUTES / "mock_priority_train.csv", index=False)
    (ROUTES / "mock_priority_choice.json").write_text(json.dumps({"chosen_method": "freq_baseline", "선택 근거": "화면·단위시험용 예시"}, ensure_ascii=False))
    test = [dict(report_id=f"DEMO_{i}", ts=pd.Timestamp("2024-10-02 17:10")+pd.Timedelta(minutes=i*4),
        date="2024-10-02", day_type="평일", hour=17+(10+i*4)//60, hour_block="17-21",
        raw_loc=spots[i % 6]["label"], spot_id=spots[i % 6]["spot_id"], fine_status="예시", split="test")
        for i in range(24)]
    train = []
    for i in range(480):
        ts = pd.Timestamp("2024-09-01")+pd.Timedelta(days=i//16, hours=[8,12,17,18,19,20][i%6], minutes=i%60)
        index = (i%16) if i%3 else i%4
        train.append(dict(report_id=f"TRAIN_DEMO_{i}", ts=ts, date=ts.date().isoformat(),
            day_type="주말" if ts.weekday()>=5 else "평일", hour=ts.hour,
            hour_block=next(b for lo,hi,b in BLOCKS if lo<=ts.hour<hi), raw_loc=spots[index]["label"],
            spot_id=spots[index]["spot_id"], fine_status="예시", split="train"))
    pd.DataFrame(train+test).to_parquet(ROUTES / "mock_reports.parquet", index=False)
    print("mock inputs ready:", ROUTES)


if __name__ == "__main__":
    main()
