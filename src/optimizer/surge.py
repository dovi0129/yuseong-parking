"""급증 지점: the month before the reference month has >= 5 reports and >= 2 x the monthly mean of the three months
before it. A spot with no report in those three months is included when the month before has >= 5 reports.

For a 현황 query ending on day E, the counted month ('직전 한 달') is the last complete calendar month up to E: E's
month when E is its last day, otherwise the month before. The reference month is the month after it. The three months
before must lie inside the data (from 2024-01), so the first counted month is 2024-04.
"""
from __future__ import annotations

import pandas as pd

MIN_RECENT = 5
RATIO = 2.0
DATA_START = pd.Period("2024-01", "M")


def counted_month(end_date) -> pd.Period | None:
    """Last complete month up to end_date, or None when its three earlier months are not all in the data."""
    end = pd.Timestamp(end_date)
    month = end.to_period("M") if end.is_month_end else end.to_period("M") - 1
    return month if month - 3 >= DATA_START else None


def surge_table(reports: pd.DataFrame, month: pd.Period) -> pd.DataFrame:
    """Per spot with a report in the four months: 직전 한 달 count, 앞 석 달 월평균 and the 급증 flag.

    Input reports need ts and spot_id. Output columns spot_id, recent, prior_mean, surge.
    """
    months = pd.to_datetime(reports.ts).dt.to_period("M")
    window = reports.loc[(months >= month - 3) & (months <= month)].assign(_m=months, spot_id=lambda f: f.spot_id.astype(str))
    recent = window.loc[window._m.eq(month)].groupby("spot_id").size()
    prior = window.loc[window._m.lt(month)].groupby("spot_id").size()
    table = pd.DataFrame({"recent": recent, "prior": prior}).fillna(0).astype(int)
    table["prior_mean"] = table.prior / 3
    table["surge"] = (table.recent >= MIN_RECENT) & (table.recent >= RATIO * table.prior_mean)
    return table.rename_axis("spot_id").reset_index()[["spot_id", "recent", "prior_mean", "surge"]]
