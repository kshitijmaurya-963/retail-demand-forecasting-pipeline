import pandas as pd
import numpy as np
from holidays import CountryHoliday

def add_calendar_features(df, country="IN"):
    cal = df.copy()
    cal["dow"] = cal["date"].dt.weekday
    cal["week"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.month
    cal["year"] = cal["date"].dt.year
    try:
        holidays = CountryHoliday(country, years=sorted(cal["year"].unique()))
        cal["is_holiday"] = cal["date"].astype("datetime64[ns]").isin(holidays).astype(int)
    except Exception:
        cal["is_holiday"] = 0
    cal["is_eoq_week"] = cal["date"].dt.quarter.isin([3]).astype(int)
    return cal

def add_lag_features(df, lags=(1,2,4,52), mas=(4,12)):
    out = df.copy()
    out = out.sort_values(["store_id","sku_id","date" if "date" in out.columns else "week_start"])
    group_cols = ["store_id","sku_id"]
    base_col = "sales"
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby(group_cols)[base_col].shift(lag)
    for w in mas:
        out[f"ma_{w}"] = out.groupby(group_cols)[base_col].shift(1).rolling(w).mean()
    return out

def weekly_aggregate(df):
    w = df.copy()
    w["week_start"] = w["date"] - pd.to_timedelta(w["date"].dt.weekday, unit="D")
    agg = (w.groupby(["store_id","sku_id","week_start"], as_index=False)
             .agg(sales=("sales","sum")))
    return agg
