import pandas as pd

def overall_metrics(df):
    overall = df.groupby("model", as_index=False).agg(mae=("mae","mean"), mape=("mape","mean"))
    return overall.sort_values("mape")
