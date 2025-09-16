import pandas as pd
from .config import RAW_DATA

def load_sales():
    df = pd.read_csv(RAW_DATA, parse_dates=["date"])
    df = df.sort_values(["store_id","sku_id","date"])
    return df
