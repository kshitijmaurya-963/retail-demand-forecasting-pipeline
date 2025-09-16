from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .config import REPORTS_DIR
from .ingestion import load_sales
from .features import weekly_aggregate, add_calendar_features, add_lag_features
from .models import sarimax_forecast, fit_lightgbm, predict_lightgbm

HORIZON_WEEKS = 4

def latest_forecast(save_plots=True):
    raw = load_sales()
    wdf = weekly_aggregate(raw)
    feat = add_calendar_features(wdf.rename(columns={"week_start":"date"}))
    feat = feat.rename(columns={"date":"week_start"})
    feat["sales"] = wdf["sales"].values
    feat = add_lag_features(feat, lags=(1,2,4,52), mas=(4,12))
    feat = feat.dropna().reset_index(drop=True)

    fcs = []
    history_cache = {}

    for (store, sku), g in feat.groupby(["store_id","sku_id"]):
        g = g.sort_values("week_start")
        ytr = g["sales"]
        # classical
        fc_sarimax = sarimax_forecast(ytr, HORIZON_WEEKS)

        # ML (train on whole series)
        try:
            model, feats = fit_lightgbm(g, target_col="sales")
            last_week = g["week_start"].max()
            future_weeks = [last_week + pd.Timedelta(days=7*(i+1)) for i in range(HORIZON_WEEKS)]
            fut = pd.DataFrame({
                "store_id": store,
                "sku_id": sku,
                "week_start": future_weeks,
                "lag_1": ytr.tail(1).values.repeat(HORIZON_WEEKS),
                "lag_2": ytr.tail(1).values.repeat(HORIZON_WEEKS),
                "lag_4": ytr.tail(1).values.repeat(HORIZON_WEEKS),
                "lag_52": ytr.tail(1).values.repeat(HORIZON_WEEKS),
                "ma_4": float(ytr.tail(4).mean()) if len(ytr)>=4 else float(ytr.mean()),
                "ma_12": float(ytr.tail(12).mean()) if len(ytr)>=12 else float(ytr.mean()),
            })
            dcal = pd.DataFrame({"date":future_weeks})
            dcal["dow"] = pd.to_datetime(dcal["date"]).dt.weekday
            dcal["week"] = pd.to_datetime(dcal["date"]).dt.isocalendar().week.astype(int)
            dcal["month"] = pd.to_datetime(dcal["date"]).dt.month
            dcal["year"] = pd.to_datetime(dcal["date"]).dt.year
            dcal["is_holiday"] = 0
            dcal["is_eoq_week"] = pd.to_datetime(dcal["date"]).dt.quarter.isin([3]).astype(int)
            fut = fut.merge(dcal.rename(columns={"date":"week_start"}), on="week_start", how="left")
            fc_lgb = predict_lightgbm(model, feats, fut.fillna(0))
        except Exception:
            fc_lgb = [np.nan] * HORIZON_WEEKS

        fcs.extend([{
            "store_id": store, "sku_id": sku, "model": "sarimax", "week_start": wk, "forecast": float(val)
        } for wk, val in zip(future_weeks, fc_sarimax)])
        fcs.extend([{
            "store_id": store, "sku_id": sku, "model": "lgbm", "week_start": wk, "forecast": float(val) if not np.isnan(val) else None
        } for wk, val in zip(future_weeks, fc_lgb)])

        history_cache[(store, sku)] = {
            "history_weeks": g["week_start"].tail(12).tolist(),   # last 12 weeks of history
            "history_sales": g["sales"].tail(12).tolist(),
            "future_weeks": future_weeks,
            "sarimax": list(fc_sarimax),
            "lgbm": list(fc_lgb)
        }

    out = pd.DataFrame(fcs)
    out_path = Path(REPORTS_DIR) / f"forecast_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out.to_csv(out_path, index=False)
    print("Forecasts written to:", out_path)

    if save_plots:
        plot_dir = Path(REPORTS_DIR) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for (store, sku), cache in history_cache.items():
            try:
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(cache["history_weeks"], cache["history_sales"], label="history", marker="o")
                ax.plot(cache["future_weeks"], cache["sarimax"], label="sarimax_fc", linestyle="--", marker="x")
                lgb_vals = cache["lgbm"]
                if any([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in lgb_vals]):
                    ax.plot(cache["future_weeks"], lgb_vals, label="lgbm_fc", linestyle=":", marker="s")
                ax.set_title(f"{store} | {sku} — last 12 weeks + {HORIZON_WEEKS}-week forecast")
                ax.set_xlabel("week_start")
                ax.set_ylabel("sales")
                ax.legend()
                fig.tight_layout()
                fname = plot_dir / f"forecast_{store}_{sku}.png"
                fig.savefig(fname)
                plt.close(fig)
            except Exception as e:
                print(f"Plotting failed for {store}_{sku}: {e}")

        print("Forecast plots saved to:", plot_dir)

if __name__ == "__main__":
    latest_forecast()
