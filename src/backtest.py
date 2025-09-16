from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .config import REPORTS_DIR, N_FOLDS, TEST_WEEKS, MIN_TRAIN_WEEKS, SARIMAX_ORDER, SARIMAX_SEASONAL
from .ingestion import load_sales
from .features import weekly_aggregate, add_calendar_features, add_lag_features
from .models import naive_forecast, seasonal_naive, sarimax_forecast, fit_lightgbm, predict_lightgbm

def rolling_origin_splits(wdf, n_folds=4, test_weeks=4, min_train_weeks=30):
    weeks = sorted(wdf["week_start"].unique())
    splits = []
    for i in range(n_folds):
        train_end_idx = len(weeks) - (n_folds - i)*test_weeks - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_weeks - 1
        if train_end_idx + 1 < min_train_weeks or test_end_idx >= len(weeks):
            continue
        splits.append((weeks[0], weeks[train_end_idx], weeks[test_start_idx], weeks[test_end_idx]))
    return splits

def evaluate_series(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return float(mae), float(mape)

def run_backtest():
    print("Loading sales...")
    raw = load_sales()
    print("Aggregating to weekly...")
    wdf = weekly_aggregate(raw)

    print("Adding calendar features...")
    feat = add_calendar_features(wdf.rename(columns={"week_start": "date"}))
    feat = feat.rename(columns={"date": "week_start"})
    feat["sales"] = wdf["sales"].values

    print("Adding lag and MA features...")
    feat = add_lag_features(feat, lags=(1,2,4,52), mas=(4,12))

    before_rows = feat.shape[0]
    feat = feat.dropna().reset_index(drop=True)
    after_rows = feat.shape[0]
    print(f"Dropped {before_rows - after_rows} rows due to lag NA. Remaining rows: {after_rows}")

    splits = rolling_origin_splits(feat, n_folds=N_FOLDS, test_weeks=TEST_WEEKS, min_train_weeks=MIN_TRAIN_WEEKS)
    if len(splits) == 0:
        raise RuntimeError("No valid rolling-origin splits found. Lower MIN_TRAIN_WEEKS or adjust N_FOLDS/TEST_WEEKS.")

    records = []
    plot_dir = Path(REPORTS_DIR) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (train_start, train_end, test_start, test_end) in splits:
        fold_tag = f"{train_end.strftime('%Y%m%d')}_{test_start.strftime('%Y%m%d')}"
        print(f"Running fold {fold_tag} ...")
        fold_records = []

        series_stats = []
        for (store, sku), g in feat.groupby(["store_id","sku_id"]):
            sub = g.sort_values("week_start")
            # filter into train/test based on the fold
            train = sub[(sub["week_start"] >= train_start) & (sub["week_start"] <= train_end)]
            test  = sub[(sub["week_start"] >= test_start) & (sub["week_start"] <= test_end)]
            if len(train) == 0 or len(test) == 0:
                continue
            series_stats.append(((store, sku), float(np.var(sub["sales"])), len(sub)))

        # pick deterministic series to plot:
        #  - first valid series (by groupby order) and up to 3 highest variance ones
        series_stats_sorted = sorted(series_stats, key=lambda x: (-x[1], -x[2]))
        plotted = set()
        to_plot = []
        if series_stats:
            to_plot.append(series_stats[0][0])
        # add top-variance up to 3 (avoid duplicates)
        for (sks, var, ln) in series_stats_sorted[:3]:
            if sks not in to_plot:
                to_plot.append(sks)

        # Now iterate per series and compute models + metrics
        for (store, sku), g in feat.groupby(["store_id","sku_id"]):
            g = g.sort_values("week_start")
            train = g[(g["week_start"] >= train_start) & (g["week_start"] <= train_end)]
            test  = g[(g["week_start"] >= test_start) & (g["week_start"] <= test_end)]
            if len(train) == 0 or len(test) == 0:
                continue

            # True series
            ytr = train["sales"]
            yts = test["sales"]

            fc_naive = naive_forecast(ytr, len(yts))
            fc_snaive = seasonal_naive(ytr, len(yts), season=52)
            fc_sarimax = sarimax_forecast(ytr, len(yts), order=SARIMAX_ORDER, seasonal=SARIMAX_SEASONAL)

            try:
                train_for_lgb = train.copy().drop(columns=["store_id","sku_id"])
                # ensure 'sales' is present in train_for_lgb, as fit_lightgbm expects it
                if "sales" not in train_for_lgb.columns:
                    train_for_lgb = train_for_lgb.assign(sales=ytr)
                model, features = fit_lightgbm(train_for_lgb, target_col="sales")
                test_for_lgb = test.copy().drop(columns=["store_id","sku_id", "sales"])
                # ensure missing features are handled in predict_lightgbm
                fc_lgb = predict_lightgbm(model, features, test_for_lgb)
            except Exception as e:
                # if LightGBM fails for this series, fallback to NaN predictions (so other models still evaluated)
                print(f"[warn] LightGBM failed for {store}_{sku} in fold {fold_tag}: {e}")
                fc_lgb = np.array([np.nan]*len(yts))

            for name, pred in [("naive", fc_naive), ("seasonal_naive", fc_snaive),
                               ("sarimax", fc_sarimax), ("lgbm", fc_lgb)]:
                try:
                    mae, mape = evaluate_series(yts.values, np.array(pred))
                except Exception:
                    mae, mape = float("nan"), float("nan")
                fold_records.append({
                    "fold": fold_tag,
                    "store_id": store,
                    "sku_id": sku,
                    "model": name,
                    "mae": mae,
                    "mape": mape
                })

            if (store, sku) in to_plot:
                try:
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(train["week_start"], ytr, label="train", linewidth=1, marker='o')
                    ax.plot(test["week_start"], yts, label="test", linewidth=1, marker='o')
                    try:
                        ax.plot(test["week_start"], fc_sarimax, label="sarimax_fc", linestyle="--", marker='x')
                    except Exception:
                        pass
                    try:
                        ax.plot(test["week_start"], fc_lgb, label="lgbm_fc", linestyle=":", marker='s')
                    except Exception:
                        pass
                    ax.set_title(f"{store} | {sku} | fold {fold_tag}")
                    ax.set_xlabel("week_start")
                    ax.set_ylabel("sales")
                    ax.legend()
                    fig.tight_layout()
                    fname = plot_dir / f"{store}_{sku}_{fold_tag}.png"
                    fig.savefig(fname)
                    plt.close(fig)
                    plotted.add((store, sku))
                except Exception as e:
                    print(f"[warn] Plotting failed for {store}_{sku} in fold {fold_tag}: {e}")

        if len(fold_records) > 0:
            rec_df = pd.DataFrame(fold_records)
            records.append(rec_df)
        else:
            print(f"[info] No records for fold {fold_tag}")

    results = pd.concat(records, ignore_index=True)
    out_csv = Path(REPORTS_DIR) / f"backtest_{splits[-1][2].strftime('%Y%m%d')}.csv"
    results.to_csv(out_csv, index=False)

    summary = (results.groupby(["store_id","sku_id","model"], as_index=False)
                      .agg(mae=("mae","mean"), mape=("mape","mean")))
    best = summary.sort_values("mape").groupby(["store_id","sku_id"]).head(1)
    best_out = Path(REPORTS_DIR) / f"model_selection_{splits[-1][2].strftime('%Y%m%d')}.csv"
    best.to_csv(best_out, index=False)

    print("Backtest written to:", out_csv)
    print("Model selection written to:", best_out)
    print("Plots (if any) are in:", plot_dir)

if __name__ == "__main__":
    run_backtest()
