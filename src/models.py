# src/models.py  (replace the functions in your file with these)

import numpy as np
import pandas as pd
import warnings
from pandas.api.types import is_numeric_dtype
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

def naive_forecast(train_series, horizon):
    last = train_series.iloc[-1]
    return np.repeat(last, horizon)

def seasonal_naive(train_series, horizon, season=52):
    if len(train_series) <= season:
        return naive_forecast(train_series, horizon)
    last_season = train_series.iloc[-season:]
    reps = int(np.ceil(horizon/season))
    return np.tile(last_season.values, reps)[:horizon]

def sarimax_forecast(train_series, horizon, order=(1,1,1), seasonal=(0,1,1,52)):
    """
    Gentle guard: if the series is very short relative to the seasonal period,
    skip SARIMAX and return seasonal_naive to avoid warnings/failures.
    """
    try:
        season = seasonal[-1] if isinstance(seasonal, (list, tuple)) else seasonal
        # if not enough history for seasonal estimation, fallback
        min_len = max(10, 2 * int(season))
        if len(train_series) < min_len:
            return seasonal_naive(train_series, horizon, season=season)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")  # reduce noisy sarimax warnings
            model = SARIMAX(train_series, order=order, seasonal_order=seasonal,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fc = res.forecast(steps=horizon)
            return fc.values
    except Exception:
        # fallback to seasonal naive if anything goes wrong
        return seasonal_naive(train_series, horizon, season=seasonal[-1])

def fit_lightgbm(train_df, target_col="sales"):
    """
    Train LightGBM on numeric features only.
    - Excludes `target_col` and `week_start`.
    - Automatically picks numeric columns; if none exist, attempts a light coercion (factorize).
    Returns (model, feature_list).
    """
    if target_col not in train_df.columns:
        raise ValueError(f"target_col '{target_col}' not in dataframe columns: {train_df.columns.tolist()}")

    # candidate features excluding target and time index
    candidate_features = [c for c in train_df.columns if c not in [target_col, "week_start"]]

    # keep numeric columns only
    numeric_features = [c for c in candidate_features if is_numeric_dtype(train_df[c])]

    # if no numeric features, try to factorize object columns (last resort)
    if len(numeric_features) == 0:
        for c in candidate_features:
            if train_df[c].dtype == "O":
                train_df[c] = pd.factorize(train_df[c])[0]
        numeric_features = [c for c in candidate_features if is_numeric_dtype(train_df[c])]

    if len(numeric_features) == 0:
        raise ValueError(f"No numeric features available for LightGBM. Candidates: {candidate_features}")

    X = train_df[numeric_features].astype(float)
    y = train_df[target_col].astype(float)

    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X, y)
    return model, numeric_features

def predict_lightgbm(model, features, df):
    """
    Predict using the LightGBM model.
    """
    X = df.copy()
    for c in features:
        if c not in X.columns:
            X[c] = 0

    X = X[features].copy()

    for c in features:
        if not is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return model.predict(X.astype(float))
