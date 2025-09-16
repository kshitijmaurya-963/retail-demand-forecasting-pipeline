Retail Demand Forecasting Micro-Pipeline (Classical + ML)
📌 Project Overview

This project builds a small, end-to-end forecasting pipeline for weekly retail demand.
It’s designed to feel like a realistic portfolio project: modular, reproducible, and insightful — but not overly perfect.

We simulate a dataset of 3 stores × 5 SKUs (~27 months of daily sales) and aggregate to weekly demand.
The pipeline benchmarks classical methods (Naive, Seasonal Naive, SARIMAX) against a machine learning model (LightGBM).

The goal is:

Data ingestion & preprocessing

Feature engineering (calendar, holidays, lags, moving averages)

Model comparisons

Rolling-origin backtesting

Error analysis & visualization

Forecast generation for future periods

🔍 Why This Approach?

Classical baselines first → to ground expectations (Naive, Seasonal Naive).

SARIMAX → interpretable seasonal model, but not heavily tuned (realistic tradeoff).

LightGBM → handles nonlinear interactions & engineered features better than pure time series models.

Rolling-origin backtest → mimics real deployment (retraining & predicting in steps).

Plots + metrics → numbers give precision, plots give intuition.

This mirrors how a DS would balance rigor with pragmatism.

⚙️ Project Structure
retail_demand_forecasting_micro_pipeline/
│
├── data/                 <- synthetic sales data (CSV)
├── reports/              <- metrics & plots after runs
│
├── src/
│   ├── data.py           <- data generation & ingestion
│   ├── features.py       <- feature engineering
│   ├── models.py         <- forecasting models
│   ├── backtest.py       <- rolling-origin evaluation
│   ├── forecast.py       <- final forecast generation
│
├── notebooks/
│   └── EDA.ipynb         <- quick exploratory notebook
│
├── requirements.txt
├── Makefile              <- run backtest/forecast easily
└── README.md

🚀 How to Run

Setup environment

python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt


Run backtest

make backtest


Outputs metrics CSV in reports/

Saves error & forecast plots

Run forecast

make forecast


Generates 4-week ahead forecasts

Saves plots per store-SKU

📊 Example Outputs

Metrics (MAE, MAPE) per model

Forecast Plots

Naive vs SARIMAX vs LightGBM

Error distributions

(You can add screenshots here after running locally and saving plots.)

✅ Key Learnings

Baselines are critical: Naive methods often surprisingly competitive.

SARIMAX can struggle with multiple SKUs/stores due to tuning complexity.

LightGBM with engineered features often generalizes better across series.

Rolling-origin evaluation avoids “data leakage” optimism.

🔮 Future Improvements

Hyperparameter tuning (Bayesian optimization / Optuna).

More advanced ML/DL (XGBoost, CatBoost, RNNs, Transformers).

Incorporate price & promotions as external regressors.

Deploy forecasts via REST API or Streamlit dashboard.

👤 Author

Project built as a portfolio case study to demonstrate end-to-end data science workflow:
from idea → pipeline → models → evaluation → storytelling.