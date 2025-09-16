# 🛒 Retail Demand Forecasting Pipeline

An end-to-end **retail demand forecasting micro-pipeline** that compares classical time series models (Naive, Seasonal Naive, SARIMAX) with machine learning (LightGBM).  
This project demonstrates the workflow of a data scientist: from data ingestion and feature engineering to model training, rolling-origin backtesting, and forecast visualization.

---

## 📌 Project Overview
- **Dataset**: Synthetic daily store-item sales (~2 years), aggregated weekly.
- **Feature Engineering**:
  - Calendar features (week, month, year, holiday flags)
  - Lags (1, 2, 4 weeks)
  - Moving averages
- **Models**:
  - Baselines: Naive, Seasonal Naive
  - Classical: SARIMAX
  - Machine Learning: LightGBM
- **Evaluation**:
  - Rolling-origin backtesting (4 folds)
  - Metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE)
- **Outputs**:
  - Metrics CSV reports
  - Forecast plots per store/SKU

---

## 🛠️ Tech Stack
- **Core**: Python 3.9+
- **Libraries**:  
  `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `lightgbm`, `matplotlib`, `seaborn`

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/retail-demand-forecasting-pipeline.git
   cd retail-demand-forecasting-pipeline
   
2. **Create virtual environment & install dependencies**
   ```bash
    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
   
3. **Run backtesting**
   ```bash
    make backtest
-Metrics saved in: reports/metrics/
-Plots saved in: reports/plots/

4. **Run forecasting (final horizon)**
   ```bash
   make forecast
-Forecasts + plots saved in: reports/forecasts/
