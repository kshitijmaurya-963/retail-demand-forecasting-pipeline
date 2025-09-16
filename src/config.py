from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DATA = DATA_DIR / "raw" / "sales.csv"

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 4
TEST_WEEKS = 4
MIN_TRAIN_WEEKS = 30

SARIMAX_ORDER = (1,1,1)
SARIMAX_SEASONAL = (0,1,1,52)
LGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
