"""Central configuration for the shopper purchase-intent classification study."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_EXPORT_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

TARGET_COLUMN = "Revenue"
HOLDOUT_FRACTION = 0.2
SEED = 2026

SESSION_COUNT_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
]

SESSION_QUALITY_FEATURES = [
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

NUMERIC_FEATURES = SESSION_COUNT_FEATURES + SESSION_QUALITY_FEATURES

# Stored as integers in the source file but semantically unordered labels,
# so they are one-hot encoded rather than scaled.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASSIFIER_REGISTRY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbours",
    "naive_bayes": "Naive Bayes (Gaussian)",
    "svm": "Support Vector Machine",
}
