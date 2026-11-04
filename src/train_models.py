"""Train, tune and persist the five classifiers compared in this assignment.

Run from the repository root:

    python -m src.train_models
"""

import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CLASSIFIER_REGISTRY,
    FEATURE_COLUMNS,
    HOLDOUT_FRACTION,
    METRICS_PATH,
    MODEL_DIR,
    RAW_DATA_PATH,
    SEED,
    TARGET_COLUMN,
    TEST_EXPORT_PATH,
)
from src.metrics import score_classifier
from src.preprocessing import build_feature_pipeline, normalise_frame

SEARCH_SPACE = {
    "logistic_regression": (
        LogisticRegression(max_iter=3000, random_state=SEED),
        {
            "learner__C": [0.1, 1.0, 10.0],
            "learner__class_weight": [None, "balanced"],
        },
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=SEED),
        {
            "learner__max_depth": [5, 8, 12, None],
            "learner__min_samples_leaf": [1, 5, 20],
            "learner__class_weight": [None, "balanced"],
        },
    ),
    "knn": (
        KNeighborsClassifier(),
        {
            "learner__n_neighbors": [5, 11, 21, 35],
            "learner__weights": ["uniform", "distance"],
        },
    ),
    "naive_bayes": (
        GaussianNB(),
        {"learner__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    ),
    # No Platt scaling: the AUC is rank-based, so raw margin distances from
    # decision_function serve it directly and avoid a 5-fold calibration refit.
    "svm": (
        SVC(kernel="rbf", random_state=SEED),
        {
            "learner__C": [1.0, 10.0],
            "learner__class_weight": [None, "balanced"],
        },
    ),
}


def load_sessions() -> pd.DataFrame:
    sessions = pd.read_csv(RAW_DATA_PATH)
    sessions = sessions.drop_duplicates().reset_index(drop=True)
    sessions[TARGET_COLUMN] = sessions[TARGET_COLUMN].astype(int)
    return sessions


def carve_holdout(sessions: pd.DataFrame):
    train_frame, holdout_frame = train_test_split(
        sessions,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=sessions[TARGET_COLUMN],
    )
    return train_frame.reset_index(drop=True), holdout_frame.reset_index(drop=True)


def tune_and_fit(key: str, train_x, train_y):
    learner, grid = SEARCH_SPACE[key]
    candidate = Pipeline(
        [("prepare", build_feature_pipeline()), ("learner", learner)]
    )
    search = GridSearchCV(
        candidate,
        param_grid=grid,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(train_x, train_y)
    elapsed = time.perf_counter() - started
    return search.best_estimator_, search.best_params_, elapsed


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions()
    train_frame, holdout_frame = carve_holdout(sessions)
    holdout_frame.to_csv(TEST_EXPORT_PATH, index=False)

    train_x = normalise_frame(train_frame)[FEATURE_COLUMNS]
    train_y = train_frame[TARGET_COLUMN]
    holdout_x = normalise_frame(holdout_frame)[FEATURE_COLUMNS]
    holdout_y = holdout_frame[TARGET_COLUMN]

    print(f"training rows: {len(train_frame)}  holdout rows: {len(holdout_frame)}")
    print(f"holdout purchase rate: {holdout_y.mean():.4f}\n")

    report = {
        "dataset": {
            "rows_total": int(len(sessions)),
            "rows_train": int(len(train_frame)),
            "rows_holdout": int(len(holdout_frame)),
            "feature_count": len(FEATURE_COLUMNS),
            "positive_rate_overall": float(sessions[TARGET_COLUMN].mean()),
            "positive_rate_holdout": float(holdout_y.mean()),
        },
        "models": {},
    }

    for key, label in CLASSIFIER_REGISTRY.items():
        fitted, chosen, elapsed = tune_and_fit(key, train_x, train_y)
        joblib.dump(fitted, MODEL_DIR / f"{key}.joblib")

        scores = score_classifier(fitted, holdout_x, holdout_y)
        report["models"][key] = {
            "label": label,
            "best_params": {k: str(v) for k, v in chosen.items()},
            "fit_seconds": round(elapsed, 2),
            "metrics": {k: round(v, 4) for k, v in scores.items()},
        }

        formatted = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"{label:<24} {formatted}   [{elapsed:.1f}s]")

    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nmetrics written to {METRICS_PATH}")
    print(f"holdout CSV written to {TEST_EXPORT_PATH}")


if __name__ == "__main__":
    main()
