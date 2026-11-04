"""The six evaluation measures reported for every classifier in this study."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def positive_class_scores(estimator, predictors) -> np.ndarray:
    """Continuous scores for the purchasing class, used for the AUC."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(predictors)[:, 1]
    return estimator.decision_function(predictors)


def score_classifier(estimator, predictors, truth) -> dict:
    predicted = estimator.predict(predictors)
    ranking = positive_class_scores(estimator, predictors)
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": roc_auc_score(truth, ranking),
        "Precision": precision_score(truth, predicted, zero_division=0),
        "Recall": recall_score(truth, predicted, zero_division=0),
        "F1": f1_score(truth, predicted, zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }
