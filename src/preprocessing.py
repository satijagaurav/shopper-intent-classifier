"""Feature preparation shared by the training script and the Streamlit app."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


class MissingColumnsError(ValueError):
    """Raised when an uploaded CSV does not carry every column the models need."""


def build_feature_pipeline() -> ColumnTransformer:
    """Scale the continuous session metrics and one-hot the unordered labels."""
    scaled_block = Pipeline([("scale", StandardScaler())])
    encoded_block = Pipeline(
        [("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("continuous", scaled_block, NUMERIC_FEATURES),
            ("labels", encoded_block, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce the raw CSV into the dtypes the fitted pipeline expects."""
    cleaned = frame.copy()
    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype(str)
    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[NUMERIC_FEATURES] = cleaned[NUMERIC_FEATURES].fillna(0)
    return cleaned


def split_features_and_target(frame: pd.DataFrame):
    """Validate an incoming frame, then separate predictors from the label."""
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise MissingColumnsError(
            "Uploaded file is missing required column(s): " + ", ".join(absent)
        )

    cleaned = normalise_frame(frame)
    predictors = cleaned[FEATURE_COLUMNS]
    labels = cleaned[TARGET_COLUMN]

    if labels.dtype == object or labels.dtype == bool or str(labels.dtype) == "str":
        truthy = {"true", "1", "yes", "y", "t"}
        labels = labels.astype(str).str.strip().str.lower().isin(truthy).astype(int)
    else:
        labels = labels.astype(int)

    return predictors, labels
