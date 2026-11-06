"""Unit tests for src/preprocessing.py.

Run from the repository root: python -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import (
    InvalidTargetError,
    MissingColumnsError,
    coerce_target,
    normalise_frame,
    split_features_and_target,
)


def _valid_row():
    row = {
        "Administrative": 2,
        "Administrative_Duration": 20.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 10,
        "ProductRelated_Duration": 300.0,
        "BounceRates": 0.01,
        "ExitRates": 0.02,
        "PageValues": 15.0,
        "SpecialDay": 0.0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 1,
        "Region": 3,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
    assert set(row) == set(FEATURE_COLUMNS)
    return row


def _frame(target_values):
    row = _valid_row()
    return pd.DataFrame([{**row, TARGET_COLUMN: value} for value in target_values])


def test_split_features_and_target_accepts_boolean_target():
    predictors, labels = split_features_and_target(_frame([True, False]))
    assert list(labels) == [1, 0]
    assert list(predictors.columns) == FEATURE_COLUMNS


def test_split_features_and_target_accepts_string_variants():
    predictors, labels = split_features_and_target(_frame(["yes", "No", "TRUE", "0"]))
    assert list(labels) == [1, 0, 1, 0]


def test_split_features_and_target_missing_column_raises():
    frame = _frame([True]).drop(columns=["PageValues"])
    with pytest.raises(MissingColumnsError, match="PageValues"):
        split_features_and_target(frame)


def test_blank_target_cell_raises_invalid_target_error_not_a_crash():
    frame = _frame([True, np.nan, False])
    with pytest.raises(InvalidTargetError, match="blank or unreadable"):
        split_features_and_target(frame)


def test_unreadable_string_target_raises_invalid_target_error():
    frame = _frame(["yes", "maybe"])
    with pytest.raises(InvalidTargetError):
        split_features_and_target(frame)


def test_coerce_target_rejects_nan_numeric_series():
    with pytest.raises(InvalidTargetError):
        coerce_target(pd.Series([1, np.nan, 0]))


def test_coerce_target_rejects_out_of_range_numeric_values():
    with pytest.raises(InvalidTargetError, match="blank or unreadable"):
        coerce_target(pd.Series([0, 1, 2]))


def test_normalise_frame_fills_missing_categorical_with_sentinel_not_literal_nan():
    frame = _frame([True])
    frame.loc[0, "Month"] = np.nan
    cleaned = normalise_frame(frame)
    assert cleaned.loc[0, "Month"] == "missing"
    assert cleaned.loc[0, "Month"] != "nan"


def test_normalise_frame_fills_missing_numeric_with_zero():
    frame = _frame([True])
    frame.loc[0, "PageValues"] = np.nan
    cleaned = normalise_frame(frame)
    assert cleaned.loc[0, "PageValues"] == 0


def test_normalise_frame_coerces_non_numeric_junk_to_zero():
    row_bad = {**_valid_row(), TARGET_COLUMN: True}
    row_bad["BounceRates"] = "not-a-number"  # simulates a malformed CSV cell
    frame = pd.DataFrame([row_bad])
    cleaned = normalise_frame(frame)
    assert cleaned.loc[0, "BounceRates"] == 0
