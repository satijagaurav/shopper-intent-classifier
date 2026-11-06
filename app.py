"""Streamlit front end for the shopper purchase-intent classifier comparison.

Loads the five pre-trained pipelines from model/ and scores whatever test CSV
is supplied. No training happens here.
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.config import CLASSIFIER_REGISTRY, METRICS_PATH, MODEL_DIR, TEST_EXPORT_PATH
from src.metrics import METRIC_ORDER, positive_class_scores, score_classifier
from src.preprocessing import (
    InvalidTargetError,
    MissingColumnsError,
    split_features_and_target,
)

st.set_page_config(
    page_title="Shopper Purchase-Intent Classifier",
    page_icon="🛒",
    layout="wide",
)

ACCENT = "#4c6ef5"
POSITIVE_LABEL = "Purchase"
NEGATIVE_LABEL = "No purchase"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.55rem; color: {ACCENT}; }}
      div[data-testid="stMetric"] {{
          background: rgba(76,110,245,0.06);
          border: 1px solid rgba(76,110,245,0.20);
          border-radius: 10px;
          padding: 12px 14px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipelines() -> dict:
    return {
        key: joblib.load(MODEL_DIR / f"{key}.joblib")
        for key in CLASSIFIER_REGISTRY
    }


@st.cache_data(show_spinner=False)
def load_training_report() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@st.cache_data(show_spinner=False)
def load_bundled_holdout() -> pd.DataFrame:
    return pd.read_csv(TEST_EXPORT_PATH)


def draw_confusion(matrix, title: str):
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title, fontsize=11)
    figure.tight_layout()
    return figure


def draw_roc(truth, ranking, auc_value: float, title: str):
    false_rate, true_rate, _ = roc_curve(truth, ranking)
    figure, axis = plt.subplots(figsize=(4.2, 3.4))
    axis.plot(false_rate, true_rate, color=ACCENT, linewidth=2, label=f"AUC = {auc_value:.4f}")
    axis.plot([0, 1], [0, 1], color="#adb5bd", linestyle="--", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title, fontsize=11)
    axis.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    return figure


st.title("🛒 Online Shopper Purchase-Intent Classifier")
st.caption(
    "Five classifiers trained on the UCI Online Shoppers Purchasing Intention "
    "dataset, compared on a held-out test split. Upload a test CSV to score the "
    "saved models — nothing is retrained in this app."
)

pipelines = load_pipelines()
training_report = load_training_report()

with st.sidebar:
    st.header("Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV (17 feature columns + Revenue target)",
        type=["csv"],
    )
    use_bundled = st.checkbox(
        "Use the bundled held-out split instead", value=uploaded is None
    )

    st.divider()
    st.header("Model")
    selected_key = st.selectbox(
        "Classifier to inspect",
        options=list(CLASSIFIER_REGISTRY),
        format_func=lambda key: CLASSIFIER_REGISTRY[key],
    )

    if training_report:
        chosen_params = training_report["models"][selected_key]["best_params"]
        st.caption("Hyperparameters selected by 5-fold grid search:")
        for name, value in chosen_params.items():
            st.caption(f"• `{name.replace('learner__', '')}` = `{value}`")

if uploaded is not None and not use_bundled:
    try:
        source_frame = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as problem:
        st.error(f"Could not read '{uploaded.name}' as a CSV: {problem}")
        st.stop()
    source_name = uploaded.name
elif use_bundled:
    source_frame = load_bundled_holdout()
    source_name = "test_data.csv (bundled held-out split)"
else:
    st.info("Upload a test CSV from the sidebar, or tick the bundled-split option.")
    st.stop()

try:
    predictors, truth = split_features_and_target(source_frame)
except (MissingColumnsError, InvalidTargetError) as problem:
    st.error(str(problem))
    st.stop()

summary_left, summary_mid, summary_right = st.columns(3)
summary_left.metric("Test rows scored", f"{len(predictors):,}")
summary_mid.metric("Feature columns", f"{predictors.shape[1]}")
summary_right.metric("Purchase rate", f"{truth.mean():.2%}")
st.caption(f"Source: **{source_name}**")

single_tab, comparison_tab, data_tab = st.tabs(
    ["Selected model", "All five models", "Data preview"]
)

with single_tab:
    label = CLASSIFIER_REGISTRY[selected_key]
    estimator = pipelines[selected_key]
    scores = score_classifier(estimator, predictors, truth)
    predictions = estimator.predict(predictors)

    st.subheader(f"{label} — evaluation metrics")
    metric_slots = st.columns(6)
    for slot, metric_name in zip(metric_slots, METRIC_ORDER):
        slot.metric(metric_name, f"{scores[metric_name]:.4f}")

    left_panel, right_panel = st.columns(2)
    with left_panel:
        st.markdown("**Confusion matrix**")
        st.pyplot(
            draw_confusion(confusion_matrix(truth, predictions), label),
            width='stretch',
        )
    with right_panel:
        st.markdown("**ROC curve**")
        st.pyplot(
            draw_roc(
                truth,
                positive_class_scores(estimator, predictors),
                scores["AUC"],
                label,
            ),
            width='stretch',
        )

    st.markdown("**Classification report**")
    report_dict = classification_report(
        truth,
        predictions,
        target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report_dict).transpose().round(4), width='stretch')

with comparison_tab:
    st.subheader("All five classifiers on the current test data")
    rows = []
    for key, name in CLASSIFIER_REGISTRY.items():
        row = {"ML Model Name": name}
        row.update(score_classifier(pipelines[key], predictors, truth))
        rows.append(row)
    comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)

    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(
            axis=0, color="rgba(76,110,245,0.22)"
        ),
        width='stretch',
    )

    focus = st.radio(
        "Chart a single metric",
        options=METRIC_ORDER,
        index=METRIC_ORDER.index("MCC"),
        horizontal=True,
    )
    chart_figure, chart_axis = plt.subplots(figsize=(8, 3.2))
    ordered = comparison[focus].sort_values()
    chart_axis.barh(ordered.index, ordered.values, color=ACCENT, height=0.55)
    chart_axis.set_xlim(0, 1)
    chart_axis.set_xlabel(focus)
    for position, value in enumerate(ordered.values):
        chart_axis.text(value + 0.012, position, f"{value:.4f}", va="center", fontsize=9)
    chart_figure.tight_layout()
    st.pyplot(chart_figure, width='stretch')

    st.download_button(
        "Download this comparison as CSV",
        comparison.to_csv().encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

with data_tab:
    st.subheader("Uploaded test data")
    st.dataframe(source_frame.head(50), width='stretch')
    st.markdown("**Target balance**")
    st.bar_chart(truth.map({0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}).value_counts())
