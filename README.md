# Online Shopper Purchase-Intent Classifier

M.Tech (AIML) — Machine Learning, Assignment 2

A comparison of five classification algorithms on real e-commerce session data,
served through an interactive Streamlit application.

**Live app:** https://shopper-intent-classifier-fubibbyroyskvvupsarr2l.streamlit.app

---

## a. Problem statement

An online retailer records the behaviour of every visitor session: how many pages
of each type were opened, how long was spent on them, the bounce and exit rates,
the average value of the pages seen, the month, the traffic source, and whether
the visitor was new or returning. Most of these sessions end without a purchase.

The task is to predict, from this session profile alone, whether the session will
end in a transaction (`Revenue = True`). This is a **binary classification**
problem.

The business value is straightforward: a model that can identify a browsing
session as likely-to-convert while it is still in progress allows the retailer to
intervene, for example by offering assistance, free shipping, or a discount at
the moment it will actually change the outcome. Equally, correctly recognising
sessions that will not convert avoids spending promotional budget on them.

The dataset is imbalanced, with roughly one purchase for every six sessions. That
shapes the whole evaluation: a model that predicts "no purchase" every single
time already scores about 84% accuracy while being completely useless. For this
reason the comparison below leans on **MCC** and **AUC**, both of which account
for the class skew, rather than on raw accuracy.

---

## b. Dataset description

| Property | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (ID 468) |
| URL | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances (raw) | 12,330 |
| Instances (after removing 125 exact duplicates) | 12,205 |
| Feature columns | **17** |
| Target column | `Revenue` (boolean) |
| Task | Binary classification |
| Missing values | None |
| Class balance | 15.63% purchase / 84.37% no purchase |
| Train / test split | Stratified 80 / 20 → 9,764 train, 2,441 held out |

Both assignment minimums are cleared: 17 features (minimum 12) and 12,205
instances (minimum 500).

Each row is one user session collected over a one-year period, deliberately
sampled so that no single campaign, special day or user profile dominates.

### Feature inventory

**Continuous session metrics (10)** — standardised before modelling:

| Feature | Meaning |
|---|---|
| `Administrative` | Number of account/admin pages opened |
| `Administrative_Duration` | Seconds spent on those pages |
| `Informational` | Number of informational pages opened |
| `Informational_Duration` | Seconds spent on those pages |
| `ProductRelated` | Number of product pages opened |
| `ProductRelated_Duration` | Seconds spent on product pages |
| `BounceRates` | Average bounce rate of the pages visited |
| `ExitRates` | Average exit rate of the pages visited |
| `PageValues` | Average Google Analytics page value of pages visited |
| `SpecialDay` | Closeness of the session date to a special day (0 to 1) |

**Unordered label columns (7)** — one-hot encoded:

| Feature | Levels | Meaning |
|---|---|---|
| `Month` | 10 | Month of the session |
| `OperatingSystems` | 8 | Anonymised OS identifier |
| `Browser` | 13 | Anonymised browser identifier |
| `Region` | 9 | Anonymised geographic region |
| `TrafficType` | 20 | Anonymised traffic source |
| `VisitorType` | 3 | Returning, new, or other |
| `Weekend` | 2 | Whether the session fell on a weekend |

`OperatingSystems`, `Browser`, `Region` and `TrafficType` are stored as integers
in the source file, but the integers are arbitrary identifiers with no ordering.
They are therefore one-hot encoded rather than scaled, so that the models are not
told that "browser 13 is greater than browser 2". One-hot expansion takes the
17 original columns to 75 model inputs.

### Preprocessing

A single scikit-learn `ColumnTransformer` handles both halves of the feature set:
`StandardScaler` on the ten continuous metrics, `OneHotEncoder(handle_unknown="ignore")`
on the seven label columns. This transformer is fitted **inside each model's
`Pipeline`**, so it is persisted together with the estimator. The Streamlit app
can therefore score a raw, untouched CSV with no manual preparation, and there is
no risk of train/serve skew.

---

## c. GitHub repository link

https://github.com/satijagaurav/shopper-intent-classifier

```
shopper-intent-classifier/
├── app.py                              Streamlit front end
├── requirements.txt                    Pinned dependencies
├── requirements-dev.txt                Adds pytest for running tests/
├── README.md
├── test_data.csv                       Held-out 20% split (2,441 rows, features + target)
├── data/
│   └── online_shoppers_intention.csv   Raw UCI dataset
├── model/
│   ├── logistic_regression.joblib      Fitted preprocessing + estimator pipeline
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metrics.json                    Metrics and chosen hyperparameters
├── notebooks/
│   └── model_development.ipynb         EDA, training and evaluation walkthrough
├── tests/
│   └── test_preprocessing.py           Unit tests for CSV validation and cleaning
└── src/
    ├── config.py                       Paths, column groups, seed
    ├── preprocessing.py                Feature pipeline and CSV validation
    ├── metrics.py                      The six evaluation measures
    └── train_models.py                 Grid search, fitting, artifact export
```

### Reproducing the results

```bash
pip install -r requirements.txt
python -m src.train_models     # retrains, rewrites model/ and test_data.csv
streamlit run app.py           # launches the app on http://localhost:8501
```

Every random operation is seeded (`SEED = 2026` in `src/config.py`), so a rerun
reproduces the table below exactly.

### Tests

`tests/test_preprocessing.py` covers the CSV-validation path: a missing column,
a blank or unreadable target cell, a missing categorical value, and non-numeric
junk in a numeric column. All of these are rejected with a clear, caught error
rather than a raw traceback.

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

---

## d. Models used

All five classifiers are trained on the same 9,764-row training split and scored
on the same untouched 2,441-row held-out split.

Each model is tuned with a **5-fold stratified grid search optimising F1**. F1 is
the right search objective under this class imbalance because it refuses to
reward a model that simply predicts the majority class. Where the estimator
supports `class_weight`, both `None` and `"balanced"` are included in the grid,
so the search decides for itself whether rebalancing helps rather than having it
imposed.

Hyperparameters selected by the search:

| Model | Selected hyperparameters |
|---|---|
| Logistic Regression | `C=0.1`, `class_weight='balanced'` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=20`, `class_weight=None` |
| K-Nearest Neighbours | `n_neighbors=5`, `weights='distance'` |
| Naive Bayes (Gaussian) | `var_smoothing=1e-3` |
| Support Vector Machine | `kernel='rbf'`, `C=1.0`, `class_weight='balanced'` |

### Comparison table

Scores on the held-out test split. Best value in each column is in bold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8619 | 0.9117 | 0.5414 | 0.7696 | 0.6357 | 0.5667 |
| Decision Tree | **0.9070** | **0.9321** | **0.7370** | 0.6309 | **0.6798** | **0.6284** |
| KNN | 0.8763 | 0.7935 | 0.6786 | 0.3979 | 0.5017 | 0.4568 |
| Naive Bayes | 0.6534 | 0.7939 | 0.2864 | **0.8141** | 0.4237 | 0.3200 |
| Support Vector Machine | 0.8800 | 0.9190 | 0.5906 | 0.7592 | 0.6644 | 0.5996 |

Supporting confusion-matrix counts, out of 2,441 sessions of which 382 were real
purchases:

| ML Model Name | True neg. | False pos. | False neg. | True pos. | Sessions flagged |
|---|---|---|---|---|---|
| Logistic Regression | 1810 | 249 | 88 | 294 | 543 |
| Decision Tree | 1973 | 86 | 141 | 241 | 327 |
| KNN | 1987 | 72 | 230 | 152 | 224 |
| Naive Bayes | 1284 | 775 | 71 | 311 | 1086 |
| Support Vector Machine | 1858 | 201 | 92 | 290 | 491 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong ranking ability (AUC 0.9117) but weak thresholded precision (0.5414). The grid search chose `class_weight='balanced'`, which pushes it into a high-recall regime: it catches 294 of 382 real purchases, at the cost of also flagging 249 sessions that never convert. The gap between its high AUC and its mid-range F1 says the ordering of its scores is good while its default 0.5 cut-point is poorly placed. Heavy regularisation (`C=0.1`) winning the search indicates the one-hot expanded space is wide enough that the model overfits without it. A useful, well-calibrated baseline, but a linear boundary cannot capture the threshold behaviour of `PageValues`. |
| **Decision Tree** | **The best model on this dataset**, leading on accuracy, AUC, F1 and MCC simultaneously. The reason is visible in its split importances: `PageValues` alone accounts for about 77% of the total, and its relationship with purchasing is sharp and threshold-like rather than gradual, which is exactly the shape an axis-aligned split captures perfectly while a linear or distance-based model has to approximate. It is also the most economical model in business terms, flagging only 327 sessions to capture 241 purchases. Notably the search chose `class_weight=None` here, meaning the tree did not need rebalancing to handle the skew. Pruning to `max_depth=5` with 20 samples per leaf is what keeps it from memorising noise. |
| **KNN** | The weakest of the usable models (MCC 0.4568) and the clearest casualty of dimensionality. After one-hot encoding, the feature space has 75 dimensions of which the large majority are sparse binary columns, and Euclidean distance becomes close to uninformative there: every point is roughly equidistant from every other. The result is a conservative classifier that achieves decent precision (0.6786) by only committing when neighbours strongly agree, but misses 230 of 382 real purchases, giving the worst recall of any model at 0.3979. Its AUC of 0.7935 is the second-lowest, confirming the problem is in the representation rather than the threshold. |
| **Naive Bayes** | Clearly the worst performer (accuracy 0.6534, MCC 0.3200), and instructive about *why*. Its conditional-independence assumption is violated badly and repeatedly here: `ProductRelated` and `ProductRelated_Duration` are near-duplicates of each other, as are `BounceRates` and `ExitRates`. Counting correlated evidence as if it were independent makes the model wildly overconfident, so it flags 1,086 of 2,441 sessions as purchases and earns a precision of just 0.2864. The Gaussian likelihood is also a poor fit for one-hot columns and for `PageValues`, which is zero for most sessions. The one thing it does well is recall (0.8141, the highest of all five) because it over-predicts the positive class, but at this precision that is not a usable trade. Its AUC of 0.7939 being far above its F1 shows the ranking is not worthless; the decision rule is. |
| **Support Vector Machine** | A close and credible runner-up, second on accuracy (0.8800), AUC (0.9190), F1 (0.6644) and MCC (0.5996). The RBF kernel does capture non-linear structure that logistic regression cannot, and it beats the linear model on every single metric, but it never closes the gap to the tree. With `class_weight='balanced'` selected, it settles into a profile very similar to logistic regression — high recall at 0.7592, moderate precision at 0.5906 — just shifted uniformly better. It is also by far the most expensive to train, roughly six times the tuning cost of the tree for a worse result, since kernel SVM scales poorly with the 9,764 training rows. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It takes four of the six metrics outright (accuracy 0.9070, AUC 0.9321, precision 0.7370, F1 0.6798) and leads MCC at 0.6284, which is the measure that matters most under 15.6% class imbalance because it stays honest when the classes are skewed. It is simultaneously the cheapest to train, the fastest to serve, and the only model whose decisions can be read off directly as rules. The practical argument is just as strong: it identifies 241 purchasing sessions while flagging only 327 in total, so an intervention budget driven by this model is spent on sessions that genuinely convert, where the SVM and logistic regression would waste roughly half of it and Naive Bayes nearly three-quarters. |

---

## Streamlit application

### Required features

| Requirement | Where it is implemented |
|---|---|
| Dataset upload option (CSV) | Sidebar file uploader; the bundled `test_data.csv` is offered as a fallback so the app is never empty on first load |
| Model selection dropdown | Sidebar selector across all five classifiers, which also displays the hyperparameters the grid search chose for the selected model |
| Display of evaluation metrics | Six metric cards (Accuracy, AUC, Precision, Recall, F1, MCC) on the *Selected model* tab, plus a full five-model table on the *All five models* tab |
| Confusion matrix / classification report | Both: an annotated confusion-matrix heatmap and a per-class classification report table, alongside a ROC curve |

### Additional features

- **All five models tab** scores every classifier on the currently loaded test
  data in one table, with the best value per metric highlighted, a selectable
  per-metric bar chart, and a CSV download of the comparison.
- **Data preview tab** shows the uploaded rows and the target balance.
- **Column validation** reports exactly which columns are missing by name if an
  incompatible CSV is uploaded, instead of failing with a stack trace.
- **Flexible target parsing** accepts `True`/`False`, `1`/`0` or `yes`/`no` in the
  `Revenue` column.

### No retraining in the app

The app calls `joblib.load` on the five saved pipelines and only ever calls
`.predict()` and `.predict_proba()` / `.decision_function()` on them. There is no
`.fit()` anywhere in `app.py`. Models are cached with `@st.cache_resource` so
they are deserialised once per session rather than per interaction.

### Uploaded CSV format

The uploaded file must contain all 17 feature columns listed above plus the
`Revenue` target column. `test_data.csv` in the repository root is exactly this
format and can be uploaded directly, or any subset of its rows.

---

## Tooling note

AUC for the SVM is computed from `decision_function` margins rather than Platt-scaled
probabilities. The AUC is rank-based, so the margins serve it directly and this
avoids an unnecessary five-fold calibration refit; the resulting score is
identical in ordering.
