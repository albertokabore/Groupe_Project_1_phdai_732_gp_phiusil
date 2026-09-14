"""Single-feature leakage screen. Run this before any modeling.

For each candidate feature, fit a depth-1 decision tree and record accuracy.
A feature that reaches near-perfect accuracy alone is not a strong predictor.
It is a restatement of the label produced during dataset construction.

Output: results/leakage_screen.csv, sorted descending.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def single_feature_accuracy(df, features=None, threshold=0.95):
    features = features or C.FEATURE_SETS["full"]
    rows = []
    y = df[C.LABEL]
    for col in features:
        x = df[[col]]
        stump = DecisionTreeClassifier(max_depth=1, random_state=C.SEED).fit(x, y)
        rows.append({"feature": col, "single_feature_accuracy": stump.score(x, y)})

    out = (
        pd.DataFrame(rows)
        .sort_values("single_feature_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    out["flagged"] = out["single_feature_accuracy"] >= threshold
    out.to_csv(C.RESULTS_DIR / "leakage_screen.csv", index=False)
    return out


def class_constancy(df, features=None):
    """Find features that take a single value across an entire class.

    This is the sharpest form of the artifact: if every legitimate row shares
    one value, the feature is a label in disguise.
    """
    features = features or C.FEATURE_SETS["full"]
    rows = []
    for col in features:
        grp = df.groupby(C.LABEL)[col]
        for label_value, series in grp:
            top_share = series.value_counts(normalize=True).iloc[0]
            rows.append(
                {
                    "feature": col,
                    "label": label_value,
                    "modal_value": series.mode().iloc[0],
                    "modal_share": top_share,
                }
            )
    out = pd.DataFrame(rows).sort_values("modal_share", ascending=False)
    out.to_csv(C.RESULTS_DIR / "class_constancy.csv", index=False)
    return out
