"""Model roster and the evaluation contract.

Every result in the report comes from evaluate(), which writes a JSON record to
results/. Report writers read those files. Nobody transcribes a number from a
Colab cell into a Word document by hand.
"""

import json
from datetime import datetime, timezone

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def build_models():
    """The agreed roster. Add a model here, not in your own notebook."""
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=C.SEED)),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(random_state=C.SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=C.SEED
        ),
    }


def evaluate(model, name, feature_set, X_train, X_test, y_train, y_test, cv=True):
    """Fit, score, persist. Returns the metrics dict."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    try:
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
    except (AttributeError, IndexError):
        auc = None

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    record = {
        "model": name,
        "feature_set": feature_set,
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "seed": C.SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    if cv and name != "baseline_majority":
        scores = cross_val_score(
            model, X_train, y_train, cv=C.CV_FOLDS, scoring="f1", n_jobs=-1
        )
        record["cv_f1_mean"] = float(np.mean(scores))
        record["cv_f1_sd"] = float(np.std(scores, ddof=1))

    out = C.RESULTS_DIR / f"metrics_{feature_set}_{name}.json"
    out.write_text(json.dumps(record, indent=2, default=float))
    return record
