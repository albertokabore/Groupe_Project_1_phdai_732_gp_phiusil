"""Part 2 hyperparameter search on the frozen stratified split.

The course brief describes Surprise's GridSearchCV over an SVD recommender. This
project is a binary classifier, so the equivalent is scikit-learn's GridSearchCV
over the Part 1 roster, scored on phishing (C.POS_LABEL) F1 rather than RMSE.

Search runs on url_only, the only feature set where the Part 1 models leave
errors to remove. full and no_derived already sit at or near 1.0, so tuning
them measures nothing; tuned configurations are refit on them afterwards only
so the report can show the three-way comparison with the same settings.

Model selection uses cross-validation on the training partition only. The test
partition is never seen here. final_eval.py opens it once.
"""

import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as C
from .data import load_data, verify_dataset
from .models import build_models
from .splits import get_xy, make_split

TUNING_DIR = C.RESULTS_DIR / "tuning"
TUNED_FEATURE_SET = "url_only"
REFIT_METRIC = "f1_phishing"

# Grids are deliberately small and centered on the Part 1 defaults so each
# search answers "does moving away from the default help", not "what is the
# global optimum". Every grid contains the Part 1 default configuration, so the
# best CV score can never be worse than the untuned model's by construction.
PARAM_GRIDS = {
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        "clf__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": [1, 5, 20],
    },
    "random_forest": {
        "n_estimators": [150, 300],
        "max_depth": [None, 30],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", 0.5],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [100, 300, 600],
        "max_leaf_nodes": [31, 127],
        "l2_regularization": [0.0, 1.0],
    },
}


def build_tuning_models():
    """Part 1 roster minus the majority baseline, plus gradient boosting.

    Gradient boosting is the one addition. Random forest and gradient boosting
    are the two standard ensemble families for tabular data, and Part 1 only
    tried one of them.
    """
    models = {k: v for k, v in build_models().items() if k != "baseline_majority"}
    # The search parallelizes over folds and candidates, so the forest itself
    # runs single-threaded to avoid oversubscribing the machine.
    models["random_forest"].set_params(n_jobs=1)
    models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        early_stopping=False, random_state=C.SEED
    )
    return models


SCORE_NAMES = ["f1_phishing", "precision_phishing", "recall_phishing",
               "roc_auc", "pr_auc_phishing"]


def scoring(estimator, X, y):
    """All CV metrics from one predict and one predict_proba call.

    A single callable, not a dict of make_scorer objects, on purpose. sklearn's
    multimetric scorer caches predict_proba output after the first scorer has
    already reduced it to one column. For models without decision_function
    (trees, forests) the "roc_auc" scorer cached P(legitimate), and a
    pos_label=0 average-precision scorer then silently read that column. The
    first grid search reported PR AUC near the base rate for exactly those two
    models. Selecting the phishing column here removes the shared cache.
    """
    pos = C.POS_LABEL
    pred = estimator.predict(X)
    p_phish = estimator.predict_proba(X)[:, list(estimator.classes_).index(pos)]
    is_phish = np.asarray(y) == pos
    return {
        "f1_phishing": f1_score(y, pred, pos_label=pos, zero_division=0),
        "precision_phishing": precision_score(y, pred, pos_label=pos, zero_division=0),
        "recall_phishing": recall_score(y, pred, pos_label=pos, zero_division=0),
        "roc_auc": roc_auc_score(is_phish, p_phish),
        "pr_auc_phishing": average_precision_score(is_phish, p_phish),
    }


def cv_splitter():
    """Shuffled, seeded folds so every teammate gets identical CV partitions."""
    return StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)


def _default_params(model, grid):
    """The untuned model's value for each searched parameter."""
    params = model.get_params()
    return {k: params[k] for k in grid}


def _row_for(results, params):
    """Locate the cv_results_ row whose parameters equal params."""
    for i, candidate in enumerate(results["params"]):
        if all(candidate[k] == v for k, v in params.items()):
            return i
    raise KeyError(f"Default configuration {params} is missing from the grid.")


def run_tuning(models=None):
    """Grid-search each model on url_only training data. Writes:

    results/tuning/cv_results_<model>.csv  every candidate, every scorer
    results/tuning_summary.csv             default vs best, per model
    results/tuning_best_params.json        consumed by final_eval.py
    results/tuning_manifest.json           provenance
    """
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    df, log = load_data(write_log=False)
    verify_dataset(df, strict=True)
    assignment = make_split(df)
    split_hash = hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest()
    X_train, _, y_train, _ = get_xy(df, assignment, TUNED_FEATURE_SET)

    roster = build_tuning_models()
    names = models or list(PARAM_GRIDS)
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "url_fingerprint": log["fingerprint"],
        "split_sha256": split_hash,
        "feature_set": TUNED_FEATURE_SET,
        "n_train": int(len(X_train)),
        "seed": C.SEED,
        "cv": f"StratifiedKFold(n_splits={C.CV_FOLDS}, shuffle=True, random_state={C.SEED})",
        "refit_metric": REFIT_METRIC,
        "positive_class": "phishing (0)",
        "param_grids": PARAM_GRIDS,
        "sklearn": sklearn.__version__,
        "completed": [],
    }
    manifest_path = C.RESULTS_DIR / "tuning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    summary, best_params = [], {}
    for name in names:
        model, grid = roster[name], PARAM_GRIDS[name]
        n_candidates = int(np.prod([len(v) for v in grid.values()]))
        print(f"Searching {name}: {n_candidates} candidates x {C.CV_FOLDS} folds", flush=True)
        search = GridSearchCV(
            clone(model), grid, scoring=scoring, refit=False,
            cv=cv_splitter(), n_jobs=-1, return_train_score=False, error_score="raise",
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        res = search.cv_results_
        table = pd.DataFrame(res).drop(columns=["params"])
        table.to_csv(TUNING_DIR / f"cv_results_{name}.csv", index=False)

        best = int(np.argmax(res[f"mean_test_{REFIT_METRIC}"]))
        default = _row_for(res, _default_params(model, grid))
        record = {"model": name, "n_candidates": n_candidates, "elapsed_seconds": elapsed}
        for tag, i in (("default", default), ("best", best)):
            for metric in SCORE_NAMES:
                record[f"{tag}_cv_{metric}"] = float(res[f"mean_test_{metric}"][i])
            record[f"{tag}_cv_f1_phishing_sd"] = float(res[f"std_test_{REFIT_METRIC}"][i])
        record["best_params"] = json.dumps(res["params"][best], default=str)
        record["default_params"] = json.dumps(res["params"][default], default=str)
        record["cv_f1_gain"] = record["best_cv_f1_phishing"] - record["default_cv_f1_phishing"]
        summary.append(record)
        best_params[name] = res["params"][best]

        manifest["completed"].append({"model": name, "elapsed_seconds": elapsed})
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"  best CV F1={record['best_cv_f1_phishing']:.6f} "
              f"(default {record['default_cv_f1_phishing']:.6f}) "
              f"{res['params'][best]} in {elapsed:.0f}s", flush=True)

    # Merge with any earlier summary so a single-model rerun does not drop the others.
    summary_path = C.RESULTS_DIR / "tuning_summary.csv"
    params_path = C.RESULTS_DIR / "tuning_best_params.json"
    new = pd.DataFrame(summary)
    if summary_path.exists() and models:
        old = pd.read_csv(summary_path)
        new = pd.concat([old[~old.model.isin(new.model)], new], ignore_index=True)
        best_params = {**json.loads(params_path.read_text()), **best_params}
    order = [m for m in PARAM_GRIDS if m in set(new.model)]
    new.set_index("model").loc[order].reset_index().to_csv(summary_path, index=False)
    params_path.write_text(json.dumps(best_params, indent=2, default=str))

    assert hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest() == split_hash
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return new


def tuned_models():
    """Roster with the searched parameters applied. Parallel forest again,
    since single fits are no longer running inside a parallel search."""
    params = json.loads((C.RESULTS_DIR / "tuning_best_params.json").read_text())
    roster = build_tuning_models()
    roster["random_forest"].set_params(n_jobs=-1)
    return {name: clone(roster[name]).set_params(**p) for name, p in params.items()}


if __name__ == "__main__":
    run_tuning()
