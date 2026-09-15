"""Notebook analysis helpers; use shared data and split contracts.

Persist supplemental diagnostics without replacing existing experiment metrics.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from . import config as C
from .data import load_data, verify_dataset
from .splits import make_split, get_xy
from .models import build_models, evaluate

def load_and_summarize():
    # Load through the shared cleaning function without replacing the saved log.
    df, cleaning_log = load_data(write_log=False)
    display(pd.Series({k: cleaning_log[k] for k in [
        "rows_raw", "cols_raw", "duplicate_urls_removed", "rows_clean",
        "n_unique_domains", "fingerprint_matches"]}, name="Value").to_frame())
    display(df[C.FEATURE_SETS["url_only"]].describe().T)
    print("Missing values:", cleaning_log["columns_with_missing"])
    verify_dataset(df, strict=True)
    return df, cleaning_log


def plot_distributions(df):
    counts = df[C.LABEL].value_counts().reindex([0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["Phishing (0)", "Legitimate (1)"], counts.values)
    axes[0].set(title="Cleaned class distribution", ylabel="URLs")
    for label, name in [(0, "Phishing"), (1, "Legitimate")]:
        axes[1].hist(df.loc[df[C.LABEL] == label, "URLLength"],
                     bins=60, range=(0, 300), alpha=0.5, label=name)
    axes[1].set(title="URL length (display limited to 0-300)",
                xlabel="Characters", ylabel="URLs")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(C.FIGURES_DIR / "notebook_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def prepare_split(df):
    # Reuse the existing assignment; do not regenerate the project's split.
    if not C.SPLIT_FILE.exists():
        raise FileNotFoundError("Expected the project's saved split_assignment.csv.")
    assignment = make_split(df, strategy="stratified")
    X_train, X_test, y_train, y_test = get_xy(df, assignment, "url_only")
    print(f"Training: {len(X_train):,}; testing: {len(X_test):,}")
    print(f"URL-only predictors: {X_train.shape[1]}")
    train_domains = set(df.loc[assignment == "train", "Domain"])
    test_domains = set(df.loc[assignment == "test", "Domain"])
    print("Recorded domains shared across partitions:", len(train_domains & test_domains))
    return assignment, X_train, X_test, y_train, y_test


def quality_diagnostics(df, assignment):
    # Additional quality checks; summaries use training observations only.
    import numpy as np
    train_frame = df.loc[assignment == "train", C.FEATURE_SETS["full"]]
    numeric = train_frame.select_dtypes(include="number")
    quality = pd.DataFrame({
        "dtype": train_frame.dtypes.astype(str),
        "missing": train_frame.isna().sum(),
        "unique_train": train_frame.nunique(),
    })
    quality["non_finite"] = (~np.isfinite(numeric)).sum()
    display(quality)
    display(numeric.describe(percentiles=[0.01, 0.5, 0.95, 0.99]).T)
    display(pd.crosstab(assignment, df[C.LABEL], normalize="index")
            .rename(columns={0: "phishing_share", 1: "legitimate_share"}))
    print("Constant training predictors:", quality.index[quality["unique_train"] <= 1].tolist())

    quality.to_csv(C.RESULTS_DIR / "notebook_training_quality.csv")
    numeric.describe(percentiles=[0.01, 0.5, 0.95, 0.99]).T.to_csv(C.RESULTS_DIR / "notebook_training_summary.csv")
    pd.crosstab(assignment, df[C.LABEL], normalize="index").to_csv(C.RESULTS_DIR / "notebook_split_proportions.csv")


def review_saved_metrics():
    # Display existing experiment artifacts; this cell does not retrain models.
    records = [json.loads(p.read_text()) for p in sorted(C.RESULTS_DIR.glob("metrics_*.json"))]
    if not records:
        raise FileNotFoundError("No saved model metrics found in results/.")
    metrics = pd.DataFrame(records)
    display(metrics[["feature_set", "model", "n_features", "accuracy",
                     "precision", "recall", "f1", "roc_auc"]].round(6))
    print("Saved precision, recall, and F1 use legitimate (1) as positive.")
    rf = next(r for r in records if r["feature_set"] == "url_only" and r["model"] == "random_forest")
    cm = rf["confusion_matrix"]
    print(f"Phishing precision: {cm['tn'] / (cm['tn'] + cm['fn']):.2%}")
    print(f"Phishing recall: {cm['tn'] / (cm['tn'] + cm['fp']):.2%}")
    return records


def phishing_summary(records):
    # Derive phishing-oriented metrics from saved confusion matrices.
    # This uses existing artifacts, not fresh predictions or a new model fit.
    rows = []
    for record in records:
        c = record["confusion_matrix"]
        detected, missed = c["tn"], c["fp"]
        false_alerts, accepted = c["fn"], c["tp"]
        precision = detected / (detected + false_alerts) if detected + false_alerts else 0.0
        recall = detected / (detected + missed) if detected + missed else 0.0
        legitimate_recall = accepted / (accepted + false_alerts) if accepted + false_alerts else 0.0
        rows.append({"features": record["feature_set"], "model": record["model"],
                     "phishing_precision": precision, "phishing_recall": recall,
                     "phishing_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                     "balanced_accuracy": (recall + legitimate_recall) / 2,
                     "missed_phishing": missed, "false_alerts": false_alerts})
    phishing_metrics = pd.DataFrame(rows)
    display(phishing_metrics.round(6))

    phishing_metrics.to_csv(C.RESULTS_DIR / "notebook_phishing_metrics.csv", index=False)


def train_initial_model(df, assignment):
    """Use the project roster and evaluation writer under a distinct run name."""
    inputs = get_xy(df, assignment, "url_only")
    model = build_models()["logistic_regression"]
    return evaluate(model, "logistic_regression_notebook", "url_only", *inputs)
