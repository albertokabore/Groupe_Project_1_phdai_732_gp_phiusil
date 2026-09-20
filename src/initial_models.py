"""Run the agreed initial-model experiment on the existing frozen splits."""
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from . import config as C
from .data import load_data, verify_dataset
from .models import build_models, evaluate
from .splits import get_xy, make_split

STRATEGIES = ("stratified", "grouped")


def run_initial_models(strategies=STRATEGIES):
    """Fit every configuration on each split, preserve earlier metrics, persist provenance."""
    if not C.SPLIT_FILE.exists():
        raise FileNotFoundError("The frozen split must already exist.")
    df, log = load_data(write_log=False)
    verify_dataset(df, strict=True)
    split_hash = hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = C.RESULTS_DIR / "initial_model_runs" / stamp
    archive.mkdir(parents=True)
    for path in C.RESULTS_DIR.glob("metrics_*.json"):
        shutil.copy2(path, archive / path.name)
    manifest = {
        "started_utc": stamp, "status": "running",
        "dataset_sha256": hashlib.sha256(C.RAW_CSV.read_bytes()).hexdigest(),
        "url_fingerprint": log["fingerprint"], "split_sha256": split_hash,
        "seed": C.SEED, "cv_folds": C.CV_FOLDS,
        "split_strategies": list(strategies),
        "positive_class": f"phishing ({C.POS_LABEL})",
        "feature_sets": C.FEATURE_SETS,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        "prior_metrics_archive": str(archive.relative_to(C.ROOT)),
        "completed": [],
    }
    manifest_path = C.RESULTS_DIR / "initial_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    records = []
    # Bound CPU use and avoid spawning multiple full forests in subprocesses.
    with parallel_backend("threading", n_jobs=4), threadpool_limits(limits=1):
        for strategy in strategies:
            assignment = make_split(df, strategy=strategy)
            for feature_set in C.FEATURE_SETS:
                inputs = get_xy(df, assignment, feature_set)
                for name, model in build_models().items():
                    if name == "random_forest":
                        model.set_params(n_jobs=4)
                    print(f"Training {strategy}/{feature_set}/{name} (including training CV)",
                          flush=True)
                    started = time.perf_counter()
                    record = evaluate(model, name, feature_set, *inputs, split=strategy)
                    record["elapsed_seconds"] = time.perf_counter() - started
                    records.append(record)
                    manifest["completed"].append({"split": strategy, "feature_set": feature_set,
                                                  "model": name,
                                                  "elapsed_seconds": record["elapsed_seconds"]})
                    manifest_path.write_text(json.dumps(manifest, indent=2))
                    print(f"Finished: accuracy={record['accuracy']:.6f}, "
                          f"F1={record['f1']:.6f}, seconds={record['elapsed_seconds']:.1f}",
                          flush=True)
    assert hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest() == split_hash
    pd.json_normalize(records).to_csv(C.RESULTS_DIR / "initial_model_comparison.csv", index=False)
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return records


def show_initial_results(split="stratified"):
    """Display the completed experiment and save model comparison figures.

    Figures are drawn for one split at a time. The returned table covers every
    split that was run.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    manifest = json.loads((C.RESULTS_DIR / "initial_model_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Initial model training has not completed.")
    table = pd.read_csv(C.RESULTS_DIR / "initial_model_comparison.csv")
    strategies = manifest.get("split_strategies", ["stratified"])
    expected = 4 * len(C.FEATURE_SETS) * len(strategies)
    if len(table) != expected:
        raise ValueError(f"Expected {expected} rows: four models, "
                         f"{len(C.FEATURE_SETS)} feature sets, {len(strategies)} splits.")
    if split not in strategies:
        raise ValueError(f"No results for split {split!r}. Ran: {strategies}.")
    for _, row in table.iterrows():
        tn, fp, fn, tp = [row[f"confusion_matrix.{k}"] for k in ("tn", "fp", "fn", "tp")]
        if tn + fp + fn + tp != row.n_test or not np.isclose((tn + tp) / row.n_test, row.accuracy):
            raise ValueError("Confusion matrix does not agree with reported metrics.")
    # Derived from the confusion matrix, which keeps sklearn's sorted-label order:
    # tn counts phishing (0) predicted as phishing. These must equal the recorded
    # precision, recall, and F1 now that evaluate() scores against phishing.
    table["phishing_precision"] = (table["confusion_matrix.tn"] /
        (table["confusion_matrix.tn"] + table["confusion_matrix.fn"])).fillna(0)
    table["phishing_recall"] = table["confusion_matrix.tn"] / (table["confusion_matrix.tn"] + table["confusion_matrix.fp"])
    table["phishing_f1"] = (2 * table["confusion_matrix.tn"] /
        (2 * table["confusion_matrix.tn"] + table["confusion_matrix.fn"] + table["confusion_matrix.fp"]))
    mismatch = ~np.isclose(table["phishing_f1"], table["f1"])
    if mismatch.any():
        raise ValueError("Derived phishing F1 disagrees with the recorded F1. "
                         "Check config.POS_LABEL and models.evaluate().")
    table.to_csv(C.RESULTS_DIR / "initial_model_comparison.csv", index=False)
    display(table[["split", "feature_set", "model", "accuracy", "phishing_precision",
                   "phishing_recall", "phishing_f1", "roc_auc", "cv_f1_mean"]].round(6))
    print(f"All metrics use phishing ({C.POS_LABEL}) as the positive class.")
    if "grouped" in strategies:
        print("Grouped runs: the train/test partition shares no domains, but the "
              "cross-validation folds inside the training set are not domain-grouped, "
              "so cv_f1_mean is not domain-clean for those rows.")
    print("Completed:", manifest["finished_utc"])

    shown = table[table.split == split]
    ax = shown.pivot(index="model", columns="feature_set", values="accuracy").plot.bar(
        figsize=(10, 5), ylim=(0, 1.05), title=f"Initial model test accuracy ({split} split)")
    ax.set_ylabel("Accuracy")
    ax.figure.tight_layout()
    ax.figure.savefig(C.FIGURES_DIR / f"initial_model_accuracy_{split}.png", dpi=150)
    plt.show()
    url_only = shown[shown.feature_set == "url_only"]
    fig, axes = plt.subplots(1, len(url_only), figsize=(15, 3.8))
    for ax, (_, row) in zip(np.atleast_1d(axes), url_only.iterrows()):
        matrix = np.array([[row["confusion_matrix.tn"], row["confusion_matrix.fp"]],
                           [row["confusion_matrix.fn"], row["confusion_matrix.tp"]]], dtype=int)
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Phishing", "Legitimate"],
               yticklabels=["Phishing", "Legitimate"], xlabel="Predicted", ylabel="Actual",
               title=row["model"].replace("_", " "))
    fig.suptitle(f"URL-only models: held-out confusion matrices ({split} split)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"initial_model_confusion_matrices_{split}.png", dpi=150)
    plt.show()
    return table


if __name__ == "__main__":
    run_initial_models()
