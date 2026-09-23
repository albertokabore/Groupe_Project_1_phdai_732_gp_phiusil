"""Part 2 final evaluation: the one place the held-out test partition is opened
for tuned models.

Order of operations matters and is enforced here:
  1. The final model is chosen from tuning_summary.csv by cross-validated F1.
     The test partition plays no part in that choice.
  2. Decision thresholds are chosen from out-of-fold training predictions,
     then applied to test unchanged.
  3. Only then are test metrics computed and written.

The brief asks for RMSE and MAE. Those are rating-regression metrics; on a
binary label they are only meaningful against the predicted probability, so
they are reported as probability_rmse (the square root of the Brier score) and
probability_mae, next to the classification metrics that actually drive the
analysis.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from . import config as C
from .data import load_data, verify_dataset
from .splits import get_xy, make_split
from .tuning import TUNED_FEATURE_SET, build_tuning_models, cv_splitter, tuned_models

N_BOOTSTRAP = 1000
# Legitimate-URL false-alarm budget for the alternative operating point. One
# in a thousand legitimate URLs blocked is a common starting target for a
# user-facing filter; the group can move it.
FPR_BUDGET = 0.001
MIN_GROUP_N = 200


# --- Metric helpers -----------------------------------------------------------

def p_phishing(model, X):
    """P(phishing). Column chosen by class value, never by position."""
    return model.predict_proba(X)[:, list(model.classes_).index(C.POS_LABEL)]


def predict_at(p_phish, threshold):
    """Label vector from P(phishing): phishing (0) at or above threshold."""
    return np.where(p_phish >= threshold, C.POS_LABEL, 1 - C.POS_LABEL)


def classification_metrics(y, p_phish, threshold=0.5):
    """Everything the report quotes for one set of predictions.

    Confusion counts are named for what they mean here, not sklearn's tn/fp
    names, because phishing is the positive class and sits at index 0.
    """
    pos = C.POS_LABEL
    y = np.asarray(y)
    pred = predict_at(p_phish, threshold)
    is_phish = (y == pos).astype(float)
    (phish_caught, phish_missed), (legit_flagged, legit_passed) = confusion_matrix(
        y, pred, labels=[pos, 1 - pos]
    )
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, pos_label=pos, zero_division=0),
        "recall": recall_score(y, pred, pos_label=pos, zero_division=0),
        "f1": f1_score(y, pred, pos_label=pos, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(is_phish, p_phish),
        "pr_auc": average_precision_score(is_phish, p_phish),
        "log_loss": log_loss(is_phish, p_phish, labels=[0.0, 1.0]),
        "brier": brier_score_loss(is_phish, p_phish),
        "probability_rmse": float(np.sqrt(np.mean((p_phish - is_phish) ** 2))),
        "probability_mae": float(np.mean(np.abs(p_phish - is_phish))),
        "legit_false_alarm_rate": legit_flagged / (legit_flagged + legit_passed),
        "phishing_miss_rate": phish_missed / (phish_caught + phish_missed),
        "phishing_caught": int(phish_caught),
        "phishing_missed": int(phish_missed),
        "legit_flagged": int(legit_flagged),
        "legit_passed": int(legit_passed),
        "n_test": int(len(y)),
    }


def bootstrap_ci(y, pred, n=N_BOOTSTRAP, seed=C.SEED):
    """Percentile 95% intervals for phishing precision, recall and F1.

    Vectorized over confusion counts: resampling rows with replacement is
    equivalent to a multinomial draw over the four confusion cells.
    """
    pos = C.POS_LABEL
    y, pred = np.asarray(y), np.asarray(pred)
    cells = np.array([
        np.sum((y == pos) & (pred == pos)), np.sum((y != pos) & (pred == pos)),
        np.sum((y == pos) & (pred != pos)), np.sum((y != pos) & (pred != pos)),
    ])
    draws = np.random.default_rng(seed).multinomial(len(y), cells / cells.sum(), size=n)
    tp, fp, fn = draws[:, 0], draws[:, 1], draws[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = {
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn),
        }
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in stats.items()}


def choose_thresholds(y_train, oof):
    """Two operating points from out-of-fold training probabilities.

    f1_optimal         maximizes phishing F1
    fpr_budget         highest recall with legitimate false-alarm rate <= FPR_BUDGET
    """
    is_phish = (np.asarray(y_train) == C.POS_LABEL)
    precision, recall, thresholds = precision_recall_curve(is_phish, oof)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    f1_thr = float(thresholds[np.argmax(f1[:-1])])
    legit = np.sort(oof[~is_phish])
    # Threshold just above the (1 - budget) quantile of legitimate scores.
    k = int(np.floor(len(legit) * (1 - FPR_BUDGET)))
    budget_thr = float(np.nextafter(legit[min(k, len(legit) - 1)], np.inf))
    return {"default": 0.5, "f1_optimal": f1_thr, "fpr_budget": budget_thr}


def group_error_rates(frame, column, min_n=MIN_GROUP_N):
    """Per-group legitimate false-alarm rate and phishing miss rate.

    frame needs columns: column, label, pred. Groups with fewer than min_n
    URLs of a class get NaN for that class's rate rather than a noisy number.
    """
    pos = C.POS_LABEL
    rows = []
    for value, g in frame.groupby(column):
        legit, phish = g[g.label != pos], g[g.label == pos]
        rows.append({
            column: value,
            "n": len(g),
            "n_legit": len(legit),
            "n_phishing": len(phish),
            "share_phishing": len(phish) / len(g),
            "legit_false_alarm_rate": (legit.pred == pos).mean() if len(legit) >= min_n else np.nan,
            "phishing_miss_rate": (phish.pred != pos).mean() if len(phish) >= min_n else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


# --- Driver -------------------------------------------------------------------

def run_final_evaluation():
    """Writes, under results/:

    final_model_comparison.csv       default vs tuned, every model, url_only test
    final_feature_set_comparison.csv tuned models on all three feature sets
    final_model.json                 chosen model, params, test metrics, CIs, thresholds
    final_thresholds.csv             test metrics at each operating point
    final_test_predictions.csv       row index, label, P(phishing) for the final model
    final_errors.csv                 misclassified test URLs
    final_error_profile.csv          feature medians by outcome
    final_group_error_rates_tld.csv  per-TLD error rates
    final_group_error_rates_https.csv
    final_permutation_importance.csv
    final_grouped_split.json         same config refit on the domain-grouped split
    """
    df, log = load_data(write_log=False)
    verify_dataset(df, strict=True)
    assignment = make_split(df)
    X_train, X_test, y_train, y_test = get_xy(df, assignment, TUNED_FEATURE_SET)

    summary = pd.read_csv(C.RESULTS_DIR / "tuning_summary.csv")
    final_name = summary.loc[summary.best_cv_f1_phishing.idxmax(), "model"]
    tuned = tuned_models()
    defaults = build_tuning_models()
    defaults["random_forest"].set_params(n_jobs=-1)

    # 1. Default vs tuned, every model, url_only.
    rows, fitted = [], {}
    for name in tuned:
        for config_name, model in (("default", clone(defaults[name])), ("tuned", tuned[name])):
            print(f"Fitting {config_name} {name} on {TUNED_FEATURE_SET}", flush=True)
            model.fit(X_train, y_train)
            rows.append({"model": name, "config": config_name,
                         **classification_metrics(y_test, p_phishing(model, X_test))})
            fitted[(name, config_name)] = model
    comparison = pd.DataFrame(rows)
    comparison.to_csv(C.RESULTS_DIR / "final_model_comparison.csv", index=False)

    # 2. Tuned settings on the other two feature sets, for the three-way table.
    rows = []
    for feature_set in C.FEATURE_SETS:
        inputs = get_xy(df, assignment, feature_set)
        for name, model in tuned.items():
            if feature_set == TUNED_FEATURE_SET:
                m = fitted[(name, "tuned")]
            else:
                print(f"Fitting tuned {name} on {feature_set}", flush=True)
                m = clone(model).fit(inputs[0], inputs[2])
            rows.append({"feature_set": feature_set, "model": name,
                         **classification_metrics(inputs[3], p_phishing(m, inputs[1]))})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "final_feature_set_comparison.csv", index=False)

    # 3. Operating points from out-of-fold training probabilities.
    final = fitted[(final_name, "tuned")]
    print(f"Out-of-fold predictions for {final_name}", flush=True)
    oof = cross_val_predict(clone(tuned[final_name]), X_train, y_train,
                            cv=cv_splitter(), method="predict_proba", n_jobs=1)
    oof = oof[:, list(np.unique(y_train)).index(C.POS_LABEL)]
    thresholds = choose_thresholds(y_train, oof)
    p_test = p_phishing(final, X_test)
    threshold_rows = [{"operating_point": k, **classification_metrics(y_test, p_test, t)}
                      for k, t in thresholds.items()]
    pd.DataFrame(threshold_rows).to_csv(C.RESULTS_DIR / "final_thresholds.csv", index=False)

    headline = classification_metrics(y_test, p_test)
    pred = predict_at(p_test, 0.5)
    record = {
        "model": final_name,
        "feature_set": TUNED_FEATURE_SET,
        "selected_by": "highest mean CV phishing F1 in tuning_summary.csv (test not used)",
        "params": json.loads(summary.set_index("model").loc[final_name, "best_params"]),
        "test_metrics": headline,
        "bootstrap_95ci": bootstrap_ci(y_test, pred),
        "n_bootstrap": N_BOOTSTRAP,
        "thresholds": thresholds,
        "fpr_budget": FPR_BUDGET,
        "url_fingerprint": log["fingerprint"],
        "seed": C.SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # 4. Predictions and error analysis.
    test = df.loc[X_test.index]
    out = pd.DataFrame({"label": y_test, "pred": pred, "p_phishing": p_test}, index=X_test.index)
    out.rename_axis("row").to_csv(C.RESULTS_DIR / "final_test_predictions.csv")
    outcome = np.select(
        [(out.label == C.POS_LABEL) & (out.pred == C.POS_LABEL),
         (out.label == C.POS_LABEL) & (out.pred != C.POS_LABEL),
         (out.label != C.POS_LABEL) & (out.pred == C.POS_LABEL)],
        ["phishing_caught", "phishing_missed", "legit_flagged"], "legit_passed",
    )
    errors = pd.DataFrame({
        "URL": test["URL"], "TLD": test["TLD"], "label": out.label,
        "p_phishing": out.p_phishing, "outcome": outcome,
    })[lambda d: d.outcome.isin(["phishing_missed", "legit_flagged"])]
    errors.sort_values("p_phishing").rename_axis("row").to_csv(
        C.RESULTS_DIR / "final_errors.csv")
    X_test.assign(outcome=outcome).groupby("outcome").median().T.rename_axis(
        "feature").to_csv(C.RESULTS_DIR / "final_error_profile.csv")
    record["error_counts"] = pd.Series(outcome).value_counts().to_dict()
    record["error_confidence"] = {
        k: {"median_p_phishing": float(g.p_phishing.median()),
            "share_confident": float(((g.p_phishing < 0.1) | (g.p_phishing > 0.9)).mean())}
        for k, g in errors.groupby("outcome")
    }

    # 5. Error rates by group: fairness to legitimate sites.
    groups = test[["TLD", "IsHTTPS"]].assign(label=out.label, pred=out.pred)
    group_error_rates(groups, "TLD").to_csv(
        C.RESULTS_DIR / "final_group_error_rates_tld.csv", index=False)
    group_error_rates(groups, "IsHTTPS").to_csv(
        C.RESULTS_DIR / "final_group_error_rates_https.csv", index=False)

    # 6. Permutation importance on a fixed test sample.
    sample = X_test.sample(n=min(10000, len(X_test)), random_state=C.SEED)
    print("Permutation importance", flush=True)
    imp = permutation_importance(
        final, sample, y_test.loc[sample.index], n_repeats=5, random_state=C.SEED,
        scoring=make_scorer(f1_score, pos_label=C.POS_LABEL, zero_division=0), n_jobs=1,
    )
    pd.DataFrame({"feature": sample.columns, "f1_drop_mean": imp.importances_mean,
                  "f1_drop_sd": imp.importances_std}).sort_values(
        "f1_drop_mean", ascending=False).to_csv(
        C.RESULTS_DIR / "final_permutation_importance.csv", index=False)

    # 7. Domain-grouped split: does the model generalize to unseen domains?
    print("Refitting on the domain-grouped split", flush=True)
    g_assign = make_split(df, strategy="grouped")
    gX_train, gX_test, gy_train, gy_test = get_xy(df, g_assign, TUNED_FEATURE_SET)
    g_model = clone(tuned[final_name]).fit(gX_train, gy_train)
    grouped = {
        "model": final_name, "split": "grouped",
        "shared_domains_train_test": int(len(
            set(df.loc[gX_train.index, "Domain"]) & set(df.loc[gX_test.index, "Domain"]))),
        "test_metrics": classification_metrics(gy_test, p_phishing(g_model, gX_test)),
        "stratified_test_metrics": headline,
    }
    (C.RESULTS_DIR / "final_grouped_split.json").write_text(
        json.dumps(grouped, indent=2, default=float))

    (C.RESULTS_DIR / "final_model.json").write_text(json.dumps(record, indent=2, default=float))
    return record


# --- Figures ------------------------------------------------------------------
# Static PNGs for a printed report, so light surface only. Colors are role
# tokens; categorical slots follow a fixed order and are never cycled.
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES_1 = "#2a78d6"
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
}


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.axisbelow": True,
    })
    return plt


def make_final_figures():
    """Render every Part 2 figure from results/ into figures/."""
    from matplotlib.colors import LinearSegmentedColormap

    plt = _style()
    R, F = C.RESULTS_DIR, C.FIGURES_DIR
    final = json.loads((R / "final_model.json").read_text())
    final_label = LABELS[final["model"]]
    m = final["test_metrics"]

    # 1. Default vs tuned CV F1: dumbbell per model.
    s = pd.read_csv(R / "tuning_summary.csv").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(s))
    ax.hlines(y, s.default_cv_f1_phishing, s.best_cv_f1_phishing, color=AXIS, lw=2, zorder=1)
    ax.scatter(s.default_cv_f1_phishing, y, s=64, color=MUTED, label="Default",
               zorder=2, edgecolor=SURFACE, linewidth=2)
    ax.scatter(s.best_cv_f1_phishing, y, s=64, color=SERIES_1, label="Tuned",
               zorder=3, edgecolor=SURFACE, linewidth=2)
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.annotate(f"{r.best_cv_f1_phishing:.4f}", (r.best_cv_f1_phishing, yi),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=9)
    ax.set_yticks(y, [LABELS[k] for k in s.model], color=INK_2)
    ax.set_xlabel("Mean 5-fold CV F1, phishing class (url_only, training partition)")
    ax.grid(axis="y", visible=False)
    lo = s[["default_cv_f1_phishing", "best_cv_f1_phishing"]].min().min()
    ax.set_xlim(lo - 0.0005, s.best_cv_f1_phishing.max() + 0.0012)
    ax.legend(loc="lower left", ncols=2)
    ax.set_title("Grid search: cross-validated F1 before and after tuning")
    fig.tight_layout()
    fig.savefig(F / "tuning_cv_f1.png", dpi=200)
    plt.close(fig)

    # 2. Confusion matrix of the final model on the test partition. Log color
    # scale so the error cells are not washed out by the correct-class cells.
    matrix = np.array([[m["phishing_caught"], m["phishing_missed"]],
                       [m["legit_flagged"], m["legit_passed"]]])
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(np.log10(matrix + 1), cmap=cmap)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if np.log10(v + 1) > 2.5 else INK)
    ax.set_xticks([0, 1], ["Phishing", "Legitimate"])
    ax.set_yticks([0, 1], ["Phishing", "Legitimate"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(f"Tuned {final_label.lower()}, test partition")
    fig.tight_layout()
    fig.savefig(F / "final_confusion_matrix.png", dpi=200)
    plt.close(fig)

    # 3. Precision-recall curve with the three operating points.
    pred = pd.read_csv(R / "final_test_predictions.csv")
    is_phish = pred.label == C.POS_LABEL
    precision, recall, _ = precision_recall_curve(is_phish, pred.p_phishing)
    thr = pd.read_csv(R / "final_thresholds.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(recall, precision, color=SERIES_1, lw=2)
    names = {"default": "Default", "f1_optimal": "F1-optimal",
             "fpr_budget": f"{FPR_BUDGET:.1%} false-alarm budget"}
    # Operating points that land on the same threshold share one marker and label.
    thr["key"] = thr.threshold.round(3)
    for i, (_, g) in enumerate(thr.groupby("key", sort=False)):
        r = g.iloc[0]
        label = " = ".join(names[k] for k in g.operating_point)
        ax.scatter(r.recall, r.precision, s=64, color=INK, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{label} (t={r.threshold:.3f})\nprecision {r.precision:.4f}, "
                    f"recall {r.recall:.4f}", (r.recall, r.precision),
                    xytext=(-12, -30 - 30 * i), textcoords="offset points", ha="right",
                    fontsize=8.5, color=INK_2,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(0.985, 1.0005)
    ax.set_ylim(0.996, 1.0003)
    ax.set_xlabel("Recall (phishing)")
    ax.set_ylabel("Precision (phishing)")
    ax.set_title(f"Precision-recall, test partition (PR AUC {m['pr_auc']:.4f})")
    fig.tight_layout()
    fig.savefig(F / "final_pr_curve.png", dpi=200)
    plt.close(fig)

    # 4. How confident are the errors? Near-zero scores on missed phishing
    # mean a threshold change cannot recover them.
    missed = pred.loc[is_phish & (pred.pred != C.POS_LABEL), "p_phishing"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(0, 0.5, 26)
    ax.hist(missed, bins=bins, color=SERIES_1, edgecolor=SURFACE, linewidth=0.6)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_yticks([0, 1, 2, 5, 10, 50, 100], ["0", "1", "2", "5", "10", "50", "100"])
    ax.axvline(0.1, color=INK, lw=1, ls=":")
    ax.annotate(f"{(missed < 0.1).sum()} of {len(missed)} misses score below 0.1;\n"
                f"{(missed >= 0.1).sum()} fall between 0.1 and 0.5",
                (0.1, 40), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Model's P(phishing) for phishing URLs it missed (decision threshold 0.5)")
    ax.set_ylabel("Missed URLs")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Missed phishing URLs (n={len(missed)}) scored as confidently legitimate")
    fig.tight_layout()
    fig.savefig(F / "final_error_confidence.png", dpi=200)
    plt.close(fig)

    # 5. Legitimate false-alarm rate by TLD: the fairness view.
    tld = pd.read_csv(R / "final_group_error_rates_tld.csv").dropna(
        subset=["legit_false_alarm_rate"]).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh([f".{t}" for t in tld.TLD], tld.legit_false_alarm_rate * 100,
            color=SERIES_1, height=0.62)
    for i, (_, r) in enumerate(tld.iterrows()):
        ax.annotate(f"{r.legit_false_alarm_rate:.2%}  (n={r.n_legit:,}, "
                    f"{r.share_phishing:.0%} of TLD is phishing)",
                    (r.legit_false_alarm_rate * 100, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    overall = m["legit_false_alarm_rate"] * 100
    ax.axvline(overall, color=INK, lw=1, ls=":")
    ax.annotate(f"all TLDs {overall:.2f}%", (overall, len(tld) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, max(tld.legit_false_alarm_rate.max() * 100 * 2.1, overall * 2))
    ax.set_xlabel("Legitimate URLs wrongly flagged as phishing (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"False alarms on legitimate sites by TLD (test, n_legit >= {MIN_GROUP_N})")
    fig.tight_layout()
    fig.savefig(F / "final_false_alarms_by_tld.png", dpi=200)
    plt.close(fig)

    # 6. Permutation importance.
    imp = pd.read_csv(R / "final_permutation_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.barh(imp.feature, imp.f1_drop_mean, xerr=imp.f1_drop_sd, color=SERIES_1,
            height=0.62, error_kw={"ecolor": INK_2, "lw": 1})
    ax.set_xlabel("Drop in phishing F1 when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("Permutation importance (top 10 features)")
    fig.tight_layout()
    fig.savefig(F / "final_permutation_importance.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_final_evaluation()
    make_final_figures()
