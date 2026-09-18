"""Exploratory figures for Section 2 of the report.

Reads only the cleaned frame returned by load_data(). Nobody re-derives
cleaning logic here; the numbers in these figures must match cleaning_log.json.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config as C


def create_figures(df):
    """Write class-balance, feature, and leakage-screen figures to figures/."""
    sns.set_theme(style="whitegrid")
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = C.FIGURES_DIR / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[C.LABEL].value_counts().rename({1: "legitimate", 0: "phishing"})
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False)
    ax.set(title="Class balance after cleaning", xlabel="Class", ylabel="Rows")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    save(fig, "class_balance")

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df = df[["URLLength", C.LABEL]].copy()
    plot_df["class"] = plot_df[C.LABEL].map({1: "legitimate", 0: "phishing"})
    sns.histplot(
        data=plot_df, x="URLLength", hue="class", element="step",
        stat="density", common_norm=False, ax=ax,
    )
    ax.set(title="URL length by class", xlabel="URL length (characters)")
    save(fig, "url_length_by_class")

    fig, ax = plt.subplots(figsize=(5, 4))
    share = df.groupby(C.LABEL)["IsHTTPS"].mean().rename({1: "legitimate", 0: "phishing"})
    sns.barplot(x=share.index, y=share.values, ax=ax, hue=share.index, legend=False)
    ax.set(title="Share of URLs served over HTTPS", xlabel="Class", ylabel="Share with IsHTTPS=1")
    ax.set_ylim(0, 1.05)
    save(fig, "https_by_class")

    from .leakage import single_feature_accuracy

    screen = single_feature_accuracy(df).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=screen, x="single_feature_accuracy", y="feature", ax=ax,
        hue="flagged", dodge=False,
    )
    ax.set(title="Single-feature accuracy (top 10)", xlabel="Accuracy", ylabel="")
    ax.axvline(0.95, color="black", linestyle="--", linewidth=1)
    save(fig, "leakage_screen_top10")

    return paths
