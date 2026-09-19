"""
src/eda.py

Exploratory Data Analysis and Visualization Pipeline
for the PhiUSIIL Phishing URL Dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import config as c
from src.data import load_data
from src.leakage import single_feature_accuracy


def load_constancy_data() -> pd.DataFrame:
    """Reads the verified class constancy table from results/."""
    constancy_path = c.RESULTS_DIR / "class_constancy.csv"
    if not constancy_path.exists():
        raise FileNotFoundError(f"Missing {constancy_path}. Ensure data steward pipeline has run.")
    return pd.read_csv(constancy_path)


def plot_modal_share_by_class(constancy_df: pd.DataFrame):
    """
    Plots a comparative bar chart of modal share per feature split by class.
    Visualizes collection artifacts where features exhibit near-100% constancy in legitimate sites.
    """
    print("[1/5] Generating modal share by class plot...")
    
    # Filter to top features showing high constancy in legitimate class (label=1)
    legit_high = constancy_df[constancy_df[c.LABEL] == 1].sort_values(by='modal_share', ascending=False)
    top_features = legit_high.head(10)['feature'].tolist()
    
    plot_df = constancy_df[constancy_df['feature'].isin(top_features)].copy()
    plot_df['Class'] = plot_df[c.LABEL].map({0: 'Phishing (0)', 1: 'Legitimate (1)'})
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_df,
        x='feature',
        y='modal_share',
        hue='Class',
        palette=['#d95f02', '#1f77b4']
    )
    plt.title("Modal Share by Feature Split by Class (Collection Artifact Diagnostics)", fontsize=12, fontweight='bold')
    plt.xlabel("Feature")
    plt.ylabel("Modal Share")
    plt.ylim(0, 1.15)
    plt.xticks(rotation=35, ha='right')
    plt.legend(title="Class")
    plt.tight_layout()
    
    out_path = c.FIGURES_DIR / "modal_share_by_class.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def plot_target_distribution(df: pd.DataFrame):
    """Plots and exports the binary class distribution."""
    print("[2/5] Generating class distribution plot...")
    target_counts = df[c.LABEL].value_counts()
    
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette='Blues_d')
    plt.title(f"Distribution of Target Variable ('{c.LABEL}')", fontsize=12, fontweight='bold')
    plt.xlabel(f"Class (0 = Phishing, 1 = Legitimate)")
    plt.ylabel("Record Count")
    plt.xticks([0, 1], ['Phishing (0)', 'Legitimate (1)'])
    
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, 
                    color='black', xytext=(0, 5), textcoords='offset points')
                    
    plt.ylim(0, max(target_counts.values) * 1.15)
    plt.tight_layout()
    out_path = c.FIGURES_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def plot_feature_boxplots(df: pd.DataFrame):
    """Plots comparative boxplots using c.URL_LEXICAL and c.PAGE_CONTENT."""
    print("[3/5] Generating lexical and DOM feature boxplots...")
    
    # 1. URL Lexical Features from config
    url_feats = [col for col in c.URL_LEXICAL if col in df.columns]
    if url_feats:
        display_feats = url_feats[:5]
        fig, axes = plt.subplots(1, len(display_feats), figsize=(18, 3.5))
        if len(display_feats) == 1:
            axes = [axes]
        for i, col in enumerate(display_feats):
            sns.boxplot(x=c.LABEL, y=col, data=df, ax=axes[i], palette='Set2', showfliers=False)
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_xticklabels(['Phishing (0)', 'Legitimate (1)'])
        plt.tight_layout()
        out_url = c.FIGURES_DIR / "url_features_boxplots.png"
        plt.savefig(out_url, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"       -> Saved: {out_url}")

    # 2. Page Content / DOM Features from config
    page_feats = [col for col in c.PAGE_CONTENT if col in df.columns]
    if page_feats:
        display_page = page_feats[:5]
        fig, axes = plt.subplots(1, len(display_page), figsize=(18, 3.5))
        if len(display_page) == 1:
            axes = [axes]
        for i, col in enumerate(display_page):
            sns.boxplot(x=c.LABEL, y=col, data=df, ax=axes[i], palette='Pastel1', showfliers=False)
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_xticklabels(['Phishing (0)', 'Legitimate (1)'])
        plt.tight_layout()
        out_page = c.FIGURES_DIR / "page_features_boxplots.png"
        plt.savefig(out_page, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"       -> Saved: {out_page}")


def plot_correlation_matrix(df: pd.DataFrame):
    """Calculates Pearson correlations and renders top-predictor heatmap."""
    print("[4/5] Generating correlation matrix heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    if c.LABEL not in numeric_df.columns:
        return

    corrs = numeric_df.corr()[c.LABEL].sort_values()
    top_negative = list(corrs.head(7).index)
    top_positive = list(corrs.tail(8).index)
    selected_features = list(dict.fromkeys(top_negative + top_positive))
    
    corr_matrix = df[selected_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='vlag', annot=True, fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix of Top Predictors with {c.LABEL}", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = c.FIGURES_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def plot_leakage_diagnostics(df: pd.DataFrame):
    """Plots leakage diagnostics by reading results/leakage_screen.csv."""
    print("[5/5] Generating leakage diagnostics plot...")
    screen_path = c.RESULTS_DIR / "leakage_screen.csv"
    if not screen_path.exists():
        # Fallback to function without fallback heuristics
        screen = single_feature_accuracy(df)
    else:
        screen = pd.read_csv(screen_path)

    top_screen = screen.head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_screen['feature'], top_screen['single_feature_accuracy'], color='firebrick')
    plt.axvline(0.95, color='black', linestyle='--', label='95% Leakage Flag Threshold')
    plt.title("Single-Feature Predictive Accuracy (Leakage Detection)", fontsize=12, fontweight='bold')
    plt.xlabel("Univariate Accuracy")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    out_path = c.FIGURES_DIR / "leakage_screen_barchart.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def run_eda_pipeline():
    """Main execution function adhering to project constants and shared artifacts."""
    c.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data without modifying cleaning log
    df, _ = load_data(write_log=False)
    
    # 2. Read constancy data from results/
    constancy_df = load_constancy_data()
    
    # 3. Generate figures
    plot_modal_share_by_class(constancy_df)
    plot_target_distribution(df)
    plot_feature_boxplots(df)
    plot_correlation_matrix(df)
    plot_leakage_diagnostics(df)
    
    print("\nAll EDA figures generated and saved to figures/.")


if __name__ == "__main__":
    run_eda_pipeline()
