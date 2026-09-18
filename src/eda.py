"""
src/eda.py

Exploratory Data Analysis, Leakage Auditing, and Figure Generation
for the PhiUSIIL Phishing URL Dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_class_constancy_table(df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """
    Computes class-constancy metrics (modal value and modal share) per feature.
    Explains collection artifacts where features exhibit 100% constancy in legitimate sites.
    """
    print("[1/5] Generating class-constancy table...")
    records = []
    
    for label in [0, 1]:
        subset = df[df['label'] == label]
        n_sub = len(subset)
        for col in df.columns:
            if col in ['label', 'URL', 'Domain', 'TLD', 'Title']:
                continue
            
            val_counts = subset[col].value_counts(dropna=False)
            if not val_counts.empty:
                modal_val = val_counts.index[0]
                modal_freq = val_counts.iloc[0]
                modal_share = modal_freq / n_sub
                
                records.append({
                    'feature': col,
                    'label': label,
                    'modal_value': modal_val,
                    'modal_share': round(modal_share, 6)
                })

    constancy_df = pd.DataFrame(records)
    out_csv = results_dir / "class_constancy.csv"
    constancy_df.to_csv(out_csv, index=False)
    print(f"       -> Saved: {out_csv}")
    return constancy_df


def plot_target_distribution(df: pd.DataFrame, fig_dir: Path):
    """Plots and exports the binary class distribution."""
    print("[2/5] Generating class distribution plot...")
    target_counts = df['label'].value_counts()
    
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette='Blues_d')
    plt.title("Distribution of Target Variable ('label')", fontsize=12, fontweight='bold')
    plt.xlabel("Class (0 = Phishing, 1 = Legitimate)")
    plt.ylabel("Record Count")
    plt.xticks([0, 1], ['Phishing (0)', 'Legitimate (1)'])
    
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, 
                    color='black', xytext=(0, 5), textcoords='offset points')
                    
    plt.ylim(0, max(target_counts.values) * 1.15)
    plt.tight_layout()
    out_path = fig_dir / "class_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def plot_feature_boxplots(df: pd.DataFrame, fig_dir: Path):
    """Plots comparative boxplots for Lexical and DOM features."""
    print("[3/5] Generating lexical and DOM feature boxplots...")
    
    # 1. URL Lexical Features
    url_feats = ['URLLength', 'DomainLength', 'NoOfSubDomain', 'NoOfLettersInURL', 'NoOfDegitsInURL']
    url_feats_exist = [f for f in url_feats if f in df.columns]
    if url_feats_exist:
        fig, axes = plt.subplots(1, len(url_feats_exist), figsize=(18, 3.5))
        for i, col in enumerate(url_feats_exist):
            sns.boxplot(x='label', y=col, data=df, ax=axes[i], palette='Set2', showfliers=False)
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_xticklabels(['Phishing (0)', 'Legitimate (1)'])
        plt.tight_layout()
        out_url = fig_dir / "url_features_boxplots.png"
        plt.savefig(out_url, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"       -> Saved: {out_url}")

    # 2. Page Content / DOM Features
    page_feats = ['LineOfCode', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfExternalRef']
    page_feats_exist = [f for f in page_feats if f in df.columns]
    if page_feats_exist:
        fig, axes = plt.subplots(1, len(page_feats_exist), figsize=(18, 3.5))
        for i, col in enumerate(page_feats_exist):
            sns.boxplot(x='label', y=col, data=df, ax=axes[i], palette='Pastel1', showfliers=False)
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_xticklabels(['Phishing (0)', 'Legitimate (1)'])
        plt.tight_layout()
        out_page = fig_dir / "page_features_boxplots.png"
        plt.savefig(out_page, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"       -> Saved: {out_page}")


def plot_correlation_matrix(df: pd.DataFrame, fig_dir: Path):
    """Calculates Pearson correlation and renders top-predictor heatmap."""
    print("[4/5] Generating correlation matrix heatmap...")
    numeric_df = df.select_dtypes(include=[np.number])
    if 'label' not in numeric_df.columns:
        return

    corrs = numeric_df.corr()['label'].sort_values()
    top_negative = list(corrs.head(7).index)
    top_positive = list(corrs.tail(8).index)
    selected_features = list(dict.fromkeys(top_negative + top_positive))
    
    corr_matrix = df[selected_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='vlag', annot=True, fmt='.2f', 
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Top Predictors with Phishing/Legitimate Label", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_path = fig_dir / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def plot_leakage_diagnostics(df: pd.DataFrame, fig_dir: Path, results_dir: Path):
    """Generates univariate predictive accuracy bar chart to isolate data leakage."""
    print("[5/5] Generating leakage diagnostics...")
    try:
        from src.leakage import single_feature_accuracy
        screen = single_feature_accuracy(df)
    except ImportError:
        records = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == 'label':
                continue
            acc = (df[col] == df['label']).mean()
            acc = max(acc, 1 - acc)
            records.append({'feature': col, 'single_feature_accuracy': acc, 'flagged': acc >= 0.95})
        screen = pd.DataFrame(records).sort_values(by='single_feature_accuracy', ascending=False)

    screen.to_csv(results_dir / "leakage_screen.csv", index=False)
    
    top_screen = screen.head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_screen['feature'], top_screen['single_feature_accuracy'], color='firebrick')
    plt.axvline(0.95, color='black', linestyle='--', label='95% Leakage Flag Threshold')
    plt.title("Single-Feature Predictive Accuracy (Leakage Detection)", fontsize=12, fontweight='bold')
    plt.xlabel("Univariate Accuracy")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    out_path = fig_dir / "leakage_screen_barchart.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"       -> Saved: {out_path}")


def run_eda_pipeline():
    """Main execution function."""
    root_dir = Path(__file__).resolve().parent.parent
    fig_dir = root_dir / "figures"
    results_dir = root_dir / "results"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from src.data import load_data
    df, _ = load_data()
    
    generate_class_constancy_table(df, results_dir)
    plot_target_distribution(df, fig_dir)
    plot_feature_boxplots(df, fig_dir)
    plot_correlation_matrix(df, fig_dir)
    plot_leakage_diagnostics(df, fig_dir, results_dir)
    
    print("\nAll EDA figures and tables saved successfully.")


if __name__ == "__main__":
    run_eda_pipeline()
