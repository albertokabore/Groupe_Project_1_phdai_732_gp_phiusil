"""Execute and document preprocessing using the project's shared contracts."""
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import load_data, verify_dataset
from .splits import get_xy, make_split


def run_preprocessing():
    """Clean, validate, export, and check train-only scaling without changing splits."""
    if not C.SPLIT_FILE.exists():
        raise FileNotFoundError("The frozen split must exist before preprocessing.")
    split_hash = hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest()
    df, log = load_data(write_log=False)
    verify_dataset(df, strict=True)
    if log["urls_with_conflicting_labels"]:
        raise ValueError("Conflicting URL labels require review before proceeding.")
    if not df[C.LABEL].isin([0, 1]).all() or df["URL"].duplicated().any():
        raise ValueError("Labels or URL uniqueness failed validation.")
    features = C.FEATURE_SETS["full"]
    numeric = df[features]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise TypeError("Configured model inputs must be numeric.")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing or non-finite model inputs require an explicit cleaning policy.")
    assignment = make_split(df)
    if assignment.isna().any() or set(assignment.unique()) != {"train", "test"}:
        raise ValueError("Invalid frozen split assignments.")

    quality = pd.DataFrame({"dtype": df.dtypes.astype(str),
                            "missing_count": df.isna().sum(),
                            "unique_count": df.nunique()})
    quality.to_csv(C.RESULTS_DIR / "preprocessing_column_quality.csv", index_label="column")
    feature_rows, scaling_rows = [], []
    for name, columns in C.FEATURE_SETS.items():
        X_train, X_test, y_train, y_test = get_xy(df, assignment, name)
        scaler = StandardScaler().fit(X_train)
        scaled_train, scaled_test = scaler.transform(X_train), scaler.transform(X_test)
        if not np.isfinite(scaled_train).all() or not np.isfinite(scaled_test).all():
            raise ValueError(f"Scaling generated invalid values for {name}.")
        for i, column in enumerate(columns):
            scaling_rows.append({"feature_set": name, "feature": column,
                                 "training_mean": scaler.mean_[i],
                                 "training_scale": scaler.scale_[i]})
        feature_rows.append({"feature_set": name, "n_features": len(columns),
                             "n_train": len(y_train), "n_test": len(y_test),
                             "scaled_inputs_finite": True})
    pd.DataFrame(scaling_rows).to_csv(C.RESULTS_DIR / "preprocessing_scaling_parameters.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(C.RESULTS_DIR / "preprocessing_feature_sets.csv", index=False)
    counts = pd.crosstab(assignment, df[C.LABEL]).rename(columns={0: "phishing", 1: "legitimate"})
    counts.to_csv(C.RESULTS_DIR / "preprocessing_split_counts.csv")
    cleaned_path = C.DATA_DIR / "PhiUSIIL_cleaned.csv"
    # Retain original row identity so the export can be matched to frozen splits.
    df.to_csv(cleaned_path, index=True, index_label="source_row_index")
    (C.RESULTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    summary = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "rows_raw": log["rows_raw"], "rows_clean": len(df),
        "duplicate_urls_removed": log["duplicate_urls_removed"],
        "conflicting_url_labels": log["urls_with_conflicting_labels"],
        "missing_values": int(df.isna().sum().sum()),
        "non_finite_model_values": int((~np.isfinite(numeric.to_numpy())).sum()),
        "class_counts": log["class_counts"], "fingerprint_matches": log["fingerprint_matches"],
        "raw_sha256": hashlib.sha256(C.RAW_CSV.read_bytes()).hexdigest(),
        "split_sha256": split_hash, "cleaned_file": str(cleaned_path.relative_to(C.ROOT)),
        "excluded_model_columns": sorted(set(df.columns) - set(features) - {C.LABEL}),
        "feature_sets": feature_rows,
        "scaling": "Fitted on training rows only; model pipelines refit within each CV fold.",
        "imputation": "Not needed: no missing values detected.",
        "outliers": "Retained; no unvalidated clipping or removal.",
        "encoding": "Configured predictors are numeric; categorical TLD is excluded.",
    }
    if hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest() != split_hash:
        raise RuntimeError("Frozen split changed during preprocessing.")
    (C.RESULTS_DIR / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def show_preprocessing_results():
    from IPython.display import display
    summary = json.loads((C.RESULTS_DIR / "preprocessing_summary.json").read_text())
    display(pd.Series({key: summary[key] for key in ["rows_raw", "rows_clean",
        "duplicate_urls_removed", "missing_values", "non_finite_model_values",
        "fingerprint_matches"]}, name="Verified result").to_frame())
    display(pd.read_csv(C.RESULTS_DIR / "preprocessing_feature_sets.csv"))
    display(pd.read_csv(C.RESULTS_DIR / "preprocessing_split_counts.csv"))
    print("Cleaned dataset:", summary["cleaned_file"])
    print(summary["scaling"])
    return summary


if __name__ == "__main__":
    print(json.dumps(run_preprocessing(), indent=2))
