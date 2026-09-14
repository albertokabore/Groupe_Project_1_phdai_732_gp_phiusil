"""Shared loader. Everyone imports load_data(). Nobody calls pd.read_csv directly.

Returns the cleaned frame plus a cleaning log dict that the preprocessing
section of the report is written from.
"""

import hashlib
import json
import warnings

import pandas as pd

from . import config as C


def fingerprint(df):
    """SHA-256 of the sorted URL column. Stable across download paths."""
    return hashlib.sha256(
        "\n".join(sorted(df["URL"].astype(str))).encode()
    ).hexdigest()


def verify_dataset(df, strict=False):
    """Confirm this copy of the dataset matches everyone else's.

    Returns a dict for the cleaning log. Warns by default rather than raising,
    because a mismatch on day one is more likely to mean UCI changed something
    than that one teammate is wrong, and blocking four people on that is worse
    than telling them. Pass strict=True once the group has agreed on a
    fingerprint and wants divergence to be fatal.
    """
    got = fingerprint(df)
    ok = (
        got == C.EXPECTED_FINGERPRINT
        and len(df) == C.EXPECTED_CLEAN_ROWS
        and {str(k): int(v) for k, v in df[C.LABEL].value_counts().items()}
        == C.EXPECTED_CLASS_COUNTS
    )
    result = {
        "fingerprint": got,
        "fingerprint_expected": C.EXPECTED_FINGERPRINT,
        "fingerprint_matches": ok,
    }
    if not ok:
        message = (
            "DATASET MISMATCH. This copy does not match the one the group's "
            f"numbers were built on.\n  expected fingerprint: {C.EXPECTED_FINGERPRINT}"
            f"\n  got                 : {got}"
            f"\n  expected clean rows : {C.EXPECTED_CLEAN_ROWS}, got {len(df)}"
            "\nDo not report metrics from this copy until the group resolves it. "
            "Delete data/PhiUSIIL_Phishing_URL_Dataset.csv and refetch first."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
    return result


def _validate_schema(df):
    """Fail loudly if a configured column is absent.

    The published column names contain spelling errors, and mirrors of the
    dataset differ. Catch that here rather than three files downstream.
    """
    expected = set(
        C.DROP_ALWAYS + C.DERIVED_SUSPECT + C.PAGE_CONTENT + C.URL_LEXICAL + [C.LABEL]
    )
    missing = sorted(expected - set(df.columns))
    extra = sorted(set(df.columns) - expected)
    if missing:
        raise KeyError(
            f"Columns named in config are absent from the CSV: {missing}. "
            f"Unaccounted-for columns present: {extra}. Fix config.py, not this file."
        )
    return {"unaccounted_columns": extra}


def _check_label_orientation(df):
    """Report which label value denotes legitimate.

    Do not assume. URLSimilarityIndex is fixed at 100 for the legitimate class
    in the published data, which gives a cheap orientation check.
    """
    by_label = df.groupby(C.LABEL)["URLSimilarityIndex"].agg(["mean", "min", "max"])
    legit_value = int(by_label["min"].idxmax())
    return {
        "urlsimilarity_by_label": by_label.to_dict(orient="index"),
        "label_value_for_legitimate": legit_value,
        "note": "Confirm against the UCI documentation before citing in the report.",
    }


def load_data(path=None, write_log=True):
    """Load, clean, and document. Returns (df, log)."""
    path = path or C.RAW_CSV
    # utf-8-sig: the published CSV carries a byte-order mark, which older
    # pandas reads into the first column name as "﻿FILENAME".
    raw = pd.read_csv(path, encoding="utf-8-sig")
    log = {"path": str(path), "rows_raw": int(len(raw)), "cols_raw": int(raw.shape[1])}
    log.update(_validate_schema(raw))

    df = raw.copy()

    # Exact duplicate rows.
    dup_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Duplicate URLs with conflicting labels would poison any split.
    url_conflicts = int(
        df.groupby("URL")[C.LABEL].nunique().gt(1).sum()
    )
    dup_urls = int(df.duplicated(subset="URL").sum())
    df = df.drop_duplicates(subset="URL", keep="first")

    missing = df.isna().sum()
    log.update(
        {
            "duplicate_rows_removed": dup_rows,
            "duplicate_urls_removed": dup_urls,
            "urls_with_conflicting_labels": url_conflicts,
            "rows_clean": int(len(df)),
            "columns_with_missing": {k: int(v) for k, v in missing[missing > 0].items()},
            "class_counts": {str(k): int(v) for k, v in df[C.LABEL].value_counts().items()},
            "n_unique_domains": int(df["Domain"].nunique()),
        }
    )
    log.update(_check_label_orientation(df))
    log.update(verify_dataset(df))

    if write_log:
        (C.RESULTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))

    return df, log


if __name__ == "__main__":
    _, log = load_data()
    print(json.dumps(log, indent=2))
