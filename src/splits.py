"""The split is frozen once and written to disk.

Every model in the report is fit on the same partition. If someone regenerates
a split with a different seed, the numbers in section 3 stop being comparable
to the numbers in section 2, and nobody notices until the night before.

Two strategies:
  stratified  - the conventional row-level split, used for headline results
  grouped     - no domain appears in both train and test

Grouped splitting matters here. Many URLs share a domain, so a row-level split
lets the model memorize domains rather than learn URL structure.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config as C


def split_path(strategy="stratified"):
    """One cache file per strategy.

    A single cache file meant that asking for the grouped split after someone
    had already frozen the stratified one silently returned the stratified
    assignment. The strategy argument was ignored and nothing said so.
    """
    if strategy == "stratified":
        return C.SPLIT_FILE
    return C.SPLIT_FILE.with_name(f"split_assignment_{strategy}.csv")


def make_split(df, strategy="stratified", overwrite=False):
    """Assign each row to train or test. Cached per strategy under results/."""
    cache = split_path(strategy)
    if cache.exists() and not overwrite:
        cached = pd.read_csv(cache, index_col=0)["split"]
        # Compare row identity, not row count. Two frames of equal length can
        # hold different rows, and reindex would then hand back NaN, which
        # evaluates as "not train" and quietly moves rows into the test set.
        if not df.index.equals(cached.index):
            raise ValueError(
                f"Cached split at {cache.name} does not cover the current frame. "
                f"cached n={len(cached)}, current n={len(df)}, "
                f"rows only in cache={len(cached.index.difference(df.index))}, "
                f"rows only in frame={len(df.index.difference(cached.index))}. "
                "Either the cleaning changed or you are on a stale CSV. Resolve "
                "before rerunning; do not pass overwrite=True to make this go away."
            )
        return cached.reindex(df.index)

    if strategy == "stratified":
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=C.TEST_SIZE,
            random_state=C.SEED,
            stratify=df[C.LABEL],
        )
    elif strategy == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.SEED)
        pos_tr, pos_te = next(gss.split(df, df[C.LABEL], groups=df["Domain"]))
        train_idx, test_idx = df.index[pos_tr], df.index[pos_te]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    assignment = pd.Series("train", index=df.index, name="split")
    assignment.loc[test_idx] = "test"
    assignment.to_frame().to_csv(cache)
    return assignment


def get_xy(df, assignment, feature_set="full"):
    """Return X_train, X_test, y_train, y_test for a named feature set."""
    features = C.FEATURE_SETS[feature_set]
    if assignment.isna().any():
        raise ValueError(
            f"{int(assignment.isna().sum())} rows have no split assignment. "
            "That means the cached split and the current frame disagree. Fix "
            "that before fitting anything."
        )
    tr = assignment == "train"
    return (
        df.loc[tr, features],
        df.loc[~tr, features],
        df.loc[tr, C.LABEL],
        df.loc[~tr, C.LABEL],
    )
