# PhDAI 732 Group Project Part 1 — Group 5

PhiUSIIL Phishing URL Dataset (UCI #967). Part 1 due Sunday 9/20, Part 2 due Sunday 9/27.

## The argument

A default classifier on this dataset scores near 1.0. That is not a result, it is a
warning. `URLSimilarityIndex` is fixed at 100 for all 134,850 legitimate rows, so the
feature restates the label, and four page-content features exceed .94 accuracy on
their own.

The obvious response is to remove the leaky features and see what the models can
still do. We ran that. It does not work, and why it does not work is the finding.

| Feature set | Random forest accuracy |
|---|---|
| `full` | 1.0000 |
| `no_derived` | .9999 |
| `url_only` | .9973 |

Stripping `URLSimilarityIndex`, every derived feature, and all twenty-six
page-content features costs about a quarter of a percentage point. Domain
memorization does not explain it either. On a domain-grouped split with zero shared
domains, `url_only` scores .9975, marginally higher.

The explanation is in `results/class_constancy.csv`. Six URL-lexical features take
one value across the entire legitimate class. Not one legitimate URL in 134,850 is
served over plain HTTP. Not one contains an ampersand, which means not one carries a
multi-parameter query string. Real web traffic does not look like that. This is a
fingerprint of collection: the legitimate class reads as an HTTPS-only crawl of bare
homepages, while the phishing class came from a feed with no such filter. No single
lexical feature passes the .95 screen, so nothing trips the alarm, yet the two
classes differ in shape before any feature is examined. See
`figures/modal_share_by_class.png`.

So the leakage is not localized in a feature group that can be named and dropped. It
is in the sampling frame. That is why removing thirty-two features changes almost
nothing, and it is why `url_only` is not a deployment estimate. It is the same
construction artifact seen through a narrower window.

The project is a leakage audit, and the claim is that this benchmark cannot be
repaired by feature ablation. The evidence is the flatness of the very table that was
expected to show a cliff.

**Positive class.** Label 1 is legitimate and label 0 is phishing.
`config.POS_LABEL = 0`, and `models.evaluate()` threads it through precision, recall,
F1, and the cross-validated F1. sklearn defaults to `pos_label=1`, which would report
detection of legitimate URLs in a paper about phishing. Accuracy and ROC AUC are
unaffected.

## Repo contract

Five rules. They exist so five people produce one set of numbers.

1. **One seed, one split.** `src/config.py` holds `SEED`. `src/splits.py` writes
   `results/split_assignment.csv` once and everyone reads it. Never regenerate a split
   without telling the group. Each strategy caches to its own file
   (`split_assignment.csv` for stratified, `split_assignment_grouped.csv` for grouped),
   so asking for one does not hand you the other.
2. **Nobody calls `pd.read_csv` on the raw file.** Import `load_data()` from `src.data`.
3. **Logic goes in `src/`, not in notebook cells.** The notebook is a driver.
4. **Results go to disk as JSON or CSV.** Report writers read `results/`. Nobody
   retypes a number out of a Colab output. Metrics filenames carry the split strategy,
   so a grouped run does not overwrite the stratified one.
5. **The dataset is not committed.** `.gitignore` blocks `data/*.csv`. The notebook
   fetches it via `ucimlrepo`, so each person downloads their own copy. `load_data()`
   fingerprints that copy and warns if it does not match the group's. The fingerprint
   is the SHA-256 of the sorted URL column, not of the file, because the direct UCI
   download and the `ucimlrepo` fetch produce byte-different files from identical
   data. Check `fingerprint_matches` in `results/cleaning_log.json` before you report
   a number.

## Layout

```
.gitignore       blocks the dataset CSV from git
requirements.txt pinned floor versions, includes ucimlrepo
src/__init__.py  makes src importable; do not delete
src/config.py    paths, seed, feature groups        (lead owns)
src/data.py      load, clean, schema validation     (steward owns)
src/splits.py    frozen train/test assignment       (steward owns)
src/leakage.py   single-feature audit               (steward owns)
src/models.py    model roster, evaluate()           (modeling owns)
src/eda.py       figures                            (EDA owns)
src/tuning.py    Part 2 grid search on url_only     (modeling owns)
src/final_eval.py Part 2 test evaluation, errors, fairness, figures
tests/           metric-helper tests; python -m pytest
notebooks/       thin Colab driver; Part 1 = Sections 1-3, Part 2 = Sections 4-8
results/         JSON and CSV outputs
figures/         PNG, referenced by filename in the report
reports/         report drafts; build_deliverable2_docx.py writes the Part 2 Word report
```

## Part 2

Tuning, final evaluation, and the report run in this order. Each step reads only
what the previous one wrote to `results/`.

```bash
python -m src.tuning                        # ~40 min on 8 cores; writes results/tuning*
python -m src.final_eval                    # ~10 min; writes results/final_*, figures/final_*
python reports/build_deliverable2_docx.py   # reports/Deliverable2_Report.docx
python -m pytest                            # needs pytest
```

The final model is picked by cross-validated F1 in `tuning_summary.csv`, and
decision thresholds come from out-of-fold training predictions. The test partition
is opened once, in `final_eval.py`. Notebook Sections 4-8 read the saved results by
default; set `RUN_SEARCH` or `RUN_FINAL` to recompute.

## Feature sets

Defined once in `config.FEATURE_SETS`. Every model runs against all three.

| Set | Contents | Question it answers |
|---|---|---|
| `full` | everything | what the benchmark reports |
| `no_derived` | drop construction-derived features | how much was the artifact |
| `url_only` | lexical features from the URL string | whether the artifact is localized |

## Roles and handoffs

| Role | Owner | Produces | Consumed by |
|---|---|---|---|
| Lead / scaffold | Eric | repo, `config.py`, Section 1, Section 4, final assembly | everyone |
| Data steward | Div | `cleaning_log.json`, frozen split, leakage screen | EDA, modeling |
| EDA | Zubair | `figures/*.png`, Section 2 exploratory half | lead |
| Modeling | Albert | `metrics_*.json`, Section 3 | lead, Part 2 tuning |
| Evaluation / ethics | Bharadwaj | metric justification, background sources | lead, Part 2 ethics |

## Schedule

| Date | Gate |
|---|---|
| Mon 9/14 | repo live, roles confirmed, everyone can run the notebook |
| Wed 9/16 | **hard gate**: cleaning done, leakage screen posted, split frozen |
| Fri 9/18 | all metrics JSON written, all figures rendered |
| Sat 9/19 | drafts merged, lead edits to one voice |
| Sun 9/20 | submit |

Wednesday is the gate that matters. Modeling cannot start before it and must not
wait after it.

## Git workflow

Branch per role: `steward/cleaning`, `eda/figures`, `model/baselines`. Pull request
into `main`, lead merges. Nobody commits directly to `main`. Keep commit messages
specific, because the Team Collaboration section of the report is easier to write
honestly when the history shows what actually happened.

## Setup

```bash
git clone https://github.com/jecollier041/phdai_732_gp_phiusil.git
cd phdai_732_gp_phiusil
pip install -r requirements.txt
python -m src.data     # prints the cleaning log
```

`python -m src.data` needs the dataset. Either run the notebook's fetch cell, or put
`PhiUSIIL_Phishing_URL_Dataset.csv` in `data/`. `config.py` will also find a copy left
at the repo root or in an unzipped `phiusiil+phishing+url+dataset/` folder. The CSV is
57 MB and is never committed.
