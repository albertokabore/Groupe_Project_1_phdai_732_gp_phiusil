# Repo review: `phdai_732_gp_phiusil`

**Reviewer:** Claude (Cowork), on request from J. Eric Collier
**Date:** September 14, 2026
**Repo reviewed:** `C:\Users\NCFI\Documents\GitHub\phdai_732_gp_phiusil`

---

## Bottom line

The repo would not have run for anyone, including you. Every module opens with
`from . import config as C`, but the modules sit loose at the repo root with no
`src/` package and no `__init__.py`, so `python -m src.data` and `from src.data
import load_data` both fail before touching a line of logic. Nothing had ever
been committed; `refs/heads/` is empty and the remote at
`github.com/jecollier041/phdai_732_gp_phiusil.git` has no objects in it. Four
teammates cloning that URL on Monday would have gotten an empty directory.

All of that is fixed and written back. The pipeline now runs end to end on the
real 235,795-row CSV, and the numbers it produces confirm the leakage thesis and
complicate it in a way that matters more than any of the mechanical problems. See
the last section before you do anything else.

The Wednesday 9/16 gate is achievable. It was not achievable from the state I
found.

---

## What was broken and is now fixed

### 1. There was no package

The intended layout puts five modules under `src/`. All five were at the repo
root instead. Relative imports need a package, so the repo was inert.

Fixed by creating the `src/` directory, moving `config.py`, `data.py`,
`splits.py`, `leakage.py`, and `models.py` into it, and adding `src/__init__.py`.

This also silently fixed a second bug. `config.ROOT` is
`Path(__file__).resolve().parents[1]`, which is correct only when `config.py`
lives one directory below the root. With `config.py` at the root, `ROOT`
resolved to `C:\Users\NCFI\Documents\GitHub`, and the `mkdir` loop at import
time would have created `data/`, `figures/`, `results/`, and `reports/` as
siblings of every other repo in your GitHub folder. Those folders do not exist
there, which is independent confirmation that `config.py` had never been
successfully imported by anyone.

### 2. The notebook pointed at a repository that does not exist

```diff
-REPO = 'https://github.com/GROUP5/phishing-part1.git'  # <-- replace
+REPO = 'https://github.com/jecollier041/phdai_732_gp_phiusil.git'
+DIR  = 'phdai_732_gp_phiusil'
```

The `%cd phishing-part1` that followed would have failed too, as would the
`cd phishing-part1` in the README setup block. Both corrected. The setup cell
now ends with a sanity print of `ROOT`, `RAW_CSV`, `SEED`, and `TEST_SIZE`, so a
teammate who runs one cell knows immediately whether the repo is wired up.

### 3. `make_split` ignored its own `strategy` argument

This is the one I would have been most annoyed to find on Friday night. The
cache check ran before the strategy dispatch and keyed on a single filename, so
once anyone froze the stratified split, every later call returned that same
assignment regardless of what was asked for:

```
assignment_grouped = make_split(df, strategy='grouped')
# returned the cached stratified assignment, silently
```

I confirmed this against the real data before fixing it: `(grouped ==
stratified).all()` returned `True`. Whoever ran the domain-grouped comparison
would have reported it as a finding, and the finding would have been the
stratified split wearing a different label.

Fixed with one cache file per strategy via a new `split_path()` helper.
`split_assignment.csv` stays the stratified file so the contract language in the
README still holds; grouped writes `split_assignment_grouped.csv`.

### 4. The stale-cache guard compared the wrong thing

```diff
-        if len(cached) == len(df):
-            return cached["split"].reindex(df.index)
+        if not df.index.equals(cached.index):
+            raise ValueError(...)
+        return cached.reindex(df.index)
```

Length equality is not row identity. Two frames of the same length holding
different rows would pass the check, `reindex` would return `NaN` for every row
not in the cache, `assignment == "train"` would evaluate those to `False`, and
they would land in the test set with nobody the wiser. The new error message
names how many rows are in each direction. `get_xy` also now refuses outright if
any assignment is `NaN`, so there is no path from a mismatched cache to a fitted
model.

### 5. No `.gitignore`, and two 57 MB copies of the dataset in the working tree

Contract rule 5 existed only in prose. There was no `.gitignore` at all, and the
CSV was sitting at the repo root *and* again inside
`phiusiil+phishing+url+dataset/`. A first `git add -A` would have staged 114 MB.
GitHub rejects files over 100 MB and warns over 50 MB, so the push would have
either failed or left you with an unrecoverable 57 MB blob in history.

Added a `.gitignore` covering `data/*.csv`, the root-level CSV by name, the
extraction folder, `__pycache__/`, and `.ipynb_checkpoints/`. It carries an
explicit comment that `results/` is *not* ignored, because the frozen split and
the metrics JSON are the handoff artifact between the person who computes a
number and the person who writes it down.

### 6. `config.RAW_CSV` pointed at a file that was not there

`RAW_CSV` was hardcoded to `data/PhiUSIIL_Phishing_URL_Dataset.csv`, and `data/`
did not exist. `_find_raw_csv()` now checks `data/`, then the repo root, then
`phiusiil+phishing+url+dataset/`, and falls back to the canonical `data/` path
when nothing is on disk, which is where the notebook's `ucimlrepo` fetch writes.
A teammate who unzips the UCI download in place is no longer blocked.

I could not move or delete your two CSV copies from here. The device bridge caps
writes at 20 MB per file, and file deletion on your machine is not enabled for
this session. Do this yourself:

```
cd C:\Users\NCFI\Documents\GitHub\phdai_732_gp_phiusil
move PhiUSIIL_Phishing_URL_Dataset.csv data\
rmdir /s /q "phiusiil+phishing+url+dataset"
```

`.gitignore` covers you either way, but carrying two 57 MB copies is pointless.

### 7. `ucimlrepo` was missing from `requirements.txt`

The notebook's second cell `pip install`s it separately, which works in Colab and
fails for anyone running locally off `requirements.txt`. Added.

### 8. Smaller items

- `pd.read_csv` now passes `encoding="utf-8-sig"`. The published CSV carries a
  byte-order mark. pandas 3.0 strips it; some 2.x builds read it into the first
  column name as `\ufeffFILENAME`, which would have made `_validate_schema` raise
  a `KeyError` on `FILENAME` and sent the steward hunting through `config.py` for
  a problem that was not there.
- `.gitkeep` files in `data/`, `results/`, `figures/`, and `reports/` so the
  directories survive a clone.
- README: layout table updated, setup block corrected, per-strategy split caching
  documented, dataset location documented.
- Notebook cell 6 carries a runtime warning. Random forest took 100 s per feature
  set on a fast multicore machine; a free Colab CPU runtime is several times
  slower. Budget roughly 45 minutes for that cell, or pass `cv=False` while
  iterating.

---

## What passed

- **Schema fidelity.** All three published misspellings are reproduced verbatim
  in `config.py`: `NoOfDegitsInURL`, `DegitRatioInURL`, `SpacialCharRatioInURL`.
  Nobody silently corrected them. Verified against the real CSV header.
- **The schema validator works as designed.** It raises `KeyError` naming the
  missing columns and reporting the unaccounted-for ones. On the real file it
  reports exactly one unaccounted column, `TLD`, which is a string field not
  named in any feature group. That is correct behavior, not a defect.
- **Seed discipline.** No literal integer `random_state=` anywhere outside
  `config.py`. No `train_test_split` outside `splits.py`. No hardcoded Windows or
  `/content/drive` paths. The only `read_csv` outside `data.py` is `splits.py`
  reading its own split cache, which is not the raw file and does not violate
  rule 2.
- **Feature lists are defined once.** Only in `config.py`. No shadow lists.
- **Label orientation is derived, not assumed.** `_check_label_orientation`
  computes it from the data. On the real file it returns
  `label_value_for_legitimate = 1`, and the supporting table shows
  `URLSimilarityIndex` mean, min, and max all exactly 100.0 for label 1 across
  all 134,850 rows. Still worth the one-line confirmation against the UCI page
  before it goes in the report.
- **The grouped split is genuinely grouped.** Now that it runs, I checked it:
  zero domains appear in both train and test.
- **`src/eda.py` left unwritten**, as instructed. Role assignments untouched.
  Feature-set membership untouched. Seed and test proportion untouched.

---

## Smoke test on the real dataset

Ran the full notebook path against your local
`PhiUSIIL_Phishing_URL_Dataset.csv`. No network.

**Cleaning.** 235,795 rows in, 56 columns. Zero exact duplicate rows. 425
duplicate URLs removed, zero of them with conflicting labels. 235,370 rows out.
Zero missing values anywhere. 220,086 unique domains. Class counts after
cleaning: 134,850 legitimate (label 1) and 100,520 phishing (label 0).

**Leakage screen.** Your thesis numbers replicate exactly.

| Feature | Single-feature accuracy |
|---|---|
| URLSimilarityIndex | .9967 |
| NoOfExternalRef | .9613 |
| LineOfCode | .9554 |
| NoOfSelfRef | .9485 |
| NoOfImage | .9426 |

**Split.** Stratified: 176,527 train, 58,843 test. Grouped: 176,906 train,
58,464 test, zero domain overlap.

**Models**, stratified split, all twelve fits. These are real numbers off your
machine's data, not estimates.

| Feature set | Model | Accuracy | F1 | ROC AUC | CV F1 mean |
|---|---|---|---|---|---|
| full | majority baseline | .5729 | .7285 | .5000 | — |
| full | logistic regression | .9999 | .9999 | 1.0000 | .9999 |
| full | decision tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| full | random forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_derived | logistic regression | .9993 | .9993 | 1.0000 | .9994 |
| no_derived | decision tree | .9990 | .9991 | .9990 | .9991 |
| no_derived | random forest | .9999 | .9999 | 1.0000 | .9999 |
| url_only | logistic regression | .9959 | .9964 | .9981 | .9966 |
| url_only | decision tree | .9972 | .9976 | .9967 | .9975 |
| url_only | random forest | .9973 | .9976 | .9981 | .9976 |

---

## The thing you need to decide before Wednesday

**The ablation does not produce the gap the argument depends on.**

The handoff says: report the naive result, show the single-feature baseline that
matches it, rebuild on URL-lexical features only, and "the gap between those
numbers is the analysis." There is no gap. Accuracy goes 1.0000 → .9999 → .9973
across the three feature sets. Stripping `URLSimilarityIndex`, every derived
feature, and all twenty-six page-content features costs you about a quarter of a
percentage point. A reader will ask what was audited.

I also checked whether domain memorization was propping up `url_only`. It is
not. On the domain-grouped split with zero shared domains, `url_only` random
forest scores .9975, marginally *higher* than on the stratified split. That rules
out the obvious alternative explanation.

The actual explanation is in `class_constancy`, and it is more interesting than
the one in the handoff. Several `URL_LEXICAL` features are constant across the
entire legitimate class:

| Feature | Value for all 134,850 legitimate rows | Share of phishing rows sharing it |
|---|---|---|
| IsHTTPS | 1 | .491 |
| IsDomainIP | 0 | .994 |
| HasObfuscation | 0 | .995 |
| NoOfObfuscatedChar | 0 | .995 |
| ObfuscationRatio | 0 | .995 |
| NoOfAmpersandInURL | 0 | .991 |

Not one legitimate URL in 134,850 is served over plain HTTP. Not one contains an
ampersand, which means not one carries a multi-parameter query string. That is
not a property of legitimate web traffic. It is a property of how the legitimate
class was harvested, almost certainly an HTTPS-only crawl of bare homepages,
while the phishing class came from a feed with no such filter. No single lexical
feature exceeds .80 accuracy on its own, so nothing trips the .95 screen, but the
two classes differ systematically in shape before any single feature is examined.

That reframes the project, and I think it strengthens it. The leakage is not
localized in a feature group you can name and remove. It is in the sampling
frame. `url_only` is not "what survives deployment," it is the same construction
artifact viewed through a narrower window, which is exactly why removing forty-one
features costs you a quarter of a point. The honest claim is that this benchmark
cannot be repaired by feature ablation, and the evidence for that claim is the
flatness of the very table the original design expected to show a cliff.

I did not change the feature sets, the taxonomy, or any framing language. That
taxonomy is still the right instrument. It is the conclusion you draw from it
that has to move, and the ablation table is still what you put in Section 3,
just read the other way around.

Two consequences for the write-up:

1. Section 1 currently promises a gap. Rewrite the promise before the EDA and
   modeling owners start drafting to it, or you will get four sections arguing
   for a finding the numbers do not support.
2. The follow-up email to Dr. Paramashivaiah is now more clearly owed, and you
   have something concrete to put in it. "The ablation was flat and here is why"
   is a better message than "we are pivoting to a leakage audit."

---

## Second thing to decide: the positive class

`evaluate()` uses sklearn's defaults, which means `pos_label=1`. Label 1 is
legitimate. Every precision, recall, and F1 in the table above therefore
describes detection of *legitimate* URLs, not phishing.

You can see it in the baseline row: the majority classifier scores recall 1.000
and F1 .7285 by predicting "legitimate" for everything. If a report section says
"F1 of .9976 for phishing detection," it will be describing the wrong class.

This changes reported numbers, so I left the code alone. The minimal fix is a
`POS_LABEL = 0` constant in `config.py` threaded through the three metric calls
in `models.py`, plus a sentence in Section 2 stating the convention. Say the
word and I will make the change and rerun. Whichever way you go, the report needs
to state explicitly which class is positive, because a reader of a phishing paper
will assume phishing is.

---

## Blockers for Monday

None, once you push. Concretely:

1. Move the CSV and delete the duplicate folder, using the two commands above.
2. `git add -A`, commit, push. Verify `git ls-files` shows no CSV.
3. Have one teammate clone fresh and run the notebook's first three cells. That
   is the only real test of whether Monday works.

`results/split_assignment.csv` and `results/split_assignment_grouped.csv` are
written and ready to commit. Treat them as provisional until the data steward
signs off on the cleaning. If they change the dedup rules, `make_split` will now
raise a named error rather than drift, which is the behavior you wanted.

I deliberately did not commit `cleaning_log.json`, `leakage_screen.csv`,
`class_constancy.csv`, or any `metrics_*.json`. Those belong to the steward and
the modeling owner. Their names should be on those commits, because Section 4
asks you to describe the collaboration process and the git history is your
evidence.

## Wednesday 9/16 gate

Achievable. Cleaning runs in about four seconds. The leakage screen runs in two.
The split is already frozen. The steward's Wednesday work is now a matter of
reviewing output and committing it under their own name, not of debugging a repo
that does not import.

The schedule risk is not Wednesday. It is Section 1, which currently argues for a
result the data does not produce. That needs your decision first, because four
people are about to start writing against it.
