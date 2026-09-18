# PhDAI 732 Group Project Part 1 — Deliverable 1

**Group 5** | PhiUSIIL Phishing URL Dataset (UCI #967) | Due Sunday 9/20

Format the final submission in APA style: 12-point Times New Roman,
double-spaced, one-inch margins. Section owners should replace the bracketed
name placeholders in Section 4 before submission.

## 1. Introduction and Problem Framing

Phishing URLs mimic legitimate sites to steal credentials and financial
information, and automated URL classifiers are a first line of defense in
browsers and email gateways (Sahingoz et al., 2019). This project builds and
evaluates a binary classifier on the PhiUSIIL Phishing URL Dataset, which
pairs 134,850 legitimate URLs with 100,520 phishing URLs and 54 candidate
features spanning URL lexical structure, domain properties, and rendered-page
content (Prasad & Chandra, 2024).

A default classifier on this dataset scores near-perfect accuracy, which is a
warning rather than a result: several features are constructed in ways that
correlate almost perfectly with the label. The project is therefore framed as
a leakage audit, not a leaderboard. We report the naive result, identify which
features are label-adjacent and why, and test whether the model still
performs once those features are removed. The gap — or, as this deliverable
shows, the near-absence of a gap — is the analysis.

## 2. Dataset Exploration and Preprocessing

The dataset was loaded with `src.data.load_data()`, which enforces a single
cleaning path so every teammate works from the same numbers
(`results/cleaning_log.json`). The raw file contains 235,795 rows and 56
columns. Cleaning removed 425 duplicate URLs (keeping the first occurrence;
zero of the duplicates carried conflicting labels) and found zero missing
values in any column, leaving 235,370 rows across 220,086 unique domains.
The published column names contain three verified spelling errors
(`NoOfDegitsInURL`, `DegitRatioInURL`, `SpacialCharRatioInURL`), which the
project reproduces verbatim rather than silently correcting, since renaming
them would break joins against the published documentation.

Class balance is moderate: 134,850 rows (57.3%) are legitimate (label 1) and
100,520 (42.7%) are phishing (label 0) (Figure: `figures/class_balance.png`).
Label orientation was confirmed from the data itself rather than assumed:
`URLSimilarityIndex` is exactly 100.0 for every legitimate row and averages
49.7 (range 0.16–100.0) for phishing rows, which only makes sense if 1 denotes
legitimate.

A single-feature screen (`src/leakage.py`, `results/leakage_screen.csv`) fits
a depth-1 decision tree on each feature alone. Three features exceed 0.95
accuracy by themselves: `URLSimilarityIndex` (0.997), `NoOfExternalRef`
(0.961), and `LineOfCode` (0.955) (Figure:
`figures/leakage_screen_top10.png`). A feature that alone reaches
near-perfect accuracy is not a strong predictor; it is a restatement of the
label produced during dataset construction (Kaufman et al., 2012).

A finer-grained check, class constancy (`results/class_constancy.csv`),
shows the leakage is broader than the three flagged features. Several
URL-lexical fields are *fixed* across the entire legitimate class but vary
in the phishing class:

| Feature | Value for all 134,850 legitimate rows | Share of phishing rows matching it |
|---|---|---|
| IsHTTPS | 1 | 49.1% |
| IsDomainIP | 0 | 99.4% |
| HasObfuscation | 0 | 99.5% |
| NoOfObfuscatedChar | 0 | 99.5% |
| ObfuscationRatio | 0.0 | 99.5% |
| NoOfAmpersandInURL | 0 | 99.1% |

Not one of the 134,850 legitimate URLs is served over plain HTTP or contains
a multi-parameter query string (Figure: `figures/https_by_class.png`; URL
length by class in `figures/url_length_by_class.png`). No single one of these
lexical features clears the 0.95 screening threshold on its own, so nothing
here is individually "leaky" by that test — but the two classes differ
systematically in shape before any single feature is examined, most plausibly
because the legitimate class was harvested from an HTTPS-only crawl of bare
homepages while the phishing class was not (a property of how the data was
collected, not of phishing itself).

## 3. Initial Model Development

Following the pattern in the notebook's Step 4, four models — a majority-class
baseline, logistic regression (standardized features), a decision tree, and a
random forest (300 trees) — were fit on a stratified 75/25 train/test split
(`results/split_assignment.csv`, seed 42; `n_train` = 176,527, `n_test` =
58,843) and evaluated against three feature sets defined once in
`config.FEATURE_SETS`: `full` (all 50 modeling features), `no_derived`
(drops the six construction-derived features including
`URLSimilarityIndex`), and `url_only` (the 18 lexical features readable from
the URL string alone, before any page is fetched). Five-fold cross-validated
F1 on the training set is reported alongside the held-out test metrics.

| Feature set | Model | Accuracy | F1 | ROC AUC | CV F1 (train) |
|---|---|---|---|---|---|
| full | majority baseline | 0.573 | 0.728 | 0.500 | — |
| full | logistic regression | 0.9999 | 0.9999 | 1.0000 | 0.9999 |
| full | decision tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| full | random forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_derived | logistic regression | 0.9993 | 0.9993 | 1.0000 | 0.9994 |
| no_derived | decision tree | 0.9990 | 0.9991 | 0.9990 | 0.9991 |
| no_derived | random forest | 0.9999 | 0.9999 | 1.0000 | 0.9999 |
| url_only | logistic regression | 0.9959 | 0.9964 | 0.9981 | 0.9966 |
| url_only | decision tree | 0.9972 | 0.9976 | 0.9967 | 0.9975 |
| url_only | random forest | 0.9973 | 0.9976 | 0.9981 | 0.9976 |

Full metrics, per-model confusion matrices, and timestamps are in
`results/metrics_<feature_set>_<model>.json`.

The result is the flat ablation the leakage screen predicted: accuracy moves
from 1.0000 to 0.9999 to 0.9973 across `full` → `no_derived` → `url_only`.
Removing all six construction-derived features and all twenty-six
page-content features — features unavailable at the moment a URL must
actually be judged, before any page is fetched — costs about a quarter of a
percentage point, not the large gap the single-feature screen alone would
suggest. This benchmark cannot be repaired by dropping a named feature group;
the classes differ systematically in shape (Section 2) rather than in one
removable column. `precision` and `recall` above use scikit-learn's default
positive class (label 1, legitimate); a report that states "recall for
phishing detection" from these numbers would be describing the wrong class,
and the team should agree on and state the intended positive class before
Part 2.

## 4. Team Collaboration Process

The repository enforces five shared-numbers rules so five people converge on
one report rather than five slightly different ones (see README, "Repo
contract"): one frozen seed and split per strategy
(`results/split_assignment.csv`, `results/split_assignment_grouped.csv`),
no direct `pd.read_csv` on the raw file outside `src/data.py`, modeling logic
kept in `src/` rather than notebook cells, results written to disk as JSON/CSV
so nobody hand-transcribes a number, and the 57 MB dataset itself never
committed — instead fingerprinted (`results/cleaning_log.json`,
`fingerprint_matches`) so five independent downloads stay comparable.

Work is organized by role (data steward, EDA, modeling, evaluation/ethics,
with [lead: Eric] owning the scaffold and final assembly) on branches per
role (e.g., `steward/colab-original-columns`, merged via pull request #1 on
2026-09-14), reviewed and merged into `main` by the lead rather than committed
directly. An independent repo review conducted 2026-09-14
(`reports/REPO_REVIEW_2026-09-14.md`) caught and fixed several
reproducibility bugs before modeling began — a split-caching bug that
silently returned the stratified split when a grouped split was requested,
and a stale-cache guard that compared split length instead of row identity —
which is why `make_split` now raises a named error on any split/frame
mismatch instead of drifting silently.

*[Data steward name]*: dataset cleaning, schema validation, leakage screen.
*[EDA owner name]*: exploratory figures (`src/eda.py`,
`figures/*.png`). *[Modeling owner name]*: model roster and evaluation
(`src/models.py`, `results/metrics_*.json`). *[Lead/Eric]*: repo scaffold,
Section 1, Section 4, final assembly and APA formatting. Because every number
above is a named git commit against a named branch, this section can be
finished by checking `git log` rather than by memory before submission.

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15, 1–21. https://doi.org/10.1145/2382577.2382579

Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security, 136*, Article 103545. https://doi.org/10.1016/j.cose.2023.103545

Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. *Expert Systems with Applications, 117*, 345–357. https://doi.org/10.1016/j.eswa.2018.09.029

Sheng, S., Wardman, B., Warner, G., Cranor, L. F., Hong, J., & Zhang, C. (2009). An empirical analysis of phishing blacklists. In *Proceedings of the Sixth Conference on Email and Anti-Spam (CEAS 2009)*.
