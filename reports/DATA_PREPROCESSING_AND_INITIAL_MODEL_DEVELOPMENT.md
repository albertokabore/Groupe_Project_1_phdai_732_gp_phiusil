# Data Preprocessing and Initial Model Development

**PhDAI 732 | Group 5 | PhiUSIIL Phishing URL Dataset**

*Draft aligned to the requested 0.5 / 1.5 / 1 / 0.5-page section allocation. Final pagination depends on font, spacing, and table placement. Complete the team contribution placeholders before submission.*

## 1. Introduction & Problem Framing

Phishing detection is a binary classification problem: given a URL and its available characteristics, predict whether it is phishing or legitimate. Missing a phishing URL can expose users to credential theft, while incorrectly blocking a legitimate URL interrupts normal browsing. The objective of this project is to explore and clean a labeled dataset, prepare reproducible model inputs, and evaluate initial classifiers using Python, pandas, scikit-learn, and Matplotlib.

The project uses the PhiUSIIL dataset. Its target encodes legitimate URLs as 1 and phishing URLs as 0, as confirmed by the [UCI dataset documentation](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). We compare models using all configured numeric predictors, a reduced set excluding suspected construction-derived predictors, and URL-only features available without retrieving webpage content. Success requires improvement over a majority-class baseline and careful interpretation of errors. Exceptionally high benchmark scores motivate investigation of possible leakage and collection bias; they do not establish that the classifier will generalize to new attacks.

## 2. Dataset Exploration & Preprocessing

### Data quality and class distribution

The shared loader, `src/data.py`, reads the CSV with UTF-8 BOM handling and validates configured column names. The saved cleaning log records 235,795 original rows and 56 columns. There were no exact duplicate rows, but 425 repeated URLs were removed, retaining the first occurrence. No duplicated URL had conflicting labels. Cleaning left 235,370 observations across 220,086 distinct recorded domains, with no missing values detected. Consequently, missing-value imputation was unnecessary for this dataset.

The cleaned sample contains 134,850 legitimate URLs (57.3%) and 100,520 phishing URLs (42.7%). This moderate imbalance makes a majority-class baseline useful: predicting every URL as legitimate already achieves approximately 57.3% accuracy. The existing class-balance figure visualizes this distribution. Other saved Matplotlib figures compare URL length and HTTPS usage by class and summarize the single-feature screen. These plots support inspection of class differences alongside aggregate metrics.

### Feature preparation

The target is separated from the predictors. `FILENAME`, `URL`, `Domain`, and `Title` are excluded from model inputs. The categorical `TLD` column is also outside the configured feature lists and is recorded as an unaccounted column in the cleaning log. Thus, the current models use selected numeric features without one-hot encoding text or categorical fields. Original feature spellings are retained so configuration and input columns remain consistent.

Three configurations provide a controlled feature comparison: `full` contains 50 predictors; `no_derived` contains 44, excluding six suspected construction-derived predictors; and `url_only` contains 18 lexical or structural URL predictors. The latter includes URL length, domain length, character counts, obfuscation indicators, and HTTPS status. The 26 webpage-content predictors require page retrieval and therefore represent a different information setting from URL-only classification.

Logistic regression uses a scikit-learn pipeline containing `StandardScaler` and the classifier. Scaling is learned from training data and is refitted within each cross-validation fold. Decision trees and random forests use the numeric inputs without scaling. The current workflow does not apply resampling, clipping, or dimensionality reduction; these operations should not be described as completed preprocessing.

### Data diagnostics and reproducibility

A depth-one decision tree fitted separately to each feature produced accuracies of 99.67% for `URLSimilarityIndex`, 96.13% for `NoOfExternalRef`, and 95.54% for `LineOfCode`. These are **in-sample exploratory scores**: `src/leakage.py` fits and scores each tree on the same data. They flag unusually strong associations but are neither held-out performance estimates nor proof of target leakage.

Class-constancy results show that every legitimate observation has `URLSimilarityIndex = 100` and `IsHTTPS = 1`. Such patterns suggest possible collection artifacts and justify feature ablation. They do not establish the collection mechanism or imply that HTTPS guarantees safety. Because this exploration examined the full dataset, subsequent test results should be treated as exploratory; future feature selection should use training data alone.

The cached stratified split allocates 176,527 observations to training and 58,843 to testing, using seed 42. The cleaning log's URL fingerprint matches the configured reference, providing a check on the cleaned URL population, although it does not verify every feature value. A domain-grouped split is also available for later evaluation; the reported results below are from the conventional row-level experiment. Grouped evaluation is needed to assess sensitivity to domains appearing across partitions.

## 3. Initial Model Development

The initial roster comprises a majority-class dummy classifier, logistic regression with a 2,000-iteration limit, a decision tree, and a random forest with 300 trees. The dummy classifier establishes a minimum comparison; logistic regression supplies a linear baseline; and the tree models capture nonlinear relationships. The project fits each model across the three feature configurations. Five-fold cross-validation on training data provides an additional F1 estimate for the learned models. No hyperparameter search is documented in this initial stage.

Selected saved test results are summarized below. F1 uses the project's positive class, **legitimate (1)**; it is not phishing-class F1.

| Features | Model | Accuracy | F1 (legitimate) | ROC AUC |
|---|---|---:|---:|---:|
| URL-only | Majority baseline | 0.5729 | 0.7285 | 0.5000 |
| URL-only | Logistic regression | 0.9959 | 0.9964 | 0.9981 |
| URL-only | Decision tree | 0.9972 | 0.9976 | 0.9967 |
| URL-only | Random forest | 0.9973 | 0.9976 | 0.9981 |
| No derived | Random forest | 0.9999 | 0.9999 | 1.0000 |
| Full | Random forest | 1.0000 | 1.0000 | 1.0000 |

The URL-only random forest is a reasonable provisional model for further investigation because it uses features available before page retrieval. Its training cross-validation F1 is 0.997606, with a standard deviation of 0.000136. Its saved test confusion matrix shows 24,990 phishing URLs correctly detected, 140 phishing URLs classified as legitimate, 20 legitimate URLs classified as phishing, and 33,693 legitimate URLs correctly accepted. Re-expressing these counts with phishing as the positive class gives 99.92% precision and 99.44% recall.

Removing derived and page-content features reduces random-forest accuracy by only about 0.27 percentage points. This small decrease indicates that strong class separation persists in the remaining predictors; it does not resolve concerns about collection bias. Next steps are domain-grouped evaluation, training-only feature screening, and testing on independently collected or later data. These initial results support continued development, not a deployment-readiness claim.

## 4. Team Collaboration Process

The repository documents a shared workflow built around centralized configuration, one cleaning function, cached split assignments, and persisted JSON/CSV results. Model logic resides in `src/`, while notebooks serve as drivers. This structure allows contributors to use consistent inputs and lets report writers trace numerical claims to saved artifacts. The README specifies role-based branches and pull-request review, and assigns Eric responsibility for the scaffold and final assembly. These are documented responsibilities; actual contributions require team confirmation.

Before submission, complete this contribution record with verifiable work:

- **Data preprocessing — [name]:** cleaning, schema checks, and dataset diagnostics; [commit or PR].
- **Exploration — [name]:** class distributions and visualizations; [commit or PR].
- **Modeling — [name]:** baseline training and metric review; [commit or PR].
- **Evaluation/reporting — [name]:** error interpretation, limitations, and report editing; [commit or PR].
- **Coordination — [name]:** integration and final review; [meeting record or PR].

Add a brief account of the team's actual communication method, one challenge encountered, and how it was resolved. Repository conventions alone cannot establish that meetings, reviews, or individual contributions occurred.

## References and supporting artifacts

Prasad, A., & Chandra, S. (2024). *PhiUSIIL Phishing URL (Website)* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Project evidence: `results/cleaning_log.json`, `results/leakage_screen.csv`, `results/class_constancy.csv`, `results/metrics_*.json`, and `src/{config,data,splits,models,leakage}.py`. This draft summarizes saved results; models were not retrained during report preparation.

Optional supporting figures: [class balance](../figures/class_balance.png), [URL length by class](../figures/url_length_by_class.png), [HTTPS by class](../figures/https_by_class.png), and [single-feature screen](../figures/leakage_screen_top10.png). Place figures outside the main text if needed to meet the page allocation.
