# PhDAI 732 Group Project Part 1 — Group 5

PhiUSIIL Phishing URL Dataset (UCI #967). Part 1 due Sunday 9/20, Part 2 due Sunday 9/27.

## The argument

A default classifier on this dataset scores near 1.0. That is not a result, it is a
warning. `URLSimilarityIndex` is fixed at a single value for the entire legitimate
class, so the feature restates the label. Several page-content features exceed .95
accuracy on their own.

So the project is not a leaderboard. It is a leakage audit. We report the naive
result, show the single-feature baseline that matches it, then rebuild on features
actually available at the moment a URL must be judged. The gap between those numbers
is what we analyze.

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
   retypes a number out of a Colab output.
5. **The dataset is not committed.** `.gitignore` blocks `data/*.csv`. The notebook
   fetches it via `ucimlrepo`.

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
notebooks/       thin Colab driver
results/         JSON and CSV outputs
figures/         PNG, referenced by filename in the report
reports/         report drafts
```

## Feature sets

Defined once in `config.FEATURE_SETS`. Every model runs against all three.

| Set | Contents | Question it answers |
|---|---|---|
| `full` | everything | what the benchmark reports |
| `no_derived` | drop construction-derived features | how much was the artifact |
| `url_only` | lexical features from the URL string | what survives deployment |

## Roles and handoffs

| Role | Owner | Produces | Consumed by |
|---|---|---|---|
| Lead / scaffold | Eric | repo, `config.py`, Section 1, Section 4, final assembly | everyone |
| Data steward | | `cleaning_log.json`, frozen split, leakage screen | EDA, modeling |
| EDA | | `figures/*.png`, Section 2 exploratory half | lead |
| Modeling | | `metrics_*.json`, Section 3 | lead, Part 2 tuning |
| Evaluation / ethics | | metric justification, background sources | lead, Part 2 ethics |

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
