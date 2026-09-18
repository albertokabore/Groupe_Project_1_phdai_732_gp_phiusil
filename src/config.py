"""Single source of truth for paths, the random seed, and feature groupings.

Nothing else in the project hardcodes a seed, a path, or a column list.
If you need to change one of these, change it here and tell the group.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# ROOT resolves to the repo root whether you run from src/, notebooks/, or Colab.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


def _find_raw_csv():
    """Canonical location is data/. Also accept a copy left where the UCI zip
    extracts it, so a teammate who unzipped in place is not blocked. The
    canonical path is returned when nothing is on disk yet, which is what the
    notebook's ucimlrepo fetch writes to.
    """
    candidates = [
        DATA_DIR / _CSV_NAME,
        ROOT / _CSV_NAME,
        ROOT / "phiusiil+phishing+url+dataset" / _CSV_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    return DATA_DIR / _CSV_NAME


RAW_CSV = _find_raw_csv()
SPLIT_FILE = RESULTS_DIR / "split_assignment.csv"

# --- Dataset fingerprint -----------------------------------------------------
# Five people download this dataset independently. Nothing else proves they got
# the same bytes, and a silent difference makes their metrics incomparable.
#
# The fingerprint is the SHA-256 of the sorted, newline-joined URL column after
# cleaning. It is deliberately not a hash of the CSV file: the UCI direct
# download and the ucimlrepo fetch produce byte-different files (column order,
# byte-order mark, float formatting) from identical data. Hashing the URL set
# survives both paths.
#
# Established 2026-09-14 from UCI #967.
EXPECTED_FINGERPRINT = "0f9dc76443ca80d8e2cd7ad7183795a7c4619c1b0b2d4a37a60635003a568a3b"
EXPECTED_CLEAN_ROWS = 235370
EXPECTED_CLASS_COUNTS = {"1": 134850, "0": 100520}

# --- Reproducibility ---------------------------------------------------------
SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# --- Columns -----------------------------------------------------------------
LABEL = "label"

# Positive class for precision/recall/F1: phishing (0), not sklearn's default
# of 1. Label 1 is legitimate. Accuracy and ROC AUC are unaffected.
POS_LABEL = 0

# Identifiers and free text. Never model on these.
DROP_ALWAYS = ["FILENAME", "URL", "Domain", "Title"]

# Derived features. These were computed during dataset construction using
# information correlated with the label. URLSimilarityIndex is the worst
# offender and is effectively a restatement of the label.
DERIVED_SUSPECT = [
    "URLSimilarityIndex",
    "TLDLegitimateProb",
    "URLCharProb",
    "CharContinuationRate",
    "URLTitleMatchScore",
    "DomainTitleMatchScore",
]

# Page-content features. These require downloading and rendering the page, so
# they are not available at the moment a URL must be judged. Including them
# changes the problem being solved.
PAGE_CONTENT = [
    "LineOfCode", "LargestLineLength", "HasTitle", "HasFavicon", "Robots",
    "IsResponsive", "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription",
    "NoOfPopup", "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet",
    "HasSubmitButton", "HasHiddenFields", "HasPasswordField", "Bank", "Pay",
    "Crypto", "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
]

# Lexical features readable from the URL string alone.
# NOTE: the published dataset contains spelling errors in three of these
# ("Degits", "Spacial"). They are reproduced verbatim on purpose.
URL_LEXICAL = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
    "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
    "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL", "IsHTTPS",
]

# --- Feature sets ------------------------------------------------------------
# Three configurations. Every model gets fit on all three. The comparison
# between them is the analytical contribution of the project.
FEATURE_SETS = {
    "full": URL_LEXICAL + DERIVED_SUSPECT + PAGE_CONTENT,
    "no_derived": URL_LEXICAL + PAGE_CONTENT,
    "url_only": URL_LEXICAL,
}
