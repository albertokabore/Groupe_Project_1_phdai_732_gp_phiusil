"""Build reports/Deliverable2_Report.docx from results/ and figures/.

Every number in the report is read from the Part 2 artifacts written by
src/tuning.py and src/final_eval.py, so the text cannot drift from the results.
Rerun after any change to results/:

    python reports/build_deliverable2_docx.py

Format: APA 7 student paper. Times New Roman 12 pt, double-spaced, 1-inch
margins, page numbers top right, APA heading levels, hanging-indent references.
"""

import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.final_eval import MIN_GROUP_N  # noqa: E402

R, F = C.RESULTS_DIR, C.FIGURES_DIR
OUT = C.REPORTS_DIR / "Deliverable2_Report.docx"
FONT = "Times New Roman"

# --- Load results --------------------------------------------------------------

S = pd.read_csv(R / "tuning_summary.csv").set_index("model")
FINAL = json.loads((R / "final_model.json").read_text())
M = FINAL["test_metrics"]
CMP = pd.read_csv(R / "final_model_comparison.csv").set_index(["model", "config"])
FSET = pd.read_csv(R / "final_feature_set_comparison.csv")
THR = pd.read_csv(R / "final_thresholds.csv").set_index("operating_point")
GROUPED = json.loads((R / "final_grouped_split.json").read_text())
TLD = pd.read_csv(R / "final_group_error_rates_tld.csv").set_index("TLD")
IMP = pd.read_csv(R / "final_permutation_importance.csv")
PRED = pd.read_csv(R / "final_test_predictions.csv")
HGB = pd.read_csv(R / "tuning" / "cv_results_hist_gradient_boosting.csv")
P1 = pd.read_csv(R / "initial_model_comparison.csv").set_index(["feature_set", "model"])
N_TRAIN = json.loads((R / "tuning_manifest.json").read_text())["n_train"]
N_URL_FEATURES = len(C.FEATURE_SETS["url_only"])

NAMES = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree",
         "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting"}
final_name = FINAL["model"]
default_final = CMP.loc[(final_name, "default")]
missed = PRED[(PRED.label == C.POS_LABEL) & (PRED.pred != C.POS_LABEL)].p_phishing
top8 = HGB.mean_test_f1_phishing.nlargest(8)
p1_rf = P1.loc[("url_only", "random_forest")]


def v(x, d=4):
    """APA style for statistics bounded by 1: no leading zero."""
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("0.") else s


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def n(x):
    return f"{int(x):,}"


WORDS = dict(enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven",
                        "eight", "nine"]))


def num(x):
    """APA: words for whole numbers below 10 that are not compared with larger ones."""
    return WORDS.get(int(x), n(x))


def precision_at(prevalence):
    caught = M["recall"] * prevalence
    return caught / (caught + M["legit_false_alarm_rate"] * (1 - prevalence))


n_phish = M["phishing_caught"] + M["phishing_missed"]
n_legit = M["legit_flagged"] + M["legit_passed"]
full_perfect = int((FSET[FSET.feature_set == "full"].f1 == 1).sum())
edu, com = TLD.loc["edu"], TLD.loc["com"]
thin = {t: int(TLD.loc[t, "n_legit"]) for t in ["app", "io", "dev"]}
top_features = IMP.feature.head(4).tolist()

# --- Document helpers -----------------------------------------------------------

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1))

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
pf = normal.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
pf.space_before = pf.space_after = Pt(0)

for name in ("Heading 1", "Heading 2", "Title", "Caption"):
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.color.rgb = FONT, Pt(12), None
    st.font.italic = False
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)
    for theme_attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(theme_attr), None)
    st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    st.paragraph_format.space_before = st.paragraph_format.space_after = Pt(0)
    st.paragraph_format.keep_with_next = True


def page_number_header(section):
    """APA: page number flush right in the header of every page."""
    p = section.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run()
    run.font.name, run.font.size = FONT, Pt(12)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


page_number_header(sec)


def rich(p, text):
    """Add text with *italic* spans (single asterisks) to paragraph p."""
    for i, chunk in enumerate(text.split("*")):
        if chunk:
            run = p.add_run(chunk)
            run.italic = i % 2 == 1
    return p


def para(text, indent=True, align=None, bold=False, keep=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    if align is not None:
        p.alignment = align
    if keep:
        p.paragraph_format.keep_with_next = True
    if bold:
        p.add_run(text).bold = True
    else:
        rich(p, text)
    return p


def h1(text):
    p = doc.add_paragraph(style="Heading 1")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text).bold = True


def label(kind, number, title):
    """APA table/figure label: bold number line, italic title line."""
    p = para(f"{kind} {number}", indent=False, bold=True, keep=True)
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.add_run(title).italic = True


def note(text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    p.add_run("Note. ").italic = True
    rich(p, text)


def set_cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, val in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), "6")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def table(header, rows, widths, font_size=10, left_cols=(0,)):
    """APA table: horizontal rules above and below the header and at the end."""
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for r_i, values in enumerate([header] + rows):
        for c_i, value in enumerate(values):
            cell = t.cell(r_i, c_i)
            cell.width = Inches(widths[c_i])
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(2)
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if c_i in left_cols
                           else WD_ALIGN_PARAGRAPH.CENTER)
            run = p.add_run(str(value))
            run.font.size = Pt(font_size)
            edges = {}
            if r_i == 0:
                edges.update(top="single", bottom="single")
            if r_i == len(rows):
                edges.update(bottom="single")
            if edges:
                set_cell_border(cell, **edges)
    return t


def figure(number, title, path, width, note_text):
    label("Figure", number, title)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.add_run().add_picture(str(path), width=Inches(width))
    note(note_text)


# --- Title page ------------------------------------------------------------------

for _ in range(3):
    doc.add_paragraph()
para("Tuning, Evaluating, and Deploying a URL-Only Phishing Classifier: "
     "Deliverable 2", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
doc.add_paragraph()
for line in ["Group 5: [Team member names]",
             "[Department, University]",
             "PhDAI 732: [Course Name]",
             "[Instructor Name]",
             "September 27, 2026"]:
    para(line, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

para("Tuning, Evaluating, and Deploying a URL-Only Phishing Classifier",
     indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
para(f"Deliverable 1 showed that near-perfect accuracy on the PhiUSIIL phishing dataset "
     f"(Prasad & Chandra, 2024) comes largely from construction artifacts, and that the "
     f"defensible estimate uses only the {N_URL_FEATURES} lexical features readable from "
     f"the URL string. This deliverable tunes that URL-only model, evaluates it once on the "
     f"held-out test partition, and asks who bears its errors and how it would behave in "
     f"production.")

# --- 1. Hyperparameter tuning ----------------------------------------------------

h1("Hyperparameter Tuning")
lr, dt, rf, gb = (S.loc[k] for k in NAMES)
n_cfg = int(S.n_candidates.sum())
para(f"The course template tunes a matrix-factorization recommender on root mean squared "
     f"error (RMSE). Our task is binary classification, so we applied the same procedure with "
     f"scikit-learn's *GridSearchCV* (Pedregosa et al., 2011), scoring phishing-class F1 with "
     f"seeded, stratified five-fold cross-validation (CV) on the URL-only training partition. "
     f"We searched the three Deliverable 1 models and added histogram gradient boosting, "
     f"{n_cfg} configurations and {n_cfg * C.CV_FOLDS} fits in total. Every grid contained "
     f"the untuned default, so each gain in Table 1 is measured on identical folds. Grid "
     f"search suits grids this small; random or Bayesian search scales better as the number "
     f"of hyperparameters grows (Bergstra & Bengio, 2012).")
para(f"Gains were real but small. Logistic regression improved most (CV F1 {v(lr.default_cv_f1_phishing)} "
     f"to {v(lr.best_cv_f1_phishing)}), entirely through weaker regularization: F1 rose "
     f"monotonically with *C*, so the lexical features are not overfitting at this sample "
     f"size. Tree and forest gains were smaller than one fold-to-fold standard deviation "
     f"(about {v(S.best_cv_f1_phishing_sd.mean())}). Gradient boosting scored highest "
     f"({v(gb.best_cv_f1_phishing)}) and was selected as the final model before the test "
     f"data were opened. Its chosen learning rate and iteration count sit at the grid edge, "
     f"but the top eight configurations fall within {v(top8.max() - top8.min(), 5)} of each "
     f"other, a plateau rather than an unexplored optimum.")

label("Table", 1, "Cross-Validated Phishing F1 Before and After Grid Search (URL-Only Features)")
rows = []
for key, row in S.iterrows():
    params = json.loads(row.best_params)
    short = ", ".join(f"{k.replace('clf__', '')}={p}" for k, p in params.items())
    rows.append([NAMES[key], int(row.n_candidates), v(row.default_cv_f1_phishing),
                 v(row.best_cv_f1_phishing), f"+{v(row.cv_f1_gain, 5)}", short])
table(["Model", "Configs", "Default F1", "Tuned F1", "Gain", "Selected hyperparameters"],
      rows, [1.35, 0.6, 0.75, 0.7, 0.6, 2.5], font_size=9, left_cols=(0, 5))
note(f"Mean over five stratified folds of the {n(N_TRAIN)}-URL training "
     f"partition. The fold-to-fold standard deviation of F1 is about "
     f"{v(S.best_cv_f1_phishing_sd.mean())} for every model.")

# --- 2. Final model evaluation ---------------------------------------------------

h1("Final Model Evaluation")
ci = FINAL["bootstrap_95ci"]["f1"]
para(f"Table 2 reports each tuned model on the {n(M['n_test'])} held-out URLs. The tuned "
     f"gradient boosting model reached F1 = {v(M['f1'])}, 95% bootstrap CI "
     f"[{v(ci[0])}, {v(ci[1])}], with precision = {v(M['precision'])}, recall = "
     f"{v(M['recall'])}, and a Matthews correlation coefficient of {v(M['mcc'])}. It missed "
     f"{n(M['phishing_missed'])} of {n(n_phish)} phishing URLs and flagged "
     f"{n(M['legit_flagged'])} of {n(n_legit)} legitimate ones (Figure 1), against "
     f"{n(default_final.phishing_missed)} and {n(default_final.legit_flagged)} before tuning "
     f"and {n(p1_rf['confusion_matrix.fp'])} and {n(p1_rf['confusion_matrix.fn'])} for the "
     f"Deliverable 1 random forest. We weight precision, recall, and precision-recall area "
     f"(PR AUC) above accuracy because the two errors carry different costs (Saito & "
     f"Rehmsmeier, 2015; Chicco & Jurman, 2020). RMSE and mean absolute error (MAE) are "
     f"defined for ratings, so we computed them on the predicted phishing probability: RMSE "
     f"fell from {v(default_final.probability_rmse)} to {v(M['probability_rmse'])} and MAE "
     f"from {v(default_final.probability_mae)} to {v(M['probability_mae'])}.")

label("Table", 2, "Tuned Models on the Held-Out Test Partition (URL-Only Features)")
rows = []
for key in NAMES:
    r = CMP.loc[(key, "tuned")]
    rows.append([NAMES[key], v(r.precision), v(r.recall), v(r.f1), v(r.pr_auc),
                 v(r.probability_rmse), v(r.probability_mae), n(r.phishing_missed),
                 n(r.legit_flagged)])
rows.append([f"{NAMES[final_name]} (default)", v(default_final.precision),
             v(default_final.recall), v(default_final.f1), v(default_final.pr_auc),
             v(default_final.probability_rmse), v(default_final.probability_mae),
             n(default_final.phishing_missed), n(default_final.legit_flagged)])
table(["Model", "Precision", "Recall", "F1", "PR AUC", "RMSE", "MAE", "Missed", "False alarms"],
      rows, [1.75, 0.7, 0.6, 0.6, 0.6, 0.55, 0.55, 0.6, 0.65], font_size=9)
note("Phishing is the positive class. RMSE and MAE are computed on the predicted phishing "
     "probability. *Missed* counts phishing URLs predicted legitimate; *false alarms* counts "
     "legitimate URLs predicted phishing.")

figure(1, f"Confusion Matrix of the Tuned {NAMES[final_name].title()} Model on the Test Partition",
       F / "final_confusion_matrix.png", 3.2,
       "Color uses a log scale so the error cells remain visible.")

bud = THR.loc["fpr_budget"]
para(f"Two results qualify the headline. First, tuning traded a little ranking quality for "
     f"decision quality: ROC AUC fell from {v(default_final.roc_auc)} to {v(M['roc_auc'])} "
     f"because the search optimized F1 at a 0.5 threshold. Second, the remaining errors are "
     f"not borderline. The F1-optimal threshold learned on training folds was "
     f"{FINAL['thresholds']['f1_optimal']:.3f}, and {int((missed < 0.1).sum())} of the "
     f"{len(missed)} missed phishing URLs received a phishing probability below 0.1 "
     f"(Figure 2). Spending a 0.1% false-alarm budget (threshold {bud.threshold:.3f}) recovers "
     f"only {int(M['phishing_missed'] - bud.phishing_missed)} more phishing URLs for "
     f"{int(bud.legit_flagged - M['legit_flagged'])} more false alarms. Comparing predictions "
     f"with actual labels shows why: the misses are short HTTPS homepages that impersonate "
     f"brands, such as appleid-find.cloud, netflix-vn.com, and amazon5551.com. Lexically they "
     f"look like the legitimate class, and none of the {N_URL_FEATURES} counts encodes a brand name.")
figure(2, "Predicted Phishing Probability for the Phishing URLs the Model Missed",
       F / "final_error_confidence.png", 5.6,
       "Vertical axis is linear from 0 to 5 and logarithmic above.")
para(f"Two checks support the result. Refit on a domain-grouped split, where no domain "
     f"appears in both partitions, the model scored F1 = {v(GROUPED['test_metrics']['f1'])}, "
     f"so it is not memorizing domains. On the full feature set, {num(full_perfect)} of the four "
     f"tuned models remain perfect, which confirms that the Deliverable 1 leakage finding "
     f"survives tuning (Kapoor & Narayanan, 2023). The areas for improvement therefore lie in "
     f"new information, such as brand-similarity and domain-registration features, not in "
     f"further tuning.")

# --- 3. Ethical considerations ---------------------------------------------------

h1("Ethical Considerations")
para(f"PhiUSIIL has no demographic attributes, but bias still enters through the kind of "
     f"site owner behind a URL. The legitimate class consists of HTTPS homepages of "
     f"established domains, while phishing URLs come from threat feeds (Prasad & Chandra, "
     f"2024), so the model partly learns the difference between two collection processes, a "
     f"sampling bias common in security datasets (Arp et al., 2022). Its false alarms are "
     f"uneven (Figure 3): {pct(edu.legit_false_alarm_rate)} of legitimate .edu URLs "
     f"({int(round(edu.legit_false_alarm_rate * edu.n_legit))} of {n(edu.n_legit)}) were "
     f"flagged, against {pct(com.legit_false_alarm_rate, 3)} for .com, because universities "
     f"use long, multi-level addresses. For newer TLDs the rate cannot be measured at all: "
     f"the test set holds {thin['app']}, {thin['io']}, and {thin['dev']} legitimate .app, "
     f".io, and .dev URLs. And because every HTTP URL in the dataset is phishing, the model "
     f"treats HTTP as proof of fraud and would block every legitimate HTTP-only site. A "
     f"blocked site loses traffic and reputation, a harm that falls unevenly across groups "
     f"(Mehrabi et al., 2021).")
figure(3, "False-Alarm Rate on Legitimate URLs by Top-Level Domain",
       F / "final_false_alarms_by_tld.png", 5.6,
       f"Test partition, TLDs with at least {MIN_GROUP_N} legitimate URLs. The dotted line is the "
       f"overall rate.")
para(f"Transparency is a partial safeguard. Four features ({', '.join(top_features)}) carry "
     f"most of the decision, which a flagged owner can be told, although the same list tells "
     f"attackers what to imitate. Privacy risk is modest for URL-only screening, but a log of "
     f"scored URLs is a browsing history and should be minimized and retained briefly. We "
     f"would publish per-TLD error rates in a model card (Mitchell et al., 2019), add "
     f"legitimate HTTP and new-TLD samples, drop or down-weight *IsHTTPS*, and give site "
     f"owners an appeal path.")

# --- 4. Real-world application ---------------------------------------------------

h1("Real-World Application")
para("The natural use is a pre-fetch screen. An email gateway, browser extension, or DNS "
     "resolver scores a URL before the page is requested, which is exactly the information "
     "setting the URL-only features simulate. Lexical models are fast and need no page "
     f"download (Sahingoz et al., 2019), so a gradient boosting model over {N_URL_FEATURES} "
     "numeric features can run inline at mail-server volume. It should not act alone. Blacklists catch known "
     "campaigns but lag new ones (Sheng et al., 2009), and our model has the opposite blind "
     "spot: clean-looking new domains. A layered design uses the classifier as the first "
     "stage and routes uncertain URLs to reputation and page-content checks.")
label("Table", 3, "Expected Precision of the Final Model at Different Phishing Base Rates")
rows = []
for prev in [n_phish / M["n_test"], 0.01, 0.001]:
    caught = M["recall"] * prev * 1e6
    fa = M["legit_false_alarm_rate"] * (1 - prev) * 1e6
    rows.append([pct(prev, 1), n(round(caught)), n(round(fa)), v(caught / (caught + fa))])
table(["Phishing share of traffic", "Caught per million URLs", "False alarms per million",
       "Precision"], rows, [1.9, 1.6, 1.6, 1.0], font_size=10)
note("Uses the test-set recall and false-alarm rate of the final model. The first row is the "
     "test partition's own class balance.")
para(f"Table 3 shows the most important production adjustment. The test set is "
     f"{pct(n_phish / M['n_test'], 0)} phishing; live traffic is far cleaner. At one phishing "
     f"URL per thousand, the same recall and false-alarm rate give a precision of about "
     f"{v(precision_at(0.001), 2)}, so roughly one alert in {num(round(1 / (1 - precision_at(0.001))))} "
     f"is false, a base-rate effect that benchmark evaluations routinely hide (Arp et al., "
     f"2022). Four modifications would make the system effective in production. First, set "
     f"the operating point by cost: quarantine in a mail gateway can tolerate the budget "
     f"threshold, while a browser interstitial needs the lower false-alarm default. Second, "
     f"add features attackers cannot cheaply change, such as domain age, certificate "
     f"transparency records, and edit distance to protected brand names, which targets the "
     f"confident misses directly. Third, plan for drift. Phishing kits change within weeks, so "
     f"score distributions and per-TLD false-alarm rates should be monitored, the model "
     f"retrained on a rolling window, and each release validated on a time-ordered split "
     f"(Gama et al., 2014). Fourth, deploy in shadow mode first, logging decisions without "
     f"blocking, then enforce once live precision is measured.")

# --- 5. Final thoughts -----------------------------------------------------------

h1("Final Thoughts and Conclusion")
CONCLUSION = [
    f"*Tuning helped, but modestly.* Grid search raised cross-validated F1 for every model. "
    f"The tuned gradient boosting model improved test F1 from {v(default_final.f1)} to "
    f"{v(M['f1'])}, cutting missed phishing URLs from {n(default_final.phishing_missed)} to "
    f"{n(M['phishing_missed'])} and false alarms from {n(default_final.legit_flagged)} to "
    f"{n(M['legit_flagged'])}.",
    f"*The remaining errors are not a tuning problem.* Of the {len(missed)} missed URLs, "
    f"{int((missed < 0.1).sum())} were scored as confidently legitimate. They are brand "
    f"impersonations on clean HTTPS homepages, which lexical counts cannot see, so further "
    f"gains require new information rather than new hyperparameters.",
    f"*The dataset shapes the model as much as phishing does.* HTTPS acts as a proxy for the "
    f"label, legitimate .edu sites are flagged far more often than .com sites, and newer "
    f"TLDs such as .app and .dev cannot be evaluated at all.",
    f"*Benchmark scores overstate field performance.* At a realistic rate of one phishing "
    f"URL per thousand, precision falls from {v(M['precision'])} to about "
    f"{v(precision_at(0.001), 2)}, so evaluation must reflect deployment base rates.",
    "*The Deliverable 1 argument holds end to end.* The contribution of this project is the "
    "audit of where a near-perfect number comes from, not the number itself.",
    "*A deployable system needs more than this model.* It would use the fast URL screen as a "
    "first stage, add brand-similarity and domain-reputation features, monitor per-group "
    "error rates and drift, and retrain on a schedule.",
    "*The work is reproducible.* All code, the frozen split, search results, and figures are "
    "in the project repository (src/tuning.py, src/final_eval.py, and Sections 4 to 8 of the "
    "project notebook), and every number in this report is generated from those files.",
]
for item in CONCLUSION:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0)
    rich(p, item)
CONCLUSION_WORDS = sum(len(item.replace("*", "").split()) for item in CONCLUSION)

# --- References ------------------------------------------------------------------

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
para("References", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
REFERENCES = [
    "Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., "
    "Cavallaro, L., & Rieck, K. (2022). Dos and don'ts of machine learning in computer "
    "security. In *Proceedings of the 31st USENIX Security Symposium*. USENIX Association. "
    "https://www.usenix.org/conference/usenixsecurity22/presentation/arp",
    "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. "
    "*Journal of Machine Learning Research, 13*, 281–305.",
    "Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient "
    "(MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics, 21*, "
    "Article 6. https://doi.org/10.1186/s12864-019-6413-7",
    "Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on "
    "concept drift adaptation. *ACM Computing Surveys, 46*(4), Article 44. "
    "https://doi.org/10.1145/2523813",
    "Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in "
    "machine-learning-based science. *Patterns, 4*(9), Article 100804. "
    "https://doi.org/10.1016/j.patter.2023.100804",
    "Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on "
    "bias and fairness in machine learning. *ACM Computing Surveys, 54*(6), Article 115. "
    "https://doi.org/10.1145/3457607",
    "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, "
    "E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of "
    "the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. "
    "https://doi.org/10.1145/3287560.3287596",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, "
    "M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., "
    "Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in "
    "Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
    "Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing "
    "URL detection framework based on similarity index and incremental learning. *Computers & "
    "Security, 136*, Article 103545. https://doi.org/10.1016/j.cose.2023.103545",
    "Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing "
    "detection from URLs. *Expert Systems with Applications, 117*, 345–357. "
    "https://doi.org/10.1016/j.eswa.2018.09.029",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than "
    "the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), "
    "Article e0118432. https://doi.org/10.1371/journal.pone.0118432",
    "Sheng, S., Wardman, B., Warner, G., Cranor, L. F., Hong, J., & Zhang, C. (2009). An "
    "empirical analysis of phishing blacklists. In *Proceedings of the Sixth Conference on "
    "Email and Anti-Spam (CEAS 2009)*.",
]
for ref in REFERENCES:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.5)
    rich(p, ref)

# --- Appendix: team contributions ----------------------------------------------------

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
para("Appendix", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
para("Team Contributions", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
para("Each member should confirm or edit their row before submission. Commit history in the "
     "project repository records who produced each artifact.")
table(["Team member", "Role", "Deliverable 2 contribution"], [
    ["[Name]", "Lead / scaffold", "Final assembly, editing to one voice, Conclusion"],
    ["[Name]", "Data steward", "Frozen split verification, domain-grouped split"],
    ["[Name]", "Modeling", "Grid search (src/tuning.py), final evaluation (src/final_eval.py)"],
    ["[Name]", "Evaluation / ethics", "Metric justification, per-TLD error analysis, Ethics"],
    ["[Name]", "EDA / application", "Figures, Real-World Application"],
], [1.4, 1.5, 3.6], font_size=10, left_cols=(0, 1, 2))

doc.save(OUT)
print(f"Wrote {OUT.relative_to(ROOT)}; conclusion is {CONCLUSION_WORDS} words")
