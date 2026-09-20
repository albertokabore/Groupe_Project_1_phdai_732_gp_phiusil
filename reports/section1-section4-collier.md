# PhDAI 732 Group Project Part 1 — Sections 1 and 4

**J. Eric Collier**
School of Computer and Information Sciences, University of the Cumberlands
PhDAI 732 - A03: Fundamentals of AI-Enabled Systems
Dr. Manjunath Paramashivaiah
September 20, 2026

---

## Section 1: Problem Framing and Project Design

The PhiUSIIL Phishing URL Dataset contains 235,795 labeled URLs, 134,850 of them
legitimate and 100,945 phishing, described by 54 features spanning URL structure,
domain properties, and rendered page content (Prasad & Chandra, 2024). It is a
popular benchmark, and published classifiers routinely report accuracy above .99 on
it. Our first run reproduced that result. A random forest fit on the complete feature
set scored 1.0000 on a held-out test partition of 58,843 records.

A perfect score on a security problem is not a finding. It is a signal that the model
has access to information it would not have at decision time. Kaufman et al. (2012)
define leakage as the introduction of information about the target that no legitimate
model could possess when a prediction is made, and Kapoor and Narayanan (2023)
document its consequences across 17 scientific fields and 294 papers, where corrected
analyses frequently erase the reported advantage of machine learning over simpler
methods. The risk is not that a leaking model fails a test. It is that the model
passes, gets published, and fails only in deployment.

Our single-feature screen located an obvious source. Fitting a classifier on
`URLSimilarityIndex` alone reaches .9967 accuracy, and the feature takes the value
100 for every one of the 134,850 legitimate records in the dataset. It does not
describe a URL. It restates the label. Four page-content features clear .94
individually: `NoOfExternalRef` at .9613, `LineOfCode` at .9554, `NoOfSelfRef` at
.9485, and `NoOfImage` at .9426. These are properties of a rendered page, which means
they are unavailable at the moment a URL must be judged, before anything has been
fetched. A model that uses them is answering a different question than the one
phishing detection poses.

The project was therefore designed as a leakage audit rather than a classification
contest. We defined three nested feature sets and fit the same four models against
each. The `full` set reproduces what the benchmark reports. The `no_derived` set
removes the six construction-derived features, including `URLSimilarityIndex`. The
`url_only` set retains the 18 lexical features readable from the URL string itself.
The design anticipated a substantial performance decline across that sequence, which
would have quantified how much of the benchmark result was artifact.

The decline did not occur. Random forest accuracy moves from 1.0000 on `full` to
.9999 on `no_derived` to .9973 on `url_only`. Removing 32 features, including the one
that restates the label and all 26 page-content features, costs roughly a quarter of
a percentage point. We tested the most plausible alternative explanation, that the
lexical models were memorizing domains rather than learning URL structure, by
refitting on a domain-grouped partition in which no domain appears in both training
and test data. Accuracy on `url_only` was .9975 under that partition, marginally
higher than under stratified sampling, which rules memorization out.

The explanation lies in the composition of the classes rather than in any feature
group. Six lexical features hold a single constant value across the entire legitimate
class. Every legitimate URL in the dataset is served over HTTPS. None contains an
ampersand, which means none carries a multi-parameter query string. None uses a bare
IP address in place of a domain, and none contains an obfuscated character. Genuine
web traffic does not distribute this way. The pattern is consistent with a legitimate
class harvested by an HTTPS-only crawl of bare homepages and a phishing class drawn
from a feed subject to no comparable filter. Supporting evidence appears in the
cleaning log: all 425 duplicate URLs removed during preprocessing belonged to the
phishing class, leaving 134,850 legitimate and 100,520 phishing records.

No individual lexical feature reaches the .95 threshold that flags leakage under a
single-feature screen, so nothing in the standard audit fires. The two classes
nevertheless differ systematically in shape before any single feature is examined.
This is leakage in the sampling frame rather than in a column, and it is not
reparable by feature ablation. The flatness of the ablation table is the evidence.
Had the artifact been localized in the features we removed, accuracy would have
fallen when we removed them.

Two consequences follow for how this report should be read. First, `url_only` cannot
be interpreted as a deployment estimate. It is the same construction artifact viewed
through a narrower window, and its .9973 accuracy carries no information about
performance on live traffic. Second, benchmark results on this dataset, including
those we reproduce here, describe a separation between two collection procedures
rather than a separation between phishing and legitimate URLs.

One reporting convention requires explicit statement because it inverts the meaning
of several standard metrics. In this dataset, label 1 denotes a legitimate URL and
label 0 denotes phishing. Scikit-learn treats the greater label as positive by
default, so precision, recall, and F1 computed without specification describe
detection of legitimate URLs. We set the positive class to phishing throughout.
Accuracy and area under the ROC curve are unaffected by the choice; the other three
metrics are not.

## Section 4: Team Collaboration and Process

Group 5 organized around a shared repository with five defined roles and a written
contract governing how numbers move between them. The contract holds that the seed
and the train/test assignment are generated once and read by everyone, that no member
loads the raw dataset directly, that analytical logic lives in the package rather than
in notebook cells, and that every reported figure is written to disk as JSON or CSV
before any section cites it. The final rule is the operative one. A report assembled
by five authors fails most often when two of them quote the same quantity from
different runs.

Work proceeded on per-role branches merged into the trunk through pull requests. The
data steward produced the cleaning log, the frozen splits, and the single-feature
leakage screen. The exploratory analysis owner produced the figures, including the
modal-share comparison that carries the collection-artifact finding. The modeling
owner produced the model roster, the evaluation function, and the metrics files. The
evaluation owner produced the justification for the reported metrics. The lead
produced the scaffold, the configuration module, and the final assembly.

Three process findings are worth recording, because each was caught by the review
workflow rather than by the individual who introduced it.

The first concerns the positive class. The evaluation function initially used
library defaults, which silently reported detection of legitimate URLs under labels
that read as phishing detection. The modeling owner identified the inversion
independently while validating output and worked around it by deriving the phishing
metrics from the confusion matrix. The defect was subsequently corrected at its
source, with the positive class defined once in configuration and threaded through
every metric call. Both discoveries were necessary. The workaround surfaced the
problem; the configuration change prevented it from recurring in later runs.

The second concerns silent failure modes in supporting code. A draft of the
exploratory analysis module contained a fallback path that, if an import failed,
substituted an arithmetically different quantity for single-feature accuracy and
wrote it to the same output file the data steward owned. The substituted values were
plausible in magnitude and would not have raised an error. Review removed the
fallback and reassigned the module to read the steward's outputs rather than
regenerate them. Two members writing the same artifact is a contract violation
regardless of whether the two results agree.

The third concerns documentation drift. The repository README stated the project's
argument, and that argument was written before the ablation was run. When the results
contradicted it, the README continued to promise a performance gap the data did not
produce, while the surrounding technical sections were updated normally. A partially
current document is more dangerous than an obviously stale one, because the updated
portions vouch for the rest. The argument section was rewritten against the measured
results before section drafting began. The general lesson, which we carried into the
remainder of the project, is that claims about results must either cite a file in the
results directory or be marked as expected and unverified.

## References

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in
machine-learning-based science. *Patterns, 4*(9), Article 100804.
https://doi.org/10.1016/j.patter.2023.100804

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining:
Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from
Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579

Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered
phishing URL detection framework based on similarity index and incremental learning.
*Computers & Security, 136*, Article 103545.
https://doi.org/10.1016/j.cose.2023.103545

Sharda, R., Delen, D., & Turban, E. (2023). *Business intelligence, analytics, data
science, and AI: A managerial perspective* (5th ed.). Pearson.
