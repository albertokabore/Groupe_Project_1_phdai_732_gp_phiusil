# Evaluation and Ethical Considerations

**Role:** Evaluation / Ethics  
**Project:** PhDAI 732 Group Project Part 1 — Group 5  
**Dataset:** PhiUSIIL Phishing URL Dataset (UCI #967)

## Evaluation Strategy

The evaluation phase is designed to determine not only how accurately the classifiers separate phishing and legitimate URLs, but also whether the observed performance is credible, reproducible, and meaningful for deployment. This distinction is central to the project because the repository frames the work as a leakage audit rather than a model leaderboard. Near-perfect test performance is therefore treated as a result that requires explanation, not as sufficient evidence that phishing detection has been solved.

The cleaned dataset contains 235,370 observations after 425 duplicate URLs were removed. The retained data include 134,850 legitimate URLs (label 1; 57.29%) and 100,520 phishing URLs (label 0; 42.71%). The project uses a frozen stratified 75/25 split with seed 42, producing 176,527 training observations and 58,843 test observations. Keeping the split fixed across models supports reproducibility and ensures that performance differences are attributable to the model or feature set rather than to different test samples.

All models are evaluated across the three feature configurations defined in `src/config.py`: `full`, `no_derived`, and `url_only`. The `full` set contains all 50 modeling features; `no_derived` removes the six construction-derived variables; and `url_only` retains 18 lexical features that can be read from the URL string. Comparing these configurations helps determine whether high benchmark performance depends on derived or page-content information.

## Metric Justification

No single performance metric is sufficient for evaluating a phishing classifier. The project therefore reports accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and five-fold cross-validated F1 for non-baseline models. Each metric answers a different evaluation question.

| Metric | Purpose | Relevance to This Project |
|---|---|---|
| Accuracy | Proportion of all URLs classified correctly | Useful as an overall benchmark, but insufficient by itself when dataset artifacts may make the classification task unusually easy. |
| Precision | Proportion of positive predictions that are correct | Measures the reliability of predictions for the designated positive class and helps quantify false-alert behavior. |
| Recall | Proportion of the positive class correctly identified | Important for understanding missed cases; for phishing-focused reporting, missed phishing URLs are especially important. |
| F1-score | Harmonic mean of precision and recall | Provides a balanced summary when both missed detections and incorrect alerts matter. |
| ROC-AUC | Threshold-independent ranking/discrimination measure | Indicates how strongly the model separates the classes across decision thresholds, but can still be inflated by dataset artifacts. |
| Confusion matrix | Counts TN, FP, FN, and TP | Makes the actual types and numbers of classification errors visible. |
| Cross-validated F1 | F1 estimated across five training folds | Provides an additional stability check beyond a single held-out test score. |

### Positive-Class Convention

The dataset uses **0 = phishing** and **1 = legitimate**. This convention is confirmed by the UCI dataset documentation. The shared `src/models.py` currently uses scikit-learn's default binary metric convention, so its stored `precision`, `recall`, and `f1` fields treat **label 1 (legitimate)** as the positive class. The confusion matrix is likewise unpacked according to the standard `[0, 1]` ordering.

This distinction is essential for correct interpretation. A statement such as “phishing recall” must not be taken directly from the default `recall` field. The repository now also contains phishing-specific calculations in `results/notebook_phishing_metrics.csv`, where phishing (label 0) is evaluated directly. The final report should identify the class convention whenever class-sensitive metrics are presented.

## Model Performance Across Feature Sets

The saved model outputs show that high performance persists even after derived and page-content features are removed. The table below reports the strongest non-baseline model in each feature configuration and includes phishing-specific precision, recall, and F1 calculated for label 0.

| Feature Set | Model | Accuracy | Phishing Precision | Phishing Recall | Phishing F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Full | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| No Derived | Random Forest | 0.9999 | 0.9999 | 0.9998 | 0.9998 | 1.0000 |
| URL Only | Random Forest | 0.9973 | 0.9992 | 0.9944 | 0.9968 | 0.9981 |

*Note.* Values are read from the repository's saved result files (`results/initial_model_comparison.csv` and `results/notebook_phishing_metrics.csv`), not transcribed from an unsaved notebook display. Rounded values are shown for readability.

The feature ablation produces only a small performance decline. Random-forest accuracy decreases from 1.0000 with the full feature set to 0.9999 without derived features and 0.9973 with URL-only features. The URL-only random forest still misses 140 phishing URLs out of 25,130 phishing test observations and generates 20 false alerts under the phishing-focused interpretation recorded in `notebook_phishing_metrics.csv`. Thus, the restricted model remains extremely strong within this dataset, but its small performance decline does not demonstrate that the underlying leakage/generalizability concern has disappeared.

## Leakage and Dataset-Construction Concerns

The single-feature leakage screen provides a major reason for caution. `URLSimilarityIndex` alone achieves approximately 0.9967 accuracy, `NoOfExternalRef` approximately 0.9613, and `LineOfCode` approximately 0.9554. These results identify features requiring scrutiny because a single variable that nearly reproduces the target can make held-out performance appear more impressive than the underlying task warrants.

High single-feature accuracy should not automatically be equated with proven leakage. The more defensible interpretation is that these variables warrant investigation of how they were created, what information they encode, and whether that information would be available in the intended deployment context. This distinction is important because predictive strength can arise from a legitimate signal, target-adjacent construction, sampling procedures, or a combination of these mechanisms.

The project's class-constancy analysis raises a broader concern that cannot be solved simply by deleting `URLSimilarityIndex`. Several lexical features are fixed across the entire legitimate class. For example, all 134,850 legitimate observations have `IsHTTPS = 1`, while only about 49.1% of phishing observations share that value. Other fields such as `IsDomainIP`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, and `NoOfAmpersandInURL` are also nearly constant for legitimate observations. These patterns are **consistent with a sampling or data-collection artifact**, although the exact collection mechanism cannot be established from the repository evidence alone.

This finding helps explain why `url_only` remains extremely accurate. The lexical feature set removes derived and rendered-page variables, but it can still encode systematic differences in how the two classes were collected. Therefore, `url_only` should be described as containing features available from the URL string at decision time, not as proof that the resulting 99%+ performance will generalize to arbitrary future phishing traffic.

## Generalizability and Validation

The project's held-out test results establish strong **within-dataset** discrimination. They do not establish equivalent performance across new time periods, independent phishing feeds, different legitimate-site sampling procedures, or adversarially adapted URLs. This distinction is especially important for cybersecurity applications because attackers can modify URL structure and infrastructure in response to detection systems.

The repository also includes a domain-grouped split with zero domain overlap between training and test sets. Such a comparison is useful because it reduces the possibility that performance is explained simply by seeing the same domains in both partitions. However, a grouped split cannot eliminate a class-wide collection artifact if the same collection process affects both training and test data. External and temporal validation would therefore provide stronger evidence of deployment generalizability.

## Ethical Considerations

### Responsible Communication of Performance

The primary ethical obligation is to communicate the model's performance without overstating what the experiment establishes. Describing the model only as “99.7% accurate” could imply that the same effectiveness should be expected against arbitrary real-world phishing attacks. The evidence supports a narrower claim: the URL-only random forest achieves approximately 99.73% accuracy on the project's frozen PhiUSIIL test split. Because the dataset contains unusually strong class patterns, this benchmark should not be presented as a guaranteed production detection rate.

This approach is consistent with the NIST AI Risk Management Framework, which emphasizes validity and reliability, security and resilience, accountability, and transparency as characteristics relevant to trustworthy AI. In this project, those principles support documenting the dataset, class convention, feature availability, leakage concerns, evaluation procedure, and limitations rather than reporting performance without context.

### False Negatives and Missed Phishing

A phishing URL classified as legitimate represents a security failure. Depending on the context, a missed phishing site can expose users to credential theft, financial fraud, malicious downloads, or unauthorized access. Consequently, phishing recall and the number of missed phishing observations should be visible in the evaluation rather than hidden within overall accuracy.

For the URL-only random forest, phishing recall is approximately 0.9944, corresponding to 140 missed phishing observations in the held-out test set. This is a very strong within-dataset result, but the remaining errors illustrate why even high accuracy does not imply zero operational risk.

### False Positives and Operational Harm

Incorrectly flagging legitimate URLs can block access to valid resources, create unnecessary user friction, and contribute to alert fatigue when deployed at scale. For the URL-only random forest, the phishing-focused results record 20 false alerts on the held-out test set. The appropriate tradeoff between missed phishing and false alerts depends on the intended deployment context and risk tolerance; it should not be determined by accuracy alone.

### Transparency and Reproducibility

The repository's shared workflow supports responsible reporting. A single seed and frozen split are used across team members, reusable logic is kept in `src/`, and results are persisted as JSON or CSV. These controls reduce accidental variation and make reported numbers traceable to specific result files. The dataset itself is not committed; instead, `load_data()` validates a cleaned-data fingerprint so independently downloaded copies can be checked for consistency.

Reproducibility is ethically relevant because a result that cannot be traced or reproduced is difficult to audit. The final paper should therefore continue to report values from the saved result artifacts rather than manually copying values from transient notebook output.

## Limitations

1. **Single-dataset evaluation.** The current results are based primarily on PhiUSIIL and should not be generalized automatically to all phishing traffic.
2. **Potential sampling artifacts.** Class-constant and near-constant lexical patterns suggest systematic differences between phishing and legitimate observations. The precise collection mechanism is not established by the repository and should not be asserted as fact.
3. **Feature availability differs by deployment stage.** Page-content variables require fetching or rendering a page, whereas lexical URL variables are available before page retrieval. These configurations represent different operational problems.
4. **Metric orientation requires explicit documentation.** Default precision, recall, and F1 in `src/models.py` use legitimate URLs (label 1) as the positive class; phishing-specific interpretations must use label 0 metrics.
5. **Concept drift and adversarial adaptation are not tested.** Future phishing campaigns may differ from the historical observations represented in the dataset.
6. **External validation is still needed.** Independent and temporally separated datasets would provide stronger evidence of real-world robustness.

## Conclusion

The evaluation results demonstrate exceptionally strong classification performance within the PhiUSIIL dataset, including after restricting models to URL lexical features. However, the project's most important analytical finding is not simply the high score. The leakage screen, class-constancy analysis, and flat feature-ablation results show that performance must be interpreted in the context of dataset construction and potential sampling artifacts.

For this reason, the final report should present accuracy, precision, recall, F1-score, ROC-AUC, and confusion-matrix information together with the class convention and feature-set definitions. Phishing-specific metrics should be used when discussing phishing detection consequences. Ethical reporting also requires distinguishing held-out benchmark performance from claims about deployment effectiveness, documenting false-negative and false-positive consequences, and clearly acknowledging limitations in external validity.

Overall, the evidence supports the conclusion that the models are highly effective **within the evaluated dataset**, while further external and temporal validation is necessary before making equivalent claims about production phishing detection.

## References

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15, 1–21. https://doi.org/10.1145/2382577.2382579

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). U.S. Department of Commerce. https://doi.org/10.6028/NIST.AI.100-1

Prasad, A., & Chandra, S. (2024). *PhiUSIIL phishing URL (website)* [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website+dataset

Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security, 136*, 103545. https://doi.org/10.1016/j.cose.2023.103545

Sahingoz, O. K., Buber, E., Demir, O., & Diri, B. (2019). Machine learning based phishing detection from URLs. *Expert Systems with Applications, 117*, 345–357. https://doi.org/10.1016/j.eswa.2018.09.029
