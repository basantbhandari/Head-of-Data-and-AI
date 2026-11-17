# Chapter 12. Model Evaluation and Experimentation

_Part III — Machine Learning_

Metrics, validation strategies, calibration, statistical model comparison, slice and fairness evaluation, A/B testing, ablations and debugging models that underperform. Senior candidates are judged on choosing the right metric for the business problem and on knowing when the numbers can be trusted.

This chapter contains **41 questions**.

## Contents

- **12.1** [How is A/B testing used to evaluate machine learning models?](#121-how-is-ab-testing-used-to-evaluate-machine-learning-models)
- **12.2** [What are precision, recall, specificity, F1 score and accuracy?](#122-what-are-precision-recall-specificity-f1-score-and-accuracy)
- **12.3** [What is the confusion matrix, why do we need it, and how do you interpret it?](#123-what-is-the-confusion-matrix-why-do-we-need-it-and-how-do-you-interpret-it)
- **12.4** [What are the common evaluation metrics for classification, and when would you prefer each?](#124-what-are-the-common-evaluation-metrics-for-classification-and-when-would-you-prefer-each)
- **12.5** [When would you use accuracy vs other metrics?](#125-when-would-you-use-accuracy-vs-other-metrics)
- **12.6** [When would you use log loss vs accuracy?](#126-when-would-you-use-log-loss-vs-accuracy)
- **12.7** [What metrics would you use for a multi-class classification problem?](#127-what-metrics-would-you-use-for-a-multi-class-classification-problem)
- **12.8** [How do you handle class imbalance, in both the evaluation and the model?](#128-how-do-you-handle-class-imbalance-in-both-the-evaluation-and-the-model)
- **12.9** [What is the ROC curve and AUC, what are their benefits, and what are their limitations?](#129-what-is-the-roc-curve-and-auc-what-are-their-benefits-and-what-are-their-limitations)
- **12.10** [What are the common regression metrics, and how do MAE, MSE and RMSE differ?](#1210-what-are-the-common-regression-metrics-and-how-do-mae-mse-and-rmse-differ)
- **12.11** [How do you choose the right evaluation metric for a given problem?](#1211-how-do-you-choose-the-right-evaluation-metric-for-a-given-problem)
- **12.12** [How do you compare two models and decide whether one is really better?](#1212-how-do-you-compare-two-models-and-decide-whether-one-is-really-better)
- **12.13** [What is cross-validation, what types are there, and why does it matter?](#1213-what-is-cross-validation-what-types-are-there-and-why-does-it-matter)
- **12.14** [How do you tune hyperparameters without overfitting the validation set?](#1214-how-do-you-tune-hyperparameters-without-overfitting-the-validation-set)
- **12.15** [How do you evaluate unsupervised learning models?](#1215-how-do-you-evaluate-unsupervised-learning-models)
- **12.16** [How do you evaluate a clustering algorithm?](#1216-how-do-you-evaluate-a-clustering-algorithm)
- **12.17** [How do you evaluate a recommendation system?](#1217-how-do-you-evaluate-a-recommendation-system)
- **12.18** [What is LLM-as-a-judge?](#1218-what-is-llm-as-a-judge)
- **12.19** [Describe your approach to debugging an underperforming ML model.](#1219-describe-your-approach-to-debugging-an-underperforming-ml-model)
- **12.20** [How is a logistic regression model evaluated?](#1220-how-is-a-logistic-regression-model-evaluated)
- **12.21** [How do you choose a classifier's decision threshold, and how do you weigh false positives against false negatives?](#1221-how-do-you-choose-a-classifiers-decision-threshold-and-how-do-you-weigh-false-positives-against-false-negatives)
- **12.22** [How can you measure the performance of a dimensionality reduction algorithm on your dataset?](#1222-how-can-you-measure-the-performance-of-a-dimensionality-reduction-algorithm-on-your-dataset)
- **12.23** [What cross-validation technique would you use on time-series data, and why does standard k-fold fail?](#1223-what-cross-validation-technique-would-you-use-on-time-series-data-and-why-does-standard-k-fold-fail)
- **12.24** [Which among these is more important: model accuracy or model performance?](#1224-which-among-these-is-more-important-model-accuracy-or-model-performance)
- **12.25** [Difference between R2 and adjusted R2? Why do you prefer adjusted R2?](#1225-difference-between-r2-and-adjusted-r2-why-do-you-prefer-adjusted-r2)
- **12.26** [When should you use PR-AUC over ROC-AUC?](#1226-when-should-you-use-pr-auc-over-roc-auc)
- **12.27** [How do you run a good ablation study?](#1227-how-do-you-run-a-good-ablation-study)
- **12.28** [What is stratified cross-validation, and when should you not use it?](#1228-what-is-stratified-cross-validation-and-when-should-you-not-use-it)
- **12.29** [Training accuracy is 99%, validation accuracy is 60%. How do you debug it?](#1229-training-accuracy-is-99-validation-accuracy-is-60-how-do-you-debug-it)
- **12.30** [Training and validation accuracy are both 50%. What could be wrong?](#1230-training-and-validation-accuracy-are-both-50-what-could-be-wrong)
- **12.31** [Your model performs well offline but poorly in production. Why?](#1231-your-model-performs-well-offline-but-poorly-in-production-why)
- **12.32** [Your model's accuracy decreases after adding more training data. Explain possible reasons.](#1232-your-models-accuracy-decreases-after-adding-more-training-data-explain-possible-reasons)
- **12.33** [A feature has extremely high importance. How do you determine whether it is legitimate or leakage?](#1233-a-feature-has-extremely-high-importance-how-do-you-determine-whether-it-is-legitimate-or-leakage)
- **12.34** [Your validation score varies significantly between folds. What does that indicate?](#1234-your-validation-score-varies-significantly-between-folds-what-does-that-indicate)
- **12.35** [Your model predicts only the majority class. How do you fix it?](#1235-your-model-predicts-only-the-majority-class-how-do-you-fix-it)
- **12.36** [What is probability calibration, how do you measure it, and how do you fix a miscalibrated model?](#1236-what-is-probability-calibration-how-do-you-measure-it-and-how-do-you-fix-a-miscalibrated-model)
- **12.37** [How do you evaluate a model across slices and check that it is fair across groups?](#1237-how-do-you-evaluate-a-model-across-slices-and-check-that-it-is-fair-across-groups)
- **12.38** [How do you evaluate whether a model will hold up under distribution shift?](#1238-how-do-you-evaluate-whether-a-model-will-hold-up-under-distribution-shift)
- **12.39** [Your offline metric improved, but the A/B test showed no gain. Why, and what do you do?](#1239-your-offline-metric-improved-but-the-ab-test-showed-no-gain-why-and-what-do-you-do)
- **12.40** [How do you evaluate a model in production when labels are delayed, scarce or affected by the model's own decisions?](#1240-how-do-you-evaluate-a-model-in-production-when-labels-are-delayed-scarce-or-affected-by-the-models-own-decisions)
- **12.41** [How do you build a trustworthy test set, and how large does it need to be?](#1241-how-do-you-build-a-trustworthy-test-set-and-how-large-does-it-need-to-be)

---

### 12.1 How is A/B testing used to evaluate machine learning models?

An A/B test randomly splits live traffic between the current model (control) and a candidate (treatment) and compares business outcomes. It is the only reliable way to learn whether a better offline metric actually changes user or business behaviour, because offline evaluation on logged data cannot capture feedback loops, latency effects or how users react to new predictions.

**How it works**

| Step | What happens |
| --- | --- |
| **Hypothesis** | "The new ranker increases purchases per session by at least 1 percent" |
| **Randomisation unit** | Usually the user (hash of user ID and experiment salt), so each person sees one model consistently |
| **Traffic split** | Start small (1 to 5 percent), ramp up once guardrails are healthy |
| **Metrics** | One primary metric (the overall evaluation criterion), secondary metrics, and guardrails (latency, errors, revenue, complaints) |
| **Analysis** | Effect size with a confidence interval, not only a p-value; segment breakdowns |
| **Decision** | Ship, iterate or roll back, documented with the evidence |

**What is different when the treatment is a model**

- **Run it offline and in shadow first.** Shadow mode (the candidate scores live traffic but its outputs are not used) catches training-serving skew, latency and crashes before any user is affected (see 26.78).
- **The offline metric is only a filter.** A gain in AUC or NDCG often does not move the business metric; the A/B test is the decision (see 12.39).
- **Feedback loops.** The model changes the data it will be retrained on. Both arms share the same logging pipeline, so make sure the control's training data is not polluted by the treatment.
- **Latency is part of the treatment.** A more accurate but slower model can lose because of the extra 80 ms; measure latency as a guardrail.
- **Ranking problems can use interleaving**, which mixes results from both rankers in one list and needs far less traffic to detect a preference.

**Statistical hygiene**

- Power analysis before launch: minimum detectable effect, baseline variance, sample size and duration covering full weekly cycles.
- No peeking and stopping at the first significant result, unless you use sequential testing.
- Check sample ratio mismatch (the split is not the 50/50 you configured), which signals a bucketing or logging bug.
- Use CUPED or other variance reduction to shorten tests, and correct for multiple metrics.
- Watch for novelty and primacy effects, and for interference between users in marketplaces or social products (see 32.10).

**Why it matters:** real-world validation, a limited blast radius for a bad model, and a decision based on the business outcome rather than a proxy. The platform side of this (assignment service, metric pipelines, SRM checks, concurrency) is covered in 28.27.

**Pitfalls interviewers look for:** declaring victory on a "positive trend" that is not significant, choosing the primary metric after seeing the results, and running a test too short to see delayed outcomes such as churn or chargebacks.

### 12.2 What are precision, recall, specificity, F1 score and accuracy?

All five are computed from the four cells of the confusion matrix (see 12.3), and each answers a different question about a binary classifier at a fixed threshold.

|  | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | True positive (TP) | False negative (FN) |
| **Actual negative** | False positive (FP) | True negative (TN) |

**Accuracy**

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

- The share of all predictions that are correct. Simple, but misleading with imbalance: if 95 percent of emails are not spam, "always not spam" scores 95 percent and catches nothing.

**Precision (positive predictive value)**

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

- "When the model says yes, how often is it right?" High precision means few false alarms.
- Matters when false positives are costly: a spam filter hiding real mail, a fraud model blocking good customers.

**Recall (sensitivity, true positive rate)**

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

- "Of all real positives, how many did we catch?" If there are 100 actual positives and the model finds 90, recall is 90 percent.
- Matters when misses are costly: cancer screening, fraud, safety incidents.

**Specificity (true negative rate)**

$$
\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}
$$

- "Of all real negatives, how many did we correctly leave alone?" It equals one minus the false positive rate, so it is the x-axis of the ROC curve read backwards. Common in medicine, paired with sensitivity.

**F1 score and F-beta**

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \qquad F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

- The harmonic mean is pulled towards the lower value: 100 percent precision with 1 percent recall gives an F1 of about 2 percent, not 50 percent.
- $F_\beta$ weights recall $\beta$ times as much as precision: F2 for screening, F0.5 when false alarms hurt more.
- Limitations: F1 ignores true negatives, depends on the threshold and the base rate, and assumes precision and recall matter equally, which is rarely true in business. MCC is a more balanced single number when both classes matter.

**The precision-recall trade-off**

- Lowering the threshold catches more positives (recall up) but lets in more false alarms (precision down); raising it does the opposite.
- Where to sit on the curve is a business decision driven by the cost of each error type (see 12.21).

**Quick reference**

| Metric | Answers | Use when |
| --- | --- | --- |
| Accuracy | How often is the model right overall? | Classes balanced, errors equally costly |
| Precision | When it says positive, is it right? | False positives are costly |
| Recall | Did it catch the real positives? | False negatives are costly |
| Specificity | Did it leave real negatives alone? | Medical tests, paired with sensitivity |
| F1 / F-beta | One number balancing both | Imbalanced data, single threshold |

```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
```

**For LLM systems:** these metrics apply directly to classification-style tasks (moderation, intent detection, extraction). Open-ended generation needs other methods (see Chapter 24).

### 12.3 What is the confusion matrix, why do we need it, and how do you interpret it?

A confusion matrix is a table of predicted classes against actual classes. We need it because a single score such as accuracy hides *which* mistakes the model makes, and the type of mistake usually matters more than the count. Every threshold metric (accuracy, precision, recall, specificity, F1) is computed from it. (Chapter 8 covers the basics in 8.21.)

**Binary case**

|  | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actual positive** | TP | FN (Type II error, a miss) |
| **Actual negative** | FP (Type I error, a false alarm) | TN |

Example for a spam filter: TP is spam caught, TN is normal mail delivered, FP is a real email sent to spam, FN is spam that reached the inbox.

**How to interpret it**

- **Diagonal** cells are correct predictions; **off-diagonal** cells are errors, and their split tells you the error profile.
- **Read it in rows and columns.** Normalising by row gives per-class recall (what fraction of each true class was caught); normalising by column gives per-class precision.
- **Look at absolute counts as well as rates.** 0.5 percent false positives on 10 million transactions is 50,000 customers wrongly blocked a day, which may break an operations team even if the rate looks small.
- **Compare against the base rate.** With 1 percent positives, a matrix dominated by TN can still hide a model that misses most positives.

**Multi-class case**

- An $N \times N$ grid. Large off-diagonal cells show which specific classes are confused (cat against fox, "billing" against "refund" intents).
- Those pairs drive action: merge classes, collect more examples, add a feature that separates them, or fix inconsistent labels.

**Turning it into a decision**

- Multiply the matrix element-wise by a **cost matrix** (cost of each FP and FN) to get the expected cost of a model at a given threshold. This is how you compare models when errors have different prices (see 12.21).
- The matrix exists only at one threshold. Changing the threshold moves counts between cells, which is what ROC and PR curves summarise.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred, normalize="true")  # rows sum to 1: per-class recall
ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
```

**Production angle:** compute the matrix per segment (region, device, customer tier) as well as globally; aggregate matrices often hide a segment where the model fails (see 12.37).

### 12.4 What are the common evaluation metrics for classification, and when would you prefer each?

Classification metrics fall into three families: threshold-based metrics that judge hard decisions, ranking metrics that judge how well scores separate the classes, and probability metrics that judge whether the scores can be trusted as probabilities. A senior answer picks from each family deliberately.

**Threshold-based (from the confusion matrix, see 12.2 and 12.3)**

- **Accuracy** = $(\text{TP} + \text{TN}) / \text{all}$. Misleading with class imbalance.
- **Precision** = $\text{TP} / (\text{TP} + \text{FP})$: use when false positives are costly.
- **Recall (sensitivity)** = $\text{TP} / (\text{TP} + \text{FN})$: use when misses are costly.
- **Specificity** = $\text{TN} / (\text{TN} + \text{FP})$.
- **F1 and F-beta**: harmonic mean of precision and recall; F-beta weights recall $\beta$ times as much.
- **MCC (Matthews correlation coefficient)**: uses all four cells and stays honest under imbalance.
- **Cohen's kappa**: agreement corrected for chance.

**Ranking quality (threshold-free)**

- **ROC-AUC**: probability a random positive is ranked above a random negative (see 12.9). Can look optimistic with heavy imbalance.
- **PR-AUC / average precision**: focuses on the positive class; preferred for rare events (see 12.26).

**Probability quality**

- **Log loss (cross-entropy)**: punishes confident wrong predictions (see 12.6).
- **Brier score** and **expected calibration error**: whether a score of 0.8 really means 80 percent (see 12.36).

**Which to prefer**

| Metric | Prefer it when | Main idea |
| --- | --- | --- |
| **Accuracy** | Classes balanced and errors equally costly | Share correct |
| **F1 / F-beta** | You need a binary decision and care about both precision and recall | Harmonic mean at one threshold |
| **ROC-AUC** | You care about ranking quality across thresholds, classes reasonably balanced | Positive ranked above negative |
| **PR-AUC** | The positive class is rare and finding positives matters | Precision-recall trade-off |
| **Log loss** | You care about probability quality, not just labels | Penalises confident errors |
| **Calibration error** | Probabilities feed decisions (pricing, risk, expected value) | Predicted against observed frequency |

**Multi-class:** macro averaging (each class equal), micro (each sample equal), weighted by support; plus top-k accuracy and per-class confusion matrices (see 12.7).

**Production angle:** choose the metric that maps to business cost, then choose the operating threshold on a validation set (for example, maximise recall subject to precision of at least 95 percent). Report one ranking metric for model selection, the threshold metrics at the operating point, and calibration if scores are consumed as probabilities. Track all of them per segment, not only globally.

### 12.5 When would you use accuracy vs other metrics?

##### Accuracy vs. Other Metrics

**Use Accuracy when:**

- Classes are **balanced** (roughly equal positive/negative counts)
- All error types (false positives vs. false negatives) have **similar cost**
- Simple overall correctness is what matters

**Avoid Accuracy when:**

- Data is **imbalanced** — e.g., 95% negative class → a model predicting all-negative gets 95% accuracy while being useless
- Different error types have **different costs** — accuracy treats FP and FN as equally bad

**Use Precision when:**

- **False positives are costly** — e.g., flagging legitimate emails as spam, wrongly accusing someone of fraud

**Use Recall when:**

- **False negatives are costly** — e.g., missing a cancer diagnosis, missing a fraud case

**Use F1 when:**

- Need a **single balanced metric** on imbalanced data, and both FP and FN matter

**Use ROC-AUC / PR-AUC when:**

- Need to evaluate performance **across all thresholds**, not just one fixed cutoff
- PR-AUC specifically preferred over ROC-AUC on **highly imbalanced** data

**Rule of thumb:** balanced data + equal error cost → accuracy; imbalanced data or unequal error cost → precision/recall/F1 depending on which error type matters more.

### 12.6 When would you use log loss vs accuracy?

##### Log Loss vs. Accuracy

**Use Log Loss when:**

- You care about **predicted probabilities**, not just the final class label — how confident the model is matters
- You want to **penalize confident wrong predictions** heavily (e.g., predicting 99% positive when actual is negative gets punished much more than predicting 51%)
- Model outputs are used **downstream** for ranking, thresholding, or risk scoring, where calibration matters
- Comparing models with similar accuracy but different confidence behavior — log loss differentiates a well-calibrated model from an overconfident one

**Use Accuracy when:**

- Only the **final hard prediction** matters, not the confidence behind it
- Stakeholders need a simple, interpretable "how often is it right" number
- Classes are balanced and probability calibration isn't a concern

**Key difference:**

- Accuracy only looks at whether `argmax(prediction) == label` — throws away probability information
- Log loss uses the full predicted probability, so two models with identical accuracy can have very different log loss if one is poorly calibrated (overconfident/underconfident)

**Rule of thumb:** accuracy for simple correctness checks; log loss when probability quality/calibration matters (e.g., risk models, ranking systems, model comparison during training).

### 12.7 What metrics would you use for a multi-class classification problem?

##### Metrics for Multi-Class Classification

**Core metrics (extended from binary):**

- **Accuracy**: overall fraction correct across all classes
- **Per-class Precision/Recall/F1**: computed separately for each class (one-vs-rest)

**Averaging methods for aggregating per-class scores:**

- **Macro-average**: simple average across classes — treats all classes equally, regardless of size (good for imbalanced classes when all matter equally)
- **Micro-average**: aggregates TP/FP/FN across all classes first, then computes metric — dominated by larger classes
- **Weighted average**: averages per-class scores weighted by class support (size) — accounts for imbalance while still reflecting overall performance

**Other useful metrics:**

- **Confusion Matrix ($N \times N$)**: shows exactly which classes get confused with which others
- **Cohen's Kappa**: agreement between predicted/actual, adjusted for chance
- **Matthews Correlation Coefficient (multi-class version)**: balanced single-score metric robust to imbalance
- **Top-k Accuracy**: correct if true label is among the top-k predicted classes (common in NLP/vision with many classes)

**Rule of thumb:**

- Balanced classes → macro-average or plain accuracy
- Imbalanced classes → weighted average, or macro if minority classes are important
- Need to see specific class confusions → confusion matrix

### 12.8 How do you handle class imbalance, in both the evaluation and the model?

Imbalance is first an evaluation problem and only second a modelling problem: with the right metric and threshold, many "imbalance" issues disappear. Fix the measurement, then the decision threshold, and only then change the data or the loss.

**1. Evaluate with imbalance-aware metrics**

- Drop plain accuracy as the headline: 99.9 percent accuracy is free when 0.1 percent of cases are fraud.
- Use precision, recall and F-beta at the operating point, **PR-AUC** rather than ROC-AUC (see 12.26), and **MCC** as a balanced single number.
- For multi-class, report **per-class** metrics and macro averages so minority classes are not drowned out.
- Use **stratified** splits (see 12.28) so every fold has enough positives, and report confidence intervals: with 50 positives in a test set, recall moves 2 percentage points per example.

**2. Move the threshold before touching the data**

- Most models rank positives well but the default 0.5 threshold is wrong for a 1 percent base rate. Choosing the threshold from costs or capacity (see 12.21) is cheap and often enough.

**3. Algorithm-level fixes**

- **Class weights or cost-sensitive loss**: `class_weight="balanced"`, `scale_pos_weight` in XGBoost, weighted cross-entropy, focal loss for dense detection.
- **Ensembles built for imbalance**: Balanced Random Forest, EasyEnsemble.
- **Anomaly detection framing** when positives are extremely rare or keep changing.

**4. Data-level fixes**

- Random undersampling of the majority (fast, can lose information), oversampling or SMOTE for the minority, augmentation for images and text, and collecting more positives (often the best fix).
- Apply resampling **only inside the training folds**, never before the split, or synthetic points leak into validation.

**The calibration trap**

Resampling and class weights change the class prior the model sees, so predicted probabilities are inflated. If the scores are used as probabilities, recalibrate on untouched validation data (see 12.36) or apply the prior correction. With undersampling rate $\beta$ for negatives, a score $p_s$ maps back to

$$
p = \frac{\beta p_s}{\beta p_s - p_s + 1}
$$

**Rule of thumb:** metrics alone do not fix imbalance, they reveal it. Combine imbalance-aware evaluation with the cheapest effective fix (threshold, then weights, then resampling). Chapter 10 covers resampling methods in detail (see 10.26 to 10.28), and 12.35 covers the model that predicts only the majority class.

### 12.9 What is the ROC curve and AUC, what are their benefits, and what are their limitations?

The **ROC curve** plots the true positive rate (recall) against the false positive rate as the decision threshold sweeps from 1 to 0. **AUC** is the area under it and equals the probability that a randomly chosen positive is scored higher than a randomly chosen negative (the Mann-Whitney U statistic, normalised).

$$
\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} \qquad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{specificity}
$$

**Reading it**

- AUC of 1.0 is perfect separation, 0.5 is random, and below 0.5 means the scores are inverted (often a label bug).
- A curve hugging the top-left corner is better; the diagonal is a random classifier.
- Rough guide sometimes quoted: 0.7 to 0.8 fair, 0.8 to 0.9 good, above 0.9 excellent. These bands depend heavily on the domain; an AUC of 0.99 on a hard problem is more often leakage than brilliance (see 12.33).

**Benefits**

- **Threshold-independent:** summarises ranking quality before the business has chosen an operating point.
- **Scale-invariant:** depends only on the order of scores, so it compares models with different score ranges.
- **Prior-insensitive:** TPR and FPR are each computed within one class, so AUC is comparable across datasets with different base rates.
- **Useful for choosing a threshold:** the curve shows the trade-off (Youden's J maximises TPR minus FPR, or pick the cost-weighted point, see 12.21).

**Limitations**

- **Optimistic under heavy imbalance.** The FPR denominator (all negatives) is huge, so thousands of false positives barely move it. A fraud model can show 0.98 AUC while precision at a usable threshold is 5 percent. Use PR-AUC there (see 12.26).
- **Ranking only, not calibration.** Good AUC says nothing about whether 0.8 means 80 percent; check log loss, Brier score or calibration error (see 12.36).
- **Averages over thresholds you will never use.** Production runs at one point, often FPR below 1 percent. Two models with equal AUC can differ greatly there, so use partial AUC or TPR at a fixed FPR.
- **Ignores costs.** It treats FP and FN as equally important.
- **Binary by nature.** Multi-class needs one-vs-rest or one-vs-one averaging, which can hide weak classes.
- **Unstable with few positives.** Bootstrap a confidence interval (see 12.12) and check AUC per segment.

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, y_score)
fpr, tpr, thr = roc_curve(y_true, y_score)
tpr_at_1pct_fpr = tpr[fpr <= 0.01].max()
partial = roc_auc_score(y_true, y_score, max_fpr=0.01)  # standardised partial AUC
```

**Production angle:** report ROC-AUC for model comparison, but choose the threshold and monitor with business-aligned metrics (precision and recall at the operating point, cost per decision) plus calibration plots.

### 12.10 What are the common regression metrics, and how do MAE, MSE and RMSE differ?

Regression metrics differ mainly in how they weight large errors and whether they are scale-free. Pick the one whose penalty matches the real cost of being wrong, and always report it next to a naive baseline.

| Metric | Formula (idea) | Notes |
| --- | --- | --- |
| **MAE** | mean of $\lvert y - \hat{y} \rvert$ | Same units as the target, robust to outliers; minimised by the median |
| **MSE** | mean of $(y - \hat{y})^2$ | Squared units; punishes large errors heavily; smooth, so used as a training loss; minimised by the mean |
| **RMSE** | $\sqrt{\text{MSE}}$ | Back in target units, still outlier-sensitive; always at least as large as MAE |
| **$R^2$** | $1 - SS_{\text{res}} / SS_{\text{tot}}$ | Share of variance explained against predicting the mean; can be negative on test data |
| **Adjusted $R^2$** | penalises extra features | Comparing linear models with different feature counts (see 12.25) |
| **MAPE** | mean of $\lvert (y - \hat{y}) / y \rvert$ | Scale-free and business-friendly, but explodes near zero and penalises over-forecasts more |
| **sMAPE, WAPE** | normalised variants | WAPE (total absolute error over total actuals) is the safer default for demand |
| **RMSLE** | RMSE on $\log(1 + y)$ | Relative error; for targets spanning orders of magnitude (prices, counts) |
| **Huber, quantile (pinball) loss** | hybrid and asymmetric | Robust fitting; quantile loss for prediction intervals and asymmetric costs |

**MAE against MSE and RMSE**

- **Linear against quadratic penalty.** Errors of 1, 1, 1 and 10 give MAE 3.25 but RMSE about 5.1: the single big miss dominates RMSE.
- **What they estimate.** A model that minimises MAE predicts the conditional median; one that minimises MSE predicts the conditional mean. With skewed targets (spend, claim size) these differ a lot, so the choice of metric changes the model you get.
- **Units.** MAE and RMSE are in target units and easy to explain; MSE is in squared units and mainly a loss.
- **Gap as a diagnostic.** RMSE much larger than MAE means a few large errors; inspect them before trusting either number.

**How to choose**

- Being off by 10 costs the same whatever the case: **MAE**.
- Big misses are disproportionately bad (stock-outs, safety margins): **RMSE**.
- Relative error matters across items of very different scale: **WAPE** or **RMSLE**, not MAPE when actuals can be zero.
- Over- and under-prediction have different costs: **quantile loss** at the right quantile.
- Explanatory power for a linear model: $R^2$, with adjusted $R^2$ for feature-count comparisons.

**Practical guidance:** report a baseline (predict the mean, the last value or the seasonal naive forecast) alongside every metric so the number has context, and look at residual plots by segment. Forecasting metrics such as MASE and backtesting are covered in 32.30.

### 12.11 How do you choose the right evaluation metric for a given problem?

Start from what a wrong prediction costs in the real use case, then pick an offline metric that is a faithful proxy for that cost, and confirm the link online. The metric is a product decision as much as a technical one, so agree it with stakeholders before modelling.

**1. Identify the problem type**

- Classification, regression, ranking, forecasting or generation each have their own metric families (see 12.4, 12.10, 12.17, Chapter 24).

**2. Understand the cost of each error**

- False positives costlier: prioritise **precision**. False negatives costlier: prioritise **recall**.
- Large errors much worse than small ones: **RMSE**. All errors equally bad: **MAE**.
- Asymmetric over- and under-prediction: **quantile loss**.

**3. Check the data**

- Imbalanced classes: precision, recall, PR-AUC, MCC rather than accuracy.
- Targets near zero or spanning orders of magnitude: avoid MAPE; use WAPE or RMSLE.

**4. Decide whether probabilities or only decisions matter**

- Scores used as probabilities (risk, pricing, expected value): **log loss**, **Brier score**, calibration error.
- Only the final decision matters: metrics at a fixed threshold.
- Only the order matters (ranking, triage queues): **ROC-AUC**, **PR-AUC**, **NDCG@k**, **precision@k**.

**5. Separate three kinds of metric**

| Role | Example | Why |
| --- | --- | --- |
| **Optimisation (training loss)** | Log loss, MSE | Differentiable and stable |
| **Selection (offline)** | PR-AUC, recall at 95 percent precision, NDCG@10 | Mirrors the decision the model supports |
| **Business (online)** | Fraud losses prevented, tickets deflected, revenue per session | What the organisation actually values |

**6. Use constraint-style metrics for hard requirements**

- "Maximise recall subject to precision of at least 95 percent", or "maximise precision at a review capacity of 500 cases a day" (precision@k).
- With an explicit cost matrix, choose the threshold that minimises expected cost, not the default 0.5 (see 12.21).

$$
\text{Expected cost} = C_{\text{FP}} \cdot \text{FP} + C_{\text{FN}} \cdot \text{FN}
$$

**7. Add guardrail metrics**

- Latency, inference cost, calibration, and performance on critical segments or protected groups (see 12.37). A model that wins the primary metric but fails a guardrail does not ship.

**8. Validate the offline-online link**

- Confirm with an A/B test that gains in the offline metric move the business KPI. If they do not, the offline metric is the wrong proxy (see 12.39).

**Pitfalls:** optimising a metric nobody downstream cares about, changing the metric after seeing results, and using one aggregate number that hides a failing segment. **Rule of thumb:** one primary metric, a few guardrails, all agreed in writing before the first experiment.

### 12.12 How do you compare two models and decide whether one is really better?

Compare them on the same data with the same protocol, measure the uncertainty of the difference, and weigh the gain against cost, latency and risk. A 0.3 point improvement on one test split is usually noise, and senior interviewers want to hear how you would tell.

**1. Make the comparison fair**

- Same train, validation and test splits, same preprocessing, same metric code, and a comparable tuning budget for each model (otherwise you compare tuning effort, not models).
- Several random seeds for models with stochastic training; report mean and spread.

**2. Use paired statistics on the same test examples**

Because both models score the same examples, compare them example by example. This removes the variation between examples and gives much more power than comparing two separate confidence intervals.

| Situation | Method |
| --- | --- |
| Two classifiers, one test set | **McNemar's test** on the disagreements: cases A gets right and B wrong ($b$) against the reverse ($c$) |
| Any metric, including AUC, F1, NDCG | **Paired bootstrap**: resample test examples with replacement, recompute both metrics, take the percentile interval of the difference |
| Per-example losses or scores | Paired t-test, or Wilcoxon signed-rank if not normal |
| Cross-validation scores | **Corrected resampled t-test** (Nadeau and Bengio) or 5x2 CV test; a naive t-test on fold scores is overconfident because folds share training data |
| AUC specifically | DeLong's test |
| Many models on many datasets | Friedman test with a post-hoc test |

$$
\chi^2 = \frac{(\lvert b - c \rvert - 1)^2}{b + c}
$$

**3. Paired bootstrap in practice**

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def paired_bootstrap(y, s_a, s_b, metric=roc_auc_score, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    diffs = []
    for _ in range(n):
        idx = rng.choice(idx_all, size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], s_b[idx]) - metric(y[idx], s_a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return np.mean(diffs), (lo, hi)  # B better if the interval excludes 0
```

If the data has groups (users, sessions, documents), resample whole groups, not rows, or the interval will be too narrow.

**4. Look beyond the aggregate**

- **Slices:** a better average can hide a regression on a key segment (see 12.37).
- **Calibration:** equal AUC can hide very different probability quality (see 12.36).
- **Robustness:** performance on out-of-time and stress test sets (see 12.38).
- **Cost:** latency, memory, training and serving cost, and maintenance burden. A model that is 0.2 points better but three times slower rarely wins.

**5. Decide on practical significance**

- Agree in advance the minimum improvement worth shipping. Statistically significant but tiny gains are not worth the complexity; large but uncertain gains need more data or an A/B test (see 12.1).
- If you compare many variants, correct for multiple comparisons or confirm the winner on a fresh holdout. LLM and prompt comparisons follow the same logic (see 24.19).

### 12.13 What is cross-validation, what types are there, and why does it matter?

Cross-validation estimates how well a model generalises by repeatedly splitting the data into training and validation parts, training on one and scoring on the other, and averaging. It gives a more reliable estimate than a single holdout split, uses scarce data efficiently, and is the standard tool for model selection and hyperparameter tuning.

**k-fold cross-validation**

1. Split the data into $k$ folds of roughly equal size (5 or 10 is typical).
2. Train on $k - 1$ folds and validate on the remaining one.
3. Repeat $k$ times so every fold is validated exactly once.
4. Report the mean **and** the standard deviation of the fold scores.

| Iteration | Training data | Validation data |
| --- | --- | --- |
| 1 | Folds 2 to 5 | Fold 1 |
| 2 | Folds 1, 3 to 5 | Fold 2 |
| 3 | Folds 1, 2, 4, 5 | Fold 3 |
| 4 | Folds 1 to 3, 5 | Fold 4 |
| 5 | Folds 1 to 4 | Fold 5 |

**Types and when to use them**

| Scheme | Use when |
| --- | --- |
| **Holdout** | Large data, or very expensive training; fast but noisy |
| **k-fold** | Default for independent rows |
| **Stratified k-fold** | Classification, especially imbalanced (see 12.28) |
| **Repeated k-fold** | Small data where the estimate is unstable (see 12.34) |
| **Leave-one-out** | Tiny datasets; nearly unbiased but high variance and expensive |
| **Group k-fold / leave-one-group-out** | Several rows per user, patient or document, so the same entity never sits on both sides |
| **Time-series split** | Temporal data; train on the past, validate on the future (see 12.23) |
| **Nested CV** | You tune hyperparameters and also need an unbiased performance estimate (see 12.14) |
| **Monte Carlo (shuffle-split)** | Many random train and validation splits of a chosen size |

**Why it matters**

- **Lower-variance estimate:** a single split can be lucky or unlucky; averaging over folds reduces that.
- **Efficient use of data:** every row is used for both training and validation.
- **Detects overfitting and instability:** a large train-CV gap or a large spread across folds is a warning.
- **Fair model comparison:** compare CV scores with paired methods (see 12.12).

**Rules that make it valid**

- Fit every preprocessing step (scaling, imputation, target encoding, feature selection, resampling) **inside** each fold by using a `Pipeline`; fitting them on the full data leaks validation information.
- Choose the split that mirrors how the model meets new data in production: new users need group splits, future periods need time splits.
- Keep a final untouched test set for the one-off estimate you report.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())
```

**Cost:** $k$ times the training cost. For large deep learning models a single well-designed holdout (or a time-based one) is usually used instead.

### 12.14 How do you tune hyperparameters without overfitting the validation set?

Every time you look at a validation score and make a decision, the validation set leaks a little into the model. With hundreds of tuning trials the best validation score becomes an optimistic estimate, so you need a protocol that keeps one estimate honest. The search methods themselves (grid, random, Bayesian, Hyperband) are covered in 11.4 and 11.5; this answer is about evaluating them correctly.

**Why it happens**

- Picking the maximum of many noisy scores selects configurations that were lucky on this particular validation sample (the winner's curse).
- The more trials and the smaller the validation set, the larger the optimism. On a 500-example validation set, a 1 point accuracy gain is about one standard error, so it is easily noise.

**Protocol**

1. **Three-way split.** Train for fitting, validation (or inner CV) for tuning and early stopping, test touched once for the final estimate.
2. **Nested cross-validation** when data is small: an inner loop tunes hyperparameters, an outer loop scores the whole tuning procedure. The outer score estimates "how good is my model-building process", not one configuration.
3. **Use CV, not a single split, for tuning** on small and medium data, so each trial is scored on more examples.
4. **Prefer simpler winners.** If several configurations are within one standard error of the best, pick the simplest or most regularised one.
5. **Limit the search to what matters.** A focused space with sensible priors (log-uniform learning rate) needs fewer trials and overfits less.
6. **Refresh the test set** if it has been used for many decisions over months; a test set reused for every release slowly becomes a validation set.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)
search = GridSearchCV(pipe, param_grid, cv=inner, scoring="average_precision")
nested_scores = cross_val_score(search, X, y, cv=outer, scoring="average_precision")
```

**Other leaks during tuning**

- Early stopping on the test set (common in deep learning code).
- Choosing the feature set, the threshold or the random seed by looking at test results.
- Preprocessing fitted outside the CV loop, which makes every trial look better than it is.

**What interviewers look for:** the distinction between tuning data and estimation data, awareness that more trials means more optimism, and a concrete habit (nested CV, a locked test set, a one-standard-error rule). Chapter 11 covers how to run the search efficiently (see 11.4 to 11.7).

### 12.15 How do you evaluate unsupervised learning models?

Without labels there is no single correct answer, so evaluate an unsupervised model against the purpose it serves: structure metrics tell you whether the output is internally coherent, but downstream usefulness and stability tell you whether it is worth having.

**Four lenses**

| Lens | What it checks | Examples |
| --- | --- | --- |
| **Internal metrics** | Compactness and separation, computed from the data alone | Silhouette, Davies-Bouldin, Calinski-Harabasz, reconstruction error |
| **External metrics** | Agreement with labels, where a labelled sample exists | ARI, NMI, purity; precision and recall for anomalies |
| **Downstream task** | Does the output improve something that has labels? | Segments that lift campaign conversion; embeddings that improve a classifier |
| **Stability** | Does the result survive resampling, seeds and time? | ARI between bootstrap runs, drift in segment sizes month to month |

**By task**

- **Clustering:** internal and external indices, choice of $k$, stability (see 12.16).
- **Dimensionality reduction:** explained variance, reconstruction error, neighbourhood preservation and downstream accuracy (see 12.22).
- **Anomaly detection:** labels are rare, so use a small labelled set of known incidents, precision@k of the top alerts reviewed by experts, and the alert volume the operations team can absorb. Synthetic injected anomalies help, but real anomalies are rarely as clean.
- **Embeddings and representation learning:** linear probes (train a simple classifier on top), retrieval metrics such as recall@k on known similar pairs, and nearest-neighbour inspection.
- **Topic models:** topic coherence plus human review of top words and documents.
- **Generative models:** likelihood where available, sample quality metrics (FID for images), and human judgement (see Chapter 24 for LLMs).

**Practical approach**

1. Write down what decision the output supports (marketing segments, alert triage, search).
2. Build a small labelled or expert-reviewed evaluation set for that decision, even 200 examples.
3. Use internal metrics only to compare configurations of the same method, because they encode assumptions (convex clusters, Euclidean distance) that may not match the data.
4. Check stability and run a domain expert review of samples from each output group.

**Pitfall:** optimising silhouette or reconstruction error for its own sake. A clustering with a high silhouette score that marketing cannot act on has failed.

### 12.16 How do you evaluate a clustering algorithm?

Clustering has no single accuracy, because there is usually no ground truth. Combine internal metrics (structure), external metrics (where some labels exist), stability, and above all a downstream or expert check of whether the clusters are useful.

**Internal metrics (no labels)**

- **Silhouette score** (from $-1$ to $1$, higher is better). For a point with mean intra-cluster distance $a$ and mean distance to the nearest other cluster $b$:

$$
s = \frac{b - a}{\max(a, b)}
$$

- **Davies-Bouldin index:** average similarity between each cluster and its most similar neighbour; lower is better.
- **Calinski-Harabasz index:** between-cluster dispersion over within-cluster dispersion; higher is better.
- **Inertia (within-cluster sum of squares):** always falls as $k$ grows, so it is only useful comparatively (elbow method).

**External metrics (a labelled sample exists)**

- **Adjusted Rand Index (ARI)** and **Normalised Mutual Information (NMI):** agreement with true labels, invariant to label permutation; ARI is adjusted for chance.
- **Homogeneity, completeness and V-measure:** whether each cluster holds one class, and whether each class stays in one cluster. Purity is simple but rewards many small clusters.

**Choosing $k$**

- Elbow on inertia, silhouette analysis across $k$, the gap statistic, BIC for Gaussian mixtures, or simply the number of segments the business can act on.

**Stability**

- Re-run on bootstrap samples, different seeds and different time windows, and measure ARI between runs. Clusters that reshuffle with each run are not real structure.

**Caveats**

- Internal metrics favour convex, similar-sized, spherical clusters, so they penalise valid DBSCAN or HDBSCAN results with odd shapes and noise points.
- All scores depend on the distance metric and feature scaling; a change in scaling can change the "best" $k$.

**Production angle:** the real test is downstream. Do the customer segments behave differently? Do they lift campaign conversion? Can a domain expert describe each cluster in one sentence after reviewing samples? Visualise with PCA or UMAP for intuition, but do not read cluster sizes or distances from t-SNE plots.

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
silhouette_score(X, labels)          # internal
davies_bouldin_score(X, labels)      # internal
adjusted_rand_score(y_true, labels)  # external, or between two runs for stability
```

### 12.17 How do you evaluate a recommendation system?

Evaluate a recommender at three levels: offline ranking metrics on historical data, beyond-accuracy properties of the lists, and online business impact. Offline metrics are a filter; the online test is the decision.

**Offline accuracy (time-based split, never random)**

- **Rating prediction:** RMSE and MAE. Largely outdated, because users see rankings, not predicted ratings.
- **Top-K ranking:**
  - **Precision@K** and **Recall@K**: share of the top K that is relevant, and share of relevant items that reach the top K.
  - **Hit rate@K:** whether at least one relevant item appears.
  - **MRR:** reciprocal rank of the first relevant item; suits "one right answer" cases.
  - **MAP:** average precision across relevant items, averaged over users.
  - **NDCG@K:** rewards placing relevant items higher, supports graded relevance; the usual headline metric.

**Beyond accuracy**

- **Coverage:** share of the catalogue ever recommended.
- **Diversity:** dissimilarity within a list. **Novelty and serendipity:** non-obvious but liked items.
- **Popularity bias and fairness to providers:** whether long-tail items and smaller sellers get exposure (see 32.20).

**Online (the real test)**

- A/B tests on CTR, conversion, watch time, add-to-cart, revenue per session and retention; interleaving for faster, more sensitive ranking comparisons.
- Guardrails: latency, complaints, unsubscribes, content-quality signals.

**Pitfalls**

- **Random splits leak the future;** split by time and evaluate on the next period.
- **Logged data is biased by the previous recommender** (exposure and position bias). Items never shown cannot be clicked, so offline metrics favour models similar to the old one. Counterfactual estimators such as inverse propensity scoring help (see 32.19).
- **Sampled metrics:** ranking the true item against 100 random negatives can change which model looks best; prefer full ranking or report both.
- **Optimising CTR alone** promotes clickbait; balance it with long-term engagement (see 32.24).
- Offline and online results often disagree; 32.21 explains why and how to narrow the gap.

```python
import numpy as np

def ndcg_at_k(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    discounts = np.log2(np.arange(2, rels.size + 2))
    dcg = (rels / discounts).sum()
    ideal = np.sort(rels)[::-1]
    idcg = (ideal / discounts[: ideal.size]).sum()
    return dcg / idcg if idcg else 0.0
```

### 12.18 What is LLM-as-a-judge?

Chapter 24 covers this in more depth, including judge calibration and bias mitigation (see 24.7).

##### LLM as a Judge

- Using a strong LLM to evaluate the outputs of another model (or itself) — grading, ranking, or comparing responses — instead of relying solely on human evaluators or traditional automated metrics

##### Why It's Used

- **Traditional metrics fall short for open-ended generation**: BLEU/ROUGE only measure surface-level word overlap, missing semantic quality, coherence, helpfulness, or correctness
- **Human evaluation is expensive and slow**: doesn't scale to evaluating thousands of model outputs quickly or repeatedly during development
- **LLMs can approximate human judgment reasonably well** on many quality dimensions (helpfulness, coherence, correctness, tone) when given clear evaluation criteria

##### Common Approaches

**1. Pointwise scoring**

- Judge LLM rates a single response on a scale (e.g., 1–10) against defined criteria (accuracy, helpfulness, safety)

**2. Pairwise comparison**

- Judge LLM is shown two responses to the same prompt and asked which is better (or a tie) — often more reliable than absolute scoring, since relative judgments tend to be more consistent

**3. Reference-based grading**

- Judge compares a model's response against a "gold" reference answer, checking for factual alignment or key content coverage

**4. Rubric-based evaluation**

- Judge is given an explicit checklist/rubric to score against, improving consistency and reducing ambiguity

##### Benefits

- **Scalable and fast**: can evaluate large volumes of outputs automatically
- **Flexible**: can assess nuanced qualities (tone, reasoning quality, instruction-following) that hard-coded metrics can't capture
- **Useful for RLHF/DPO-style training**: can generate preference labels at scale, reducing reliance on costly human annotation (related to RLAIF)

##### Limitations and Risks

- **Self-preference bias**: judge models tend to favor outputs that resemble their own style/patterns, including favoring outputs from the same model family
- **Position bias**: in pairwise comparisons, judges can be biased toward whichever response appears first/second
- **Verbosity bias**: judges often favor longer responses, even when not actually better
- **Not perfectly aligned with human judgment**: correlates well in many cases, but can diverge on nuanced or subjective quality dimensions
- **Susceptible to prompt sensitivity**: judgment quality depends heavily on how the evaluation prompt/rubric is written

##### Mitigations

- Randomize response order in pairwise comparisons to reduce position bias
- Use multiple judge models or ensemble judgments
- Calibrate against a sample of human-labeled data to check judge reliability
- Use detailed, explicit rubrics rather than vague scoring instructions

### 12.19 Describe your approach to debugging an underperforming ML model.

Work systematically, from the cheapest and most likely causes to the most expensive.

1. **Check the problem definition and metric.** Is the metric aligned with the business goal? Is the baseline sensible? Is the target right?
2. **Compare against baselines:** majority class, simple heuristics, logistic regression. If a complex model barely beats them, suspect the data or the features.
3. **Diagnose bias against variance with learning curves:**
   - *High train error* (underfitting): more capacity, better features, less regularisation, longer training.
   - *Low train error, high validation error* (overfitting): more data, regularisation, augmentation, simpler model, early stopping.
4. **Inspect the data:** label quality (re-label a sample), leakage, train/validation/test distribution mismatch, duplicates across splits, class imbalance, missing values, wrong joins, feature bugs. Most problems live here.
5. **Error analysis:** slice the errors by segment (region, device, class, input length). Look at the worst-scored examples manually and group failure modes. This shows whether you need more data for a slice, a new feature, or label fixes.
6. **Sanity checks on the pipeline:** overfit a tiny batch (if you can't, there's a bug), check that preprocessing matches between training and inference, shuffle labels (performance should drop to chance), check loss curves, the learning rate and gradient norms.
7. **Feature analysis:** importance and SHAP to spot leakage or features that do nothing; add domain features.
8. **Model and hyperparameters:** systematic tuning (Optuna), trying other model families or ensembles. Do this *after* the data checks.
9. **If it's production-only underperformance:** training/serving skew, drift, upstream pipeline changes, or feedback loops. Compare logged serving features with offline features.

Document each hypothesis and its result. Fix the largest error bucket first.

### 12.20 How is a logistic regression model evaluated?

Evaluate logistic regression on three fronts: how well it discriminates (classification and ranking metrics), how good its probabilities are (calibration and log loss, which is its natural strength), and whether the fitted model itself is sound (coefficients, fit statistics and assumptions). The third part is what distinguishes logistic regression from a black-box classifier.

**1. Discrimination**

- At a chosen threshold: accuracy, precision, recall, specificity and F1 from the confusion matrix (see 12.2 and 12.3). The threshold should come from business costs, not default to 0.5 (see 12.21).
- Across thresholds: ROC-AUC, which for a binary model equals the concordance statistic (c-statistic), and PR-AUC when positives are rare (see 12.9 and 12.26).

**2. Probability quality**

- **Log loss** is exactly the negative log-likelihood the model was fitted to minimise, so it is the most direct measure of fit on held-out data.
- **Brier score** and a **reliability diagram** (see 12.36). Logistic regression is usually well calibrated on data like its training data, but class weights, resampling or regularisation strength can break that.

**3. Model fit and inference (statistical view)**

- **Coefficients:** each is a change in log-odds per unit of the feature; $e^{\beta}$ is an odds ratio. Check signs and magnitudes against domain knowledge.
- **Significance:** Wald tests and confidence intervals per coefficient; likelihood-ratio tests for nested models (deviance difference follows a chi-squared distribution).
- **Deviance and pseudo-$`R^2`$** (McFadden), and **AIC or BIC** to compare models with different features.
- **Goodness of fit:** the Hosmer-Lemeshow test is traditional, though it depends on the binning; calibration plots are more informative.

**4. Assumption checks**

- Linearity of features in the log-odds (check with binned plots or splines).
- **Multicollinearity** inflates coefficient variance and makes interpretation unstable (see 10.36).
- Complete or quasi-complete separation makes coefficients diverge; regularisation fixes it.
- Independent observations; repeated measures need mixed or clustered models.

```python
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
print(model.summary())            # coefficients, z-tests, pseudo R-squared, log-likelihood
p = model.predict(sm.add_constant(X_test))
print(roc_auc_score(y_test, p), log_loss(y_test, p), brier_score_loss(y_test, p))
```

**Practical guidance:** in regulated settings (credit, insurance) the coefficient review, calibration by risk band and stability over time often matter more than a small gain in AUC.

### 12.21 How do you choose a classifier's decision threshold, and how do you weigh false positives against false negatives?

The threshold turns a score into a decision, so it should be set by the cost of each kind of mistake and by operational limits, not left at 0.5. Whether a false positive or a false negative is "better" has no general answer: it depends on what each one costs in the specific application.

**False positive against false negative**

| | False positive (false alarm) | False negative (miss) |
| --- | --- | --- |
| **Disease screening** | Extra test, anxiety | Untreated illness: far worse |
| **Fraud** | Good customer blocked, call-centre cost | Fraud loss |
| **Spam filter** | Important email hidden: often worse | One spam reaches the inbox |
| **Criminal justice** | Innocent person convicted: worse by design | Guilty person released |

**The principled default: expected cost**

With a calibrated probability $p$, predict positive when the expected cost of acting is lower than the cost of not acting. With costs $C_{\text{FP}}$ and $C_{\text{FN}}$, and zero cost for correct decisions, the optimal threshold is:

$$
t^{\ast} = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}
$$

If a missed fraud costs 200 dollars and a false block costs 10, $t^{\ast}$ is about 0.048, far below 0.5. This needs calibrated probabilities (see 12.36); otherwise sweep thresholds on validation data and pick the one with the lowest total cost.

**Other methods for choosing the threshold**

1. **Cost sweep.** When probabilities are not calibrated, compute total cost on validation data for each candidate threshold and take the minimum (see the code below).
2. **Constraint-based.** Maximise recall subject to precision of at least 95 percent, or keep the false positive rate below 1 percent. Common when a regulator or product owner sets a floor.
3. **Capacity-based.** If the review team can handle 500 cases a day, the threshold is the score of the 500th highest case (precision@k). Very common in fraud, compliance and sales lead scoring.
4. **Metric-maximising.** Maximise F1 or F-beta on the precision-recall curve, or Youden's J (TPR minus FPR) on the ROC curve, when no costs are available.
5. **Multiple thresholds (reject option).** Auto-approve below one score, auto-block above another, and send the middle band to humans. This often beats any single threshold.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y_val, p_val)
ok = prec[:-1] >= 0.95
t_constraint = thr[ok][np.argmax(rec[:-1][ok])] if ok.any() else None

costs = [(10 * ((p_val >= t) & (y_val == 0)).sum() + 200 * ((p_val < t) & (y_val == 1)).sum(), t)
         for t in np.linspace(0.01, 0.99, 99)]
t_cost = min(costs)[1]
```

**Production pitfalls**

- **Choose on validation data, never on the test set.**
- **Thresholds drift.** When the base rate or the score distribution shifts, the same threshold produces a different alert volume and precision. Monitor the positive rate and re-tune on fresh labelled data.
- **Retraining moves scores.** A new model version needs its own threshold, or a calibration step that keeps scores comparable across versions.
- **Different segments may need different thresholds**, but per-group thresholds on protected attributes raise legal and fairness questions (see 12.37).

### 12.22 How can you measure the performance of a dimensionality reduction algorithm on your dataset?

It depends on *why* you are reducing dimensions:

**1. Downstream task performance (most important in practice).** Train the same model on the reduced and original features and compare the validation metric, training and inference time, and memory. If accuracy holds while latency drops, the reduction worked. Put the reducer inside the CV pipeline to avoid leakage.

**2. Information retained.**
- **PCA:** explained variance ratio, cumulative variance (e.g. keep 95%), scree plot.
- **Reconstruction error:** project down and back up ($\hat X =$ `inverse_transform`) and compute MSE. This works for PCA, autoencoders and kernel PCA (approximate pre-image).

**3. Neighbourhood/structure preservation** (for manifold methods such as t-SNE, UMAP, Isomap):
- **Trustworthiness and continuity**: are the k-nearest neighbours preserved?
- kNN classification accuracy in the embedded space.
- Correlation of pairwise distances before and after (Shepard diagram).

**4. Visual and cluster quality:** silhouette score of known classes in 2-D; check plots qualitatively (but don't over-read t-SNE cluster sizes and distances).

**Tips:** run hyperparameter search over the number of components with the downstream metric as the objective, and check stability across random seeds.

```python
from sklearn.manifold import trustworthiness
trustworthiness(X, X_emb, n_neighbors=10)
```

### 12.23 What cross-validation technique would you use on time-series data, and why does standard k-fold fail?

Standard k-fold assumes rows are independent and order does not matter. Time series break both assumptions: random folds let the model train on the future and predict the past (leakage), and autocorrelation means neighbouring rows in train and validation are near-copies, so scores are inflated. Use schemes that preserve time order.

**The failure**

```text
Data:        2021 -> 2022 -> 2023 -> 2024 -> 2025
Random fold: train on 2021, 2022, 2024, 2025   validate on 2023
             -> the model has seen what happened after 2023
```

**Order-preserving schemes**

- **Expanding window (forward chaining, rolling origin):** train on everything up to a cut-off, test on the next horizon, move the cut-off forward. This is scikit-learn's `TimeSeriesSplit` and mimics periodic retraining.

```text
Fold 1: train 2021            | validate 2022
Fold 2: train 2021-2022       | validate 2023
Fold 3: train 2021-2023       | validate 2024
Fold 4: train 2021-2024       | validate 2025
```

- **Sliding (rolling) window:** fixed-length training window that moves forward; better when old data is stale or the process is non-stationary.
- **Gap or embargo:** leave a gap between train and test equal to the label horizon or the feature look-back, so overlapping windows do not leak. In finance this is **purged k-fold with embargo** (Lopez de Prado).
- **Blocked CV:** contiguous blocks with gaps, when you need more folds and some leakage risk is acceptable.
- **Many series (stores, products):** split by time across all series; optionally also hold out whole series to test cold start.

**Best practices**

- Match the test window to the real forecast horizon and the retraining cadence.
- Compute features with only the information available at each cut-off (point-in-time correct), and fit preprocessing inside each fold.
- Keep a final out-of-time holdout, and look at the metric per fold: seasonality and regime changes show up as unstable folds.
- Compare against naive and seasonal-naive baselines (see 32.30 for forecast backtesting).

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=30, gap=7)
for tr, te in tscv.split(X):
    model.fit(X[tr], y[tr])
    print(score(model, X[te], y[te]))
```

**Where else this matters:** any data with a time element, such as churn, fraud and credit, should be validated out of time, even if it is not formally a time series.

### 12.24 Which among these is more important: model accuracy or model performance?

It depends on how you define the terms, and that is the point of the question.

- **Model accuracy** is one specific metric: the fraction of correct predictions.
- **Model performance** is broader: how well the model serves its purpose. That includes the *right* metric for the problem (precision/recall, F1, AUC, RMSE, calibration) and also latency, throughput, cost, robustness, fairness and business impact.

**Performance matters more**, because accuracy can be misleading:
- **Imbalanced data:** with 1% fraud, predicting "not fraud" always gives 99% accuracy and catches nothing. Recall, precision and PR-AUC tell the real story.
- **Asymmetric costs:** a missed cancer diagnosis (FN) costs far more than a false alarm (FP), so optimize recall at an acceptable precision, not accuracy.
- **Operational constraints:** a model that is 0.5% more accurate but 10× slower may break a 50 ms SLA or blow the inference budget. Fraud scoring, ad ranking and search all have hard latency limits.
- **Business outcome:** the goal is revenue, retention or reduced risk. A slightly less accurate but interpretable, stable model may win in regulated settings.

**How I'd frame it in practice:** agree on the success metric with stakeholders up front (e.g. "recall ≥ 90% at ≤ 5% FPR, p95 latency < 100 ms"), select models on that, and validate with an A/B test on the business KPI.

### 12.25 Difference between R2 and adjusted R2? Why do you prefer adjusted R2?

**$R^2$** measures the fraction of the target's variance explained by the model:

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

Its problem: on training data, **adding any feature never decreases $R^2$**, even a column of random noise, because least squares can always set its coefficient to zero or exploit chance correlation.

**Adjusted $R^2$** penalizes the number of predictors $p$ relative to sample size $n$:

$$
R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}
$$

It only increases if a new feature improves the fit more than would be expected by chance; useless features make it drop.

**Why prefer adjusted $R^2$:**

- Fair comparison of linear models with **different numbers of features**.
- Guards against overfitting by feature stuffing, especially when $p$ is not small relative to $n$.

**Caveats:** both are in-sample measures for linear models. For real model selection, held-out metrics (cross-validated RMSE/MAE) or AIC/BIC are more reliable, and neither $R^2$ tells you whether the model is unbiased or whether residual assumptions hold. With $n \gg p$ the two are nearly identical.

### 12.26 When should you use PR-AUC over ROC-AUC?

Use **PR-AUC (average precision)** when the **positive class is rare** and you care mainly about how well the model finds positives, not about the negatives.

**Why:** ROC uses $\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$. With 1M negatives and 1k positives, 10k false positives is only a 1% FPR, so the ROC curve looks great. **Precision** = $\text{TP} / (\text{TP} + \text{FP})$ exposes that same situation: 10k false alarms against 1k true positives means precision under 10%. PR curves ignore true negatives entirely, so they reflect what an analyst reviewing alerts actually experiences.

**Choose PR-AUC when**

- There is severe imbalance: fraud, rare disease screening, anomaly detection, click/conversion prediction, retrieval and search (relevant documents are few).
- **False positives have a real cost per alert** (investigation queues, user friction).
- You want to compare models on the *positive-class* performance that matters.

**Choose ROC-AUC when**

- Classes are reasonably balanced, or both classes matter equally.
- You want a metric that is invariant to the base rate (comparing across populations with different prevalence). PR-AUC changes with prevalence: a random model's PR-AUC equals the positive rate, not 0.5.

**Practical notes**

- Always report the **baseline** (prevalence) alongside PR-AUC. 0.30 is excellent when prevalence is 0.1%, and poor when it is 25%.
- Use `average_precision_score` (step-wise) rather than trapezoidal interpolation of the PR curve, which is optimistic.
- Finally, choose an operating threshold by business constraint, e.g. "maximize recall subject to precision ≥ 80%", and monitor precision and recall at that point in production.

### 12.27 How do you run a good ablation study?

An **ablation study** removes or changes one component at a time to measure its causal contribution to the result.

**Principles**

1. **Start from a strong, well-tuned full system** and define the primary metric (plus a few secondary ones such as latency and cost) *before* running anything.
2. **Change one thing at a time.** Remove a component, replace it with a simple baseline, or vary one hyperparameter. Keep everything else identical: data splits, compute budget, training steps, preprocessing, evaluation code.
3. **Fair tuning:** if removing a component changes the optimal learning rate or other settings, give each variant a comparable tuning budget, or the comparison is biased toward the configuration you tuned most.
4. **Multiple seeds and uncertainty:** run 3–5 seeds and report mean ± std or confidence intervals. Use paired tests or bootstrap on the same eval set. Many "gains" of 0.3% are noise.
5. **Use held-out data** that was not used for model selection, so you don't overfit the ablations to the dev set.
6. **Measure interactions:** components can be redundant or synergistic. When interactions matter, add pairwise ablations or a factorial design for the key factors.
7. **Report everything:** a table with the full model, each "−component" variant, and the deltas, including negative and neutral results. Note the compute cost.

**Example (RAG system):**

| Variant | Answer accuracy | Faithfulness | p95 latency |
| --- | --- | --- | --- |
| Full (hybrid + rerank + query rewrite) | 82.1 ± 0.6 | 0.91 | 1.8 s |
| − reranker | 76.4 | 0.85 | 1.2 s |
| − BM25 (dense only) | 79.0 | 0.89 | 1.6 s |
| − query rewrite | 81.7 | 0.90 | 1.5 s |

**Conclusion from the table:** the reranker is essential. Query rewriting adds 0.3 s for a gain that isn't significant, so it is a candidate for removal.

**Pitfalls:** confounding changes (a new data version alongside the ablation), comparing to a weak baseline, cherry-picking seeds, and ablating only on easy benchmarks. Ablations drive **simplification decisions** as much as paper claims.

### 12.28 What is stratified cross-validation, and when should you not use it?

#### Stratified Cross-Validation

**Stratified cross-validation** keeps approximately the **same class distribution in every fold** as in the original dataset.

Example with 20% positive and 80% negative:

```txt
Dataset: 20% positive / 80% negative

Fold 1: 20% / 80%
Fold 2: 20% / 80%
Fold 3: 20% / 80%
...
```

This is especially useful for **classification with imbalanced classes**, because every fold gets enough minority-class examples.

#### When NOT to use it

**1. Time-series data**

Don't randomly stratify time-series observations:

```txt
2021 → 2022 → 2023 → 2024 → 2025
```

You need chronological/walk-forward validation instead.

**2. Regression**

Standard stratified CV is designed for **categorical labels**. For continuous targets, use regular K-fold or specialized approaches such as binning the target when appropriate.

**3. Grouped data**

If multiple samples belong to the same person, patient, customer, device, etc., random stratification can put the same entity into both train and validation:

```txt
Customer A → Train
Customer A → Validation
```

Use **GroupKFold** or **StratifiedGroupKFold** when both grouping and class balance matter.

### 12.29 Training accuracy is 99%, validation accuracy is 60%. How do you debug it?

A 99% vs 60% gap is classic **overfitting**, but I first rule out data problems before adding regularisation.

**1. Check the data split:**
- Is validation from the same distribution? (different time period, source, class balance)
- Are there duplicates or near-duplicates inside train (memorisation) — or is validation genuinely harder?
- Leakage in train only (e.g. ID-like feature that encodes the label in train but not val)?
- Is validation preprocessed the same way (same scaler/tokenizer, no fitting on val separately)?

**2. Check the baseline:** what does a simple model get? If logistic regression gets 58%, the task/features may just be weak.

**3. If it is overfitting (high variance):**
- More data or data augmentation.
- Reduce capacity: smaller model, shallower trees, fewer features.
- Regularisation: L1/L2/weight decay, dropout, max depth/min samples per leaf, early stopping.
- Feature selection — remove noisy, high-cardinality or ID features.
- Cross-validation to get a reliable estimate; ensembling (bagging).

**4. Diagnostics:**
- Learning curves (train/val vs data size and epochs).
- Per-class/segment validation errors and confusion matrix.
- Feature importance — a dominant suspicious feature often indicates leakage or spurious correlation.

**5. Label quality:** noisy labels are memorised in training but can't generalise.

Fix one thing at a time and track experiments so the cause is clear.

### 12.30 Training and validation accuracy are both 50%. What could be wrong?

Both at 50% (for a binary balanced task) means the model is performing at **chance** — it's learning nothing. Underfitting is possible, but this usually signals a bug.

**Data/label problems:**
- Labels shuffled or misaligned with inputs (e.g. after a join or sort, or data loader shuffling X and y separately).
- Wrong label encoding, or all labels the same in a batch due to unshuffled, sorted data.
- Features carry no signal (wrong columns, all constants/NaNs filled with zeros, normalisation bug).
- Target is genuinely unpredictable from the given features.

**Training/pipeline bugs:**
- Learning rate too high (diverges) or too low (no movement); loss not decreasing at all.
- Wrong loss for the output (e.g. softmax + `BCELoss`, logits passed to a loss expecting probabilities), wrong output activation.
- Gradients not flowing: frozen layers, forgot `optimizer.step()` / `zero_grad()`, dead ReLUs, vanishing gradients.
- Evaluation bug: comparing predictions with wrong labels, or `argmax` on the wrong axis.

**Model capacity:** model too simple (linear model on nonlinear XOR-like data) → genuine underfitting.

**Debugging steps:**
1. Inspect samples and label distribution manually.
2. Try a trivial baseline (majority class, logistic regression).
3. **Overfit a tiny batch** (10–100 samples): a correct pipeline should reach ~100% train accuracy. If it can't, there's a bug.
4. Check loss curves and gradient norms.
5. Then increase capacity, train longer, or engineer better features.

### 12.31 Your model performs well offline but poorly in production. Why?

Good offline, bad in production almost always means **the production data or setup differs from the offline evaluation**.

**Common causes:**
- **Training-serving skew:** features computed differently online (different code, units, defaults, stale feature store values, missing-value handling).
- **Data leakage offline:** features available in historical data that aren't available at prediction time (future information, post-event fields), inflating offline metrics.
- **Wrong validation scheme:** random split on time-dependent data instead of a time-based split; duplicates across train/test; tuning on the test set.
- **Distribution shift:** production users, seasons, products or behaviour differ from training data (covariate shift, label shift, concept drift).
- **Feedback loops:** the model's decisions change future data (e.g. recommendations shape clicks, fraud blocks change attacker behaviour).
- **Metric mismatch:** offline metric (AUC) doesn't reflect the business objective or the chosen threshold; class priors differ. (12.39 covers the case where the offline metric improves but the A/B test does not.)
- **Engineering bugs:** different model version, preprocessing missing, timeouts causing fallbacks, latency forcing truncated inputs.

**How to investigate:**
1. Log production inputs and predictions; recompute offline predictions on those exact inputs and compare.
2. Compare feature distributions train vs production (PSI/KS).
3. Re-evaluate with a time-based backtest that mimics production.
4. Segment errors by user group, time, channel.

**Prevention:** feature store, point-in-time joins, shadow deployments before launch, monitoring and canary rollouts.

### 12.32 Your model's accuracy decreases after adding more training data. Explain possible reasons.

Normally more data helps, so a drop is a signal to inspect the new data and the evaluation:

- **Lower-quality or noisy labels** in the new data (different labelling team, weak/automatic labels).
- **Different distribution:** new data comes from another source, region, time period or device, so the model now fits a mixture that matches the validation set less well. Or the validation set is from the old distribution and the new data shifted the model.
- **Class imbalance changes:** the new data changes class priors, shifting the decision threshold.
- **Duplicates / leakage:** test examples (or near-duplicates) previously in training removed, or new data leaks into the validation set — or a re-split moved hard examples into validation, making scores incomparable.
- **Preprocessing inconsistency:** new data encoded differently (units, categories, text normalisation, missing values).
- **Concept drift:** the relationship between features and label changed over time; old and new data conflict.
- **Model/training not adjusted:** fixed number of epochs or learning-rate schedule now under-trains; regularisation tuned for smaller data; capacity too low to fit more diverse data.
- **Random variance:** the difference is within noise — check with cross-validation / multiple seeds and confidence intervals.

**How to diagnose:** evaluate on a fixed, unchanged test set; train on old vs new data separately and compare; check label agreement on a sample of new data; compare distributions; check learning curves. Then clean or reweight the new data, or train with domain indicators.

### 12.33 A feature has extremely high importance. How do you determine whether it is legitimate or leakage?

An extremely important feature is either a genuine strong signal or — very often — **target leakage**. I check:

**1. Timing / availability:** Is the feature's value known **at prediction time** in production? Trace its lineage. Examples of leakage: "account_closed_date" when predicting churn, "number of chargebacks" when predicting fraud, a status field updated after the outcome.

**2. Domain sense-check:** does the relationship make causal or business sense? Ask domain experts. A support-ticket count that predicts churn may be legit; a "refund_issued" flag predicting a complaint is probably leakage.

**3. Too-good-to-be-true metrics:** near-perfect AUC (e.g. > 0.99) on a hard problem is a red flag. Train a model on **only that feature** — if it alone almost perfectly predicts the label, investigate.

**4. Time-based validation:** use point-in-time correct feature snapshots and an out-of-time test set. If performance collapses when features are computed as of the prediction timestamp, it was leakage.

**5. Proxy / ID leakage:** IDs, row order, timestamps, file paths or data-source flags that correlate with the label because of how the data was collected (e.g. all positives came from one system).

**6. Inspect the relationship:** partial dependence/SHAP plots; look for perfect thresholds or values only present for one class (e.g. missing for all negatives).

**7. Compare with production:** shadow-score live data — leaked features often have different distributions or are missing online.

If legitimate, keep it but monitor it closely and consider robustness (the model shouldn't fail entirely if that feature breaks).

### 12.34 Your validation score varies significantly between folds. What does that indicate?

High variance across folds means your performance estimate is **unstable** — the score depends heavily on *which* rows land in validation. Typical causes:

- **Small dataset / small folds:** each fold has too few examples (or too few positives) for a stable metric.
- **High-variance model:** a deep tree or overfit model reacts strongly to the particular training split.
- **Heterogeneous data:** subgroups (regions, customers, time periods) are unevenly distributed across folds.
- **Leakage or grouping problems:** the same user/patient appears in train and validation in some folds but not others.
- **Class imbalance without stratification:** some folds get almost no minority examples.
- **Temporal drift:** if data is time-ordered, some folds are "easier" periods than others.

**What to do**

1. Report mean **and** std (or a confidence interval) rather than a single number; use repeated k-fold to tighten the estimate.
2. Use `StratifiedKFold` for imbalance, `GroupKFold` for grouped entities, `TimeSeriesSplit` for temporal data.
3. Inspect the worst fold: what is different about its data?
4. Reduce model variance (regularization, ensembling, simpler model) if the instability is model-driven.

**Production angle:** large fold variance is a warning that production performance will also swing as the input mix changes — treat it as a robustness problem, not just an evaluation nuisance.

### 12.35 Your model predicts only the majority class. How do you fix it?

Predicting only the majority class is the classic symptom of **severe class imbalance** combined with an objective (or metric) that rewards it: with 99% negatives, "always negative" gives 99% accuracy and a low unweighted loss.

**Diagnose first**
- Check the class distribution and the confusion matrix; switch from accuracy to **precision/recall, F1, PR-AUC**.
- Inspect predicted probabilities — often the model *does* rank positives higher, but the default 0.5 threshold hides it.
- Rule out bugs: label encoding errors, features that carry no signal, a broken pipeline feeding constant inputs, or a learning rate so high the model collapsed to the prior.

**Fixes**
1. **Threshold tuning** on a validation set (optimize F1 or a cost-weighted metric) — cheapest and often enough.
2. **Class weights / cost-sensitive loss** (`class_weight='balanced'`, `scale_pos_weight` in XGBoost, weighted BCE, focal loss).
3. **Resampling:** undersample the majority, oversample the minority, or SMOTE — applied only to the training folds.
4. **Better features** if the model genuinely cannot separate the classes.
5. Reframe as **anomaly detection** when positives are extremely rare.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr, y_tr)
probs = clf.predict_proba(X_val)[:, 1]
# choose the threshold that maximizes F1 on validation instead of 0.5
```

**Production angle:** pick the threshold from the business cost of FP vs FN, and recalibrate as the base rate drifts.

### 12.36 What is probability calibration, how do you measure it, and how do you fix a miscalibrated model?

A model is **calibrated** when its scores mean what they say: of all cases scored 0.8, about 80 percent are positive. Discrimination (AUC) and calibration are separate properties, and a model can rank perfectly while being badly calibrated. Calibration matters whenever a probability feeds a decision: expected-cost thresholds, pricing, credit limits, risk bands, or combining scores from several models.

**Why models are miscalibrated**

- **Boosted trees and random forests** tend to push scores away from or towards the extremes; SVM scores are not probabilities at all.
- **Modern deep networks** are often overconfident, especially when trained long with little regularisation.
- **Class weights, resampling and focal loss** change the effective class prior, inflating positive scores (see 12.8).
- **Distribution shift:** a model calibrated last year drifts as the base rate changes.

**How to measure it**

- **Reliability diagram:** bin predictions by score (or by quantile), plot mean predicted probability against observed positive rate. A calibrated model lies on the diagonal; points below it mean overconfidence.
- **Expected calibration error (ECE):** the weighted average gap between confidence and accuracy over $B$ bins, where $n_b$ is the number of predictions in bin $b$:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left\lvert \text{acc}(b) - \text{conf}(b) \right\rvert
$$

- **Brier score** (mean squared error of the probability) and **log loss**: proper scoring rules that reward both calibration and discrimination.
- ECE depends on the binning and is noisy for small samples; use quantile bins, report a bootstrap interval, and look at the diagram, not just the number. Check calibration **per segment** too: a model can be calibrated overall and badly off for one region.

**How to fix it (post-hoc, on held-out data)**

| Method | How it works | When to use |
| --- | --- | --- |
| **Platt scaling** | Fit a logistic regression on the model's score | Small calibration sets; sigmoid-shaped distortion |
| **Isotonic regression** | Fit a monotone step function | Larger sets (thousands of examples); arbitrary monotone distortion; can overfit small data |
| **Temperature scaling** | Divide logits by one learned scalar $T$ | Deep networks and multi-class; keeps the ranking unchanged |
| **Prior correction** | Adjust for known resampling rate or new base rate | After undersampling, or when prevalence shifts |

All of them must be fitted on data **not** used for training, such as a separate calibration split or cross-validated predictions. Because they are monotone, Platt, isotonic and temperature scaling leave ROC-AUC (essentially) unchanged while improving log loss and ECE.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

cal = CalibratedClassifierCV(base_model, method="isotonic", cv=5).fit(X_train, y_train)
p = cal.predict_proba(X_test)[:, 1]
frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
print(brier_score_loss(y_test, p))
```

**Production guidance**

- Recalibrate after every retrain and monitor calibration over time; a shift in base rate shows up as a calibration drift before AUC moves.
- If downstream systems consume the score as a probability, make calibration a release gate, not an afterthought.
- Calibration within groups is also a fairness criterion, and it can conflict with equalised error rates (see 12.37).

### 12.37 How do you evaluate a model across slices and check that it is fair across groups?

An aggregate metric is an average over very different populations, and averages hide failures. Slice-based evaluation breaks performance down by meaningful segments, and fairness evaluation is the special case where the segments are groups that must not be treated unjustly. Both should be part of every model review, not a one-off audit.

**1. Choose the slices**

- **Business segments:** region, channel, device, customer tier, product category, new against returning users.
- **Data properties:** input length, language, image quality, missing-feature patterns, rare categories.
- **Protected or sensitive groups:** gender, age band, ethnicity, disability, where you are legally allowed and able to measure them.
- **Discovered slices:** cluster the errors or train a shallow tree to predict "model was wrong" to find weak regions nobody thought of.

**2. Measure properly**

- Report the same metrics per slice with **confidence intervals** and **sample sizes**; a slice with 40 examples showing a 10 point drop may be noise.
- Look at the **worst slice** and the **gap** to the overall figure, not only the average.
- Compare against the previous model per slice, so a release cannot improve the average while regressing a key segment.

**3. Fairness metrics (for classifiers)**

| Criterion | Requires equal across groups | Typical use |
| --- | --- | --- |
| **Demographic parity** | Positive prediction rate | Allocation where base rates should not matter |
| **Equal opportunity** | True positive rate | Benefits (loans approved for those who would repay) |
| **Equalised odds** | True positive and false positive rates | Punitive decisions (fraud flags, risk scores) |
| **Predictive parity** | Precision | When a positive prediction triggers costly action |
| **Calibration within groups** | Observed rate at each score | Risk scores used by humans |

When base rates differ between groups, a non-trivial classifier cannot satisfy calibration within groups and equal error rates at the same time (Kleinberg et al., Chouldechova). The choice of criterion is therefore a policy decision to make with legal, risk and product owners, not a purely technical one (see 24.27).

**4. Common causes of slice failures**

- Under-representation in training data, label bias (historical decisions used as labels), proxy features (postcode for ethnicity), and measurement differences (sensors or forms that work worse for some groups).

**5. What to do about a gap**

- Collect or reweight data for the weak slice, fix label quality, remove or constrain proxy features, or train with fairness constraints.
- Post-processing (group-specific thresholds) can equalise error rates, but using protected attributes at decision time may be unlawful in some domains; check first.
- Mitigation methods in depth are covered in 30.4 and 30.23 (intersectional groups).

```python
import pandas as pd
from sklearn.metrics import recall_score, precision_score

df = pd.DataFrame({"y": y_test, "pred": y_pred, "group": groups})
report = df.groupby("group").apply(lambda g: pd.Series({
    "n": len(g),
    "positive_rate": g.pred.mean(),
    "tpr": recall_score(g.y, g.pred, zero_division=0),
    "precision": precision_score(g.y, g.pred, zero_division=0),
}))
print(report)
```

**Production guidance:** define the critical slices and acceptable gaps before training, add them as release gates in the evaluation pipeline, and keep monitoring them after launch because drift often hits one segment first (see 24.24).

### 12.38 How do you evaluate whether a model will hold up under distribution shift?

A random test split drawn from the same data as training measures in-distribution performance, which is usually the best case. Production data drifts, so a senior evaluation deliberately builds test sets that resemble the future and the edges of the input space, and reports how much performance degrades.

**Types of shift to test for** (detailed definitions in 26.65)

- **Covariate shift:** inputs change, the input-label relationship does not (new device mix, new markets).
- **Label (prior) shift:** class proportions change (fraud rate doubles in a holiday season).
- **Concept drift:** the relationship itself changes (fraudsters adapt, a policy changes).
- **Subpopulation shift:** the mix of known groups changes, so a weak group becomes more common.

**Build test sets that stress the model**

| Test set | How | What it tells you |
| --- | --- | --- |
| **Out-of-time** | Train on months 1 to 9, test on 10 to 12 | Realistic decay over the retraining interval |
| **Out-of-domain** | Hold out a whole region, store, hospital or customer segment | Behaviour on populations not seen in training |
| **Group holdout** | Hold out entire users or entities | Cold start and memorisation |
| **Stress and perturbation** | Noise, missing features, typos, image corruptions, unit changes | Robustness to messy real inputs |
| **Reweighted test** | Weight test examples to match the expected future mix | Performance under a forecast population change |

**Techniques**

- **Measure the degradation curve:** score the model on successive months after the training cut-off. The slope tells you how often to retrain.
- **Adversarial validation:** train a classifier to separate training data from recent production data; if it succeeds, the features it uses are the ones that shifted (see 10.37).
- **Importance weighting:** estimate $w(x) = p_{\text{target}}(x) / p_{\text{source}}(x)$ with that domain classifier and compute a weighted test metric to estimate performance on the new population without new labels. It breaks down when the target has regions the source never covered.
- **Label shift estimation:** methods such as black-box shift estimation use the confusion matrix and the new prediction distribution to estimate the new class priors, which you can then use to recalibrate.
- **Worst-group metrics:** report the minimum over groups, not only the average, because shift usually hurts the weakest group first.

**Design responses when the model is fragile**

- Prefer features that are stable over time; drop features whose meaning drifts (IDs, campaign codes).
- Train on more diverse data, augment for expected perturbations, and use time-decayed sample weights.
- Add monitoring on input drift (PSI, KS) and prediction drift, with retraining triggers (see 26.64 and 26.72).
- Add an abstain or fallback path when inputs are far from the training data.

**What interviewers look for:** the recognition that a random split overstates production performance, a concrete out-of-time or out-of-domain protocol, and a plan for what happens when the evaluation shows the model is fragile.

### 12.39 Your offline metric improved, but the A/B test showed no gain. Why, and what do you do?

This is common: in many organisations a large share of offline wins do not move the online metric. It usually means the offline metric is an imperfect proxy for what users and the business respond to, or that the online test cannot see the effect. Diagnose which before running more experiments.

**Reasons the offline metric did not translate**

- **Proxy mismatch.** Offline AUC on clicks, but the business metric is purchases or retention; better click prediction may simply surface more clickbait.
- **Improvement in the wrong region.** AUC rose across thresholds, but the model operates at the top 1 percent, where nothing changed. Check the metric at the operating point (precision@k, TPR at a fixed FPR).
- **Saturation.** The decision barely changes: the new model reorders items below the fold, or the old and new models make the same decision for 98 percent of traffic. Measure the share of decisions that actually differ.
- **Biased offline data.** Logged data only contains outcomes for items the old model chose to show (selection and position bias), so the offline test rewards models that imitate the old one (see 32.19 and 32.21).
- **Leakage or skew.** Offline features were computed with information not available at serving time, or the online feature pipeline differs from the offline one (see 12.31).
- **Latency or availability cost.** The new model is slower or times out more often and falls back to defaults, eating the gain.
- **Calibration change.** Downstream systems (auctions, pricing, thresholds) consume the score as a probability; a better-ranking but differently calibrated model makes worse decisions.

**Reasons the online test could not see it**

- **Underpowered test:** the expected effect is smaller than the minimum detectable effect for the traffic and duration.
- **Dilution:** only a fraction of users in the treatment arm ever encounter the model; analyse triggered users.
- **Novelty or learning effects, interference between users, or short duration** for metrics that take weeks to move.

**What to do**

1. Check the plumbing: shadow-compare online and offline predictions on the same requests, and check the sample ratio and exposure logging.
2. Quantify how many decisions changed and look at a sample of them by hand.
3. Re-evaluate offline at the operating point and on the segments with the most traffic.
4. Check the power calculation; if the test was underpowered, use variance reduction (CUPED), triggered analysis or interleaving for rankers (see 28.27).
5. Over time, **measure the offline-online correlation** across past experiments and pick or design offline metrics that predict online wins (counterfactual or off-policy evaluation, metrics at the operating point, long-term proxies).

**Leadership angle:** treat this as a signal about the evaluation system, not a failed model. A team that keeps shipping offline wins that never show up online is optimising the wrong target, and fixing the offline metric is often the highest-leverage work available.

### 12.40 How do you evaluate a model in production when labels are delayed, scarce or affected by the model's own decisions?

Many production models never see clean, timely labels: chargebacks arrive weeks later, loan defaults take months, and a fraud model that blocks a transaction prevents you from ever learning whether it was fraud. Evaluation then combines early proxies, mature labels, deliberate label collection and counterfactual corrections.

**1. Delayed labels**

- Define a **label maturity window** (for example 90 days for chargebacks) and only compute final metrics on cohorts old enough to be labelled; otherwise recent cohorts look artificially good or bad.
- Track metrics by **prediction cohort** (week scored), not by label arrival date.
- Use **early proxies** that correlate with the final label: early delinquency for default, first-week engagement for churn, customer disputes for fraud. Validate each proxy against mature labels before trusting it.
- Monitor **leading indicators** while you wait: input drift, prediction drift, score distribution and alert volume (see 26.64).

**2. Scarce labels**

- **Sampled human review:** label a random sample of predictions each week. Stratify by score band so rare high-risk cases are covered, then weight back to the population.
- **Active evaluation:** spend the labelling budget where it reduces uncertainty most (disagreement between models, borderline scores), keeping the importance weights so estimates stay unbiased.
- **Golden sets:** a stable, expert-labelled set re-scored on every model version for regression testing.

**3. Labels censored by the model's own decisions**

- When the model's action hides the outcome (blocked transactions, rejected loan applicants, items never recommended), the labelled data only covers cases the model allowed.
- **Holdout or exploration traffic:** let a small random share of decisions bypass the model (approve a small sample of borderline cases, randomise a small slice of recommendations) to observe true outcomes. Agree the cost of this with the business explicitly, since it is the price of unbiased evaluation.
- **Off-policy evaluation:** with logged propensities, inverse propensity scoring or doubly robust estimators estimate how a new policy would have performed (see 32.4 and 32.5).
- **Reject inference** in credit scoring is the classic version of this problem; treat its results with caution.

**4. Report with honesty about uncertainty**

- State which metrics are final, which are proxies and which are estimated from samples, with confidence intervals.

```python
import numpy as np

def weighted_precision(sample_df):
    # sample_df: reviewed alerts with columns is_fraud (0/1) and weight (1 / sampling rate of its score band)
    w = sample_df["weight"].to_numpy()
    return np.sum(w * sample_df["is_fraud"].to_numpy()) / np.sum(w)
```

**What interviewers look for:** awareness that the training labels are themselves shaped by past models, a concrete plan for unbiased label collection, and the discipline to compare cohorts only after their labels have matured.

### 12.41 How do you build a trustworthy test set, and how large does it need to be?

A test set is only useful if it represents the decisions the model will make, has correct labels, and is large enough that differences you care about are bigger than the noise. Most "model improvements" that vanish in production come from test sets that fail one of those three conditions.

**Representativeness**

- Sample from the population the model will serve, in the period it will serve (out of time where possible, see 12.38).
- Split by entity (user, patient, document, session) so near-duplicates do not appear on both sides; deduplicate exact and near copies across splits.
- Over-sample rare but critical cases (high-value fraud, safety-critical classes) into a separate stratum, and report it separately rather than distorting the headline metric.

**Label quality**

- Use written labelling guidelines, two or more annotators for a sample, adjudication of disagreements, and inter-annotator agreement (Cohen's kappa) as a quality signal.
- Test labels deserve more care than training labels: label noise in the test set caps the measurable accuracy and can reverse model rankings.
- Review a sample of "model errors": a surprising share are often label errors.

**Size: how many examples?**

For a proportion metric such as accuracy or recall with true value $p$ on $n$ examples, the standard error is:

$$
\text{SE} = \sqrt{\frac{p(1 - p)}{n}}
$$

| Accuracy around | Test examples | Approximate 95 percent interval |
| --- | --- | --- |
| 90 percent | 500 | plus or minus 2.6 points |
| 90 percent | 2,000 | plus or minus 1.3 points |
| 90 percent | 10,000 | plus or minus 0.6 points |

- For recall, $n$ is the number of **positives**, not total rows: 100 positives give an interval of about 6 points at 90 percent recall.
- To detect a difference between two models, paired tests on the same examples need far fewer examples than comparing two independent intervals (see 12.12).
- Rule of thumb: size the test set so that the smallest improvement you would act on is clearly larger than the interval width.

```python
import numpy as np
def n_needed(p, half_width, z=1.96):
    return int(np.ceil(z**2 * p * (1 - p) / half_width**2))
n_needed(0.9, 0.01)  # about 3,458 examples for plus or minus 1 point at 90 percent
```

**Keeping it trustworthy over time**

- **Version and lock it:** store the exact IDs, labels and labelling guideline version, so results are reproducible.
- **Limit reuse:** every decision made on the test set leaks information into model choices (see 12.14). Keep a validation set for iteration and a test set for release decisions, and refresh the test set periodically from recent data.
- **Keep a stable regression set** alongside a fresh one: the stable set detects regressions between versions, the fresh one tracks today's distribution.

**What interviewers look for:** reasoning about uncertainty with numbers, not just "use a big test set", and processes for label quality, freshness and leakage.

---

← [Chapter 11. Optimization and Training Dynamics](11-optimization-and-training-dynamics.md) · [Table of contents](../README.md) · [Chapter 13. Deep Learning](13-deep-learning.md) →
