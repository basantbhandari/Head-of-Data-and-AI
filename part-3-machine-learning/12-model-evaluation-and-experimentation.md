# Chapter 12. Model Evaluation and Experimentation

_Part III — Machine Learning_

Metrics, validation strategies, A/B testing, ablations and debugging models that underperform. Senior candidates are judged on choosing the right metric for the business problem.

This chapter contains **58 questions**.

## Contents

- **12.1** [How is A/B testing used in machine learning?](#121-how-is-ab-testing-used-in-machine-learning)
- **12.2** [What are precision, recall, F1 score, and accuracy?](#122-what-are-precision-recall-f1-score-and-accuracy)
- **12.3** [What is the confusion matrix, and how do you interpret it?](#123-what-is-the-confusion-matrix-and-how-do-you-interpret-it)
- **12.4** [What are common evaluation metrics for Classification?](#124-what-are-common-evaluation-metrics-for-classification)
- **12.5** [When would you use accuracy vs other metrics?](#125-when-would-you-use-accuracy-vs-other-metrics)
- **12.6** [When would you use log loss vs accuracy?](#126-when-would-you-use-log-loss-vs-accuracy)
- **12.7** [What metrics would you use for a multi-class classification problem?](#127-what-metrics-would-you-use-for-a-multi-class-classification-problem)
- **12.8** [How do you handle class imbalance in classification metrics?](#128-how-do-you-handle-class-imbalance-in-classification-metrics)
- **12.9** [What is the ROC curve? What is AUC?](#129-what-is-the-roc-curve-what-is-auc)
- **12.10** [How do you handle imbalanced datasets?](#1210-how-do-you-handle-imbalanced-datasets)
- **12.11** [What are common evaluation metrics for Regression?](#1211-what-are-common-evaluation-metrics-for-regression)
- **12.12** [What's the difference between MAE, MSE, and RMSE?](#1212-whats-the-difference-between-mae-mse-and-rmse)
- **12.13** [How do you choose the right evaluation metric for a given problem?](#1213-how-do-you-choose-the-right-evaluation-metric-for-a-given-problem)
- **12.14** [How do you compare the performance of different models?](#1214-how-do-you-compare-the-performance-of-different-models)
- **12.15** [Explain cross-validation and its importance.](#1215-explain-cross-validation-and-its-importance)
- **12.16** [What is Hyperparameter Tuning?](#1216-what-is-hyperparameter-tuning)
- **12.17** [How do you evaluate unsupervised learning models?](#1217-how-do-you-evaluate-unsupervised-learning-models)
- **12.18** [How do you evaluate a clustering algorithm?](#1218-how-do-you-evaluate-a-clustering-algorithm)
- **12.19** [What metrics would you use for a recommendation system?](#1219-what-metrics-would-you-use-for-a-recommendation-system)
- **12.20** [What is A/B testing in the context of ML?](#1220-what-is-ab-testing-in-the-context-of-ml)
- **12.21** [What is LLM-as-a-judge?](#1221-what-is-llm-as-a-judge)
- **12.22** [Describe your approach to debugging an underperforming ML model.](#1222-describe-your-approach-to-debugging-an-underperforming-ml-model)
- **12.23** [How is a logistic regression model evaluated?](#1223-how-is-a-logistic-regression-model-evaluated)
- **12.24** [What is cross validation?](#1224-what-is-cross-validation)
- **12.25** [What is the confusion matrix?](#1225-what-is-the-confusion-matrix)
- **12.26** [What does the area under the ROC curve indicate?](#1226-what-does-the-area-under-the-roc-curve-indicate)
- **12.27** [What is sensitivity (Recall)?](#1227-what-is-sensitivity-recall)
- **12.28** [What is specificity?](#1228-what-is-specificity)
- **12.29** [How would you define Precision and Recall?](#1229-how-would-you-define-precision-and-recall)
- **12.30** [Why is the ROC in a curve important?](#1230-why-is-the-roc-in-a-curve-important)
- **12.31** [How would you describe an F1 score, and how would you use it?](#1231-how-would-you-describe-an-f1-score-and-how-would-you-use-it)
- **12.32** [What methods can be used to find the threshold of a classifier?](#1232-what-methods-can-be-used-to-find-the-threshold-of-a-classifier)
- **12.33** [Which is better to have? A false positive or false negative?](#1233-which-is-better-to-have-a-false-positive-or-false-negative)
- **12.34** [Why do we need the confusion matrix?](#1234-why-do-we-need-the-confusion-matrix)
- **12.35** [How is data divided in cross-validation?](#1235-how-is-data-divided-in-cross-validation)
- **12.36** [How do you measure the accuracy of a Clustering Algorithm?](#1236-how-do-you-measure-the-accuracy-of-a-clustering-algorithm)
- **12.37** [How do you measure the accuracy of a recommendation engine?](#1237-how-do-you-measure-the-accuracy-of-a-recommendation-engine)
- **12.38** [How can you measure the performance of a dimensionality reduction algorithm on your dataset?](#1238-how-can-you-measure-the-performance-of-a-dimensionality-reduction-algorithm-on-your-dataset)
- **12.39** [What is the ROC-AUC curve? List some of its benefits.](#1239-what-is-the-roc-auc-curve-list-some-of-its-benefits)
- **12.40** [What cross-validation technique would you use on a time series data set?](#1240-what-cross-validation-technique-would-you-use-on-a-time-series-data-set)
- **12.41** [Which among these is more important: model accuracy or model performance?](#1241-which-among-these-is-more-important-model-accuracy-or-model-performance)
- **12.42** [What is cross-validation and its types?](#1242-what-is-cross-validation-and-its-types)
- **12.43** [What are the different evaluation metrics for a regression model?](#1243-what-are-the-different-evaluation-metrics-for-a-regression-model)
- **12.44** [What are the different evaluation metrics for a classification model?](#1244-what-are-the-different-evaluation-metrics-for-a-classification-model)
- **12.45** [Difference between R2 and adjusted R2? Why do you prefer adjusted R2?](#1245-difference-between-r2-and-adjusted-r2-why-do-you-prefer-adjusted-r2)
- **12.46** [What is ROC-AUC, and what are its limitations?](#1246-what-is-roc-auc-and-what-are-its-limitations)
- **12.47** [When should you use PR-AUC over ROC-AUC?](#1247-when-should-you-use-pr-auc-over-roc-auc)
- **12.48** [How do you run a good ablation study?](#1248-how-do-you-run-a-good-ablation-study)
- **12.49** [When would you prefer F1, ROC-AUC, PR-AUC, log loss, or calibration error?](#1249-when-would-you-prefer-f1-roc-auc-pr-auc-log-loss-or-calibration-error)
- **12.50** [How does cross-validation fail when dealing with time-series data?](#1250-how-does-cross-validation-fail-when-dealing-with-time-series-data)
- **12.51** [What is stratified cross-validation, and when should you not use it?](#1251-what-is-stratified-cross-validation-and-when-should-you-not-use-it)
- **12.52** [Training accuracy is 99%, validation accuracy is 60%. How do you debug it?](#1252-training-accuracy-is-99-validation-accuracy-is-60-how-do-you-debug-it)
- **12.53** [Training and validation accuracy are both 50%. What could be wrong?](#1253-training-and-validation-accuracy-are-both-50-what-could-be-wrong)
- **12.54** [Your model performs well offline but poorly in production. Why?](#1254-your-model-performs-well-offline-but-poorly-in-production-why)
- **12.55** [Your model's accuracy decreases after adding more training data. Explain possible reasons.](#1255-your-models-accuracy-decreases-after-adding-more-training-data-explain-possible-reasons)
- **12.56** [A feature has extremely high importance. How do you determine whether it is legitimate or leakage?](#1256-a-feature-has-extremely-high-importance-how-do-you-determine-whether-it-is-legitimate-or-leakage)
- **12.57** [Your validation score varies significantly between folds. What does that indicate?](#1257-your-validation-score-varies-significantly-between-folds-what-does-that-indicate)
- **12.58** [Your model predicts only the majority class. How do you fix it?](#1258-your-model-predicts-only-the-majority-class-how-do-you-fix-it)

---

### 12.1 How is A/B testing used in machine learning?

A/B testing compares two versions of a model in a **live environment** to determine which performs better in practice.

---

#### How It Works

| Step | Description |
| --- | --- |
| **Version A (Control)** | The existing or baseline model currently in production |
| **Version B (Treatment)** | The new/updated model with different features, algorithms, or hyperparameters |
| **Traffic Split** | Incoming real-world user requests are randomly split between the two models |
| **Measurement** | Performance is compared using business or technical metrics |

**Common metrics used:**

- Conversion rate
- Latency
- Click-through rate (CTR)
- Prediction accuracy

---

#### Why It Matters

- **Real-World Validation** — Offline metrics (accuracy, F1 score) from lab testing don't always guarantee real-world success
- **Risk Mitigation** — Limits the blast radius of a bad update by exposing only a small % of users to the new model
- **Data-Driven Decisions** — Confirms whether a model change genuinely improves user experience or business outcomes before a full rollout

---

#### Quick Summary

> A/B testing = **Control vs. Treatment model** → **Random traffic split** → **Compare real metrics** → **Roll out the winner with confidence**

### 12.2 What are precision, recall, F1 score, and accuracy?

##### Precision, Recall, F1 Score, and Accuracy

These are standard metrics for evaluating classification models, built from four basic outcomes of a prediction.

##### The Confusion Matrix Foundation

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

---

##### Accuracy

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- The fraction of **all correct** predictions — both positives and negatives
- **Simple and intuitive**, but misleading on **imbalanced datasets**
- Example: if 95% of emails are "not spam," a model that always predicts "not spam" gets 95% accuracy while being useless at catching actual spam

---

##### Precision

`Precision = TP / (TP + FP)`

- Of all the items the model **predicted as positive**, what fraction were **actually positive**?
- Answer: "When the model says yes, how often is it right?"
- High precision = few false alarms (few false positives)
- Matters most when **false positives are costly** — e.g., a spam filter wrongly flagging an important email as spam

---

##### Recall (Sensitivity / True Positive Rate)

`Recall = TP / (TP + FN)`

- Of all the items that were **actually positive**, what fraction did the model **correctly identify**?
- Answers: "Of all the real positives, how many did the model catch?"
- High recall = few missed cases (few false negatives)
- Matters most when **false negatives are costly** — e.g., missing a cancer diagnosis, missing a fraudulent transaction

---

##### F1 Score

`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

- The **harmonic mean** of precision and recall — combines both into a single number
- Harmonic mean (rather than simple average) penalizes extreme imbalance between precision and recall — a model with 100% precision but 1% recall gets a low F1, not a mediocre-looking 50%
- Useful when you need a single balanced metric and both false positives and false negatives matter, especially on imbalanced datasets where accuracy is misleading

---

##### The Precision-Recall Trade-off

- These two metrics often pull in opposite directions, controlled by a model's decision threshold
- **Lowering the threshold** for predicting "positive" → catches more true positives (higher recall) but also more false positives (lower precision)
- **Raising the threshold** → fewer false positives (higher precision) but misses more true positives (lower recall)
- Choosing where to sit on this trade-off depends on the cost of each error type in the specific application

---

##### Quick Reference Table

| Metric | Formula | Answers | Best when... |
| --- | --- | --- | --- |
| Accuracy | (TP+TN)/(All) | How often is the model right overall? | Classes are balanced |
| Precision | TP/(TP+FP) | When it says positive, is it right? | False positives are costly |
| Recall | TP/(TP+FN) | Did it catch all actual positives? | False negatives are costly |
| F1 Score | 2·P·R/(P+R) | Balanced measure of both | Need one number, imbalanced data |

##### Relevance to LLM Evaluation

- These metrics apply directly to classification-style LLM tasks (e.g., content moderation, intent classification, entity extraction, spam/toxicity detection)
- For generative tasks (open-ended text generation), these metrics don't directly apply — different metrics are used instead (BLEU, ROUGE, perplexity, or LLM-as-judge evaluations), since there's no simple binary correct/incorrect classification to measure

### 12.3 What is the confusion matrix, and how do you interpret it?

##### Confusion Matrix

- A table showing a classifier's predictions vs. actual labels, broken into four outcomes (binary case):

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

**How to interpret:**

- Diagonal (TP, TN) = correct predictions
- Off-diagonal (FP, FN) = errors — shows _what kind_ of mistakes the model makes, not just how many
- Reveals issues hidden by accuracy alone (e.g., high accuracy but poor detection of the positive class on imbalanced data)
- Precision, recall, F1, and accuracy are all calculated directly from these four values

**Multi-class version:**

- Expands to an N×N grid; diagonal = correct, off-diagonal shows which specific classes get confused with each other

**Used for:** classification-style LLM tasks (moderation, intent detection, sentiment analysis) — not applicable to open-ended generation.

### 12.4 What are common evaluation metrics for Classification?

##### Common Classification Evaluation Metrics

- **Accuracy**: `(TP+TN)/Total` — overall correctness; misleading on imbalanced data
- **Precision**: `TP/(TP+FP)` — of predicted positives, how many were correct
- **Recall**: `TP/(TP+FN)` — of actual positives, how many were caught
- **F1 Score**: `2·P·R/(P+R)` — harmonic mean of precision & recall
- **Confusion Matrix**: table of TP/TN/FP/FN, foundation for above metrics
- **ROC-AUC**: plots True Positive Rate vs. False Positive Rate across thresholds; AUC = overall separability (1.0 = perfect, 0.5 = random)
- **PR-AUC**: precision vs. recall across thresholds; better than ROC-AUC for highly imbalanced data
- **Specificity**: `TN/(TN+FP)` — true negative rate, how well negatives are identified
- **Log Loss (Cross-Entropy)**: penalizes confident wrong predictions, uses predicted probabilities not just labels

**Choosing a metric depends on:** class balance, cost of false positives vs. false negatives, and whether probability calibration matters.

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

- **Confusion Matrix (N×N)**: shows exactly which classes get confused with which others
- **Cohen's Kappa**: agreement between predicted/actual, adjusted for chance
- **Matthews Correlation Coefficient (multi-class version)**: balanced single-score metric robust to imbalance
- **Top-k Accuracy**: correct if true label is among the top-k predicted classes (common in NLP/vision with many classes)

**Rule of thumb:**

- Balanced classes → macro-average or plain accuracy
- Imbalanced classes → weighted average, or macro if minority classes are important
- Need to see specific class confusions → confusion matrix

### 12.8 How do you handle class imbalance in classification metrics?

##### Handling Class Imbalance in Classification Metrics

**Choose the right metrics:**

- Avoid plain accuracy — misleading when one class dominates
- Use **Precision, Recall, F1** (especially per-class or macro-averaged) instead
- Use **PR-AUC** rather than ROC-AUC — more sensitive to minority class performance
- Use **Matthews Correlation Coefficient (MCC)** — robust single metric under imbalance
- Use **macro-averaging** (not micro) so minority classes aren't drowned out by majority class

**Handling the data/model itself (not just metrics):**

- **Resampling**: oversample minority class (e.g., SMOTE) or undersample majority class
- **Class weighting**: assign higher loss weight to minority class during training
- **Threshold tuning**: adjust decision threshold instead of using default 0.5, based on precision/recall trade-off needed
- **Stratified sampling**: ensure train/test splits preserve class proportions

**Reporting:**

- Always report **per-class** precision/recall/F1, not just an aggregate — reveals if the model silently fails on the minority class
- Confusion matrix is essential — shows exactly how minority class errors manifest

**Rule of thumb:** metrics alone don't fix imbalance — they just reveal it; combine imbalance-aware metrics with data-level or algorithm-level techniques.

### 12.9 What is the ROC curve? What is AUC?

##### ROC Curve

- **Receiver Operating Characteristic curve**: plots **True Positive Rate (Recall)** vs. **False Positive Rate** at every possible classification threshold
- `TPR = TP/(TP+FN)`, `FPR = FP/(FP+TN)`
- As the threshold changes, both TPR and FPR shift — the curve traces this trade-off across all thresholds

##### AUC (Area Under the Curve)

- A single number summarizing the ROC curve — the area under it, ranging from 0 to 1
- Represents the probability that the model ranks a random positive example higher than a random negative example

**Interpretation:**

- **AUC = 1.0**: perfect classifier (perfect separation)
- **AUC = 0.5**: no better than random guessing
- **AUC \< 0.5**: worse than random (predictions inverted)

**Why it's useful:**

- Threshold-independent — evaluates ranking quality across all decision thresholds, not just one fixed cutoff
- Good for comparing models regardless of what threshold will eventually be chosen

**Limitation:**

- Can be overly optimistic on **highly imbalanced data** — PR-AUC is preferred in that case, since ROC-AUC can look good even when the minority (positive) class is poorly predicted

### 12.10 How do you handle imbalanced datasets?

##### Handling Imbalanced Datasets

**Data-level techniques:**

- **Oversampling** minority class (random duplication, or synthetic like **SMOTE**)
- **Undersampling** majority class (random removal, or informed methods like Tomek links)
- **Data augmentation** for minority class (especially in images/text)
- **Collect more minority-class data** if possible

**Algorithm-level techniques:**

- **Class weighting**: assign higher loss penalty to minority class errors during training
- **Cost-sensitive learning**: explicitly define different misclassification costs per class
- **Ensemble methods**: bagging/boosting variants designed for imbalance (e.g., Balanced Random Forest, EasyEnsemble)
- **Anomaly detection framing**: treat rare class as an outlier detection problem instead of standard classification, if extremely rare

**Threshold/decision-level:**

- **Adjust classification threshold** instead of using default 0.5, based on desired precision/recall trade-off

**Evaluation-level:**

- Use **precision, recall, F1, PR-AUC, MCC** instead of accuracy
- **Stratified sampling** for train/test splits to preserve class ratios
- Report **per-class metrics**, not just aggregates

**Rule of thumb:** combine a data or algorithm-level fix with imbalance-aware evaluation metrics — neither alone is usually sufficient.

### 12.11 What are common evaluation metrics for Regression?

##### Common Regression Evaluation Metrics

- **MAE (Mean Absolute Error)**: `mean(|actual - predicted|)` — average magnitude of errors, easy to interpret, robust to outliers
- **MSE (Mean Squared Error)**: `mean((actual - predicted)²)` — penalizes larger errors more heavily due to squaring
- **RMSE (Root Mean Squared Error)**: `√MSE` — same units as the target variable, easier to interpret than MSE while still penalizing large errors
- **R² (R-squared / Coefficient of Determination)**: proportion of variance in the target explained by the model; ranges roughly 0–1 (can be negative for very poor fits); 1 = perfect fit
- **Adjusted R²**: R² adjusted for number of predictors — penalizes adding useless features, useful for comparing models with different feature counts
- **MAPE (Mean Absolute Percentage Error)**: average absolute error as a percentage of actual values — scale-independent, but unstable when actual values are near zero
- **Huber Loss**: combines MAE and MSE — quadratic for small errors, linear for large errors, robust to outliers while still differentiable

**Choosing between them:**

- **MAE**: when outliers shouldn't dominate the error signal
- **RMSE/MSE**: when large errors are especially undesirable and should be penalized more
- **R²**: when you want to know how well the model explains variance relative to a baseline
- **MAPE**: when comparing errors across different scales/units (e.g., forecasting)

**Rule of thumb:** use RMSE/MAE for absolute error magnitude, R² for explanatory power, MAPE for scale-independent comparison — and check for outliers before choosing, since MSE/RMSE are sensitive to them.

### 12.12 What's the difference between MAE, MSE, and RMSE?

##### MAE vs MSE vs RMSE

| Metric | Formula | Key Trait |
| --- | --- | --- |
| **MAE** | `mean(\|actual - predicted\|)` | Average absolute error; treats all errors linearly |
| **MSE** | `mean((actual - predicted)²)` | Squares errors → penalizes large errors much more; not in original units |
| **RMSE** | `√MSE` | Same as MSE but square-rooted back to original units — easier to interpret |

**Key differences:**

- **MAE** is robust to outliers — a single huge error contributes proportionally to its size
- **MSE/RMSE** amplify the effect of outliers — a large error contributes disproportionately due to squaring
- **MAE and RMSE** are both in the same units as the target variable; **MSE** is in squared units (harder to interpret directly)

**When to use which:**

- **MAE**: when all errors should be weighted equally, or outliers are expected/not particularly bad
- **MSE/RMSE**: when large errors are especially costly and should be penalized more heavily
- **RMSE over MSE**: almost always preferred for reporting/interpretation, since it's in the original scale

### 12.13 How do you choose the right evaluation metric for a given problem?

##### Choosing the Right Evaluation Metric

**1. Identify the problem type**

- Classification vs. Regression vs. Ranking vs. Generation — different metric families apply

**2. Consider class/data balance**

- Balanced → accuracy can work
- Imbalanced → precision/recall/F1/PR-AUC (classification), or check for outlier sensitivity (regression)

**3. Understand the cost of different errors**

- False positives costlier → prioritize **precision**
- False negatives costlier → prioritize **recall**
- Large errors much worse than small ones → **MSE/RMSE** (regression)
- All errors equally bad → **MAE** or **F1**

**4. Consider whether probability/confidence matters**

- Need calibrated probabilities (e.g., risk scoring) → **log loss**
- Only final decision matters → accuracy/F1

**5. Check if a single threshold will be used**

- Yes → precision/recall/F1 at that threshold
- No, need overall ranking quality → **ROC-AUC** (or **PR-AUC** if imbalanced)

**6. Match metric to business/stakeholder goals**

- Interpretability needed → MAE, accuracy, simple ratios
- Technical optimization → RMSE, log loss, F1

**7. Use multiple metrics together**

- No single metric tells the whole story — pair a threshold-dependent metric (F1) with a threshold-independent one (AUC), or report per-class breakdowns alongside an aggregate

**Rule of thumb:** start from what a wrong prediction _costs_ in the real-world use case, then pick the metric that best reflects minimizing

…that cost. The note above ends mid-sentence. The intended closing point: **the offline metric should be a proxy for the business cost of errors.** In practice:

- **Separate the optimization metric from the reporting metric.** Train on a differentiable loss (log loss, MSE), select models on the metric that reflects the cost (PR-AUC, recall@precision≥X, cost-weighted error), and report a business KPI (fraud $ caught, tickets deflected).
- **Use constraint-style metrics when there's a hard requirement.** For example, "maximize recall subject to precision ≥ 95%" or "maximize precision at a review capacity of 500 cases/day" (precision@k).
- **Cost-weighted evaluation.** With an explicit cost matrix (C_FP, C_FN), choose the threshold that minimizes `C_FP·FP + C_FN·FN`, not the default 0.5.
- **Ranking/recommendation:** NDCG@k, MAP, MRR. **Forecasting:** MAPE/sMAPE (beware zeros), or MASE. **LLM/generation:** task success rate, groundedness/faithfulness and human or LLM-as-judge preference, not just BLEU/ROUGE.
- **Validate the offline–online link.** Confirm with an A/B test that gains in the offline metric move the online KPI. If they don't, the metric is the wrong proxy.

### 12.14 How do you compare the performance of different models?

##### Comparing Model Performance

**1. Use consistent, relevant metrics**

- Same metric(s) across all models, chosen based on the problem's cost structure (see prior answer)
- Use multiple metrics together (e.g., F1 + AUC) rather than relying on one number

**2. Use the same evaluation data**

- Same test/validation set, same preprocessing, same class distribution — otherwise comparisons aren't fair

**3. Check statistical significance**

- Small differences in a metric could be noise — use techniques like **cross-validation** (report mean ± std across folds) or statistical tests (e.g., paired t-test, McNemar's test for classifiers) to confirm a difference is real, not random variation

**4. Consider computational cost, not just accuracy metrics**

- Inference latency, memory footprint, training cost — a slightly less accurate but much faster/cheaper model may be preferable in production

**5. Evaluate on multiple slices, not just the aggregate**

- Per-class or per-subgroup performance (e.g., does Model A do better on minority classes than Model B?) — an aggregate score can hide important differences

**6. Check calibration, not just discrimination**

- Two models can have similar accuracy/AUC but very different confidence calibration (use log loss or calibration curves to check)

**7. Consider robustness**

- Test on out-of-distribution or edge-case data, not just the standard test set — a model that's marginally better on average but fragile to distribution shift may be riskier

**Rule of thumb:** compare using the same data/metrics, validate differences are statistically meaningful, and weigh accuracy trade-offs against cost, latency, and robustness for the actual deployment context.

### 12.15 Explain cross-validation and its importance.

##### Cross-Validation

- A technique for evaluating model performance by splitting data into multiple train/test partitions, rather than relying on a single train/test split

**Most common form: K-Fold Cross-Validation**

- Split data into `k` equal folds
- Train on `k-1` folds, test on the remaining fold
- Repeat `k` times, each fold used once as the test set
- Average the results across all `k` runs for a final performance estimate

**Other variants:**

- **Stratified K-Fold**: preserves class proportions in each fold — important for imbalanced data
- **Leave-One-Out (LOOCV)**: k = number of samples; each sample tested individually (expensive, used for small datasets)
- **Time Series Cross-Validation**: respects temporal order (train on past, test on future) — no random shuffling

##### Why It's Important

- **More reliable performance estimate**: a single train/test split can give a lucky or unlucky result depending on which data ends up in the test set; averaging across folds reduces this variance
- **Uses data more efficiently**: every sample gets used for both training and testing across folds, valuable when data is limited
- **Detects overfitting**: large gaps between training and cross-validation performance signal the model isn't generalizing
- **Enables fair model/hyperparameter comparison**: comparing average CV scores is more trustworthy than comparing single-split results

**Rule of thumb:** always prefer cross-validation over a single train/test split when data is limited or when making important model-selection/hyperparameter decisions.

### 12.16 What is Hyperparameter Tuning?

##### Hyperparameter Tuning

- The process of finding the best set of **hyperparameters** — settings configured _before_ training that control the learning process itself (not learned from data, unlike model weights)
- Examples: learning rate, batch size, number of layers, regularization strength, tree depth, number of epochs

##### Why It Matters

- Model performance can vary significantly depending on hyperparameter choices — poor settings can cause underfitting, overfitting, slow convergence, or training instability
- The "best" hyperparameters differ per dataset/task — no universal default works everywhere

##### Common Tuning Methods

**1. Grid Search**

- Exhaustively tries every combination from a predefined set of values for each hyperparameter
- Simple, guaranteed to check all specified combos, but computationally expensive as the search space grows

**2. Random Search**

- Randomly samples combinations from the hyperparameter space
- Often more efficient than grid search, especially when only a few hyperparameters actually matter

**3. Bayesian Optimization**

- Builds a probabilistic model of how hyperparameters affect performance, using past results to intelligently choose the next combination to try
- More sample-efficient than grid/random search, good for expensive-to-train models

**4. Successive Halving / Hyperband**

- Starts many configurations with limited resources (e.g., few epochs), progressively eliminates poor performers, and allocates more resources to promising ones
- Efficient for large search spaces with costly training runs

**5. Manual tuning**

- Based on experience/intuition, adjusting one hyperparameter at a time based on observed results

##### Best Practices

- Always tune using a **validation set** (or cross-validation) — never the test set, to avoid overfitting to test data
- Search space and method should scale with compute budget — Bayesian/Hyperband for expensive models, grid/random search for cheaper ones

### 12.17 How do you evaluate unsupervised learning models?

##### Evaluating Unsupervised Learning Models

Harder than supervised evaluation since there are no ground-truth labels to compare against.

**Internal metrics (no labels needed, based on data structure alone):**

- **Silhouette Score**: measures how similar a point is to its own cluster vs. other clusters; ranges -1 to 1, higher is better
- **Davies-Bouldin Index**: ratio of within-cluster to between-cluster distances; lower is better
- **Inertia / Within-Cluster Sum of Squares (WCSS)**: used in the **Elbow Method** to help pick number of clusters (k) in k-means
- **Calinski-Harabasz Index**: ratio of between-cluster to within-cluster dispersion; higher is better

**External metrics (when ground-truth labels exist, for validation purposes):**

- **Adjusted Rand Index (ARI)**: measures similarity between predicted clusters and true labels, adjusted for chance
- **Normalized Mutual Information (NMI)**: measures shared information between predicted clusters and true labels
- **Homogeneity, Completeness, V-measure**: assess whether clusters contain only one class (homogeneity) and whether all members of a class are in one cluster (completeness)

**For dimensionality reduction (e.g., PCA, autoencoders):**

- **Reconstruction error**: how well original data can be recovered from the reduced representation
- **Explained variance ratio**: how much of the original data's variance is captured

**For anomaly detection:**

- Precision/recall/F1 if some labeled anomalies exist for validation
- Otherwise, domain expert review of flagged anomalies

**Practical approach:**

- Visual inspection (t-SNE/UMAP plots, cluster visualization)
- Domain expert validation of results
- Stability check: does the same method give consistent results on data subsets/resamples?

**Rule of thumb:** use internal metrics to compare configurations of the same algorithm, external metrics if any labels exist for sanity-checking, and always pair metrics with domain expert judgment since "correctness" is inherently more subjective without ground truth.

### 12.18 How do you evaluate a clustering algorithm?

##### Evaluating a Clustering Algorithm

**Internal metrics (no ground truth needed):**

- **Silhouette Score**: how similar a point is to its own cluster vs. nearest other cluster; range -1 to 1, higher = better-defined clusters
- **Davies-Bouldin Index**: ratio of within-cluster to between-cluster distance; lower = better
- **Calinski-Harabasz Index**: between-cluster vs. within-cluster dispersion ratio; higher = better
- **Inertia (WCSS)**: sum of squared distances to cluster centroids; used in **Elbow Method** to pick optimal k

**External metrics (if true labels are available):**

- **Adjusted Rand Index (ARI)**: agreement between predicted and true clusters, adjusted for chance
- **Normalized Mutual Information (NMI)**: shared information between predicted and true labels
- **Homogeneity / Completeness / V-measure**: purity of clusters vs. whether same-class points stay together

**Practical/qualitative checks:**

- Visualize clusters (t-SNE, UMAP, PCA projection)
- Check cluster stability across different runs/data subsets
- Domain expert review — do clusters make sense practically?

**Choosing k (number of clusters):**

- Elbow method (inertia), silhouette analysis across different k values, or domain knowledge

**Rule of thumb:** combine an internal metric (like silhouette score) with visual inspection and domain validation — clustering "correctness" is inherently more subjective without ground truth labels.

### 12.19 What metrics would you use for a recommendation system?

##### Metrics for Recommendation Systems

**Accuracy-based (rating prediction):**

- **RMSE / MAE**: how close predicted ratings are to actual ratings

**Ranking-based (top-N recommendations):**

- **Precision@K**: of top-K recommended items, how many were relevant
- **Recall@K**: of all relevant items, how many appeared in top-K
- **MAP (Mean Average Precision)**: average precision across users, rewards relevant items ranked higher
- **NDCG (Normalized Discounted Cumulative Gain)**: rewards relevant items appearing higher in the ranked list, accounts for graded relevance
- **MRR (Mean Reciprocal Rank)**: focuses on rank of the _first_ relevant item

**Beyond accuracy (business/UX-relevant):**

- **Coverage**: fraction of catalog the system actually recommends (avoids over-recommending only popular items)
- **Diversity**: how different recommended items are from each other within a list
- **Novelty/Serendipity**: how non-obvious or surprising good recommendations are
- **Click-Through Rate (CTR)**: real-world engagement with recommendations
- **Conversion Rate**: recommendations leading to actual purchases/actions

**Online evaluation:**

- **A/B testing**: measures real business impact (revenue, engagement, retention) — the ultimate test beyond offline metrics

### 12.20 What is A/B testing in the context of ML?

##### A/B Testing in ML

- A controlled experiment comparing two (or more) versions of a model/system by randomly splitting real users/traffic between them, measuring which performs better on real-world outcomes

##### How It Works

- **Group A (control)**: existing model/system
- **Group B (treatment)**: new model/system being tested
- Users are randomly assigned to one group, unaware of which version they're seeing
- Measure a target metric (e.g., click-through rate, conversion, revenue, engagement) for each group over a set period
- Use statistical significance testing to determine if the difference is real, not due to random chance

##### Why It's Used in ML

- **Offline metrics don't always predict real-world impact**: a model with better offline accuracy/AUC/NDCG might not actually improve user engagement or business outcomes once deployed
- **Ground truth for business impact**: directly measures what matters (revenue, retention, satisfaction) rather than a proxy metric
- **De-risks deployment**: catches unexpected negative effects before a full rollout

##### Key Considerations

- **Sample size**: needs enough users/traffic to detect a statistically significant difference
- **Randomization**: must be truly random to avoid selection bias
- **Duration**: long enough to capture representative behavior (avoid weekday/weekend effects, novelty effects)
- **Statistical significance**: use proper hypothesis testing (e.g., t-test, chi-squared) and account for multiple comparisons if testing many metrics
- **Guardrail metrics**: monitor for unintended negative side effects, not just the primary target metric

### 12.21 What is LLM-as-a-judge?

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

### 12.22 Describe your approach to debugging an underperforming ML model.

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

### 12.23 How is a logistic regression model evaluated?

#### Evaluation of Logistic Regression(supervised, binary classification)

- **Accuracy:**

  - Measures the overall percentage of correctly classified observations.
  - (TP + TN) / ( TP + TN + FP + FN )
- **Precision (P):**

  - Measures how precise my positive predictions are.
  - TP / (TP + FP)
  - Matters most when false positives are costly
- **Recall (Sensitivity, R):**

  - Measures how many of the actual positives did I find/recall
  - TP / (TP + FN)
  - Matters most when false negatives are costly
- **F1-Score:**

  - Harmonic mean of precision and recall; useful for imbalanced datasets.
  - 2 \* P \* R / ( P + R )
- **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve, binary classification):**

  - ROC
    - **Receiver Operating Characteristic (ROC)** is a graph.
    - It plots the **True Positive Rate (TPR)** (sensitivity or recall) on the y-axis.
    - It plots the **False Positive Rate (FPR)** (1 - specificity) on the x-axis.
    - It shows how a model trades off correct positive guesses against false alarms as you change the decision boundary.
  - AUC
    - **Area Under the Curve (AUC)** measures the entire two-dimensional area underneath the ROC curve.
    - It gives a single number score from **0 to 1**.
    - It tells you the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example
  - **Score Interpretation**
    - **1.0**: A perfect model.
    - **0.9 to 1.0**: Excellent performance.
    - **0.8 to 0.9**: Good/acceptable performance.
    - **0.7 to 0.8**: Fair performance.
    - **0.5**: Random guessing (no better than a coin flip).
    - **Below 0.5**: Worse than random guessing (often means labels are inverted)
- **Log Loss (Cross-Entropy):**

  - Evaluates the quality of predicted probabilities
  - Lower is better.
- **Confusion Matrix:**

  - Shows **TP, TN, FP, and FN** to understand classification errors.

  |  | **Predicted Positive** | **Predicted Negative** |
  | --- | --- | --- |
  | **Actual Positive** | **TP** – True Positive | **FN** – False Negative (**Type II Error**) |
  | **Actual Negative** | **FP** – False Positive (**Type I Error**) | **TN** – True Negative |

### 12.24 What is cross validation?

#### Cross-Validation

- **Cross-validation** is a technique used to evaluate a machine learning model by dividing the dataset into multiple subsets (**folds**).
- The model is trained on some folds and **validated on the remaining fold**.
- This process is repeated so that each fold is used for validation.
- The results are then **averaged** to obtain a more reliable estimate of model performance.
- It helps detect **overfitting** and is commonly used for **model selection and hyperparameter tuning**.

#### Example: 5-Fold Cross-Validation

| Iteration | Training Data | Validation Data |
| --- | --- | --- |
| 1 | Folds 2–5 | Fold 1 |
| 2 | Folds 1, 3–5 | Fold 2 |
| 3 | Folds 1–2, 4–5 | Fold 3 |
| 4 | Folds 1–3, 5 | Fold 4 |
| 5 | Folds 1–4 | Fold 5 |

### 12.25 What is the confusion matrix?

> **See also:** [Question 8.41 — What is Confusion Matrix?](08-machine-learning-fundamentals.md#841-what-is-confusion-matrix), which covers the same question from another angle.

A **Confusion Matrix** is a table used to evaluate the performance of a **classification model** by comparing the model's predicted classes with the actual classes.

#### Binary Classification

|  | **Predicted Positive** | **Predicted Negative** |
| --- | --- | --- |
| **Actual Positive** | **True Positive (TP)** | **False Negative (FN)** |
| **Actual Negative** | **False Positive (FP)** | **True Negative (TN)** |

#### Four components

- **True Positive (TP):** Model predicts positive, and it is actually positive.
- **True Negative (TN):** Model predicts negative, and it is actually negative.
- **False Positive (FP):** Model predicts positive, but it is actually negative.
- **False Negative (FN):** Model predicts negative, but it is actually positive.

#### Example

For a **spam email classifier**:

- **TP:** Spam correctly identified as spam.
- **TN:** Normal email correctly identified as normal.
- **FP:** Normal email incorrectly classified as spam.
- **FN:** Spam incorrectly classified as normal.

From these four values, we can calculate **Accuracy, Precision, Recall, and F1-score**.

### 12.26 What does the area under the ROC curve indicate?

- **AUC (Area Under the ROC Curve)** indicates how well a classification model can **distinguish between positive and negative classes**.
- AUC ranges from **0 to 1**:
  - **1.0** → Perfect classifier
  - **0.5** → Random guessing
  - **Below 0.5** → Worse than random
- **Higher AUC means better ability to separate the classes** across different classification thresholds.

**In simple terms:** AUC tells us **how well the model separates the two classes**.

### 12.27 What is sensitivity (Recall)?

- **Sensitivity**, also called **Recall** or **True Positive Rate (TPR)**
- How many of the actual positives did I recall/find?
- TP / (TP + FN)
- High sensitivity means the model produces **fewer false negatives**

**Example:**

If there are 100 actual positive cases and the model correctly finds 90 of them, the

**Recall/Sensitivity = 90%**

### 12.28 What is specificity?

- **Specificity**, also called **True Negative Rate (TNR)**
- **How many of the actual negatives did I** find?
- **TN / (TN + FP)**
- High specificity means the model produces **fewer false positives**.

**Example:**

If there are 100 actual negative cases and the model correctly identifies 90 of them, the **Specificity = 90%**.

### 12.29 How would you define Precision and Recall?

#### Precision

- **Precision** measures how precise my positive prediction are?
- Formula: **TP / (TP + FP)**
- High precision means **fewer false positives**.

#### Recall

- **Recall** is also called **Sensitivity**.
- It measures how many of the **actual positives did I recall/find?**
- Formula: **TP / (TP + FN)**
- High recall means **fewer false negatives**.

### 12.30 Why is the ROC in a curve important?

#### Why is the ROC Curve Important?

- **ROC (Receiver Operating Characteristic) curve** is used to evaluate the performance of a **binary classification model**.
- It shows the trade-off between:
  - **True Positive Rate (TPR / Recall)** — how many actual positives are correctly identified.
  - **False Positive Rate (FPR)** — how many actual negatives are incorrectly classified as positive.
- The ROC curve is useful because it evaluates the model across **different classification thresholds**, rather than using only one threshold.
- A curve closer to the **top-left corner** generally indicates better classification performance.
- **AUC (Area Under the ROC Curve)** summarizes the ROC curve into a single value:
  - **1.0** → Perfect classifier
  - **0.5** → Random guessing
  - **Below 0.5** → Worse than random

### 12.31 How would you describe an F1 score, and how would you use it?

#### F1 Score

- **F1 Score** is a classification metric that combines **Precision and Recall** into a single score.
- It is the **harmonic mean of Precision and Recall**.
- F1 Score ranges from **0 to 1**:
  - **1** → Perfect Precision and Recall
  - **0** → Very poor performance
- It is especially useful when:
  - The dataset is **imbalanced**.
  - Both **false positives and false negatives** are important.
  - You want a balance between **Precision and Recall**.

##### How to Use It

- Use **F1 Score** when you need a single metric that considers both Precision and Recall.
- A **higher F1 Score** generally indicates better classification performance.
- If Precision and Recall are very different, the F1 Score will be closer to the **lower value**.

**In simple terms:** **F1 Score tells you how well a classification model balances Precision and Recall.**

**One-line interview answer:**

**F1 Score is the harmonic mean of Precision and Recall, commonly used when both false positives and false negatives matter, especially with imbalanced datasets.**

### 12.32 What methods can be used to find the threshold of a classifier?

#### Methods to Find the Threshold of a Classifier

The **classification threshold** determines when a predicted probability is converted into a class label.

Common methods include:

- **Default threshold (0.5):** Predict positive if probability ≥ 0.5; otherwise negative.
- **ROC Curve:** Choose the threshold that provides a good balance between **True Positive Rate (TPR)** and **False Positive Rate (FPR)**.
- **Youden’s J Statistic:** Selects the threshold that maximizes **TPR − FPR**.
- **Precision-Recall Curve:** Useful for imbalanced datasets; choose a threshold that provides the desired **precision-recall trade-off**.
- **F1 Score:** Select the threshold that gives the highest **F1 score** when balancing precision and recall is important.
- **Cost-Based Thresholding:** Choose a threshold based on the different costs of **false positives and false negatives**.
- **Business/Application Requirements:** Set the threshold according to the desired level of precision, recall, sensitivity, or specificity.

### 12.33 Which is better to have? A false positive or false negative?

#### False Positive vs False Negative

- **False Positive (FP):** The model says **“Yes”**, but the correct answer is **“No.”**
  - Example: A healthy person is predicted to have a disease.
- **False Negative (FN):** The model says **“No”**, but the correct answer is **“Yes.”**
  - Example: A person with a disease is predicted to be healthy.

#### Which is better?

- It **depends on the situation**.
- If missing a positive case is dangerous, **False Positive is better**.
  - Example: Disease detection → better to have a false alarm than miss a sick person.
- If false alarms are more costly, **False Negative may be better**.

**Simple terms:**

**False Positive = false alarm.**

**False Negative = missed positive case.**

The better one depends on which mistake is less costly.

### 12.34 Why do we need the confusion matrix?

- A **confusion matrix** helps evaluate the performance of a **classification model**.
- It shows how many predictions were:
  - **True Positive (TP):** Correctly predicted positive.
  - **True Negative (TN):** Correctly predicted negative.
  - **False Positive (FP):** Predicted positive but actually negative.
  - **False Negative (FN):** Predicted negative but actually positive.
- It helps identify **what types of mistakes** the model is making.
- It is used to calculate important metrics such as **Accuracy, Precision, Recall, and Specificity**.

### 12.35 How is data divided in cross-validation?

#### How Is Data Divided in Cross-Validation?

- In **k-fold cross-validation**, the dataset is divided into **k equal or nearly equal parts called folds**.
- The model is trained on **k − 1 folds** and tested on the **remaining fold**.
- This process is repeated **k times**, so every fold is used once as the validation set.
- The performance from all folds is then **averaged** to get the final evaluation.

**Example: 5-Fold Cross-Validation**

- Divide data into **5 folds**.
- Train on 4 folds and validate on 1 fold.
- Repeat this **5 times**, changing the validation fold each time.
- Average the 5 results.

### 12.36 How do you measure the accuracy of a Clustering Algorithm?

Clustering has no single "accuracy" because there is usually no ground truth. Use two families of metrics:

**Internal metrics (no labels)** — judge compactness and separation:
- **Silhouette score** ($-1$ to $1$): $(b-a)/\max(a,b)$, where $a$ is mean intra-cluster distance and $b$ the mean distance to the nearest other cluster. Higher is better.
- **Davies–Bouldin index**: average similarity between each cluster and its most similar one. Lower is better.
- **Calinski–Harabasz**: ratio of between-cluster to within-cluster dispersion. Higher is better.
- **Inertia / WCSS**: used in the elbow method; always decreases with $k$, so only useful comparatively.

**External metrics (labels available, e.g. a labelled sample)**:
- **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)**: compare the clustering with true labels and are invariant to label permutation.
- **Purity, homogeneity, completeness, V-measure**.

**Caveats:** internal metrics favour convex, spherical clusters (they will rate DBSCAN's odd-shaped clusters poorly), and they depend on the distance metric and on feature scaling.

**Production angle:** the real test is usually downstream. Do customer segments behave differently? Do they lift campaign conversion? Is the clustering stable across bootstrap samples or time windows (measure ARI between runs)? Have a domain expert review samples from each cluster.

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score
silhouette_score(X, labels)          # internal
adjusted_rand_score(y_true, labels)  # external
```

### 12.37 How do you measure the accuracy of a recommendation engine?

Evaluate a recommender at three levels: offline, online and business.

**Offline (historical data, time-based split):**
- **Rating prediction:** RMSE/MAE (mostly outdated; users care about rankings, not exact ratings).
- **Top-K ranking:** Precision@K, Recall@K, Hit Rate@K, **MAP**, **MRR**, **NDCG@K** (rewards placing relevant items higher).
- **Beyond accuracy:** coverage (share of the catalogue ever recommended), diversity (intra-list dissimilarity), novelty, serendipity, popularity bias, fairness across item providers.

**Online (the real test):**
- **A/B tests** on CTR, conversion, watch time, add-to-cart, revenue per session, retention; interleaving for faster, more sensitive ranking comparisons.
- Guardrail metrics: latency, complaints, unsubscribes.

**Pitfalls:**
- **Random splits leak the future**; split by time.
- Logged data is biased by the previous recommender (exposure/selection bias); counterfactual estimators (IPS) help offline.
- Offline gains often don't carry over online, so treat offline metrics as a filter and A/B tests as the decision.
- Optimizing CTR alone can promote clickbait; balance it with long-term engagement metrics.

```python
def ndcg_at_k(rels, k):
    import numpy as np
    rels = np.asarray(rels)[:k]
    dcg = (rels / np.log2(np.arange(2, rels.size + 2))).sum()
    ideal = np.sort(rels)[::-1]
    idcg = (ideal / np.log2(np.arange(2, ideal.size + 2))).sum()
    return dcg / idcg if idcg else 0.0
```

### 12.38 How can you measure the performance of a dimensionality reduction algorithm on your dataset?

It depends on *why* you are reducing dimensions:

**1. Downstream task performance (most important in practice).** Train the same model on the reduced and original features and compare the validation metric, training and inference time, and memory. If accuracy holds while latency drops, the reduction worked. Put the reducer inside the CV pipeline to avoid leakage.

**2. Information retained.**
- **PCA:** explained variance ratio, cumulative variance (e.g. keep 95%), scree plot.
- **Reconstruction error:** project down and back up ($\hat X = $ `inverse_transform`) and compute MSE. This works for PCA, autoencoders and kernel PCA (approximate pre-image).

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

### 12.39 What is the ROC-AUC curve? List some of its benefits.

The **ROC curve** plots the **True Positive Rate** (recall) against the **False Positive Rate** ($FP/(FP+TN)$) as the classification threshold sweeps from 1 to 0. **AUC** is the area under it: the probability that the model ranks a random positive above a random negative. 0.5 is random and 1.0 is perfect.

**Benefits:**
- **Threshold-independent:** summarizes ranking quality over all thresholds, which is useful before the business has picked an operating point.
- **Scale-invariant:** depends only on the ranking of scores, not on calibration.
- **Insensitive to class prior** in the sense that TPR and FPR are each computed within a class, so AUC is comparable across datasets with different base rates.
- Gives a single number for model comparison and model selection.
- The curve itself helps choose a threshold based on the cost of FP vs FN (e.g. Youden's J, or a cost-weighted point).

**Limitations to mention:**
- With **heavy class imbalance**, a tiny FPR can still mean many false positives, so ROC-AUC looks optimistic. Prefer **PR-AUC** when positives are rare (fraud, disease).
- It ignores calibration: a model can have great AUC and badly calibrated probabilities.
- It weights all thresholds equally, though only a narrow region matters in production. Use partial AUC or precision at a fixed recall instead.

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_true, y_score)
fpr, tpr, thr = roc_curve(y_true, y_score)
```

### 12.40 What cross-validation technique would you use on a time series data set?

Standard k-fold shuffles the rows, so the model trains on the future and tests on the past (leakage), and autocorrelation inflates the scores. Use **order-preserving** schemes:

- **Expanding-window (forward chaining / rolling-origin):** train on $[t_0, t_k]$, test on $(t_k, t_{k+h}]$, then grow the training window. This is scikit-learn's `TimeSeriesSplit`, and it mimics periodic retraining in production.
- **Sliding (rolling) window:** fixed-length training window that moves forward. Better when old data is stale or the process is non-stationary.
- **Gap / embargo:** leave a gap between train and test equal to the label horizon or the feature lookback, so overlapping windows don't leak (common in finance; **purged k-fold** from López de Prado).
- **Blocked CV:** contiguous blocks with gaps, when you need more folds.
- **Grouped temporal splits** for many series (e.g. per store): split by time across all series, and optionally also hold out whole series to test cold start.

**Best practices:**
- Match the test window length to the real forecast horizon.
- Fit all preprocessing and feature engineering inside each fold.
- Keep a final out-of-time holdout and check metric stability across folds (seasonality).

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=30, gap=7)
for tr, te in tscv.split(X):
    model.fit(X[tr], y[tr]); score(model, X[te], y[te])
```

### 12.41 Which among these is more important: model accuracy or model performance?

It depends on how you define the terms, and that is the point of the question.

- **Model accuracy** is one specific metric: the fraction of correct predictions.
- **Model performance** is broader: how well the model serves its purpose. That includes the *right* metric for the problem (precision/recall, F1, AUC, RMSE, calibration) and also latency, throughput, cost, robustness, fairness and business impact.

**Performance matters more**, because accuracy can be misleading:
- **Imbalanced data:** with 1% fraud, predicting "not fraud" always gives 99% accuracy and catches nothing. Recall, precision and PR-AUC tell the real story.
- **Asymmetric costs:** a missed cancer diagnosis (FN) costs far more than a false alarm (FP), so optimize recall at an acceptable precision, not accuracy.
- **Operational constraints:** a model that is 0.5% more accurate but 10× slower may break a 50 ms SLA or blow the inference budget. Fraud scoring, ad ranking and search all have hard latency limits.
- **Business outcome:** the goal is revenue, retention or reduced risk. A slightly less accurate but interpretable, stable model may win in regulated settings.

**How I'd frame it in practice:** agree on the success metric with stakeholders up front (e.g. "recall ≥ 90% at ≤ 5% FPR, p95 latency < 100 ms"), select models on that, and validate with an A/B test on the business KPI.

### 12.42 What is cross-validation and its types?

**Cross-validation** estimates how well a model generalizes by repeatedly splitting the data into training and validation parts, training on one and evaluating on the other, then averaging. It gives a lower-variance estimate than a single holdout split and is used for model selection and hyperparameter tuning.

**Types:**
- **Holdout:** one train/validation split. Fast, but a noisy estimate.
- **k-Fold:** split into $k$ folds (5 or 10 typical) and rotate which fold is validation. Every point is validated exactly once.
- **Stratified k-Fold:** preserves class proportions in each fold. The default for classification, especially with imbalance.
- **Leave-One-Out (LOOCV):** $k = n$. Nearly unbiased but high variance and expensive; for tiny datasets.
- **Leave-P-Out:** all combinations of $p$ held-out points; combinatorially expensive.
- **Repeated (stratified) k-Fold:** repeat with different shuffles for a more stable estimate.
- **Group k-Fold / Leave-One-Group-Out:** keeps all rows of a group (patient, user, document) in the same fold to prevent entity leakage.
- **Time-series split (forward chaining, rolling window, with a gap or purge):** respects temporal order.
- **Nested CV:** an inner loop tunes hyperparameters and an outer loop estimates performance, giving an unbiased estimate when tuning.
- **Monte Carlo (shuffle-split):** random repeated train/test splits.

**Key rule:** fit all preprocessing inside each fold (use a `Pipeline`), and choose the split strategy that mirrors how the model will face new data in production.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc")
```

### 12.43 What are the different evaluation metrics for a regression model?

**Common regression metrics:**

| Metric | Formula (idea) | Notes |
| --- | --- | --- |
| **MAE** | mean of $\lvert y-\hat y\rvert$ | Same units as target, robust to outliers, median-optimal |
| **MSE** | mean of $(y-\hat y)^2$ | Penalizes large errors heavily; differentiable, used as a loss |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as target, still outlier-sensitive |
| **$R^2$** | $1-\frac{SS_{res}}{SS_{tot}}$ | Share of variance explained; can be negative on test data |
| **Adjusted $R^2$** | penalizes extra features | For comparing linear models with different feature counts |
| **MAPE** | mean of $\lvert (y-\hat y)/y\rvert$ | Scale-free and business-friendly, but explodes near $y=0$ and is asymmetric |
| **sMAPE / WAPE** | normalized variants | Safer alternatives for forecasting |
| **RMSLE** | RMSE on $\log(1+y)$ | Relative errors; good for targets spanning orders of magnitude (prices, counts) |
| **Huber / quantile loss** | hybrid / asymmetric | Robust fitting; quantile (pinball) loss for prediction intervals |

**How to choose:** pick the metric that matches the cost of errors. If being off by 10 units costs the same whatever the size, use MAE. If big misses are disproportionately bad (inventory stock-outs), use RMSE. If relative error matters (demand across products of very different volume), use WAPE or RMSLE. Always report a baseline (predict the mean or last value) alongside the metric so the number has context.

### 12.44 What are the different evaluation metrics for a classification model?

**Threshold-based (from the confusion matrix):**

- **Accuracy** = (TP+TN)/all. Misleading with class imbalance.
- **Precision** = TP/(TP+FP): of what we flagged, how much was right. Use when false positives are costly (spam filter hiding real mail).
- **Recall / sensitivity** = TP/(TP+FN): how much of the positive class we caught. Use when misses are costly (fraud, cancer screening).
- **Specificity** = TN/(TN+FP).
- **F1** = harmonic mean of precision and recall; **F-beta** weights recall $\beta$ times as much.
- **MCC (Matthews correlation)**: balanced single number that stays honest under imbalance.
- **Cohen's kappa**: agreement corrected for chance.

**Threshold-free (ranking quality):**

- **ROC-AUC**: probability a random positive is ranked above a random negative. Can look optimistic with heavy imbalance.
- **PR-AUC / average precision**: focuses on the positive class; preferred for rare events.

**Probability quality:**

- **Log loss (cross-entropy)**: punishes confident wrong predictions.
- **Brier score** and **calibration error (ECE)**: whether a 0.8 score really means 80%.

**Multiclass:** macro averaging (each class equal), micro (each sample equal), weighted; plus top-k accuracy and per-class confusion matrices.

**Production angle:** choose the metric that maps to business cost, then pick the operating threshold on a validation set (e.g. maximize recall subject to precision >= 95%). Track it per segment, not only globally.

### 12.45 Difference between R2 and adjusted R2? Why do you prefer adjusted R2?

**$R^2$** measures the fraction of the target's variance explained by the model:

$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

Its problem: on training data, **adding any feature never decreases $R^2$**, even a column of random noise, because least squares can always set its coefficient to zero or exploit chance correlation.

**Adjusted $R^2$** penalizes the number of predictors $p$ relative to sample size $n$:

$R^2_{adj} = 1 - (1-R^2)\frac{n-1}{n-p-1}$

It only increases if a new feature improves the fit more than would be expected by chance; useless features make it drop.

**Why prefer adjusted $R^2$:**

- Fair comparison of linear models with **different numbers of features**.
- Guards against overfitting by feature stuffing, especially when $p$ is not small relative to $n$.

**Caveats:** both are in-sample measures for linear models. For real model selection, held-out metrics (cross-validated RMSE/MAE) or AIC/BIC are more reliable, and neither $R^2$ tells you whether the model is unbiased or whether residual assumptions hold. With $n \gg p$ the two are nearly identical.

### 12.46 What is ROC-AUC, and what are its limitations?

**ROC curve:** plots **TPR (recall)** against **FPR** = FP/(FP+TN) as the classification threshold sweeps from 1 to 0. **AUC** is the area under it. It equals the probability that a randomly chosen positive is scored higher than a randomly chosen negative (equivalent to the Mann-Whitney U statistic). 0.5 is random and 1.0 is perfect.

**Why it's popular:** threshold-independent, scale-invariant (only ranking matters), and insensitive to class prior, so it is comparable across datasets with different base rates.

**Limitations**

- **Misleading under heavy class imbalance:** FPR's denominator (TN) is huge, so thousands of false positives barely move FPR. A fraud model can show 0.98 AUC while precision at a usable threshold is 5%. Use **PR-AUC** there.
- **Ranking only, not calibration:** a model with good AUC can output badly calibrated probabilities. If you need risk scores or expected-cost decisions, also check log loss, Brier score or ECE.
- **Summarizes all thresholds, including irrelevant ones:** in practice you operate at one point (e.g., FPR < 1%). Two models with equal AUC can differ a lot in that region, so use **partial AUC** or TPR@fixed-FPR.
- **Ignores costs:** treats FP and FN as equally important. Business decisions need cost-weighted metrics.
- **Binary by nature:** multiclass needs one-vs-rest or one-vs-one averaging (macro/micro), which can hide poor classes.
- **Can be gamed or unstable:** with few positives the confidence intervals are wide (bootstrap them), and AUC doesn't reveal *which* segments fail.

**Production angle:** report ROC-AUC for model comparison, but choose the threshold and monitor with business-aligned metrics (precision/recall at the operating point, cost per decision), plus calibration plots.

### 12.47 When should you use PR-AUC over ROC-AUC?

Use **PR-AUC (average precision)** when the **positive class is rare** and you care mainly about how well the model finds positives, not about the negatives.

**Why:** ROC uses FPR = FP/(FP+TN). With 1M negatives and 1k positives, 10k false positives is only a 1% FPR, so the ROC curve looks great. **Precision** = TP/(TP+FP) exposes that same situation: 10k false alarms against 1k true positives means precision under 10%. PR curves ignore true negatives entirely, so they reflect what an analyst reviewing alerts actually experiences.

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

### 12.48 How do you run a good ablation study?

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

### 12.49 When would you prefer F1, ROC-AUC, PR-AUC, log loss, or calibration error?

| Metric | Prefer it when | Main idea |
| --- | --- | --- |
| **F1** | You need a **binary decision** and care about both precision and recall | `2PR/(P+R)` |
| **ROC-AUC** | You care about **ranking quality** across thresholds, especially with reasonably balanced classes | Probability that a positive ranks above a negative |
| **PR-AUC** | The positive class is **rare/imbalanced** and detecting positives matters | Measures precision-recall tradeoff |
| **Log Loss** | You care about **probability quality**, not just correct labels | Penalizes confident wrong predictions heavily |
| **Calibration Error** | You need **reliable probabilities** for decisions | Checks whether predicted probabilities match actual frequencies |

### 12.50 How does cross-validation fail when dealing with time-series data?

#### How Cross-Validation Fails with Time-Series Data

Standard **K-fold cross-validation** assumes that observations are independent and that the order of the data does not matter.

Time-series data violates this assumption because **the future can depend on the past**.

#### The problem

Suppose we have:

```txt
2021 → 2022 → 2023 → 2024 → 2025
```

With random K-fold CV, you might get:

```txt
Training: 2021, 2022, 2024, 2025
Validation: 2023
```

The model is being trained using **2024 and 2025 to predict 2023**.

That's **future information leakage**.

#### Correct approach: Walk-forward validation

Instead, preserve chronological order:

```txt
Fold 1:
Train:      2021
Validate:          2022

Fold 2:
Train:      2021 → 2022
Validate:                  2023

Fold 3:
Train:      2021 → 2023
Validate:                          2024

Fold 4:
Train:      2021 → 2024
Validate:                                  2025
```

This better simulates the real-world situation:

> **Train on the past → predict the future.**

### 12.51 What is stratified cross-validation, and when should you not use it?

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

### 12.52 Training accuracy is 99%, validation accuracy is 60%. How do you debug it?

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

### 12.53 Training and validation accuracy are both 50%. What could be wrong?

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

### 12.54 Your model performs well offline but poorly in production. Why?

Good offline, bad in production almost always means **the production data or setup differs from the offline evaluation**.

**Common causes:**
- **Training-serving skew:** features computed differently online (different code, units, defaults, stale feature store values, missing-value handling).
- **Data leakage offline:** features available in historical data that aren't available at prediction time (future information, post-event fields), inflating offline metrics.
- **Wrong validation scheme:** random split on time-dependent data instead of a time-based split; duplicates across train/test; tuning on the test set.
- **Distribution shift:** production users, seasons, products or behaviour differ from training data (covariate shift, label shift, concept drift).
- **Feedback loops:** the model's decisions change future data (e.g. recommendations shape clicks, fraud blocks change attacker behaviour).
- **Metric mismatch:** offline metric (AUC) doesn't reflect the business objective or the chosen threshold; class priors differ.
- **Engineering bugs:** different model version, preprocessing missing, timeouts causing fallbacks, latency forcing truncated inputs.

**How to investigate:**
1. Log production inputs and predictions; recompute offline predictions on those exact inputs and compare.
2. Compare feature distributions train vs production (PSI/KS).
3. Re-evaluate with a time-based backtest that mimics production.
4. Segment errors by user group, time, channel.

**Prevention:** feature store, point-in-time joins, shadow deployments before launch, monitoring and canary rollouts.

### 12.55 Your model's accuracy decreases after adding more training data. Explain possible reasons.

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

### 12.56 A feature has extremely high importance. How do you determine whether it is legitimate or leakage?

An extremely important feature is either a genuine strong signal or — very often — **target leakage**. I check:

**1. Timing / availability:** Is the feature's value known **at prediction time** in production? Trace its lineage. Examples of leakage: "account_closed_date" when predicting churn, "number of chargebacks" when predicting fraud, a status field updated after the outcome.

**2. Domain sense-check:** does the relationship make causal or business sense? Ask domain experts. A support-ticket count that predicts churn may be legit; a "refund_issued" flag predicting a complaint is probably leakage.

**3. Too-good-to-be-true metrics:** near-perfect AUC (e.g. > 0.99) on a hard problem is a red flag. Train a model on **only that feature** — if it alone almost perfectly predicts the label, investigate.

**4. Time-based validation:** use point-in-time correct feature snapshots and an out-of-time test set. If performance collapses when features are computed as of the prediction timestamp, it was leakage.

**5. Proxy / ID leakage:** IDs, row order, timestamps, file paths or data-source flags that correlate with the label because of how the data was collected (e.g. all positives came from one system).

**6. Inspect the relationship:** partial dependence/SHAP plots; look for perfect thresholds or values only present for one class (e.g. missing for all negatives).

**7. Compare with production:** shadow-score live data — leaked features often have different distributions or are missing online.

If legitimate, keep it but monitor it closely and consider robustness (the model shouldn't fail entirely if that feature breaks).

### 12.57 Your validation score varies significantly between folds. What does that indicate?

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

### 12.58 Your model predicts only the majority class. How do you fix it?

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

---

← [Chapter 11. Optimization and Training Dynamics](11-optimization-and-training-dynamics.md) · [Table of contents](../README.md) · [Chapter 13. Deep Learning](13-deep-learning.md) →
