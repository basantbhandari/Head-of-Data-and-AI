# Model Evaluation

Model evaluation interview questions: classification and regression metrics, class imbalance, ROC/AUC, metric selection, model comparison, cross-validation, hyperparameter tuning, unsupervised and recommender evaluation, A/B testing and LLM-as-a-judge.

### Q1. What are precision, recall, F1 score, and accuracy?

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

### Q2. What is the confusion matrix, and how do you interpret it?

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

### Q3. What are common evaluation metrics for Classification?

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

### Q4. When would you use accuracy vs other metrics?

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

### Q5. When would you use log loss vs accuracy?

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

### Q6. What metrics would you use for a multi-class classification problem?

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

### Q7. How do you handle class imbalance in classification metrics?

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

### Q8. What is the ROC curve? What is AUC?

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

### Q9. How do you handle imbalanced datasets?

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

### Q10. What are common evaluation metrics for Regression?

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

### Q11. What's the difference between MAE, MSE, and RMSE?

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

### Q12. How do you choose the right evaluation metric for a given problem?

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

> ✍️ _Drafted addition._

…that cost. The note above ends mid-sentence. The intended closing point: **the offline metric should be a proxy for the business cost of errors.** In practice:

- **Separate the optimization metric from the reporting metric.** Train on a differentiable loss (log loss, MSE), select models on the metric that reflects the cost (PR-AUC, recall@precision≥X, cost-weighted error), and report a business KPI (fraud $ caught, tickets deflected).
- **Use constraint-style metrics when there's a hard requirement.** For example, "maximize recall subject to precision ≥ 95%" or "maximize precision at a review capacity of 500 cases/day" (precision@k).
- **Cost-weighted evaluation.** With an explicit cost matrix (C_FP, C_FN), choose the threshold that minimizes `C_FP·FP + C_FN·FN`, not the default 0.5.
- **Ranking/recommendation:** NDCG@k, MAP, MRR. **Forecasting:** MAPE/sMAPE (beware zeros), or MASE. **LLM/generation:** task success rate, groundedness/faithfulness and human or LLM-as-judge preference, not just BLEU/ROUGE.
- **Validate the offline–online link.** Confirm with an A/B test that gains in the offline metric move the online KPI. If they don't, the metric is the wrong proxy.

### Q13. How do you compare the performance of different models?

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

### Q14. Explain cross-validation and its importance.

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

### Q15. What is Hyperparameter Tuning?

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

### Q16. How do you evaluate unsupervised learning models?

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

### Q17. How do you evaluate a clustering algorithm?

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

### Q18. What metrics would you use for a recommendation system?

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

### Q19. What is A/B testing in the context of ML?

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

### Q20. LLM as a Judge

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
