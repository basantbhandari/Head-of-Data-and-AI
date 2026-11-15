# Commonly Asked ML Interview Questions

Frequently asked ML interview questions, from classical ML and statistics to deep learning, LLMs, agents and system design.

### Q1. Difference between convex and non-convex cost functions; what does it mean when a cost function is non-convex?
> ✍️ _Drafted answer (not from the original notes)._

**Convex cost function:** any line segment between two points on its graph lies on or above the graph — formally $f(\lambda a + (1-\lambda) b) \le \lambda f(a) + (1-\lambda) f(b)$. It has a single basin, so **every local minimum is the global minimum** and gradient descent with a suitable learning rate converges to it. Examples: MSE for linear regression, log loss for logistic regression, the hinge loss for linear SVMs (the regularized versions too).

**Non-convex cost function:** the surface has multiple valleys, plateaus and **saddle points**. Implications:
- Gradient descent can get stuck in a **local minimum** or slow down on plateaus/saddles; the result depends on **initialization** and hyperparameters.
- No guarantee of reaching the global optimum, and different runs give different models.
- Examples: neural-network losses (because of hidden layers and non-linear activations), k-means objective, matrix factorization, GMM likelihood.

**Why we still succeed with non-convex problems**
- In high-dimensional neural nets, most critical points are saddles rather than bad local minima, and many local minima have similar, near-optimal loss.
- Practical tools: SGD noise, momentum/Adam, good initialization (He/Xavier), learning-rate schedules, multiple random restarts (k-means++), and batch normalization/residual connections that smooth the landscape.

**Interview takeaway:** convex = optimization is "solved" and reproducible; non-convex = optimization becomes part of the modeling problem, so you report variance across seeds and tune the optimizer.

### Q2. How do you decide when to go for deep learning for a project?
> ✍️ _Drafted answer (not from the original notes)._

Deep learning is worth it when the **problem, data and constraints** line up; otherwise gradient-boosted trees or linear models usually win on cost and maintainability.

**Go for deep learning when**
- **Unstructured data:** images, audio, text, video, graphs — where representation learning beats hand-made features.
- **Lots of (labeled or self-supervised) data**, or a pretrained foundation model you can fine-tune or prompt (transfer learning lowers the data requirement drastically).
- **Complex, non-linear patterns** with high-dimensional inputs (sequences, spatial structure).
- **Multimodal or generative tasks** (captioning, chat, summarization, image generation).
- You have **GPU budget** for training and serving, and latency requirements that allow it.

**Prefer classical ML when**
- Data is **tabular** and modest in size — XGBoost/LightGBM typically match or beat neural nets.
- **Interpretability** or regulatory explainability is a must (credit, insurance).
- Data is small, the deadline is tight, or the team lacks DL/MLOps experience.
- A simple baseline already meets the business target.

**Process I follow**
1. Define the metric and business target; build a quick **baseline** (heuristic, logistic regression, GBM).
2. Check whether a **pretrained model / API** solves it (embeddings + classifier, LLM prompting).
3. Move to custom DL only if the gap to the target justifies the extra training, serving and monitoring cost.

**Example:** churn prediction on CRM tables → GBM. Defect detection on factory images → fine-tune a pretrained CNN/ViT or YOLO.

### Q3. Give an example where the median is a better measure than the mean
> ✍️ _Drafted answer (not from the original notes)._

The **median is better when the distribution is skewed or has outliers**, because the mean is pulled toward extreme values while the median depends only on the middle observation.

**Classic example — income:** salaries in a team of ten are 40k–60k, plus a CEO at 5M. The mean is ~550k, which describes nobody; the median (~50k) represents a "typical" employee. That is why governments report **median household income** and **median house prices**.

**Other examples**
- **Latency / response times:** a few slow requests inflate the mean; teams report p50 (the median) alongside p95/p99.
- **Time-on-site, order values, insurance claims, hospital stays** — all long-tailed.
- **Imputing missing values** in a skewed feature: median imputation is more robust than mean imputation.
- **Data with recording errors** (e.g., an age of 999).

**When the mean is still preferred:** roughly symmetric data without outliers; when you need totals (total revenue = mean × count); or when extremes genuinely matter (expected cost in insurance).

**Practice:** look at the histogram or box plot, report both mean and median when they differ, and consider a log transform for heavily skewed features.

### Q4. What do you mean by the unreasonable effectiveness of data?
> ✍️ _Drafted answer (not from the original notes)._

"**The Unreasonable Effectiveness of Data**" is a 2009 paper by Halevy, Norvig and Pereira (Google), echoing Wigner's "unreasonable effectiveness of mathematics". The thesis: for many real-world problems, **simple models trained on very large datasets outperform sophisticated models trained on less data**.

**Key ideas**
- Evidence from Banko & Brill (2001): on a word-disambiguation task, simple algorithms' accuracy kept rising with more data, and the ranking of algorithms changed — data volume mattered more than algorithm choice.
- Web-scale text (n-gram models for translation, spelling correction, search) captured the messy, long-tailed reality of language better than hand-crafted rules or elaborate grammars.
- "Follow the data": instead of building complex theories of the domain, memorize and generalize from huge amounts of examples.

**Modern relevance**
- It anticipated **scaling laws** and LLMs: performance improves predictably with more data and compute.
- Qualifications today: data **quality** and deduplication matter as much as quantity (Chinchilla, Phi); diminishing returns follow power laws; labeled data is expensive; and more data can amplify bias.

**Practical takeaway:** before spending weeks tuning a model, ask whether acquiring, labeling or cleaning more representative data would yield a bigger gain — it often does. Learning curves tell you which regime you are in.

### Q5. What is an OOB error, and how is it useful?
> ✍️ _Drafted answer (not from the original notes)._

**OOB (out-of-bag) error** is a built-in validation estimate for bagging ensembles such as **Random Forests**.

**How it works**
- Each tree is trained on a **bootstrap sample** (drawn with replacement) of the training set. About $1 - (1-1/n)^n \approx 63.2\%$ of the rows appear in each sample; the remaining **~36.8% are "out-of-bag"** for that tree.
- For each training row, aggregate predictions **only from the trees that did not see it**, and compare with the true label.
- The OOB error is the error rate (or MSE) over all rows computed this way.

**Why it's useful**
- **Free validation:** an almost unbiased estimate of generalization error without a separate hold-out set or k-fold CV — valuable when data is scarce or training is expensive.
- **Hyperparameter tuning:** compare `n_estimators`, `max_features`, `min_samples_leaf` by OOB score.
- **Permutation feature importance** can be computed on OOB samples.
- Tracking OOB error versus number of trees shows when adding more trees stops helping.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1).fit(X, y)
print(rf.oob_score_)   # accuracy on out-of-bag samples
```

**Caveats:** it assumes i.i.d. rows — it is optimistic for time-series or grouped data (use time-based/grouped CV instead); it can be slightly pessimistic with few trees; and it does not replace a final untouched test set for reporting.

### Q6. In what scenario should a decision tree be preferred over random forest?
> ✍️ _Drafted answer (not from the original notes)._

A single decision tree is preferable to a random forest when **interpretability, simplicity or speed** matter more than the last few points of accuracy:

- **Explainability and auditability:** a shallow tree can be printed as if-then rules and shown to regulators, doctors or business users (credit decisions, clinical triage, policy rules). A forest of 500 trees is a black box without SHAP-style tools.
- **Rules to hand off:** the tree can be converted into SQL `CASE` statements or business rules engines.
- **Low-latency or constrained environments:** a single tree is tiny and predicts in microseconds — useful on embedded devices or in very high-QPS paths.
- **Small, clean data with a simple structure:** if a depth-3 tree already captures the signal, a forest adds complexity without real gain.
- **Exploratory analysis:** quickly understanding which features and thresholds split the data.
- **Teaching or baselines.**

**Trade-off:** single trees have **high variance** — small data changes produce very different trees, and they overfit easily. Mitigate with pruning (`max_depth`, `min_samples_leaf`, cost-complexity `ccp_alpha`) and cross-validation.

**Middle ground:** train a forest or GBM for accuracy, then **distill** it into a surrogate tree for explanation, or use a shallow tree when its validated performance is within your tolerance of the ensemble's.

### Q7. Why is Logistic Regression called regression?
> ✍️ _Drafted answer (not from the original notes)._

Because at its core it **is a regression model — a linear regression on the log-odds** — and only becomes a classifier when you put a threshold on its output.

$\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \quad\Rightarrow\quad p = \sigma(\beta^T x) = \frac{1}{1+e^{-\beta^T x}}$

- It models a **continuous quantity**: the probability $p$ (equivalently, the log-odds, which ranges over all reals) as a linear function of the features.
- It belongs to the family of **Generalized Linear Models (GLMs)**, alongside linear regression (identity link) and Poisson regression (log link); logistic regression uses the **logit link** with a Bernoulli outcome. Statisticians call all of these "regression".
- Coefficients are interpreted like regression coefficients: a one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$.
- Classification is a separate **decision step**: predict class 1 if $p > t$ (e.g., 0.5, or a threshold tuned to business cost).

**Why this matters in practice:** because it outputs (reasonably calibrated) probabilities, logistic regression is used for risk scoring, ranking and expected-value decisions — not just hard labels. It is trained with maximum likelihood (log loss), not least squares.

### Q8. What is Online Machine Learning? How is it different from Offline machine learning? List some of its applications
> ✍️ _Drafted answer (not from the original notes)._

**Online learning** updates the model **incrementally as each new example (or mini-batch) arrives**, typically discarding the data after the update. **Offline (batch) learning** trains on a fixed, complete dataset at once and is retrained periodically from scratch.

| | Offline / batch | Online |
| --- | --- | --- |
| Data | Full historical dataset available | Streaming, one example or mini-batch at a time |
| Updates | Periodic full retrains (daily/weekly) | Continuous, incremental |
| Adaptation to drift | Slow (until next retrain) | Fast |
| Compute/memory | High per retrain; needs all data | Low per update; constant memory |
| Stability/reproducibility | High, easier to validate | Risk of drift, catastrophic forgetting, poisoning |
| Algorithms | Most models | SGD-based linear models, online gradient descent, Hoeffding trees, bandits, `partial_fit` in scikit-learn, River, Vowpal Wabbit |

**Applications of online learning**
- **Ad click-through / recommendation ranking** (rapidly changing user interests).
- **Fraud and spam detection** where attackers adapt.
- **Stock/price prediction and dynamic pricing.**
- **News/feed personalization**, contextual bandits for A/B testing.
- **IoT/sensor anomaly detection** on streams; **datasets too large to fit in memory** (out-of-core learning).

**Production concerns:** the learning rate controls how fast it forgets; guard against bad data with validation gates and anomaly checks; keep shadow evaluation and rollback snapshots. Many teams use a hybrid approach: a periodic offline model plus online fine-tuning or recalibration.

### Q9. What is the No Free Lunch Theorem?
> ✍️ _Drafted answer (not from the original notes)._

The **No Free Lunch (NFL) theorem** (Wolpert & Macready, 1997, for optimization; Wolpert, 1996, for supervised learning) states that **averaged over all possible problems (all possible data-generating functions), every learning algorithm has the same expected performance**. No algorithm is universally best: any gain on one class of problems is paid for by losses on another.

**What it means in practice**
- An algorithm only does well because its **inductive bias** (assumptions) matches the structure of the real problem: CNNs assume spatial locality; linear models assume additive effects; trees assume axis-aligned splits; k-NN assumes smoothness.
- Real-world problems are not uniformly distributed over "all possible functions" — they have structure — so some algorithms are reliably good in particular domains (GBMs on tabular data, Transformers on language).
- You must **empirically compare** candidate models on your data with proper validation rather than assuming a favorite always wins.

**Interview framing:** "NFL is why I start with a baseline and a few model families suited to the data type, evaluate them with cross-validation on the metric that matters, and pick based on evidence — while also weighing interpretability, latency and maintenance cost."

**Common misreading:** NFL does *not* say all algorithms are equally good on *your* problem — only when averaged over all conceivable problems.

### Q10. Imagine you are working with a laptop with 2 GB RAM; how would you process a dataset of 10GB?
> ✍️ _Drafted answer (not from the original notes)._

Don't load it all at once — **stream, reduce, or push the compute elsewhere**.

**1. Read less**
- Load only the needed columns (`usecols`) and downcast dtypes (`float64→float32`, `int64→int8/16`, strings→`category`); this alone often cuts memory 3–5×.
- Filter rows early; sample for exploration and prototyping.

**2. Process in chunks (out-of-core)**
```python
import pandas as pd
totals = None
for chunk in pd.read_csv("big.csv", usecols=["user", "amount"], chunksize=500_000):
    agg = chunk.groupby("user")["amount"].sum()
    totals = agg if totals is None else totals.add(agg, fill_value=0)
```

**3. Use out-of-core / lazy engines**
- **Polars (lazy/streaming), DuckDB, Dask, Vaex** process data larger than RAM; DuckDB can query CSV/Parquet directly with SQL and spill to disk.
- Convert to **Parquet** (columnar, compressed) once; later reads touch only the needed columns and row groups.

**4. Train incrementally**
- Models with `partial_fit` (SGDClassifier, MiniBatchKMeans, Naive Bayes), XGBoost/LightGBM external-memory mode, or PyTorch `DataLoader` streaming from disk.

**5. Reduce the data**
- Aggregate to the modeling grain, feature hashing, dimensionality reduction, or a stratified sample — often a 1–2 GB sample is statistically sufficient.

**6. Move compute**
- Use a cloud VM/notebook, Spark/Databricks, or BigQuery/Snowflake to do heavy joins and aggregations, then pull the smaller result locally.

**Interview signal:** mention profiling memory (`df.info(memory_usage='deep')`), and that the right answer often is "do the heavy lifting in the warehouse".

### Q11. What are the main differences between Structured and Unstructured Data?
> ✍️ _Drafted answer (not from the original notes)._

| | Structured data | Unstructured data |
| --- | --- | --- |
| Format | Predefined schema: rows and columns with fixed types | No predefined data model |
| Examples | Relational tables, CRM records, transactions, sensor readings, spreadsheets | Text, emails, PDFs, images, audio, video, social posts |
| Storage | RDBMS (Postgres, MySQL), data warehouses (Snowflake, BigQuery) | Data lakes / object storage (S3, ADLS), document stores, vector DBs |
| Querying | SQL, easy to filter, aggregate and join | Needs parsing, NLP, CV, OCR or embeddings to extract meaning |
| Analysis/ML | Classical ML (GBMs, regression) on engineered features | Deep learning, foundation models, embeddings |
| Volume | Smaller share of enterprise data | Estimated ~80–90% of enterprise data |
| Quality control | Schema validation, constraints | Harder: content-level checks, metadata |

**Semi-structured data** sits in between: JSON, XML, logs, Avro/Parquet with nested fields — self-describing tags but a flexible schema (stored in document DBs or as VARIANT columns in warehouses).

**Why it matters for a Data/AI lead**
- Architecture: a **lakehouse** (Delta/Iceberg) lets you keep both kinds and govern them together.
- GenAI has made unstructured data far more valuable: RAG, document extraction and VLMs turn PDFs and images into structured fields or searchable embeddings.
- Governance differs: PII is easy to locate in columns but hidden inside free text and scans, so unstructured data needs detection and redaction pipelines.

### Q12. What are the main points of difference between Bagging and Boosting?
> ✍️ _Drafted answer (not from the original notes)._

Both are ensemble methods that combine many weak/base learners, but they attack different errors.

| | Bagging (Bootstrap Aggregating) | Boosting |
| --- | --- | --- |
| Training | **Parallel**, independent models | **Sequential**, each model depends on the previous ones |
| Data | Each model sees a bootstrap sample (with replacement) | All data, but **reweighted** (AdaBoost) or fit to **residuals/gradients** (Gradient Boosting) |
| Goal | Reduce **variance** | Reduce **bias** (and variance) |
| Base learners | Strong, low-bias, high-variance models (deep trees) | Weak learners (shallow trees/stumps) |
| Combination | Simple average / majority vote | Weighted sum of learners |
| Overfitting | Robust; more trees rarely hurt | Can overfit with too many rounds; needs learning rate, early stopping, regularization |
| Noise/outliers | Robust | Sensitive (it keeps focusing on hard/mislabeled points) |
| Speed | Easily parallelized | Sequential across rounds (parallel within a tree in XGBoost/LightGBM) |
| Examples | Random Forest, Extra Trees, bagged models | AdaBoost, GBM, XGBoost, LightGBM, CatBoost |

**When to use which**
- **Bagging/Random Forest:** strong, low-tuning baseline; noisy data; when you want stability and OOB error estimates.
- **Boosting:** usually the top accuracy on tabular data when tuned carefully (learning rate, depth, n_estimators with early stopping, subsampling).

**One-liner:** bagging = many independent experts voting to cancel out noise; boosting = a team where each new member fixes the previous members' mistakes.

### Q13. How do you measure the accuracy of a Clustering Algorithm?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q14. What is Matrix Factorization, and where is it used in Machine Learning?
> ✍️ _Drafted answer (not from the original notes)._

**Matrix factorization** decomposes a matrix $R$ ($m \times n$) into a product of lower-rank matrices, typically $R \approx U V^T$ with $U \in \mathbb{R}^{m\times k}$, $V \in \mathbb{R}^{n\times k}$ and $k \ll m,n$. Each row of $U$ and $V$ is a latent vector, and their dot product reconstructs an entry.

**Where it is used:**
- **Recommender systems (collaborative filtering):** user–item rating matrix → user and item embeddings; predicted rating $\hat r_{ui} = u_u^T v_i$ (+ biases). Made famous by the Netflix Prize. Trained with SGD or **ALS** (Alternating Least Squares, easy to parallelize, e.g. Spark MLlib), with L2 regularization, fitting only the observed entries.
- **Dimensionality reduction:** SVD/PCA, truncated SVD for sparse text matrices (LSA/topic modeling).
- **NMF (non-negative MF):** interpretable parts-based factors for topic modeling, audio source separation, image parts.
- **Compression:** low-rank approximation of neural network weights (the same idea behind LoRA).
- **Imputation** of missing values in matrices.

**Tradeoffs:** it is fast, scalable and gives reusable embeddings, but has a **cold-start problem** (new users/items have no vector), is linear (dot-product interactions only), and implicit feedback (clicks) needs weighted variants (e.g. implicit ALS). In production, hybrid models (factorization machines, two-tower networks) add side features to solve cold start.

### Q15. How do you measure the accuracy of a recommendation engine?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q16. What are some ways to make your model more robust to outliers?
> ✍️ _Drafted answer (not from the original notes)._

**First decide whether the outliers are errors or real signal.** Fix or drop data-entry errors. Keep genuine extremes (fraud, large orders) and make the model handle them.

**Data-level:**
- **Transform** skewed features: log, Box-Cox/Yeo-Johnson, rank/quantile transforms.
- **Winsorize / clip** at percentiles (e.g. 1st–99th), using thresholds fitted on training data only.
- **Robust scaling** (median/IQR, `RobustScaler`) instead of mean/std.
- Binning continuous features.

**Model-level:**
- **Robust losses:** MAE/L1, **Huber**, quantile loss instead of MSE, which squares large errors and lets them dominate.
- **Tree-based models** (RF, GBMs) are insensitive to outliers in *features*, because splits depend only on ordering (target outliers still matter for regression).
- **Robust regression:** RANSAC, Theil–Sen, Huber regressor.
- Regularization to stop the model chasing extreme points.
- Ensembles/bagging to average out the influence of individual points.

**Evaluation:** report median absolute error or quantile metrics next to RMSE, and look at error by segment.

**Production angle:** outliers at inference time may be **out-of-distribution** inputs. Add input validation, range checks and drift monitoring, and fall back or flag for review rather than trusting extrapolation.

```python
from sklearn.linear_model import HuberRegressor
model = HuberRegressor(epsilon=1.35).fit(X, y)
```

### Q17. How can you measure the performance of a dimensionality reduction algorithm on your dataset?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q18. What is Data Leakage? List some ways by which you can overcome this problem.
> ✍️ _Drafted answer (not from the original notes)._

**Data leakage** happens when information that would not be available at prediction time gets into training. The result is optimistic offline metrics that collapse in production.

**Common forms:**
- **Target leakage:** features derived from the label or recorded after the outcome (e.g. "refund_issued" when predicting fraud).
- **Train–test contamination:** fitting scalers, imputers, encoders or feature selection on the full dataset before splitting.
- **Temporal leakage:** random splits on time-dependent data; features built with future windows.
- **Group leakage:** the same patient, user or document appears in both train and test.
- **Duplicates / near-duplicates** across splits.

**How to prevent it:**
1. **Split first**, then do all preprocessing inside a `Pipeline` so it is fitted only on training folds.
2. Use **time-based splits** (`TimeSeriesSplit`, rolling-origin) and **point-in-time correct** feature joins (feature stores with as-of joins).
3. Use **group-aware CV** (`GroupKFold`) for entity-level data.
4. Audit features: for each one ask "would I know this at prediction time?" Treat suspiciously high importance or near-perfect AUC as a red flag.
5. Deduplicate before splitting.
6. Keep a truly untouched holdout set and validate with a shadow deployment.

```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
pipe = make_pipeline(StandardScaler(), LogisticRegression())
cross_val_score(pipe, X, y, cv=TimeSeriesSplit(5))
```

### Q19. What are the different types of bias in Machine Learning?
> ✍️ _Drafted answer (not from the original notes)._

"Bias" means two different things in ML. Separate them in an interview.

**1. Statistical bias (bias–variance):** systematic error from overly simple assumptions, which leads to underfitting. Also the bias/intercept term in models.

**2. Data and societal biases**, which cause unfair or wrong models:
- **Selection / sampling bias:** the training data doesn't represent the population (e.g. a loan model trained only on approved applicants is *survivorship* or *reject-inference* bias).
- **Historical bias:** the data faithfully reflects past discrimination (hiring data).
- **Measurement bias:** proxies or labels are measured differently across groups (arrests as a proxy for crime).
- **Label / annotator bias:** subjective or inconsistent labelling.
- **Representation bias:** under-represented groups get poor accuracy (face recognition on darker skin tones).
- **Exclusion bias:** dropping features or records that matter.
- **Confirmation bias:** analysts favour results that match their expectations.
- **Aggregation bias:** one model for heterogeneous subgroups.
- **Evaluation bias:** benchmarks that don't reflect the deployment population.
- **Feedback-loop / automation bias:** model outputs shape future training data (recommenders, predictive policing), and humans over-trust model output.
- **Algorithmic / inductive bias:** assumptions built into the learner.

**Mitigation:** audit data coverage, use stratified sampling, compute per-group metrics (equal opportunity, demographic parity), reweight or resample, apply fairness constraints, use diverse annotators with clear guidelines, monitor slices in production, and keep a human review for high-stakes decisions.

### Q20. How do you approach a categorical feature with high cardinality?
> ✍️ _Drafted answer (not from the original notes)._

High-cardinality categoricals (zip codes, user IDs, product SKUs) make one-hot encoding explode in dimension and sparsity, and the rare levels overfit.

**Options, roughly in order of preference:**
- **Group rare levels** into "Other" (frequency threshold), then one-hot the rest.
- **Frequency / count encoding:** replace a level with how often it occurs. Cheap and works well with trees.
- **Target (mean) encoding:** replace a level with the mean target, **smoothed** toward the global mean and computed **out-of-fold** (or with CatBoost's ordered target statistics) to avoid leakage.
- **Hashing trick:** hash into a fixed number of buckets. Memory-bounded and handles unseen levels, at the cost of collisions.
- **Learned embeddings:** entity embeddings in neural networks (standard for IDs in recommenders and two-tower models).
- **Domain hierarchy:** map zip → region, SKU → category, or join external attributes such as population or income for a zip.
- **Native support:** CatBoost and LightGBM handle categoricals directly.

**Production concerns:**
- Unseen categories at inference time: always have an "unknown" bucket or a fallback.
- Category drift: new SKUs appear daily, so hashing or embeddings with periodic retraining hold up better.
- Store encoding mappings with the model artifact for training/serving parity.

```python
from sklearn.preprocessing import TargetEncoder  # sklearn >= 1.3, cross-fitted internally
X_enc = TargetEncoder(smooth="auto").fit_transform(X[["zip"]], y)
```

### Q21. Explain Pruning in Decision Trees and how it is done
> ✍️ _Drafted answer (not from the original notes)._

**Pruning** reduces the size of a decision tree to stop it from memorizing noise. Fully grown trees have low bias and very high variance.

**1. Pre-pruning (early stopping)** — stop growing during construction:
- `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`
- `min_impurity_decrease`: only split if the impurity gain exceeds a threshold.

It is fast, but it can be short-sighted: a split that looks weak now may enable strong splits below it (the "horizon effect").

**2. Post-pruning** — grow the full tree, then remove subtrees that don't help:
- **Cost-complexity pruning (CART, weakest-link):** minimize $R_\alpha(T) = R(T) + \alpha |T|$, where $R(T)$ is the training error and $|T|$ the number of leaves. Increasing $\alpha$ gives a nested sequence of smaller trees. Pick $\alpha$ by cross-validation.
- **Reduced-error pruning:** replace a subtree with a leaf if accuracy on a validation set doesn't drop.
- **Pessimistic error pruning** (C4.5): uses a statistical upper bound on the error, so no validation set is needed.

```python
from sklearn.tree import DecisionTreeClassifier
path = DecisionTreeClassifier().cost_complexity_pruning_path(X_tr, y_tr)
# grid-search ccp_alpha over path.ccp_alphas with CV
clf = DecisionTreeClassifier(ccp_alpha=best_alpha).fit(X_tr, y_tr)
```

**Practical note:** inside ensembles, random forests usually use unpruned trees (bagging handles the variance), while gradient boosting uses shallow, pre-pruned trees. Pruning matters most when you need a single interpretable tree.

### Q22. What is the ROC-AUC curve? List some of its benefits.
> ✍️ _Drafted answer (not from the original notes)._

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

### Q23. What are kernels in SVM? Can you list some popular SVM kernels?
> ✍️ _Drafted answer (not from the original notes)._

A **kernel** is a function $K(x, x') = \langle \phi(x), \phi(x') \rangle$ that computes an inner product in a (possibly infinite-dimensional) feature space without computing $\phi$ explicitly. Because the SVM dual depends on the data only through dot products, swapping in a kernel lets a linear max-margin classifier learn **non-linear decision boundaries**. This is the kernel trick. A valid kernel must be positive semi-definite (Mercer's condition).

**Popular kernels:**
| Kernel | Formula | When to use |
| --- | --- | --- |
| Linear | $x^T x'$ | High-dimensional sparse data (text), many features; fastest |
| Polynomial | $(\gamma x^T x' + r)^d$ | Feature interactions of degree $d$; can be numerically unstable for large $d$ |
| RBF / Gaussian | $\exp(-\gamma \lVert x-x'\rVert^2)$ | Default for non-linear problems; $\gamma$ controls locality |
| Sigmoid | $\tanh(\gamma x^T x' + r)$ | Neural-net-like; not always PSD, rarely best |
| Others | Laplacian, chi-squared (histograms), string/graph kernels | Domain-specific data |

**Practical tradeoffs:**
- Kernel SVMs need $O(n^2)$ memory and $O(n^2)$–$O(n^3)$ training time, so they don't scale past roughly $10^5$ samples. For big data use a linear SVM, or approximate the kernel with random Fourier features or Nyström and then a linear model.
- Tune $C$ and $\gamma$ jointly with grid search on a log scale, and **scale the features** first.

### Q24. What is the difference between Gini Impurity and Entropy? Which one is better and why?
> ✍️ _Drafted answer (not from the original notes)._

Both measure node impurity for choosing splits in decision trees.

- **Gini impurity:** $G = 1 - \sum_k p_k^2$, the probability of misclassifying a random sample if it were labelled at random from the node's class distribution. Range $[0, 1 - 1/K]$ (0.5 for binary).
- **Entropy:** $H = -\sum_k p_k \log_2 p_k$, the information content; splitting maximizes **information gain** (the reduction in entropy). Range $[0, \log_2 K]$ (1 for binary).

**Differences:**
- **Computation:** Gini avoids logarithms, so it is slightly faster. This is why it is the default in CART/scikit-learn.
- **Behaviour:** entropy penalizes mixed nodes a little more and tends to produce slightly more balanced trees. Gini tends to isolate the most frequent class in its own branch.
- **Origin:** Gini comes from CART; entropy/information gain from ID3/C4.5 (C4.5 uses gain ratio to correct the bias toward many-valued features).

**Which is better?** Neither, in general. Empirical studies show they pick the same split in roughly 98% of cases and the accuracy differences are negligible. Choose Gini by default for speed and treat the criterion as a hyperparameter if you care. Pruning, depth limits and ensembling matter far more than the impurity measure.

```python
DecisionTreeClassifier(criterion="gini")      # or "entropy" / "log_loss"
```

### Q25. Why does L2 regularization give sparse coefficients?
> ✍️ _Drafted answer (not from the original notes)._

**The premise is wrong: L2 (Ridge) regularization does *not* give sparse coefficients. L1 (Lasso) does.**

**Why L2 isn't sparse:** the penalty $\lambda \sum w_j^2$ has gradient $2\lambda w_j$, which shrinks toward zero in proportion to the weight. As $w_j$ gets small the push gets small too, so weights become tiny but almost never exactly zero. Ridge shrinks all coefficients smoothly and spreads weight across correlated features.

**Why L1 is sparse:**
- **Gradient view:** the penalty $\lambda \sum |w_j|$ has a constant-magnitude (sub)gradient $\lambda \,\text{sign}(w_j)$, so it keeps pushing small weights at the same rate until they reach exactly 0. The solution is the soft-thresholding operator: $w_j = \text{sign}(z)\max(|z| - \lambda, 0)$.
- **Geometric view:** the constrained problem $\min \text{loss}$ s.t. $\lVert w\rVert_1 \le t$ has a diamond-shaped feasible region with corners on the axes. The elliptical loss contours usually touch it at a corner, where some coordinates are exactly zero. The L2 ball is round, with no corners, so contact points are generically off-axis.
- **Bayesian view:** L1 equals a Laplace prior (peaked at 0) and L2 a Gaussian prior.

**Practical use:** use L1 for feature selection or interpretability, L2 for stability with multicollinearity, and **Elastic Net** for both (sparse, but it handles groups of correlated features better than Lasso, which picks one arbitrarily).

### Q26. List some ways using which you can improve a model’s performance.
> ✍️ _Drafted answer (not from the original notes)._

Work systematically: diagnose first, then pick the lever.

**1. Diagnose:** compare train vs validation error. A high-bias model underfits; a high-variance model overfits. Do error analysis on misclassified examples and slices. Learning curves show whether more data would help.

**2. Data (usually the biggest win):**
- More or better labelled data; fix label noise; active learning.
- Better feature engineering: domain features, interactions, aggregates, time features.
- Handle missing values, outliers and class imbalance properly.
- Data augmentation (images, text, audio).

**3. Model:**
- **Underfitting:** a more expressive model, more features, less regularization, longer training.
- **Overfitting:** regularization (L1/L2, dropout), simpler model, early stopping, more data.
- Try stronger algorithm families (gradient boosting for tabular data, pretrained models for text and vision with fine-tuning).

**4. Hyperparameter tuning:** random search or Bayesian optimization (Optuna) with proper CV.

**5. Ensembling:** bagging, boosting, stacking or blending diverse models.

**6. Objective and threshold:** optimize the loss that matches the business metric, tune the decision threshold, calibrate probabilities.

**7. Validation hygiene:** make sure the CV scheme reflects production (time splits, groups) and that there is no leakage. Many "improvements" are just overfitting the validation set.

**Production angle:** measure gains against latency, cost and maintainability, and confirm them with an A/B test, because an offline lift is not a business lift.

### Q27. Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?
> ✍️ _Drafted answer (not from the original notes)._

**Standard PCA will usually do poorly** on a highly non-linear dataset. PCA finds orthogonal *linear* directions of maximum variance. If the data lies on a curved manifold (a Swiss roll, concentric circles), the important structure isn't aligned with any linear subspace. PCA will either need many components or squash distinct regions on top of each other.

That said, PCA can still be useful as a **first step**: to remove noise and redundant dimensions (e.g. 10,000 → 50) before a non-linear method, or simply because a "non-linear" dataset sometimes still has most of its variance in a few linear directions. Check the explained-variance curve and the downstream metric rather than assuming.

**Better options for non-linear structure:**
- **Kernel PCA** (RBF or polynomial kernel): PCA in an implicit feature space, which can unroll curved structure. Tune the kernel and $\gamma$ (e.g. by reconstruction error or downstream accuracy). It costs $O(n^2)$ memory.
- **Manifold learning:** Isomap, LLE, **t-SNE** and **UMAP** (t-SNE and UMAP mainly for visualization; UMAP also scales and supports `transform`).
- **Autoencoders:** neural non-linear compression; scale to large data and support reconstruction.

```python
from sklearn.decomposition import KernelPCA
X_k = KernelPCA(n_components=2, kernel="rbf", gamma=15).fit_transform(X)
```

### Q28. What’s the difference between probability and likelihood?
> ✍️ _Drafted answer (not from the original notes)._

They use the same function $p(x \mid \theta)$, read in two different directions.

- **Probability:** fix the parameters $\theta$ and ask how likely different *data* outcomes are. "Given a fair coin ($p = 0.5$), what is the chance of 7 heads in 10 flips?" Probabilities over outcomes sum or integrate to 1.
- **Likelihood:** fix the observed *data* and treat it as a function of the parameters: $L(\theta \mid x) = p(x \mid \theta)$. "Given I saw 7 heads in 10 flips, how plausible is $p = 0.5$ vs $p = 0.7$?" Likelihood is **not** a probability distribution over $\theta$ and need not sum to 1.

**In ML:**
- **Maximum Likelihood Estimation:** $\hat\theta = \arg\max_\theta \prod_i p(x_i \mid \theta)$, in practice by maximizing the **log-likelihood** (sums are numerically stable). Minimizing cross-entropy or MSE equals MLE under Bernoulli or Gaussian noise assumptions.
- For a probability over parameters, use Bayes: posterior $\propto$ likelihood $\times$ prior (MAP / Bayesian inference).

**Example:** 7 heads in 10 flips gives $L(0.7) = \binom{10}{7}0.7^7 0.3^3 \approx 0.267$ and $L(0.5) \approx 0.117$, so $p = 0.7$ is more likely given the data. The MLE is $\hat p = 0.7$.

### Q29. What cross-validation technique would you use on a time series data set?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q30. Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?
> ✍️ _Drafted answer (not from the original notes)._

**Generally only approximately, and for some methods not at all.** Reducing dimensions throws information away, so an exact inverse is impossible unless the dropped components carried zero variance.

- **PCA:** yes, approximately. $X_{rec} = Z W^T + \mu$, where $W$ holds the kept components. The **reconstruction error** equals the variance in the discarded components. Keeping 95% of the variance means losing about 5%. This is used for denoising and compression (`pca.inverse_transform`).
- **Truncated SVD / NMF:** same idea, $X \approx U\Sigma V^T$ or $X \approx WH$.
- **Autoencoders:** the decoder *is* the learned inverse, giving an approximate reconstruction.
- **Kernel PCA:** no direct inverse, because the components live in an implicit feature space. You learn an approximate **pre-image** mapping (`fit_inverse_transform=True` in scikit-learn trains a regression back to input space).
- **t-SNE / UMAP:** t-SNE has no inverse mapping at all, since it optimizes an embedding of the given points only. UMAP offers an approximate `inverse_transform`.
- **Feature selection** (dropping columns): the dropped columns are simply gone.

**Practical use:** reconstruction error is a good **anomaly-detection** signal. Points the reduced model can't reconstruct well are unusual (common with PCA or autoencoders on sensor or transaction data).

```python
X_rec = pca.inverse_transform(pca.transform(X))
err = ((X - X_rec) ** 2).mean(axis=1)
```

### Q31. Why do we always need the intercept term in a regression model??
> ✍️ _Drafted answer (not from the original notes)._

The intercept $\beta_0$ is the predicted value when all features are 0. Without it, the regression line or hyperplane is **forced through the origin**, which is almost never justified.

**Why we keep it:**
- **Unbiased fit:** if the true relationship has a non-zero offset, dropping the intercept biases *every* slope coefficient to compensate, and the predictions become systematically wrong.
- **Zero-mean residuals:** with an intercept, OLS guarantees the residuals sum to zero and the fit passes through $(\bar x, \bar y)$.
- **Valid statistics:** $R^2$ and the usual F-tests assume an intercept. Without one, $R^2$ is computed against 0 instead of $\bar y$ and can be misleadingly high, or even negative in some software.
- **Categorical encoding:** with dummy variables, the intercept represents the baseline category. This is why you drop one dummy, to avoid the dummy-variable trap.
- In logistic regression, the intercept captures the base-rate log-odds.

**When you might drop it (rarely):**
- Physics or domain knowledge guarantees $y = 0$ when $x = 0$ (e.g. distance = speed × time), and even then test it.
- All features and the target are mean-centered: the intercept is then 0 anyway.
- You include a full set of one-hot dummies (the intercept is then redundant).

Interpretation caveat: the intercept itself often has no practical meaning (age = 0, house size = 0). Centering features makes it the "average case" prediction.

### Q32. Which among these is more important: model accuracy or model performance?
> ✍️ _Drafted answer (not from the original notes)._

It depends on how you define the terms, and that is the point of the question.

- **Model accuracy** is one specific metric: the fraction of correct predictions.
- **Model performance** is broader: how well the model serves its purpose. That includes the *right* metric for the problem (precision/recall, F1, AUC, RMSE, calibration) and also latency, throughput, cost, robustness, fairness and business impact.

**Performance matters more**, because accuracy can be misleading:
- **Imbalanced data:** with 1% fraud, predicting "not fraud" always gives 99% accuracy and catches nothing. Recall, precision and PR-AUC tell the real story.
- **Asymmetric costs:** a missed cancer diagnosis (FN) costs far more than a false alarm (FP), so optimize recall at an acceptable precision, not accuracy.
- **Operational constraints:** a model that is 0.5% more accurate but 10× slower may break a 50 ms SLA or blow the inference budget. Fraud scoring, ad ranking and search all have hard latency limits.
- **Business outcome:** the goal is revenue, retention or reduced risk. A slightly less accurate but interpretable, stable model may win in regulated settings.

**How I'd frame it in practice:** agree on the success metric with stakeholders up front (e.g. "recall ≥ 90% at ≤ 5% FPR, p95 latency < 100 ms"), select models on that, and validate with an A/B test on the business KPI.

### Q33. What is active learning, and where is it useful?
> ✍️ _Drafted answer (not from the original notes)._

**Active learning** is a semi-supervised loop in which the model **chooses which unlabelled examples a human should label next**, so you reach a target accuracy with far fewer labels.

**Loop:**
1. Train on a small labelled seed set.
2. Score the unlabelled pool.
3. Select the most informative samples using a query strategy.
4. Send them to annotators, add them to the training set, retrain, repeat until budget or performance targets are met.

**Query strategies:**
- **Uncertainty sampling:** least confidence, smallest margin between the top-2 classes, or highest entropy.
- **Query-by-committee:** samples where an ensemble disagrees most.
- **Expected model change / expected error reduction:** more principled but expensive.
- **Diversity / density-based** (core-set, clustering): avoids picking many near-duplicate uncertain points. In practice, combine uncertainty with diversity and do batch selection.

**Where it's useful:** labelling is expensive or needs experts (medical imaging, legal document review, fraud investigation), unlabelled data is plentiful (support tickets, images, logs), and domains shift over time (refresh labels on new patterns). It is common in document AI and in LLM evaluation and fine-tuning data curation.

**Pitfalls:** the selected samples are biased (not i.i.d.), so keep a **random, separately labelled test set**. Cold start needs a reasonable seed. Pure uncertainty sampling can chase label noise and outliers. Factor annotator latency into the batch size.

### Q34. Why is Ridge Regression called Ridge?
> ✍️ _Drafted answer (not from the original notes)._

There are two complementary explanations.

**1. The "ridge" added to the matrix (the original, historical reason).** The OLS solution $\hat\beta = (X^TX)^{-1}X^Ty$ becomes unstable when features are collinear, because $X^TX$ is near-singular. Hoerl and Kennard (1970) proposed
$$\hat\beta_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$
Adding $\lambda I$ puts a constant "ridge" along the diagonal of $X^TX$. This lifts every eigenvalue by $\lambda$, makes the matrix invertible and well-conditioned, and stabilizes the coefficients. The name also echoes Hoerl's earlier "ridge analysis" of response surfaces, where the path of optimal estimates traces a ridge.

**2. The ridge in the likelihood surface.** With multicollinearity, the loss surface has a long flat valley or ridge: many coefficient combinations fit almost equally well, so the estimates have huge variance. The L2 penalty $\lambda\lVert\beta\rVert_2^2$ adds curvature, picks the smallest-norm solution along that ridge, and shrinks the coefficients.

**Why it matters:** ridge trades a little bias for a large reduction in variance, keeps all features (no sparsity, unlike Lasso), and is the go-to when predictors are highly correlated. Equivalent views: L2 regularization, a Gaussian prior on the weights (MAP), and weight decay in neural networks.

### Q35. Does it make any sense to chain two different dimensionality reduction algorithms?
> ✍️ _Drafted answer (not from the original notes)._

**Yes, and it is common practice.** The typical pattern is a **cheap linear method first, then an expensive non-linear one**:

- **PCA → t-SNE / UMAP:** reduce 10,000-dimensional data (gene expression, embeddings, image pixels) to roughly 50 PCA components, then run t-SNE or UMAP to 2-D. PCA removes noise and redundant dimensions and makes the neighbour computations much faster. scikit-learn's t-SNE docs recommend exactly this.
- **Feature selection → PCA:** drop irrelevant or noisy features first, then decorrelate.
- **TF-IDF → Truncated SVD (LSA) → UMAP** for visualizing text corpora.
- **Autoencoder → clustering or t-SNE** on the latent space.

**Why it helps:** speed and memory (manifold methods scale roughly $O(n^2)$ or worse with dimension-dependent distance costs), denoising, and avoiding the curse of dimensionality in distance-based methods.

**Cautions:**
- The first step can discard non-linear structure the second needs, so keep enough components (e.g. 50–100 or 90%+ variance).
- Fit the chain on training data only and apply it consistently at inference. t-SNE has no `transform` for new points, which is a reason to prefer UMAP or parametric methods in production.
- Validate with the downstream metric or neighbourhood-preservation scores, not only pretty plots.

```python
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import umap
reducer = make_pipeline(PCA(n_components=50), umap.UMAP(n_components=2))
```

### Q36. Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?
> ✍️ _Drafted answer (not from the original notes)._

**Yes. Bagging is embarrassingly parallel.** Each base model is trained independently on its own bootstrap sample, so there are no dependencies between estimators during training.

**How:**
- **Single machine:** `n_jobs=-1` in scikit-learn's `BaggingClassifier` / `RandomForestClassifier` uses all cores.
- **Multiple servers:** split the $n$ estimators across workers (e.g. 500 trees → 10 servers × 50 trees), each drawing its own bootstraps with different seeds, and then **combine the predictions** (average probabilities, or majority vote). Tools: Spark MLlib random forests, Dask/joblib distributed backends, Ray, or simple job arrays.
- **Prediction** parallelizes the same way: each server scores with its trees and the results are aggregated.

**Contrast with boosting:** boosting (AdaBoost, GBMs) is **sequential**, because each learner fits the errors of the previous ones, so you can't distribute across estimators. Systems like XGBoost and LightGBM parallelize *within* a tree instead (split finding over features or histograms, data-parallel gradient sums).

**Practical considerations:**
- Data distribution: each worker needs access to the data (or its partition); shipping big data can dominate the runtime.
- Use distinct random seeds per worker so the ensembles aren't identical.
- If workers sample from different data partitions, bootstrap samples are no longer drawn from the full dataset. That is usually fine with i.i.d. shuffled partitions.
- Out-of-bag (OOB) evaluation requires tracking which rows each tree saw.

### Q37. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?
> ✍️ _Drafted answer (not from the original notes)._

**No.** Decision trees are **invariant to monotonic feature scaling**. A split asks "is $x_j \le t$?", and scaling $x_j$ (standardization, min-max, even a log transform) just moves the threshold $t$. The ordering of samples, and therefore the partitions and impurity reductions, stays exactly the same. Scaling won't change the tree at all, apart from floating-point ties.

**What actually fixes an underfitting tree** (high bias):
- **Loosen the regularization:** increase `max_depth` and `max_leaf_nodes`, decrease `min_samples_leaf`, `min_samples_split` and `min_impurity_decrease`, reduce `ccp_alpha`.
- **Better features:** trees struggle with diagonal or additive relationships, because they need many axis-aligned splits. Engineered features like ratios, differences, interactions and aggregates help a lot.
- **Switch to ensembles:** gradient boosting reduces bias by adding trees sequentially. Random forests mostly reduce variance, so they help less with pure underfitting.
- Check data quality and label noise. Sometimes the signal just isn't in the features.

**When scaling does matter:** distance- or gradient-based models such as KNN, SVM, linear and logistic regression with regularization, neural networks, PCA and k-means. In a mixed pipeline (e.g. trees stacked with logistic regression) you scale for the other models, not for the tree.

### Q38. Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?
> ✍️ _Drafted answer (not from the original notes)._

**Increase γ and increase C.** Both reduce regularization and make the model more complex.

**γ (gamma)** in the RBF kernel $K(x,x') = \exp(-\gamma\lVert x - x'\rVert^2)$ controls how far a single training example's influence reaches.
- **Small γ:** wide, smooth Gaussians, a nearly linear and very smooth boundary, which can underfit.
- **Large γ:** narrow bells, so each point influences only its neighbourhood and the boundary becomes wiggly; too large overfits.
- Underfitting → **increase γ**.

**C** is the inverse regularization strength: the penalty for margin violations in the soft-margin objective $\frac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i$.
- **Small C:** wide margin, tolerates misclassifications, strong regularization, can underfit.
- **Large C:** penalizes errors heavily, narrower margin, fits the training data more closely.
- Underfitting → **increase C**.

**In practice:** tune them jointly on a log grid (e.g. $C \in \{0.1, 1, 10, 100\}$, $\gamma \in \{10^{-3}, \dots, 10\}$) with cross-validation, because they interact. Also make sure the features are **scaled**: unscaled features make γ meaningless and a common "underfit" cause. If the model still underfits after that, the features may lack signal.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 0.1, 1]}, cv=5)
```

### Q39. What is cross-validation and its types?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q40. How do we interpret weights in linear models?
> ✍️ _Drafted answer (not from the original notes)._

In a linear model $\hat y = \beta_0 + \sum_j \beta_j x_j$, the weight $\beta_j$ is **the expected change in the prediction for a one-unit increase in $x_j$, holding all other features constant**.

**Variants:**
- **Logistic regression:** $\beta_j$ is the change in **log-odds**. $e^{\beta_j}$ is the **odds ratio**: e.g. $\beta = 0.7$ means each unit multiplies the odds by about 2.
- **Log-transformed target** ($\log y$): a unit change in $x_j$ changes $y$ by about $100\cdot\beta_j\%$ (exactly $e^{\beta_j} - 1$).
- **Log-log:** $\beta_j$ is an **elasticity**: a 1% change in $x_j$ gives a $\beta_j\%$ change in $y$.
- **Dummy variable:** the difference from the baseline category.
- **Interaction terms:** the effect of $x_j$ depends on the value of the interacting feature.

**Caveats (where interviews dig):**
- **Scale:** raw weights aren't comparable across features in different units. Standardize the features to compare magnitudes (the effect of a 1-SD change).
- **Multicollinearity:** correlated features make weights unstable, so signs can flip and the "holding others constant" reading becomes unrealistic. Check VIF.
- **Regularization** shrinks weights (Ridge) or zeroes them (Lasso), so the magnitudes are biased.
- **Statistical significance:** look at standard errors, confidence intervals and p-values, not just point estimates.
- **Correlation ≠ causation:** weights describe associations conditional on the included features. Omitted-variable bias can distort them.

For complex models, SHAP or partial dependence give comparable per-feature effect interpretations.

### Q41. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?
> ✍️ _Drafted answer (not from the original notes)._

(This refers to the classic comparison of **Batch GD, Stochastic GD and Mini-batch GD**, as in *Hands-On ML*.)

**Reaches the vicinity of the optimum fastest: Stochastic GD**, closely followed by **Mini-batch GD**. SGD updates after *every* example, so it makes $n$ updates per epoch versus one for batch GD. On large datasets it gets close to the minimum after a fraction of an epoch. Mini-batch GD is similar per update and in wall-clock time is often the fastest in practice, thanks to vectorized GPU computation.

**Actually converges: Batch GD.** With a suitable fixed learning rate on a convex problem (e.g. linear regression's MSE), it follows the true gradient and settles exactly at the minimum. SGD and mini-batch GD use noisy gradient estimates, so with a constant learning rate they **keep bouncing around** the minimum. Mini-batch bounces less than SGD.

**However:** SGD and mini-batch GD *will* converge if you use a **learning-rate schedule** that decays over time (e.g. $\eta_t \propto 1/t$, satisfying $\sum \eta_t = \infty$, $\sum \eta_t^2 < \infty$), or progressively increase the batch size.

**Tradeoffs:**
| | Batch GD | Mini-batch GD | SGD |
| --- | --- | --- | --- |
| Cost per step | Full dataset | Batch (32–4096) | 1 sample |
| Path | Smooth | Mildly noisy | Very noisy |
| Large data | Slow / memory-bound | Best in practice | Fast, but poor hardware utilization |
| Non-convex | Can stall at saddles | Noise helps escape | Noise helps escape |

In deep learning, mini-batch with a schedule (warmup plus cosine decay) and an optimizer such as Adam or AdamW is the standard.

### Q42. Why is it important to scale the inputs when using SVMs?
> ✍️ _Drafted answer (not from the original notes)._

SVMs maximize the **geometric margin**, which is measured with Euclidean distances in feature space. If one feature ranges from 0 to 100,000 (income) and another from 0 to 1 (a ratio), the large-scale feature dominates the distances and the margin. The SVM effectively ignores the small-scale features and finds a poor boundary.

**Specific effects:**
- **Linear SVM:** the margin $2/\lVert w\rVert$ and the regularization on $w$ treat all dimensions equally, so the weight needed for a small-scale feature is heavily penalized and it gets suppressed.
- **RBF kernel:** $\exp(-\gamma\lVert x - x'\rVert^2)$ depends on raw distances. Without scaling, one feature determines the kernel values and a single $\gamma$ can't suit all dimensions.
- **Polynomial kernel:** large values raised to powers cause numerical overflow and instability.
- **Optimization:** solvers (SMO, liblinear) converge much more slowly on badly conditioned data.

**Practice:**
- Use **standardization** (zero mean, unit variance) or min-max scaling to $[0,1]$ or $[-1,1]$ (the LIBSVM guide recommends this).
- **Fit the scaler on training data only** and apply the same transform at inference. Put it in a pipeline to avoid leakage and training/serving skew.
- For sparse data (TF-IDF), scale without centering (`MaxAbsScaler`) to keep it sparse.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))
```

Tree-based models, by contrast, don't need scaling.

### Q43. What is OvR and OvO for multiclass classification, and which machine learning algorithm supports this
> ✍️ _Drafted answer (not from the original notes)._

These are two strategies for turning **binary classifiers into multiclass classifiers**.

**One-vs-Rest (OvR, a.k.a. One-vs-All):**
- Train $K$ binary classifiers, each separating one class from all others.
- Predict the class whose classifier gives the highest score or probability.
- **Pros:** only $K$ models, simple. **Cons:** each model trains on the full dataset, it creates artificial class imbalance (1 vs $K-1$), and the scores from different models may not be calibrated against each other.

**One-vs-One (OvO):**
- Train a classifier for every pair of classes: $K(K-1)/2$ models.
- Predict by majority voting (ties broken by confidence).
- **Pros:** each model trains only on two classes' data (small and balanced), which suits algorithms that scale badly with $n$. **Cons:** the number of models grows quadratically (100 classes → 4,950 models).

**Which algorithms use which:**
- **SVMs** are inherently binary. scikit-learn's `SVC` uses **OvO** internally (kernel SVMs scale superlinearly in $n$, so small pairwise problems are cheaper). `LinearSVC` uses **OvR**.
- **Logistic regression** can do OvR, or native multinomial (softmax), which is scikit-learn's default now.
- **Perceptron, SGDClassifier:** OvR.
- **Natively multiclass, no wrapper needed:** decision trees, random forests, gradient boosting (one tree per class per round, softmax), KNN, Naive Bayes, neural networks (softmax output).

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
OneVsRestClassifier(LogisticRegression()).fit(X, y)
OneVsOneClassifier(SVC()).fit(X, y)
```

### Q44. How will you do feature selection using Lasso Regression?
> ✍️ _Drafted answer (not from the original notes)._

Lasso adds an L1 penalty, $\min_\beta \frac{1}{2n}\lVert y - X\beta\rVert^2 + \alpha\lVert\beta\rVert_1$. The L1 term drives the coefficients of uninformative features **exactly to zero**, so the features with non-zero coefficients are the selected set.

**Steps:**
1. **Standardize the features.** The penalty is scale-dependent, and unscaled features get penalized unevenly.
2. **Choose α by cross-validation** (`LassoCV`, or `LogisticRegression(penalty="l1")` for classification). A larger α means fewer features. Use the "1-SE rule" (the largest α within one standard error of the best CV score) if you want a sparser, more robust set.
3. **Keep the features with $\beta_j \ne 0$.** Optionally plot the **regularization path** to see the order in which features enter.
4. Optionally **refit** an unpenalized or different model on the selected features (relaxed Lasso), because Lasso shrinks the surviving coefficients.

```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), SelectFromModel(LassoCV(cv=5)), Ridge())
pipe.fit(X_train, y_train)
selected = X.columns[pipe[1].get_support()]
```

**Caveats:**
- **Correlated features:** Lasso tends to pick one arbitrarily and drop the others, and the selection is unstable across samples. Use **Elastic Net**, or **stability selection** (Lasso on bootstraps, keeping features selected in, say, more than 70% of runs).
- It captures only linear (or log-odds-linear) relationships, so non-linear signal can be dropped.
- Do selection **inside the CV loop**. Selecting on the full dataset and then cross-validating leaks information and inflates the scores.
- When $p > n$, Lasso selects at most $n$ features.

### Q45. What is the difference between standard scaler and minmax scaler? What will you do if there is a categorical variable?
> ✍️ _Drafted answer (not from the original notes)._

**StandardScaler** (z-score): $x' = (x - \mu)/\sigma$, giving zero mean and unit variance with an unbounded range.
**MinMaxScaler**: $x' = (x - x_{min})/(x_{max} - x_{min})$, mapping to a fixed range, usually $[0,1]$.

| | StandardScaler | MinMaxScaler |
| --- | --- | --- |
| Output range | Unbounded (mostly −3 to 3) | Bounded [0, 1] |
| Outliers | Affected (mean/std), but less squashed | Very sensitive: one outlier compresses everything else |
| Good for | Linear/logistic regression, SVM, PCA, roughly Gaussian features, gradient descent | Neural nets that expect bounded inputs, image pixels, KNN with known bounds |
| Shape | Preserved (not made normal) | Preserved |

With heavy outliers, use **RobustScaler** (median/IQR). For sparse data, use **MaxAbsScaler**. Neither scaler makes data Gaussian; for that, use power or quantile transforms.

**Categorical variables:**
- **Don't scale raw categories or label codes.** Encode them first: one-hot for low cardinality, ordinal encoding only for genuinely ordered categories, target, frequency or hashing encoding for high cardinality.
- One-hot (0/1) columns generally **don't need scaling**. Some people standardize them for regularized linear models so the penalty treats them comparably, but this loses sparsity and interpretability.
- Apply the scaler only to numeric columns with a `ColumnTransformer`, and fit everything on the training data only.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
```

Tree-based models need neither scaling nor one-hot encoding in many implementations.

### Q46. What types of models tend to overfit?
> ✍️ _Drafted answer (not from the original notes)._

Models with **high capacity and few built-in constraints** relative to the amount of data overfit most easily.

**Prone to overfitting:**
- **Deep, unpruned decision trees:** can create one leaf per sample. The classic high-variance model.
- **KNN with small k** (k = 1 memorizes the training set).
- **Kernel SVM with large γ and large C.**
- **High-degree polynomial regression** and linear models with many features relative to samples ($p \gg n$) and no regularization.
- **Large neural networks** trained long on small datasets without dropout, weight decay, augmentation or early stopping.
- **Gradient boosting** with too many rounds, deep trees or a high learning rate. Boosting keeps fitting residuals, including the noise.
- **Non-parametric and flexible models** in general (Gaussian processes with short length scales, splines with many knots).
- Complex ensembles like stacking, if the meta-learner is fit on in-sample predictions.

**Less prone:** regularized linear models, Naive Bayes, shallow trees, random forests (bagging averages away variance), large-k KNN. These tend to underfit instead.

**Aggravating factors (model-independent):** small or noisy datasets, too many features, data leakage, heavy hyperparameter tuning on one validation set, and training on non-representative data.

**Controls:** regularization, pruning, early stopping, dropout, cross-validation, more data or augmentation, feature selection and ensembling. The bias–variance tradeoff says to match model capacity to data size and noise.

### Q47. What are some advantages and Disadvantages of regression models and tree-based models?
> ✍️ _Drafted answer (not from the original notes)._

**Regression (linear/logistic) models**

*Advantages:*
- Highly **interpretable**: coefficients give direction and magnitude, with confidence intervals and p-values. Preferred in regulated domains (credit scoring).
- Fast to train and serve, small memory footprint, works well with few samples.
- Extrapolates (linearly) beyond the training range.
- Well-calibrated probabilities (logistic regression).
- Works well on high-dimensional sparse data (text).

*Disadvantages:*
- Assumes linearity and additivity: needs manual feature engineering for interactions and non-linearities.
- Sensitive to outliers, multicollinearity and unscaled features.
- Needs careful handling of missing values and categorical encoding.

**Tree-based models (decision trees, RF, GBMs such as XGBoost/LightGBM)**

*Advantages:*
- Capture **non-linearities and interactions automatically**.
- Robust to outliers in features and monotonic transformations; **no scaling needed**.
- Handle mixed feature types, and some implementations handle missing values natively.
- **State of the art on tabular data** (GBMs).
- Built-in feature importance; SHAP for explanations.

*Disadvantages:*
- **Cannot extrapolate**: predictions are piecewise-constant within the training target range (bad for trending time series).
- Single trees have high variance and are unstable. Ensembles lose interpretability.
- Larger models and slower inference for big ensembles; many hyperparameters to tune.
- Probabilities are often poorly calibrated (RF, boosted trees) and need Platt or isotonic calibration.
- Weak on very sparse, high-dimensional data compared with linear models.

**Rule of thumb:** start with a regularized linear baseline for interpretability and sanity, move to GBMs for maximum tabular accuracy, and use a hybrid (linear model plus tree-engineered features, or GBM with monotonic constraints) when you need both.

### Q48. What are some important hyperparameters for XGBOOST
> ✍️ _Drafted answer (not from the original notes)._

Group them by what they control:

**Boosting process**
- `n_estimators` (number of rounds) and `learning_rate` / `eta` (0.01–0.3). They trade off: a lower learning rate needs more trees but generalizes better. Use **early stopping** (`early_stopping_rounds`) on a validation set instead of fixing the tree count.

**Tree complexity (main overfitting controls)**
- `max_depth` (typically 3–10): depth of each tree, i.e. the order of interactions.
- `min_child_weight`: minimum sum of instance weights (hessian) in a leaf. Higher is more conservative.
- `gamma` / `min_split_loss`: minimum loss reduction needed to make a split.
- `max_leaves`, `grow_policy` (depthwise vs lossguide).

**Randomness (variance reduction)**
- `subsample` (row sampling per tree, 0.5–1.0).
- `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` (feature sampling).

**Regularization**
- `reg_lambda` (L2 on leaf weights, default 1) and `reg_alpha` (L1 on leaf weights).

**Problem-specific**
- `objective` (e.g. `binary:logistic`, `reg:squarederror`, `rank:ndcg`) and `eval_metric`.
- `scale_pos_weight` for class imbalance (≈ negatives/positives).
- `max_delta_step`: helps logistic regression with extreme imbalance.
- `monotone_constraints`, `interaction_constraints` for domain knowledge or regulation.

**System**
- `tree_method="hist"` (fast, and `device="cuda"` for GPU), `max_bin`, `n_jobs`.

**Tuning recipe:** fix a moderately high learning rate (0.1) with early stopping, then tune `max_depth` and `min_child_weight`, then `gamma`, then the subsample and colsample parameters, then the regularization, and finally lower the learning rate and raise the rounds. Use Optuna or Bayesian search with CV.

### Q49. Can you tell the complete life cycle of a data science project?
> ✍️ _Drafted answer (not from the original notes)._

A typical end-to-end lifecycle (CRISP-DM extended with MLOps):

1. **Problem framing:** understand the business objective, define success metrics (business KPI and ML metric), constraints (latency, cost, interpretability, regulation), and whether ML is even needed. Agree on a baseline (e.g. the current rule-based system).
2. **Data acquisition:** identify sources, access, governance and privacy (PII, consent), and labelling strategy. Check volume, freshness and point-in-time availability.
3. **Data understanding / EDA:** distributions, missing values, outliers, leakage risks, class balance, data quality issues. Iterate with domain experts.
4. **Data preparation and feature engineering:** cleaning, imputation, encoding, scaling, feature creation, train/validation/test split that mirrors production (time or group splits). Build reproducible pipelines, possibly a feature store.
5. **Modeling:** start with simple baselines, then iterate on algorithms. Hyperparameter tuning with proper cross-validation, experiment tracking (MLflow, W&B).
6. **Evaluation:** offline metrics on a holdout set, error analysis by slice, fairness and robustness checks, calibration. Sign-off against the agreed success criteria.
7. **Deployment:** package the model (container, model registry), serve it batch or real-time, test it (unit, integration, data contracts), roll out via shadow, canary or A/B test.
8. **Monitoring:** data and prediction drift, model performance as labels arrive, latency, cost, errors. Alerting and dashboards.
9. **Maintenance and iteration:** retraining (scheduled or drift-triggered), feedback loops, versioning of data, code and models, rollback plans, eventually deprecation.
10. **Communication throughout:** stakeholder updates, documentation (model cards), and translating results into business impact.

The process is **iterative, not linear**. Most time goes into the data steps, and most failures come from poor problem framing or missing monitoring, not from the choice of algorithm.

### Q50. What are the properties of a good ML model?
> ✍️ _Drafted answer (not from the original notes)._

A good ML model is judged by more than its validation score:

- **Generalizes:** performs well on unseen data from the production distribution, not only on the test split (low variance, low enough bias).
- **Fit for the business metric:** optimizes the thing the business cares about (cost of errors, revenue, latency), not just accuracy.
- **Robust:** stable under noise, outliers, missing values and moderate distribution shift; small input changes don't flip predictions wildly.
- **Calibrated:** predicted probabilities mean what they say, so thresholds and downstream decisions are trustworthy.
- **Interpretable enough:** stakeholders and regulators can understand why it made a decision (inherently simple models or SHAP-style explanations).
- **Fair:** no unjustified performance gaps across user groups.
- **Efficient:** meets latency, memory and cost budgets at serving time; trains in a reasonable time.
- **Reproducible and maintainable:** versioned data, code and config; simple enough to retrain and debug.
- **Monitorable:** you can detect drift and degradation after deployment.

**Interview angle:** the "best" model is often not the most accurate one. A gradient-boosted model 0.5% worse in AUC but 10x cheaper, explainable and easy to retrain usually wins in production. Frame the answer as a tradeoff between accuracy, cost, risk and maintainability.

### Q51. What are the different evaluation metrics for a regression model?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q52. What are the different evaluation metrics for a classification model?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q53. Difference between R2 and adjusted R2? Why do you prefer adjusted R2?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q54. List some of the drawbacks of a Linear model
> ✍️ _Drafted answer (not from the original notes)._

**Drawbacks of linear models (linear/logistic regression):**

- **Assumes a linear relationship** between features and target (or log-odds). Real relationships are often nonlinear, causing underfitting (high bias) unless you hand-engineer polynomial terms, splines or interactions.
- **No automatic interactions:** the effect of one feature cannot depend on another unless you add the cross term yourself.
- **Sensitive to outliers** (squared loss), which can drag the fit.
- **Multicollinearity** makes coefficients unstable and hard to interpret.
- **Statistical assumptions** (independent errors, homoscedasticity, normal residuals) are needed for valid confidence intervals and p-values.
- **Needs preprocessing:** scaling for regularization, encoding of categoricals, imputation of missing values.
- **Extrapolation risk:** predictions grow without bound outside the training range.
- **Limited capacity** for high-dimensional raw data like images, audio or text without feature engineering or embeddings.

**Why they're still used:** fast, cheap, highly interpretable, robust with little data, easy to regularize (Ridge/Lasso) and to deploy. In production they're often a strong baseline, a calibrated scoring layer, or the model of choice in regulated domains (credit scoring) where explainability outweighs a few points of accuracy.

### Q55. What do you mean by the curse of Dimensionality?
> ✍️ _Drafted answer (not from the original notes)._

The **curse of dimensionality** is the set of problems that appear as the number of features grows:

- **Data becomes sparse:** the volume of the space grows exponentially, so a fixed number of samples covers it ever more thinly. To keep the same density in 10 dimensions as with 10 points in 1 dimension you'd need $10^{10}$ points.
- **Distances lose meaning:** in high dimensions, the nearest and farthest neighbours of a point end up at almost the same distance, which breaks k-NN, k-means and other distance-based methods.
- **Overfitting risk rises:** with many features and few samples, the model can always find spurious patterns (the $p \gg n$ problem).
- **Compute and memory cost** grow, and visualization becomes impossible.

**Mitigations:**

- Feature selection (filter, wrapper, L1 regularization).
- Dimensionality reduction (PCA, autoencoders, UMAP for visualization).
- Regularization and simpler models.
- Learned embeddings instead of huge one-hot/sparse vectors.
- More data, or domain knowledge to engineer fewer, better features.

**Nuance:** real data usually lies near a low-dimensional manifold, which is why deep nets work on million-pixel images. The curse bites hardest for methods that rely on raw Euclidean distance over many uninformative features.

### Q56. Explain the kernel trick in SVM
> ✍️ _Drafted answer (not from the original notes)._

An SVM's dual formulation and decision function only use **dot products** between data points:

$f(x) = \text{sign}\left(\sum_i \alpha_i y_i \langle x_i, x\rangle + b\right)$

If the data isn't linearly separable, we could map it to a higher-dimensional space $\phi(x)$ where it is, and then compute $\langle \phi(x_i), \phi(x)\rangle$. The **kernel trick** replaces that dot product with a kernel function $K(x_i, x) = \langle \phi(x_i), \phi(x)\rangle$ computed directly in the original space, so we **never construct $\phi(x)$ explicitly**.

**Example:** for $x \in \mathbb{R}^2$, the polynomial kernel $K(x,z) = (x \cdot z)^2$ equals the dot product of the explicit features $(x_1^2, \sqrt2 x_1x_2, x_2^2)$. The RBF kernel $K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$ corresponds to an **infinite-dimensional** feature space, which would be impossible to compute explicitly.

**Requirements:** $K$ must be a valid (positive semi-definite, Mercer) kernel.

**Tradeoffs:**

- Lets a linear algorithm learn nonlinear boundaries cheaply.
- Training needs the $n \times n$ kernel matrix, so cost is roughly $O(n^2)$ to $O(n^3)$; kernel SVMs don't scale well past ~100k samples. In practice, use linear SVMs, random Fourier features / Nystrom approximations, or gradient boosting / neural nets at scale.
- Kernel choice and hyperparameters ($\gamma$, $C$) need tuning.

### Q57. Why is it sometimes needed to scale or normalise features?
> ✍️ _Drafted answer (not from the original notes)._

Feature scaling puts features on comparable ranges. It's needed when an algorithm is sensitive to the **magnitude** of features:

- **Distance-based models** (k-NN, k-means, SVM with RBF kernel): a feature measured in thousands (income) would dominate one measured in units (age), regardless of importance.
- **Gradient-descent-trained models** (linear/logistic regression, neural nets): unscaled features create an elongated loss surface, so gradient descent zig-zags and converges slowly or needs a tiny learning rate.
- **Regularized models** (Ridge, Lasso): the penalty on coefficients is only fair if features share a scale; otherwise small-scale features are penalized more.
- **PCA:** it finds directions of maximum variance, so large-scale features dominate the components.

**Not needed for:** tree-based models (decision trees, random forests, gradient boosting), since splits depend on the order of values, not their scale. Naive Bayes is also mostly unaffected.

**Common methods:** standardization (zero mean, unit variance), min-max scaling to [0, 1], robust scaling (median/IQR, outlier-resistant), log transforms for skewed data.

**Production pitfall:** fit the scaler **only on the training data** (inside the CV pipeline) and save it with the model, otherwise you leak test information and create training/serving skew.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
model = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(X_train, y_train)
```

### Q58. Why do binary\_crossentropy and categorical\_crossentropy give different performances for the same problem?
> ✍️ _Drafted answer (not from the original notes)._

Usually because they are being used for **different problem formulations**, or because of how Keras computes accuracy.

**1. The metric, not the model, differs (the classic Keras gotcha).** With `metrics=['accuracy']`, Keras picks the accuracy function from the loss. With `binary_crossentropy` on a multi-class softmax output, it uses **binary accuracy**, which thresholds each of the $K$ outputs independently at 0.5 and averages them. For 10 classes, predicting all zeros already scores ~90% "accuracy". The model isn't better; the reported number is misleading. Fix: specify `CategoricalAccuracy` explicitly.

**2. They optimize different things.**

- `categorical_crossentropy` + **softmax**: one label per sample; classes compete, probabilities sum to 1. Loss $= -\sum_k y_k \log p_k$.
- `binary_crossentropy` + **sigmoid**: each output is an independent yes/no problem (multi-label). Loss sums $K$ separate Bernoulli losses, also penalizing the probability mass on negative classes.

Using the wrong pairing gives different gradients and a different optimum, so training dynamics and final performance differ.

**Rule of thumb:**

| Task | Output | Loss |
| --- | --- | --- |
| Binary | 1 sigmoid | binary_crossentropy |
| Multi-class (one label) | K softmax | categorical / sparse_categorical_crossentropy |
| Multi-label | K sigmoid | binary_crossentropy |

For two classes, 1 sigmoid + BCE and 2 softmax + CCE are mathematically equivalent.

### Q59. [Why does one-hot encoding improve machine learning performance?](https://stackoverflow.com/questions/17469835/why-does-one-hot-encoding-improve-machine-learning-performance)
> ✍️ _Drafted answer (not from the original notes)._

One-hot encoding turns a categorical feature with $k$ values into $k$ binary columns. It helps because:

- **Removes a fake ordering.** Label encoding (red=0, green=1, blue=2) tells a linear model or a distance-based model that blue is "twice" green and that green lies between red and blue. One-hot treats categories as unordered and equidistant.
- **Lets linear models learn a separate weight per category.** With label encoding, a linear model can only learn one slope across the codes, i.e. a monotonic effect. With one-hot, each category gets its own coefficient, so the model can capture arbitrary, nonlinear effects of the category. It effectively increases model capacity in the right way.
- **Works with any algorithm that expects numeric input** without introducing bias from arbitrary integer codes.

**When it doesn't help or hurts:**

- **High cardinality** (zip codes, user IDs): thousands of sparse columns, memory blow-up and overfitting. Prefer target/mean encoding (with out-of-fold fitting to avoid leakage), hashing, frequency encoding or learned embeddings.
- **Tree-based models** can often split label-encoded categories fine, and LightGBM/CatBoost handle categoricals natively, often better than one-hot.
- **Dummy-variable trap** in unregularized linear regression: drop one column (`drop='first'`) to avoid perfect collinearity with the intercept.
- In production, handle unseen categories (`handle_unknown='ignore'`).

### Q60. Considering the long list of machine learning algorithms, given a data set, how do you decide which one to use?
> ✍️ _Drafted answer (not from the original notes)._

There's no universally best algorithm (No Free Lunch), so the choice is driven by the problem, the data and the constraints, then confirmed empirically.

**1. Define the problem:** supervised (classification, regression, ranking), unsupervised (clustering, anomaly detection) or sequential decision-making (RL). What metric and what error costs matter?

**2. Look at the data:**

- **Size:** small data favours simple, regularized models (logistic regression, Naive Bayes, SVM); large data allows boosting and deep learning.
- **Type:** tabular → gradient boosting (XGBoost/LightGBM/CatBoost) is the usual winner; images → CNNs/ViTs; text → Transformers or TF-IDF + linear baseline; time series → seasonal baselines, ARIMA, boosting with lag features.
- **Dimensionality, sparsity, missing values, categorical cardinality, label noise, class imbalance.**

**3. Constraints:** interpretability or regulatory needs, latency and memory at inference, training budget, how often you must retrain.

**4. Empirical process:**

- Start with a trivial baseline (majority class, mean) and a simple model.
- Try a few strong candidate families with cross-validation and the right metric.
- Tune the best one or two; compare on a held-out test set.
- Do error analysis before adding complexity.

**Example:** churn prediction on 200k rows of tabular CRM data with a requirement to explain decisions → logistic regression baseline, then LightGBM with SHAP explanations; pick LightGBM if the uplift justifies it.

### Q61. Differentiate between wide and tall data formats?
> ✍️ _Drafted answer (not from the original notes)._

Wide and tall (long) are two layouts of the same data.

**Wide format:** one row per entity, with a column for each variable or time point.

| customer | jan | feb | mar |
| --- | --- | --- | --- |
| A | 10 | 12 | 9 |

**Tall / long format:** one row per observation, with identifier, variable name and value columns.

| customer | month | sales |
| --- | --- | --- |
| A | jan | 10 |
| A | feb | 12 |
| A | mar | 9 |

**Differences and uses:**

- **Tall** is "tidy" data: easy to filter, group, aggregate and append new time periods without schema changes. Preferred for storage in databases, time-series tools, plotting libraries (seaborn, ggplot) and event logs.
- **Wide** is what most ML models expect: one row per sample with a fixed feature vector. Also easier for humans to read in spreadsheets.
- Wide tables become unwieldy with many variables and contain many nulls when data is sparse.

**Converting:**

```python
long = wide.melt(id_vars="customer", var_name="month", value_name="sales")
wide = long.pivot(index="customer", columns="month", values="sales")
```

In feature engineering pipelines you often store data tall (events) and pivot to wide (one feature row per entity) right before training.

### Q62. How will you know which machine learning algorithm to choose for your classification problem?
> ✍️ _Drafted answer (not from the original notes)._

Choosing a classifier is a mix of constraints and experiments:

**Questions to ask first**

- **How much data?** Few hundred rows → logistic regression, Naive Bayes, small trees, SVM. Hundreds of thousands+ → gradient boosting, neural nets.
- **What kind of data?** Tabular → gradient-boosted trees (LightGBM/XGBoost/CatBoost) are the default strong choice. Text → TF-IDF + logistic regression baseline, then fine-tuned Transformers. Images/audio → CNNs/ViTs, usually via transfer learning.
- **Is the boundary likely linear?** If yes, linear models; if not, trees, kernel SVM, neural nets.
- **Interpretability needs?** Regulated decisions → logistic regression, shallow trees, or boosting + SHAP.
- **Latency/memory at inference?** Edge or high-QPS services favour small linear or tree models.
- **Imbalance, noise, missing values, high-cardinality categoricals?** Trees handle missing values and mixed types well; Naive Bayes is robust with tiny data; k-NN suffers in high dimensions.
- **Do you need calibrated probabilities?** Logistic regression is naturally calibrated; others may need Platt/isotonic calibration.

**Then validate empirically**

1. Baseline (majority class / simple rule).
2. Shortlist 2–4 model families, evaluate with stratified k-fold on the business metric (e.g. PR-AUC for rare fraud).
3. Tune, inspect errors per segment, check calibration and latency.
4. Pick the simplest model within noise of the best.

**Example:** loan default with 50k rows, mixed features, explainability required → logistic regression baseline vs. LightGBM with monotonic constraints and SHAP.

### Q63. How will you find the correlation between a categorical variable and a continuous variable?
> ✍️ _Drafted answer (not from the original notes)._

It depends on whether the categorical variable is binary, nominal or ordinal:

- **Binary categorical (2 levels):** use the **point-biserial correlation**, which is Pearson correlation with the category coded 0/1. Ranges from -1 to 1.
- **Nominal with k levels:** use the **correlation ratio $\eta$ (eta)**, derived from ANOVA: the share of variance in the continuous variable explained by group membership.

  $\eta^2 = \frac{SS_{between}}{SS_{total}}$, ranging 0–1. Test significance with a one-way **ANOVA F-test** (or **Kruskal-Wallis** if the data is non-normal).
- **Ordinal categorical:** use **Spearman's rank correlation** or **Kendall's tau**, which respect order without assuming equal spacing.
- **Model-based alternatives:** mutual information (captures nonlinear dependence), or the AUC of predicting the binary category from the continuous variable.

```python
import pandas as pd
def correlation_ratio(cat, num):
    df = pd.DataFrame({"c": cat, "y": num})
    grand = df.y.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in df.groupby("c").y)
    ss_total = ((df.y - grand) ** 2).sum()
    return (ss_between / ss_total) ** 0.5
```

**Interview caveats:** don't just label-encode a nominal variable and compute Pearson, since the result depends on the arbitrary codes. Association isn't causation, and with large samples tiny effects become statistically significant, so report the effect size ($\eta$), not just the p-value.

### Q64. What are the differences between “Bayesian” and “Frequentist” approaches for Machine Learning?
> ✍️ _Drafted answer (not from the original notes)._

The two schools differ in **what probability means**.

| | Frequentist | Bayesian |
| --- | --- | --- |
| Probability | Long-run frequency of repeated events | Degree of belief, updated with evidence |
| Parameters | Fixed but unknown constants | Random variables with distributions |
| Prior knowledge | Not formally used | Encoded in a **prior** $p(\theta)$ |
| Output | Point estimate (e.g. MLE) + confidence interval / p-value | Full **posterior** $p(\theta \mid D) \propto p(D \mid \theta)p(\theta)$ + credible interval |
| Typical ML | MLE, standard SGD-trained nets, hypothesis tests | Bayesian linear regression, Gaussian processes, Bayesian optimization, MCMC/variational inference |

**Links in practice:**

- **MAP estimation** = MLE + prior. L2 regularization equals a Gaussian prior on the weights; L1 equals a Laplace prior. So most regularized ML is "Bayesian-flavoured".
- With lots of data the likelihood dominates and both give similar answers.

**Tradeoffs:**

- **Bayesian pros:** principled uncertainty estimates, natural use of prior knowledge, works well with small data, good for sequential updating (A/B tests, bandits like Thompson sampling).
- **Bayesian cons:** priors can be subjective; exact posteriors are usually intractable, so MCMC or variational approximations are slow or approximate at scale.
- **Frequentist pros:** simpler, scalable, objective procedures, well-established tests.
- **Frequentist cons:** confidence intervals and p-values are easy to misinterpret; no direct probability statements about parameters.

**Production example:** Bayesian A/B testing gives "P(variant B is better) = 94%", which is often easier for stakeholders than a p-value.

### Q65. What is the difference between a Gaussian Mixture Model and the K-Means Algorithm?
> ✍️ _Drafted answer (not from the original notes)._

Both cluster data into $k$ groups, but GMM is a probabilistic generalization of k-means.

| | K-Means | Gaussian Mixture Model |
| --- | --- | --- |
| Assignment | **Hard**: each point belongs to one cluster | **Soft**: probability of belonging to each component |
| Cluster shape | Spherical, similar size (Euclidean distance to centroid) | Elliptical, any orientation and size (full covariance per component) |
| Parameters | Centroids | Means, covariances, mixing weights |
| Algorithm | Lloyd's algorithm (assign, update) | Expectation-Maximization (E-step: responsibilities, M-step: parameters) |
| Objective | Minimize within-cluster sum of squares | Maximize data likelihood |
| Output | Labels | Labels + probabilities + a density model |
| Cost | Fast, scales to large data | Slower; covariance estimation costly in high dimensions |
| Choosing $k$ | Elbow, silhouette | **BIC/AIC** (proper likelihood-based model selection) |

**Relationship:** k-means is the limiting case of a GMM with equal, spherical covariances shrinking to zero (so responsibilities become 0/1).

**When to use GMM:** overlapping or elongated clusters, when you need uncertainty of assignment (e.g. a customer that's 60% "bargain hunter", 40% "loyal"), or when you want a density estimate for anomaly detection (low likelihood = anomaly).

**When to use k-means:** large datasets, roughly spherical clusters, need for speed and simplicity. A common pattern is initializing GMM with k-means centroids.

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4, covariance_type="full").fit(X)
probs = gmm.predict_proba(X)
```

### Q66. Is more data always better?
> ✍️ _Drafted answer (not from the original notes)._

**No.** More data usually helps, but not always, and not for free.

**When more data helps:**

- The model has **high variance** (overfitting): learning curves show a train/validation gap that shrinks as data grows.
- The new data covers **underrepresented cases** (rare classes, new regions, edge cases).
- High-capacity models (deep nets, LLMs) that keep improving with scale.

**When it doesn't help or hurts:**

- **High bias:** a too-simple model plateaus; learning curves are flat and close together. You need better features or a more expressive model, not more rows.
- **Low quality:** noisy labels, duplicates, or wrong data can lower performance. Quality beats quantity (a key lesson from LLM data curation).
- **Distribution mismatch:** data from a different population, an outdated period, or another domain can pull the model away from what production sees.
- **Bias amplification:** more of a skewed source makes the imbalance worse.
- **Cost:** labeling, storage, compute, privacy and compliance risk all grow; returns are typically diminishing (often roughly logarithmic).

**How to decide in practice:** plot a **learning curve** (validation score vs. training-set size). If it's still rising, collecting data is worthwhile; if it has flattened, invest in features, model changes or label quality instead. Targeted collection via active learning or error analysis usually beats collecting more of the same.

### Q67. How can you determine which features are the most important in your model?
> ✍️ _Drafted answer (not from the original notes)._

Several methods, each with caveats:

**Model-specific**

- **Linear models:** magnitude of coefficients **after standardizing** features (and only if multicollinearity is low).
- **Tree ensembles:** impurity-based importance (split gain). Fast, but biased toward high-cardinality and continuous features, and computed on training data.

**Model-agnostic**

- **Permutation importance:** shuffle one feature on a validation set and measure the metric drop. Reflects actual predictive value; can underestimate correlated features (the model uses the twin).
- **SHAP values:** game-theoretic attribution per prediction; aggregate mean $\lvert\text{SHAP}\rvert$ for global importance, and shows direction and interactions. TreeSHAP is fast for tree models.
- **Drop-column importance:** retrain without the feature; most faithful but expensive.
- **Partial dependence / ICE plots:** show how predictions change with a feature.

**Statistical / filter methods** (before modeling): mutual information, correlation, chi-square, ANOVA F-test.

**Embedded selection:** L1 (Lasso) shrinks irrelevant coefficients to zero.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val, n_repeats=10, scoring="roc_auc")
```

**Pitfalls to mention:** importance is not causality; correlated features split or hide importance; a suspiciously dominant feature often signals **data leakage** (e.g. a field populated after the outcome). Validate on held-out data and check stability across CV folds.

### Q68. Which hyperparameter tuning strategies (in general) do you know?
> ✍️ _Drafted answer (not from the original notes)._

**Main strategies:**

- **Manual tuning:** guided by experience; fine for a few knobs, not reproducible.
- **Grid search:** try every combination on a grid. Exhaustive and parallel, but cost grows exponentially with the number of hyperparameters and wastes trials on unimportant ones.
- **Random search:** sample configurations from distributions. Usually beats grid search for the same budget because only a few hyperparameters really matter (Bergstra & Bengio, 2012).
- **Bayesian optimization:** build a surrogate model (Gaussian process, TPE) of score vs. hyperparameters and pick the next trial with an acquisition function (expected improvement). Sample-efficient for expensive training runs. Tools: Optuna, Hyperopt, scikit-optimize.
- **Multi-fidelity / early stopping:** Successive Halving, **Hyperband**, ASHA: start many configs on small budgets (few epochs, data subsets) and only continue the promising ones. Often combined with Bayesian optimization (BOHB, Optuna pruners).
- **Population-based training (PBT):** evolve hyperparameters during training (used for RL and large nets).
- **Evolutionary / genetic algorithms**, and **gradient-based** HPO (rare in practice).

**Good practice:**

- Tune with **cross-validation or a fixed validation set**, never the test set.
- Use log-uniform ranges for learning rates and regularization strengths.
- Tune the most influential parameters first (learning rate, depth, regularization).
- Track every trial (MLflow/W&B) and fix seeds; report variance across seeds.

```python
import optuna
def objective(trial):
    params = {"learning_rate": trial.suggest_float("lr", 1e-3, 0.3, log=True),
              "max_depth": trial.suggest_int("max_depth", 3, 10)}
    return cv_score(params)
optuna.create_study(direction="maximize").optimize(objective, n_trials=100)
```

### Q69. How to select K for K-means?
> ✍️ _Drafted answer (not from the original notes)._

$k$ is a hyperparameter, so you combine quantitative criteria with domain sense:

- **Elbow method:** plot within-cluster sum of squares (inertia) vs. $k$ and pick the point where adding clusters stops giving big drops. Easy, but the elbow is often ambiguous.
- **Silhouette score:** for each point, $(b-a)/\max(a,b)$ where $a$ is mean intra-cluster distance and $b$ is mean distance to the nearest other cluster. Choose the $k$ that maximizes the average (range -1 to 1).
- **Gap statistic:** compares inertia against what you'd get on uniformly random reference data; choose the smallest $k$ within one standard error of the maximum gap.
- **Davies-Bouldin** (lower is better) and **Calinski-Harabasz** (higher is better) indices.
- **Information criteria:** if you switch to a Gaussian Mixture Model, use BIC/AIC.
- **Stability:** run on bootstrap samples; good $k$ values give consistent clusters.
- **Business/domain constraints:** marketing may only be able to act on 4–6 segments; that can override a marginally better statistical choice.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
scores = {k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
          for k in range(2, 11)}
best_k = max(scores, key=scores.get)
```

**Caveats:** scale features first; k-means assumes roughly spherical clusters, so if no $k$ looks good, the issue may be the algorithm (try DBSCAN/HDBSCAN or GMM) rather than $k$.

### Q70. Describe the differences between and use cases for box plots and histograms
> ✍️ _Drafted answer (not from the original notes)._

Both show the distribution of a numeric variable, at different levels of detail.

**Histogram**

- Bins the values and shows the count (or density) in each bin.
- Reveals the **shape**: skewness, multiple modes (bimodality), gaps, heavy tails.
- Sensitive to bin width choice; awkward for comparing many groups.
- **Use when:** exploring one variable in depth, checking normality before modeling, spotting bimodal populations (e.g. two customer types), choosing transformations (log for right skew).

**Box plot**

- Five-number summary: median, Q1, Q3 (the box = IQR), whiskers (typically to 1.5×IQR) and individual **outliers** beyond them.
- Compact and robust; makes **comparing many groups side by side** easy.
- Hides shape: a bimodal and a unimodal distribution can produce identical boxes.
- **Use when:** comparing a metric across categories (latency per region, salary per department), quickly flagging outliers, dashboards with limited space.

**Practical tip:** violin plots or box plots overlaid with strip/jitter points combine both views. In EDA I'd use histograms per feature to understand shape and box plots of the feature by target class to see which features separate classes.

### Q71. How would you differentiate between multi-label and multi-class classification?
> ✍️ _Drafted answer (not from the original notes)._

**Multi-class:** each sample belongs to **exactly one** of $K > 2$ mutually exclusive classes.
Example: classifying a digit as 0–9, or a ticket as Billing / Technical / Sales.

- Output layer: $K$ units with **softmax** (probabilities sum to 1); predict the argmax.
- Loss: categorical cross-entropy.
- Metrics: accuracy, macro/micro F1, confusion matrix.
- Strategies for binary algorithms: One-vs-Rest, One-vs-One.

**Multi-label:** each sample can have **zero, one or several** labels at once.
Example: tagging a movie as Action + Comedy, or an image containing both a dog and a car.

- Output layer: $K$ independent **sigmoid** units; a threshold per label (tuned separately, not necessarily 0.5).
- Loss: binary cross-entropy summed over labels.
- Metrics: Hamming loss, subset accuracy (exact match), per-label and micro/macro F1, precision@k.
- Strategies: Binary Relevance (one classifier per label), Classifier Chains (capture label correlations), or a single network with sigmoid outputs.

**Quick test:** "Can two labels be true at the same time?" If yes, it's multi-label.

**Production gotchas:** multi-label often has strong label imbalance (rare tags), so threshold tuning per label and label co-occurrence analysis matter. Using softmax for a multi-label problem silently forces labels to compete and caps recall.

### Q72. What is KL divergence, and how would you define its use case in ML?
> ✍️ _Drafted answer (not from the original notes)._

**KL divergence** measures how much one probability distribution $P$ differs from a reference distribution $Q$:

$D_{KL}(P \Vert Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$

Interpretation: the expected extra bits (nats) needed to encode samples from $P$ using a code optimized for $Q$.

**Properties:** always $\ge 0$, zero only if $P = Q$; **not symmetric** ($D_{KL}(P\Vert Q) \ne D_{KL}(Q\Vert P)$), so it's not a true distance; infinite if $Q(x) = 0$ where $P(x) > 0$. Forward KL is "mean-seeking" (covers all modes); reverse KL is "mode-seeking".

**Use cases in ML:**

- **Cross-entropy loss:** $H(P,Q) = H(P) + D_{KL}(P\Vert Q)$, so minimizing cross-entropy against fixed labels minimizes KL.
- **VAEs:** the KL term pushes the encoder's latent distribution toward the prior $\mathcal{N}(0, I)$.
- **Knowledge distillation:** KL between teacher and student softened output distributions.
- **RLHF / PPO / DPO:** a KL penalty keeps the fine-tuned LLM close to the reference model, preventing reward hacking and drift.
- **Variational inference:** minimize $D_{KL}(q \Vert p)$ to approximate posteriors.
- **Drift monitoring:** compare feature or prediction distributions between training and production (often with the symmetric, bounded Jensen-Shannon divergence or PSI instead).
- **t-SNE:** minimizes KL between high- and low-dimensional neighbour distributions.

```python
import numpy as np
def kl(p, q, eps=1e-12):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    return float(np.sum(p * np.log(p / q)))
```

### Q73. Can you define the concept of Undersampling and Oversampling?
> ✍️ _Drafted answer (not from the original notes)._

Both are **resampling** techniques for imbalanced classification that change the class distribution of the **training** data.

**Undersampling:** remove samples from the majority class.

- Random undersampling, or smarter variants: Tomek links, NearMiss, Edited Nearest Neighbours (remove borderline/noisy majority points).
- **Pros:** smaller, faster training set. **Cons:** throws away information; can increase variance. Works best when the majority class is huge (e.g. billions of non-fraud transactions). Ensembles like EasyEnsemble / balanced random forest reduce the information loss by undersampling differently per model.

**Oversampling:** add samples to the minority class.

- Random oversampling (duplicates), **SMOTE** (interpolates synthetic points between minority neighbours), ADASYN, borderline-SMOTE; data augmentation for images/text.
- **Pros:** keeps all data. **Cons:** duplicates encourage overfitting; SMOTE can create unrealistic points and noise in overlapping regions, and works poorly with high-dimensional or categorical data.

**Critical rules:**

- Resample **only the training folds, inside cross-validation**, never before splitting; otherwise synthetic copies leak into validation and inflate scores.
- Evaluate on the **original distribution** with PR-AUC, recall, F1, not accuracy.
- Resampling distorts predicted probabilities, so **recalibrate** if you need real probabilities.

**Alternatives that often work as well or better:** class weights in the loss, threshold tuning, focal loss, and strong gradient-boosting models.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
pipe = Pipeline([("smote", SMOTE()), ("clf", model)])  # resampling happens per CV fold
```

### Q74. Explain the difference between Normalization and Standardization

> 🔁 _Already answered in [06-deep-learning.md → Q50](ml-engineering/06-deep-learning.md#q50-normalization-vs-standardization); skipped here to avoid duplication._

### Q75. List the most popular distribution curves along with scenarios where you will use them in an algorithm.
> ✍️ _Drafted answer (not from the original notes)._

| Distribution | Type | Describes | Where it shows up in ML |
| --- | --- | --- | --- |
| **Normal (Gaussian)** | Continuous | Sums of many small effects (CLT) | Linear regression residuals, weight initialization, GMMs, Gaussian Naive Bayes, Gaussian processes, VAE latent prior, anomaly detection via z-scores |
| **Bernoulli** | Discrete | One yes/no trial | Binary classification output (logistic regression, sigmoid + BCE), dropout masks |
| **Binomial** | Discrete | Number of successes in $n$ trials | Conversion counts, A/B test analysis |
| **Categorical / Multinomial** | Discrete | One of $K$ outcomes / counts over $K$ | Softmax classifiers, LLM next-token sampling, Multinomial Naive Bayes for text |
| **Poisson** | Discrete | Count of events per interval | Modeling calls/orders per hour, Poisson regression, demand forecasting |
| **Exponential** | Continuous | Time between Poisson events | Survival/time-to-event, queueing and latency modeling |
| **Uniform** | Both | Equal likelihood | Random initialization, random search for hyperparameters, simulation |
| **Beta** | Continuous on [0,1] | Distribution of a probability | Bayesian A/B testing, Thompson sampling for bandits |
| **Dirichlet** | Multivariate on simplex | Distribution over probability vectors | Topic models (LDA), Bayesian priors for categorical data |
| **Gamma** | Continuous, positive | Waiting times, positive skewed values | Insurance claim sizes, Bayesian priors for precision |
| **Student's t** | Continuous, heavy tails | Small-sample means | t-tests, t-SNE's low-dimensional similarities, robust regression |
| **Log-normal / power law (Zipf)** | Continuous / discrete | Skewed positive values / long tails | Income, prices, latency; word frequencies, item popularity in recommenders |

**Why it matters:** the assumed distribution determines the right loss (MSE ↔ Gaussian, cross-entropy ↔ Bernoulli/categorical, Poisson deviance ↔ counts), the right transformation (log for log-normal data) and the right statistical test.

### Q76. List all types of popular recommendation systems?
> ✍️ _Drafted answer (not from the original notes)._

**Main families of recommendation systems:**

1. **Popularity / rule-based:** trending or best-selling items. Simple, strong baseline, solves cold start, but not personalized.
2. **Content-based filtering:** recommend items similar to what the user liked, using item features (genre, text, embeddings). Works for new items; tends to over-specialize (filter bubble).
3. **Collaborative filtering (CF):** uses user-item interactions only.
   - **Memory-based:** user-user or item-item similarity (Amazon's "people who bought this also bought").
   - **Model-based:** matrix factorization (SVD, ALS), factorization machines.
   - Suffers from cold start and sparsity.
4. **Hybrid systems:** combine content and CF (weighted, switching, or feature-level), e.g. Netflix.
5. **Deep learning recommenders:** two-tower models (user/item embeddings for retrieval), Wide & Deep, DLRM, neural CF, autoencoders.
6. **Sequential / session-based:** model the order of interactions with RNNs or Transformers (GRU4Rec, SASRec, BERT4Rec); good for anonymous sessions and next-item prediction.
7. **Knowledge-based / constraint-based:** use explicit requirements and domain rules (real estate, cars, financial products) where interactions are rare.
8. **Graph-based:** GNNs over user-item graphs (PinSage, LightGCN).
9. **Context-aware and RL/bandit-based:** use time, device, location; contextual bandits balance exploration vs. exploitation.
10. **LLM-based / conversational recommenders:** generate or rerank recommendations from natural-language preferences.

**Production architecture:** multi-stage: **candidate generation** (two-tower ANN retrieval, CF, popularity) → **ranking** (gradient boosting or deep model on rich features) → **re-ranking** for diversity, freshness, business rules. Evaluate offline with precision@k, recall@k, NDCG, MAP and online with A/B tests on CTR, conversion, retention.

### Q77. Which metrics can be used to measure correlation of categorical data?
> ✍️ _Drafted answer (not from the original notes)._

**Categorical vs. categorical (nominal):**

- **Chi-square test of independence:** tests whether an association exists (gives a p-value, not strength).
- **Cramér's V:** effect size from chi-square, range 0–1, works for any table size:
  $V = \sqrt{\frac{\chi^2/n}{\min(r-1, c-1)}}$ (use the bias-corrected version for small samples).
- **Phi coefficient ($\phi$):** for 2×2 tables; equals Pearson on two binary variables.
- **Contingency coefficient** and **Tschuprow's T:** older alternatives to Cramér's V.
- **Theil's U (uncertainty coefficient):** asymmetric, based on entropy: how much knowing X reduces uncertainty about Y. Useful because association can be directional.
- **Mutual information:** general, captures any dependence; normalized MI for comparability.
- **Fisher's exact test:** instead of chi-square when expected cell counts are small (< 5).

**Ordinal vs. ordinal:** Spearman's rho, Kendall's tau-b, Goodman-Kruskal gamma.

**Categorical vs. continuous:** point-biserial (binary), correlation ratio $\eta$ / ANOVA.

```python
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
def cramers_v(x, y):
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t)[0]
    n = t.values.sum()
    return np.sqrt(chi2 / n / (min(t.shape) - 1))
```

**Caveat:** with large samples chi-square is almost always significant, so report an effect size like Cramér's V. Label-encoding nominal categories and computing Pearson is meaningless.

### Q78. Which type of sampling is better for a classification model, and why?
> ✍️ _Drafted answer (not from the original notes)._

There's no single best sampling method; it depends on the goal. For **building train/validation/test splits** for classification, **stratified sampling** is usually the better default.

**Stratified sampling**

- Preserves the class proportions in every split and CV fold.
- Essential with imbalanced data: a random split of a dataset with 1% positives can leave a validation fold with almost no positives, making metrics noisy or undefined.
- Gives lower-variance, more reliable estimates of performance.

**Simple random sampling**

- Fine when the dataset is large and classes are balanced; unbiased but can produce unlucky splits on small data.

**Exceptions and other cases**

- **Time-dependent data:** use **time-based splits** (train on the past, validate on the future), not random or stratified shuffling, to avoid leakage.
- **Grouped data** (several rows per user/patient): use **group-aware splits** (GroupKFold, StratifiedGroupKFold) so the same entity isn't in train and test.
- **Changing the training distribution** for imbalance (under/oversampling, SMOTE) is a separate step, applied only to the training folds; evaluation should reflect the real-world class ratio.
- **Data collection/labeling:** stratify by important segments; active learning can choose the most informative samples.

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Q79. What is the common idea behind GPUs and TPUs?
There is processor and there is memory. The amount of time it take to move a floating point number from memory to processor, with in that time we can perform hundreds of floating point operations in processor. So, we can perform batch processing.

### Q80. RNN vs LSTM vs GRU vs transformers?
RNN: Processes sequences step by step, passing a hidden state forward. Simple, but suffers from vanishing/exploding gradients, so it forgets long-range context.

LSTM: An RNN with a separate cell state and three gates (input, forget, output) that control what to keep or discard. Handles long dependencies much better, but is heavier (more parameters) and still sequential.

GRU: A slimmed-down LSTM with two gates (update, reset) and no separate cell state. Fewer parameters, trains faster, and often performs about as well as LSTM on smaller datasets.

Transformer: Drops recurrence entirely. Self-attention lets every token look at every other token directly, and positional encodings supply order. Fully parallelizable, captures long-range dependencies easily, and scales extremely well, but attention cost grows quadratically with sequence length.

### Q81. Federated learning?

> 🔁 _See also [07-ai-security-and-ml-topics.md → Q2](../backend-engineering/core/07-ai-security-and-ml-topics.md#q2-federated-learning) (separate notes on the same question)._
Federated learning is a way to train a machine learning model across many devices or organizations without ever collecting their raw data in one place. Instead of sending data to a central server, the model goes to the data.

**How it works**

1. A central server sends the current global model to a set of participants (phones, hospitals, banks, etc.).
2. Each participant trains the model locally on its own data for a few steps.
3. Each sends back only the model update (weights or gradients), not the data.
4. The server aggregates the updates, most commonly by averaging them (the FedAvg algorithm), into a new global model.
5. The process repeats for many rounds.

**Why people use it**

- **Privacy:** sensitive data (medical records, keyboard typing, financial transactions) stays where it was created.
- **Regulation:** helps with rules like GDPR and HIPAA that restrict moving data.
- **Bandwidth:** sending model updates can be cheaper than uploading huge raw datasets.
- **Access to more data:** organizations that can’t share data can still collaborate on a shared model.

**Main flavors**

- **Cross-device:** millions of unreliable, small devices (e.g., Google’s Gboard next-word prediction).
- **Cross-silo:** a handful of reliable organizations (e.g., hospitals jointly training a diagnostic model).
- **Horizontal vs. vertical:** horizontal means participants have the same features but different users; vertical means they have different features about the same users.

**Challenges**

- **Non-IID data:** each client’s data looks different, which can slow or destabilize training.
- **Communication cost:** many rounds of sending models back and forth is slow, especially on mobile networks.
- **Device heterogeneity:** varying hardware, availability, and dropouts.
- **Privacy is not automatic:** model updates can leak information (via gradient inversion or membership inference attacks), so it’s often combined with differential privacy, secure aggregation, or homomorphic encryption.
- **Security:** malicious clients can poison the model with bad updates.

**Popular tools:** TensorFlow Federated, Flower, PySyft, FedML, NVIDIA FLARE.

### Q82. Adversarial machine learning?
The study of vulnerabilities in artificial intelligence and machine learning models

### Q83. AI ontology?
> ✍️ _Drafted answer (not from the original notes)._

In AI, an **ontology** is a formal, explicit model of a domain: the **concepts** (classes), their **properties**, the **relationships** between them, and **rules/constraints**, expressed in a machine-readable way so software can reason over it.

**Example (finance):** `Company` —*issues*→ `Invoice`; `Invoice` *hasAmount* `Money`; `Subsidiary` *subClassOf* `Company`; rule: every `Invoice` must have exactly one `Issuer`.

**Building blocks and standards:** classes and hierarchies (is-a), object/data properties, axioms; languages like **RDF, RDFS, OWL**; queried with **SPARQL**; validated with SHACL. Well-known ontologies: schema.org, SNOMED CT (medicine), FIBO (finance), Wikidata.

**Ontology vs. related terms:**

- **Taxonomy:** just a hierarchy; an ontology adds arbitrary relations and logic.
- **Knowledge graph:** the actual data (instances/facts) organized according to an ontology (the schema).

**Why it matters in modern AI/LLM systems:**

- **GraphRAG / knowledge graphs:** an ontology defines which entity and relation types the LLM extracts, enabling multi-hop, relationship-aware retrieval.
- **Grounding and consistency:** constrains LLM outputs to valid entities and relations; reduces hallucinated or inconsistent facts.
- **Data integration:** a shared vocabulary aligns data from different systems (the semantic layer of a data platform).
- **Explainability and compliance:** rules are explicit and auditable, important in regulated domains.
- **Agents and tools:** an ontology (e.g. Palantir-style) gives agents typed objects and actions to operate on rather than raw tables.

**Tradeoffs:** expensive to design and maintain, requires domain experts, can be rigid. A pragmatic approach is a lightweight ontology for core entities, extended with LLM-assisted extraction and human review.

### Q84. Consciousness?
> ✍️ _Drafted answer (not from the original notes)._

In an AI interview this usually means: **can AI systems be conscious, and how should engineers think about the claim?**

**What the term means:** consciousness is subjective experience, "what it is like" to be something (phenomenal consciousness), as distinct from intelligence or behaviour. Chalmers' "hard problem" is explaining why physical processing produces experience at all. Related but different: **self-awareness**, **sentience** (capacity to feel), and **access consciousness** (information available for reasoning and report).

**Main theories, and what they say about AI:**

- **Global Workspace Theory:** consciousness = information broadcast to many specialized modules; some argue Transformer-plus-agent architectures could partially satisfy this.
- **Integrated Information Theory (IIT):** consciousness = integrated causal structure ($\Phi$); suggests standard digital hardware has very low integration regardless of behaviour.
- **Higher-order theories:** requires representations of one's own mental states.
- **Biological views:** consciousness may depend on living, embodied substrates.

**Current consensus:** there is no scientific test for consciousness and no agreement that today's LLMs are conscious. LLMs can **talk** convincingly about feelings because they were trained on human text, so self-reports are weak evidence. Researchers (e.g. Butlin, Long et al., 2023) propose checklists of "indicator properties" from these theories; current systems satisfy few.

**Why engineers should care:**

- Avoid anthropomorphizing in product design and user communications (users over-trusting or forming attachments).
- Don't confuse fluent behaviour with understanding when evaluating reliability.
- Model welfare is an emerging research and policy topic; some labs study it precautionarily.

A good interview answer is calibrated: explain the distinctions, note the uncertainty, and tie it back to practical concerns like trust, safety and honest UX.

### Q85. Unsupervised learning through compression?
> ✍️ _Drafted answer (not from the original notes)._

The idea: **learning = compression**. A model that can compress data well must have captured its underlying structure, so you can learn useful representations without labels by training a model to compress (or predict) the data.

**Theory:**

- **Minimum Description Length (MDL) / Kolmogorov complexity:** the best explanation of data is the one that gives the shortest total description (model + data given the model). Regularities are exactly what allow compression.
- **Prediction ⇔ compression:** with arithmetic coding, a model that assigns probability $p(x)$ can encode $x$ in about $-\log_2 p(x)$ bits. So minimizing cross-entropy (next-token prediction) **is** minimizing compressed size. An LLM's training loss is literally its compression rate in bits per token.
- **Ilya Sutskever's argument:** a good compressor of a dataset X will find the shared structure that also helps with a related task Y, which explains why unsupervised pretraining transfers to downstream tasks.

**Practical forms:**

- **Autoencoders / VAEs:** squeeze input through a bottleneck and reconstruct; the bottleneck code is the learned representation.
- **PCA:** optimal linear compression.
- **Language-model pretraining** (GPT): predict the next token, i.e. compress text; representations emerge for free.
- **Masked modeling** (BERT, MAE) and **JEPA**-style latent prediction.
- Clustering / vector quantization (VQ-VAE codebooks).

**Evidence:** DeepMind's "Language Modeling Is Compression" (2023) showed LLMs compress images and audio better than PNG/FLAC when used as compressors with arithmetic coding.

**Caveats:** compressing raw pixels perfectly wastes capacity on irrelevant details (noise, texture), which motivates predicting in latent space (JEPA) instead of reconstructing inputs; and good compression doesn't guarantee representations aligned with the downstream task.

### Q86. Static vs. continuous batching in LLMs?
> ✍️ _Drafted answer (not from the original notes)._

Batching groups requests so the GPU processes them together, which raises throughput because LLM decoding is memory-bandwidth-bound (weights are loaded once per step for the whole batch).

**Static batching**

- Collect a batch of requests, run them together until **every** sequence finishes, then start the next batch.
- Problems: sequences have different output lengths, so short requests finish early and their slots sit idle (padding waste) while the longest request runs; new requests wait for the whole batch, increasing latency. GPU utilization is poor for chat workloads with highly variable lengths.
- Fine for offline jobs with similar lengths (bulk embedding, batch scoring).
- **Dynamic batching** is a variant: wait up to a short timeout to fill a batch, but it still runs batch-to-completion.

**Continuous batching (iteration-level / in-flight batching)**

- Scheduling happens **every decoding step**: as soon as a sequence emits its end token, it is evicted and a waiting request is inserted into the free slot.
- The batch stays full, giving much higher throughput (vLLM reported up to ~10–20x over naive static batching) and lower queueing latency.
- Requires careful **KV-cache memory management**, because sequences of different lengths come and go. **PagedAttention** (vLLM) allocates KV cache in fixed-size blocks to avoid fragmentation. Prefill of new requests is often split into chunks (**chunked prefill**) so it doesn't stall ongoing decodes.
- Implemented in vLLM, TGI, TensorRT-LLM, SGLang.

**Tradeoffs to mention:** larger batches increase throughput but also per-token latency (inter-token latency), so production systems cap batch size / tokens per step to meet p95 latency SLOs; scheduling policies (priority, fairness, preemption with KV swapping or recomputation) matter under load.

### Q87. BERT architecture?
> ✍️ _Drafted answer (not from the original notes)._

**BERT** (Bidirectional Encoder Representations from Transformers, Google 2018) is an **encoder-only Transformer** pre-trained to produce deep bidirectional contextual embeddings.

**Architecture**

- **Input:** WordPiece tokens (30k vocab) with special tokens: `[CLS]` at the start (its final vector summarizes the sequence) and `[SEP]` between/after segments.
- **Embeddings:** token embedding + **learned absolute position embedding** (max 512 positions) + **segment embedding** (sentence A or B), summed and layer-normalized.
- **Stack of Transformer encoder blocks**, each with multi-head **bidirectional** self-attention (no causal mask, every token attends to every token) and a feed-forward network (GELU), with residual connections and Post-LayerNorm.
- **Sizes:** BERT-Base: 12 layers, hidden 768, 12 heads, ~110M parameters. BERT-Large: 24 layers, hidden 1024, 16 heads, ~340M.

**Pre-training objectives:**

- **Masked Language Modeling (MLM):** mask 15% of tokens (80% `[MASK]`, 10% random, 10% unchanged) and predict them using both left and right context.
- **Next Sentence Prediction (NSP):** predict whether sentence B follows A.

**Fine-tuning:** add a small head: `[CLS]` vector → classifier for sentence classification; per-token outputs for NER; start/end span prediction for extractive QA.

**Strengths and limits:** excellent for understanding tasks (classification, NER, retrieval, reranking as a cross-encoder); bidirectionality makes it unsuited to text generation; 512-token limit; quadratic attention. Successors: RoBERTa, ALBERT, DistilBERT, DeBERTa, ModernBERT. In production, BERT-family models remain the workhorses for embeddings, classifiers and rerankers because they're cheap compared with LLMs.

### Q88. What is next-sentence prediction in BERT?
> ✍️ _Drafted answer (not from the original notes)._

**Next-Sentence Prediction (NSP)** is BERT's second pre-training task (alongside masked language modeling). Its goal was to teach the model relationships **between** sentences, useful for tasks like question answering and natural language inference.

**How it works:**

- Build input pairs `[CLS] sentence A [SEP] sentence B [SEP]`, with segment embeddings marking A and B.
- 50% of the time B is the **actual next sentence** from the corpus (label `IsNext`); 50% it's a **random sentence** from another document (`NotNext`).
- The final hidden state of `[CLS]` goes through a binary classifier; the loss is binary cross-entropy, added to the MLM loss.

```text
[CLS] The man went to the store [SEP] He bought a gallon of milk [SEP]   -> IsNext
[CLS] The man went to the store [SEP] Penguins are flightless birds [SEP] -> NotNext
```

**Why it was later dropped:**

- **RoBERTa** (2019) showed removing NSP and training on contiguous full-length text matched or improved downstream performance.
- The task is too easy: random sentences usually come from a different topic, so the model learns **topic detection** rather than coherence.
- It also forced shorter segments, hurting long-range learning.
- **ALBERT** replaced it with **Sentence Order Prediction (SOP)**: the two segments are consecutive, but sometimes swapped, forcing the model to learn discourse coherence.

**Interview takeaway:** NSP is a good example of a pre-training objective that sounded useful but whose benefit came mostly from confounding signals; ablations (RoBERTa) revealed it.

### Q89. Hooks, Coroutine, Plugins, Rules, Skills?
> ✍️ _Drafted answer (not from the original notes)._

These are building blocks for **extending and controlling AI agents / coding assistants** (Claude Code, Cursor, agent frameworks), plus coroutines as the async primitive under them.

- **Hooks:** user-defined code that the agent **harness** runs automatically at lifecycle events, e.g. `PreToolUse` (before a tool call), `PostToolUse`, session start, on stop. Because they're executed deterministically by the harness, not by the model, they enforce policy reliably: block `rm -rf`, run a formatter/linter after every edit, require tests before finishing, log tool calls for audit. Think git hooks or web middleware.
- **Coroutines:** functions that can pause and resume (`async def` / `await` in Python). Agent runtimes use them to run many LLM calls, tool calls and streams concurrently on one thread without blocking: e.g. `asyncio.gather` over parallel tool calls, streaming tokens while executing tools. Key for latency and throughput in I/O-bound agent systems.
- **Plugins:** packaged extensions that add capabilities to the agent or app: new tools, MCP servers, commands, UI, integrations. They are installable and distributable bundles (often containing skills, hooks and commands together).
- **Rules:** persistent instructions loaded into the agent's context (e.g. `CLAUDE.md`, `.cursor/rules`, system prompts): coding conventions, project facts, "always use pnpm". They are **soft guidance**, interpreted by the model, so they can be ignored; anything that must always happen belongs in a hook.
- **Skills:** packaged, on-demand know-how: a folder with instructions (e.g. `SKILL.md`), scripts and resources that the agent loads **only when relevant** (progressive disclosure), keeping the context window small. E.g. a "pdf" skill with extraction scripts.

**Design principle:** rules and skills shape *what the model decides*; hooks enforce *what the harness guarantees*; plugins *distribute* capabilities; coroutines make it all *run concurrently*.

```python
async def run_tools(calls):
    return await asyncio.gather(*(execute(c) for c in calls))  # parallel tool calls
```

### Q90. Reliability (trust model out), Controllability (control the model)?
> ✍️ _Drafted answer (not from the original notes)._

Two complementary properties for deploying LLMs/agents safely:

**Reliability: can you trust the model's output?**

The system produces correct, consistent, grounded results across inputs and over time, and fails predictably when it can't.

- **Threats:** hallucinations, non-determinism, sensitivity to prompt wording, silent degradation after model or data changes, overconfident answers.
- **Techniques:** RAG with grounded citations; structured outputs validated by schemas (Pydantic) with repair loops; self-consistency or verifier models; calibrated confidence with abstention ("I don't know"); golden-dataset evals and regression tests in CI; monitoring of quality, hallucination rate and drift; fallbacks and retries.

**Controllability: can you steer and constrain what the model does?**

Ability to direct behaviour (style, scope, actions) and bound it so it stays within policy.

- **Threats:** instruction drift, prompt injection and jailbreaks, agents taking unintended actions, reward hacking.
- **Techniques:** clear system prompts and few-shot examples; constrained decoding and JSON schemas; fine-tuning (SFT/DPO) for consistent behaviour; guardrails on input and output (classifiers, policy filters); least-privilege tools, sandboxing, max-step limits; **human-in-the-loop** approval for high-stakes actions; deterministic hooks in the harness; steering methods like activation steering or system-level routing.

**Relationship:** controllability is a prerequisite for reliability (you can't trust what you can't constrain), but a controllable model can still be confidently wrong, so reliability needs measurement on top. In regulated domains (fintech, healthcare) both are requirements, captured as an evaluation plan (accuracy, groundedness, policy-violation rate) and an audit trail of every prompt, tool call and decision.

### Q91. SAM vs YOLO?
> ✍️ _Drafted answer (not from the original notes)._

They solve different problems and are often combined rather than competing.

| | **YOLO** (You Only Look Once) | **SAM** (Segment Anything Model, Meta) |
| --- | --- | --- |
| Task | Object **detection** (boxes + class labels); newer versions add instance segmentation | Promptable **segmentation** (pixel masks), class-agnostic |
| Input | Image | Image + prompt (points, boxes, or masks; SAM 2 also handles video) |
| Output | Boxes, classes, confidences | High-quality masks, no class names |
| Training | Supervised on a fixed label set (e.g., COCO); fine-tune for new classes | Trained on SA-1B (1B+ masks); zero-shot on new objects |
| Speed | Real-time, edge-friendly (single-stage CNN) | Heavy ViT image encoder; the mask decoder is light, but encoding is the bottleneck (MobileSAM / EfficientSAM help) |

**When to use which**

- Need to know *what* and *where* fast (counting cars, safety-gear detection on a camera feed): **YOLO**.
- Need precise object outlines with no task-specific labels (annotation tooling, medical or industrial masks, background removal): **SAM**.

**Common production pattern:** YOLO (or Grounding DINO for open-vocabulary text prompts) proposes boxes, and SAM turns each box into a precise mask. This "detect then segment" pipeline is also a fast way to bootstrap segmentation labels: SAM-assisted labeling cuts annotation time a lot, and a human only corrects the masks.

**Trade-off:** YOLO needs labeled data for your classes but runs cheaply. SAM is zero-shot for shapes but gives you no semantics and costs more compute per image.

### Q92. NCM(next concept modeling) vs LLM?
> ✍️ _Drafted answer (not from the original notes)._

**LLM (next-token modeling):** a Transformer trained to predict the next *token* (a subword). Reasoning, planning and abstraction emerge implicitly from that objective, and generation happens one token at a time.

**Next-concept modeling (NCM):** the idea of predicting the next *concept*, meaning a higher-level semantic unit such as a sentence or an idea embedding, instead of the next token. The best-known example is Meta's **Large Concept Model (LCM, 2024)**. It encodes each sentence into a language- and modality-agnostic embedding space (SONAR), trains a model (MSE-regression or diffusion-based) to predict the *next sentence embedding*, then decodes that embedding back to text in any supported language.

| | LLM (token-level) | NCM / LCM (concept-level) |
| --- | --- | --- |
| Unit of prediction | Subword token | Sentence/concept embedding |
| Sequence length | Long (many tokens) | Much shorter (one step per sentence), so attention is cheaper |
| Language dependence | Tied to the tokenizer/vocabulary | Concept space shared across languages and modalities |
| Planning | Implicit | Explicitly hierarchical: plan at the idea level, render the words later |
| Maturity | Production-grade, huge ecosystem | Research stage; quality depends on the encoder/decoder and lags strong LLMs on fluency |

**Why it matters:** humans plan at the level of ideas, not words. Concept-level modeling promises better long-form coherence, cheaper long-context processing, and zero-shot multilinguality.

**Caveats:** embedding-space prediction is lossy (fine detail such as numbers or exact wording can drift), the field has few tools and benchmarks yet, and in production today it is a research direction, not a replacement for LLMs.

### Q93. World Models vs. JEPA?
> ✍️ _Drafted answer (not from the original notes)._

**World model:** an internal model that predicts how an environment evolves, $p(s_{t+1} \mid s_t, a_t)$, so an agent can *imagine* outcomes and plan before acting. Examples include Ha & Schmidhuber's World Models, Dreamer (RL in a learned latent space), MuZero (planning with a learned dynamics model), and video-generation "world simulators" such as Genie and Sora-style models. Many of these are **generative**: they reconstruct or predict pixels or observations.

**JEPA (Joint-Embedding Predictive Architecture, Yann LeCun):** a self-supervised architecture that predicts the **representation** of a missing or future part of the input from the representation of the visible part, *in embedding space*, never in pixel space. Implementations include I-JEPA (images) and V-JEPA / V-JEPA 2 (video, used for robot planning).

```text
context x --enc--> s_x --predictor(z)--> ŝ_y   ≈   s_y <--enc(EMA)-- target y
loss = || ŝ_y - s_y ||   (no pixel reconstruction)
```

**How they relate:** JEPA is a *way to build* a world model. LeCun's argument is that a world model should predict abstract states, not every pixel, because much of the future is unpredictable detail (the exact texture of leaves). Generative world models spend capacity on that detail. JEPA discards it and keeps what is predictable and useful for planning.

| | Generative world model | JEPA |
| --- | --- | --- |
| Predicts | Observations (pixels/tokens) or a reconstructable latent | Latent embeddings only |
| Strength | Can render and simulate; easy to inspect visually | Efficient, semantic features; less wasted capacity |
| Risk | Blurry or hallucinated details; expensive | Representation collapse (handled with EMA target encoders and masking); cannot render outputs directly |

**Interview angle:** both target the same goal, agents that understand and plan in the physical world. JEPA is a non-generative, representation-level approach to building that world model.

### Q94. What is circuit/graph(operation like addition, multiplication, activation function are represented by nodes of the graph are called gates) search( process of systematically exploring nodes and edges of graph to determine the valid path from starting point to end point ) ? How does backpropagation solve circuit search?
backpropagation solves a computational circuit by reversing the flow of data to instantly calculate exactly how to fix the circuit's errors

### Q95. Why is AI infra moving towards Rust?
> ✍️ _Drafted answer (not from the original notes)._

Python remains the *interface* for AI, but more of the *infrastructure underneath* is being written in Rust:

- **Performance without a GC:** C/C++-level speed and predictable latency (no garbage-collection pauses), which matters for tokenizers, data loaders, inference servers and vector databases.
- **Memory safety:** the ownership model removes whole classes of bugs (use-after-free, data races) that plague C++ infra. This matters for long-running, multi-tenant services where a segfault means an outage or a security hole.
- **Fearless concurrency:** the compiler enforces thread safety, so heavy parallel work (tokenizing millions of docs, async request handling) is easier to get right than in C++. It also sidesteps Python's GIL.
- **Great Python interop:** PyO3 and maturin make it easy to ship Rust-backed Python packages, so users keep a Python API while the hot path runs in Rust.
- **Tooling and deployment:** Cargo, single static binaries, small containers, and WebAssembly targets for edge or in-browser inference.

**Real examples:** Hugging Face `tokenizers` and `safetensors`, Hugging Face TGI's router, `candle` (an ML framework), Polars (a DataFrame library), vector DBs (Qdrant, LanceDB), `uv`/`ruff` (Python tooling), and parts of OpenAI's and other vendors' serving stacks.

**Trade-offs:** a steeper learning curve, slower development than Python, a smaller ML ecosystem, and CUDA kernels are still mostly C++/CUDA (or Triton). So the realistic pattern is **Python for research and orchestration, Rust for performance-critical, safety-critical plumbing**, rather than a full rewrite.

### RL
https://spinningup.openai.com/en/latest/

### Q96. Function approximator?
Neural networks

### Q97. Hopfield networks vs. Boltzmann Machine
> ✍️ _Drafted answer (not from the original notes)._

All three are **energy-based** neural networks from the 1980s with symmetric weights. They differ in whether they are deterministic or stochastic, and in what they are used for.

**Hopfield network**

- A fully connected, recurrent network of binary (±1) units with symmetric weights and no self-connections.
- Energy: $E = -\tfrac12 \sum_{i,j} w_{ij} s_i s_j - \sum_i b_i s_i$. Updates are **deterministic** and always lower the energy, so the network settles into a local minimum.
- Weights are set by a Hebbian rule ($w_{ij} \propto \sum_\mu \xi_i^\mu \xi_j^\mu$), so the stored patterns become attractors. Use: **associative / content-addressable memory**. Give it a noisy or partial pattern and it recalls the closest stored one.
- Limits: classical capacity is about $0.14N$ patterns, and it suffers from spurious attractors. *Modern Hopfield networks* have exponential capacity, and their update rule is mathematically equivalent to Transformer attention.

**Boltzmann machine (BM)**

- Same energy function, but units are **stochastic**: $p(s_i=1) = \sigma(\Delta E_i / T)$. It also has **hidden units** in addition to visible ones.
- It is a **generative model**: it learns a probability distribution $p(v) \propto \sum_h e^{-E(v,h)}$ rather than storing specific patterns.
- Learning needs expectations under the model distribution (sampled via MCMC/Gibbs), which is extremely slow for a fully connected graph. As a result it is mostly theoretical.

**Restricted Boltzmann machine (RBM)**

- A BM with a **bipartite** graph: there are visible↔hidden connections but none within a layer.
- That makes hidden units conditionally independent given the visible units (and vice versa), so block Gibbs sampling is cheap. Training uses **Contrastive Divergence (CD-k)** (Hinton).
- Uses: feature learning, collaborative filtering (the Netflix Prize), and stacked as **Deep Belief Networks** for layer-wise pretraining (around 2006), which helped spark the deep learning revival.

| | Hopfield | Boltzmann | RBM |
| --- | --- | --- | --- |
| Units | Deterministic | Stochastic | Stochastic |
| Hidden units | No | Yes | Yes (bipartite) |
| Purpose | Memory / pattern completion | Generative modeling | Practical generative modeling / features |
| Training | Hebbian, one-shot | Very slow MCMC | Contrastive Divergence |

**Today:** they are largely historical. VAEs, GANs and diffusion models replaced them for generative modeling, but the energy-based view and the Hopfield ↔ attention link still come up in interviews.

### Q98. Content-addressable memory vs. parametric memory
> ✍️ _Drafted answer (not from the original notes)._

These are two ways a system can "remember" information.

**Content-addressable memory (CAM)**: you retrieve an item by its *content*, not by an address. A partial or noisy cue returns the best-matching stored item.

- Classic examples: Hopfield networks (pattern completion) and hardware CAM/TCAM in network routers.
- Modern examples: attention (query·key similarity selects values), vector databases, kNN-LM, memory networks, and **RAG**. You look up stored chunks by embedding similarity.
- Properties: knowledge is **explicit and inspectable**, easy to **update or delete** (add or remove a document), and attributable (you can cite the source). Costs: lookup latency, storage, and dependence on retrieval quality.

**Parametric memory**: knowledge is stored *implicitly in the model's weights* by training.

- Example: an LLM "knowing" that Paris is the capital of France.
- Properties: fast at inference (no lookup), generalizes and composes knowledge, and is compressed. But it is **opaque**, **hard to update** (needs fine-tuning or model editing, and risks catastrophic forgetting), frozen at the training cutoff, and prone to **hallucination** because there is no explicit source.

| | Content-addressable (non-parametric) | Parametric |
| --- | --- | --- |
| Where stored | External store / keys | Model weights |
| Update cost | Cheap (insert/delete) | Expensive (retrain/fine-tune) |
| Traceability | High (citations) | Low |
| Generalization | Limited to what is stored | Strong |
| Failure mode | Retrieval misses | Hallucination, staleness |

**Production takeaway:** modern LLM systems are **hybrid**. Parametric memory provides language skill and reasoning, and content-addressable memory (RAG, tool lookups, agent memory stores) provides fresh, private, auditable facts. That is why "prompting vs. RAG vs. fine-tuning" is mostly a question of which memory type should hold a given piece of knowledge.

### Q99. How does BERT work, and what are its pre-training objectives (MLM, NSP)?
> ✍️ _Drafted answer (not from the original notes)._

**Architecture:** BERT (Devlin et al., 2018) is a **Transformer encoder-only** stack: 12 layers, hidden size 768, 12 heads, about 110M parameters for Base; 24 layers, 1024, 16 heads, about 340M for Large. It uses full **bidirectional self-attention**, so every token sees left *and* right context.

**Input representation:** WordPiece tokens (about a 30k vocabulary), plus segment embeddings (sentence A/B), plus learned position embeddings (max 512). Special tokens: `[CLS]` at the start (its final vector is used for classification) and `[SEP]` between and after sentences.

```text
[CLS] my dog is [MASK] [SEP] he likes play ##ing [SEP]
```

**Pre-training objectives**

1. **Masked Language Modeling (MLM):** randomly select 15% of tokens. Of those, 80% become `[MASK]`, 10% become a random token, and 10% stay unchanged (this reduces the pretrain/fine-tune mismatch, since `[MASK]` never appears at fine-tuning time). The model predicts the original tokens with cross-entropy. This is what makes deep bidirectional context possible; a left-to-right LM cannot "see the answer".
2. **Next Sentence Prediction (NSP):** given sentences A and B, predict from `[CLS]` whether B actually follows A (50% real, 50% random). It was meant to teach inter-sentence relationships for QA and NLI.

Training data was BooksCorpus plus English Wikipedia (about 3.3B words).

**Fine-tuning:** add a small head (classification on `[CLS]`, span start/end for QA, per-token tags for NER) and fine-tune all weights briefly.

**Caveats worth mentioning:** RoBERTa later showed **NSP adds little or even hurts**. MLM only learns from about 15% of tokens per pass, which is sample-inefficient (ELECTRA addresses this). BERT also can't generate text naturally. It is still a strong, cheap backbone for classification, NER, reranking (cross-encoders) and embeddings.

### Q100. How does RoBERTa differ from BERT?
> ✍️ _Drafted answer (not from the original notes)._

**RoBERTa** ("Robustly optimized BERT approach", Facebook AI, 2019) keeps BERT's **architecture unchanged** and shows BERT was significantly *undertrained*. The gains come purely from the training recipe:

- **Removed NSP:** trains with MLM only, on full contiguous sentences packed up to 512 tokens (FULL-SENTENCES). Dropping NSP matched or improved downstream performance.
- **Dynamic masking:** a new mask pattern is generated every time a sequence is fed, instead of BERT's static masks fixed at preprocessing time.
- **Much more data:** about 160GB of text (adds CC-News, OpenWebText, Stories) versus BERT's about 16GB.
- **Longer training with bigger batches:** batch size of 8k sequences, more steps, and a tuned learning rate and Adam $\beta_2$. Large batches improved both perplexity and end-task accuracy.
- **Tokenizer:** byte-level BPE with about a 50k vocabulary (GPT-2 style) instead of a 30k WordPiece. It has no unknown tokens, but a slightly larger embedding matrix.

**Result:** state-of-the-art on GLUE, SQuAD and RACE at the time, with zero architecture changes.

**Interview takeaway:** data scale, training duration and objective choices can matter as much as architecture, and ablations should be done carefully (the NSP finding). In practice RoBERTa, and DeBERTa after it, became the default encoder backbones for classification, NER and cross-encoder rerankers.

### Q101. Why do Transformers use layer normalization instead of batch normalization, and what is Pre-LN vs Post-LN?
> ✍️ _Drafted answer (not from the original notes)._

**Why LayerNorm instead of BatchNorm**

- **BatchNorm** normalizes each feature across the *batch* (and spatial dims). In NLP, sequences have variable length and padding, so batch statistics get mixed with padding and are noisy. Statistics also depend on batch composition and size, which hurts at small per-device batches and with gradient accumulation.
- At **inference**, BN relies on running averages from training. With autoregressive decoding (one token at a time, batch size 1, changing distributions) those statistics mismatch badly.
- **LayerNorm** normalizes across the *feature dimension of each token independently*: $\text{LN}(x) = \gamma \frac{x-\mu}{\sigma} + \beta$, with $\mu, \sigma$ computed over the hidden size. That makes it batch-size independent, identical at train and test time, and naturally handles variable lengths. Many modern LLMs use **RMSNorm**, which drops mean-centering and is cheaper with similar quality (LLaMA, Mistral).

**Post-LN vs Pre-LN**

```text
Post-LN (original Transformer):  x = LN(x + Sublayer(x))
Pre-LN  (GPT-2 onwards):         x = x + Sublayer(LN(x))
```

- **Post-LN:** the normalization sits on the residual path, so gradients near the output are large and those near the input shrink with depth. It needs **careful learning-rate warmup**, is unstable for deep stacks, but when it trains successfully it can reach slightly better final quality.
- **Pre-LN:** the residual stream is an identity path, so gradients flow cleanly to every layer. It is much more stable, trains deep models with little or no warmup, and is the default in almost all LLMs. One downside: the residual stream's magnitude grows with depth and later layers can contribute less, so a final LayerNorm is added before the LM head. Variants such as DeepNorm and sandwich-LN try to get the best of both.

**One-liner:** LN because NLP batches are variable and inference is per-token; Pre-LN because stability at depth matters more than the small quality edge of Post-LN.

### Q102. What is the difference between SFT and RL-based tuning (RLHF/DPO)?
> ✍️ _Drafted answer (not from the original notes)._

**SFT (Supervised Fine-Tuning):** train on curated (prompt, ideal response) pairs with standard next-token cross-entropy on the response tokens. It is **imitation learning**: "write like this demonstration."

- Pros: simple, stable, cheap, and good for teaching format, style, domain and instruction-following.
- Cons: capped by demo quality, it teaches only *what to do*, never *what not to do*, can encourage confident hallucination (the model imitates confident answers even where it lacks the knowledge), and needs expensive expert-written demonstrations.

**RL-based / preference tuning:** optimize the model using *relative* feedback about which outputs are better.

- **RLHF (PPO):** collect human comparisons, train a **reward model**, then optimize the policy with PPO to maximize reward minus a **KL penalty** to the SFT model, which prevents drift and reward hacking. It is powerful, but uses four models in memory (policy, reference, reward, value), is unstable, and is sensitive to hyperparameters.
- **DPO:** a closed-form reformulation that trains directly on (chosen, rejected) pairs with a classification-style loss. There is no reward model and no sampling loop, so it is far simpler and more stable. It is offline, though, so it is limited to the data distribution it was given.
- **RLVR / GRPO:** RL with programmatic rewards (unit tests, math answer checks), the main driver of recent reasoning models.

| | SFT | RLHF / DPO |
| --- | --- | --- |
| Signal | Absolute demos | Relative preferences / rewards |
| Learns | Imitation | Ranking of good vs bad; can exceed demo quality |
| Stability / cost | High / low | Lower (PPO) to medium (DPO) / higher |
| Typical use | First stage | After SFT, for helpfulness, harmlessness, reasoning |

**Practice:** the standard pipeline is pretrain → SFT → preference tuning. SFT gets the model into the right distribution, and preference/RL tuning refines it. Comparisons are also easier for humans to give than perfect demonstrations.

### Q103. What is InfoNCE loss, and where is it used?
> ✍️ _Drafted answer (not from the original notes)._

**InfoNCE** (from Contrastive Predictive Coding, van den Oord et al., 2018) is the standard **contrastive loss**. Given an anchor $q$, one positive $k^+$ and $N-1$ negatives, it is a softmax cross-entropy that asks the model to pick out the positive:

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q, k_j)/\tau)}$$

- $\text{sim}$ is usually cosine similarity (dot product of L2-normalized embeddings).
- $\tau$ is the **temperature**. A low $\tau$ sharpens the distribution and focuses on hard negatives; a high $\tau$ makes it smoother. It is often learned (CLIP).
- Minimizing it maximizes a lower bound on the mutual information between the anchor and positive views: $I \ge \log N - \mathcal{L}$, so **more negatives give a tighter bound**.

```python
import torch, torch.nn.functional as F
def info_nce(q, k, tau=0.07):           # q, k: [B, D], row i of k is the positive for row i of q
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    logits = q @ k.T / tau               # [B, B]; off-diagonal entries are in-batch negatives
    return F.cross_entropy(logits, torch.arange(q.size(0), device=q.device))
```

**Where it's used**

- **Self-supervised vision:** SimCLR (two augmentations of one image), MoCo (momentum queue of negatives).
- **Multimodal:** CLIP/ALIGN (a symmetric InfoNCE over image-text pairs).
- **Retrieval and embeddings:** dense retrievers (DPR), sentence embeddings (SimCSE, E5, BGE), with query/positive pairs plus in-batch and hard negatives.
- **Audio/speech:** wav2vec 2.0, CPC. **RL:** contrastive goal-conditioned objectives.

**Practical issues:** quality depends on **batch size / number of negatives** (hence large batches, memory queues, cross-device gathering), **hard-negative mining** (but watch for false negatives, such as duplicates that are actually positives), and tuning $\tau$.

### Q104. How does BPE tokenization work?
> ✍️ _Drafted answer (not from the original notes)._

**Byte-Pair Encoding** builds a subword vocabulary by repeatedly merging the most frequent adjacent symbol pair.

**Training**

1. Pre-tokenize the corpus (split on whitespace and punctuation, e.g. GPT-2's regex) and count word frequencies.
2. Start with a base vocabulary of characters, or the 256 **bytes** in byte-level BPE, so any string is encodable and there is never an `[UNK]`.
3. Count all adjacent symbol pairs (weighted by word frequency). Merge the most frequent pair into a new symbol and record the merge rule.
4. Repeat until the target vocabulary size is reached (e.g., 32k–200k).

```text
corpus: low×5 lower×2 newest×6 widest×3
merge 1: (e,s) → es     merge 2: (es,t) → est    merge 3: (l,o) → lo ...
```

**Encoding:** split the new text into base symbols, then apply the learned merges **in the order they were learned** (by merge rank) until none apply. Decoding just concatenates the tokens (bytes), then UTF-8 decodes.

**Why it works:** frequent words become single tokens, while rare words break into meaningful pieces (`unhappiness → un + happi + ness`). That balances vocabulary size against sequence length and handles open vocabularies.

**Trade-offs and production notes**

- A larger vocabulary means shorter sequences (cheaper attention, more context) but a bigger embedding/LM-head matrix and rare, undertrained tokens.
- Tokenization is greedy and frequency-driven, not linguistic. It causes quirks such as poor arithmetic on numbers split oddly, spelling and character-counting failures, and non-English text costing more tokens (a fairness and cost issue).
- Related schemes: **WordPiece** (BERT; merges by likelihood gain), **Unigram LM** (SentencePiece; prunes a large vocabulary probabilistically). Libraries: `tiktoken`, Hugging Face `tokenizers` (Rust), SentencePiece.

### Q105. How do you implement a tokenizer (training, encoding, decoding, special tokens)?
> ✍️ _Drafted answer (not from the original notes)._

A tokenizer has four parts: a **normalizer**, a **pre-tokenizer**, a **model** (BPE/Unigram/WordPiece) and a **post-processor** for special tokens. A minimal byte-level BPE:

```python
import re
from collections import Counter

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+""")

class BPETokenizer:
    def __init__(self, special=("<|bos|>", "<|eos|>", "<|pad|>")):
        self.merges, self.special = {}, list(special)

    def train(self, text, vocab_size):
        words = Counter(tuple(w.encode()) for w in PAT.findall(text))   # bytes 0..255
        self.vocab = {i: bytes([i]) for i in range(256)}
        nxt = 256
        while nxt < vocab_size - len(self.special):
            pairs = Counter()
            for w, f in words.items():
                for p in zip(w, w[1:]): pairs[p] += f
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = nxt; self.vocab[nxt] = self.vocab[best[0]] + self.vocab[best[1]]
            words = Counter({self._merge(w, best, nxt): f for w, f in words.items()})
            nxt += 1
        self.special_ids = {s: nxt + i for i, s in enumerate(self.special)}

    @staticmethod
    def _merge(w, pair, new):
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair: out.append(new); i += 2
            else: out.append(w[i]); i += 1
        return tuple(out)

    def encode(self, text):
        ids = []
        for chunk in re.split("(" + "|".join(map(re.escape, self.special)) + ")", text):
            if chunk in self.special_ids: ids.append(self.special_ids[chunk]); continue
            for w in PAT.findall(chunk):
                t = tuple(w.encode())
                while len(t) > 1:          # apply the lowest-rank (earliest-learned) merge first
                    pair = min(zip(t, t[1:]), key=lambda p: self.merges.get(p, 1e18))
                    if pair not in self.merges: break
                    t = self._merge(t, pair, self.merges[pair])
                ids.extend(t)
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in self.special_ids.items()}
        return b"".join(inv[i].encode() if i in inv else self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

**Key design points to discuss**

- **Special tokens** (BOS/EOS/PAD, chat-role markers, tool-call tags) are matched *before* BPE and must never be produced from user text by accident, or you open a prompt-injection hole. Make "allowed special tokens" an explicit flag.
- **Round-trip guarantee:** `decode(encode(x)) == x`. Byte-level BPE guarantees it; test it with fuzzing on Unicode and emoji.
- **Performance:** the naive training loop is $O(\text{merges} \times \text{corpus})$. Real implementations update pair counts incrementally with a heap, and are written in Rust/C++ (HF `tokenizers`, `tiktoken`).
- **Streaming decode:** a multi-byte UTF-8 character can span several tokens, so buffer bytes before emitting text.
- **Versioning:** the tokenizer is part of the model contract. Changing it means retraining the embeddings.

### Q106. What does the LM head do, and what is weight tying?
> ✍️ _Drafted answer (not from the original notes)._

**LM head:** the final layer of a language model. It maps each position's final hidden state $h \in \mathbb{R}^{d}$ to **logits over the vocabulary**:

$$\text{logits} = h W_{\text{out}}^\top (+ b), \quad W_{\text{out}} \in \mathbb{R}^{|V| \times d}, \qquad p(\text{next token}) = \text{softmax}(\text{logits}/T)$$

During training these logits go into cross-entropy against the next token. At inference they feed sampling (greedy, temperature, top-k/top-p). It is usually preceded by a final LayerNorm/RMSNorm.

**Weight tying:** reuse the **input embedding matrix** $E \in \mathbb{R}^{|V| \times d}$ as the output projection, $W_{\text{out}} = E$ (Press & Wolf 2017; Inan et al. 2017).

```python
class TinyLM(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.embed.weight   # tied: same Parameter object
```

**Why tie**

- **Fewer parameters:** $|V| \times d$ can be huge (128k × 4096 ≈ 0.5B params), which matters a lot for small models.
- **Regularization and better perplexity:** the input and output token representations live in one space, and rare tokens get gradient signal from both ends.
- Intuition: a token's output score is the dot product between the context vector and that token's embedding, a natural similarity.

**Why many large LLMs *untie* (e.g., LLaMA 2/3 at larger sizes)**

- At scale the parameter saving is relatively small, and separate matrices give extra capacity, because input semantics and output prediction need somewhat different geometry.
- It interacts with optimization (embedding vs. output gradient scales) and with sharding in tensor parallelism.

**Practical gotchas:** when you resize the vocabulary (adding special tokens) you must resize both matrices and keep them tied. LoRA or quantization configs must handle the shared parameter correctly. The LM head is also often the single biggest matmul at decode time for small models.

### Q107. How would you implement a regex matcher supporting `.` and your PyTorch training loss is NaN or not decreasing. How do you debug it?
> ✍️ _Drafted answer (not from the original notes)._

This title merges two separate questions, so both are answered: (a) a regex matcher supporting `.` and `*`, and (b) debugging a PyTorch loss that is NaN or not decreasing.

**(a) Regex matcher with `.` (any char) and `*` (zero or more of the preceding element), full match**

Use DP: `dp[i][j]` means `s[i:]` matches `p[j:]`. Time and space are $O(|s|\cdot|p|)$.

```python
from functools import lru_cache

def is_match(s: str, p: str) -> bool:
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], '.')
        if j + 1 < len(p) and p[j + 1] == '*':
            # skip "x*" entirely, or consume one char and stay on "x*"
            return dp(i, j + 2) or (first and dp(i + 1, j))
        return first and dp(i + 1, j + 1)
    return dp(0, 0)

assert is_match("aab", "c*a*b") and is_match("ab", ".*") and not is_match("mississippi", "mis*is*p*.")
```

Discuss: the recursive version without memoization is exponential, a bottom-up table avoids recursion limits, and real regex engines use NFA/Thompson construction for linear-time guarantees, which avoids catastrophic backtracking (ReDoS).

**(b) Loss is NaN or not decreasing: a debugging checklist**

1. **Overfit one small batch first.** If the loss can't go to about 0 on 1–10 examples, there is a bug (wrong labels, detached graph, wrong loss), not a tuning problem.
2. **Data:** check inputs and labels for NaN/Inf (`torch.isfinite(x).all()`), label ranges (class ids < num_classes), normalization, and that shuffling did not misalign x and y.
3. **Loss setup:** passing softmax outputs to `CrossEntropyLoss` (it expects logits); `log(0)` in a custom loss (add eps or use `log_softmax`/`BCEWithLogitsLoss`); division by zero in normalization.
4. **Training loop:** forgetting `optimizer.zero_grad()`, calling `model.eval()` by mistake, frozen params (`requires_grad=False`), params missing from the optimizer, or `.detach()`/`.item()` breaking the graph.
5. **Optimization:** a learning rate that is too high (NaN/divergence) or too low (flat loss). Run an LR range test, add warmup, use gradient clipping (`clip_grad_norm_(…, 1.0)`), and log the gradient norm per layer to spot explosion or vanishing.
6. **Numerics:** fp16 overflow, so use bf16 or `GradScaler`. Turn on `torch.autograd.set_detect_anomaly(True)` to find the op that first produces NaN, and check BatchNorm with batch size 1.
7. **Initialization and architecture:** bad init, missing normalization or residuals in deep nets, dead ReLUs.

Log loss, gradient norm, LR and activation stats (W&B/TensorBoard) so the failure step is visible, and fix seeds to reproduce it.

### Q108. Design a conversational assistant (memory, context management, safety, latency).
> ✍️ _Drafted answer (not from the original notes)._

**Clarify:** domain (customer support vs. general chat), scale (QPS, users), latency SLO (e.g., first token < 1s), modalities, compliance (PII, regulated data).

**Architecture**

```text
Client ⇄ (WebSocket/SSE streaming) ⇄ API Gateway (auth, rate limit)
   → Orchestrator ── Input guardrails (PII redaction, injection/jailbreak, moderation)
        ├─ Context builder: system prompt + user profile + memory + retrieved docs + recent turns
        ├─ Router: small model for simple intents, frontier model for hard ones
        ├─ Tools (search, CRM, booking) via function calling / MCP
        └─ LLM inference (vLLM/hosted API) → Output guardrails → stream to user
   → Async: memory writer, summarizer, eval/logging pipeline
```

**Memory**

- **Short-term:** recent turns kept verbatim inside a token budget.
- **Rolling summary:** older turns compressed by a cheap model as the conversation grows.
- **Long-term:** extracted facts and preferences stored per user (a key-value store plus a vector store), retrieved by relevance, with user-visible controls to view and delete (GDPR) and TTLs.
- **Episodic:** past sessions stored as summaries, so the assistant can say "last time you asked about…".

**Context management:** a fixed token budget per section (system prompt, memory, retrieval, history), reranking to keep only the top-N chunks, the most relevant content placed at the start or end (lost-in-the-middle), and prompt caching for the static prefix.

**Safety:** layered input and output classifiers, treat retrieved and tool content as untrusted data (injection), least-privilege tools with confirmation for side-effecting actions, grounded answers with citations, refusal and escalation-to-human paths, and an audit log.

**Latency:** streaming tokens, a parallel pipeline (retrieval, memory lookup and moderation run concurrently), small-model routing, semantic and prompt caching, speculative decoding / continuous batching on self-hosted models, and asynchronous memory writes off the critical path. Measure TTFT and p95/p99 latency.

**Evaluation and operations:** golden conversation sets and LLM-as-judge (calibrated to human ratings), regression tests in CI for prompt changes, online metrics (CSAT, resolution rate, escalation rate, thumbs up/down), tracing of every turn (Langfuse/OTel), and cost per conversation dashboards.

### Q109. Design a coding agent.
> ✍️ _Drafted answer (not from the original notes)._

**Goal:** an agent that takes a task ("fix this failing test", "add an endpoint") and produces a verified code change in a real repository.

**Core loop (ReAct-style):** plan → act with tools → observe the results → repeat until tests pass or budget is exhausted → present a diff or PR.

**Tools**

- Repo navigation: `ls`, `grep`/ripgrep, code search (symbol index via tree-sitter / LSP, plus embeddings for semantic search).
- File read and **edit** (search-and-replace or patch edits rather than whole-file rewrites, which saves tokens and reduces clobbering).
- **Shell execution in a sandbox:** run tests, linters, type-checkers, build.
- Optional: git operations, web or documentation lookup, issue tracker access.

**Context management:** repos don't fit in context. Retrieve just enough: a repo map (file tree plus key signatures), files relevant to the task, and test output. Summarize long tool outputs, keep a scratchpad or plan file, and compact the history on long tasks.

**Verification is the key differentiator:** the agent writes or reproduces a failing test first, then makes the change, then runs the tests, lint and type checks. Execution feedback is the reward signal. It retries with the error output as context, and caps iterations and tokens.

**Safety and sandboxing:** run in an isolated container or VM with no production credentials, network egress restricted, resource limits and timeouts. Any destructive action (force push, delete, deploy) requires human approval. Treat repository content and web pages as untrusted (prompt injection via README or comments). Human review of the final PR.

**Architecture options:** a single agent with good tools is often best. Sub-agents (explorer, implementer, reviewer) help for large tasks and keep each one's context clean. Use a strong model for planning and editing, and cheaper models for summarizing and search.

**Evaluation:** SWE-bench-style task suites (resolved rate), internal golden tasks, metrics for test pass rate, diff size, number of iterations, cost per task, human acceptance rate of PRs, and regressions caught in CI. Trace every step to debug failure modes such as wrong file edited, test hallucination, or loops.

### Q110. How does prompt caching work, and when does it help?
> ✍️ _Drafted answer (not from the original notes)._

**What it is:** during prefill an LLM computes **KV-cache tensors** for every prompt token. Prompt caching stores the KV cache for a prompt **prefix**, so a later request with the *exact same prefix* skips recomputing it and only processes the new suffix.

**How it works**

- Matching is **prefix-based and exact** (token-identical, usually in blocks of tokens). Any change early in the prompt invalidates everything after it.
- Hosted APIs (Anthropic, OpenAI, Gemini) cache automatically above a minimum length, or via explicit cache breakpoints. Cached input tokens are billed at a steep discount (about 10–50% of the normal price) with a TTL of minutes, extendable on some providers. Anthropic charges a small premium to write to the cache.
- Self-hosted: vLLM *automatic prefix caching* and SGLang *RadixAttention* keep KV blocks in GPU memory and share them across requests using a hash or radix tree of prefixes.

**When it helps**

- Long, **static** system prompts, tool definitions, few-shot examples.
- RAG or document Q&A where many questions hit the same large document.
- Multi-turn chat and **agents**, where each turn resends the growing history and only the tail is new.
- Effects: **lower TTFT** (often 50–80% for long prefixes) and **lower cost**.

**When it doesn't:** short prompts, prompts where variable content (timestamps, user names, retrieved chunks) comes *first*, low request rates where entries expire before reuse, and highly diverse prompts.

**Design tips:** order the prompt **static → semi-static → dynamic** (system prompt, tools, documents, then history, then the user query). Keep tool definitions and serialization deterministic (stable JSON key order). Track the cache hit rate as a metric.

**Don't confuse with:** *semantic caching*, which returns a stored **response** for a semantically similar query. That skips the model entirely but risks stale or wrong answers. Prompt caching still runs the model, so the output is unaffected.

### Q111. Design a distributed rate limiter.
> ✍️ _Drafted answer (not from the original notes)._

**Requirements:** limit requests per key (user, API key, IP, tenant) across many stateless API servers, e.g. 100 req/min. Low added latency (< a few ms), high availability, roughly accurate globally, and support for different limits per tier.

**Algorithms**

| Algorithm | Notes |
| --- | --- |
| Fixed window counter | Simple `INCR key:minute`; allows 2× bursts at window edges |
| Sliding window log | Exact; stores every timestamp, so memory-heavy |
| Sliding window counter | Weighted mix of current and previous window; good accuracy, O(1) memory |
| **Token bucket** | Refill rate + burst capacity; the industry default (allows controlled bursts) |
| Leaky bucket | Smooths output rate; good for shaping traffic to downstreams |

**Architecture**

```text
Client → LB → API servers (rate-limit middleware) → Redis Cluster (sharded by key)
                                   ↘ config service (limits per tier, cached locally)
```

- State lives in a **central fast store (Redis)**, sharded by rate-limit key so one key lives on one shard.
- **Atomicity:** do check-and-decrement in a single **Lua script** (or `INCR` + `EXPIRE`) to avoid races between servers:

```lua
-- token bucket: KEYS[1]=bucket, ARGV: capacity, refill_per_ms, now_ms, cost
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[1])
local ts = tonumber(b[2]) or tonumber(ARGV[3])
tokens = math.min(tonumber(ARGV[1]), tokens + (tonumber(ARGV[3]) - ts) * tonumber(ARGV[2]))
local ok = tokens >= tonumber(ARGV[4])
if ok then tokens = tokens - tonumber(ARGV[4]) end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
redis.call('PEXPIRE', KEYS[1], 60000)
return ok and 1 or 0
```

**Scaling and trade-offs**

- **Latency vs accuracy:** for very high QPS, keep **local in-memory buckets** and sync to Redis asynchronously, or give each node a slice of the global quota. This trades a small over-admission for no network hop.
- **Failure mode:** if Redis is down, *fail open* (availability) for most APIs, or *fail closed* for expensive or abusive endpoints such as LLM calls or login. Use circuit breakers.
- **Multi-region:** per-region limits with approximate global aggregation; strict global consistency is too costly.
- **Hot keys:** a single tenant flooding one shard needs local pre-limiting.
- **Clock:** use Redis server time, or tolerate skew.
- **Responses:** return `429` with `Retry-After` and `X-RateLimit-Remaining` headers. Clients back off with jitter.
- **LLM-specific:** limit **tokens per minute** as well as requests, using the estimated cost of the request.

### Q112. What is ROC-AUC, and what are its limitations?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q113. When should you use PR-AUC over ROC-AUC?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q114. How do you detect hallucinations?
> ✍️ _Drafted answer (not from the original notes)._

Detect hallucinations at several layers. No single method is enough.

**1. Groundedness against sources (RAG and summarization)**

- Split the answer into atomic claims and check each against the retrieved context with an **NLI/entailment model** or an **LLM judge** ("supported / contradicted / not found"). Examples: RAGAS faithfulness, TruLens groundedness, SummaC, AlignScore.
- **Citation verification:** require citations per claim and check that the cited span actually supports the claim.

**2. Reference-free signals (no source available)**

- **Self-consistency:** sample N answers; disagreement between samples signals hallucination (SelfCheckGPT). Semantic entropy clusters the answers by meaning before measuring the spread.
- **Token-level uncertainty:** low log-probabilities or high entropy on key entities and numbers. It is cheap but only loosely calibrated.
- **External verification:** check claims against a search engine, knowledge base, or database (e.g., FActScore-style factual precision).
- **Model internals:** linear probes on hidden states that predict truthfulness (research stage).

**3. Structural and domain checks:** validate generated SQL, code, URLs, IDs, citations and numbers against systems of record (does that invoice ID exist? does the code compile? do the totals add up?). These are cheap and very high precision.

**4. Human and offline evaluation:** golden datasets with known answers, adversarial "unanswerable" questions to test refusals, and human review of samples to calibrate the automated judges.

**In production**

- Score groundedness on sampled or all traffic and dashboard the **hallucination rate** over time.
- Gate high-stakes responses: if the score falls below a threshold, regenerate, answer "I don't know", or escalate to a human.
- Log failures into the regression suite.

**Trade-off:** judge-based checks add latency and cost, so run lightweight checks inline and heavier ones asynchronously.

### Q115. How do you run a good ablation study?
> ✍️ _Drafted answer (not from the original notes)._

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

### Q116. How do you evaluate LLM outputs (metrics, LLM-as-judge, human eval)?
> ✍️ _Drafted answer (not from the original notes)._

Combine three layers: **automatic metrics**, **LLM-as-judge**, and **human evaluation**, run both offline and online.

**1. Automatic / reference-based metrics**

- Exact match, F1, accuracy for closed-form QA and classification. **Execution-based** checks for code (pass@k, unit tests) and SQL (result match).
- Lexical overlap (BLEU, ROUGE) and embedding similarity (BERTScore) are cheap but correlate weakly with quality for open-ended generation.
- Task-specific: JSON-schema validity, citation correctness, groundedness or faithfulness (NLI/RAGAS), toxicity classifiers, length and format compliance.

**2. LLM-as-judge**

- A strong model grades outputs against a **rubric** (pointwise 1–5 per criterion, such as correctness, helpfulness, faithfulness, tone) or compares two outputs (**pairwise**, which is more reliable for A/B tests).
- Make it robust: give clear rubrics with examples, ask for reasoning before the score, use reference answers when available, randomize order and swap positions (position bias), control for **verbosity and self-preference bias** (use a judge from a different model family), and use low temperature.
- **Calibrate:** measure the judge's agreement with human labels (Cohen's κ / Spearman) on a sample before trusting it. Recheck after any prompt or model change.

**3. Human evaluation**

- The gold standard for subjective or high-stakes quality: expert annotators, clear guidelines, blind and randomized comparisons, inter-annotator agreement measured, enough samples for significance.
- It is expensive, so use it to build golden datasets, calibrate judges, and audit edge cases.

**Operationalizing**

- **Golden dataset** (representative plus adversarial/edge cases), versioned, run in **CI** on every prompt or model change, with regression thresholds.
- **Online:** user feedback (thumbs, edits, retries), task success and escalation rates, A/B tests, sampled production traces scored by the judge, all on dashboards.
- Always track quality together with **latency, cost and safety**. A "better" output that is 3× slower may not ship.

### Q117. What does TensorRT (TensorRT-LLM) do?
> ✍️ _Drafted answer (not from the original notes)._

**TensorRT** is NVIDIA's SDK for **optimizing and running trained neural networks for inference on NVIDIA GPUs**. It takes a model (typically via ONNX or framework integrations such as Torch-TensorRT) and compiles it into an optimized **engine** for a specific GPU:

- **Graph optimizations:** layer and kernel **fusion** (e.g., conv + bias + ReLU in one kernel), constant folding, removing no-op layers.
- **Precision calibration:** FP16, BF16, INT8 (with calibration data), FP8 on Hopper/Blackwell, and INT4 weight-only.
- **Kernel auto-tuning:** benchmarks candidate kernels and picks the fastest for the target GPU, batch size and shapes.
- **Memory optimization:** reuses tensor memory, and supports dynamic shapes through optimization profiles.

**TensorRT-LLM** is the open-source library built on it specifically for **LLM inference**:

- Optimized Transformer kernels, fused attention (FlashAttention-style), **paged KV cache**, KV-cache quantization and reuse (prefix caching).
- **In-flight (continuous) batching**, chunked prefill, **speculative decoding** (draft models, Medusa, EAGLE), and disaggregated prefill/decode.
- Quantization (FP8, INT8 SmoothQuant, INT4 AWQ/GPTQ) and **tensor/pipeline parallelism** across multiple GPUs and nodes.
- Served via **Triton Inference Server** or NVIDIA NIM, with an OpenAI-compatible API.

**Trade-offs**

- Often delivers the best throughput and latency on NVIDIA hardware, frequently ahead of generic PyTorch serving.
- But: NVIDIA lock-in, engines are tied to a GPU type and TensorRT version (rebuild when either changes), a longer build and compile step, more operational complexity, and support for new model architectures can lag behind.
- Alternatives: **vLLM** and **SGLang** (easier to use, broad model support, close performance), ONNX Runtime, and llama.cpp for CPU/edge.

**When to use it:** high-volume, latency-sensitive production on fixed NVIDIA hardware where the extra engineering effort pays off in GPU cost.

### Q118. What are the main techniques for optimizing LLM inference overall (latency, throughput, cost)?
> ✍️ _Drafted answer (not from the original notes)._

First identify what you are optimizing: **latency** (TTFT, inter-token latency), **throughput** (tokens/s per GPU), or **cost** ($ per 1M tokens). Prefill is compute-bound; decode is **memory-bandwidth-bound** (reading weights and the KV cache for every token).

**Model-level**

- **Quantization:** FP8/INT8 weights and activations, INT4 weight-only (AWQ/GPTQ), KV-cache quantization. Gives 2–4× less memory and faster decode for a small quality loss.
- **Smaller or distilled models**, and **routing** (simple queries to a small model, hard ones to a large model).
- **Architecture choices:** GQA/MQA (smaller KV cache), MoE (fewer active parameters per token), sliding-window attention.

**Serving and system-level**

- **Continuous (in-flight) batching:** raises GPU utilization dramatically compared with static batching.
- **PagedAttention:** near-zero KV-cache fragmentation, so more concurrent sequences fit.
- **Optimized kernels:** FlashAttention, fused ops, CUDA graphs (vLLM, SGLang, TensorRT-LLM).
- **Prefix / prompt caching:** reuse the KV cache of shared system prompts and documents.
- **Speculative decoding:** a draft model, Medusa/EAGLE heads or n-gram lookup proposes tokens and the large model verifies several at once. Gives 2–3× lower latency with identical output distribution.
- **Chunked prefill** and **disaggregated prefill/decode** on separate GPU pools, so long prompts don't stall ongoing decodes.
- **Parallelism:** tensor parallelism for latency, replicas (data parallelism) for throughput, pipeline/expert parallelism for very large models.

**Application-level**

- Shorter prompts (context budgeting, compression, fewer retrieved chunks), **output length limits**, structured outputs, stop sequences.
- **Semantic caching** of full responses, **streaming** to improve perceived latency, **batch APIs** (about 50% cheaper) for offline jobs.
- Parallelize tool calls and retrieval, and avoid unnecessary agent steps.

**Trade-offs to discuss:** bigger batches raise throughput but hurt per-request latency, so tune to the SLO. Quantization and distillation can cost quality, so evaluate on *your* tasks. Every optimization should be measured on a load test (p50/p95/p99 TTFT, tokens/s, GPU utilization, cost per request).

### Q119. How do you build a spam detection system?
> ✍️ _Drafted answer (not from the original notes)._

**1. Problem framing:** binary (or multi-class: spam, phishing, promo, ham) classification on messages or emails. **Precision matters most**: putting a legitimate email in spam is very costly. So pick an operating point such as precision ≥ 99% and maximize recall. Latency target is under about 50 ms per message at high volume. The setting is **adversarial**: spammers adapt.

**2. Data and labels:** user "report spam / not spam" actions (noisy, biased), honeypot accounts, analyst labels, blocklists. Use time-based splits, deduplicate near-identical campaigns (avoid leakage), and handle class imbalance by weighting rather than naive oversampling.

**3. Features / signals**

- **Content:** TF-IDF n-grams, embeddings (a fine-tuned DistilBERT or similar), URLs and their reputation, attachments, obfuscation (l33t speak, hidden text, Unicode homoglyphs).
- **Sender and metadata:** sender reputation and domain age, SPF/DKIM/DMARC results, IP reputation, sending volume and velocity, header anomalies.
- **Behavioral / graph:** recipient engagement, how many users received similar content (campaign clustering via MinHash), sender-recipient graph.

**4. Models (layered)**

- **Rules and blocklists** for known-bad items. Fast, explainable.
- **GBDT** (LightGBM) on reputation and metadata features, plus a **text model** (logistic regression on TF-IDF as a baseline, a transformer for quality). Combine via stacking.
- Optional: an **LLM** for borderline cases or a phishing-intent explanation, where cost allows.

**5. Serving:** a real-time scoring service with a feature store (precomputed reputations), thresholds for inbox / spam folder / quarantine / block, and per-user personalization (what one user calls spam is a newsletter to another).

**6. Evaluation and monitoring:** precision, recall, PR-AUC, false-positive rate on important mail, user complaint rate. Online A/B tests. Monitor **drift** closely: new campaigns appear daily.

**7. Adversarial loop:** frequent retraining (daily, or online learning), fast rule deployment for emerging campaigns, adversarial testing, rate limits on suspicious senders, and keep the exact features secret so they are harder to game.

### Q120. How do you approach text summarization?
> ✍️ _Drafted answer (not from the original notes)._

**1. Clarify requirements:** extractive or abstractive? Single or multiple documents? Length and format (bullets, TL;DR, structured fields)? Domain (legal, medical, meetings)? Faithfulness requirements (regulated domains allow zero hallucination)? Volume, latency, cost?

**2. Approaches**

- **Extractive:** select the key sentences (TextRank / LexRank, or a supervised sentence scorer such as BERTSum). Faithful by construction, cheap and explainable, but choppy and redundant.
- **Abstractive with seq2seq models:** fine-tuned BART, T5 or PEGASUS. Fluent and good for fixed domains with training data, but can hallucinate.
- **LLM prompting (the default today):** instruction-tuned LLMs with clear prompts (audience, length, format, "only use information in the text"), few-shot examples, and structured output (JSON fields such as decisions and action items).

**3. Long documents**

- **Map-reduce:** summarize chunks, then summarize the summaries.
- **Refine:** iteratively update a running summary chunk by chunk.
- **Hierarchical:** section → document → corpus.
- **Long-context models:** simpler, but watch for lost-in-the-middle effects and cost.
- **Query-focused summarization:** retrieve only the relevant parts first (RAG).

**4. Evaluation**

- ROUGE or BERTScore against references (weak signal for LLM outputs).
- **Faithfulness:** NLI-based checks (SummaC, AlignScore), QA-based checks (QAGS, QuestEval), LLM-as-judge with a rubric (coverage, faithfulness, conciseness, coherence).
- Human review on a sample, especially in high-stakes domains.

**5. Production concerns:** hallucinated numbers, names and dates (verify entities against the source, or add citations per bullet), omission of critical information, PII handling, prompt injection inside the documents being summarized, model choice by cost (a cheap model for bulk work, a stronger one for important documents), caching, and monitoring of user edits and feedback as a quality signal.

### Q121. How are knowledge graphs used with NLP?
> ✍️ _Drafted answer (not from the original notes)._

A **knowledge graph (KG)** stores facts as entities and typed relations, e.g. `(Marie Curie, wonPrize, Nobel Prize in Physics)`. NLP and KGs help each other in both directions.

**NLP → building KGs**

- **Named Entity Recognition** to find entities, and **entity linking / disambiguation** to map each mention to a canonical node ("Apple" the company vs. the fruit; Wikidata IDs).
- **Relation extraction / open information extraction** to pull (subject, relation, object) triples from text, now often done with LLMs and structured outputs.
- **Coreference resolution**, **entity resolution** (deduplicating nodes), and **schema/ontology alignment**.

**KGs → improving NLP**

- **Question answering over KGs (KGQA):** translate natural language to SPARQL or Cypher (text-to-query with LLMs), giving precise answers for multi-hop and aggregate questions.
- **GraphRAG / KG-augmented RAG:** retrieve entities and their neighborhoods or community summaries instead of (or as well as) text chunks. This is good for multi-hop ("which suppliers of our top customer had incidents?") and global sensemaking questions.
- **Grounding and hallucination reduction:** check generated claims against KG facts, and constrain outputs to valid entities.
- **Search and recommendation:** entity-aware query understanding, knowledge panels, query expansion via related entities, explainable recommendations (paths in the graph).
- **Knowledge-enhanced language models:** inject entity embeddings (ERNIE, KnowBERT), or use the KG for consistency in domain-specific models (biomedical, finance).

**Trade-offs:** KGs give precision, explainability, multi-hop reasoning and easy updates of facts. They are costly to build and maintain, have incomplete coverage, are brittle to extraction errors, and need schema governance.

**Production pattern:** combine them. Run LLM-based extraction with human validation for critical relations, store the result in a graph DB (Neo4j, Neptune), and use hybrid retrieval (vector + graph traversal) at query time.

### Q122. How do you build a harmful-content detection system?
> ✍️ _Drafted answer (not from the original notes)._

**1. Scope and policy first:** define the taxonomy (hate, harassment, violence, self-harm, sexual content, CSAM, extremism, spam, misinformation) with written policy guidelines and severity levels. The policy drives labeling, thresholds and actions. Cover multiple modalities (text, image, video, audio) and languages.

**2. Data and labeling:** expert annotators with clear guidelines, inter-annotator agreement tracking, annotator wellbeing safeguards, active learning to pick borderline examples, adversarial and red-team data, and rare-class oversampling. Severe classes are extremely imbalanced.

**3. Model stack (cascade for cost and latency)**

- **Hash matching** of known-bad content (PhotoDNA / perceptual hashes for CSAM and terror content, shared via industry hash databases). Exact and fast.
- **Lightweight classifiers** on everything: fine-tuned multilingual transformers (e.g., XLM-R) for text, CNN/ViT for images, multi-label, one head per policy.
- **Heavier models / LLM judges** (e.g., Llama Guard-style policy-conditioned classifiers, VLMs) for the uncertain middle band or new policies (zero/few-shot).
- **Context and behavioral signals:** user history, account age, report velocity, network or coordinated behavior.

**4. Decisioning:** per-class thresholds tied to actions: allow, down-rank, blur/interstitial, age-gate, remove, escalate to human review, report to authorities (CSAM, which is legally required). Severe harms get high-recall thresholds; borderline speech gets high-precision thresholds to limit over-moderation.

**5. Human in the loop:** review queues prioritized by severity and reach, appeals, reviewer decisions fed back as labels. Track reviewer accuracy.

**6. Evaluation and fairness:** per-class precision and recall, PR-AUC, prevalence estimation through sampling, latency. Do **bias audits across dialects and identity terms** (e.g., AAVE falsely flagged, reclaimed slurs) and per-language quality.

**7. Adversarial robustness and operations:** obfuscation (misspellings, emoji, text in images, so add OCR), evolving slang, coordinated attacks. Mitigate with frequent retraining, rapid rule updates, drift monitoring, and transparency reports.

### Q123. How would you solve entity resolution?
> ✍️ _Drafted answer (not from the original notes)._

**Entity resolution (ER)** identifies records that refer to the same real-world entity across or within datasets (customers, companies, products) despite typos, abbreviations, missing fields and format differences.

**Pipeline**

1. **Standardize / clean:** normalize case, punctuation, addresses (libpostal), phone numbers (E.164), names (nicknames such as Bob → Robert), company suffixes (Inc, Ltd).
2. **Blocking / candidate generation:** comparing all pairs is $O(n^2)$, which is infeasible. Group by blocking keys (zip code, first 3 letters plus year), sorted neighborhood, **LSH/MinHash**, or **embedding ANN search** to produce candidate pairs. Tune for high recall.
3. **Pairwise matching:** compute similarity features per field (Jaro-Winkler, Levenshtein, TF-IDF cosine, phonetic codes such as Soundex/Metaphone, numeric and date distances, embedding similarity). Then classify:
   - Rules or **probabilistic Fellegi-Sunter** models (Splink: unsupervised EM, explainable match weights), or
   - Supervised models (GBDT, or fine-tuned transformers such as Ditto on serialized record pairs), or
   - LLMs to adjudicate hard pairs (expensive, so use them for the uncertain band only).
4. **Clustering:** turn pairwise matches into entities via connected components, or correlation clustering or thresholded graph clustering to avoid "chain" errors (A≈B, B≈C, but A≠C).
5. **Canonicalization / golden record:** survivorship rules decide which value wins (most recent, most trusted source), and lineage is kept back to the source records.

**Evaluation:** pairwise precision and recall, cluster metrics (B³, pairwise F1) on a labeled sample. Active learning to label the most informative uncertain pairs. **Clerical review** queue for mid-confidence matches.

**Production concerns:** incremental or streaming ER for new records (match against the existing index rather than re-running everything), stable entity IDs over time, scalability (Spark/Splink, Zingg), PII and privacy (hashing, access controls), and explainability for regulated uses (KYC/AML).

**Trade-off:** false merges (combining two customers) are usually worse than missed matches, so set thresholds asymmetrically.

### Q124. How did you make the agentic RAG system reliable, trustworthy, explainable, responsible, and aligned?
> ✍️ _Drafted answer (not from the original notes)._

This is behavioural, so answer in **STAR** form with concrete numbers. Template to adapt:

**Situation / Task:** "We built an agentic RAG assistant that answered [domain, e.g., policy/compliance] questions for [N users], using tools such as search, a SQL lookup, and ticket creation. Early pilots showed about [X]% unsupported answers and occasional wrong tool actions, which was unacceptable in a regulated setting."

**Actions, grouped by property**

- **Reliable:** built a golden dataset of [~500] real questions with gold sources and answers. Separated retrieval metrics (recall@k) from generation metrics (faithfulness, correctness). Added hybrid search plus a cross-encoder reranker. Used Pydantic-validated tool calls with retries, max-step limits, timeouts and fallbacks. Ran a regression suite in CI on every prompt or model change.
- **Trustworthy:** grounded answers with **per-claim citations**, verified automatically. Added an "I don't know / escalate" path when retrieval confidence or the groundedness score fell below a threshold. Calibrated an LLM judge against human labels (κ ≈ [0.7]).
- **Explainable:** full tracing (Langfuse/OpenTelemetry) of retrieved chunks, tool calls and reasoning summaries. The UI showed sources and *why* an action was proposed. Every decision was auditable.
- **Responsible:** PII redaction before logging and before external API calls, role-based document access enforced at retrieval time (no cross-tenant leakage), prompt-injection defenses (retrieved content treated as data, tool allowlists), bias and safety red-teaming, data-retention policies.
- **Aligned:** worked with domain experts and stakeholders to define success and the policy. Required human-in-the-loop approval for side-effecting actions (sending, updating records). Collected user feedback and turned failure cases into new eval items and prompt or data fixes.

**Result:** "Hallucination rate fell from [X]% to [Y]%, answer accuracy rose to [Z]% on the golden set, escalation rate settled at [~8]%, p95 latency stayed under [N] s, and the system passed compliance review and rolled out to [N] teams."

**Reflection:** "The biggest lesson was that evaluation infrastructure, not model choice, drove reliability. Next I'd invest earlier in [online evaluation / better document parsing]."

Keep it truthful: replace the placeholders with your real numbers, and be ready to go deep on one item, such as how you calibrated the judge.

### 20 Neural Network Algorithms
| # | Algorithm | Category | Core Idea | Typical Use |
| --- | --- | --- | --- | --- |
| 1 | Perceptron | Basics | Single neuron: weighted sum plus threshold | Linear binary classification |
| 2 | Multi-Layer Perceptron (MLP) | Basics | Stacked fully connected layers with nonlinearities | Tabular data, function approximation |
| 3 | Convolutional Neural Network (CNN) | Basics | Shared convolutional filters that detect local patterns | Image classification, vision |
| 4 | Recurrent Neural Network (RNN) | Basics | Hidden state passed step to step through a sequence | Time series, early NLP |
| 5 | Long Short-Term Memory (LSTM) | Basics | RNN with gates and a memory cell for long-range information | Speech, forecasting, text |
| 6 | Gated Recurrent Unit (GRU) | Sequence & Language | Simplified LSTM with fewer gates | Same as LSTM, faster to train |
| 7 | Transformer | Sequence & Language | Attention-based encoder/decoder, processes tokens in parallel | Translation, LLMs, vision |
| 8 | Self-Attention | Sequence & Language | Each token weighs every other token via query/key/value | Core building block of Transformers |
| 9 | BERT | Sequence & Language | Bidirectional Transformer encoder, pretrained by masked-word prediction | Classification, search, QA |
| 10 | GPT | Sequence & Language | Autoregressive Transformer decoder that predicts the next token | Text generation, chat, code |
| 11 | ResNet | Advanced Architectures | Skip connections let very deep networks train | Image recognition backbones |
| 12 | InceptionNet | Advanced Architectures | Parallel filters of different sizes in one block | Efficient image classification |
| 13 | Generative Adversarial Network (GAN) | Advanced Architectures | Generator and discriminator trained against each other | Image synthesis, data augmentation |
| 14 | Autoencoder | Advanced Architectures | Compress to a latent code, then reconstruct | Denoising, anomaly detection, embeddings |
| 15 | Deep Q-Network (DQN) | Advanced Architectures | Neural network approximating Q-values in reinforcement learning | Games, control |
| 16 | Attention Is All You Need | Modern Deep Learning | The 2017 paper that introduced the Transformer (a paper, not an algorithm) | See #7 |
| 17 | Deep Belief Network | Modern Deep Learning | Stacked restricted Boltzmann machines, pretrained layer by layer | Historical unsupervised pretraining |
| 18 | U-Net | Modern Deep Learning | Encoder-decoder with skip connections between matching levels | Medical and other image segmentation |
| 19 | YOLO | Modern Deep Learning | Single-pass object detection with boxes and classes together | Real-time detection |
| 20 | Diffusion Models | Modern Deep Learning | Learn to reverse a gradual noising process | Image, audio, and video generation |

### 5 Layers of Agentic Memory
1\. Working Memory

Stores the information needed for the agent’s current task, including intermediate steps and active context.

2\. Short-Term Memory

Maintains recent conversation history and context so the agent can respond consistently throughout an interaction.

3\. Long-Term Memory

Stores persistent information, such as user preferences, facts, and knowledge that may be useful in future sessions.

4\. Episodic Memory

Records past experiences, including what happened, which actions were taken, and what outcomes followed.

5\. Semantic Memory

Stores structured facts and concepts that help the agent retrieve and apply knowledge across different tasks.

## Covered elsewhere

These questions are answered by another question in this repo:

- What is the difference between Parametric and Non Parametric Algorithms? → see [01-turing-100-interview-questions.md Q84](resources/01-turing-100-interview-questions.md#q84-how-can-you-differentiate-between-a-parametric-and-a-non-parametric-model)
- Give an example of when a false positive is more crucial than a false negative, and vice versa → see [01-turing-100-interview-questions.md Q88](resources/01-turing-100-interview-questions.md#q88-which-is-better-to-have-a-false-positive-or-false-negative)
- Why is “Naive” Bayes naive? → see [01-turing-100-interview-questions.md Q50](resources/01-turing-100-interview-questions.md#q50-why-is-the-naive-bayes-method-naive)
- Why is KNN known as a lazy learning technique? → see [01-turing-100-interview-questions.md Q92](resources/01-turing-100-interview-questions.md#q92-why-do-you-think-instance-based-learning-algorithm-is-sometimes-referred-to-as-lazy-learning-algorithm)
- What do you mean by semi-supervised learning? → see [01-turing-100-interview-questions.md Q24](resources/01-turing-100-interview-questions.md#q24-what-is-semi-supervised-learning)
- What are the assumptions of linear regression? → see [01-turing-100-interview-questions.md Q6](resources/01-turing-100-interview-questions.md#q6-to-start-linear-regression-you-would-need-to-make-some-assumptions-what-are-those-assumptions)
- What is an Imbalanced Dataset, and how can one deal with this problem? → see [advanced-ml.md Q5](advanced-ml.md#q5-how-would-you-handle-severe-class-imbalance)
- What is Multicollinearity? How to detect it? List some techniques to overcome Multicollinearity. → see [01-turing-100-interview-questions.md Q7](resources/01-turing-100-interview-questions.md#q7-what-is-multicollinearity-and-how-will-you-handle-it-in-your-regression-model)
- List some ways using which you can reduce overfitting in a model. → see [01-turing-100-interview-questions.md Q87](resources/01-turing-100-interview-questions.md#q87-how-can-overfitting-be-avoided-in-ml)
- When Your Dataset Is Suffering From High Variance, How Would You Handle It? → see [01-turing-100-interview-questions.md Q89](resources/01-turing-100-interview-questions.md#q89-how-would-you-handle-a-dataset-suffering-from-high-variance)
- State the differences between causality and correlation? → see [01-turing-100-interview-questions.md Q16](resources/01-turing-100-interview-questions.md#q16-what-is-the-difference-between-correlation-covariance-and-causality)
- What is a p-value, and why is it important? → see [01-turing-100-interview-questions.md Q30](resources/01-turing-100-interview-questions.md#q30-what-is-p-value)
- What is the difference between loss function and cost function? → see [01-turing-100-interview-questions.md Q78](resources/01-turing-100-interview-questions.md#q78-explain-the-difference-between-the-loss-functions-and-cost-functions)
- What are the common ways to handle missing data in a dataset? → see [01-turing-100-interview-questions.md Q3](resources/01-turing-100-interview-questions.md#q3-you-have-found-missing-data-in-your-dataset-how-will-you-handle-it)
- What do you mean by the bias-variance tradeoff? → see [advanced-ml.md Q1](advanced-ml.md#q1-explain-the-bias-variance-tradeoff-mathematically-and-practically)
- What is the main difference between Machine Learning and Data Mining? → see [01-turing-100-interview-questions.md Q64](resources/01-turing-100-interview-questions.md#q64-how-would-you-differentiate-data-mining-from-machine-learning)
- What is the difference between Type 1 and Type 2 error? → see [01-turing-100-interview-questions.md Q23](resources/01-turing-100-interview-questions.md#q23-what-are-the-differences-between-type-i-error-and-type-ii-error)
- What is the difference between a Generative model vs. a Discriminative model? → see [advanced-ml.md Q11](advanced-ml.md#q11-what-is-the-difference-between-generative-and-discriminative-models)
- What is the difference between inductive machine learning and deductive machine learning? → see [01-turing-100-interview-questions.md Q63](resources/01-turing-100-interview-questions.md#q63-what-is-deductive-learning-and-inductive-learning)
- What is the difference between Covariance and Correlation → see [01-turing-100-interview-questions.md Q16](resources/01-turing-100-interview-questions.md#q16-what-is-the-difference-between-correlation-covariance-and-causality)
- What is the difference between stochastic gradient descent (SGD) and gradient descent? → see [01-turing-100-interview-questions.md Q47](resources/01-turing-100-interview-questions.md#q47-what-are-the-differences-between-stochastic-gradient-descent-sgd-and-gradient-descent-gd)
- Segmentation vs detection? → see [02-ai-engineer-senior-level.md Q14](role-prep/02-ai-engineer-senior-level.md#q14-whats-the-difference-between-object-detection-image-classification-and-segmentation-and-when-would-you-use-each)
- Speculative Decoding? → see [advanced-ml.md Q143](advanced-ml.md#q143-what-is-speculative-decoding)
- Why use multi-head attention instead of a single attention head? → see [advanced-ml.md Q67](advanced-ml.md#q67-why-are-multiple-attention-heads-useful)
- What are the differences between LSTM and GRU? → see [Q80](#q80-rnn-vs-lstm-vs-gru-vs-transformers)
- How does LoRA work? → see [advanced-ml.md Q152](advanced-ml.md#q152-explain-lora-and-qlora)
- What is catastrophic forgetting, and how do you mitigate it? → see [advanced-ml.md Q85](advanced-ml.md#q85-what-is-catastrophic-forgetting-during-fine-tuning)
- What is a KV cache, and why does it matter? → see [advanced-ml.md Q69](advanced-ml.md#q69-why-does-kv-cache-significantly-improve-autoregressive-generation)
- Design a RAG system for internal documents. → see [advanced-ml.md Q109](advanced-ml.md#q109-design-an-enterprise-rag-platform)
- What is quantization, and what are the trade-offs? → see [advanced-ml.md Q80](advanced-ml.md#q80-what-is-quantization)
- Why separate prefill and decode? → see [advanced-ml.md Q165](advanced-ml.md#q165-what-is-llm-inference-explain-prefill-vs-decode)
- Why do language models hallucinate? → see [advanced-ml.md Q87](advanced-ml.md#q87-what-causes-llm-hallucinations)
- Hopfield network, Boltzmann machine, Restricted Boltzmann machine? → see [Q97](#q97-hopfield-networks-vs-boltzmann-machine)
