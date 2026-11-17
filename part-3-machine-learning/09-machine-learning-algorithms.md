# Chapter 9. Machine Learning Algorithms

_Part III — Machine Learning_

Linear models, trees and ensembles (random forests and gradient boosting internals), SVMs, Bayesian methods, clustering, dimensionality reduction, anomaly detection and recommender basics: how each works, when to use it, how to interpret it and how it fails.

This chapter contains **57 questions**.

## Contents

- **9.1** [How does K-Nearest Neighbours (KNN) work, how does k affect overfitting, and why is it called a lazy learner?](#91-how-does-k-nearest-neighbours-knn-work-how-does-k-affect-overfitting-and-why-is-it-called-a-lazy-learner)
- **9.2** [What is the difference between k-NN and k-means?](#92-what-is-the-difference-between-k-nn-and-k-means)
- **9.3** [How does a decision tree work, and what are its advantages and disadvantages?](#93-how-does-a-decision-tree-work-and-what-are-its-advantages-and-disadvantages)
- **9.4** [Explain how Decision Trees make splits and handle categorical features.](#94-explain-how-decision-trees-make-splits-and-handle-categorical-features)
- **9.5** [How does the random forest algorithm work, and how does it improve on a single decision tree?](#95-how-does-the-random-forest-algorithm-work-and-how-does-it-improve-on-a-single-decision-tree)
- **9.6** [What are ensemble methods, which techniques can combine multiple models, and why do they work?](#96-what-are-ensemble-methods-which-techniques-can-combine-multiple-models-and-why-do-they-work)
- **9.7** [What is the difference between bagging and boosting?](#97-what-is-the-difference-between-bagging-and-boosting)
- **9.8** [What is gradient boosting, and how does XGBoost work?](#98-what-is-gradient-boosting-and-how-does-xgboost-work)
- **9.9** [What are the key hyperparameters for XGBoost, and how do you tune them?](#99-what-are-the-key-hyperparameters-for-xgboost-and-how-do-you-tune-them)
- **9.10** [How does gradient boosting differ from random forest, and what are its advantages?](#910-how-does-gradient-boosting-differ-from-random-forest-and-what-are-its-advantages)
- **9.11** [How does logistic regression work, how does it differ from linear regression, and can it handle more than two classes?](#911-how-does-logistic-regression-work-how-does-it-differ-from-linear-regression-and-can-it-handle-more-than-two-classes)
- **9.12** [Explain R-squared and adjusted R-squared.](#912-explain-r-squared-and-adjusted-r-squared)
- **9.13** [What is multicollinearity, how do you detect it, and how do you handle it in a regression model?](#913-what-is-multicollinearity-how-do-you-detect-it-and-how-do-you-handle-it-in-a-regression-model)
- **9.14** [Explain K-Means Clustering. How does it work? Limitations?](#914-explain-k-means-clustering-how-does-it-work-limitations)
- **9.15** [How do support vector machines work, and what are support vectors?](#915-how-do-support-vector-machines-work-and-what-are-support-vectors)
- **9.16** [What is the decision boundary in classifiers?](#916-what-is-the-decision-boundary-in-classifiers)
- **9.17** [How does Naive Bayes work, why is it called naive, and what is Bayes' theorem in this context?](#917-how-does-naive-bayes-work-why-is-it-called-naive-and-what-is-bayes-theorem-in-this-context)
- **9.18** [What is Dimensionality Reduction?](#918-what-is-dimensionality-reduction)
- **9.19** [How does PCA work, and when would you use it?](#919-how-does-pca-work-and-when-would-you-use-it)
- **9.20** [Explain Gradient Descent and its variants.](#920-explain-gradient-descent-and-its-variants)
- **9.21** [What is the ROC curve and AUC, and how are they interpreted?](#921-what-is-the-roc-curve-and-auc-and-how-are-they-interpreted)
- **9.22** [What assumptions does linear regression make, and what happens when they are violated?](#922-what-assumptions-does-linear-regression-make-and-what-happens-when-they-are-violated)
- **9.23** [Explain why the performance of XGBoost is better than that of SVM?](#923-explain-why-the-performance-of-xgboost-is-better-than-that-of-svm)
- **9.24** [What are the most widely used machine learning algorithms, and when is each the right default?](#924-what-are-the-most-widely-used-machine-learning-algorithms-and-when-is-each-the-right-default)
- **9.25** [Which similarity measures are used in recommender systems?](#925-which-similarity-measures-are-used-in-recommender-systems)
- **9.26** [What is a Bayesian network, and what is it also called?](#926-what-is-a-bayesian-network-and-what-is-it-also-called)
- **9.27** [What is clustering, and how do the main clustering algorithms compare?](#927-what-is-clustering-and-how-do-the-main-clustering-algorithms-compare)
- **9.28** [What is a kernel SVM, what is the kernel trick, and which kernels are common?](#928-what-is-a-kernel-svm-what-is-the-kernel-trick-and-which-kernels-are-common)
- **9.29** [What are content-based and collaborative filtering, and what powers "people who bought this also bought"?](#929-what-are-content-based-and-collaborative-filtering-and-what-powers-people-who-bought-this-also-bought)
- **9.30** [What are the two components of Bayesian logic, and how do they combine?](#930-what-are-the-two-components-of-bayesian-logic-and-how-do-they-combine)
- **9.31** [What is a recommender system, and what are the main types?](#931-what-is-a-recommender-system-and-what-are-the-main-types)
- **9.32** [Explain Genetic programming as it relates to Evolutionary algorithm.](#932-explain-genetic-programming-as-it-relates-to-evolutionary-algorithm)
- **9.33** [What is an OOB error, and how is it useful?](#933-what-is-an-oob-error-and-how-is-it-useful)
- **9.34** [In what scenario should a decision tree be preferred over random forest?](#934-in-what-scenario-should-a-decision-tree-be-preferred-over-random-forest)
- **9.35** [Why is Logistic Regression called regression?](#935-why-is-logistic-regression-called-regression)
- **9.36** [What is Matrix Factorization, and where is it used in Machine Learning?](#936-what-is-matrix-factorization-and-where-is-it-used-in-machine-learning)
- **9.37** [Explain Pruning in Decision Trees and how it is done](#937-explain-pruning-in-decision-trees-and-how-it-is-done)
- **9.38** [What is the difference between Gini Impurity and Entropy? Which one is better and why?](#938-what-is-the-difference-between-gini-impurity-and-entropy-which-one-is-better-and-why)
- **9.39** [Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?](#939-can-pca-be-used-to-reduce-the-dimensionality-of-a-highly-nonlinear-dataset)
- **9.40** [Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?](#940-once-a-datasets-dimensionality-has-been-reduced-is-it-possible-to-reverse-the-operation-if-so-how-if-not-why)
- **9.41** [Why do we always need the intercept term in a regression model?](#941-why-do-we-always-need-the-intercept-term-in-a-regression-model)
- **9.42** [Why is Ridge Regression called Ridge?](#942-why-is-ridge-regression-called-ridge)
- **9.43** [Does it make any sense to chain two different dimensionality reduction algorithms?](#943-does-it-make-any-sense-to-chain-two-different-dimensionality-reduction-algorithms)
- **9.44** [Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?](#944-is-it-possible-to-speed-up-training-of-a-bagging-ensemble-by-distributing-it-across-multiple-servers)
- **9.45** [If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?](#945-if-a-decision-tree-is-underfitting-the-training-set-is-it-a-good-idea-to-try-scaling-the-input-features)
- **9.46** [Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?](#946-say-you-trained-an-svm-classifier-with-an-rbf-kernel-it-seems-to-underfit-the-training-set-should-you-increase-or-decrease-γ-gamma-what-about-c)
- **9.47** [How do we interpret weights in linear models?](#947-how-do-we-interpret-weights-in-linear-models)
- **9.48** [Why is it important to scale the inputs when using SVMs?](#948-why-is-it-important-to-scale-the-inputs-when-using-svms)
- **9.49** [What is OvR and OvO for multiclass classification, and which machine learning algorithms use them?](#949-what-is-ovr-and-ovo-for-multiclass-classification-and-which-machine-learning-algorithms-use-them)
- **9.50** [What are some advantages and Disadvantages of regression models and tree-based models?](#950-what-are-some-advantages-and-disadvantages-of-regression-models-and-tree-based-models)
- **9.51** [List some of the drawbacks of a Linear model](#951-list-some-of-the-drawbacks-of-a-linear-model)
- **9.52** [What is the difference between a Gaussian Mixture Model and the K-Means Algorithm?](#952-what-is-the-difference-between-a-gaussian-mixture-model-and-the-k-means-algorithm)
- **9.53** [How to select K for K-means?](#953-how-to-select-k-for-k-means)
- **9.54** [How do XGBoost, LightGBM and CatBoost differ internally, and how do you choose between them?](#954-how-do-xgboost-lightgbm-and-catboost-differ-internally-and-how-do-you-choose-between-them)
- **9.55** [How do you interpret tree ensembles with feature importance and SHAP, and what are the pitfalls?](#955-how-do-you-interpret-tree-ensembles-with-feature-importance-and-shap-and-what-are-the-pitfalls)
- **9.56** [Which anomaly detection algorithms are there, how do they compare, and how do you choose one?](#956-which-anomaly-detection-algorithms-are-there-how-do-they-compare-and-how-do-you-choose-one)
- **9.57** [How do you approach a new tabular prediction problem in practice, and when is gradient boosting not the right choice?](#957-how-do-you-approach-a-new-tabular-prediction-problem-in-practice-and-when-is-gradient-boosting-not-the-right-choice)

---

### 9.1 How does K-Nearest Neighbours (KNN) work, how does k affect overfitting, and why is it called a lazy learner?

KNN is a non-parametric, instance-based algorithm: it stores the training set and, for a new point, predicts from the $k$ closest training examples (majority vote for classification, mean or distance-weighted mean for regression). There is no model fitting step, which is why it is called a **lazy learner**.

**How a prediction is made**

1. Compute the distance from the query to every training point (Euclidean, Manhattan, Minkowski, cosine for text or embeddings).
2. Take the $k$ nearest.
3. Classification: majority vote, optionally weighted by $1/d$. Regression: average of their targets.

**Effect of k (bias and variance)**

| k | Behaviour | Risk |
| --- | --- | --- |
| Small (k = 1) | Very jagged boundary that follows individual points, training error is zero | Overfits, sensitive to noise and mislabelled points |
| Large | Smooth boundary, averages over a wide area | Underfits; at $k = n$ it always predicts the majority class |

Choose $k$ by cross-validation; odd values avoid ties in binary problems, and $\sqrt{n}$ is only a rough starting point.

**Why "lazy" (instance-based) learning**

- Training is just storing the data, so it is instant, but all the work happens at prediction time.
- Eager learners (logistic regression, trees, neural networks) compress the data into parameters up front and predict cheaply.
- Consequence: naive prediction costs $O(nd)$ per query and memory grows with the dataset.

**Practical issues**

- **Scaling is mandatory:** distances are dominated by large-range features, so standardise first (see 10.5).
- **Curse of dimensionality:** in high dimensions all points are roughly equidistant and "nearest" loses meaning (see 8.27). Reduce dimensions or use learned embeddings.
- **Speed at scale:** KD-trees and ball trees help in low dimensions; for millions of high-dimensional vectors use approximate nearest neighbour indexes (HNSW, IVF, FAISS). This is exactly how embedding retrieval in RAG and recommender candidate generation works.
- **Imbalanced classes:** the majority class wins votes; use distance weighting or class weights.
- **Categorical features:** need a suitable distance (Hamming, Gower) or encoding.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": [3, 5, 11, 21, 41],
                           "kneighborsclassifier__weights": ["uniform", "distance"]}, cv=5)
grid.fit(X_train, y_train)
```

**Where it is still used:** small tabular problems as a baseline, imputation (`KNNImputer`), anomaly scores (distance to the k-th neighbour), and anywhere similarity search over embeddings is the real task.

### 9.2 What is the difference between k-NN and k-means?

#### Core difference

|  | **k-NN** | **k-means** |
| --- | --- | --- |
| **Type** | Supervised (classification/regression) | Unsupervised (clustering) |
| **Goal** | Predict a label/value for a new point | Group unlabeled points into k clusters |
| **What "k" means** | Number of neighbors to consult | Number of clusters to form |
| **Needs labeled data?** | Yes | No |
| **Training phase** | None really — it's "lazy," just stores the data | Yes — iteratively fits cluster centroids |
| **Output** | A prediction per query point | A cluster assignment per point + centroids |

#### How each actually works

**k-NN:**

- Given a new point, find its k closest points in the *labeled* training set
- Classification → majority vote; regression → average
- No model is built ahead of time — all the "work" happens at prediction time (hence "lazy learner")

**k-means:**

- Given unlabeled data, initialize k centroids (randomly, or via k-means++)
- Repeat: assign each point to its nearest centroid → recompute centroids as the mean of assigned points
- Converges when assignments stop changing
- This is "eager" — it builds a model (centroids) upfront

#### Where confusion usually comes from

- Both rely on **distance metrics** (usually Euclidean) — that's the main surface similarity
- Both have a **k that needs tuning** — but tuning k in k-NN is about bias/variance on predictions, while tuning k in k-means is about how many natural groupings exist in the data (elbow method, silhouette score)
- **k-means uses k-NN-style distance calculations internally** during the assignment step, which can make them feel related — but the *purpose* of the k is completely different

#### Quick way to remember

- k-**NN**: k = how many neighbors *vote* on a label you don't know
- k-**means**: k = how many *groups* you're trying to discover

### 9.3 How does a decision tree work, and what are its advantages and disadvantages?

A decision tree predicts by asking a sequence of yes/no questions about the features. Training greedily chooses, at each node, the split that most reduces impurity, and recursively repeats on each child until a stopping rule is hit; each leaf stores a prediction (the majority class or class proportions, or the mean target for regression).

**Structure**

- **Root node:** the first split, chosen over the whole training set.
- **Internal nodes:** a test such as "income > 50k?".
- **Leaves:** the final prediction for every sample that reaches them.

```text
                income > 50k?
               /            \
            yes              no
         age > 30?         reject
        /        \
   approve      review
```

**Training (CART, the scikit-learn algorithm)**

1. For every feature and every candidate threshold, compute the impurity of the two child nodes.
2. Pick the split with the largest weighted impurity decrease: Gini or entropy for classification (see 9.38), variance (MSE) reduction for regression.
3. Recurse on each child.
4. Stop at `max_depth`, `min_samples_leaf`, `min_impurity_decrease`, or when a node is pure; optionally post-prune (see 9.37).

The greedy search is not globally optimal; finding the optimal tree is NP-hard.

**Advantages**

- Readable rules that can be shown to business users or regulators when the tree is shallow.
- Handles numeric and categorical features, non-linear effects and interactions without feature engineering.
- No scaling needed; invariant to monotonic transforms of features (see 9.45).
- Fast to predict; works for classification and regression.
- Building block for random forests and gradient boosting, which fix most of its weaknesses.

**Disadvantages**

- **High variance:** a small change in the data can produce a very different tree.
- **Overfits** when grown deep; a fully grown tree memorises the training set.
- **Axis-aligned splits:** a diagonal boundary such as $x_1 \gt x_2$ needs a staircase of many splits.
- **Cannot extrapolate:** predictions are piecewise constant, bounded by the training targets.
- **Biased impurity-based importance** towards high-cardinality features.
- Probability estimates are coarse (a leaf with 3 samples gives 0, 0.33, 0.67 or 1).

**What interviewers look for:** that you know a single tree is rarely the final model, why ensembles help (variance reduction in random forests, bias reduction in boosting), and when a single tree is still the right choice (see 9.34).

### 9.4 Explain how Decision Trees make splits and handle categorical features.

**How Decision Trees Make Splits**

- At each node, algorithm evaluates all possible splits across all features
- Selects split that maximizes **information gain** (or minimizes impurity) between parent and child nodes
- **Impurity measures**:
  - **Gini Impurity**: $\text{Gini} = 1 - \sum_i p_i^2$ — lower is purer
  - **Entropy**: $-\sum_i p_i \log_2(p_i)$, used with $\text{Information Gain} = \text{Entropy}(\text{parent}) - \text{weighted Entropy}(\text{children})$
  - **Variance Reduction**: used for regression trees
- Process repeats recursively for each child node (greedy, top-down approach)

**Handling Categorical Features**

- **Binary/low-cardinality categories**: Split into subsets, e.g., \{A\} vs \{B, C\}
- **One-hot encoding**: Convert category into multiple binary features before splitting
- **Ordinal encoding**: If categories have natural order, map to integers
- **High-cardinality categories**:
  - Some libraries (e.g., LightGBM, CatBoost) handle natively via optimal category grouping
  - Target/mean encoding sometimes used to convert categories to numeric based on target statistic
- Tree evaluates each possible category split/grouping and picks the one minimizing impurity

### 9.5 How does the random forest algorithm work, and how does it improve on a single decision tree?

A random forest trains many deep decision trees on bootstrap samples of the data, with a random subset of features considered at each split, and averages their predictions (majority vote or averaged probabilities for classification, mean for regression). Averaging many low-bias, high-variance and weakly correlated trees cuts variance sharply while keeping bias roughly the same.

**Algorithm**

1. For $b = 1 \ldots B$: draw a bootstrap sample of $n$ rows with replacement (about 63 percent unique rows).
2. Grow a deep tree on it; at every split consider only `max_features` randomly chosen features (by default $\sqrt{p}$ for classification in scikit-learn, all features for regression).
3. Predict by averaging the $B$ trees.

**Why it reduces variance**

For $B$ trees each with variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the average is

$$
\operatorname{Var}(\bar f) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2
$$

- More trees shrink the second term, so adding trees never causes overfitting in the usual sense; it just stops helping.
- The first term only falls if the trees are **decorrelated**, which is why feature subsampling matters. Bagging alone (all features) gives trees that all split on the same strong feature first.

**Key hyperparameters**

| Parameter | Effect |
| --- | --- |
| `n_estimators` | More is better until the out-of-bag error flattens (often 300 to 1,000) |
| `max_features` | Main lever on tree correlation; lower means more diverse, more biased trees |
| `min_samples_leaf`, `max_depth` | Smooth leaves; helps with noisy targets and calibration |
| `class_weight` | Imbalanced classes |

**Strengths:** strong default with little tuning, robust to noise and outliers, parallel training, a free validation estimate from out-of-bag samples (see 9.33).

**Weaknesses and pitfalls**

- Usually a few points behind tuned gradient boosting on tabular data (see 9.10).
- Large models (hundreds of deep trees) mean memory and latency cost.
- Cannot extrapolate beyond the training target range.
- Impurity-based `feature_importances_` are biased towards continuous and high-cardinality features and are computed on training data; prefer permutation importance on a held-out set or SHAP (see 9.55).
- Probabilities are pulled away from 0 and 1; calibrate if you need them.

**Variant:** Extra Trees also randomise the split thresholds, which decorrelates trees further and trains faster.

### 9.6 What are ensemble methods, which techniques can combine multiple models, and why do they work?

An ensemble combines several models so that their individual errors partly cancel. It works when the members are individually reasonable and their errors are not perfectly correlated; the main families are bagging, boosting, stacking and voting or averaging.

**Techniques**

| Technique | How models are trained | Combined by | Mainly reduces | Examples |
| --- | --- | --- | --- | --- |
| Bagging | Independently, in parallel, on bootstrap samples | Average or majority vote | Variance | Random forest, Extra Trees |
| Boosting | Sequentially, each fixing the previous errors | Weighted sum | Bias (and variance with shrinkage) | AdaBoost, XGBoost, LightGBM, CatBoost |
| Stacking | Different model types on the same data | A meta-model trained on their out-of-fold predictions | Both, by learning which model to trust where | Logistic regression on top of GBM, kNN and a neural net |
| Voting or averaging | Independently | Hard vote, soft vote (average probabilities) or weighted average | Variance | Blends in competitions |

**Why ensembles work**

- For $B$ models with errors of variance $\sigma^2$ and correlation $\rho$, the averaged error variance is $\rho\sigma^2 + (1-\rho)\sigma^2/B$. Diversity (low $\rho$) is what makes the gain.
- Boosting builds an additive model stage by stage, so it can represent functions no single shallow tree can.
- Different algorithms make different mistakes: a GBM and a linear model fail on different inputs.

**Stacking done correctly**

1. Produce **out-of-fold** predictions for each base model with K-fold cross-validation; never fit the meta-model on in-sample predictions or it learns to trust the most overfitted model.
2. Train a simple meta-model (regularised logistic or linear regression) on those predictions.
3. Refit the base models on all training data for inference.

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(n_estimators=500)),
                ("gbm", HistGradientBoostingClassifier())],
    final_estimator=LogisticRegression(), cv=5, stack_method="predict_proba")
```

**Production trade-offs**

- Each member adds training cost, inference latency, monitoring and retraining work. A 0.2 percent AUC gain from a five-model stack is often not worth it outside competitions.
- Soft voting needs comparable, calibrated probabilities.
- Ensembles are harder to explain; SHAP still works but per model.

**Rule of thumb:** a single well-tuned gradient boosting model is the usual production choice on tabular data; bag it with a few seeds if you need stability, and only stack when the measured gain pays for the complexity.

### 9.7 What is the difference between bagging and boosting?

Both combine many base learners, but bagging trains them independently on resampled data and averages them to cut **variance**, while boosting trains them sequentially, each one focused on what the current ensemble gets wrong, to cut **bias**.

| | Bagging (bootstrap aggregating) | Boosting |
| --- | --- | --- |
| Training | **Parallel**, independent models | **Sequential**, each model depends on the previous ones |
| Data | Each model sees a bootstrap sample (with replacement) | All data, but **reweighted** (AdaBoost) or fitted to **residuals or gradients** (gradient boosting) |
| Goal | Reduce variance | Reduce bias (and variance via shrinkage and subsampling) |
| Base learners | Strong, low-bias, high-variance models (deep trees) | Weak learners (shallow trees or stumps) |
| Combination | Simple average or majority vote | Weighted sum of learners |
| Overfitting | Robust; more trees rarely hurt | Can overfit with too many rounds; needs learning rate, early stopping, regularisation |
| Noise and outliers | Robust | Sensitive, because it keeps focusing on hard and possibly mislabelled points |
| Speed | Easily parallelised across models (see 9.44) | Sequential across rounds; parallel within a tree in XGBoost and LightGBM |
| Examples | Random forest, Extra Trees, bagged models | AdaBoost, GBM, XGBoost, LightGBM, CatBoost |

**When to use which**

- **Bagging or random forest:** strong low-tuning baseline, noisy labels, when you want stability and out-of-bag error estimates.
- **Boosting:** usually the best accuracy on tabular data when tuned carefully (learning rate, depth, number of rounds with early stopping, subsampling).

**One-liner:** bagging is many independent experts voting so their noise cancels; boosting is a team where each new member fixes the previous members' mistakes.

### 9.8 What is gradient boosting, and how does XGBoost work?

Gradient boosting builds an additive model of shallow trees one at a time, where each new tree is fitted to the **negative gradient of the loss** with respect to the current predictions. For squared error that gradient is just the residual, so each tree literally predicts "what the model still gets wrong"; for other losses (log loss, quantile, ranking) it is the generalised residual.

**The algorithm**

1. Start with a constant prediction $F_0$ (the mean, or the log-odds of the base rate).
2. For $m = 1 \ldots M$: compute pseudo-residuals $r_i = -\partial L(y_i, F(x_i)) / \partial F(x_i)$.
3. Fit a small regression tree $h_m$ to the $r_i$.
4. Update $F_m(x) = F_{m-1}(x) + \eta \thinspace h_m(x)$, with learning rate $\eta$ (shrinkage, typically 0.01 to 0.1).
5. Stop when validation loss stops improving (early stopping).

This is gradient descent in function space: each tree is a step in the direction that most reduces the loss.

**What XGBoost adds**

**Second-order objective.** XGBoost uses both the gradient $g_i$ and the Hessian $h_i$ of the loss (a Newton step) and adds a penalty on tree complexity:

$$
\text{Obj} = \sum_i \left[g_i f(x_i) + \tfrac{1}{2} h_i f(x_i)^2\right] + \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2
$$

Here $T$ is the number of leaves and $w_j$ the leaf weights. This gives a closed-form optimal leaf weight $w_j = -G_j / (H_j + \lambda)$, where $G_j$ and $H_j$ sum the gradients and Hessians in leaf $j$, and an exact gain formula for scoring each split.

**Other additions**

- **Regularisation:** L2 (`lambda`) and L1 (`alpha`) on leaf weights, and `gamma` as the minimum gain to keep a split; splits with gain below `gamma` are pruned after growing to `max_depth`.
- **Sparsity-aware splits:** missing values are sent down a learned default direction at each split.
- **Speed:** column blocks, parallel split finding within a tree, the `hist` method (binned features), GPU training, out-of-core data.
- **Row and column subsampling** (stochastic gradient boosting) for variance reduction.

**Why it is the tabular default:** it handles mixed features, missing values, non-linearities and interactions with little preprocessing, and with early stopping it is hard to beat (see 13.90). The differences between XGBoost, LightGBM and CatBoost are covered in 9.54.

**Pitfalls:** overfits with deep trees and a high learning rate; needs a proper validation set for early stopping (time-based for temporal data); cannot extrapolate beyond the training range; raw probabilities may need calibration.

### 9.9 What are the key hyperparameters for XGBoost, and how do you tune them?

Group them by what they control:

**Boosting process**

- `n_estimators` (number of rounds) and `learning_rate` / `eta` (0.01 to 0.3). They trade off: a lower learning rate needs more trees but generalises better. Use **early stopping** (`early_stopping_rounds`) on a validation set instead of fixing the tree count.

**Tree complexity (the main overfitting controls)**

- `max_depth` (typically 3 to 10): depth of each tree, i.e. the order of interactions.
- `min_child_weight`: minimum sum of instance weights (Hessian) in a leaf. Higher is more conservative.
- `gamma` / `min_split_loss`: minimum loss reduction needed to keep a split.
- `max_leaves`, `grow_policy` (`depthwise` or `lossguide`, the latter is LightGBM-style leaf-wise growth).

**Randomness (variance reduction)**

- `subsample` (row sampling per tree, 0.5 to 1.0).
- `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` (feature sampling).

**Regularisation**

- `reg_lambda` (L2 on leaf weights, default 1) and `reg_alpha` (L1 on leaf weights).

**Problem-specific**

- `objective` (e.g. `binary:logistic`, `reg:squarederror`, `rank:ndcg`) and `eval_metric`.
- `scale_pos_weight` for class imbalance (roughly negatives divided by positives); note it distorts probabilities, so recalibrate if you need them.
- `max_delta_step`: helps logistic objectives with extreme imbalance.
- `monotone_constraints`, `interaction_constraints` for domain knowledge or regulation.

**System**

- `tree_method="hist"` (fast, and `device="cuda"` for GPU), `max_bin`, `n_jobs`.

**Tuning recipe**

1. Fix a moderately high learning rate (0.1) with early stopping.
2. Tune `max_depth` and `min_child_weight`, then `gamma`.
3. Tune `subsample` and the `colsample` parameters, then the regularisation.
4. Lower the learning rate (e.g. 0.02) and let early stopping raise the number of rounds.

Use Optuna or another Bayesian search with cross-validation (see 11.5), and keep the validation split faithful to production (time-based for temporal data), or the tuned settings will not transfer.

### 9.10 How does gradient boosting differ from random forest, and what are its advantages?

Both are ensembles of decision trees, but a random forest averages many **deep, independent** trees to reduce variance, while gradient boosting adds many **shallow, sequential** trees, each correcting the current ensemble, to reduce bias.

| Aspect | Random forest | Gradient boosting |
| --- | --- | --- |
| Tree building | Parallel, independent | Sequential, each fits the gradient of the loss |
| Trees | Deep, low bias, high variance | Shallow (depth 3 to 8), high bias individually |
| Main effect | Reduces variance | Reduces bias |
| Loss function | Implicit (impurity) | Explicit and pluggable: log loss, quantile, Poisson, Tweedie, ranking, custom |
| Accuracy on tabular data | Very good | Usually best when tuned |
| Tuning | Easy, robust defaults | Sensitive: learning rate, depth, rounds, regularisation |
| Overfitting with more trees | Effectively no | Yes, needs early stopping |
| Noisy labels | Robust | More sensitive |
| Training | Embarrassingly parallel | Sequential across rounds (parallel within a tree) |
| Validation estimate | Free out-of-bag error | Needs a validation set |

**Advantages of gradient boosting**

- Directly optimises the loss you care about, including ranking and custom business losses.
- Typically a few points better than random forest on structured data.
- Supports monotonic and interaction constraints, native missing values and categorical handling (LightGBM, CatBoost).
- Smaller models at equal accuracy (hundreds of shallow trees rather than hundreds of deep ones), so often faster inference.

**When random forest is the better pick**

- You need a strong model with almost no tuning or no clean validation set.
- Labels are very noisy.
- You want a quick baseline or a stable feature-screening model.

**In practice:** start with a random forest or default LightGBM as a baseline, then tune a gradient boosting model with early stopping; the gap tells you how much tuning is worth.

### 9.11 How does logistic regression work, how does it differ from linear regression, and can it handle more than two classes?

Logistic regression models the probability of the positive class by passing a linear score through the sigmoid: $z = w^\top x + b$ and $p = \sigma(z) = 1/(1 + e^{-z})$. It is trained by maximum likelihood (minimising log loss), and a threshold on $p$ turns it into a classifier.

**How it works**

1. Linear score $z$, which is the **log-odds**: $\log \frac{p}{1-p} = w^\top x + b$.
2. Sigmoid squashes it into $(0, 1)$.
3. Loss is binary cross-entropy, $-[y \log p + (1-y)\log(1-p)]$, which is convex, so gradient methods (L-BFGS, SGD) find the global optimum.
4. Decision: predict 1 if $p \gt t$, where $t$ is 0.5 by default but should be set from business costs (see 12.21).
5. The decision boundary $w^\top x + b = 0$ is linear in the features.

**Compared with linear regression**

| Aspect | Linear regression | Logistic regression |
| --- | --- | --- |
| Target | Continuous | Binary (or categorical) |
| Output | Any real number | Probability between 0 and 1 |
| Model | $y = w^\top x + b$ | $\log\frac{p}{1-p} = w^\top x + b$ |
| Loss | Squared error (least squares) | Log loss (maximum likelihood) |
| Coefficient meaning | Change in $y$ per unit of $x$ | Change in log-odds; $e^{w_j}$ is an odds ratio |
| GLM view | Gaussian outcome, identity link | Bernoulli outcome, logit link |

Using linear regression for a 0/1 target gives predictions outside $[0, 1]$ and is pulled by extreme points, which is why the logit link is used.

**More than two classes**

- **Multinomial (softmax) regression:** one weight vector per class, $p_k = e^{z_k} / \sum_j e^{z_j}$, trained with categorical cross-entropy. This is scikit-learn's default for multiclass.
- **One-vs-rest:** $K$ binary models, pick the highest score (see 9.49).
- **Ordinal logistic regression** when classes are ordered (ratings, severity).

**Practical points**

- Regularise (L2 by default, L1 for sparsity) and scale features so the penalty is fair.
- Perfect separation makes unregularised weights diverge.
- Add interactions or splines for non-linear effects, or move to trees.
- It gives reasonably calibrated probabilities, which is why it remains the default for credit scoring and as a calibration layer (see 9.35).

### 9.12 Explain R-squared and adjusted R-squared.

**R-squared vs Adjusted R-squared**

**R-squared ($R^2$)**

- Measures the proportion of variance in the dependent variable explained by the model
- Formula: $R^2 = 1 - \frac{SS_{\text{residual}}}{SS_{\text{total}}}$
- Range: 0 to 1 (higher = better fit)
- **Problem**: Always increases (or stays same) when adding more features, even irrelevant ones

**Adjusted R-squared**

- Modifies $R^2$ to penalize adding unnecessary/irrelevant features
- Formula: $\text{Adj } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}$
  - $n$ = number of observations, $k$ = number of predictors
- Increases only if new feature improves model more than expected by chance
- Better for comparing models with different numbers of features

### 9.13 What is multicollinearity, how do you detect it, and how do you handle it in a regression model?

Multicollinearity means two or more predictors are highly correlated, so the model cannot separate their individual effects. Predictions can still be fine, but coefficients become unstable (large standard errors, signs that flip between samples) and their interpretation breaks down.

**Why it hurts:** with collinear columns $X^\top X$ is near-singular, so small changes in the data produce large swings in $\hat\beta = (X^\top X)^{-1} X^\top y$.

**How to detect it**

- **Correlation matrix or heatmap:** pairwise correlations above about 0.8. Misses relationships involving three or more variables.
- **Variance inflation factor:** $\text{VIF}_j = 1/(1 - R_j^2)$, where $R_j^2$ comes from regressing feature $j$ on all others. VIF above 5 is a warning, above 10 is serious. Tolerance is $1/\text{VIF}$.
- **Condition number** of the (scaled) design matrix: above about 30 indicates a problem; small eigenvalues show which combination is degenerate.
- **Symptoms:** coefficients with implausible signs, large standard errors while the overall F-test is significant, big coefficient changes when one feature is added or removed.

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

Xc = add_constant(X)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])], index=Xc.columns)
```

**How to handle it**

| Method | When |
| --- | --- |
| Drop one of the correlated features | Redundant measures of the same thing |
| Combine them (ratio, average, index) | They represent one concept |
| Ridge (L2) regularisation | You care about prediction and stable coefficients; spreads weight across correlated features |
| Lasso or elastic net | You want sparsity; lasso picks one arbitrarily, elastic net keeps groups |
| PCA or PLS | Many correlated features; costs interpretability |
| Centre variables | Collinearity created by polynomial or interaction terms |
| More data | Reduces standard errors, not the correlation itself |

**When it does not matter:** if you only need predictions and the correlation structure is stable in production, multicollinearity is largely harmless; tree ensembles are unaffected for accuracy, though their importance scores split between correlated features. It matters when you interpret coefficients, make causal claims or the correlation might change. See 10.36 for the feature-engineering view.

### 9.14 Explain K-Means Clustering. How does it work? Limitations?

**K-Means Clustering**

- Unsupervised algorithm that partitions data into K clusters based on feature similarity

**How it works**

1. Choose K (number of clusters), initialize K centroids (randomly or via K-Means++)
2. **Assignment step**: Assign each point to nearest centroid (Euclidean distance)
3. **Update step**: Recompute centroids as mean of points in each cluster
4. Repeat steps 2–3 until centroids stabilize (convergence) or max iterations reached
5. Objective: minimize within-cluster sum of squares (WCSS/inertia)

**Choosing K**

- Elbow method (plot WCSS vs K, find "elbow")
- Silhouette score

**Limitations**

- Must specify K in advance
- Sensitive to initial centroid placement (can converge to local optima) — mitigated by K-Means++
- Assumes spherical, equally-sized clusters — struggles with irregular shapes
- Sensitive to outliers (mean-based)
- Requires feature scaling
- Struggles with clusters of varying density/size
- Only works well with numeric data

### 9.15 How do support vector machines work, and what are support vectors?

A support vector machine finds the hyperplane that separates the classes with the **largest margin**, the widest gap between the boundary and the nearest points of each class. Those nearest points are the **support vectors**: they alone define the boundary, and every other training point could be removed without changing the model.

**Hard margin:** for labels $y_i \in \lbrace -1, 1\rbrace$, solve

$$
\min_{w,b} \tfrac{1}{2}\lVert w\rVert^2 \quad \text{subject to} \quad y_i (w^\top x_i + b) \ge 1
$$

The margin width is $2/\lVert w\rVert$, so minimising $\lVert w\rVert$ maximises it.

**Soft margin (real data overlaps):** add slack variables $\xi_i$ and a penalty $C$:

$$
\min_{w,b,\xi} \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i \quad \text{subject to} \quad y_i (w^\top x_i + b) \ge 1 - \xi_i\quad \xi_i \ge 0
$$

Equivalently, minimise hinge loss $\max(0, 1 - y_i f(x_i))$ plus L2 regularisation. Small $C$ means a wide margin that tolerates errors (more regularisation); large $C$ means a narrow margin that fits training data closely.

**Support vectors in detail**

- Points on the margin or inside it (and misclassified points) have non-zero dual coefficients $\alpha_i$; all others have $\alpha_i = 0$.
- The decision function is $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$ over support vectors only, so prediction cost scales with their number.
- Many support vectors relative to $n$ suggests heavy class overlap or too small a $C$ or $\gamma$.

**Non-linear boundaries:** the dual depends on data only through dot products, so a kernel can replace them (see 9.28 for kernels and the kernel trick).

**Strengths:** effective in high dimensions (text), robust margin-based generalisation, convex optimisation with a unique solution, memory-light at inference when support vectors are few.

**Weaknesses:** kernel SVMs scale roughly $O(n^2)$ to $O(n^3)$, so they are impractical beyond about $10^5$ rows; need feature scaling (see 9.48); outputs are scores, not probabilities (Platt scaling is needed); sensitive to $C$ and $\gamma$ (see 9.46). On large tabular data gradient boosting usually wins (see 9.23).

**Variants:** SVR (epsilon-insensitive regression), one-class SVM (novelty detection), `LinearSVC` and SGD with hinge loss for large sparse data.

### 9.16 What is the decision boundary in classifiers?

- The surface/line that separates different predicted classes in feature space
- Points on one side classified as one class, other side as another class
- **Linear boundary**: straight line/hyperplane (e.g., Logistic Regression, Linear SVM)
- **Non-linear boundary**: curved/complex surface (e.g., Kernel SVM, Decision Trees, Neural Networks)
- Determined by model parameters learned during training
- Simpler boundaries → more generalizable but may underfit; complex boundaries → fit training data closely but may overfit

### 9.17 How does Naive Bayes work, why is it called naive, and what is Bayes' theorem in this context?

Naive Bayes is a probabilistic classifier that applies Bayes' theorem and assumes the features are **conditionally independent given the class**. That assumption is "naive" because it is almost never true (words in an email are correlated), yet the classifier often ranks classes well anyway.

**Bayes' theorem**

$$
P(C \mid X) = \frac{P(X \mid C)\thinspace P(C)}{P(X)}
$$

- $P(C)$: prior, the class frequency before seeing features.
- $P(X \mid C)$: likelihood of the features given the class.
- $P(X)$: evidence, the same for every class, so it can be ignored when picking the argmax.
- $P(C \mid X)$: posterior.

**The naive step:** with features $x_1 \ldots x_d$,

$$
P(C \mid x_1, \ldots, x_d) \propto P(C)\prod_{j=1}^{d} P(x_j \mid C)
$$

so instead of estimating a joint distribution over all features (impossible with limited data) you estimate $d$ one-dimensional distributions per class. In practice you sum log-probabilities to avoid numerical underflow.

**Variants**

| Variant | Feature model | Typical use |
| --- | --- | --- |
| Multinomial | Counts (word frequencies) | Text classification, spam |
| Bernoulli | Binary presence or absence | Short texts, binary features |
| Gaussian | Continuous, normal per class | Simple numeric baselines |
| Complement | Multinomial variant for imbalanced text | Imbalanced document classes |
| Categorical | Discrete categories | Categorical tabular features |

**Laplace (additive) smoothing:** a word never seen with a class would give $P = 0$ and wipe out the whole product. Adding $\alpha$ (often 1) to every count fixes this.

**Example:** an email contains "free" and "prize". With $P(\text{spam}) = 0.3$, $P(\text{free} \mid \text{spam}) = 0.4$, $P(\text{prize} \mid \text{spam}) = 0.2$, and for ham $0.7$, $0.05$, $0.01$: spam scores $0.3 \times 0.4 \times 0.2 = 0.024$, ham scores $0.7 \times 0.05 \times 0.01 = 0.00035$, so it is classified as spam.

**Strengths:** trains in one pass over the data, handles very high dimensions and small data, supports incremental updates (`partial_fit`), a strong and cheap text baseline.

**Weaknesses**

- Correlated features are double-counted, so probabilities are pushed towards 0 and 1 and are poorly calibrated; use them for ranking or calibrate them.
- Cannot learn interactions between features.
- Gaussian NB fails on skewed or multi-modal numeric features.
- Usually beaten by logistic regression once there is enough data.

### 9.18 What is Dimensionality Reduction?

**Dimensionality Reduction**

- Technique to reduce number of input features while preserving important information/structure
- Helps combat curse of dimensionality, reduces overfitting, speeds up training, aids visualization

**Main approaches**

- **Feature Selection**: Choose subset of original features (filter, wrapper, embedded methods)
- **Feature Extraction**: Transform features into new, smaller set

**Common techniques**

- **PCA (Principal Component Analysis)**: Linear technique, projects data onto directions of max variance
- **t-SNE**: Non-linear, mainly for visualization, preserves local structure
- **UMAP**: Non-linear, faster than t-SNE, preserves both local and global structure
- **LDA (Linear Discriminant Analysis)**: Supervised, maximizes class separability
- **Autoencoders**: Neural network-based, learns compressed representation

**Benefits**

- Reduces computation cost and storage
- Removes noise/redundant features
- Helps visualization (2D/3D)
- Mitigates overfitting

### 9.19 How does PCA work, and when would you use it?

Principal component analysis finds orthogonal directions (principal components) along which the data varies most, and projects the data onto the first $k$ of them. It turns many correlated features into a few uncorrelated ones while keeping as much variance as possible.

**How it works**

1. Centre the data (and standardise if features are on different scales, otherwise large-unit features dominate).
2. Compute the covariance matrix $\Sigma = \frac{1}{n-1}X^\top X$.
3. Find its eigenvectors (directions) and eigenvalues (variance along each direction).
4. Sort by eigenvalue and keep the top $k$ eigenvectors $W_k$.
5. Project: $Z = X W_k$.

In practice libraries compute the SVD $X = U\Sigma V^\top$ directly, which is more numerically stable; the right singular vectors are the components and the squared singular values divided by $n-1$ are the variances. Randomised SVD scales to large data.

**Choosing k:** keep enough components to explain a target share of variance (say 90 to 95 percent), look for an elbow in the scree plot, or choose $k$ by the downstream model's cross-validated score.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca = make_pipeline(StandardScaler(), PCA(n_components=0.95))  # keep 95 percent of variance
Z = pca.fit_transform(X_train)
print(pca[-1].n_components_, pca[-1].explained_variance_ratio_.cumsum()[-1])
```

**When to use it**

- Many correlated numeric features (sensor channels, spectra, financial factors): compress and remove multicollinearity before a linear model.
- Before distance-based methods (KNN, k-means) to reduce noise and the curse of dimensionality.
- Visualisation in 2-D or 3-D, and as a first step before t-SNE or UMAP (see 9.43).
- Denoising and compression, and anomaly detection via reconstruction error (see 9.40).

**Limitations and pitfalls**

- Linear only; curved structure needs kernel PCA, UMAP or autoencoders (see 9.39).
- Maximum variance is not maximum predictive power: a low-variance direction can carry the signal. PCA is unsupervised; LDA or PLS use the target.
- Components are mixtures of features and hard to explain to stakeholders.
- Sensitive to scaling and outliers (robust PCA exists).
- Fit on training data only and reuse the fitted transform at inference, or you leak information.
- Rarely helps tree ensembles, which handle many features well.

### 9.20 Explain Gradient Descent and its variants.

**Gradient Descent**

- Optimization algorithm that iteratively adjusts model parameters to minimize a loss function
- Computes gradient (partial derivatives) of loss w.r.t. parameters, updates in opposite direction of gradient
- Update rule: $\theta = \theta - \alpha \nabla J(\theta)$, where $\alpha$ = learning rate

**Variants**

**Batch Gradient Descent**

- Uses entire training dataset to compute gradient per update
- Stable convergence but slow, memory-intensive for large datasets

**Stochastic Gradient Descent (SGD)**

- Uses one random sample per update
- Fast, noisy updates — can escape local minima but has high variance

**Mini-Batch Gradient Descent**

- Uses small batch (e.g., 32, 64, 128 samples) per update
- Balance between stability (batch) and speed (SGD) — most commonly used

**Advanced optimizers (build on GD)**

- **Momentum**: adds fraction of previous update to accelerate convergence, dampens oscillations
- **RMSProp**: adapts learning rate per parameter using moving average of squared gradients
- **Adam**: combines Momentum + RMSProp — adaptive learning rates, most widely used in deep learning

### 9.21 What is the ROC curve and AUC, and how are they interpreted?

The ROC curve plots the **true positive rate** against the **false positive rate** as the classification threshold moves from 1 to 0; AUC is the area under it, a single threshold-free number for how well the model ranks positives above negatives.

- $\text{TPR} = \text{TP}/(\text{TP} + \text{FN})$, also called recall or sensitivity.
- $\text{FPR} = \text{FP}/(\text{FP} + \text{TN})$, which equals 1 minus specificity (specificity is $`\text{TN}/(\text{TN} + \text{FP})`$).

**Interpretation**

- AUC equals the probability that a randomly chosen positive gets a higher score than a randomly chosen negative.
- 0.5 is random ranking, 1.0 is perfect, below 0.5 means the scores are inverted.
- It is threshold-independent and insensitive to monotonic rescaling of scores, so it says nothing about calibration.

**Limitations**

- With heavy class imbalance, a large number of true negatives keeps FPR small, so ROC-AUC can look excellent while precision is poor. Use PR-AUC when the positive class is rare and false positives are costly (see 12.26).
- It averages over thresholds you would never use; for an operating point, report precision and recall at that threshold.

See 12.9 and 12.26 for the full treatment of ROC-AUC and its limitations.

### 9.22 What assumptions does linear regression make, and what happens when they are violated?

Ordinary least squares needs very few assumptions to produce a best-fit line, but its **coefficient estimates, standard errors and p-values** are only trustworthy if the following hold. Knowing the consequence of each violation, and the fix, is what interviewers look for.

| Assumption | What it means | If violated | Diagnose | Fix |
| --- | --- | --- | --- | --- |
| Linearity | $E[y \mid x]$ is linear in the parameters | Biased predictions and coefficients | Residuals versus fitted plot shows a curve | Transform features, add polynomial or spline terms, use trees |
| Independence of errors | Residuals are not correlated with each other | Standard errors too small, false significance | Durbin-Watson, autocorrelation plot (time series), clustered data | Time-series models, clustered or HAC standard errors, mixed models |
| Homoscedasticity | Constant error variance | Coefficients still unbiased but standard errors wrong | Funnel shape in residual plot, Breusch-Pagan test | Log-transform target, weighted least squares, robust (HC) standard errors |
| Normality of errors | Residuals roughly normal | Matters for small-sample inference only; large samples rely on the central limit theorem | Q-Q plot | Transform target, bootstrap confidence intervals |
| No perfect multicollinearity | No predictor is an exact linear combination of others | Coefficients undefined or unstable | VIF, condition number (see 9.13) | Drop or combine features, ridge |
| Exogeneity | Errors uncorrelated with predictors (no omitted confounders) | Biased coefficients, wrong causal conclusions | Domain reasoning | Add confounders, instruments (see Chapter 32) |

Also check for **influential outliers** (Cook's distance, leverage): a few points can tilt the whole line.

**Residual definition:**

$$
e_i = y_i - \hat{y}_i
$$

**Practical view:** if the goal is prediction, violations mainly show up as worse validation error and you can switch models. If the goal is explanation or inference (pricing elasticity, effect of a policy), the assumptions are essential, and exogeneity is the one most often broken in business data.

### 9.23 Explain why the performance of XGBoost is better than that of SVM?

#### Why XGBoost Can Outperform SVM

There is **no universal rule** that XGBoost is better than SVM; performance depends on the dataset and problem. However, XGBoost often performs better on **tabular data** because:

| Aspect | **XGBoost** | **SVM** |
| --- | --- | --- |
| **Data type** | Excellent for structured/tabular data | Effective for smaller, well-separated datasets |
| **Nonlinearity** | Naturally captures complex nonlinear relationships | Requires appropriate kernel |
| **Feature scaling** | Generally not required | Usually required |
| **Large datasets** | Scales well | Can become computationally expensive |
| **Missing values** | Handles them effectively | Usually requires preprocessing |
| **Feature interactions** | Captures automatically | Depends on kernel |
| **Training** | Highly optimized and parallelizable | Can be expensive for large datasets |

### 9.24 What are the most widely used machine learning algorithms, and when is each the right default?

A handful of algorithm families covers most real work. The useful answer is not a list but when each is the sensible first choice.

| Algorithm | Typical use | Why choose it | Watch out for |
| --- | --- | --- | --- |
| Linear and logistic regression | Baselines, scoring, pricing, inference | Fast, interpretable, calibrated probabilities, works on sparse text | Needs feature engineering for non-linearity |
| Decision tree | Rules for humans, quick exploration | Readable if shallow | High variance |
| Random forest | Robust tabular baseline | Little tuning, out-of-bag validation | Large models, cannot extrapolate |
| Gradient boosting (XGBoost, LightGBM, CatBoost) | Most tabular prediction | Best accuracy on structured data | Needs tuning and early stopping |
| KNN | Similarity search, small baselines, imputation | No training | Slow at scale, needs scaling |
| SVM | Medium-size high-dimensional data, text | Strong margins | Does not scale to millions of rows |
| Naive Bayes | Text baseline | Very fast, works on little data | Poor probabilities |
| k-means, DBSCAN, GMM | Segmentation, clustering | Unsupervised structure | Choosing k, scaling |
| PCA, UMAP | Compression, visualisation | Reduce noise and dimensions | Interpretability |
| Neural networks | Images, text, audio, very large data | Learn representations | Data, compute and tuning cost |

**Typical progression for a new problem:** a simple baseline (majority class or linear model), then gradient boosting for tabular data or a pretrained neural network for unstructured data, and only then more complex options. See 8.47 for choosing an algorithm and 13.90 for trees versus deep learning on tabular data.

### 9.25 Which similarity measures are used in recommender systems?

Neighbourhood-based and content-based recommenders need a way to say "these two users" or "these two items" are alike. The choice of measure depends on whether the data are ratings, binary interactions or dense embeddings.

| Measure | Formula idea | Best for | Notes |
| --- | --- | --- | --- |
| Cosine similarity | $\frac{a \cdot b}{\lVert a\rVert \lVert b\rVert}$ | Sparse interaction vectors, TF-IDF, embeddings | Ignores magnitude; the default for embeddings |
| Adjusted cosine or Pearson correlation | Cosine after subtracting each user's mean rating | Explicit ratings | Removes "generous rater" bias |
| Jaccard | $\lvert A \cap B\rvert / \lvert A \cup B\rvert$ | Binary data (bought or not) | Simple, works on sets |
| Dot product | $a \cdot b$ | Learned embeddings (matrix factorisation, two-tower) | Rewards popular items with large norms |
| Euclidean or Manhattan distance | Distance in feature space | Dense item features | Needs scaling; less common for sparse data |

**Practical issues**

- **Popularity bias:** raw co-occurrence makes best-sellers similar to everything; normalise (cosine, lift, or PMI) or penalise popular items.
- **Sparsity:** two users with one shared item can look perfectly similar; apply significance weighting or shrinkage when overlap is small.
- **Scale:** all-pairs similarity is $O(n^2)$; compute item-item similarities offline in batch, or use approximate nearest neighbour search over embeddings.

Collaborative and content-based approaches that use these measures are described in 9.29; modern learned-embedding recommenders are covered in Chapter 32.

### 9.26 What is a Bayesian network, and what is it also called?

A Bayesian network (also called a **Bayes network, Bayes net, belief network** or **Bayesian belief network**; when edges are given a causal reading, a **causal Bayesian network**) is a probabilistic graphical model that represents a set of variables and their conditional dependencies as a **directed acyclic graph (DAG)**.

**Components**

- **Nodes:** random variables (disease, symptom, test result).
- **Directed edges:** direct dependencies; a node is conditionally independent of its non-descendants given its parents.
- **Conditional probability tables or distributions:** $P(X_i \mid \text{parents}(X_i))$ for each node.

The joint distribution factorises as

$$
P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{parents}(X_i))
$$

which needs far fewer parameters than the full joint table.

**Example:**

```text
   Rain      Sprinkler
      \        /
       WetGrass
```

Observing wet grass raises the probability of both rain and sprinkler; then learning it rained lowers the probability of the sprinkler ("explaining away"), even though rain and sprinkler are independent a priori.

**What you do with it**

- **Inference:** compute posteriors given evidence, exactly (variable elimination, junction tree) or approximately (sampling, belief propagation) for large graphs.
- **Parameter learning:** estimate the tables from data (maximum likelihood or Bayesian with priors).
- **Structure learning:** search for the DAG from data (score-based or constraint-based); hard and data-hungry, so expert-defined structure is common.

**Uses:** medical diagnosis, fault diagnosis, risk modelling, decision support, and as the foundation for causal graphs (see 32.1 and 32.2).

**Limitations:** structure learning is difficult, exact inference is NP-hard in general, and continuous variables need parametric assumptions or discretisation. Naive Bayes (see 9.17) is the simplest Bayesian network: a class node with every feature as an independent child.

### 9.27 What is clustering, and how do the main clustering algorithms compare?

Clustering is unsupervised learning that groups points so that members of a cluster are more similar to each other than to points in other clusters. There are no labels, so the definition of "similar" (features, scaling, distance) and the choice of algorithm determine the result.

**Families and algorithms**

| Family | Algorithm | Cluster shape | Need k? | Handles noise | Scales to |
| --- | --- | --- | --- | --- | --- |
| Partitioning | k-means, mini-batch k-means | Convex, similar size | Yes | No (outliers pull centroids) | Millions |
| Partitioning | k-medoids, k-prototypes | Convex; any distance, mixed types | Yes | Better | Thousands to millions |
| Hierarchical | Agglomerative (Ward, average, single linkage) | Depends on linkage | Cut the dendrogram | No | About $10^4$ to $10^5$ ($O(n^2)$ memory) |
| Density-based | DBSCAN, HDBSCAN, OPTICS | Arbitrary | No | Yes, labels noise | Large with indexing |
| Model-based | Gaussian mixture | Elliptical, soft assignments | Yes (BIC helps) | No | Medium |
| Graph-based | Spectral clustering | Arbitrary, connectivity-based | Yes | No | Small to medium |
| Grid-based | STING, CLIQUE | Dense grid cells | No | Yes | Large, low dimensions |

**Choosing in practice**

- Large data, roughly round clusters, need speed: **k-means** (see 9.14, 9.53).
- Unknown number of clusters, irregular shapes, noise to discard: **HDBSCAN**.
- Need probabilities or overlapping segments: **Gaussian mixtures** (see 9.52).
- Need a hierarchy or a small data set to explore: **agglomerative** with a dendrogram.
- Text or embeddings: normalise vectors and use cosine distance; reduce dimensions first (UMAP then HDBSCAN is a common pattern).

**Pitfalls**

- Clustering always returns clusters, even on random data; check stability across seeds and bootstrap samples.
- Scaling and feature choice matter more than the algorithm.
- Evaluate with silhouette or Davies-Bouldin, but above all with whether segments are actionable and stable over time (see 12.16).

**Applications:** customer segmentation, document and topic grouping, image segmentation, anomaly detection (points far from any cluster), and cold-start grouping in recommenders.

### 9.28 What is a kernel SVM, what is the kernel trick, and which kernels are common?

A kernel SVM is an SVM whose dot products are replaced by a **kernel function** $K(x, x') = \langle \phi(x), \phi(x')\rangle$, which computes an inner product in a higher (possibly infinite) dimensional feature space without ever building $\phi(x)$. A linear max-margin classifier in that space becomes a non-linear boundary in the original space. This substitution is the **kernel trick**.

**Why it works:** the SVM dual problem and its decision function use the data only through dot products:

$$
f(x) = \operatorname{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)
$$

so any valid kernel can be plugged in. A kernel is valid if it is symmetric and positive semi-definite (Mercer's condition).

**Example:** for $x \in \mathbb{R}^2$, the kernel $K(x,z) = (x \cdot z)^2$ equals the dot product of the explicit features $(x_1^2, \sqrt{2}\thinspace x_1 x_2, x_2^2)$. The RBF kernel corresponds to an infinite-dimensional feature space, which could never be computed explicitly.

**Common kernels**

| Kernel | Formula | When to use |
| --- | --- | --- |
| Linear | $x^\top x'$ | High-dimensional sparse data (text), many features; fastest |
| Polynomial | $(\gamma x^\top x' + r)^d$ | Feature interactions of degree $d$; unstable for large $d$ |
| RBF (Gaussian) | $\exp(-\gamma \lVert x-x'\rVert^2)$ | Default for non-linear problems; $\gamma$ controls locality |
| Sigmoid | $\tanh(\gamma x^\top x' + r)$ | Neural-network-like; not always PSD, rarely best |
| Others | Laplacian, chi-squared, string and graph kernels | Histograms, sequences, graphs |

**Trade-offs**

- Training needs the $n \times n$ kernel matrix: $O(n^2)$ memory and $O(n^2)$ to $O(n^3)$ time, so kernel SVMs are impractical past about $10^5$ samples. At scale use a linear SVM, approximate the kernel with random Fourier features or Nystroem and a linear model, or switch to gradient boosting or neural networks.
- $C$ and $\gamma$ interact and must be tuned together on a log grid, after scaling the features (see 9.46 and 9.48).
- Kernels are not SVM-only: kernel PCA, kernel ridge regression and Gaussian processes use the same trick.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
```

### 9.29 What are content-based and collaborative filtering, and what powers "people who bought this also bought"?

Both are classic recommender approaches. **Content-based filtering** recommends items whose attributes resemble what the user already liked; **collaborative filtering** ignores item attributes and recommends from patterns in many users' interactions.

**Content-based filtering**

- Build an item representation from attributes (genre, brand, text, image embeddings) and a user profile from the items they engaged with.
- Score candidate items by similarity to the profile (see 9.25).
- Works for new items with no interactions, and recommendations are explainable ("because you watched...").
- Over-specialises (filter bubble), needs good item features, and cannot capture quality or taste that features miss.

**Collaborative filtering**

- **User-user:** find users with similar histories and recommend what they liked.
- **Item-item:** find items co-consumed by the same users. More stable than user-user because item relationships change slowly, and similarities can be precomputed.
- **Model-based:** learn latent user and item vectors with matrix factorisation (see 9.36) or neural models.
- Finds non-obvious relationships, but suffers from cold start for new users and items, sparsity, and popularity bias.

| | Content-based | Collaborative |
| --- | --- | --- |
| Data | Item features plus the user's own history | User-item interaction matrix |
| New item | Works | Cold start |
| New user | Needs some history | Cold start |
| Serendipity | Low | Higher |
| Explainability | High | Medium ("people like you...") |

**"People who bought this also bought"** is **item-to-item collaborative filtering**, popularised by Amazon: for each item, find the items most often bought by the same customers, normalised so that best-sellers do not dominate (cosine, lift or conditional probability), precompute the top neighbours offline, and look them up at request time. A related method is association rule mining (support, confidence, lift) on baskets, which is used for "frequently bought together".

**In practice** most systems are hybrids: collaborative signals for users with history, content features for cold start, combined in a learned ranking model. Cold start and modern hybrids are covered in 32.18 and 32.14.

### 9.30 What are the two components of Bayesian logic, and how do they combine?

Bayesian reasoning combines a **prior** (what you believed before the evidence) with a **likelihood** (how probable the evidence is under each hypothesis) to produce a **posterior** (the updated belief):

$$
P(H \mid E) = \frac{P(E \mid H)\thinspace P(H)}{P(E)}
$$

- **Prior $P(H)$:** base rates, past data or expert judgement.
- **Likelihood $P(E \mid H)$:** a model of how the data arise if $H$ is true.
- **Evidence $P(E)$:** a normalising constant, the total probability of the data across all hypotheses.

**Worked example (base-rate neglect).** A disease affects 1 in 1,000 people. A test detects it 99 percent of the time and gives false positives 5 percent of the time. After a positive result:

$$
P(D \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.05 \times 0.999} \approx 0.019
$$

Only about 2 percent of positives are real, because the prior is so small. The same logic explains why fraud and anomaly alerts on rare events are mostly false alarms unless the false positive rate is tiny.

**Where it appears in ML:** Naive Bayes (see 9.17), Bayesian networks (see 9.26), regularisation as a prior (ridge is a Gaussian prior on weights), Bayesian optimisation for hyperparameters, and Bayesian A/B testing. Chapter 1 covers Bayes' theorem and Bayesian inference in more depth.

### 9.31 What is a recommender system, and what are the main types?

A recommender system ranks items (products, videos, articles, jobs) for a user in a context, using past behaviour, item attributes and context, so that the most relevant items appear first. This question gives the map; Chapter 32 covers the modern techniques in depth (32.13 to 32.24).

**Main families**

1. **Popularity and rules:** trending or best-selling items. A strong baseline and the fallback for cold start, but not personalised.
2. **Content-based filtering:** items similar to what the user liked, using item features (see 9.29).
3. **Collaborative filtering:** neighbourhood methods (user-user, item-item) and matrix factorisation (see 9.36, 32.13).
4. **Hybrid:** combine content and collaborative signals, by weighting, switching or as features in one model.
5. **Deep learning:** two-tower retrieval models and deep ranking models (32.14, 32.17).
6. **Sequential and session-based:** model the order of interactions with RNNs or Transformers (32.16).
7. **Knowledge- or constraint-based:** explicit requirements for rare, high-value purchases (houses, cars, financial products).
8. **Graph-based:** graph neural networks over user-item graphs.
9. **Context-aware and bandit-based:** use time, device and location; contextual bandits balance exploration and exploitation.
10. **LLM-based and conversational:** generate or re-rank from natural-language preferences (32.23).

**Typical production architecture**

```text
millions of items
   -> candidate generation (ANN over embeddings, item-item CF, popularity)   ~1,000 items
   -> ranking (gradient boosting or deep model on rich features)             ~100 items
   -> re-ranking (diversity, freshness, business rules, fairness)            ~10 shown
```

**Evaluation:** offline with recall@k, NDCG and MAP on time-based splits, then online A/B tests on click-through, conversion and retention (see 12.17 and 32.21).

**Common problems:** cold start (32.18), popularity and position bias (32.19), feedback loops, and optimising short-term clicks at the expense of long-term value (32.24).

### 9.32 Explain Genetic programming as it relates to Evolutionary algorithm.

#### Genetic Programming in Evolutionary Algorithms

- **Genetic Programming (GP)** is a type of **Evolutionary Algorithm (EA)** that automatically evolves **computer programs or mathematical expressions** to solve a problem.
- It is inspired by **natural evolution**, where better solutions are selected and improved over generations.
- GP represents solutions as **programs or expression trees**, rather than just numbers or fixed-length strings.

#### How Genetic Programming Works

1. **Initialize:** Create a population of random programs.
2. **Evaluate:** Calculate the **fitness** of each program.
3. **Select:** Choose the better-performing programs.
4. **Crossover:** Combine parts of two programs to create new programs.
5. **Mutation:** Make small random changes to programs.
6. **Repeat:** Continue for multiple generations until a good solution is found.

#### Example

Suppose we want to find a mathematical expression that predicts house prices.

GP might start with random expressions such as:

- $x + 5$
- $x \times 2$
- $(x + 3) \times y$

It evaluates them and evolves better expressions over generations.

### 9.33 What is an OOB error, and how is it useful?

**OOB (out-of-bag) error** is a built-in validation estimate for bagging ensembles such as **Random Forests**.

**How it works**
- Each tree is trained on a **bootstrap sample** (drawn with replacement) of the training set. About $1 - (1-1/n)^n \approx 0.632$ (63.2%) of the rows appear in each sample; the remaining **~36.8% are "out-of-bag"** for that tree.
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

### 9.34 In what scenario should a decision tree be preferred over random forest?

A single decision tree is preferable to a random forest when **interpretability, simplicity or speed** matter more than the last few points of accuracy:

- **Explainability and auditability:** a shallow tree can be printed as if-then rules and shown to regulators, doctors or business users (credit decisions, clinical triage, policy rules). A forest of 500 trees is a black box without SHAP-style tools.
- **Rules to hand off:** the tree can be converted into SQL `CASE` statements or business rules engines.
- **Low-latency or constrained environments:** a single tree is tiny and predicts in microseconds — useful on embedded devices or in very high-QPS paths.
- **Small, clean data with a simple structure:** if a depth-3 tree already captures the signal, a forest adds complexity without real gain.
- **Exploratory analysis:** quickly understanding which features and thresholds split the data.
- **Teaching or baselines.**

**Trade-off:** single trees have **high variance** — small data changes produce very different trees, and they overfit easily. Mitigate with pruning (`max_depth`, `min_samples_leaf`, cost-complexity `ccp_alpha`) and cross-validation.

**Middle ground:** train a forest or GBM for accuracy, then **distill** it into a surrogate tree for explanation, or use a shallow tree when its validated performance is within your tolerance of the ensemble's.

### 9.35 Why is Logistic Regression called regression?

Because at its core it **is a regression model — a linear regression on the log-odds** — and only becomes a classifier when you put a threshold on its output.

$$
\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \quad\Rightarrow\quad p = \sigma(\beta^T x) = \frac{1}{1+e^{-\beta^T x}}
$$

- It models a **continuous quantity**: the probability $p$ (equivalently, the log-odds, which ranges over all reals) as a linear function of the features.
- It belongs to the family of **Generalized Linear Models (GLMs)**, alongside linear regression (identity link) and Poisson regression (log link); logistic regression uses the **logit link** with a Bernoulli outcome. Statisticians call all of these "regression".
- Coefficients are interpreted like regression coefficients: a one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$.
- Classification is a separate **decision step**: predict class 1 if $p \gt t$ (e.g., 0.5, or a threshold tuned to business cost).

**Why this matters in practice:** because it outputs (reasonably calibrated) probabilities, logistic regression is used for risk scoring, ranking and expected-value decisions — not just hard labels. It is trained with maximum likelihood (log loss), not least squares.

### 9.36 What is Matrix Factorization, and where is it used in Machine Learning?

**Matrix factorization** decomposes a matrix $R$ ($m \times n$) into a product of lower-rank matrices, typically $R \approx U V^T$ with $U \in \mathbb{R}^{m\times k}$, $V \in \mathbb{R}^{n\times k}$ and $k \ll m,n$. Each row of $U$ and $V$ is a latent vector, and their dot product reconstructs an entry.

**Where it is used:**
- **Recommender systems (collaborative filtering):** user–item rating matrix → user and item embeddings; predicted rating $\hat r_{ui} = u_u^T v_i$ (+ biases). Made famous by the Netflix Prize. Trained with SGD or **ALS** (Alternating Least Squares, easy to parallelize, e.g. Spark MLlib), with L2 regularization, fitting only the observed entries.
- **Dimensionality reduction:** SVD/PCA, truncated SVD for sparse text matrices (LSA/topic modeling).
- **NMF (non-negative MF):** interpretable parts-based factors for topic modeling, audio source separation, image parts.
- **Compression:** low-rank approximation of neural network weights (the same idea behind LoRA).
- **Imputation** of missing values in matrices.

**Tradeoffs:** it is fast, scalable and gives reusable embeddings, but has a **cold-start problem** (new users/items have no vector), is linear (dot-product interactions only), and implicit feedback (clicks) needs weighted variants (e.g. implicit ALS). In production, hybrid models (factorization machines, two-tower networks) add side features to solve cold start.

### 9.37 Explain Pruning in Decision Trees and how it is done

**Pruning** reduces the size of a decision tree to stop it from memorizing noise. Fully grown trees have low bias and very high variance.

**1. Pre-pruning (early stopping)** — stop growing during construction:
- `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`
- `min_impurity_decrease`: only split if the impurity gain exceeds a threshold.

It is fast, but it can be short-sighted: a split that looks weak now may enable strong splits below it (the "horizon effect").

**2. Post-pruning** — grow the full tree, then remove subtrees that don't help:
- **Cost-complexity pruning (CART, weakest-link):** minimize $R_\alpha(T) = R(T) + \alpha \lvert T \rvert$, where $R(T)$ is the training error and $\lvert T \rvert$ the number of leaves. Increasing $\alpha$ gives a nested sequence of smaller trees. Pick $\alpha$ by cross-validation.
- **Reduced-error pruning:** replace a subtree with a leaf if accuracy on a validation set doesn't drop.
- **Pessimistic error pruning** (C4.5): uses a statistical upper bound on the error, so no validation set is needed.

```python
from sklearn.tree import DecisionTreeClassifier
path = DecisionTreeClassifier().cost_complexity_pruning_path(X_tr, y_tr)
# grid-search ccp_alpha over path.ccp_alphas with CV
clf = DecisionTreeClassifier(ccp_alpha=best_alpha).fit(X_tr, y_tr)
```

**Practical note:** inside ensembles, random forests usually use unpruned trees (bagging handles the variance), while gradient boosting uses shallow, pre-pruned trees. Pruning matters most when you need a single interpretable tree.

### 9.38 What is the difference between Gini Impurity and Entropy? Which one is better and why?

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

### 9.39 Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?

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

### 9.40 Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?

**Generally only approximately, and for some methods not at all.** Reducing dimensions throws information away, so an exact inverse is impossible unless the dropped components carried zero variance.

- **PCA:** yes, approximately. $X_{\text{rec}} = Z W^T + \mu$, where $W$ holds the kept components. The **reconstruction error** equals the variance in the discarded components. Keeping 95% of the variance means losing about 5%. This is used for denoising and compression (`pca.inverse_transform`).
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

### 9.41 Why do we always need the intercept term in a regression model?

The intercept $\beta_0$ is the predicted value when all features are 0. Without it, the regression line or hyperplane is **forced through the origin**, which is almost never justified.

**Why we keep it:**
- **Unbiased fit:** if the true relationship has a non-zero offset, dropping the intercept biases *every* slope coefficient to compensate, and the predictions become systematically wrong.
- **Zero-mean residuals:** with an intercept, OLS guarantees the residuals sum to zero and the fit passes through $(\bar x, \bar y)$.
- **Valid statistics:** $R^2$ and the usual F-tests assume an intercept. Without one, $R^2$ is computed against 0 instead of $\bar y$ and can be misleadingly high, or even negative in some software.
- **Categorical encoding:** with dummy variables, the intercept represents the baseline category. This is why you drop one dummy, to avoid the dummy-variable trap.
- In logistic regression, the intercept captures the base-rate log-odds.

**When you might drop it (rarely):**
- Physics or domain knowledge guarantees $y = 0$ when $x = 0$ (e.g. $\text{distance} = \text{speed} \times \text{time}$), and even then test it.
- All features and the target are mean-centered: the intercept is then 0 anyway.
- You include a full set of one-hot dummies (the intercept is then redundant).

Interpretation caveat: the intercept itself often has no practical meaning (age = 0, house size = 0). Centering features makes it the "average case" prediction.

### 9.42 Why is Ridge Regression called Ridge?

There are two complementary explanations.

**1. The "ridge" added to the matrix (the original, historical reason).** The OLS solution $\hat\beta = (X^TX)^{-1}X^Ty$ becomes unstable when features are collinear, because $X^TX$ is near-singular. Hoerl and Kennard (1970) proposed

$$
\hat\beta_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty
$$

Adding $\lambda I$ puts a constant "ridge" along the diagonal of $X^TX$. This lifts every eigenvalue by $\lambda$, makes the matrix invertible and well-conditioned, and stabilizes the coefficients. The name also echoes Hoerl's earlier "ridge analysis" of response surfaces, where the path of optimal estimates traces a ridge.

**2. The ridge in the likelihood surface.** With multicollinearity, the loss surface has a long flat valley or ridge: many coefficient combinations fit almost equally well, so the estimates have huge variance. The L2 penalty $\lambda\lVert\beta\rVert_2^2$ adds curvature, picks the smallest-norm solution along that ridge, and shrinks the coefficients.

**Why it matters:** ridge trades a little bias for a large reduction in variance, keeps all features (no sparsity, unlike Lasso), and is the go-to when predictors are highly correlated. Equivalent views: L2 regularization, a Gaussian prior on the weights (MAP), and weight decay in neural networks.

### 9.43 Does it make any sense to chain two different dimensionality reduction algorithms?

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

### 9.44 Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?

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

### 9.45 If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?

**No.** Decision trees are **invariant to monotonic feature scaling**. A split asks "is $x_j \le t$?", and scaling $x_j$ (standardization, min-max, even a log transform) just moves the threshold $t$. The ordering of samples, and therefore the partitions and impurity reductions, stays exactly the same. Scaling won't change the tree at all, apart from floating-point ties.

**What actually fixes an underfitting tree** (high bias):
- **Loosen the regularization:** increase `max_depth` and `max_leaf_nodes`, decrease `min_samples_leaf`, `min_samples_split` and `min_impurity_decrease`, reduce `ccp_alpha`.
- **Better features:** trees struggle with diagonal or additive relationships, because they need many axis-aligned splits. Engineered features like ratios, differences, interactions and aggregates help a lot.
- **Switch to ensembles:** gradient boosting reduces bias by adding trees sequentially. Random forests mostly reduce variance, so they help less with pure underfitting.
- Check data quality and label noise. Sometimes the signal just isn't in the features.

**When scaling does matter:** distance- or gradient-based models such as KNN, SVM, linear and logistic regression with regularization, neural networks, PCA and k-means. In a mixed pipeline (e.g. trees stacked with logistic regression) you scale for the other models, not for the tree.

### 9.46 Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?

**Increase $\gamma$ and increase $C$.** Both reduce regularization and make the model more complex.

**$\gamma$ (gamma)** in the RBF kernel $K(x,x') = \exp(-\gamma\lVert x - x'\rVert^2)$ controls how far a single training example's influence reaches.
- **Small $\gamma$:** wide, smooth Gaussians, a nearly linear and very smooth boundary, which can underfit.
- **Large $\gamma$:** narrow bells, so each point influences only its neighbourhood and the boundary becomes wiggly; too large overfits.
- Underfitting → **increase $\gamma$**.

**C** is the inverse regularization strength: the penalty for margin violations in the soft-margin objective $\frac{1}{2}\lVert w\rVert^2 + C\sum_i \xi_i$.
- **Small C:** wide margin, tolerates misclassifications, strong regularization, can underfit.
- **Large C:** penalizes errors heavily, narrower margin, fits the training data more closely.
- Underfitting → **increase C**.

**In practice:** tune them jointly on a log grid (e.g. $C \in \lbrace 0.1, 1, 10, 100\rbrace$, $\gamma \in \lbrace 10^{-3}, \dots, 10\rbrace$) with cross-validation, because they interact. Also make sure the features are **scaled**: unscaled features make $\gamma$ meaningless and a common "underfit" cause. If the model still underfits after that, the features may lack signal.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [1e-3, 1e-2, 0.1, 1]}, cv=5)
```

### 9.47 How do we interpret weights in linear models?

In a linear model $\hat y = \beta_0 + \sum_j \beta_j x_j$, the weight $\beta_j$ is **the expected change in the prediction for a one-unit increase in $x_j$, holding all other features constant**.

**Variants:**
- **Logistic regression:** $\beta_j$ is the change in **log-odds**. $e^{\beta_j}$ is the **odds ratio**: e.g. $\beta = 0.7$ means each unit multiplies the odds by about 2.
- **Log-transformed target** ($\log y$): a unit change in $x_j$ changes $y$ by about $100\cdot\beta_j$ percent (exactly $100(e^{\beta_j} - 1)$ percent).
- **Log-log:** $\beta_j$ is an **elasticity**: a 1% change in $x_j$ gives a $\beta_j$ percent change in $y$.
- **Dummy variable:** the difference from the baseline category.
- **Interaction terms:** the effect of $x_j$ depends on the value of the interacting feature.

**Caveats (where interviews dig):**
- **Scale:** raw weights aren't comparable across features in different units. Standardize the features to compare magnitudes (the effect of a 1-SD change).
- **Multicollinearity:** correlated features make weights unstable, so signs can flip and the "holding others constant" reading becomes unrealistic. Check VIF.
- **Regularization** shrinks weights (Ridge) or zeroes them (Lasso), so the magnitudes are biased.
- **Statistical significance:** look at standard errors, confidence intervals and p-values, not just point estimates.
- **Correlation ≠ causation:** weights describe associations conditional on the included features. Omitted-variable bias can distort them.

For complex models, SHAP or partial dependence give comparable per-feature effect interpretations.

### 9.48 Why is it important to scale the inputs when using SVMs?

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

### 9.49 What is OvR and OvO for multiclass classification, and which machine learning algorithms use them?

These are two strategies for turning **binary classifiers into multiclass classifiers**.

**One-vs-Rest (OvR, also called One-vs-All):**
- Train $K$ binary classifiers, each separating one class from all others.
- Predict the class whose classifier gives the highest score or probability.
- **Pros:** only $K$ models, simple. **Cons:** each model trains on the full dataset, it creates artificial class imbalance (1 vs $K-1$), and the scores from different models may not be calibrated against each other.

**One-vs-One (OvO):**
- Train a classifier for every pair of classes: $K(K-1)/2$ models.
- Predict by majority voting (ties broken by confidence).
- **Pros:** each model trains only on two classes' data (small and balanced), which suits algorithms that scale badly with $n$. **Cons:** the number of models grows quadratically (100 classes give 4,950 models).

**Which algorithms use which:**
- **SVMs** are inherently binary. scikit-learn's `SVC` uses **OvO** internally (kernel SVMs scale superlinearly in $n$, so small pairwise problems are cheaper). `LinearSVC` uses **OvR**.
- **Logistic regression** can do OvR, or native multinomial (softmax), which is scikit-learn's default now.
- **Perceptron, SGDClassifier:** OvR.
- **Natively multiclass, no wrapper needed:** decision trees, random forests, gradient boosting (one tree per class per round, softmax), KNN, Naive Bayes, neural networks (softmax output).

**Multilabel is different:** when one sample can have several labels at once (an image tagged "dog", "outdoor" and "beach"), train one independent binary classifier per label (binary relevance, `MultiOutputClassifier`) or a classifier chain that feeds earlier label predictions into later ones. OvR picks one winner; multilabel thresholds each label separately.

```python
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
OneVsRestClassifier(LogisticRegression()).fit(X, y)
OneVsOneClassifier(SVC()).fit(X, y)
```

### 9.50 What are some advantages and Disadvantages of regression models and tree-based models?

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

### 9.51 List some of the drawbacks of a Linear model

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

### 9.52 What is the difference between a Gaussian Mixture Model and the K-Means Algorithm?

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

### 9.53 How to select K for K-means?

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

### 9.54 How do XGBoost, LightGBM and CatBoost differ internally, and how do you choose between them?

All three implement gradient-boosted decision trees with second-order (gradient and Hessian) leaf estimates and similar regularisation, so on most problems their tuned accuracy is close. They differ in **how trees are grown, how splits are found and how categorical features are handled**, which drives speed, memory and behaviour on small or categorical-heavy data.

**Side-by-side**

| Aspect | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- |
| Tree growth | Level-wise (depth-wise) by default; `lossguide` optional | **Leaf-wise** (best-first): always split the leaf with the largest gain | **Symmetric (oblivious) trees**: the same split condition across a whole level |
| Split finding | Exact greedy or **histogram** (`hist`, the modern default) | Histogram-based, plus GOSS and EFB | Histogram-based on quantised features |
| Categorical features | Native support with `hist` (newer versions); otherwise encode | Native: sorts categories by gradient statistics to find a good partition | **Ordered target statistics**: target encoding computed on a random permutation using only "earlier" rows |
| Leakage control | Standard | Standard | **Ordered boosting** to avoid prediction shift |
| Missing values | Learned default direction | Learned default direction | Treated as min or max by default |
| Speed | Fast, strong GPU support | Usually fastest on CPU for large data | Slower to train on CPU, fast GPU, very fast inference |
| Defaults | Need some tuning | Need care with `num_leaves` on small data | Strong out of the box |

**Key mechanisms explained**

- **Histogram splits:** each feature is bucketed into up to `max_bin` bins (255 by default) once. Split search then scans bins instead of sorted values, cutting cost from roughly $O(n)$ to $O(\text{bins})$ per feature and node, and the histogram of one child can be obtained by subtracting the sibling's from the parent's.
- **Leaf-wise growth:** reaches lower training loss with the same number of leaves, but grows deep, unbalanced trees that overfit small data sets. Control it with `num_leaves` (keep it well below $2^{\text{max depth}}$), `min_data_in_leaf` and `max_depth`.
- **GOSS (gradient-based one-side sampling):** keep all rows with large gradients, randomly sample the rows with small gradients and up-weight them, so the split statistics stay roughly unbiased while using fewer rows.
- **EFB (exclusive feature bundling):** sparse features that are rarely non-zero together (for example one-hot columns) are bundled into one feature, reducing the effective feature count.
- **Ordered boosting:** in standard boosting, the gradient for a row is computed by a model that was trained on that same row, so residuals on training data are biased compared with unseen data. CatBoost maintains models trained on prefixes of a random permutation and computes each row's residual from a model that has not seen it. The same idea, applied to target encoding, prevents the leakage that naive mean encoding causes (see 10.6).
- **Oblivious trees:** every node at a given depth uses the same feature and threshold, so a tree is a lookup table indexed by $d$ binary tests. This acts as regularisation and makes inference very fast and branch-free.

**How to choose**

- **Many high-cardinality categorical features, small to medium data, little tuning time:** CatBoost.
- **Very large data (tens of millions of rows), training speed and memory matter:** LightGBM.
- **Mature ecosystem, distributed training (Spark, Dask), GPU, broad deployment tooling:** XGBoost.
- **Inside scikit-learn without extra dependencies:** `HistGradientBoostingClassifier`, which follows the LightGBM design.

**Practical guidance**

- Benchmark two of them with the same validation split and early stopping; differences are often within noise, so pick on operational grounds (serving format, latency, team familiarity).
- Leaf-wise growth plus a high `num_leaves` on 10,000 rows is a common cause of overfitting in LightGBM.
- Native categorical handling can overfit rare categories; set minimum counts or smoothing.
- Check inference format: all three export to ONNX or have native C APIs, and Treelite can compile XGBoost and LightGBM models for low-latency serving.

**What interviewers look for:** that you know the libraries are close in accuracy, can explain histogram splits, leaf-wise growth and ordered boosting in a sentence each, and choose based on data shape and operations rather than habit.

### 9.55 How do you interpret tree ensembles with feature importance and SHAP, and what are the pitfalls?

Tree ensembles offer several importance measures, and they answer different questions. Impurity-based importance is cheap but biased, permutation importance measures dependence of the score on a feature, and SHAP gives additive per-prediction attributions. All of them explain **the model**, not the real world, and all of them struggle with correlated features.

**The main tools**

| Method | What it measures | Scope | Main weakness |
| --- | --- | --- | --- |
| Impurity (gain) importance | Total loss reduction from splits on the feature | Global | Biased towards continuous and high-cardinality features; computed on training data, so it rewards overfitting |
| Split count | How often a feature is used | Global | Ignores how useful each split is |
| Permutation importance | Drop in validation score when the feature is shuffled | Global | Shuffling correlated features creates unrealistic rows; correlated features share and hide importance |
| SHAP (TreeSHAP) | Shapley-value contribution of each feature to one prediction relative to a baseline | Local and global | Credit splitting under correlation; depends on baseline and output scale |
| Partial dependence and ICE | Average and per-row prediction as one feature varies | Global and local | Misleading when features are correlated |
| ALE plots | Local effect within realistic ranges | Global | Harder to explain to stakeholders |

**How SHAP works for trees**

- Shapley values from cooperative game theory distribute the difference between a prediction and the average prediction among the features, satisfying additivity: $f(x) = \phi_0 + \sum_j \phi_j$.
- TreeSHAP computes them exactly for tree ensembles in polynomial time rather than over all $2^d$ feature subsets.
- Global importance is the mean absolute SHAP value; beeswarm plots show direction and spread; SHAP interaction values separate pairwise effects.

```python
import shap
explainer = shap.TreeExplainer(model)            # exact for tree models
sv = explainer(X_valid)                          # explain held-out data, not training data
shap.plots.beeswarm(sv)                          # global view with direction
shap.plots.waterfall(sv[0])                      # reason codes for one prediction
```

**Pitfalls**

1. **Correlated features:** with two near-duplicate features the model may use either, and credit is split arbitrarily between them. Neither looks important on its own, and dropping one changes the other's importance. Group correlated features before interpreting, or compute grouped permutation importance.
2. **Path-dependent versus interventional SHAP:** the default tree path method respects the data's correlations as seen in the trees; the interventional method (with a background data set) breaks them. They can give different answers; choose deliberately and state which you used.
3. **Output scale:** for classifiers SHAP values are usually in log-odds, not probability; translating them to "percentage points" in a business deck is wrong.
4. **Not causal:** a high SHAP value for "number of support calls" in a churn model does not mean reducing calls reduces churn. For intervention questions use causal methods (see Chapter 32).
5. **Leakage shows up as importance:** a feature with suspiciously dominant importance is often a leak (see 12.33).
6. **Instability:** importances can change a lot between retrains or seeds; look at rankings across bootstraps before acting.
7. **Training versus held-out data:** compute importance on validation data; training-set importance reflects memorisation.
8. **Explanations can be gamed or misleading** for adversarial or out-of-distribution inputs; they are approximations, not proofs.

**When explanations must be reliable** (credit, insurance, healthcare), prefer models that are interpretable by construction or constrained: monotonic constraints in gradient boosting, explainable boosting machines or GAMs, or a scorecard. Use SHAP for reason codes, but validate that they agree with domain knowledge (see 30.44).

**What interviewers look for:** knowing why default `feature_importances_` can mislead, what SHAP does and does not tell you, and how you would explain a decision to a regulator.

### 9.56 Which anomaly detection algorithms are there, how do they compare, and how do you choose one?

Anomaly detection finds observations that do not fit the normal pattern. Because labelled anomalies are rare, most methods are unsupervised or semi-supervised: they model "normal" and score how far each point is from it. The right algorithm depends on the data type (tabular, time series, high-dimensional), whether any labels exist and whether you need explanations. See 8.23 for the basic definition.

**Algorithm families**

| Method | Idea | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Robust statistics (z-score, median absolute deviation, IQR) | Distance from a robust centre per feature | Simple, explainable, fast | Univariate; misses unusual combinations |
| Isolation Forest | Random splits isolate anomalies in fewer steps; score is average path length | Fast, scales, handles many features, little tuning | Struggles with local anomalies in dense regions; axis-aligned splits |
| Local Outlier Factor (LOF) | Compares a point's local density with its neighbours' | Finds local anomalies between clusters of different density | $O(n^2)$ without indexing; sensitive to k and scaling |
| k-NN distance | Distance to the k-th nearest neighbour | Simple, strong baseline | Slow on large data; curse of dimensionality |
| One-class SVM | Learns a boundary around normal data | Works with kernels on medium data | Hard to tune, does not scale, sensitive to scaling |
| Gaussian mixture or density models | Low likelihood under a fitted density | Probabilistic score | Assumes a distributional shape |
| PCA or autoencoder reconstruction | Points the model cannot reconstruct are unusual | Handles correlated, high-dimensional data, images, sensors | Can learn to reconstruct anomalies too; threshold needs care |
| Forecast residuals (time series) | Large deviation from a seasonal forecast | Handles trend and seasonality | Needs a good forecaster; level shifts confuse it |
| Supervised classifier | Trained on labelled fraud or failures | Most accurate when labels exist | Labels are scarce, delayed and biased; misses new patterns |

**How to choose**

- **Tabular data, no labels, need a fast first model:** Isolation Forest, compared with a robust-statistics baseline.
- **Clusters of different density:** LOF or k-NN distance.
- **High-dimensional sensors, images or logs:** autoencoder or PCA reconstruction error.
- **Time series:** decompose or forecast, then score residuals (see 32.34).
- **Some labels available:** use them to choose and threshold the unsupervised score, then move to a supervised or semi-supervised model as labels accumulate; combine both, since supervised models catch known patterns and unsupervised ones catch new ones.

**Evaluation and thresholds**

- With few labels, use precision at k (the top alerts analysts can review) and PR-AUC, not accuracy or ROC-AUC.
- The threshold is a capacity decision: if analysts can review 200 alerts a day, set the threshold to produce about 200. The `contamination` parameter in scikit-learn only sets this cut-off, it does not improve the model.
- Base rates make most alerts false positives even for good models (see 9.30), so measure alert precision explicitly.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(X_train)
scores = -iso.score_samples(X_new)               # higher means more anomalous
alerts = X_new[scores >= np.quantile(scores, 0.995)]   # top 0.5 percent for review
```

**Production pitfalls**

- "Normal" drifts (seasonality, product launches), so retrain or use rolling baselines.
- Unscaled features dominate distance-based methods.
- An anomaly score without an explanation is hard to act on; attach the top contributing features.
- Feedback from analysts is the most valuable label source; capture it.

System-level designs are in 28.20, 28.23 and 29.40.

### 9.57 How do you approach a new tabular prediction problem in practice, and when is gradient boosting not the right choice?

For most structured, tabular problems the winning recipe is a careful validation design, a simple baseline, then a tuned gradient boosting model with good features. Most of the real gains come from the validation split, leakage control and features, not from the algorithm choice. This question is the hands-on workflow; see 8.47 for general algorithm selection and 13.90 for trees versus deep learning.

**Workflow**

1. **Frame the decision:** what action the prediction drives, the cost of each error type, and the metric that reflects it (see 12.11).
2. **Design validation to mirror production:** time-based splits when the future is predicted from the past, grouped splits when the same customer or patient appears many times. A random split on such data gives optimistic results that disappear in production.
3. **Check for leakage:** features that are only known after the prediction time, IDs that encode the label, aggregates computed over the whole data set (see 8.39 and 10.30).
4. **Baselines:** majority class or last value, then a regularised linear model. If gradient boosting cannot beat it clearly, the problem is the data.
5. **Gradient boosting with defaults** (LightGBM, XGBoost or CatBoost) and early stopping.
6. **Feature engineering:** aggregates over time windows, ratios, recency and frequency, domain features. This usually beats hyperparameter tuning.
7. **Tune** a few key parameters with Bayesian search (see 9.9), keeping the same validation design.
8. **Calibrate** if decisions use probabilities, choose thresholds from business costs, check performance on important segments.
9. **Explain and constrain:** SHAP reason codes, monotonic constraints where the domain requires them.
10. **Productionise:** same feature code for training and serving (see 10.34), monitoring for drift and performance.

**When something other than gradient boosting is better**

| Situation | Better choice | Why |
| --- | --- | --- |
| Very small data (hundreds of rows) | Regularised linear model, or a tabular foundation model such as TabPFN | Trees overfit; strong priors help |
| Need to extrapolate beyond the training range (trends, prices) | Linear model, or linear trend plus boosting on residuals | Trees predict a constant outside the seen range |
| Very wide sparse data (text, millions of one-hot columns) | Linear model or factorisation machine | Trees split on one sparse feature at a time |
| Regulation demands a transparent model | Scorecard, GAM or explainable boosting machine | Inspectable by construction |
| Rows also contain text, images or long sequences | Neural network, or embeddings fed into boosting | Representation learning |
| Very large data with many categorical IDs and online updates | Deep models with embeddings (as in recommenders) | Scale and continual training |
| Strict latency on tiny devices | Small tree or linear model | Hundreds of trees cost memory and time |

**Common mistakes**

- Tuning heavily on a validation split that does not match production.
- Treating a 0.002 AUC gain as real without checking variance across folds (see 12.34).
- Ignoring calibration and thresholds, so a better AUC gives no business gain.
- Skipping the linear baseline, which hides that the signal is weak.

**What interviewers look for:** a validation-first mindset, a default you can justify (gradient boosting), and the judgement to recognise when the default is wrong.

---

← [Chapter 8. Machine Learning Fundamentals](08-machine-learning-fundamentals.md) · [Table of contents](../README.md) · [Chapter 10. Data Preprocessing and Feature Engineering](10-data-preprocessing-and-feature-engineering.md) →
