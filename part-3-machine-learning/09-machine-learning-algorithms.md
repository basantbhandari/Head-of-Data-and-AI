# Chapter 9. Machine Learning Algorithms

_Part III — Machine Learning_

Linear models, trees and ensembles, SVMs, Bayesian methods, clustering, dimensionality reduction and recommender systems: how each works, when to use it and how it fails.

This chapter contains **81 questions**.

## Contents

- **9.1** [What is the KNN (K-Nearest Neighbors) classifier?](#91-what-is-the-knn-k-nearest-neighbors-classifier)
- **9.2** [Is KNN prone to overfitting or underfitting?](#92-is-knn-prone-to-overfitting-or-underfitting)
- **9.3** [What is the difference between k-NN and k-means?](#93-what-is-the-difference-between-k-nn-and-k-means)
- **9.4** [How does a Decision Tree algorithm work?](#94-how-does-a-decision-tree-algorithm-work)
- **9.5** [Explain how Decision Trees make splits and handle categorical features.](#95-explain-how-decision-trees-make-splits-and-handle-categorical-features)
- **9.6** [How does the Random Forest algorithm work? How does it improve over Decision Trees? How does it reduce variance?](#96-how-does-the-random-forest-algorithm-work-how-does-it-improve-over-decision-trees-how-does-it-reduce-variance)
- **9.7** [Explain Ensemble Methods. Why are they powerful?](#97-explain-ensemble-methods-why-are-they-powerful)
- **9.8** [What is the difference between bagging and boosting?](#98-what-is-the-difference-between-bagging-and-boosting)
- **9.9** [What is Gradient Boosting? How does XGBoost work?](#99-what-is-gradient-boosting-how-does-xgboost-work)
- **9.10** [What are the key hyperparameters for XGBoost?](#910-what-are-the-key-hyperparameters-for-xgboost)
- **9.11** [Explain Gradient Boosting and its advantages over Random Forests.](#911-explain-gradient-boosting-and-its-advantages-over-random-forests)
- **9.12** [Explain how Logistic Regression differs from Linear Regression.](#912-explain-how-logistic-regression-differs-from-linear-regression)
- **9.13** [How does logistic regression work?](#913-how-does-logistic-regression-work)
- **9.14** [Explain R-squared and adjusted R-squared.](#914-explain-r-squared-and-adjusted-r-squared)
- **9.15** [How do you check for multicollinearity in regression models?](#915-how-do-you-check-for-multicollinearity-in-regression-models)
- **9.16** [How does K-Nearest Neighbors (KNN) work?](#916-how-does-k-nearest-neighbors-knn-work)
- **9.17** [Explain K-Means Clustering. How does it work? Limitations?](#917-explain-k-means-clustering-how-does-it-work-limitations)
- **9.18** [Explain Support Vector Machines (SVM). What is the kernel trick?](#918-explain-support-vector-machines-svm-what-is-the-kernel-trick)
- **9.19** [What is the decision boundary in classifiers?](#919-what-is-the-decision-boundary-in-classifiers)
- **9.20** [Explain Naive Bayes.](#920-explain-naive-bayes)
- **9.21** [What is Dimensionality Reduction?](#921-what-is-dimensionality-reduction)
- **9.22** [Explain PCA (Principal Component Analysis). How does it work? When would you use it?](#922-explain-pca-principal-component-analysis-how-does-it-work-when-would-you-use-it)
- **9.23** [Explain Gradient Descent and its variants.](#923-explain-gradient-descent-and-its-variants)
- **9.24** [What is the ROC-AUC curve, and how is it interpreted?](#924-what-is-the-roc-auc-curve-and-how-is-it-interpreted)
- **9.25** [Explain Decision Tree Classification.](#925-explain-decision-tree-classification)
- **9.26** [To start Linear Regression, you would need to make some assumptions. What are those assumptions?](#926-to-start-linear-regression-you-would-need-to-make-some-assumptions-what-are-those-assumptions)
- **9.27** [What is multicollinearity and how will you handle it in your regression model?](#927-what-is-multicollinearity-and-how-will-you-handle-it-in-your-regression-model)
- **9.28** [Explain why the performance of XGBoost is better than that of SVM?](#928-explain-why-the-performance-of-xgboost-is-better-than-that-of-svm)
- **9.29** [What are five popular algorithms used in Machine Learning?](#929-what-are-five-popular-algorithms-used-in-machine-learning)
- **9.30** [List the differences between KNN and k-means clustering.](#930-list-the-differences-between-knn-and-k-means-clustering)
- **9.31** [What is a PCA?](#931-what-is-a-pca)
- **9.32** [What are support vectors in SVM (Support Vector Machine)?](#932-what-are-support-vectors-in-svm-support-vector-machine)
- **9.33** [What techniques are used to find resemblance in the recommendation system?](#933-what-techniques-are-used-to-find-resemblance-in-the-recommendation-system)
- **9.34** [What is another name for a Bayesian Network?](#934-what-is-another-name-for-a-bayesian-network)
- **9.35** [What is ensemble learning?](#935-what-is-ensemble-learning)
- **9.36** [What is clustering?](#936-what-is-clustering)
- **9.37** [What is the Bayesian Network?](#937-what-is-the-bayesian-network)
- **9.38** [What is dimension reduction in ML?](#938-what-is-dimension-reduction-in-ml)
- **9.39** [Why is the Naive Bayes Method ‘Naive’?](#939-why-is-the-naive-bayes-method-naive)
- **9.40** [State the Bayes’ theorem for Naive Bayes Method.](#940-state-the-bayes-theorem-for-naive-bayes-method)
- **9.41** [Explain Kernel SVM.](#941-explain-kernel-svm)
- **9.42** [What are some types of clustering?](#942-what-are-some-types-of-clustering)
- **9.43** [What is content-based filtering and collaborative filtering?](#943-what-is-content-based-filtering-and-collaborative-filtering)
- **9.44** [What are the two components of Bayesian logic?](#944-what-are-the-two-components-of-bayesian-logic)
- **9.45** [How would you describe a Recommender system?](#945-how-would-you-describe-a-recommender-system)
- **9.46** [Do you consider decision trees to have any advantages or disadvantages? If yes, what are they?](#946-do-you-consider-decision-trees-to-have-any-advantages-or-disadvantages-if-yes-what-are-they)
- **9.47** [What is a random forest, and how does it work?](#947-what-is-a-random-forest-and-how-does-it-work)
- **9.48** [What ensemble techniques can be used to aggregate multiple models?](#948-what-ensemble-techniques-can-be-used-to-aggregate-multiple-models)
- **9.49** [Do you think logistic regression can be used for more than two classes?](#949-do-you-think-logistic-regression-can-be-used-for-more-than-two-classes)
- **9.50** [Explain Genetic programming as it relates to Evolutionary algorithm.](#950-explain-genetic-programming-as-it-relates-to-evolutionary-algorithm)
- **9.51** [What are some classification methods that SVM can handle?](#951-what-are-some-classification-methods-that-svm-can-handle)
- **9.52** [Why do you think instance-based learning algorithm is sometimes referred to as Lazy learning algorithm?](#952-why-do-you-think-instance-based-learning-algorithm-is-sometimes-referred-to-as-lazy-learning-algorithm)
- **9.53** [Explain the reason for pruning of a decision tree.](#953-explain-the-reason-for-pruning-of-a-decision-tree)
- **9.54** [What differentiates Gradient boosting and Random forest?](#954-what-differentiates-gradient-boosting-and-random-forest)
- **9.55** [When recommendations like “....people who did this also bought this” comes up on a website. What sort of algorithm propels that?](#955-when-recommendations-like-people-who-did-this-also-bought-this-comes-up-on-a-website-what-sort-of-algorithm-propels-that)
- **9.56** [What is an OOB error, and how is it useful?](#956-what-is-an-oob-error-and-how-is-it-useful)
- **9.57** [In what scenario should a decision tree be preferred over random forest?](#957-in-what-scenario-should-a-decision-tree-be-preferred-over-random-forest)
- **9.58** [Why is Logistic Regression called regression?](#958-why-is-logistic-regression-called-regression)
- **9.59** [What are the main points of difference between Bagging and Boosting?](#959-what-are-the-main-points-of-difference-between-bagging-and-boosting)
- **9.60** [What is Matrix Factorization, and where is it used in Machine Learning?](#960-what-is-matrix-factorization-and-where-is-it-used-in-machine-learning)
- **9.61** [Explain Pruning in Decision Trees and how it is done](#961-explain-pruning-in-decision-trees-and-how-it-is-done)
- **9.62** [What are kernels in SVM? Can you list some popular SVM kernels?](#962-what-are-kernels-in-svm-can-you-list-some-popular-svm-kernels)
- **9.63** [What is the difference between Gini Impurity and Entropy? Which one is better and why?](#963-what-is-the-difference-between-gini-impurity-and-entropy-which-one-is-better-and-why)
- **9.64** [Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?](#964-can-pca-be-used-to-reduce-the-dimensionality-of-a-highly-nonlinear-dataset)
- **9.65** [Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?](#965-once-a-datasets-dimensionality-has-been-reduced-is-it-possible-to-reverse-the-operation-if-so-how-if-not-why)
- **9.66** [Why do we always need the intercept term in a regression model?](#966-why-do-we-always-need-the-intercept-term-in-a-regression-model)
- **9.67** [Why is Ridge Regression called Ridge?](#967-why-is-ridge-regression-called-ridge)
- **9.68** [Does it make any sense to chain two different dimensionality reduction algorithms?](#968-does-it-make-any-sense-to-chain-two-different-dimensionality-reduction-algorithms)
- **9.69** [Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?](#969-is-it-possible-to-speed-up-training-of-a-bagging-ensemble-by-distributing-it-across-multiple-servers)
- **9.70** [If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?](#970-if-a-decision-tree-is-underfitting-the-training-set-is-it-a-good-idea-to-try-scaling-the-input-features)
- **9.71** [Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?](#971-say-you-trained-an-svm-classifier-with-an-rbf-kernel-it-seems-to-underfit-the-training-set-should-you-increase-or-decrease-γ-gamma-what-about-c)
- **9.72** [How do we interpret weights in linear models?](#972-how-do-we-interpret-weights-in-linear-models)
- **9.73** [Why is it important to scale the inputs when using SVMs?](#973-why-is-it-important-to-scale-the-inputs-when-using-svms)
- **9.74** [What is OvR and OvO for multiclass classification, and which machine learning algorithm supports this](#974-what-is-ovr-and-ovo-for-multiclass-classification-and-which-machine-learning-algorithm-supports-this)
- **9.75** [What are some advantages and Disadvantages of regression models and tree-based models?](#975-what-are-some-advantages-and-disadvantages-of-regression-models-and-tree-based-models)
- **9.76** [What are some important hyperparameters for XGBOOST](#976-what-are-some-important-hyperparameters-for-xgboost)
- **9.77** [List some of the drawbacks of a Linear model](#977-list-some-of-the-drawbacks-of-a-linear-model)
- **9.78** [Explain the kernel trick in SVM](#978-explain-the-kernel-trick-in-svm)
- **9.79** [What is the difference between a Gaussian Mixture Model and the K-Means Algorithm?](#979-what-is-the-difference-between-a-gaussian-mixture-model-and-the-k-means-algorithm)
- **9.80** [How to select K for K-means?](#980-how-to-select-k-for-k-means)
- **9.81** [List all types of popular recommendation systems?](#981-list-all-types-of-popular-recommendation-systems)

---

### 9.1 What is the KNN (K-Nearest Neighbors) classifier?

A simple, non-parametric learning algorithm that classifies a new data point based on the majority class among its "k" closest training examples in feature space. It doesn't build an explicit general rule during training (it's "lazy learning") — instead, it stores the training data and makes decisions at prediction time by comparing distances to neighbors.

### 9.2 Is KNN prone to overfitting or underfitting?

It depends on k. A small k (e.g., k=1) closely follows individual training points and is prone to overfitting (sensitive to noise). A large k smooths the decision boundary more, which can lead to underfitting if k is too large relative to the dataset.

### 9.3 What is the difference between k-NN and k-means?

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

### 9.4 How does a Decision Tree algorithm work?

- Tree-structured model that splits data based on feature values to make predictions
- Starts at root node, recursively splits data using feature thresholds that best separate classes/values
- **Split criteria**: Gini Impurity, Entropy/Information Gain (classification), Variance Reduction (regression)
- Each internal node = decision/test on a feature; each leaf node = final prediction
- Splitting continues until stopping condition (max depth, min samples, pure node)
- **Pros**: Easy to interpret, no feature scaling needed
- **Cons**: Prone to overfitting (addressed via pruning, max depth, ensemble methods like Random Forest)

### 9.5 Explain how Decision Trees make splits and handle categorical features.

**How Decision Trees Make Splits**

- At each node, algorithm evaluates all possible splits across all features
- Selects split that maximizes **information gain** (or minimizes impurity) between parent and child nodes
- **Impurity measures**:
  - **Gini Impurity**: Gini = 1 − Σ(pᵢ)² — lower is purer
  - **Entropy**: −Σ pᵢ log₂(pᵢ), used with Information Gain = Entropy(parent) − weighted Entropy(children)
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

### 9.6 How does the Random Forest algorithm work? How does it improve over Decision Trees? How does it reduce variance?

**Random Forest**

- Ensemble method that builds multiple decision trees and combines their predictions
- **Prediction**: Classification → majority vote; Regression → average of tree outputs

**How it works**

1. **Bagging (Bootstrap Aggregating)**: Each tree trained on random sample (with replacement) of training data
2. **Feature randomness**: At each split, only a random subset of features considered (not all features)
3. Trees grown deep, largely uncorrelated with each other
4. Final output aggregates all trees' predictions

**Improvement over single Decision Trees**

- Single trees overfit easily (high variance, low bias)
- Random Forest reduces overfitting by averaging many diverse trees
- More robust, generalizes better to unseen data

**How it reduces variance**

- Averaging predictions from many decorrelated trees cancels out individual trees' errors/noise
- Bootstrap sampling + feature randomness ensures trees are diverse (not identical)
- Variance of average of N independent-ish estimators ≈ Variance(single tree) / N (reduced further since trees aren't fully independent, but still significant reduction)
- Bias stays roughly similar to individual trees, but variance drops sharply → better generalization

### 9.7 Explain Ensemble Methods. Why are they powerful?

**Ensemble Methods**

- Combine multiple models (weak/base learners) to produce a stronger, more accurate model
- Based on idea that aggregating diverse models reduces error better than any single model

**Main types**

- **Bagging (Bootstrap Aggregating)**: Train models in parallel on random data subsets, aggregate (vote/average) → reduces variance. Ex: Random Forest
- **Boosting**: Train models sequentially, each correcting errors of previous → reduces bias. Ex: AdaBoost, Gradient Boosting, XGBoost
- **Stacking**: Train multiple different models, use a meta-model to combine their outputs
- **Voting**: Combine predictions via majority vote (classification) or averaging (regression)

**Why powerful**

- Reduces variance (bagging) and/or bias (boosting)
- Individual model errors cancel out when aggregated
- More robust to noise and overfitting than single models
- Captures diverse patterns different models learn

### 9.8 What is the difference between bagging and boosting?

**Bagging vs Boosting**

| Aspect | Bagging | Boosting |
| --- | --- | --- |
| Training | Parallel — models trained independently | Sequential — each model corrects previous one's errors |
| Data sampling | Random bootstrap samples (with replacement) | Reweighted data — misclassified samples get more weight |
| Goal | Reduce variance | Reduce bias (and variance) |
| Model weighting | Equal weight to all models | Weighted based on model performance |
| Overfitting | Less prone to overfitting | More prone to overfitting (if not tuned) |
| Examples | Random Forest | AdaBoost, Gradient Boosting, XGBoost |
| Combination | Averaging/majority vote | Weighted sum of predictions |

### 9.9 What is Gradient Boosting? How does XGBoost work?

**Gradient Boosting**

- Ensemble technique that builds models sequentially, each new model trained to correct residual errors of previous models
- Uses gradient descent to minimize loss function — each new tree fits the negative gradient (residuals) of the loss
- Combines weak learners (typically shallow trees) into a strong learner
- Final prediction = weighted sum of all trees' outputs

**XGBoost (Extreme Gradient Boosting)**

- Optimized, regularized implementation of gradient boosting
- **Key improvements over vanilla GBM**:
  - **Regularization** (L1 & L2) built into objective function → reduces overfitting
  - **Second-order gradients** (uses both gradient and Hessian) → faster, more accurate convergence
  - **Parallel processing** for tree construction → faster training
  - **Handles missing values** automatically (learns default split direction)
  - **Tree pruning**: uses max\_depth then prunes back (unlike greedy pure gain-based growth)
  - **Built-in cross-validation** support
- Widely used for structured/tabular data competitions (Kaggle) due to speed + performance

### 9.10 What are the key hyperparameters for XGBoost?

**Key XGBoost Hyperparameters**

**Tree structure**

- `max_depth` — max depth of each tree (controls complexity/overfitting)
- `min_child_weight` — min sum of instance weight needed in a child node
- `gamma` — min loss reduction required to make a split (higher = more conservative)

**Boosting process**

- `n_estimators` — number of trees/boosting rounds
- `learning_rate` (eta) — shrinks contribution of each tree (lower = more robust, needs more trees)

**Regularization**

- `lambda` (reg\_lambda) — L2 regularization on weights
- `alpha` (reg\_alpha) — L1 regularization on weights

**Sampling**

- `subsample` — fraction of rows sampled per tree (prevents overfitting)
- `colsample_bytree` — fraction of features sampled per tree

**Other**

- `objective` — loss function (e.g., binary:logistic, reg:squarederror)
- `eval_metric` — evaluation metric (e.g., auc, rmse, logloss)
- `early_stopping_rounds` — stop training if validation score doesn't improve

### 9.11 Explain Gradient Boosting and its advantages over Random Forests.

**Gradient Boosting vs Random Forest**

**Gradient Boosting recap**

- Sequential ensemble — each tree corrects errors (residuals) of previous trees
- Uses gradient descent to minimize loss function
- Combines many weak learners (shallow trees) into strong learner

**Advantages over Random Forest**

| Aspect | Gradient Boosting | Random Forest |
| --- | --- | --- |
| Bias | Lower (actively reduces bias) | Higher (relies on averaging) |
| Accuracy | Often higher with proper tuning | Slightly lower typically |
| Learning | Sequential, learns from mistakes | Parallel, independent trees |
| Flexibility | Supports custom loss functions | Limited to standard splits |
| Handles imbalanced data | Better (via loss weighting) | Weaker by default |

- Gradient Boosting typically achieves better predictive performance since it directly optimizes the loss function
- More flexible — can plug in different loss functions for different problems (regression, ranking, classification)
- Trade-off: more prone to overfitting, sensitive to hyperparameters, slower to train (sequential), less parallelizable than Random Forest

### 9.12 Explain how Logistic Regression differs from Linear Regression.

**Logistic Regression vs Linear Regression**

| Aspect | Linear Regression | Logistic Regression |
| --- | --- | --- |
| Output | Continuous numeric value | Probability (0 to 1) → class label |
| Use case | Regression problems | Classification problems |
| Function | Linear: y = wx + b | Sigmoid applied to linear combo: 1/(1+e\^-(wx+b)) |
| Loss function | MSE (Mean Squared Error) | Cross-Entropy / Log Loss |
| Output range | −∞ to +∞ | 0 to 1 |
| Decision boundary | Not applicable | Linear boundary separating classes |

### 9.13 How does logistic regression work?

- Classification algorithm that predicts probability of an input belonging to a class
- Computes linear combination: z = w·x + b
- Applies **sigmoid function** to squash output between 0 and 1: p = 1/(1+e\^-z)
- Predicts class based on threshold (typically 0.5): p ≥ 0.5 → class 1, else class 0
- **Training**: minimizes **cross-entropy (log) loss** using gradient descent to find optimal weights
- Decision boundary is linear (in feature space)
- Extends to multi-class via **Softmax** (Softmax Regression / Multinomial Logistic Regression)

### 9.14 Explain R-squared and adjusted R-squared.

**R-squared vs Adjusted R-squared**

**R-squared (R²)**

- Measures the proportion of variance in the dependent variable explained by the model
- Formula: R² = 1 − (SS\_residual / SS\_total)
- Range: 0 to 1 (higher = better fit)
- **Problem**: Always increases (or stays same) when adding more features, even irrelevant ones

**Adjusted R-squared**

- Modifies R² to penalize adding unnecessary/irrelevant features
- Formula: Adj R² = 1 − \[(1−R²)(n−1) / (n−k−1)\]
  - n = number of observations, k = number of predictors
- Increases only if new feature improves model more than expected by chance
- Better for comparing models with different numbers of features

### 9.15 How do you check for multicollinearity in regression models?

**Checking for Multicollinearity**

- Multicollinearity: high correlation between independent variables, making it hard to isolate individual effects

**Methods to detect**

- **Correlation matrix**: Check pairwise correlation coefficients between features (>0.7–0.8 suggests issue)
- **VIF (Variance Inflation Factor)**:
  - Formula: VIF = 1 / (1 − R²ᵢ), where R²ᵢ is from regressing feature i on all other features
  - VIF > 5 or 10 → high multicollinearity
- **Tolerance**: 1/VIF — low tolerance indicates multicollinearity
- **Eigenvalues/Condition number**: Small eigenvalues of correlation matrix indicate multicollinearity; condition number > 30 is concerning
- **Correlation heatmap**: Visual inspection via seaborn/matplotlib

**How to fix**

- Remove/combine highly correlated features
- Use PCA to decorrelate features
- Apply L2 regularization (Ridge) — handles multicollinearity well
- Increase sample size

### 9.16 How does K-Nearest Neighbors (KNN) work?

**K-Nearest Neighbors (KNN)**

- Instance-based, non-parametric algorithm used for classification and regression
- No explicit training phase — stores entire training dataset ("lazy learner")

**How it works**

1. For a new data point, compute the distance (Euclidean, Manhattan, etc.) to all training points
2. Select K nearest neighbors
3. **Classification**: Assign majority class among K neighbors
4. **Regression**: Average the values of K neighbors

**Key considerations**

- Choice of **K**: small K → sensitive to noise (overfitting); large K → smoother but may underfit
- **Distance metric** matters: Euclidean (default), Manhattan, Minkowski, Cosine
- Requires **feature scaling** (normalization/standardization) since distance-based
- Sensitive to curse of dimensionality (high dimensions reduce meaningfulness of distance)
- Computationally expensive at prediction time (must compare to all points)

### 9.17 Explain K-Means Clustering. How does it work? Limitations?

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

### 9.18 Explain Support Vector Machines (SVM). What is the kernel trick?

**Support Vector Machines (SVM)**

- Supervised algorithm for classification (and regression) that finds optimal hyperplane separating classes
- **Hyperplane**: decision boundary that maximizes the margin (distance) between classes
- **Support vectors**: data points closest to hyperplane that define the margin
- Objective: maximize margin while minimizing classification error
- **Soft margin (C parameter)**: allows some misclassification for better generalization — controls trade-off between margin width and errors

**Kernel Trick**

- Technique to handle non-linearly separable data without explicitly transforming to higher dimensions
- Uses a **kernel function** to compute similarity (dot product) between points as if they were in higher-dimensional space, without actually computing the transformation (computationally efficient)
- **Common kernels**:
  - Linear: for linearly separable data
  - Polynomial: captures polynomial relationships
  - RBF (Radial Basis Function/Gaussian): most common, handles complex non-linear boundaries
  - Sigmoid
- Allows SVM to draw non-linear decision boundaries in original feature space

### 9.19 What is the decision boundary in classifiers?

- The surface/line that separates different predicted classes in feature space
- Points on one side classified as one class, other side as another class
- **Linear boundary**: straight line/hyperplane (e.g., Logistic Regression, Linear SVM)
- **Non-linear boundary**: curved/complex surface (e.g., Kernel SVM, Decision Trees, Neural Networks)
- Determined by model parameters learned during training
- Simpler boundaries → more generalizable but may underfit; complex boundaries → fit training data closely but may overfit

### 9.20 Explain Naive Bayes.

**Naive Bayes**

- Probabilistic classifier based on Bayes' Theorem, assuming feature independence ("naive" assumption)
- **Bayes' Theorem**: P(y\|X) = \[P(X\|y) × P(y)\] / P(X)
- Predicts class with highest posterior probability given features
- **"Naive" assumption**: all features are conditionally independent given the class (rarely true in practice, but works well anyway)

**Common variants**

- **Gaussian NB**: assumes continuous features follow normal distribution
- **Multinomial NB**: for discrete counts (e.g., word counts in text classification)
- **Bernoulli NB**: for binary/boolean features

**Pros**

- Fast, simple, works well with high-dimensional data (e.g., text classification, spam filtering)
- Requires small amount of training data

**Cons**

- Independence assumption rarely holds in real data
- Can be outperformed by more sophisticated models when feature correlations matter

### 9.21 What is Dimensionality Reduction?

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

### 9.22 Explain PCA (Principal Component Analysis). How does it work? When would you use it?

**PCA (Principal Component Analysis)**

- Unsupervised linear technique for dimensionality reduction
- Transforms correlated features into a smaller set of uncorrelated variables called **principal components**
- Principal components are ordered by variance explained (PC1 captures most variance, PC2 next, etc.)

**How it works**

1. Standardize the data (mean = 0, variance = 1)
2. Compute covariance matrix of features
3. Calculate eigenvectors and eigenvalues of covariance matrix
4. Eigenvectors = principal components (directions); eigenvalues = variance explained by each
5. Select top K components (based on explained variance) and project data onto them

**Key concepts**

- **Explained variance ratio**: how much variance each PC captures
- Components are orthogonal (uncorrelated) to each other

**When to use**

- High-dimensional data with correlated features
- Need to speed up training / reduce storage
- Data visualization (reduce to 2D/3D)
- Noise reduction
- Before applying distance-based algorithms (KNN, K-Means) to reduce curse of dimensionality

**Limitations**

- Assumes linear relationships
- Components can be hard to interpret
- Sensitive to feature scaling (must standardize first)

### 9.23 Explain Gradient Descent and its variants.

**Gradient Descent**

- Optimization algorithm that iteratively adjusts model parameters to minimize a loss function
- Computes gradient (partial derivatives) of loss w.r.t. parameters, updates in opposite direction of gradient
- Update rule: θ = θ − α∇J(θ), where α = learning rate

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

### 9.24 What is the ROC-AUC curve, and how is it interpreted?

**ROC-AUC Curve**

- **ROC (Receiver Operating Characteristic) curve**: plots True Positive Rate (Recall) vs False Positive Rate at various classification thresholds
  - Recall (Sensitivity) , TPR = TP / (TP + FN)
  - Specificity, FPR = FP / (FP + TN)
- **AUC (Area Under Curve)**: single value summarizing overall model performance across all thresholds
  - Range: 0 to 1
  - AUC = 0.5 → random guessing (no better than chance)
  - AUC = 1.0 → perfect classifier
  - AUC closer to 1 → better model at distinguishing classes

**Interpretation**

- Measures the model's ability to rank positive instances higher than negative ones, regardless of threshold
- Useful for comparing models independent of chosen classification threshold
- Good for imbalanced datasets (better than accuracy alone), but can be misleading with extreme imbalance — PR-AUC sometimes preferred

### 9.25 Explain Decision Tree Classification.

#### Decision Tree Classification

- A **supervised learning algorithm** used to classify observations into predefined categories.
- It recursively splits the dataset based on **feature values** to create decision rules.
- **Root node:** Represents the first and most important decision.
- **Internal nodes:** Represent feature-based decision conditions.
- **Branches:** Represent the outcomes of those conditions.
- **Leaf nodes:** Represent the final predicted class.
- Common splitting criteria include **Gini Impurity** and **Entropy/Information Gain**.
- **Example:** Classifying a customer as *Buy* or *Not Buy* based on age, income, and purchase history.

#### Decision Tree in Machine Learning

- A **Decision Tree** is a **supervised machine learning algorithm** used for **classification and regression**.
- It makes predictions by asking a series of **questions or conditions** about the input features.
- It has three main components:
  - **Root node:** The starting point where the first decision is made.
  - **Decision/internal nodes:** Conditions used to split the data.
  - **Leaf nodes:** Final predictions or outcomes.
- The tree selects features and split points that best separate the data, commonly using measures such as **Gini impurity, entropy, or information gain**.
- Decision Trees are easy to understand and interpret because their decisions can be followed step by step.
- They can easily **overfit**, especially when the tree becomes very deep.
- Overfitting can be reduced using **maximum depth, minimum samples per split/leaf, pruning**, etc.
- Example:
  - **Income > 50K?**
  - Yes → **Age > 30?**
  - Yes → Approve loan
  - No → Reject loan

### 9.26 To start Linear Regression, you would need to make some assumptions. What are those assumptions?

#### Assumptions of Linear Regression

| **Assumption** | **Explanation** |
| --- | --- |
| **Linearity** | The relationship between independent and dependent variables is linear. |
| **Independence** | Observations/errors are independent of each other. |
| **Homoscedasticity** | The variance of residuals remains constant across predicted values. |
| **Normality of Residuals** | Residuals are approximately normally distributed. |
| **No Multicollinearity** | Independent variables should not be highly correlated with each other. |
| **No Significant Outliers** | Extreme observations should not disproportionately influence the regression model. |

Residual = Actual − Predicted

### 9.27 What is multicollinearity and how will you handle it in your regression model?

#### Multicollinearity

- **Multicollinearity** occurs when two or more independent variables in a regression model are **highly correlated with each other**.
- It makes it difficult to determine the **individual effect** of each feature on the dependent variable.
- It can lead to **unstable coefficients, large standard errors, and unreliable interpretations**.

#### How to Handle It

| Method | Explanation |
| --- | --- |
| **Correlation Matrix** | Identify highly correlated independent variables. |
| **VIF (Variance Inflation Factor)** | Quantify multicollinearity. A high VIF indicates a problem. |
| **Remove Features** | Remove one of the highly correlated variables if it is not essential. |
| **Combine Features** | Create a single feature from correlated variables when appropriate. |
| **Regularization** | Use **Ridge (L2)** or **Lasso (L1)** regression to reduce the impact of correlated features. |
| **Dimensionality Reduction** | Use techniques such as **PCA** when many correlated features exist. |

### 9.28 Explain why the performance of XGBoost is better than that of SVM?

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

### 9.29 What are five popular algorithms used in Machine Learning?

Five popular Machine Learning algorithms are:

1. **Linear Regression** – Predicts continuous numerical values.
2. **Logistic Regression** – Used mainly for binary classification.
3. **Decision Trees** – Used for both classification and regression.
4. **Random Forest** – An ensemble of multiple decision trees; works for classification and regression.
5. **Support Vector Machines (SVM)** – Mainly used for classification, but also supports regression (SVR).

**Easy way to remember:**

**Linear Regression → Logistic Regression → Decision Tree → Random Forest → SVM**.

### 9.30 List the differences between KNN and k-means clustering.

#### KNN vs K-Means Clustering

| Aspect | **KNN (K-Nearest Neighbors)** | **K-Means Clustering** |
| --- | --- | --- |
| **Type** | Supervised learning | Unsupervised learning |
| **Purpose** | Classification or regression | Clustering |
| **Training data** | Requires labeled data | Does not require labels |
| **Meaning of K** | Number of nearest neighbors considered | Number of clusters to create |
| **How it works** | Predicts based on the closest training examples | Assigns data points to the nearest cluster centroid |
| **Output** | Class label or numerical value | Cluster assignment |
| **Training** | Almost no explicit training; computation happens during prediction | Iteratively learns cluster centroids |
| **Example** | Classify an email as spam/not spam | Group customers into different segments |
| **Common metric** | Euclidean distance, Manhattan distance, etc. | Usually Euclidean distance |
| **Main challenge** | Choosing an appropriate K and handling high-dimensional data | Choosing the number of clusters K |

#### Easy way to remember

- **KNN:** *“Which labeled points are closest to this new point?”*
- **K-Means:** *“Which cluster does this point belong to?”*

**Key difference:** KNN uses **labeled data to make predictions**, while K-Means uses **unlabeled data to discover groups**.

### 9.31 What is a PCA?

#### PCA (Principal Component Analysis)

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms a dataset with many correlated features into a smaller set of **uncorrelated variables called principal components**, while preserving as much of the data's variance as possible.

#### How PCA works

1. **Standardize the data** if features have different scales.
2. Calculate the **covariance matrix**.
3. Compute its **eigenvectors and eigenvalues**.
4. Rank the eigenvectors by their eigenvalues.
5. Select the top k eigenvectors as the **principal components**.
6. Project the original data onto these components.

The first principal component captures the **maximum variance**, the second captures the maximum remaining variance while being orthogonal to the first, and so on.

#### Example

Suppose a dataset has **100 features**. PCA might reduce it to **10 principal components** while retaining, for example, 95% of the original variance.

#### Why use PCA?

- Reduce dimensionality
- Remove redundant information
- Reduce computational cost
- Help visualize high-dimensional data
- Reduce the effect of multicollinearity
- Sometimes improve model performance

#### Key point

> **PCA converts many potentially correlated features into fewer uncorrelated principal components that retain most of the important information (variance) in the original data.**

### 9.32 What are support vectors in SVM (Support Vector Machine)?

- **Support vectors** are the data points closest to the decision boundary in an SVM.
- They determine the **position and orientation of the optimal decision boundary**.
- They define the **margin** between the classes.
- They are the most important training points for the SVM classifier.

### 9.33 What techniques are used to find resemblance in the recommendation system?

- Recommendation systems use **similarity techniques** to find how similar users, items, or their features are.

Common techniques include:

- **Cosine Similarity** — Measures similarity between item or user vectors based on their angle.
- **Pearson Correlation** — Measures how similarly two users or items behave or are rated.
- **Jaccard Similarity** — Measures similarity between sets, such as products two users have purchased.
- **Euclidean Distance** — Measures the distance between feature vectors; smaller distance means more similar.
- **Manhattan Distance** — Measures similarity based on the sum of absolute differences between features.
- **Collaborative Filtering** — Finds similar users or items from their interaction history.
- **Content-Based Filtering** — Finds items similar based on their attributes, such as category, description, or genre.
- **Embedding Similarity** — Uses vector embeddings from models such as neural networks to find semantically similar items.

### 9.34 What is another name for a Bayesian Network?

- A **Bayesian Network** is also called a **Bayes Network**, **Bayesian Belief Network (BBN)**, Casual network, Bayes net, Belief Propagation Network or **Belief Network**.
- It is a **probabilistic graphical model** that represents relationships and conditional dependencies between variables.

### 9.35 What is ensemble learning?

- **Ensemble learning** is a machine learning technique that **combines multiple models** to make better and more reliable predictions.
- Instead of relying on a single model, it combines the predictions of several models to improve **accuracy, stability, and generalization**.
- Common ensemble methods include:
  - **Bagging** → Trains multiple models **independently** on different samples of the data and combines their predictions. It mainly helps **reduce variance**.
    - Example: **Random Forest**, which combines many decision trees.
  - **Boosting** → Trains models **sequentially**, where each new model focuses on correcting the errors of previous models. It mainly helps **reduce bias** and improve predictive performance.
    - Examples: **XGBoost, AdaBoost, Gradient Boosting**.
  - **Stacking** → Combines predictions from **different types of models** and uses another model, called a **meta-model**, to learn how to combine their predictions effectively.
    - Example: **Random Forest + SVM + XGBoost → Meta-model → Final prediction**.
- **In simple terms:**
  - **Bagging** → Models work **independently**.
  - **Boosting** → Models work **sequentially to correct errors**.
  - **Stacking** → **Different models are combined using a meta-model**.

### 9.36 What is clustering?

- **Clustering** is an **unsupervised machine learning technique** that groups similar data points into **clusters**.
- The goal is to make:
  - Data points **within the same cluster** as similar as possible.
  - Data points **from different clusters** as different as possible.
- Clustering does **not require labeled data**. The algorithm discovers the groups automatically.
- Common clustering algorithms include:
  - **K-Means** → Divides data into a predefined number of clusters.
  - **Hierarchical Clustering** → Creates a hierarchy of clusters.
  - **DBSCAN** → Groups closely packed points and identifies noise/outliers.
  - **Gaussian Mixture Model (GMM)** → Groups data based on probability distributions.
- Common applications include:
  - **Customer segmentation**
  - **Document/news grouping**
  - **Image segmentation**
  - **Anomaly detection**
  - **Recommendation systems**

### 9.37 What is the Bayesian Network?

- A **Bayesian Network** is a **probabilistic graphical model** that represents relationships and dependencies between variables using a **directed acyclic graph (DAG)**.
- It consists of:
  - **Nodes** → Represent variables or events.
  - **Directed edges** → Represent probabilistic dependencies between variables.
  - **Conditional probability distributions (CPDs)** → Describe how each variable depends on its parent variables.
- Bayesian Networks are used to **reason under uncertainty** and calculate the probability of events based on available evidence.
- **Example:**`Rain → Wet Grass`<br>If we know that it is raining, the probability that the grass is wet increases.
- Common applications include:
  - Medical diagnosis
  - Risk assessment
  - Fraud detection
  - Fault diagnosis
  - Decision support systems

### 9.38 What is dimension reduction in ML?

- **Dimension reduction** is the process of **reducing the number of features (dimensions) in a dataset** while preserving as much useful information as possible.
- It is useful when a dataset has **many features**, some of which may be redundant or irrelevant.
- Main benefits include:
  - Reduces **computational complexity**
  - Removes **redundant or noisy features**
  - Helps reduce **overfitting**
  - Makes data easier to **visualize**
  - Can improve model training speed
- Common dimension reduction techniques include:
  - **PCA (Principal Component Analysis)** → Creates new features called principal components that capture the most variance in the data.
  - **LDA (Linear Discriminant Analysis)** → Reduces dimensions while trying to preserve class separability.
  - **t-SNE** → Mainly used to visualize high-dimensional data in 2D or 3D.
  - **UMAP** → Used for dimensionality reduction and visualization while preserving important data structure.

**Example:** If a dataset has **100 features**, dimension reduction might transform it into **10 important components** while retaining most of the useful information.

### 9.39 Why is the Naive Bayes Method ‘Naive’?

#### Why is Naive Bayes called “Naive”?

- Naive Bayes is called **“naive”** because it makes a **strong assumption that all features are independent of each other**, given the class.
- In real-world data, features are often **related or correlated**.
- Despite this unrealistic assumption, Naive Bayes can still perform very well, especially in **text classification, spam detection, and sentiment analysis**.

**Example:**

For email spam detection, suppose the features are:

- Contains “free”
- Contains “offer”
- Contains “prize”

Naive Bayes assumes these features are **independent of each other** when determining whether the email is spam.

### 9.40 State the Bayes’ theorem for Naive Bayes Method.

#### Bayes’ Theorem for Naive Bayes

Bayes’ theorem is:

**P(C \| X) = \[P(X \| C) × P(C)\] / P(X)**

Where:

- **P(C \| X)** → Probability of class **C** given the features **X** (posterior probability)
- **P(X \| C)** → Probability of observing features **X** given class **C** (likelihood)
- **P(C)** → Probability of class **C** before seeing the features (prior probability)
- **P(X)** → Probability of observing the features **X** (evidence)

In **Naive Bayes**, the feature probability is further simplified by assuming that the features are **conditionally independent given the class**.

**In simple terms:** Bayes’ theorem calculates **how likely a class is after seeing the input features**.

### 9.41 Explain Kernel SVM.

#### Kernel SVM

- **Kernel SVM (Support Vector Machine)** is an extension of SVM that can classify data that is **not linearly separable**.
- A normal SVM tries to find the **best separating hyperplane** between classes.
- When data cannot be separated by a straight line or flat plane, Kernel SVM uses the **kernel trick**.
- The **kernel function** effectively maps the original data into a **higher-dimensional feature space**, where the data may become linearly separable.
- Common kernel functions include:
  - **Linear Kernel**
  - **Polynomial Kernel**
  - **RBF (Gaussian) Kernel**
  - **Sigmoid Kernel**
- Kernel SVM mainly depends on **support vectors**, which are the data points closest to the decision boundary.
- It is useful for **nonlinear classification problems**.

### 9.42 What are some types of clustering?

#### Types of Clustering

Common types of clustering in machine learning include:

- **Partitioning Clustering:** Divides data into a fixed number of clusters.
  - Example: **K-Means**
- **Hierarchical Clustering:** Creates a hierarchy of clusters, either by merging smaller clusters or splitting larger ones.
  - Example: **Agglomerative Clustering**
- **Density-Based Clustering:** Forms clusters based on areas of high data density and identifies sparse points as noise or outliers.
  - Examples: **DBSCAN, OPTICS**
- **Model-Based Clustering:** Assumes the data comes from a mixture of underlying probability distributions.
  - Example: **Gaussian Mixture Model (GMM)**
- **Grid-Based Clustering:** Divides the data space into grid cells and forms clusters based on dense cells.
  - Example: **STING**

### 9.43 What is content-based filtering and collaborative filtering?

#### Content-Based Filtering vs Collaborative Filtering

##### 1. Content-Based Filtering

- Recommends items based on the **features or characteristics of items the user has liked before**.
- It focuses on the **user's own preferences and item attributes**.
- Example: If you frequently watch **action movies**, the system recommends more action movies.
- It does not require information about other users.

**In simple terms:** **“You liked this item, so I recommend similar items.”**

##### 2. Collaborative Filtering

- Recommends items based on the **behavior and preferences of multiple users**.
- It finds users with **similar preferences** and recommends items they liked.
- Example: If users who liked the same movies as you also liked another movie, that movie may be recommended to you.
- It can be based on **user-user** or **item-item** similarity.

**In simple terms:** **“Users similar to you liked this item, so I recommend it to you.”**

##### Key Difference

| Content-Based | Collaborative |
| --- | --- |
| Uses **item features** | Uses **user-item interactions** |
| Focuses on the **same user's history** | Learns from **multiple users' behavior** |
| Recommends similar items | Recommends items based on similar users/items |
| Less affected by new users | Can struggle with new users/items (**cold start**) |

### 9.44 What are the two components of Bayesian logic?

#### Two Components of Bayesian Logic

The two main components of Bayesian logic are:

1. **Prior Probability**
   - Represents what we **already know or believe** about an event before seeing new evidence.
   - Example: The probability that a person has a particular disease before a medical test.
2. **Likelihood**
   - Represents how **likely the observed evidence is**, assuming a particular hypothesis or event is true.
   - Example: How likely a positive test result is if the person actually has the disease.

**In simple terms:**

**Prior = what we know before the evidence**

**Likelihood = how well the evidence supports the hypothesis**

### 9.45 How would you describe a Recommender system?

#### Recommender System

- A **Recommender System** is a machine learning system that **suggests relevant items to users based on their preferences, behavior, or interests**.
- It analyzes information such as:
  - User interactions and history
  - Item features
  - Ratings, clicks, views, or purchases
  - Preferences of similar users
- Common approaches include:
  - **Content-Based Filtering:** Recommends items similar to what the user previously liked.
  - **Collaborative Filtering:** Recommends items based on the behavior of similar users or items.
  - **Hybrid Recommendation:** Combines multiple recommendation approaches.
- Examples include recommending **movies, products, songs, news, or videos**.

### 9.46 Do you consider decision trees to have any advantages or disadvantages? If yes, what are they?

#### Advantages and Disadvantages of Decision Trees

##### Advantages

- **Easy to understand and interpret:** Decisions can be followed step by step.
- **Works for classification and regression:** Can predict both categories and numerical values.
- **Requires little data preprocessing:** Usually does not require feature scaling or normalization.
- **Handles numerical and categorical features:** Can work with different types of input variables.
- **Captures nonlinear relationships:** Can model complex relationships between features and the target.
- **Feature selection:** Automatically selects important features when creating splits.

##### Disadvantages

- **Prone to overfitting:** Deep trees can memorize the training data.
- **High variance:** Small changes in training data can produce a very different tree.
- **Can become complex:** Large trees can be difficult to interpret.
- **Greedy splitting:** Decisions made at each split may not lead to the globally optimal tree.
- **Sensitive to noisy data:** Noise can lead to unnecessary or poor splits.
- **Less stable than ensemble methods:** A single decision tree may perform worse than methods such as Random Forest or Gradient Boosting.

### 9.47 What is a random forest, and how does it work?

#### Random Forest

- **Random Forest** is an **ensemble learning algorithm** that combines multiple **Decision Trees** to make more accurate and robust predictions.
- It can be used for both **classification and regression**.
- Random Forest is a **bagging** (Bootstrap Aggregating) algorithm.

##### How Random Forest Works

1. It creates **multiple decision trees** using different random samples of the training data.
2. At each split, each tree considers a **random subset of features** rather than all features.
3. Each tree makes its own prediction.
4. The Random Forest combines the predictions:
   - **Classification:** Uses **majority voting**.
   - **Regression:** Uses the **average prediction**.
5. Combining many trees helps **reduce overfitting and variance** compared with using a single decision tree.

##### Key Advantages

- Reduces overfitting compared with a single Decision Tree.
- Handles nonlinear relationships well.
- Works with many features.
- Generally robust to noise and outliers.
- Provides estimates of **feature importance**.

### 9.48 What ensemble techniques can be used to aggregate multiple models?

#### Ensemble Techniques

The core ensemble techniques used to combine multiple machine learning models are **Bagging, Boosting, Stacking, and Voting/Averaging**.

##### 1. Bagging — Bootstrap Aggregating

- Trains multiple models **independently and in parallel** on different bootstrap samples of the training data.
- Each sample is created by **sampling with replacement**.
- Predictions are combined using:
  - **Majority voting** for classification.
  - **Averaging** for regression.
- **Main goal:** Reduce **variance** and overfitting.
- **Example:** Random Forest.

##### 2. Boosting

- Trains models **sequentially**, where each new model focuses on correcting errors made by previous models.
- The models are combined to form a stronger learner.
- **Main goal:** Reduce **bias** and improve predictive performance.
- **Examples:** AdaBoost, Gradient Boosting, XGBoost, LightGBM.

##### 3. Stacking

- Combines predictions from **different types of base models**.
- The predictions of the base models are used as inputs to another model called a **meta-learner**.
- The meta-learner produces the final prediction.
- **Main goal:** Combine the strengths of different algorithms to improve predictive performance.
- **Example:** SVM + Random Forest + Neural Network → Meta-learner → Final prediction.

##### 4. Voting and Averaging

- Combines predictions from multiple models directly.
- **Hard Voting:** The class receiving the most votes is selected.
- **Soft Voting:** Class probabilities from models are combined, usually by averaging them.
- **Averaging:** Predictions are averaged for regression.
- **Weighted Averaging:** Better-performing models can be given higher weights.

##### Quick Comparison

| Technique | How Models Work | Main Purpose |
| --- | --- | --- |
| **Bagging** | Independent / Parallel | Reduce **variance** |
| **Boosting** | Sequential / Dependent | Reduce **bias** |
| **Stacking** | Base models + Meta-model | Combine different model strengths |
| **Voting/Averaging** | Combine predictions directly | Improve overall performance |

**In simple terms:**

- **Bagging:** Build models independently and combine them.
- **Boosting:** Build models sequentially, correcting previous errors.
- **Stacking:** Use different models and let another model combine their predictions.
- **Voting/Averaging:** Combine model predictions directly.

### 9.49 Do you think logistic regression can be used for more than two classes?

Yes. **Logistic Regression can be extended to handle multiclass classification.**

- **Binary Logistic Regression:** Used for **2 classes**, such as Yes/No.
- **Multinomial Logistic Regression:** Used when there are **3 or more classes** with no natural ordering.
  - Example: Classifying an image as **Cat, Dog, or Horse**.
- **One-vs-Rest (OvR):** Builds one binary logistic regression model for each class and selects the class with the highest probability.
- **Multinomial/Softmax:** Directly calculates probabilities for all classes and selects the class with the highest probability.

### 9.50 Explain Genetic programming as it relates to Evolutionary algorithm.

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

- `x + 5`
- `x × 2`
- `(x + 3) × y`

It evaluates them and evolves better expressions over generations.

### 9.51 What are some classification methods that SVM can handle?

#### Classification Methods SVM Can Handle

- **Binary Classification:** Classifies data into **two classes**.
  - Example: Spam vs Not Spam.
- **Multiclass Classification:** Classifies data into **more than two classes**, using methods such as **One-vs-One (OvO)** or **One-vs-Rest (OvR)**.
  - Example: Cat vs Dog vs Horse.
- **Multilabel Classification:** Assigns **multiple labels to the same data point** using multiple binary SVM classifiers.
  - Example: An image can have labels such as **Animal, Outdoor, and Dog**.

### 9.52 Why do you think instance-based learning algorithm is sometimes referred to as Lazy learning algorithm?

#### Why Is Instance-Based Learning Called Lazy Learning?

- **Instance-based learning** stores the training data instead of building a general model during training.
- It performs most of the **computation when making a prediction**, rather than during training.
- Therefore, the **training phase is fast**, but the **prediction phase can be slower**.
- This is why it is called **Lazy Learning** — the algorithm “waits” until a prediction is needed before doing most of the work.
- Example: **K-Nearest Neighbors (KNN)**.

### 9.53 Explain the reason for pruning of a decision tree.

- **Pruning** means removing unnecessary branches or nodes from a decision tree.
- It is mainly done to **reduce overfitting**.
- A very deep tree may learn **noise and small details** from the training data.
- Pruning makes the tree **simpler and easier to interpret**.
- It can improve the model's **performance on unseen data**.
- It can also reduce **computational complexity**.

### 9.54 What differentiates Gradient boosting and Random forest?

#### Gradient Boosting vs Random Forest

Both are **ensemble learning methods based on decision trees**, but they work differently.

- **Random Forest:**
  - Builds many decision trees **independently and in parallel**.
  - Each tree is trained on a different random sample of data/features.
  - Combines the predictions of all trees.
  - Mainly helps **reduce variance and overfitting**.
  - Generally easier to tune and less sensitive to hyperparameters.
- **Gradient Boosting:**
  - Builds trees **sequentially**, one after another.
  - Each new tree focuses on **correcting the errors of previous trees**.
  - Combines the trees to make the final prediction.
  - Mainly helps **reduce bias and improve predictive accuracy**.
  - Usually requires more careful tuning.

| Feature | Random Forest | Gradient Boosting |
| --- | --- | --- |
| Tree building | Parallel/independent | Sequential |
| Main focus | Reduce variance | Reduce bias |
| Error correction | No | Yes |
| Overfitting | More resistant | Can overfit if not tuned |
| Tuning | Easier | More sensitive |

**Simple terms:**

**Random Forest = many independent trees working together.**

**Gradient Boosting = trees built one after another, each correcting previous errors.**

### 9.55 When recommendations like “....people who did this also bought this” comes up on a website. What sort of algorithm propels that?

#### Algorithm Behind “People Who Did This Also Bought This”

- This type of recommendation is mainly powered by **Collaborative Filtering**.
- More specifically, it is often called **Item-to-Item Collaborative Filtering**.
- The system analyzes the behavior of many users, such as:
  - Products they bought
  - Products they viewed
  - Products they added to carts
- It identifies products that are **frequently associated with each other**.
- For example:
  - Many users who bought **Product A** also bought **Product B**.
  - The system recommends **Product B** to someone who buys **Product A**.

### 9.56 What is an OOB error, and how is it useful?

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

### 9.57 In what scenario should a decision tree be preferred over random forest?

A single decision tree is preferable to a random forest when **interpretability, simplicity or speed** matter more than the last few points of accuracy:

- **Explainability and auditability:** a shallow tree can be printed as if-then rules and shown to regulators, doctors or business users (credit decisions, clinical triage, policy rules). A forest of 500 trees is a black box without SHAP-style tools.
- **Rules to hand off:** the tree can be converted into SQL `CASE` statements or business rules engines.
- **Low-latency or constrained environments:** a single tree is tiny and predicts in microseconds — useful on embedded devices or in very high-QPS paths.
- **Small, clean data with a simple structure:** if a depth-3 tree already captures the signal, a forest adds complexity without real gain.
- **Exploratory analysis:** quickly understanding which features and thresholds split the data.
- **Teaching or baselines.**

**Trade-off:** single trees have **high variance** — small data changes produce very different trees, and they overfit easily. Mitigate with pruning (`max_depth`, `min_samples_leaf`, cost-complexity `ccp_alpha`) and cross-validation.

**Middle ground:** train a forest or GBM for accuracy, then **distill** it into a surrogate tree for explanation, or use a shallow tree when its validated performance is within your tolerance of the ensemble's.

### 9.58 Why is Logistic Regression called regression?

Because at its core it **is a regression model — a linear regression on the log-odds** — and only becomes a classifier when you put a threshold on its output.

$\log\frac{p}{1-p} = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n \quad\Rightarrow\quad p = \sigma(\beta^T x) = \frac{1}{1+e^{-\beta^T x}}$

- It models a **continuous quantity**: the probability $p$ (equivalently, the log-odds, which ranges over all reals) as a linear function of the features.
- It belongs to the family of **Generalized Linear Models (GLMs)**, alongside linear regression (identity link) and Poisson regression (log link); logistic regression uses the **logit link** with a Bernoulli outcome. Statisticians call all of these "regression".
- Coefficients are interpreted like regression coefficients: a one-unit increase in $x_j$ multiplies the odds by $e^{\beta_j}$.
- Classification is a separate **decision step**: predict class 1 if $p > t$ (e.g., 0.5, or a threshold tuned to business cost).

**Why this matters in practice:** because it outputs (reasonably calibrated) probabilities, logistic regression is used for risk scoring, ranking and expected-value decisions — not just hard labels. It is trained with maximum likelihood (log loss), not least squares.

### 9.59 What are the main points of difference between Bagging and Boosting?

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

### 9.60 What is Matrix Factorization, and where is it used in Machine Learning?

**Matrix factorization** decomposes a matrix $R$ ($m \times n$) into a product of lower-rank matrices, typically $R \approx U V^T$ with $U \in \mathbb{R}^{m\times k}$, $V \in \mathbb{R}^{n\times k}$ and $k \ll m,n$. Each row of $U$ and $V$ is a latent vector, and their dot product reconstructs an entry.

**Where it is used:**
- **Recommender systems (collaborative filtering):** user–item rating matrix → user and item embeddings; predicted rating $\hat r_{ui} = u_u^T v_i$ (+ biases). Made famous by the Netflix Prize. Trained with SGD or **ALS** (Alternating Least Squares, easy to parallelize, e.g. Spark MLlib), with L2 regularization, fitting only the observed entries.
- **Dimensionality reduction:** SVD/PCA, truncated SVD for sparse text matrices (LSA/topic modeling).
- **NMF (non-negative MF):** interpretable parts-based factors for topic modeling, audio source separation, image parts.
- **Compression:** low-rank approximation of neural network weights (the same idea behind LoRA).
- **Imputation** of missing values in matrices.

**Tradeoffs:** it is fast, scalable and gives reusable embeddings, but has a **cold-start problem** (new users/items have no vector), is linear (dot-product interactions only), and implicit feedback (clicks) needs weighted variants (e.g. implicit ALS). In production, hybrid models (factorization machines, two-tower networks) add side features to solve cold start.

### 9.61 Explain Pruning in Decision Trees and how it is done

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

### 9.62 What are kernels in SVM? Can you list some popular SVM kernels?

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

### 9.63 What is the difference between Gini Impurity and Entropy? Which one is better and why?

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

### 9.64 Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?

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

### 9.65 Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?

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

### 9.66 Why do we always need the intercept term in a regression model?

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

### 9.67 Why is Ridge Regression called Ridge?

There are two complementary explanations.

**1. The "ridge" added to the matrix (the original, historical reason).** The OLS solution $\hat\beta = (X^TX)^{-1}X^Ty$ becomes unstable when features are collinear, because $X^TX$ is near-singular. Hoerl and Kennard (1970) proposed
$$\hat\beta_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$$
Adding $\lambda I$ puts a constant "ridge" along the diagonal of $X^TX$. This lifts every eigenvalue by $\lambda$, makes the matrix invertible and well-conditioned, and stabilizes the coefficients. The name also echoes Hoerl's earlier "ridge analysis" of response surfaces, where the path of optimal estimates traces a ridge.

**2. The ridge in the likelihood surface.** With multicollinearity, the loss surface has a long flat valley or ridge: many coefficient combinations fit almost equally well, so the estimates have huge variance. The L2 penalty $\lambda\lVert\beta\rVert_2^2$ adds curvature, picks the smallest-norm solution along that ridge, and shrinks the coefficients.

**Why it matters:** ridge trades a little bias for a large reduction in variance, keeps all features (no sparsity, unlike Lasso), and is the go-to when predictors are highly correlated. Equivalent views: L2 regularization, a Gaussian prior on the weights (MAP), and weight decay in neural networks.

### 9.68 Does it make any sense to chain two different dimensionality reduction algorithms?

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

### 9.69 Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?

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

### 9.70 If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?

**No.** Decision trees are **invariant to monotonic feature scaling**. A split asks "is $x_j \le t$?", and scaling $x_j$ (standardization, min-max, even a log transform) just moves the threshold $t$. The ordering of samples, and therefore the partitions and impurity reductions, stays exactly the same. Scaling won't change the tree at all, apart from floating-point ties.

**What actually fixes an underfitting tree** (high bias):
- **Loosen the regularization:** increase `max_depth` and `max_leaf_nodes`, decrease `min_samples_leaf`, `min_samples_split` and `min_impurity_decrease`, reduce `ccp_alpha`.
- **Better features:** trees struggle with diagonal or additive relationships, because they need many axis-aligned splits. Engineered features like ratios, differences, interactions and aggregates help a lot.
- **Switch to ensembles:** gradient boosting reduces bias by adding trees sequentially. Random forests mostly reduce variance, so they help less with pure underfitting.
- Check data quality and label noise. Sometimes the signal just isn't in the features.

**When scaling does matter:** distance- or gradient-based models such as KNN, SVM, linear and logistic regression with regularization, neural networks, PCA and k-means. In a mixed pipeline (e.g. trees stacked with logistic regression) you scale for the other models, not for the tree.

### 9.71 Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?

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

### 9.72 How do we interpret weights in linear models?

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

### 9.73 Why is it important to scale the inputs when using SVMs?

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

### 9.74 What is OvR and OvO for multiclass classification, and which machine learning algorithm supports this

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

### 9.75 What are some advantages and Disadvantages of regression models and tree-based models?

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

### 9.76 What are some important hyperparameters for XGBOOST

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

### 9.77 List some of the drawbacks of a Linear model

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

### 9.78 Explain the kernel trick in SVM

An SVM's dual formulation and decision function only use **dot products** between data points:

$f(x) = \text{sign}\left(\sum_i \alpha_i y_i \langle x_i, x\rangle + b\right)$

If the data isn't linearly separable, we could map it to a higher-dimensional space $\phi(x)$ where it is, and then compute $\langle \phi(x_i), \phi(x)\rangle$. The **kernel trick** replaces that dot product with a kernel function $K(x_i, x) = \langle \phi(x_i), \phi(x)\rangle$ computed directly in the original space, so we **never construct $\phi(x)$ explicitly**.

**Example:** for $x \in \mathbb{R}^2$, the polynomial kernel $K(x,z) = (x \cdot z)^2$ equals the dot product of the explicit features $(x_1^2, \sqrt2 x_1x_2, x_2^2)$. The RBF kernel $K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$ corresponds to an **infinite-dimensional** feature space, which would be impossible to compute explicitly.

**Requirements:** $K$ must be a valid (positive semi-definite, Mercer) kernel.

**Tradeoffs:**

- Lets a linear algorithm learn nonlinear boundaries cheaply.
- Training needs the $n \times n$ kernel matrix, so cost is roughly $O(n^2)$ to $O(n^3)$; kernel SVMs don't scale well past ~100k samples. In practice, use linear SVMs, random Fourier features / Nystrom approximations, or gradient boosting / neural nets at scale.
- Kernel choice and hyperparameters ($\gamma$, $C$) need tuning.

### 9.79 What is the difference between a Gaussian Mixture Model and the K-Means Algorithm?

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

### 9.80 How to select K for K-means?

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

### 9.81 List all types of popular recommendation systems?

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

---

← [Chapter 8. Machine Learning Fundamentals](08-machine-learning-fundamentals.md) · [Table of contents](../README.md) · [Chapter 10. Data Preprocessing and Feature Engineering](10-data-preprocessing-and-feature-engineering.md) →
