<!-- notion-id: 3d667418285480e78060edb1702ef241 -->
# Algorithms

Classical ML algorithms: decision trees, random forests, ensembles and gradient boosting (XGBoost), linear/logistic regression, KNN, K-Means, SVM, Naive Bayes, PCA and dimensionality reduction, gradient descent, and ROC-AUC.

### Q1. How does a Decision Tree algorithm work?

- Tree-structured model that splits data based on feature values to make predictions
- Starts at root node, recursively splits data using feature thresholds that best separate classes/values
- **Split criteria**: Gini Impurity, Entropy/Information Gain (classification), Variance Reduction (regression)
- Each internal node = decision/test on a feature; each leaf node = final prediction
- Splitting continues until stopping condition (max depth, min samples, pure node)
- **Pros**: Easy to interpret, no feature scaling needed
- **Cons**: Prone to overfitting (addressed via pruning, max depth, ensemble methods like Random Forest)

### Q2. Explain how Decision Trees make splits and handle categorical features.

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

### Q3. How does the Random Forest algorithm work? How does it improve over Decision Trees? How does it reduce variance?

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

### Q4. Explain Ensemble Methods. Why are they powerful?

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

### Q5. What is the difference between bagging and boosting?

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

### Q6. What is Gradient Boosting? How does XGBoost work?

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

### Q7. What are the key hyperparameters for XGBoost?

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

### Q8. Explain Gradient Boosting and its advantages over Random Forests.

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

### Q9. Explain how Logistic Regression differs from Linear Regression.

**Logistic Regression vs Linear Regression**

| Aspect | Linear Regression | Logistic Regression |
| --- | --- | --- |
| Output | Continuous numeric value | Probability (0 to 1) → class label |
| Use case | Regression problems | Classification problems |
| Function | Linear: y = wx + b | Sigmoid applied to linear combo: 1/(1+e\^-(wx+b)) |
| Loss function | MSE (Mean Squared Error) | Cross-Entropy / Log Loss |
| Output range | −∞ to +∞ | 0 to 1 |
| Decision boundary | Not applicable | Linear boundary separating classes |

### Q10. How does logistic regression work?

- Classification algorithm that predicts probability of an input belonging to a class
- Computes linear combination: z = w·x + b
- Applies **sigmoid function** to squash output between 0 and 1: p = 1/(1+e\^-z)
- Predicts class based on threshold (typically 0.5): p ≥ 0.5 → class 1, else class 0
- **Training**: minimizes **cross-entropy (log) loss** using gradient descent to find optimal weights
- Decision boundary is linear (in feature space)
- Extends to multi-class via **Softmax** (Softmax Regression / Multinomial Logistic Regression)

### Q11. Explain R-squared and adjusted R-squared.

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

### Q12. How do you check for multicollinearity in regression models?

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

### Q13. How does K-Nearest Neighbors (KNN) work?

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

### Q14. Explain K-Means Clustering. How does it work? Limitations?

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

### Q15. Explain Support Vector Machines (SVM). What is the kernel trick?

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

### Q16. What is the decision boundary in classifiers?

- The surface/line that separates different predicted classes in feature space
- Points on one side classified as one class, other side as another class
- **Linear boundary**: straight line/hyperplane (e.g., Logistic Regression, Linear SVM)
- **Non-linear boundary**: curved/complex surface (e.g., Kernel SVM, Decision Trees, Neural Networks)
- Determined by model parameters learned during training
- Simpler boundaries → more generalizable but may underfit; complex boundaries → fit training data closely but may overfit

### Q17. Explain Naive Bayes.

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

### Q18. What is Dimensionality Reduction?

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

### Q19. Explain PCA (Principal Component Analysis). How does it work? When would you use it?

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

### Q20. Explain Gradient Descent and its variants.

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

### Q21. What is the ROC-AUC curve, and how is it interpreted?

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
