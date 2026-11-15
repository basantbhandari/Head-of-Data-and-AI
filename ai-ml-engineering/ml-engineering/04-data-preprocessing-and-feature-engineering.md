# Data Preprocessing and Feature Engineering

Data preprocessing and feature engineering: encoding categorical features, missing data, outliers, scaling, feature creation and selection, imbalanced classes, and train/validation/test splits.

### Q1. What is Feature Engineering?

**Feature Engineering**

- Process of creating, transforming, or selecting features from raw data to improve model performance
- Uses domain knowledge to make data more informative for ML algorithms

**Common techniques**

- **Feature creation**: combining/deriving new features (e.g., date → day/month/year)
- **Encoding categorical variables**: one-hot, label, target encoding
- **Scaling/normalization**: StandardScaler, MinMaxScaler
- **Handling missing values**: imputation (mean, median, mode, model-based)
- **Binning**: converting continuous variables into categories
- **Feature selection**: removing irrelevant/redundant features
- **Polynomial features**: capturing non-linear relationships
- **Log/power transforms**: handling skewed distributions
- Often has bigger impact on model performance than algorithm choice itself

### Q2. What is one-hot encoding? When should you use it?

**One-Hot Encoding**

- Converts categorical variables into binary vectors — one column per category, 1 if present, 0 otherwise
- Example: Color \[Red, Blue, Green\] → Red=\[1,0,0\], Blue=\[0,1,0\], Green=\[0,0,1\]
- Avoids implying false ordinal relationships (unlike label encoding)

**When to use**

- Nominal categorical data (no inherent order) — e.g., color, city, category
- Low-to-medium cardinality features (few unique categories)
- Algorithms sensitive to numeric magnitude (Linear/Logistic Regression, SVM, Neural Networks)

**When NOT to use**

- High-cardinality features (many unique values) → creates too many sparse columns (use target/embedding encoding instead)
- Tree-based models often handle categorical/label encoding fine without one-hot
- Can lead to curse of dimensionality if overused

### Q3. How do you deal with missing data?

**Handling Missing Data**

**Detection**

- Check missing percentage per column, understand pattern (MCAR, MAR, MNAR)

**Deletion**

- **Listwise deletion**: remove rows with missing values (if missing % is small)
- **Column deletion**: drop feature if too many values missing (e.g., >50%)

**Imputation**

- **Numerical**: mean, median (robust to outliers), mode
- **Categorical**: mode, "Unknown" category
- **Model-based**: KNN imputation, regression imputation, MICE (Multiple Imputation by Chained Equations)
- **Forward/backward fill**: for time-series data

**Advanced approaches**

- **Indicator variable**: add binary flag column marking whether value was missing (preserves missingness signal)
- **Algorithms that handle missing values natively**: XGBoost, LightGBM, CatBoost

**Considerations**

- Choose method based on why data is missing and how much
- Avoid data leakage — impute using only training set statistics

### Q4. How do you handle Outliers?

**Handling Outliers**

**Detection**

- **Statistical**: Z-score (>3 std dev), IQR method (below Q1−1.5×IQR or above Q3+1.5×IQR)
- **Visualization**: Box plots, scatter plots, histograms
- **Model-based**: Isolation Forest, DBSCAN, LOF (Local Outlier Factor)

**Handling methods**

- **Removal**: Drop outliers if clearly errors/noise (use cautiously — may lose valid info)
- **Capping/Winsorization**: Cap values at a percentile threshold (e.g., 1st/99th percentile)
- **Transformation**: Log, square root, Box-Cox — reduces impact of extreme values
- **Imputation**: Replace with median/mean
- **Robust algorithms**: Use models less sensitive to outliers (e.g., median-based, tree-based models, Huber loss)
- **Separate modeling**: Treat outliers as distinct segment if meaningful (e.g., fraud)

**Considerations**

- Understand if outlier is genuine signal (e.g., fraud) vs error before removing
- Domain knowledge crucial in deciding how to treat

### Q5. Explain Feature Scaling. Why is it needed?

**Feature Scaling**

- Process of transforming features to a common scale/range
- Needed because features with larger ranges can dominate distance-based or gradient-based algorithms

**Why needed**

- Distance-based algorithms (KNN, K-Means, SVM) treat larger-scale features as more "important"
- Gradient descent converges faster/more stably when features are on similar scales
- Regularization (L1/L2) penalizes weights unfairly if features aren't scaled

**Common techniques**

- **Standardization (Z-score)**: (x − mean) / std → mean=0, std=1; handles outliers better
- **Normalization (Min-Max)**: (x − min) / (max − min) → scales to \[0,1\]; sensitive to outliers
- **Robust Scaling**: uses median and IQR → robust to outliers

**Not needed for**

- Tree-based algorithms (Decision Trees, Random Forest, XGBoost) — splits are scale-invariant

### Q6. One-Hot, Label, Target, and K-Fold Target Encoding

**Categorical Encoding Methods**

**One-Hot Encoding**

- Binary column per category (1/0)
- No ordinal assumption; good for low-cardinality, nominal data
- Can cause high dimensionality with many categories

**Label Encoding**

- Assigns an integer to each category (e.g., Red=0, Blue=1, Green=2)
- Compact, but implies a false ordinal relationship
- Works well for tree-based models; risky for linear/distance-based models

**Target Encoding**

- Replaces category with mean of target variable for that category
- Handles high cardinality well, captures predictive signal
- Risk: **data leakage**/overfitting (target info leaks into features)

**K-Fold Target Encoding**

- Fixes leakage by computing target encoding using out-of-fold data (K-fold CV approach)
- For each fold, encode using target means computed from the other K−1 folds
- Reduces overfitting risk compared to naive target encoding
- More robust for high-cardinality categorical features in production models

### Q7. How do you handle Categorical Features?

**Handling Categorical Features**

**Encoding methods**

- **One-Hot Encoding**: low-cardinality, nominal data
- **Label Encoding**: ordinal data or tree-based models
- **Target/Mean Encoding**: high-cardinality features (use K-fold to avoid leakage)
- **Frequency/Count Encoding**: replace category with its frequency count
- **Binary Encoding**: converts categories to binary code (fewer columns than one-hot)
- **Embedding (Entity Embeddings)**: for deep learning, learns dense vector representation per category

**High-cardinality strategies**

- Group rare categories into "Other"
- Use hashing trick (feature hashing)
- Target encoding with regularization/smoothing

**Considerations**

- Choice depends on: cardinality, algorithm type (tree vs linear/distance-based), presence of order
- Always fit encoders on training data only (avoid data leakage)
- Handle unseen categories at inference time (e.g., map to "Unknown"/default)

### Q8. Explain Feature selection vs. feature extraction.

**Feature Selection vs Feature Extraction**

| Aspect | Feature Selection | Feature Extraction |
| --- | --- | --- |
| Definition | Selects subset of original features | Creates new features by transforming original ones |
| Output | Original features (unchanged) | New, transformed features |
| Interpretability | High (features remain original) | Lower (new features may not be interpretable) |
| Examples | Filter, Wrapper, Embedded methods | PCA, LDA, Autoencoders |
| Goal | Remove irrelevant/redundant features | Reduce dimensionality while preserving info |

**Feature Selection methods**

- **Filter**: statistical tests (correlation, chi-square, ANOVA) — independent of model
- **Wrapper**: uses model performance to select features (e.g., RFE — Recursive Feature Elimination)
- **Embedded**: selection happens during training (e.g., L1/Lasso, tree feature importance)

**Feature Extraction methods**

- PCA, LDA, t-SNE, Autoencoders, Kernel PCA

### Q9. How would you create new features from existing ones?

**Creating New Features (Feature Engineering)**

**Mathematical transformations**

- Ratios, differences, sums (e.g., price/sqft, income−expenses)
- Log, square root, power transforms (handle skewness)
- Polynomial features (x², x³, interaction terms x1×x2)

**Date/time features**

- Extract day, month, year, weekday, quarter, hour
- Time since event, is\_weekend, is\_holiday
- Cyclical encoding (sin/cos) for periodic features like hour/month

**Text features**

- Word count, character count, sentiment score
- TF-IDF, n-grams, embeddings

**Aggregation features**

- Group-by statistics: mean, sum, count, std per category (e.g., avg purchase per customer)
- Rolling window stats for time-series (moving average, lag features)

**Binning/Discretization**

- Convert continuous → categorical buckets (e.g., age groups)

**Domain-specific features**

- Business logic-driven combinations (e.g., BMI from height/weight)
- Interaction features between related variables

**Encoding-based**

- Combine categorical features (e.g., city + product\_type)

### Q10. How do you approach a dataset with highly imbalanced classes?

**Handling Imbalanced Classes**

**Data-level approaches**

- **Oversampling**: duplicate/synthesize minority class samples (SMOTE, ADASYN)
- **Undersampling**: reduce majority class samples (random, Tomek links, NearMiss)
- **Combination**: SMOTE + undersampling (e.g., SMOTEENN)

**Algorithm-level approaches**

- **Class weights**: penalize misclassification of minority class more (`class_weight='balanced'`)
- **Cost-sensitive learning**: custom loss functions weighting errors differently
- **Ensemble methods**: Balanced Random Forest, EasyEnsemble, RUSBoost

**Evaluation approach**

- Avoid accuracy (misleading) — use Precision, Recall, F1, PR-AUC, ROC-AUC
- Confusion matrix to understand error types
- Use stratified k-fold CV to preserve class ratio in splits

**Other strategies**

- Anomaly detection framing (if extreme imbalance, e.g., fraud)
- Collect more minority class data if possible
- Threshold tuning (adjust decision threshold instead of default 0.5)

### Q11. How do you select features for a model?

**Feature Selection Approaches**

**Filter methods** (statistical, model-independent)

- Correlation with target (Pearson, Spearman)
- Chi-square test (categorical features)
- ANOVA F-test
- Mutual Information
- Variance Threshold (remove low-variance features)

**Wrapper methods** (model performance-based)

- Recursive Feature Elimination (RFE)
- Forward selection / Backward elimination
- Exhaustive search (computationally expensive)

**Embedded methods** (selection during training)

- L1/Lasso regularization (shrinks coefficients to zero)
- Tree-based feature importance (Random Forest, XGBoost)
- SHAP values for feature contribution

**Practical considerations**

- Remove highly correlated/redundant features (multicollinearity)
- Domain knowledge to guide relevant feature choice
- Cross-validate feature selection to avoid overfitting/leakage
- Balance interpretability vs performance needs

### Q12. Why and how do you split data into a train, test, and validation set?

**Train, Validation, Test Split**

**Why**

- **Train**: used to fit model parameters
- **Validation**: used to tune hyperparameters, select the best model, and prevent overfitting to the test set
- **Test**: final unbiased evaluation of model performance on unseen data
- Prevents overfitting and gives a realistic estimate of real-world performance

**How**

- Common split ratio: 60/20/20 or 70/15/15 or 80/10/10 (depends on dataset size)
- **Random split**: shuffle, then split (for i.i.d. data)
- **Stratified split**: preserves class distribution (important for imbalanced classification)
- **Time-based split**: for time-series — train on past, test on future (no random shuffling, avoids leakage)
- **K-Fold Cross-Validation**: alternative to fixed validation set — rotates validation fold across K splits (uses train+val more efficiently)

**Key rules**

- Test set should only be used once, at the very end
- No data leakage — preprocessing (scaling, encoding) is fit only on the train set and applied to val/test
- Larger datasets can use smaller validation/test proportions (still enough absolute samples)
