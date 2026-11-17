# Chapter 10. Data Preprocessing and Feature Engineering

_Part III — Machine Learning_

Missing values, outliers, encoding, scaling, imbalance and feature selection, plus leakage, temporal, event-log, text and geospatial features, training-serving consistency and data validation: the work that usually decides model quality.

This chapter contains **38 questions**.

## Contents

- **10.1** [What is Feature Engineering?](#101-what-is-feature-engineering)
- **10.2** [What is one-hot encoding? When should you use it?](#102-what-is-one-hot-encoding-when-should-you-use-it)
- **10.3** [How do you deal with missing data?](#103-how-do-you-deal-with-missing-data)
- **10.4** [How do you handle Outliers?](#104-how-do-you-handle-outliers)
- **10.5** [Explain Feature Scaling. Why is it needed?](#105-explain-feature-scaling-why-is-it-needed)
- **10.6** [Compare one-hot, label, target and K-fold target encoding.](#106-compare-one-hot-label-target-and-k-fold-target-encoding)
- **10.7** [How do you handle Categorical Features?](#107-how-do-you-handle-categorical-features)
- **10.8** [Explain Feature selection vs. feature extraction.](#108-explain-feature-selection-vs-feature-extraction)
- **10.9** [How would you create new features from existing ones?](#109-how-would-you-create-new-features-from-existing-ones)
- **10.10** [How do you select features for a model?](#1010-how-do-you-select-features-for-a-model)
- **10.11** [Why and how do you split data into a train, test, and validation set?](#1011-why-and-how-do-you-split-data-into-a-train-test-and-validation-set)
- **10.12** [How do you deal with noisy data in machine learning models?](#1012-how-do-you-deal-with-noisy-data-in-machine-learning-models)
- **10.13** [What functions are used to perform categorical conversion into factor?](#1013-what-functions-are-used-to-perform-categorical-conversion-into-factor)
- **10.14** [Do you believe treating a categorical variable as a continuous variable will result in a better predictive model?](#1014-do-you-believe-treating-a-categorical-variable-as-a-continuous-variable-will-result-in-a-better-predictive-model)
- **10.15** [How does a Box-Cox transformation occur?](#1015-how-does-a-box-cox-transformation-occur)
- **10.16** [What functions can be used to identify, drop, and replace missing or corrupted values?](#1016-what-functions-can-be-used-to-identify-drop-and-replace-missing-or-corrupted-values)
- **10.17** [Imagine you are working with a laptop with 2 GB RAM; how would you process a dataset of 10GB?](#1017-imagine-you-are-working-with-a-laptop-with-2-gb-ram-how-would-you-process-a-dataset-of-10gb)
- **10.18** [What are some ways to make your model more robust to outliers?](#1018-what-are-some-ways-to-make-your-model-more-robust-to-outliers)
- **10.19** [How do you approach a categorical feature with high cardinality?](#1019-how-do-you-approach-a-categorical-feature-with-high-cardinality)
- **10.20** [How will you do feature selection using Lasso Regression?](#1020-how-will-you-do-feature-selection-using-lasso-regression)
- **10.21** [What is the difference between standard scaler and minmax scaler? What will you do if there is a categorical variable?](#1021-what-is-the-difference-between-standard-scaler-and-minmax-scaler-what-will-you-do-if-there-is-a-categorical-variable)
- **10.22** [Why is it sometimes needed to scale or normalise features?](#1022-why-is-it-sometimes-needed-to-scale-or-normalise-features)
- **10.23** [Why does one-hot encoding improve machine learning performance?](#1023-why-does-one-hot-encoding-improve-machine-learning-performance)
- **10.24** [Differentiate between wide and tall data formats?](#1024-differentiate-between-wide-and-tall-data-formats)
- **10.25** [How can you determine which features are the most important in your model?](#1025-how-can-you-determine-which-features-are-the-most-important-in-your-model)
- **10.26** [Can you define the concept of Undersampling and Oversampling?](#1026-can-you-define-the-concept-of-undersampling-and-oversampling)
- **10.27** [Which type of sampling is better for a classification model, and why?](#1027-which-type-of-sampling-is-better-for-a-classification-model-and-why)
- **10.28** [How would you handle severe class imbalance?](#1028-how-would-you-handle-severe-class-imbalance)
- **10.29** [What is Binning?](#1029-what-is-binning)
- **10.30** [What is data leakage in feature engineering, and how do you detect and prevent it?](#1030-what-is-data-leakage-in-feature-engineering-and-how-do-you-detect-and-prevent-it)
- **10.31** [How do you engineer features for time series and temporal data?](#1031-how-do-you-engineer-features-for-time-series-and-temporal-data)
- **10.32** [How do you create features from event logs and transactional data?](#1032-how-do-you-create-features-from-event-logs-and-transactional-data)
- **10.33** [How do you turn text into features for classical machine learning models?](#1033-how-do-you-turn-text-into-features-for-classical-machine-learning-models)
- **10.34** [How do you keep preprocessing consistent between training and serving?](#1034-how-do-you-keep-preprocessing-consistent-between-training-and-serving)
- **10.35** [How do you engineer features from geospatial data?](#1035-how-do-you-engineer-features-from-geospatial-data)
- **10.36** [What is multicollinearity, how do you detect it, and when does it matter?](#1036-what-is-multicollinearity-how-do-you-detect-it-and-when-does-it-matter)
- **10.37** [What is adversarial validation, and how do you use it to detect distribution shift?](#1037-what-is-adversarial-validation-and-how-do-you-use-it-to-detect-distribution-shift)
- **10.38** [How do you validate data quality for ML features before training and serving?](#1038-how-do-you-validate-data-quality-for-ml-features-before-training-and-serving)

---

### 10.1 What is Feature Engineering?

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

### 10.2 What is one-hot encoding? When should you use it?

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

### 10.3 How do you deal with missing data?

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
- **Interpolation**: linear or time-based, for continuous and time-series data

**Advanced approaches**

- **Indicator variable**: add binary flag column marking whether value was missing (preserves missingness signal)
- **Algorithms that handle missing values natively**: XGBoost, LightGBM, CatBoost

**Considerations**

- Choose method based on why data is missing and how much
- Avoid data leakage — impute using only training set statistics

### 10.4 How do you handle Outliers?

**Handling Outliers**

An **outlier** is an observation that differs markedly from the rest of the data. It may be a data entry or measurement error, a rare but genuine event, or natural variation. Outliers strongly affect statistics and models based on means, squared errors or distances.

**Detection**

- **Statistical**: Z-score ($\lvert z \rvert > 3$ std dev), IQR method (below $Q_1 - 1.5 \times \text{IQR}$ or above $Q_3 + 1.5 \times \text{IQR}$)
- **Visualization**: Box plots, scatter plots, histograms
- **Model-based**: Isolation Forest, DBSCAN, LOF (Local Outlier Factor), One-Class SVM

**Handling methods**

- **Removal**: Drop outliers if clearly errors/noise (use cautiously — may lose valid info)
- **Capping/Winsorization**: Cap values at a percentile threshold (e.g., 1st/99th percentile)
- **Transformation**: Log, square root, Box-Cox — reduces impact of extreme values
- **Imputation**: Replace with median/mean
- **Robust algorithms**: Use models less sensitive to outliers (e.g., median-based, tree-based models, Huber loss)
- **Separate modeling**: Treat outliers as distinct segment if meaningful (e.g., fraud)

**Considerations**

- Understand if outlier is genuine signal (e.g., fraud) vs error before removing: correct or remove errors, keep genuine extremes
- Domain knowledge crucial in deciding how to treat
- Compare model performance with and without the treatment to measure its real impact
- Fit thresholds (percentiles, IQR bounds) on training data only, and reuse them at inference

### 10.5 Explain Feature Scaling. Why is it needed?

**Feature Scaling**

- Process of transforming features to a common scale/range
- Needed because features with larger ranges can dominate distance-based or gradient-based algorithms

**Why needed**

- Distance-based algorithms (KNN, K-Means, SVM) treat larger-scale features as more "important"
- Gradient descent converges faster/more stably when features are on similar scales
- Regularization (L1/L2) penalizes weights unfairly if features aren't scaled

**Common techniques**

- **Standardization (Z-score)**: $\frac{x - \text{mean}}{\text{std}}$ → $\text{mean} = 0$, $\text{std} = 1$; handles outliers better
- **Normalization (Min-Max)**: $\frac{x - \min}{\max - \min}$ → scales to $[0, 1]$; sensitive to outliers
- **Robust Scaling**: uses median and IQR → robust to outliers

**Not needed for**

- Tree-based algorithms (Decision Trees, Random Forest, XGBoost) — splits are scale-invariant

### 10.6 Compare one-hot, label, target and K-fold target encoding.

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
- For each fold, encode using target means computed from the other $K - 1$ folds
- Reduces overfitting risk compared to naive target encoding
- More robust for high-cardinality categorical features in production models

### 10.7 How do you handle Categorical Features?

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

### 10.8 Explain Feature selection vs. feature extraction.

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

### 10.9 How would you create new features from existing ones?

**Creating New Features (Feature Engineering)**

**Mathematical transformations**

- Ratios, differences, sums (e.g., $\text{price} / \text{sqft}$, $\text{income} - \text{expenses}$)
- Log, square root, power transforms (handle skewness)
- Polynomial features ($x^2$, $x^3$, interaction terms $x_1 \times x_2$)

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

### 10.10 How do you select features for a model?

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

### 10.11 Why and how do you split data into a train, test, and validation set?

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

### 10.12 How do you deal with noisy data in machine learning models?

Separate **feature noise** (measurement errors, outliers, missing values) from **label noise** (wrong annotations). Label noise is usually more damaging.

**Diagnose first:**
- Profile the data: distributions, impossible values, duplicates, nulls.
- Measure annotator agreement (Cohen's kappa) on a re-labelled sample.
- Use **confident learning** (cleanlab) or high-loss examples during training to surface likely mislabelled samples, then review them.

**Feature noise:**
- Validation rules and data contracts upstream (Great Expectations / dbt tests) so that garbage is caught at the source.
- Robust scaling, winsorising or clipping outliers, log transforms; median instead of mean imputation, plus missingness indicator features.
- Smoothing and aggregation (rolling windows) for sensor and time-series noise.
- Tree-based models are naturally fairly robust to outliers.

**Label noise:**
- Fix the labelling process: clearer guidelines, gold questions, consensus or majority vote, expert adjudication.
- Robust losses (MAE, Huber, generalized cross-entropy, symmetric CE), label smoothing.
- Down-weight or drop suspected noisy samples. Co-teaching or ensembles that disagree flag the doubtful ones.
- Weak supervision (Snorkel) models the accuracy of each noisy source explicitly.

**Modelling:** regularisation, early stopping (networks memorise noise late in training), and ensembles.

**Key principle (data-centric AI):** cleaning 5% of the labels often beats a month of model tuning. Keep a small, very clean **test set**, otherwise you cannot even measure progress.

### 10.13 What functions are used to perform categorical conversion into factor?

#### Functions Used to Convert Categorical Data into Factors

In **R**, the `factor()` function is used to convert categorical data into a **factor**.

- **`factor()`** — converts a categorical variable into a factor.
- **`as.factor()`** — converts an existing variable into a factor.
- **`ordered()`** — creates an **ordered factor** when categories have a meaningful order.

**Example:**

```txt
gender<-factor(c("Male","Female","Male"))
```

**Why convert categorical variables into factors?** A factor tells R that the values are categories with a fixed set of levels, not numeric quantities. Modelling functions such as `lm()` and `glm()` then create the right dummy variables automatically, summaries count by level, and ordered factors record a natural ranking (Low, Medium, High). In Python, the equivalent is pandas' `category` dtype, which also saves memory and is used natively by LightGBM and CatBoost.

### 10.14 Do you believe treating a categorical variable as a continuous variable will result in a better predictive model?

- **Usually, no.** Treating a categorical variable as continuous does not necessarily produce a better predictive model.
- A categorical variable represents **groups or categories**, not numerical quantities.
- If categories are converted directly into numbers, the model may incorrectly assume **order and distance** between them.
- For example, assigning **Low = 1, Medium = 2, High = 3** assumes the difference between Low and Medium is the same as between Medium and High.
- Instead, use appropriate encoding methods such as **One-Hot Encoding** or **Ordinal Encoding**, depending on the nature of the categories.

### 10.15 How does a Box-Cox transformation occur?

#### Box-Cox Transformation

- **Box-Cox transformation** is a statistical technique used to **transform data so that it is closer to a normal distribution**.
- It is mainly used when the data is **skewed** or when the variance is not constant.
- It works by applying a **power transformation** to the data.
- The transformation uses a parameter called **lambda ($\lambda$)**.
- Different values of $\lambda$ produce different transformations.
- The value of $\lambda$ is usually chosen to find the transformation that gives the **best approximation to normality**.
- Box-Cox requires the data to contain **positive values**.

**Common transformations:**

- $\lambda = 1$ → No major transformation
- $\lambda = 0$ → Log transformation
- $\lambda = 0.5$ → Square-root transformation
- $\lambda = -1$ → Reciprocal transformation

**Simple terms:**

**Box-Cox changes the scale of the data using a suitable power so that the data becomes more normally distributed and has more stable variance.**

### 10.16 What functions can be used to identify, drop, and replace missing or corrupted values?

#### Functions to Handle Missing or Corrupted Values

If you are asking about **Python/Pandas**, common functions are:

- **Identify missing values:**
  - `isnull()` / `isna()` → identifies missing values.
  - `notnull()` / `notna()` → identifies non-missing values.
- **Drop missing values:**
  - `dropna()` → removes rows or columns containing missing values.
- **Replace missing values:**
  - `fillna()` → replaces missing values with a specified value, such as the **mean, median, or mode**.
- **Replace corrupted/incorrect values:**
  - `replace()` → replaces specific incorrect values with correct values.

**Example:**

```py
df.isnull()              # Identify missing values
df.dropna()              # Drop missing values
df.fillna(0)             # Replace missing values with 0
df.replace(-999, None)   # Replace corrupted value
```

**Simple terms:**

**`isnull()` → identify, `dropna()` → drop, `fillna()` → fill, `replace()` → replace.**

### 10.17 Imagine you are working with a laptop with 2 GB RAM; how would you process a dataset of 10GB?

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

### 10.18 What are some ways to make your model more robust to outliers?

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

### 10.19 How do you approach a categorical feature with high cardinality?

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

### 10.20 How will you do feature selection using Lasso Regression?

Lasso adds an L1 penalty, $\min_\beta \frac{1}{2n}\lVert y - X\beta\rVert^2 + \alpha\lVert\beta\rVert_1$. The L1 term drives the coefficients of uninformative features **exactly to zero**, so the features with non-zero coefficients are the selected set.

**Steps:**
1. **Standardize the features.** The penalty is scale-dependent, and unscaled features get penalized unevenly.
2. **Choose $\alpha$ by cross-validation** (`LassoCV`, or `LogisticRegression(penalty="l1")` for classification). A larger $\alpha$ means fewer features. Use the "1-SE rule" (the largest $\alpha$ within one standard error of the best CV score) if you want a sparser, more robust set.
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

### 10.21 What is the difference between standard scaler and minmax scaler? What will you do if there is a categorical variable?

**StandardScaler** (z-score): $x' = \frac{x - \mu}{\sigma}$, giving zero mean and unit variance with an unbounded range.
**MinMaxScaler**: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$, mapping to a fixed range, usually $[0,1]$.

| | StandardScaler | MinMaxScaler |
| --- | --- | --- |
| Output range | Unbounded (mostly $-3$ to $3$) | Bounded $[0, 1]$ |
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

### 10.22 Why is it sometimes needed to scale or normalise features?

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

### 10.23 Why does one-hot encoding improve machine learning performance?

One-hot encoding turns a categorical feature with $k$ values into $k$ binary columns. It helps because:

- **Removes a fake ordering.** Label encoding (red=0, green=1, blue=2) tells a linear model or a distance-based model that blue is "twice" green and that green lies between red and blue. One-hot treats categories as unordered and equidistant.
- **Lets linear models learn a separate weight per category.** With label encoding, a linear model can only learn one slope across the codes, i.e. a monotonic effect. With one-hot, each category gets its own coefficient, so the model can capture arbitrary, nonlinear effects of the category. It effectively increases model capacity in the right way.
- **Works with any algorithm that expects numeric input** without introducing bias from arbitrary integer codes.

**When it doesn't help or hurts:**

- **High cardinality** (zip codes, user IDs): thousands of sparse columns, memory blow-up and overfitting. Prefer target/mean encoding (with out-of-fold fitting to avoid leakage), hashing, frequency encoding or learned embeddings.
- **Tree-based models** can often split label-encoded categories fine, and LightGBM/CatBoost handle categoricals natively, often better than one-hot.
- **Dummy-variable trap** in unregularized linear regression: drop one column (`drop='first'`) to avoid perfect collinearity with the intercept.
- In production, handle unseen categories (`handle_unknown='ignore'`).

### 10.24 Differentiate between wide and tall data formats?

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

### 10.25 How can you determine which features are the most important in your model?

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

### 10.26 Can you define the concept of Undersampling and Oversampling?

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

### 10.27 Which type of sampling is better for a classification model, and why?

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

### 10.28 How would you handle severe class imbalance?

#### Handling Severe Class Imbalance

For example, fraud detection where **0.1% = fraud** and **99.9% = normal**.

1. **Use the right metrics**
   - Avoid relying on accuracy.
   - Use **Precision, Recall, F1, PR-AUC**, and confusion matrix.
   - ROC-AUC can be useful but may look overly optimistic with extreme imbalance.
2. **Class weighting**
   Give the minority class higher loss:

   ```py
   LogisticRegression(class_weight="balanced")
   ```

   For neural networks:

   ```py
   loss=WeightedCrossEntropy(class_weights)
   ```
3. **Resampling**
   - **Oversampling:** Random oversampling, SMOTE
   - **Undersampling:** Reduce majority samples
   - Combine both when appropriate.
4. **Threshold tuning**
   Don't automatically use `threshold = 0.5`.

   For fraud detection, you might lower the threshold to improve recall:

   ```txt
   probability > 0.2 → fraud
   ```
5. **Stratified splitting**
   Preserve class proportions:

   ```py
   train_test_split(X,y,stratify=y)
   ```
6. **Use appropriate loss functions**<br>For deep learning, **Focal Loss** is useful because it focuses more on difficult/minority examples.
7. **Use cost-sensitive learning**<br>If a false negative costs much more than a false positive, explicitly incorporate those business costs into the objective.
8. **Use imbalance-aware ensembles**<br>Balanced Random Forest, EasyEnsemble and RUSBoost undersample differently for each model, which reduces the information lost by undersampling.
9. **Reframe the problem when positives are extremely rare**<br>Anomaly detection, or collecting more minority-class examples (for example through targeted labelling), can work better than any resampling trick.

### 10.29 What is Binning?

**Binning** is the process of grouping continuous values into discrete intervals or "bins."

Example:

```txt
Age:
0–18
19–30
31–45
46–60
60+
```

#### Common types

**Equal-width binning**

Each bin has the same numerical range.

**Equal-frequency binning**

Each bin contains approximately the same number of observations.

#### Applications

- Feature engineering
- Data preprocessing
- Histograms
- Outlier handling
- Risk scoring
- Calibration analysis

Binning can simplify noisy continuous variables, but excessive binning can cause **information loss** and introduce artificial boundaries.

### 10.30 What is data leakage in feature engineering, and how do you detect and prevent it?

**Data leakage** happens when training data contains information that will **not be available at prediction time**. The model looks excellent offline and then fails in production. It is one of the most common and most expensive mistakes in applied ML, and interviewers often probe it with case studies.

**Types of leakage, with examples**

| Type | Example | Why it leaks |
| --- | --- | --- |
| Target leakage | Predicting loan default using "number of collection calls" | Collection calls happen *because* of default |
| Temporal leakage | Using a customer's 30-day spend computed over a window that ends after the prediction date | Uses the future |
| Preprocessing leakage | Fitting a scaler, imputer, target encoder or feature selector on the full dataset before splitting | Test statistics flow into training |
| Duplicate or entity leakage | The same patient, user or near-duplicate image in both train and test | The model memorises the entity, not the pattern |
| Proxy and ID leakage | A record ID, file path or timestamp that correlates with the label because of how data was collected (for example, all positive scans came from one hospital) | Learns the collection process, not the signal |
| Label-definition leakage | A churn label defined by "account closed", with a feature "account status" | The feature is the label in disguise |

**Warning signs**

- Validation performance that is suspiciously high, or much higher than published results or simple baselines.
- One feature dominating importance far more than domain intuition suggests.
- A large gap between offline metrics and live performance.

**Prevention**

- **Point-in-time correctness:** for each training example, define a prediction timestamp and build every feature using only data available before it. Feature stores with as-of joins enforce this.
- **Ask of every feature:** "Would I have this value, with this exact freshness, at the moment of prediction?"
- **Split first, then fit:** put all preprocessing (imputation, scaling, encoding, selection, resampling) inside a pipeline fitted on training folds only.
- **Split by time and by entity:** time-based splits for anything temporal; group-aware splits (GroupKFold) when entities repeat.
- **Deduplicate** across splits, including near-duplicates.
- **Shadow-test before launch:** run the model on live traffic and compare live performance with offline estimates before trusting it.

### 10.31 How do you engineer features for time series and temporal data?

Temporal features are often the most predictive in business problems (demand, churn, fraud, maintenance), and also the easiest place to leak the future. Every feature must be computed **as of the prediction time**.

**Feature families**

- **Lags:** the value 1, 7, 28 or 364 days ago. Choose lags that match the forecast horizon: to forecast 14 days ahead, the most recent usable lag is 14 days.
- **Rolling window statistics:** mean, median, standard deviation, minimum, maximum, count and quantiles over the last 7, 28 and 90 days, computed only on data before the prediction time.
- **Trends and changes:** differences and ratios between windows (last 7 days divided by previous 28 days), slopes, and exponentially weighted moving averages.
- **Calendar:** day of week, month, week of year, holidays and the days before and after them, paydays, school terms, promotions and events.
- **Cyclical encoding:** for hour of day or day of year, use $\sin(2\pi t / P)$ and $\cos(2\pi t / P)$ so that 23:00 and 00:00 are close. Fourier terms at several frequencies capture complex seasonality.
- **Time since events:** days since the last purchase, last login, last failure, last price change.
- **Counts within windows:** number of transactions in the last hour, distinct merchants in the last day.
- **Static and hierarchical context:** product category, store size and region, so that a global model can share patterns across series.

**Common pitfalls**

- **Leakage through windows:** a centred rolling mean or a window that includes the target day.
- **Train and serve mismatch:** offline features computed with complete, corrected data, while online features see late-arriving or incomplete data. Simulate production latency when building training data.
- **Random cross-validation:** always use time-based backtesting (rolling origin).
- **Too many highly correlated windows:** they add little and increase overfitting; select a few windows per signal.

```python
import numpy as np
df = df.sort_values(["store_id", "date"])
g = df.groupby("store_id")["sales"]
df["lag_7"] = g.shift(7)
df["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())  # shift(1) excludes today
df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
```

### 10.32 How do you create features from event logs and transactional data?

Much business data arrives as **events**: clicks, transactions, logins, support tickets, sensor readings. Models usually predict at the level of an **entity** (customer, account, device) at a point in time, so the core task is aggregating many events into one row per entity and prediction time.

**Approach**

1. **Define the entity and the prediction time** (for example, each customer on the first day of each month).
2. **Define windows** ending at the prediction time (last 1, 7, 30, 90 days).
3. **Aggregate** events within each window.

**Useful aggregate families**

- **Counts and sums:** number of transactions, total spend, number of logins.
- **Recency, frequency, monetary value (RFM):** days since last event, events per period, average value.
- **Distribution statistics:** mean, median, maximum, standard deviation, share of weekend or night activity.
- **Diversity:** number of distinct products, merchants, devices, countries.
- **Ratios and trends:** this month against the previous three months, refunds divided by purchases, failed divided by total logins.
- **Sequences:** the last N event types, time between events, or embeddings learned from event sequences.
- **Category breakdowns:** spend per product category as shares of total spend.

**Scaling it**

- Compute aggregates in SQL or Spark with window functions or pre-aggregated daily tables (daily summaries rolled up into longer windows are much cheaper than scanning raw events each time).
- For real-time models, maintain streaming aggregates (Flink, Spark Streaming) in an online store, with the same definitions used offline.
- Tools such as Featuretools (deep feature synthesis) can generate many aggregations automatically, but generated features still need selection and leakage review.

**Pitfalls:** windows that include events after the prediction time; events that arrive late (a refund recorded days later); entities with no events (decide between zero and missing, and add indicators); and very many generated features that overfit.

```python
import pandas as pd

# snapshots: one row per (customer_id, as_of_date); transactions: customer_id, txn_id, txn_date, amount, merchant_id
joined = snapshots.merge(transactions, on="customer_id", how="left")
in_window = (joined["txn_date"] < joined["as_of_date"]) & (
    joined["txn_date"] >= joined["as_of_date"] - pd.Timedelta(days=30))   # strictly before the prediction time
window = joined[in_window]

feats = (window.groupby(["customer_id", "as_of_date"])
         .agg(txn_cnt_30d=("txn_id", "count"),
              spend_30d=("amount", "sum"),
              merchants_30d=("merchant_id", "nunique"),
              last_txn_date=("txn_date", "max"))
         .reset_index())
feats["days_since_last_txn"] = (feats["as_of_date"] - feats["last_txn_date"]).dt.days

# keep customers with no events in the window: counts become 0, recency stays missing
features = snapshots.merge(feats.drop(columns="last_txn_date"), on=["customer_id", "as_of_date"], how="left")
features[["txn_cnt_30d", "spend_30d", "merchants_30d"]] = features[["txn_cnt_30d", "spend_30d", "merchants_30d"]].fillna(0)
```

### 10.33 How do you turn text into features for classical machine learning models?

Text fields (product descriptions, support tickets, reviews, notes) often carry strong signal for tabular models. The options trade simplicity, cost and quality.

| Method | How it works | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Bag of words and TF-IDF (with n-grams) | Sparse vector of term weights | Fast, strong baseline, interpretable | No semantics, large sparse matrices |
| Hashing vectoriser | Hashes terms into a fixed number of columns | Bounded memory, works on streams, no vocabulary to store | Collisions, not interpretable |
| Hand-crafted features | Length, number of digits, presence of keywords, sentiment, readability | Cheap, interpretable, domain knowledge | Limited coverage |
| Pretrained embeddings | Encode text with a sentence embedding model, use the vector as features | Captures meaning and synonyms, few labels needed | Dense vectors, inference cost, less interpretable |
| LLM-extracted attributes | Use an LLM to extract structured fields (category, intent, urgency) | Turns free text into clean categorical features | Cost, latency, needs validation |

**Practical guidance**

- Start with TF-IDF and a linear model as a baseline; it is often surprisingly hard to beat on keyword-driven tasks.
- For gradient-boosted trees, high-dimensional sparse TF-IDF works poorly; reduce it (truncated SVD) or use embeddings, or feed a separate text model's predicted probability as a single stacked feature (computed out-of-fold to avoid leakage).
- With embeddings, consider reducing dimensions (PCA) before mixing with tabular features, and check that the embedding model suits the domain and language.
- Fit vocabularies, IDF weights and any reducers on training data only.
- Monitor text features for drift: new products, new slang and template changes shift vocabularies quickly.

### 10.34 How do you keep preprocessing consistent between training and serving?

**Training and serving skew** happens when the features a model sees in production are computed differently from those it was trained on. The model silently degrades, often without errors. Preventing it is mostly an engineering discipline.

**Common causes**

- Preprocessing written twice: in Python notebooks for training and in another language or service for serving.
- Statistics (means for imputation, scaling parameters, category vocabularies) recomputed at serving time on different data.
- Different handling of missing values, unseen categories, time zones or units.
- Features available offline with full history but computed online from partial or delayed data.

**Prevention**

- **Package preprocessing with the model.** In scikit-learn, put every step in a `Pipeline` with a `ColumnTransformer` and save it as one artefact; in deep learning, include preprocessing layers in the model graph or export it with the model.
- **Fit once, reuse everywhere:** learned parameters (scaler means, encoder vocabularies, clipping thresholds) are fitted on training data and loaded at serving, never recomputed.
- **Handle the unexpected explicitly:** unseen categories (`handle_unknown="ignore"` or an "other" bucket), missing values, and out-of-range inputs.
- **Single feature definitions** for offline and online computation through a feature store or shared library.
- **Log served features** and use the logged values for future training, which removes a whole class of skew.
- **Detect skew:** compare distributions of training features with served features (statistics, PSI), and periodically recompute offline features for a sample of served requests and compare them value by value.
- **Version** preprocessing code, fitted artefacts and the model together, and deploy them as one unit.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
])
model = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
joblib.dump(model, "model_v3.joblib")   # one artefact: preprocessing + model
```

### 10.35 How do you engineer features from geospatial data?

Location drives many outcomes: delivery times, prices, fraud risk, demand, insurance risk. Raw latitude and longitude are rarely the best features on their own.

**Feature types**

- **Raw coordinates:** usable by tree models, which can split on regions, but linear models cannot use them meaningfully. Rotated coordinates (for example at 15 and 30 degrees) help trees find diagonal boundaries.
- **Distances:** to key points (city centre, nearest store, warehouse, airport), using haversine distance for great-circle distance, or road-network travel time from a routing engine where it matters.
- **Spatial indexes as categories:** geohash or **H3** hexagon cells at a chosen resolution. Use them as categorical features with target or frequency encoding, or as join keys for aggregates.
- **Neighbourhood aggregates:** counts and statistics within a radius or cell: number of competitors, population density, average house price, historical demand, crime rates.
- **Spatial clusters:** k-means or DBSCAN on coordinates to create region labels.
- **External context by location:** weather, demographics, points of interest from open data.

**Pitfalls**

- **Spatial leakage:** neighbouring points are highly correlated, so random splits overstate performance. Use spatial block cross-validation (hold out whole regions or cells).
- **Target encoding of small cells** overfits; smooth towards the parent cell's average and fit out-of-fold.
- **Coordinate quality:** default values (0, 0), swapped latitude and longitude, and low-precision geocoding.
- **Privacy:** precise locations can identify individuals; aggregate or coarsen where possible.

```python
import numpy as np
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))
```

### 10.36 What is multicollinearity, how do you detect it, and when does it matter?

**Multicollinearity** means two or more features are highly correlated, so one can be predicted well from the others (for example, total spend, number of orders, and average order value).

**When it matters**

- **Interpretation of linear models:** coefficients become unstable and their standard errors large. A coefficient can flip sign between samples, and individual coefficients cannot be interpreted as "the effect of this feature holding others fixed". This matters for statistical inference, credit scorecards and any explanation given to regulators or stakeholders.
- **Feature importance:** importance is split arbitrarily among correlated features, so each looks less important than the underlying signal. Permutation importance can understate them, and SHAP values spread credit across them.
- **Numerical stability** in unregularised linear regression when correlation is near perfect.

**When it matters less**

- **Pure prediction:** if the goal is accuracy and the correlation structure is stable in production, multicollinearity usually does not hurt predictive performance much, especially with regularisation or tree-based models.

**Detection**

- **Correlation matrix** for pairwise relationships (misses relationships involving three or more features).
- **Variance inflation factor (VIF):** for feature $j$, $\text{VIF}_j = \frac{1}{1 - R_j^2}$, where $R_j^2$ comes from regressing feature $j$ on all the others. Values above about 5 to 10 indicate a problem.
- **Condition number** of the feature matrix.

**Remedies**

- Remove or combine redundant features (keep the most interpretable, or build a ratio or index).
- **Ridge regularisation** stabilises coefficients; Lasso picks one feature from a correlated group (somewhat arbitrarily); Elastic Net keeps groups together.
- **PCA** or other dimensionality reduction, at the cost of interpretability.
- For importance analysis, group correlated features and compute importance per group.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns)}
```

### 10.37 What is adversarial validation, and how do you use it to detect distribution shift?

**Adversarial validation** checks whether the training data and another dataset (test set, recent production data, a new market) come from the same distribution, by training a classifier to tell them apart.

**Method**

1. Combine the two datasets, labelling rows from the training data 0 and rows from the other data 1 (drop the real target).
2. Train a classifier (gradient-boosted trees work well) with cross-validation to predict the label.
3. Look at the ROC-AUC:
   - **About 0.5:** the classifier cannot tell them apart; the distributions are similar.
   - **Well above 0.5:** there is a detectable shift.
4. Inspect **feature importance** of the classifier to see which features differ most.

**What to do with the result**

- **Find leaks and artefacts:** IDs, timestamps or row indexes that differ between sets often top the importance list; remove them.
- **Fix or drop drifting features** that would behave differently in production (a feature computed differently after a system change).
- **Build a better validation set:** select training rows that the classifier thinks look most like the test data, so validation reflects the target distribution.
- **Reweight training data:** use the classifier's probabilities as importance weights, $w = \frac{p}{1 - p}$, to emphasise training rows that resemble the target distribution (covariate shift correction).
- **Monitor production:** run it periodically between training data and recent traffic as a drift check that captures multivariate changes that per-feature statistics miss.

**Caveats:** it detects shift in features (covariate shift), not changes in the relationship between features and target (concept drift); and some shift is expected and harmless, so judge it by its impact on model performance.

### 10.38 How do you validate data quality for ML features before training and serving?

Bad data causes more production ML failures than bad models. Data validation turns assumptions about data into **automated checks** that run in pipelines, so problems are caught before they reach training or predictions.

**What to check**

| Category | Examples |
| --- | --- |
| Schema | Expected columns present, correct types, no unexpected new columns |
| Completeness | Null rate per column within expected bounds; row counts within expected range |
| Validity | Values within ranges (age 0 to 120), allowed categories, valid formats (emails, dates), units |
| Uniqueness and keys | No duplicate primary keys; joins do not multiply rows |
| Consistency | Cross-field rules (end date after start date; totals equal sum of parts) |
| Freshness | Latest timestamp within the expected delay |
| Distribution | Mean, quantiles and category frequencies compared with a reference; PSI or statistical tests for drift |
| Label quality | Label rate within expected range; label delay; annotation agreement |

**Where to run checks**

- **At ingestion and in transformation pipelines** (data contracts with source teams, dbt tests, Great Expectations, Pandera, TFX Data Validation).
- **Before training:** block training if the dataset fails critical checks, and record a data quality report with each model version.
- **At serving:** validate incoming requests (schema, ranges, missing fields) and either reject, impute with logged warnings, or fall back to a simpler model.
- **Continuously in production:** monitor feature distributions and missing rates against training data, with alerts.

**Good practice**

- Separate **blocking** checks (schema broken, keys duplicated) from **warning** checks (moderate distribution shift), so pipelines do not stop for harmless changes.
- Derive initial expectations by profiling historical data, then refine them with domain experts.
- Every incident caused by bad data should become a new automated check.
- Assign owners: each critical dataset needs someone responsible for fixing failed checks.

```python
import pandera as pa
schema = pa.DataFrameSchema({
    "age": pa.Column(int, pa.Check.in_range(0, 120)),
    "country": pa.Column(str, pa.Check.isin(["NP", "IN", "US", "GB"])),
    "amount": pa.Column(float, pa.Check.ge(0), nullable=False),
})
schema.validate(df)
```

---

← [Chapter 9. Machine Learning Algorithms](09-machine-learning-algorithms.md) · [Table of contents](../README.md) · [Chapter 11. Optimization and Training Dynamics](11-optimization-and-training-dynamics.md) →
