# Turing 100 interview questions

Answers to the Turing list of ~100 common ML / data-science interview questions.

### Q1. Differentiate between Training Sets and Test Sets?
| Aspect | Training Set | Test Set |
| --- | --- | --- |
| **Purpose** | Train the model | Evaluate the model |
| **Data** | Seen by the model | Unseen by the model |
| **Used for** | Learning parameters/patterns | Measuring generalization |
| **Model exposure** | During training | Only after training |
| **Typical split** | 80% | 20% |
| **Main concern** | Overfitting | Data leakage |

### Q2. Define Bias and Variance.
| Aspect | **Bias** | **Variance** |
| --- | --- | --- |
| **Definition** | Error from overly simplistic assumptions | Error from being too sensitive to training data |
| **Model behavior** | **Underfitting** | **Overfitting** |
| **Training error** | High | Low |
| **Test error** | High | High |
| **Example** | Linear model for a complex nonlinear problem | Very complex model fitting noise |
| **Goal** | Reduce bias by increasing model complexity | Reduce variance using regularization/more data |

**In short:**

- **High Bias → Underfitting**
- **High Variance → Overfitting**

### Q3. You have found missing data in your dataset. How will you handle it?
| Approach | When to Use |
| --- | --- |
| **Delete rows** | Few missing values and data is not important |
| **Mean/Median imputation** | Missing numerical values |
| **Mode imputation** | Missing categorical values |
| **Forward/Backward fill** | Time-series data |
| **Interpolation** | Continuous/time-series data |
| **Model-based imputation** | Large or complex missing patterns |
| **Missing indicator** | When missingness itself may contain information |

**Best practice:** First understand **why the data is missing (MCAR, MAR, or MNAR)**, then choose the appropriate method. Avoid blindly replacing all missing values with the mean.

### Q4. Explain Decision Tree Classification.
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

### Q5. How is a logistic regression model evaluated?
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

### Q6. To start Linear Regression, you would need to make some assumptions. What are those assumptions?
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

### Q7. What is multicollinearity and how will you handle it in your regression model?
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

### Q8. Explain why the performance of XGBoost is better than that of SVM?
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

### Q9. Why is an encoder-decoder model used for NLP?
#### Why Use an Encoder-Decoder Model for NLP?

- **Encoder:** Converts the input sequence into a meaningful **contextual representation**.
- **Decoder:** Generates the output sequence **token by token** based on that representation.
- **Handles variable-length sequences:** Input and output can have different lengths.
- **Captures context:** The encoder understands relationships between words in the input.
- **Suitable for sequence-to-sequence tasks:** Such as **machine translation, text summarization, and question answering**.
- **Example:**`English sentence → Encoder → Context representation → Decoder → French sentence`

### Q10. What are Machine Learning and Artificial Intelligence?
#### Artificial Intelligence (AI) vs. Machine Learning (ML)

| **Aspect** | **Artificial Intelligence (AI)** | **Machine Learning (ML)** |
| --- | --- | --- |
| **Definition** | A broader field focused on creating systems that perform tasks requiring human-like intelligence. | A subset of AI where systems learn patterns from data to make predictions or decisions. |
| **Goal** | Simulate capabilities such as reasoning, planning, perception, and decision-making. | Learn from data and improve performance without explicitly programming every rule. |
| **Relationship** | **AI is the broader concept.** | **ML is a subset of AI.** |
| **Example** | Autonomous systems, AI assistants, game-playing systems. | Spam detection, house-price prediction, recommendation systems. |

### Q11. Differentiate between Deep Learning and Machine Learning?
#### Deep Learning vs. Machine Learning

| **Aspect** | **Machine Learning (ML)** | **Deep Learning (DL)** |
| --- | --- | --- |
| **Definition** | Algorithms that learn patterns from data | A subset of ML using multi-layer neural networks |
| **Feature Engineering** | Usually requires manual feature engineering | Automatically learns features from raw data |
| **Data Requirement** | Generally works well with smaller datasets | Usually requires large datasets |
| **Computational Power** | Lower computational requirements | Higher computational requirements, often GPUs |
| **Model Examples** | Linear Regression, Decision Trees, SVM, XGBoost | CNN, RNN, Transformer, Autoencoder |
| **Best suited for** | Structured/tabular data | Unstructured data such as images, audio, and text |
| **Interpretability** | Often more interpretable | Generally more difficult to interpret |

### Q12. What is cross validation?
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

### Q13. What are the types of Machine learning?
#### Types of Machine Learning

| **Type** | **Definition** | **Examples** |
| --- | --- | --- |
| **Supervised Learning** | Learns from **labeled data** to predict outputs. | Classification, Regression |
| **Unsupervised Learning** | Finds patterns or structures in **unlabeled data**. | Clustering, Dimensionality Reduction |
| **Semi-Supervised Learning** | Uses a combination of **labeled and unlabeled data**. | Image and text classification |
| **Reinforcement Learning** | Learns by interacting with an environment using **rewards and penalties**. | Robotics, Game Playing, Autonomous Systems |

### Q14. Differentiate between Supervised and Unsupervised learning.

> 🔁 _See also [02-fundamentals-of-machine-learning.md → Q16](../ml-engineering/02-fundamentals-of-machine-learning.md#q16-differentiate-between-supervised-and-unsupervised-learning) (separate notes on the same question)._
#### Supervised vs. Unsupervised Learning

| **Aspect** | **Supervised Learning** | **Unsupervised Learning** |
| --- | --- | --- |
| **Data** | Uses **labeled data** | Uses **unlabeled data** |
| **Goal** | Predict a target/output | Discover hidden patterns or structure |
| **Training** | Learns from input-output pairs | Learns from relationships within the data |
| **Main Tasks** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Examples** | Linear Regression, Decision Tree, XGBoost, SVM | K-Means, DBSCAN, PCA |
| **Evaluation** | Accuracy, F1, RMSE, etc. | Silhouette Score, clustering analysis, etc. |

### Q15. What is Selection Bias?
#### Selection Bias

- **Selection bias** occurs when the data selected for a study or model is **not representative of the target population**.
- It happens when certain groups are **systematically overrepresented or underrepresented**.
- This can lead to **biased conclusions and poor generalization**.

**Example:**

If you survey customer satisfaction only among customers who voluntarily respond online, the results may not represent all customers.

**How to reduce it:**

- Use **random sampling** where possible.
- Ensure the sample represents the target population.
- Monitor for **underrepresented groups**.
- Use appropriate sampling or weighting techniques.

### Q16. What is the difference between Correlation, Covariance, and Causality?
#### **Correlation vs Covariance vs Causality**

| Aspect | **Covariance** | **Correlation** | **Causality** |
| --- | --- | --- | --- |
| **Meaning** | Measures whether two variables move together | Measures the **strength and direction** of a linear relationship | Means a change in one variable **causes** a change in another |
| **Direction** | Positive / Negative | Positive / Negative | Cause → Effect |
| **Strength** | Difficult to interpret because scale-dependent | Easy to interpret, from `-1` to `+1` | Not represented by a simple fixed range |
| **Range** | `(-∞, +∞)` | `[-1, +1]` | No fixed range |
| **Units** | Depends on units of X and Y | Unitless | Depends on the causal effect being measured |
| **Standardized?** | No | Yes | Not simply a statistical standardization |
| **Does it prove cause-effect?** | No | **No** | **Yes**, when supported by valid causal evidence |
| **Typical use** | Understand joint variation | Measure linear association | Determine whether changing X produces a change in Y |

For example, whether **increasing advertising expenditure actually causes an increase in sales**, rather than simply observing that advertising and sales are correlated.

#### Simple Example

Suppose we observe:

> **Ice cream sales ↑** and **drowning incidents ↑**

- **Covariance:** Positive → both tend to increase together.
- **Correlation:** Positive → there is a positive linear association.
- **Causality:** ❌ Ice cream sales do not cause drowning. A **confounding variable**, such as hot weather, can increase both.

#### Key Relationship

**Covariance → standardized → Correlation**

But:

**Correlation ≠ Causality**

### Q17. What is the difference between supervised and reinforcement learning?
#### Supervised Learning vs Reinforcement Learning

| Aspect | **Supervised Learning** | **Reinforcement Learning (RL)** |
| --- | --- | --- |
| **Learning from** | Labeled data | Interaction with an environment |
| **Feedback** | Correct target/label | Reward or penalty |
| **Goal** | Learn a mapping `X → Y` | Learn actions that maximize cumulative reward |
| **Training data** | Input-output pairs | States, actions, rewards, next states |
| **Decision making** | Usually no sequential decision-making | Sequential decision-making |
| **Learning signal** | Prediction error/loss | Reward signal |
| **Main concept** | Minimize prediction error | Maximize expected cumulative reward |
| **Examples** | Image classification, fraud detection, price prediction | Robotics, game playing, autonomous decision-making |
| **Typical algorithms** | Linear Regression, Decision Trees, XGBoost, Neural Networks | Q-Learning, DQN, Policy Gradient, PPO |

### Q18. What are the requirements of reinforcement learning environments?
#### Requirements of a Reinforcement Learning Environment

A reinforcement learning (RL) environment should provide the components necessary for an **agent to observe, act, and receive feedback**.

| Component | Description |
| --- | --- |
| **State** `S` | Represents the current situation of the environment |
| **Action** `A` | Set of actions the agent can take |
| **Reward** `R` | Feedback received after taking an action |
| **Transition** | Defines how the environment changes from one state to another |
| **Policy** `π(a\|s)` | Strategy used by the agent to choose actions |
| **Initial State** | Defines where an episode starts |
| **Terminal State** | Defines when an episode ends |
| **Dynamics** | Rules governing state transitions and rewards |

#### Markov Decision Process (MDP)

Most RL environments are modeled as an **MDP**:

`M = (S, A, P, R, γ)`

where:

- `S` = set of states
- `A` = set of actions
- `P(s'|s,a)` = transition probability
- `R(s,a)` = reward function
- `γ` = discount factor

The environment essentially performs:

`State → Action → Reward + Next State`

#### Example: Robot Navigation

- **State:** Robot's current position and surroundings
- **Actions:** Move forward, backward, left, right
- **Reward:** `+10` for reaching the target, `1` for each step
- **Transition:** Robot moves to a new position
- **Terminal state:** Target reached or maximum steps exceeded

### Q19. What different targets do classification and regression algorithms require?
The main difference is **the type of target (output) variable they predict**:

| Aspect | Classification | Regression |
| --- | --- | --- |
| **Target variable** | **Categorical / discrete** | **Continuous / numerical** |
| **Goal** | Predict a **class or category** | Predict a **numeric value** |
| **Example target** | `Spam` / `Not Spam` | House price = `$250,000` |
| **Output** | Class label or probability | Real-valued number |
| **Common algorithms** | Logistic Regression, Decision Tree, Random Forest, SVM, Neural Networks | Linear Regression, Decision Tree, Random Forest, SVR, Neural Networks |
| **Common metrics** | Accuracy, Precision, Recall, F1, ROC-AUC | MAE, MSE, RMSE, R² |

#### Simple examples

**Classification:**

> Predict whether a customer will **churn or not**.

Target: y ∈ \{0, 1\}

**Regression:**

> Predict a customer's **annual spending**.

Target: y ∈ R

### Q20. What are five popular algorithms used in Machine Learning?
Five popular Machine Learning algorithms are:

1. **Linear Regression** – Predicts continuous numerical values.
2. **Logistic Regression** – Used mainly for binary classification.
3. **Decision Trees** – Used for both classification and regression.
4. **Random Forest** – An ensemble of multiple decision trees; works for classification and regression.
5. **Support Vector Machines (SVM)** – Mainly used for classification, but also supports regression (SVR).

**Easy way to remember:**

**Linear Regression → Logistic Regression → Decision Tree → Random Forest → SVM**.

### Q21. What is the confusion matrix?

> 🔁 _See also [02-fundamentals-of-machine-learning.md → Q30](../ml-engineering/02-fundamentals-of-machine-learning.md#q30-what-is-confusion-matrix) (separate notes on the same question)._
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

### Q22. List the differences between KNN and k-means clustering.
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

### Q23. What are the differences between Type I error and Type II error?
#### Type I Error vs Type II Error

| Aspect | **Type I Error (α)** | **Type II Error (β)** |
| --- | --- | --- |
| **Meaning** | Rejecting a **true null hypothesis** | Failing to reject a **false null hypothesis** |
| **Also called** | **False Positive** | **False Negative** |
| **Occurs when** | We conclude there is an effect when there isn't one | We conclude there is no sufficient evidence of an effect when one actually exists |
| **Probability** | α (significance level) | β |
| **Related concept** | Significance level | Statistical power = 1−β1-\\beta |

#### Simple example

Suppose we test whether a new medicine works:

- **Type I Error:** We conclude the medicine works, but actually it **doesn't**. → **False Positive**
- **Type II Error:** We conclude there isn't enough evidence that it works, but actually it **does**. → **False Negative**

#### Easy way to remember

> **Type I = False Alarm** 🚨
>
> **Type II = Missed Detection** 🎯

**In short:** Type I error means **detecting something that isn't there**, while Type II error means **missing something that is there**.

### Q24. What is semi-supervised learning?
#### Semi-Supervised Learning

**Semi-supervised learning** is a Machine Learning approach that uses a **small amount of labeled data** together with a **large amount of unlabeled data** to train a model.

#### How it works

For example, suppose we have **10,000 images**:

- **500 images** → labeled (e.g., cat/dog)
- **9,500 images** → unlabeled

The model learns from the labeled examples and uses the structure of the unlabeled data to improve its predictions.

#### Comparison

| Learning Type | Labeled Data | Unlabeled Data |
| --- | --- | --- |
| **Supervised** | ✅ Large amount | ❌ Usually not required |
| **Unsupervised** | ❌ None | ✅ Large amount |
| **Semi-supervised** | ✅ Small amount | ✅ Large amount |

#### Common techniques

- **Pseudo-labeling**
- **Self-training**
- **Consistency regularization**
- **Label propagation**

#### Example

In **medical image classification**, labeling thousands of X-rays can be expensive because it requires experts. A model can use a smaller set of expert-labeled X-rays along with many unlabeled X-rays.

### Q25. Where is semi-supervised learning applied?
#### Applications of Semi-Supervised Learning

Semi-supervised learning is useful when **labeled data is expensive or difficult to obtain**, but large amounts of unlabeled data are available.

Common applications include:

1. **Medical Imaging** – Classifying X-rays, MRI scans, and CT scans with limited expert-labeled data.
2. **Image Classification** – Classifying large collections of images when only some are labeled.
3. **Speech Recognition** – Learning from a small amount of transcribed speech and large amounts of untranscribed audio.
4. **Natural Language Processing (NLP)** – Text classification, sentiment analysis, and document categorization with limited labeled text.
5. **Fraud Detection** – Detecting unusual transactions when only a small portion of transactions have confirmed labels.
6. **Web Content Classification** – Categorizing webpages or online content using limited labeled examples.
7. **Cybersecurity** – Detecting malicious or abnormal network activity when labeled attack data is limited.

### Q26. Difference between Stemming vs Lemmatization?

> 🔁 _See also [07-nlp.md → Q14](../ml-engineering/07-nlp.md#q14-what-is-stemming-vs-lemmatization) (separate notes on the same question)._
#### Stemming vs Lemmatization

Both **stemming** and **lemmatization** are NLP techniques used to reduce words to their **base or root form**, but they work differently.

| Aspect | Stemming | Lemmatization |
| --- | --- | --- |
| **Approach** | Removes prefixes/suffixes using rules | Uses vocabulary and linguistic analysis |
| **Output** | May produce an invalid word | Produces a valid dictionary word |
| **Accuracy** | Generally lower | Generally higher |
| **Speed** | Faster | Slower |
| **Context** | Usually does not consider context | Can consider context and part of speech |
| **Example** | `studies → studi` | `studies → study` |
| **Another example** | `running → run` | `running → run` |

#### Example

**Stemming:**

> `connection`, `connected`, `connecting` → `connect`

It simply removes word endings according to predefined rules.

**Lemmatization:**

> `better` → `good`

It uses linguistic knowledge to identify the correct **lemma** (base dictionary form).

### Q27. What is a PCA?
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

### Q28. What are support vectors in SVM (Support Vector Machine)?
- **Support vectors** are the data points closest to the decision boundary in an SVM.
- They determine the **position and orientation of the optimal decision boundary**.
- They define the **margin** between the classes.
- They are the most important training points for the SVM classifier.

### Q29. In terms of access, how are array and linked lists different?
- **Array:** Provides **direct/random access** to elements using an index, typically in constant time.
- **Linked List:** Provides **sequential access**; you must traverse nodes from the beginning or another known node.
- **Key difference:** Arrays are faster for accessing elements by index, while linked lists are slower because they require traversal.

### Q30. What is P-value?
- **P-value** is the probability of obtaining a result **at least as extreme as the observed result**, assuming the **null hypothesis (H₀) is true**.
- In simple term: **How surprising is my result if the null hypothesis is actually true?**
- It helps determine whether the observed result provides enough evidence to **reject the null hypothesis**.
- A **small p-value** indicates stronger evidence against H₀.
- Common significance level: **α = 0.05**

  - **p-value \< 0.05:** Reject H₀
  - **p-value ≥ 0.05:** Fail to reject H₀
- A p-value **does not** represent the probability that the null hypothesis is true.
- It is **not** the probability that your test result is a random fluke.

### Q31. What techniques are used to find resemblance in the recommendation system?
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

### Q32. What is the difference between Regression and Classification?

> 🔁 _See also [02-fundamentals-of-machine-learning.md → Q19](../ml-engineering/02-fundamentals-of-machine-learning.md#q19-what-is-the-difference-between-classification-and-regression) (separate notes on the same question)._
#### Regression vs Classification

| Regression | Classification |
| --- | --- |
| Predicts a **continuous numerical value** | Predicts a **category or class** |
| Output can be any numerical value | Output belongs to a fixed set of classes |
| Example: Predict house **price** | Example: Predict whether a customer will **churn or not** |
| Common algorithms: Linear Regression, Ridge, Lasso | Common algorithms: Logistic Regression, Decision Tree, SVM |
| Metrics: MAE, MSE, RMSE, R² | Metrics: Accuracy, Precision, Recall, F1-score |

**Simple way to remember:**

- **Regression → "How much?"**<br>Example: How much will the house cost? → \$250,000
- **Classification → "Which category?"**<br>Example: Is the email spam? → Yes/No

### Q33. What does the area under the ROC curve indicate?
- **AUC (Area Under the ROC Curve)** indicates how well a classification model can **distinguish between positive and negative classes**.
- AUC ranges from **0 to 1**:
  - **1.0** → Perfect classifier
  - **0.5** → Random guessing
  - **Below 0.5** → Worse than random
- **Higher AUC means better ability to separate the classes** across different classification thresholds.

**In simple terms:** AUC tells us **how well the model separates the two classes**.

### Q34. What is a neural network?
- A **Neural Network** is a machine learning model inspired by the way the human brain processes information.
- It consists of interconnected **neurons (nodes)** organized into layers:
  - **Input layer** → receives the input data.
  - **Hidden layers** → learn patterns and relationships from the data.
  - **Output layer** → produces the final prediction or result.
- During training, the network **adjusts its weights** to reduce prediction errors.
- Neural networks are widely used in **image recognition, natural language processing, speech recognition, recommendation systems, and generative AI**.

**In simple terms:** A neural network learns patterns from data and uses those patterns to make predictions.

### Q35. What is an Outlier?
- An **outlier** is a data point that is **significantly different from most other observations** in a dataset.
- It may occur due to:
  - **Data entry or measurement errors**
  - **Unusual but genuine events**
  - **Natural variation in the data**
- Outliers can **strongly affect statistical analysis and machine learning models**, especially models based on averages or distances.

**Example:** If most students score between 60 and 90, but one student scores 5, that score may be an **outlier**.

**In simple terms:** An outlier is a **data point that unusually differs from the rest of the data**.

### Q36. What is another name for a Bayesian Network?
- A **Bayesian Network** is also called a **Bayes Network**, **Bayesian Belief Network (BBN)**, Casual network, Bayes net, Belief Propagation Network or **Belief Network**.
- It is a **probabilistic graphical model** that represents relationships and conditional dependencies between variables.

### Q37. What is ensemble learning?
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

### Q38. What is clustering?
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

### Q39. How would you define collinearity?
- **Collinearity** occurs when **two or more independent variables in a dataset are highly correlated with each other**.
- In a regression model, this can make it difficult to determine the **individual effect of each independent variable** on the target variable.
- High collinearity can lead to:
  - Unstable model coefficients
  - Large standard errors
  - Difficulty interpreting the importance of individual features
- **Example:** If a regression model uses both **age** and **years of experience**, these variables may be highly correlated, creating collinearity.

### Q40. What is overfitting?

> 🔁 _See also [01-ai-engineering.md → Q8](../ml-engineering/01-ai-engineering.md#q8-what-is-overfitting) (separate notes on the same question)._
- **Overfitting** occurs when a machine learning model learns the **training data too closely**, including its noise and random patterns, instead of learning general patterns.
- As a result:
  - **Training performance** → Very high
  - **Test/validation performance** → Poor
- An overfitted model performs well on data it has seen but **fails to generalize to new, unseen data**.
- Common causes include:
  - Model is **too complex**
  - Training for **too many iterations/epochs**
  - **Too little training data**
  - Too many irrelevant features
- Common ways to reduce overfitting:
  - Use **regularization**
  - Collect more training data
  - Reduce model complexity
  - Use **early stopping**
  - Use **cross-validation**
  - Apply **dropout** in neural networks

### Q41. What is the Bayesian Network?
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

### Q42. What is the time series?
- A **time series** is a sequence of data points recorded **over time**, usually at regular intervals.
- The observations are ordered chronologically, such as **daily, weekly, monthly, or yearly**.
- Time series data is used to understand **patterns, trends, and changes over time**.
- Common components of a time series include:
  - **Trend** → Long-term upward or downward movement.
  - **Seasonality** → Repeating patterns at regular intervals.
  - **Cyclical patterns** → Longer-term fluctuations that do not necessarily repeat at fixed intervals.
  - **Noise/Random variation** → Unpredictable fluctuations.
- Common applications include:
  - **Stock price forecasting**
  - **Sales and demand forecasting**
  - **Weather forecasting**
  - **Electricity consumption prediction**
  - **Economic forecasting**
- Common time-series models include **ARIMA, SARIMA, Exponential Smoothing, Prophet, and LSTM**.

### Q43. What is dimension reduction in ML?
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

### Q44. What is underfitting?

> 🔁 _See also [01-ai-engineering.md → Q9](../ml-engineering/01-ai-engineering.md#q9-what-is-underfitting) (separate notes on the same question)._
- **Underfitting** occurs when a machine learning model is **too simple to learn the important patterns** in the training data.
- It is associated with **high bias and low variance**.
- As a result:
  - **Training performance** → Poor
  - **Test/validation performance** → Poor
- Common causes include:
  - Model is **too simple**
  - Insufficient training
  - Too few or poor-quality features
  - Excessive regularization
- Ways to reduce underfitting:
  - Use a **more complex model**
  - Add relevant features
  - Train for longer
  - Reduce excessive regularization
  - Use a more suitable algorithm

**In simple terms:** Underfitting means the model **has high bias and is too simple to capture the important patterns in the data**.

### Q45. What is sensitivity (Recall)?
- **Sensitivity**, also called **Recall** or **True Positive Rate (TPR)**
- How many of the actual positives did I recall/find?
- TP / (TP + FN)
- High sensitivity means the model produces **fewer false negatives**

**Example:**

If there are 100 actual positive cases and the model correctly finds 90 of them, the

**Recall/Sensitivity = 90%**

### Q46. What is specificity?
- **Specificity**, also called **True Negative Rate (TNR)**
- **How many of the actual negatives did I** find?
- **TN / (TN + FP)**
- High specificity means the model produces **fewer false positives**.

**Example:**

If there are 100 actual negative cases and the model correctly identifies 90 of them, the **Specificity = 90%**.

### Q47. What are the differences between stochastic gradient descent (SGD) and gradient descent (GD)?
- **radient Descent (GD)** and **Stochastic Gradient Descent (SGD)** are optimization algorithms used to **minimize the loss function** by updating model parameters.

| Feature | Gradient Descent (GD) | Stochastic Gradient Descent (SGD) |
| --- | --- | --- |
| Data used per update | **Entire dataset** | **One training example** |
| Updates | One update after processing all data | Update after each example |
| Speed per update | Slower | Faster |
| Memory usage | Higher | Lower |
| Convergence | More stable/smooth | More noisy/fluctuating |
| Large datasets | Computationally expensive | More suitable |
| Local minima | Can get stuck | Noise can sometimes help escape local minima |

#### Gradient Descent

- Calculates the gradient using **all training examples**.
- Then updates the model parameters.
- Gives a more **accurate and stable** direction toward the minimum.
- Can be slow when the dataset is very large.

#### Stochastic Gradient Descent

- Calculates the gradient using **one randomly selected training example at a time**.
- Updates the model parameters immediately after each example.
- Much faster for large datasets.
- Because each update uses only one example, the optimization path is **noisy** and may fluctuate around the minimum.

**In simple terms:**

- **GD:** Look at **all data → calculate gradient → update model**.
- **SGD:** Look at **one data point → calculate gradient → update model → repeat**.

### Q48. What is an Array?
- An **Array** is a data structure that stores **multiple elements of the same type** in a **contiguous block of memory**.
- Each element is accessed using an **index**.
- In most programming languages, array indexing starts from **0**.
- Arrays provide **fast access to elements** because an element can be directly accessed using its index.
- The size of a traditional array is usually **fixed** when it is created.
- Common operations include:
  - **Access** → Get an element using its index.
  - **Update** → Change an element.
  - **Search** → Find an element.
  - **Insertion/Deletion** → Add or remove elements, which may require shifting other elements.

**Example:**

`[10, 20, 30, 40]`

- Index `0` → `10`
- Index `1` → `20`
- Index `2` → `30`
- Index `3` → `40`

### Q49. What is a linked list?
- A **Linked List** is a data structure made up of a sequence of **nodes**, where each node contains:
  - **Data** → Stores the actual value.
  - **Pointer/Reference** → Stores the address/reference of the next node.
- Unlike arrays, linked-list nodes **do not need to be stored in contiguous memory locations**.
- The first node is called the **head**, and the last node usually points to **NULL**.
- Common types include:
  - **Singly Linked List** → Each node points to the next node.
  - **Doubly Linked List** → Each node points to both the previous and next nodes.
  - **Circular Linked List** → The last node points back to the first node.
- **Insertion and deletion** can be efficient because elements do not need to be shifted, provided you already have the relevant node/reference.
- **Accessing an element by position is slower** than in an array because you generally have to traverse the nodes sequentially.

**Example:**

`10 → 20 → 30 → 40 → NULL`

### Q50. Why is the Naive Bayes Method ‘Naive’?
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

### Q51. State the Bayes’ theorem for Naive Bayes Method.
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

### Q52. What is Static Memory Allocation?
#### Static Memory Allocation

- **Static Memory Allocation** is the process of allocating memory for variables **before the program starts running**.
- The **size of memory is fixed** and generally cannot be changed during program execution.
- Memory is usually allocated in the **stack or static memory area**, depending on the variable.
- It is generally **faster** because the memory size is known in advance.
- Example: Declaring an array with a fixed size, such as `int arr[10]`.

### Q53. How would you define Dynamic Memory Allocation?
#### Dynamic Memory Allocation

- **Dynamic Memory Allocation** is the process of allocating memory **during program execution (runtime)**.
- The amount of memory can be **decided based on the program's needs**.
- Memory is typically allocated from the **heap**.
- It is useful when the required memory size is **not known in advance**.
- The allocated memory can generally be **released when it is no longer needed**.
- Example: Creating an array at runtime based on user input.

### Q54. How would you define Precision and Recall?
#### Precision

- **Precision** measures how precise my positive prediction are?
- Formula: **TP / (TP + FP)**
- High precision means **fewer false positives**.

#### Recall

- **Recall** is also called **Sensitivity**.
- It measures how many of the **actual positives did I recall/find?**
- Formula: **TP / (TP + FN)**
- High recall means **fewer false negatives**.

### Q55. What are some tools used to discover outliers?
Common tools and techniques for detecting outliers include:

- **Box Plot:** Identifies outliers using the distribution of the data.
- **Scatter Plot:** Helps visually identify unusual data points.
- **Z-Score:** Detects observations that are far from the mean.
- **IQR (Interquartile Range):** Identifies values that fall unusually far outside the middle 50% of the data.
- **Histogram:** Helps spot values that are unusually far from the main distribution.
- **Isolation Forest:** A machine learning algorithm specifically designed for anomaly/outlier detection.
- **DBSCAN:** Can identify points that do not belong to any dense cluster as outliers.
- **One-Class SVM:** Detects observations that differ significantly from the normal data pattern.

### Q56. What is the Fourier Transform?
#### Fourier Transform

- **Fourier Transform** is a mathematical technique used to convert a signal from the **time domain** into the **frequency domain**.
- It tells us **which frequencies are present in a signal and how strong they are**.
- It is commonly used for **audio processing, image processing, signal processing, communications, and time-series analysis**.
- For example, an audio signal that looks complex in the time domain can be broken down into its individual **frequency components**.
- The **Fast Fourier Transform (FFT)** is an efficient algorithm used to calculate the Fourier Transform.

### Q57. What is Inductive Logic Programming?
#### Inductive Logic Programming (ILP)

- **Inductive Logic Programming (ILP)** is a branch of **Artificial Intelligence and Machine Learning** that learns **logical rules from examples and background knowledge**.
- It combines **machine learning with logic programming**.
- ILP uses:
  - **Positive examples** — cases the rule should explain.
  - **Negative examples** — cases the rule should not explain.
  - **Background knowledge** — existing facts and relationships.
- From these, ILP automatically learns a **hypothesis (logic program)** that can be used to make predictions on new data.
- It is useful in domains such as **natural language processing, bioinformatics, and software engineering**.

**In simple terms:** ILP **learns logical rules from examples and existing knowledge, then uses those rules to make predictions on new cases**.

### Q58. Explain Kernel SVM.
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

### Q59. What are some types of clustering?
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

### Q60. How would you describe reinforcement learning?
#### Reinforcement Learning (RL)

- **Reinforcement Learning (RL)** is a type of machine learning where an **agent learns by interacting with an environment**.
- The agent takes **actions** and receives **rewards or penalties** based on those actions.
- The goal is to learn a **strategy (policy)** that maximizes the **total reward over time**.
- Unlike supervised learning, RL does **not require labeled input-output pairs**.
- RL involves key components:
  - **Agent:** Learner or decision-maker.
  - **Environment:** The world in which the agent operates.
  - **State:** Current situation of the environment.
  - **Action:** What the agent can do.
  - **Reward:** Feedback received after taking an action.
  - **Policy:** Strategy used to choose actions.

**Example:** In a game, an agent takes actions, receives rewards for good moves, and learns over time which actions lead to better outcomes.

### Q61. What are some methods to improve inference time?
#### Methods to Improve Inference Time

Common methods to reduce **model inference latency** include:

- **Model Quantization:** Reduce model precision, such as FP32 to FP16 or INT8, to make computation faster.
- **Model Pruning:** Remove unnecessary weights or parameters from the model.
- **Knowledge Distillation:** Train a smaller model (student) to reproduce the behavior of a larger model (teacher).
- **Smaller Model Architecture:** Use a model with fewer layers or parameters.
- **Batching:** Process multiple requests together to improve hardware utilization.
- **Caching:** Cache frequently used results or intermediate computations to avoid repeated work.
- **Efficient Hardware:** Use GPUs, TPUs, or specialized inference accelerators.
- **Optimized Inference Engines:** Use tools such as TensorRT, ONNX Runtime, or optimized serving frameworks.
- **Parallelism:** Run independent computations simultaneously.
- **Reduce Input Size:** Use shorter sequences, smaller images, or fewer unnecessary features.

### Q62. What is content-based filtering and collaborative filtering?
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

### Q63. What is deductive learning and inductive learning?
#### Deductive Learning vs Inductive Learning

##### 1. Deductive Learning

- **Deductive learning** starts with **general rules or existing knowledge** and applies them to reach a **specific conclusion**.
- The conclusion is logically derived from the given rules and facts.
- If the rules and facts are correct, the conclusion is guaranteed to be correct.

**Example:**

- Rule: All humans are mortal.
- Fact: Socrates is a human.
- Conclusion: Socrates is mortal.

**In simple terms:** **General rule → Specific conclusion**

##### 2. Inductive Learning

- **Inductive learning** starts with **specific examples or observations** and learns a **general rule or pattern** from them.
- The learned rule is then used to make predictions about new cases.
- The conclusion is **probable**, not guaranteed.

**Example:**

- Observation: Several observed birds can fly.
- Learned pattern: Birds generally can fly.
- Prediction: A new bird may be able to fly.

**In simple terms:** **Specific examples → General rule**

##### Key Difference

| Deductive Learning | Inductive Learning |
| --- | --- |
| General → Specific | Specific → General |
| Applies existing rules | Learns new rules/patterns |
| Conclusion is logically guaranteed | Conclusion is probabilistic |
| Knowledge-driven | Data/example-driven |

### Q64. How would you differentiate data mining from Machine learning?
#### Data Mining vs Machine Learning

##### 1. Data Mining

- **Data mining** is the process of discovering **useful patterns, relationships, trends, and insights** from large datasets.
- It focuses on **extracting valuable information** from existing data.
- Common techniques include **clustering, association rule mining, anomaly detection, and statistical analysis**.
- Example: An online store analyzes purchase data to discover that customers who buy **laptops often buy laptop bags**.

**In simple terms:** **Data mining finds useful patterns and insights from data.**

##### 2. Machine Learning

- **Machine Learning (ML)** is a branch of AI where algorithms **learn patterns from data and use them to make predictions or decisions**.
- It focuses on building models that can **generalize to new, unseen data**.
- Common techniques include **classification, regression, clustering, and reinforcement learning**.
- Example: An online store trains a model to **predict which products a customer is likely to buy**.

**In simple terms:** **Machine learning learns from data to make predictions or decisions.**

##### Key Difference

| Data Mining | Machine Learning |
| --- | --- |
| Discovers patterns and insights | Learns patterns to make predictions/decisions |
| Mainly focuses on **understanding data** | Mainly focuses on **learning and generalization** |
| Often exploratory | Often predictive |
| Uses statistical and computational techniques | Uses algorithms that learn from data |
| Example: Find purchasing patterns | Example: Predict future purchases |

### Q65. Why is the ROC in a curve important?
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

### Q66. Why does overfitting occur in ML?
#### Why Does Overfitting Occur in ML?

- **Overfitting occurs when a model learns the training data too closely**, including its noise and random variations.
- As a result, the model performs **very well on training data but poorly on unseen test data**.
- **Overfitting = Low Bias + High Variance**
- Common causes include:
  - **Model is too complex** for the amount of data available.
  - **Too little training data**.
  - **Too many irrelevant or noisy features**.
  - **Training for too long**, especially in neural networks.
  - **Insufficient regularization**.
- Overfitting means the model has **low training error but high test/validation error**.
- It can be reduced using:
  - More training data
  - Simpler models
  - Regularization
  - Feature selection
  - Cross-validation
  - Early stopping
  - Dropout for neural networks

### Q67. What are the functions of unsupervised learning?
#### Functions of Unsupervised Learning

- **Clustering:** Groups similar data points together without predefined labels.
  - Example: Customer segmentation using K-Means.
- **Dimensionality Reduction:** Reduces the number of features while preserving important information.
  - Example: PCA.
- **Association Rule Learning:** Discovers relationships or patterns between variables.
  - Example: Finding products frequently purchased together.
- **Anomaly Detection:** Identifies unusual or abnormal data points.
  - Example: Detecting fraudulent transactions.
- **Data Exploration:** Helps discover hidden patterns, structures, and relationships in data.
- **Feature Learning:** Automatically learns useful representations or features from raw data.

### Q68. What are some functions of supervised learning?
#### Functions of Supervised Learning

- **Classification:** Predicts a **category or class** from labeled data.
  - Example: Spam vs. Not Spam.
- **Regression:** Predicts a **continuous numerical value**.
  - Example: Predicting house prices.
- **Prediction:** Uses learned patterns from historical data to predict outcomes for **new, unseen data**.
- **Pattern Learning:** Learns the relationship between **input features and known target labels**.
- **Decision Making:** Helps systems make automated decisions based on learned patterns.
  - Example: Credit risk prediction.
- **Anomaly Detection:** Can identify unusual cases when trained with appropriate labeled examples.

### Q69. What are the two components of Bayesian logic?
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

### Q70. How would you describe a Recommender system?
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

### Q71. What functions are used to perform categorical conversion into factor?
#### Functions Used to Convert Categorical Data into Factors

In **R**, the `factor()` function is used to convert categorical data into a **factor**.

- **`factor()`** — converts a categorical variable into a factor.
- **`as.factor()`** — converts an existing variable into a factor.
- **`ordered()`** — creates an **ordered factor** when categories have a meaningful order.

**Example:**

```txt
gender<-factor(c("Male","Female","Male"))
```

### Q72. Are elements stored consecutively in linked lists?
#### Are Elements Stored Consecutively in Linked Lists?

- **No**, elements in a linked list are **not necessarily stored consecutively** in memory.
- Each element is stored in a **node**, and each node contains:
  - The **data**
  - A **pointer/reference** to the next node
- Nodes can be located at **different memory addresses**.
- The pointers/references connect the nodes together.
- This is different from an **array**, where elements are typically stored in consecutive memory locations.

### Q73. What is Regularization in ML?
#### Regularization in Machine Learning

- **Regularization** is a technique used to **reduce overfitting** and improve a model’s ability to **generalize to unseen data**.
- It adds a **penalty for model complexity**, encouraging the model to learn simpler patterns.
- Common types include:
  - **L1 Regularization (Lasso):** Can make some feature weights exactly zero, helping with feature selection.
  - **L2 Regularization (Ridge):** Shrinks model weights toward zero without usually making them exactly zero.
  - **Dropout:** Randomly deactivates some neurons during training in neural networks.
- Regularization helps prevent the model from **memorizing noise in the training data**.

### Q74. Do you consider decision trees to have any advantages or disadvantages? If yes, what are they?
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

### Q75. What do you understand about the exploding gradient technique problem in the back propagation technique?
#### Exploding Gradient Problem

- **Exploding gradients** occur during **backpropagation** when the gradients become **extremely large** as they are propagated backward through the network.
- This is common in **very deep neural networks** and **Recurrent Neural Networks (RNNs)**.
- Large gradients cause the model's **weights to change by very large amounts** during training.
- This can make training **unstable**, causing the loss to become extremely large or even result in **NaN** values.
- Common causes include:
  - Very large weights
  - Deep networks
  - Repeated multiplication of large gradients, especially in RNNs
- Common solutions include:
  - **Gradient clipping:** Limits gradients to a maximum value.
  - **Proper weight initialization**
  - **Lower learning rate**
  - **Normalization techniques**
  - Using architectures such as **LSTM or GRU** for sequence problems.

### Q76. What do you understand about the Bias-Variance Tradeoff?
#### Bias-Variance Tradeoff

- The **Bias-Variance Tradeoff** is the balance between **bias** and **variance** that affects a machine learning model's ability to generalize to unseen data.

##### Bias

- **Bias** is the error caused by a model being **too simple** and making strong assumptions.
- High bias leads to **underfitting**.
- The model performs poorly on both training and test data.

##### Variance

- **Variance** is the error caused by a model being **too sensitive to the training data**.
- High variance leads to **overfitting**.
- The model performs very well on training data but poorly on test data.

##### Tradeoff

- Increasing model complexity generally **reduces bias but increases variance**.
- Decreasing model complexity generally **increases bias but reduces variance**.
- The goal is to find the **right level of complexity** that gives good performance on unseen data.

**In simple terms:**

- **High Bias + Low Variance → Underfitting**
- **Low Bias + High Variance → Overfitting**
- **Balanced Bias + Variance → Good Generalization**

### Q77. How would you describe an F1 score, and how would you use it?
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

### Q78. Explain the difference between the Loss functions and Cost functions.
#### Loss Function vs Cost Function

##### Loss Function

- A **Loss Function** measures the **error for a single training example**.
- It tells us how far the model's prediction is from the actual value.
- It is calculated **for individual data points**.
- Example: The error between the predicted price and actual price for **one house**.

##### Cost Function

- A **Cost Function** measures the **overall error of the model across the entire training dataset**.
- It is usually calculated by taking the **average of the loss values** for all training examples.
- It gives us a single measure of how well the model performs on the whole dataset.
- The model training process tries to **minimize the cost function**.

##### Key Difference

| Loss Function | Cost Function |
| --- | --- |
| Error for **one example** | Overall error for **the dataset** |
| Calculated per data point | Usually aggregated/averaged across data |
| Measures individual prediction error | Measures overall model performance |

**In simple terms:**

**Loss = error for one example**

**Cost = overall/average error across the dataset**

### Q79. How do you handle outlier values?
#### How to Handle Outlier Values

Outliers should not always be removed. The appropriate approach depends on **why the outlier exists and the business context**.

- **Identify the outliers** using methods such as IQR, Z-score, box plots, or Isolation Forest.
- **Investigate the cause:**
  - If it is a **data entry or measurement error**, correct or remove it.
  - If it represents a **genuine observation**, consider keeping it.
- **Remove outliers** when they are clearly invalid or irrelevant.
- **Transform the data** using methods such as logarithmic transformation when appropriate.
- **Cap or Winsorize** extreme values to reduce their influence.
- **Use robust methods** such as median, IQR-based methods, or robust machine learning algorithms that are less sensitive to outliers.
- **Compare model performance** with and without outliers to determine their impact.

**In simple terms:** First **identify and understand the outlier**, then decide whether to **remove, transform, cap, or keep it** based on its cause and impact.

### Q80. What is a random forest, and how does it work?
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

### Q81. What ensemble techniques can be used to aggregate multiple models?
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

### Q82. What methods can be used to find the threshold of a classifier?
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

### Q83. How can you check for the Normality of a dataset?
#### Methods to Check Normality of a Dataset

- **Histogram:** Check whether the data looks approximately **bell-shaped and symmetric**.
- **Q-Q Plot:** If the points are close to a straight line, the data is approximately normally distributed.
- **Shapiro-Wilk Test:** Statistical test commonly used to check normality.
- **Anderson-Darling Test:** Tests how well the data fits a normal distribution.
- **Kolmogorov-Smirnov Test:** Compares the data distribution with a theoretical normal distribution.
- **D’Agostino Skewness Test:** Checks whether the data has **significant skewness**.
- **Skewness and Kurtosis:** Values close to those expected for a normal distribution suggest approximate normality.

### Q84. How can you differentiate between a parametric and a non-parametric model?
#### Parametric vs Non-Parametric Models

- **Parametric model:**
  - Assumes the data follows a specific distribution or relationship.
  - Uses a **fixed number of parameters** to describe the model.
  - Parameters usually have a clear statistical interpretation.
  - Examples: **Linear Regression, Logistic Regression, Naive Bayes**.
- **Non-parametric model:**
  - Does **not assume a specific distribution** for the data.
  - Model complexity can grow with the amount of data.
  - More flexible and can capture complex patterns.
  - Examples: **K-Nearest Neighbors (KNN), Decision Trees, Random Forest, DBSCAN**.

| Feature | Parametric | Non-parametric |
| --- | --- | --- |
| Distribution assumption | Usually required | Usually not required |
| Number of parameters | Fixed | Can grow with data |
| Flexibility | Lower | Higher |
| Data requirement | Usually less | Usually more |
| Examples | Linear Regression, Logistic Regression | KNN, Decision Tree, Random Forest |

### Q85. Do you think logistic regression can be used for more than two classes?
Yes. **Logistic Regression can be extended to handle multiclass classification.**

- **Binary Logistic Regression:** Used for **2 classes**, such as Yes/No.
- **Multinomial Logistic Regression:** Used when there are **3 or more classes** with no natural ordering.
  - Example: Classifying an image as **Cat, Dog, or Horse**.
- **One-vs-Rest (OvR):** Builds one binary logistic regression model for each class and selects the class with the highest probability.
- **Multinomial/Softmax:** Directly calculates probabilities for all classes and selects the class with the highest probability.

### Q86. What differences exist between Softmax and Sigmoid functions?
#### Softmax vs Sigmoid

- **Sigmoid:**
  - Produces an output between **0 and 1**.
  - Mainly used for **binary classification**.
  - Can also be used for **multi-label classification**, where multiple classes can be positive at the same time.
  - Each output is treated independently.
  - Example: **Cat = 0.8, Dog = 0.7** — both can be positive.
- **Softmax:**
  - Produces a probability for **each class**, with all probabilities adding up to **1**.
  - Mainly used for **multi-class classification**.
  - Classes are treated as mutually exclusive.
  - Example: **Cat = 0.7, Dog = 0.2, Horse = 0.1**.

| Feature | Sigmoid | Softmax |
| --- | --- | --- |
| Output | 0 to 1 | 0 to 1 |
| Sum of outputs | Not necessarily 1 | Always 1 |
| Typical use | Binary / Multi-label | Multi-class |
| Classes | Independent | Mutually exclusive |
| Example | Spam / Not Spam | Cat / Dog / Horse |

**Simple terms:**

**Sigmoid → independent probabilities.**

**Softmax → probabilities competing among multiple classes.**

### Q87. How can overfitting be avoided in ML?
- **Use more training data:** Helps the model learn general patterns instead of memorizing the training data.
- **Use a simpler model:** Reduce model complexity to prevent learning noise.
- **Regularization:** Use **L1, L2, or Elastic Net** to penalize overly complex models.
- **Cross-validation:** Use techniques such as **k-fold cross-validation** to check how well the model generalizes.
- **Feature selection:** Remove irrelevant or noisy features.
- **Early stopping:** Stop training when validation performance starts getting worse.
- **Dropout:** Randomly deactivate some neurons during neural-network training.
- **Data augmentation:** Create variations of training data, especially for images and other structured data.
- **Pruning:** Remove unnecessary branches or parameters from complex models such as decision trees or neural networks.
- **Ensembling:** Combine multiple models to improve generalization and reduce the risk of relying too heavily on one model.

### Q88. Which is better to have? A false positive or false negative?
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

### Q89. How would you handle a dataset suffering from high variance?
High variance usually means the **model is overfitting** — it performs very well on training data but poorly on unseen data.

- **Use more training data:** Helps the model learn general patterns.
- **Use a simpler model:** Reduce model complexity.
- **Use regularization:** Apply **L1, L2, or Elastic Net**.
- **Feature selection:** Remove irrelevant or noisy features.
- **Cross-validation:** Use **k-fold cross-validation** to evaluate generalization.
- **Early stopping:** Stop training when validation performance starts getting worse.
- **Dropout:** Useful for neural networks to reduce overfitting.
- **Pruning:** Reduce unnecessary branches in decision trees.
- **Ensembling:** Techniques like **Bagging and Random Forest** can reduce variance.

#### High Variance and Low Bias

A model with **high variance and low bias** is usually **overfitting** the training data.

To handle it:

- **Use more training data:** Helps the model learn general patterns instead of memorizing the training data.
- **Reduce model complexity:** Use a simpler model or reduce the number of parameters.
- **Apply regularization:** Use **L1, L2, or Elastic Net** to control model complexity.
- **Use feature selection:** Remove irrelevant or noisy features.
- **Use cross-validation:** Helps check whether the model generalizes well to unseen data.
- **Early stopping:** Stop training when validation performance starts getting worse.
- **Use dropout:** Useful for neural networks to reduce overfitting.
- **Use pruning:** Reduce unnecessary branches in decision trees.
- **Use ensembling:** Methods such as **Bagging and Random Forest** can help reduce variance.

### Q90. Explain Genetic programming as it relates to Evolutionary algorithm.
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

### Q91. What are some classification methods that SVM can handle?
#### Classification Methods SVM Can Handle

- **Binary Classification:** Classifies data into **two classes**.
  - Example: Spam vs Not Spam.
- **Multiclass Classification:** Classifies data into **more than two classes**, using methods such as **One-vs-One (OvO)** or **One-vs-Rest (OvR)**.
  - Example: Cat vs Dog vs Horse.
- **Multilabel Classification:** Assigns **multiple labels to the same data point** using multiple binary SVM classifiers.
  - Example: An image can have labels such as **Animal, Outdoor, and Dog**.

### Q92. Why do you think instance-based learning algorithm is sometimes referred to as Lazy learning algorithm?
#### Why Is Instance-Based Learning Called Lazy Learning?

- **Instance-based learning** stores the training data instead of building a general model during training.
- It performs most of the **computation when making a prediction**, rather than during training.
- Therefore, the **training phase is fast**, but the **prediction phase can be slower**.
- This is why it is called **Lazy Learning** — the algorithm “waits” until a prediction is needed before doing most of the work.
- Example: **K-Nearest Neighbors (KNN)**.

### Q93. Explain the reason for pruning of a decision tree.
- **Pruning** means removing unnecessary branches or nodes from a decision tree.
- It is mainly done to **reduce overfitting**.
- A very deep tree may learn **noise and small details** from the training data.
- Pruning makes the tree **simpler and easier to interpret**.
- It can improve the model's **performance on unseen data**.
- It can also reduce **computational complexity**.

### Q94. How Regularization reduces the cost term?
- Regularization **adds a penalty term** to the original cost function.
- This penalty increases when the model's **weights become too large**.
- To minimize the total cost, the model tries to keep the weights **small**.
- Smaller weights make the model **less complex** and reduce its tendency to overfit.
- **L1 regularization:** Can make some weights exactly zero.
- **L2 regularization:** Shrinks weights toward zero.

### Q95. What's the need to convert categorical variables into factors?
#### Why Convert Categorical Variables into Factors?

- **Factors** are used to represent **categorical data** as distinct groups or categories.
- They help statistical and ML functions **understand that the values represent categories**, not continuous numbers.
- They can store a fixed set of **levels**, such as Male/Female or Low/Medium/High.
- They are especially important in **R**, where many statistical functions treat factors appropriately.
- Factors can be **unordered** or **ordered**, depending on whether the categories have a natural ranking.

**Example:**

```txt
gender<-factor(c("Male","Female","Male"))
```

**Simple terms:**

**Converting categorical variables into factors tells R that the values represent categories rather than numerical quantities.**

### Q96. Do you believe treating a categorical variable as a continuous variable will result in a better predictive model?
- **Usually, no.** Treating a categorical variable as continuous does not necessarily produce a better predictive model.
- A categorical variable represents **groups or categories**, not numerical quantities.
- If categories are converted directly into numbers, the model may incorrectly assume **order and distance** between them.
- For example, assigning **Low = 1, Medium = 2, High = 3** assumes the difference between Low and Medium is the same as between Medium and High.
- Instead, use appropriate encoding methods such as **One-Hot Encoding** or **Ordinal Encoding**, depending on the nature of the categories.

### Q97. Why do we need the confusion matrix?
- A **confusion matrix** helps evaluate the performance of a **classification model**.
- It shows how many predictions were:
  - **True Positive (TP):** Correctly predicted positive.
  - **True Negative (TN):** Correctly predicted negative.
  - **False Positive (FP):** Predicted positive but actually negative.
  - **False Negative (FN):** Predicted negative but actually positive.
- It helps identify **what types of mistakes** the model is making.
- It is used to calculate important metrics such as **Accuracy, Precision, Recall, and Specificity**.

### Q98. What differentiates Gradient boosting and Random forest?
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

### Q99. How does a Box-Cox transformation occur?
#### Box-Cox Transformation

- **Box-Cox transformation** is a statistical technique used to **transform data so that it is closer to a normal distribution**.
- It is mainly used when the data is **skewed** or when the variance is not constant.
- It works by applying a **power transformation** to the data.
- The transformation uses a parameter called **lambda (λ)**.
- Different values of λ produce different transformations.
- The value of λ is usually chosen to find the transformation that gives the **best approximation to normality**.
- Box-Cox requires the data to contain **positive values**.

**Common transformations:**

- λ = 1 → No major transformation
- λ = 0 → Log transformation
- λ = 0.5 → Square-root transformation
- λ = -1 → Reciprocal transformation

**Simple terms:**

**Box-Cox changes the scale of the data using a suitable power so that the data becomes more normally distributed and has more stable variance.**

### Q100. What functions can be used to identify, drop, and replace missing or corrupted values?
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

### Q101. How is data divided in cross-validation?
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

### Q102. A game exists where you are required to roll two six-faced dice. You stand a chance to win \$50 if the sum of the two dice equals 7. However, you must pay \$4 to play each time you roll both dice. Would you partake in this game? Note: If you play 6 times, what is the probability of making money from this game?
#### Should You Play the Game?

- There are **36 possible outcomes** when rolling two dice.
- A sum of **7** can occur in **6 ways**:
  - (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
- Therefore, the probability of winning on one roll is **6/36 = 1/6**.
- You win **\$50** and pay **\$4** per game.

For **6 games**:

- Total cost = 6 × \$4 = **\$24**
- Let **X** = number of wins.
- You make money when your winnings exceed \$24:
  - 1 win → \$50 − \$24 = **\$26 profit**
  - So you need **at least 1 win**.

Probability of making money:

- Probability of no wins in 6 rolls = **(5/6)⁶ ≈ 0.335**
- Probability of at least one win = **1 − (5/6)⁶ ≈ 0.665**

#### Final Answer

**Probability of making money after 6 plays ≈ 66.5%.**

So, based purely on the probability of making money in 6 plays, **yes, I would play**.

### Q103. Assuming you have a two-column table of users and their friends coupled with a two-column table of users and their liked pages. Write a SQL query to make recommendations using pages that your friends have liked when it should not recommend pages that you have liked.
Assume the tables are:

- `friends(user_id, friend_id)`
- `likes(user_id, page_id)`

A SQL query would be:

```sql
SELECT DISTINCT
    f.user_id,
    l.page_id
FROM friends f
JOIN likes l
    ON f.friend_id = l.user_id
WHERE NOT EXISTS (
    SELECT 1
    FROM likes my_likes
    WHERE my_likes.user_id = f.user_id
      AND my_likes.page_id = l.page_id
);
```

#### How it works

- Join `friends` with `likes` to find **pages liked by your friends**.
- `NOT EXISTS` removes pages that **you have already liked**.
- `DISTINCT` prevents duplicate recommendations when multiple friends like the same page.

### Q104. When recommendations like “....people who did this also bought this” comes up on a website. What sort of algorithm propels that?
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
