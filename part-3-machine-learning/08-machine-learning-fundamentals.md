# Chapter 8. Machine Learning Fundamentals

_Part III — Machine Learning_

The learning problem, bias and variance, overfitting, regularization, data leakage and the ML project lifecycle. Expect these in every ML loop, often as the opening questions.

This chapter contains **99 questions**.

## Contents

- **8.1** [Why do we need Machine Learning?](#81-why-do-we-need-machine-learning)
- **8.2** [What is Machine Learning, formally?](#82-what-is-machine-learning-formally)
- **8.3** [Give some real-world applications of Machine Learning.](#83-give-some-real-world-applications-of-machine-learning)
- **8.4** [What is the difference between Deduction and Induction?](#84-what-is-the-difference-between-deduction-and-induction)
- **8.5** [How does Machine Learning relate to induction?](#85-how-does-machine-learning-relate-to-induction)
- **8.6** [Why should Machine Learning work at all?](#86-why-should-machine-learning-work-at-all)
- **8.7** [What is Overfitting?](#87-what-is-overfitting)
- **8.8** [What is Underfitting?](#88-what-is-underfitting)
- **8.9** [How would you find a good balance between overfitting and underfitting?](#89-how-would-you-find-a-good-balance-between-overfitting-and-underfitting)
- **8.10** [What is the difference between parametric and non-parametric models?](#810-what-is-the-difference-between-parametric-and-non-parametric-models)
- **8.11** [What is the difference between online and offline learning?](#811-what-is-the-difference-between-online-and-offline-learning)
- **8.12** [What is an Epoch?](#812-what-is-an-epoch)
- **8.13** [What is a Batch?](#813-what-is-a-batch)
- **8.14** [What is Batch Size?](#814-what-is-batch-size)
- **8.15** [What is an Iteration?](#815-what-is-an-iteration)
- **8.16** [If I have 1,000 training examples and a batch size of 100, how many iterations are in one epoch?](#816-if-i-have-1000-training-examples-and-a-batch-size-of-100-how-many-iterations-are-in-one-epoch)
- **8.17** [If I train for 5 epochs with the setup above, how many total iterations happen?](#817-if-i-train-for-5-epochs-with-the-setup-above-how-many-total-iterations-happen)
- **8.18** [Why not just feed the entire dataset in one go instead of using batches?](#818-why-not-just-feed-the-entire-dataset-in-one-go-instead-of-using-batches)
- **8.19** [What's the trade-off between large and small batch sizes?](#819-whats-the-trade-off-between-large-and-small-batch-sizes)
- **8.20** [What is (Batch) Gradient Descent?](#820-what-is-batch-gradient-descent)
- **8.21** [What is Stochastic Gradient Descent (SGD)?](#821-what-is-stochastic-gradient-descent-sgd)
- **8.22** [What is Mini-Batch Gradient Descent?](#822-what-is-mini-batch-gradient-descent)
- **8.23** [Which GD is used more now?](#823-which-gd-is-used-more-now)
- **8.24** [What are embeddings in Machine Learning?](#824-what-are-embeddings-in-machine-learning)
- **8.25** [What is the Softmax Activation Function?](#825-what-is-the-softmax-activation-function)
- **8.26** [What is Machine Learning?](#826-what-is-machine-learning)
- **8.27** [Differentiate between Supervised and Unsupervised Learning.](#827-differentiate-between-supervised-and-unsupervised-learning)
- **8.28** [What is Reinforcement Learning?](#828-what-is-reinforcement-learning)
- **8.29** [What is Bias?](#829-what-is-bias)
- **8.30** [What is the difference between Classification and Regression?](#830-what-is-the-difference-between-classification-and-regression)
- **8.31** [Explain Overfitting and Underfitting. How can you prevent them?](#831-explain-overfitting-and-underfitting-how-can-you-prevent-them)
- **8.32** [What Are L1 and L2 Loss Functions?](#832-what-are-l1-and-l2-loss-functions)
- **8.33** [What is Regularization? Explain L1 (Lasso) and L2 (Ridge) regularization.](#833-what-is-regularization-explain-l1-lasso-and-l2-ridge-regularization)
- **8.34** [What are Loss Functions and Cost Functions? Explain the key difference between them.](#834-what-are-loss-functions-and-cost-functions-explain-the-key-difference-between-them)
- **8.35** [What are dropouts?](#835-what-are-dropouts)
- **8.36** [What is a Perceptron?](#836-what-is-a-perceptron)
- **8.37** [Explain Multilayer Perception (MLP).](#837-explain-multilayer-perception-mlp)
- **8.38** [What is Cross-Entropy?](#838-what-is-cross-entropy)
- **8.39** [What are Logits?](#839-what-are-logits)
- **8.40** [Explain Cross-Validation. Why is it used?](#840-explain-cross-validation-why-is-it-used)
- **8.41** [What is Confusion Matrix?](#841-what-is-confusion-matrix)
- **8.42** [What are precision, recall, and F1-score?](#842-what-are-precision-recall-and-f1-score)
- **8.43** [What is anomaly detection?](#843-what-is-anomaly-detection)
- **8.44** [What is the difference between policy-based and value-based methods?](#844-what-is-the-difference-between-policy-based-and-value-based-methods)
- **8.45** [What is Q-Learning?](#845-what-is-q-learning)
- **8.46** [Explain the concept of exploration vs. exploitation.](#846-explain-the-concept-of-exploration-vs-exploitation)
- **8.47** [Explain the curse of dimensionality and how to address it.](#847-explain-the-curse-of-dimensionality-and-how-to-address-it)
- **8.48** [Explain Local Loss, Focal Loss, and Gradient Blending in the context of Multi-Task Learning.](#848-explain-local-loss-focal-loss-and-gradient-blending-in-the-context-of-multi-task-learning)
- **8.49** [Explain Contrastive Learning](#849-explain-contrastive-learning)
- **8.50** [What is Generative AI?](#850-what-is-generative-ai)
- **8.51** [Differentiate between Training Sets and Test Sets?](#851-differentiate-between-training-sets-and-test-sets)
- **8.52** [Define Bias and Variance.](#852-define-bias-and-variance)
- **8.53** [What are Machine Learning and Artificial Intelligence?](#853-what-are-machine-learning-and-artificial-intelligence)
- **8.54** [Differentiate between Deep Learning and Machine Learning?](#854-differentiate-between-deep-learning-and-machine-learning)
- **8.55** [What are the types of Machine learning?](#855-what-are-the-types-of-machine-learning)
- **8.56** [What different targets do classification and regression algorithms require?](#856-what-different-targets-do-classification-and-regression-algorithms-require)
- **8.57** [What is semi-supervised learning?](#857-what-is-semi-supervised-learning)
- **8.58** [Where is semi-supervised learning applied?](#858-where-is-semi-supervised-learning-applied)
- **8.59** [What is Inductive Logic Programming?](#859-what-is-inductive-logic-programming)
- **8.60** [What is deductive learning and inductive learning?](#860-what-is-deductive-learning-and-inductive-learning)
- **8.61** [How would you differentiate data mining from Machine learning?](#861-how-would-you-differentiate-data-mining-from-machine-learning)
- **8.62** [Why does overfitting occur in ML?](#862-why-does-overfitting-occur-in-ml)
- **8.63** [What are the functions of unsupervised learning?](#863-what-are-the-functions-of-unsupervised-learning)
- **8.64** [What are some functions of supervised learning?](#864-what-are-some-functions-of-supervised-learning)
- **8.65** [What is Regularization in ML?](#865-what-is-regularization-in-ml)
- **8.66** [What do you understand about the Bias-Variance Tradeoff?](#866-what-do-you-understand-about-the-bias-variance-tradeoff)
- **8.67** [How can you differentiate between a parametric and a non-parametric model?](#867-how-can-you-differentiate-between-a-parametric-and-a-non-parametric-model)
- **8.68** [How can overfitting be avoided in ML?](#868-how-can-overfitting-be-avoided-in-ml)
- **8.69** [How would you handle a dataset suffering from high variance?](#869-how-would-you-handle-a-dataset-suffering-from-high-variance)
- **8.70** [How Regularization reduces the cost term?](#870-how-regularization-reduces-the-cost-term)
- **8.71** [What do you mean by the unreasonable effectiveness of data?](#871-what-do-you-mean-by-the-unreasonable-effectiveness-of-data)
- **8.72** [What is Online Machine Learning? How is it different from Offline machine learning? List some of its applications](#872-what-is-online-machine-learning-how-is-it-different-from-offline-machine-learning-list-some-of-its-applications)
- **8.73** [What is the No Free Lunch Theorem?](#873-what-is-the-no-free-lunch-theorem)
- **8.74** [What are the main differences between Structured and Unstructured Data?](#874-what-are-the-main-differences-between-structured-and-unstructured-data)
- **8.75** [What is Data Leakage? List some ways by which you can overcome this problem.](#875-what-is-data-leakage-list-some-ways-by-which-you-can-overcome-this-problem)
- **8.76** [What are the different types of bias in Machine Learning?](#876-what-are-the-different-types-of-bias-in-machine-learning)
- **8.77** [Why does L2 regularization give sparse coefficients?](#877-why-does-l2-regularization-give-sparse-coefficients)
- **8.78** [List some ways using which you can improve a model’s performance.](#878-list-some-ways-using-which-you-can-improve-a-models-performance)
- **8.79** [What is active learning, and where is it useful?](#879-what-is-active-learning-and-where-is-it-useful)
- **8.80** [What types of models tend to overfit?](#880-what-types-of-models-tend-to-overfit)
- **8.81** [Can you tell the complete life cycle of a data science project?](#881-can-you-tell-the-complete-life-cycle-of-a-data-science-project)
- **8.82** [What are the properties of a good ML model?](#882-what-are-the-properties-of-a-good-ml-model)
- **8.83** [What do you mean by the curse of Dimensionality?](#883-what-do-you-mean-by-the-curse-of-dimensionality)
- **8.84** [Considering the long list of machine learning algorithms, given a data set, how do you decide which one to use?](#884-considering-the-long-list-of-machine-learning-algorithms-given-a-data-set-how-do-you-decide-which-one-to-use)
- **8.85** [How will you know which machine learning algorithm to choose for your classification problem?](#885-how-will-you-know-which-machine-learning-algorithm-to-choose-for-your-classification-problem)
- **8.86** [Is more data always better?](#886-is-more-data-always-better)
- **8.87** [How would you differentiate between multi-label and multi-class classification?](#887-how-would-you-differentiate-between-multi-label-and-multi-class-classification)
- **8.88** [What is an AI ontology?](#888-what-is-an-ai-ontology)
- **8.89** [Can AI systems be conscious?](#889-can-ai-systems-be-conscious)
- **8.90** [What is unsupervised learning through compression?](#890-what-is-unsupervised-learning-through-compression)
- **8.91** [Explain the bias-variance tradeoff mathematically and practically.](#891-explain-the-bias-variance-tradeoff-mathematically-and-practically)
- **8.92** [Why can a model with lower training loss have worse test performance?](#892-why-can-a-model-with-lower-training-loss-have-worse-test-performance)
- **8.93** [How do you detect overfitting beyond simply comparing train and validation accuracy?](#893-how-do-you-detect-overfitting-beyond-simply-comparing-train-and-validation-accuracy)
- **8.94** [What is data leakage? Give examples that are difficult to detect.](#894-what-is-data-leakage-give-examples-that-are-difficult-to-detect)
- **8.95** [What is the difference between model capacity and model complexity?](#895-what-is-the-difference-between-model-capacity-and-model-complexity)
- **8.96** [Why does regularization reduce variance?](#896-why-does-regularization-reduce-variance)
- **8.97** [Explain L1 vs L2 regularization geometrically.](#897-explain-l1-vs-l2-regularization-geometrically)
- **8.98** [What is the difference between generative and discriminative models?](#898-what-is-the-difference-between-generative-and-discriminative-models)
- **8.99** [What is the singularity in AI?](#899-what-is-the-singularity-in-ai)

---

### 8.1 Why do we need Machine Learning?

Many tasks are too complex to solve with explicitly hand-coded rules (e.g., recognizing handwriting, detecting objects, driving a car). Machine Learning lets a computer learn how to perform such tasks directly from training examples, instead of a programmer trying to anticipate every rule manually. This makes it possible to build systems for spam filtering, bioinformatics, medical image analysis, language processing, and game-playing agents like AlphaGo, where explicit programming is impractical.

### 8.2 What is Machine Learning, formally?

A set of algorithms that allow a computer to learn a specific task from training examples. "Learning" here doesn't mean memorizing the seen examples — it means generalizing to previously unseen instances. Ideally, the computer should use the training examples to extract a general rule describing how the task should be performed correctly.

### 8.3 Give some real-world applications of Machine Learning.

Handwritten letter recognition, object detection, self-driving cars, spam filtering, bioinformatics, medical image analysis, natural language processing, and game-playing systems like AlphaGo and chess engines.

### 8.4 What is the difference between Deduction and Induction?

- **Deduction** reasons from one or more general premises to a logically certain conclusion. If the premises are true, the conclusion is guaranteed to be true.
- **Induction** reasons from specific examples to construct or evaluate a general proposition. We can never be fully certain the conclusion is correct — it's a generalization, not a guarantee.

### 8.5 How does Machine Learning relate to induction?

Machine Learning can be viewed as an attempt to automate the process of inductive inference — using specific training examples to derive a general rule or model, without any certainty that the rule is perfectly correct.

### 8.6 Why should Machine Learning work at all?

It only works if there is actually something to learn — i.e., if the data contains an underlying pattern or structure. If the examples were pure noise with no consistent relationship between inputs and outputs, no algorithm could extract a meaningful general rule.

### 8.7 What is Overfitting?

Overfitting happens when a model is too complex and fits the training data extremely well — including its noise — rather than the true underlying pattern. It has low training (approximation) error but high test (estimation) error, because it fails to generalize to unseen data.

#### Additional notes

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

### 8.8 What is Underfitting?

Underfitting happens when a model is too simplistic to capture the underlying pattern. It tends to be stable with respect to noise, so the training error is high, but paradoxically the "estimation gap" (variance) can be low — the model consistently fails, both on training and test data.

#### Additional notes

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

### 8.9 How would you find a good balance between overfitting and underfitting?

By choosing model complexity that minimizes total error — balancing bias (underfitting) and variance (overfitting) — often through techniques like cross-validation, regularization, or pruning, rather than maximizing fit to the training set alone.

### 8.10 What is the difference between parametric and non-parametric models?

**Parametric models:**

- Assume a fixed functional form with a **fixed, finite number of parameters**, decided *before* seeing the data
- Learning = estimating those parameters from data
- Once trained, the training data can be discarded — the parameters *are* the model
- Model complexity doesn't grow with more data

**Non-parametric models:**

- Don't assume a fixed functional form (or make much weaker assumptions)
- Number of "parameters" effectively **grows with the size of the data**
- Often need to keep the training data around at prediction time
- Model complexity can grow as you get more data

#### Examples

| Parametric | Non-parametric |
| --- | --- |
| Linear regression | k-NN |
| Logistic regression | Decision trees (can grow unbounded) |
| Naive Bayes (with assumed distribution) | Kernel density estimation |
| Neural nets with fixed architecture\* | k-means (arguably semi-parametric — see below) |

> ✍️ _Drafted addition._

**About the two footnoted rows in the table:**

- **\*Neural nets with a fixed architecture** are technically parametric: the parameter count is set before training and does not grow with the data. In practice, very large networks behave a lot like non-parametric models. They can memorize their training set, and when there is more data we usually pick a bigger architecture, so their effective capacity grows with the data.
- **k-means as "semi-parametric":** once trained, the model is just k centroids (k × d numbers). That is a fixed, finite set of parameters, which points to parametric. But the number of clusters is often chosen from the data (elbow or silhouette), and k-means assumes no distributional form beyond spherical, similarly sized clusters. So it sits between the two. A Gaussian Mixture Model is the clearly parametric probabilistic counterpart.

**Trade-offs to mention in an interview:**

- **Parametric:** fast, small and easy to interpret. It works with little data but has high bias if the assumed form is wrong (for example, a linear model on a non-linear relationship).
- **Non-parametric:** flexible, with low bias. It needs more data and is prone to overfitting without controls such as k, tree depth or kernel bandwidth. Inference can be expensive because k-NN and KDE keep the whole training set, which matters for latency and memory in production.

### 8.11 What is the difference between online and offline learning?

| Aspect | **Online Learning** | **Offline Learning (Batch Learning)** |
| --- | --- | --- |
| **Definition** | Model learns continuously from new data | Model learns from a fixed dataset |
| **Data** | Data arrives incrementally/continuously | Data is collected first, then used for training |
| **Training** | Updates model frequently | Retrains periodically |
| **Memory** | Usually requires less memory | Can require large memory/storage |
| **Adaptability** | Quickly adapts to changing patterns | Does not automatically adapt to new patterns |
| **Computation** | Small, frequent updates | Large, less frequent computations |
| **Use case** | Streaming data, real-time systems | Static or periodically updated datasets |
| **Example** | Fraud detection adapting to new transactions | Training an image classifier on a fixed dataset |

### 8.12 What is an Epoch?

One complete pass through the entire training dataset. E.g., 5 epochs means the model has seen every training example 5 times.

### 8.13 What is a Batch?

A single chunk or group of training examples processed together in one pass before the model updates its weights.

### 8.14 What is Batch Size?

The number of training examples in one batch. E.g., batch size = 100 means the model processes 100 examples at a time.

### 8.15 What is an Iteration?

One weight update, which happens after the model processes one batch.

### 8.16 If I have 1,000 training examples and a batch size of 100, how many iterations are in one epoch?

10 iterations (1,000 ÷ 100 = 10 batches, and each batch = 1 iteration).

### 8.17 If I train for 5 epochs with the setup above, how many total iterations happen?

50 total iterations (5 epochs × 10 iterations per epoch).

### 8.18 Why not just feed the entire dataset in one go instead of using batches?

Large datasets often don't fit in memory (especially GPU memory), and smaller batches allow more frequent weight updates, which can speed up learning.

### 8.19 What's the trade-off between large and small batch sizes?

Large batches give smoother, more accurate gradient estimates but are slower per update. Small batches update faster but produce noisier gradients (batch size = 1 is called stochastic gradient descent).

### 8.20 What is (Batch) Gradient Descent?

An optimization method where the model computes the gradient using the **entire training dataset** before making a single weight update.

### 8.21 What is Stochastic Gradient Descent (SGD)?

An optimization method where the model computes the gradient and updates weights using **just one training example** at a time (batch size = 1).

<!-- Merged from 05-optimization.md, where the same question appeared again ("What is stochastic gradient descent (SGD)?"). Original content follows. -->

**Stochastic Gradient Descent (SGD)**

- Variant of gradient descent that updates parameters using one random training sample at a time (instead of full dataset)
- Update rule: θ = θ − α∇J(θ; xᵢ, yᵢ) for a single sample i
- Much faster per update than Batch GD, especially for large datasets
- Noisy updates — high variance in path to minimum, but noise can help escape local minima/saddle points
- Requires more iterations to converge, but each iteration is cheap
- Often used with mini-batches in practice (Mini-Batch SGD) for balance between speed and stability

### 8.22 What is Mini-Batch Gradient Descent?

A middle ground — the gradient is computed on a small batch (e.g., 32, 64, 100 examples) instead of the whole dataset or a single example. This is what's used in most real-world deep learning.

### 8.23 Which GD is used more now?

lmost always **mini-batch gradient descent** — it balances the computational efficiency of SGD with the more stable convergence of full-batch gradient descent. It's often just called "SGD" loosely in frameworks like PyTorch/TensorFlow, even though it technically uses mini-batches.

### 8.24 What are embeddings in Machine Learning?

> **See also:** [Question 17.15 — What are embeddings?](../part-4-generative-ai/17-large-language-model-foundations.md#1715-what-are-embeddings), which covers the same question from another angle.

- Dense, low-dimensional vector representations of data (text, images, users, etc.) that capture semantic meaning
- Convert discrete/high-dimensional data into continuous numerical form usable by ML models
- Similar items lie close together in vector space (measured via cosine similarity, dot product, etc.)
- Learned through neural networks (e.g., Word2Vec, BERT, transformers) during training
- Reduce dimensionality while preserving meaningful relationships (e.g., king − man + woman ≈ queen)
- Used in search, recommendations, clustering, RAG, and vector databases

### 8.25 What is the Softmax Activation Function?

- Converts a vector of raw scores (logits) into a probability distribution
- Formula: softmax(zᵢ) = e\^zᵢ / Σ e\^zⱼ (for all j classes)
- Output values are between 0 and 1, and sum to 1
- Used in the output layer for multi-class classification problems
- Amplifies differences — larger logits get pushed closer to 1, smaller ones toward 0
- Often paired with cross-entropy loss during training

### 8.26 What is Machine Learning?

- A field of AI where systems learn patterns from data to make predictions or decisions, without being explicitly programmed with rules
- Models improve performance on a task through experience (training data)
- Main types: supervised, unsupervised, and reinforcement learning
- Core process: input data → model learns patterns → outputs predictions on new data

### 8.27 Differentiate between Supervised and Unsupervised Learning.

**Supervised vs Unsupervised Learning**

| Aspect | Supervised | Unsupervised |
| --- | --- | --- |
| Data | Labeled (input-output pairs) | Unlabeled (input only) |
| Goal | Predict output for new inputs | Find hidden patterns/structure |
| Examples | Classification, Regression | Clustering, Dimensionality Reduction |
| Algorithms | Linear Regression, SVM, Decision Trees | K-Means, PCA, Hierarchical Clustering |
| Feedback | Direct error signal (label vs prediction) | No explicit feedback |
| Use cases | Spam detection, price prediction | Customer segmentation, anomaly detection |

#### Additional notes

#### Supervised vs. Unsupervised Learning

| **Aspect** | **Supervised Learning** | **Unsupervised Learning** |
| --- | --- | --- |
| **Data** | Uses **labeled data** | Uses **unlabeled data** |
| **Goal** | Predict a target/output | Discover hidden patterns or structure |
| **Training** | Learns from input-output pairs | Learns from relationships within the data |
| **Main Tasks** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Examples** | Linear Regression, Decision Tree, XGBoost, SVM | K-Means, DBSCAN, PCA |
| **Evaluation** | Accuracy, F1, RMSE, etc. | Silhouette Score, clustering analysis, etc. |

### 8.28 What is Reinforcement Learning?

> **See also:** [Question 17.78 — What is reinforcement learning?](../part-4-generative-ai/17-large-language-model-foundations.md#1778-what-is-reinforcement-learning), which covers the same question from another angle.

- Learning paradigm where an agent learns to make decisions by interacting with an environment
- Agent takes actions → receives rewards/penalties → learns policy to maximize cumulative reward
- No labeled data — learns through trial and error
- Key components: agent, environment, state, action, reward, policy
- Examples: game playing (AlphaGo), robotics, recommendation systems

### 8.29 What is Bias?

- Error introduced when a model makes overly simplistic assumptions about the data
- Represents the gap between average model prediction and true value
- High bias → underfitting (model too simple, misses patterns)
- Leads to poor performance on both training and test data
- Trade-off exists with variance (Bias-Variance tradeoff)

### 8.30 What is the difference between Classification and Regression?

**Classification vs Regression**

| Aspect | Classification | Regression |
| --- | --- | --- |
| Output | Discrete categories/classes | Continuous numerical value |
| Example | Spam vs Not Spam, Cat vs Dog | House price, Temperature prediction |
| Algorithms | Logistic Regression, SVM, Decision Trees | Linear Regression, Ridge, Lasso |
| Evaluation | Accuracy, Precision, Recall, F1 | MSE, RMSE, MAE, R² |
| Output type | Class labels/probabilities | Real numbers |

#### Additional notes

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

### 8.31 Explain Overfitting and Underfitting. How can you prevent them?

**Overfitting vs Underfitting**

**Overfitting**

- Model learns training data too well, including noise
- Performs great on training data, poorly on test/unseen data
- Caused by: too complex model, too little data, too many features

**Underfitting**

- Model too simple to capture underlying patterns
- Performs poorly on both training and test data
- Caused by: too simple model, insufficient training, high bias

**Prevention**

| Overfitting | Underfitting |
| --- | --- |
| Regularization (L1/L2) | Increase model complexity |
| Cross-validation | Add more features |
| More training data | Train longer |
| Dropout (neural nets) | Reduce regularization |
| Early stopping | Use better features |
| Reduce model complexity |  |

### 8.32 What Are L1 and L2 Loss Functions?

**L1 and L2 Loss Functions**

**L1 Loss (Mean Absolute Error)**

- Sum/average of absolute differences: L1 = Σ\|y\_true − y\_pred\|
- Robust to outliers
- Produces sparse solutions (used in L1 regularization/Lasso)
- Gradient is constant, not smooth at zero

**L2 Loss (Mean Squared Error)**

- Sum/average of squared differences: L2 = Σ(y\_true − y\_pred)²
- Penalizes larger errors more heavily
- Sensitive to outliers
- Smooth gradient, easier to optimize (used in L2 regularization/Ridge)

### 8.33 What is Regularization? Explain L1 (Lasso) and L2 (Ridge) regularization.

**Regularization**

- Technique to prevent overfitting by adding a penalty term to the loss function based on model weights
- Discourages overly complex models by shrinking coefficients
- Controlled by hyperparameter λ (lambda) — higher λ = more penalty

**L1 (Lasso)**

- Adds penalty = λ Σ\|wᵢ\|
- Shrinks some coefficients to exactly zero → performs feature selection
- Produces sparse models

**L2 (Ridge)**

- Adds penalty = λ Σwᵢ²
- Shrinks coefficients toward zero but rarely to exactly zero
- Handles multicollinearity well, keeps all features

### 8.34 What are Loss Functions and Cost Functions? Explain the key difference between them.

**Loss Function vs Cost Function**

- **Loss function**: Error for a single training example
- **Cost function**: Average (or sum) of loss over the entire training dataset
- Loss = per-sample error; Cost = overall model error used to guide training
- Both measure how far predictions are from actual values, but at different scales (individual vs aggregate)

### 8.35 What are dropouts?

- Regularization technique used in neural networks to prevent overfitting
- Randomly "drops" (deactivates) a fraction of neurons during each training iteration
- Forces network to not rely on specific neurons, improving generalization
- Dropout rate (p) typically 0.2–0.5
- Only applied during training; all neurons active during inference (with scaling)

### 8.36 What is a Perceptron?

- Simplest form of a neural network — a single-layer binary classifier
- Takes weighted inputs, sums them, adds bias, passes through activation function (step function)
- Formula: output = f(Σ(wᵢxᵢ) + b)
- Can only solve linearly separable problems (e.g., fails on XOR)
- Foundation/building block for multi-layer neural networks (MLPs)

### 8.37 Explain Multilayer Perception (MLP).

- Feedforward neural network with multiple layers: input, one or more hidden layers, output
- Each layer fully connected to the next; uses non-linear activation functions (ReLU, sigmoid, tanh)
- Can solve non-linearly separable problems (unlike single perceptron)
- Trained via backpropagation + gradient descent
- Forms the basis of deep learning architectures

### 8.38 What is Cross-Entropy?

> **See also:** [Question 17.74 — What is entropy vs. cross-entropy?](../part-4-generative-ai/17-large-language-model-foundations.md#1774-what-is-entropy-vs-cross-entropy), which covers the same question from another angle.

- Loss function that measures difference between predicted probability distribution and true distribution
- Formula (binary): L = −\[y log(p) + (1−y) log(1−p)\]
- Formula (multi-class): L = −Σ yᵢ log(pᵢ)
- Penalizes confident wrong predictions heavily
- Commonly used with Softmax for classification tasks

### 8.39 What are Logits?

- Raw, unnormalized output scores from a model's final layer, before activation (like softmax/sigmoid)
- Can be any real number (positive, negative, unbounded)
- Converted to probabilities by applying softmax (multi-class) or sigmoid (binary)
- Used directly in some loss functions (e.g., `CrossEntropyLoss` in PyTorch expects logits, not probabilities)

### 8.40 Explain Cross-Validation. Why is it used?

**Cross-Validation**

- Technique to evaluate model performance by splitting data into multiple train/test subsets
- Most common: **k-fold CV** — data split into k folds, model trained on k-1 folds, tested on remaining fold, repeated k times
- Final performance = average across all folds

**Why used**

- Gives more reliable performance estimate than a single train/test split
- Reduces risk of overfitting to one particular split
- Makes better use of limited data
- Helps in hyperparameter tuning and model selection

### 8.41 What is Confusion Matrix?

> **See also:** [Question 12.25 — What is the confusion matrix?](12-model-evaluation-and-experimentation.md#1225-what-is-the-confusion-matrix), which covers the same question from another angle.

**Confusion Matrix**

- Table used to evaluate classification model performance by comparing actual vs predicted labels

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

- **TP**: Correctly predicted positive
- **TN**: Correctly predicted negative
- **FP**: Predicted positive, actually negative (Type I error)
- **FN**: Predicted negative, actually positive (Type II error)
- Used to derive accuracy, precision, recall, F1-score

### 8.42 What are precision, recall, and F1-score?

**Precision, Recall, F1-Score**

- **Precision**: Of predicted positives, how many are actually correct → TP / (TP + FP), how precise are my positive predictions?
- **Recall**: Of actual positives, how many were correctly identified → TP / (TP + FN), how many of the actual positive did i find/recall?
- **F1-Score**: Harmonic mean of precision and recall → 2 × (P × R) / (P + R)
- Precision matters when false positives are costly (e.g., spam detection)
- Recall matters when false negatives are costly (e.g., disease detection)
- F1 balances both, useful for imbalanced datasets

### 8.43 What is anomaly detection?

- Technique to identify data points, events, or observations that deviate significantly from normal patterns
- Also called outlier detection
- **Methods**:
  - Statistical (z-score, IQR)
  - Distance-based (k-NN, clustering)
  - Density-based (DBSCAN, LOF)
  - ML-based (Isolation Forest, One-Class SVM, Autoencoders)
- **Use cases**: Fraud detection, network intrusion, equipment failure, medical diagnosis, quality control

### 8.44 What is the difference between policy-based and value-based methods?

**Policy-Based vs Value-Based Methods (Reinforcement Learning)**

| Aspect | Policy-Based | Value-Based |
| --- | --- | --- |
| Learns | Policy directly (mapping states → actions) | Value function (expected reward of states/actions) |
| Action selection | Sampled directly from policy | Derived from value function (e.g., pick max-value action) |
| Action space | Works well with continuous action spaces | Better suited for discrete action spaces |
| Output | Probability distribution over actions | Estimated value (Q-value) |
| Examples | REINFORCE, PPO, A2C (actor part) | Q-Learning, DQN, SARSA |
| Convergence | Can converge to local optima, higher variance | More stable but can be less flexible |
| Exploration | Naturally stochastic (built-in exploration) | Needs explicit strategy (e.g., ε-greedy) |

### 8.45 What is Q-Learning?

- Model-free, value-based reinforcement learning algorithm
- Learns a Q-value function Q(s, a) representing expected cumulative reward for taking action a in state s
- **Update rule**: Q(s,a) ← Q(s,a) + α\[r + γ·max Q(s',a') − Q(s,a)\]
  - α = learning rate, γ = discount factor, r = reward
- Off-policy — learns optimal policy regardless of agent's actions during exploration
- Uses a Q-table (or neural network in Deep Q-Learning/DQN) to store/approximate values
- Agent picks action with highest Q-value (often via ε-greedy for exploration)

### 8.46 Explain the concept of exploration vs. exploitation.

- Core dilemma in reinforcement learning: whether to try new actions or use known best actions
- **Exploration**: Trying new/unknown actions to discover potentially better rewards
- **Exploitation**: Choosing the best-known action to maximize immediate reward
- Too much exploration → slow convergence, inefficient
- Too much exploitation → risk getting stuck in suboptimal policy (local optima)
- **Common strategies**: ε-greedy (random action with probability ε), decaying ε, softmax action selection, UCB (Upper Confidence Bound)

### 8.47 Explain the curse of dimensionality and how to address it.

**Curse of Dimensionality**

- As the number of features/dimensions increases, data becomes sparse in the feature space
- Distance metrics become less meaningful (points appear equidistant)
- Requires exponentially more data to maintain statistical significance
- Leads to overfitting, increased computation, and poor model performance

**How to address it**

- **Dimensionality reduction**: PCA, t-SNE, UMAP, LDA
- **Feature selection**: Remove irrelevant/redundant features
- **Regularization**: L1 (Lasso) to shrink less important features
- **Increase data**: More samples relative to dimensions
- **Autoencoders**: Learn compressed representations

### 8.48 Explain Local Loss, Focal Loss, and Gradient Blending in the context of Multi-Task Learning.

**Local Loss**

- Loss computed independently for each task/branch before combining into total loss
- Helps each task-specific head learn effectively without being dominated by others
- Often combined via weighted sum: L\_total = Σ wᵢ × Lᵢ (local losses)

**Focal Loss**

- Modified cross-entropy loss that down-weights easy/well-classified examples
- Formula: FL = −α(1−p)\^γ log(p)
- Focuses training on hard/misclassified examples
- Useful in multi-task settings with class imbalance across tasks

**Gradient Blending**

- Technique to combine gradients from multiple tasks/modalities to prevent one task from dominating training
- Addresses overfitting/imbalance when tasks learn at different rates (e.g., multimodal learning — audio vs video)
- Computes optimal blend of per-modality/task losses based on overfitting-to-generalization ratio
- Introduced by Facebook AI (Wang et al.) for audio-visual multi-task learning

### 8.49 Explain Contrastive Learning

- Self-supervised learning technique that learns representations by comparing similar vs dissimilar pairs
- Pulls similar (positive) pairs closer together in embedding space, pushes dissimilar (negative) pairs apart
- Doesn't require labeled data — positive pairs often created via augmentation (e.g., two crops of same image)
- **Common loss**: InfoNCE / Contrastive loss (based on similarity scores, e.g., cosine similarity)
- **Examples**: SimCLR, MoCo (images), CLIP (text-image)
- **Use cases**: Representation learning, embedding models, self-supervised pretraining

### 8.50 What is Generative AI?

- Subset of AI focused on creating new content (text, images, audio, video, code) rather than just analyzing/predicting
- Learns underlying patterns/distribution of training data to generate novel, similar outputs
- **Key architectures**: GANs, VAEs, Diffusion Models, Transformers (GPT, etc.)
- **Examples**: ChatGPT (text), DALL-E/Midjourney (images), Stable Diffusion, GitHub Copilot (code)
- **Use cases**: Content creation, code generation, data augmentation, chatbots, design

### 8.51 Differentiate between Training Sets and Test Sets?

| Aspect | Training Set | Test Set |
| --- | --- | --- |
| **Purpose** | Train the model | Evaluate the model |
| **Data** | Seen by the model | Unseen by the model |
| **Used for** | Learning parameters/patterns | Measuring generalization |
| **Model exposure** | During training | Only after training |
| **Typical split** | 80% | 20% |
| **Main concern** | Overfitting | Data leakage |

### 8.52 Define Bias and Variance.

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

### 8.53 What are Machine Learning and Artificial Intelligence?

#### Artificial Intelligence (AI) vs. Machine Learning (ML)

| **Aspect** | **Artificial Intelligence (AI)** | **Machine Learning (ML)** |
| --- | --- | --- |
| **Definition** | A broader field focused on creating systems that perform tasks requiring human-like intelligence. | A subset of AI where systems learn patterns from data to make predictions or decisions. |
| **Goal** | Simulate capabilities such as reasoning, planning, perception, and decision-making. | Learn from data and improve performance without explicitly programming every rule. |
| **Relationship** | **AI is the broader concept.** | **ML is a subset of AI.** |
| **Example** | Autonomous systems, AI assistants, game-playing systems. | Spam detection, house-price prediction, recommendation systems. |

### 8.54 Differentiate between Deep Learning and Machine Learning?

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

### 8.55 What are the types of Machine learning?

#### Types of Machine Learning

| **Type** | **Definition** | **Examples** |
| --- | --- | --- |
| **Supervised Learning** | Learns from **labeled data** to predict outputs. | Classification, Regression |
| **Unsupervised Learning** | Finds patterns or structures in **unlabeled data**. | Clustering, Dimensionality Reduction |
| **Semi-Supervised Learning** | Uses a combination of **labeled and unlabeled data**. | Image and text classification |
| **Reinforcement Learning** | Learns by interacting with an environment using **rewards and penalties**. | Robotics, Game Playing, Autonomous Systems |

### 8.56 What different targets do classification and regression algorithms require?

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

### 8.57 What is semi-supervised learning?

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

### 8.58 Where is semi-supervised learning applied?

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

### 8.59 What is Inductive Logic Programming?

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

### 8.60 What is deductive learning and inductive learning?

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

### 8.61 How would you differentiate data mining from Machine learning?

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

### 8.62 Why does overfitting occur in ML?

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

### 8.63 What are the functions of unsupervised learning?

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

### 8.64 What are some functions of supervised learning?

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

### 8.65 What is Regularization in ML?

#### Regularization in Machine Learning

- **Regularization** is a technique used to **reduce overfitting** and improve a model’s ability to **generalize to unseen data**.
- It adds a **penalty for model complexity**, encouraging the model to learn simpler patterns.
- Common types include:
  - **L1 Regularization (Lasso):** Can make some feature weights exactly zero, helping with feature selection.
  - **L2 Regularization (Ridge):** Shrinks model weights toward zero without usually making them exactly zero.
  - **Dropout:** Randomly deactivates some neurons during training in neural networks.
- Regularization helps prevent the model from **memorizing noise in the training data**.

### 8.66 What do you understand about the Bias-Variance Tradeoff?

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

### 8.67 How can you differentiate between a parametric and a non-parametric model?

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

### 8.68 How can overfitting be avoided in ML?

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

### 8.69 How would you handle a dataset suffering from high variance?

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

### 8.70 How Regularization reduces the cost term?

- Regularization **adds a penalty term** to the original cost function.
- This penalty increases when the model's **weights become too large**.
- To minimize the total cost, the model tries to keep the weights **small**.
- Smaller weights make the model **less complex** and reduce its tendency to overfit.
- **L1 regularization:** Can make some weights exactly zero.
- **L2 regularization:** Shrinks weights toward zero.

### 8.71 What do you mean by the unreasonable effectiveness of data?

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

### 8.72 What is Online Machine Learning? How is it different from Offline machine learning? List some of its applications

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

### 8.73 What is the No Free Lunch Theorem?

> ✍️ _Drafted answer (not from the original notes)._

The **No Free Lunch (NFL) theorem** (Wolpert & Macready, 1997, for optimization; Wolpert, 1996, for supervised learning) states that **averaged over all possible problems (all possible data-generating functions), every learning algorithm has the same expected performance**. No algorithm is universally best: any gain on one class of problems is paid for by losses on another.

**What it means in practice**
- An algorithm only does well because its **inductive bias** (assumptions) matches the structure of the real problem: CNNs assume spatial locality; linear models assume additive effects; trees assume axis-aligned splits; k-NN assumes smoothness.
- Real-world problems are not uniformly distributed over "all possible functions" — they have structure — so some algorithms are reliably good in particular domains (GBMs on tabular data, Transformers on language).
- You must **empirically compare** candidate models on your data with proper validation rather than assuming a favorite always wins.

**Interview framing:** "NFL is why I start with a baseline and a few model families suited to the data type, evaluate them with cross-validation on the metric that matters, and pick based on evidence — while also weighing interpretability, latency and maintenance cost."

**Common misreading:** NFL does *not* say all algorithms are equally good on *your* problem — only when averaged over all conceivable problems.

### 8.74 What are the main differences between Structured and Unstructured Data?

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

### 8.75 What is Data Leakage? List some ways by which you can overcome this problem.

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

### 8.76 What are the different types of bias in Machine Learning?

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

### 8.77 Why does L2 regularization give sparse coefficients?

> ✍️ _Drafted answer (not from the original notes)._

**The premise is wrong: L2 (Ridge) regularization does *not* give sparse coefficients. L1 (Lasso) does.**

**Why L2 isn't sparse:** the penalty $\lambda \sum w_j^2$ has gradient $2\lambda w_j$, which shrinks toward zero in proportion to the weight. As $w_j$ gets small the push gets small too, so weights become tiny but almost never exactly zero. Ridge shrinks all coefficients smoothly and spreads weight across correlated features.

**Why L1 is sparse:**
- **Gradient view:** the penalty $\lambda \sum |w_j|$ has a constant-magnitude (sub)gradient $\lambda \,\text{sign}(w_j)$, so it keeps pushing small weights at the same rate until they reach exactly 0. The solution is the soft-thresholding operator: $w_j = \text{sign}(z)\max(|z| - \lambda, 0)$.
- **Geometric view:** the constrained problem $\min \text{loss}$ s.t. $\lVert w\rVert_1 \le t$ has a diamond-shaped feasible region with corners on the axes. The elliptical loss contours usually touch it at a corner, where some coordinates are exactly zero. The L2 ball is round, with no corners, so contact points are generically off-axis.
- **Bayesian view:** L1 equals a Laplace prior (peaked at 0) and L2 a Gaussian prior.

**Practical use:** use L1 for feature selection or interpretability, L2 for stability with multicollinearity, and **Elastic Net** for both (sparse, but it handles groups of correlated features better than Lasso, which picks one arbitrarily).

### 8.78 List some ways using which you can improve a model’s performance.

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

### 8.79 What is active learning, and where is it useful?

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

### 8.80 What types of models tend to overfit?

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

### 8.81 Can you tell the complete life cycle of a data science project?

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

### 8.82 What are the properties of a good ML model?

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

### 8.83 What do you mean by the curse of Dimensionality?

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

### 8.84 Considering the long list of machine learning algorithms, given a data set, how do you decide which one to use?

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

### 8.85 How will you know which machine learning algorithm to choose for your classification problem?

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

### 8.86 Is more data always better?

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

### 8.87 How would you differentiate between multi-label and multi-class classification?

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

### 8.88 What is an AI ontology?

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

### 8.89 Can AI systems be conscious?

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

### 8.90 What is unsupervised learning through compression?

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

### 8.91 Explain the bias-variance tradeoff mathematically and practically.

#### Bias-Variance Tradeoff

Prediction error can be decomposed as:

$\boxed{ \text{MSE}=\text{Bias}^2+\text{Variance}+\text{Irreducible Noise} }$

- **Bias** → error from overly simple assumptions → **underfitting**
- **Variance** → error from being too sensitive to training data → **overfitting**
- **Noise** → randomness that cannot be eliminated

#### As model complexity increases

$\text{Complexity}\uparrow \Rightarrow \text{Bias}\downarrow,\quad \text{Variance}\uparrow$

So:

```txt
Simple model  → High bias, low variance
Optimal model → Balanced
Complex model → Low bias, high variance
```

**Goal:** Find a model complexity that minimizes **generalization/test error**.

$Test Error≈Bias^2+Variance+Noise$

### 8.92 Why can a model with lower training loss have worse test performance?

**training loss measures fit to the training data**, while **test performance measures generalization to unseen data**.

#### Intuition

Suppose two models:

| Model | Training loss | Test loss |
| --- | --- | --- |
| Model A | 0.05 | 0.20 |
| Model B | 0.15 | 0.12 |

Model A fits the training data better, but its test performance is worse because it has **overfit**.

#### Mathematically

Training minimizes empirical risk:

$\hat{R}(f)=\frac{1}{n}\sum_{i=1}^{n}L(f(x_i),y_i)$

But what we actually care about is the expected/test risk:

$R(f)=\mathbb{E}_{(x,y)\sim P}[L(f(x),y)]$

A model can have:

$\hat{R}(f_A)<\hat{R}(f_B)$

while:

$R(f_A)>R(f_B)$

because fAf\_A has learned **training-specific patterns/noise** rather than general patterns.

#### Why this happens

- **Overfitting:** Model memorizes training examples.
- **High variance:** Small changes in data can significantly change the model.
- **Noise fitting:** Model learns random patterns in labels/features.
- **Distribution differences:** Test data may differ from training data.
- **Insufficient regularization:** Model has too much freedom to fit the training set.

#### Key idea

> **Lower training loss ≠ better generalization.**

### 8.93 How do you detect overfitting beyond simply comparing train and validation accuracy?

You can detect overfitting using several signals beyond just train vs validation accuracy:

1. **Train vs validation loss curves**
   - Training loss keeps decreasing.
   - Validation loss starts increasing.
   - This is a strong overfitting signal.
2. **Learning curves**<br>Plot performance against training-set size.
   - Large train/validation gap that persists → likely high variance/overfitting.
   - Validation performance improves as more data is added → model may be data-limited.
3. **Cross-validation**<br>Train using multiple folds.
   - Large variation in validation performance across folds suggests high variance and possible overfitting.
4. **Performance on truly unseen data**<br>Evaluate on a held-out test set that was never used for hyperparameter tuning.
   - Large drop from validation to test performance can indicate validation-set overfitting.
5. **Regularization sensitivity**<br>Increase dropout, weight decay, data augmentation, or reduce model capacity.
   - If generalization improves substantially, the original model was likely overfitting.
6. **Training-set memorization**<br>Check whether the model performs unusually well on individual training examples but poorly on perturbed or similar unseen examples.
7. **Prediction confidence**<br>An overfit classifier may produce **very high-confidence predictions on incorrect test examples**.

### 8.94 What is data leakage? Give examples that are difficult to detect.

#### Data Leakage

**Data leakage** happens when information that should be unavailable to the model at prediction time indirectly enters the training process. It often produces **unrealistically good validation/test performance**.

#### Difficult-to-detect examples

1. **Post-outcome features**
   - Predict loan default using `days_overdue`.
   - The value may only exist **after** the customer has already started defaulting.
2. **Aggregated features using future data**
   - Predict January customer behavior using `customer_total_transactions`, where the total was calculated using the **whole year**.
3. **Preprocessing before splitting**
   ```py
   scaler.fit(X)
   X = scaler.transform(X)
   train_test_split(X, y)
   ```

   The scaler has learned statistics from the test set.
4. **Target-derived features**
   - Predict fraud using `fraud_investigation_status`.
   - The investigation status may be created because the transaction was already suspected to be fraud.
5. **Duplicate or near-duplicate samples**
   - The same patient, transaction, image, or document appears in both train and test.
   - Even if IDs differ, the underlying information can be almost identical.
6. **Entity leakage**
   - Randomly split transactions instead of splitting by customer.
   - The model sees the same customer's behavior in both training and validation.
7. **Time leakage**
   - Predict an event in March using a feature calculated from April data.
   - This is especially easy to miss in time-series pipelines.
8. **Feature engineering with the target**
   ```txt
   target = customer_churn
   feature = average_support_tickets_before_churn
   ```

   If the feature calculation accidentally includes tickets created after churn, it leaks the target.
9. **Imputation leakage**
   - Missing values are filled using statistics calculated from the complete dataset rather than the training set only.
10. **Model-selection leakage**

- You repeatedly tune the model based on test-set performance.
- Even without directly training on test examples, the test set gradually influences your decisions.

### 8.95 What is the difference between model capacity and model complexity?

#### Model Capacity vs Model Complexity

**Model capacity** = how much a model is **capable of learning**.

**Model complexity** = how **complicated the particular model/solution actually is**.

|  | Capacity | Complexity |
| --- | --- | --- |
| Meaning | Potential ability to represent functions | Complexity of the learned solution |
| Example | A 100-layer neural network has high capacity | A trained model may use a relatively simple decision boundary |
| Controlled by | Parameters, architecture, hypothesis class | Architecture + learned weights + regularization |
| Main concern | Underfitting vs ability to fit data | Overfitting and generalization |

#### Example

Consider polynomial regression:

```txt
y = w₀ + w₁x + w₂x² + ... + w₁₀x¹⁰
```

A degree-10 polynomial has **higher capacity** than a degree-2 polynomial because it can represent more functions.

But if the learned degree-10 polynomial happens to have small higher-order coefficients:

```txt
w₇ ≈ 0
w₈ ≈ 0
w₉ ≈ 0
w₁₀ ≈ 0
```

the **effective complexity of the learned solution** may still be low.

### 8.96 Why does regularization reduce variance?

Regularization reduces **variance** because it prevents the model from becoming overly sensitive to the particular training dataset.

#### Intuition

A high-variance model learns not only the true pattern but also **noise and small fluctuations** in the training data.

Regularization adds a constraint or penalty that discourages overly flexible solutions:

$$
\text{Loss}_{\text{regularized}} = \text{Training Loss} + \lambda \cdot \text{Complexity Penalty}
$$

For example, L2 regularization adds:

$$
\lambda \sum_i w_i^2
$$

This encourages smaller weights.

#### Why does that reduce variance?

Suppose you train the same model on slightly different training datasets:

```txt
Without regularization:

Dataset A → Model A
Dataset B → Model B
Dataset C → Model C

Models are very different → high variance
```

With regularization:

```txt
Dataset A → Model A'
Dataset B → Model B'
Dataset C → Model C'

Models are more similar → lower variance
```

The penalty makes the model **less sensitive to individual training examples**, so changes in the training data cause smaller changes in the learned function.

### 8.97 Explain L1 vs L2 regularization geometrically.

#### L1 vs L2 Regularization Geometrically

Think of regularization as putting a **constraint around the possible weight values**.

##### L1 Regularization

$\lambda \sum_i |w_i|$

For two weights $w_1, w_2$:

$|w_1| + |w_2| \leq c$

The constraint forms a **diamond**:

```txt
          w₂
           |
          /\
         /  \
        /    \
-------+------+------- w₁
        \    /
         \  /
          \/
```

The diamond has **sharp corners on the axes**.

Because the optimum often lands on these corners:

$w_1 = 0 \quad \text{or} \quad w_2 = 0$

So **L1 encourages sparse models**, effectively performing feature selection.

---

##### L2 Regularization

$\lambda \sum_i w_i^2$

For two weights:

$w_1^2 + w_2^2 \leq c$

The constraint forms a **circle**:

```txt
          w₂
           |
        .-----.
      .'       '.
     /           \
----+-------------+---- w₁
     \           /
      '.       .'
        '-----'
```

The boundary has no sharp corners, so the optimum is **less likely to land exactly on an axis**.

Therefore, L2 usually makes weights **small**, but does not usually make them exactly zero.

**Easy memory trick:**

> **L1 = diamond = sparse**
>
> **L2 = circle = small weights**

### 8.98 What is the difference between generative and discriminative models?

The key difference is **what probability distribution they learn**.

|  | Generative | Discriminative |
| --- | --- | --- |
| Learns | $P(X,Y)$ or $P(X\mid Y)$ | $P(Y\mid X)$ |
| Goal | Model how data is generated | Separate/predict classes |
| Can generate data? | Yes | Usually no |
| Examples | Naive Bayes, GMM, VAE, GAN, LLMs | Logistic Regression, SVM, discriminative neural networks |

#### 1. Generative model

A generative model tries to understand **how the data is generated**.

For classification, it can learn:

$P(X\mid Y)P(Y)$

and then use Bayes' rule:

$P(Y\mid X)=\frac{P(X\mid Y)P(Y)}{P(X)}$

**Example:** Naive Bayes learns what spam and non-spam emails look like, then estimates whether a new email is spam.

Generative models can potentially **create new samples** similar to the training data.

---

#### 2. Discriminative model

A discriminative model directly learns the relationship between input and target:

$P(Y\mid X)$

or directly learns a decision boundary:

$f(X)\rightarrow Y$

**Example:** Logistic regression learns:

$P(\text{spam}\mid\text{email})$

without explicitly modeling how spam emails themselves are generated.

### 8.99 What is the singularity in AI?

**AI Singularity** refers to a hypothetical point where **artificial intelligence becomes capable of improving its own intelligence**, potentially leading to extremely rapid technological progress that humans can no longer effectively predict or control.

#### Key idea

```txt
Advanced AI
    ↓
AI improves AI
    ↓
More capable AI
    ↓
Faster AI improvement
    ↓
Rapid intelligence growth
```

#### Important points

- **Self-improvement:** AI could improve its own algorithms, architecture, or capabilities.
- **Recursive improvement:** Improvements could enable even faster improvements.
- **Superintelligence:** AI might eventually exceed human intellectual capabilities across many domains.
- **Unpredictability:** Future technological and societal changes could become difficult to predict.
- **Not achieved:** The AI singularity is currently **hypothetical**, not an established event or technology.

---

← [Chapter 7. Pipelines and Orchestration](../part-2-data-engineering/07-pipelines-and-orchestration.md) · [Table of contents](../README.md) · [Chapter 9. Machine Learning Algorithms](09-machine-learning-algorithms.md) →
