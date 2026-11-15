<!-- notion-id: 3d667418285480e8a34ac9ef5cdbe5fd -->
# Fundamentals of Machine Learning

ML fundamentals: training terminology (epochs, batches, gradient descent variants), learning paradigms, bias/overfitting, losses and regularization, evaluation metrics, reinforcement learning basics, and representation learning.

### Q1. What is an Epoch?

One complete pass through the entire training dataset. E.g., 5 epochs means the model has seen every training example 5 times.

### Q2. What is a Batch?

A single chunk or group of training examples processed together in one pass before the model updates its weights.

### Q3. What is Batch Size?

The number of training examples in one batch. E.g., batch size = 100 means the model processes 100 examples at a time.

### Q4. What is an Iteration?

One weight update, which happens after the model processes one batch.

### Q5. If I have 1,000 training examples and a batch size of 100, how many iterations are in one epoch?

10 iterations (1,000 ÷ 100 = 10 batches, and each batch = 1 iteration).

### Q6. If I train for 5 epochs with the setup above, how many total iterations happen?

50 total iterations (5 epochs × 10 iterations per epoch).

### Q7. Why not just feed the entire dataset in one go instead of using batches?

Large datasets often don't fit in memory (especially GPU memory), and smaller batches allow more frequent weight updates, which can speed up learning.

### Q8. What's the trade-off between large and small batch sizes?

Large batches give smoother, more accurate gradient estimates but are slower per update. Small batches update faster but produce noisier gradients (batch size = 1 is called stochastic gradient descent).

### Q9. What is (Batch) Gradient Descent?

An optimization method where the model computes the gradient using the **entire training dataset** before making a single weight update.

### Q10. What is Stochastic Gradient Descent (SGD)?

An optimization method where the model computes the gradient and updates weights using **just one training example** at a time (batch size = 1).

<!-- Merged from 05-optimization.md, where the same question appeared again ("What is stochastic gradient descent (SGD)?"). Verbatim Notion content follows. -->

**Stochastic Gradient Descent (SGD)**

- Variant of gradient descent that updates parameters using one random training sample at a time (instead of full dataset)
- Update rule: θ = θ − α∇J(θ; xᵢ, yᵢ) for a single sample i
- Much faster per update than Batch GD, especially for large datasets
- Noisy updates — high variance in path to minimum, but noise can help escape local minima/saddle points
- Requires more iterations to converge, but each iteration is cheap
- Often used with mini-batches in practice (Mini-Batch SGD) for balance between speed and stability

### Q11. What is Mini-Batch Gradient Descent?

A middle ground — the gradient is computed on a small batch (e.g., 32, 64, 100 examples) instead of the whole dataset or a single example. This is what's used in most real-world deep learning.

### Q12. Which GD is used more now?

lmost always **mini-batch gradient descent** — it balances the computational efficiency of SGD with the more stable convergence of full-batch gradient descent. It's often just called "SGD" loosely in frameworks like PyTorch/TensorFlow, even though it technically uses mini-batches.

### Q13. What are embeddings in Machine Learning?

> 🔁 _See also [01-llm-fundamentals.md → Q15](../ai-engineering/01-llm-fundamentals.md#q15-what-are-embeddings) (separate Notion notes on the same question)._

- Dense, low-dimensional vector representations of data (text, images, users, etc.) that capture semantic meaning
- Convert discrete/high-dimensional data into continuous numerical form usable by ML models
- Similar items lie close together in vector space (measured via cosine similarity, dot product, etc.)
- Learned through neural networks (e.g., Word2Vec, BERT, transformers) during training
- Reduce dimensionality while preserving meaningful relationships (e.g., king − man + woman ≈ queen)
- Used in search, recommendations, clustering, RAG, and vector databases

### Q14. What is the Softmax Activation Function?

- Converts a vector of raw scores (logits) into a probability distribution
- Formula: softmax(zᵢ) = e\^zᵢ / Σ e\^zⱼ (for all j classes)
- Output values are between 0 and 1, and sum to 1
- Used in the output layer for multi-class classification problems
- Amplifies differences — larger logits get pushed closer to 1, smaller ones toward 0
- Often paired with cross-entropy loss during training

### Q15. What is Machine Learning?

- A field of AI where systems learn patterns from data to make predictions or decisions, without being explicitly programmed with rules
- Models improve performance on a task through experience (training data)
- Main types: supervised, unsupervised, and reinforcement learning
- Core process: input data → model learns patterns → outputs predictions on new data

### Q16. Differentiate between Supervised and Unsupervised Learning.

> 🔁 _See also [01-turing-100-interview-questions.md → Q14](../resources/01-turing-100-interview-questions.md#q14-differentiate-between-supervised-and-unsupervised-learning) (separate Notion notes on the same question)._

**Supervised vs Unsupervised Learning**

| Aspect | Supervised | Unsupervised |
| --- | --- | --- |
| Data | Labeled (input-output pairs) | Unlabeled (input only) |
| Goal | Predict output for new inputs | Find hidden patterns/structure |
| Examples | Classification, Regression | Clustering, Dimensionality Reduction |
| Algorithms | Linear Regression, SVM, Decision Trees | K-Means, PCA, Hierarchical Clustering |
| Feedback | Direct error signal (label vs prediction) | No explicit feedback |
| Use cases | Spam detection, price prediction | Customer segmentation, anomaly detection |

### Q17. What is Reinforcement Learning?

> 🔁 _See also [01-llm-fundamentals.md → Q78](../ai-engineering/01-llm-fundamentals.md#q78-what-is-reinforcement-learning) (separate Notion notes on the same question)._

- Learning paradigm where an agent learns to make decisions by interacting with an environment
- Agent takes actions → receives rewards/penalties → learns policy to maximize cumulative reward
- No labeled data — learns through trial and error
- Key components: agent, environment, state, action, reward, policy
- Examples: game playing (AlphaGo), robotics, recommendation systems

### Q18. What is Bias?

- Error introduced when a model makes overly simplistic assumptions about the data
- Represents the gap between average model prediction and true value
- High bias → underfitting (model too simple, misses patterns)
- Leads to poor performance on both training and test data
- Trade-off exists with variance (Bias-Variance tradeoff)

### Q19. What is the difference between Classification and Regression?

> 🔁 _See also [01-turing-100-interview-questions.md → Q32](../resources/01-turing-100-interview-questions.md#q32-what-is-the-difference-between-regression-and-classification) (separate Notion notes on the same question)._

**Classification vs Regression**

| Aspect | Classification | Regression |
| --- | --- | --- |
| Output | Discrete categories/classes | Continuous numerical value |
| Example | Spam vs Not Spam, Cat vs Dog | House price, Temperature prediction |
| Algorithms | Logistic Regression, SVM, Decision Trees | Linear Regression, Ridge, Lasso |
| Evaluation | Accuracy, Precision, Recall, F1 | MSE, RMSE, MAE, R² |
| Output type | Class labels/probabilities | Real numbers |

### Q20. Explain Overfitting and Underfitting. How can you prevent them?

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

### Q21. What Are L1 and L2 Loss Functions?

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

### Q22. What is Regularization? Explain L1 (Lasso) and L2 (Ridge) regularization.

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

### Q23. What are Loss Functions and Cost Functions? Explain the key difference between them.

**Loss Function vs Cost Function**

- **Loss function**: Error for a single training example
- **Cost function**: Average (or sum) of loss over the entire training dataset
- Loss = per-sample error; Cost = overall model error used to guide training
- Both measure how far predictions are from actual values, but at different scales (individual vs aggregate)

### Q24. What are dropouts?

- Regularization technique used in neural networks to prevent overfitting
- Randomly "drops" (deactivates) a fraction of neurons during each training iteration
- Forces network to not rely on specific neurons, improving generalization
- Dropout rate (p) typically 0.2–0.5
- Only applied during training; all neurons active during inference (with scaling)

### Q25. What is a Perceptron?

- Simplest form of a neural network — a single-layer binary classifier
- Takes weighted inputs, sums them, adds bias, passes through activation function (step function)
- Formula: output = f(Σ(wᵢxᵢ) + b)
- Can only solve linearly separable problems (e.g., fails on XOR)
- Foundation/building block for multi-layer neural networks (MLPs)

### Q26. Explain Multilayer Perception (MLP).

- Feedforward neural network with multiple layers: input, one or more hidden layers, output
- Each layer fully connected to the next; uses non-linear activation functions (ReLU, sigmoid, tanh)
- Can solve non-linearly separable problems (unlike single perceptron)
- Trained via backpropagation + gradient descent
- Forms the basis of deep learning architectures

### Q27. What is Cross-Entropy?

> 🔁 _See also [01-llm-fundamentals.md → Q74](../ai-engineering/01-llm-fundamentals.md#q74-what-is-entropy-vs-cross-entropy) (separate Notion notes on the same question)._

- Loss function that measures difference between predicted probability distribution and true distribution
- Formula (binary): L = −\[y log(p) + (1−y) log(1−p)\]
- Formula (multi-class): L = −Σ yᵢ log(pᵢ)
- Penalizes confident wrong predictions heavily
- Commonly used with Softmax for classification tasks

### Q28. What are Logits?

- Raw, unnormalized output scores from a model's final layer, before activation (like softmax/sigmoid)
- Can be any real number (positive, negative, unbounded)
- Converted to probabilities by applying softmax (multi-class) or sigmoid (binary)
- Used directly in some loss functions (e.g., `CrossEntropyLoss` in PyTorch expects logits, not probabilities)

### Q29. Explain Cross-Validation. Why is it used?

**Cross-Validation**

- Technique to evaluate model performance by splitting data into multiple train/test subsets
- Most common: **k-fold CV** — data split into k folds, model trained on k-1 folds, tested on remaining fold, repeated k times
- Final performance = average across all folds

**Why used**

- Gives more reliable performance estimate than a single train/test split
- Reduces risk of overfitting to one particular split
- Makes better use of limited data
- Helps in hyperparameter tuning and model selection

### Q30. What is Confusion Matrix?

> 🔁 _See also [01-turing-100-interview-questions.md → Q21](../resources/01-turing-100-interview-questions.md#q21-what-is-the-confusion-matrix) (separate Notion notes on the same question)._

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

### Q31. What are precision, recall, and F1-score?

**Precision, Recall, F1-Score**

- **Precision**: Of predicted positives, how many are actually correct → TP / (TP + FP), how precise are my positive predictions?
- **Recall**: Of actual positives, how many were correctly identified → TP / (TP + FN), how many of the actual positive did i find/recall?
- **F1-Score**: Harmonic mean of precision and recall → 2 × (P × R) / (P + R)
- Precision matters when false positives are costly (e.g., spam detection)
- Recall matters when false negatives are costly (e.g., disease detection)
- F1 balances both, useful for imbalanced datasets

### Q32. What is anomaly detection?

- Technique to identify data points, events, or observations that deviate significantly from normal patterns
- Also called outlier detection
- **Methods**:
  - Statistical (z-score, IQR)
  - Distance-based (k-NN, clustering)
  - Density-based (DBSCAN, LOF)
  - ML-based (Isolation Forest, One-Class SVM, Autoencoders)
- **Use cases**: Fraud detection, network intrusion, equipment failure, medical diagnosis, quality control

### Q33. What is the difference between policy-based and value-based methods?

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

### Q34. What is Q-Learning?

- Model-free, value-based reinforcement learning algorithm
- Learns a Q-value function Q(s, a) representing expected cumulative reward for taking action a in state s
- **Update rule**: Q(s,a) ← Q(s,a) + α\[r + γ·max Q(s',a') − Q(s,a)\]
  - α = learning rate, γ = discount factor, r = reward
- Off-policy — learns optimal policy regardless of agent's actions during exploration
- Uses a Q-table (or neural network in Deep Q-Learning/DQN) to store/approximate values
- Agent picks action with highest Q-value (often via ε-greedy for exploration)

### Q35. Explain the concept of exploration vs. exploitation.

- Core dilemma in reinforcement learning: whether to try new actions or use known best actions
- **Exploration**: Trying new/unknown actions to discover potentially better rewards
- **Exploitation**: Choosing the best-known action to maximize immediate reward
- Too much exploration → slow convergence, inefficient
- Too much exploitation → risk getting stuck in suboptimal policy (local optima)
- **Common strategies**: ε-greedy (random action with probability ε), decaying ε, softmax action selection, UCB (Upper Confidence Bound)

### Q36. Explain the curse of dimensionality and how to address it.

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

### Q37. Explain Local Loss, Focal Loss, and Gradient Blending in the context of Multi-Task Learning.

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

### Q38. Explain Contrastive Learning

- Self-supervised learning technique that learns representations by comparing similar vs dissimilar pairs
- Pulls similar (positive) pairs closer together in embedding space, pushes dissimilar (negative) pairs apart
- Doesn't require labeled data — positive pairs often created via augmentation (e.g., two crops of same image)
- **Common loss**: InfoNCE / Contrastive loss (based on similarity scores, e.g., cosine similarity)
- **Examples**: SimCLR, MoCo (images), CLIP (text-image)
- **Use cases**: Representation learning, embedding models, self-supervised pretraining

### Q39. What is Generative AI?

- Subset of AI focused on creating new content (text, images, audio, video, code) rather than just analyzing/predicting
- Learns underlying patterns/distribution of training data to generate novel, similar outputs
- **Key architectures**: GANs, VAEs, Diffusion Models, Transformers (GPT, etc.)
- **Examples**: ChatGPT (text), DALL-E/Midjourney (images), Stable Diffusion, GitHub Copilot (code)
- **Use cases**: Content creation, code generation, data augmentation, chatbots, design
