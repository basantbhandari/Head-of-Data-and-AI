# Optimization

Optimization interview questions: gradient descent, vanishing gradients, learning rates, hyperparameter tuning (grid/random/Bayesian, TPE), quantization, fairness, and adaptive optimizers (Adam, RMSprop, Adagrad).

### Q1. What is gradient descent? How does it work?

**Gradient Descent**

- Optimization algorithm used to minimize a loss function by iteratively updating model parameters
- Moves parameters in the direction opposite to the gradient (steepest descent)

**How it works**

1. Initialize parameters (weights) randomly
2. Compute gradient (partial derivative) of loss function w.r.t. each parameter
3. Update parameters: θ = θ − α∇J(θ)
   - α = learning rate (step size)
   - ∇J(θ) = gradient of cost function
4. Repeat until convergence (loss stops decreasing significantly or max iterations reached)

**Key considerations**

- Learning rate too high → overshoots minimum, diverges
- Learning rate too low → slow convergence
- Can get stuck in local minima (less issue in deep learning with high-dimensional loss surfaces)
- Variants: Batch, Stochastic (SGD), Mini-Batch, and adaptive optimizers (Adam, RMSProp)

> This set also asks "What is stochastic gradient descent (SGD)?" here. Its answer was merged into [02 · Fundamentals of Machine Learning: What is Stochastic Gradient Descent (SGD)?](02-fundamentals-of-machine-learning.md#q10-what-is-stochastic-gradient-descent-sgd), where the same question appears first.

### Q2. What are vanishing gradients?

**Vanishing Gradients**

- Problem where gradients become extremely small during backpropagation, especially in deep networks
- As gradients are multiplied through many layers (chain rule), they shrink exponentially toward zero
- Early layers receive near-zero gradient updates → learn very slowly or stop learning entirely
- Common with **sigmoid/tanh** activations (derivatives \< 1, compound over layers)

**Causes**

- Deep networks (many layers)
- Saturating activation functions (sigmoid, tanh) where gradient ≈ 0 at extremes
- Poor weight initialization

**Solutions**

- Use **ReLU** (and variants like Leaky ReLU) — doesn't saturate for positive inputs
- **Batch Normalization** — stabilizes/normalizes layer inputs
- **Residual connections/skip connections** (ResNets) — allow gradients to flow directly
- **Proper weight initialization** (Xavier/Glorot, He initialization)
- **LSTM/GRU** for RNNs — designed to combat vanishing gradients in sequential data
- Gradient clipping (helps more with exploding gradients)

### Q3. What is a learning rate? How to choose a good one?

**Learning Rate**

- Hyperparameter controlling step size of parameter updates during gradient descent
- Update rule: θ = θ − α∇J(θ), where α = learning rate

**Effects**

- **Too high**: overshoots minimum, causes divergence/oscillation, unstable training
- **Too low**: slow convergence, may get stuck, wastes compute time

**How to choose**

- **Grid/random search**: try values like 0.1, 0.01, 0.001, 0.0001
- **Learning rate schedules**: decay LR over time (step decay, exponential decay, cosine annealing)
- **Warm-up**: start small, gradually increase (common in transformers)
- **Adaptive optimizers**: Adam, RMSProp — automatically adjust LR per parameter
- **LR range test**: gradually increase LR, plot loss vs LR, pick value just before loss diverges (used in fast.ai/cyclical LR)
- Monitor training/validation loss curves to detect instability (too high) or slow progress (too low)

### Q4. How does the learning rate affect model training?

**Effect of Learning Rate on Training**

- **Too high**:
  - Overshoots the minimum, loss oscillates or diverges
  - Training becomes unstable, may never converge
- **Too low**:
  - Very slow convergence, takes many epochs
  - Risk of getting stuck in local minima/plateaus
  - Wastes computation time
- **Optimal LR**:
  - Steady, efficient decrease in loss
  - Faster convergence to good minimum
- **Practical effects**
  - Affects training time and final model quality
  - Interacts with batch size (larger batch often needs larger LR)
  - Often combined with LR schedules (decay) — high LR early for fast progress, lower LR later for fine-tuning/stability

### Q5. How do you approach hyperparameter tuning?

**Hyperparameter Tuning Approaches**

**Search strategies**

- **Grid Search**: exhaustive search over predefined parameter grid — thorough but computationally expensive
- **Random Search**: samples random combinations — often more efficient than grid search for high-dimensional spaces
- **Bayesian Optimization**: builds probabilistic model of objective function, chooses next params based on past results (e.g., Optuna, Hyperopt) — more sample-efficient
- **Genetic Algorithms**: evolutionary approach, mutates/combines good parameter sets

**Process**

1. Define hyperparameter search space (ranges, distributions)
2. Choose evaluation metric aligned with business goal
3. Use cross-validation to evaluate each combination (avoid overfitting to single split)
4. Select best-performing combination on validation set
5. Final evaluation on held-out test set

**Practical tips**

- Start with coarse search, then refine around promising region
- Use early stopping to save compute (skip unpromising trials early)
- Prioritize most impactful hyperparameters first (e.g., learning rate before minor ones)
- Tools: Optuna, Ray Tune, scikit-learn's GridSearchCV/RandomizedSearchCV, Hyperopt

### Q6. What is model quantization, and when would you use it?

**Model Quantization**

- Technique to reduce model size and speed up inference by converting weights/activations from high precision (FP32) to lower precision (FP16, INT8, INT4)
- Reduces memory footprint and computational cost with minimal accuracy loss

**Types**

- **Post-Training Quantization (PTQ)**: Quantize after training a full-precision model — fast, simple
- **Quantization-Aware Training (QAT)**: Simulates quantization during training — better accuracy retention
- **Dynamic Quantization**: Weights quantized ahead of time, activations quantized at runtime
- **Static Quantization**: Both weights and activations quantized ahead of time using calibration data

**When to use**

- Deploying models on edge devices/mobile (limited memory, compute)
- Reducing inference latency for production serving
- Lowering cloud inference costs at scale
- Running large models (LLMs) on consumer hardware

**Trade-offs**

- Some accuracy degradation (usually small, more with lower bit-widths like INT4)
- Not all hardware supports all quantization types equally
- Higher compression (INT4, INT8) = more speed/memory savings but higher accuracy risk

### Q7. How do you ensure fairness and reduce bias in ML models?

**Ensuring Fairness & Reducing Bias in ML Models**

**Data-level approaches**

- Audit training data for representation bias (check demographic distributions)
- Collect more diverse/balanced data across sensitive groups
- Remove/reduce proxy variables that correlate with sensitive attributes (zip code → race proxy)
- Re-sampling techniques to balance underrepresented groups

**Model-level approaches**

- **Fairness-aware algorithms**: constrained optimization during training (e.g., adversarial debiasing)
- **Regularization techniques**: penalize unfair predictions across groups
- Use fairness metrics as part of loss function/objective

**Evaluation metrics**

- **Demographic parity**: equal positive prediction rate across groups
- **Equal opportunity**: equal true positive rate across groups
- **Equalized odds**: equal TPR and FPR across groups
- **Disparate impact ratio**: compare outcome rates between groups

**Post-processing approaches**

- Adjust decision thresholds per group to equalize outcomes
- Calibrate predictions per subgroup

**Process/governance**

- Diverse team involvement in model design/review
- Regular bias audits and monitoring in production
- Use explainability tools (SHAP, LIME) to understand feature influence
- Document model limitations (model cards, datasheets)

### Q8. Explain Grid Search vs Random Search vs Bayesian Optimization.

**Grid Search vs Random Search vs Bayesian Optimization**

| Aspect | Grid Search | Random Search | Bayesian Optimization |
| --- | --- | --- | --- |
| Method | Exhaustive search over all parameter combinations | Random sampling of parameter combinations | Probabilistic model guides search based on past results |
| Coverage | Complete (within grid) | Partial, random | Targeted, informed |
| Efficiency | Low — expensive, scales poorly with dimensions | Higher — often finds good params faster | Highest — sample-efficient |
| Best for | Small search spaces, few hyperparameters | Larger/high-dimensional spaces | Expensive-to-evaluate models (deep learning) |
| How next trial chosen | Fixed grid, no learning from past trials | Random, no learning from past trials | Uses surrogate model (e.g., Gaussian Process) + acquisition function to pick promising next trial |
| Parallelizable | Yes | Yes | Harder (sequential by nature, though batched variants exist) |
| Tools | GridSearchCV | RandomizedSearchCV | Optuna, Hyperopt, scikit-optimize |

- **Grid Search**: guarantees finding best combo within grid, but computationally infeasible for many hyperparameters
- **Random Search**: often outperforms grid search in high dimensions since not all hyperparameters matter equally
- **Bayesian Optimization**: balances exploration/exploitation, converges to good solutions with fewer trials — ideal for costly model training (e.g., deep learning, LLM fine-tuning)

### Q9. Explain TPE hyperparameter optimization.

**TPE (Tree-structured Parzen Estimator)**

- A Bayesian optimization algorithm for hyperparameter tuning (used by default in Optuna, Hyperopt)
- Sequential Model-Based Optimization (SMBO) approach — builds probabilistic models to guide search

**How it works**

1. Evaluate a few random hyperparameter configurations initially, record results (loss values)
2. Split observed trials into two groups based on a performance threshold (quantile γ, e.g., top 15%):
   - **l(x)**: density of hyperparameters that gave **good** results
   - **g(x)**: density of hyperparameters that gave **poor** results
3. Model l(x) and g(x) separately (instead of one joint model like Gaussian Process)
4. Select next hyperparameters that **maximize l(x)/g(x)** ratio (Expected Improvement proxy) — favors regions likely to yield good results and unlikely to yield bad ones
5. Evaluate new point, add to observations, repeat

**Why it's used**

- Handles **conditional/hierarchical hyperparameters** well (e.g., different params depending on model type) — advantage over Gaussian Process-based BO
- More scalable to high-dimensional, mixed-type (categorical + continuous) search spaces
- Efficient — needs fewer trials than grid/random search to find good hyperparameters

**vs Gaussian Process-based BO**

- GP models p(y\|x) directly; TPE models p(x\|y) via l(x) and g(x) — makes TPE better suited for tree-structured/conditional search spaces

### Q10. Explain Bayesian Optimization.

**Bayesian Optimization**

- Sequential, sample-efficient strategy for optimizing expensive black-box functions (e.g., hyperparameter tuning, where each evaluation = full model training)
- Balances **exploration** (trying uncertain regions) and **exploitation** (refining known good regions)

**How it works**

1. Build a **surrogate model** (probabilistic model) approximating the true objective function — commonly a **Gaussian Process (GP)**
2. Use an **acquisition function** to decide the next point to evaluate, based on surrogate's predicted mean and uncertainty
   - Common acquisition functions: Expected Improvement (EI), Upper Confidence Bound (UCB), Probability of Improvement (PI)
3. Evaluate the true objective function at the chosen point (e.g., train model with those hyperparameters)
4. Update surrogate model with new observation
5. Repeat until budget exhausted or convergence

**Key components**

- **Surrogate model**: Gaussian Process (most common), Random Forest, or TPE-based density estimators
- **Acquisition function**: guides exploration/exploitation trade-off

**Advantages**

- Requires far fewer evaluations than grid/random search to find good hyperparameters
- Well-suited for expensive-to-evaluate functions (deep learning, LLM training)

**Limitations**

- GP-based BO scales poorly with high dimensions and large number of observations (cubic complexity)
- Sequential by nature — harder to parallelize (though batch variants exist)
- Choice of surrogate model/kernel can affect performance

### Q11. Explain Adam Optimizer.

**Adam Optimizer (Adaptive Moment Estimation)**

- Adaptive learning rate optimization algorithm combining **Momentum** + **RMSProp**
- Most widely used optimizer in deep learning

**How it works**

- Maintains two moving averages per parameter:
  - **First moment (m)**: exponential moving average of gradients (like Momentum) → mₜ = β₁mₜ₋₁ + (1−β₁)gₜ
  - **Second moment (v)**: exponential moving average of squared gradients (like RMSProp) → vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²
- **Bias correction** (since m, v initialized at 0, biased toward 0 early on):
  - m̂ₜ = mₜ / (1−β₁ᵗ), v̂ₜ = vₜ / (1−β₂ᵗ)
- **Update rule**: θ = θ − α · m̂ₜ / (√v̂ₜ + ε)

**Hyperparameters**

- α (learning rate) — typically 0.001
- β₁ — decay rate for first moment, typically 0.9
- β₂ — decay rate for second moment, typically 0.999
- ε — small constant to prevent division by zero (e.g., 1e-8)

**Advantages**

- Adaptive per-parameter learning rates — works well with sparse gradients
- Combines benefits of Momentum (fast convergence) + RMSProp (handles varying gradient scales)
- Requires little tuning, works well out-of-the-box

**Limitations**

- Can generalize worse than SGD with momentum in some cases (sharper minima)
- Extra memory overhead (stores m and v per parameter)

### Q12. Explain the RMSprop Optimizer.

**RMSprop (Root Mean Square Propagation)**

- Adaptive learning rate optimizer that adjusts learning rate per parameter based on recent gradient magnitudes
- Designed to fix Adagrad's problem of learning rate shrinking too aggressively over time

**How it works**

- Maintains exponential moving average of **squared gradients**:
  - vₜ = β·vₜ₋₁ + (1−β)·gₜ²
- **Update rule**: θ = θ − α · gₜ / (√vₜ + ε)
  - α = learning rate, β = decay rate (typically 0.9), ε = small constant (avoid division by zero)

**Key idea**

- Parameters with large recent gradients → get smaller effective learning rate
- Parameters with small recent gradients → get larger effective learning rate
- Normalizes step size based on gradient magnitude history

**Advantages**

- Handles non-stationary objectives well (good for RNNs)
- Prevents learning rate from shrinking too fast (unlike Adagrad, uses moving average not cumulative sum)
- Works well with mini-batch training, reduces oscillations

**Limitations**

- Still requires tuning learning rate
- Adam (which builds on RMSprop + Momentum) generally preferred in most modern applications

### Q13. What is Adagrad Optimizer?

**Adagrad (Adaptive Gradient Algorithm)**

- Adaptive learning rate optimizer that scales the learning rate per parameter based on the **cumulative sum of past squared gradients**

**How it works**

- Accumulates squared gradients: Gₜ = Gₜ₋₁ + gₜ²
- **Update rule**: θ = θ − α · gₜ / (√Gₜ + ε)
  - α = initial learning rate, ε = small constant (to avoid division by zero)

**Key idea**

- Parameters with large historical gradients → smaller effective learning rate
- Parameters with small/infrequent gradients → larger effective learning rate
- Good for sparse features (e.g., NLP, where rare words get larger updates)

**Advantages**

- No manual learning rate tuning per parameter needed
- Well-suited for sparse data (sparse gradients get bigger updates)

**Limitations**

- **Learning rate shrinks monotonically** — since Gₜ only grows, learning rate can become too small, causing training to stall prematurely
- This limitation led to the development of **RMSprop** and **Adam**, which use moving averages instead of cumulative sums
