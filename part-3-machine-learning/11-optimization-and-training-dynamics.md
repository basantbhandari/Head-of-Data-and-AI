# Chapter 11. Optimization and Training Dynamics

_Part III — Machine Learning_

Gradient descent and its variants, learning-rate schedules, loss landscapes, generalization of over-parameterized models and hyperparameter tuning.

This chapter contains **38 questions**.

## Contents

- **11.1** [What is gradient descent? How does it work?](#111-what-is-gradient-descent-how-does-it-work)
- **11.2** [What are vanishing gradients?](#112-what-are-vanishing-gradients)
- **11.3** [What is a learning rate? How to choose a good one?](#113-what-is-a-learning-rate-how-to-choose-a-good-one)
- **11.4** [How does the learning rate affect model training?](#114-how-does-the-learning-rate-affect-model-training)
- **11.5** [How do you approach hyperparameter tuning?](#115-how-do-you-approach-hyperparameter-tuning)
- **11.6** [What is model quantization, and when would you use it?](#116-what-is-model-quantization-and-when-would-you-use-it)
- **11.7** [How do you ensure fairness and reduce bias in ML models?](#117-how-do-you-ensure-fairness-and-reduce-bias-in-ml-models)
- **11.8** [Explain Grid Search vs Random Search vs Bayesian Optimization.](#118-explain-grid-search-vs-random-search-vs-bayesian-optimization)
- **11.9** [Explain TPE hyperparameter optimization.](#119-explain-tpe-hyperparameter-optimization)
- **11.10** [Explain Bayesian Optimization.](#1110-explain-bayesian-optimization)
- **11.11** [Explain Adam Optimizer.](#1111-explain-adam-optimizer)
- **11.12** [Explain the RMSprop Optimizer.](#1112-explain-the-rmsprop-optimizer)
- **11.13** [What is Adagrad Optimizer?](#1113-what-is-adagrad-optimizer)
- **11.14** [What strategies would you use to optimize the training time for a deep learning model?](#1114-what-strategies-would-you-use-to-optimize-the-training-time-for-a-deep-learning-model)
- **11.15** [What are the differences between stochastic gradient descent (SGD) and gradient descent (GD)?](#1115-what-are-the-differences-between-stochastic-gradient-descent-sgd-and-gradient-descent-gd)
- **11.16** [Explain the difference between the Loss functions and Cost functions.](#1116-explain-the-difference-between-the-loss-functions-and-cost-functions)
- **11.17** [Difference between convex and non-convex cost functions; what does it mean when a cost function is non-convex?](#1117-difference-between-convex-and-non-convex-cost-functions-what-does-it-mean-when-a-cost-function-is-non-convex)
- **11.18** [Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?](#1118-which-gradient-descent-algorithm-among-those-we-discussed-will-reach-the-vicinity-of-the-optimal-solution-the-fastest-which-will-actually-converge)
- **11.19** [Which hyperparameter tuning strategies (in general) do you know?](#1119-which-hyperparameter-tuning-strategies-in-general-do-you-know)
- **11.20** [Explain how gradient descent actually works.](#1120-explain-how-gradient-descent-actually-works)
- **11.21** [Why does SGD sometimes generalize better than full-batch gradient descent?](#1121-why-does-sgd-sometimes-generalize-better-than-full-batch-gradient-descent)
- **11.22** [Compare SGD, Momentum, Adam, AdamW, and Adagrad.](#1122-compare-sgd-momentum-adam-adamw-and-adagrad)
- **11.23** [Why is AdamW generally preferred over Adam with L2 regularization?](#1123-why-is-adamw-generally-preferred-over-adam-with-l2-regularization)
- **11.24** [What causes vanishing and exploding gradients?](#1124-what-causes-vanishing-and-exploding-gradients)
- **11.25** [How does gradient clipping solve exploding gradients?](#1125-how-does-gradient-clipping-solve-exploding-gradients)
- **11.26** [What is a learning-rate scheduler, and why is it important?](#1126-what-is-a-learning-rate-scheduler-and-why-is-it-important)
- **11.27** [Compare cosine decay, step decay, warmup, and one-cycle learning rates.](#1127-compare-cosine-decay-step-decay-warmup-and-one-cycle-learning-rates)
- **11.28** [Why can a very high learning rate prevent convergence?](#1128-why-can-a-very-high-learning-rate-prevent-convergence)
- **11.29** [What happens when the learning rate is extremely small?](#1129-what-happens-when-the-learning-rate-is-extremely-small)
- **11.30** [What is the difference between a local minimum, saddle point, and global minimum?](#1130-what-is-the-difference-between-a-local-minimum-saddle-point-and-global-minimum)
- **11.31** [Why are saddle points particularly relevant in high-dimensional neural networks?](#1131-why-are-saddle-points-particularly-relevant-in-high-dimensional-neural-networks)
- **11.32** [What is sharpness of a loss landscape?](#1132-what-is-sharpness-of-a-loss-landscape)
- **11.33** [Why can flatter minima sometimes generalize better?](#1133-why-can-flatter-minima-sometimes-generalize-better)
- **11.34** [What is the relationship between batch size and generalization?](#1134-what-is-the-relationship-between-batch-size-and-generalization)
- **11.35** [Why does overparameterization not necessarily cause worse generalization?](#1135-why-does-overparameterization-not-necessarily-cause-worse-generalization)
- **11.36** [Explain the double-descent phenomenon.](#1136-explain-the-double-descent-phenomenon)
- **11.37** [What is the implicit bias of gradient descent?](#1137-what-is-the-implicit-bias-of-gradient-descent)
- **11.38** [Why can SGD find solutions that generalize despite having enormous parameter counts?](#1138-why-can-sgd-find-solutions-that-generalize-despite-having-enormous-parameter-counts)

---

### 11.1 What is gradient descent? How does it work?

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

> This set also asks "What is stochastic gradient descent (SGD)?" here. Its answer was merged into [02 · Fundamentals of Machine Learning: What is Stochastic Gradient Descent (SGD)?](08-machine-learning-fundamentals.md#821-what-is-stochastic-gradient-descent-sgd), where the same question appears first.

### 11.2 What are vanishing gradients?

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

### 11.3 What is a learning rate? How to choose a good one?

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

### 11.4 How does the learning rate affect model training?

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

### 11.5 How do you approach hyperparameter tuning?

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

### 11.6 What is model quantization, and when would you use it?

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

### 11.7 How do you ensure fairness and reduce bias in ML models?

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

### 11.8 Explain Grid Search vs Random Search vs Bayesian Optimization.

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

### 11.9 Explain TPE hyperparameter optimization.

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

### 11.10 Explain Bayesian Optimization.

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

### 11.11 Explain Adam Optimizer.

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

### 11.12 Explain the RMSprop Optimizer.

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

### 11.13 What is Adagrad Optimizer?

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

### 11.14 What strategies would you use to optimize the training time for a deep learning model?

> ✍️ _Drafted answer (not from the original notes)._

**Profile before optimising** (PyTorch Profiler, `nvidia-smi`, GPU utilisation). Find out whether the job is input-bound, compute-bound or communication-bound.

**Input pipeline:** more DataLoader workers, `pin_memory`, prefetching, preprocessed and cached datasets, GPU-side augmentation (DALI), and sharded streaming formats.

**Compute efficiency:**
- **Mixed precision** (bf16/fp16 with AMP): often a 2–3× speedup and half the memory.
- `torch.compile`, fused kernels and FlashAttention for transformers.
- Larger batch sizes to fill the GPU, with LR scaling and warmup (LARS/LAMB for very large batches). Use gradient accumulation if memory is tight.
- Gradient checkpointing to trade compute for memory, allowing larger batches or models.
- Channels-last memory format for CNNs, and tensor-core-friendly dimensions (multiples of 8).

**Algorithmic:**
- Transfer learning or fine-tuning instead of training from scratch, and parameter-efficient fine-tuning (LoRA).
- A better optimiser and schedule (AdamW with one-cycle or cosine), and early stopping.
- Smaller or distilled architectures. Progressive resizing (train on low resolution first).
- Data curation: deduplicate and drop low-value samples.

**Scale out:** DDP for data parallelism; FSDP/DeepSpeed ZeRO when the model doesn't fit; fast interconnects (NVLink, InfiniBand).

**Experiment efficiency:** tune hyperparameters on a subset or smaller model, use early-stopping search (ASHA/Hyperband via Ray Tune or Optuna), and use spot instances with checkpointing to cut cost.

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(x).loss
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
```

### 11.15 What are the differences between stochastic gradient descent (SGD) and gradient descent (GD)?

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

### 11.16 Explain the difference between the Loss functions and Cost functions.

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

### 11.17 Difference between convex and non-convex cost functions; what does it mean when a cost function is non-convex?

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

### 11.18 Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?

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

### 11.19 Which hyperparameter tuning strategies (in general) do you know?

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

### 11.20 Explain how gradient descent actually works.

#### Gradient Descent

**Gradient descent** is an optimization algorithm used to find model parameters that minimize a loss function.

Imagine the loss function as a **hill/valley**. We want to move downhill until we reach a minimum.

#### How it works

1. **Initialize parameters**
   ```txt
   w = random values
   ```
2. **Calculate the loss**<br>Measure how wrong the model's predictions are.
3. **Calculate the gradient**
   The gradient tells us **which direction increases the loss most quickly**.
4. **Move in the opposite direction**
   $w := w - \eta \frac{\partial L}{\partial w}$

   where:

   - $w$ = model parameter
   - $L$ = loss
   - $\eta$ = learning rate
   - $\frac{\partial L}{\partial w} $= gradient
5. **Repeat**
   Continue updating the parameters until the loss stops improving.

#### Intuition

```txt
High Loss
    ●
     \
      \
       ●
        \
         ●  ← minimum
```

The **gradient tells you which way is uphill**, so gradient descent moves in the **opposite direction**.

#### Learning rate matters

```txt
Too small → very slow
Good      → converges efficiently
Too large → may overshoot/diverge
```

### 11.21 Why does SGD sometimes generalize better than full-batch gradient descent?

|  | **Full-Batch GD** | **SGD** | **Mini-Batch SGD** |
| --- | --- | --- | --- |
| Samples/update | Entire dataset | 1 | Small batch, e.g. 32/64/128 |
| Gradient | Exact | Very noisy | Moderately noisy |
| Updates | 1 per epoch | Many | Many |
| Memory | High | Very low | Moderate |
| Training speed | Slow for large data | Fast updates | Usually fastest in practice |
| Convergence | Smooth | Noisy | Relatively stable |
| Generalization | Can be weaker | Often good | Often good |
| Practical use | Less common for large data | Less common alone | **Most common** |

Full Batch<br>All data → Gradient → Update

SGD<br>1 sample → Gradient → Update<br>1 sample → Gradient → Update<br>1 sample → Gradient → Update

Mini-Batch SGD<br>32 samples → Gradient → Update<br>32 samples → Gradient → Update<br>32 samples → Gradient → Update

#### Why can SGD generalize better?

**SGD (Stochastic Gradient Descent)** uses a small random batch of training examples for each update, so its gradient contains **noise**.

That noise can actually help optimization:

- **Avoids sharp/local minima:** Noise can push the model away from solutions that fit training data too specifically.
- **Acts like implicit regularization:** The randomness discourages overly complex solutions.
- **Explores more solutions:** Instead of following one perfectly precise gradient path, SGD takes slightly noisy steps.
- **Often favors flatter minima:** Flatter solutions tend to be less sensitive to small parameter changes and often generalize better.

#### Full-batch vs SGD

```txt
Full Batch:
Exact gradient → smooth optimization → can fit training data closely

SGD:
Noisy gradient → more exploration → implicit regularization
```

#### Important nuance

SGD **does not always generalize better**. Its advantage depends on the architecture, learning rate, batch size, optimizer, training duration, and dataset.

### 11.22 Compare SGD, Momentum, Adam, AdamW, and Adagrad.

#### SGD vs Momentum vs Adam vs AdamW vs Adagrad

| Optimizer | Main idea | Learning rate | Strength | Weakness |
| --- | --- | --- | --- | --- |
| **SGD** | Follow gradient directly | Fixed/global | Simple, often good generalization | Can be slow |
| **Momentum** | Accumulate past gradients | Global | Faster, smoother convergence | Needs tuning |
| **Adagrad** | Adapt LR per parameter | Adaptive | Good for sparse features | LR can become too small |
| **Adam** | Momentum + adaptive LR | Adaptive | Fast, robust, easy to tune | Can generalize differently from SGD |
| **AdamW** | Adam + **decoupled weight decay** | Adaptive | Strong default for many deep-learning models | Still needs tuning |

#### 1. SGD

$\theta_{t+1}=\theta_t-\eta g_t$

Uses only the current gradient.

**Intuition:** Take a step directly downhill.

---

#### 2. Momentum

Keeps a moving average of previous gradients:

$v_t=\beta v_{t-1}+g_t$

$\theta_{t+1}=\theta_t-\eta v_t$

**Intuition:** Like a ball gaining velocity while moving downhill.

It reduces oscillations and can accelerate movement in consistent directions.

---

#### 3. Adagrad

Adjusts the learning rate separately for each parameter based on accumulated squared gradients:

$\theta_{t+1} $ = $\theta_t- \frac{\eta}{\sqrt{G_t+\epsilon}}g_t$

Frequently updated parameters get smaller learning rates.

**Good for:** sparse features, such as NLP/count-based features.

**Problem:** The accumulated gradient keeps growing, so learning rates can eventually become extremely small.

---

#### 4. Adam

Combines **Momentum + adaptive learning rates**.

It maintains:

$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$

$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$

Then updates using both estimates.

**Intuition:**

> Momentum tells you **which direction to move**, while the adaptive term controls **how large the step should be for each parameter**.

---

#### 5. AdamW

AdamW is essentially **Adam with decoupled weight decay**.

Instead of mixing L2 regularization directly into Adam's gradient, weight decay is applied separately:

$\theta \leftarrow \theta-\eta(\text{Adam update})-\eta\lambda\theta$

This makes weight decay behave more predictably and is widely used for modern deep-learning models.

#### Easy interview summary

> **SGD:** simple gradient updates
>
> **Momentum:** SGD + accumulated direction
>
> **Adagrad:** parameter-specific learning rates
>
> **Adam:** Momentum + adaptive learning rates
>
> **AdamW:** Adam + decoupled weight decay

### 11.23 Why is AdamW generally preferred over Adam with L2 regularization?

#### Why AdamW over Adam + L2?

The key difference is **how weight decay is applied**.

With **Adam + L2**, the regularization term is added to the gradient:

$g_t' = g_t + \lambda\theta_t$

Adam then **rescales this combined gradient** using its adaptive moments.

With **AdamW**, weight decay is applied **separately**:

$\theta_{t+1} $ = $\theta_t-\eta\cdot AdamUpdate-\eta\lambda\theta_t$

#### Why is this better?

Because in Adam, the adaptive scaling means the effective strength of L2 regularization can vary across parameters.

AdamW **decouples weight decay from the adaptive gradient update**, making λ\\lambda behave more directly as a weight-decay hyperparameter.

### 11.24 What causes vanishing and exploding gradients?

#### Vanishing vs Exploding Gradients

They happen mainly because gradients are **multiplied repeatedly through many layers/time steps** during backpropagation.

For a deep network:

$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial W_n} \prod_{i=2}^{n} \frac{\partial h_i}{\partial h_{i-1}}$

#### 1. Vanishing gradients

If the derivatives are mostly **less than 1**:

$0.5^{20} \approx 9.5\times10^{-7}$

The gradient becomes extremely small.

**Causes:**

- Deep networks
- Sigmoid/tanh saturation
- Poor initialization
- Repeated multiplication by small derivatives

**Effect:** Early layers learn very slowly or stop learning.

#### 2. Exploding gradients

If the derivatives are mostly **greater than 1**:

$2^{20} = 1,048,576$

The gradient becomes extremely large.

**Causes:**

- Large weights
- Poor initialization
- Very deep networks
- Long sequences in RNNs

**Effect:** Unstable training, huge parameter updates, or `NaN` losses.

#### Common solutions

**Vanishing:**

- ReLU/GELU activations
- Xavier/He initialization
- Residual connections
- Normalization
- LSTM/GRU for long sequences

**Exploding:**

- Gradient clipping
- Proper initialization
- Smaller learning rate
- Normalization

### 11.25 How does gradient clipping solve exploding gradients?

**Gradient clipping limits the size of the gradient before updating the model parameters.**

Suppose the gradient becomes huge:

$\|g\| = 1000$

Instead of applying the full gradient, we set a maximum norm, say $5$:

$g_{\text{clipped}} $ = $g\cdot\frac{5}{\|g\|}$

So:

$\|g_{\text{clipped}}\|=5$

#### Intuition

```txt
Normal gradient:
        ↓
       1000
        ↓
Huge parameter update → unstable training

With clipping:
        ↓
       1000
        ↓
      clipped
        ↓
         5
        ↓
Controlled update
```

The **direction is preserved**, but the magnitude is limited.

#### Why it helps

Without clipping:

$\theta_{t+1}=\theta_t-\eta g$

A huge gg causes a huge parameter update.

With clipping:

$\theta_{t+1}=\theta_t-\eta g_{\text{clipped}}$

the update remains bounded.

### 11.26 What is a learning-rate scheduler, and why is it important?

A **learning-rate scheduler** automatically changes the learning rate during training instead of keeping it fixed.

$\theta_{t+1}=\theta_t-\eta_t\nabla L(\theta_t)$

where $\eta_t$ changes over time.

#### Why is it important?

- **Large LR initially** → faster learning
- **Smaller LR later** → more stable convergence
- Helps avoid **overshooting** the minimum
- Can improve **final model performance**

#### Common schedulers

- **Step Decay:** Reduce LR every few epochs
- **Exponential Decay:** Gradually decrease LR
- **Cosine Annealing:** Smoothly decrease LR following a cosine curve
- **ReduceLROnPlateau:** Reduce LR when validation performance stops improving
- **Warmup:** Start with a small LR, then increase it before normal scheduling

#### Simple intuition

```txt
Early training:   LR = 0.001 → move quickly
Later training:   LR = 0.0001 → make fine adjustments
```

### 11.27 Compare cosine decay, step decay, warmup, and one-cycle learning rates.

#### Learning-rate schedules comparison

| Schedule | How it changes LR | Main idea | Best use |
| --- | --- | --- | --- |
| **Cosine Decay** | Smoothly decreases LR following a cosine curve | Large → gradually smaller | Deep learning, fine-tuning |
| **Step Decay** | Drops LR by a fixed factor at specific epochs | Sudden LR reductions | Simple CNN training |
| **Warmup** | Starts with small LR, gradually increases to target LR | Avoid unstable early updates | Transformers, large models |
| **One-Cycle** | LR increases → reaches maximum → decreases to very small value | Explore quickly, then fine-tune | Fast training, CNNs/transformers |

#### 1. Cosine Decay

LR decreases smoothly:

$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min}) \left(1+\cos\frac{\pi t}{T}\right)$

**Shape:** `High → smooth decrease → very low`

Good because there are no abrupt LR changes.

---

#### 2. Step Decay

Keep LR constant, then suddenly reduce it:

$\eta_t = \eta_0 \gamma^k$

where $k$ increases at predefined epochs.

Example:

```txt
Epoch:  0 ---- 30 ---- 60 ---- 90
LR:    .1       .01     .001    .0001
```

Simple and effective, but requires choosing decay points.

---

#### 3. Warmup

Start with a small LR and gradually increase:

```txt
LR
│       ______
│      /
│     /
│____/
└──────────────
  warmup
```

Commonly used with **Transformers**, because a large LR at the beginning can cause unstable training.

Often combined with cosine decay:

```txt
Warmup → Cosine Decay
small → peak → gradually smaller
```

---

#### 4. One-Cycle

LR first **increases**, then **decreases**:

```txt
LR
│       /\
│      /  \
│     /    \
│____/      \____
└────────────────
```

Usually:

$\text{Low LR} \rightarrow \text{High LR} \rightarrow \text{Very Low LR}$

The increasing phase helps exploration, while the decreasing phase helps convergence.

#### Interview summary

- **Cosine:** smooth decay
- **Step:** sudden drops
- **Warmup:** gradually increase LR at the beginning
- **One-Cycle:** increase first, then decrease

### 11.28 Why can a very high learning rate prevent convergence?

A **very high learning rate** can prevent convergence because each parameter update becomes too large.

Gradient descent:

$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$

where η\\eta is the learning rate.

#### What happens with a high LR?

```txt
Loss
│       /\      /\
│      /  \    /  \
│  ___/    \__/    \__
│
└──────────────────────
       parameter
```

Instead of moving toward the minimum, the optimizer can **overshoot it repeatedly**.

#### Main problems

1. **Overshooting**<br>Updates jump across the minimum.
2. **Oscillation**<br>Parameters bounce between opposite sides of the minimum.
3. **Divergence**<br>Updates become increasingly large and loss can explode.
4. **Numerical instability**<br>Very large updates can produce `NaN` or `Inf`.

#### Simple intuition

- **LR too small:** converges slowly
- **Good LR:** converges efficiently
- **LR too large:** overshoots or diverges

### 11.29 What happens when the learning rate is extremely small?

When the **learning rate is extremely small**, parameter updates become tiny:

$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$

#### What happens?

1. **Very slow convergence**<br>The model makes tiny steps toward the minimum.
2. **Training takes much longer**<br>It may require many more iterations.
3. **Can appear stuck**<br>Loss may decrease so slowly that training looks like it has stopped.
4. **May stop before reaching a good solution**<br>With a fixed training budget, the model may not have enough time to reach a good region.

#### Intuition

```txt
Too small:  ──→──→──→──→──→  very slow
Good:       ─────→────→────→  efficient
Too large:  ←────→────←────→  unstable
```

### 11.30 What is the difference between a local minimum, saddle point, and global minimum?

#### Local Minimum vs Saddle Point vs Global Minimum

| Point | Meaning | Key property |
| --- | --- | --- |
| **Global minimum** | Lowest loss anywhere | Best possible loss |
| **Local minimum** | Lowest loss in its nearby region | May not be globally best |
| **Saddle point** | Minimum in one direction, maximum in another | Gradient can be zero but not a minimum |

#### 1. Global Minimum

$L(\theta^*) \leq L(\theta), \quad \forall \theta$

It has the **lowest possible loss across the entire parameter space**.

```txt
Loss
│\        /
│ \______/
│    ↑
│ Global minimum
└──────────────
```

#### 2. Local Minimum

A point that is lower than its **neighbors**, but another region may have an even lower loss.

```txt
Loss
│\    /\       /\
│ \__/  \_____/  \__
│   ↑
│ Local minimum
└──────────────────
```

#### 3. Saddle Point

A point that behaves like a minimum in one direction and a maximum in another.

For example:

$f(x,y)=x^2-y^2$

At (0,0):

- Along x: f increases → looks like a minimum
- Along y: f decreases → looks like a maximum

So it is a **saddle point**.

#### Interview shortcut

> **Global minimum:** best everywhere.
>
> **Local minimum:** best nearby.
>
> **Saddle point:** down in one direction, up in another.

### 11.31 Why are saddle points particularly relevant in high-dimensional neural networks?

Saddle points are important in neural networks because the parameter space has **millions or billions of dimensions**.

#### Why?

At a saddle point:

$\nabla L(\theta) \approx 0$

so gradient-based optimizers can make very small updates.

But unlike a local minimum, the curvature differs across directions:

$\lambda_i > 0 \quad \text{in some directions}$

$\lambda_j < 0 \quad \text{in other directions}$

where $\lambda$ are eigenvalues of the Hessian.

#### Why high dimensions make this important

In high-dimensional spaces:

- There are **many possible directions** to move.
- A point can be flat in some directions and descending in others.
- Saddle points become much more common than strict local minima.
- Gradients can become very small near saddle regions, slowing optimization.

#### Simple intuition

Imagine a mountain pass:

```txt
        \     /
         \   /
          \_/
          / \
         /   \
```

You are at the center:

- Moving one direction takes you **down**
- Moving another direction takes you **up**

That's a saddle point.

### 11.32 What is sharpness of a loss landscape?

#### Sharpness of a Loss Landscape

**Sharpness** describes **how quickly the loss changes when we slightly change the model parameters**.

#### Intuition

- **Sharp minimum:** Small parameter changes → large increase in loss.
- **Flat minimum:** Small parameter changes → small increase in loss.

```txt
Sharp                    Flat

Loss                      Loss
│   /\                    │      __
│  /  \                   │    _/  \_
│_/    \_                 │___/      \___
└───────                  └──────────────
```

Mathematically, sharpness is related to the **curvature** of the loss, often measured using the Hessian:

$H = \nabla^2 L(\theta)$

A large maximum eigenvalue:

$\lambda_{\max}(H)$

generally indicates **sharper curvature**.

#### Why does it matter?

A commonly discussed observation is:

> **Flatter solutions often show better robustness to small parameter perturbations and can correlate with better generalization.**

However, **sharpness is parameterization-dependent**, so it should not be interpreted as a universal measure of generalization by itself.

### 11.33 Why can flatter minima sometimes generalize better?

#### Why can flatter minima generalize better?

A **flat minimum** means small changes in the model parameters cause only a small increase in loss.

```txt
Sharp minimum          Flat minimum

Loss                    Loss
│   /\                  │      ______
│  /  \                 │    _/      \_
│_/    \_               │___/          \___
└───────                └──────────────────
```

#### Why might this help?

1. **Less sensitive to parameter changes**<br>Small perturbations do not significantly hurt performance.
2. **More robust solutions**<br>The model is not relying on extremely precise parameter values.
3. **May capture broader patterns**<br>Instead of fitting very specific training details, the solution can represent patterns that remain useful on unseen data.

#### Intuition

Think of fitting training data:

- **Sharp minimum:** "I need these exact parameter values."
- **Flat minimum:** "A whole neighborhood of parameter values works well."

The second behavior can be more robust to unseen data.

#### Important caveat

The statement **"flatter = better generalization" is not universally true**. Sharpness depends on parameterization and scaling of the model, so modern research treats it as a useful optimization/generalization concept rather than a standalone guarantee.

### 11.34 What is the relationship between batch size and generalization?

#### Batch size and generalization

**Batch size affects the noise in gradient updates, which can influence how well the model generalizes to unseen data.**

- **Small batch size** → noisier gradients → often acts like a form of regularization → can improve generalization.
- **Large batch size** → smoother, more accurate gradient estimates → faster hardware utilization, but can sometimes lead to poorer generalization.

#### Why?

For batch size B, the gradient is approximately:

$g_B = \frac{1}{B}\sum_{i=1}^{B}\nabla L_i$

As B increases, the variance of the gradient estimate generally decreases:

$\mathrm{Var}(g_B) \propto \frac{1}{B}$

So:

| Batch size | Gradient noise | Typical effect |
| --- | --- | --- |
| Small | High | More exploration, potentially better generalization |
| Large | Low | More stable updates, potentially worse generalization |

**Key intuition:**

Small batches introduce useful randomness into optimization, which can help the model avoid settling into solutions that fit the training data too specifically.

However, **batch size does not universally determine generalization**. Learning rate, optimizer, training duration, weight decay, architecture, and the learning-rate schedule also matter.

### 11.35 Why does overparameterization not necessarily cause worse generalization?

> ✍️ _Drafted answer (not from the original notes)._

Classical theory (VC dimension, bias-variance) says more parameters than data points should overfit. Modern deep nets with far more parameters than examples still generalize well, because **parameter count is a poor measure of effective capacity**:

- **Implicit regularization:** among the infinitely many solutions that fit the training data, gradient descent tends to find "simple" ones (low norm, low complexity, flat minima).
- **Interpolation can be benign:** in high dimensions, a model can fit noisy labels with tiny, spread-out components that barely affect predictions on new inputs ("benign overfitting").
- **Double descent:** test error rises near the interpolation threshold (#params ≈ #samples) and then *falls again* as the model grows, because larger models can pick smoother interpolating solutions.
- **Architectural inductive biases:** convolutions, attention, weight sharing and normalization constrain the hypothesis space to functions that suit the data.
- **Explicit regularization and training tricks:** weight decay, data augmentation, dropout, early stopping.
- **Optimization benefits:** overparameterized loss landscapes have more connected, flatter minima, which makes good solutions easier to reach.

**Caveat:** overparameterization does not *guarantee* generalization — with little data, label noise and no regularization, big models still overfit. Practically: scale model and data together, keep validation monitoring, and rely on weight decay and augmentation.

### 11.36 Explain the double-descent phenomenon.

> ✍️ _Drafted answer (not from the original notes)._

**Double descent** is the observation that test error does not follow the classic U-curve as model capacity grows:

1. **Classical regime:** as capacity grows, test error first falls and then rises (the usual bias-variance U).
2. **Interpolation threshold:** around the point where the model can just barely fit the training set exactly (#params ≈ #samples), test error **peaks** — the model is forced into a single, highly contorted solution that fits the noise.
3. **Modern (overparameterized) regime:** past the threshold, test error **decreases again**, often below the classical optimum, because many interpolating solutions exist and the optimizer picks a smooth, low-norm one.

```txt
test error
   |\        /\
   | \      /  \
   |  \____/    \________
   +----------------------- model size
        classical  ^  modern
              interpolation threshold
```

**Variants**
- **Model-wise:** varying parameter count (above).
- **Epoch-wise:** test error can rise and fall again as training continues.
- **Sample-wise:** adding data can temporarily *hurt* a model near its threshold.

**Why it matters:** the peak is strongest with label noise and little regularization; proper weight decay or early stopping can flatten it. Practically, it justifies "go bigger and regularize" rather than shrinking models to avoid overfitting, and warns that small changes in data size can produce surprising regressions near the threshold.

### 11.37 What is the implicit bias of gradient descent?

> ✍️ _Drafted answer (not from the original notes)._

When many parameter settings achieve zero (or equal) training loss, **gradient descent does not pick one at random** — its dynamics favor particular solutions. That preference, which comes without any explicit regularizer, is its *implicit bias*.

**Known results**
- **Linear regression (underdetermined), started from zero:** GD converges to the **minimum-L2-norm** solution that interpolates the data.
- **Logistic regression on separable data:** the weights diverge in norm, but their direction converges to the **max-margin (hard-margin SVM)** solution.
- **Matrix factorization / deep linear nets:** GD tends toward **low-rank** solutions (a nuclear-norm-like bias).
- **Deep nets:** SGD noise and step size bias toward **flatter minima**; there is also a "simplicity bias" toward learning low-frequency, simple functions first.

**What controls it**
- Initialization scale (small init → rich, feature-learning regime; large init → kernel/"lazy" regime).
- Learning rate and batch size (larger LR/smaller batch → flatter minima).
- Optimizer choice (Adam has a different implicit bias than SGD; it sometimes generalizes worse on vision tasks).
- Architecture and parameterization.

**Why it matters:** it is a key explanation for why overparameterized networks generalize — the training algorithm itself acts as a regularizer. Practically, hyperparameters like LR, batch size, warmup and init are not just speed knobs; they change *which* solution you end up with.

### 11.38 Why can SGD find solutions that generalize despite having enormous parameter counts?

> ✍️ _Drafted answer (not from the original notes)._

Huge networks can memorize random labels (Zhang et al., 2017), yet on real data SGD finds solutions that generalize. The main explanations:

- **Implicit regularization of SGD:** gradient noise from mini-batches acts like a regularizer, steering the optimizer away from sharp minima toward **flat minima**, which are more robust to small parameter and data perturbations.
- **Simplicity bias:** networks trained with SGD learn simple, low-frequency patterns first and fit noise later — which is why early stopping works.
- **Structure in real data:** natural data lies on low-dimensional manifolds with shared patterns; fitting the signal is "easier" than memorizing noise, so SGD reaches it first.
- **Effective capacity is far below parameter count:** weight sharing, normalization and architectural priors (convolution, attention) restrict the functions that are actually reachable.
- **Benign overparameterization:** wide nets have many global minima, many of them low-norm and smooth; SGD from small random init lands near those.
- **Explicit help:** weight decay, augmentation, dropout and learning-rate schedules reinforce the bias.

**Evidence and practice:** the ratio of LR to batch size correlates with generalization; very large batches can generalize worse unless you scale LR and add warmup. Tools such as SAM (Sharpness-Aware Minimization) explicitly seek flat minima.

**Caveat:** these are partial explanations; theory still does not fully account for deep-learning generalization.

---

← [Chapter 10. Data Preprocessing and Feature Engineering](10-data-preprocessing-and-feature-engineering.md) · [Table of contents](../README.md) · [Chapter 12. Model Evaluation and Experimentation](12-model-evaluation-and-experimentation.md) →
