# Chapter 11. Optimization and Training Dynamics

_Part III — Machine Learning_

Gradient descent and its variants, convergence and conditioning, second-order and newer optimisers, learning-rate and batch-size choices, weight decay and weight averaging, loss landscapes, generalization of over-parameterized models, hyperparameter tuning and reading training curves.

This chapter contains **41 questions**.

## Contents

- **11.1** [What is gradient descent? How does it work?](#111-what-is-gradient-descent-how-does-it-work)
- **11.2** [What are vanishing and exploding gradients, and what causes them?](#112-what-are-vanishing-and-exploding-gradients-and-what-causes-them)
- **11.3** [What is a learning rate? How to choose a good one?](#113-what-is-a-learning-rate-how-to-choose-a-good-one)
- **11.4** [How do you approach hyperparameter tuning?](#114-how-do-you-approach-hyperparameter-tuning)
- **11.5** [Explain Grid Search vs Random Search vs Bayesian Optimization.](#115-explain-grid-search-vs-random-search-vs-bayesian-optimization)
- **11.6** [Explain TPE hyperparameter optimization.](#116-explain-tpe-hyperparameter-optimization)
- **11.7** [Explain Bayesian Optimization.](#117-explain-bayesian-optimization)
- **11.8** [Explain Adam Optimizer.](#118-explain-adam-optimizer)
- **11.9** [Explain the RMSprop Optimizer.](#119-explain-the-rmsprop-optimizer)
- **11.10** [What is Adagrad Optimizer?](#1110-what-is-adagrad-optimizer)
- **11.11** [What strategies would you use to optimize the training time for a deep learning model?](#1111-what-strategies-would-you-use-to-optimize-the-training-time-for-a-deep-learning-model)
- **11.12** [What are the differences between stochastic gradient descent (SGD) and gradient descent (GD)?](#1112-what-are-the-differences-between-stochastic-gradient-descent-sgd-and-gradient-descent-gd)
- **11.13** [Explain the difference between the Loss functions and Cost functions.](#1113-explain-the-difference-between-the-loss-functions-and-cost-functions)
- **11.14** [Difference between convex and non-convex cost functions; what does it mean when a cost function is non-convex?](#1114-difference-between-convex-and-non-convex-cost-functions-what-does-it-mean-when-a-cost-function-is-non-convex)
- **11.15** [Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?](#1115-which-gradient-descent-algorithm-among-those-we-discussed-will-reach-the-vicinity-of-the-optimal-solution-the-fastest-which-will-actually-converge)
- **11.16** [Why does SGD sometimes generalize better than full-batch gradient descent?](#1116-why-does-sgd-sometimes-generalize-better-than-full-batch-gradient-descent)
- **11.17** [Compare SGD, Momentum, Adam, AdamW, and Adagrad.](#1117-compare-sgd-momentum-adam-adamw-and-adagrad)
- **11.18** [Why is AdamW generally preferred over Adam with L2 regularization?](#1118-why-is-adamw-generally-preferred-over-adam-with-l2-regularization)
- **11.19** [How does gradient clipping solve exploding gradients?](#1119-how-does-gradient-clipping-solve-exploding-gradients)
- **11.20** [What is a learning-rate scheduler, and why is it important?](#1120-what-is-a-learning-rate-scheduler-and-why-is-it-important)
- **11.21** [Compare cosine decay, step decay, warmup, and one-cycle learning rates.](#1121-compare-cosine-decay-step-decay-warmup-and-one-cycle-learning-rates)
- **11.22** [Why can a very high learning rate prevent convergence?](#1122-why-can-a-very-high-learning-rate-prevent-convergence)
- **11.23** [What happens when the learning rate is extremely small?](#1123-what-happens-when-the-learning-rate-is-extremely-small)
- **11.24** [What is the difference between a local minimum, saddle point, and global minimum?](#1124-what-is-the-difference-between-a-local-minimum-saddle-point-and-global-minimum)
- **11.25** [Why are saddle points particularly relevant in high-dimensional neural networks?](#1125-why-are-saddle-points-particularly-relevant-in-high-dimensional-neural-networks)
- **11.26** [What is sharpness of a loss landscape?](#1126-what-is-sharpness-of-a-loss-landscape)
- **11.27** [Why can flatter minima sometimes generalize better?](#1127-why-can-flatter-minima-sometimes-generalize-better)
- **11.28** [What is the relationship between batch size and generalization?](#1128-what-is-the-relationship-between-batch-size-and-generalization)
- **11.29** [Why does overparameterization not necessarily cause worse generalization?](#1129-why-does-overparameterization-not-necessarily-cause-worse-generalization)
- **11.30** [Explain the double-descent phenomenon.](#1130-explain-the-double-descent-phenomenon)
- **11.31** [What is the implicit bias of gradient descent?](#1131-what-is-the-implicit-bias-of-gradient-descent)
- **11.32** [Why can SGD find solutions that generalize despite having enormous parameter counts?](#1132-why-can-sgd-find-solutions-that-generalize-despite-having-enormous-parameter-counts)
- **11.33** [What determines how fast gradient descent converges, and what is the condition number?](#1133-what-determines-how-fast-gradient-descent-converges-and-what-is-the-condition-number)
- **11.34** [How should batch size and learning rate be chosen together?](#1134-how-should-batch-size-and-learning-rate-be-chosen-together)
- **11.35** [What are second-order optimisation methods, and why are they rarely used for deep learning?](#1135-what-are-second-order-optimisation-methods-and-why-are-they-rarely-used-for-deep-learning)
- **11.36** [What newer optimisers have appeared beyond AdamW, and how should you evaluate claims about them?](#1136-what-newer-optimisers-have-appeared-beyond-adamw-and-how-should-you-evaluate-claims-about-them)
- **11.37** [How should weight decay be applied in practice?](#1137-how-should-weight-decay-be-applied-in-practice)
- **11.38** [What are exponential moving averages of weights, stochastic weight averaging and model soups?](#1138-what-are-exponential-moving-averages-of-weights-stochastic-weight-averaging-and-model-soups)
- **11.39** [How does gradient accumulation work, and what are its pitfalls?](#1139-how-does-gradient-accumulation-work-and-what-are-its-pitfalls)
- **11.40** [How do you optimise a model when the metric you care about is not differentiable?](#1140-how-do-you-optimise-a-model-when-the-metric-you-care-about-is-not-differentiable)
- **11.41** [How do you read training curves to diagnose problems in model training?](#1141-how-do-you-read-training-curves-to-diagnose-problems-in-model-training)

---

### 11.1 What is gradient descent? How does it work?

#### Gradient Descent

**Gradient descent** is an optimization algorithm used to find model parameters that minimize a loss function.

Imagine the loss function as a **hill/valley**. We want to move downhill until we reach a minimum.

#### How it works

1. **Initialize parameters**: $w = \text{random values}$
2. **Calculate the loss**<br>Measure how wrong the model's predictions are.
3. **Calculate the gradient**
   The gradient tells us **which direction increases the loss most quickly**.
4. **Move in the opposite direction**

   ```math
   w := w - \eta \frac{\partial L}{\partial w}
   ```

   where:

   - $w$ = model parameter
   - $L$ = loss
   - $\eta$ = learning rate
   - $\frac{\partial L}{\partial w}$ = gradient
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

### 11.2 What are vanishing and exploding gradients, and what causes them?

#### Vanishing vs Exploding Gradients

They happen mainly because gradients are **multiplied repeatedly through many layers/time steps** during backpropagation.

For a deep network:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial W_n} \prod_{i=2}^{n} \frac{\partial h_i}{\partial h_{i-1}}
$$

#### 1. Vanishing gradients

If the derivatives are mostly **less than 1**:

$$
0.5^{20} \approx 9.5 \times 10^{-7}
$$

The gradient becomes extremely small.

**Causes:**

- Deep networks
- Sigmoid/tanh saturation
- Poor initialization
- Repeated multiplication by small derivatives

**Effect:** Early layers learn very slowly or stop learning.

#### 2. Exploding gradients

If the derivatives are mostly **greater than 1**:

$$
2^{20} = 1{,}048{,}576
$$

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

### 11.3 What is a learning rate? How to choose a good one?

**Learning Rate**

- Hyperparameter controlling step size of parameter updates during gradient descent
- Update rule: $\theta = \theta - \alpha \nabla J(\theta)$, where $\alpha$ = learning rate

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
- **Interaction with batch size:** larger batches usually tolerate, and need, a larger learning rate (see the batch size question in this chapter)
- **Schedules:** a high learning rate early gives fast progress; a lower one later gives stable convergence

### 11.4 How do you approach hyperparameter tuning?

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

**Process:**

1. Define hyperparameter search space (ranges, distributions)
2. Choose evaluation metric aligned with business goal
3. Use cross-validation to evaluate each combination (avoid overfitting to single split)
4. Select best-performing combination on validation set
5. Final evaluation on held-out test set

### 11.5 Explain Grid Search vs Random Search vs Bayesian Optimization.

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

### 11.6 Explain TPE hyperparameter optimization.

**TPE (Tree-structured Parzen Estimator)**

- A Bayesian optimization algorithm for hyperparameter tuning (used by default in Optuna, Hyperopt)
- Sequential Model-Based Optimization (SMBO) approach — builds probabilistic models to guide search

**How it works**

1. Evaluate a few random hyperparameter configurations initially, record results (loss values)
2. Split observed trials into two groups based on a performance threshold (quantile $\gamma$, e.g., top 15%):
   - $l(x)$: density of hyperparameters that gave **good** results
   - $g(x)$: density of hyperparameters that gave **poor** results
3. Model $l(x)$ and $g(x)$ separately (instead of one joint model like Gaussian Process)
4. Select next hyperparameters that **maximize the ratio $l(x)/g(x)$** (Expected Improvement proxy) — favors regions likely to yield good results and unlikely to yield bad ones
5. Evaluate new point, add to observations, repeat

**Why it's used**

- Handles **conditional/hierarchical hyperparameters** well (e.g., different params depending on model type) — advantage over Gaussian Process-based BO
- More scalable to high-dimensional, mixed-type (categorical + continuous) search spaces
- Efficient — needs fewer trials than grid/random search to find good hyperparameters

**vs Gaussian Process-based BO**

- GP models $p(y \mid x)$ directly; TPE models $p(x \mid y)$ via $l(x)$ and $g(x)$ — makes TPE better suited for tree-structured/conditional search spaces

### 11.7 Explain Bayesian Optimization.

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

### 11.8 Explain Adam Optimizer.

**Adam Optimizer (Adaptive Moment Estimation)**

- Adaptive learning rate optimization algorithm combining **Momentum** + **RMSProp**
- Most widely used optimizer in deep learning

**How it works**

- Maintains two moving averages per parameter:
  - **First moment ($m$)**: exponential moving average of gradients (like Momentum) → $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
  - **Second moment ($v$)**: exponential moving average of squared gradients (like RMSProp) → $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- **Bias correction** (since $m$, $v$ initialized at 0, biased toward 0 early on):
  - $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$
- **Update rule**:

  ```math
  \theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  ```

**Hyperparameters**

- $\alpha$ (learning rate) — typically 0.001
- $\beta_1$ — decay rate for first moment, typically 0.9
- $\beta_2$ — decay rate for second moment, typically 0.999
- $\epsilon$ — small constant to prevent division by zero (e.g., 1e-8)

**Advantages**

- Adaptive per-parameter learning rates — works well with sparse gradients
- Combines benefits of Momentum (fast convergence) + RMSProp (handles varying gradient scales)
- Requires little tuning, works well out-of-the-box

**Limitations**

- Can generalize worse than SGD with momentum in some cases (sharper minima)
- Extra memory overhead (stores $m$ and $v$ per parameter)

### 11.9 Explain the RMSprop Optimizer.

**RMSprop (Root Mean Square Propagation)**

- Adaptive learning rate optimizer that adjusts learning rate per parameter based on recent gradient magnitudes
- Designed to fix Adagrad's problem of learning rate shrinking too aggressively over time

**How it works**

- Maintains exponential moving average of **squared gradients**:
  - $v_t = \beta v_{t-1} + (1-\beta) g_t^2$
- **Update rule**:

  ```math
  \theta_{t+1} = \theta_t - \alpha \frac{g_t}{\sqrt{v_t} + \epsilon}
  ```

  - $\alpha$ = learning rate, $\beta$ = decay rate (typically 0.9), $\epsilon$ = small constant (avoid division by zero)

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

### 11.10 What is Adagrad Optimizer?

**Adagrad (Adaptive Gradient Algorithm)**

- Adaptive learning rate optimizer that scales the learning rate per parameter based on the **cumulative sum of past squared gradients**

**How it works**

- Accumulates squared gradients: $G_t = G_{t-1} + g_t^2$
- **Update rule**:

  ```math
  \theta_{t+1} = \theta_t - \alpha \frac{g_t}{\sqrt{G_t} + \epsilon}
  ```

  - $\alpha$ = initial learning rate, $\epsilon$ = small constant (to avoid division by zero)

**Key idea**

- Parameters with large historical gradients → smaller effective learning rate
- Parameters with small/infrequent gradients → larger effective learning rate
- Good for sparse features (e.g., NLP, where rare words get larger updates)

**Advantages**

- No manual learning rate tuning per parameter needed
- Well-suited for sparse data (sparse gradients get bigger updates)

**Limitations**

- **Learning rate shrinks monotonically** — since $G_t$ only grows, learning rate can become too small, causing training to stall prematurely
- This limitation led to the development of **RMSprop** and **Adam**, which use moving averages instead of cumulative sums

### 11.11 What strategies would you use to optimize the training time for a deep learning model?

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

### 11.12 What are the differences between stochastic gradient descent (SGD) and gradient descent (GD)?

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

### 11.13 Explain the difference between the Loss functions and Cost functions.

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

### 11.14 Difference between convex and non-convex cost functions; what does it mean when a cost function is non-convex?

**Convex cost function:** any line segment between two points on its graph lies on or above the graph — formally $f(\lambda a + (1-\lambda) b) \le \lambda f(a) + (1-\lambda) f(b)$. It has a single basin, so **every local minimum is the global minimum** and gradient descent with a suitable learning rate converges to it. Examples: MSE for linear regression, log loss for logistic regression, the hinge loss for linear SVMs (the regularized versions too).

**Non-convex cost function:** the surface has multiple valleys, plateaus and **saddle points**. Implications:
- Gradient descent can get stuck in a **local minimum** or slow down on plateaus/saddles; the result depends on **initialization** and hyperparameters.
- No guarantee of reaching the global optimum, and different runs give different models.
- Examples: neural-network losses (because of hidden layers and non-linear activations), k-means objective, matrix factorization, GMM likelihood.

**Why we still succeed with non-convex problems**
- In high-dimensional neural nets, most critical points are saddles rather than bad local minima, and many local minima have similar, near-optimal loss.
- Practical tools: SGD noise, momentum/Adam, good initialization (He/Xavier), learning-rate schedules, multiple random restarts (k-means++), and batch normalization/residual connections that smooth the landscape.

**Interview takeaway:** convex = optimization is "solved" and reproducible; non-convex = optimization becomes part of the modeling problem, so you report variance across seeds and tune the optimizer.

### 11.15 Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?

(This refers to the classic comparison of **Batch GD, Stochastic GD and Mini-batch GD**, as in *Hands-On ML*.)

**Reaches the vicinity of the optimum fastest: Stochastic GD**, closely followed by **Mini-batch GD**. SGD updates after *every* example, so it makes $n$ updates per epoch versus one for batch GD. On large datasets it gets close to the minimum after a fraction of an epoch. Mini-batch GD is similar per update and in wall-clock time is often the fastest in practice, thanks to vectorized GPU computation.

**Actually converges: Batch GD.** With a suitable fixed learning rate on a convex problem (e.g. linear regression's MSE), it follows the true gradient and settles exactly at the minimum. SGD and mini-batch GD use noisy gradient estimates, so with a constant learning rate they **keep bouncing around** the minimum. Mini-batch bounces less than SGD.

**However:** SGD and mini-batch GD *will* converge if you use a **learning-rate schedule** that decays over time (e.g. $\eta_t \propto 1/t$, satisfying $\sum \eta_t = \infty$, $\sum \eta_t^2 \lt \infty$), or progressively increase the batch size.

**Tradeoffs:**
| | Batch GD | Mini-batch GD | SGD |
| --- | --- | --- | --- |
| Cost per step | Full dataset | Batch (32–4096) | 1 sample |
| Path | Smooth | Mildly noisy | Very noisy |
| Large data | Slow / memory-bound | Best in practice | Fast, but poor hardware utilization |
| Non-convex | Can stall at saddles | Noise helps escape | Noise helps escape |

In deep learning, mini-batch with a schedule (warmup plus cosine decay) and an optimizer such as Adam or AdamW is the standard.

### 11.16 Why does SGD sometimes generalize better than full-batch gradient descent?

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

### 11.17 Compare SGD, Momentum, Adam, AdamW, and Adagrad.

#### SGD vs Momentum vs Adam vs AdamW vs Adagrad

| Optimizer | Main idea | Learning rate | Strength | Weakness |
| --- | --- | --- | --- | --- |
| **SGD** | Follow gradient directly | Fixed/global | Simple, often good generalization | Can be slow |
| **Momentum** | Accumulate past gradients | Global | Faster, smoother convergence | Needs tuning |
| **Adagrad** | Adapt LR per parameter | Adaptive | Good for sparse features | LR can become too small |
| **Adam** | Momentum + adaptive LR | Adaptive | Fast, robust, easy to tune | Can generalize differently from SGD |
| **AdamW** | Adam + **decoupled weight decay** | Adaptive | Strong default for many deep-learning models | Still needs tuning |

#### 1. SGD

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

Uses only the current gradient.

**Intuition:** Take a step directly downhill.

---

#### 2. Momentum

Keeps a moving average of previous gradients:

$$
v_t = \beta v_{t-1} + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_t
$$

**Intuition:** Like a ball gaining velocity while moving downhill.

It reduces oscillations and can accelerate movement in consistent directions.

---

#### 3. Adagrad

Adjusts the learning rate separately for each parameter based on accumulated squared gradients:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t
$$

Frequently updated parameters get smaller learning rates.

**Good for:** sparse features, such as NLP/count-based features.

**Problem:** The accumulated gradient keeps growing, so learning rates can eventually become extremely small.

---

#### 4. Adam

Combines **Momentum + adaptive learning rates**.

It maintains:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

Then updates using both estimates.

**Intuition:**

> Momentum tells you **which direction to move**, while the adaptive term controls **how large the step should be for each parameter**.

---

#### 5. AdamW

AdamW is essentially **Adam with decoupled weight decay**.

Instead of mixing L2 regularization directly into Adam's gradient, weight decay is applied separately:

$$
\theta \leftarrow \theta - \eta \cdot \text{AdamUpdate} - \eta \lambda \theta
$$

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

### 11.18 Why is AdamW generally preferred over Adam with L2 regularization?

#### Why AdamW over Adam + L2?

The key difference is **how weight decay is applied**.

With **Adam + L2**, the regularization term is added to the gradient:

$$
g_t' = g_t + \lambda\theta_t
$$

Adam then **rescales this combined gradient** using its adaptive moments.

With **AdamW**, weight decay is applied **separately**:

$$
\theta_{t+1} = \theta_t - \eta \cdot \text{AdamUpdate} - \eta \lambda \theta_t
$$

#### Why is this better?

Because in Adam, the adaptive scaling means the effective strength of L2 regularization can vary across parameters.

AdamW **decouples weight decay from the adaptive gradient update**, making $\lambda$ behave more directly as a weight-decay hyperparameter.

### 11.19 How does gradient clipping solve exploding gradients?

**Gradient clipping limits the size of the gradient before updating the model parameters.**

Suppose the gradient becomes huge:

$$
\lVert g \rVert = 1000
$$

Instead of applying the full gradient, we set a maximum norm, say $5$:

$$
g_{\text{clipped}} = g \cdot \frac{5}{\lVert g \rVert}
$$

So:

$$
\lVert g_{\text{clipped}} \rVert = 5
$$

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

$$
\theta_{t+1} = \theta_t - \eta g
$$

A huge $g$ causes a huge parameter update.

With clipping:

$$
\theta_{t+1} = \theta_t - \eta g_{\text{clipped}}
$$

the update remains bounded.

### 11.20 What is a learning-rate scheduler, and why is it important?

A **learning-rate scheduler** automatically changes the learning rate during training instead of keeping it fixed.

$$
\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)
$$

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

### 11.21 Compare cosine decay, step decay, warmup, and one-cycle learning rates.

#### Learning-rate schedules comparison

| Schedule | How it changes LR | Main idea | Best use |
| --- | --- | --- | --- |
| **Cosine Decay** | Smoothly decreases LR following a cosine curve | Large → gradually smaller | Deep learning, fine-tuning |
| **Step Decay** | Drops LR by a fixed factor at specific epochs | Sudden LR reductions | Simple CNN training |
| **Warmup** | Starts with small LR, gradually increases to target LR | Avoid unstable early updates | Transformers, large models |
| **One-Cycle** | LR increases → reaches maximum → decreases to very small value | Explore quickly, then fine-tune | Fast training, CNNs/transformers |

#### 1. Cosine Decay

LR decreases smoothly:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min}) \left(1+\cos\frac{\pi t}{T}\right)
$$

**Shape:** `High → smooth decrease → very low`

Good because there are no abrupt LR changes.

---

#### 2. Step Decay

Keep LR constant, then suddenly reduce it:

$$
\eta_t = \eta_0 \gamma^k
$$

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

$$
\text{Low LR} \rightarrow \text{High LR} \rightarrow \text{Very Low LR}
$$

The increasing phase helps exploration, while the decreasing phase helps convergence.

#### Interview summary

- **Cosine:** smooth decay
- **Step:** sudden drops
- **Warmup:** gradually increase LR at the beginning
- **One-Cycle:** increase first, then decrease

### 11.22 Why can a very high learning rate prevent convergence?

A **very high learning rate** can prevent convergence because each parameter update becomes too large.

Gradient descent:

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

where $\eta$ is the learning rate.

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

### 11.23 What happens when the learning rate is extremely small?

When the **learning rate is extremely small**, parameter updates become tiny:

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

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

### 11.24 What is the difference between a local minimum, saddle point, and global minimum?

#### Local Minimum vs Saddle Point vs Global Minimum

| Point | Meaning | Key property |
| --- | --- | --- |
| **Global minimum** | Lowest loss anywhere | Best possible loss |
| **Local minimum** | Lowest loss in its nearby region | May not be globally best |
| **Saddle point** | Minimum in one direction, maximum in another | Gradient can be zero but not a minimum |

#### 1. Global Minimum

$$
L(\theta^{\ast}) \leq L(\theta), \quad \forall \theta
$$

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

$$
f(x, y) = x^2 - y^2
$$

At $(0, 0)$:

- Along $x$: $f$ increases → looks like a minimum
- Along $y$: $f$ decreases → looks like a maximum

So it is a **saddle point**.

#### Interview shortcut

> **Global minimum:** best everywhere.
>
> **Local minimum:** best nearby.
>
> **Saddle point:** down in one direction, up in another.

### 11.25 Why are saddle points particularly relevant in high-dimensional neural networks?

Saddle points are important in neural networks because the parameter space has **millions or billions of dimensions**.

#### Why?

At a saddle point:

$$
\nabla L(\theta) \approx 0
$$

so gradient-based optimizers can make very small updates.

But unlike a local minimum, the curvature differs across directions:

$$
\lambda_i > 0 \quad \text{in some directions}
$$

$$
\lambda_j < 0 \quad \text{in other directions}
$$

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

### 11.26 What is sharpness of a loss landscape?

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

$$
H = \nabla^2 L(\theta)
$$

A large maximum eigenvalue:

$$
\lambda_{\max}(H)
$$

generally indicates **sharper curvature**.

#### Why does it matter?

A commonly discussed observation is:

> **Flatter solutions often show better robustness to small parameter perturbations and can correlate with better generalization.**

However, **sharpness is parameterization-dependent**, so it should not be interpreted as a universal measure of generalization by itself.

### 11.27 Why can flatter minima sometimes generalize better?

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

### 11.28 What is the relationship between batch size and generalization?

#### Batch size and generalization

**Batch size affects the noise in gradient updates, which can influence how well the model generalizes to unseen data.**

- **Small batch size** → noisier gradients → often acts like a form of regularization → can improve generalization.
- **Large batch size** → smoother, more accurate gradient estimates → faster hardware utilization, but can sometimes lead to poorer generalization.

#### Why?

For batch size $B$, the gradient is approximately:

$$
g_B = \frac{1}{B}\sum_{i=1}^{B}\nabla L_i
$$

As $B$ increases, the variance of the gradient estimate generally decreases:

$$
\mathrm{Var}(g_B) \propto \frac{1}{B}
$$

So:

| Batch size | Gradient noise | Typical effect |
| --- | --- | --- |
| Small | High | More exploration, potentially better generalization |
| Large | Low | More stable updates, potentially worse generalization |

**Key intuition:**

Small batches introduce useful randomness into optimization, which can help the model avoid settling into solutions that fit the training data too specifically.

However, **batch size does not universally determine generalization**. Learning rate, optimizer, training duration, weight decay, architecture, and the learning-rate schedule also matter.

### 11.29 Why does overparameterization not necessarily cause worse generalization?

Classical theory (VC dimension, bias-variance) says more parameters than data points should overfit. Modern deep nets with far more parameters than examples still generalize well, because **parameter count is a poor measure of effective capacity**:

- **Implicit regularization:** among the infinitely many solutions that fit the training data, gradient descent tends to find "simple" ones (low norm, low complexity, flat minima).
- **Interpolation can be benign:** in high dimensions, a model can fit noisy labels with tiny, spread-out components that barely affect predictions on new inputs ("benign overfitting").
- **Double descent:** test error rises near the interpolation threshold ($\text{number of parameters} \approx \text{number of samples}$) and then *falls again* as the model grows, because larger models can pick smoother interpolating solutions.
- **Architectural inductive biases:** convolutions, attention, weight sharing and normalization constrain the hypothesis space to functions that suit the data.
- **Explicit regularization and training tricks:** weight decay, data augmentation, dropout, early stopping.
- **Optimization benefits:** overparameterized loss landscapes have more connected, flatter minima, which makes good solutions easier to reach.

**Caveat:** overparameterization does not *guarantee* generalization — with little data, label noise and no regularization, big models still overfit. Practically: scale model and data together, keep validation monitoring, and rely on weight decay and augmentation.

### 11.30 Explain the double-descent phenomenon.

**Double descent** is the observation that test error does not follow the classic U-curve as model capacity grows:

1. **Classical regime:** as capacity grows, test error first falls and then rises (the usual bias-variance U).
2. **Interpolation threshold:** around the point where the model can just barely fit the training set exactly ($\text{number of parameters} \approx \text{number of samples}$), test error **peaks** — the model is forced into a single, highly contorted solution that fits the noise.
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

### 11.31 What is the implicit bias of gradient descent?

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

### 11.32 Why can SGD find solutions that generalize despite having enormous parameter counts?

Huge networks can memorize random labels (Zhang et al., 2017), yet on real data SGD finds solutions that generalize. The main explanations:

- **Implicit regularization of SGD:** gradient noise from mini-batches acts like a regularizer, steering the optimizer away from sharp minima toward **flat minima**, which are more robust to small parameter and data perturbations.
- **Simplicity bias:** networks trained with SGD learn simple, low-frequency patterns first and fit noise later — which is why early stopping works.
- **Structure in real data:** natural data lies on low-dimensional manifolds with shared patterns; fitting the signal is "easier" than memorizing noise, so SGD reaches it first.
- **Effective capacity is far below parameter count:** weight sharing, normalization and architectural priors (convolution, attention) restrict the functions that are actually reachable.
- **Benign overparameterization:** wide nets have many global minima, many of them low-norm and smooth; SGD from small random init lands near those.
- **Explicit help:** weight decay, augmentation, dropout and learning-rate schedules reinforce the bias.

**Evidence and practice:** the ratio of LR to batch size correlates with generalization; very large batches can generalize worse unless you scale LR and add warmup. Tools such as SAM (Sharpness-Aware Minimization) explicitly seek flat minima.

**Caveat:** these are partial explanations; theory still does not fully account for deep-learning generalization.

### 11.33 What determines how fast gradient descent converges, and what is the condition number?

Convergence speed depends on the **geometry of the loss function**: how curved it is, and how different the curvature is in different directions. Classical results for convex functions give the vocabulary interviewers expect.

**Key definitions**

- **$L$-smooth:** the gradient does not change too fast; the largest curvature (largest Hessian eigenvalue) is at most $L$.
- **$\mu$-strongly convex:** the function curves upwards at least as much as a quadratic with curvature $\mu$ (smallest Hessian eigenvalue at least $\mu \gt 0$).
- **Condition number:** $\kappa = L / \mu$, the ratio of the steepest to the flattest curvature.

**Convergence rates of gradient descent (step size $1/L$)**

| Setting | Rate | Meaning |
| --- | --- | --- |
| Convex, $L$-smooth | $f(x_k) - f^{\ast} \le \frac{L \lVert x_0 - x^{\ast} \rVert^2}{2k}$, so $O(1/k)$ | Sublinear: 10 times more steps for 10 times less error |
| Strongly convex | $O\left((1 - 1/\kappa)^k\right)$ | Linear (geometric) convergence; about $\kappa$ steps per constant-factor improvement |
| With Nesterov momentum | $O(1/k^2)$ convex; $O\left((1 - 1/\sqrt{\kappa})^k\right)$ strongly convex | Momentum replaces $\kappa$ with $\sqrt{\kappa}$, a large gain when $\kappa$ is big |
| Stochastic gradient descent | $O(1/\sqrt{k})$ convex; $O(1/k)$ strongly convex with decaying step size | Gradient noise slows final convergence, which is why learning rates are decayed |

**Why the condition number matters**

- With a large $\kappa$, the loss surface is a long, narrow valley. A step size small enough to be stable in the steep direction ($\eta \lt 2/L$) makes progress painfully slow along the flat direction, and the path zig-zags across the valley.
- This is exactly what happens with **unscaled features** in linear models: a feature measured in thousands creates a steep direction, another in fractions a flat one.

**Remedies, all of which reduce the effective condition number**

- **Feature scaling and normalisation** (standardisation; batch and layer normalisation in networks).
- **Momentum** (heavy ball, Nesterov), which accelerates along consistent directions and damps oscillation.
- **Preconditioning:** adaptive optimisers (Adam, RMSprop, Adagrad) rescale each coordinate by an estimate of its gradient scale, a cheap diagonal approximation of second-order information; full second-order methods do this exactly (see the second-order methods question).

Deep networks are non-convex, so these rates do not apply directly, but the intuition (curvature, conditioning, noise) explains most practical behaviour, including why warmup, normalisation and adaptive methods help.

### 11.34 How should batch size and learning rate be chosen together?

Batch size and learning rate are coupled: changing one usually requires changing the other.

**Why they interact**

- A mini-batch gradient is a noisy estimate of the true gradient; larger batches reduce the noise (variance falls roughly in proportion to $1/B$).
- With less noise, the optimiser can take larger steps safely. With a small batch, large steps amplify noise and destabilise training.

**Scaling rules**

- **Linear scaling rule (SGD):** when multiplying the batch size by $k$, multiply the learning rate by $k$. Goyal et al. (2017) trained ResNet-50 on ImageNet with batches up to 8,192 this way, using a **warmup** period because the large learning rate is unstable at the start.
- **Square-root scaling** is a common heuristic for Adam-style optimisers: multiply the learning rate by $\sqrt{k}$.
- Both rules break down beyond a certain batch size.

**The critical batch size**

- Below a critical batch size, doubling the batch roughly halves the number of steps needed (near-perfect scaling). Above it, larger batches give diminishing returns: more compute per step, almost no reduction in steps.
- The critical batch size can be estimated from the **gradient noise scale** (the ratio of gradient variance to squared gradient magnitude, McCandlish et al., 2018). It tends to **grow during training** as the loss falls, which is why some large-scale training runs increase batch size over time.

**Very large batches**

- Layer-wise adaptive methods (LARS for SGD, LAMB for Adam) scale each layer's update by the ratio of weight norm to update norm, which made very large batch training of ResNets and BERT practical.

**Generalisation**

- Very large batches without retuning often generalise worse (see 11.28); careful learning rate scaling, warmup and longer schedules close most of the gap.

**Practical procedure**

1. Pick the largest batch that fits in memory and gives good hardware utilisation (use gradient accumulation if you need more).
2. Tune the learning rate at that batch size with a short sweep on a log scale.
3. When changing batch size later, rescale the learning rate as a starting point, keep or extend warmup, and re-verify with a short sweep.

### 11.35 What are second-order optimisation methods, and why are they rarely used for deep learning?

First-order methods use only the gradient. **Second-order methods** also use curvature (the Hessian matrix of second derivatives) to choose both the direction and the size of each step.

**Newton's method**

$\theta_{t+1} = \theta_t - H^{-1} \nabla L(\theta_t)$

- Rescales the step by the inverse curvature in every direction, so it is unaffected by the condition number and converges **quadratically** near a minimum of a well-behaved convex function.
- **Cost:** for $n$ parameters, the Hessian has $n^2$ entries and inverting it costs $O(n^3)$. For a model with a billion parameters this is impossible.
- On non-convex losses, Newton's method can move towards saddle points or maxima, because negative curvature flips the step direction.

**Practical approximations**

| Method | Idea | Where it is used |
| --- | --- | --- |
| Quasi-Newton (BFGS, L-BFGS) | Build an approximation of the inverse Hessian from recent gradient differences; L-BFGS stores only a few vectors | Full-batch, smooth problems: logistic regression (scikit-learn's default `lbfgs` solver), small models, scientific computing |
| Gauss-Newton and Levenberg-Marquardt | Approximate the Hessian for least-squares problems | Curve fitting, small neural networks |
| Natural gradient | Precondition with the Fisher information matrix, so steps are measured in distribution space | Theory behind K-FAC, policy-gradient methods in RL (TRPO) |
| K-FAC | Approximate the Fisher matrix of each layer by a Kronecker product of two small matrices | Research and some large-scale training |
| Shampoo, SOAP | Kronecker-factored preconditioners per weight matrix | Competitive results in optimiser benchmarks such as AlgoPerf; used in some large training runs |
| Diagonal approximations (Adam, Adagrad, RMSprop) | Per-parameter scaling from gradient statistics | The default in deep learning |

**Why first-order methods dominate deep learning**

- Memory and compute: even approximate curvature is expensive at billions of parameters.
- **Stochastic gradients:** curvature estimates from small noisy mini-batches are unreliable, and quasi-Newton methods assume consistent gradients.
- Non-convexity makes exact curvature steps risky.
- Adam with good normalisation, initialisation and learning rate schedules already works well, and the engineering ecosystem is built around it.

That said, structured preconditioners (Shampoo and its successors) are an active area, and matrix-aware optimisers are increasingly used in large-scale training.

### 11.36 What newer optimisers have appeared beyond AdamW, and how should you evaluate claims about them?

AdamW has been the default for training transformers for years, but several alternatives target lower memory, faster training or better stability.

| Optimiser | Key idea | Notes |
| --- | --- | --- |
| Adafactor | Factorises Adam's second-moment matrix into row and column statistics | Large memory savings; used to train T5 |
| Lion | Discovered by automated program search; the update is the **sign** of an interpolation of momentum and gradient; keeps only momentum | Less memory than Adam; needs a smaller learning rate and larger weight decay |
| Sophia | Uses a cheap estimate of the diagonal Hessian as a preconditioner, with clipping | Reported faster LLM pretraining in its paper; results vary in independent tests |
| Shampoo and SOAP | Kronecker-factored preconditioning per weight matrix | Strong results in optimiser benchmarks; more compute and engineering |
| Muon | For 2D weight matrices, orthogonalises the momentum update (using a few Newton-Schulz iterations) so all directions are updated at a similar scale | Popular in fast training "speedrun" results and used, with stability modifications, in some large open models; embeddings, normalisation and output layers usually still use AdamW |
| Schedule-free optimisers | Replace the learning rate schedule with a form of iterate averaging | Removes the need to choose the training length in advance |

**How to evaluate claims**

- **Tune the baseline properly.** Many reported gains disappear when AdamW receives the same tuning budget (learning rate, warmup, weight decay, betas).
- **Compare at equal wall-clock time and memory**, not just equal steps; a method that needs fewer steps but more compute per step may not save time.
- **Check scale and duration:** an advantage on small models or short runs may not hold at larger scale or longer training.
- **Look at independent benchmarks** (for example AlgoPerf, which measures time to reach a target across several workloads) and replications.
- **Consider stability and hyperparameter sensitivity:** an optimiser that wins only with narrow settings is costly in practice.

**Practical advice:** use AdamW with a well-tuned schedule as the default; try an alternative when memory is the bottleneck (Adafactor, Lion) or when training cost is large enough that a few percent matters, and validate on your own workload.

### 11.37 How should weight decay be applied in practice?

Weight decay shrinks weights towards zero each step. It looks simple, but several practical details strongly affect results.

**Decoupled weight decay (AdamW)**

- In SGD, L2 regularisation and weight decay are equivalent. In Adam they are not: an L2 penalty added to the loss gets divided by Adam's adaptive scaling, so parameters with large gradients are barely regularised (see 11.18).
- AdamW applies decay directly to the weights: $\theta \leftarrow \theta - \eta \lambda \theta$, separately from the gradient step. Use AdamW, not Adam with an L2 term.

**Which parameters to decay**

- **Decay:** weight matrices of linear and convolution layers.
- **Usually exclude:** biases, normalisation gains and offsets (LayerNorm, BatchNorm, RMSNorm), and often embeddings. Decaying normalisation gains pushes them towards zero, which works against their purpose.

```python
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim < 2 or "norm" in name or "bias" in name else decay).append(p)
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4, betas=(0.9, 0.95))
```

**Interaction with normalisation**

- When a layer is followed by batch or layer normalisation, its output is unchanged if the weights are scaled. Weight decay then does not regularise the function directly; it keeps the weight norm small, which **increases the effective learning rate** (the effective step size grows as the weight norm shrinks). Weight decay and learning rate therefore interact strongly in normalised networks.

**Typical values**

- AdamW for transformers: 0.01 to 0.1. Fine-tuning: often smaller (0 to 0.01), because the goal is to stay close to the pretrained weights.
- In AdamW, the effective decay per step is $\eta \lambda$, so changing the learning rate or schedule also changes the regularisation strength. Retune weight decay when the learning rate changes significantly.

### 11.38 What are exponential moving averages of weights, stochastic weight averaging and model soups?

Several techniques improve a model by **averaging weights** rather than choosing a single final checkpoint. Averaging cancels some of the noise of stochastic training and tends to land in flatter, better-generalising regions of the loss surface.

**Exponential moving average (EMA) of weights**

- Maintain a shadow copy: $\theta_{\text{EMA}} \leftarrow \beta \thinspace \theta_{\text{EMA}} + (1 - \beta) \thinspace \theta$ after each step, with $\beta$ typically 0.999 to 0.9999.
- Evaluate and deploy the EMA weights, while training continues on the raw weights.
- Widely used in diffusion models, object detectors and self-supervised learning (the "teacher" network in methods such as BYOL and DINO is an EMA of the student).
- Costs one extra copy of the parameters.

**Stochastic weight averaging (SWA)**

- After the main training phase, continue training with a constant or cyclical learning rate and average the weights collected at regular intervals.
- Often improves generalisation at almost no cost. For networks with BatchNorm, **recompute the normalisation statistics** for the averaged weights with a pass over the training data, since the stored statistics belong to individual checkpoints.

**Model soups**

- Average the weights of **several models fine-tuned from the same pretrained model** with different hyperparameters.
- A "greedy soup" adds models one at a time and keeps each only if validation accuracy improves.
- Gives much of the benefit of an ensemble with the inference cost of a single model.

**Why averaging works:** models trained from the same starting point usually lie in the same basin of the loss surface, so the average of their weights is also a good solution, typically nearer the centre of the basin. Averaging models trained from **different random initialisations** does not work in general, because their weights are not aligned.

**Related:** checkpoint averaging of the last few saved checkpoints is common in machine translation and language model training, and weight averaging is the basis of model merging methods for LLMs (see Chapter 17).

### 11.39 How does gradient accumulation work, and what are its pitfalls?

**Gradient accumulation** simulates a large batch on limited memory. Instead of one optimiser step per micro-batch, gradients from $k$ micro-batches are added up and a single step is taken, giving an effective batch size of $k \times$ micro-batch size.

```python
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = model(**batch).loss / accum_steps   # scale so the sum equals the mean
    loss.backward()
    if (i + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()                        # schedule counts optimiser steps
        optimizer.zero_grad()
```

**Pitfalls**

- **Loss scaling:** divide each micro-batch loss by the number of accumulation steps, or gradients are $k$ times too large.
- **Variable-length sequences:** if each micro-batch loss is averaged over its own tokens, accumulating those averages weights short and long micro-batches equally, which differs from a true large batch where every token counts equally. The correct approach is to sum token losses and divide by the total number of tokens across all accumulated micro-batches. This subtle bug affected popular training libraries until it was fixed.
- **BatchNorm:** statistics are computed per micro-batch, not over the effective batch, so accumulation does not reproduce large-batch BatchNorm behaviour. Layer normalisation does not have this problem.
- **Schedulers and logging:** step the learning rate scheduler and count warmup in optimiser steps, not micro-batches.
- **Gradient clipping:** clip once, after accumulation, on the total gradient.
- **Distributed training:** in data-parallel training, skip the gradient synchronisation on intermediate micro-batches (for example with `no_sync()` in PyTorch DDP) and synchronise only before the optimiser step, to avoid wasted communication.

**Trade-off:** accumulation saves memory but not time; $k$ micro-batches take about $k$ times as long as one. It is the right tool when the desired batch size does not fit, not a way to speed up training.

### 11.40 How do you optimise a model when the metric you care about is not differentiable?

Business and evaluation metrics (accuracy, F1, precision at a threshold, NDCG, BLEU, intersection over union, profit) are usually **not differentiable**, so gradient descent cannot optimise them directly. Several strategies bridge the gap.

**1. Optimise a surrogate loss, then tune the decision**

- Train with a smooth loss that is well aligned with the metric (cross-entropy for accuracy and AUC, squared error for RMSE, pinball loss for quantiles).
- Then optimise the **decision layer** separately: choose the classification threshold that maximises F1 or expected profit on validation data, or apply cost-sensitive decision rules. This is often the simplest and most effective approach.

**2. Weight the loss by business cost**

- Class weights or per-example weights that reflect the cost of each error type (a missed fraud case costs more than a false alarm).

**3. Smooth approximations of the metric**

- Soft F1 (computed from probabilities instead of hard predictions), soft Dice and Lovász losses for segmentation IoU, differentiable sorting and ranking approximations.

**4. Metric-aware gradients for ranking**

- LambdaRank and LambdaMART scale pairwise gradients by how much swapping two items would change NDCG, effectively optimising a non-differentiable ranking metric.

**5. Reinforcement learning**

- Treat the model as a policy and the metric as a reward. REINFORCE-style methods optimise sequence metrics such as BLEU or task success (self-critical sequence training for captioning; RL with verifiable rewards for LLM reasoning). Higher variance and more complex to train.

**6. Black-box optimisation for a few parameters**

- When only a handful of parameters matter (thresholds, blending weights, business rules), use grid search, Bayesian optimisation or evolutionary methods directly on the metric.

**Caution:** optimising directly for a narrow metric can exploit its weaknesses (Goodhart's law). Keep guardrail metrics, and check that improvements on the proxy translate into real outcomes through online tests.

### 11.41 How do you read training curves to diagnose problems in model training?

Training curves (loss and metrics over steps, for training and validation) are the main diagnostic tool. Reading them well saves days of guesswork. NaN and divergence debugging is covered in Chapter 13; this question is about the broader shapes.

| Pattern | Likely cause | What to try |
| --- | --- | --- |
| Training loss flat from the start | Learning rate far too low; bug (frozen parameters, wrong labels, loss not connected) | Overfit a tiny batch first; check gradients are non-zero; increase learning rate |
| Training loss decreases very slowly, smoothly | Learning rate too low, poor conditioning (unscaled inputs), model too small | Learning rate range test; normalise inputs; larger model |
| Training loss noisy, oscillating, not improving | Learning rate too high; batch too small | Lower learning rate, add warmup, increase batch size |
| Sudden spikes then recovery | Bad batches, learning rate too high for current curvature, optimiser instability | Gradient clipping, lower peak learning rate, inspect data in the spike batch |
| Training and validation both high and close | Underfitting | More capacity, more features, train longer, less regularisation |
| Training loss keeps falling, validation rises | Overfitting | More data or augmentation, regularisation, early stopping |
| Validation better than training | Regularisation active only in training (dropout, augmentation), or validation easier or leaking | Usually fine; check for leakage if large |
| Validation loss rises while validation accuracy still improves | Model becoming over-confident on some examples | Watch calibration; choose checkpoints by the metric that matters |
| Plateau then sudden drop | Learning rate schedule step; the model "discovering" a solution (common in transformers and grokking-style tasks) | Be patient; compare with schedule events |

**Signals beyond the loss**

- **Gradient norm** over time: steadily growing norms or spikes warn of instability before the loss diverges.
- **Update-to-weight ratio** (size of each update relative to the weight, roughly $10^{-3}$ is a common rule of thumb): much larger suggests instability, much smaller suggests the layer is barely learning.
- **Activation and weight statistics per layer:** dead ReLUs, saturated activations, exploding norms.
- **Learning rate** plotted alongside the loss, to link changes to schedule events.

**Good habits:** log at fine granularity, compare runs on the same chart, fix random seeds to separate real effects from noise, and run several seeds before trusting small differences.

---

← [Chapter 10. Data Preprocessing and Feature Engineering](10-data-preprocessing-and-feature-engineering.md) · [Table of contents](../README.md) · [Chapter 12. Model Evaluation and Experimentation](12-model-evaluation-and-experimentation.md) →
