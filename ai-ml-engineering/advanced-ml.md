# advanced ML interview question

Advanced ML, deep learning, computer vision, Transformer/LLM, RAG, MLOps and ML system-design interview questions.

### Q1. Explain the bias-variance tradeoff mathematically and practically.
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

### Q2. Why can a model with lower training loss have worse test performance?
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

### Q3. How do you detect overfitting beyond simply comparing train and validation accuracy?
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

### Q4. What is data leakage? Give examples that are difficult to detect.
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

### Q5. How would you handle severe class imbalance?
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

### Q6. When would you prefer F1, ROC-AUC, PR-AUC, log loss, or calibration error?
| Metric | Prefer it when | Main idea |
| --- | --- | --- |
| **F1** | You need a **binary decision** and care about both precision and recall | `2PR/(P+R)` |
| **ROC-AUC** | You care about **ranking quality** across thresholds, especially with reasonably balanced classes | Probability that a positive ranks above a negative |
| **PR-AUC** | The positive class is **rare/imbalanced** and detecting positives matters | Measures precision-recall tradeoff |
| **Log Loss** | You care about **probability quality**, not just correct labels | Penalizes confident wrong predictions heavily |
| **Calibration Error** | You need **reliable probabilities** for decisions | Checks whether predicted probabilities match actual frequencies |

### Q7. What is the difference between model capacity and model complexity?
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

### Q8. Why does regularization reduce variance?
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

### Q9. Explain L1 vs L2 regularization geometrically.
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

### Q10. Why does feature scaling matter for some algorithms but not others?
Feature scaling matters when an algorithm is sensitive to the **magnitude or distance of features**.

#### Scaling matters for

**1. Distance-based algorithms**

Examples:

- KNN
- K-Means
- DBSCAN

Suppose:

$\text{Age}=25,\quad \text{Income}=100000$

Income dominates the Euclidean distance simply because its numerical scale is much larger.

Scaling puts features on comparable ranges.

---

**2. Gradient-based algorithms**

Examples:

- Linear/Logistic Regression
- Neural Networks
- SVM

If one feature is much larger than another, the loss surface can become poorly shaped:

```txt
Unscaled → elongated contour → slow/unstable optimization
Scaled   → more circular contour → faster optimization
```

Scaling generally makes gradient descent converge more efficiently.

---

**3. Regularized models**

For L1/L2:

$L = Loss + \lambda\sum_i |w_i|$

or

$L = Loss + \lambda\sum_i w_i^2$

Without scaling, a feature's numerical unit can affect the size of its weight and therefore how strongly regularization affects it.

---

#### Scaling usually does NOT matter for

**Tree-based algorithms:**

- Decision Trees
- Random Forest
- XGBoost
- LightGBM

A tree typically asks questions like:

$x < 50000?$

If you transform the feature:

$x' = x/1000$

the equivalent split is:

$x' < 50$

The ordering of values hasn't changed, so the tree can make essentially the same splits.

### Q11. What is the difference between generative and discriminative models?
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

### Q12. How does cross-validation fail when dealing with time-series data?
#### How Cross-Validation Fails with Time-Series Data

Standard **K-fold cross-validation** assumes that observations are independent and that the order of the data does not matter.

Time-series data violates this assumption because **the future can depend on the past**.

#### The problem

Suppose we have:

```txt
2021 → 2022 → 2023 → 2024 → 2025
```

With random K-fold CV, you might get:

```txt
Training: 2021, 2022, 2024, 2025
Validation: 2023
```

The model is being trained using **2024 and 2025 to predict 2023**.

That's **future information leakage**.

#### Correct approach: Walk-forward validation

Instead, preserve chronological order:

```txt
Fold 1:
Train:      2021
Validate:          2022

Fold 2:
Train:      2021 → 2022
Validate:                  2023

Fold 3:
Train:      2021 → 2023
Validate:                          2024

Fold 4:
Train:      2021 → 2024
Validate:                                  2025
```

This better simulates the real-world situation:

> **Train on the past → predict the future.**

### Q13. What is stratified cross-validation, and when should you not use it?
#### Stratified Cross-Validation

**Stratified cross-validation** keeps approximately the **same class distribution in every fold** as in the original dataset.

Example with 20% positive and 80% negative:

```txt
Dataset: 20% positive / 80% negative

Fold 1: 20% / 80%
Fold 2: 20% / 80%
Fold 3: 20% / 80%
...
```

This is especially useful for **classification with imbalanced classes**, because every fold gets enough minority-class examples.

#### When NOT to use it

**1. Time-series data**

Don't randomly stratify time-series observations:

```txt
2021 → 2022 → 2023 → 2024 → 2025
```

You need chronological/walk-forward validation instead.

**2. Regression**

Standard stratified CV is designed for **categorical labels**. For continuous targets, use regular K-fold or specialized approaches such as binning the target when appropriate.

**3. Grouped data**

If multiple samples belong to the same person, patient, customer, device, etc., random stratification can put the same entity into both train and validation:

```txt
Customer A → Train
Customer A → Validation
```

Use **GroupKFold** or **StratifiedGroupKFold** when both grouping and class balance matter.

### Q14. How would you detect distribution shift between training and production data?
#### Detecting Distribution Shift

**Distribution shift** occurs when production data differs from training data.

I would check:

- **Feature distributions:** Mean, variance, quantiles, missing values, category frequencies.
- **Statistical tests:** KS test, PSI, Wasserstein distance, Jensen-Shannon divergence.
- **Prediction distribution:** Compare model outputs between training and production.
- **Performance:** Monitor precision, recall, F1, PR-AUC when production labels become available.
- **Train-vs-production classifier:** If a classifier can easily distinguish them, there is likely a shift.

**Key types:**

$\text{Covariate shift: } P(X) \text{ changes}$

$\text{Label shift: } P(Y) \text{ changes}$

$Concept drift: P(Y∣X) changes$

### Q15. What is the difference between covariate shift, label shift, and concept drift?
#### Covariate Shift vs Label Shift vs Concept Drift

| Type | What changes? | Example |
| --- | --- | --- |
| **Covariate shift** | $P(X)$ changes | Customer demographics change, but the relationship to fraud stays the same |
| **Label shift** | $P(Y)$ changes | Fraud rate increases from 1% → 5%, but $P(X\mid Y)$ stays similar |
| **Concept drift** | $P(Y\mid X)$ changes | The same customer behavior that previously indicated fraud no longer does |

### Q16. Explain how gradient descent actually works.
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

### Q17. Why does SGD sometimes generalize better than full-batch gradient descent?
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

### Q18. Compare SGD, Momentum, Adam, AdamW, and Adagrad.
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

### Q19. Why is AdamW generally preferred over Adam with L2 regularization?
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

### Q20. What causes vanishing and exploding gradients?
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

### Q21. How does gradient clipping solve exploding gradients?
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

### Q22. What is a learning-rate scheduler, and why is it important?
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

### Q23. Compare cosine decay, step decay, warmup, and one-cycle learning rates.
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

### Q24. Why can a very high learning rate prevent convergence?
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

### Q25. What happens when the learning rate is extremely small?
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

### Q26. What is the difference between a local minimum, saddle point, and global minimum?
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

### Q27. Why are saddle points particularly relevant in high-dimensional neural networks?
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

### Q28. What is sharpness of a loss landscape?
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

### Q29. Why can flatter minima sometimes generalize better?
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

### Q30. What is the relationship between batch size and generalization?
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

### Q31. Explain backpropagation from first principles.
**Backpropagation is an efficient way to calculate how much each neural-network parameter contributed to the final error, so we can update those parameters to reduce the error.**

The core idea is simply the **chain rule of calculus**.

#### 1. Forward pass

Suppose a simple network is:

$x \rightarrow z \rightarrow \hat{y} \rightarrow L$

For one neuron:

$z = wx+b$

$\hat{y}=f(z)$

$L=L(\hat{y},y)$

The forward pass calculates the prediction and then the loss.

---

#### 2. What do we actually need?

To update $w$, gradient descent needs:

$\frac{\partial L}{\partial w}$

This answers:

> "If I slightly change $w$, how much does the loss change?"

But $w$ doesn't directly produce the loss. It affects $z$, which affects $\hat y$, which affects $L$.

So we use the chain rule:

$\boxed{ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat y} \frac{\partial \hat y}{\partial z} \frac{\partial z}{\partial w} }$

This is the essence of backpropagation.

---

#### 3. Example

Consider:

$z=wx+b$

$\hat y=z$

and squared-error loss:

$L=\frac{1}{2}(\hat y-y)^2$

We calculate derivatives backward.

First:

$\frac{\partial L}{\partial \hat y}=\hat y-y$

Then:

$\frac{\partial \hat y}{\partial z}=1$

And:

$\frac{\partial z}{\partial w}=x$

Therefore:

$\frac{\partial L}{\partial w} $ = $(\hat y-y)(1)(x)$

$\boxed{ \frac{\partial L}{\partial w}=(\hat y-y)x }$

Similarly:

$\boxed{ \frac{\partial L}{\partial b}=\hat y-y }$

---

#### 4. Then gradient descent updates the parameters

Once backpropagation gives us the gradients:

$w \leftarrow w-\eta\frac{\partial L}{\partial w}$

$b \leftarrow b-\eta\frac{\partial L}{\partial b}$

where η\\eta is the learning rate.

---

#### 5. Why "back" propagation?

The network computes values **forward**:

$x \rightarrow z_1 \rightarrow z_2 \rightarrow \hat y \rightarrow L$

But gradients are computed **backward**:

$L \rightarrow \hat y \rightarrow z_2 \rightarrow z_1 \rightarrow w$

Each layer receives a gradient from the layer after it and uses the chain rule to calculate its own gradient.

For a layer:

$z = Wx+b$

$a=f(z)$

the backward pass essentially determines:

$\frac{\partial L}{\partial W}, \quad \frac{\partial L}{\partial b}, \quad \frac{\partial L}{\partial x}$

The last one is important because it allows the gradient to continue propagating toward earlier layers.

---

#### 6. The key intuition

Think of the network as a sequence of functions:

$L=f_3(f_2(f_1(x)))$

Backpropagation asks:

> **How sensitive is the final loss to each intermediate value and parameter?**

The chain rule decomposes this sensitivity:

$\frac{dL}{dx} $ = $\frac{dL}{df_2} \frac{df_2}{df_1} \frac{df_1}{dx}$

So **backpropagation is essentially repeated application of the chain rule, performed efficiently from the output layer back toward the input.**

### Q32. Why do neural networks need nonlinear activation functions?
Neural networks need **nonlinear activation functions** because without them, stacking multiple layers would still produce only a **linear transformation**.

#### Without nonlinear activation

Suppose:

$h = W_1x+b_1$

$y = W_2h+b_2$

Substitute hh:

$y=W_2(W_1x+b_1)+b_2$

$y=(W_2W_1)x+(W_2b_1+b_2)$

This is still just:

$y=Wx+b$

So even with 100 layers, a network containing only linear operations can be mathematically reduced to **one linear layer**.

#### With nonlinear activation

Add an activation such as ReLU:

$h=\text{ReLU}(W_1x+b_1)$

$y=W_2h+b_2$

Now the network can represent **nonlinear functions and complex decision boundaries**.

For example, nonlinear activations allow networks to learn:

- XOR-like relationships
- Curved decision boundaries
- Image features
- Speech patterns
- Complex interactions between features

#### Intuition

Think of each layer as transforming the data.

**Linear only:**

$x \rightarrow \text{line} \rightarrow \text{line} \rightarrow \text{line}$

Still fundamentally a linear mapping.

**With nonlinearities:**

$x \rightarrow \text{transform} \rightarrow \text{bend} \rightarrow \text{transform} \rightarrow \text{bend}$

The network can progressively build complex functions.

#### Common nonlinear activations

$\text{ReLU}(x)=\max(0,x)$

$\text{sigmoid}(x)=\frac{1}{1+e^{-x}}$

$\text{tanh}(x)$

Modern deep networks commonly use **ReLU and its variants** such as GELU, SiLU, and Leaky ReLU.

### Q33. Compare ReLU, Leaky ReLU, GELU, SiLU, and sigmoid.
#### ReLU vs Leaky ReLU vs GELU vs SiLU vs Sigmoid

| Activation | Formula | Output Range | Main Advantage | Main Limitation | Common Use |
| --- | --- | --- | --- | --- | --- |
| **ReLU** | $f(x)=\max(0,x)$ | $[0,\infty)$ | Simple and fast | **Dying ReLU** for negative inputs | CNNs, MLPs |
| **Leaky ReLU** | $f(x)=\max(\alpha x,x)$ | $(-\infty,\infty)$ | Keeps a small gradient for x\<0x\<0 | Requires choosing α\\alpha | CNNs, MLPs |
| **GELU** | $f(x)=x\Phi(x)$ | $(-\infty,\infty)$ | Smooth, probabilistic gating | More computationally expensive than ReLU | **Transformers, BERT** |
| **SiLU / Swish** | $f(x)=x\sigma(x)$ | $(-\infty,\infty)$ | Smooth and non-monotonic | More expensive than ReLU | **Modern deep learning, LLMs** |
| **Sigmoid** | $f(x)=\frac{1}{1+e^{-x}}$ | $(0,1)$ | Interpretable as probability | Vanishing gradients when saturated | Binary classification output |

#### Key differences

**1. ReLU**

- Negative values → 0.
- Positive values → unchanged.
- Very cheap computationally.
- Can cause **dying neurons** because the gradient is 0 for .
  $x<0$

**2. Leaky ReLU**

- Similar to ReLU, but allows a small negative output.
- Usually  $\alpha$ is small, such as $0.01$.
- Reduces the dying-ReLU problem.

**3. GELU**

- Smoothly gates inputs instead of using a hard cutoff.
- Widely used in **Transformer architectures**, including BERT-style models.
- Approximation often used: $\operatorname{GELU}(x)\approx \frac{x}{2}\left(1+\tanh\left[ \sqrt{\frac{2}{\pi}} \left(x+0.044715x^3\right) \right]\right)$

**4. SiLU / Swish**

- Defined as: $\operatorname{SiLU}(x)=x\sigma(x)$
- Smooth and non-monotonic.
- Used extensively in modern neural networks.

**5. Sigmoid**

- Squashes values into $(0,1)$.
- Useful when the output represents a **binary probability**.
- Usually avoided as the main hidden-layer activation in deep networks because of **vanishing gradients**.

#### Interview takeaway

> **ReLU:** fast and simple
>
> **Leaky ReLU:** ReLU with a small negative gradient
>
> **GELU:** smooth gating, common in Transformers
>
> **SiLU:** smooth, non-monotonic activation used in modern architectures
>
> **Sigmoid:** mainly for binary-probability outputs

### Q34. Why can sigmoid cause vanishing gradients?
#### Why can Sigmoid cause vanishing gradients?

The sigmoid function is:

$\sigma(x)=\frac{1}{1+e^{-x}}$

Its derivative is:

$\sigma'(x)=\sigma(x)(1-\sigma(x))$

The **maximum derivative is only $0.25$**, occurring at $x=0$.

For very large positive or negative values, sigmoid **saturates**:

- $x \gg 0$ → $\sigma(x)\approx1$ →$\sigma'(x)\approx0$
- $x \ll 0 $→ $\sigma(x)\approx0$ →$\sigma'(x)\approx0$

During backpropagation, gradients are multiplied through many layers:

$\frac{\partial L}{\partial x} $ = $\frac{\partial L}{\partial y} \prod_i \frac{\partial y_i}{\partial x_i}$

Since sigmoid derivatives are often **much smaller than 1**, repeatedly multiplying them can make the gradient become extremely small.

**Example:**

If each layer contributes a gradient of $0.1$:

$0.1^{10}=10^{-10}$

So early layers receive almost no gradient and learn very slowly.

### Q35. What is Batch Normalization, and why does it help?
#### What is Batch Normalization?

**Batch Normalization (BatchNorm)** normalizes a neural network's activations using the **mean and variance of the current mini-batch**, then applies learnable scale and shift parameters.

For a 2D activation matrix, imagine:

- **Rows** → training examples in the mini-batch
- **Columns** → features

BatchNorm typically normalizes **each feature across the mini-batch**.

##### How Does BatchNorm Work?

For each feature:

**1. Calculate batch statistics**

Compute the mean and variance across the mini-batch:

\[$\mu_B=\frac{1}{m}\sum_{i=1}^{m}x_i$\]

\[$\sigma_B^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$\]

**2. Normalize**

Standardize the activation:

\[$\hat{x}_i$ = $\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$\]

This gives the activation approximately **zero mean and unit variance** within the batch.

**3. Scale and shift**

BatchNorm then applies learnable parameters:

\[$y_i=\gamma\hat{x}_i+\beta$\]

where:

- $(\gamma) $→ learnable **scale**
- $(\beta)$ → learnable **shift**
- $(\epsilon)$ → small constant for numerical stability

The learnable $(\gamma)$ and $(\beta)$ allow the network to choose an appropriate distribution rather than being forced to use exactly zero mean and unit variance.

---

#### Why Does BatchNorm Help?

- **Stabilizes activations:** Keeps intermediate activations at a more controlled scale.
- **Improves gradient flow:** Can make optimization more stable, particularly in deep networks.
- **Enables higher learning rates:** Training can often use larger learning rates without becoming unstable.
- **Speeds up convergence:** Optimization often reaches a good solution faster.
- **Reduces sensitivity to initialization:** Training becomes less dependent on the exact initial scale of the weights.
- **Provides mild regularization:** Batch-to-batch variation in the estimated statistics introduces noise that can have a regularizing effect.

> **Important:** BatchNorm does not simply "prevent activations from exploding or shrinking to zero." Its main effect is to normalize activations and improve the conditioning/stability of optimization.

---

#### Training vs. Inference

During **training**, BatchNorm uses statistics calculated from the current mini-batch.

During **inference**, it normally uses **running estimates of the mean and variance** accumulated during training.

This allows inference to work even when processing a single example.

---

#### Why Do Transformers Usually Use LayerNorm Instead?

Transformers generally use **Layer Normalization (LayerNorm)** rather than BatchNorm because LayerNorm is independent of the mini-batch.

##### Key reasons:

- **Independent of batch size:** LayerNorm works even with a batch size of 1.
- **Works naturally with variable-length sequences:** Normalization does not depend on statistics across different examples in the batch.
- **Suitable for autoregressive generation:** Each token's representation can be normalized without requiring statistics from other sequences or examples.
- **Well suited to sequence models:** It normalizes across the features of each individual token representation rather than across the batch.

##### BatchNorm vs. LayerNorm

|  | **BatchNorm** | **LayerNorm** |
| --- | --- | --- |
| Normalization basis | Across mini-batch | Across features |
| Depends on batch size | Yes | No |
| Training statistics | Current batch | Individual sample |
| Inference | Running statistics | Same normalization principle |
| Commonly used in | CNNs, vision | Transformers, NLP |
| Works well with batch size 1 | Not ideal | Yes |

### Q36. Compare BatchNorm, LayerNorm, GroupNorm, and RMSNorm.
##### BatchNorm vs LayerNorm vs GroupNorm vs RMSNorm

All four normalization methods help control the scale of neural-network activations, but they differ in **which dimensions they normalize over**.

| Feature | **BatchNorm** | **LayerNorm** | **GroupNorm** | **RMSNorm** |
| --- | --- | --- | --- | --- |
| Normalizes over | Batch dimension | Feature dimension | Groups of channels | Feature dimension |
| Depends on batch size | **Yes** | No | No | No |
| Uses mean? | Yes | Yes | Yes | **No** |
| Uses variance? | Yes | Yes | Yes | RMS only |
| Learnable scale | ✓ | ✓ | ✓ | ✓ |
| Learnable shift | ✓ | ✓ | ✓ | Usually **No** |
| Common use | CNNs | Transformers | CNNs / vision | Modern LLMs |
| Main advantage | Efficient training | Batch-independent | Good for small batches | Simpler/faster than LayerNorm |

---

#### 1. Batch Normalization

For each feature/channel, BatchNorm computes statistics across the **mini-batch**:

$\hat{x}= \frac{x-\mu_B} {\sqrt{\sigma_B^2+\epsilon}}$

Then:

$y=\gamma\hat{x}+\beta$

**Key point:** BatchNorm depends on batch statistics.

**Best known for:** CNNs and computer vision.

**Limitation:** Performance can be affected by very small batch sizes.

---

#### 2. Layer Normalization

LayerNorm normalizes the features **within each individual sample**.

$\mu=\frac{1}{d}\sum_{i=1}^{d}x_i$

$\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$

$\operatorname{LayerNorm}(x)$ = $\gamma \frac{x-\mu} {\sqrt{\sigma^2+\epsilon}} +\beta$

**Key point:** It does **not depend on other samples in the batch**.

**Best known for:** Transformers and sequence models.

---

#### 3. Group Normalization

GroupNorm divides channels into **groups** and computes normalization statistics within each group.

For example, with 32 channels, you might divide them into 8 groups of 4 channels.

Conceptually:

$\text{Channels} \rightarrow \text{Groups} \rightarrow \text{Normalize each group}$

**Key point:** It provides a middle ground between BatchNorm and LayerNorm.

It does not depend on the batch size, making it useful when CNNs are trained with **small batches**.

**Best known for:** Computer vision, especially object detection and segmentation.

---

#### 4. RMSNorm

RMSNorm simplifies LayerNorm by removing the **mean-centering** step.

Instead of calculating:

$\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$

RMSNorm uses the root mean square:

$\operatorname{RMS}(x) $ = $\sqrt{ \frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon }$

Then:

$\operatorname{RMSNorm}(x) $ = $\gamma \frac{x} {\operatorname{RMS}(x)}$

So RMSNorm **rescales** the activations but does not subtract their mean.

**Key point:** It is computationally simpler than LayerNorm.

**Best known for:** Modern Transformer/LLM architectures.

---

#### The easiest way to remember them

> **BatchNorm:** Normalize across the **batch**.
>
> **LayerNorm:** Normalize across the **features** of each sample.
>
> **GroupNorm:** Normalize across **groups of channels**.
>
> **RMSNorm:** Normalize by **RMS without subtracting the mean**.

### Q37. Why is LayerNorm preferred over BatchNorm in Transformers?
**LayerNorm is preferred because it normalizes each token independently across its feature dimensions, while BatchNorm depends on statistics computed across the mini-batch.** This makes LayerNorm much better suited to Transformer architectures.

#### Key reasons

1. **Independent of batch size**
   - BatchNorm depends on the other samples in the batch.
   - LayerNorm works even with **batch size = 1**.
   - This is useful for training and inference with varying batch sizes.
2. **Works naturally with variable-length sequences**
   - Transformer inputs can have different sequence lengths.
   - BatchNorm's batch-level statistics can be awkward when sequences are padded or masked.
   - LayerNorm normalizes each token's representation independently.
3. **Suitable for autoregressive generation**
   - During LLM inference, tokens may be generated **one at a time**.
   - LayerNorm does not require statistics from other examples or tokens in the batch.
4. **Consistent between training and inference**
   - BatchNorm uses batch statistics during training but running statistics during inference.
   - LayerNorm uses the same normalization principle during both training and inference.
5. **Works well with Transformer representations**
   - A token is represented as a vector of hidden features: $x \in \mathbb{R}^{d_{\text{model}}}$

- LayerNorm normalizes those $d_{\text{model}}$ features for each token independently.

#### BatchNorm vs LayerNorm

|  | **BatchNorm** | **LayerNorm** |
| --- | --- | --- |
| Normalizes | Across batch | Across features |
| Depends on batch | **Yes** | **No** |
| Batch size = 1 | Problematic | Works |
| Variable sequences | Less natural | Well suited |
| Autoregressive LLM inference | Less suitable | Well suited |
| Training/inference behavior | Different statistics | Same principle |
| Typical use | CNNs | Transformers |

### Q38. What is residual learning, and why are skip connections so effective?
#### What is Residual Learning?

**Residual learning** is the idea of having a neural network learn the **difference (residual)** between the desired output and the input, rather than learning the complete transformation directly.

Instead of learning:

$H(x)$

a residual block learns:

$F(x)=H(x)-x$

and produces:

$y=F(x)+x$

This is the core idea behind **ResNet**.

#### What is a Skip Connection?

A **skip connection** provides a shortcut that carries the input directly to a later layer:

 $y=x+F(x)$

The input $x$ effectively **skips** one or more transformations and is added to their output.

#### Why Are Skip Connections So Effective?

**1. Improve gradient flow**

During backpropagation:

$\frac{\partial y}{\partial x} $ = $ 1+\frac{\partial F(x)}{\partial x}$

The **$1$** provides a direct gradient path, helping gradients propagate through very deep networks.

**2. Reduce the vanishing-gradient problem**

Without a shortcut, gradients must pass through many transformations. With a residual connection, part of the gradient can flow directly through the identity path.

**3. Make deep networks easier to optimize**

Learning:

$H(x)=x+F(x)$

can be easier than learning $H(x)$ directly, especially when the desired transformation is close to the identity.

If the optimal transformation is approximately:

$H(x)\approx x$

the network only needs to learn:

$F(x)\approx0$

rather than reconstructing the identity transformation from scratch.

**4. Preserve information**

The original representation can bypass transformations that might otherwise distort or lose useful information.

**5. Enable very deep architectures**

Skip connections are a fundamental component of architectures such as **ResNet** and **Transformers**.

#### In Transformers

A simplified Transformer sublayer looks like:

$x' = x + F(x)$

where $F(x)$ could be:

- Multi-Head Self-Attention
- Feed-Forward Network (FFN)

Modern Transformers commonly combine residual connections with normalization, for example:

$x′=x+Attention(LayerNorm(x))$

### Q39. Why do deeper networks sometimes perform worse even when optimization should theoretically be easier?
#### Why can deeper networks perform worse?

A deeper network has **greater representational capacity**, so in theory it should be able to represent everything a shallower network can. However, **representational capacity is not the same as optimization being easy**.

This is known as the **degradation problem**.

#### 1. Optimization becomes harder

Adding layers introduces more parameters and nonlinear transformations. Even if a deeper network could theoretically reproduce the shallower network, gradient-based optimization may have difficulty finding those parameters.

For example, suppose the added layers should simply perform an identity mapping:

$H(x)=x$

A plain deep network may struggle to learn this identity transformation.

#### 2. Vanishing or exploding gradients

During backpropagation, gradients are repeatedly multiplied through layers:

$\frac{\partial L}{\partial x} $ = $\prod_i \frac{\partial h_i}{\partial h_{i-1}} \frac{\partial L}{\partial h_n}$

If these derivatives are consistently smaller than 1, gradients can **vanish**.

If they are consistently larger than 1, gradients can **explode**.

Normalization, careful initialization, and appropriate activations help, but very deep plain networks can still be difficult to optimize.

#### 3. Error accumulation

Each additional layer applies another transformation. Small optimization errors can accumulate through many layers, making it harder for the network to preserve useful representations.

#### 4. Overfitting can also occur

A deeper network has more parameters and therefore can have higher capacity. With insufficient data or inadequate regularization, it may fit training data too closely and perform worse on unseen data.

However, **overfitting is not the same as the degradation problem**.

The degradation problem can occur when the **training error itself becomes worse** as depth increases. That indicates an optimization issue rather than simply poor generalization.

#### Why do residual connections help?

A residual block learns:

$y=x+F(x)$

If additional layers are not useful, they can learn:

$F(x)\approx0$

so that:

$y\approx x$

This makes it much easier for a deeper network to preserve the behavior of a shallower network.

The gradient also gets a direct path:

$\frac{\partial y}{\partial x} $ = $1+\frac{\partial F(x)}{\partial x}$

The identity term provides a strong route for gradient propagation.

### Q40. Explain dropout mathematically.
#### Dropout — Mathematical Explanation

**Dropout** is a regularization technique that randomly deactivates a fraction of neurons during training to reduce overfitting.

Suppose a layer produces:

$h = f(Wx+b)$

During training, we generate a random **Bernoulli mask**:

$m_i \sim \text{Bernoulli}(1-p)$

where:

- $p$ = dropout probability
- $1-p $ = probability that a neuron is kept
- $m_i \in \{0,1\}$

The mask is applied to the activations.

#### Inverted Dropout

Modern implementations typically use **inverted dropout**:

$\tilde{h}_i $ = $\frac{m_i}{1-p}h_i$

or in vector form:

$\boxed{ \tilde{h} = \frac{m\odot h}{1-p} }$

where $\odot$ represents element-wise multiplication.

The scaling factor $\frac{1}{1-p}$ keeps the **expected activation approximately unchanged**:

$[\tilde{h}_i] $ = $E\left[\frac{m_i h_i}{1-p}\right]$

Since:

$E[m_i]=1-p$

we get:

$E[\tilde{h}_i] $ = $\frac{(1-p)h_i}{1-p}$ = $h_i$

#### Example

If:

$p=0.5$

then each neuron has a 50% probability of being dropped.

Suppose:

$h=[2,4,6,8]$

and the sampled mask is:

$m=[1,0,1,0]$

Then:

$\tilde{h} $ = $\frac{[1,0,1,0]\odot[2,4,6,8]}{0.5}$

$\tilde{h}=[4,0,12,0]$

So some neurons are temporarily deactivated while the remaining activations are scaled up.

#### Training vs. Inference

**During training:**

$\tilde{h}=\frac{m\odot h}{1-p}$

A new random mask is typically sampled for each training pass.

**During inference:**

$\tilde{h}=h$

No neurons are randomly dropped because the scaling was already handled during training.

#### Why Does Dropout Reduce Overfitting?

Dropout prevents the network from relying too heavily on particular neurons or specific combinations of features.

You can think of training as approximately optimizing many different **thinned subnetworks** that share parameters.

This encourages the network to learn more robust representations.

### Q41. Is dropout still useful in modern large Transformers?
Yes, but **its role has changed in modern large Transformers**.

#### Dropout in modern Transformers

Dropout is still a valid regularization technique, but many **large-scale LLMs use little or no dropout during pretraining**.

| Situation | Dropout usefulness |
| --- | --- |
| Small Transformer | **Often useful** |
| Limited training data | **Often useful** |
| Fine-tuning | **Can be useful** |
| Large-scale LLM pretraining | **Often low or zero** |
| Overfitting concern | **Can help** |
| Very large datasets | Often less necessary |

#### Why is dropout often reduced or removed?

**1. Massive datasets already provide regularization**

Large language models are trained on enormous datasets. The diversity of training examples can reduce overfitting, so aggressive dropout may not be necessary.

**2. It introduces noise during training**

Dropout randomly removes activations:

h\~=m⊙h1−p\\tilde\{h\}=\\frac\{m\\odot h\}\{1-p\}

For very large models, this additional noise can sometimes make optimization less efficient.

**3. Other regularization techniques are important**

Modern LLM training commonly relies on techniques such as:

- Large and diverse datasets
- Weight decay
- Careful initialization
- Learning-rate schedules
- Data filtering
- Early stopping or training-duration control
- Parameter-efficient fine-tuning methods

#### What about fine-tuning?

Dropout can become more useful when fine-tuning a model on a **small dataset**, because overfitting becomes a bigger concern.

However, whether to use it depends on the model, dataset size, training objective, and existing regularization.

#### Important distinction

**Dropout is not what makes Transformers generalize well by itself.**

For large LLMs, the combination of **scale + data + optimization + regularization choices** matters much more than simply adding dropout.

### Q42. What is knowledge distillation?
#### What is Knowledge Distillation?

**Knowledge distillation (KD)** is a model-compression technique where a smaller **student model** learns to reproduce the behavior of a larger, more capable **teacher model**.

Instead of learning only from the hard labels, the student learns from the teacher's **soft predictions**, which contain information about how the teacher distributes probability across different classes.

#### Basic idea

$\text{Teacher model} \rightarrow \text{Soft targets} \rightarrow \text{Student model}$

For example, suppose the correct class is **cat**.

A hard label might be:

$[0,1,0]$

The teacher might produce:

$[0.05,0.90,0.05]$

The student learns from this richer probability distribution.

#### Temperature

Knowledge distillation commonly uses a **temperature $T$** to soften the probability distribution:

$p_i^{(T)} $ = $\frac{\exp(z_i/T)} {\sum_j\exp(z_j/T)}$

where:

- $z_i$ = model logits
- $T$ = temperature
- Higher $T$ → softer probability distribution

At $T=1$, this is the normal softmax.

#### Distillation Loss

A common objective combines the normal supervised loss with the teacher's soft predictions:

$\mathcal{L} $ = $\alpha\mathcal{L}_{CE} + (1-\alpha)\mathcal{L}_{KD}$

where:

$\mathcal{L}_{KD} $ = $T^2 D_{KL} \left( p_{\text{teacher}}^{(T)} \parallel p_{\text{student}}^{(T)} \right)$

So the student learns from both:

1. **Ground-truth labels**
2. **Teacher's behavior**

#### Why is it useful?

- **Model compression:** Smaller models can approximate larger models.
- **Lower inference cost:** Fewer parameters and FLOPs.
- **Lower latency:** Useful for edge/mobile deployment.
- **Knowledge transfer:** The student can learn useful patterns from the teacher's outputs.
- **Deploying large-model capabilities:** A smaller model can inherit some behavior of a larger model.

#### In LLMs

For language models, distillation can involve more than simply matching class probabilities. A student may learn from:

- Teacher-generated responses
- Token-level probability distributions/logits
- Intermediate representations
- Reasoning traces or demonstrations
- Preference or instruction-following behavior

For example:

$\text{Large LLM} \rightarrow \text{generated training data} \rightarrow \text{smaller LLM}$

This is one approach used to transfer capabilities from a larger model to a smaller one.

### Q43. Explain teacher-student training.
#### Teacher–Student Training

**Teacher–student training** is a framework where a **teacher model provides guidance or supervision to a student model**.

The teacher is usually larger, more accurate, or already trained, while the student is smaller or designed to be more efficient.

$\text{Teacher} \rightarrow \text{Knowledge/Targets} \rightarrow \text{Student}$

#### How it works

**1. Train or obtain the teacher**

The teacher is first trained or selected as a strong model.

$f_T(x)\rightarrow z_T$

where $z_T$ represents the teacher's output.

**2. Generate teacher targets**

The teacher processes training examples and produces predictions:

$p_T=\operatorname{softmax}\left(\frac{z_T}{T}\right)$

The temperature $T$ can soften the distribution and expose more information about the teacher's relative preferences.

**3. Train the student**

The student produces:

$p_S=\operatorname{softmax}\left(\frac{z_S}{T}\right)$

The student is trained to match the teacher:

$\mathcal{L}_{KD} $ = $T^2 D_{KL}(p_T\parallel p_S)$

Often, ground-truth labels are also included:

$\mathcal{L} $ = $\alpha\mathcal{L}_{CE} + (1-\alpha)\mathcal{L}_{KD}$

#### Why does this work?

The teacher provides **soft information** that ordinary labels don't contain.

For example, instead of only:

$\text{label}=\text{cat}$

the teacher might indicate:

$P(\text{cat})=0.90,\quad P(\text{dog})=0.08,\quad P(\text{car})=0.02$

The student learns not only the correct answer but also the teacher's **similarity structure and decision behavior**.

#### Teacher–Student in LLMs

For LLMs, teacher–student training can work at several levels:

| Knowledge transferred | Example |
| --- | --- |
| **Logits** | Student matches teacher token probabilities |
| **Generated text** | Student learns from teacher-generated responses |
| **Hidden representations** | Student matches intermediate representations |
| **Reasoning behavior** | Student learns from teacher-generated reasoning examples |
| **Instruction following** | Student trains on teacher-generated instruction/response pairs |

A common LLM workflow is:

$\text{Large Teacher LLM} \rightarrow \text{Generate high-quality examples} \rightarrow \text{Train Student LLM}$

#### Teacher–Student vs Knowledge Distillation

**Teacher–student training** is the broader framework.

**Knowledge distillation** is a specific technique within that framework that explicitly transfers knowledge from teacher to student, commonly through **soft targets, logits, or representations**.

### Q44. What is early stopping actually doing from an optimization perspective?
**Early stopping** stops gradient-based training before the model reaches the minimum training loss, usually when validation performance stops improving.

From an optimization perspective, it acts as a form of **implicit regularization**.

#### 1. Without early stopping

Suppose we optimize:

$\theta_{t+1}$ = $\theta_t-\eta\nabla_\theta L_{\text{train}}(\theta_t)$

As training continues:

$Ltrain↓L_{\text{train}}\downarrow$

But after some point, validation performance may stop improving:

$Lval↑L_{\text{val}}\uparrow$

The model starts fitting increasingly specific patterns in the training data.

#### 2. Early stopping

We monitor validation performance and stop when it no longer improves:

$t^*$ = $\arg\min_t L_{\text{val}}(\theta_t)$

Then we use: $\theta_{t^*}$

instead of the parameters obtained after fully optimizing the training loss.

#### Why is this regularization?

The model has not had unlimited optimization time to fit the training data.

Early stopping effectively **restricts the set of parameter values the optimizer can reach from the initialization**.

So instead of: $\theta_0 \rightarrow \theta_1 \rightarrow \theta_2 \rightarrow \cdots \rightarrow \theta_{\text{train optimum}}$

we select an earlier point: $\theta_0 \rightarrow \cdots \rightarrow \boxed{\theta_{t^*}}$

This can prevent the model from fitting increasingly complex or noisy patterns.

#### Connection to Gradient Descent

For some simple models, the connection can be made mathematically explicit.

Consider least-squares regression: $L(\theta)$=$\frac{1}{2}\|X\theta-y\|^2$

Gradient descent gives:

$\theta_{t+1} $ = $ \theta_t-\eta X^T(X\theta_t-y)$

Starting from $\theta_0=0$, the solution after a finite number of iterations is biased toward certain lower-complexity solutions.

Thus, **finite optimization time itself can act like regularization**.

This is sometimes called **implicit regularization** or **iterative regularization**.

#### Early Stopping vs Explicit Regularization

| Technique | Mechanism |
| --- | --- |
| **Early stopping** | Stop optimization early |
| **Weight decay / L2** | Penalize large weights |
| **Dropout** | Add stochastic noise / remove activations |
| **Data augmentation** | Increase effective training diversity |

Early stopping does **not** explicitly add a term such as:

$L_{\text{total}} $ = $L_{\text{train}} + \lambda\|\theta\|^2$

Instead, the **optimization trajectory and stopping time** provide the regularizing effect.

#### Important distinction

Early stopping is **not simply "stopping when training loss gets bad."**

Usually:

- Training loss → continues decreasing
- Validation loss → decreases initially
- Validation loss → eventually stops improving or increases
- Training stops → parameters from the best validation point are retained

### Q45. How would you debug a neural network whose loss suddenly becomes NaN?
> ✍️ _Drafted answer (not from the original notes)._

Treat it as a bisection problem: find the **first step** and the **first tensor** that goes non-finite.

1. **Reproduce deterministically** — fix seeds, save the batch and checkpoint just before the blow-up, and replay that step.
2. **Turn on anomaly detection** — `torch.autograd.set_detect_anomaly(True)` points at the backward op that produced NaN; add forward hooks that assert `torch.isfinite` on activations.
3. **Check the data** — NaN/Inf in inputs or labels, division by zero in preprocessing, labels outside the class range, empty batches.
4. **Check numerically unsafe ops** — `log(0)`, `sqrt` of negative, `exp` overflow, softmax on huge logits, division by a tiny std in a custom norm. Use fused stable versions (`log_softmax`, `BCEWithLogitsLoss`, `cross_entropy`) and add `eps`.
5. **Check optimisation** — learning rate too high (loss spikes before NaN), missing warmup, exploding gradients. Log the global gradient norm; add gradient clipping and lower the LR.
6. **Mixed precision** — FP16 overflow is a classic cause. Use a dynamic `GradScaler`, keep loss/softmax/norms in FP32, or switch to BF16.
7. **Optimizer state** — Adam with a tiny `eps` in FP16, or a corrupted checkpoint, can produce Inf in the second moment.

```python
for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        print("non-finite grad in", name)
```

**Production habit:** log loss, grad-norm and LR every step, and have the training loop skip and alert on non-finite losses rather than silently corrupting weights.

### Q46. Why do CNNs work well for images?
> ✍️ _Drafted answer (not from the original notes)._

CNNs encode **priors that match the structure of natural images**:

- **Locality** — a pixel is most related to its neighbours, so small kernels (3×3) look at local patches first.
- **Weight sharing** — the same filter slides over the image, so an edge detector learned in one place works everywhere. This gives **translation equivariance** and cuts parameters massively (a 3×3×64×64 conv has ~37k weights regardless of image size, versus millions for a dense layer).
- **Hierarchy** — stacking layers grows the receptive field, so features compose: edges → textures → parts → objects.
- **Pooling / striding** — adds some tolerance to small shifts and reduces compute.

**Tradeoffs:** these inductive biases make CNNs very **data-efficient**, which is why they beat Vision Transformers on small/medium datasets. ViTs have weaker priors and need more data or pretraining, but scale better and model global context earlier. Modern designs (ConvNeXt, hybrid CNN-ViT) borrow from both.

**Production angle:** CNNs are still the default for edge and real-time vision (YOLO, MobileNet, EfficientNet) because convolution is extremely well optimised on GPUs, NPUs and mobile accelerators.

### Q47. Explain convolution, stride, padding, dilation, and receptive field.
> ✍️ _Drafted answer (not from the original notes)._

- **Convolution** — slide a learned kernel (e.g. 3×3×C_in) over the input and take a dot product at each position, producing one output channel per kernel. It detects the same local pattern everywhere.
- **Stride** — how far the kernel moves each step. Stride 2 halves the spatial resolution (cheap downsampling).
- **Padding** — adds border pixels (usually zeros) so the kernel can be centred on edge pixels and the output size is controlled.
- **Dilation** — spaces out kernel taps (dilation 2 means a 3×3 kernel covers a 5×5 area). It enlarges the receptive field without more parameters or downsampling; used in segmentation (DeepLab) and WaveNet.
- **Receptive field** — the region of the input that affects one output unit. It grows with depth, kernel size, stride and dilation.

**Output size:**

$$O = \left\lfloor \frac{I + 2P - D(K-1) - 1}{S} \right\rfloor + 1$$

**Receptive field** (recursive): $r_l = r_{l-1} + (K_l - 1)\,D_l \prod_{i<l} S_i$.

**Example:** three stacked 3×3 convs (stride 1) have a 7×7 receptive field with fewer parameters and more nonlinearity than one 7×7 conv, which is the VGG insight.

**Practical note:** the *effective* receptive field is smaller than the theoretical one (centre pixels dominate), which is one reason global-context modules and attention help on large objects.

### Q48. What is the difference between same and valid convolution?
> ✍️ _Drafted answer (not from the original notes)._

- **Valid convolution** — no padding. The kernel only sits where it fully fits, so the output shrinks: $O = I - K + 1$ (stride 1). A 32×32 input with a 3×3 kernel gives 30×30.
- **Same convolution** — pad so the output has the **same spatial size** as the input (stride 1). For odd kernel $K$, pad $P = (K-1)/2$ on each side (3×3 → pad 1).

**Tradeoffs:**

- *Valid* has no artificial border values, but features shrink every layer, which limits depth and loses border information.
- *Same* keeps shapes stable, which makes deep stacks and skip connections (ResNet, U-Net) easy to wire, at the cost of edge artefacts from the zero padding. Reflect or replicate padding can reduce those.

```python
import torch.nn as nn
nn.Conv2d(64, 64, kernel_size=3, padding=0)       # valid
nn.Conv2d(64, 64, kernel_size=3, padding="same")  # same (stride 1)
```

Note that with stride > 1, "same" in TensorFlow means $O = \lceil I/S \rceil$, and the padding can be asymmetric, which is a common source of PyTorch/TF porting bugs.

### Q49. How does pooling provide translation invariance?
> ✍️ _Drafted answer (not from the original notes)._

Strictly, pooling gives only **local, approximate** translation invariance.

- Convolution is **equivariant**: shift the input and the feature map shifts by the same amount.
- **Max pooling** takes the maximum over a window (e.g. 2×2). If a feature moves by a pixel but stays inside the same window, the pooled output is unchanged. So small shifts are absorbed.
- Stacking conv + pool layers compounds this, and **global average pooling** at the end discards position entirely, so the classifier sees "is the feature present" rather than "where".

**Caveats worth raising in an interview:**

- Invariance is only to small shifts. Strided pooling causes **aliasing**, and a 1-pixel shift can noticeably change CNN outputs (Zhang 2019, "Making CNNs Shift-Invariant Again" fixes this with blur-pooling).
- Invariance is not always wanted. Detection and segmentation need location, so they rely on skip connections or avoid aggressive pooling.
- Most modern robustness to translation actually comes from **data augmentation** (random crops and shifts) and global pooling, not from the pooling layers themselves.

### Q50. Why did ResNet outperform many earlier CNN architectures?
> ✍️ _Drafted answer (not from the original notes)._

**Problem:** before ResNet, making plain networks deeper (beyond ~20 layers) made **training** error worse. This is the *degradation problem*: an optimisation difficulty, not overfitting.

**Key idea — residual learning:** each block learns a residual $F(x)$ and outputs $y = F(x) + x$ through an identity skip connection.

- If the best mapping is close to identity, the block only has to push $F(x)$ towards 0, which is easy.
- Gradients flow straight through the identity path ($\partial y/\partial x = \partial F/\partial x + I$), which eases vanishing gradients and lets 100+ (even 1000) layer networks train.
- The network behaves like an ensemble of many shallower paths, which smooths the loss landscape.

**Other ingredients:** BatchNorm after every conv, bottleneck blocks (1×1 → 3×3 → 1×1) for cheap depth, and global average pooling instead of huge FC layers.

**Result:** ResNet-152 won ImageNet 2015 with 3.57% top-5 error, with fewer FLOPs than VGG.

**Legacy:** skip connections are now everywhere: Transformers (residual stream), U-Net, DenseNet and diffusion models. ResNet-50 remains a standard production and benchmark backbone because it is simple and well supported by hardware.

### Q51. Compare ResNet, EfficientNet, Vision Transformer, and ConvNeXt.
> ✍️ _Drafted answer (not from the original notes)._

| Model | Core idea | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **ResNet** (2015) | Residual blocks with skip connections | Simple, robust, well optimised everywhere, good transfer | Less accurate per FLOP than newer models |
| **EfficientNet** (2019) | Compound scaling of depth, width and resolution; MBConv + squeeze-excitation | Excellent accuracy per parameter/FLOP; good for mobile | Depthwise convs are memory-bound, so it is often slower on GPU than FLOPs suggest; harder to train |
| **Vision Transformer (ViT)** (2020) | Split image into 16×16 patches, treat them as tokens, apply a standard Transformer | Global context from layer 1, scales very well with data, shares stack with LLMs (CLIP, multimodal) | Weak inductive bias, so it needs large pretraining or heavy augmentation; quadratic cost in patches at high resolution |
| **ConvNeXt** (2022) | A ResNet "modernised" with Transformer design choices: 7×7 depthwise convs, LayerNorm, GELU, inverted bottlenecks, AdamW training | Matches Swin/ViT accuracy while staying a pure CNN; efficient at high resolution; easy to deploy | Less natural for multimodal token pipelines |

**How I would choose:**

- **Limited data, edge deployment** → EfficientNet or MobileNet-style, or ResNet for simplicity.
- **Large pretraining or multimodal** → ViT (often a CLIP or DINOv2 backbone).
- **Dense prediction at high resolution, strong general backbone** → ConvNeXt or Swin.

**Takeaway:** much of the ViT vs CNN gap came from training recipes (augmentation, AdamW, long schedules), which is exactly what ConvNeXt showed.

### Q52. Explain IoU, precision, recall, and mAP for object detection.
> ✍️ _Drafted answer (not from the original notes)._

- **IoU (Intersection over Union)** — overlap between a predicted box and a ground-truth box: $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$. A prediction counts as a **true positive** if IoU ≥ a threshold (e.g. 0.5) and the class matches. Each ground truth can be matched only once.
- **Precision** — $TP / (TP + FP)$: of the boxes predicted, how many are correct. Duplicates and wrong boxes are FPs.
- **Recall** — $TP / (TP + FN)$: of the real objects, how many were found.
- **AP (Average Precision)** — sort detections by confidence, sweep the threshold, plot the precision-recall curve, and take the area under it (interpolated). Computed **per class**.
- **mAP** — the mean of AP over classes.
  - **PASCAL VOC:** mAP@0.5.
  - **COCO:** mAP@[0.5:0.95], the average over IoU thresholds 0.5, 0.55, …, 0.95. This rewards tight localisation. COCO also reports AP_small, AP_medium and AP_large.

**Production angle:** mAP is a ranking metric that summarises all thresholds. In deployment you pick **one** confidence threshold, so also report precision and recall at the operating point, per-class performance (rare classes hide inside the mean) and latency. Business cost often depends on FP vs FN asymmetry, for example missed defects vs false alarms.

### Q53. How does NMS work?
> ✍️ _Drafted answer (not from the original notes)._

**Non-Maximum Suppression** removes duplicate detections of the same object.

**Algorithm (per class):**

1. Drop boxes below a confidence threshold.
2. Sort the remaining boxes by score.
3. Take the highest-scoring box and keep it.
4. Remove every remaining box whose IoU with it exceeds a threshold (e.g. 0.5).
5. Repeat until no boxes are left.

```python
def nms(boxes, scores, iou_thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        ious = iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return keep
```

**Tradeoffs and variants:**

- **Threshold too low** suppresses true objects that overlap (crowds, stacked items). **Too high** leaves duplicates.
- **Soft-NMS** decays the scores of overlapping boxes instead of deleting them, which helps crowded scenes.
- **Class-agnostic NMS** suppresses across classes; **batched NMS** runs per class efficiently.
- **DIoU-NMS** also considers centre distance.
- **NMS-free detectors** (DETR, YOLOv10) learn one-to-one matching, removing this heuristic, its latency and its hard-to-export post-processing.

**Production:** NMS can dominate latency on edge devices with many candidate boxes, so use the framework's fused implementation (`torchvision.ops.nms`, TensorRT's NMS plugin).

### Q54. What is data augmentation, and why does it improve generalization?
> ✍️ _Drafted answer (not from the original notes)._

**Data augmentation** creates label-preserving variants of training examples so the model sees more of the input distribution.

- **Vision:** random crop/resize, flips, rotation, colour jitter, blur, Cutout, **Mixup/CutMix**, RandAugment.
- **Text:** back-translation, synonym replacement, token dropout, paraphrasing with an LLM.
- **Audio:** time shift, noise, SpecAugment.
- **Tabular:** SMOTE, noise injection (use with care).

**Why it improves generalization:**

- **Encodes invariances** you know are true ("a flipped cat is still a cat"), so the model does not have to learn them from data.
- **Acts as a regulariser** — more effective data and fewer spurious cues to memorise, which reduces variance and overfitting.
- **Covers distribution shift** — lighting, camera and orientation differences in production.
- Mixup and CutMix also improve **calibration** and smooth decision boundaries.

**Production angle:** design augmentations from the **failure modes** you see in production (motion blur, glare, low light) rather than a generic list. Apply them on the fly on GPU (DALI, Kornia) to avoid data-loader bottlenecks. Never augment validation or test sets, except deliberately for test-time augmentation.

### Q55. When can augmentation actually hurt model performance?
> ✍️ _Drafted answer (not from the original notes)._

Augmentation hurts when it **breaks the label, breaks the distribution, or adds noise the model cannot use**.

- **Label-changing transforms** — horizontal flip on text or digits ("b" vs "d"), rotating a 6 into a 9, flipping left/right in medical images where laterality matters, or vertical flips for street scenes.
- **Destroying the signal** — colour jitter when colour is the label (ripe vs unripe fruit, skin lesions), or aggressive crops that remove the object, which produces mislabeled samples.
- **Distribution mismatch** — augmentations that never occur in production move capacity away from the real distribution. Strong augmentation on a small model can cause **underfitting**.
- **Too strong for the capacity or the schedule** — heavy RandAugment or Mixup on small models or short training lowers accuracy. The best strength usually scales with model size.
- **Leakage** — augmenting before the train/test split puts near-duplicates in both sets and inflates metrics.
- **Detection and segmentation** — transforming the image but not the boxes or masks consistently.
- **Text** — synonym swaps can flip sentiment or meaning ("not bad").
- **Class imbalance** — oversampling plus augmentation of a rare class can overfit to augmentation artefacts.

**Mitigation:** treat augmentation as a hyperparameter, run ablations, check per-class metrics, and visually inspect augmented batches before training.

### Q56. Explain the architecture of CLIP.
> ✍️ _Drafted answer (not from the original notes)._

**CLIP (Contrastive Language-Image Pretraining, OpenAI 2021)** is a **dual-encoder** model trained on ~400M image-text pairs from the web.

- **Image encoder** — a ResNet or ViT that maps an image to a vector.
- **Text encoder** — a Transformer that maps a caption to a vector (the EOS token representation).
- **Projection heads** — linear layers map both into a **shared embedding space** (e.g. 512-d), followed by L2 normalisation.
- **Training** — for a batch of $N$ pairs, compute the $N \times N$ cosine-similarity matrix scaled by a learned temperature. The matching pairs are on the diagonal. Apply a **symmetric cross-entropy (InfoNCE)** loss over rows (image→text) and columns (text→image).

```python
logits = (img_emb @ txt_emb.T) * logit_scale.exp()
labels = torch.arange(N)
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

**Properties:**

- Encoders are independent, so image and text embeddings can be **precomputed and indexed**, which suits retrieval at scale.
- Enables zero-shot classification, text-to-image search, and serves as a backbone for multimodal LLMs (e.g. the LLaVA vision encoder) and the text conditioning in diffusion models.

**Limitations:** weak at counting, spatial relations and fine-grained text in images; inherits web-data biases; a bag-of-concepts behaviour because there is no cross-attention between modalities.

### Q57. Why does CLIP use contrastive learning?
> ✍️ _Drafted answer (not from the original notes)._

**Because the supervision is noisy, open-ended web captions, and contrastive learning turns them into an efficient, scalable training signal.**

- **Predicting exact captions is too hard.** The OpenAI authors first tried generating captions; it learned about 3× slower for zero-shot accuracy. The model wastes capacity on wording, not on meaning.
- **A contrastive objective only asks "which caption matches this image?"** among the batch. That is a much easier proxy task that still forces the embeddings to capture semantics.
- **No fixed label set** — any text becomes a class, which is what makes zero-shot transfer possible.
- **Aligned shared space** — matching pairs pulled together and non-matching pairs pushed apart gives embeddings where cosine similarity is meaningful across modalities, which is ideal for retrieval.
- **Scales well** — every other sample in the batch is a free negative. CLIP used a 32k batch, so each step had ~1 billion negative pairs.

**Tradeoffs:** quality depends on large batches (expensive multi-GPU all-gather), false negatives in the batch (two near-identical captions), and the temperature setting. Follow-ups address this: **SigLIP** uses a pairwise sigmoid loss that works with smaller batches; **ALIGN** scales to noisier data; and **BLIP/CoCa** add captioning losses back for generative ability.

### Q58. How would you use CLIP for zero-shot image classification?
> ✍️ _Drafted answer (not from the original notes)._

**Turn class names into text prompts, embed everything, and pick the nearest text.**

1. For each class, write prompts like `"a photo of a {label}."`. **Prompt ensembling** (averaging the embeddings of many templates: "a blurry photo of a…", "a close-up of a…") usually adds 1–5% accuracy.
2. Encode the prompts with the text encoder, L2-normalise them, and cache the class-embedding matrix.
3. Encode the image, normalise it, compute cosine similarities, scale by the temperature, and apply softmax.

```python
import torch, open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
labels = ["cat", "dog", "car"]
with torch.no_grad():
    t = model.encode_text(tok([f"a photo of a {l}." for l in labels]))
    t = t / t.norm(dim=-1, keepdim=True)
    i = model.encode_image(preprocess(img).unsqueeze(0))
    i = i / i.norm(dim=-1, keepdim=True)
    probs = (100 * i @ t.T).softmax(dim=-1)
```

**Practical points:**

- Class names matter. Use descriptive, unambiguous names ("crane (bird)"), and add domain context ("a satellite photo of…").
- Zero-shot is weak on specialised domains (medical, industrial defects). Improve it cheaply with a **linear probe** on frozen embeddings, few-shot adapters (Tip-Adapter, CoOp), or fine-tuning.
- Calibrate a reject threshold for out-of-set images, because softmax over a closed label list always picks something.

### Q59. Explain self-attention mathematically.
The core operation is:

$Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

#### Transformer Self-Attention

Self-attention allows each token to determine **how much attention to give to every other token** in the sequence.

Given input embeddings:

$X \in \mathbb{R}^{n \times d_{\text{model}}}$

means **$X$ is a matrix of real numbers with $n$ rows and $d_{\text{model}}$ columns.**

we create **Query, Key, and Value** matrices:

$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$

where $W_Q,W_K,W_V$ are learned projection matrices.

#### 1. Compute attention scores

Compare each query with every key using the dot product: $S = QK^T$

Scale the scores to prevent large values: $S = \frac{QK^T}{\sqrt{d_k}}$

where $d_k$ is the dimension of each key vector.

#### 2. Apply softmax

Convert scores into attention probabilities:

$A = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$

Each row of $A $ represents how much one token attends to all tokens.

#### 3. Weighted sum of values

$\boxed{ \operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V }$

So the overall flow is:

$X \rightarrow Q,K,V \rightarrow QK^T \rightarrow \text{Scale} \rightarrow \text{Softmax} \rightarrow AV$

#### Why Q,K,VQ,K,V?

- **Query:** What information is this token looking for?
- **Key:** What information does each token contain?
- **Value:** What information should actually be retrieved?

#### Multi-Head Attention

Instead of performing one attention operation, Transformers use multiple heads:

$head_i = \operatorname{Attention} (XW_Q^i,XW_K^i,XW_V^i)$

Then concatenate them: $\operatorname{MHA}(X) = \operatorname{Concat}(head_1,\ldots,head_h)W_O$

Different heads can learn different relationships, such as syntax, semantic relationships, or long-range dependencies.

### Q60. Why divide by $\sqrt{d_k}$?
> ✍️ _Drafted answer (not from the original notes)._

Scaled dot-product attention is $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$. The $\sqrt{d_k}$ is there to keep the **variance of the logits** around 1.

**Derivation:** if the components of $q$ and $k$ are independent with mean 0 and variance 1, then

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \quad \text{Var}(q \cdot k) = d_k.$$

So the standard deviation grows like $\sqrt{d_k}$. With $d_k = 128$ the logits have std ≈ 11.

**Why that is a problem:** large logits push softmax into **saturation**. One position gets almost all the probability mass and the rest get ~0. The softmax Jacobian $\text{diag}(p) - pp^\top$ then becomes nearly zero, so **gradients vanish** and training is slow or unstable. Attention also becomes prematurely "hard".

Dividing by $\sqrt{d_k}$ restores unit variance, so the softmax temperature does not depend on head dimension.

**Related practice:** some models instead apply **QK-LayerNorm** or learn the temperature for extra stability at large scale. In mixed precision, the scaling also helps avoid FP16 overflow in the logits, which is why FlashAttention applies the scale inside the fused kernel.

### Q61. What is the difference between self-attention and cross-attention?
> ✍️ _Drafted answer (not from the original notes)._

Both use the same operation, $\text{softmax}(QK^\top/\sqrt{d_k})V$. The difference is **where Q, K and V come from**.

| | Self-attention | Cross-attention |
| --- | --- | --- |
| Q from | Sequence X | Sequence X (e.g. the decoder) |
| K, V from | The **same** sequence X | A **different** sequence Y (e.g. encoder output, image features) |
| Purpose | Relate tokens within one sequence | Condition one sequence on another |
| Shape | $n \times n$ attention map | $n \times m$ attention map |

**Examples:**

- **Self-attention** — every token in a sentence attends to the other tokens (BERT, GPT). In decoders it is **causally masked**.
- **Cross-attention** — in the original encoder-decoder Transformer for translation, the decoder queries attend over the encoder's source-sentence states. In **Stable Diffusion**, image latents (Q) attend to CLIP text embeddings (K, V). In Flamingo and some VLMs, text attends to image features.

**Production angle:** cross-attention K/V over a fixed context (the encoder output or image) can be **computed once and cached** for the whole generation. Decoder-only LLMs avoid cross-attention entirely by concatenating the context into the prompt and using self-attention, which simplifies the architecture but makes context length the cost driver.

### Q62. Why do Transformers need positional information?
> ✍️ _Drafted answer (not from the original notes)._

Self-attention is **permutation-equivariant**: it computes weighted sums over a *set* of tokens. If you shuffle the input tokens, the outputs shuffle in the same way, and nothing in $\text{softmax}(QK^\top)V$ depends on order. Without extra information, "dog bites man" and "man bites dog" look identical.

RNNs get order from sequential processing and CNNs from local kernels. Transformers process all tokens in parallel, so order must be **injected explicitly**.

**Ways to add it:**

- **Absolute positional encodings** — sinusoidal (original Transformer) or learned vectors added to token embeddings (BERT, GPT-2).
- **Relative encodings** — bias attention by the distance $i - j$ (T5, Transformer-XL).
- **RoPE** — rotate Q and K by position-dependent angles so the dot product depends on relative offset (LLaMA, most modern LLMs).
- **ALiBi** — add a linear distance penalty to attention logits.

**Nuance:** a causally masked decoder can infer some position implicitly (by counting how many tokens are visible), and "NoPE" models work at small scale. In practice explicit encodings train better and matter a lot for **long-context extrapolation**, which is why choices like RoPE scaling (NTK, YaRN) are central when extending context windows.

### Q63. Compare absolute positional embeddings, RoPE, ALiBi, and relative positional encoding.
> ✍️ _Drafted answer (not from the original notes)._

| Method | How it works | Pros | Cons | Used in |
| --- | --- | --- | --- | --- |
| **Absolute (sinusoidal / learned)** | Add a position vector $p_i$ to token embedding $x_i$ at the input | Simple, cheap | Learned: cannot go beyond max trained length. Both extrapolate poorly; position is mixed with content | Original Transformer, BERT, GPT-2 |
| **Relative positional encoding** | Add a learned bias or embedding based on distance $i-j$ inside attention | Captures what usually matters (relative offset); better generalisation | Extra compute/memory; harder with efficient kernels | Transformer-XL, T5 (bucketed bias) |
| **RoPE** (rotary) | Rotate pairs of dimensions of Q and K by angle $m\theta_k$ for position $m$; then $q_m^\top k_n$ depends only on $m-n$ | Relative by construction, no extra parameters, works with KV cache and FlashAttention, extendable via NTK/YaRN/position interpolation | Degrades beyond trained length without scaling tricks | LLaMA, Mistral, Qwen, GPT-NeoX |
| **ALiBi** | Add a fixed, head-specific linear penalty $-m\cdot\lvert i-j\rvert$ to attention logits | No parameters; extrapolates well to longer inputs ("train short, test long") | Strong recency bias can hurt long-range retrieval; somewhat lower quality than RoPE at scale | BLOOM, MPT |

**Interview summary:** absolute encodings say *where* a token is; relative, RoPE and ALiBi say *how far apart* two tokens are, which generalises better. RoPE is the current default because it is parameter-free, efficient and has well-understood context-extension methods.

### Q64. Why does self-attention have $O(n^2)$ complexity?
> ✍️ _Drafted answer (not from the original notes)._

For a sequence of $n$ tokens with head dimension $d$:

1. **Scores:** $QK^\top$ multiplies an $n \times d$ matrix by a $d \times n$ matrix, producing an **$n \times n$** matrix. Every token is compared with every other token, which is $O(n^2 d)$ compute.
2. **Softmax** over each row of the $n \times n$ matrix is $O(n^2)$.
3. **Weighted sum** with $V$ ($n \times n$ times $n \times d$) is $O(n^2 d)$.

So **time is $O(n^2 d)$** and naïvely **memory is $O(n^2)$** to store the attention matrix, per head and per layer. Compare the feed-forward layers at $O(n d^2)$: attention dominates once $n$ exceeds roughly $d$ (a few thousand tokens).

**Example:** going from 4k to 32k tokens is 8× the length but **64×** the attention FLOPs.

**What changes this:**

- **FlashAttention** keeps compute at $O(n^2)$ but cuts memory to $O(n)$ by never materialising the full matrix.
- **Sparse, sliding-window, linear and low-rank attention**, or state-space models (Mamba), reduce compute to ~$O(n)$ or $O(n \log n)$, usually with some quality tradeoff.
- **At inference with a KV cache**, each new token costs $O(n)$, but over the full generation it is still quadratic in total.

### Q65. What techniques reduce Transformer attention complexity?
> ✍️ _Drafted answer (not from the original notes)._

**Exact but more efficient (same result):**

- **FlashAttention** — tiled, fused kernel; $O(n)$ memory and much less HBM traffic. Still $O(n^2)$ FLOPs.
- **KV cache + GQA/MQA** — cut inference memory and bandwidth rather than asymptotic cost.

**Sparse patterns:**

- **Sliding-window / local attention** — each token attends to $w$ neighbours, giving $O(nw)$ (Longformer, Mistral).
- **Global + local + random tokens** — BigBird, Longformer. Stays expressive with near-linear cost.
- **Dilated / strided / block-sparse** attention (Sparse Transformer).

**Approximations:**

- **Low-rank** — project K and V to a fixed length $k$ (Linformer): $O(nk)$.
- **Kernel / linear attention** — replace softmax with a feature map so $(QK^\top)V = Q(K^\top V)$: $O(nd^2)$ (Performer, Linear Transformer).
- **Hashing / clustering** — attend only within LSH buckets (Reformer).

**Architectural alternatives:**

- **State-space models and RNN-like designs** — Mamba, RWKV, RetNet: linear time, constant-memory inference.
- **Hybrids** — mostly linear or SSM layers with a few full-attention layers (Jamba).
- **Context management** — retrieval, summarisation and memory tokens so you do not need long context at all.

**Tradeoff:** approximate and linear methods often lose quality on precise long-range retrieval ("needle in a haystack"). That is why frontier LLMs mostly keep exact attention plus FlashAttention, GQA, sliding windows and ring/sequence parallelism, rather than replacing softmax attention.

### Q66. Explain Multi-Head Attention.
> ✍️ _Drafted answer (not from the original notes)._

Instead of one attention over the full model width $d_{model}$, run $h$ attentions in parallel, each in a lower-dimensional subspace $d_k = d_{model}/h$.

$$\text{head}_i = \text{softmax}\left(\frac{(XW_i^Q)(XW_i^K)^\top}{\sqrt{d_k}}\right)XW_i^V$$

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O$$

**Steps:**

1. Project the input into Q, K and V (one big matmul in practice), then reshape to $(batch, h, n, d_k)$.
2. Compute scaled dot-product attention independently per head, with masks as needed.
3. Concatenate the heads and mix them with the output projection $W^O$.

```python
q, k, v = qkv(x).view(B, n, 3, h, dk).permute(2, 0, 3, 1, 4)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out = out.transpose(1, 2).reshape(B, n, h * dk)
y = proj(out)
```

**Cost:** roughly the same FLOPs and parameters as one full-width head, because the dimensions are split, not duplicated.

**Example:** GPT-3 175B uses 96 heads of size 128 ($d_{model} = 12288$). LLaMA-2 70B uses 64 query heads but only 8 KV heads (GQA) to shrink the KV cache.

### Q67. Why are multiple attention heads useful?
> ✍️ _Drafted answer (not from the original notes)._

A single head produces **one** softmax distribution per token, so it must average all the relationships it cares about into one weighted sum. Multiple heads let the model attend to **different things at the same time**.

- **Different relation types in parallel** — research has found heads specialised in previous-token, syntactic dependency (subject–verb), coreference, positional and **induction heads** (copying patterns, key for in-context learning).
- **Different representation subspaces** — each head has its own projections, so similarity is measured in different learned spaces.
- **Avoids averaging away information** — one head attending to two important tokens blurs them; two heads can each focus sharply.
- **Same cost** — heads split $d_{model}$, so you get this diversity at about the cost of one full-width head, and the computation parallelises well on GPUs.
- **Ensemble-like robustness.**

**Nuance:** many heads are redundant. Michel et al. (2019) pruned a large fraction of heads with little loss, and too many tiny heads ($d_k$ too small) hurts quality. That redundancy motivates **MQA/GQA** (share K/V across heads) and **head pruning** for cheaper inference.

### Q68. What is KV caching during inference?
> ✍️ _Drafted answer (not from the original notes)._

In autoregressive decoding, each new token must attend to all previous tokens. A **KV cache** stores the **Key and Value tensors** of every previous token, for every layer and head, so they are computed once and reused instead of reprocessing the whole prefix at each step.

**How it works:**

1. **Prefill** — run the prompt through the model in parallel, compute K and V for all prompt tokens and store them.
2. **Decode** — for each new token, compute only its Q, K and V, append K and V to the cache, and attend $q_t$ against all cached keys and values.

Per-step attention cost drops from reprocessing $t$ tokens to processing one token against $t$ cached entries.

**Memory cost:**

$$\text{KV bytes} = 2 \times L \times n_{kv\_heads} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{bytes}$$

**Example:** LLaMA-2 7B (32 layers, 32 heads × 128, FP16) uses ~0.5 MB per token, so ~2 GB for one 4k-token sequence. At serving scale **the KV cache, not the weights, often limits batch size and context length.**

**Optimisations:** GQA/MQA (fewer KV heads), **PagedAttention** (vLLM, no fragmentation), KV quantization (FP8/INT8), prefix/prompt caching across requests, sliding-window eviction and offloading to CPU memory.

### Q69. Why does KV cache significantly improve autoregressive generation?
#### KV Cache

**KV caching improves LLM inference by avoiding recomputation of previous tokens during autoregressive decoding.**

In the Transformer attention mechanism, each token produces **Key (K)** and **Value (V)** vectors. During generation:

- Without KV cache → the model repeatedly computes K/V for **all previous tokens** at every new token.
- With KV cache → previously computed K/V vectors are **stored in GPU memory** and reused.
- Only the **new token's K/V** needs to be computed.

#### Example

Generating: `I → I am → I am learning → I am learning AI`

When generating **AI**, the model reuses the cached K/V for: `I am learning`

and computes K/V only for **AI**.

#### Benefits

- **Lower decoding latency**
- Avoids redundant computation
- Improves **tokens/second**
- Especially important for long contexts and autoregressive generation

**Trade-off:** KV caching consumes significant **GPU memory**, which can become a bottleneck for large models and long context windows.

### Q70. What is Grouped Query Attention (GQA)?
> ✍️ _Drafted answer (not from the original notes)._

**Grouped Query Attention (GQA)** is a middle ground between multi-head (MHA) and multi-query attention (MQA). Query heads are split into $G$ groups, and **each group shares one Key/Value head**.

- MHA: $h$ query heads, $h$ KV heads.
- **GQA: $h$ query heads, $G$ KV heads** ($1 < G < h$).
- MQA: $h$ query heads, 1 KV head.

**Why:** during decoding, the bottleneck is **reading the KV cache from memory**, not FLOPs. The KV cache scales with the number of KV heads, so GQA cuts it by $h/G$.

**Example:** LLaMA-2 70B has 64 query heads and 8 KV heads, an **8× smaller KV cache**. That allows bigger batches and longer contexts on the same GPU, with quality close to full MHA.

**Implementation:** the K and V tensors are broadcast/repeated across the query heads within each group (`repeat_kv`), or the kernel indexes them directly. FlashAttention and vLLM support GQA natively.

**Uptraining:** Ainslie et al. (2023) showed you can convert an MHA checkpoint to GQA by **mean-pooling the KV heads** within each group and continuing training on ~5% of the original compute.

**Tradeoff:** fewer KV heads reduce representational capacity slightly. GQA recovers most of the quality lost by MQA, which is why it is the default in LLaMA-2/3, Mistral, Qwen and Gemma.

### Q71. Compare MHA, MQA, and GQA.
> ✍️ _Drafted answer (not from the original notes)._

| | **MHA** (multi-head) | **GQA** (grouped-query) | **MQA** (multi-query) |
| --- | --- | --- | --- |
| Query heads | $h$ | $h$ | $h$ |
| KV heads | $h$ | $G$ (e.g. 8) | 1 |
| KV cache size | 1× (baseline) | $G/h$ (e.g. 1/8) | $1/h$ |
| Decode speed | Slowest (memory-bound) | Fast | Fastest |
| Quality | Best | Close to MHA | Noticeably lower; can be unstable |
| Tensor-parallel fit | Easy | Good if $G$ ≥ number of TP shards | KV head must be replicated on every shard |
| Examples | GPT-3, original Transformer, BERT | LLaMA-2/3 70B, Mistral, Qwen | PaLM, Falcon-7B, StarCoder |

**Why it matters:** in autoregressive inference each generated token reads the full KV cache, so decode throughput is **memory-bandwidth-bound**. Fewer KV heads means less memory traffic, larger batches and longer contexts.

**How to choose:**

- **Training a small or encoder model where inference cost is not critical** → MHA.
- **Serving an LLM at scale** → GQA is the sweet spot; pick $G$ as a multiple of your tensor-parallel degree.
- **Extreme memory constraints** → MQA, accepting some quality loss.

A further step is **MLA (Multi-head Latent Attention, DeepSeek-V2/V3)**, which compresses K/V into a low-rank latent per token, shrinking the cache even more while keeping per-head expressivity.

### Q72. What is FlashAttention?
> ✍️ _Drafted answer (not from the original notes)._

**FlashAttention** (Dao et al., 2022) is an **exact**, IO-aware GPU kernel for attention. It computes the same output as standard attention but much faster and with far less memory.

**Problem with standard attention:** it materialises the $n \times n$ score and probability matrices in GPU **HBM** (slow, large memory). It writes and reads them several times (matmul → mask → softmax → dropout → matmul), so it is **memory-bandwidth-bound** and uses $O(n^2)$ memory.

**Key ideas:**

- **Tiling** — split Q, K and V into blocks that fit in on-chip **SRAM**. Compute attention block by block, never storing the full matrix.
- **Online softmax** — keep a running max and running sum per row so softmax can be computed incrementally across blocks and rescaled correctly.
- **Kernel fusion** — matmul, mask, softmax, dropout and matmul run in one kernel.
- **Recomputation** — in the backward pass, recompute attention blocks instead of storing the $n \times n$ matrix (trading cheap FLOPs for expensive memory).

**Impact:** memory is $O(n)$ instead of $O(n^2)$, there is a 2–4× wall-clock speed-up, and long-context training becomes feasible.

**Versions:** FA-2 (better parallelism and work partitioning), FA-3 (Hopper-specific: asynchrony, FP8).

**In practice:** use it via `torch.nn.functional.scaled_dot_product_attention`, `flash-attn`, or inference engines (vLLM, TensorRT-LLM). It is now the default in essentially every serious LLM stack.

### Q73. Why is FlashAttention faster without changing the attention result mathematically?
> ✍️ _Drafted answer (not from the original notes)._

Because it changes **how data moves**, not **what is computed**. Attention on modern GPUs is limited by **memory bandwidth**, not arithmetic.

**Where the time goes in standard attention:**

- An A100 does ~312 TFLOPs (BF16) but reads HBM at only ~2 TB/s. On-chip SRAM is ~10× faster but only ~20 MB in total.
- Standard attention writes $S = QK^\top$ ($n \times n$) to HBM, reads it back for softmax, writes $P$, and reads it again for $PV$. For long sequences that traffic dominates the runtime.

**What FlashAttention does:**

1. **Tiling** — load blocks of Q, K and V into SRAM and do all the work there.
2. **Online softmax** — softmax needs the row max and row sum over *all* keys. FlashAttention keeps running statistics $m$ (max) and $\ell$ (sum). When a new block arrives with a larger max, previous partial results are rescaled by $e^{m_{old} - m_{new}}$. This is algebraically **identical** to the full softmax.
3. **Fusion** — one kernel, so intermediates never touch HBM.
4. **Recompute in backward** — cheaper than reading a stored $n \times n$ matrix.

**Result:** HBM accesses drop from $\Theta(nd + n^2)$ to $\Theta(n^2 d^2 / M)$ (M = SRAM size). FLOPs actually *increase* slightly because of recomputation, yet wall-clock time falls 2–4×.

The output is exact up to floating-point rounding: results are not bit-identical because summation order changes, but the method is not an approximation.

### Q74. What is PagedAttention?
> ✍️ _Drafted answer (not from the original notes)._

**PagedAttention** (Kwon et al., 2023, the core of **vLLM**) manages the KV cache like an operating system manages **virtual memory pages**.

**Problem:** naïve servers pre-allocate one **contiguous** KV buffer per request sized for the maximum length (e.g. 2k–32k tokens). Because output length is unknown:

- **Internal fragmentation** — reserved space that is never used.
- **External fragmentation** — free gaps between buffers that are too small to reuse.
- The vLLM paper measured only **20–40%** of KV memory holding real tokens in existing systems.

**How PagedAttention works:**

- The KV cache is split into fixed-size **blocks** (e.g. 16 tokens each), stored anywhere in GPU memory.
- Each sequence has a **block table** that maps logical blocks to physical blocks, like a page table.
- Blocks are allocated **on demand** as tokens are generated; waste is limited to the last partial block (under 4%).
- The attention kernel is modified to gather K and V from non-contiguous blocks through the block table.
- **Copy-on-write sharing** — sequences with a common prefix (same system prompt, parallel sampling, beam search) share physical blocks with reference counts.

**Result:** near-zero waste, so many more concurrent sequences fit on the GPU. vLLM reported 2–4× throughput over prior systems (up to ~24× vs naïve HF Transformers).

### Q75. How does PagedAttention improve LLM serving?
> ✍️ _Drafted answer (not from the original notes)._

LLM decoding is memory-bound, so **throughput scales with how many sequences you can batch at once**, and batch size is limited by KV-cache memory. PagedAttention frees that memory.

**How it improves serving:**

- **Higher batch size → higher throughput.** Blocks are allocated on demand rather than reserved for the max length, so KV utilisation goes from ~20–40% to over 96%. More concurrent requests per GPU means a lower cost per token.
- **Pairs with continuous batching.** Requests join and leave the running batch at token granularity. Because memory is paged, a finished sequence's blocks are returned immediately and reused by new requests, with no compaction or copying.
- **Prefix sharing.** Shared system prompts, few-shot examples and RAG templates are stored once and referenced by many requests (copy-on-write). This is the basis for automatic **prefix caching**.
- **Parallel sampling and beam search** share the prompt's blocks across candidates, reducing memory by up to ~55% for those workloads.
- **Preemption and swapping.** Under memory pressure, whole blocks can be swapped to CPU memory or recomputed, which degrades gracefully instead of failing with OOM.
- **Predictable capacity planning** — you can compute how many tokens fit in the cache and set admission control on that.

**Tradeoff:** the indirection adds some kernel overhead (gathering non-contiguous blocks), which is small compared with the batching gains. vLLM, TensorRT-LLM, TGI and SGLang all use paged KV caches today.

### Q76. What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?
> ✍️ _Drafted answer (not from the original notes)._

| | **Encoder-only** | **Decoder-only** | **Encoder-decoder** |
| --- | --- | --- | --- |
| Attention | Bidirectional self-attention | **Causal** (masked) self-attention | Bidirectional encoder + causal decoder with **cross-attention** to the encoder |
| Pretraining | Masked language modelling (BERT), replaced-token detection (ELECTRA) | Next-token prediction | Span corruption / denoising (T5, BART) |
| Output | Contextual embeddings per token | Generated text, token by token | Generated text conditioned on an encoded input |
| Best for | Classification, NER, embeddings, retrieval, reranking | Open-ended generation, chat, code, in-context learning | Input→output transformation: translation, summarisation, speech-to-text |
| Examples | BERT, RoBERTa, DeBERTa, sentence-transformers | GPT, LLaMA, Claude, Mistral | T5, BART, mT5, Whisper, original Transformer |

**Production angle:**

- **Encoder-only** models are small, fast and cheap. They are still the right choice for embeddings, classifiers and cross-encoder rerankers in RAG.
- **Decoder-only** models are the general-purpose LLM default because one model handles any task via prompting.
- **Encoder-decoder** models are efficient when the input is long and the output short (the input is encoded once). They remain strong for translation and ASR.

### Q77. Why are decoder-only architectures dominant for modern LLMs?
> ✍️ _Drafted answer (not from the original notes)._

1. **Simplest objective that scales.** Next-token prediction on raw text needs no labels, uses every token as a training signal (unlike MLM, which trains on ~15%), and its loss follows smooth scaling laws.
2. **Generation is native.** Training (predict the next token) and inference (generate the next token) are the same task, so there is no train/inference gap.
3. **One model, every task.** Any task can be phrased as "text in, text out" via prompting. That enables **in-context learning** and instruction-following without task-specific heads, which emerged strongly in GPT-3.
4. **Engineering efficiency.** One homogeneous stack is easier to scale, shard and optimise. The KV cache makes autoregressive inference efficient, and an ecosystem (FlashAttention, vLLM, speculative decoding) is built around it.
5. **Empirical results.** At equal compute, decoder-only models have done as well as or better than encoder-decoder models for zero-shot generalisation (Wang et al., 2022). Most labs converged on the design, which compounded tooling advantages.
6. **Long-context and multimodal extension** is straightforward: just prepend tokens (images, documents, tool outputs).

**Caveats:** causal attention is less efficient for pure understanding tasks. Encoders still win for embeddings and classification per dollar, and encoder-decoder models remain strong for translation and ASR (Whisper). Decoder-only dominance is partly a scaling and ecosystem effect, not proof that it is strictly better.

### Q78. What is the difference between pretraining, instruction tuning, and alignment?
> ✍️ _Drafted answer (not from the original notes)._

| Stage | Data | Objective | What it gives you |
| --- | --- | --- | --- |
| **Pretraining** | Trillions of tokens of unlabeled text and code | Next-token prediction (self-supervised) | Knowledge, language and reasoning ability. A **base model** that continues text rather than follows instructions |
| **Instruction tuning (SFT)** | ~10k–1M curated (instruction, response) pairs, often multi-turn chats | Supervised cross-entropy on the response tokens | Follows instructions, adopts a chat format, uses tools, answers in the desired style |
| **Alignment / preference tuning** | Human or AI preference comparisons, verifiable rewards | RLHF (reward model + PPO), DPO, RLAIF / Constitutional AI, RLVR | Helpful, harmless and honest behaviour; refusing harmful requests; better reasoning when rewarded for verified correctness |

**Intuition:** pretraining builds **capability**, SFT teaches the **format and behaviour**, and alignment shapes **preferences and values** ("which of several valid answers is best").

**Practical points:**

- Compute is heavily skewed to pretraining (over 95%). Post-training is cheap but determines most of the user-perceived quality.
- SFT alone can only imitate demonstrations. Preference optimisation can push beyond them and reduce harmful or hallucinated outputs.
- Risks: the **alignment tax** (small capability loss), **reward hacking**, and over-refusal.
- In industry, teams rarely pretrain. They pick an aligned open model and do SFT, LoRA or DPO on domain data.

### Q79. Explain supervised fine-tuning (SFT).
> ✍️ _Drafted answer (not from the original notes)._

**Supervised fine-tuning (SFT)** continues training a pretrained model on **labeled input→output examples** with the standard next-token cross-entropy loss, usually computing the loss **only on the response tokens** (prompt tokens are masked).

$$\mathcal{L} = -\sum_{t \in \text{response}} \log p_\theta(y_t \mid x, y_{<t})$$

**Uses:** instruction-following and chat formatting (turning a base model into an assistant), domain adaptation (legal, medical, internal style), structured output, tool calling, and distilling a larger model's outputs into a smaller one.

**Practical recipe:**

- **Data quality beats quantity.** A few thousand clean, diverse, consistent examples often beat 100k noisy ones (the LIMA paper used 1k). Deduplicate and remove contradictions.
- Use the model's **chat template** exactly; template mismatch is a common silent failure.
- Low learning rate (~1e-5 to 2e-4 with LoRA), 1–3 epochs, early stopping on a held-out set; packing sequences for efficiency.
- **Parameter-efficient fine-tuning (LoRA/QLoRA)** for cost and to reduce forgetting.

**Limitations:** SFT only imitates the demonstrations. It can increase hallucination when it teaches facts the model does not know, and it can cause **catastrophic forgetting** of general skills. Mix in general data and evaluate on general benchmarks as well as the target task. Use preference tuning (DPO/RLHF) afterwards for quality beyond imitation.

### Q80. What is quantization?

> 🔁 _Already answered in [01-llm-fundamentals.md → Q6](ai-engineering/01-llm-fundamentals.md#q6-what-is-quantization); skipped here to avoid duplication._

### Q81. Compare INT8, INT4, FP16, BF16, and FP8.
> ✍️ _Drafted answer (not from the original notes)._

| Format | Bits | Layout | Range / precision | Typical use |
| --- | --- | --- | --- | --- |
| **FP16** | 16 | 1 sign, 5 exponent, 10 mantissa | Max ~65,504; good precision, **narrow range** | Inference; mixed-precision training (needs loss scaling) |
| **BF16** | 16 | 1 sign, **8 exponent**, 7 mantissa | Same range as FP32, lower precision | Default for LLM training and inference on A100/H100/TPU; no loss scaling needed |
| **FP8** | 8 | E4M3 (precision) or E5M2 (range) | Very coarse; needs per-tensor or per-block scaling | Training and inference on H100/Blackwell (Transformer Engine); W8A8 serving at ~2× BF16 throughput |
| **INT8** | 8 | Integer + scale (+ zero-point) | 256 levels, uniform spacing | PTQ for CNNs and LLMs (LLM.int8(), SmoothQuant); broad hardware support including CPUs and edge |
| **INT4** | 4 | Integer + per-group scale | 16 levels | Weight-only LLM inference (GPTQ, AWQ, GGUF Q4); 4× smaller than FP16 |

**Key tradeoffs:**

- **Range vs precision:** FP16 overflows (hence NaNs in training) while BF16 rarely does, at the cost of mantissa precision.
- **Floating vs integer:** FP8 handles outliers better than INT8 thanks to its exponent. Integer formats need careful scale selection.
- **Hardware support decides speed:** FP8 needs Hopper or newer. INT4 is mostly a *memory* win (weights are dequantized to FP16 for matmul) unless kernels are specialised.

**Rule of thumb:** train in BF16 (or FP8 at frontier scale), serve in FP8 or INT8 when quality matters and hardware supports it, and use INT4 weight-only to fit large models on limited GPUs, always validating on task evals.

### Q82. What is quantization-aware training vs post-training quantization?
> ✍️ _Drafted answer (not from the original notes)._

| | **Post-training quantization (PTQ)** | **Quantization-aware training (QAT)** |
| --- | --- | --- |
| When | After training, on a frozen model | During training or fine-tuning |
| Data | Small calibration set (~100–1000 samples) or none | Full or substantial training data |
| Cost | Minutes to hours | A training run (expensive for LLMs) |
| Accuracy | Good at 8-bit; can degrade at 4-bit or below | Best, especially at low bit-widths |
| Examples | Min-max / percentile calibration, GPTQ, AWQ, SmoothQuant | Fake-quant nodes with the straight-through estimator; QLoRA-style low-bit fine-tuning (adjacent) |

**How QAT works:** insert **fake-quantize** ops (quantize → dequantize) in the forward pass so the model "feels" the rounding error. Backprop uses the **straight-through estimator** (treating round() as identity for gradients), so the weights learn to be robust to quantization.

**How PTQ works:** run calibration data to collect activation ranges and choose scales and zero-points per tensor, channel or group. Advanced LLM PTQ methods minimise layer-wise reconstruction error (GPTQ uses second-order information) or protect salient weight channels (AWQ).

**Decision rule:**

- Start with **PTQ**. It is cheap, and INT8/FP8 or INT4 weight-only with GPTQ/AWQ is usually within ~1% on LLMs.
- Use **QAT** when PTQ loses too much accuracy: very low bit-widths (INT4 activations, 2–3 bit), small models (mobile CNNs, where each layer matters) or edge targets with strict integer-only hardware. Examples include Google's QAT Gemma checkpoints and MobileNet on-device.

Always compare against the FP16 baseline on the task's own evaluation, not just perplexity.

### Q83. What is RLHF?
**RLHF** is a technique for aligning an AI model with human preferences by using human feedback as the reward signal.

#### Typical pipeline

1. **Pretraining** — Train the model on large-scale data.
2. **Supervised Fine-Tuning (SFT)** — Fine-tune on high-quality human demonstrations.
3. **Preference collection** — Humans rank multiple model responses.
4. **Reward modeling** — Train a reward model to predict those preferences.
5. **RL optimization** — Optimize the language model against the learned reward, traditionally using PPO.

RLHF is useful when the desired behavior is difficult to specify with a deterministic reward, such as **helpfulness, harmlessness, instruction following, and response quality**.

### Q84. What is model collapse?
> ✍️ _Drafted answer (not from the original notes)._

**Model collapse** is the degradation that happens when generative models are **trained recursively on data generated by earlier models** instead of real human data (Shumailov et al., *Nature* 2024).

**Mechanism:**

- Each generation samples from an approximation of the previous distribution. Sampling and estimation errors compound, and the **tails of the distribution disappear first**: rare words, minority viewpoints and unusual cases.
- Over generations the distribution narrows and drifts. Outputs become bland and repetitive, and eventually nonsensical, with variance collapsing towards a few modes.
- The causes are statistical approximation error, function-approximation error and sampling choices (low temperature, top-p truncation), which all cut off the tails.

**Why it matters now:** more and more of the web is LLM-generated, so naïve future crawls get contaminated. Heavy use of synthetic data for SFT or pretraining can also reduce diversity.

**Mitigations:**

- **Keep real data in the mix** — accumulating synthetic data *alongside* the original data, rather than replacing it, largely avoids collapse (Gerstgrasser et al., 2024).
- **Provenance tracking and filtering** — watermarking, AI-text detection, data deduplication, and favouring pre-2022 or curated human sources.
- **Quality-controlled synthetic data** — verification (unit tests, math checkers), diversity-promoting generation, rejection sampling and human review.
- **Monitor diversity metrics** (entropy, distinct-n, tail-class coverage) across data and model versions.

Do not confuse it with **mode collapse** in GANs, which is a single training run failing to cover the modes, though both lose diversity.

### Q85. What is catastrophic forgetting during fine-tuning?
> ✍️ _Drafted answer (not from the original notes)._

**Catastrophic forgetting** is when fine-tuning a pretrained model on a new task overwrites the weights that encoded earlier knowledge, so performance on the original capabilities (general reasoning, other languages, safety behaviour, instruction following) drops sharply.

**Why it happens:** gradient descent only optimises the loss in front of it. With a narrow fine-tuning dataset, a high learning rate or many epochs, parameters drift far from the pretrained solution in directions the new loss doesn't penalise but old tasks depended on. Full fine-tuning of all weights makes this worse than parameter-efficient methods.

**Typical symptoms in production:**
- A model fine-tuned on support tickets now writes everything in ticket style and fails at simple maths.
- Safety/refusal behaviour from alignment degrades ("fine-tuning removes guardrails").
- General benchmarks (MMLU, your own general eval set) drop several points.

**Mitigations:**
- **Parameter-efficient fine-tuning** (LoRA/adapters): base weights stay frozen, so drift is bounded by the low-rank update.
- **Replay / data mixing:** mix 5–30% general instruction data into the fine-tuning set.
- **Regularise towards the base model:** lower learning rate, fewer epochs, EWC (penalise changes to parameters important for old tasks), or a KL penalty to the reference model.
- **Measure it:** always run a "general capability" regression suite alongside the task eval before shipping.

**Tradeoff:** stronger protection against forgetting usually means less specialisation; tune the mix ratio and LoRA rank against both evals.

### Q86. How would you fine-tune an LLM while preserving its general capabilities?
> ✍️ _Drafted answer (not from the original notes)._

The goal is task gains without regressing general skills, so I treat it as a two-sided optimisation with evaluation on both sides.

1. **Check fine-tuning is needed.** Try prompting, few-shot examples and RAG first; fine-tune for consistent behaviour/format, latency or cost.
2. **Use parameter-efficient methods.** LoRA/QLoRA on attention (and optionally MLP) projections keeps base weights frozen, limits drift and lets you hot-swap adapters per task.
3. **Curate and mix data.** High-quality, deduplicated, diverse task examples (a few thousand clean examples beat 100k noisy ones); mix in general instruction and safety data (replay); keep the base model's chat template.
4. **Optimise conservatively.** Low learning rate (≈1e-4 for LoRA, ≤1e-5 for full FT), warmup + cosine decay, 1–3 epochs, early stopping on validation. For preference tuning (DPO/PPO) keep a KL constraint to the reference model.
5. **Evaluate both axes.**
   - Task eval: golden set, task metrics, calibrated LLM-as-judge.
   - Regression eval: general benchmarks (MMLU, GSM8K, IFEval), safety/refusal tests, real production prompts.
   - Pick the checkpoint maximising task gain subject to a regression budget (e.g. < 1 point drop).
6. **Ship safely.** Version data, adapter and base model together; canary the new model; keep the base model as fallback.

If forgetting is unacceptable, route only task traffic to the adapter, or use model merging (e.g. TIES/DARE) and re-evaluate.

### Q87. What causes LLM hallucinations?
> ✍️ _Drafted answer (not from the original notes)._

An LLM is trained to produce **plausible next tokens**, not true statements, so hallucination is a natural failure mode. Main causes:

- **Training objective:** next-token likelihood rewards fluent continuations; nothing forces "I don't know", and RLHF can reward confident, helpful-sounding answers.
- **Knowledge gaps and staleness:** rare, long-tail, post-cutoff or private facts were never (or weakly) learned, so the model interpolates.
- **Lossy parametric memory:** knowledge is compressed into weights; similar entities blur (wrong dates, merged people, invented but realistic-looking citations).
- **Decoding:** high temperature/top-p can pick wrong low-probability tokens; once emitted, autoregressive generation commits and builds on them (snowballing).
- **Prompt/context issues:** ambiguous questions, false premises, contradictory or irrelevant context, or key facts lost in a long context.
- **Poor calibration:** confidence doesn't track accuracy, so the model can't reliably abstain.
- **Noisy training data:** errors and fiction in the corpus get reproduced.

**In RAG** add: retrieval misses (answer not in context), the model overriding context with priors, and chunking that splits key facts.

**Production takeaway:** treat hallucination as a systems problem — grounding with citations, abstention instructions, low temperature for factual tasks, verification steps and continuous groundedness evaluation.

### Q88. Can hallucinations be completely eliminated?
> ✍️ _Drafted answer (not from the original notes)._

**No, not completely** — they can be reduced and contained, not eliminated.

**Why not:**
- Generation is probabilistic and the model has no built-in notion of truth.
- Its knowledge is incomplete and compressed; questions outside it invite interpolation.
- Even with perfect retrieval it can misread, over-generalise or combine facts wrongly.
- Theoretical results suggest any computable LLM used as a general question answerer will hallucinate on some inputs.

**What you can do:**
- **Ground** answers in authoritative retrieved sources and require citations.
- **Constrain** outputs: structured outputs, closed-set classification, tool calls to databases/calculators instead of free recall.
- **Allow abstention:** instruct and reward "I don't know"; apply confidence thresholds.
- **Verify:** groundedness checks (NLI or LLM judge vs context), self-consistency, rule checks.
- **Human-in-the-loop** for high-stakes decisions.
- **Monitor** hallucination rate on sampled production traffic.

**Interview framing:** aim for a hallucination rate that is *acceptable for the use case and detectable when it happens*. A creative-writing tool tolerates it; a tax or medical assistant needs grounding, verification and escalation so the residual risk is managed.

### Q89. How would you evaluate an LLM beyond accuracy?
> ✍️ _Drafted answer (not from the original notes)._

Accuracy alone misses most of what makes an LLM useful and safe. I evaluate across:

- **Correctness/factuality:** exact match/F1 where there's one answer; rubric grading otherwise.
- **Groundedness/faithfulness:** claims supported by provided context (critical for RAG).
- **Relevance and completeness:** answers the actual question, all parts.
- **Instruction following and format:** schema validity, length, tone, language.
- **Reasoning quality:** step correctness; pass@k and unit tests for code.
- **Safety:** toxicity, bias, jailbreak and prompt-injection resistance, PII leakage, and correct (not excessive) refusals.
- **Calibration/abstention:** says "I don't know" when it should.
- **Robustness/consistency:** stable under paraphrases, typos, reordered context, repeated runs.
- **Efficiency:** latency p50/p95, tokens, cost per request.
- **User/business outcomes:** task completion, thumbs-up rate, escalation rate, deflection, conversion.

**How:** golden datasets per capability built from real queries and edge cases; automated metrics where possible; **LLM-as-judge** with explicit rubrics, calibrated against human labels; pairwise human evaluation for subjective quality; online A/B tests and monitoring. Track everything per model/prompt version so regressions show up before release.

### Q90. What is LLM-as-a-judge, and what are its limitations?
> ✍️ _Drafted answer (not from the original notes)._

**LLM-as-a-judge** uses a (usually strong) LLM to grade another model's output — pointwise against a rubric ("score faithfulness to the context 1–5") or pairwise ("which answer is better, A or B?"). It scales evaluation of open-ended outputs far cheaper and faster than human raters and correlates reasonably with humans when designed well.

**Limitations:**
- **Position bias** in pairwise comparisons → swap order and average.
- **Verbosity bias:** rewards longer, confident answers.
- **Self-preference** for outputs from its own model family.
- **Can't verify unknown facts:** may approve fluent hallucinations.
- **Inconsistency:** sensitive to prompt wording and sampling; fine-grained scales are noisy.
- **Weak in specialised domains** (legal, medical, code correctness) without references.
- **Cost/latency** at scale, and **Goodhart risk** if you optimise the system against the same judge.

**Good practice:**
- Clear rubrics with examples; prefer binary or pairwise judgments over 1–10 scores.
- Give references (gold answer, retrieved context) — reference-guided judging is far more reliable.
- **Calibrate** against a human-labelled set, report agreement (e.g. Cohen's kappa), re-check when the judge model changes.
- Different model family from the generator, temperature 0, reasoning before verdict.
- Use deterministic checks (unit tests, schema validation, exact match) wherever possible; keep judges for the rest.

### Q91. How would you build an evaluation framework for an enterprise LLM application?
> ✍️ _Drafted answer (not from the original notes)._

I build it as a layered, versioned system that runs offline (pre-release) and online (production).

1. **Define success with stakeholders:** correctness, groundedness, format compliance, safety, latency, cost and a business KPI (e.g. ticket deflection), with thresholds and a regression budget.
2. **Datasets:**
   - **Golden set:** hundreds to a few thousand real queries with reference answers/sources, stratified by intent, difficulty and risk, reviewed by domain experts.
   - **Adversarial set:** prompt injection, jailbreaks, PII, out-of-scope, false premises.
   - Continuously add production failures.
3. **Metrics per component:**
   - Retrieval: recall@k, precision@k, MRR.
   - Generation: faithfulness, answer relevance, correctness (rubric-based LLM judge calibrated vs humans).
   - Deterministic checks: JSON schema, citation presence, forbidden-content rules.
   - Ops: latency, tokens, cost, error rates.
4. **CI/CD gating:** every prompt, model, retriever or chunking change runs the suite and is compared to baseline; failing thresholds block release. Store results keyed by prompt/model/index/data versions.
5. **Human review loop:** periodic expert review of samples to catch what judges miss and recalibrate them.
6. **Online evaluation:** tracing (Langfuse/LangSmith/OpenTelemetry), sampled automated grading of live traffic, user feedback, A/B tests, dashboards and alerts on quality, safety and cost drift.
7. **Governance:** audit logs, documented eval results per release, escalation paths for high-risk failures — essential in regulated domains.

### Q92. Explain a complete RAG architecture.
Typical flow:

```javascript
Documents
   ↓
Parsing
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Database
   ↓
Query
   ↓
Retrieval
   ↓
Reranking
   ↓
Context Construction
   ↓
LLM
   ↓
Answer
```

#### RAG End-to-End

**Retrieval-Augmented Generation (RAG)** combines information retrieval with an LLM so the model can answer using external knowledge instead of relying only on its pretrained knowledge.

##### 1. Indexing / Ingestion

```txt
Documents
   ↓
Parsing / OCR
   ↓
Cleaning & Metadata
   ↓
Chunking
   ↓
Embedding Model
   ↓
Vector DB / Search Index
```

- Load documents from PDFs, web pages, databases, etc.
- Parse and clean the content.
- Split documents into meaningful chunks.
- Generate embeddings for each chunk.
- Store chunks, embeddings, and metadata in a vector database/search engine.

##### 2. Query Processing

```txt
User Query
    ↓
Query Rewriting / Expansion
    ↓
Retrieval
    ↓
Reranking
    ↓
Top-N Context
```

- Convert the query into an embedding.
- Retrieve relevant chunks using **dense, sparse, or hybrid retrieval**.
- Optionally use a **cross-encoder reranker** to improve precision.
- Select the best chunks for the LLM context.

##### 3. Generation

```txt
User Query + Retrieved Context
              ↓
             LLM
              ↓
        Answer + Citations
```

The LLM generates an answer based primarily on the retrieved evidence, ideally refusing to answer when sufficient evidence is unavailable.

---

#### How to Evaluate RAG

Evaluate **retrieval and generation separately**, then evaluate the complete system.

| Layer | Metrics | What it tells you |
| --- | --- | --- |
| **Retrieval** | Recall@K, Precision@K, MRR, nDCG | Did we retrieve the right documents? |
| **Context** | Context relevance, context precision, context recall | Is the retrieved context useful and sufficient? |
| **Generation** | Faithfulness/groundedness, answer correctness, relevance | Is the answer supported and correct? |
| **End-to-end** | Task success, human/LLM judge, user feedback | Does the complete RAG system solve the user's problem? |
| **Production** | p50/p95 latency, throughput, token usage, cost, errors | Is it efficient and reliable? |

##### Example debugging

If the answer is wrong:

```txt
Wrong Answer
    ↓
Was the correct document retrieved?
    ├── No → Retrieval problem
    │        → embeddings / chunking / query / search
    │
    └── Yes
         ↓
    Was the evidence sufficient and relevant?
         ├── No → Context construction problem
         │
         └── Yes
              ↓
         Did the LLM use the evidence correctly?
              └── No → Generation / prompting problem
```

##### Production evaluation loop

**Golden dataset → Retrieval evaluation → Generation evaluation → Human/LLM evaluation → Production monitoring → Regression tests**

### Q93. Why can naive vector search produce poor results?
> ✍️ _Drafted answer (not from the original notes)._

Naive vector search = embed the query, take top-k by cosine similarity, stuff into the prompt. It often underperforms because:

- **Similarity ≠ relevance:** "how to cancel a subscription" and "how to start a subscription" embed close together but answer different questions.
- **Exact terms are smoothed out:** product codes, error IDs, names, numbers and acronyms are poorly represented, so keyword matches are missed.
- **Query–document asymmetry:** short questions and long passages occupy different regions of the space.
- **Bad chunking:** chunks that split a fact or mix topics give diluted embeddings.
- **Domain mismatch:** general embedding models miss legal, medical or internal jargon.
- **No metadata awareness:** ignores recency, permissions, document type and version, so stale or unauthorised content surfaces.
- **ANN approximation and fixed k:** approximate indexes miss neighbours; a fixed k returns junk when few chunks are relevant and too little when many are.
- **Redundancy:** near-duplicates fill the context with the same information.

**Fixes:** hybrid search (BM25 + dense), cross-encoder reranking, query rewriting/HyDE/multi-query, structure-aware chunking with metadata, domain-adapted embeddings, metadata filters, MMR for diversity, and measuring recall@k on a labelled set.

### Q94. Compare dense retrieval vs sparse retrieval.
#### Dense vs. Sparse vs. Hybrid Retrieval

| Feature | **Dense Retrieval** | **Sparse Retrieval** | **Hybrid Retrieval** |
| --- | --- | --- | --- |
| Representation | Dense embeddings | Sparse term-weight vectors | Dense + sparse |
| Matching | Semantic similarity | Keyword/token matching | Semantic + lexical |
| Good at | Meaning and paraphrases | Exact terms, IDs, names | Both |
| Common methods | Bi-encoder + vector DB | BM25, TF-IDF | BM25 + vector search |
| Weakness | Can miss exact terms | Poor semantic understanding | More complexity |

#### Core Mechanics

- **Dense:** Converts query and documents into embeddings and retrieves based on vector similarity.*Example:* `"How can I reset my password?"` can retrieve `"Steps to recover account credentials"`.
- **Sparse:** Matches important words/tokens between query and documents.*Example:* Searching `"ERR_CONNECTION_REFUSED"` benefits from exact lexical matching.
- **Hybrid:** Runs both retrieval methods and combines their rankings/scores, often using **Reciprocal Rank Fusion (RRF)** or a weighted score.

#### Production RAG

```txt
                ┌─ Dense Retrieval ─┐
User Query ─────┤                   ├─→ Fusion/Reranking → LLM
                └─ Sparse Retrieval ┘
```

**Key idea:**

**Dense = semantic meaning**

**Sparse = exact terminology**

**Hybrid = robust retrieval using both**

### Q95. What is reranking, and why does it improve RAG?
> ✍️ _Drafted answer (not from the original notes)._

**Reranking** is a second-stage scoring step: a cheap first-stage retriever (BM25, dense or hybrid) fetches a broad candidate set (e.g. top 50–200), then a more accurate, more expensive model re-scores each (query, passage) pair and only the top-N go to the LLM.

**Why it improves RAG:**
- Bi-encoders encode query and document independently, so they can't model fine-grained interactions. A **cross-encoder** (or LLM reranker) reads both together with full attention and judges whether the passage actually answers the question.
- It raises **precision at the top**, which matters because context is limited and noisy context causes hallucination and lost-in-the-middle failures. Fewer, better chunks also cut token cost.
- It decouples goals: set first-stage k high for **recall**, rely on the reranker for **precision**.

**Options:** cross-encoders (bge-reranker, ms-marco MiniLM), hosted rerankers (Cohere Rerank), late interaction (ColBERT), LLM listwise reranking.

**Tradeoffs:** extra latency (tens to hundreds of ms per ~100 pairs), cost and infra; mitigate with fewer candidates, batching, distilled rerankers and caching.

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base")
scores = reranker.predict([(query, c.text) for c in candidates])
top = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])][:5]
```

Validate the gain with nDCG@k/MRR and end-to-end answer quality.

### Q96. Compare bi-encoder vs cross-encoder retrieval.
#### Bi-Encoder vs. Cross-Encoder Reranking

| Feature | Bi-Encoder | Cross-Encoder |
| --- | --- | --- |
| **Input** | Query and document encoded independently | Query and document processed together |
| **Output** | Dense embedding vector | Relevance score |
| **Speed** | Fast | Slower |
| **Precomputation** | Document embeddings can be precomputed | Cannot precompute document-query scores |
| **Scalability** | Excellent for large-scale retrieval | Limited for large-scale retrieval |
| **Accuracy** | Good semantic matching | Usually better fine-grained relevance |
| **Typical role** | Initial retrieval | Reranking |

#### Core Mechanics

- **Bi-Encoder:** Encodes the query and documents separately, then compares their embeddings using **cosine similarity, dot product, or another distance metric**. It is optimized for **high recall and scalable retrieval**.
- **Cross-Encoder:** Processes the **query-document pair jointly**, allowing attention between query and document tokens. It produces a relevance score and is optimized for **ranking precision**.

#### Production RAG Pipeline

```txt
User Query
    ↓
Bi-Encoder
    ↓
Top-K Candidates
    ↓
Cross-Encoder Reranker
    ↓
Top-N Relevant Chunks
    ↓
LLM
```

**Key idea:**

**Bi-encoder = retrieve efficiently**

**Cross-encoder = rank accurately**

### Q97. What is contextual compression?
> ✍️ _Drafted answer (not from the original notes)._

**Contextual compression** shrinks retrieved documents to only what is relevant to the current query before they reach the LLM.

**Techniques:**
- **Filtering:** drop chunks below a relevance threshold (reranker score, embedding similarity, LLM yes/no).
- **Extractive compression:** keep only query-relevant sentences/spans.
- **Abstractive compression:** summarise each chunk with respect to the query.
- **Token-level compression:** e.g. LLMLingua removes low-information tokens.
- **Deduplication:** merge near-duplicate content.

**Benefits:** more distinct evidence fits in the window; less noise (fewer hallucinations and lost-in-the-middle failures); lower token cost and latency of the main call.

**Tradeoffs:**
- Extra model calls add latency/cost, sometimes more than they save.
- Risk of dropping a crucial qualifier or exception, hurting faithfulness.
- Abstractive summaries can hallucinate and lose exact wording needed for citations.

**Production tip:** start with cheap filtering and reranking; add LLM compression only when context budgets are tight, and judge it by faithfulness and correctness, not just token savings. LangChain implements this as `ContextualCompressionRetriever` wrapping a base retriever with a compressor.

### Q98. What is the lost-in-the-middle problem?
#### Lost-in-the-Middle

LLMs may use information at the **beginning and end of a long context better than information buried in the middle**.

**Strategies:**

- Put the most relevant context near the **beginning/end**.
- Retrieve fewer, higher-quality chunks.
- **Rerank** retrieved documents before passing them to the LLM.
- Summarize or compress less-important context.
- Split complex queries into smaller retrieval tasks.

#### Long-Context Strategies

For large documents or long conversations:

- **Hierarchical retrieval:** retrieve documents → sections → relevant chunks.
- **Context compression:** remove redundant or irrelevant information.
- **Query-focused retrieval:** retrieve only information relevant to the current question.
- **Chunk strategically:** preserve semantic boundaries rather than using arbitrary sizes.
- **Summarization:** maintain compact summaries of older conversation/history.
- **Long-context model:** use when the task genuinely requires broad context.

### Q99. How would you select an optimal chunk size?
> ✍️ _Drafted answer (not from the original notes)._

There's no universal best size; chunk size is a hyperparameter tuned against retrieval and answer quality.

**Start from data and queries:**
- **Structure:** split on natural boundaries (headings, sections, paragraphs, table rows, functions) rather than fixed character counts.
- **Query type:** factoid lookups favour smaller chunks (~200–400 tokens); explanatory/synthesis questions favour larger (~500–1,000).
- **Embedding limits:** stay well within the model's max input; long inputs dilute embeddings.
- **Context budget:** chunk size × top-k must fit with the prompt and answer.

**Experiment:**
1. Build a labelled set (questions, gold passages, gold answers).
2. Try a grid, e.g. 256/512/1024 tokens with 10–20% overlap, plus a structure-aware splitter.
3. Measure recall@k/MRR, end-to-end faithfulness and correctness, cost and latency.
4. Choose the smallest size that keeps recall high and answers complete.

**Useful patterns:**
- **Small-to-big / parent-document retrieval:** match on small chunks, return the parent section to the LLM.
- **Contextual chunk headers:** prepend document title/section path to each chunk.
- Different strategies per document type (contracts vs FAQs vs code).

Re-tune when the corpus, embedding model or query mix changes.

### Q100. What happens if chunks are too small?
> ✍️ _Drafted answer (not from the original notes)._

**Chunks that are too small** (a sentence or < ~100 tokens):

- **Lost context:** "It must be renewed every 12 months" no longer says what "it" is, so both the embedding and the LLM lack meaning.
- **Fragmented answers:** needed facts are spread over many chunks; top-k may get only some, giving incomplete or wrong answers.
- **Noisier retrieval:** short texts produce less reliable embeddings that match on surface words.
- **Overhead:** many more vectors to store and search; the context fills with tiny, often redundant pieces.
- **Harder reasoning and citation:** the model must stitch fragments together, raising hallucination risk.

**Upside:** very precise matching for factoid lookups with little irrelevant text.

**Mitigations:** overlap, prepend document/section titles, parent-document (small-to-big) retrieval that sends the surrounding section, or larger top-k plus reranking.

### Q101. What happens if chunks are too large?
> ✍️ _Drafted answer (not from the original notes)._

**Chunks that are too large** (whole pages or thousands of tokens):

- **Diluted embeddings:** one vector represents many topics, so similarity to a specific query weakens and the right chunk ranks lower.
- **Noisy context:** lots of irrelevant text distracts the LLM, increasing hallucinations and lost-in-the-middle failures.
- **Less coverage:** only a few chunks fit the context budget, so fewer distinct sources are seen.
- **Higher cost and latency** from more input tokens.
- **Silent truncation** beyond the embedding model's max input length.
- **Coarse citations:** hard to point to the exact supporting passage.

**Upside:** more context per chunk, good for explanatory questions within one section.

**Mitigations:** structure-aware splitting, hierarchical retrieval (retrieve section, then sub-chunk), reranking and contextual compression, and tuning size on an eval set.

### Q102. Explain retrieval precision, retrieval recall, faithfulness, and answer relevance.
> ✍️ _Drafted answer (not from the original notes)._

These four metrics separate RAG quality into **retrieval** and **generation**:

**Retrieval precision** — of the retrieved chunks, how many are relevant?
$\text{Precision@k} = \frac{\#\text{relevant in top-}k}{k}$
Low precision = noisy context, more tokens, more distraction.

**Retrieval recall** — of all relevant chunks, how many were retrieved?
$\text{Recall@k} = \frac{\#\text{relevant in top-}k}{\#\text{relevant total}}$
Low recall = the answer isn't in the context, so the LLM can't answer correctly (or hallucinates). Usually the first thing to fix.

**Faithfulness (groundedness)** — are the answer's claims supported by the retrieved context? Split the answer into claims and check each against the context (NLI model or LLM judge): supported / total. Low faithfulness = hallucination, even if the answer happens to be true.

**Answer relevance** — does the answer address the user's question (not off-topic, evasive or padded)? Measured with an LLM judge, or (RAGAS) by generating questions from the answer and comparing them to the original via embeddings.

**Diagnosis:**
- Low recall → chunking, embeddings, hybrid search, query rewriting.
- High recall, low precision → reranking/filtering.
- Good retrieval, low faithfulness → prompt, grounding or model issue.
- Faithful but irrelevant → prompt or query understanding.

Also track **answer correctness** vs a gold answer. Tools: RAGAS, TruLens, DeepEval — with judges calibrated against human labels.

### Q103. What is GraphRAG, and when is it better than traditional RAG?
|  | **Traditional RAG** | **GraphRAG** |
| --- | --- | --- |
| Knowledge representation | Document chunks + embeddings | **Entities + relationships + graph** |
| Retrieval | Similarity/vector search | Graph traversal + semantic search |
| Best for | Specific factual lookups | Multi-hop and relationship-based questions |
| Context | Individual relevant chunks | Connected entities, relationships, and summaries |
| Complexity | Lower | Higher |
| Storage | Vector DB | Graph DB + often vector index |

**Example:**

Question: *“How did Company A's acquisition of Company B affect its relationship with Supplier C?”*

- **Traditional RAG:** retrieves chunks mentioning the companies.
- **GraphRAG:** can traverse **Company A → acquired → Company B → supplied by → Supplier C** and reason over those relationships.

### Q104. How would you handle tables, PDFs, images, and scanned documents in RAG?
> ✍️ _Drafted answer (not from the original notes)._

Enterprise corpora are messy, so I build a **document-type-aware ingestion pipeline**, not "extract text and chunk".

**Digital PDFs:** layout-aware parsers (Docling, Unstructured, PyMuPDF, pdfplumber, Azure Document Intelligence, AWS Textract) to preserve reading order, headings and lists and strip headers/footers. Chunk by section; keep page numbers for citations.

**Scanned documents:** detect pages with no text layer, pre-process (deskew, denoise), OCR (Tesseract, PaddleOCR, cloud OCR or a VLM). Keep OCR confidence; route low-confidence pages to a stronger model or human review.

**Tables:**
- Extract as structured rows/columns, not flattened text.
- Store as Markdown/HTML for the LLM plus a natural-language summary for embedding.
- Keep small tables whole; chunk large ones by rows, repeating the header.
- For numeric/aggregate questions, load tables into SQL and let an agent query them.

**Images, charts, diagrams:** caption with a VLM and embed the caption (or use multimodal embeddings); store a link to the original so a VLM can inspect it at answer time.

**Cross-cutting:**
- Rich metadata (source, page, section, type, date, ACLs) for filtering and citations.
- Multi-vector indexing: summaries for retrieval, raw content for generation.
- Evaluate per document type — parsing failures are often the biggest source of bad answers.

### Q105. How would you prevent an LLM from answering using its pretrained knowledge instead of retrieved documents?
> ✍️ _Drafted answer (not from the original notes)._

You can't fully stop a model from using its priors, but you can constrain it strongly and detect violations.

- **Prompting:** "Answer only from the context below. If it isn't there, say you don't have enough information." Delimit context clearly, require **citations** (chunk IDs) for each claim, and include few-shot examples of abstaining.
- **Fix retrieval first:** most prior-knowledge answers happen when retrieval misses. Improve recall (hybrid, reranking, query rewriting) and abstain when the top relevance score is below a threshold.
- **Structured output:** e.g. `{"answer": ..., "citations": [...], "answerable": bool}`; reject answers without valid citations.
- **Verification:** an NLI model or LLM judge checks each claim against the cited chunks; unsupported answers are regenerated, trimmed or replaced with a refusal.
- **Model choice/tuning:** pick models that follow grounding instructions well; fine-tune on context-grounded QA including unanswerable examples; use low temperature.
- **Monitor:** track faithfulness on sampled traffic; keep "answer not in docs" cases in the eval set.

**Tradeoff:** stricter grounding raises refusal rates. Tune per use case (strict for legal/finance, looser for general assistants) and route abstentions to search results or a human.

### Q106. Design a real-time fraud detection system.
> ✍️ _Drafted answer (not from the original notes)._

**Requirements:** score card/payment transactions in < 100 ms p99 at thousands of TPS; high recall on fraud with tolerable false-positive rate; explainability and audit for regulators; adapt to evolving fraud patterns.

**Architecture:**
```txt
Payment API → Kafka → Feature service (online store: Redis/Feast)
                    → Scoring service (rules + model) → decision: approve / step-up (3DS/OTP) / decline / review
                    → Event log → offline lake (training, analytics)
```

**Features:**
- Real-time aggregates via streaming (Flink/Spark Streaming): txn count/amount per card in last 1 min/1 h/24 h, new device/IP/merchant flags, geo-velocity ("impossible travel").
- Static/profile: account age, historical spend, merchant risk.
- Graph features: shared devices/cards/emails across accounts (fraud rings).
- Same feature definitions offline and online (feature store) to avoid training-serving skew.

**Models:** a rules engine for known patterns and hard policy; a gradient-boosted model (XGBoost/LightGBM) as the main scorer — fast, strong on tabular data, explainable with SHAP; optionally GNN or sequence models offline, and anomaly detection for novel patterns.

**Key challenges:**
- **Extreme imbalance (~0.1%):** class weights, PR-AUC metrics, cost-sensitive thresholds (cost of fraud vs friction).
- **Label delay:** chargebacks arrive weeks later — use delayed labels plus analyst feedback; train on time-based splits.
- **Adversarial drift:** monitor score distributions and feature drift; retrain frequently; champion/challenger.

**Operations:** shadow-deploy new models, canary, latency budgets with fallback to rules if the model times out, case-management UI for analysts, full audit log of features, score and reason codes per decision.

### Q107. Design a recommendation system for millions of users.
> ✍️ _Drafted answer (not from the original notes)._

**Goal:** personalised top-N items for millions of users and items at low latency (~100 ms), optimising engagement while respecting business rules (diversity, freshness, inventory).

**Multi-stage funnel:**
```txt
Millions of items → Candidate generation (~1,000) → Ranking (~100) → Re-ranking/business rules → Top-N
```

1. **Candidate generation (high recall, cheap):**
   - Two-tower model: user tower and item tower produce embeddings; serve with ANN (FAISS, ScaNN, Milvus).
   - Collaborative filtering / matrix factorisation, item-to-item co-occurrence ("bought together"), trending/popular, content-based for new items.
   - Union multiple sources.
2. **Ranking (high precision):** GBDT or deep model (DLRM, DCN, Wide & Deep) with rich features — user history, item features, context (time, device), cross features; predicts multiple objectives (click, purchase, watch time) combined into a score.
3. **Re-ranking:** diversity (MMR), freshness, deduplication, fairness, sponsored items, filtering already-purchased or out-of-stock items.

**Data and infra:**
- Event streaming (Kafka) for impressions/clicks; feature store for online/offline consistency; batch-precompute embeddings and candidates for heavy users, real-time features for session intent.
- Logging of impressions (not just clicks) for unbiased training.

**Cold start:** content features, popularity by segment, onboarding preferences, exploration (bandits).

**Evaluation:** offline recall@k, NDCG, AUC; but decisions come from **online A/B tests** on business metrics (CTR, conversion, retention). Watch for feedback loops and popularity bias — keep some exploration traffic.

### Q108. Design a large-scale document classification system.
> ✍️ _Drafted answer (not from the original notes)._

**Scenario:** classify millions of documents (emails, contracts, tickets, invoices) into tens to hundreds of classes, possibly multi-label, with high throughput and auditability.

**Pipeline:**
```txt
Ingest (S3/queue) → Parse/OCR → Clean + language detect → Feature/embedding → Classifier → Confidence routing → Store labels + index
```

**Modelling choices (cheapest that works):**
- **Baseline:** TF-IDF + logistic regression/linear SVM — fast, strong, explainable.
- **Fine-tuned encoder** (DeBERTa/BERT/multilingual) for higher accuracy; truncate or chunk long docs and aggregate (max/mean pooling or hierarchical model).
- **Embeddings + lightweight classifier** for many classes and easy retraining.
- **LLMs** zero/few-shot for bootstrapping labels, rare classes or long-tail; distil into a smaller model for volume because LLMs are expensive at millions of docs.
- Hierarchical classification for large taxonomies.

**Labels:** start with weak supervision / LLM labelling, then human review via active learning (label the most uncertain samples).

**Scale and ops:**
- Batch inference on Spark/Ray with GPU workers; streaming path for new docs; ONNX/quantisation for throughput.
- **Confidence thresholds:** auto-accept high confidence, route low confidence to humans; their corrections feed retraining.
- Calibrate probabilities; per-class precision/recall and confusion matrix, macro-F1 for imbalanced classes.
- Monitor class distribution and drift; version models and data; keep audit trail of predictions.

### Q109. Design an enterprise RAG platform.
> ✍️ _Drafted answer (not from the original notes)._

**Goal:** a shared platform letting many teams build grounded Q&A/assistants over internal content (Confluence, SharePoint, Drive, tickets, databases) with security, quality and cost control.

**Components:**
1. **Connectors and ingestion:** scheduled + event-driven sync; parsing (PDF, Office, HTML, OCR); deduplication; structure-aware chunking; metadata (source, owner, timestamps, **ACLs**); incremental re-indexing and deletion propagation.
2. **Indexing:** embeddings (versioned model) in a vector store plus BM25/keyword index; per-tenant or per-domain namespaces.
3. **Retrieval service:** query rewriting, hybrid search, **permission filtering at query time** (user's groups must match document ACLs), cross-encoder reranking, MMR.
4. **Generation service:** model gateway/router (multiple providers, fallback, model routing by complexity), prompt templates under version control, grounded answers with citations, abstention when evidence is weak, streaming.
5. **Guardrails:** PII redaction, prompt-injection defences (treat retrieved content as data), output filtering, rate limiting.
6. **Observability and evaluation:** end-to-end tracing, golden sets per use case, retrieval and faithfulness metrics in CI, sampled online evaluation, user feedback, cost per query dashboards.
7. **Platform APIs/SDK:** self-service onboarding of sources, configurable pipelines per team, chat UI and API.

**Non-functional:** data residency and tenant isolation, audit logs, SSO, latency targets (first token < 1–2 s), caching (embedding, retrieval, semantic/prompt caching), autoscaling.

**Key tradeoffs:** central platform vs team autonomy (offer defaults with overridable pipelines); freshness vs indexing cost; managed vector DB vs self-hosted; stricter grounding vs refusal rate.

### Q110. Design a real-time anomaly detection pipeline.
> ✍️ _Drafted answer (not from the original notes)._

**Use case:** detect anomalies in metrics, logs, transactions or IoT sensor streams within seconds, with low false-alert rates.

**Architecture:**
```txt
Sources → Kafka/Kinesis → Stream processor (Flink/Spark Streaming)
        → Feature windows → Detector(s) → Alert scoring/dedup → Alerting (PagerDuty/Slack) + dashboards
        → Raw + scored events to lake for training/backtesting
```

**Feature engineering on the stream:** sliding/tumbling windows (counts, means, percentiles, rates of change), seasonality-aware baselines (hour-of-day, day-of-week), per-entity state (per device/user/service).

**Detectors (layered, cheap first):**
- **Statistical:** z-score/robust MAD, EWMA, seasonal decomposition (STL), control charts — fast and explainable.
- **ML:** Isolation Forest, one-class models, autoencoders (reconstruction error), forecasting models (Prophet, LSTM) flagging large residuals.
- **Rules** for known critical conditions.
- Models trained offline, scored online; state in a fast store (Redis/RocksDB).

**Reducing noise:** require persistence (N anomalous windows), combine signals, group related alerts, severity scoring, suppression during deploys/maintenance, and feedback buttons so operators label alerts.

**Challenges:** few labels (mostly unsupervised; evaluate on injected anomalies and historical incidents with precision/recall and detection delay), concept drift (seasonality changes, new normal → periodic retraining or adaptive baselines), late/out-of-order events (watermarks), and scale (partition by entity key).

### Q111. Design a personalized search engine.
> ✍️ _Drafted answer (not from the original notes)._

**Goal:** return results ranked by relevance to the query **and** the individual user's preferences, within ~100–200 ms.

**Pipeline:**
```txt
Query → Query understanding → Retrieval (lexical + semantic) → Personalised ranking → Re-ranking → Results
```

1. **Query understanding:** spell correction, tokenisation, synonym expansion, intent and entity detection, category prediction, query embedding.
2. **Candidate retrieval (recall):** BM25 in Elasticsearch/OpenSearch plus dense retrieval via ANN; filters (availability, locale, permissions); a few hundred to a thousand candidates.
3. **Personalised ranking:** learning-to-rank (LambdaMART/GBDT or a neural ranker) with features:
   - Query–document: BM25 score, embedding similarity, field matches.
   - Document: popularity, quality, freshness, price.
   - **User:** embedding from history (clicks, purchases), preferred brands/categories, location, device, session context.
   - User–document cross features (affinity of the user to the item's category).
4. **Re-ranking:** diversity, business rules, de-duplication, fairness.

**Data:** click/purchase logs as implicit labels with **position-bias correction** (inverse propensity weighting, randomised interleaving); feature store for online/offline parity; user profiles updated in near real time.

**Evaluation:** offline NDCG/MRR on held-out logs; online A/B and interleaving tests on CTR, conversion, zero-result rate, time to click.

**Concerns:** cold-start users (fall back to non-personalised ranking), privacy and consent for using history, filter bubbles (keep some exploration), latency budget (precompute user embeddings, cache popular queries).

### Q112. Design a model serving infrastructure supporting 10,000 requests/second.
> ✍️ _Drafted answer (not from the original notes)._

**Targets:** 10k RPS, e.g. p99 < 100 ms, 99.9%+ availability, cost-efficient.

**Architecture:**
```txt
Clients → CDN/edge → API gateway / L7 load balancer (auth, rate limit)
        → Stateless inference service pods (Kubernetes, HPA/KEDA autoscaling)
        → Model server (Triton / TorchServe / vLLM for LLMs) on CPU/GPU nodes
        → Feature store (Redis) + cache · Metrics/tracing · Model registry
```

**Throughput techniques:**
- **Dynamic batching** in the model server (batch requests arriving within a few ms) — the biggest GPU win.
- **Model optimisation:** ONNX Runtime/TensorRT, quantisation (FP16/INT8), distillation, compiled graphs.
- **Caching:** response cache for repeated inputs, precomputed features/embeddings.
- **Async I/O** and gRPC between internal services; connection pooling.
- Right-size hardware: CPU for small tabular models, GPUs for deep models.

**Capacity planning:** benchmark one replica (e.g. 500 RPS at target latency) → ~20 replicas at peak, plus 30–50% headroom and N+1 across availability zones. Autoscale on RPS/queue depth/GPU utilisation, not just CPU.

**Reliability:** health/readiness probes, timeouts, retries with backoff, circuit breakers, load shedding and graceful degradation (fallback to a simpler model or cached/default result), multi-AZ/region.

**Deployment and ops:** model registry and versioned images; canary/blue-green rollout; shadow traffic for new models; monitor latency percentiles, error rate, throughput, saturation, prediction distributions and drift; load-test before launch.

### Q113. How would you handle training-serving skew?
> ✍️ _Drafted answer (not from the original notes)._

**Training-serving skew** is any difference between how data/features look at training time and at inference time, causing a model that looked good offline to underperform in production.

**Common causes:**
- **Different code paths:** features computed in Spark/SQL for training and re-implemented in Java/Python for serving.
- **Time leakage:** training features computed with information not available at prediction time (e.g. daily aggregates including later events).
- **Data freshness:** online features stale or computed over different windows.
- **Preprocessing mismatch:** different tokenizer, scaler, vocabulary or missing-value handling.
- **Distribution changes** between training data and live traffic.

**How to prevent it:**
- **Single source of feature definitions** via a feature store (Feast, Tecton, Databricks) serving the same logic offline and online.
- **Point-in-time correct joins** for training data.
- **Package preprocessing with the model** (sklearn Pipeline, TF Transform, model graph includes tokenisation) so serving can't diverge.
- **Log features at serving time** and train on those logged features ("log and wait").
- Shared libraries and unit tests asserting offline and online feature values match for the same entity/time.

**How to detect it:** compare feature distributions between training data and serving logs (PSI, KS tests), compare online predictions with offline predictions on the same inputs (shadow scoring), and monitor per-feature null rates and ranges. Alert on divergence before it shows up as a business metric drop.

### Q114. How would you version models and datasets?
> ✍️ _Drafted answer (not from the original notes)._

Goal: any prediction must be **reproducible and traceable** to the exact code, data, config and model that produced it.

**Datasets:**
- Store immutable snapshots; never overwrite. Use table formats with time travel (Delta Lake, Iceberg, Hudi) or tools like DVC/lakeFS for files.
- Record dataset version (snapshot ID/commit hash), schema, row counts, statistics and lineage (source tables, transformation code version).
- Version labels and train/val/test split definitions too.

**Models:**
- **Model registry** (MLflow, SageMaker, Vertex, W&B) storing artifact, semantic version, stage (staging/production/archived), metrics and links to training run.
- Each training run logs: git commit, dataset version, hyperparameters, environment (Docker image, library versions), random seeds, evaluation results.
- Package the model with its preprocessing and a signature (input/output schema).

**Prompts and LLM apps:** version prompts, retrieval configs, embedding model and index version together as a deployable unit.

**Serving and audit:** each prediction log contains model version (and feature/data versions where relevant), enabling rollback and debugging.

**Practices:** promotion via CI/CD with automated evaluation gates, approvals for regulated domains, retention policies for old versions, and model cards documenting intended use and limitations.

### Q115. How would you perform online vs offline inference?
> ✍️ _Drafted answer (not from the original notes)._

**Offline (batch) inference:** predictions computed on a schedule for many records at once and stored for later lookup.
- Examples: nightly churn scores, weekly demand forecasts, precomputed recommendations.
- Implementation: Spark/Ray/Beam jobs or warehouse SQL UDFs orchestrated by Airflow/Dagster; results written to a table, cache or key-value store.
- **Pros:** high throughput, cheap (spot instances, large batches), simple, no strict latency requirements, easy to validate before publishing.
- **Cons:** stale predictions; wasted compute on entities never queried; can't react to fresh context.

**Online (real-time) inference:** predictions computed per request, synchronously.
- Examples: fraud scoring at payment time, search ranking, chatbots.
- Implementation: model behind a REST/gRPC service (FastAPI, Triton, KServe), online feature store, autoscaling.
- **Pros:** fresh, uses request context; only computes what's needed.
- **Cons:** latency SLAs, high-availability ops, more expensive per prediction, training-serving skew risk.

**Middle grounds:**
- **Streaming/near-real-time:** score events from Kafka with Flink within seconds; write to a store.
- **Hybrid:** precompute heavy parts in batch (user embeddings, candidates) and do lightweight real-time ranking with fresh features.
- **Async online:** queue requests for heavy models (e.g. document processing) and notify when done.

**Decision drivers:** freshness requirement, latency, request volume and pattern, cost, and how far ahead inputs are known.

### Q116. How would you deploy a model using blue-green or canary deployment?
> ✍️ _Drafted answer (not from the original notes)._

**Blue-green deployment:** run two identical environments. **Blue** serves production with model v1; deploy model v2 to **green**, run smoke tests and validation against it, then switch the router/load balancer to send 100% of traffic to green. Blue remains on standby for instant rollback.
- **Pros:** simple, near-zero downtime, instant rollback.
- **Cons:** double infrastructure during the switch; all users exposed to v2 at once, so issues only visible under real traffic hit everyone.

**Canary deployment:** route a small slice of traffic to v2 (e.g. 1% → 5% → 25% → 100%) while monitoring.
- Compare v2 vs v1 on technical metrics (latency, errors, memory) and model metrics (prediction distribution, confidence, business KPIs such as conversion or fraud catch rate).
- Automated promotion/rollback based on thresholds (Argo Rollouts, Flagger, Istio/Service Mesh traffic splitting, SageMaker/Vertex traffic splitting).
- **Pros:** limits blast radius, real-traffic validation. **Cons:** slower, needs good monitoring and enough traffic for statistical confidence; sticky routing may be needed for consistent user experience.

**ML-specific additions:**
- **Shadow deployment first:** v2 receives mirrored traffic, predictions are logged but not used — validates latency and prediction agreement with zero user risk.
- Beyond canary, use **A/B tests** for business impact with proper statistics.
- Version model + preprocessing + features together; ensure the feature pipeline is compatible with both versions during the rollout; keep v1 warm until v2 is proven.

### Q117. How would you monitor model performance after deployment?
> ✍️ _Drafted answer (not from the original notes)._

Monitoring has to cover the **system**, the **data** and the **model's quality**, because models fail silently.

**1. Operational health:** latency (p50/p95/p99), throughput, error/timeout rates, resource utilisation (CPU/GPU/memory), queue depth, cost.

**2. Data quality and drift:**
- Schema checks, null rates, ranges, category cardinality (Great Expectations, Evidently, WhyLabs).
- Input drift vs training baseline per feature (PSI, KS test, Jensen-Shannon).
- Training-serving skew checks.

**3. Prediction monitoring:** distribution of scores/classes, confidence, rate of positive predictions — changes are an early proxy when labels are delayed.

**4. Model performance with ground truth:** join predictions with labels when they arrive (immediately for clicks, weeks later for chargebacks/churn); track accuracy, precision/recall, AUC, calibration, RMSE by segment (region, device, customer type) to catch localised degradation and fairness issues.

**5. Business KPIs:** conversion, revenue, fraud losses, user feedback — the ultimate measure.

**Process:**
- Log every prediction with model version, features and request ID.
- Dashboards and alerts with sensible thresholds; on-call runbooks.
- Scheduled evaluation on fresh labelled samples; human review for LLM outputs.
- Triggers for retraining or rollback; champion/challenger comparisons.

### Q118. What metrics would you monitor for an ML service?
> ✍️ _Drafted answer (not from the original notes)._

I group them into four layers:

**Service (SRE golden signals):**
- Latency p50/p95/p99 (and time-to-first-token for LLMs)
- Throughput (RPS, tokens/s)
- Error rate (5xx, timeouts, fallbacks triggered)
- Saturation: CPU/GPU utilisation, GPU memory, queue length, autoscaling events
- Availability/SLO burn rate; cost per 1k predictions

**Data/input:**
- Schema violations, missing values, out-of-range values
- Feature drift (PSI/KS per feature), new unseen categories
- Feature freshness/staleness from the feature store
- Input volume anomalies

**Model/output:**
- Prediction distribution, class balance, mean score, confidence/entropy
- Online performance when labels arrive: accuracy, precision/recall, AUC, RMSE, calibration (by segment)
- For LLMs: groundedness/hallucination rate, refusal rate, toxicity, token usage, tool-call failures, user thumbs up/down

**Business:**
- The KPI the model drives: CTR, conversion, fraud loss, approval rate, deflection rate, revenue

Alert on SLO breaches and significant drift; review business metrics alongside model metrics because a model can be "accurate" yet hurt the business (or vice versa). Always slice metrics by model version and key segments.

### Q119. How would you detect data drift automatically?
> ✍️ _Drafted answer (not from the original notes)._

**Approach:** compare the distribution of recent production data against a reference (training data or a stable recent window) on a schedule or streaming, and alert when the difference is significant and meaningful.

**Statistical tests / distances per feature:**
- **Numerical:** Kolmogorov–Smirnov test, Wasserstein distance, PSI (Population Stability Index; rule of thumb: < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant).
- **Categorical:** chi-squared test, PSI, Jensen–Shannon divergence; flag new categories.
- **Multivariate:** train a **domain classifier** to distinguish reference vs production rows — AUC well above 0.5 means drift, and its feature importances show what drifted. Or monitor drift in embedding space (MMD, centroid distance) for text/images.
- **Prediction drift:** distribution of model outputs — a cheap early signal when labels are delayed.
- **Concept drift:** when labels arrive, track performance over time (and detectors like DDM/ADWIN on error rates).

**Implementation:**
```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(train_df["amount"], prod_df["amount"])
```
Tools: Evidently, WhyLabs, NannyML (estimates performance without labels), Arize, SageMaker Model Monitor.

**Practicalities:**
- With large samples, p-values flag tiny, harmless shifts → use effect-size thresholds (PSI, distance) and weight by feature importance.
- Handle seasonality with appropriate reference windows.
- Alert → investigate (pipeline bug vs real-world change) → retrain or fix. Many "drift" alerts are actually upstream data bugs.

### Q120. How would you roll back a model that performs badly in production?
> ✍️ _Drafted answer (not from the original notes)._

**Preparation (before anything goes wrong):**
- Every model version is immutable in a **model registry**, together with its preprocessing, feature definitions and container image.
- Deploy via blue-green/canary so the previous version stays deployed (warm) for a while.
- Routing is configuration-driven (traffic split / alias like `production` → v12), so rollback is a config change, not a rebuild.
- Automated rollback triggers on canary metrics (error rate, latency, prediction distribution, business KPI guardrails).
- A runbook with owners and a decision threshold.

**During the incident:**
1. **Detect and confirm:** alerts on metrics, verify it's the model and not upstream data (a bad feature pipeline will also break the old model).
2. **Roll back:** shift traffic to the last known good version (or to a fallback such as a rules engine/simpler model) — ideally one command or automatic.
3. **Check compatibility:** old model must still match the current feature schema; if the feature pipeline changed too, roll back that as well (version them together).
4. **Communicate** to stakeholders; record the incident.

**After:**
- Root-cause analysis: data drift, bug in training data, offline/online skew, insufficient evaluation coverage.
- Add the failing cases to the evaluation/regression suite and strengthen release gates (shadow testing, longer canary).
- Reprocess or correct decisions made by the bad model if needed (important in finance/healthcare), using prediction logs keyed by model version.

### Q121. How would you reduce inference latency without significantly reducing accuracy?
> ✍️ _Drafted answer (not from the original notes)._

I start by **profiling** where time goes (preprocessing, network, feature fetch, model compute, postprocessing), then apply the cheapest fix with the least accuracy loss.

**Model-level:**
- **Graph optimisation/compilation:** ONNX Runtime, TensorRT, `torch.compile`, operator fusion — often 2–5× with no accuracy loss.
- **Lower precision:** FP16/BF16 (essentially lossless), INT8 quantisation with calibration or QAT (usually < 1% loss).
- **Distillation:** train a smaller student on the teacher's outputs.
- **Pruning** (structured) and efficient architectures (e.g. DistilBERT, MobileNet).
- **Early exit / cascades:** cheap model handles easy cases, expensive one only for hard ones.

**LLM-specific:** KV caching, continuous batching, FlashAttention, PagedAttention (vLLM), speculative decoding, prompt caching, shorter prompts, streaming, model routing to smaller models.

**Serving/system-level:**
- Dynamic batching (latency/throughput tradeoff), warm instances, GPU/accelerator choice.
- Cache results and precompute features/embeddings.
- Reduce network hops (co-locate feature store), use gRPC, async I/O.
- Parallelise independent steps (feature fetch, retrieval).

**Process:** define latency budget (p99), benchmark on representative inputs, verify accuracy on the eval set after each optimisation, and roll out with canary. The usual order of payoff: system fixes and compilation → FP16 → INT8 → distillation.

### Q122. Training accuracy is 99%, validation accuracy is 60%. How do you debug it?
> ✍️ _Drafted answer (not from the original notes)._

A 99% vs 60% gap is classic **overfitting**, but I first rule out data problems before adding regularisation.

**1. Check the data split:**
- Is validation from the same distribution? (different time period, source, class balance)
- Are there duplicates or near-duplicates inside train (memorisation) — or is validation genuinely harder?
- Leakage in train only (e.g. ID-like feature that encodes the label in train but not val)?
- Is validation preprocessed the same way (same scaler/tokenizer, no fitting on val separately)?

**2. Check the baseline:** what does a simple model get? If logistic regression gets 58%, the task/features may just be weak.

**3. If it is overfitting (high variance):**
- More data or data augmentation.
- Reduce capacity: smaller model, shallower trees, fewer features.
- Regularisation: L1/L2/weight decay, dropout, max depth/min samples per leaf, early stopping.
- Feature selection — remove noisy, high-cardinality or ID features.
- Cross-validation to get a reliable estimate; ensembling (bagging).

**4. Diagnostics:**
- Learning curves (train/val vs data size and epochs).
- Per-class/segment validation errors and confusion matrix.
- Feature importance — a dominant suspicious feature often indicates leakage or spurious correlation.

**5. Label quality:** noisy labels are memorised in training but can't generalise.

Fix one thing at a time and track experiments so the cause is clear.

### Q123. Training and validation accuracy are both 50%. What could be wrong?
> ✍️ _Drafted answer (not from the original notes)._

Both at 50% (for a binary balanced task) means the model is performing at **chance** — it's learning nothing. Underfitting is possible, but this usually signals a bug.

**Data/label problems:**
- Labels shuffled or misaligned with inputs (e.g. after a join or sort, or data loader shuffling X and y separately).
- Wrong label encoding, or all labels the same in a batch due to unshuffled, sorted data.
- Features carry no signal (wrong columns, all constants/NaNs filled with zeros, normalisation bug).
- Target is genuinely unpredictable from the given features.

**Training/pipeline bugs:**
- Learning rate too high (diverges) or too low (no movement); loss not decreasing at all.
- Wrong loss for the output (e.g. softmax + `BCELoss`, logits passed to a loss expecting probabilities), wrong output activation.
- Gradients not flowing: frozen layers, forgot `optimizer.step()` / `zero_grad()`, dead ReLUs, vanishing gradients.
- Evaluation bug: comparing predictions with wrong labels, or `argmax` on the wrong axis.

**Model capacity:** model too simple (linear model on nonlinear XOR-like data) → genuine underfitting.

**Debugging steps:**
1. Inspect samples and label distribution manually.
2. Try a trivial baseline (majority class, logistic regression).
3. **Overfit a tiny batch** (10–100 samples): a correct pipeline should reach ~100% train accuracy. If it can't, there's a bug.
4. Check loss curves and gradient norms.
5. Then increase capacity, train longer, or engineer better features.

### Q124. Your model performs well offline but poorly in production. Why?
> ✍️ _Drafted answer (not from the original notes)._

Good offline, bad in production almost always means **the production data or setup differs from the offline evaluation**.

**Common causes:**
- **Training-serving skew:** features computed differently online (different code, units, defaults, stale feature store values, missing-value handling).
- **Data leakage offline:** features available in historical data that aren't available at prediction time (future information, post-event fields), inflating offline metrics.
- **Wrong validation scheme:** random split on time-dependent data instead of a time-based split; duplicates across train/test; tuning on the test set.
- **Distribution shift:** production users, seasons, products or behaviour differ from training data (covariate shift, label shift, concept drift).
- **Feedback loops:** the model's decisions change future data (e.g. recommendations shape clicks, fraud blocks change attacker behaviour).
- **Metric mismatch:** offline metric (AUC) doesn't reflect the business objective or the chosen threshold; class priors differ.
- **Engineering bugs:** different model version, preprocessing missing, timeouts causing fallbacks, latency forcing truncated inputs.

**How to investigate:**
1. Log production inputs and predictions; recompute offline predictions on those exact inputs and compare.
2. Compare feature distributions train vs production (PSI/KS).
3. Re-evaluate with a time-based backtest that mimics production.
4. Segment errors by user group, time, channel.

**Prevention:** feature store, point-in-time joins, shadow deployments before launch, monitoring and canary rollouts.

### Q125. Your model's accuracy decreases after adding more training data. Explain possible reasons.
> ✍️ _Drafted answer (not from the original notes)._

Normally more data helps, so a drop is a signal to inspect the new data and the evaluation:

- **Lower-quality or noisy labels** in the new data (different labelling team, weak/automatic labels).
- **Different distribution:** new data comes from another source, region, time period or device, so the model now fits a mixture that matches the validation set less well. Or the validation set is from the old distribution and the new data shifted the model.
- **Class imbalance changes:** the new data changes class priors, shifting the decision threshold.
- **Duplicates / leakage:** test examples (or near-duplicates) previously in training removed, or new data leaks into the validation set — or a re-split moved hard examples into validation, making scores incomparable.
- **Preprocessing inconsistency:** new data encoded differently (units, categories, text normalisation, missing values).
- **Concept drift:** the relationship between features and label changed over time; old and new data conflict.
- **Model/training not adjusted:** fixed number of epochs or learning-rate schedule now under-trains; regularisation tuned for smaller data; capacity too low to fit more diverse data.
- **Random variance:** the difference is within noise — check with cross-validation / multiple seeds and confidence intervals.

**How to diagnose:** evaluate on a fixed, unchanged test set; train on old vs new data separately and compare; check label agreement on a sample of new data; compare distributions; check learning curves. Then clean or reweight the new data, or train with domain indicators.

### Q126. A feature has extremely high importance. How do you determine whether it is legitimate or leakage?
> ✍️ _Drafted answer (not from the original notes)._

An extremely important feature is either a genuine strong signal or — very often — **target leakage**. I check:

**1. Timing / availability:** Is the feature's value known **at prediction time** in production? Trace its lineage. Examples of leakage: "account_closed_date" when predicting churn, "number of chargebacks" when predicting fraud, a status field updated after the outcome.

**2. Domain sense-check:** does the relationship make causal or business sense? Ask domain experts. A support-ticket count that predicts churn may be legit; a "refund_issued" flag predicting a complaint is probably leakage.

**3. Too-good-to-be-true metrics:** near-perfect AUC (e.g. > 0.99) on a hard problem is a red flag. Train a model on **only that feature** — if it alone almost perfectly predicts the label, investigate.

**4. Time-based validation:** use point-in-time correct feature snapshots and an out-of-time test set. If performance collapses when features are computed as of the prediction timestamp, it was leakage.

**5. Proxy / ID leakage:** IDs, row order, timestamps, file paths or data-source flags that correlate with the label because of how the data was collected (e.g. all positives came from one system).

**6. Inspect the relationship:** partial dependence/SHAP plots; look for perfect thresholds or values only present for one class (e.g. missing for all negatives).

**7. Compare with production:** shadow-score live data — leaked features often have different distributions or are missing online.

If legitimate, keep it but monitor it closely and consider robustness (the model shouldn't fail entirely if that feature breaks).

### Q127. Your validation score varies significantly between folds. What does that indicate?
> ✍️ _Drafted answer (not from the original notes)._

High variance across folds means your performance estimate is **unstable** — the score depends heavily on *which* rows land in validation. Typical causes:

- **Small dataset / small folds:** each fold has too few examples (or too few positives) for a stable metric.
- **High-variance model:** a deep tree or overfit model reacts strongly to the particular training split.
- **Heterogeneous data:** subgroups (regions, customers, time periods) are unevenly distributed across folds.
- **Leakage or grouping problems:** the same user/patient appears in train and validation in some folds but not others.
- **Class imbalance without stratification:** some folds get almost no minority examples.
- **Temporal drift:** if data is time-ordered, some folds are "easier" periods than others.

**What to do**

1. Report mean **and** std (or a confidence interval) rather than a single number; use repeated k-fold to tighten the estimate.
2. Use `StratifiedKFold` for imbalance, `GroupKFold` for grouped entities, `TimeSeriesSplit` for temporal data.
3. Inspect the worst fold: what is different about its data?
4. Reduce model variance (regularization, ensembling, simpler model) if the instability is model-driven.

**Production angle:** large fold variance is a warning that production performance will also swing as the input mix changes — treat it as a robustness problem, not just an evaluation nuisance.

### Q128. Your model predicts only the majority class. How do you fix it?
> ✍️ _Drafted answer (not from the original notes)._

Predicting only the majority class is the classic symptom of **severe class imbalance** combined with an objective (or metric) that rewards it: with 99% negatives, "always negative" gives 99% accuracy and a low unweighted loss.

**Diagnose first**
- Check the class distribution and the confusion matrix; switch from accuracy to **precision/recall, F1, PR-AUC**.
- Inspect predicted probabilities — often the model *does* rank positives higher, but the default 0.5 threshold hides it.
- Rule out bugs: label encoding errors, features that carry no signal, a broken pipeline feeding constant inputs, or a learning rate so high the model collapsed to the prior.

**Fixes**
1. **Threshold tuning** on a validation set (optimize F1 or a cost-weighted metric) — cheapest and often enough.
2. **Class weights / cost-sensitive loss** (`class_weight='balanced'`, `scale_pos_weight` in XGBoost, weighted BCE, focal loss).
3. **Resampling:** undersample the majority, oversample the minority, or SMOTE — applied only to the training folds.
4. **Better features** if the model genuinely cannot separate the classes.
5. Reframe as **anomaly detection** when positives are extremely rare.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr, y_tr)
probs = clf.predict_proba(X_val)[:, 1]
# choose the threshold that maximizes F1 on validation instead of 0.5
```

**Production angle:** pick the threshold from the business cost of FP vs FN, and recalibrate as the base rate drifts.

### Q129. Your embeddings retrieve semantically similar but factually incorrect documents. How do you improve retrieval?
> ✍️ _Drafted answer (not from the original notes)._

This is the gap between **semantic similarity** and **factual relevance**: embeddings place "Q3 revenue rose 5%" near "Q3 revenue fell 5%", and an old policy near the current one. Fixes, roughly in order of payoff:

1. **Hybrid retrieval:** add BM25/sparse search so exact entities, numbers, IDs and dates count; fuse with RRF.
2. **Cross-encoder reranking:** rerank the top 50–100 candidates with a model that reads query and passage together and can catch negation, wrong entity or wrong period.
3. **Metadata filtering:** filter by date, version, product, jurisdiction or tenant before (or during) vector search; mark superseded documents and drop them.
4. **Better chunking:** keep the claim and its qualifiers (entity, period, units) in the same chunk; prepend titles/section headers ("contextual chunk headers").
5. **Domain-adapted embeddings:** fine-tune the embedding model with contrastive pairs, including **hard negatives** that look similar but are factually wrong.
6. **Query rewriting:** extract entities and constraints from the query and turn them into filters or keyword terms.
7. **Data hygiene:** deduplicate and remove stale or contradictory sources — retrieval cannot beat a corpus that contains wrong facts.

**Measure it:** build a labeled set with gold passages and track recall@k and precision@k (or nDCG) separately from answer quality, including adversarial "near miss" queries. Downstream, ask the generator to cite passages and verify that claims are supported.

### Q130. Your RAG system retrieves the correct document but the LLM gives the wrong answer. Where is the problem?
> ✍️ _Drafted answer (not from the original notes)._

If retrieval is correct, the failure is in **context construction or generation**. Debug in that order:

**1. Did the right passage actually reach the prompt?**
- The document was retrieved, but the relevant chunk may have been cut by the reranker, truncated by the token budget, or split across chunk boundaries.
- Log the exact final prompt — not just the retriever output.

**2. Is the context noisy or contradictory?**
- Other retrieved chunks may contradict the right one or distract the model; the key passage may be buried in the middle (*lost in the middle*).
- Fix: fewer, reranked chunks; put the best evidence first; deduplicate.

**3. Is the prompt telling the model to ground?**
- Without explicit instructions the model blends parametric knowledge with the context. Require "answer only from the context, cite sources, say 'I don't know' otherwise".

**4. Is the model capable of the reasoning?**
- Multi-hop reasoning, arithmetic over tables, or reading complex layouts (parsed PDFs) can exceed a small model. Try a stronger model, chain-of-thought, or tool use (a calculator, SQL).

**5. Parsing quality:** tables or numbers mangled during ingestion look "retrieved" but are unreadable.

**6. Decoding settings:** high temperature, or truncated outputs because of max tokens.

**How to tell:** measure **faithfulness/groundedness** (is the answer supported by the given context?) separately from **context recall**. High context recall plus low faithfulness pins the problem on generation. Use traces (Langfuse/LangSmith) to inspect failing cases and add them to a regression set.

### Q131. The LLM answers correctly with 5 retrieved documents but incorrectly with 20. Why?
> ✍️ _Drafted answer (not from the original notes)._

More context is not free. Likely reasons:

- **Distractors and noise:** documents 6–20 are less relevant; some are near-misses or contradict the right answer, and the model gets pulled toward them.
- **Lost in the middle:** LLMs attend best to the start and end of long contexts; the correct passage may now sit in the middle.
- **Attention dilution:** relevant evidence is a smaller fraction of the prompt, so the signal-to-noise ratio drops.
- **Conflicting information:** outdated versions or different entities introduce contradictions the model resolves wrongly.
- **Context-length limits:** the prompt may be truncated (silently, by the framework or the model limit), cutting the instructions or the key chunk.
- **Instruction drift:** long contexts push system instructions far from the question, weakening grounding behavior.

**Fixes**
- Retrieve more broadly (top-50) but **rerank** and pass only the top 3–8 to the model.
- Order chunks by relevance, putting the best near the start (or the end, next to the question).
- Deduplicate and filter by metadata (date, version).
- Use contextual compression: extract only the relevant sentences.
- Repeat the key instructions after the context.
- Tune *k* empirically: plot answer accuracy and faithfulness against *k* on an eval set — the curve usually peaks and then declines.

**Takeaway:** optimize for **precision of context**, not volume; retrieval recall matters, but the generator needs a short, clean, well-ordered context.

### Q132. Why does overparameterization not necessarily cause worse generalization?
> ✍️ _Drafted answer (not from the original notes)._

Classical theory (VC dimension, bias-variance) says more parameters than data points should overfit. Modern deep nets with far more parameters than examples still generalize well, because **parameter count is a poor measure of effective capacity**:

- **Implicit regularization:** among the infinitely many solutions that fit the training data, gradient descent tends to find "simple" ones (low norm, low complexity, flat minima).
- **Interpolation can be benign:** in high dimensions, a model can fit noisy labels with tiny, spread-out components that barely affect predictions on new inputs ("benign overfitting").
- **Double descent:** test error rises near the interpolation threshold (#params ≈ #samples) and then *falls again* as the model grows, because larger models can pick smoother interpolating solutions.
- **Architectural inductive biases:** convolutions, attention, weight sharing and normalization constrain the hypothesis space to functions that suit the data.
- **Explicit regularization and training tricks:** weight decay, data augmentation, dropout, early stopping.
- **Optimization benefits:** overparameterized loss landscapes have more connected, flatter minima, which makes good solutions easier to reach.

**Caveat:** overparameterization does not *guarantee* generalization — with little data, label noise and no regularization, big models still overfit. Practically: scale model and data together, keep validation monitoring, and rely on weight decay and augmentation.

### Q133. Explain the double-descent phenomenon.
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

### Q134. What is the implicit bias of gradient descent?
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

### Q135. Why can SGD find solutions that generalize despite having enormous parameter counts?
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

### Q136. What is the relationship between model scaling and emergent capabilities?
> ✍️ _Drafted answer (not from the original notes)._

**Emergent capabilities** are abilities that appear to be absent in smaller models and show up abruptly once models pass a certain scale (parameters, data, compute) — for example multi-step arithmetic, chain-of-thought reasoning, or in-context learning of new tasks (Wei et al., 2022).

**Relationship to scaling**
- **Loss scales smoothly:** pretraining loss decreases predictably as a power law in compute (the scaling laws).
- **Task metrics can look discontinuous:** a smooth gain in per-token accuracy can turn into a sharp jump on an all-or-nothing metric. If a 5-step problem needs every step right, accuracy ≈ $p^5$, which stays near zero and then rises quickly as $p$ improves.
- **The "mirage" critique (Schaeffer et al., 2023):** much apparent emergence comes from discontinuous metrics (exact match) and coarse model sampling; with continuous metrics (log-likelihood, partial credit), improvement often looks gradual and predictable.

**Practical view**
- Some capabilities genuinely need a threshold of competence across several sub-skills before the full task works.
- Capability gains also come from data quality, instruction tuning and RLHF, not just size.
- For planning: forecast with continuous proxy metrics on small models; don't assume a capability will appear at a given scale, and don't assume it won't.
- For safety: unpredictable capability jumps are why labs run dangerous-capability evals at each scale-up.

### Q137. Explain the scaling laws for LLMs.
> ✍️ _Drafted answer (not from the original notes)._

**Scaling laws** are empirical power-law relationships between a language model's pretraining loss and the resources used to train it.

**Kaplan et al. (OpenAI, 2020)**
$L(N) \approx (N_c/N)^{\alpha_N}$, $L(D) \approx (D_c/D)^{\alpha_D}$, $L(C) \approx (C_c/C)^{\alpha_C}$
where $N$ = non-embedding parameters, $D$ = tokens, $C$ = compute. Key findings:
- Loss falls smoothly as a power law over many orders of magnitude when the other factors aren't bottlenecks.
- Architecture details (depth vs width) matter much less than scale.
- They recommended putting most extra compute into **bigger models** (trained on relatively fewer tokens).

**Chinchilla (Hoffmann et al., DeepMind, 2022)** revised this: for a fixed compute budget $C \approx 6ND$, parameters and tokens should grow **roughly equally**, at about **20 tokens per parameter**. Many earlier models (GPT-3, Gopher) were undertrained.

**Common fitted form**
$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$ with $E$ the irreducible loss.

**How they're used**
- Predict the loss of a large run from small pilot runs; choose model size and token count for a budget.
- Since inference cost matters, modern practice **over-trains** smaller models far beyond 20 tokens/param (LLaMA 3 8B: ~15T tokens) to make serving cheaper.

**Limits:** they predict loss, not specific downstream abilities; they assume fixed data quality; and the supply of high-quality data is becoming the bottleneck (driving synthetic data and test-time compute scaling).

### Q138. What is the Chinchilla scaling principle?
> ✍️ _Drafted answer (not from the original notes)._

The **Chinchilla principle** (Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models*): for a fixed training compute budget, **model size and number of training tokens should be scaled in equal proportion**.

- Training compute ≈ $C \approx 6 N D$ (N = parameters, D = tokens).
- Compute-optimal: $N_{opt} \propto C^{0.5}$, $D_{opt} \propto C^{0.5}$.
- Rule of thumb: **~20 training tokens per parameter**.

**Evidence:** Chinchilla (70B params, 1.4T tokens) used the same compute as Gopher (280B params, 300B tokens) and outperformed it across most benchmarks — and was 4× cheaper to serve. It showed that GPT-3-era models were heavily **undertrained**, following Kaplan et al.'s advice to favor parameters over data.

**Nuances**
- "Compute-optimal" minimizes **training** compute for a target loss; it ignores inference cost. If a model will serve billions of requests, it is cheaper overall to train a **smaller model on far more tokens** (LLaMA, Mistral and Phi-style models go to 100–2000 tokens/param).
- It assumes unlimited, unique, high-quality data; data quality and repetition change the optimum.
- The exact coefficients depend on the data, tokenizer and architecture; re-fit them with pilot runs before a big training run.

**Interview one-liner:** "Chinchilla says double the data whenever you double the parameters — about 20 tokens per parameter — but production teams deliberately over-train smaller models because inference dominates lifetime cost."

### Q139. Why can a smaller model outperform a larger model?
> ✍️ _Drafted answer (not from the original notes)._

Parameter count is only one driver of quality. A smaller model can win because of:

- **Training data and tokens:** a smaller model trained on far more (or better) tokens beats an undertrained large one — Chinchilla 70B beat Gopher 280B; LLaMA-13B beat GPT-3 175B on many benchmarks.
- **Data quality:** curated, deduplicated, filtered or synthetic "textbook-quality" data (the Phi series) can be worth more than scale.
- **Task specialization:** a 7B model fine-tuned on a narrow domain (classification, extraction, SQL for one schema) often beats a general frontier model on that task.
- **Distillation:** learning from a stronger teacher's outputs/logits transfers capability into a compact student.
- **Post-training:** better instruction tuning, RLHF/DPO and reasoning RL can matter more than base size.
- **Architecture and context use:** better attention variants, longer effective context, and retrieval (RAG) give a small model access to knowledge it doesn't store.
- **Inference strategy:** test-time compute (self-consistency, verification, reasoning traces) lets a small model "think longer".
- **Evaluation artifacts:** benchmark contamination or metric quirks can also make a small model look better than it is — always validate on your own held-out data.

**Production angle:** smaller models bring lower latency and cost, on-prem deployment, and easier fine-tuning. A standard pattern is a **router**: a small specialized model for the bulk of traffic, escalating hard cases to a large model.

### Q140. What is mixture-of-experts (MoE)?
> ✍️ _Drafted answer (not from the original notes)._

A **Mixture-of-Experts (MoE)** layer replaces a single dense feed-forward block with **many parallel "expert" FFNs plus a learned router (gating network)** that sends each token to only a few experts.

```txt
token h ─► router: softmax(W_g · h) ─► top-k experts (e.g., 2 of 8)
                                        │
           y = Σ_{i∈topk} g_i(h) · Expert_i(h)
```

**Key properties**
- **Sparse activation:** each token uses only *k* experts, so compute per token ≈ that of a much smaller dense model, while total parameters (capacity/knowledge) are much larger.
- In Transformers, MoE usually replaces the FFN in some or all layers; attention stays dense.
- Examples: Switch Transformer (top-1), GShard, Mixtral 8×7B (top-2: ~47B total, ~13B active), DeepSeek-V3 (fine-grained + shared experts), and reportedly several frontier models.

**Benefits:** better quality per training FLOP; faster training to a target loss; experts can specialize (often by token type or syntax rather than clean "topics").

**Costs and challenges**
- **Memory:** all experts must be stored, so VRAM needs follow *total* params, not active ones.
- **Load balancing:** routers collapse onto a few experts without auxiliary balancing losses or capacity limits.
- **Communication:** expert parallelism needs all-to-all exchanges across GPUs.
- **Training instability and harder fine-tuning;** batching at inference is less efficient when tokens scatter across experts.

### Q141. Why does MoE allow larger parameter counts without proportionally increasing compute?
> ✍️ _Drafted answer (not from the original notes)._

Because **compute scales with the parameters a token actually touches, not with the parameters that exist**.

- In a dense Transformer, every token passes through every weight: FLOPs per token ≈ $2N$.
- In an MoE layer with $E$ experts and top-$k$ routing, each token passes through only $k$ experts. FFN FLOPs per token scale with $k$, not $E$. Total parameters grow roughly linearly with $E$, while active parameters stay about constant.

**Example — Mixtral 8×7B:** ~47B total parameters, but with top-2 routing only ~13B are **active** per token, so inference compute is roughly that of a 13B dense model while capacity is much larger. DeepSeek-V3: 671B total, ~37B active.

**Why extra parameters still help:** more experts = more total capacity to store knowledge and specialized features; the router conditionally picks the relevant subset per token (*conditional computation*).

**What does NOT scale down**
- **Memory:** all experts must sit in GPU memory (or be offloaded), so serving needs VRAM for total params.
- **Communication:** expert parallelism adds all-to-all traffic between devices.
- **Router overhead** (small), plus wasted capacity from load imbalance or dropped tokens.
- At small batch sizes decoding is memory-bandwidth bound, and touching different experts per token reduces weight reuse.

**Summary:** MoE decouples **capacity** (total params) from **cost per token** (active params); the price is memory footprint and systems complexity.

### Q142. Explain expert routing and load balancing in MoE.
> ✍️ _Drafted answer (not from the original notes)._

**Routing:** a gating network computes scores for each token, $g = \text{softmax}(W_g h)$, and the token is sent to the **top-k** experts (k=1 in Switch, k=2 in Mixtral). The outputs are combined with the (renormalized) gate weights: $y = \sum_{i \in \text{topk}} g_i E_i(h)$. Variants: noisy top-k (adds noise for exploration), expert-choice routing (experts pick tokens, which balances by construction), and fine-grained plus **shared experts** (DeepSeek), where a few experts process every token.

**Why load balancing is needed:** routers are self-reinforcing — popular experts get more gradient, get better and attract even more tokens. Results: **expert collapse** (a few experts do everything, others are wasted), GPU hotspots, and dropped tokens.

**Balancing techniques**
- **Auxiliary load-balancing loss** (Switch): $L_{aux} = \alpha \cdot E \sum_i f_i P_i$, where $f_i$ = fraction of tokens routed to expert *i* and $P_i$ = mean router probability for *i*; minimized when routing is uniform.
- **Router z-loss:** penalizes large router logits for numerical stability.
- **Capacity factor:** each expert accepts at most $C = \text{cf} \cdot \frac{\text{tokens}}{E}$ tokens per batch; overflow tokens are dropped (they pass through the residual) or rerouted. A higher factor drops fewer tokens but costs memory and padding.
- **Auxiliary-loss-free balancing** (DeepSeek-V3): a per-expert bias added to routing scores is adjusted dynamically to even out load, avoiding interference from the extra loss.
- **Expert-choice routing:** guarantees an even load.

**Production concerns:** monitor per-expert utilization and token-drop rate; at inference, balance experts across GPUs (expert parallelism) and replicate hot experts.

### Q143. What is speculative decoding?
> ✍️ _Drafted answer (not from the original notes)._

**Speculative decoding** speeds up autoregressive generation without changing the output distribution. A small, fast **draft model** proposes several tokens ahead; the large **target model** checks them all in **one forward pass** and accepts the longest correct prefix.

**Algorithm (per step)**
1. The draft model generates *k* candidate tokens $x_1..x_k$ autoregressively (cheap).
2. The target model runs once over the prefix plus all *k* candidates, producing its distributions at every position in parallel.
3. Each token is accepted with probability $\min(1, p_{target}(x_i)/p_{draft}(x_i))$; at the first rejection, a replacement is sampled from the normalized residual $\max(0, p_{target} - p_{draft})$ and the rest are discarded.
4. If all *k* are accepted, the target's next-token distribution gives a bonus token for free.

This rejection-sampling scheme guarantees the output is **distributed exactly as if sampled from the target model alone** (with greedy decoding: identical output).

**Variants**
- **Self-speculative / Medusa:** extra decoding heads on the target model propose tokens; no separate draft model.
- **EAGLE:** drafts at the feature level for higher acceptance rates.
- **Lookahead decoding / n-gram (prompt lookup):** draft tokens are copied from the prompt — great for summarization, code edits and RAG.

**Requirements:** the draft must share the tokenizer and agree with the target often (acceptance rate α typically 0.6–0.8). Supported in vLLM, TensorRT-LLM, SGLang and TGI.

### Q144. How can speculative decoding reduce LLM inference latency?
> ✍️ _Drafted answer (not from the original notes)._

Decoding in a large LLM is **memory-bandwidth bound**: each new token requires streaming all the weights (and the KV cache) from HBM to compute just one token, so the GPU's arithmetic units sit mostly idle. Speculative decoding converts that idle compute into extra tokens.

**Why it's faster**
- Verifying *k* draft tokens in one target forward pass costs about the same wall-clock time as generating **one** token (the weights are read once; the extra FLOPs are nearly free when memory bound).
- If on average $\bar{a}$ draft tokens are accepted, each expensive target pass yields $\bar{a}+1$ tokens instead of 1.
- Expected tokens per target pass with acceptance rate α and draft length k: $\frac{1-\alpha^{k+1}}{1-\alpha}$. With α = 0.8 and k = 4 that is ~3.4 tokens per pass → typical **2–3× lower latency**.
- The output is unchanged (exact under rejection sampling), so there is no quality trade-off.

**When it helps / hurts**
- Best at **low batch sizes / latency-sensitive** serving, where the GPU has spare compute.
- Gains shrink at high batch sizes (the GPU is already compute-bound) and with low acceptance rates (creative, high-temperature sampling, or a mismatched draft model).
- Costs: memory and compute for the draft model, and engineering complexity in batching and KV-cache management.

**Production tuning:** choose a draft model ~10–50× smaller from the same family; tune *k* (3–8); monitor acceptance rate per workload; use n-gram/prompt-lookup drafting for input-grounded tasks (RAG, code editing), where acceptance is very high.

### Q145. What is continuous batching?
> ✍️ _Drafted answer (not from the original notes)._

**Continuous (in-flight / iteration-level) batching** schedules LLM requests **per decoding step** instead of per batch.

**Problem with static batching:** a batch of requests starts together and the batch is held until the *longest* sequence finishes. Short requests sit idle, finished slots are padding, new requests wait in the queue → poor GPU utilization and high latency.

**Continuous batching**
- After every forward step (one token), the scheduler **removes finished sequences and inserts new requests** into the free slots.
- The batch composition changes every iteration, so the GPU always works on a full batch of *active* sequences.
- New requests' **prefill** can be interleaved or chunked alongside other requests' **decode** steps (chunked prefill) to avoid latency spikes.

**Enablers**
- **PagedAttention (vLLM):** the KV cache is allocated in fixed-size blocks, so sequences of varying lengths can join and leave without memory fragmentation.
- A scheduler with policies for admission, preemption (swap or recompute KV when memory runs out) and priorities.

**Impact:** throughput gains of up to ~10–20× over naive static batching (per the Orca/vLLM papers), lower queueing latency, and better time-to-first-token for short requests.

**Trade-offs:** larger batches raise throughput but also per-token latency (TPOT); tune max batch size and max batched tokens against your SLOs (p95 TTFT, tokens/sec). Implemented in vLLM, TGI, TensorRT-LLM and SGLang.

### Q146. Why is GPU memory often the bottleneck in LLM inference?
> ✍️ _Drafted answer (not from the original notes)._

Two things must live in GPU memory (HBM) during inference, and both are large:

**1. Model weights**
- Memory ≈ params × bytes per param: a 70B model needs ~140 GB in FP16 — more than one 80 GB GPU — so it needs tensor parallelism or quantization.

**2. KV cache** — grows with batch size × sequence length:
$\text{KV bytes} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{bytes}$
- Llama-2-70B (80 layers, 8 KV heads with GQA, d=128, FP16): ~0.33 MB per token → a 4k context is ~1.3 GB per sequence. Thirty concurrent requests consume tens of GB.
- The KV cache limits **how many concurrent requests (batch size)** you can serve, which directly caps throughput.

**3. Bandwidth, not just capacity**
- During decode, each token requires reading **all weights plus the KV cache** from HBM, while doing only ~2 FLOPs per parameter. Arithmetic intensity is tiny, so token speed ≈ memory bandwidth / bytes read. An H100 has ~3.35 TB/s HBM but ~1,000 TFLOPS of BF16 compute — compute sits idle.

**Mitigations**
- **Quantization** (INT8/INT4/FP8) of weights and the KV cache.
- **GQA/MQA** to shrink the KV cache; **PagedAttention** to avoid fragmentation; prefix caching to share common prompts.
- **Batching** to amortize weight reads over many sequences; **speculative decoding** to get more tokens per weight read.
- Tensor/pipeline parallelism across GPUs; KV offloading for long contexts.

### Q147. Compare compute-bound vs memory-bandwidth-bound workloads.
> ✍️ _Drafted answer (not from the original notes)._

The **roofline model** classifies a workload by its **arithmetic intensity** = FLOPs performed per byte moved from memory.

| | Compute-bound | Memory-bandwidth-bound |
| --- | --- | --- |
| Bottleneck | ALU / tensor-core throughput | Moving data from HBM/DRAM |
| Arithmetic intensity | High (above the hardware's ridge point) | Low |
| Examples | Large matmuls, training with big batches, **LLM prefill**, convolutions | **LLM decode at small batch**, elementwise ops, LayerNorm, softmax, embedding lookups |
| Faster if you… | add FLOPS, use lower precision math (FP8 tensor cores), better kernels | reduce bytes (quantization, fusion), increase reuse (batching), more bandwidth (HBM3) |

**Ridge point:** peak FLOPS ÷ bandwidth. An H100 has ~990 TFLOPS BF16 and ~3.35 TB/s → ~300 FLOPs/byte. Below that, you are memory bound.

**LLM example**
- **Prefill** processes all prompt tokens at once: matrix-matrix products with high intensity → compute-bound.
- **Decode** at batch 1 is a matrix-vector product: each weight (2 bytes) is used for ~2 FLOPs → intensity ≈ 1 → heavily memory-bound. Increasing the batch raises intensity linearly until it hits the compute roof (often at batch ~100–300).

**Practical implications**
- Optimizations must match the bottleneck: kernel fusion (FlashAttention) and quantization help memory-bound phases; FP8 and better parallelism help compute-bound ones.
- This is why **disaggregated prefill/decode** serving puts the two phases on differently tuned hardware or pools.
- Profile (Nsight, PyTorch profiler) before optimizing: check achieved bandwidth vs achieved FLOPS.

### Q148. What is the difference between training FLOPs and inference FLOPs?
> ✍️ _Drafted answer (not from the original notes)._

For a dense Transformer with $N$ parameters:

**Training FLOPs ≈ $6ND$** (D = training tokens)
- Forward pass ≈ $2N$ FLOPs per token (one multiply plus one add per parameter).
- Backward pass ≈ $4N$ per token (gradients with respect to both activations and weights).
- Example: a 7B model on 2T tokens ≈ $6 \times 7\times10^9 \times 2\times10^{12} \approx 8.4\times10^{22}$ FLOPs — a one-time cost, but enormous.

**Inference FLOPs ≈ $2N$ per generated token** (forward only)
- Plus attention cost that grows with context: ≈ $2 \cdot n_{layers} \cdot \text{ctx} \cdot d_{model}$ per token (more for long contexts).
- Generating 1,000 tokens with a 7B model ≈ $1.4\times10^{13}$ FLOPs.

**Key differences**
- **Scale vs frequency:** training is a one-off cost; inference is paid per request, forever. For a popular product, cumulative inference compute can exceed training compute — which is why teams over-train smaller models (beyond Chinchilla-optimal).
- **Bottleneck:** training is mostly compute-bound (big batches, high utilization, MFU 40–60%). Inference decode is usually **memory-bandwidth bound**, so FLOPs alone underestimate latency; prefill is compute-bound.
- **Memory:** training also stores optimizer states, gradients and activations (~16–20 bytes per param with Adam in mixed precision); inference stores weights plus the KV cache.
- **MoE:** use *active* parameters in both formulas.

**Use in practice:** estimate training cost = FLOPs ÷ (GPUs × peak FLOPS × MFU); estimate inference cost from tokens/day × 2N, then sanity-check against bandwidth limits.

### Q149. How would you estimate the GPU requirements for serving an LLM?
> ✍️ _Drafted answer (not from the original notes)._

Work through **memory first, then throughput**, then validate with a benchmark.

**1. Weights:** params × bytes. A 70B model: FP16 ≈ 140 GB, FP8/INT8 ≈ 70 GB, INT4 ≈ 35–40 GB.

**2. KV cache per token:** $2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{bytes}$.
Llama-3-70B (80 layers, 8 KV heads, d=128, FP16) ≈ 320 KB per token.
**Total KV** = per-token × (avg context length) × (concurrent sequences). 32 concurrent users × 8k tokens ≈ 84 GB.

**3. Overhead:** activations, CUDA graphs, fragmentation → leave ~10–20% headroom (vLLM's `gpu_memory_utilization` ≈ 0.9).

**4. Memory total → GPU count:** 140 + 84 + ~20 ≈ 244 GB → 4× H100 80 GB (tensor-parallel 4), or 2× H100 with FP8 weights and a smaller KV budget.

**5. Throughput check (decode is bandwidth-bound):**
- Tokens/s per sequence ≈ effective bandwidth ÷ bytes read per step. With TP=4 (~13 TB/s aggregate) reading ~140 GB → ~60–90 tokens/s at batch 1 (in practice lower).
- Batching amortizes weight reads: aggregate throughput rises nearly linearly with batch size until compute-bound or out of KV memory.
- Required capacity = peak requests/s × avg output tokens ÷ tokens/s per replica → number of replicas.

**6. Latency SLOs:** TTFT (prefill, compute-bound; scales with prompt length) and TPOT (decode). Long prompts may need chunked prefill or separate prefill nodes.

**7. Validate:** load test with real prompt/response length distributions (vLLM benchmark, GenAI-Perf), measure p95 TTFT/TPOT, then add autoscaling and ~30% headroom for peaks and failover.

**Levers:** quantization, GQA models, prefix caching, speculative decoding, smaller distilled models, routing easy traffic to cheaper models.

### Q150. What happens to model quality when you aggressively quantize a model?
> ✍️ _Drafted answer (not from the original notes)._

Quality degrades **gracefully down to ~8 bits, noticeably at 4 bits, and sharply below that** — but how much depends on the method, the model size and the task.

**Typical pattern (weight-only quantization)**
- **FP16/BF16 → INT8/FP8:** near-lossless (<1% on benchmarks) with decent calibration.
- **INT4 (GPTQ, AWQ, NF4):** small perplexity increase; usually acceptable for chat, a few points lost on reasoning/math/code.
- **3-bit and below:** perplexity rises quickly; models become repetitive, lose factual recall, follow instructions badly, and can produce incoherent output (unless you use specialized methods such as QuIP#/AQLM, or quantization-aware training).

**Why it hurts**
- **Outliers:** a few activation channels and weights have huge magnitudes; uniform low-bit grids clip them or waste resolution (hence LLM.int8(), SmoothQuant, AWQ).
- **Error accumulation** across many layers and long generations.
- **Activation and KV-cache quantization** is harder than weight quantization; KV quantization can hurt long-context retrieval.
- **Small models suffer more** than large ones (less redundancy).
- Benchmark averages hide damage on **tail capabilities**: multilingual, long context, precise arithmetic, tool-call formatting, and safety behavior.

**Production practice**
- Evaluate the quantized model on **your own task eval set**, not just perplexity; check structured-output validity and refusal behavior.
- Prefer calibrated methods (AWQ/GPTQ) with representative calibration data; keep sensitive layers (embeddings, LM head, first/last layers) in higher precision.
- Compare "bigger model at 4-bit" vs "smaller model at 16-bit" for the same memory — the former often wins.
- Consider FP8 on H100-class GPUs as a near-free default.

### Q151. How would you determine whether to fine-tune, RAG, prompt-engineer, or train from scratch?
| Approach | Choose when | Best for |
| --- | --- | --- |
| **Prompting** | The model already has the knowledge/capability, but needs better instructions or context | Behavior, formatting, reasoning instructions, simple tasks |
| **RAG** | The model needs **external, private, current, or large amounts of knowledge** | Company documents, policies, databases, knowledge bases |
| **Fine-tuning** | You need to change the model's **learned behavior or output patterns** consistently | Domain-specific style, classification, structured outputs, specialized behavior |

#### Simple rule

**Prompting → tell the model what to do**

**RAG → give the model information it doesn't know**

**Fine-tuning → teach the model how to consistently behave**

#### Example

Suppose you're building a **company support chatbot**:

- **Prompting:** "Answer professionally and cite the relevant policy."
- **RAG:** Retrieve the company's latest HR policies and provide them to the LLM.
- **Fine-tuning:** Train the model to consistently classify tickets into `HR`, `IT`, `Finance`, etc.

> ✍️ _Drafted addition._

**Train from scratch (pretraining your own model)** is the fourth option, and rarely the right one:
- **Choose it only when:** you have a genuinely different modality or language/tokenization that existing models handle badly (e.g., genomics, proprietary time-series, a low-resource script); you have **massive in-domain data** (hundreds of billions of tokens or more); strict sovereignty, licensing or IP rules forbid using third-party weights; or a tiny, highly specialized model must run on the edge.
- **Cost:** millions of dollars of compute, a dedicated research/infra team, months of time, and you must also rebuild instruction tuning, safety alignment and evaluation. Continued pretraining (domain-adaptive pretraining) of an open model usually captures most of the benefit at a fraction of the cost.

**Decision order (escalate only when the cheaper option fails your eval):**
1. **Prompt engineering** — always first; establish a baseline with a golden eval set.
2. **RAG** — if failures are about missing, private, changing or citable knowledge.
3. **Fine-tuning (LoRA/PEFT, then full)** — if failures are about behavior, format, style, latency/cost (distilling into a smaller model), or a narrow task with enough labeled examples (hundreds to thousands).
4. **Continued pretraining** — if the domain vocabulary and distribution are far from the base model's.
5. **Train from scratch** — only if all of the above fail *and* the data, budget and strategic need exist.

**Combine them:** most production systems use prompting + RAG + a fine-tuned model together. Decide with data: measure each failure category on the eval set and pick the option that addresses the dominant one at the lowest total cost (build plus ongoing inference plus maintenance).

### Q152. Explain LoRA and QLoRA.
#### LoRA (Low-Rank Adaptation)

**LoRA** is a parameter-efficient fine-tuning (PEFT) technique that freezes the original LLM weights and learns small **low-rank matrices** instead of updating all parameters.

For a weight matrix WW:

$W' = W + BA$

where, $A$ and $B$ are much smaller matrices.

**Key points:**

- Base model weights remain **frozen**.
- Only the LoRA adapters are trained.
- Requires much less GPU memory and storage.
- Faster and cheaper than full fine-tuning.
- Adapters can be swapped for different tasks.
- Commonly applied to attention projections such as **Q, K, V, and O**.

---

#### QLoRA (Quantized LoRA)

**QLoRA** combines **LoRA + quantization** to fine-tune large models with significantly less GPU memory.

Typical process:

1. Quantize the pretrained model to **4-bit** precision.
2. Keep the quantized base model frozen.
3. Add trainable LoRA adapters.
4. Train only the LoRA parameters.
5. During computation, use higher-precision representations as needed.

**Key points:**

- The base model is typically stored in **4-bit NF4** quantization.
- LoRA adapters remain trainable.
- Uses much less VRAM than standard LoRA.
- Makes fine-tuning relatively large LLMs possible on smaller GPUs.
- **QLoRA does not mean quantizing the LoRA adapters themselves**; the major memory saving comes from quantizing the frozen base model.

#### LoRA vs QLoRA

| Feature | LoRA | QLoRA |
| --- | --- | --- |
| Base model | Full/normal precision | 4-bit quantized |
| Base weights | Frozen | Frozen |
| Trainable parameters | LoRA adapters | LoRA adapters |
| VRAM usage | Low | **Very low** |
| Fine-tuning speed | Fast | Fast, with quantization overhead |
| Main advantage | Efficient fine-tuning | **Memory-efficient fine-tuning of large LLMs** |

### Q153. Explain RLHF, PPO, DPO, and reward hacking.
#### 1. RLHF — Reinforcement Learning from Human Feedback

RLHF aligns an LLM with human preferences.

**Typical pipeline:**

```txt
Pretrained LLM
     ↓
Supervised Fine-Tuning (SFT)
     ↓
Human preference data
     ↓
Reward Model
     ↓
RL optimization (usually PPO)
     ↓
Aligned LLM
```

Humans compare multiple model responses, e.g., **A is better than B**. These preferences train a **reward model**, which then provides a reward signal for optimizing the LLM.

---

#### 2. PPO — Proximal Policy Optimization

**PPO** is an RL algorithm commonly used in traditional RLHF.

The LLM is treated as a **policy** that generates tokens, and the reward model evaluates the generated response.

PPO tries to improve the policy while preventing it from changing too aggressively from the previous policy.

A **KL penalty** is commonly used to keep the optimized model close to the reference/SFT model.

**Key idea:**

> PPO = optimize the LLM against a learned reward while constraining policy updates.

---

#### 3. DPO — Direct Preference Optimization

DPO uses human preference pairs **directly**, without explicitly training a separate reward model or running an RL loop like PPO.

Given:

```txt
Prompt
 ├── Chosen response
 └── Rejected response
```

DPO increases the probability of the preferred response relative to the rejected response.

**RLHF/PPO:**

```txt
Preferences → Reward Model → PPO → LLM
```

**DPO:**

```txt
Preferences → DPO → LLM
```

So DPO is generally **simpler and more stable to train** than PPO-based RLHF.

---

#### 4. Reward Hacking

**Reward hacking** occurs when the model learns to maximize the reward signal without actually achieving the intended objective.

Example:

> A reward model favors detailed answers, so the LLM learns to produce unnecessarily long responses—even when the additional content is incorrect or irrelevant.

The model is effectively exploiting a weakness in the reward function.

Common causes:

- Incomplete reward design
- Reward-model bias
- Distribution shift
- Proxy objective ≠ actual objective
- Over-optimization against the reward model

#### Relationship

```txt
RLHF
 └── PPO → optimize reward model
 └── DPO → directly optimize preferences

Reward hacking
 └── Risk when optimizing imperfect reward signals
```

### Q154. HyDE, query decomposition, multi-query, and step-back prompting.
#### RAG Query Optimization Techniques

- **HyDE (Hypothetical Document Embeddings):** Generate a hypothetical answer/document first, then embed it and use that embedding for retrieval. Useful when the **user query and relevant documents use different terminology**.
- **Query Decomposition:** Break a complex question into **smaller sub-questions**, retrieve answers for each, then combine them. Useful for **multi-step questions**.
- **Multi-Query:** Generate several alternative versions of the same query and retrieve documents for each. Improves **recall** when one wording may miss relevant documents.
- **Step-Back Prompting:** Transform a specific question into a **broader conceptual question** before retrieval/reasoning. Helps when answering requires **general principles or background knowledge**.

**Quick distinction:**

**HyDE → hypothetical document**

**Decomposition → smaller questions**

**Multi-query → different phrasings**

**Step-back → broader question**

### Q155. How would you debug an LLM/RAG system that hallucinates?
#### Debugging Hallucinations in an LLM/RAG System

1. **Check retrieval**
   - Are the correct documents being retrieved?
   - Check **Recall@K, Precision@K, and context relevance**.
2. **Check retrieved context**
   - Is the required answer actually present?
   - Check for noisy, incomplete, outdated, or conflicting chunks.
3. **Check generation**
   - Verify whether the model's answer is supported by the retrieved context.
   - Reduce unnecessary generation freedom through better prompts, temperature, and structured output.
4. **Check the pipeline**
   - Inspect query rewriting, chunking, embeddings, reranking, and context assembly.
   - Trace the complete request to identify where incorrect information enters.
5. **Add guardrails**
   - Require citations/evidence.
   - Instruct the model to say **"I don't know"** when evidence is insufficient.
   - Add factuality/groundedness evaluation.

### Q156. How would you design production LLM serving?
#### Production LLM Serving Architecture

**Client → API Gateway → LLM Router → Inference Servers → Model**

Key components:

- **API Gateway:** Authentication, rate limiting, request validation.
- **LLM Router:** Load balancing, model selection, fallback, and request prioritization.
- **Inference:** vLLM/TGI with **continuous batching, KV caching, PagedAttention, and quantization**.
- **GPU layer:** Multiple GPU replicas with autoscaling based on traffic and GPU utilization.
- **Streaming:** Return tokens progressively using SSE/WebSocket.
- **Caching:** Cache repeated prompts/responses where appropriate.
- **Observability:** Track latency (p50/p95/p99), tokens/sec, GPU utilization, errors, token usage, and cost.
- **Reliability:** Timeouts, retries, circuit breakers, health checks, and graceful degradation.
- **Security:** Authentication, authorization, PII protection, prompt-injection controls, and tenant isolation.

### Q157. KV cache, PagedAttention, FlashAttention, GQA, and quantization.
- **KV Cache:** Stores past **Key/Value states** during generation so they don't need to be recomputed.
- **PagedAttention:** Organizes KV-cache memory into fixed-size **blocks/pages**, allowing efficient allocation and sharing.
- **FlashAttention:** Uses **tiling and fused GPU operations** to reduce memory I/O during attention computation.
- **GQA (Grouped Query Attention):** Multiple Q heads share fewer K/V heads, reducing KV-cache memory while retaining much of MHA's quality.
- **Quantization:** Converts weights/activations to lower precision such as **INT8 or INT4**, reducing memory and compute requirements.

### Q158. How do you evaluate an LLM application in production?
Evaluate it across **quality, safety, performance, and business metrics**:

1. **Response quality**
   - Correctness/factuality
   - Relevance
   - Completeness
   - Groundedness/hallucination rate
   - LLM-as-a-judge + human evaluation
2. **RAG-specific metrics**
   - Retrieval precision/recall
   - Context relevance
   - Context completeness
   - Faithfulness to retrieved documents
   - Answer correctness
3. **Safety**
   - Toxic or harmful outputs
   - Prompt injection resistance
   - PII/data leakage
   - Jailbreak detection
4. **Production performance**
   - **Latency:** p50, p95, p99
   - Throughput/requests per second
   - Token usage
   - Error and timeout rate
   - Cost per request
5. **User/business metrics**
   - User satisfaction
   - Task completion rate
   - Conversation abandonment
   - Escalation-to-human rate
   - Conversion or business KPI

#### Production approach

Use **continuous evaluation** rather than evaluating only before deployment:

**Logs/Traces → Sampling → Automated evaluation → Human feedback → Regression tests → Monitoring → Model/prompt update**

### Q159. What is RLVR (Reinforcement Learning with Verifiable Rewards)?
**RLVR** uses rewards that can be **automatically verified** rather than relying primarily on subjective human preferences.

For example:

- Mathematics → verify whether the final answer is correct.
- Coding → run unit tests.
- Formal reasoning → check a proof.
- Structured output → validate against a schema.

#### RLHF vs RLVR

| Aspect | RLHF | RLVR |
| --- | --- | --- |
| Reward source | Human preferences | Verifiable outcome |
| Reward quality | Subjective | More objective |
| Scalability | Expensive | Highly scalable |
| Example | Helpful response | Correct mathematical answer |
| Main challenge | Human preference modeling | Designing reliable verifiers |

RLVR is particularly attractive for reasoning tasks because the reward can be generated automatically at large scale.

### Q160. What is RLCD (Reinforcement Learning for Calibrated Decisions)?
**RLCD** can refer to reinforcement-learning approaches that optimize an agent not only for making good decisions but also for producing **well-calibrated confidence or uncertainty estimates**.

The goal is to avoid situations where an AI is highly confident in an incorrect decision.

For example:

> Prediction: "Fraud"
>
> Confidence: 99%

If the model is actually correct only 70% of the time at that confidence level, it is **poorly calibrated**.

#### Why it matters

RLCD is useful in decision-making systems where uncertainty matters, such as:

- Autonomous agents
- Medical decision support
- Financial decision systems
- Recommendation systems
- Risk-sensitive AI

Accuracy alone isn't sufficient for high-stakes decision systems; we also want the confidence associated with decisions to be reliable.

### Q161. What is ImageNet?
**ImageNet** is a large-scale image dataset and benchmark that played a major role in the development of modern deep learning for computer vision.

It contains millions of labeled images organized into thousands of object categories.

#### Why is ImageNet important?

It helped establish:

- Large-scale supervised computer vision
- CNN-based image classification
- Transfer learning
- Standardized benchmarking

The **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** was particularly influential in the development of CNN architectures.

ImageNet's importance is not just the dataset itself; it demonstrated that **large datasets + deep neural networks + GPU computation** could dramatically outperform traditional computer-vision pipelines.

### Q162. What is AlexNet and why is it important?
**AlexNet** is a deep convolutional neural network that achieved a breakthrough result in the 2012 ImageNet competition.

#### Important techniques

- Deep CNN architecture
- **ReLU** activations
- GPU-based training
- Dropout
- Data augmentation
- Max pooling
- Local response normalization

Its success demonstrated the effectiveness of deep CNNs for large-scale image recognition.

#### Why was ReLU important?

Compared with sigmoid/tanh, ReLU reduces saturation-related gradient problems and makes deep networks easier to optimize.

AlexNet was a major milestone that helped trigger the modern deep-learning revolution in computer vision.

### Q163. What is Binning?
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

### Q164. What is the difference between System 1 and System 2 thinking?
The distinction comes from the dual-process model of cognition popularized by Daniel Kahneman.

| System 1 | System 2 |
| --- | --- |
| Fast | Slow |
| Automatic | Deliberate |
| Intuitive | Analytical |
| Low cognitive effort | Higher cognitive effort |
| Pattern recognition | Step-by-step reasoning |
| Usually inexpensive | More computationally expensive |

#### AI analogy

**System 1-like AI:**

- Fast classification
- Simple prediction
- Direct response generation

**System 2-like AI:**

- Planning
- Multi-step reasoning
- Tool use
- Verification
- Iterative problem solving

Modern AI agents often combine fast generation with additional reasoning, planning, tool use, or verification loops rather than treating inference as a single forward pass.

### Q165. What is LLM inference? Explain prefill vs. decode.
LLM inference is the process of generating tokens from a trained language model.

It is generally divided into **prefill** and **decode** phases.

#### Prefill

The model processes the entire input prompt in parallel.

```txt
Prompt → Transformer → KV Cache
```

Prefill is generally **compute-intensive** because many prompt tokens are processed together.

#### Decode

The model generates output **one token at a time**.

```txt
Token 1 → Token 2 → Token 3 → ...
```

The KV cache stores previously computed attention information, avoiding recomputing the entire sequence.

Decode is often **memory-bandwidth and latency sensitive**.

#### Key metrics

**Time to First Token (TTFT):** Measures how long it takes to generate the first token.

**Time Per Output Token (TPOT):** Measures generation latency per token.

**Throughput:** Often measured in tokens/second.

Optimizing LLM serving requires balancing **prefill throughput, decode latency, KV-cache memory, batching, GPU utilization, and concurrency**.

### Q166. When can SQL indexing make a SELECT query slower?
- **Low selectivity** – If a query returns many rows, a sequential scan may be faster.
  ```sql
  SELECT * FROM users WHERE gender = 'M';
  ```
- **Very small tables** – Full table scans can be cheaper than index lookups.
- **Poorly designed indexes** – Indexes that don't match the query pattern provide little benefit.
- **Non-sargable predicates** – Functions on indexed columns may prevent efficient index usage.
  ```sql
  WHERE LOWER(name) = 'john';
  ```
- **Too many indexes** – Increase storage usage and can negatively affect overall database performance.

#### Senior-level approach

- Don't assume **"Index = faster."**
- Use:
  ```sql
  EXPLAIN ANALYZE
  SELECT ...;
  ```
- Check whether the optimizer chooses **Sequential Scan, Index Scan, or Index Only Scan**.

### Q167. What are Graph Neural Networks (GNNs)?
A **Graph Neural Network (GNN)** is a neural network designed to learn from graph-structured data.

A graph consists of:

```txt
Nodes + Edges + Features
```

Examples:

- Social networks
- Recommendation systems
- Fraud networks
- Knowledge graphs
- Molecular structures
- Transportation networks

#### Core idea

GNNs use **message passing**.

Each node aggregates information from its neighboring nodes:

```txt
Neighbor features
       ↓
Aggregation
       ↓
Node representation
       ↓
Updated node embedding
```

A simplified formulation is:

$h_v^{(l+1)} $ = $\text{UPDATE} \left( h_v^{(l)}, \text{AGGREGATE}\left(\{h_u^{(l)}:u\in N(v)\}\right) \right)$

#### Common GNN architectures

- **GCN** — Graph Convolutional Network
- **GAT** — Graph Attention Network
- **GraphSAGE**
- **GIN** — Graph Isomorphism Network

#### Common tasks

**Node classification:** Predict a label for a node.

**Link prediction:** Predict whether an edge should exist.

**Graph classification:** Classify an entire graph.

GNNs are especially useful when **relationships between entities contain predictive information**, which conventional tabular models may fail to capture.

### Q168. What is Unlimited-OCR, and what is the open-source Baidu OCR model?
#### Unlimited-OCR

- **Unlimited-OCR** is Baidu's open-source **vision-language document parsing model**.
- Its goal is **“one-shot long-horizon parsing”**—processing complex documents rather than treating OCR as simple character recognition.
- It supports:
  - **Single-image document parsing**
  - **Multi-page document parsing**
  - **PDF parsing**
  - Long outputs with up to **32K tokens** in the provided inference configuration.
- The model is available as : **`baidu/Unlimited-OCR`** and can be used with Transformers, vLLM, and SGLang.

#### What is the open-source Baidu OCR model?

The model you're referring to is:

> **Baidu Unlimited-OCR (`baidu/Unlimited-OCR`)**

It is better described as a **VLM-based document parser/OCR model**, rather than a traditional OCR engine.

**Traditional OCR:**

```txt
Image → Text
```

**Unlimited-OCR:**

```txt
Document/Image
      ↓
Vision-Language Model
      ↓
Document Parsing
      ↓
Structured textual representation
```

For your **RAG/document AI** context, the important point is that Unlimited-OCR can serve as the **document ingestion/parsing layer** before chunking, embedding, and retrieval.

### Q169. Singularity in AI?
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

### Q170. Singular Value Decomposition (SVD)?

> 🔁 _See also [01-llm-fundamentals.md → Q72](ai-engineering/01-llm-fundamentals.md#q72-what-is-svd-singular-value-decomposition) (separate notes on the same question)._
**Singular Value Decomposition (SVD)** is a matrix factorization technique that decomposes a matrix into three matrices:

$A = U\Sigma V^T$

Where:

- **$A$** = original matrix
- **$U$** = left singular vectors
- **$\Sigma$** = diagonal matrix containing **singular values**
- **$V^T$** = transpose of the right singular vectors

##### Why SVD is useful in AI/ML

1. **Dimensionality Reduction**
   - Keeps the largest singular values and corresponding vectors.
   - Reduces high-dimensional data while preserving important information.
   - Used in PCA and latent semantic analysis.
2. **Data Compression**
   - Approximates a large matrix using only its top  singular values: $A \approx U_k\Sigma_kV_k^T$
3. **Recommendation Systems**
   - Factorizes user-item matrices to discover latent user preferences and item characteristics.
4. **NLP**
   - Used in **Latent Semantic Analysis (LSA)** to identify latent relationships between words and documents.
5. **Computer Vision**
   - Can compress images and extract important visual features.
6. **ML Model Compression**
   - Large weight matrices can be approximated with lower-rank matrices, reducing parameters and computation.

##### Simple intuition

Think of SVD as finding the **most important directions/features in a matrix**.

The singular values tell you how important each direction is. Large singular values represent more significant information, while very small ones can often be discarded.

Let's do a **complete SVD calculation** from start to finish using: $A= \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix}$

We want: $\boxed{A=U\Sigma V^T}$

##### 1. Calculate $A^T$

$A^T= \begin{bmatrix} 3&1\\ 2&4 \end{bmatrix}$

##### 2. Calculate $AA^T$

$AA^T= \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix} \begin{bmatrix} 3&1\\ 2&4 \end{bmatrix} = \begin{bmatrix} 13&11\\ 11&17 \end{bmatrix}$

Find its eigenvalues:

$\det \begin{bmatrix} 13-\lambda&11\\ 11&17-\lambda \end{bmatrix}=0$

$(13-\lambda)(17-\lambda)-121=0$

$\lambda^2-30\lambda+100=0$

Therefore:

$\lambda_1=15+5\sqrt5\approx26.180$

$\lambda_2=15-5\sqrt5\approx3.820$

---

##### 3. Calculate singular values

Singular values are: $\sigma_i=\sqrt{\lambda_i}$

Therefore:

$\sigma_1=\sqrt{26.180}\approx5.117$

$\sigma_2=\sqrt{3.820}\approx1.955$

So: $\boxed{ \Sigma= \begin{bmatrix} 5.117&0\\ 0&1.955 \end{bmatrix}}$

---

##### 4. Find UU

The columns of $U$ are the **normalized eigenvectors of $AA^T$**.

For $\lambda_1$: $\begin{bmatrix} 13-26.180&11\\ 11&17-26.180 \end{bmatrix} \begin{bmatrix} x\\y \end{bmatrix}=0$

This gives approximately: $y=1.198x$

After normalization: $u_1= \begin{bmatrix} 0.641\\ 0.768 \end{bmatrix}$

For $\lambda_2$: $u_2= \begin{bmatrix} 0.768\\ -0.641 \end{bmatrix}$

Therefore: $\boxed{ U= \begin{bmatrix} 0.641&0.768\\ 0.768&-0.641 \end{bmatrix}}$

---

##### 5. Find VV

Now calculate: $A^TA= \begin{bmatrix} 3&1\\ 2&4 \end{bmatrix} \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix}$= $\begin{bmatrix} 10&10\\ 10&20 \end{bmatrix}$

The eigenvalues are the same: $26.180,\quad3.820$

For $\lambda_1=26.180$:

$(10-26.180)x+10y=0$

$y=1.618x$

After normalization: $v_1= \begin{bmatrix} 0.526\\ 0.851 \end{bmatrix}$

For $\lambda_2=3.820$: $v_2= \begin{bmatrix} 0.851\\ -0.526 \end{bmatrix}$

Thus: $\boxed{ V= \begin{bmatrix} 0.526&0.851\\ 0.851&-0.526 \end{bmatrix}}$

Therefore: $\boxed{ V^T= \begin{bmatrix} 0.526&0.851\\ 0.851&-0.526 \end{bmatrix}}$

---

#### 6. Final SVD

Putting everything together: $\boxed{ A=U\Sigma V^T }$

$\boxed{ \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix} \approx \begin{bmatrix} 0.641&0.768\\ 0.768&-0.641 \end{bmatrix} \begin{bmatrix} 5.117&0\\ 0&1.955 \end{bmatrix} \begin{bmatrix} 0.526&0.851\\ 0.851&-0.526 \end{bmatrix} }$

If you multiply these three matrices, you recover approximately: $\begin{bmatrix} 3&2\\ 1&4 \end{bmatrix}$

##### The whole process to memorize

$\boxed{A\rightarrow AA^T\rightarrow \text{eigenvectors}\rightarrow U}$

$\boxed{A\rightarrow A^TA\rightarrow \text{eigenvectors}\rightarrow V}$

$\boxed{\text{eigenvalues}\rightarrow\sqrt{\text{eigenvalues}}\rightarrow\Sigma}$

So the key relationship is: $\boxed{A=U\Sigma V^T}$

**AI connection:** Once you have, $U,\Sigma,V^T$ you can keep only the largest kk singular values to create a low-rank approximation: $A\approx U_k\Sigma_kV_k^T$

This is the core idea behind **SVD-based dimensionality reduction, compression, LSA, recommendation systems, and low-rank neural-network compression**.

### Q171. What is Open Knowledge Format (OKF), and how is it different from RAG?
**OKF** is a structured way to **represent and organize knowledge** so that machines and AI systems can easily understand, search, and exchange it.

It can represent:

- **Entities** — people, companies, products
- **Facts** — attributes and information
- **Relationships** — connections between entities
- **Sources/metadata** — where the information came from

#### OKF vs RAG

|  | OKF | RAG |
| --- | --- | --- |
| Purpose | Structure knowledge | Retrieve knowledge for an LLM |
| Focus | **Knowledge representation** | **Knowledge retrieval + generation** |
| Typical data | Entities, facts, relationships | Documents, chunks, vectors, databases |
| LLM required | No | Usually yes |

**Example:**

```txt
OKF:
Einstein → developed → Theory of Relativity
```

A RAG system could **retrieve this knowledge** when a user asks:

"What did Einstein develop?"

and provide it to the LLM to generate the answer.

## Covered elsewhere

These questions are answered by another question in this repo:

- What is the difference between object detection, segmentation, and classification? → see [02-ai-engineer-senior-level.md Q14](role-prep/02-ai-engineer-senior-level.md#q14-whats-the-difference-between-object-detection-image-classification-and-segmentation-and-when-would-you-use-each)
- What is LoRA, and why does it reduce fine-tuning cost? → see [Q152](#q152-explain-lora-and-qlora)
- Explain QLoRA. → see [Q152](#q152-explain-lora-and-qlora)
- Explain the RLHF pipeline. → see [Q153](#q153-explain-rlhf-ppo-dpo-and-reward-hacking)
- What is PPO, and why was it used for RLHF? → see [Q153](#q153-explain-rlhf-ppo-dpo-and-reward-hacking)
- What is DPO, and how does it differ from PPO-based RLHF? → see [Q153](#q153-explain-rlhf-ppo-dpo-and-reward-hacking)
- What is reward hacking? → see [Q153](#q153-explain-rlhf-ppo-dpo-and-reward-hacking)
- What is hybrid search? → see [Q94](#q94-compare-dense-retrieval-vs-sparse-retrieval)
- What is HyDE? → see [Q154](#q154-hyde-query-decomposition-multi-query-and-step-back-prompting)
- What is query decomposition? → see [Q154](#q154-hyde-query-decomposition-multi-query-and-step-back-prompting)
- What is step-back prompting? → see [Q154](#q154-hyde-query-decomposition-multi-query-and-step-back-prompting)
- What is multi-query retrieval? → see [Q154](#q154-hyde-query-decomposition-multi-query-and-step-back-prompting)
- How would you evaluate a RAG system separately from the LLM? → see [Q92](#q92-explain-a-complete-rag-architecture)
- Training loss suddenly becomes NaN. What do you investigate? → see [Q45](#q45-how-would-you-debug-a-neural-network-whose-loss-suddenly-becomes-nan)
