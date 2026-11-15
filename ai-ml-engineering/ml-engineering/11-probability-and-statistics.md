<!-- notion-id: 3d4674182854803ab005c946791afc47 -->
# Probability and Statistics

Probability and statistics interview questions: bias-variance, probability distributions, descriptive statistics, correlation, hypothesis testing and p-values, confidence intervals, Bayesian vs. frequentist inference, MLE/MAP, CLT, sampling and the bootstrap.

### Q1. Explain the Bias-Variance Tradeoff.

##### Bias-Variance Tradeoff

The **bias-variance tradeoff** is the balance between two types of model error that determines how well a machine learning model generalizes to unseen data.

- **Bias:** Error caused by overly simplistic assumptions in the model.
  - High bias → **underfitting**
  - Example: Using a linear model for a highly nonlinear problem.
- **Variance:** Error caused by the model being too sensitive to the training data.
  - High variance → **overfitting**
  - Example: A very complex model memorizing the training data, including noise.
- **Tradeoff:**
  - Increasing model complexity usually **reduces bias** but **increases variance**.
  - Decreasing model complexity usually **increases bias** but **reduces variance**.
  - The goal is to find a model complexity that achieves a good balance and minimizes **generalization error**.

**In short:**

> **High bias → underfitting; high variance → overfitting; the goal is to balance both for good generalization.**

### Q2. Explain different probability distributions (Normal, Binomial, Poisson, Uniform).

##### Different Probability Distributions

A **probability distribution** describes how likely different values or outcomes of a random variable are.

- **Normal Distribution**
  - Continuous probability distribution with a symmetric, bell-shaped curve.
  - Defined by its **mean** and **standard deviation**.
  - Values close to the mean are more likely than extreme values.
  - Commonly used for measurements such as height, weight, test scores, and measurement errors.
- **Binomial Distribution**
  - Discrete distribution that models the number of **successes in a fixed number of independent trials**.
  - Each trial has two possible outcomes: success or failure.
  - Example: Number of heads obtained in 10 coin flips.
  - Parameters: number of trials `n` and probability of success `p`.
- **Poisson Distribution**
  - Discrete distribution that models the number of times an **event occurs within a fixed interval** of time or space.
  - Useful when events occur independently at a roughly constant average rate.
  - Example: Number of customer arrivals at a store per hour.
  - Parameter: average event rate `λ`.
- **Uniform Distribution**
  - Distribution where all outcomes within a specified range are **equally likely**.
  - Can be discrete or continuous.
  - Example: A fair six-sided die has a discrete uniform distribution.
  - A random number between 0 and 1 is an example of a continuous uniform distribution.

**In short:**

| Distribution | Type | Typical Use |
| --- | --- | --- |
| Normal | Continuous | Measurements and natural variation |
| Binomial | Discrete | Number of successes in fixed trials |
| Poisson | Discrete | Number of events in an interval |
| Uniform | Discrete/Continuous | Equally likely outcomes |

### Q3. What is the normal distribution and its functions?

##### Normal Distribution

A **normal distribution** is a continuous probability distribution with a **symmetric, bell-shaped curve**. It is widely used to model continuous data such as measurements, errors, and test scores.

- It is **symmetric around the mean**.
- **Mean = Median = Mode**.
- It is determined by:
  - **Mean (μ):** controls the center.
  - **Standard deviation (σ):** controls the spread.
- Most observations are close to the mean, while extreme values are less likely.

##### Important Functions

1. **Probability Density Function (PDF)**
   - Describes the likelihood density of values.
   - The total area under the curve equals **1**.
2. **Cumulative Distribution Function (CDF)**
   - Gives the probability that a random variable is **less than or equal to a given value**.
3. **Quantile Function (PPF)**
   - The inverse of the CDF.
   - Finds the value corresponding to a given probability or percentile.
4. **Z-score**
   - Measures how many standard deviations a value is away from the mean.
   - Useful for standardizing values and comparing observations.

##### 68–95–99.7 Rule

- **68%** of values → within 1 standard deviation
- **95%** → within 2 standard deviations
- **99.7%** → within 3 standard deviations

**In short:** The normal distribution models continuous data around a mean, and its **PDF, CDF, PPF, and z-score** are commonly used for probability and statistical analysis.

### Q4. What is an exponential distribution?

##### Exponential Distribution

The **exponential distribution** is a continuous probability distribution used to model the **time until an event occurs** when events happen independently at a constant average rate.

- It is commonly used for **waiting-time problems**.
- Example: Time until the next customer arrives at a store.
- It is controlled by a **rate parameter (λ)**.
- A larger λ means events happen more frequently, so the expected waiting time is shorter.
- It is **right-skewed**: many observations are small, with fewer very large values.
- It has the **memoryless property**, meaning the probability of waiting an additional amount of time does not depend on how long you have already waited.

##### Common Applications

- Time between customer arrivals
- Time until a machine fails
- Network request arrival times
- Waiting time between events

### Q5. What is the binomial distribution?

##### Binomial Distribution

The **binomial distribution** is a discrete probability distribution that models the **number of successes in a fixed number of independent trials**, where each trial has only two possible outcomes: **success or failure**.

- **Fixed number of trials:** `n`
- **Two possible outcomes:** success or failure
- **Constant probability of success:** `p`
- **Trials are independent** of each other.
- The random variable represents the **number of successes**.

**Example:**

If you flip a coin 10 times, the binomial distribution can determine the probability of getting exactly 6 heads.

##### Applications

- Number of successful predictions out of multiple predictions
- Number of defective products in a sample
- Number of customers who respond to an offer
- Number of heads in repeated coin flips

### Q6. What is the Bernoulli distribution?

##### Bernoulli Distribution

The **Bernoulli distribution** is a discrete probability distribution that models a **single experiment with only two possible outcomes**:

- **Success:** represented by `1`
- **Failure:** represented by `0`
- The probability of success is **p**.
- The probability of failure is **1 − p**.

**Example:**

For a single coin toss:

- Heads = success
- Tails = failure

##### Bernoulli vs. Binomial

- **Bernoulli:** One trial.
- **Binomial:** Multiple independent Bernoulli trials and counts the total number of successes.

### Q7. What is the multinomial distribution?

##### Multinomial Distribution

The **multinomial distribution** is a generalization of the **binomial distribution** that models the number of outcomes in **multiple categories** over a fixed number of independent trials.

- Each trial has **more than two possible outcomes**.
- The number of trials is fixed.
- Each trial belongs to exactly **one category**.
- Each category has its own probability.
- The probabilities of all categories add up to **1**.

**Example:**

If a six-sided die is rolled 20 times, the multinomial distribution can model how many times each number (1–6) appears.

##### Binomial vs. Multinomial

- **Binomial:** 2 possible categories → success/failure.
- **Multinomial:** 3 or more possible categories.

### Q8. What is the lognormal distribution?

##### Lognormal Distribution

The **lognormal distribution** is a continuous probability distribution for a positive-valued variable whose **natural logarithm follows a normal distribution**.

- The variable can only take **positive values** (`X > 0`).
- It is typically **right-skewed**, with a long tail toward larger values.
- It is useful when values grow through **multiplicative effects** rather than additive effects.
- If `X` is lognormally distributed, then `ln(X)` follows a normal distribution.

##### Examples

- Income and wealth distributions
- Stock prices and financial returns
- Product lifetimes
- File sizes and network traffic
- Biological measurements

##### Normal vs. Lognormal

- **Normal:** Can take positive or negative values and is symmetric.
- **Lognormal:** Only positive values and is usually right-skewed.

### Q9. What is a logistic distribution?

##### Logistic Distribution

The **logistic distribution** is a continuous probability distribution that is **symmetric and bell-shaped**, similar to the normal distribution, but has **heavier tails**.

- It is centered around a **location parameter**, which determines the center.
- A **scale parameter** controls how spread out the distribution is.
- It is symmetric around its center.
- Its heavier tails make extreme values somewhat more likely than under a normal distribution.
- Its **CDF has an S-shaped (sigmoid) curve**, which is especially useful in classification.

##### Applications

- **Logistic regression**
- Binary classification
- Modeling probabilities
- Growth and dose-response models
- Statistical modeling of bounded probabilities

**Normal vs. Logistic:**

- **Normal:** lighter tails.
- **Logistic:** heavier tails and a sigmoid CDF.

### Q10. What is the gamma distribution and its functions?

##### Gamma Distribution

The **gamma distribution** is a continuous probability distribution used to model **positive-valued quantities**, especially **waiting time until multiple events occur**.

- It only takes **positive values** (`X > 0`).
- It is generally **right-skewed**, although its shape can vary.
- It is defined by two parameters:
  - **Shape (α):** controls the shape of the distribution.
  - **Rate (β):** controls the rate at which events occur.
- It generalizes the **exponential distribution**. The exponential distribution models the waiting time for one event, while the gamma distribution can model the waiting time for multiple events.

##### Common Functions

1. **Probability Density Function (PDF)**
   - Describes the probability density of different values.
   - Used to calculate probabilities over ranges.
2. **Cumulative Distribution Function (CDF)**
   - Gives the probability that the random variable is **less than or equal to a given value**.
3. **Quantile Function (PPF)**
   - The inverse of the CDF.
   - Finds the value corresponding to a given probability or percentile.

##### Applications

- Waiting time for multiple events
- Reliability and failure-time analysis
- Queueing systems
- Insurance claim modeling
- Bayesian statistics
- Modeling rainfall and other positive continuous measurements

### Q11. Poisson distribution and its function.

##### Poisson Distribution

The **Poisson distribution** is a discrete probability distribution used to model the **number of times an event occurs within a fixed interval of time, space, or area**, when events occur independently at a constant average rate.

- It is a **discrete distribution**.
- The random variable represents the **number of events**: `0, 1, 2, ...`
- It has one parameter, **λ (lambda)**, which represents the average number of events in the interval.
- The mean and variance are both **λ**.
- Examples:
  - Number of customers arriving at a store per hour.
  - Number of calls received by a call center per minute.
  - Number of defects in a manufactured product.

##### Main Function

**Probability Mass Function (PMF)**

The PMF gives the probability of observing exactly **k events** in the interval.

**P(X = k) = (e⁻λ λᵏ) / k!**

where:

- `λ` = average event rate
- `k` = number of events
- `e` = Euler's number

##### Other Functions

- **CDF:** Gives the probability of observing **k or fewer events**.
- **Mean:** `λ`
- **Variance:** `λ`
- **Standard deviation:** `√λ`

### Q12. When would you use a Poisson distribution over a Binomial distribution?

##### Poisson vs. Binomial Distribution

Use a **Poisson distribution** when you want to model the **number of events occurring within a fixed interval**, while a **Binomial distribution** is used when you have a **fixed number of trials** with two possible outcomes.

- **Use Binomial when:**
  - The number of trials is fixed (`n`).
  - Each trial has two outcomes: success/failure.
  - The probability of success (`p`) is known or estimated.
  - Example: Number of defective products among **100 inspected products**.
- **Use Poisson when:**
  - You are counting events within a fixed **time, space, area, or volume**.
  - There is no fixed number of trials.
  - Events occur independently at a roughly constant average rate (`λ`).
  - Example: Number of customers arriving at a store **per hour**.

##### Simple Rule

> **Fixed number of trials → Binomial**
>
> **Fixed interval and count of events → Poisson**

### Q13. What is the variance?

##### Variance

**Variance** is a measure of how much values in a dataset **spread out from their mean**.

- **Low variance:** Values are close to the mean.
- **High variance:** Values are far from the mean.
- It is calculated using the **average of the squared differences from the mean**.
- Variance is always **non-negative**.
- Its unit is the **square of the original unit**.

**Example:**

- `48, 49, 50, 51, 52` → low variance
- `10, 30, 50, 70, 90` → high variance

### Q14. What is stddev?

##### Standard Deviation (Std Dev)

**Standard deviation** is a measure of how much the values in a dataset **spread out from the mean**.

- **Low standard deviation:** Values are close to the mean.
- **High standard deviation:** Values are more spread out.
- It is the **square root of the variance**.
- Unlike variance, standard deviation has the **same unit as the original data**, making it easier to interpret.

**Example:**

- Mean = 50, standard deviation = 2 → values are generally close to 50.
- Mean = 50, standard deviation = 20 → values are much more spread out.

### Q15. Explain the difference between mean, median, and mode.

##### Mean vs. Median vs. Mode

These are three common measures of **central tendency** used to describe the center of a dataset.

- **Mean**
  - The **average** of all values.
  - Add all values and divide by the number of values.
  - Sensitive to **outliers**.
  - Example: `2, 4, 6` → Mean = `4`.
- **Median**
  - The **middle value** when data is arranged in ascending or descending order.
  - Less affected by outliers than the mean.
  - Example: `2, 4, 6` → Median = `4`.
  - For an even number of values, take the average of the two middle values.
- **Mode**
  - The value that occurs **most frequently**.
  - A dataset can have one mode, multiple modes, or no mode.
  - Example: `2, 3, 3, 5` → Mode = `3`.

##### When to use

| Measure | Best used when |
| --- | --- |
| **Mean** | Data is relatively symmetric without major outliers |
| **Median** | Data is skewed or contains outliers |
| **Mode** | You want the most frequent value or category |

**In short:**

> **Mean = average, Median = middle, Mode = most frequent.**

### Q16. What is the difference between correlation and covariance?

##### Correlation vs. Covariance

Both **covariance** and **correlation** measure how two variables change together, but they differ in scale and interpretation.

- **Covariance**
  - Shows the **direction** of the relationship between two variables.
  - Positive covariance → variables tend to increase together.
  - Negative covariance → one tends to increase when the other decreases.
  - Its value depends on the **units and scale** of the variables, so it is difficult to compare across datasets.
- **Correlation**
  - Shows both the **direction and strength** of the linear relationship.
  - Always ranges from **−1 to +1**.
  - `+1` → perfect positive correlation.
  - `0` → no linear correlation.
  - `−1` → perfect negative correlation.
  - It is **unitless**, making it easier to compare relationships.

##### Key Difference

| Covariance | Correlation |
| --- | --- |
| Measures direction | Measures direction + strength |
| Range is unbounded | Range: −1 to +1 |
| Depends on units | Unitless |
| Harder to interpret | Easier to interpret |

### Q17. What does a correlation coefficient of +1, 0, and -1 indicate?

The **correlation coefficient** measures the direction and strength of a **linear relationship** between two variables. It ranges from **−1 to +1**.

- **+1: Perfect positive correlation**
  - Both variables move in exactly the same direction.
  - As one increases, the other also increases proportionally.
- **0: No linear correlation**
  - There is no **linear relationship** between the variables.
  - Note: A coefficient of 0 does **not necessarily mean there is no relationship at all**; a nonlinear relationship may still exist.
- **−1: Perfect negative correlation**
  - The variables move in exactly opposite directions.
  - As one increases, the other decreases proportionally.

### Q18. Explain Correlation vs. causation.

##### Correlation vs. Causation

- **Correlation** means two variables are **associated** or tend to change together.
- **Causation** means a change in one variable **directly produces a change** in another variable.

##### Example

Suppose data shows that **ice cream sales increase when the number of sunburn cases increases**.

- There is a **correlation** between ice cream sales and sunburn cases.
- But buying ice cream does **not cause** sunburn.
- A third variable, **hot weather**, can cause both more ice cream sales and more sunburn.

##### Key Difference

| Correlation | Causation |
| --- | --- |
| Variables are associated | One variable causes a change in another |
| Does not establish cause and effect | Establishes a cause-and-effect relationship |
| Can be caused by confounding variables | Requires stronger evidence, often from controlled experiments |

### Q19. What are Type I and Type II errors?

##### Type I and Type II Errors

- **Type I Error (False Positive):**
  - It predicts positive when the actual is negative
  - Example: A medical test says a person has cancer although that person is normal.
  - Rejecting the null hypothesis when it is actually true.
  - You conclude there is an effect when there is actually no effect.
  - Probability of Type I error = **α (significance level)**.
- **Type II Error (False Negative):**
  - It predicts negative when the actual is positive
  - Example: A medical test says a person is normal although that person has cancer.
  - Failing to reject the null hypothesis when it is actually false.
  - You conclude there is no effect when there actually is one.
  - Probability of Type II error = **β**.

| Actual / Predicted | Positive | Negative |
| --- | --- | --- |
| **Positive** | True Positive (TP) | **False Negative (FN) → Type II Error** |
| **Negative** | **False Positive (FP) → Type I Error** | True Negative (TN) |

### Q20. What is a p-value? What does statistical significance mean?

##### P-value

- It is the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true.
- Null hypothesis, **H₀** ⇒ no effect, no difference, or no relationship
- Alternative hypothesis, **H1** ⇒ there is an effect, there is a difference, or there is a relationship
- A **small p-value** means the observed result would be unlikely if H₀ were true.
- A **large p-value** means there is not enough evidence to reject H₀.
- Common significance level: **α = 0.05**.

##### Statistical Significance

A result is **statistically significant** when the p-value is smaller than the chosen significance level α.

- **p-value \< 0.05 → Statistically significant → Reject H₀, Accept H1**
- **p-value ≥ 0.05 → Not statistically significant → Fail to reject H₀, Reject H1**

**p-value \< α → evidence against H₀ → statistically significant.**

### Q21. Explain p-values and their limitations.

##### P-value

- A **p-value** measures how unusual the observed data would be **assuming the null hypothesis (H₀) is true**.
- **Small p-value:** Data provides stronger evidence against H₀.
- **Large p-value:** Data does not provide sufficient evidence to reject H₀.
- Common significance level: **α = 0.05**.
  - **p \< 0.05 → statistically significant**
  - **p ≥ 0.05 → not statistically significant**

##### Limitations of P-values

- **Does not measure the size of an effect:** A tiny effect can have a very small p-value with a large sample.
- **Does not measure practical importance:** Statistical significance does not necessarily mean the result is useful or meaningful in practice.
- **Does not give the probability that H₀ is true:** A p-value is not `P(H₀ is true)`.
- **Depends on sample size:** Larger samples can produce very small p-values even for weak effects.
- **Sensitive to study design and assumptions:** Violating statistical assumptions can make p-values misleading.
- **Multiple testing problem:** Testing many hypotheses increases the chance of obtaining a significant result by chance.
- **Not a measure of certainty:** A p-value of 0.01 does not mean there is a 99% probability that the result is correct.

**Short:**

**P-value tells how compatible the data is with H₀, but it does not tell you whether an effect is large, important, or certainly true.**

### Q22. What is hypothesis testing, and when is it used in ML?

##### Hypothesis Testing

- **Hypothesis testing** is a statistical method used to determine whether there is enough evidence in sample data to support a claim about a population.
- It starts with two hypotheses:
  - **Null hypothesis (H₀):** No significant effect or difference exists.
  - **Alternative hypothesis (H₁):** A significant effect or difference exists.
- A statistical test produces a **p-value**, which is compared with a significance level such as **α = 0.05**.
- If **p \< α**, reject H₀.
- If **p ≥ α**, fail to reject H₀.

##### When is it used in ML?

- **Feature selection:** Test whether a feature has a statistically significant relationship with the target.
- **Model comparison:** Determine whether one model significantly outperforms another.
- **A/B testing:** Test whether a new model or feature improves performance compared with the existing version.
- **Model evaluation:** Determine whether observed performance differences are likely due to chance.
- **Experimentation:** Test whether a treatment, intervention, or ML system produces a meaningful effect.

### Q23. What statistical tests would you use to compare two models?

##### Statistical Tests to Compare Two ML Models

It depends on **what you are comparing** and the type of predictions.

- **Paired t-test**
  - Compare the mean performance of two models across multiple folds/runs.
  - Example: Compare accuracy or RMSE from cross-validation.
  - Assumes approximately normally distributed differences.
- **Wilcoxon signed-rank test**
  - Non-parametric alternative to the paired t-test.
  - Useful when performance differences are not normally distributed.
- **McNemar's test**
  - Used for **two classifiers on the same test samples**.
  - Compares their classification errors.
  - Particularly useful for binary classification.
- **Paired bootstrap test**
  - Resample test examples and compare model performance.
  - Useful for metrics such as accuracy, F1, AUC, or RMSE without strong distributional assumptions.
- **Permutation test**
  - Randomly permutes model labels and evaluates whether the observed performance difference is likely due to chance.
  - Flexible and useful for model comparison.

### Q24. How do you assess if a feature is statistically significant?

##### Assessing Whether a Feature Is Statistically Significant

- **Fit a statistical model**, such as linear or logistic regression, including the feature.
- Examine the feature's **coefficient** and its **p-value**.
- **p-value \< α** (commonly 0.05) → feature is statistically significant.
- **p-value ≥ α** → insufficient evidence that the feature has a statistically significant effect, given the model and assumptions.
- Use a **confidence interval** for the coefficient:
  - If the interval **does not include 0**, the coefficient is statistically significant at the corresponding confidence level.
  - If it includes 0, it is not.

##### In ML

Statistical significance alone does **not** mean a feature improves the model.

Also check:

- **Cross-validation performance**
- **Feature importance / SHAP**
- **Effect size**
- **Multicollinearity**
- **Out-of-sample performance**

### Q25. What is a confidence interval, and how is it used?

##### Confidence Interval

- A **confidence interval (CI)** is a range of values used to estimate an unknown population parameter, such as a **mean, proportion, or regression coefficient**.
- It gives an estimate along with the **uncertainty** around that estimate.
- A **95% confidence interval** means that if we repeatedly collected samples and constructed intervals using the same method, about **95% of those intervals would contain the true parameter**.

##### Example

Suppose a model's estimated accuracy improvement is **5%**, with a **95% CI of \[2%, 8%\]**.

- Estimated effect = **5%**
- Plausible range = **2% to 8%**
- Since the interval does **not include 0**, the improvement is statistically significant at the 5% level.

##### How it is used

- Estimate population parameters.
- Measure **uncertainty and precision**.
- Compare effects between models or groups.
- Assess statistical significance.
- In regression, check whether a coefficient's CI includes **0**.

**Short:**

**A confidence interval gives a range of plausible values for a population parameter and shows how uncertain the estimate is.**

### Q26. What are z-score and t-score? When do you use each?

##### Z-score

- Measures how many **standard deviations** a value is from the population mean.
- Used when the **population standard deviation (σ) is known** or when working with a standardized population.
- Follows the **standard normal distribution**.

##### T-score

- Measures how many standard errors a sample estimate is from the hypothesized population value.
- Used when the **population standard deviation is unknown** and must be estimated from the sample.
- Follows a **t-distribution**, which has heavier tails than the normal distribution.
- Especially useful for **small samples**.

##### Z-score vs T-score

|  | Z-score | T-score |
| --- | --- | --- |
| Standard deviation | Population σ known | Population σ unknown |
| Distribution | Normal | t-distribution |
| Small samples | Less appropriate | Appropriate |
| Large samples | Common | Also approaches normal |

**Short:**

**Use Z when σ is known; use T when σ is unknown and estimated from the sample.**

### Q27. Explain Bayes' Theorem. How is it relevant to Naive Bayes or Bayesian methods?

##### Bayes' Theorem

Bayes' Theorem describes how to **update the probability of a hypothesis when new evidence is observed**.

- **Prior probability:** `P(A)` — probability of the hypothesis before seeing new evidence.
- **Likelihood:** `P(B|A)` — probability of observing the evidence if the hypothesis is true.
- **Posterior probability:** `P(A|B)` — updated probability of the hypothesis after seeing the evidence.
- **Evidence:** `P(B)` — overall probability of observing the evidence.

##### Relevance to Naive Bayes

- **Naive Bayes** uses Bayes' Theorem for classification.
- It calculates the probability of each class given the observed features.
- It assumes the features are **conditionally independent given the class**.
- The class with the highest posterior probability is selected.

Example: For spam detection, it calculates whether an email is more likely to be **Spam** or **Not Spam** based on its words/features.

##### Relevance to Bayesian Methods

Bayesian methods follow the principle:

**Prior belief + New evidence → Updated belief (Posterior)**

They are used for parameter estimation, prediction, uncertainty quantification, and statistical inference.

**Short:**

**Bayes' Theorem updates probabilities using new evidence. Naive Bayes applies it to classification, while Bayesian methods use it broadly for statistical inference and updating beliefs.**

### Q28. What's the difference between MLE and MAP estimation?

##### MLE vs MAP Estimation

- **MLE (Maximum Likelihood Estimation)**
  - Finds the parameter values that make the observed data **most likely**.
  - Uses only the observed data.
  - Does not include prior beliefs about the parameters.
  - Objective: maximize `P(Data | Parameters)`.
- **MAP (Maximum A Posteriori) Estimation**
  - Finds the parameter values that are **most probable after considering both data and prior beliefs**.
  - Combines likelihood with a prior distribution.
  - Objective: maximize `P(Parameters | Data)`.

##### Key Difference

|  | MLE | MAP |
| --- | --- | --- |
| Uses data | Yes | Yes |
| Uses prior | No | Yes |
| Goal | Maximize likelihood | Maximize posterior |
| Regularization | No inherent prior | Often equivalent to regularization |

**Relationship:**

**MAP = MLE + Prior Information**

If the prior is **uniform**, MAP and MLE give the same estimate.

**Short:**

**MLE relies only on the data; MAP combines the data with prior beliefs about the parameters.**

### Q29. What is Maximum Likelihood Estimation (MLE)?

##### Maximum Likelihood Estimation (MLE)

- **MLE** is a statistical method for estimating unknown model parameters from observed data.
- It chooses the parameter values that make the **observed data most likely**.
- It maximizes the **likelihood function**:

  `L(θ) = P(Data | θ)`

- The parameter estimate is:

  `θ̂ = argmaxθ L(θ)`

##### Example

Suppose you toss a coin 10 times and get **7 heads**.

- MLE estimates the probability of heads as **p = 0.7**.
- This value makes the observed result of 7 heads out of 10 most likely.

##### In ML

MLE is widely used for:

- Logistic regression
- Linear regression under Gaussian noise
- Naive Bayes
- Gaussian mixture models
- Many probabilistic models

**Short:**

**MLE chooses the model parameters that make the observed training data most likely.**

### Q30. Explain the Bayesian vs. Frequentist approach in statistics.

##### Bayesian vs. Frequentist Approach

|  | **Frequentist** | **Bayesian** |
| --- | --- | --- |
| Parameters | Fixed but unknown | Treated as uncertain/random variables |
| Prior knowledge | Not used as a probability distribution | Uses a **prior distribution** |
| Data | Used to estimate parameters | Updates prior beliefs |
| Main result | Estimate, confidence interval, p-value | Posterior distribution, credible interval |
| Common methods | MLE, hypothesis testing | MAP, Bayesian inference |

##### Frequentist Approach

- Treats the unknown parameter as **fixed**.
- Uses observed data to estimate the parameter.
- Example: **MLE** finds the parameter that makes the observed data most likely.
- Uses **confidence intervals** and **p-values** for inference.

##### Bayesian Approach

- Treats the parameter as having a **probability distribution representing uncertainty**.
- Starts with a **prior**.
- Combines prior and observed data to obtain a **posterior**.

**Prior + Data → Posterior**

- **MAP** estimates the parameter with the highest posterior probability.

##### Simple Example

Suppose you want to estimate the probability that a coin lands heads.

- **Frequentist:** Estimate the probability from the observed coin flips.
- **Bayesian:** Start with a prior belief about the probability, then update it using the observed flips.

**Short:**

**Frequentist = infer parameters from data without a prior distribution.**

**Bayesian = combine prior beliefs with data to obtain updated beliefs.**

### Q31. What is the Central Limit Theorem(CLT)? Why is it important?

##### Central Limit Theorem (CLT)

- The **Central Limit Theorem** states that when we take sufficiently large random samples from a population, the **sampling distribution of the sample mean approaches a normal distribution**, regardless of the population's original distribution.
- As sample size increases, the sample mean becomes more normally distributed.
- The mean of the sampling distribution is approximately the **population mean**.
- The standard error decreases as sample size increases.

##### Why is it important?

- Allows us to use the **normal distribution** for statistical inference even when the population is not normally distributed.
- Forms the foundation for:
  - **Confidence intervals**
  - **Hypothesis testing**
  - **Estimating population parameters**
  - **A/B testing**
  - Many statistical and ML methods

##### Example

Even if individual customer purchase amounts are highly skewed, the **average purchase amount across many sufficiently large random samples** will tend to follow a normal distribution.

**Short:**

**CLT says that the distribution of sample means becomes approximately normal as sample size increases, making statistical inference possible for many real-world problems.**

### Q32. What are sampling techniques?

##### Sampling Techniques

**Sampling** is the process of selecting a subset of a population to study, instead of collecting data from the entire population.

##### 1. Probability Sampling

Every member has a known chance of being selected.

- **Simple Random Sampling:** Every member has an equal chance of selection.
- **Systematic Sampling:** Select every `k`th member after a random starting point.
- **Stratified Sampling:** Divide the population into groups (strata) and randomly sample from each group.
- **Cluster Sampling:** Divide the population into clusters and randomly select entire clusters.
- **Multistage Sampling:** Sampling is performed in multiple stages, often combining different techniques.

##### 2. Non-Probability Sampling

Not every member has a known or equal chance of selection.

- **Convenience Sampling:** Select participants who are easily available.
- **Purposive/Judgment Sampling:** Select participants based on specific criteria.
- **Quota Sampling:** Select a predetermined number from different groups.
- **Snowball Sampling:** Existing participants recruit additional participants.

##### Short

**Probability sampling → random selection and better generalizability.**

**Non-probability sampling → selection based on convenience or judgment and may introduce sampling bias.**

### Q33. What is the bootstrap method, and how is it used?

##### Bootstrap Method

- **Bootstrap** is a resampling technique used to estimate the **uncertainty and variability** of a statistic.
- It repeatedly creates new samples **with replacement** from the original dataset.
- The statistic of interest is calculated for each resampled dataset.
- The resulting distribution, called the **bootstrap distribution**, is used to estimate uncertainty.

##### How it is used

1. Start with the original dataset.
2. Randomly sample from it **with replacement** to create a new dataset of the same size.
3. Calculate the statistic, such as mean, median, accuracy, or model performance.
4. Repeat this process many times.
5. Use the resulting distribution to calculate:
   - **Confidence intervals**
   - **Standard errors**
   - **Bias estimates**
   - **Model performance uncertainty**

##### In ML

Bootstrap can be used to:

- Compare the performance of two models.
- Estimate confidence intervals for metrics such as accuracy, F1, or AUC.
- Assess the stability of model predictions.
- Build **bagging** methods such as Random Forests.

**Short:**

**Bootstrap repeatedly resamples the observed data with replacement to estimate the sampling distribution and uncertainty of a statistic.**
