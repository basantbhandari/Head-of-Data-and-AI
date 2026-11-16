# Chapter 1. Mathematics, Probability and Statistics

_Part I — Foundations_

Linear algebra, probability, distributions, hypothesis testing and statistical reasoning. Interviewers use these questions to check that a candidate can reason about uncertainty and explain why a method works, not just call a library.

This chapter contains **53 questions**.

## Contents

- **1.1** [Explain the Bias-Variance Tradeoff.](#11-explain-the-bias-variance-tradeoff)
- **1.2** [Explain different probability distributions (Normal, Binomial, Poisson, Uniform).](#12-explain-different-probability-distributions-normal-binomial-poisson-uniform)
- **1.3** [What is the normal distribution and its functions?](#13-what-is-the-normal-distribution-and-its-functions)
- **1.4** [What is an exponential distribution?](#14-what-is-an-exponential-distribution)
- **1.5** [What is the binomial distribution?](#15-what-is-the-binomial-distribution)
- **1.6** [What is the Bernoulli distribution?](#16-what-is-the-bernoulli-distribution)
- **1.7** [What is the multinomial distribution?](#17-what-is-the-multinomial-distribution)
- **1.8** [What is the lognormal distribution?](#18-what-is-the-lognormal-distribution)
- **1.9** [What is a logistic distribution?](#19-what-is-a-logistic-distribution)
- **1.10** [What is the gamma distribution and its functions?](#110-what-is-the-gamma-distribution-and-its-functions)
- **1.11** [What is the Poisson distribution, and what is its probability function?](#111-what-is-the-poisson-distribution-and-what-is-its-probability-function)
- **1.12** [When would you use a Poisson distribution over a Binomial distribution?](#112-when-would-you-use-a-poisson-distribution-over-a-binomial-distribution)
- **1.13** [What is the variance?](#113-what-is-the-variance)
- **1.14** [What is stddev?](#114-what-is-stddev)
- **1.15** [Explain the difference between mean, median, and mode.](#115-explain-the-difference-between-mean-median-and-mode)
- **1.16** [What is the difference between correlation and covariance?](#116-what-is-the-difference-between-correlation-and-covariance)
- **1.17** [What does a correlation coefficient of +1, 0, and -1 indicate?](#117-what-does-a-correlation-coefficient-of-1-0-and--1-indicate)
- **1.18** [Explain Correlation vs. causation.](#118-explain-correlation-vs-causation)
- **1.19** [What are Type I and Type II errors?](#119-what-are-type-i-and-type-ii-errors)
- **1.20** [What is a p-value? What does statistical significance mean?](#120-what-is-a-p-value-what-does-statistical-significance-mean)
- **1.21** [Explain p-values and their limitations.](#121-explain-p-values-and-their-limitations)
- **1.22** [What is hypothesis testing, and when is it used in ML?](#122-what-is-hypothesis-testing-and-when-is-it-used-in-ml)
- **1.23** [What statistical tests would you use to compare two models?](#123-what-statistical-tests-would-you-use-to-compare-two-models)
- **1.24** [How do you assess if a feature is statistically significant?](#124-how-do-you-assess-if-a-feature-is-statistically-significant)
- **1.25** [What is a confidence interval, and how is it used?](#125-what-is-a-confidence-interval-and-how-is-it-used)
- **1.26** [What are z-score and t-score? When do you use each?](#126-what-are-z-score-and-t-score-when-do-you-use-each)
- **1.27** [Explain Bayes' Theorem. How is it relevant to Naive Bayes or Bayesian methods?](#127-explain-bayes-theorem-how-is-it-relevant-to-naive-bayes-or-bayesian-methods)
- **1.28** [What's the difference between MLE and MAP estimation?](#128-whats-the-difference-between-mle-and-map-estimation)
- **1.29** [What is Maximum Likelihood Estimation (MLE)?](#129-what-is-maximum-likelihood-estimation-mle)
- **1.30** [Explain the Bayesian vs. Frequentist approach in statistics.](#130-explain-the-bayesian-vs-frequentist-approach-in-statistics)
- **1.31** [What is the Central Limit Theorem(CLT)? Why is it important?](#131-what-is-the-central-limit-theoremclt-why-is-it-important)
- **1.32** [What are sampling techniques?](#132-what-are-sampling-techniques)
- **1.33** [What is the bootstrap method, and how is it used?](#133-what-is-the-bootstrap-method-and-how-is-it-used)
- **1.34** [What are eigenvalues and eigenvectors?](#134-what-are-eigenvalues-and-eigenvectors)
- **1.35** [What is Selection Bias?](#135-what-is-selection-bias)
- **1.36** [What is the difference between Correlation, Covariance, and Causality?](#136-what-is-the-difference-between-correlation-covariance-and-causality)
- **1.37** [What are the differences between Type I error and Type II error?](#137-what-are-the-differences-between-type-i-error-and-type-ii-error)
- **1.38** [What is P-value?](#138-what-is-p-value)
- **1.39** [How would you define collinearity?](#139-how-would-you-define-collinearity)
- **1.40** [What is the time series?](#140-what-is-the-time-series)
- **1.41** [What is the Fourier Transform?](#141-what-is-the-fourier-transform)
- **1.42** [How can you check for the Normality of a dataset?](#142-how-can-you-check-for-the-normality-of-a-dataset)
- **1.43** [A game exists where you are required to roll two six-faced dice. You stand a chance to win \$50 if the sum of the two dice equals 7. However, you must pay \$4 to play each time you roll both dice. Would you partake in this game? Note: If you play 6 times, what is the probability of making money from this game?](#143-a-game-exists-where-you-are-required-to-roll-two-six-faced-dice-you-stand-a-chance-to-win-50-if-the-sum-of-the-two-dice-equals-7-however-you-must-pay-4-to-play-each-time-you-roll-both-dice-would-you-partake-in-this-game-note-if-you-play-6-times-what-is-the-probability-of-making-money-from-this-game)
- **1.44** [Give an example where the median is a better measure than the mean](#144-give-an-example-where-the-median-is-a-better-measure-than-the-mean)
- **1.45** [What’s the difference between probability and likelihood?](#145-whats-the-difference-between-probability-and-likelihood)
- **1.46** [How will you find the correlation between a categorical variable and a continuous variable?](#146-how-will-you-find-the-correlation-between-a-categorical-variable-and-a-continuous-variable)
- **1.47** [What are the differences between “Bayesian” and “Frequentist” approaches for Machine Learning?](#147-what-are-the-differences-between-bayesian-and-frequentist-approaches-for-machine-learning)
- **1.48** [Describe the differences between and use cases for box plots and histograms](#148-describe-the-differences-between-and-use-cases-for-box-plots-and-histograms)
- **1.49** [What is KL divergence, and how would you define its use case in ML?](#149-what-is-kl-divergence-and-how-would-you-define-its-use-case-in-ml)
- **1.50** [List the most popular distribution curves along with scenarios where you will use them in an algorithm.](#150-list-the-most-popular-distribution-curves-along-with-scenarios-where-you-will-use-them-in-an-algorithm)
- **1.51** [Which metrics can be used to measure correlation of categorical data?](#151-which-metrics-can-be-used-to-measure-correlation-of-categorical-data)
- **1.52** [Where does linear algebra/statistics actually show up in day-to-day LLM/CV engineering work (not just theory)?](#152-where-does-linear-algebrastatistics-actually-show-up-in-day-to-day-llmcv-engineering-work-not-just-theory)
- **1.53** [What is Singular Value Decomposition (SVD)?](#153-what-is-singular-value-decomposition-svd)

---

### 1.1 Explain the Bias-Variance Tradeoff.

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

### 1.2 Explain different probability distributions (Normal, Binomial, Poisson, Uniform).

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
  - Parameters: number of trials $n$ and probability of success $p$.
- **Poisson Distribution**
  - Discrete distribution that models the number of times an **event occurs within a fixed interval** of time or space.
  - Useful when events occur independently at a roughly constant average rate.
  - Example: Number of customer arrivals at a store per hour.
  - Parameter: average event rate $\lambda$.
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

### 1.3 What is the normal distribution and its functions?

##### Normal Distribution

A **normal distribution** is a continuous probability distribution with a **symmetric, bell-shaped curve**. It is widely used to model continuous data such as measurements, errors, and test scores.

- It is **symmetric around the mean**.
- **Mean = Median = Mode**.
- It is determined by:
  - **Mean ($\mu$):** controls the center.
  - **Standard deviation ($\sigma$):** controls the spread.
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

### 1.4 What is an exponential distribution?

##### Exponential Distribution

The **exponential distribution** is a continuous probability distribution used to model the **time until an event occurs** when events happen independently at a constant average rate.

- It is commonly used for **waiting-time problems**.
- Example: Time until the next customer arrives at a store.
- It is controlled by a **rate parameter ($\lambda$)**.
- A larger $\lambda$ means events happen more frequently, so the expected waiting time is shorter.
- It is **right-skewed**: many observations are small, with fewer very large values.
- It has the **memoryless property**, meaning the probability of waiting an additional amount of time does not depend on how long you have already waited.

##### Common Applications

- Time between customer arrivals
- Time until a machine fails
- Network request arrival times
- Waiting time between events

### 1.5 What is the binomial distribution?

##### Binomial Distribution

The **binomial distribution** is a discrete probability distribution that models the **number of successes in a fixed number of independent trials**, where each trial has only two possible outcomes: **success or failure**.

- **Fixed number of trials:** $n$
- **Two possible outcomes:** success or failure
- **Constant probability of success:** $p$
- **Trials are independent** of each other.
- The random variable represents the **number of successes**.

**Example:**

If you flip a coin 10 times, the binomial distribution can determine the probability of getting exactly 6 heads.

##### Applications

- Number of successful predictions out of multiple predictions
- Number of defective products in a sample
- Number of customers who respond to an offer
- Number of heads in repeated coin flips

### 1.6 What is the Bernoulli distribution?

##### Bernoulli Distribution

The **Bernoulli distribution** is a discrete probability distribution that models a **single experiment with only two possible outcomes**:

- **Success:** represented by `1`
- **Failure:** represented by `0`
- The probability of success is **$p$**.
- The probability of failure is **$1 - p$**.

**Example:**

For a single coin toss:

- Heads = success
- Tails = failure

##### Bernoulli vs. Binomial

- **Bernoulli:** One trial.
- **Binomial:** Multiple independent Bernoulli trials and counts the total number of successes.

### 1.7 What is the multinomial distribution?

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

### 1.8 What is the lognormal distribution?

##### Lognormal Distribution

The **lognormal distribution** is a continuous probability distribution for a positive-valued variable whose **natural logarithm follows a normal distribution**.

- The variable can only take **positive values** ($X > 0$).
- It is typically **right-skewed**, with a long tail toward larger values.
- It is useful when values grow through **multiplicative effects** rather than additive effects.
- If $X$ is lognormally distributed, then $\ln(X)$ follows a normal distribution.

##### Examples

- Income and wealth distributions
- Stock prices and financial returns
- Product lifetimes
- File sizes and network traffic
- Biological measurements

##### Normal vs. Lognormal

- **Normal:** Can take positive or negative values and is symmetric.
- **Lognormal:** Only positive values and is usually right-skewed.

### 1.9 What is a logistic distribution?

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

### 1.10 What is the gamma distribution and its functions?

##### Gamma Distribution

The **gamma distribution** is a continuous probability distribution used to model **positive-valued quantities**, especially **waiting time until multiple events occur**.

- It only takes **positive values** ($X > 0$).
- It is generally **right-skewed**, although its shape can vary.
- It is defined by two parameters:
  - **Shape ($\alpha$):** controls the shape of the distribution.
  - **Rate ($\beta$):** controls the rate at which events occur.
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

### 1.11 What is the Poisson distribution, and what is its probability function?

##### Poisson Distribution

The **Poisson distribution** is a discrete probability distribution used to model the **number of times an event occurs within a fixed interval of time, space, or area**, when events occur independently at a constant average rate.

- It is a **discrete distribution**.
- The random variable represents the **number of events**: $0, 1, 2, \ldots$
- It has one parameter, **$\lambda$ (lambda)**, which represents the average number of events in the interval.
- The mean and variance are both **$\lambda$**.
- Examples:
  - Number of customers arriving at a store per hour.
  - Number of calls received by a call center per minute.
  - Number of defects in a manufactured product.

##### Main Function

**Probability Mass Function (PMF)**

The PMF gives the probability of observing exactly **k events** in the interval.

$$
P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}
$$

where:

- $\lambda$ = average event rate
- $k$ = number of events
- $e$ = Euler's number

##### Other Functions

- **CDF:** Gives the probability of observing **k or fewer events**.
- **Mean:** $\lambda$
- **Variance:** $\lambda$
- **Standard deviation:** $\sqrt{\lambda}$

### 1.12 When would you use a Poisson distribution over a Binomial distribution?

##### Poisson vs. Binomial Distribution

Use a **Poisson distribution** when you want to model the **number of events occurring within a fixed interval**, while a **Binomial distribution** is used when you have a **fixed number of trials** with two possible outcomes.

- **Use Binomial when:**
  - The number of trials is fixed ($n$).
  - Each trial has two outcomes: success/failure.
  - The probability of success ($p$) is known or estimated.
  - Example: Number of defective products among **100 inspected products**.
- **Use Poisson when:**
  - You are counting events within a fixed **time, space, area, or volume**.
  - There is no fixed number of trials.
  - Events occur independently at a roughly constant average rate ($\lambda$).
  - Example: Number of customers arriving at a store **per hour**.

##### Simple Rule

> **Fixed number of trials → Binomial**
>
> **Fixed interval and count of events → Poisson**

### 1.13 What is the variance?

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

### 1.14 What is stddev?

##### Standard Deviation (Std Dev)

**Standard deviation** is a measure of how much the values in a dataset **spread out from the mean**.

- **Low standard deviation:** Values are close to the mean.
- **High standard deviation:** Values are more spread out.
- It is the **square root of the variance**.
- Unlike variance, standard deviation has the **same unit as the original data**, making it easier to interpret.

**Example:**

- Mean = 50, standard deviation = 2 → values are generally close to 50.
- Mean = 50, standard deviation = 20 → values are much more spread out.

### 1.15 Explain the difference between mean, median, and mode.

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

### 1.16 What is the difference between correlation and covariance?

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

### 1.17 What does a correlation coefficient of +1, 0, and -1 indicate?

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

### 1.18 Explain Correlation vs. causation.

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

### 1.19 What are Type I and Type II errors?

##### Type I and Type II Errors

- **Type I Error (False Positive):**
  - It predicts positive when the actual is negative
  - Example: A medical test says a person has cancer although that person is normal.
  - Rejecting the null hypothesis when it is actually true.
  - You conclude there is an effect when there is actually no effect.
  - Probability of Type I error = **$\alpha$ (significance level)**.
- **Type II Error (False Negative):**
  - It predicts negative when the actual is positive
  - Example: A medical test says a person is normal although that person has cancer.
  - Failing to reject the null hypothesis when it is actually false.
  - You conclude there is no effect when there actually is one.
  - Probability of Type II error = **$\beta$**.

| Actual / Predicted | Positive | Negative |
| --- | --- | --- |
| **Positive** | True Positive (TP) | **False Negative (FN) → Type II Error** |
| **Negative** | **False Positive (FP) → Type I Error** | True Negative (TN) |

### 1.20 What is a p-value? What does statistical significance mean?

##### P-value

- It is the probability of obtaining a result at least as extreme as the observed result, assuming the null hypothesis is true.
- Null hypothesis, **$H_0$** ⇒ no effect, no difference, or no relationship
- Alternative hypothesis, **$H_1$** ⇒ there is an effect, there is a difference, or there is a relationship
- A **small p-value** means the observed result would be unlikely if $H_0$ were true.
- A **large p-value** means there is not enough evidence to reject $H_0$.
- Common significance level: **$\alpha = 0.05$**.

##### Statistical Significance

A result is **statistically significant** when the p-value is smaller than the chosen significance level $\alpha$.

- **p-value $< 0.05$ → Statistically significant → Reject $H_0$, Accept $H_1$**
- **p-value $\geq 0.05$ → Not statistically significant → Fail to reject $H_0$, Reject $H_1$**

**p-value $< \alpha$ → evidence against $H_0$ → statistically significant.**

### 1.21 Explain p-values and their limitations.

##### P-value

- A **p-value** measures how unusual the observed data would be **assuming the null hypothesis ($H_0$) is true**.
- **Small p-value:** Data provides stronger evidence against $H_0$.
- **Large p-value:** Data does not provide sufficient evidence to reject $H_0$.
- Common significance level: **$\alpha = 0.05$**.
  - **$p < 0.05$ → statistically significant**
  - **$p \geq 0.05$ → not statistically significant**

##### Limitations of P-values

- **Does not measure the size of an effect:** A tiny effect can have a very small p-value with a large sample.
- **Does not measure practical importance:** Statistical significance does not necessarily mean the result is useful or meaningful in practice.
- **Does not give the probability that $H_0$ is true:** A p-value is not $P(H_0 \text{ is true})$.
- **Depends on sample size:** Larger samples can produce very small p-values even for weak effects.
- **Sensitive to study design and assumptions:** Violating statistical assumptions can make p-values misleading.
- **Multiple testing problem:** Testing many hypotheses increases the chance of obtaining a significant result by chance.
- **Not a measure of certainty:** A p-value of 0.01 does not mean there is a 99% probability that the result is correct.

**Short:**

**P-value tells how compatible the data is with $H_0$, but it does not tell you whether an effect is large, important, or certainly true.**

### 1.22 What is hypothesis testing, and when is it used in ML?

##### Hypothesis Testing

- **Hypothesis testing** is a statistical method used to determine whether there is enough evidence in sample data to support a claim about a population.
- It starts with two hypotheses:
  - **Null hypothesis ($H_0$):** No significant effect or difference exists.
  - **Alternative hypothesis ($H_1$):** A significant effect or difference exists.
- A statistical test produces a **p-value**, which is compared with a significance level such as **$\alpha = 0.05$**.
- If **$p < \alpha$**, reject $H_0$.
- If **$p \geq \alpha$**, fail to reject $H_0$.

##### When is it used in ML?

- **Feature selection:** Test whether a feature has a statistically significant relationship with the target.
- **Model comparison:** Determine whether one model significantly outperforms another.
- **A/B testing:** Test whether a new model or feature improves performance compared with the existing version.
- **Model evaluation:** Determine whether observed performance differences are likely due to chance.
- **Experimentation:** Test whether a treatment, intervention, or ML system produces a meaningful effect.

### 1.23 What statistical tests would you use to compare two models?

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

### 1.24 How do you assess if a feature is statistically significant?

##### Assessing Whether a Feature Is Statistically Significant

- **Fit a statistical model**, such as linear or logistic regression, including the feature.
- Examine the feature's **coefficient** and its **p-value**.
- **p-value $< \alpha$** (commonly 0.05) → feature is statistically significant.
- **p-value $\geq \alpha$** → insufficient evidence that the feature has a statistically significant effect, given the model and assumptions.
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

### 1.25 What is a confidence interval, and how is it used?

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

### 1.26 What are z-score and t-score? When do you use each?

##### Z-score

- Measures how many **standard deviations** a value is from the population mean.
- Used when the **population standard deviation ($\sigma$) is known** or when working with a standardized population.
- Follows the **standard normal distribution**.

##### T-score

- Measures how many standard errors a sample estimate is from the hypothesized population value.
- Used when the **population standard deviation is unknown** and must be estimated from the sample.
- Follows a **t-distribution**, which has heavier tails than the normal distribution.
- Especially useful for **small samples**.

##### Z-score vs T-score

|  | Z-score | T-score |
| --- | --- | --- |
| Standard deviation | Population $\sigma$ known | Population $\sigma$ unknown |
| Distribution | Normal | t-distribution |
| Small samples | Less appropriate | Appropriate |
| Large samples | Common | Also approaches normal |

**Short:**

**Use Z when $\sigma$ is known; use T when $\sigma$ is unknown and estimated from the sample.**

### 1.27 Explain Bayes' Theorem. How is it relevant to Naive Bayes or Bayesian methods?

##### Bayes' Theorem

Bayes' Theorem describes how to **update the probability of a hypothesis when new evidence is observed**.

- **Prior probability:** $P(A)$ — probability of the hypothesis before seeing new evidence.
- **Likelihood:** $P(B \mid A)$ — probability of observing the evidence if the hypothesis is true.
- **Posterior probability:** $P(A \mid B)$ — updated probability of the hypothesis after seeing the evidence.
- **Evidence:** $P(B)$ — overall probability of observing the evidence.

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

### 1.28 What's the difference between MLE and MAP estimation?

##### MLE vs MAP Estimation

- **MLE (Maximum Likelihood Estimation)**
  - Finds the parameter values that make the observed data **most likely**.
  - Uses only the observed data.
  - Does not include prior beliefs about the parameters.
  - Objective: maximize $P(\text{Data} \mid \text{Parameters})$.
- **MAP (Maximum A Posteriori) Estimation**
  - Finds the parameter values that are **most probable after considering both data and prior beliefs**.
  - Combines likelihood with a prior distribution.
  - Objective: maximize $P(\text{Parameters} \mid \text{Data})$.

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

### 1.29 What is Maximum Likelihood Estimation (MLE)?

##### Maximum Likelihood Estimation (MLE)

- **MLE** is a statistical method for estimating unknown model parameters from observed data.
- It chooses the parameter values that make the **observed data most likely**.
- It maximizes the **likelihood function**:

  $$
  L(\theta) = P(\text{Data} \mid \theta)
  $$

- The parameter estimate is:

  $$
  \hat{\theta} = \arg\max_{\theta} L(\theta)
  $$

##### Example

Suppose you toss a coin 10 times and get **7 heads**.

- MLE estimates the probability of heads as **$p = 0.7$**.
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

### 1.30 Explain the Bayesian vs. Frequentist approach in statistics.

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

### 1.31 What is the Central Limit Theorem(CLT)? Why is it important?

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

### 1.32 What are sampling techniques?

##### Sampling Techniques

**Sampling** is the process of selecting a subset of a population to study, instead of collecting data from the entire population.

##### 1. Probability Sampling

Every member has a known chance of being selected.

- **Simple Random Sampling:** Every member has an equal chance of selection.
- **Systematic Sampling:** Select every $k$th member after a random starting point.
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

### 1.33 What is the bootstrap method, and how is it used?

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

### 1.34 What are eigenvalues and eigenvectors?

**Eigenvalues and Eigenvectors**

- **Eigenvector:** A non-zero vector whose direction does not change when a matrix transformation is applied to it. Only its magnitude may change.
- **Eigenvalue:** The scalar that tells how much the corresponding eigenvector is stretched or compressed by the transformation.
- They are defined by: $Av = \lambda v$
  - $A$ = square matrix
  - $v$ = eigenvector
  - $\lambda$ = eigenvalue
- **Eigenvalues** are found by solving the characteristic equation: $\det(A - \lambda I) = 0$
- **Eigenvectors** are then found by solving: $(A - \lambda I)v = 0$
- **In ML**, eigenvalues and eigenvectors are important in:
  - PCA and dimensionality reduction
  - Covariance matrix analysis
  - Spectral clustering
  - Graph neural networks
  - Stability analysis
  - Optimization and numerical methods

### 1.35 What is Selection Bias?

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

### 1.36 What is the difference between Correlation, Covariance, and Causality?

#### **Correlation vs Covariance vs Causality**

| Aspect | **Covariance** | **Correlation** | **Causality** |
| --- | --- | --- | --- |
| **Meaning** | Measures whether two variables move together | Measures the **strength and direction** of a linear relationship | Means a change in one variable **causes** a change in another |
| **Direction** | Positive / Negative | Positive / Negative | Cause → Effect |
| **Strength** | Difficult to interpret because scale-dependent | Easy to interpret, from $-1$ to $+1$ | Not represented by a simple fixed range |
| **Range** | $(-\infty, +\infty)$ | $[-1, +1]$ | No fixed range |
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
- **Causality:** No. Ice cream sales do not cause drowning. A **confounding variable**, such as hot weather, can increase both.

#### Key Relationship

**Covariance → standardized → Correlation**

But:

**Correlation ≠ Causality**

### 1.37 What are the differences between Type I error and Type II error?

#### Type I Error vs Type II Error

| Aspect | **Type I Error ($\alpha$)** | **Type II Error ($\beta$)** |
| --- | --- | --- |
| **Meaning** | Rejecting a **true null hypothesis** | Failing to reject a **false null hypothesis** |
| **Also called** | **False Positive** | **False Negative** |
| **Occurs when** | We conclude there is an effect when there isn't one | We conclude there is no sufficient evidence of an effect when one actually exists |
| **Probability** | $\alpha$ (significance level) | $\beta$ |
| **Related concept** | Significance level | Statistical power $= 1 - \beta$ |

#### Simple example

Suppose we test whether a new medicine works:

- **Type I Error:** We conclude the medicine works, but actually it **doesn't**. → **False Positive**
- **Type II Error:** We conclude there isn't enough evidence that it works, but actually it **does**. → **False Negative**

#### Easy way to remember

> **Type I = False Alarm**
>
> **Type II = Missed Detection**

**In short:** Type I error means **detecting something that isn't there**, while Type II error means **missing something that is there**.

### 1.38 What is P-value?

- **P-value** is the probability of obtaining a result **at least as extreme as the observed result**, assuming the **null hypothesis ($H_0$) is true**.
- In simple term: **How surprising is my result if the null hypothesis is actually true?**
- It helps determine whether the observed result provides enough evidence to **reject the null hypothesis**.
- A **small p-value** indicates stronger evidence against $H_0$.
- Common significance level: **$\alpha = 0.05$**

  - **p-value $< 0.05$:** Reject $H_0$
  - **p-value $\geq 0.05$:** Fail to reject $H_0$
- A p-value **does not** represent the probability that the null hypothesis is true.
- It is **not** the probability that your test result is a random fluke.

### 1.39 How would you define collinearity?

- **Collinearity** occurs when **two or more independent variables in a dataset are highly correlated with each other**.
- In a regression model, this can make it difficult to determine the **individual effect of each independent variable** on the target variable.
- High collinearity can lead to:
  - Unstable model coefficients
  - Large standard errors
  - Difficulty interpreting the importance of individual features
- **Example:** If a regression model uses both **age** and **years of experience**, these variables may be highly correlated, creating collinearity.

### 1.40 What is the time series?

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

### 1.41 What is the Fourier Transform?

#### Fourier Transform

- **Fourier Transform** is a mathematical technique used to convert a signal from the **time domain** into the **frequency domain**.
- It tells us **which frequencies are present in a signal and how strong they are**.
- It is commonly used for **audio processing, image processing, signal processing, communications, and time-series analysis**.
- For example, an audio signal that looks complex in the time domain can be broken down into its individual **frequency components**.
- The **Fast Fourier Transform (FFT)** is an efficient algorithm used to calculate the Fourier Transform.

### 1.42 How can you check for the Normality of a dataset?

#### Methods to Check Normality of a Dataset

- **Histogram:** Check whether the data looks approximately **bell-shaped and symmetric**.
- **Q-Q Plot:** If the points are close to a straight line, the data is approximately normally distributed.
- **Shapiro-Wilk Test:** Statistical test commonly used to check normality.
- **Anderson-Darling Test:** Tests how well the data fits a normal distribution.
- **Kolmogorov-Smirnov Test:** Compares the data distribution with a theoretical normal distribution.
- **D’Agostino Skewness Test:** Checks whether the data has **significant skewness**.
- **Skewness and Kurtosis:** Values close to those expected for a normal distribution suggest approximate normality.

### 1.43 A game exists where you are required to roll two six-faced dice. You stand a chance to win \$50 if the sum of the two dice equals 7. However, you must pay \$4 to play each time you roll both dice. Would you partake in this game? Note: If you play 6 times, what is the probability of making money from this game?

#### Should You Play the Game?

- There are **36 possible outcomes** when rolling two dice.
- A sum of **7** can occur in **6 ways**:
  - (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)
- Therefore, the probability of winning on one roll is **$\frac{6}{36} = \frac{1}{6}$**.
- You win **\$50** and pay **\$4** per game.

For **6 games**:

- Total cost = 6 × \$4 = **\$24**
- Let **X** = number of wins.
- You make money when your winnings exceed \$24:
  - 1 win → \$50 − \$24 = **\$26 profit**
  - So you need **at least 1 win**.

Probability of making money:

- Probability of no wins in 6 rolls = **$(5/6)^6 \approx 0.335$**
- Probability of at least one win = **$1 - (5/6)^6 \approx 0.665$**

#### Final Answer

**Probability of making money after 6 plays ≈ 66.5%.**

So, based purely on the probability of making money in 6 plays, **yes, I would play**.

### 1.44 Give an example where the median is a better measure than the mean

The **median is better when the distribution is skewed or has outliers**, because the mean is pulled toward extreme values while the median depends only on the middle observation.

**Classic example — income:** salaries in a team of ten are 40k–60k, plus a CEO at 5M. The mean is ~550k, which describes nobody; the median (~50k) represents a "typical" employee. That is why governments report **median household income** and **median house prices**.

**Other examples**
- **Latency / response times:** a few slow requests inflate the mean; teams report p50 (the median) alongside p95/p99.
- **Time-on-site, order values, insurance claims, hospital stays** — all long-tailed.
- **Imputing missing values** in a skewed feature: median imputation is more robust than mean imputation.
- **Data with recording errors** (e.g., an age of 999).

**When the mean is still preferred:** roughly symmetric data without outliers; when you need totals ($\text{total revenue} = \text{mean} \times \text{count}$); or when extremes genuinely matter (expected cost in insurance).

**Practice:** look at the histogram or box plot, report both mean and median when they differ, and consider a log transform for heavily skewed features.

### 1.45 What’s the difference between probability and likelihood?

They use the same function $p(x \mid \theta)$, read in two different directions.

- **Probability:** fix the parameters $\theta$ and ask how likely different *data* outcomes are. "Given a fair coin ($p = 0.5$), what is the chance of 7 heads in 10 flips?" Probabilities over outcomes sum or integrate to 1.
- **Likelihood:** fix the observed *data* and treat it as a function of the parameters: $L(\theta \mid x) = p(x \mid \theta)$. "Given I saw 7 heads in 10 flips, how plausible is $p = 0.5$ vs $p = 0.7$?" Likelihood is **not** a probability distribution over $\theta$ and need not sum to 1.

**In ML:**
- **Maximum Likelihood Estimation:** $\hat\theta = \arg\max_\theta \prod_i p(x_i \mid \theta)$, in practice by maximizing the **log-likelihood** (sums are numerically stable). Minimizing cross-entropy or MSE equals MLE under Bernoulli or Gaussian noise assumptions.
- For a probability over parameters, use Bayes: posterior $\propto$ likelihood $\times$ prior (MAP / Bayesian inference).

**Example:** 7 heads in 10 flips gives $L(0.7) = \binom{10}{7}0.7^7 0.3^3 \approx 0.267$ and $L(0.5) \approx 0.117$, so $p = 0.7$ is more likely given the data. The MLE is $\hat p = 0.7$.

### 1.46 How will you find the correlation between a categorical variable and a continuous variable?

It depends on whether the categorical variable is binary, nominal or ordinal:

- **Binary categorical (2 levels):** use the **point-biserial correlation**, which is Pearson correlation with the category coded 0/1. Ranges from -1 to 1.
- **Nominal with k levels:** use the **correlation ratio $\eta$ (eta)**, derived from ANOVA: the share of variance in the continuous variable explained by group membership.

  $\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}$, ranging 0–1. Test significance with a one-way **ANOVA F-test** (or **Kruskal-Wallis** if the data is non-normal).
- **Ordinal categorical:** use **Spearman's rank correlation** or **Kendall's tau**, which respect order without assuming equal spacing.
- **Model-based alternatives:** mutual information (captures nonlinear dependence), or the AUC of predicting the binary category from the continuous variable.

```python
import pandas as pd
def correlation_ratio(cat, num):
    df = pd.DataFrame({"c": cat, "y": num})
    grand = df.y.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in df.groupby("c").y)
    ss_total = ((df.y - grand) ** 2).sum()
    return (ss_between / ss_total) ** 0.5
```

**Interview caveats:** don't just label-encode a nominal variable and compute Pearson, since the result depends on the arbitrary codes. Association isn't causation, and with large samples tiny effects become statistically significant, so report the effect size ($\eta$), not just the p-value.

### 1.47 What are the differences between “Bayesian” and “Frequentist” approaches for Machine Learning?

The two schools differ in **what probability means**.

| | Frequentist | Bayesian |
| --- | --- | --- |
| Probability | Long-run frequency of repeated events | Degree of belief, updated with evidence |
| Parameters | Fixed but unknown constants | Random variables with distributions |
| Prior knowledge | Not formally used | Encoded in a **prior** $p(\theta)$ |
| Output | Point estimate (e.g. MLE) + confidence interval / p-value | Full **posterior** $p(\theta \mid D) \propto p(D \mid \theta)p(\theta)$ + credible interval |
| Typical ML | MLE, standard SGD-trained nets, hypothesis tests | Bayesian linear regression, Gaussian processes, Bayesian optimization, MCMC/variational inference |

**Links in practice:**

- **MAP estimation** = MLE + prior. L2 regularization equals a Gaussian prior on the weights; L1 equals a Laplace prior. So most regularized ML is "Bayesian-flavoured".
- With lots of data the likelihood dominates and both give similar answers.

**Tradeoffs:**

- **Bayesian pros:** principled uncertainty estimates, natural use of prior knowledge, works well with small data, good for sequential updating (A/B tests, bandits like Thompson sampling).
- **Bayesian cons:** priors can be subjective; exact posteriors are usually intractable, so MCMC or variational approximations are slow or approximate at scale.
- **Frequentist pros:** simpler, scalable, objective procedures, well-established tests.
- **Frequentist cons:** confidence intervals and p-values are easy to misinterpret; no direct probability statements about parameters.

**Production example:** Bayesian A/B testing gives "$P(\text{variant B is better}) = 94\%$", which is often easier for stakeholders than a p-value.

### 1.48 Describe the differences between and use cases for box plots and histograms

Both show the distribution of a numeric variable, at different levels of detail.

**Histogram**

- Bins the values and shows the count (or density) in each bin.
- Reveals the **shape**: skewness, multiple modes (bimodality), gaps, heavy tails.
- Sensitive to bin width choice; awkward for comparing many groups.
- **Use when:** exploring one variable in depth, checking normality before modeling, spotting bimodal populations (e.g. two customer types), choosing transformations (log for right skew).

**Box plot**

- Five-number summary: median, Q1, Q3 (the box = IQR), whiskers (typically to $1.5 \times \text{IQR}$) and individual **outliers** beyond them.
- Compact and robust; makes **comparing many groups side by side** easy.
- Hides shape: a bimodal and a unimodal distribution can produce identical boxes.
- **Use when:** comparing a metric across categories (latency per region, salary per department), quickly flagging outliers, dashboards with limited space.

**Practical tip:** violin plots or box plots overlaid with strip/jitter points combine both views. In EDA I'd use histograms per feature to understand shape and box plots of the feature by target class to see which features separate classes.

### 1.49 What is KL divergence, and how would you define its use case in ML?

**KL divergence** measures how much one probability distribution $P$ differs from a reference distribution $Q$:

$$
D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

Interpretation: the expected extra bits (nats) needed to encode samples from $P$ using a code optimized for $Q$.

**Properties:** always $\ge 0$, zero only if $P = Q$; **not symmetric** ($D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$), so it's not a true distance; infinite if $Q(x) = 0$ where $P(x) > 0$. Forward KL is "mean-seeking" (covers all modes); reverse KL is "mode-seeking".

**Use cases in ML:**

- **Cross-entropy loss:** $H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$, so minimizing cross-entropy against fixed labels minimizes KL.
- **VAEs:** the KL term pushes the encoder's latent distribution toward the prior $\mathcal{N}(0, I)$.
- **Knowledge distillation:** KL between teacher and student softened output distributions.
- **RLHF / PPO / DPO:** a KL penalty keeps the fine-tuned LLM close to the reference model, preventing reward hacking and drift.
- **Variational inference:** minimize $D_{\text{KL}}(q \parallel p)$ to approximate posteriors.
- **Drift monitoring:** compare feature or prediction distributions between training and production (often with the symmetric, bounded Jensen-Shannon divergence or PSI instead).
- **t-SNE:** minimizes KL between high- and low-dimensional neighbour distributions.

```python
import numpy as np
def kl(p, q, eps=1e-12):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    return float(np.sum(p * np.log(p / q)))
```

### 1.50 List the most popular distribution curves along with scenarios where you will use them in an algorithm.

| Distribution | Type | Describes | Where it shows up in ML |
| --- | --- | --- | --- |
| **Normal (Gaussian)** | Continuous | Sums of many small effects (CLT) | Linear regression residuals, weight initialization, GMMs, Gaussian Naive Bayes, Gaussian processes, VAE latent prior, anomaly detection via z-scores |
| **Bernoulli** | Discrete | One yes/no trial | Binary classification output (logistic regression, sigmoid + BCE), dropout masks |
| **Binomial** | Discrete | Number of successes in $n$ trials | Conversion counts, A/B test analysis |
| **Categorical / Multinomial** | Discrete | One of $K$ outcomes / counts over $K$ | Softmax classifiers, LLM next-token sampling, Multinomial Naive Bayes for text |
| **Poisson** | Discrete | Count of events per interval | Modeling calls/orders per hour, Poisson regression, demand forecasting |
| **Exponential** | Continuous | Time between Poisson events | Survival/time-to-event, queueing and latency modeling |
| **Uniform** | Both | Equal likelihood | Random initialization, random search for hyperparameters, simulation |
| **Beta** | Continuous on $[0, 1]$ | Distribution of a probability | Bayesian A/B testing, Thompson sampling for bandits |
| **Dirichlet** | Multivariate on simplex | Distribution over probability vectors | Topic models (LDA), Bayesian priors for categorical data |
| **Gamma** | Continuous, positive | Waiting times, positive skewed values | Insurance claim sizes, Bayesian priors for precision |
| **Student's t** | Continuous, heavy tails | Small-sample means | t-tests, t-SNE's low-dimensional similarities, robust regression |
| **Log-normal / power law (Zipf)** | Continuous / discrete | Skewed positive values / long tails | Income, prices, latency; word frequencies, item popularity in recommenders |

**Why it matters:** the assumed distribution determines the right loss (MSE ↔ Gaussian, cross-entropy ↔ Bernoulli/categorical, Poisson deviance ↔ counts), the right transformation (log for log-normal data) and the right statistical test.

### 1.51 Which metrics can be used to measure correlation of categorical data?

**Categorical vs. categorical (nominal):**

- **Chi-square test of independence:** tests whether an association exists (gives a p-value, not strength).
- **Cramér's V:** effect size from chi-square, range 0–1, works for any table size (use the bias-corrected version for small samples):

  $$
  V = \sqrt{\frac{\chi^2/n}{\min(r-1, c-1)}}
  $$

- **Phi coefficient ($\phi$):** for 2×2 tables; equals Pearson on two binary variables.
- **Contingency coefficient** and **Tschuprow's T:** older alternatives to Cramér's V.
- **Theil's U (uncertainty coefficient):** asymmetric, based on entropy: how much knowing X reduces uncertainty about Y. Useful because association can be directional.
- **Mutual information:** general, captures any dependence; normalized MI for comparability.
- **Fisher's exact test:** instead of chi-square when expected cell counts are small (< 5).

**Ordinal vs. ordinal:** Spearman's rho, Kendall's tau-b, Goodman-Kruskal gamma.

**Categorical vs. continuous:** point-biserial (binary), correlation ratio $\eta$ / ANOVA.

```python
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
def cramers_v(x, y):
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t)[0]
    n = t.values.sum()
    return np.sqrt(chi2 / n / (min(t.shape) - 1))
```

**Caveat:** with large samples chi-square is almost always significant, so report an effect size like Cramér's V. Label-encoding nominal categories and computing Pearson is meaningless.

### 1.52 Where does linear algebra/statistics actually show up in day-to-day LLM/CV engineering work (not just theory)?

Linear algebra: embeddings are vectors, similarity search relies on dot product/cosine similarity, attention mechanisms are matrix multiplications, and dimensionality reduction (PCA) is used for visualizing/analyzing embeddings. Statistics/probability: evaluating model outputs (confidence calibration, sampling temperature/top-p in generation), A/B testing changes to a RAG pipeline or prompt, understanding precision/recall/F1 and ROC curves for CV classifiers, and reasoning about uncertainty in retrieval relevance scores. Calculus underlies backpropagation/gradient descent when fine-tuning models — understanding learning rate effects, loss landscapes, and why training might diverge.

### 1.53 What is Singular Value Decomposition (SVD)?

> **See also:** [Question 17.72 — What is SVD (singular value decomposition)?](../part-4-generative-ai/17-large-language-model-foundations.md#1772-what-is-svd-singular-value-decomposition), which covers the same question from another angle.

**Singular Value Decomposition (SVD)** is a matrix factorization technique that decomposes a matrix into three matrices:

$$
A = U\Sigma V^T
$$

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
   - Approximates a large matrix using only its top $k$ singular values: $A \approx U_k \Sigma_k V_k^T$
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

$$
A^T= \begin{bmatrix} 3&1\\ 2&4 \end{bmatrix}
$$

##### 2. Calculate $AA^T$

$$
AA^T= \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix} \begin{bmatrix} 3&1\\ 2&4 \end{bmatrix} = \begin{bmatrix} 13&11\\ 11&17 \end{bmatrix}
$$

Find its eigenvalues:

$$
\det \begin{bmatrix} 13-\lambda&11\\ 11&17-\lambda \end{bmatrix}=0
$$

$$
(13-\lambda)(17-\lambda)-121=0
$$

$$
\lambda^2-30\lambda+100=0
$$

Therefore:

$$
\lambda_1=15+5\sqrt5\approx26.180
$$

$$
\lambda_2=15-5\sqrt5\approx3.820
$$

---

##### 3. Calculate singular values

Singular values are: $\sigma_i=\sqrt{\lambda_i}$

Therefore:

$$
\sigma_1=\sqrt{26.180}\approx5.117
$$

$$
\sigma_2=\sqrt{3.820}\approx1.955
$$

So: $\boxed{ \Sigma= \begin{bmatrix} 5.117&0\\ 0&1.955 \end{bmatrix}}$

---

##### 4. Find $U$

The columns of $U$ are the **normalized eigenvectors of $AA^T$**.

For $\lambda_1$: $\begin{bmatrix} 13-26.180&11\\ 11&17-26.180 \end{bmatrix} \begin{bmatrix} x\\y \end{bmatrix}=0$

This gives approximately: $y=1.198x$

After normalization: $u_1= \begin{bmatrix} 0.641\\ 0.768 \end{bmatrix}$

For $\lambda_2$: $u_2= \begin{bmatrix} 0.768\\ -0.641 \end{bmatrix}$

Therefore: $\boxed{ U= \begin{bmatrix} 0.641&0.768\\ 0.768&-0.641 \end{bmatrix}}$

---

##### 5. Find $V$

Now calculate: $A^TA= \begin{bmatrix} 3&1\\ 2&4 \end{bmatrix} \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix} = \begin{bmatrix} 10&10\\ 10&20 \end{bmatrix}$

The eigenvalues are the same: $26.180,\quad3.820$

For $\lambda_1=26.180$:

$$
(10-26.180)x+10y=0
$$

$$
y=1.618x
$$

After normalization: $v_1= \begin{bmatrix} 0.526\\ 0.851 \end{bmatrix}$

For $\lambda_2=3.820$: $v_2= \begin{bmatrix} 0.851\\ -0.526 \end{bmatrix}$

Thus: $\boxed{ V= \begin{bmatrix} 0.526&0.851\\ 0.851&-0.526 \end{bmatrix}}$

Therefore: $\boxed{ V^T= \begin{bmatrix} 0.526&0.851\\ 0.851&-0.526 \end{bmatrix}}$

---

#### 6. Final SVD

Putting everything together: $\boxed{ A=U\Sigma V^T }$

$$
\boxed{ \begin{bmatrix} 3&2\\ 1&4 \end{bmatrix} \approx \begin{bmatrix} 0.641&0.768\\ 0.768&-0.641 \end{bmatrix} \begin{bmatrix} 5.117&0\\ 0&1.955 \end{bmatrix} \begin{bmatrix} 0.526&0.851\\ 0.851&-0.526 \end{bmatrix} }
$$

If you multiply these three matrices, you recover approximately: $\begin{bmatrix} 3&2\\ 1&4 \end{bmatrix}$

##### The whole process to memorize

$$
\boxed{A\rightarrow AA^T\rightarrow \text{eigenvectors}\rightarrow U}
$$

$$
\boxed{A\rightarrow A^TA\rightarrow \text{eigenvectors}\rightarrow V}
$$

$$
\boxed{\text{eigenvalues}\rightarrow\sqrt{\text{eigenvalues}}\rightarrow\Sigma}
$$

So the key relationship is: $\boxed{A=U\Sigma V^T}$

**AI connection:** Once you have $U, \Sigma, V^T$, you can keep only the largest $k$ singular values to create a low-rank approximation: $A\approx U_k\Sigma_kV_k^T$

This is the core idea behind **SVD-based dimensionality reduction, compression, LSA, recommendation systems, and low-rank neural-network compression**.

---

[Table of contents](../README.md) · [Chapter 2. Python Engineering](02-python-engineering.md) →
