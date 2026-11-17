# Chapter 1. Mathematics, Probability and Statistics

_Part I — Foundations_

Linear algebra, probability, distributions, hypothesis testing, power and multiple testing, Bayesian and frequentist inference, and the statistical reasoning mistakes that mislead business decisions. Interviewers use these questions to check that a candidate can reason about uncertainty and explain why a method works, not just call a library.

This chapter contains **49 questions**.

## Contents

- **1.1** [Explain the bias-variance trade-off.](#11-explain-the-bias-variance-trade-off)
- **1.2** [What are the most common probability distributions, and where do they appear in machine learning?](#12-what-are-the-most-common-probability-distributions-and-where-do-they-appear-in-machine-learning)
- **1.3** [What is the normal distribution and its functions?](#13-what-is-the-normal-distribution-and-its-functions)
- **1.4** [What is the exponential distribution?](#14-what-is-the-exponential-distribution)
- **1.5** [What is the binomial distribution?](#15-what-is-the-binomial-distribution)
- **1.6** [What is the Bernoulli distribution?](#16-what-is-the-bernoulli-distribution)
- **1.7** [What is the multinomial distribution?](#17-what-is-the-multinomial-distribution)
- **1.8** [What is the lognormal distribution?](#18-what-is-the-lognormal-distribution)
- **1.9** [What is a logistic distribution?](#19-what-is-a-logistic-distribution)
- **1.10** [What is the gamma distribution and its functions?](#110-what-is-the-gamma-distribution-and-its-functions)
- **1.11** [What is the Poisson distribution, and what is its probability function?](#111-what-is-the-poisson-distribution-and-what-is-its-probability-function)
- **1.12** [When would you use a Poisson distribution over a Binomial distribution?](#112-when-would-you-use-a-poisson-distribution-over-a-binomial-distribution)
- **1.13** [What are variance and standard deviation?](#113-what-are-variance-and-standard-deviation)
- **1.14** [Explain the difference between mean, median, and mode.](#114-explain-the-difference-between-mean-median-and-mode)
- **1.15** [What is the difference between covariance and correlation, and what do correlation values of +1, 0 and -1 mean?](#115-what-is-the-difference-between-covariance-and-correlation-and-what-do-correlation-values-of-1-0-and--1-mean)
- **1.16** [Explain correlation versus causation.](#116-explain-correlation-versus-causation)
- **1.17** [What are Type I and Type II errors, and how do they relate to statistical power?](#117-what-are-type-i-and-type-ii-errors-and-how-do-they-relate-to-statistical-power)
- **1.18** [What is a p-value, what does statistical significance mean, and what are the limitations of p-values?](#118-what-is-a-p-value-what-does-statistical-significance-mean-and-what-are-the-limitations-of-p-values)
- **1.19** [What is hypothesis testing, and when is it used in ML?](#119-what-is-hypothesis-testing-and-when-is-it-used-in-ml)
- **1.20** [What statistical tests would you use to compare two models?](#120-what-statistical-tests-would-you-use-to-compare-two-models)
- **1.21** [How do you assess if a feature is statistically significant?](#121-how-do-you-assess-if-a-feature-is-statistically-significant)
- **1.22** [What is a confidence interval, and how is it used?](#122-what-is-a-confidence-interval-and-how-is-it-used)
- **1.23** [What are z-score and t-score? When do you use each?](#123-what-are-z-score-and-t-score-when-do-you-use-each)
- **1.24** [Explain Bayes' theorem. How is it relevant to Naive Bayes and Bayesian methods?](#124-explain-bayes-theorem-how-is-it-relevant-to-naive-bayes-and-bayesian-methods)
- **1.25** [What are maximum likelihood estimation (MLE) and maximum a posteriori (MAP) estimation, and how do they differ?](#125-what-are-maximum-likelihood-estimation-mle-and-maximum-a-posteriori-map-estimation-and-how-do-they-differ)
- **1.26** [How do the Bayesian and frequentist approaches differ, in statistics and in machine learning practice?](#126-how-do-the-bayesian-and-frequentist-approaches-differ-in-statistics-and-in-machine-learning-practice)
- **1.27** [What is the Central Limit Theorem(CLT)? Why is it important?](#127-what-is-the-central-limit-theoremclt-why-is-it-important)
- **1.28** [What are sampling techniques?](#128-what-are-sampling-techniques)
- **1.29** [What is the bootstrap, and how do you build bootstrap confidence intervals?](#129-what-is-the-bootstrap-and-how-do-you-build-bootstrap-confidence-intervals)
- **1.30** [What are eigenvalues and eigenvectors?](#130-what-are-eigenvalues-and-eigenvectors)
- **1.31** [What is Selection Bias?](#131-what-is-selection-bias)
- **1.32** [What is collinearity, and why does it matter?](#132-what-is-collinearity-and-why-does-it-matter)
- **1.33** [What is a time series, and what makes time-series data different?](#133-what-is-a-time-series-and-what-makes-time-series-data-different)
- **1.34** [What is the Fourier transform, and where is it used in data science and ML?](#134-what-is-the-fourier-transform-and-where-is-it-used-in-data-science-and-ml)
- **1.35** [How can you check whether a data set is normally distributed, and when does it matter?](#135-how-can-you-check-whether-a-data-set-is-normally-distributed-and-when-does-it-matter)
- **1.36** [You can roll two dice for \$4 and win \$50 if the sum is 7. Would you play, and what is the probability of making money if you play six times?](#136-you-can-roll-two-dice-for-4-and-win-50-if-the-sum-is-7-would-you-play-and-what-is-the-probability-of-making-money-if-you-play-six-times)
- **1.37** [Give an example where the median is a better measure than the mean](#137-give-an-example-where-the-median-is-a-better-measure-than-the-mean)
- **1.38** [What’s the difference between probability and likelihood?](#138-whats-the-difference-between-probability-and-likelihood)
- **1.39** [How will you find the correlation between a categorical variable and a continuous variable?](#139-how-will-you-find-the-correlation-between-a-categorical-variable-and-a-continuous-variable)
- **1.40** [Describe the differences between and use cases for box plots and histograms](#140-describe-the-differences-between-and-use-cases-for-box-plots-and-histograms)
- **1.41** [What is KL divergence, and how would you define its use case in ML?](#141-what-is-kl-divergence-and-how-would-you-define-its-use-case-in-ml)
- **1.42** [Which metrics can be used to measure correlation of categorical data?](#142-which-metrics-can-be-used-to-measure-correlation-of-categorical-data)
- **1.43** [Where do linear algebra, probability and statistics actually show up in day-to-day LLM and computer vision engineering work?](#143-where-do-linear-algebra-probability-and-statistics-actually-show-up-in-day-to-day-llm-and-computer-vision-engineering-work)
- **1.44** [What is Singular Value Decomposition (SVD)?](#144-what-is-singular-value-decomposition-svd)
- **1.45** [What is statistical power, and how do you calculate the sample size for an experiment?](#145-what-is-statistical-power-and-how-do-you-calculate-the-sample-size-for-an-experiment)
- **1.46** [What is the multiple testing problem, and how do you control the false discovery rate?](#146-what-is-the-multiple-testing-problem-and-how-do-you-control-the-false-discovery-rate)
- **1.47** [What are heavy-tailed distributions, and why can the mean mislead?](#147-what-are-heavy-tailed-distributions-and-why-can-the-mean-mislead)
- **1.48** [What are the most common statistical mistakes in business analysis, and how do you avoid them?](#148-what-are-the-most-common-statistical-mistakes-in-business-analysis-and-how-do-you-avoid-them)
- **1.49** [How do you calculate confidence intervals for ratio metrics and clustered data, and what is the delta method?](#149-how-do-you-calculate-confidence-intervals-for-ratio-metrics-and-clustered-data-and-what-is-the-delta-method)

---

### 1.1 Explain the bias-variance trade-off.

The expected prediction error of a model on new data splits into three parts: **bias** (error from wrong or too-simple assumptions), **variance** (error from sensitivity to the particular training sample) and **irreducible noise**. Making a model more flexible usually lowers bias and raises variance, so the goal is the complexity that minimises their sum on unseen data.

For squared error, at a point $x$ with true function $f$ and noise variance $\sigma^2$:

$$
E\left[(y - \hat f(x))^2\right] = \left(E[\hat f(x)] - f(x)\right)^2 + \operatorname{Var}(\hat f(x)) + \sigma^2
$$

| | High bias | High variance |
| --- | --- | --- |
| Symptom | Underfitting: training and validation error both high | Overfitting: low training error, much higher validation error |
| Example | Linear model on a strongly non-linear problem | Deep unpruned tree, 1-nearest neighbour |
| Fixes | More features, more flexible model, less regularisation, boosting | More data, regularisation, simpler model, bagging, early stopping |

**Practical points**

- Diagnose with learning curves: if training and validation errors converge at a high value, more data will not help (bias); a persistent gap means variance, and more data or regularisation will help.
- Ensembles attack each side: bagging and random forests reduce variance, boosting reduces bias (see 9.7).
- Modern over-parameterised networks can generalise well despite fitting training data perfectly (double descent, see 11.30), so the classical U-shaped curve is a guide, not a law.

See 8.6 for the full machine learning treatment.

### 1.2 What are the most common probability distributions, and where do they appear in machine learning?

A probability distribution describes how likely each value of a random variable is. Knowing the common ones matters because the assumed distribution of the data or target decides the right loss function, transformation and statistical test.

| Distribution | Type | Describes | Where it shows up in ML |
| --- | --- | --- | --- |
| **Normal (Gaussian)** | Continuous | Sums of many small effects (central limit theorem) | Regression residuals, weight initialisation, Gaussian mixtures, Gaussian Naive Bayes, Gaussian processes, VAE prior, z-score anomaly detection |
| **Bernoulli** | Discrete | One yes or no trial | Binary classification output (sigmoid with binary cross-entropy), dropout masks |
| **Binomial** | Discrete | Number of successes in $n$ trials | Conversion counts, A/B test analysis |
| **Categorical or multinomial** | Discrete | One of $K$ outcomes, or counts over $K$ | Softmax classifiers, LLM next-token sampling, multinomial Naive Bayes |
| **Poisson** | Discrete | Count of events per interval | Orders or calls per hour, Poisson regression, demand forecasting |
| **Exponential** | Continuous | Time between Poisson events | Time-to-event, queueing and latency |
| **Uniform** | Both | Equally likely outcomes | Random initialisation, random hyperparameter search, simulation |
| **Beta** | Continuous on $[0, 1]$ | Uncertainty about a probability | Bayesian A/B testing, Thompson sampling |
| **Dirichlet** | Multivariate on the simplex | Uncertainty about a probability vector | Topic models (LDA), priors for categorical data |
| **Gamma** | Continuous, positive | Waiting time for several events, positive skewed values | Insurance claims, priors for precision or rates |
| **Student's t** | Continuous, heavy tails | Means of small samples | t-tests, t-SNE, robust regression |
| **Log-normal, power law (Zipf)** | Continuous or discrete | Skewed positive values, long tails | Income, prices, latency; word frequencies, item popularity (see 1.47) |

**Quick distinctions**

- Fixed number of trials and a count of successes: binomial. Count of events in an interval with no fixed number of trials: Poisson (see 1.12).
- Single trial: Bernoulli; many categories: categorical or multinomial.
- Waiting time for one event: exponential; for several: gamma.

**Why it matters:** the distribution maps to the loss (squared error assumes Gaussian noise, cross-entropy assumes Bernoulli or categorical, Poisson deviance suits counts), to the transformation (log for log-normal data) and to the test (t-test for means of roughly normal data, chi-square for counts). Questions 1.3 to 1.11 cover the individual distributions.

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

### 1.4 What is the exponential distribution?

The exponential distribution models the **waiting time until the next event** when events occur independently at a constant average rate $\lambda$ (a Poisson process). It is the continuous counterpart of "time between Poisson arrivals".

**Formulas** for $x \ge 0$:

$$
f(x) = \lambda e^{-\lambda x}, \qquad F(x) = P(X \le x) = 1 - e^{-\lambda x}
$$

- Mean $1/\lambda$, variance $1/\lambda^2$, so the standard deviation equals the mean.
- Right-skewed: many short waits, a few long ones. Median is $\ln 2 / \lambda$, below the mean.

**Memoryless property:** $P(X \gt s + t \mid X \gt s) = P(X \gt t)$. Having waited 10 minutes tells you nothing about how much longer you will wait. It is the only continuous distribution with this property, which is exactly why it is often a poor model for component lifetimes that wear out.

**Example:** if a support queue receives 12 tickets per hour ($\lambda = 0.2$ per minute), the probability that the next ticket arrives within 5 minutes is $1 - e^{-1} \approx 0.63$.

**Applications:** time between customer arrivals or requests, queueing models, time to failure for components with a constant hazard rate, survival analysis baselines.

**Related distributions:** the sum of $k$ independent exponentials is gamma distributed (see 1.10); Weibull generalises it to increasing or decreasing hazard rates, which is more realistic for wear-out or early failures.

### 1.5 What is the binomial distribution?

The binomial distribution gives the probability of observing $k$ successes in $n$ **independent** trials when each trial succeeds with the same probability $p$. It is the sum of $n$ independent Bernoulli trials.

**Probability mass function**

$$
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \qquad k = 0, 1, \ldots, n
$$

- Mean $np$, variance $np(1-p)$.
- Assumptions: fixed $n$, two outcomes per trial, constant $p$, independence.

**Example:** a model is 80 percent accurate; on 10 independent cases, the probability it gets exactly 8 right is $\binom{10}{8}0.8^8 0.2^2 \approx 0.30$.

**Approximations**

- **Normal:** when $np$ and $n(1-p)$ are both at least about 10, $X$ is approximately normal with mean $np$ and variance $np(1-p)$. This underlies the usual z-test for conversion rates in A/B tests.
- **Poisson:** when $n$ is large and $p$ small, $X$ is approximately Poisson with $\lambda = np$ (see 1.12).

**Applications:** conversion counts in experiments, defect counts in a sample, number of correct predictions on a test set (which gives a confidence interval for accuracy).

**Pitfall:** independence and constant $p$ often fail in practice. Clicks from the same user are correlated, so treating every page view as an independent trial understates the variance; analyse at the level of randomisation (the user) instead (see 1.49).

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

- The variable can only take **positive values** ($X \gt 0$).
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

- It only takes **positive values** ($X \gt 0$).
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

**Link between them:** when $n$ is large and $p$ is small, Binomial($n$, $p$) is well approximated by Poisson with $\lambda = np$ (a rule of thumb is $n \ge 20$ and $p \le 0.05$). For example, the number of fraudulent transactions among 100,000 with a fraud rate of 0.01 percent is approximately Poisson with $\lambda = 10$.

**Check the assumption:** real count data are often **overdispersed** (variance larger than the mean) because events cluster, for example a few heavy users generate many orders. Then use a negative binomial model instead of Poisson.

### 1.13 What are variance and standard deviation?

Both measure how spread out values are around the mean. **Variance** is the average squared deviation from the mean; **standard deviation** is its square root, which brings the measure back to the original units.

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2, \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar x)^2
$$

- $\sigma^2$ is the population variance; $s^2$ is the sample estimate. Dividing by $n - 1$ (Bessel's correction) makes $s^2$ unbiased, because deviations from the sample mean are slightly smaller than deviations from the true mean.
- Standard deviation $\sigma = \sqrt{\sigma^2}$ has the same unit as the data. For roughly normal data, about 68, 95 and 99.7 percent of values lie within one, two and three standard deviations of the mean.

**Example:** `48, 49, 50, 51, 52` and `10, 30, 50, 70, 90` both have mean 50, but sample standard deviations of about 1.6 and 31.6.

**Related quantities**

- **Standard error** of the mean, $s/\sqrt{n}$: the uncertainty of the estimated mean, not the spread of the data. Confusing the two is a common mistake in reports.
- **Coefficient of variation** $s / \bar x$: spread relative to the mean, for comparing variables on different scales.
- Variance of a sum: $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X, Y)$, the basis of portfolio risk and of ensemble variance reduction.

**Pitfalls:** both are sensitive to outliers (squares amplify large deviations); for skewed or heavy-tailed data use the interquartile range or median absolute deviation. In pandas, `Series.std()` uses $n - 1$ by default while NumPy's `np.std` uses $n$ unless you pass `ddof=1`.

### 1.14 Explain the difference between mean, median, and mode.

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

### 1.15 What is the difference between covariance and correlation, and what do correlation values of +1, 0 and -1 mean?

Both measure how two variables move together. **Covariance** gives the direction of the linear relationship in the variables' own units; **correlation** is covariance scaled by both standard deviations, so it is unit-free, lies between -1 and +1 and also shows strength.

$$
\operatorname{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)], \qquad \rho_{XY} = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

| | Covariance | Correlation |
| --- | --- | --- |
| Range | Unbounded | -1 to +1 |
| Units | Product of the two variables' units | None |
| Changes when you rescale a variable | Yes | No |
| Use | Covariance matrices (PCA, portfolio risk, Gaussian models) | Comparing strength of relationships |

**Interpreting the correlation coefficient**

- **+1:** perfect positive linear relationship; all points lie on an upward line.
- **0:** no linear relationship. There can still be a strong non-linear one: $y = x^2$ on a range symmetric around zero has correlation 0.
- **-1:** perfect negative linear relationship.

**Pitfalls**

- **Outliers:** a single extreme point can create or destroy a correlation; Anscombe's quartet has four very different data sets with identical correlations. Always plot.
- **Non-linear or monotonic relationships:** use Spearman's rank correlation or Kendall's tau; for general dependence use mutual information.
- **Restricted range:** correlation among only top performers understates the population relationship.
- **Aggregation:** correlations of group averages are usually much stronger than correlations between individuals (ecological fallacy).
- **Correlation is not causation** (see 1.16).

```python
import pandas as pd
df[["x", "y"]].corr(method="pearson")    # or "spearman", "kendall"
df[["x", "y"]].cov()
```

### 1.16 Explain correlation versus causation.

**Correlation** means two variables move together; **causation** means changing one would change the other. Correlation is a property of observed data; causation is a claim about what would happen under an intervention, and it needs either an experiment or strong assumptions.

**Why correlated variables may not be causal**

- **Confounding:** a third variable drives both. Ice cream sales and drownings both rise in hot weather.
- **Reverse causation:** hospitals have more sick people, but hospitals do not cause illness.
- **Selection or collider bias:** conditioning on a common effect creates a spurious link (among admitted students, test scores and grades can appear negatively related).
- **Coincidence and multiple testing:** search enough variable pairs and some will correlate by chance (see 1.46).
- **Simpson's paradox:** a trend in every subgroup reverses in the aggregate (see 32.2).

| | Correlation | Causation |
| --- | --- | --- |
| Question | Do they move together? | What happens if I change X? |
| Evidence | Observational data | Randomised experiments, or causal designs on observational data |
| Use | Prediction, monitoring | Decisions and interventions |

**How to establish causation**

1. **Randomised experiments** (A/B tests): randomisation breaks the link with confounders.
2. When experiments are impossible: difference-in-differences, instrumental variables, regression discontinuity, matching or weighting, each with its own assumptions (see Chapter 32).
3. Supporting evidence: a plausible mechanism, dose-response, temporal order, consistency across settings.

**Why it matters for ML:** predictive models exploit correlation, which is fine for forecasting but not for decisions. A churn model may find that customers who call support churn more; reducing calls will not reduce churn. Feature importance and SHAP values are not causal effects.

### 1.17 What are Type I and Type II errors, and how do they relate to statistical power?

A **Type I error** is rejecting a null hypothesis that is actually true (a false positive); a **Type II error** is failing to reject a null hypothesis that is actually false (a false negative). Their probabilities are $\alpha$ and $\beta$, and **power** $= 1 - \beta$ is the probability of detecting a real effect.

| | $H_0$ true (no effect) | $H_0$ false (real effect) |
| --- | --- | --- |
| Reject $H_0$ | **Type I error** (false positive), probability $\alpha$ | Correct detection, probability $1 - \beta$ (power) |
| Fail to reject $H_0$ | Correct | **Type II error** (false negative), probability $\beta$ |

**Examples**

- Medical test: Type I says a healthy person has the disease; Type II misses a sick person.
- A/B test: Type I ships a feature that does nothing (or harms); Type II abandons a feature that works.
- Classification: a false positive is a Type I error, a false negative is Type II.

**Easy memory aid:** Type I is a false alarm, Type II is a missed detection.

**The trade-off**

- For a fixed sample size, lowering $\alpha$ (being stricter) raises $\beta$.
- The only ways to reduce both are **more data**, **less noise** (variance reduction, better measurement) or a **larger true effect**.
- Power depends on $\alpha$, sample size, effect size and variance (see 1.45).

**Choosing which error to protect against** is a business decision: in fraud screening a missed fraud (Type II) may cost more than a false alarm; in drug approval a false claim of efficacy (Type I) is worse. Set $\alpha$ and the target power accordingly, rather than defaulting to 0.05 and 80 percent without thought.

### 1.18 What is a p-value, what does statistical significance mean, and what are the limitations of p-values?

A **p-value** is the probability of seeing a result at least as extreme as the one observed **if the null hypothesis were true**. A result is **statistically significant** when the p-value is below a significance level $\alpha$ chosen in advance (often 0.05), meaning the data would be surprising under the null.

**How to read it**

- $H_0$: no effect, no difference or no relationship. $H_1$: there is one.
- Small p-value: the data are hard to explain by chance alone under $H_0$, so reject $H_0$.
- Large p-value: the data are compatible with $H_0$; you **fail to reject** it. This is not proof that $H_0$ is true, and it is not "accepting" anything; the study may simply be underpowered.

**What a p-value is not**

- Not the probability that $H_0$ is true.
- Not the probability that the result is a fluke, or that it will replicate.
- Not a measure of effect size or importance: $p = 0.01$ does not mean a 99 percent chance that the effect is real.

**Limitations**

- **Sample size dependence:** with millions of users, trivially small effects become significant; with small samples, important effects are not. Always report the effect size with a confidence interval.
- **Multiple testing:** testing many metrics or segments guarantees some false positives (see 1.46).
- **Peeking and p-hacking:** stopping a test when $p$ dips below 0.05, or trying many analyses and reporting the best, inflates the false positive rate far beyond $\alpha$.
- **Assumption sensitivity:** independence, the correct variance estimate and distributional assumptions all matter; correlated observations (several events per user) make p-values too small.
- **Arbitrary threshold:** $p = 0.049$ and $p = 0.051$ are practically the same evidence.

**Good practice:** pre-register the hypothesis, metric, sample size and $\alpha$; report the effect estimate, its confidence interval and the p-value together; consider practical significance (is the effect worth shipping?) separately from statistical significance.

```python
from scipy import stats
t, p = stats.ttest_ind(treatment, control, equal_var=False)   # Welch's t-test
```

### 1.19 What is hypothesis testing, and when is it used in ML?

##### Hypothesis Testing

- **Hypothesis testing** is a statistical method used to determine whether there is enough evidence in sample data to support a claim about a population.
- It starts with two hypotheses:
  - **Null hypothesis ($H_0$):** No significant effect or difference exists.
  - **Alternative hypothesis ($H_1$):** A significant effect or difference exists.
- A statistical test produces a **p-value**, which is compared with a significance level such as **$\alpha = 0.05$**.
- If **$p \lt \alpha$**, reject $H_0$.
- If **$p \geq \alpha$**, fail to reject $H_0$.

##### When is it used in ML?

- **Feature selection:** Test whether a feature has a statistically significant relationship with the target.
- **Model comparison:** Determine whether one model significantly outperforms another.
- **A/B testing:** Test whether a new model or feature improves performance compared with the existing version.
- **Model evaluation:** Determine whether observed performance differences are likely due to chance.
- **Experimentation:** Test whether a treatment, intervention, or ML system produces a meaningful effect.

### 1.20 What statistical tests would you use to compare two models?

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

### 1.21 How do you assess if a feature is statistically significant?

##### Assessing Whether a Feature Is Statistically Significant

- **Fit a statistical model**, such as linear or logistic regression, including the feature.
- Examine the feature's **coefficient** and its **p-value**.
- **p-value $\lt \alpha$** (commonly 0.05) → feature is statistically significant.
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

### 1.22 What is a confidence interval, and how is it used?

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

### 1.23 What are z-score and t-score? When do you use each?

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

### 1.24 Explain Bayes' theorem. How is it relevant to Naive Bayes and Bayesian methods?

Bayes' theorem tells you how to update the probability of a hypothesis $A$ after observing evidence $B$:

$$
P(A \mid B) = \frac{P(B \mid A)\thinspace P(A)}{P(B)}, \qquad P(B) = \sum_{a} P(B \mid a)\thinspace P(a)
$$

- **Prior** $P(A)$: belief before the evidence.
- **Likelihood** $P(B \mid A)$: how probable the evidence is if $A$ holds.
- **Evidence** $P(B)$: total probability of the evidence, a normalising constant.
- **Posterior** $P(A \mid B)$: updated belief.

**Worked example (fraud alerts).** 0.5 percent of transactions are fraudulent. A detector flags 90 percent of fraud and 2 percent of legitimate transactions. For a flagged transaction:

$$
P(\text{fraud} \mid \text{flag}) = \frac{0.9 \times 0.005}{0.9 \times 0.005 + 0.02 \times 0.995} \approx 0.18
$$

Only about 18 percent of alerts are real fraud. Ignoring the prior (**base-rate neglect**) is one of the most common reasoning errors in business and medicine.

**Relevance to Naive Bayes:** the classifier picks the class $C$ maximising $P(C)\prod_j P(x_j \mid C)$, applying Bayes' theorem with the assumption that features are conditionally independent given the class (see 9.17).

**Relevance to Bayesian methods**

- **Parameter inference:** treat parameters as random, $p(\theta \mid D) \propto p(D \mid \theta)\thinspace p(\theta)$, and report the full posterior (see 1.26).
- **Regularisation as a prior:** MAP estimation with a Gaussian prior is ridge regression (see 1.25).
- **Sequential updating:** yesterday's posterior is today's prior, used in Bayesian A/B testing and Thompson sampling for bandits.
- **Bayesian optimisation** of hyperparameters and **probabilistic graphical models** (see 9.26).

**What interviewers look for:** the formula, a base-rate example worked through, and the link to precision: in classification, $P(\text{positive} \mid \text{flagged})$ is precision, which collapses when the positive class is rare.

### 1.25 What are maximum likelihood estimation (MLE) and maximum a posteriori (MAP) estimation, and how do they differ?

**MLE** picks the parameters that make the observed data most probable. **MAP** picks the parameters that are most probable given the data **and a prior**, i.e. it maximises the posterior. MAP is MLE plus a penalty from the prior, which is exactly what regularisation is.

$$
\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i \mid \theta)
$$

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta \left[\sum_i \log p(x_i \mid \theta) + \log p(\theta)\right]
$$

Logs turn products into sums, which is numerically stable and easier to differentiate.

**Example (coin):** 7 heads in 10 flips. MLE gives $\hat p = 7/10 = 0.7$. With a Beta(2, 2) prior (a mild belief that the coin is fair), MAP gives $(7 + 1)/(10 + 2) \approx 0.67$, pulled towards 0.5. With 7,000 heads in 10,000 flips both give about 0.7: the data overwhelm the prior.

**Losses you already know are MLE**

| Model assumption | Negative log-likelihood |
| --- | --- |
| Gaussian noise | Mean squared error (linear regression) |
| Bernoulli outcome | Binary cross-entropy (logistic regression) |
| Categorical outcome | Categorical cross-entropy (softmax classifiers, LLM pre-training) |
| Poisson counts | Poisson deviance |

**Priors you already know are regularisers**

- Gaussian prior on weights gives an L2 penalty (ridge, weight decay).
- Laplace prior gives an L1 penalty (lasso), which encourages sparsity.
- A uniform prior gives MAP equal to MLE.

| | MLE | MAP |
| --- | --- | --- |
| Uses a prior | No | Yes |
| Small data | Can overfit (a coin with 3 heads in 3 flips gives $\hat p = 1$) | More stable |
| Large data | Same as MAP | Prior matters little |
| Output | Point estimate | Point estimate (the posterior mode), not the full posterior |

**Properties of MLE:** consistent and asymptotically efficient under regularity conditions, invariant to reparameterisation, but biased in small samples (the MLE of variance divides by $n$, see 1.13).

**Limitation of MAP:** it is still a point estimate and ignores posterior uncertainty; full Bayesian inference keeps the whole posterior (see 1.26). See 1.38 for the distinction between probability and likelihood.

### 1.26 How do the Bayesian and frequentist approaches differ, in statistics and in machine learning practice?

The two schools differ in **what probability means**. Frequentists treat probability as long-run frequency and parameters as fixed unknown constants; Bayesians treat probability as a degree of belief and parameters as random variables with distributions that are updated by data.

| | Frequentist | Bayesian |
| --- | --- | --- |
| Probability | Long-run frequency of repeated events | Degree of belief, updated with evidence |
| Parameters | Fixed but unknown | Random variables with a prior $p(\theta)$ |
| Output | Point estimate (MLE), confidence interval, p-value | Posterior $p(\theta \mid D) \propto p(D \mid \theta)\thinspace p(\theta)$, credible interval |
| Interval meaning | 95 percent of intervals built this way would contain the true value | 95 percent probability the parameter lies in this interval, given model and prior |
| Typical ML | MLE-trained models, hypothesis tests, bootstrap | Bayesian regression, Gaussian processes, Bayesian optimisation, MCMC, variational inference |

**Coin example:** a frequentist estimates $\hat p$ from the flips and builds a confidence interval; a Bayesian starts with a Beta prior, updates it to a Beta posterior and reads off any probability statement directly.

**In practice**

- **MAP links them:** L2 regularisation is a Gaussian prior, L1 a Laplace prior (see 1.25). Most regularised ML is Bayesian in flavour even when no one calls it that.
- **With lots of data** the likelihood dominates and both give nearly the same answers; the choice matters most with small data, rare events or strong prior knowledge.
- **A/B testing:** frequentist tests need a fixed sample size and careful handling of peeking; Bayesian analysis reports "probability that B beats A is 94 percent" and expected loss, which stakeholders find easier, but it does not remove the risk of stopping early on noise, and the prior must be justified.
- **Hierarchical (multilevel) models** borrow strength across groups, e.g. estimating conversion for thousands of small segments or stores; this is a major practical Bayesian win.
- **Uncertainty for decisions:** Bayesian posteriors feed naturally into expected-value calculations, bandits (Thompson sampling) and forecasting with uncertainty.

**Trade-offs**

- **Bayesian strengths:** direct probability statements, natural use of prior knowledge, good with small data, sequential updating.
- **Bayesian weaknesses:** priors can be contested; exact posteriors are usually intractable, so MCMC is slow and variational inference is approximate; harder to explain the machinery.
- **Frequentist strengths:** standard, fast, well-understood procedures with known error rates; regulators and journals expect them.
- **Frequentist weaknesses:** p-values and confidence intervals are routinely misinterpreted; no direct probability statements about parameters.

**What interviewers look for:** you can state the philosophical difference in one sentence, interpret a confidence interval correctly, and choose pragmatically: frequentist for standard, high-volume experiments; Bayesian for small data, hierarchical structure, sequential decisions or when stakeholders need probability statements.

### 1.27 What is the Central Limit Theorem(CLT)? Why is it important?

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

##### Caveats

- The CLT needs independent observations and a **finite variance**. Correlated data (several events per user, time series) need cluster-aware or time-series methods.
- "Sufficiently large" depends on skew: 30 is a myth for heavy-tailed data such as revenue, where thousands of observations may be needed and a few extreme values still dominate (see 1.47).
- It describes the sampling distribution of the **mean**, not the data: the data themselves do not become normal.

**Short:**

**CLT says that the distribution of sample means becomes approximately normal as sample size increases, making statistical inference possible for many real-world problems.**

### 1.28 What are sampling techniques?

##### Sampling Techniques

**Sampling** is the process of selecting a subset of a population to study, instead of collecting data from the entire population.

##### 1. Probability Sampling

Every member has a known chance of being selected.

- **Simple Random Sampling:** Every member has an equal chance of selection.
- **Systematic Sampling:** Select every $`k`$th member after a random starting point.
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

### 1.29 What is the bootstrap, and how do you build bootstrap confidence intervals?

The bootstrap estimates the sampling distribution of a statistic by **resampling the observed data with replacement** many times and recomputing the statistic each time. It gives standard errors and confidence intervals for almost any statistic (median, AUC, a ratio, a model's accuracy) without a closed-form formula.

**Procedure**

1. From a sample of size $n$, draw $n$ rows with replacement.
2. Compute the statistic on the resample.
3. Repeat $B$ times (1,000 for standard errors, 5,000 to 10,000 for interval endpoints).
4. Use the spread of the $B$ values as the estimate of uncertainty.

**Types of bootstrap confidence interval**

| Method | How | Notes |
| --- | --- | --- |
| Percentile | 2.5th and 97.5th percentiles of the bootstrap statistics | Simple; can be off for skewed or biased statistics |
| Basic (reverse percentile) | Reflect the percentiles around the estimate | Corrects some bias, can give impossible values |
| Normal | Estimate plus or minus 1.96 bootstrap standard errors | Assumes a roughly normal sampling distribution |
| BCa (bias-corrected and accelerated) | Adjusts percentiles for bias and skew | Usually the most accurate; default in `scipy.stats.bootstrap` |
| Studentised (bootstrap-t) | Bootstraps a t-like statistic | Accurate but needs a standard error inside each resample |

```python
import numpy as np
from scipy.stats import bootstrap

res = bootstrap((latencies,), np.median, n_resamples=10_000, confidence_level=0.95, method="BCa")
print(res.confidence_interval)

# paired bootstrap for the AUC difference between two models on the same test set
rng = np.random.default_rng(0)
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
```

**Where it is used in ML**

- Confidence intervals for evaluation metrics (AUC, F1, NDCG) and for differences between models, using a **paired** bootstrap on the same test rows.
- A/B test metrics that are medians, percentiles or ratios.
- Bagging and random forests train each model on a bootstrap sample (see 9.5).

**Pitfalls**

- **Resample the independent unit:** if users have many sessions, resample users, not sessions (cluster bootstrap); otherwise intervals are too narrow.
- **Time series:** ordinary bootstrap destroys autocorrelation; use block bootstrap.
- **Extremes:** the bootstrap is unreliable for the maximum, extreme quantiles and heavy-tailed data with infinite variance.
- **Small samples:** the bootstrap cannot create information the sample does not contain; with 15 points intervals are rough whatever the method.
- **Leakage in model evaluation:** when bootstrapping a training procedure, rows that appear in the training resample must not appear in its evaluation set (use the out-of-bag rows).

### 1.30 What are eigenvalues and eigenvectors?

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

### 1.31 What is Selection Bias?

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

### 1.32 What is collinearity, and why does it matter?

Collinearity means two predictors are strongly linearly related; **multicollinearity** extends this to one predictor being close to a linear combination of several others. It does not usually hurt predictions, but it makes individual coefficients unstable and hard to interpret.

**Example:** in a salary model, age and years of experience are highly correlated. The model cannot tell whether salary rises with age or with experience, so it may give a large positive weight to one and a negative weight to the other, and the split changes from sample to sample.

**Consequences**

- Inflated standard errors and wide confidence intervals for coefficients.
- Coefficients that flip sign or change a lot when a feature or a few rows are added.
- Individually non-significant coefficients even though the features are jointly significant.
- With **perfect** collinearity (e.g. including all dummy columns plus an intercept, or a feature that is the sum of two others) the least squares solution is not unique.

**Detection:** pairwise correlation matrix, variance inflation factors (above 5 to 10 is a warning), condition number of the scaled design matrix.

**Remedies:** drop or combine redundant features, use ridge or elastic net, use PCA or PLS, or accept it if you only need predictions and the correlation is stable in production.

See 9.13 for detection and handling in regression models and 10.36 for the feature-engineering view.

### 1.33 What is a time series, and what makes time-series data different?

A time series is a sequence of observations indexed by time, usually at regular intervals (sales per day, CPU load per minute). What makes it different from ordinary tabular data is that **observations are not independent**: today's value depends on yesterday's, which changes how you model, validate and test.

**Components**

- **Trend:** long-term increase or decrease.
- **Seasonality:** fixed-period patterns (daily, weekly, yearly).
- **Cycles:** longer fluctuations without a fixed period (business cycles).
- **Noise:** the remaining irregular variation.

Additive decomposition writes $y_t = T_t + S_t + R_t$; multiplicative decomposition, $y_t = T_t \times S_t \times R_t$, suits seasonal swings that grow with the level (often handled with a log transform).

**Key concepts**

- **Autocorrelation:** correlation of the series with its own lags; ACF and PACF plots guide model choice.
- **Stationarity:** constant mean, variance and autocorrelation over time. Many classical models assume it; differencing and transforms help achieve it (see 32.25).
- **Structural breaks:** sudden changes in level or behaviour from launches, policy changes or shocks such as the pandemic (see 32.34).

**Practical consequences**

- **Validation must respect time:** train on the past and test on the future with rolling-origin backtests; random K-fold leaks the future (see 12.23).
- **Statistical tests that assume independence** (t-tests, naive confidence intervals) understate uncertainty on autocorrelated data.
- **Features:** lags, rolling statistics, calendar effects, holidays; all computed only from information available at prediction time.
- **Spurious correlation:** two trending series correlate strongly even when unrelated; difference them before correlating.

**Models:** seasonal naive and exponential smoothing baselines, ARIMA, gradient boosting with lag features, and deep or foundation models for many series. Chapter 32 (32.25 to 32.36) covers forecasting in depth.

### 1.34 What is the Fourier transform, and where is it used in data science and ML?

The Fourier transform rewrites a signal as a sum of sine and cosine waves, converting it from the **time (or space) domain** to the **frequency domain**. It shows which frequencies are present and how strong they are.

For a sampled signal $x_0 \ldots x_{N-1}$, the discrete Fourier transform is

$$
X_k = \sum_{n=0}^{N-1} x_n \thinspace e^{-2\pi i k n / N}, \qquad k = 0, \ldots, N-1
$$

The magnitude $\lvert X_k\rvert$ is the strength of frequency $k$, and the phase gives its shift. The **fast Fourier transform (FFT)** computes this in $O(N \log N)$ instead of $O(N^2)$, which is what makes it practical.

**Key properties**

- **Convolution theorem:** convolution in the time domain is multiplication in the frequency domain, so large convolutions and filters are computed quickly via the FFT.
- **Sampling (Nyquist):** you can only resolve frequencies up to half the sampling rate; higher frequencies alias into lower ones.
- **Spectral leakage:** a finite window smears frequencies; windowing functions (Hann, Hamming) reduce it.

**Uses in data science and ML**

- **Time series:** detect dominant seasonal periods (a peak at 1/7 per day reveals weekly seasonality) and build Fourier features (sine and cosine terms) for regression or forecasting models, as Prophet does for seasonality.
- **Audio and speech:** spectrograms and mel-frequency features via the short-time Fourier transform are the input to many speech models.
- **Images:** compression (JPEG uses the related discrete cosine transform), filtering, and frequency-domain augmentation.
- **Sensor and vibration data:** predictive maintenance looks for new frequency peaks in machine vibrations.
- **Deep learning:** FFT-based convolutions, Fourier neural operators for PDEs, and the sinusoidal positional encodings in Transformers use the same idea.

```python
import numpy as np
spectrum = np.abs(np.fft.rfft(y - y.mean()))
freqs = np.fft.rfftfreq(len(y), d=1.0)            # d = sampling interval (e.g. 1 day)
dominant_period = 1 / freqs[np.argmax(spectrum[1:]) + 1]
```

### 1.35 How can you check whether a data set is normally distributed, and when does it matter?

Use visual checks first and formal tests second, and remember that what usually matters is not the raw data but the **residuals** or the **sampling distribution** of the statistic you are testing.

**Visual checks**

- **Histogram or density plot:** roughly symmetric, single-peaked, bell shape.
- **Q-Q plot:** sample quantiles against normal quantiles; points on a straight line indicate normality. Curvature shows skew; S-shapes show heavy or light tails. This is the most informative single check.
- **Skewness and excess kurtosis:** both near 0 for a normal distribution.

**Formal tests**

| Test | Notes |
| --- | --- |
| Shapiro-Wilk | Most powerful general test for small to medium samples |
| Anderson-Darling | Gives more weight to the tails |
| D'Agostino-Pearson | Combines skewness and kurtosis |
| Kolmogorov-Smirnov | Weak for normality; requires known parameters (use Lilliefors if estimated) |

```python
from scipy import stats
import statsmodels.api as sm
stat, p = stats.shapiro(residuals)                 # small p suggests non-normality
sm.qqplot(residuals, line="s")
```

**The large-sample trap:** with thousands of rows every test rejects normality because of tiny, harmless deviations; with 20 rows they rarely reject even clear non-normality. Use tests as a supplement to plots, not a gate.

**When normality matters**

- Linear regression inference: the **residuals** should be roughly normal for small-sample confidence intervals and p-values; the predictors and target do not need to be normal.
- t-tests with small samples. With large samples the central limit theorem makes the sample mean approximately normal anyway, unless the data are heavy-tailed (see 1.27).
- Methods that assume Gaussian data explicitly: Gaussian Naive Bayes, LDA, z-score anomaly thresholds, Gaussian process noise.

**When it does not matter:** tree ensembles and most ML models make no normality assumption.

**If data are not normal:** log or Box-Cox transforms for right skew (see 10.15), non-parametric tests (Mann-Whitney, Wilcoxon), bootstrap confidence intervals (see 1.29), or a model with a suitable distribution (Poisson, gamma).

### 1.36 You can roll two dice for \$4 and win \$50 if the sum is 7. Would you play, and what is the probability of making money if you play six times?

**Yes: the game has positive expected value, and you make money in six plays about two-thirds of the time.**

**Single roll**

- 36 equally likely outcomes; a sum of 7 occurs in 6 of them, so $P(\text{win}) = 1/6$.
- You pay the \$4 fee on every roll and receive \$50 on a win, so the net outcome of one roll is +\$46 with probability 1/6 and -\$4 with probability 5/6.
- Expected value per roll: $50/6 - 4 \approx 4.33$ dollars, so the game favours the player by about \$4.33 per play.

**Six plays**

- Total cost is \$24. Let $W$ be the number of wins, $W \sim \text{Binomial}(6, 1/6)$.
- Net result is $50W - 24$, which is positive whenever $W \ge 1$.

$$
P(W \ge 1) = 1 - \left(\frac{5}{6}\right)^6 \approx 1 - 0.335 = 0.665
$$

So there is about a 66.5 percent chance of finishing ahead, and the expected profit over six plays is $6 \times 4.33 \approx 26$ dollars.

**Variance matters too:** about one time in three you lose all \$24. A positive expected value is a good bet for someone who can play many times or absorb the loss; with a small bankroll, the risk of ruin matters. This is the same reasoning used for any business decision with uncertain payoffs: compute the expected value, then look at the spread of outcomes.

**What interviewers look for:** setting up the sample space correctly, using the complement for "at least one", stating the expected value, and noticing the interpretation question (is the \$4 fee deducted from the \$50 prize or not).

### 1.37 Give an example where the median is a better measure than the mean

The **median is better when the distribution is skewed or has outliers**, because the mean is pulled toward extreme values while the median depends only on the middle observation.

**Classic example — income:** salaries in a team of ten are 40k–60k, plus a CEO at 5M. The mean is ~550k, which describes nobody; the median (~50k) represents a "typical" employee. That is why governments report **median household income** and **median house prices**.

**Other examples**
- **Latency / response times:** a few slow requests inflate the mean; teams report p50 (the median) alongside p95/p99.
- **Time-on-site, order values, insurance claims, hospital stays** — all long-tailed.
- **Imputing missing values** in a skewed feature: median imputation is more robust than mean imputation.
- **Data with recording errors** (e.g., an age of 999).

**When the mean is still preferred:** roughly symmetric data without outliers; when you need totals ($\text{total revenue} = \text{mean} \times \text{count}$); or when extremes genuinely matter (expected cost in insurance).

**Practice:** look at the histogram or box plot, report both mean and median when they differ, and consider a log transform for heavily skewed features.

### 1.38 What’s the difference between probability and likelihood?

They use the same function $p(x \mid \theta)$, read in two different directions.

- **Probability:** fix the parameters $\theta$ and ask how likely different *data* outcomes are. "Given a fair coin ($p = 0.5$), what is the chance of 7 heads in 10 flips?" Probabilities over outcomes sum or integrate to 1.
- **Likelihood:** fix the observed *data* and treat it as a function of the parameters: $L(\theta \mid x) = p(x \mid \theta)$. "Given I saw 7 heads in 10 flips, how plausible is $p = 0.5$ vs $p = 0.7$?" Likelihood is **not** a probability distribution over $\theta$ and need not sum to 1.

**In ML:**
- **Maximum Likelihood Estimation:** $\hat\theta = \arg\max_\theta \prod_i p(x_i \mid \theta)$, in practice by maximizing the **log-likelihood** (sums are numerically stable). Minimizing cross-entropy or MSE equals MLE under Bernoulli or Gaussian noise assumptions.
- For a probability over parameters, use Bayes: posterior $\propto$ likelihood $\times$ prior (MAP / Bayesian inference).

**Example:** 7 heads in 10 flips gives $L(0.7) = \binom{10}{7}0.7^7 0.3^3 \approx 0.267$ and $L(0.5) \approx 0.117$, so $p = 0.7$ is more likely given the data. The MLE is $\hat p = 0.7$.

### 1.39 How will you find the correlation between a categorical variable and a continuous variable?

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

### 1.40 Describe the differences between and use cases for box plots and histograms

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

### 1.41 What is KL divergence, and how would you define its use case in ML?

**KL divergence** measures how much one probability distribution $P$ differs from a reference distribution $Q$:

$$
D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

Interpretation: the expected extra bits (nats) needed to encode samples from $P$ using a code optimized for $Q$.

**Properties:** always $\ge 0$, zero only if $P = Q$; **not symmetric** ($`D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)`$), so it's not a true distance; infinite if $Q(x) = 0$ where $P(x) \gt 0$. Forward KL is "mean-seeking" (covers all modes); reverse KL is "mode-seeking".

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

### 1.42 Which metrics can be used to measure correlation of categorical data?

**Categorical vs. categorical (nominal):**

- **Chi-square test of independence:** tests whether an association exists (gives a p-value, not strength).
- **Cramér's V:** effect size from chi-square, range 0–1, works for any table size (use the bias-corrected version for small samples):

  ```math
  V = \sqrt{\frac{\chi^2/n}{\min(r-1, c-1)}}
  ```

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

### 1.43 Where do linear algebra, probability and statistics actually show up in day-to-day LLM and computer vision engineering work?

They show up constantly, mostly as practical judgement rather than derivations: reasoning about vectors and matrices when working with embeddings and model internals, and about uncertainty and variance when evaluating and shipping changes.

**Linear algebra**

- **Embeddings and retrieval:** text and images become vectors; similarity is a dot product or cosine; normalising vectors makes the two equivalent. Approximate nearest neighbour indexes (HNSW, IVF) and quantisation are geometry problems.
- **Attention:** $\operatorname{softmax}(QK^\top / \sqrt{d})V$ is matrix multiplication; the $\sqrt{d}$ scaling keeps dot products from growing with dimension.
- **Low-rank structure:** LoRA fine-tuning adds a low-rank update $BA$ to a weight matrix; SVD-based compression and PCA for inspecting embedding spaces rely on the same idea (see 1.44).
- **Shapes and memory:** estimating parameter counts, activation memory and FLOPs from matrix dimensions to plan GPU budgets.
- **Computer vision:** convolutions, camera projections and homographies, and image transforms are all linear maps.

**Probability**

- **Generation:** the model outputs a categorical distribution over tokens; temperature, top-k and top-p reshape it. Log-probabilities support confidence estimates and perplexity.
- **Calibration:** whether a classifier's 0.9 score is right 90 percent of the time decides whether you can threshold it.
- **Loss functions:** cross-entropy is a negative log-likelihood; KL divergence appears in distillation and RLHF (see 1.41).

**Statistics**

- **Evaluation with uncertainty:** an eval set of 200 questions gives an accuracy confidence interval of roughly plus or minus 7 points, so a 2-point gain between prompts is noise. Use bootstrap intervals and paired comparisons (see 1.29 and 1.20).
- **A/B tests** of a RAG pipeline or prompt change: power, sample size and multiple metrics (see 1.45 and 1.46).
- **LLM-as-judge agreement** with humans: correlation, Cohen's kappa and position-bias checks.
- **Monitoring:** detecting drift in input or embedding distributions with statistical distances.

**Calculus and optimisation:** backpropagation and learning-rate behaviour during fine-tuning, diagnosing divergence and loss spikes (see Chapter 11).

**What interviewers look for:** concrete examples from your own work, especially catching a false improvement by quantifying evaluation noise.

### 1.44 What is Singular Value Decomposition (SVD)?

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

Let's do a **complete SVD calculation** from start to finish using:

$$
A= \begin{bmatrix} 3&2\cr 1&4 \end{bmatrix}
$$

We want: $\boxed{A=U\Sigma V^T}$

##### 1. Calculate $A^T$

$$
A^T= \begin{bmatrix} 3&1\cr 2&4 \end{bmatrix}
$$

##### 2. Calculate $AA^T$

$$
AA^T= \begin{bmatrix} 3&2\cr 1&4 \end{bmatrix} \begin{bmatrix} 3&1\cr 2&4 \end{bmatrix} = \begin{bmatrix} 13&11\cr 11&17 \end{bmatrix}
$$

Find its eigenvalues:

$$
\det \begin{bmatrix} 13-\lambda&11\cr 11&17-\lambda \end{bmatrix}=0
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

So:

$$
\boxed{ \Sigma= \begin{bmatrix} 5.117&0\cr 0&1.955 \end{bmatrix}}
$$

---

##### 4. Find $U$

The columns of $U$ are the **normalized eigenvectors of $AA^T$**.

For $\lambda_1$:

$$
\begin{bmatrix} 13-26.180&11\cr 11&17-26.180 \end{bmatrix} \begin{bmatrix} x\cr y \end{bmatrix}=0
$$

This gives approximately: $y=1.198x$

After normalization:

$$
u_1= \begin{bmatrix} 0.641\cr 0.768 \end{bmatrix}
$$

For $\lambda_2$:

$$
u_2= \begin{bmatrix} 0.768\cr -0.641 \end{bmatrix}
$$

Therefore:

$$
\boxed{ U= \begin{bmatrix} 0.641&0.768\cr 0.768&-0.641 \end{bmatrix}}
$$

---

##### 5. Find $V$

Now calculate:

$$
A^TA= \begin{bmatrix} 3&1\cr 2&4 \end{bmatrix} \begin{bmatrix} 3&2\cr 1&4 \end{bmatrix} = \begin{bmatrix} 10&10\cr 10&20 \end{bmatrix}
$$

The eigenvalues are the same: $26.180,\quad3.820$

For $\lambda_1=26.180$:

$$
(10-26.180)x+10y=0
$$

$$
y=1.618x
$$

After normalization:

$$
v_1= \begin{bmatrix} 0.526\cr 0.851 \end{bmatrix}
$$

For $\lambda_2=3.820$:

$$
v_2= \begin{bmatrix} 0.851\cr -0.526 \end{bmatrix}
$$

Thus:

$$
\boxed{ V= \begin{bmatrix} 0.526&0.851\cr 0.851&-0.526 \end{bmatrix}}
$$

Therefore:

$$
\boxed{ V^T= \begin{bmatrix} 0.526&0.851\cr 0.851&-0.526 \end{bmatrix}}
$$

---

#### 6. Final SVD

Putting everything together: $\boxed{ A=U\Sigma V^T }$

$$
\boxed{ \begin{bmatrix} 3&2\cr 1&4 \end{bmatrix} \approx \begin{bmatrix} 0.641&0.768\cr 0.768&-0.641 \end{bmatrix} \begin{bmatrix} 5.117&0\cr 0&1.955 \end{bmatrix} \begin{bmatrix} 0.526&0.851\cr 0.851&-0.526 \end{bmatrix} }
$$

If you multiply these three matrices, you recover approximately:

$$
\begin{bmatrix} 3&2\cr 1&4 \end{bmatrix}
$$

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

### 1.45 What is statistical power, and how do you calculate the sample size for an experiment?

**Power** is the probability that a test detects an effect of a given size when that effect really exists, $1 - \beta$. A sample size calculation chooses $n$ so that the smallest effect worth acting on, the **minimum detectable effect (MDE)**, would be detected with the desired power (usually 80 percent) at the chosen significance level (usually 5 percent).

**The four linked quantities:** significance level $\alpha$, power $1 - \beta$, effect size $\delta$, and noise $\sigma$. Fix any three and the fourth follows; together they set $n$.

**Formula for comparing two means** (per group, two-sided test):

$$
n = \frac{2\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \sigma^2}{\delta^2}
$$

With $\alpha = 0.05$ and 80 percent power, $(1.96 + 0.84)^2 \approx 7.84$, giving the handy rule $n \approx 16\sigma^2/\delta^2$ per group. For a conversion rate, $\sigma^2 = p(1 - p)$.

**Worked example:** baseline conversion 5 percent, and you want to detect a 5 percent relative lift (5.00 to 5.25 percent, so $\delta = 0.0025$). Then $\sigma^2 = 0.0475$ and $n \approx 16 \times 0.0475 / 0.0025^2$, about 121,600 users per group. Halving the MDE quadruples the sample size, which is why small lifts on low base rates need very large experiments.

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

es = proportion_effectsize(0.0525, 0.05)          # Cohen's h
n = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1.0)
print(round(n))                                    # about 122,000 per group
```

**How to get more power without more users**

- **Variance reduction:** CUPED uses a pre-experiment covariate (last month's spend) to reduce variance by a factor of $1 - \rho^2$, where $\rho$ is its correlation with the metric; stratification and regression adjustment do similar things.
- **More sensitive metrics:** a proxy closer to the change (clicks on the new widget) moves more than revenue.
- **Cap or transform heavy-tailed metrics** (see 1.47).
- **Run longer**, or use a unit with less noise.
- **One-sided tests** only when you genuinely would not act on a negative result.

**Pitfalls**

- **Underpowered tests** mostly return "not significant", and the significant results they do find overestimate the true effect (the winner's curse or Type M error).
- **Post-hoc power** computed from the observed effect adds nothing beyond the p-value; plan power before the test.
- **Wrong unit of analysis:** if you randomise users but analyse page views, the variance is underestimated and the planned sample size is wrong (see 1.49).
- **Peeking** at results and stopping early changes the error rates; use a fixed horizon or a sequential design.
- **Choose the MDE from the business case** (the smallest lift that pays for the change), not from what makes the sample size convenient.

**What interviewers look for:** the four quantities and their trade-off, a quick estimate with $16\sigma^2/\delta^2$, and practical levers such as CUPED when traffic is limited.

### 1.46 What is the multiple testing problem, and how do you control the false discovery rate?

If you run many hypothesis tests at $\alpha = 0.05$, some will be significant by chance even when nothing is going on. With $m$ independent true-null tests, the chance of at least one false positive is $1 - 0.95^m$: about 40 percent for 10 tests and 64 percent for 20. Correction methods control either the probability of any false positive or the expected share of false positives among discoveries.

**Where it happens in practice**

- An A/B test with 20 metrics, or the same test sliced by 15 segments ("it worked for Android users in Germany").
- Feature screening across thousands of candidate features, genomics, or monitoring many dashboards for anomalies.
- Repeatedly checking an experiment and stopping when $p$ dips below 0.05 (peeking), which is multiple testing over time.
- Trying many model variants and reporting the best on the same test set.

**Two error rates**

| Error rate | Definition | Methods | Use when |
| --- | --- | --- | --- |
| Family-wise error rate (FWER) | Probability of at least one false positive | Bonferroni, Holm | Few tests, any false claim is costly (primary metrics, regulatory decisions) |
| False discovery rate (FDR) | Expected proportion of false positives among the rejected hypotheses | Benjamini-Hochberg, Benjamini-Yekutieli | Many tests, exploratory screening where some false leads are acceptable |

**Methods**

- **Bonferroni:** test each at $\alpha/m$. Simple and valid under any dependence, but very conservative.
- **Holm (step-down):** sort p-values ascending; compare the $k$-th smallest with $\alpha/(m - k + 1)$ and stop at the first failure. Same guarantee as Bonferroni, uniformly more powerful.
- **Benjamini-Hochberg:** sort p-values $p_{(1)} \le \ldots \le p_{(m)}$, find the largest $k$ with $p_{(k)} \le (k/m)\thinspace q$ and reject hypotheses 1 to $k$. Controls FDR at level $q$ under independence or positive dependence; Benjamini-Yekutieli handles arbitrary dependence at a cost in power.

```python
from statsmodels.stats.multitest import multipletests
reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")   # or "holm", "bonferroni"
```

**Practical approach for experiments**

1. Pre-register **one primary metric** (or a small set) and its analysis; secondary and guardrail metrics are reported but interpreted more cautiously.
2. Treat segment analyses as hypothesis-generating; confirm surprising segment effects in a follow-up test.
3. For peeking, use sequential methods designed for it (group sequential designs with alpha spending, or always-valid p-values) rather than repeated fixed-horizon tests.
4. Keep a final untouched test set for model selection, or account for the number of variants tried.

**Pitfalls:** corrections reduce power, so required sample sizes grow; correcting for metrics you would never act on wastes power; and no correction fixes undisclosed analyses (the "garden of forking paths"). Transparency about how many things were tried matters as much as the method.

### 1.47 What are heavy-tailed distributions, and why can the mean mislead?

A distribution is heavy-tailed when extreme values are far more likely than under a normal distribution, so a few observations can dominate totals and averages. Revenue per customer, insurance claims, file sizes, request latency, city sizes and item popularity are typical. For such data the sample mean is unstable, slow to converge and often unrepresentative.

**Examples and degrees of heaviness**

- **Log-normal:** all moments finite, but strongly skewed; the mean sits well above the median.
- **Power law (Pareto):** $P(X \gt x) \propto x^{-\alpha}$. If $\alpha \le 2$ the variance is infinite; if $\alpha \le 1$ even the mean is infinite. Word frequencies (Zipf), wealth and network degrees look like this.
- **Student's t** with few degrees of freedom, **Cauchy** (no mean at all).

**Why the mean misleads**

- **Dominated by a few points:** if 1 percent of customers produce 50 percent of revenue, average revenue per user mainly measures whether a few "whales" landed in your sample.
- **Slow convergence:** the central limit theorem still applies when the variance is finite, but may need enormous samples before the mean looks normal; with infinite variance it does not apply in its usual form, and the sample mean keeps jumping as new extremes arrive.
- **High variance in experiments:** one big spender in the treatment group can create a "significant" revenue lift or hide a real one.
- **The mean describes no one:** "average session length is 40 minutes" when the median is 4.

**What to use instead or as well**

| Need | Tool |
| --- | --- |
| Typical value | Median, other quantiles (p90, p99 for latency) |
| Robust average | Trimmed or winsorised mean, Hodges-Lehmann estimator |
| Modelling | Log transform; Tweedie, gamma or log-normal models; quantile regression |
| Concentration | Share of total from the top 1 or 10 percent, Lorenz curve, Gini coefficient |
| Tail risk | Extreme value theory, tail index estimates (Hill estimator), expected shortfall |
| Experiments | Cap (winsorise) at a pre-agreed percentile, CUPED, bootstrap intervals, or test on a transformed metric |

**Business reality:** totals still matter (total revenue is mean times count), so you cannot ignore the mean, but you should report it with its uncertainty and alongside the median and concentration. When you cap a metric in an A/B test you change the question from "effect on revenue" to "effect on revenue excluding the extreme tail"; say so explicitly.

**Diagnose:** plot on log-log axes (a straight tail suggests a power law), compare mean and median, check how much the mean moves when you remove the top 0.1 percent, and look at the maximum relative to the sum.

**What interviewers look for:** recognising heavy tails in common business metrics, explaining why the average is unstable, and choosing robust summaries and experiment designs. See 1.37 for the median versus mean basics.

### 1.48 What are the most common statistical mistakes in business analysis, and how do you avoid them?

Most wrong conclusions in analytics come from a small set of recurring mistakes: comparing the wrong groups, ignoring randomness, misreading aggregates and confusing correlation with cause. A senior analyst is expected to spot them in other people's work as well as their own.

| Mistake | What happens | Example | Guard |
| --- | --- | --- | --- |
| Simpson's paradox | A trend in every subgroup reverses when groups are combined | Overall conversion falls while it rises in every channel, because the mix shifted to a low-converting channel | Break results down by key segments and mix (see 32.2) |
| Selection bias | The analysed group differs systematically from the population | Users who opted into a new feature are already more engaged | Randomise, or compare like with like |
| Survivorship bias | Only the "survivors" are analysed | Studying current customers to learn why customers stay | Include churned and failed cases |
| Regression to the mean | Extreme values move towards average on their own | Stores with the worst month "improve" after a coaching programme | Use a control group |
| Before-and-after comparisons | Seasonality, trends or other launches explain the change | Sales up after a redesign launched in December | Experiments, or difference-in-differences (see 32.6) |
| Base-rate neglect | Ignoring how rare the event is | A 95 percent accurate fraud model where most alerts are false | Compute precision at the real base rate (see 1.24) |
| Multiple comparisons | Slicing until something is significant | "Worked for iOS users aged 25 to 34 in Spain" | Pre-register, correct, replicate (see 1.46) |
| Peeking | Stopping a test the moment it looks significant | False positive rate several times the nominal 5 percent | Fixed horizon or sequential design |
| Wrong unit of analysis | Analysing page views when users were randomised | Confidence intervals far too narrow | Analyse at the randomisation unit, delta method (see 1.49) |
| Averaging averages | Unweighted mean of ratios | Mean of per-store conversion rates treats a 10-visit store like a 10,000-visit store | Ratio of totals, or weight properly |
| Relative versus absolute | "Risk doubled" from 0.01 to 0.02 percent | Overreaction to a tiny absolute change | Report both |
| Ignoring uncertainty | Point estimates without intervals | Ranking regions by a noisy metric each month | Confidence intervals, shrinkage for small groups |
| Small denominators | Extreme rates from tiny samples | The "best" store has 3 sales from 3 visits | Minimum sample thresholds, Bayesian shrinkage |
| Correlation as causation | Acting on an association | "Users who use feature X retain better, so push X" | Experiments or causal methods (see 1.16 and Chapter 32) |
| Novelty and primacy effects | Short-term reaction differs from long-term | A new UI gets a spike of curious clicks | Run longer, look at effect over time |
| Sample ratio mismatch | Groups have different sizes than designed | 50/50 split shows 52/48 because of a logging bug | Check SRM with a chi-square test before reading results |
| Goodhart's law | The metric becomes the target and stops measuring the goal | Optimising clicks increases clickbait | Guardrail metrics, long-term outcomes |

**A useful habit (Twyman's law):** any figure that looks surprisingly interesting is probably wrong. Check data quality, joins, filters, time windows and definitions before celebrating.

**Checklist before presenting a result**

1. What is the comparison, and are the groups comparable?
2. How big is the effect, with what uncertainty, at what unit of analysis?
3. How many things were tried before this one?
4. Does it hold across segments and time periods, and could mix shifts explain it?
5. Is there a plausible mechanism, and what would falsify it?

**What interviewers look for:** concrete examples from your own experience, especially a time you caught a wrong conclusion before it drove a decision, and a habit of quantifying uncertainty.

### 1.49 How do you calculate confidence intervals for ratio metrics and clustered data, and what is the delta method?

Many business metrics are **ratios of two random quantities** measured on a different unit from the one that was randomised: click-through rate (clicks per page view), revenue per session, average order value. If users are randomised but you treat every page view as independent, the standard error is too small, sometimes by a factor of two or more, and you get false positives. The fix is to compute variance at the randomisation unit, with the **delta method**, a cluster-robust estimator or a cluster bootstrap.

**Why naive formulas fail**

- Page views from the same user are correlated (heavy users click more), so they are not independent Bernoulli trials.
- The denominator (page views) is itself random and differs between groups.
- The effective sample size is closer to the number of users than the number of events.

**The delta method:** for per-user totals $Y_i$ (clicks) and $X_i$ (views) over $n$ users, the metric is $R = \bar Y / \bar X$. A first-order Taylor expansion gives

$$
\operatorname{Var}(R) \approx \frac{1}{\mu_X^2}\left[\operatorname{Var}(\bar Y) - 2\frac{\mu_Y}{\mu_X}\operatorname{Cov}(\bar Y, \bar X) + \frac{\mu_Y^2}{\mu_X^2}\operatorname{Var}(\bar X)\right]
$$

where $\operatorname{Var}(\bar Y) = s_Y^2/n$ and so on, all computed from **per-user** values. A confidence interval is then $R \pm 1.96\sqrt{\operatorname{Var}(R)}$, and the difference between two independent groups has variance equal to the sum of their variances.

```python
import numpy as np

def ratio_var(y, x):
    """Delta-method variance of sum(y)/sum(x) with per-user arrays y and x."""
    n = len(y)
    mx, my = x.mean(), y.mean()
    vy, vx = y.var(ddof=1) / n, x.var(ddof=1) / n
    cov = np.cov(y, x, ddof=1)[0, 1] / n
    return (vy - 2 * (my / mx) * cov + (my / mx) ** 2 * vx) / mx ** 2

r_t, r_c = y_t.sum() / x_t.sum(), y_c.sum() / x_c.sum()
se = np.sqrt(ratio_var(y_t, x_t) + ratio_var(y_c, x_c))
ci = (r_t - r_c - 1.96 * se, r_t - r_c + 1.96 * se)
```

**Alternatives**

- **Cluster bootstrap:** resample users (with all their events), recompute the ratio; simple and flexible for medians and percentiles too (see 1.29).
- **Regression with cluster-robust standard errors:** regress the event-level outcome on the treatment indicator with errors clustered by user; this also allows covariate adjustment.
- **Change the metric to the randomisation unit:** e.g. average clicks per user, which is a simple mean with a standard t-test, if it answers the business question.

**Other clustered situations:** students within schools, patients within hospitals, stores within regions, and switchback or geo experiments where whole cities are randomised. The rule is the same: the effective sample size is driven by the number of independent clusters, not the number of rows, and with few clusters (under about 30 to 50) even cluster-robust methods need small-sample corrections.

**Pitfalls**

- The delta method is a large-sample approximation; with very few users or a denominator near zero, use the bootstrap.
- Mean of per-user ratios and ratio of totals are different metrics with different answers; choose deliberately and keep the definition fixed.
- Heavy-tailed numerators (revenue) make any variance estimate noisy (see 1.47).

**What interviewers look for:** recognising the mismatch between analysis unit and randomisation unit, knowing the delta method exists and roughly what it does, and checking this before trusting an experiment's p-value.

---

[Table of contents](../README.md) · [Chapter 2. Python Engineering](02-python-engineering.md) →
