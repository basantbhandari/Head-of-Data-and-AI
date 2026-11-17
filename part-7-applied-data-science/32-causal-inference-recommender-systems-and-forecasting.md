# Chapter 32. Causal Inference, Recommender Systems and Forecasting

_Part VII — Applied Data Science_

Three applied disciplines that data leaders are expected to master. Causal inference (questions 32.1 to 32.12): potential outcomes, causal graphs, propensity methods, double machine learning, difference-in-differences, synthetic control, instrumental variables, uplift modelling, experiments with interference and marketing measurement. Recommender systems (32.13 to 32.24): implicit feedback, two-tower retrieval, negative sampling, sequential and generative models, ranking architectures, cold start, bias, beyond-accuracy objectives, evaluation, embedding scale, LLMs and long-term value. Forecasting (32.25 to 32.36): ARIMA and exponential smoothing, global models, deep learning and foundation models, probabilistic forecasts, metrics and backtesting, hierarchies, intermittent demand, external drivers, structural breaks, new products and forecasting in production.

This chapter contains **36 questions**.

## Contents

- **32.1** [What is causal inference, and how do the potential outcomes and causal graph frameworks describe it?](#321-what-is-causal-inference-and-how-do-the-potential-outcomes-and-causal-graph-frameworks-describe-it)
- **32.2** [What are confounders, colliders and mediators, and how do you decide which variables to control for?](#322-what-are-confounders-colliders-and-mediators-and-how-do-you-decide-which-variables-to-control-for)
- **32.3** [What assumptions are needed to estimate causal effects from observational data?](#323-what-assumptions-are-needed-to-estimate-causal-effects-from-observational-data)
- **32.4** [How do propensity score matching and inverse probability weighting work?](#324-how-do-propensity-score-matching-and-inverse-probability-weighting-work)
- **32.5** [What are doubly robust estimators and double machine learning?](#325-what-are-doubly-robust-estimators-and-double-machine-learning)
- **32.6** [How does difference-in-differences work, and what can go wrong?](#326-how-does-difference-in-differences-work-and-what-can-go-wrong)
- **32.7** [What is synthetic control, and when would you use it?](#327-what-is-synthetic-control-and-when-would-you-use-it)
- **32.8** [What are instrumental variables and regression discontinuity designs?](#328-what-are-instrumental-variables-and-regression-discontinuity-designs)
- **32.9** [What is uplift modelling, and how do you estimate heterogeneous treatment effects?](#329-what-is-uplift-modelling-and-how-do-you-estimate-heterogeneous-treatment-effects)
- **32.10** [How do you run experiments when users or units interfere with each other?](#3210-how-do-you-run-experiments-when-users-or-units-interfere-with-each-other)
- **32.11** [How do you measure the true incremental impact of marketing?](#3211-how-do-you-measure-the-true-incremental-impact-of-marketing)
- **32.12** [How do you validate a causal analysis and communicate its uncertainty?](#3212-how-do-you-validate-a-causal-analysis-and-communicate-its-uncertainty)
- **32.13** [How does matrix factorisation work for implicit feedback, and what are ALS and BPR?](#3213-how-does-matrix-factorisation-work-for-implicit-feedback-and-what-are-als-and-bpr)
- **32.14** [How are two-tower retrieval models trained, and what is sampling bias correction?](#3214-how-are-two-tower-retrieval-models-trained-and-what-is-sampling-bias-correction)
- **32.15** [How do you choose negative examples for training recommender models?](#3215-how-do-you-choose-negative-examples-for-training-recommender-models)
- **32.16** [How do sequential and generative recommenders work?](#3216-how-do-sequential-and-generative-recommenders-work)
- **32.17** [How do ranking models handle feature interactions and multiple objectives?](#3217-how-do-ranking-models-handle-feature-interactions-and-multiple-objectives)
- **32.18** [How do you handle cold start for new users and new items?](#3218-how-do-you-handle-cold-start-for-new-users-and-new-items)
- **32.19** [What biases affect recommender systems, and how do you correct them?](#3219-what-biases-affect-recommender-systems-and-how-do-you-correct-them)
- **32.20** [Beyond accuracy, what objectives should a recommender optimise: diversity, novelty, serendipity and fairness?](#3220-beyond-accuracy-what-objectives-should-a-recommender-optimise-diversity-novelty-serendipity-and-fairness)
- **32.21** [How do you evaluate recommender systems offline, and why do offline and online results often disagree?](#3221-how-do-you-evaluate-recommender-systems-offline-and-why-do-offline-and-online-results-often-disagree)
- **32.22** [How do you scale embedding tables for recommendation models with billions of IDs?](#3222-how-do-you-scale-embedding-tables-for-recommendation-models-with-billions-of-ids)
- **32.23** [How do large language models change recommender systems?](#3223-how-do-large-language-models-change-recommender-systems)
- **32.24** [How do you balance short-term engagement with long-term user value in recommendations?](#3224-how-do-you-balance-short-term-engagement-with-long-term-user-value-in-recommendations)
- **32.25** [How do ARIMA models work, and what do stationarity and differencing mean?](#3225-how-do-arima-models-work-and-what-do-stationarity-and-differencing-mean)
- **32.26** [How does exponential smoothing work, and what is the ETS framework?](#3226-how-does-exponential-smoothing-work-and-what-is-the-ets-framework)
- **32.27** [Should you use local statistical models or global machine learning models for forecasting many series?](#3227-should-you-use-local-statistical-models-or-global-machine-learning-models-for-forecasting-many-series)
- **32.28** [What deep learning models and foundation models are used for time-series forecasting?](#3228-what-deep-learning-models-and-foundation-models-are-used-for-time-series-forecasting)
- **32.29** [How do you produce and evaluate probabilistic forecasts and prediction intervals?](#3229-how-do-you-produce-and-evaluate-probabilistic-forecasts-and-prediction-intervals)
- **32.30** [Which forecast accuracy metrics should you use, and how do you design a backtest?](#3230-which-forecast-accuracy-metrics-should-you-use-and-how-do-you-design-a-backtest)
- **32.31** [What is hierarchical forecasting, and how does forecast reconciliation work?](#3231-what-is-hierarchical-forecasting-and-how-does-forecast-reconciliation-work)
- **32.32** [How do you forecast intermittent demand?](#3232-how-do-you-forecast-intermittent-demand)
- **32.33** [How do you use external drivers such as promotions, prices, holidays and weather in forecasts?](#3233-how-do-you-use-external-drivers-such-as-promotions-prices-holidays-and-weather-in-forecasts)
- **32.34** [How do you handle structural breaks, anomalies and regime changes in time series?](#3234-how-do-you-handle-structural-breaks-anomalies-and-regime-changes-in-time-series)
- **32.35** [How do you forecast demand for new products with no sales history?](#3235-how-do-you-forecast-demand-for-new-products-with-no-sales-history)
- **32.36** [How do you run forecasting in production, and how do you make forecasts useful for business planning?](#3236-how-do-you-run-forecasting-in-production-and-how-do-you-make-forecasts-useful-for-business-planning)

---

### 32.1 What is causal inference, and how do the potential outcomes and causal graph frameworks describe it?

**Causal inference** estimates what *would happen* if we intervened: changed a price, launched a feature, sent a discount. Predictive models answer "what is likely to happen given what we observe"; causal questions ask "what difference does our action make". Correlation alone cannot answer that (see 1.16), because the same correlation can come from causation, reverse causation, confounding or selection.

**Potential outcomes framework (Neyman, Rubin)**

- Each unit $i$ has two potential outcomes: $Y_i(1)$ if treated and $Y_i(0)$ if not.
- The **individual treatment effect** is $Y_i(1) - Y_i(0)$.
- **The fundamental problem of causal inference:** we only ever observe one of the two for each unit. The other is a counterfactual.
- So we estimate averages:
  - **ATE** (average treatment effect): $E[Y(1) - Y(0)]$ over the population.
  - **ATT** (on the treated): the average effect among those who actually received the treatment.
  - **CATE** (conditional): $E[Y(1) - Y(0) \mid X = x]$, the effect for a subgroup with features $x$, which is the basis for personalisation and uplift modelling.

**Causal graph framework (Pearl)**

- A **directed acyclic graph (DAG)** encodes assumptions about which variables cause which.
- The **do-operator** distinguishes intervening, $P(Y \mid do(T = t))$, from observing, $P(Y \mid T = t)$.
- Graphical rules (such as the backdoor criterion) tell you which variables to adjust for to identify an effect from observational data.

The two frameworks are compatible: graphs are a convenient way to state and check the assumptions; potential outcomes give the estimands and estimators.

**The hierarchy of evidence in practice**

1. **Randomised experiments** (A/B tests): randomisation makes treatment independent of potential outcomes, so a difference in means estimates the ATE. The gold standard when feasible.
2. **Quasi-experiments:** natural experiments, difference-in-differences, regression discontinuity, instrumental variables, synthetic control.
3. **Observational adjustment:** matching, weighting, regression, double machine learning, which are valid only if all confounders are measured.

**Why it matters for a data leader:** many business questions are causal (did the campaign work, what is the effect of price on demand, which customers should get the retention offer), and answering them with correlational models leads to confident, wrong decisions.

### 32.2 What are confounders, colliders and mediators, and how do you decide which variables to control for?

Choosing control variables is not "add everything available". Some variables must be adjusted for; others must **not** be, or they create bias. A causal graph makes the decision explicit.

**Three basic structures** (treatment $T$, outcome $Y$)

| Structure | Graph | Example | Adjust for it? |
| --- | --- | --- | --- |
| Confounder | $T \leftarrow C \rightarrow Y$ | Customer engagement drives both receiving a promotion and spending more | **Yes**, otherwise the effect is biased |
| Mediator | $T \rightarrow M \rightarrow Y$ | A discount increases basket size, which increases revenue | **No**, if you want the total effect (adjusting removes part of the effect) |
| Collider | $T \rightarrow K \leftarrow Y$ | Both the feature and high spend make users more likely to answer a survey | **No**; conditioning on it (or filtering on it) creates a spurious association |

**The backdoor criterion:** to estimate the effect of $T$ on $Y$, adjust for a set of variables that blocks every "backdoor path" (a path from $T$ to $Y$ that starts with an arrow into $T$), and that contains no descendants of $T$.

**Practical rules**

- Adjust for **pre-treatment** variables that affect both treatment and outcome.
- Do not adjust for variables **caused by the treatment** (mediators, post-treatment outcomes).
- Be wary of **selection**: analysing only customers who responded, only users who stayed, or only approved loans is conditioning on a collider and can create effects that do not exist (survivorship bias).
- Variables that affect only the treatment (instruments) do not need adjustment and can increase variance if included.

**Simpson's paradox** shows why this matters: a treatment can look harmful overall but beneficial within every subgroup (or the reverse), because the subgroups have different treatment rates and baseline outcomes. Whether the aggregated or the stratified answer is right depends on the causal structure, not on the data alone.

**Tools:** draw the DAG with domain experts before analysis; DoWhy and DAGitty can derive valid adjustment sets from a graph.

```python
import networkx as nx

dag = nx.DiGraph([("engagement", "promo"), ("engagement", "spend"),
                  ("promo", "basket_size"), ("basket_size", "spend"), ("promo", "spend")])
# engagement: confounder (adjust); basket_size: mediator (do not adjust for the total effect)
print(list(nx.all_simple_paths(dag.to_undirected(), "promo", "spend")))
```

### 32.3 What assumptions are needed to estimate causal effects from observational data?

Without randomisation, causal estimates rest on assumptions that cannot be fully tested with the data. A senior answer names them and explains how to probe each one.

**1. Unconfoundedness (ignorability, conditional exchangeability)**

- Given the observed covariates $X$, treatment assignment is independent of the potential outcomes: $(Y(1), Y(0)) \perp T \mid X$.
- In words: **all confounders are measured**. This is the strongest and most fragile assumption.
- Probe it with domain knowledge (a DAG), negative control outcomes, and sensitivity analysis (how strong would an unmeasured confounder need to be to change the conclusion?).

**2. Positivity (overlap)**

- Every unit has a non-zero probability of receiving each treatment level: $0 \lt P(T = 1 \mid X) \lt 1$.
- If some customer segments never receive the promotion, there is no data to estimate the effect for them.
- Check by plotting propensity score distributions for treated and control groups; trim or restrict the population where overlap is poor.

**3. SUTVA (stable unit treatment value assumption)**

- **No interference:** one unit's treatment does not affect another unit's outcome. Violated in marketplaces, social networks and shared inventory.
- **Consistency:** there is one well-defined version of the treatment. "Received a discount" is ambiguous if discounts ranged from 5% to 50%.

**4. Correct model specification** (for regression-based estimators)

- Outcome and propensity models must capture the relationships well enough. Flexible ML models and doubly robust estimators reduce this risk.

**5. No measurement or selection problems**

- Treatment and outcome are measured correctly; the analysed sample is not selected on post-treatment variables.

**How to present it:** state the assumptions explicitly with the result, say which are most doubtful, and show sensitivity analysis. Executives should understand that an observational estimate is conditional on these assumptions, unlike a well-run experiment.

### 32.4 How do propensity score matching and inverse probability weighting work?

Both methods use the **propensity score**, $e(x) = P(T = 1 \mid X = x)$, the probability of receiving treatment given covariates. Rosenbaum and Rubin showed that if unconfoundedness holds given $X$, it also holds given $e(X)$, so balancing on a single score is enough.

**Steps common to both**

1. Fit a model of treatment on pre-treatment covariates (logistic regression or gradient-boosted trees, calibrated).
2. Check **overlap**: plot propensity distributions for treated and control units; trim units with scores near 0 or 1.

**Propensity score matching**

- Match each treated unit with one or more control units with similar propensity scores (nearest neighbour, often within a caliper).
- Estimate the ATT as the average outcome difference within matched pairs.
- Check **covariate balance** after matching with standardised mean differences (below about 0.1 is a common target).
- Simple to explain, but discards unmatched data and is sensitive to matching choices.

**Inverse probability weighting (IPW)**

- Weight each unit by the inverse probability of the treatment it actually received: treated units by $1/e(x)$, control units by $1/(1 - e(x))$. This creates a pseudo-population in which treatment is independent of $X$.
- $\hat{\tau}_{\text{ATE}} = \frac{1}{n} \sum_i \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1 - T_i) Y_i}{1 - e(X_i)} \right]$
- Uses all data, but extreme weights (scores near 0 or 1) cause high variance. Use stabilised weights, trimming or weight clipping.

**Pitfalls**

- Neither method fixes **unmeasured confounding**.
- Including post-treatment variables in the propensity model introduces bias.
- A propensity model that predicts treatment almost perfectly signals poor overlap, not a good model.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression(max_iter=1000).fit(X, t)
e = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)       # clip to limit extreme weights

ate_ipw = np.mean(t * y / e - (1 - t) * y / (1 - e))

# balance check: standardised mean difference of each covariate after weighting
w = np.where(t == 1, 1 / e, 1 / (1 - e))
def smd(col):
    m1 = np.average(col[t == 1], weights=w[t == 1]); m0 = np.average(col[t == 0], weights=w[t == 0])
    return (m1 - m0) / np.sqrt((col[t == 1].var() + col[t == 0].var()) / 2)
print(ate_ipw, [round(smd(X[:, j]), 3) for j in range(X.shape[1])])
```

### 32.5 What are doubly robust estimators and double machine learning?

Both combine an **outcome model** and a **propensity model** so that the causal estimate stays reliable even when one of them is imperfect, and both allow flexible machine learning models without biasing the result.

**Doubly robust estimation (AIPW)**

The augmented inverse probability weighted estimator:

$\hat{\tau} = \frac{1}{n} \sum_i \left[ \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1 - T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)} \right]$

- $\hat{\mu}_1$ and $\hat{\mu}_0$ predict the outcome under treatment and control; $\hat{e}$ is the propensity score.
- **Doubly robust:** the estimate is consistent if *either* the outcome model *or* the propensity model is correct.
- The weighted residual terms correct the outcome model's errors using the propensity score.

**Double (debiased) machine learning (DML)**

Chernozhukov et al. (2018) give a general recipe for using ML models in causal estimation without the regularisation bias they introduce:

1. Predict the outcome from covariates, $\hat{m}(X) \approx E[Y \mid X]$, and the treatment from covariates, $\hat{e}(X) \approx E[T \mid X]$, with any ML models.
2. Compute **residuals**: $\tilde{Y} = Y - \hat{m}(X)$ and $\tilde{T} = T - \hat{e}(X)$.
3. Regress $\tilde{Y}$ on $\tilde{T}$; the slope is the treatment effect (in a partially linear model).
4. Use **cross-fitting**: fit the nuisance models on one fold and compute residuals on another, rotating folds. This prevents overfitting from leaking into the estimate and gives valid confidence intervals.

**Why it matters**

- Lets you use gradient-boosted trees or neural networks to capture complex confounding with many covariates, while still getting a valid effect estimate and confidence interval.
- Extends to heterogeneous effects (DML-based CATE estimators, R-learner).
- Libraries: EconML, DoubleML, CausalML.

**It does not remove the core assumption:** all confounders must still be in $X$.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold

y_res, t_res = np.zeros(len(y)), np.zeros(len(y))
for train, test in KFold(5, shuffle=True, random_state=0).split(X):          # cross-fitting
    m = GradientBoostingRegressor().fit(X[train], y[train])
    e = GradientBoostingClassifier().fit(X[train], t[train])
    y_res[test] = y[test] - m.predict(X[test])
    t_res[test] = t[test] - e.predict_proba(X[test])[:, 1]

theta = np.sum(t_res * y_res) / np.sum(t_res ** 2)                           # effect estimate
se = np.sqrt(np.mean((y_res - theta * t_res) ** 2 * t_res ** 2)) / (np.mean(t_res ** 2) * np.sqrt(len(y)))
print(f"effect = {theta:.3f} +/- {1.96 * se:.3f}")
```

### 32.6 How does difference-in-differences work, and what can go wrong?

**Difference-in-differences (DiD)** estimates a causal effect when a treatment starts at a known time for one group but not another, for example a feature launched in some countries, or a policy applied in some stores.

**The estimate**

$`\hat{\tau} = (\bar{Y}_{\text{treated, after}} - \bar{Y}_{\text{treated, before}}) - (\bar{Y}_{\text{control, after}} - \bar{Y}_{\text{control, before}})`$

The control group's change estimates what would have happened to the treated group without treatment. Subtracting it removes time trends common to both groups and fixed differences between the groups.

**The key assumption: parallel trends**

- Without the treatment, the treated and control groups would have followed **parallel paths** (the same change over time), even if their levels differ.
- It cannot be proven, but can be checked: plot both groups before the treatment; run an **event study** with leads and lags; non-zero "effects" before treatment suggest the assumption fails.

**Regression form**

$Y_{it} = \alpha_i + \lambda_t + \tau \cdot D_{it} + \varepsilon_{it}$, with unit fixed effects $\alpha_i$, time fixed effects $\lambda_t$ and treatment indicator $D_{it}$. Cluster standard errors by unit, because observations of the same unit over time are correlated.

**What can go wrong**

- **Different pre-trends:** the groups were already diverging; consider synthetic control or matching on pre-period trends.
- **Anticipation:** units change behaviour before the official start.
- **Spillovers:** treatment affects the control group (for example, customers shift purchases between treated and untreated stores).
- **Composition changes:** different customers in the groups before and after.
- **Staggered adoption:** when units are treated at different times, the standard two-way fixed effects regression can be badly biased if effects vary over time, because already-treated units act as controls. Use estimators designed for this (Callaway and Sant'Anna, Sun and Abraham).
- **Few clusters:** with only a handful of treated units, standard errors are unreliable; use permutation or wild bootstrap inference.

```python
import statsmodels.formula.api as smf

# df: one row per store and week, with columns sales, treated (group), post (period), store_id, week
model = smf.ols("sales ~ treated:post + C(store_id) + C(week)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["store_id"]})
print(model.params["treated:post"], model.conf_int().loc["treated:post"].tolist())
```

### 32.7 What is synthetic control, and when would you use it?

**Synthetic control** (Abadie and colleagues) estimates the effect of an intervention on a **single unit or a few units** (a country, a city, a large client) when no single comparison unit resembles it well.

**Idea**

- Build a "synthetic" version of the treated unit as a **weighted combination of untreated units** (the donor pool).
- Choose non-negative weights that sum to one so that the synthetic unit matches the treated unit's outcome (and key predictors) closely **before** the intervention.
- After the intervention, the gap between the treated unit and its synthetic control estimates the effect.

**Example:** a new pricing model launched only in Germany. Combine other European markets (for example 40% Austria, 35% Netherlands, 25% Belgium) to reproduce Germany's pre-launch sales; the post-launch gap is the estimated effect.

**Inference**

- Standard errors do not apply with one treated unit. Use **placebo tests**: apply the same method to each donor unit as if it were treated, and check whether the real treated unit's gap is unusually large compared with the placebo gaps.
- Placebo-in-time tests (pretend the intervention happened earlier) check for spurious effects.

**When to use it**

- One or a few treated aggregate units, a long pre-period, and a reasonable donor pool not affected by the treatment.
- Geographic launches, policy changes, market-level marketing campaigns.

**Related methods**

- **Bayesian structural time series** (Google's CausalImpact): models the treated series using control series as covariates, and gives posterior intervals for the effect.
- **Synthetic difference-in-differences:** combines unit weights (like synthetic control) with time weights and DiD, often more robust.

**Pitfalls:** poor pre-period fit (the method should not be trusted then); donors affected by spillovers; interpolation bias when the treated unit is outside the range of donors; and cherry-picking the donor pool.

```python
import numpy as np
from scipy.optimize import minimize

# Y0_pre: (T_pre, J) donor outcomes before launch; y1_pre: (T_pre,) treated unit before launch
J = Y0_pre.shape[1]
loss = lambda w: np.sum((y1_pre - Y0_pre @ w) ** 2)
res = minimize(loss, np.full(J, 1 / J), bounds=[(0, 1)] * J,
               constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
w = res.x
effect = y1_post - Y0_post @ w                 # estimated effect for each post-launch period
```

### 32.8 What are instrumental variables and regression discontinuity designs?

Both are quasi-experimental methods that recover causal effects when there is **unmeasured confounding**, by exploiting a source of variation in treatment that is as good as random.

**Instrumental variables (IV)**

An instrument $Z$ affects the treatment $T$ but affects the outcome $Y$ only through $T$.

Requirements:
1. **Relevance:** $Z$ is strongly correlated with $T$ (check the first-stage F-statistic; a common rule of thumb is above 10).
2. **Exclusion:** $Z$ affects $Y$ only through $T$ (untestable; argued from domain knowledge).
3. **Independence:** $Z$ is as good as randomly assigned with respect to unmeasured confounders.

**Estimation: two-stage least squares (2SLS)**

1. Regress $T$ on $Z$ (and covariates), get predicted $\hat{T}$.
2. Regress $Y$ on $\hat{T}$; the coefficient is the causal effect.

**Examples:** random encouragement (a randomised email inviting users to try a feature, where the feature use is the treatment); an A/B test with imperfect compliance, where assignment is the instrument for actual exposure; distance to a facility as an instrument for using it.

**Interpretation:** with heterogeneous effects, IV estimates the **local average treatment effect (LATE)**, the effect for "compliers" whose treatment is changed by the instrument, not for everyone.

**Regression discontinuity design (RDD)**

Treatment is assigned by a threshold on a running variable: customers with a credit score above 700 get an offer; students above a cut-off get a scholarship.

- Units just above and just below the threshold are very similar, so comparing their outcomes estimates the effect **at the threshold**.
- **Sharp RDD:** treatment is fully determined by the threshold. **Fuzzy RDD:** the probability of treatment jumps at the threshold, analysed like IV.
- Estimate with local linear regression on each side within a bandwidth (data-driven bandwidth selection).

**Checks for RDD:** no manipulation of the running variable around the threshold (density test, such as McCrary's); covariates continuous at the threshold; robustness to bandwidth choice; placebo thresholds.

**Business uses:** eligibility rules, score-based decisions (credit, fraud, loyalty tiers) and automated thresholds are common, which makes RDD especially practical inside companies.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# 2SLS by hand (no covariates): z instrument, t treatment, y outcome
t_hat = LinearRegression().fit(z.reshape(-1, 1), t).predict(z.reshape(-1, 1))
effect = LinearRegression().fit(t_hat.reshape(-1, 1), y).coef_[0]
# note: standard errors from this naive second stage are wrong; use a 2SLS library (linearmodels IV2SLS)
```

### 32.9 What is uplift modelling, and how do you estimate heterogeneous treatment effects?

Average effects hide large differences between people. **Uplift modelling** estimates the **conditional average treatment effect (CATE)**, $\tau(x) = E[Y(1) - Y(0) \mid X = x]$, so you can target treatments (discounts, retention calls, ads) at the people whose behaviour actually changes.

**Why response models are not enough:** a model predicting who will buy targets "sure things" who would buy anyway. Uplift modelling distinguishes four groups:

| Group | Without treatment | With treatment | Target? |
| --- | --- | --- | --- |
| Persuadables | No | Yes | **Yes** |
| Sure things | Yes | Yes | No (wasted cost) |
| Lost causes | No | No | No |
| Sleeping dogs | Yes | No | **Never** (treatment causes harm, for example a retention call that prompts cancellation) |

**Data:** ideally from a **randomised experiment** (treatment assigned at random), so the estimates are unconfounded. Observational data requires the adjustments covered earlier.

**Meta-learners**

- **S-learner:** one model with treatment as a feature; uplift is the prediction with $T = 1$ minus with $T = 0$. Simple, but may ignore the treatment if its effect is small.
- **T-learner:** separate models for treated and control; uplift is the difference. Noisy when groups are imbalanced.
- **X-learner:** imputes individual effects using the other group's model, then models them; good when one group is much larger.
- **R-learner and DR-learner:** use residualisation or doubly robust pseudo-outcomes; strong theoretical properties.
- **Causal forests** (Wager and Athey): random forests that split to maximise treatment effect differences, with confidence intervals.

**Evaluation** (individual effects are never observed)

- **Uplift (Qini) curves:** rank people by predicted uplift, and compare treated and control outcomes cumulatively; the area under the curve (AUUC, Qini coefficient) summarises it.
- Check that the top-ranked segments show a large observed difference in a held-out experiment.

**Deployment:** treat the top segment by predicted uplift subject to budget, keep a random holdout to measure real incremental impact, and include cost: target where uplift times value exceeds the treatment cost.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# T-learner on randomised campaign data: X features, t assignment (0/1), y conversion (0/1)
m1 = GradientBoostingClassifier().fit(X[t == 1], y[t == 1])
m0 = GradientBoostingClassifier().fit(X[t == 0], y[t == 0])
uplift = m1.predict_proba(X_new)[:, 1] - m0.predict_proba(X_new)[:, 1]

value, cost = 40.0, 2.0
target = uplift * value > cost                     # treat only where expected incremental value beats cost
```

### 32.10 How do you run experiments when users or units interfere with each other?

Standard A/B tests (see 28.27) assume **SUTVA**: one user's treatment does not affect another's outcome. In many important products this fails, and naive tests give badly biased answers.

**Where interference occurs**

- **Two-sided marketplaces:** a pricing or ranking change for some riders affects drivers available to all riders; treated buyers buy inventory that control buyers then cannot.
- **Social networks:** treated users share content with control users.
- **Shared resources:** ads auctions and budgets, delivery capacity, inventory.
- **Learning systems:** a recommender trained on data from both arms leaks the treatment's effect into the control.

**Symptoms:** the experiment shows a positive effect, but after full launch the aggregate metric does not move (for example, treated buyers simply took supply from control buyers, so the total did not grow).

**Designs that reduce bias**

| Design | How it works | Trade-off |
| --- | --- | --- |
| Cluster randomisation | Randomise groups that interact mostly within themselves (cities, regions, social graph clusters) | Fewer units, so higher variance |
| Switchback experiments | Alternate the whole market between treatment and control over time windows (for example 30-minute or hourly blocks) | Carryover effects between windows; need washout periods and careful analysis |
| Two-sided randomisation | Randomise both buyers and sellers to estimate and correct marketplace bias | More complex design and analysis |
| Budget or supply splitting | Give each arm its own share of budget or inventory (common for ads) | Not always possible |
| Ego-network randomisation | Treat a user and their close network together | Needs graph data; complex |
| Holdout regions or long-term holdouts | Keep an untreated region or group to measure aggregate effects | Slower, less precise |

**Analysis considerations**

- The unit of analysis must match the unit of randomisation (cluster-level statistics, cluster-robust standard errors).
- Switchbacks require adjusting for time effects and dropping or modelling transition periods.
- Variance reduction (CUPED with pre-period data) helps compensate for fewer randomisation units.

**Leadership angle:** for marketplace and network products, invest in experimentation designs that measure **global** effects; decisions based on biased user-level tests can systematically favour changes that only redistribute value.

### 32.11 How do you measure the true incremental impact of marketing?

Marketing measurement asks a causal question: **how many extra sales did this spend cause?** Common reporting methods answer a different question, which leads to large budget misallocations.

**Attribution is not incrementality**

- **Last-click and multi-touch attribution** assign credit for conversions to touchpoints that happened before them. They do not ask whether the conversion would have happened anyway. Branded search ads, for example, get credit for customers already looking for the brand.
- **Incrementality** is the difference between outcomes with and without the marketing, which requires a counterfactual.

**Methods**

1. **Randomised holdouts:** withhold a campaign (emails, push notifications, retargeting) from a random group and compare. Cleanest when you control delivery.
2. **Platform lift studies:** advertising platforms run randomised conversion lift tests by withholding ads from a control group (often using "ghost ads" to record who would have seen an ad).
3. **Geo experiments:** turn spend up or down in randomly chosen regions, and compare with control regions (with DiD or synthetic control). Works when user-level randomisation is impossible (TV, out-of-home, privacy restrictions).
4. **Marketing mix modelling (MMM):** a regression of sales on spend by channel over time, with controls for seasonality, price, promotions and macro factors. Key components:
   - **Adstock:** effects carry over into later periods (geometric decay).
   - **Saturation:** diminishing returns as spend increases (Hill or logistic curves).
   - **Bayesian estimation** with priors, which helps with limited data (open-source tools include Meta's Robyn, Google's Meridian and PyMC-Marketing).
   - **Calibration:** use lift and geo experiment results as priors or constraints, because MMM alone is observational and can be confounded (spend often increases when demand is already high).

**A practical measurement system**

- MMM for strategic budget allocation across channels, updated quarterly.
- Experiments to validate and calibrate MMM, prioritising the largest channels and the most uncertain estimates.
- Attribution only for fast, tactical optimisation within a channel, with awareness of its bias.

**Outputs that executives use:** incremental return on ad spend (iROAS), marginal return curves per channel (where the next dollar works best), and recommended budget reallocations with uncertainty ranges.

```python
import numpy as np

def adstock(spend, decay=0.5):
    out = np.zeros_like(spend, dtype=float)
    for i, s in enumerate(spend):
        out[i] = s + (decay * out[i - 1] if i else 0.0)    # carry-over from previous periods
    return out

def hill(x, half_sat, slope=1.0):
    return x ** slope / (x ** slope + half_sat ** slope)   # diminishing returns

media_effect = hill(adstock(weekly_tv_spend, decay=0.6), half_sat=50_000)
```

### 32.12 How do you validate a causal analysis and communicate its uncertainty?

Observational causal estimates depend on assumptions that cannot be tested directly, so credibility comes from **stress-testing** the result and being transparent about what could break it.

**Validation techniques**

- **Placebo (falsification) tests:**
  - *Placebo treatment:* replace the real treatment with a random one; the estimated effect should be near zero.
  - *Placebo outcome:* estimate the effect on an outcome the treatment cannot plausibly affect (for example, purchases made before the treatment); a non-zero effect suggests confounding.
  - *Placebo time:* pretend the intervention happened earlier.
- **Negative controls:** variables known to have no causal relationship, used to detect hidden bias.
- **Refutation tests** (as in DoWhy): add a random common cause, use a random subset of data, replace the treatment with a placebo; a robust estimate should not change much.
- **Sensitivity analysis:** quantify how strong an unmeasured confounder would have to be to explain away the effect. The **E-value** (VanderWeele and Ding) gives the minimum strength of association, on the risk ratio scale, that a confounder would need with both treatment and outcome.
- **Method triangulation:** compare several approaches with different assumptions (matching, DML, DiD); agreement increases confidence.
- **Benchmark against experiments:** where some experiments exist, check whether the observational method reproduces their results.

**Communicating to decision-makers**

- State the **question** in plain words ("How many extra renewals did the new onboarding cause in Q3?").
- Give the estimate **with a range**, and say what the range covers (statistical uncertainty only, not assumption failures).
- List the **key assumptions** and how plausible they are; flag the biggest threat.
- Show the **robustness checks** briefly.
- Translate to the **decision**: "Even at the low end of the range, the programme pays for itself", or "The evidence is too weak to justify scaling; we recommend a randomised test in two regions."
- When the stakes are high and an experiment is possible, recommend running one rather than relying on observational analysis.

```python
import numpy as np

def e_value(rr):
    """E-value for an observed risk ratio (VanderWeele and Ding, 2017)."""
    rr = rr if rr >= 1 else 1 / rr
    return rr + np.sqrt(rr * (rr - 1))

print(e_value(1.8))   # 3.0: a confounder would need RR of about 3 with both treatment and outcome to explain the effect away
```

### 32.13 How does matrix factorisation work for implicit feedback, and what are ALS and BPR?

Basic matrix factorisation (see 9.36) learns user and item vectors whose dot product reconstructs explicit ratings. Most real data is **implicit feedback**: clicks, views, purchases, listening time. Implicit data needs different handling.

**Why implicit feedback is different**

- There are **no negative labels**: a missing interaction may mean "not interested" or simply "never seen".
- Values express **confidence**, not preference: playing a song 50 times signals more certainty than playing it once.
- The matrix is extremely sparse, and every unobserved cell matters, so methods must handle all of them efficiently.

**Weighted matrix factorisation with ALS (Hu, Koren and Volinsky, 2008)**

- Binary preference $p_{ui} = 1$ if user $u$ interacted with item $i$, else 0.
- Confidence $c_{ui} = 1 + \alpha r_{ui}$, where $r_{ui}$ is the interaction count.
- Minimise $\sum_{u,i} c_{ui} (p_{ui} - x_u^\top y_i)^2 + \lambda (\sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2)$ over **all** user-item pairs.
- **Alternating least squares:** fix item vectors and solve each user's vector in closed form, then fix users and solve items. Each step is a least-squares problem, parallelises well (Spark MLlib's ALS), and a computational trick avoids iterating over all zero entries explicitly.

**Bayesian personalised ranking (BPR, Rendle et al., 2009)**

- Treats recommendation as **ranking**: for a user, an item they interacted with should score higher than one they did not.
- Samples triples $(u, i, j)$ with $i$ observed and $j$ unobserved, and maximises $`\sum \ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})`$ with regularisation, using stochastic gradient descent.
- Optimises a pairwise ranking objective related to AUC, which aligns better with top-N recommendation than squared error.

**Choosing in practice**

- ALS: strong, fast, simple baseline for implicit data; easy to scale; tune factors, $\lambda$ and $\alpha$.
- BPR: directly optimises ranking; sensitive to negative sampling.
- Both are **strong baselines** that well-tuned studies show are often competitive with much more complex neural models. Always include them.

```python
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# interactions: arrays of user_idx, item_idx, counts
user_items = csr_matrix((counts, (user_idx, item_idx)))
model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=40.0, iterations=20)
model.fit(user_items)
ids, scores = model.recommend(userid=42, user_items=user_items[42], N=10, filter_already_liked_items=True)
```

### 32.14 How are two-tower retrieval models trained, and what is sampling bias correction?

**Two-tower (dual encoder) models** are the standard first stage of large recommender systems. A **user tower** encodes the user and context into a vector; an **item tower** encodes an item into a vector in the same space. Relevance is their dot product, so item vectors can be precomputed and searched with approximate nearest neighbour indexes.

**Inputs**

- User tower: user ID embedding, history (pooled or sequence-encoded embeddings of recent items), demographics, context (time, device, surface).
- Item tower: item ID embedding plus content features (category, text and image embeddings, price, age). Content features help new items, which have no ID history.

**Training with in-batch negatives**

- Each training example is a positive (user, item) pair from logs.
- For a batch of $B$ pairs, every other item in the batch serves as a negative for each user. A softmax over the $B$ items gives a sampled softmax loss:

$\mathcal{L} = -\log \frac{\exp(s(u, i^+) / \tau)}{\sum_{j \in \text{batch}} \exp(s(u, j) / \tau)}$

- Cheap: the negatives' embeddings are already computed.

**Sampling bias correction (logQ correction)**

- In-batch negatives are sampled in proportion to item **popularity** (popular items appear in more batches). The model then over-penalises popular items and under-penalises rare ones.
- Correction (Yi et al., 2019, YouTube): subtract $\log Q(j)$ from each logit, where $Q(j)$ is the item's sampling probability (estimated by streaming frequency counts). This makes the sampled softmax approximate the full softmax.

**Other important details**

- **Temperature** $\tau$ strongly affects results; tune it.
- **L2-normalise** embeddings to stabilise training and use cosine similarity.
- Add **random negatives** from the whole catalogue (mixed negative sampling) so that rare items are also seen as negatives.
- Item embeddings change as the model trains; refresh the ANN index regularly and keep model and index versions aligned.

**Limitations:** the dot product cannot capture rich user-item feature interactions; that is the job of the later ranking stage.

```python
import torch
import torch.nn.functional as F

def in_batch_loss(user_emb, item_emb, item_log_q, temperature=0.05):
    u = F.normalize(user_emb, dim=-1)
    v = F.normalize(item_emb, dim=-1)
    logits = u @ v.T / temperature                  # (B, B): row i's positive is column i
    logits = logits - item_log_q.unsqueeze(0)       # logQ correction for popularity-biased sampling
    labels = torch.arange(len(u), device=u.device)
    return F.cross_entropy(logits, labels)
```

### 32.15 How do you choose negative examples for training recommender models?

With implicit feedback, positives are logged interactions, and **negatives must be chosen**. The choice changes what the model learns as much as the architecture does.

**Sources of negatives**

| Negative type | Description | Effect |
| --- | --- | --- |
| Random (uniform) | Random items from the catalogue | Easy; teaches coarse relevance; most are trivially irrelevant |
| Popularity-sampled / in-batch | Items sampled roughly by popularity | Harder; needs logQ correction to avoid popularity bias |
| Impressed but not clicked | Items shown to the user who ignored them | Real, informative negatives; affected by position bias (the user may not have seen them) |
| Hard negatives | Items the current model scores highly but the user did not choose | Teaches fine distinctions; risk of **false negatives** |
| Explicit negatives | Dislikes, skips, "not interested", returns | Strongest signal, but rare |

**Stage-specific guidance**

- **Retrieval (candidate generation):** must separate relevant items from millions of irrelevant ones, so random and in-batch negatives matter; add some hard negatives for sharper boundaries.
- **Ranking:** works on candidates that are all somewhat relevant, so **impressions without engagement** are the natural negatives, with features for position to correct bias.

**Pitfalls**

- **False negatives:** many unobserved items would be liked if shown. Very hard negatives mined from the model's top results often include them, which hurts training. Mine from moderately ranked items, or filter by similarity to positives.
- **Too easy:** only random negatives produce a model that cannot rank among plausible items.
- **Mismatch with serving:** negatives should resemble the items the model must distinguish between at serving time.
- **Weak signals treated as negative:** a short view may be mild interest; consider graded labels (purchase over click over view) or weights.

**Evaluation tip:** report how results change with the negative sampling strategy; it is a key hyperparameter, not an implementation detail.

### 32.16 How do sequential and generative recommenders work?

User behaviour is a **sequence**: what someone watched or bought in the last few minutes often predicts the next action better than their long-term profile. Sequential recommenders model the order of interactions.

**Evolution of sequential models**

- **Markov chains and item-to-item transitions:** "people who viewed A next viewed B". Simple and strong for short sessions.
- **GRU4Rec:** recurrent networks over session click sequences, trained to predict the next item.
- **SASRec:** a causal (left-to-right) transformer over the user's item sequence, predicting the next item at each position.
- **BERT4Rec:** a bidirectional transformer trained by masking items in the sequence and predicting them.
- Modern industrial systems feed long behaviour sequences (hundreds to thousands of events) into transformer encoders in the ranking stage, with attention between the candidate item and the history (target attention, as in Alibaba's DIN and its successors).

**Generative recommenders**

Recent work treats recommendation like language modelling:

- **Semantic IDs:** instead of an arbitrary ID per item, each item gets a short sequence of discrete codes learned from its content embedding (for example with residual quantisation). Similar items share code prefixes.
- **Generative retrieval:** a sequence model reads the user's history (as semantic IDs) and **generates the semantic ID of the next item** token by token, with beam search producing candidates (as in Google's TIGER). This replaces the ANN index with generation and helps new items, which get meaningful codes from their content.
- **Large sequential models:** Meta's generative recommender work (HSTU) reframes ranking and retrieval as sequential transduction over very long action sequences and reports quality improving with model scale, similar to scaling laws in language models.

**Practical considerations**

- Sequence length versus latency: long histories are expensive at serving; cache encoded history, or summarise older behaviour.
- Combine short-term sequence signals with long-term preferences and context.
- Training data leakage: evaluate with a time-based split, predicting interactions after a cut-off.
- Strong, well-tuned baselines (item-to-item, SASRec) should always be compared before adopting complex generative architectures.

### 32.17 How do ranking models handle feature interactions and multiple objectives?

The ranking stage scores a few hundred candidates with rich features. Two design questions dominate: how to model **interactions between features**, and how to optimise **several objectives** at once (click, watch time, purchase, satisfaction).

**Feature interaction architectures**

| Model | Idea |
| --- | --- |
| Logistic regression with crosses | Hand-crafted feature crosses; strong, fast baseline |
| Factorisation machines (FM) | Every pair of features interacts through the dot product of learned embeddings: $\sum_{i\lt j} \langle v_i, v_j \rangle x_i x_j$; handles sparse data well |
| Wide & Deep | A linear "wide" part for memorising crosses plus a deep MLP for generalisation |
| DeepFM | An FM component and a deep component sharing embeddings, no manual crosses |
| DCN-v2 | Explicit cross layers, $x_{l+1} = x_0 \odot (W x_l + b) + x_l$, learn bounded-degree interactions efficiently, alongside an MLP |
| DLRM | Embedding tables for categorical features, an MLP for dense features, pairwise dot products between all embeddings, then a top MLP |
| Attention-based (DIN, transformers) | Attention between the candidate item and the user's behaviour history |

**Multi-task learning**

Optimising only clicks favours clickbait; real objectives are several, and their labels differ in volume and delay.

- **Shared-bottom:** shared layers feed task-specific heads. Simple, but conflicting tasks can hurt each other (negative transfer).
- **MMoE (multi-gate mixture of experts):** several expert networks shared across tasks, with a separate gating network per task choosing how to mix them. Handles task conflicts better.
- **PLE (progressive layered extraction):** separates shared and task-specific experts explicitly, reducing negative transfer further.
- **Conditional tasks:** model purchase given click (as in ESMM), so the conversion model is trained on the full impression space and avoids sample selection bias.

**Combining objectives into one score**

- Weighted sum or product of predicted probabilities and values, for example $\text{score} = P(\text{click})^{a} \cdot E[\text{watch time}]^{b} \cdot (1 - P(\text{dislike}))^{c}$.
- Tune weights with online experiments against long-term metrics, not offline metrics.
- Keep predictions **calibrated**, because weights assume meaningful probabilities.

```python
import torch
import torch.nn as nn

class CrossLayer(nn.Module):                         # DCN-v2 cross layer
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x0, xl):
        return x0 * self.linear(xl) + xl

class MMoE(nn.Module):
    def __init__(self, dim, n_experts=4, n_tasks=2, hidden=128):
        super().__init__()
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(dim, hidden), nn.ReLU()) for _ in range(n_experts))
        self.gates = nn.ModuleList(nn.Linear(dim, n_experts) for _ in range(n_tasks))
        self.heads = nn.ModuleList(nn.Linear(hidden, 1) for _ in range(n_tasks))
    def forward(self, x):
        expert_out = torch.stack([e(x) for e in self.experts], dim=1)          # (B, E, H)
        outs = []
        for gate, head in zip(self.gates, self.heads):
            w = torch.softmax(gate(x), dim=-1).unsqueeze(-1)                     # (B, E, 1)
            outs.append(torch.sigmoid(head((w * expert_out).sum(dim=1))))
        return outs                                                              # one probability per task
```

### 32.18 How do you handle cold start for new users and new items?

**Cold start** is the lack of interaction history: a new user, a new item, or a new market. Collaborative filtering depends on history, so cold start needs other signals and deliberate exploration.

**New items**

- **Content-based representations:** text, image and attribute embeddings place a new item near similar existing items from the moment it is created. Two-tower models with content features in the item tower handle this naturally.
- **Embedding warm-up:** initialise a new item's ID embedding from the average of similar items' embeddings, or from a model that maps content features to ID embedding space.
- **Exploration traffic:** reserve a small share of impressions for new items (bandits, boosted slots) so they can collect feedback quickly.
- **Semantic IDs** in generative recommenders give new items meaningful codes from content.
- **Creator or brand priors:** a new item from a popular creator or brand starts with a reasonable prior.

**New users**

- **Context:** location, device, time, language, referral source, landing page or campaign.
- **Popular and trending items** by segment as a strong baseline.
- **Onboarding:** ask for a few interests or let users pick items; keep it short.
- **Fast session adaptation:** sequential models that update recommendations within the first session based on early clicks.
- **Cross-domain signals:** behaviour from another product or surface the user already uses, where privacy rules allow.
- **Meta-learning and fast adaptation** approaches learn to personalise from very few interactions.

**New system or market (no data at all)**

- Rules and editorial curation, content-based similarity, popularity from comparable markets, then transition to learned models as data accumulates.

**Measuring cold-start performance**

- Evaluate separately on new users and new items (for example, users with fewer than five interactions); average metrics are dominated by warm users.
- Track how quickly new items reach a fair share of exposure, and how quickly new users reach engagement levels of established users.

**Business framing:** cold start matters most where catalogue turnover is high (news, marketplaces, events, fashion) and for growth, because every new user starts cold.

### 32.19 What biases affect recommender systems, and how do you correct them?

Recommender systems learn from data that **they themselves helped create**: users can only click what was shown, in the position it was shown. This creates biases that, uncorrected, reinforce themselves.

**Main biases**

| Bias | What happens |
| --- | --- |
| Position bias | Items at the top get more clicks regardless of relevance |
| Exposure (selection) bias | No feedback exists for items never shown; missing data is not random |
| Popularity bias | Popular items get more exposure, more clicks and more training signal, so they get recommended even more |
| Conformity bias | Visible popularity signals (ratings, view counts) influence user behaviour |
| Presentation bias | Thumbnails, layout and surface affect engagement independently of relevance |

**The feedback loop:** the model recommends, users respond to what was recommended, and the next model is trained on those responses. Over time this narrows recommendations (filter bubbles), starves new and niche items, and makes offline metrics look better than real quality.

**Corrections**

- **Position bias:**
  - Train with position as a feature, then set it to a fixed value at serving.
  - A separate "shallow tower" models position effects and is dropped at serving.
  - Inverse propensity scoring, with examination probabilities estimated from randomised experiments (swapping positions on small traffic) or click models.
- **Exposure bias:** inverse propensity weighting of observed interactions by their exposure probability; log the propensity with every impression.
- **Popularity bias:** logQ correction in retrieval, regularisation or re-ranking towards long-tail items, and exposure-aware objectives.
- **Breaking the loop:** reserve **exploration** traffic (random or bandit-driven) that produces less biased data, and use it for training and evaluation.

**Evaluation:** off-policy evaluation on exploration data, and online A/B tests measuring coverage, diversity and long-term engagement as well as clicks.

```python
import torch
import torch.nn as nn

class RankerWithPositionTower(nn.Module):
    def __init__(self, main: nn.Module, n_positions=50):
        super().__init__()
        self.main = main
        self.position_bias = nn.Embedding(n_positions, 1)    # learns how much each slot inflates clicks
    def forward(self, features, position=None):
        logit = self.main(features)
        if position is not None:                              # training: add position effect
            logit = logit + self.position_bias(position).squeeze(-1)
        return logit                                          # serving: call without position
```

### 32.20 Beyond accuracy, what objectives should a recommender optimise: diversity, novelty, serendipity and fairness?

A recommender that only maximises predicted clicks tends to show near-duplicates, popular items and safe choices. Users get bored, catalogues go unused, and suppliers or creators are treated unfairly. **Beyond-accuracy objectives** address this.

**Definitions and measures**

- **Diversity:** how different the items in one list are; measured by average pairwise dissimilarity (intra-list diversity), category or topic coverage.
- **Novelty:** recommending items the user is unlikely to know; often measured by the inverse popularity of recommended items.
- **Serendipity:** relevant **and** unexpected; harder to measure, sometimes through user surveys or items outside the user's usual categories that still earn engagement.
- **Coverage:** share of the catalogue that gets recommended to anyone.
- **Calibration:** the mix of recommended categories matches the user's interest mix (a user who watches 70% drama and 30% comedy should not get 100% drama).
- **Provider fairness and exposure:** how exposure is distributed across sellers, creators or new entrants; important in marketplaces and for regulatory scrutiny.

**How to achieve them**

- **Re-ranking** after the main ranker:
  - **Maximal marginal relevance (MMR):** select items one at a time, balancing relevance against similarity to items already selected: $\text{MMR} = \arg\max_i \left[ \lambda \cdot \text{rel}(i) - (1 - \lambda) \max_{j \in S} \text{sim}(i, j) \right]$.
  - Determinantal point processes for diversity-aware slate selection.
  - Business rules: category caps, limits on items from one seller, freshness quotas.
- **Multi-objective scoring** with explicit weights for diversity or novelty.
- **Exposure constraints** for provider fairness, ensuring minimum exposure for new or small suppliers.

**Trade-offs and evaluation**

- More diversity usually lowers short-term click-through slightly but can improve long-term engagement, retention and catalogue health. Only **long-running online experiments** reveal the true trade-off.
- Report a dashboard of metrics, not one number: engagement, diversity, coverage, exposure distribution, and user satisfaction.

```python
import numpy as np

def mmr(scores, item_emb, k=10, lam=0.7):
    item_emb = item_emb / np.linalg.norm(item_emb, axis=1, keepdims=True)
    selected, candidates = [], list(range(len(scores)))
    while candidates and len(selected) < k:
        def gain(i):
            redundancy = max((item_emb[i] @ item_emb[j] for j in selected), default=0.0)
            return lam * scores[i] - (1 - lam) * redundancy
        best = max(candidates, key=gain)
        selected.append(best); candidates.remove(best)
    return selected
```

### 32.21 How do you evaluate recommender systems offline, and why do offline and online results often disagree?

Basic ranking metrics are covered in 12.17. At a senior level, the hard part is designing an **evaluation protocol** that predicts online results.

**Protocol choices that matter**

- **Time-based splits:** train on interactions before a cut-off and test on those after. Random splits leak future information (the model learns from interactions that happened after the ones it is tested on) and inflate results.
- **Leave-last-out** per user is common in research but can still leak global trends; a global time cut-off is more realistic.
- **Full ranking versus sampled metrics:** many papers rank the true item against 100 randomly sampled negatives. Krichene and Rendle (2020) showed sampled metrics can **change the ranking of models** compared with evaluation over the full catalogue. Use full-catalogue ranking (or very large samples) where possible.
- **Strong baselines:** popularity, item-to-item and well-tuned matrix factorisation. Several studies found that many published neural models fail to beat properly tuned simple baselines.
- **Segment analysis:** new versus established users, head versus tail items.

**Why offline and online disagree**

- **Logged data reflects the old policy:** offline tests only reward recommending items users actually interacted with, which were the items the old system showed. A new model that recommends good items the old system never showed gets no credit.
- **Proxy metrics:** offline click prediction does not capture satisfaction, diversity, or long-term effects.
- **Feedback effects:** online, recommendations change user behaviour; offline evaluation is static.
- **Serving differences:** latency, feature freshness, and business rules applied after ranking.

**Closing the gap**

- **Off-policy evaluation (OPE):** with logged propensities (the probability that the old policy showed each item), inverse propensity scoring and doubly robust estimators estimate how a new policy would perform. Requires stochastic logging policies with enough exploration.
- **Interleaving** for sensitive online comparison of rankers with little traffic.
- **Replay evaluation** on randomised exploration traffic.
- Track the historical correlation between offline metric gains and online A/B results, and use it to decide which offline metrics to trust.

```python
import numpy as np

def recall_at_k(recommended, relevant, k=10):
    hits = [len(set(r[:k]) & rel) / min(len(rel), k) for r, rel in zip(recommended, relevant) if rel]
    return float(np.mean(hits))

def ips_value(logged_rewards, logged_propensity, new_policy_prob):
    """Off-policy estimate of a new policy's average reward from logged data."""
    w = np.clip(new_policy_prob / logged_propensity, 0, 10)      # clip to limit variance
    return float(np.mean(w * logged_rewards))
```

### 32.22 How do you scale embedding tables for recommendation models with billions of IDs?

In large recommenders, most parameters are in **embedding tables** for categorical features: user IDs, item IDs, queries, creators. With hundreds of millions of users and items, these tables reach hundreds of gigabytes to terabytes, far larger than the dense layers, and they dominate memory and serving cost.

**Size estimate:** 1 billion IDs × 128 dimensions × 4 bytes (float32) ≈ 512 GB for one table, before optimiser states (Adam roughly triples it during training).

**Techniques to reduce size**

- **Hashing trick:** map IDs into a fixed number of buckets with a hash function. Bounded memory and no vocabulary management, but collisions merge unrelated IDs.
- **Multiple hashing and compositional embeddings:** combine embeddings from several smaller tables (for example the quotient-remainder trick: one table indexed by $\text{id} \div m$, another by $\text{id} \bmod m$, combined by multiplication or concatenation). Fewer collisions for the same memory.
- **Mixed-dimension embeddings:** frequent IDs get larger vectors; rare IDs get small ones, projected to a common size.
- **Frequency filtering and eviction:** only create embeddings for IDs seen at least N times; evict IDs not seen recently (important in streaming training where new IDs appear constantly).
- **Quantisation:** store embeddings in int8 or lower precision for serving, and sometimes training.
- **Feature design:** replace raw IDs with content or cluster features where IDs add little.

**Distributing tables**

- **Sharding strategies:** table-wise (whole tables on different devices), row-wise (split a large table's rows across devices), column-wise (split embedding dimensions), or combinations. Libraries such as TorchRec plan sharding automatically based on table sizes and access patterns.
- **Hybrid parallelism:** model parallelism for embedding tables, data parallelism for dense layers, with all-to-all communication to exchange looked-up embeddings.
- **Caching:** frequent IDs follow a power law, so keeping hot rows in GPU memory and the rest in CPU memory or SSD works well.

**Serving considerations:** keep embedding lookup latency low with in-memory stores or co-located parameter servers; version tables with the model; handle unseen IDs with a default or hashed embedding.

```python
import torch
import torch.nn as nn

class QREmbedding(nn.Module):
    """Quotient-remainder compositional embedding: two small tables instead of one huge one."""
    def __init__(self, num_ids, dim, m=100_000):
        super().__init__()
        self.m = m
        self.q = nn.Embedding((num_ids + m - 1) // m, dim)
        self.r = nn.Embedding(m, dim)
    def forward(self, ids):
        return self.q(ids // self.m) * self.r(ids % self.m)
```

### 32.23 How do large language models change recommender systems?

LLMs are changing recommendation in several ways, from feature generation to conversational interfaces. A senior answer separates what works in production today from what is still mostly research.

**1. LLMs as feature and content generators (widely used)**

- Generate or clean item metadata: categories, attributes, tags, summaries from descriptions and reviews.
- Produce **text embeddings** of items and user histories that improve content-based retrieval and cold start.
- Infer user interests from free text (search queries, reviews, support chats) as features.
- Cheap enough when done offline in batch.

**2. LLMs as rankers or re-rankers**

- Give the model the user's history and a short list of candidates in the prompt, and ask it to rank or select.
- Strong zero-shot ability and world knowledge, useful for small catalogues, cold start, and explaining recommendations.
- Limitations: cost and latency at scale, position and popularity biases in the prompt, limited context for long histories, and difficulty using collaborative signals that a trained ranker captures well. Usually used for the final step on a handful of items, or distilled into a smaller model.

**3. Conversational recommendation**

- Users describe what they want in natural language ("a light rain jacket for hiking under 100 dollars"), and an LLM-based agent asks clarifying questions, calls search and recommendation tools, and explains choices.
- The LLM orchestrates; retrieval and ranking systems still provide the candidates and personalisation.

**4. Generative recommendation**

- Sequence models over semantic IDs generate the next item directly (see the sequential and generative recommenders question), borrowing architecture and scaling ideas from language models.

**5. Explanations**

- Natural-language reasons for a recommendation ("because you liked X and this has a similar style"), grounded in actual features to avoid invented reasons.

**Risks and practical concerns**

- Hallucinated items (recommending products that do not exist): constrain outputs to valid catalogue IDs.
- Cost and latency: use LLMs offline or for small candidate sets; cache results.
- Evaluation: measure with the same offline and online protocols as any recommender, not with impressions from demos.
- Privacy: user histories in prompts must follow data policies, especially with external APIs.

### 32.24 How do you balance short-term engagement with long-term user value in recommendations?

Most recommenders optimise **immediate** signals: clicks, watch time in the session, purchases today. These are easy to measure but can diverge from what the business and users actually value over months: satisfaction, trust, retention and healthy habits.

**How short-term optimisation goes wrong**

- **Clickbait and sensational content** win clicks but reduce trust and satisfaction.
- **Over-exploitation** of known interests narrows content and causes boredom.
- **Aggressive notifications and autoplay** increase sessions while increasing churn and complaints later.
- **Marketplace effects:** favouring a few sellers maximises immediate conversions but drives others away, reducing future selection.

**Approaches**

1. **Better objectives**
   - Include negative and quality signals in the score: hides, dislikes, "not interested", returns, short dwell after clicking, survey responses.
   - Weight objectives that correlate with long-term outcomes (for example satisfied sessions rather than raw time spent).

2. **Surrogate metrics for long-term outcomes**
   - Long-term outcomes (90-day retention) take too long to use in each experiment. Build **surrogate indices**: combinations of short-term metrics that predict the long-term outcome, validated on past experiments where both were observed.

3. **Long-term holdouts and experiments**
   - Keep a small group on an older or different policy for months to measure cumulative effects that short tests miss (novelty effects fade; learning effects build up).
   - Run some experiments long enough to observe retention, not just engagement.

4. **Reinforcement learning and value modelling**
   - Model the effect of today's recommendation on future engagement (user state transitions), using off-policy RL methods carefully, since online exploration is costly (see 16.47).

5. **Guardrail metrics**
   - Every launch decision checks user satisfaction, complaint rates, unsubscribe and notification-disable rates, content quality and diversity, alongside the primary metric.

**Leadership angle:** the choice of objective is a product and ethical decision, not only a technical one. Make objective weights explicit, review them with product and policy leaders, and report long-term metrics to executives alongside short-term engagement.

### 32.25 How do ARIMA models work, and what do stationarity and differencing mean?

**ARIMA** (autoregressive integrated moving average) models a time series using its own past values and past forecast errors. It remains a standard baseline and a common interview topic, alongside the overall forecasting approach in 28.15.

**Stationarity**

- A series is (weakly) **stationary** if its mean, variance and autocorrelation do not change over time.
- Most ARIMA theory assumes stationarity. Trends and changing variance break it.
- **Tests:** the augmented Dickey-Fuller (ADF) test has the null hypothesis of a unit root (non-stationary); the KPSS test has the null of stationarity. Using both gives a more reliable picture.
- **Fixes:** **differencing** ($y_t - y_{t-1}$) removes trends; **seasonal differencing** ($y_t - y_{t-s}$) removes seasonal patterns; a **log or Box-Cox transform** stabilises growing variance.

**The components of ARIMA(p, d, q)**

- **AR(p):** the value depends linearly on the previous $p$ values: $y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t$.
- **I(d):** the series is differenced $d$ times to become stationary.
- **MA(q):** the value depends on the previous $q$ forecast errors: $+ \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}$.
- **SARIMA(p, d, q)(P, D, Q)$`_s`$** adds the same structure at the seasonal lag $s$ (for example 7 for daily data with weekly seasonality, 12 for monthly data with yearly seasonality).
- **ARIMAX / regression with ARIMA errors** adds external regressors (price, promotions, holidays).

**Choosing the orders**

- Classical approach: inspect the autocorrelation function (ACF) and partial autocorrelation function (PACF) of the differenced series. A PACF cut-off after lag $p$ suggests AR(p); an ACF cut-off after lag $q$ suggests MA(q).
- Practical approach: automatic search over orders using an information criterion (AIC or AICc), as in `auto_arima`-style tools, then check residuals.

**Diagnostics:** residuals should look like white noise (no remaining autocorrelation; Ljung-Box test), with roughly constant variance.

**Strengths and limits:** interpretable, fast, strong on short, regular series; gives prediction intervals. Weak with many related series, complex seasonality (multiple periods), many external drivers or non-linear effects.

```python
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

print("ADF p-value:", adfuller(y)[1], "KPSS p-value:", kpss(y, nlags="auto")[1])
model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
forecast = model.get_forecast(steps=28)
mean, interval = forecast.predicted_mean, forecast.conf_int(alpha=0.2)   # 80% prediction interval
```

### 32.26 How does exponential smoothing work, and what is the ETS framework?

**Exponential smoothing** forecasts with weighted averages of past observations, where weights decrease exponentially with age. It is simple, robust and often hard to beat for individual series.

**The family**

- **Simple exponential smoothing** (level only): $\ell_t = \alpha y_t + (1 - \alpha) \ell_{t-1}$, forecast $\hat{y}_{t+h} = \ell_t$. The smoothing parameter $\alpha$ sets how fast the level adapts.
- **Holt's linear trend:** adds a trend component $b_t$, updated with its own smoothing parameter $\beta$. A **damped trend** (parameter $\phi \lt 1$) flattens the trend over long horizons and is usually more accurate than an undamped trend, which extrapolates too aggressively.
- **Holt-Winters:** adds a seasonal component with parameter $\gamma$, either **additive** (seasonal swings of constant size) or **multiplicative** (swings proportional to the level).

**The ETS framework (Error, Trend, Seasonal)**

- Each component can be None (N), Additive (A), Additive damped (Ad, for trend) or Multiplicative (M). For example ETS(A,Ad,M) has additive errors, a damped trend and multiplicative seasonality.
- Formulated as **state space models**, which gives maximum likelihood estimation, model selection by AIC across all combinations, and proper **prediction intervals**.
- Automatic ETS selection (as in the `forecast` package in R and `statsforecast` in Python) is a standard, strong baseline.

**ETS compared with ARIMA**

- ETS describes the series by its level, trend and seasonality; ARIMA by autocorrelation. Some models overlap (simple exponential smoothing equals ARIMA(0,1,1)).
- ETS handles changing seasonality and multiplicative patterns naturally; ARIMA handles complex autocorrelation and external regressors more naturally.
- In forecasting competitions, combinations of ETS, ARIMA and simple methods have often outperformed any single one.

**Limits:** one series at a time, a single seasonal period in the basic form (TBATS and MSTL handle several), and no external regressors in standard ETS.

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, SeasonalNaive

# df columns: unique_id, ds (date), y
sf = StatsForecast(models=[AutoETS(season_length=7), AutoARIMA(season_length=7), SeasonalNaive(7)], freq="D")
forecasts = sf.forecast(df=df, h=28, level=[80, 95])
```

### 32.27 Should you use local statistical models or global machine learning models for forecasting many series?

When a business forecasts thousands or millions of series (store-product pairs, sensors, accounts), the key architecture choice is between **local** models (one model per series) and **global** models (one model trained across all series).

| Aspect | Local (ETS, ARIMA per series) | Global (one GBDT or neural network for all series) |
| --- | --- | --- |
| Data per model | Only that series' history | All series together |
| Short or sparse histories | Weak; little to learn from | Strong; borrows patterns from similar series |
| Cross-series features | None | Category, store, price, promotion and calendar features shared across series |
| Cold start | Cannot forecast without history | Can forecast using attributes |
| Interpretability | High per series | Lower; needs tools such as SHAP |
| Operations | Millions of small models to fit and monitor | One or a few models to train, deploy and monitor |
| Strong when | Few, long, regular series | Many related series with external drivers |

**Evidence from competitions:** in the M5 competition (Walmart sales, 2020), most top solutions used **global LightGBM models** with lag, rolling, calendar and price features, trained with Tweedie or similar losses, often ensembled across several training windows. Earlier, in M4, the winning entry was a hybrid of exponential smoothing and a global recurrent network.

**Building a global ML forecaster**

- Features: lags at or beyond the forecast horizon, rolling statistics, calendar, holidays, prices and promotions, static attributes and identifiers (see 10.31 for feature construction).
- **Multi-step strategies:** *recursive* (predict one step, feed it back as a lag; errors compound), *direct* (a separate model or target for each horizon), or *multi-output*.
- Normalise series to comparable scales (divide by each series' recent mean) so the model learns shapes rather than magnitudes.
- Loss matched to the data: Tweedie or Poisson for counts with many zeros; quantile loss for intervals.

**Practical recommendation:** start with strong local baselines (seasonal naive, ETS), build a global gradient-boosted model, and compare with backtesting. Ensembles of the two are often best.

```python
import lightgbm as lgb

features = ["lag_28", "lag_35", "roll_mean_28", "roll_std_28", "dow", "month", "price", "promo", "store_id", "dept_id"]
model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=2000, learning_rate=0.03)
model.fit(train[features], train["sales"], eval_set=[(valid[features], valid["sales"])],
          callbacks=[lgb.early_stopping(100)], categorical_feature=["store_id", "dept_id"])
```

### 32.28 What deep learning models and foundation models are used for time-series forecasting?

Deep learning has produced several strong forecasting architectures, and, more recently, pretrained **time-series foundation models** that forecast new series zero-shot.

**Notable architectures**

| Model | Idea | Strengths |
| --- | --- | --- |
| DeepAR | Autoregressive recurrent network trained across many series, outputting a probability distribution at each step | Probabilistic forecasts; global model for many related series |
| N-BEATS / N-HiTS | Stacks of fully connected blocks with backward and forward residual links; N-HiTS adds multi-rate sampling for long horizons | Strong accuracy, simple, optionally interpretable (trend and seasonality blocks) |
| Temporal Fusion Transformer (TFT) | Attention plus gating, with separate handling of static, known-future and observed inputs | Many covariates, interpretable attention and variable importance |
| PatchTST | Transformer over patches (short windows) of each series, channel-independent | Strong on long-horizon benchmarks |
| Simple linear models (DLinear and others) | Linear layers on decomposed series | Showed many complex transformers were not better than well-designed linear baselines on popular benchmarks |

**Time-series foundation models**

- Pretrained on very large collections of real and synthetic series, then used **zero-shot** on new series or fine-tuned.
- Examples: Google's TimesFM (decoder-only transformer over patches), Amazon's Chronos (scales and quantises values into tokens and uses a language-model architecture, producing probabilistic forecasts), Salesforce's Moirai, and others.
- **Useful for:** new products or domains with little history, rapid prototyping, many heterogeneous series without time to build a custom model, and as an extra ensemble member.
- **Limitations:** results versus strong tuned baselines are mixed and domain-dependent; limited support for rich covariates in some models; inference cost; and less transparency.

**When deep learning is worth it**

- Many related series, long histories, complex patterns and rich covariates.
- Probabilistic forecasts needed at scale.
- Team capacity to tune and maintain neural models.

**When it is not:** a few series, short histories, or when strong statistical and gradient-boosted baselines already meet the accuracy need. Always benchmark against seasonal naive, ETS and a global GBDT under the same backtesting protocol.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

nf = NeuralForecast(models=[NHITS(h=28, input_size=112, max_steps=1000)], freq="D")
nf.fit(df=train_df)                 # columns: unique_id, ds, y
forecasts = nf.predict()
```

### 32.29 How do you produce and evaluate probabilistic forecasts and prediction intervals?

Decisions need to know **how uncertain** a forecast is. Inventory, staffing and capacity planning all choose a quantity based on a service level, which requires a distribution, not just a single number.

**Ways to produce uncertainty**

- **Model-based intervals:** ARIMA and ETS state space models give analytical prediction intervals (which assume their error distributions are correct).
- **Quantile regression:** train models with the **pinball (quantile) loss** for each quantile needed (for example P10, P50, P90). Gradient-boosted trees and neural networks support this directly.

  $L_q(y, \hat{y}) = \max\left(q (y - \hat{y}), (q - 1)(y - \hat{y})\right)$

- **Distributional models:** predict the parameters of a distribution (DeepAR predicts negative binomial or Gaussian parameters).
- **Simulation and bootstrapping:** resample residuals to generate future paths.
- **Conformal prediction:** wrap any point forecaster and calibrate intervals using errors on recent held-out data. Standard conformal methods assume exchangeable data, which time series violate, so time-series variants (adaptive conformal inference, rolling calibration windows) adjust the interval width as errors change.

**Evaluation**

- **Coverage (calibration):** an 80% interval should contain the actual value about 80% of the time. Check by horizon and segment.
- **Sharpness:** among calibrated forecasts, narrower intervals are better.
- **Pinball loss** averaged over quantiles, or the **continuous ranked probability score (CRPS)**, which evaluates the whole predictive distribution and reduces to absolute error for a point forecast.
- **Weighted scaled pinball loss** was used in the M5 uncertainty competition to compare across series of different scales.

**From forecast to decision**

- The right quantile depends on costs: for a newsvendor-style inventory decision, the optimal order is the quantile $q = \frac{c_u}{c_u + c_o}$, where $c_u$ is the cost of understocking and $c_o$ the cost of overstocking.

```python
import numpy as np
import lightgbm as lgb

quantiles = [0.1, 0.5, 0.9]
models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=500).fit(X_train, y_train) for q in quantiles}
preds = {q: m.predict(X_test) for q, m in models.items()}

coverage_80 = np.mean((y_test >= preds[0.1]) & (y_test <= preds[0.9]))     # should be close to 0.8
def pinball(y, yhat, q):
    d = y - yhat
    return np.mean(np.maximum(q * d, (q - 1) * d))
print(coverage_80, np.mean([pinball(y_test, preds[q], q) for q in quantiles]))
```

### 32.30 Which forecast accuracy metrics should you use, and how do you design a backtest?

Choosing metrics and a validation design is where many forecasting projects go wrong. Metrics should reflect the business decision, and validation must mimic how forecasts are produced in production.

**Point forecast metrics**

| Metric | Formula idea | Strengths | Weaknesses |
| --- | --- | --- | --- |
| MAE | Mean absolute error | Easy to explain, in original units | Not comparable across series of different scale |
| RMSE | Root mean squared error | Penalises large errors | Sensitive to outliers; scale-dependent |
| MAPE | Mean absolute percentage error | Scale-free, intuitive | Undefined or explosive near zero; penalises over-forecasts more than under-forecasts |
| sMAPE | Symmetric percentage error | Bounded | Still unstable near zero; not truly symmetric |
| WAPE | $\sum \lvert e \rvert / \sum \lvert y \rvert$ | Robust with many small values; common in retail | Dominated by high-volume series |
| MASE | MAE divided by the in-sample MAE of a seasonal naive forecast | Scale-free, works with zeros, below 1 means better than naive | Less intuitive to stakeholders |
| Bias | $\sum e / \sum y$ | Shows systematic over- or under-forecasting | Errors can cancel out |

**Choosing metrics**

- Measure at the **level of the decision** (for example store-product over the replenishment lead time, not daily totals).
- Weight by business value (revenue, margin) if errors on some series matter more.
- Always report **bias** alongside an accuracy metric; a forecast can have good average error but consistently under-forecast.
- Compare against a **naive benchmark**; the improvement over naive is often more meaningful than the raw error.

**Backtesting design (rolling-origin evaluation)**

- Simulate the production process: train on data up to a cut-off, forecast the next horizon, move the cut-off forward, repeat (see 12.23 for why random cross-validation fails on time series).
- Use several origins covering different seasons and events.
- Respect **data availability**: features and actuals must be as they were known at each cut-off, including reporting delays.
- Use the same horizon and frequency as production (for example weekly forecasts for 1 to 8 weeks ahead).
- Report errors by horizon: accuracy usually degrades with lead time.

**Forecast value added (FVA):** compare each step of the forecasting process (naive, statistical model, ML model, planner override) to see whether each one actually improves accuracy.

```python
import numpy as np

def mase(y_true, y_pred, y_train, m=7):
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))          # in-sample seasonal naive MAE
    return np.mean(np.abs(y_true - y_pred)) / scale

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def bias(y_true, y_pred):
    return np.sum(y_pred - y_true) / np.sum(y_true)               # positive means over-forecasting
```

### 32.31 What is hierarchical forecasting, and how does forecast reconciliation work?

Business forecasts usually form a **hierarchy**: products roll up to categories and departments; stores roll up to regions and the country; days roll up to weeks and months. Different teams use different levels (store managers use store-product forecasts; finance uses national totals), and the forecasts must **add up consistently**.

**The problem:** forecasting each level independently produces numbers that do not sum: the store forecasts may add up to more than the national forecast. Plans built on inconsistent numbers conflict.

**Traditional approaches**

- **Bottom-up:** forecast the lowest level and sum upwards. No information lost at the bottom, but bottom-level series are noisy, so aggregates can be poor.
- **Top-down:** forecast the total and split it down using historical or forecast proportions. Stable totals, but loses detail at lower levels and handles changing mixes poorly.
- **Middle-out:** forecast a middle level, sum upwards and split downwards.

**Optimal reconciliation**

- Forecast **every** level independently (base forecasts), then adjust them so they are coherent, using information from all levels.
- With summing matrix $S$ (how bottom series add up to every level) and base forecasts $\hat{y}$, reconciled forecasts are $\tilde{y} = S G \hat{y}$ for a matrix $G$.
- **MinT (minimum trace)** (Wickramasuriya, Athanasopoulos and Hyndman, 2019) chooses $G$ using the covariance of base forecast errors, minimising the total variance of reconciled errors. Simpler variants use OLS or weight each series by its error variance (WLS).
- Reconciliation often improves accuracy at all levels, not just consistency.

**Temporal hierarchies:** the same idea applies across time scales: reconcile daily, weekly and monthly forecasts of the same series.

**Practical considerations**

- Very large hierarchies (millions of bottom series) need sparse or approximate reconciliation methods.
- Non-negativity: reconciliation can produce negative values for low-volume series; use constrained methods or clip and re-reconcile.
- Global ML models trained on bottom-level series with aggregate-level features, followed by reconciliation, are a common modern setup.

```python
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace

# Y_hat_df: base forecasts for every level; Y_fitted_df: in-sample fits; S_df, tags: hierarchy structure
hrec = HierarchicalReconciliation(reconcilers=[BottomUp(), MinTrace(method="mint_shrink")])
reconciled = hrec.reconcile(Y_hat_df=Y_hat_df, Y_df=Y_fitted_df, S=S_df, tags=tags)
```

### 32.32 How do you forecast intermittent demand?

**Intermittent demand** has many periods with zero demand and occasional non-zero values: spare parts, slow-moving retail items, B2B orders. Standard methods fail: they forecast small positive values every period that match nothing, and percentage error metrics break on zeros.

**Classifying demand patterns** (Syntetos, Boylan and Croston)

- **ADI** (average demand interval): average number of periods between non-zero demands.
- **CV²**: squared coefficient of variation of the non-zero demand sizes.
- Using cut-offs of ADI = 1.32 and CV² = 0.49: **smooth** (frequent, stable), **erratic** (frequent, variable sizes), **intermittent** (infrequent, stable sizes), **lumpy** (infrequent and variable). Method choice depends on the class.

**Methods**

- **Croston's method:** separately smooth the non-zero demand sizes and the intervals between them; the forecast is size divided by interval (a demand rate per period).
- **SBA (Syntetos-Boylan approximation):** multiplies Croston's forecast by $(1 - \alpha/2)$ to correct its known upward bias.
- **TSB (Teunter-Syntetos-Babai):** smooths the **probability** of demand in each period instead of the interval, so the forecast decays when demand stops; better for items becoming obsolete.
- **Temporal aggregation (ADIDA):** forecast at a coarser level (weekly or monthly instead of daily), where demand is less intermittent, then divide back down.
- **Global ML models** with Tweedie or Poisson losses and features from similar items, which share information across many sparse series (as in M5).
- **Probabilistic approaches:** for inventory decisions, model the full distribution (for example the probability of zero demand plus a size distribution), because the decision depends on the chance of any demand at all.

**Evaluation**

- Avoid MAPE. Use MASE, scaled pinball loss, or, best, **inventory simulation** (service level achieved, stock held, stockouts) under the planned replenishment policy.
- Note that a forecast of zero can have a low absolute error and still be useless; judge methods by the decisions they lead to.

```python
import numpy as np

def croston_sba(y, alpha=0.1):
    size, interval, q = None, None, 1
    for v in y:
        if v > 0:
            size = v if size is None else alpha * v + (1 - alpha) * size
            interval = q if interval is None else alpha * q + (1 - alpha) * interval
            q = 1
        else:
            q += 1
    if size is None:
        return 0.0
    return (1 - alpha / 2) * size / interval       # SBA bias correction of Croston's rate
```

### 32.33 How do you use external drivers such as promotions, prices, holidays and weather in forecasts?

External variables (exogenous regressors, covariates) often explain most of the variation in demand: promotions can multiply sales several times, and holidays shift demand between days. Using them well requires care about **what is known in the future**.

**Types of drivers**

| Type | Examples | Known at forecast time? |
| --- | --- | --- |
| Known future (calendar) | Day of week, holidays, school terms, paydays | Yes, with certainty |
| Planned (controlled by the business) | Prices, promotions, marketing campaigns, store openings | Yes, if plans are shared in time and kept up to date |
| Uncertain external | Weather, competitor actions, macroeconomic indicators | No; they must themselves be forecast |
| Static attributes | Product category, store size, region | Yes |

**Key issues**

- **Unknown future values:** a model that uses actual weather in training must use **forecast** weather in production. Train with forecasts as they were available at the time (or simulate their error), otherwise backtests overstate accuracy.
- **Plan reliability:** promotion calendars change late; build processes to get the latest plans, and measure accuracy with plans as they were known at forecast time.
- **Promotion effects:** model uplift during the promotion, **pull-forward** (dips before and after as customers stock up or wait), **cannibalisation** of similar products, and **halo** effects on complementary products.
- **Holiday effects:** include windows before and after each holiday, and handle moving holidays (Easter, Lunar New Year, Eid) by date, not fixed calendar positions.
- **Price elasticity:** price changes are partly confounded with demand (prices are raised when demand is high); for pricing decisions, causal estimation is needed (see the causal inference questions in this chapter and 28.28).

**Scenario forecasting**

- Because planned drivers are under the business's control, forecasts can answer "what if": forecast demand under different promotion or price plans to support decisions, not just predict a single future.

**Which models handle drivers well:** regression with ARIMA errors (ARIMAX), Prophet-style additive models with holiday and regressor terms, global gradient-boosted models with driver features, and neural models with explicit known-future inputs (TFT).

```python
import pandas as pd

# known-future features must be available for every future date in the horizon
future = pd.DataFrame({"ds": pd.date_range("2027-01-01", periods=28, freq="D")})
future["dow"] = future["ds"].dt.dayofweek
future["is_holiday"] = future["ds"].isin(holiday_dates).astype(int)
future = future.merge(promo_plan, on="ds", how="left").fillna({"promo_depth": 0.0})   # latest promotion plan
future = future.merge(weather_forecast, on="ds", how="left")                          # forecast weather, not actuals
```

### 32.34 How do you handle structural breaks, anomalies and regime changes in time series?

Forecasting models assume the future resembles the past. **Structural breaks** (a pandemic, a new competitor, a pricing change, a website redesign, a change in how data is recorded) violate that assumption, and **anomalies** (one-off spikes, outages, data errors) distort models trained on them.

**Detection**

- **Changepoint detection** algorithms find points where the mean, variance or trend changes: PELT and binary segmentation (offline), Bayesian online changepoint detection (streaming).
- **Residual monitoring:** a sustained run of large forecast errors in one direction signals a break.
- **Domain events:** keep a log of known events (launches, outages, tracking changes); many breaks are known in advance.
- **Anomaly detection** for single points: robust z-scores on residuals after removing seasonality (for example STL decomposition with median-based thresholds).

**Handling anomalies in training data**

- Fix data errors at the source when possible.
- Replace or down-weight anomalous points (interpolation, or seasonal values), or add indicator variables for known one-off events (outage, stockout).
- **Stockouts** deserve special care: sales of zero during a stockout are censored demand, not low demand.

**Handling structural breaks**

- **Intervention variables:** step or ramp indicators for a known change, so the model learns the shift instead of averaging across it.
- **Shorter training windows** or higher weights on recent data after a break.
- **Models that adapt:** exponential smoothing with higher smoothing parameters, frequent retraining, online learning.
- **Exclude or separately model abnormal periods** (for example treat a pandemic period with indicators, or exclude it when estimating seasonality).
- **Judgemental adjustment** for unprecedented changes, documented and measured for value added.

**Communicating in a break:** widen prediction intervals, increase forecast refresh frequency, provide scenarios (fast recovery, slow recovery), and tell stakeholders explicitly that historical accuracy no longer applies until the new pattern stabilises.

```python
import numpy as np
import ruptures as rpt

algo = rpt.Pelt(model="rbf").fit(np.asarray(y).reshape(-1, 1))
breakpoints = algo.predict(pen=10)           # indexes where the series' behaviour changes

from statsmodels.tsa.seasonal import STL
resid = STL(y, period=7, robust=True).fit().resid
mad = np.median(np.abs(resid - np.median(resid)))
anomalies = np.abs(resid - np.median(resid)) > 3.5 * 1.4826 * mad   # robust z-score above 3.5
```

### 32.35 How do you forecast demand for new products with no sales history?

New product forecasting is a cold-start problem: launches, seasonal fashion items, new stores or new markets. There is no history, yet decisions about initial stock, production and marketing must be made before launch.

**Approaches**

1. **Analogous products ("looks like")**
   - Find similar past products by attributes (category, price band, brand, colour, size range, target customer) or by text and image embeddings.
   - Use their launch curves, scaled by expected differences in price, distribution and marketing.
   - Simple, explainable, and often the planners' current method, so it is a good baseline.

2. **Attribute-based global models**
   - Train a model across past launches to predict sales from product attributes, price, launch timing, number of stores, marketing spend and early signals.
   - Global ML models naturally handle new items through shared features.

3. **Launch curve models**
   - Model the shape of the sales curve over the product life cycle: ramp-up, peak, decline. Bass diffusion models describe adoption through innovation and imitation effects; empirical curve clustering groups past launches into typical shapes.
   - Forecast the total volume and the shape separately.

4. **Pre-launch signals**
   - Pre-orders, waitlists, product page views, search interest, social media attention, test-market results and retailer orders.

5. **Fast updating after launch**
   - The first days or weeks of sales are highly informative. Update forecasts frequently in early life (Bayesian updating of the prior from analogues with observed sales).

6. **Time-series foundation models** and similar-series transfer learning can provide reasonable zero-shot shapes.

**Managing uncertainty**

- Uncertainty is highest at launch: produce wide intervals and scenarios.
- Inventory strategy matters as much as the forecast: smaller initial allocations with fast replenishment, or flexible production, reduce the cost of forecast error.

**Evaluation:** backtest by simulating past launches, forecasting each with only information available before its launch, and measure accuracy over the first weeks and the full life cycle.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# product_emb: embeddings of past products (attributes, text, images); curves: their first 12 weeks of sales
nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(product_emb)
dist, idx = nn.kneighbors(new_product_emb.reshape(1, -1))
weights = 1 - dist[0]
launch_curve = np.average(curves[idx[0]], axis=0, weights=weights)          # analogue-based forecast
launch_curve *= expected_store_count / np.mean(store_counts[idx[0]])        # scale for distribution differences
```

### 32.36 How do you run forecasting in production, and how do you make forecasts useful for business planning?

A forecast is only valuable if it improves decisions. Operating a forecasting system well is as much about process, trust and communication as about models, which makes it a frequent topic for data leaders.

**Production pipeline**

```text
data ingestion (sales, prices, promotions, calendars, stock)
  ─► data validation (missing stores, late feeds, outliers, stockouts flagged)
  ─► feature building (point-in-time correct)
  ─► model training on a schedule (for example weekly) + forecasting on a schedule (for example daily)
  ─► reconciliation across hierarchies ─► business rules (minimums, known closures)
  ─► publish to planning systems with version, model and data timestamps
  ─► planner review and overrides (logged with reasons)
  ─► actuals arrive ─► accuracy tracking ─► alerts and retraining decisions
```

**Monitoring**

- Accuracy and bias by segment, horizon and level, compared with naive benchmarks.
- Data quality checks before forecasts are published; a missing sales feed should block publication rather than produce wrong forecasts.
- Drift in input features and in the error distribution; changepoint alerts.
- Fallback models (seasonal naive or the previous forecast) when the main pipeline fails.

**Working with planners**

- **Overrides are information:** planners know about events the model does not. Log every override with a reason, and measure its forecast value added. Often overrides improve forecasts for promotions and events but worsen them otherwise; feedback helps both sides improve.
- **Consensus forecasting** (as in sales and operations planning): one agreed number, combining the statistical forecast with commercial input, used by every function.
- **Explainability:** show the drivers of a forecast (seasonality, promotion uplift, trend) so planners can judge it.

**Communicating uncertainty**

- Give ranges and scenarios, not single numbers: "We expect 12,000 units; there is an 80% chance of between 10,500 and 13,800."
- Tie uncertainty to decisions: the stock level that gives a 95% service level, the staffing level for a busy-day scenario.
- Explain accuracy in business terms (lost sales, excess inventory) rather than only in percentages.

**Measuring success:** business outcomes the forecasts support (stockouts, inventory turns, waste, service levels, staffing costs), forecast adoption by planners, and time spent on manual adjustments, as well as accuracy metrics.

```python
import pandas as pd

def forecast_value_added(df):
    """df columns: actual, naive, statistical, final (after overrides), segment."""
    out = {}
    for name in ["naive", "statistical", "final"]:
        out[name] = (df[name] - df["actual"]).abs().sum() / df["actual"].abs().sum()     # WAPE per step
    return pd.Series(out)

print(df.groupby("segment").apply(forecast_value_added))
```

---

← [Chapter 31. Behavioural, Leadership and Scenario Questions](../part-6-professional-practice/31-behavioral-leadership-and-scenarios.md) · [Table of contents](../README.md)
