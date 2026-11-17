# Chapter 8. Machine Learning Fundamentals

_Part III — Machine Learning_

The learning problem and generalisation theory, bias and variance, overfitting, regularisation, losses, data leakage, problem framing and the ML project lifecycle. Expect these in every ML loop, often as the opening questions.

This chapter contains **62 questions**.

## Contents

- **8.1** [Why do we need machine learning, and where is it used?](#81-why-do-we-need-machine-learning-and-where-is-it-used)
- **8.2** [What is machine learning, formally, and how does it relate to AI and deep learning?](#82-what-is-machine-learning-formally-and-how-does-it-relate-to-ai-and-deep-learning)
- **8.3** [What is the difference between deduction and induction, and how does machine learning relate to induction?](#83-what-is-the-difference-between-deduction-and-induction-and-how-does-machine-learning-relate-to-induction)
- **8.4** [Why should machine learning work at all? Explain generalisation, PAC learning and VC dimension at an interview level.](#84-why-should-machine-learning-work-at-all-explain-generalisation-pac-learning-and-vc-dimension-at-an-interview-level)
- **8.5** [What are overfitting and underfitting, why do they occur, and how do you fix each?](#85-what-are-overfitting-and-underfitting-why-do-they-occur-and-how-do-you-fix-each)
- **8.6** [Explain the bias-variance trade-off mathematically and practically.](#86-explain-the-bias-variance-trade-off-mathematically-and-practically)
- **8.7** [What is the difference between parametric and non-parametric models?](#87-what-is-the-difference-between-parametric-and-non-parametric-models)
- **8.8** [What is the difference between online and offline learning, and where is online learning used?](#88-what-is-the-difference-between-online-and-offline-learning-and-where-is-online-learning-used)
- **8.9** [What are epochs, batches, batch size and iterations, and how do you choose the batch size?](#89-what-are-epochs-batches-batch-size-and-iterations-and-how-do-you-choose-the-batch-size)
- **8.10** [Compare batch, stochastic and mini-batch gradient descent. Which is used in practice?](#810-compare-batch-stochastic-and-mini-batch-gradient-descent-which-is-used-in-practice)
- **8.11** [What are embeddings in Machine Learning?](#811-what-are-embeddings-in-machine-learning)
- **8.12** [What are logits, softmax and cross-entropy, and how do they fit together?](#812-what-are-logits-softmax-and-cross-entropy-and-how-do-they-fit-together)
- **8.13** [What are the main types of machine learning, and how do supervised and unsupervised learning differ?](#813-what-are-the-main-types-of-machine-learning-and-how-do-supervised-and-unsupervised-learning-differ)
- **8.14** [What is Reinforcement Learning?](#814-what-is-reinforcement-learning)
- **8.15** [What is the difference between classification and regression, and what targets does each require?](#815-what-is-the-difference-between-classification-and-regression-and-what-targets-does-each-require)
- **8.16** [What are L1 and L2 loss functions, and when would you use each?](#816-what-are-l1-and-l2-loss-functions-and-when-would-you-use-each)
- **8.17** [What is regularisation? Explain L1 (lasso), L2 (ridge) and elastic net, and how the penalty changes the cost.](#817-what-is-regularisation-explain-l1-lasso-l2-ridge-and-elastic-net-and-how-the-penalty-changes-the-cost)
- **8.18** [What is the difference between a loss function, a cost function and an objective function?](#818-what-is-the-difference-between-a-loss-function-a-cost-function-and-an-objective-function)
- **8.19** [What is a perceptron, and how does a multilayer perceptron overcome its limits?](#819-what-is-a-perceptron-and-how-does-a-multilayer-perceptron-overcome-its-limits)
- **8.20** [Explain Cross-Validation. Why is it used?](#820-explain-cross-validation-why-is-it-used)
- **8.21** [What is Confusion Matrix?](#821-what-is-confusion-matrix)
- **8.22** [What are precision, recall, and F1-score?](#822-what-are-precision-recall-and-f1-score)
- **8.23** [What is anomaly detection?](#823-what-is-anomaly-detection)
- **8.24** [What is the difference between policy-based and value-based methods?](#824-what-is-the-difference-between-policy-based-and-value-based-methods)
- **8.25** [What is Q-Learning?](#825-what-is-q-learning)
- **8.26** [Explain the concept of exploration vs. exploitation.](#826-explain-the-concept-of-exploration-vs-exploitation)
- **8.27** [What is the curse of dimensionality, and how do you address it?](#827-what-is-the-curse-of-dimensionality-and-how-do-you-address-it)
- **8.28** [How do you combine losses in multi-task learning? Explain per-task losses, focal loss and gradient blending.](#828-how-do-you-combine-losses-in-multi-task-learning-explain-per-task-losses-focal-loss-and-gradient-blending)
- **8.29** [Explain Contrastive Learning](#829-explain-contrastive-learning)
- **8.30** [What is Generative AI?](#830-what-is-generative-ai)
- **8.31** [What are the training, validation and test sets, and how should you split data?](#831-what-are-the-training-validation-and-test-sets-and-how-should-you-split-data)
- **8.32** [Differentiate between Deep Learning and Machine Learning?](#832-differentiate-between-deep-learning-and-machine-learning)
- **8.33** [What is semi-supervised learning, and where is it applied?](#833-what-is-semi-supervised-learning-and-where-is-it-applied)
- **8.34** [What is Inductive Logic Programming?](#834-what-is-inductive-logic-programming)
- **8.35** [How would you differentiate data mining from Machine learning?](#835-how-would-you-differentiate-data-mining-from-machine-learning)
- **8.36** [What do you mean by the unreasonable effectiveness of data?](#836-what-do-you-mean-by-the-unreasonable-effectiveness-of-data)
- **8.37** [What is the No Free Lunch theorem, and what does inductive bias mean in practice?](#837-what-is-the-no-free-lunch-theorem-and-what-does-inductive-bias-mean-in-practice)
- **8.38** [What are the main differences between Structured and Unstructured Data?](#838-what-are-the-main-differences-between-structured-and-unstructured-data)
- **8.39** [What is data leakage, which forms are hard to detect, and how do you prevent it?](#839-what-is-data-leakage-which-forms-are-hard-to-detect-and-how-do-you-prevent-it)
- **8.40** [What are the different types of bias in Machine Learning?](#840-what-are-the-different-types-of-bias-in-machine-learning)
- **8.41** [Why does L1 regularisation give sparse coefficients while L2 does not? Explain geometrically.](#841-why-does-l1-regularisation-give-sparse-coefficients-while-l2-does-not-explain-geometrically)
- **8.42** [List some ways using which you can improve a model’s performance.](#842-list-some-ways-using-which-you-can-improve-a-models-performance)
- **8.43** [What is active learning, and where is it useful?](#843-what-is-active-learning-and-where-is-it-useful)
- **8.44** [What types of models tend to overfit?](#844-what-types-of-models-tend-to-overfit)
- **8.45** [Can you tell the complete life cycle of a data science project?](#845-can-you-tell-the-complete-life-cycle-of-a-data-science-project)
- **8.46** [What are the properties of a good ML model?](#846-what-are-the-properties-of-a-good-ml-model)
- **8.47** [Given a dataset, how do you decide which machine learning algorithm to use?](#847-given-a-dataset-how-do-you-decide-which-machine-learning-algorithm-to-use)
- **8.48** [Is more data always better?](#848-is-more-data-always-better)
- **8.49** [How would you differentiate between multi-label and multi-class classification?](#849-how-would-you-differentiate-between-multi-label-and-multi-class-classification)
- **8.50** [What is an AI ontology?](#850-what-is-an-ai-ontology)
- **8.51** [Can AI systems be conscious?](#851-can-ai-systems-be-conscious)
- **8.52** [What is unsupervised learning through compression?](#852-what-is-unsupervised-learning-through-compression)
- **8.53** [Why can a model with lower training loss have worse test performance?](#853-why-can-a-model-with-lower-training-loss-have-worse-test-performance)
- **8.54** [How do you detect overfitting beyond simply comparing train and validation accuracy?](#854-how-do-you-detect-overfitting-beyond-simply-comparing-train-and-validation-accuracy)
- **8.55** [What is the difference between model capacity and model complexity?](#855-what-is-the-difference-between-model-capacity-and-model-complexity)
- **8.56** [Why does regularization reduce variance?](#856-why-does-regularization-reduce-variance)
- **8.57** [What is the difference between generative and discriminative models?](#857-what-is-the-difference-between-generative-and-discriminative-models)
- **8.58** [How do you frame a business problem as a machine learning problem, and when should you not use machine learning?](#858-how-do-you-frame-a-business-problem-as-a-machine-learning-problem-and-when-should-you-not-use-machine-learning)
- **8.59** [What is the difference between data-centric and model-centric improvement, and when should you focus on each?](#859-what-is-the-difference-between-data-centric-and-model-centric-improvement-and-when-should-you-focus-on-each)
- **8.60** [How do you use learning curves to diagnose whether more data, more capacity or better features will help?](#860-how-do-you-use-learning-curves-to-diagnose-whether-more-data-more-capacity-or-better-features-will-help)
- **8.61** [What is Bayes error, how do you estimate the best achievable performance, and how do you know when to stop improving a model?](#861-what-is-bayes-error-how-do-you-estimate-the-best-achievable-performance-and-how-do-you-know-when-to-stop-improving-a-model)
- **8.62** [What is the i.i.d. assumption, and what happens to generalisation when it breaks?](#862-what-is-the-iid-assumption-and-what-happens-to-generalisation-when-it-breaks)

---

### 8.1 Why do we need machine learning, and where is it used?

We need machine learning when the rules that map inputs to outputs are too complex, too numerous or too fast-changing to write by hand, but examples of the right behaviour are plentiful. Instead of a programmer anticipating every case, the system learns the mapping from data and keeps improving as more data arrives.

**Traditional programming versus machine learning:**

```text
Traditional:   rules + data      -> program -> answers
ML:            data + answers    -> training -> model (the learned rules)
               model + new data  -> predictions
```

**When ML is the right tool:**

- **Rules are too complex to write down:** recognising handwriting, faces or speech. Nobody can write the if-else rules for "this is a cat".
- **Rules change constantly:** spam, fraud and abuse patterns shift as attackers adapt; a learned model can be retrained instead of rewritten.
- **Decisions must be personalised at scale:** ranking feeds, recommending products or pricing for millions of users, each with different behaviour.
- **Patterns are hidden in large data:** predicting machine failure from thousands of sensor signals, or protein structure from sequences.
- **Humans are the bottleneck:** triaging support tickets, extracting fields from invoices, transcribing calls.

**Typical applications by domain:**

| Domain | Examples |
| --- | --- |
| Perception | Image classification, object detection, OCR, speech recognition |
| Language | Search, translation, summarisation, chatbots, document extraction |
| Commerce | Recommendations, demand forecasting, dynamic pricing, churn prediction |
| Finance | Fraud detection, credit scoring, anti-money laundering, algorithmic trading |
| Healthcare | Medical image analysis, risk scores, drug discovery, protein folding |
| Operations | Predictive maintenance, route optimisation, anomaly detection, quality control |
| Autonomy and games | Self-driving perception and planning, robotics, AlphaGo-style game play |

**When ML is the wrong tool:** the logic is simple and stable enough for rules, the cost of an error is unacceptable without guarantees, there is little or no representative data, or nobody will act on the prediction. Framing this decision is covered in 8.58.

**What interviewers look for:** not a list of buzzwords but the reasoning: ML trades explicit logic for learned statistical patterns, which buys flexibility and scale at the cost of guarantees, interpretability and a dependency on data quality.

### 8.2 What is machine learning, formally, and how does it relate to AI and deep learning?

Machine learning is the study of algorithms that improve their performance on a task from experience (data), without being explicitly programmed with the rules. "Learning" does not mean memorising the training examples; it means extracting a general rule that performs well on new, unseen examples from the same distribution.

**Mitchell's classic definition (1997):** a program learns from experience E with respect to a class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E.

- **Spam filter:** T = classify emails, P = precision and recall on new mail, E = a corpus of labelled emails.
- **Chess engine:** T = play chess, P = win rate, E = games played against itself.

**Statistical-learning formulation:** we have examples $(x_i, y_i)$ drawn from an unknown distribution $P$, a hypothesis class $\mathcal{H}$ of candidate functions and a loss $L$. We want the function with the lowest expected (true) risk, but we can only measure the empirical risk on the sample:

$$
R(f) = \mathbb{E}_{(x,y) \sim P}\left[L(f(x), y)\right], \qquad \hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n} L(f(x_i), y_i)
$$

Training is (regularised) empirical risk minimisation: pick $\hat{f}$ in $\mathcal{H}$ that makes $\hat{R}$ small. The central question of ML is when a small $\hat{R}$ implies a small $R$, which is generalisation (see 8.4).

**The three ingredients of every learner:**

1. **Representation:** the hypothesis class (linear models, trees, neural networks).
2. **Evaluation:** the loss or objective that scores a hypothesis.
3. **Optimisation:** the search procedure (closed form, gradient descent, greedy splitting).

**How ML relates to AI and deep learning:**

```text
Artificial intelligence  (any technique that makes machines act intelligently:
 |                        search, planning, logic, knowledge graphs, ML)
 +-- Machine learning    (learns behaviour from data)
      +-- Deep learning  (ML with multi-layer neural networks that learn their own features)
           +-- Foundation models / generative AI (very large pretrained deep networks)
```

| Aspect | Artificial intelligence | Machine learning |
| --- | --- | --- |
| Scope | The broad goal: systems that perceive, reason, plan and act | One family of methods for reaching that goal |
| Approach | Can be rule-based, search-based, logic-based or learned | Always learned from data |
| Example | A chess engine using hand-written search and evaluation | A spam filter trained on labelled email |

**Main types:** supervised, unsupervised, semi-supervised, self-supervised and reinforcement learning (see 8.13). Deep learning versus classical ML is compared in 8.32.

**Interview tip:** give the one-line intuition first, then the formal E/T/P or empirical-risk view. Mentioning that the goal is generalisation, not fitting the training data, is what separates a strong answer.

### 8.3 What is the difference between deduction and induction, and how does machine learning relate to induction?

Deduction reasons from general rules to a specific conclusion that is guaranteed if the premises are true. Induction reasons from specific observations to a general rule that is probable but never certain. Supervised machine learning is automated induction: it generalises from a finite sample of examples to a rule meant to hold on unseen cases.

| | Deduction | Induction |
| --- | --- | --- |
| Direction | General to specific | Specific to general |
| Certainty | Conclusion is guaranteed if premises are true | Conclusion is probable and can be overturned by new data |
| Adds new knowledge? | No, it makes implicit knowledge explicit | Yes, it proposes a new general rule |
| Example | All humans are mortal; Socrates is human; so Socrates is mortal | Every swan I have seen is white; so all swans are white (until a black swan appears) |
| In AI | Expert systems, logic programming, theorem provers, rule engines | Machine learning, statistical estimation |

A third mode, **abduction**, infers the most plausible explanation for an observation (the lawn is wet, so it probably rained). Diagnosis systems and root-cause analysis are abductive.

**How ML relates to induction:**

- A model is fitted to training examples and then applied to new inputs. That leap from "seen" to "unseen" is inductive.
- Because induction is never certain, ML outputs are statistical: we report expected error, confidence intervals and calibration rather than proofs.
- Hume's **problem of induction** says no finite set of observations logically justifies a general rule. ML sidesteps it with assumptions: the future resembles the past (training and test data come from the same distribution) and the true rule lies in, or near, the model family we chose.
- Those assumptions are the learner's **inductive bias** (see 8.37). Without some bias, induction is impossible: infinitely many functions agree with any finite training set and disagree everywhere else.

**Deduction still has a place:** once a model is trained, applying it is deductive (rule plus input gives output). Hybrid systems combine both: business rules or ontologies (see 8.50) constrain learned models, and inductive logic programming (see 8.34) learns logical rules that are then used deductively.

**Practical consequence:** an inductive system can be confidently wrong when the world changes. That is why production ML needs held-out evaluation, monitoring for distribution shift and fallbacks, whereas a deductive rule engine needs only its rules to be correct.

### 8.4 Why should machine learning work at all? Explain generalisation, PAC learning and VC dimension at an interview level.

Machine learning works when three things hold: the data contains a real, stable pattern; the training and future data come from the same distribution; and the model family is rich enough to capture the pattern but constrained enough that the sample pins it down. Learning theory makes the last point precise: with enough data relative to model capacity, low training error implies low test error with high probability.

**Why it can fail entirely:** if labels were pure noise, or the future distribution unrelated to the past, no algorithm could beat guessing. The No Free Lunch theorem (see 8.37) says that averaged over all possible problems no learner wins; ML works because real problems have structure that our inductive biases match.

**Generalisation gap:** the quantity we care about is the difference between true risk $R(h)$ and training risk $\hat{R}(h)$. For a single fixed hypothesis, Hoeffding's inequality says the training average concentrates around the true risk at rate $1/\sqrt{n}$. The difficulty is that we choose $h$ after looking at the data, so we need the bound to hold for every hypothesis in the class at once.

**PAC learning (Probably Approximately Correct, Valiant 1984):** a class is PAC-learnable if, for any accuracy $\varepsilon$ and confidence $\delta$, enough samples guarantee that with probability at least $1 - \delta$ the learned hypothesis has error at most $\varepsilon$. For a finite hypothesis class, a union bound gives, simultaneously for all $h$ in $\mathcal{H}$:

$$
R(h) \le \hat{R}(h) + \sqrt{\frac{\ln \lvert \mathcal{H} \rvert + \ln(2/\delta)}{2n}}
$$

Read it as: the gap shrinks with more data $n$ and grows (only logarithmically) with the size of the model class.

**VC dimension (Vapnik and Chervonenkis):** most model classes are infinite, so $\ln \lvert \mathcal{H} \rvert$ is replaced by a measure of capacity. The VC dimension is the largest number of points the class can **shatter**, that is, label in every possible way.

| Model class | VC dimension |
| --- | --- |
| Threshold on a line | 1 |
| Linear classifier in $d$ dimensions (with bias) | $d + 1$ |
| Axis-aligned rectangles in 2D | 4 |
| Sine classifiers $\text{sign}(\sin(\omega x))$ | Infinite, despite one parameter |

The VC bound has the same shape: the gap is roughly of order $\sqrt{(d_{VC}\log n + \log(1/\delta))/n}$. A rule of thumb that follows is that you want many more examples than the effective capacity of the model.

**What this explains in practice:**

- Why overfitting happens: high capacity relative to $n$ makes the gap large (see 8.5 and 8.6).
- Why regularisation, simpler models and more data all help: they reduce effective capacity or increase $n$.
- Why the sine example matters: capacity is not the same as parameter count.

**Where the classical theory falls short:** modern neural networks have enormous VC dimension and can fit random labels (Zhang et al., 2017), yet they generalise well on real data. The classical bounds are vacuous for them. Current explanations involve the implicit bias of SGD, flat minima, margin and norm-based bounds, PAC-Bayes and double descent (see 11.29, 11.30 and 11.31).

**What interviewers look for:** at a senior level, not derivations but the intuition (gap shrinks with data, grows with capacity), awareness of the i.i.d. assumption behind every guarantee, and honesty that deep learning generalisation is only partly explained by classical theory.

### 8.5 What are overfitting and underfitting, why do they occur, and how do you fix each?

Overfitting is when a model learns the training data too closely, including its noise, so it performs well on training data and poorly on unseen data (low bias, high variance). Underfitting is when a model is too simple or too constrained to capture the real pattern, so it performs poorly on both (high bias, low variance). The aim is the capacity that minimises error on unseen data.

**Symptoms:**

| | Underfitting | Good fit | Overfitting |
| --- | --- | --- | --- |
| Training error | High | Low | Very low |
| Validation error | High, close to training | Low, close to training | Much higher than training |
| Bias and variance | High bias, low variance | Balanced | Low bias, high variance |
| Learning curves | Both plateau high and close together | Converge at a low error | Large gap that narrows slowly with more data |

"High" is relative to a reference such as human-level or Bayes error (see 8.61). A 10 percent error is underfitting if experts achieve 2 percent, and fine if the labels are 10 percent noisy.

**Why overfitting occurs:**

- Model capacity is large relative to the amount of data (deep unpruned trees, 1-NN, huge networks on small datasets; see 8.44).
- Too many irrelevant or noisy features, especially when features outnumber samples.
- Training too long without early stopping, so the network starts memorising.
- Too little regularisation.
- Noisy labels, which a flexible model will memorise.
- Indirect overfitting: tuning many hyperparameters against one validation set, or data leakage that makes validation look better than reality (see 8.39).

**Why underfitting occurs:**

- The model family cannot express the relationship (a linear model on a strongly non-linear problem).
- Missing or poor features: the signal is not in the inputs given to the model.
- Too much regularisation, or training stopped too early.
- Optimisation problems: learning rate too low or too high, bad initialisation, vanishing gradients.

**Fixes:**

| Fix overfitting (reduce variance) | Fix underfitting (reduce bias) |
| --- | --- |
| More, or more diverse, training data; data augmentation | More expressive model (deeper trees, more layers, kernels) |
| Regularisation (L1, L2, elastic net, weight decay) | Less regularisation |
| Simpler model, pruning, fewer features | Better features, interactions, domain signals |
| Early stopping | Train longer, fix the optimiser and learning rate |
| Dropout (neural networks) | Remove dropout or reduce its rate |
| Bagging and random forests | Boosting (sequentially reduces bias) |
| Cross-validation for honest model selection | Check the data pipeline: wrong joins or missing signal often look like underfitting |

**A practical diagnostic order:**

1. Make sure the model can overfit a tiny subset. If it cannot, there is a bug or an optimisation problem, not a bias problem.
2. Compare training error to a target (human level, a strong baseline). If far off, attack bias first.
3. Compare validation to training error. If the gap is large, attack variance.
4. Plot learning curves to decide whether more data will help (see 8.60).
5. Check for leakage and train/serve skew before trusting any improvement.

**Pitfall:** "overfitting" also happens to the validation set when you iterate on it many times. Keep a final test set that is used once, and treat large validation-to-test drops as a warning (see 8.54).

### 8.6 Explain the bias-variance trade-off mathematically and practically.

The bias-variance trade-off says that a model's expected error on unseen data comes from three sources: bias (systematic error from wrong or overly simple assumptions), variance (sensitivity to the particular training sample) and irreducible noise. Increasing model flexibility usually lowers bias but raises variance, so good generalisation means choosing the capacity, regularisation and data size that minimise their sum.

**Definitions:**

- **Bias:** how far the average prediction (averaged over many hypothetical training sets) is from the truth. High bias means underfitting.
- **Variance:** how much predictions change when the model is trained on a different sample of the same size. High variance means overfitting.
- **Noise:** randomness in the target that no model can predict, which sets a floor on achievable error.

**Derivation for squared error.** Assume $y = f(x) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$ and $\text{Var}(\varepsilon) = \sigma^2$. Let $\hat{f}$ be trained on a random dataset $D$. At a point $x$, taking expectation over datasets and noise:

$$
\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}_D[\hat{f}(x)] - f(x)\right)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D\left[\left(\hat{f}(x) - \mathbb{E}_D[\hat{f}(x)]\right)^2\right]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}
$$

The cross terms vanish because the noise has zero mean and is independent of $D$. Averaging over $x$ gives the expected test MSE.

**A concrete example, k-nearest neighbours regression:** the variance term is roughly $\sigma^2 / k$, so a larger $k$ lowers variance, while averaging over more distant neighbours raises bias. $k = 1$ has near-zero bias and high variance; $k = n$ predicts the global mean, with high bias and low variance.

**How common levers move the two terms:**

| Lever | Bias | Variance |
| --- | --- | --- |
| More flexible model (deeper trees, more features, larger network) | Down | Up |
| Stronger regularisation (larger $\lambda$, more dropout) | Up | Down |
| More training data | Unchanged | Down |
| Bagging and random forests | Roughly unchanged | Down |
| Boosting | Down | Can rise with many rounds |
| Better features | Down | Usually unchanged |

**Simulating it in Python** makes the idea concrete in an interview:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x_test = np.linspace(0, 1, 200).reshape(-1, 1)

for depth in [1, 3, 10]:
    preds = []
    for _ in range(200):                      # 200 independent training sets
        x = rng.uniform(0, 1, (50, 1))
        y = f(x).ravel() + rng.normal(0, 0.3, 50)
        preds.append(DecisionTreeRegressor(max_depth=depth).fit(x, y).predict(x_test))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - f(x_test).ravel()) ** 2)
    var = np.mean(preds.var(0))
    print(f"depth={depth:2d}  bias^2={bias2:.3f}  variance={var:.3f}")
```

Shallow trees show high bias and low variance; deep trees the opposite.

**Practical use:**

1. Estimate the noise floor (human-level or Bayes error).
2. The gap between training error and that floor approximates avoidable bias.
3. The gap between validation and training error approximates variance.
4. Attack the larger of the two first (see 8.5 for the fixes).

**Nuances a senior candidate should mention:**

- The clean decomposition holds for squared error. For 0-1 loss and cross-entropy, analogous decompositions exist but are messier; the intuition still applies.
- Very large over-parameterised models break the classic U-shaped curve: test error can fall again past the interpolation threshold (double descent, see 11.30). Implicit regularisation from SGD keeps effective variance low (see 11.31).
- Bias in this sense is statistical. It is unrelated to fairness bias in data (see 8.40) and to the bias (intercept) term of a model.

### 8.7 What is the difference between parametric and non-parametric models?

A parametric model assumes a fixed functional form with a fixed, finite number of parameters chosen before seeing the data; learning means estimating those parameters. A non-parametric model makes weaker assumptions about the form, and its effective number of parameters (its complexity) grows with the amount of data.

**Parametric models:**

- Assume a specific form or distribution (linear relationship, Gaussian class densities).
- Fixed, finite number of parameters, decided before training.
- Once trained, the training data can be discarded: the parameters are the model.
- Parameters often have a clear statistical interpretation (coefficients, means, variances).
- Complexity does not grow with more data.

**Non-parametric models:**

- Make few or no assumptions about the form of the function or distribution.
- The effective number of parameters grows with the size of the data.
- Often keep the training data (or a large summary of it) at prediction time.
- More flexible, so they need more data and more care to avoid overfitting.

| Feature | Parametric | Non-parametric |
| --- | --- | --- |
| Distribution or form assumption | Required | Weak or none |
| Number of parameters | Fixed | Grows with data |
| Flexibility | Lower | Higher |
| Data requirement | Usually less | Usually more |
| Inference cost | Small and constant | Can grow with the training set |
| Examples | Linear and logistic regression, Naive Bayes, Gaussian Mixture Models | k-NN, decision trees, random forests, kernel density estimation, Gaussian processes, DBSCAN |

**Edge cases worth discussing:**

| Parametric | Non-parametric |
| --- | --- |
| Linear regression | k-NN |
| Logistic regression | Decision trees (can grow unbounded) |
| Naive Bayes (with assumed distribution) | Kernel density estimation |
| Neural nets with fixed architecture\* | k-means (arguably semi-parametric, see below) |

- **\*Neural nets with a fixed architecture** are technically parametric: the parameter count is set before training and does not grow with the data. In practice, very large networks behave a lot like non-parametric models. They can memorise their training set, and when there is more data we usually pick a bigger architecture, so their effective capacity grows with the data.
- **k-means as "semi-parametric":** once trained, the model is just k centroids (k × d numbers). That is a fixed, finite set of parameters, which points to parametric. But the number of clusters is often chosen from the data (elbow or silhouette), and k-means assumes no distributional form beyond spherical, similarly sized clusters. So it sits between the two. A Gaussian Mixture Model is the clearly parametric probabilistic counterpart.
- **Kernel SVMs** keep a subset of training points (the support vectors), whose number can grow with the data, so they behave non-parametrically.

**Trade-offs to mention in an interview:**

- **Parametric:** fast, small and easy to interpret. It works with little data but has high bias if the assumed form is wrong (for example, a linear model on a non-linear relationship).
- **Non-parametric:** flexible, with low bias. It needs more data and is prone to overfitting without controls such as k, tree depth or kernel bandwidth. Inference can be expensive because k-NN and KDE keep the whole training set, which matters for latency and memory in production.

### 8.8 What is the difference between online and offline learning, and where is online learning used?

**Online learning** updates the model incrementally as each new example (or mini-batch) arrives, typically discarding the data after the update. **Offline (batch) learning** trains on a fixed, complete dataset at once and is retrained periodically, usually from scratch.

| | Offline / batch | Online |
| --- | --- | --- |
| Data | Full historical dataset available | Streaming, one example or mini-batch at a time |
| Updates | Periodic full retrains (daily or weekly) | Continuous, incremental |
| Adaptation to drift | Slow (until the next retrain) | Fast |
| Compute and memory | High per retrain; needs all data | Low per update; constant memory |
| Stability and reproducibility | High, easier to validate | Risk of drift, catastrophic forgetting and data poisoning |
| Algorithms | Most models | SGD-based linear models, online gradient descent, Hoeffding trees, bandits, `partial_fit` in scikit-learn, River, Vowpal Wabbit |
| Example | Training an image classifier on a fixed dataset | Fraud detection adapting to new transactions |

**Applications of online learning:**

- **Ad click-through and recommendation ranking**, where user interests change quickly.
- **Fraud and spam detection**, where attackers adapt.
- **Stock and price prediction, dynamic pricing.**
- **News and feed personalisation**, and contextual bandits as an adaptive alternative to A/B tests.
- **IoT and sensor anomaly detection** on streams, and **datasets too large to fit in memory** (out-of-core learning).

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss="log_loss")
for X_batch, y_batch in stream_of_batches():          # e.g. read from Kafka or files
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Production concerns:**

- The learning rate controls how fast the model forgets; too high and it chases noise, too low and it does not adapt.
- Guard against bad or adversarial data with validation gates and anomaly checks before updates are applied.
- Keep shadow evaluation, model snapshots and fast rollback.
- Labels often arrive late (a chargeback weeks after a transaction), so "online" is limited by feedback delay.
- Many teams use a hybrid: a periodic offline model plus online fine-tuning or recalibration of the last layer or bias terms.

### 8.9 What are epochs, batches, batch size and iterations, and how do you choose the batch size?

An **epoch** is one full pass over the training set. A **batch** (mini-batch) is a subset of examples processed together before one weight update. **Batch size** is the number of examples in a batch. An **iteration** (step) is one weight update, that is, one batch processed.

**The relationship:**

$$
\text{iterations per epoch} = \left\lceil \frac{N}{B} \right\rceil, \qquad \text{total iterations} = \text{epochs} \times \left\lceil \frac{N}{B} \right\rceil
$$

**Worked example:** 1,000 training examples with batch size 100 give 10 iterations per epoch. Training for 5 epochs gives 50 iterations. With 1,050 examples there are 11 iterations per epoch, the last batch holding 50 examples, unless you set `drop_last=True`.

```python
from torch.utils.data import DataLoader

loader = DataLoader(train_ds, batch_size=100, shuffle=True, drop_last=False)
for epoch in range(5):
    for X, y in loader:            # one iteration per batch
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
```

**Why not feed the whole dataset at once?**

- It usually does not fit in GPU memory: activations for every example must be stored for backpropagation.
- One update per epoch is slow; mini-batches give many cheaper, slightly noisy updates and reach a good solution faster in wall-clock time.
- Some gradient noise helps optimisation escape sharp minima and saddle points (see 11.16).

**Batch size trade-offs:**

| | Small batches (for example 16 to 64) | Large batches (for example 1,024 and up) |
| --- | --- | --- |
| Gradient estimate | Noisy (noise falls roughly as $1/\sqrt{B}$) | Accurate, smooth |
| Updates per epoch | Many | Few |
| Hardware use | Poor GPU utilisation | High throughput, good parallelism |
| Memory | Low | High |
| Generalisation | Noise acts as a regulariser | Can converge to sharper minima without retuning (see 11.28) |
| Learning rate | Smaller | Larger, scaled with batch size plus warmup (see 11.34) |

**How to choose in practice:**

1. Start with the largest batch that fits comfortably in memory with a typical value (32 to 512 for most supervised tasks).
2. When you change batch size, retune the learning rate; the linear scaling rule (multiply the learning rate by the same factor) with warmup is a common starting point.
3. If the batch you need does not fit, use gradient accumulation (see 11.39) or mixed precision.
4. Watch for batch-dependent layers: batch normalisation behaves badly with very small batches (see 13.13).
5. Measure by time-to-target-accuracy, not by epochs, because a larger batch does fewer steps per epoch.

**Pitfall:** comparing runs "at 10 epochs" with different batch sizes compares different numbers of updates. Compare at equal compute or equal samples seen.

### 8.10 Compare batch, stochastic and mini-batch gradient descent. Which is used in practice?

All three minimise the loss by stepping against the gradient; they differ in how many examples are used to estimate each gradient. Batch gradient descent uses the whole dataset, stochastic gradient descent (SGD) uses one example, and mini-batch gradient descent uses a small batch. Almost all modern training uses mini-batches.

The general update with learning rate $\alpha$ over a set of examples $S$ is:

$$
\theta \leftarrow \theta - \alpha \thinspace \frac{1}{\lvert S \rvert} \sum_{i \in S} \nabla_{\theta} L(f_{\theta}(x_i), y_i)
$$

where $S$ is the full dataset (batch GD), a single random example (SGD) or a random mini-batch.

| | Batch GD | Stochastic GD | Mini-batch GD |
| --- | --- | --- | --- |
| Examples per update | All $N$ | 1 | $B$, typically 32 to 4,096 |
| Gradient quality | Exact | Very noisy | Noisy but usable |
| Cost per update | Very high | Very low | Moderate |
| Updates per epoch | 1 | $N$ | $N / B$ |
| Convergence path | Smooth, deterministic | Erratic, fluctuates around the minimum | Fairly smooth |
| Hardware | Memory-bound for large data | Cannot use vectorised GPU maths well | Matches GPU parallelism |
| Escaping saddles and sharp minima | Poor | Good | Good |
| Typical use | Small convex problems, L-BFGS | Online learning, streaming | Deep learning and most large-scale ML |

**Points to make about each:**

- **Batch GD:** with a suitable learning rate it converges steadily to the minimum of a convex loss, but each step needs a full pass over the data, which is impractical for millions of examples.
- **SGD:** each update is cheap and the noise can help escape poor regions, but the path oscillates. A decaying learning rate is needed to settle (see 11.20).
- **Mini-batch GD:** balances both and uses vectorised hardware efficiently. It is the default.

**Terminology trap:** frameworks call the optimiser "SGD" (`torch.optim.SGD`) even though it is used with mini-batches. In practice, "SGD" usually means mini-batch SGD, often with momentum.

**What is used now:** mini-batch training with an adaptive or momentum-based optimiser, usually AdamW for Transformers and SGD with momentum for many vision models (see 11.17). Full-batch methods survive for small convex problems, for example L-BFGS in scikit-learn's logistic regression.

See 11.12 for a deeper comparison of SGD and GD, and 11.16 for why SGD noise can improve generalisation.

### 8.11 What are embeddings in Machine Learning?

> **See also:** [Question 17.15 — What are embeddings?](../part-4-generative-ai/17-large-language-model-foundations.md#1715-what-are-embeddings), which covers the same question from another angle.

An **embedding** is a learned, dense vector representation of an object (a word, sentence, image, product, user or graph node) in which **geometric closeness reflects similarity that matters for the task**. Embeddings turn discrete or high-dimensional inputs into compact continuous features that models can compare, search and learn from.

**How embeddings are learned**

- **As a by-product of a supervised task:** an embedding layer (a lookup table of trainable vectors) is trained along with the rest of a network, for example user and item IDs in a recommender.
- **Self-supervised objectives:** word2vec predicts neighbouring words; language models learn contextual token embeddings; contrastive learning pulls matching pairs together and pushes others apart (see 8.29).
- **Pretrained encoders:** sentence-embedding models and image encoders (for example CLIP) produce general-purpose embeddings that can be reused without training.

**Key properties**

- **Dimensionality:** typically 64 to 4,096 dimensions; far smaller than one-hot encodings of large vocabularies or catalogues.
- **Similarity:** cosine similarity or dot product; the right choice depends on how the model was trained.
- **Static versus contextual:** word2vec gives one vector per word; transformer models give a different vector for the same word in different contexts ("bank" of a river versus a bank account).
- **Task-specific geometry:** an embedding trained for one notion of similarity may be poor for another (see 19.24).

**Where they are used**

- Features for downstream models (text, categorical IDs, images).
- Semantic search, retrieval-augmented generation and recommendation (nearest-neighbour search, see Chapter 19).
- Clustering, deduplication, anomaly detection and visualisation.

**Pitfalls**

- Embeddings inherit **biases** in their training data.
- Vectors from different models (or model versions) are **not comparable**; changing the model means re-embedding everything.
- High similarity does not guarantee relevance or correctness; evaluate on your own task.

### 8.12 What are logits, softmax and cross-entropy, and how do they fit together?

Logits are the raw, unnormalised scores a model outputs; softmax turns them into a probability distribution over classes; cross-entropy measures how much probability that distribution assigns to the correct class and is the loss minimised for classification. Together they form the standard output stack of almost every classifier and language model.

**Logits:** real numbers of any sign and size, one per class, produced by the final linear layer. For binary classification there is often a single logit, which is the log-odds: $z = \log \frac{p}{1 - p}$.

**Softmax** converts a logit vector $z$ into probabilities:

$$
\text{softmax}(z)_i = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}
$$

- Outputs lie between 0 and 1 and sum to 1.
- It is invariant to adding the same constant to all logits, which is used for numerical stability: subtract $\max_j z_j$ before exponentiating.
- It amplifies differences: the largest logit takes most of the mass.
- **Temperature** $T$ (normally 1) controls sharpness: $T \lt 1$ makes the distribution peakier, $T \gt 1$ flatter. This is what LLM sampling temperature changes.
- For two classes, softmax reduces to the sigmoid of the logit difference.

**Cross-entropy** between the true distribution $y$ (usually one-hot) and the prediction $p$:

$$
L_{\text{CE}} = -\sum_{i} y_i \log p_i, \qquad L_{\text{BCE}} = -\left[y \log p + (1 - y)\log(1 - p)\right]
$$

- With a one-hot label it is simply $-\log p_{\text{correct}}$: confident wrong predictions are punished very heavily.
- It equals the negative log-likelihood, so minimising it is maximum likelihood estimation.
- Cross-entropy equals the entropy of $y$ plus the KL divergence from $y$ to $p$, so minimising it minimises the KL divergence (see 17.74).

**Why the combination is so convenient:** the gradient of softmax followed by cross-entropy with respect to the logits is simply $p - y$. It is bounded and well behaved, unlike MSE on sigmoid outputs, which saturates and learns slowly when the model is confidently wrong.

**In code, pass logits, not probabilities:**

```python
import torch
import torch.nn as nn

logits = model(x)                          # shape (batch, num_classes), no softmax
loss = nn.CrossEntropyLoss()(logits, y)    # applies log-softmax internally, stable

binary_logit = model_bin(x).squeeze(-1)
loss_bin = nn.BCEWithLogitsLoss()(binary_logit, y_bin.float())

probs = torch.softmax(logits, dim=-1)      # only for reporting or decisions
```

`CrossEntropyLoss` combines `log_softmax` and `NLLLoss` using the log-sum-exp trick. Applying softmax yourself and then taking the log loses precision and can produce `-inf` and NaN losses.

**Practical extras:**

- **Label smoothing** replaces one-hot targets with, for example, 0.9 and 0.1 spread over other classes, which reduces over-confidence.
- **Calibration:** softmax outputs from modern networks are often over-confident; temperature scaling on a validation set fixes this cheaply.
- **Multi-label problems** use independent sigmoids with binary cross-entropy, not softmax (see 8.49).
- **Class imbalance:** use class weights or focal loss (see 8.28).

### 8.13 What are the main types of machine learning, and how do supervised and unsupervised learning differ?

The main types are defined by the feedback signal available: supervised learning has labelled examples, unsupervised learning has only inputs, semi-supervised learning mixes a few labels with many unlabelled examples, self-supervised learning creates labels from the data itself, and reinforcement learning learns from rewards gained by acting in an environment.

| Type | Data and signal | Goal | Typical tasks and algorithms |
| --- | --- | --- | --- |
| Supervised | Labelled input-output pairs | Predict the output for new inputs | Classification, regression; linear models, trees, gradient boosting, neural networks |
| Unsupervised | Inputs only | Discover structure | Clustering (k-means, DBSCAN), dimensionality reduction (PCA, UMAP), density estimation, association rules |
| Semi-supervised | Few labels, many unlabelled examples | Supervised performance with fewer labels | Pseudo-labelling, consistency regularisation, label propagation (see 8.33) |
| Self-supervised | Labels generated from the data (next token, masked patch, augmented pair) | Learn general representations | LLM pretraining, BERT, contrastive learning (see 8.29), masked autoencoders |
| Reinforcement | Scalar rewards from interaction | Learn a policy maximising cumulative reward | Q-learning, policy gradients, PPO (see 8.14) |

**Supervised versus unsupervised in more detail:**

| Aspect | Supervised | Unsupervised |
| --- | --- | --- |
| Data | Labelled | Unlabelled |
| Feedback | Direct error signal (prediction versus label) | No ground truth |
| Main functions | Classification, regression, ranking, forecasting, risk scoring | Clustering, dimensionality reduction, anomaly detection, association rules, feature learning, exploration |
| Examples | Spam detection, house prices, credit risk | Customer segmentation, market-basket analysis, fraud outliers |
| Evaluation | Accuracy, F1, ROC-AUC, RMSE on held-out labels | Silhouette score, reconstruction error, downstream usefulness, human review |
| Main cost | Labelling | Interpreting and validating the output |

**Points that show seniority:**

- **Labels are the bottleneck.** Most real projects are limited by label cost and quality, which is why semi-supervised learning, active learning (see 8.43) and weak supervision matter.
- **The boundaries blur.** Self-supervised pretraining followed by supervised fine-tuning is now the dominant recipe for text, images and audio. LLMs are trained self-supervised, then with supervised fine-tuning and RL from feedback.
- **Unsupervised results are hard to validate.** Clusters always appear, whether or not they are meaningful; judge them by stability and by whether they drive a decision.
- **Anomaly detection spans both:** unsupervised when you have no labelled anomalies, supervised when you have enough labelled fraud, and semi-supervised when you train only on known-normal data.

### 8.14 What is Reinforcement Learning?

> **See also:** [Question 17.78 — What is reinforcement learning?](../part-4-generative-ai/17-large-language-model-foundations.md#1778-what-is-reinforcement-learning), which covers the same question from another angle.

**Reinforcement learning (RL)** trains an **agent** to choose **actions** in an **environment** so as to maximise **cumulative reward** over time. Unlike supervised learning, there are no labelled correct actions: the agent learns from the consequences of its own actions, and rewards may arrive long after the actions that caused them. Chapter 16 covers RL in depth; this is the overview.

**Core concepts**

- **State:** what the agent observes about the environment.
- **Action:** what the agent can do.
- **Reward:** a scalar signal of how good the immediate outcome was.
- **Policy:** the agent's strategy, mapping states to actions.
- **Value function:** expected future reward from a state (or state and action) under a policy.
- **Discount factor:** how much future rewards count compared with immediate ones.

**How it differs from other learning**

| Aspect | Supervised learning | Reinforcement learning |
| --- | --- | --- |
| Feedback | Correct label for each example | Reward for the outcome, often delayed |
| Data | Fixed dataset | Generated by the agent's own behaviour |
| Main difficulty | Generalisation | Credit assignment and exploration (see 8.26) |

**Main families of methods**

- **Value-based:** learn action values and act greedily (Q-learning, DQN).
- **Policy-based:** optimise the policy directly (policy gradients, PPO).
- **Actor-critic:** combine both.
- **Model-based:** learn a model of the environment and plan with it (for example MuZero).

**Where it is used**

- Games and simulation (AlphaGo, AlphaZero), robotics and control, resource allocation, and **training large language models** (RL from human feedback and RL with verifiable rewards, see Chapter 22).
- In business problems, simpler **bandits** often suffice for decisions such as which content or offer to show (see 16.74 and 28.37).

**Practical caveats:** RL needs many interactions, careful reward design (agents exploit flawed rewards), and safe exploration; offline RL and simulators are used when real-world exploration is costly.

### 8.15 What is the difference between classification and regression, and what targets does each require?

Classification predicts a discrete category (a class label or class probabilities); regression predicts a continuous numeric value. The difference is in the type of target variable, which then drives the loss function, the output layer and the evaluation metrics.

| Aspect | Classification | Regression |
| --- | --- | --- |
| Target variable | Categorical or discrete, for example $y \in \lbrace 0, 1 \rbrace$ or one of $K$ classes | Continuous, $y \in \mathbb{R}$ |
| Question answered | "Which category?" | "How much?" |
| Example | Will this customer churn? Is this email spam? | What will this customer spend next year? What is this house worth? |
| Output | Class label, or a probability per class | A real number (or a predictive distribution) |
| Typical loss | Cross-entropy (log loss), hinge | Squared error, absolute error, Huber |
| Output layer | Sigmoid or softmax | Linear (identity) |
| Algorithms | Logistic regression, decision trees, random forests, gradient boosting, SVM, neural networks | Linear regression, ridge, lasso, trees and forests, gradient boosting, SVR, neural networks |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss | MAE, MSE, RMSE, MAPE, $R^2$ |

**Nuances interviewers like to probe:**

- **Logistic regression is a classifier** despite its name; it regresses the log-odds and thresholds the probability (see 9.11).
- **Many algorithms do both:** trees split to minimise impurity (classification) or variance (regression); neural networks just change the output layer and loss.
- **Not every numeric target is regression.** Ratings from 1 to 5 are ordinal; treat them with ordinal regression or as classification, depending on whether the gaps are meaningful. Counts are better modelled with Poisson or Tweedie losses than plain MSE.
- **Converting between them:** a regression output can be thresholded into classes (predicted spend above a level means "high value"), and a classification probability is itself a regression on a 0-1 scale. Bucketing a continuous target into classes throws away information and should be a deliberate choice.
- **Probabilistic outputs matter:** in both cases, predicting a distribution (calibrated class probabilities, or quantiles for regression) is often more useful for decisions than a point prediction.
- **Related task types:** ranking (order items), multi-label classification (several labels per item, see 8.49) and forecasting (regression over time with temporal validation).

### 8.16 What are L1 and L2 loss functions, and when would you use each?

L1 loss (mean absolute error) penalises the absolute difference between prediction and target; L2 loss (mean squared error) penalises the squared difference. The choice decides what the model estimates: minimising L2 gives the conditional **mean** of the target, minimising L1 gives the conditional **median**, which is why L1 is far more robust to outliers.

$$
\text{L1 (MAE)} = \frac{1}{n}\sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert, \qquad \text{L2 (MSE)} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

| | L1 / MAE | L2 / MSE |
| --- | --- | --- |
| Optimal constant prediction | Median | Mean |
| Sensitivity to outliers | Low: an error of 100 costs 100 | High: an error of 100 costs 10,000 |
| Gradient | Constant magnitude ($\pm 1$), undefined at 0 | Proportional to the error, smooth everywhere |
| Optimisation | Harder: non-smooth, needs subgradients or smaller learning rates near the optimum | Easy: smooth and convex, closed form for linear regression |
| Statistical view | Maximum likelihood under Laplace noise | Maximum likelihood under Gaussian noise |
| Interpretation | Same units as the target | Squared units; take the root (RMSE) for reporting |

**When to use which:**

- **L2** when large errors are disproportionately costly, noise is roughly Gaussian and outliers are rare or meaningful.
- **L1** when the data has outliers or heavy tails that you do not want to dominate the fit, or when the business cares about typical (median) error, such as delivery time estimates.
- **Huber loss** (quadratic near zero, linear beyond a threshold $\delta$) combines smooth optimisation with robustness; **quantile (pinball) loss** generalises L1 to any percentile and is used for prediction intervals and demand forecasting.

```python
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

ols = LinearRegression().fit(X, y)                    # L2
huber = HuberRegressor(epsilon=1.35).fit(X, y)        # robust
median = QuantileRegressor(quantile=0.5, alpha=0).fit(X, y)  # L1
```

**Common confusion to correct:** L1 and L2 **losses** act on the prediction errors; L1 and L2 **regularisation** act on the model weights. An L1 loss does not make weights sparse; an L1 penalty on the weights does (see 8.17 and 8.41). The two can be combined freely, for example MSE loss with an L1 penalty (lasso).

**Metric versus loss:** you can train with one and report another. A model trained on MSE will look worse on MAE than one trained on MAE, so train with the loss that matches the metric the business uses (see 12.12).

### 8.17 What is regularisation? Explain L1 (lasso), L2 (ridge) and elastic net, and how the penalty changes the cost.

Regularisation is any technique that constrains a model to reduce overfitting and improve generalisation, usually at the cost of a slightly higher training error. The classic form adds a penalty on the size of the weights to the cost function, so the optimiser trades data fit against simplicity, controlled by a hyperparameter $\lambda$.

**The regularised cost:**

$$
J(w) = \underbrace{\frac{1}{n}\sum_{i=1}^{n} L(f_w(x_i), y_i)}_{\text{data fit}} + \lambda \thinspace \Omega(w)
$$

| | L1 (lasso) | L2 (ridge) | Elastic net |
| --- | --- | --- | --- |
| Penalty $\Omega(w)$ | $\sum_j \lvert w_j \rvert$ | $\sum_j w_j^2$ | $\alpha \sum_j \lvert w_j \rvert + (1 - \alpha)\sum_j w_j^2$ |
| Effect on weights | Drives some exactly to zero | Shrinks all smoothly towards zero | Sparse, but keeps groups of correlated features |
| Feature selection | Yes | No | Yes |
| Correlated features | Picks one arbitrarily, unstable | Spreads weight across them | Handles them well |
| Bayesian view | Laplace prior on weights | Gaussian prior on weights | Mixture of both |
| Closed form (linear regression) | No | Yes: $(X^\top X + \lambda I)^{-1} X^\top y$ | No |

**How the penalty reduces the cost term and the weights:** to minimise $J$, the optimiser must justify every unit of weight by a matching reduction in data loss. For L2, the gradient step becomes:

$$
w \leftarrow w - \eta\left(\nabla L + 2\lambda w\right) = (1 - 2\eta\lambda)\thinspace w - \eta \nabla L
$$

Each step multiplies the weights by a factor slightly below 1, which is why L2 is called **weight decay**. Weights that the data does not strongly support decay towards zero. The L1 gradient has constant magnitude $\lambda$, so it pushes small weights all the way to zero (see 8.41 for why).

**Why it helps:** smaller weights mean a smoother function that reacts less to small input changes and to the quirks of one training sample, which lowers variance at the cost of a little bias (see 8.56).

**Other regularisers:**

- **Dropout** in neural networks (see 13.10 and 13.73).
- **Early stopping**, which behaves like an L2 penalty for gradient descent (see 13.77).
- **Data augmentation**, **label smoothing**, **noise injection**.
- **Structural limits:** maximum tree depth, minimum samples per leaf, pruning, fewer features.
- **Ensembling** (bagging) and **batch normalisation** have regularising side effects.

**Practical guidance:**

1. **Scale features first.** Penalties compare weights directly, so unscaled features are penalised unequally.
2. **Do not penalise the intercept.**
3. **Tune $\lambda$ by cross-validation** on a log scale. In scikit-learn, `alpha` is $\lambda$ for `Ridge` and `Lasso`, but `C` in `LogisticRegression` and SVMs is the inverse ($C = 1/\lambda$), so smaller C means stronger regularisation.
4. **With Adam, use decoupled weight decay (AdamW)** rather than adding L2 to the loss (see 11.18).

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5))
model.fit(X_train, y_train)
```

**Pitfall:** too much regularisation causes underfitting; if training error is far above the target, reduce $\lambda$ before adding model complexity.

### 8.18 What is the difference between a loss function, a cost function and an objective function?

A loss function measures the error on a single example; a cost function aggregates the loss over the training set (usually the average, often plus a regularisation term); and the objective function is whatever quantity the optimiser actually minimises or maximises. In everyday use the terms are often interchanged, but the distinction matters when you design or debug training.

| Term | Scope | Example |
| --- | --- | --- |
| Loss $L(\hat{y}_i, y_i)$ | One example | Squared error $(y_i - \hat{y}_i)^2$; log loss for one prediction |
| Cost $J(\theta)$ | Whole training set (or a batch) | Mean squared error over all examples, plus $\lambda \lVert w \rVert^2$ |
| Objective | What the optimiser targets | Minimise the cost; or maximise a likelihood or reward |

$$
J(\theta) = \frac{1}{n}\sum_{i=1}^{n} L(f_{\theta}(x_i), y_i) + \lambda\thinspace \Omega(\theta)
$$

The data part of $J$ is the **empirical risk**; its expectation over the true distribution is the **true risk** we actually care about (see 8.2).

**Why the distinction matters in practice:**

- **Mini-batch training** optimises a noisy estimate of the cost computed on each batch, not the full cost.
- **Losses are often surrogates.** Accuracy (0-1 loss) is not differentiable, so we train on cross-entropy or hinge loss as a convex surrogate and evaluate on accuracy. A model can improve its loss while the metric stays flat.
- **The metric the business uses may differ again** (profit, recall at a fixed precision). Align them through the loss choice, sample weights, threshold tuning or a final selection step (see 11.40).
- **Per-example losses are a debugging tool:** sorting training examples by loss surfaces mislabelled data and hard cases.
- **Reduction matters:** `reduction="sum"` versus `"mean"` changes the effective learning rate when batch size changes.

See 11.13 for the same distinction from the optimisation angle.

### 8.19 What is a perceptron, and how does a multilayer perceptron overcome its limits?

A perceptron is the simplest neural unit: a linear binary classifier that computes a weighted sum of its inputs plus a bias and applies a step function. A multilayer perceptron (MLP) stacks layers of such units with non-linear activations, which lets it represent non-linear decision boundaries that a single perceptron cannot.

**The perceptron (Rosenblatt, 1958):**

$$
\hat{y} = \text{step}\left(\sum_{i} w_i x_i + b\right), \qquad w \leftarrow w + \eta\thinspace (y - \hat{y})\thinspace x
$$

- The update changes weights only on mistakes.
- **Perceptron convergence theorem:** if the data is linearly separable, the algorithm finds a separating hyperplane in a finite number of steps. If it is not separable, it never converges.
- **Limit:** it can only learn linear boundaries. Minsky and Papert (1969) famously showed it cannot learn XOR, which contributed to the first "AI winter".

**The multilayer perceptron:**

```text
input x  ->  [hidden layer: W1 x + b1 -> ReLU]  ->  [hidden layer ...]  ->  [output: softmax or linear]
```

- Fully connected feed-forward layers with non-linear activations (ReLU, GELU, tanh).
- Trained with backpropagation and gradient descent, which needs differentiable activations, so the step function is replaced (see 13.64).
- **Solves XOR:** one hidden layer with two units can carve the plane into the regions XOR needs.
- **Universal approximation theorem:** one hidden layer with enough units can approximate any continuous function on a compact domain. In practice, depth is far more parameter-efficient than width.
- **Why the non-linearity is essential:** without it, stacked linear layers collapse into a single linear map (see 13.65).

| | Perceptron | MLP |
| --- | --- | --- |
| Layers | One (no hidden layer) | One or more hidden layers |
| Activation | Step function | Differentiable non-linearities |
| Decision boundary | Linear | Arbitrary non-linear |
| Training | Perceptron rule | Backpropagation with gradient descent |
| Output | Hard 0/1 label | Probabilities or real values |

```python
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])                                   # XOR
print(Perceptron().fit(X, y).score(X, y))                    # at most 0.75
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs", random_state=1)
print(mlp.fit(X, y).score(X, y))                             # typically 1.0
```

**Where MLPs are used today:** as the feed-forward blocks inside every Transformer layer, as classification heads on top of pretrained encoders, and as baselines for tabular data, where gradient-boosted trees usually still win (see 13.90).

### 8.20 Explain Cross-Validation. Why is it used?

**Cross-Validation**

- Technique to evaluate model performance by splitting data into multiple train/test subsets
- Most common: **k-fold CV** — data split into k folds, model trained on k-1 folds, tested on remaining fold, repeated k times
- Final performance = average across all folds

**Why used**

- Gives more reliable performance estimate than a single train/test split
- Reduces risk of overfitting to one particular split
- Makes better use of limited data
- Helps in hyperparameter tuning and model selection

### 8.21 What is Confusion Matrix?

> **See also:** [Question 12.3 — What is the confusion matrix, why do we need it, and how do you interpret it?](12-model-evaluation-and-experimentation.md#123-what-is-the-confusion-matrix-why-do-we-need-it-and-how-do-you-interpret-it), which covers the same question from another angle.

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

### 8.22 What are precision, recall, and F1-score?

**Precision, Recall, F1-Score**

- **Precision**: Of predicted positives, how many are actually correct → $\frac{\text{TP}}{\text{TP} + \text{FP}}$, how precise are my positive predictions?
- **Recall**: Of actual positives, how many were correctly identified → $\frac{\text{TP}}{\text{TP} + \text{FN}}$, how many of the actual positive did i find/recall?
- **F1-Score**: Harmonic mean of precision and recall → $\frac{2 \times P \times R}{P + R}$
- Precision matters when false positives are costly (e.g., spam detection)
- Recall matters when false negatives are costly (e.g., disease detection)
- F1 balances both, useful for imbalanced datasets

### 8.23 What is anomaly detection?

**Anomaly detection** (outlier or novelty detection) finds observations that deviate from expected behaviour: fraudulent transactions, failing machines, intrusions, data quality errors. The hard part is rarely the algorithm; it is that anomalies are **rare, varied and often unlabelled**, and that "normal" changes over time.

**Problem settings**

- **Unsupervised:** no labels; model normal behaviour and flag what does not fit. The most common case.
- **Semi-supervised (novelty detection):** train only on known-normal data.
- **Supervised:** enough labelled anomalies exist to train a classifier; usually limited to known patterns.

**Method families**

| Family | Examples | Good for |
| --- | --- | --- |
| Statistical | z-score, robust z-score with the median absolute deviation, IQR rules, seasonal baselines | Simple, explainable single metrics |
| Distance and density | k-nearest neighbours distance, Local Outlier Factor, DBSCAN | Moderate dimensions, local anomalies |
| Isolation-based | Isolation Forest | Tabular data at scale; fast and robust default |
| Boundary-based | One-Class SVM | Smaller datasets with clear normal regions |
| Reconstruction-based | Autoencoders, PCA reconstruction error | High-dimensional data, images, sensor signals |
| Forecast-based | Large residuals from a time-series forecast | Metrics with trend and seasonality |

**Practical design points**

- **Point, contextual and collective anomalies differ:** a temperature of 30 degrees is normal in summer and anomalous in winter (contextual); a sequence of normal-looking events can be anomalous together (collective).
- **Thresholds drive alert volume:** choose them from the cost of missed anomalies versus analyst time; alert fatigue kills anomaly systems.
- **Evaluation without labels:** inject synthetic anomalies, use past incidents, and track precision of alerts reviewed by people.
- **Drift:** normal behaviour changes; retrain or use adaptive baselines.
- **Explainability:** say *why* something was flagged (which feature, how far from normal).

Chapter 9 compares the algorithms in more detail, and 28.20, 28.23 and 29.40 design production anomaly detection systems.

### 8.24 What is the difference between policy-based and value-based methods?

**Policy-Based vs Value-Based Methods (Reinforcement Learning)**

| Aspect | Policy-Based | Value-Based |
| --- | --- | --- |
| Learns | Policy directly (mapping states → actions) | Value function (expected reward of states/actions) |
| Action selection | Sampled directly from policy | Derived from value function (e.g., pick max-value action) |
| Action space | Works well with continuous action spaces | Better suited for discrete action spaces |
| Output | Probability distribution over actions | Estimated value (Q-value) |
| Examples | REINFORCE, PPO, A2C (actor part) | Q-Learning, DQN, SARSA |
| Convergence | Can converge to local optima, higher variance | More stable but can be less flexible |
| Exploration | Naturally stochastic (built-in exploration) | Needs explicit strategy (e.g., $\varepsilon$-greedy) |

### 8.25 What is Q-Learning?

- Model-free, value-based reinforcement learning algorithm
- Learns a Q-value function $Q(s, a)$ representing expected cumulative reward for taking action $a$ in state $s$
- **Update rule**: $Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]$
  - $\alpha$ = learning rate, $\gamma$ = discount factor, $r$ = reward
- Off-policy — learns optimal policy regardless of agent's actions during exploration
- Uses a Q-table (or neural network in Deep Q-Learning/DQN) to store/approximate values
- Agent picks action with highest Q-value (often via $\varepsilon$-greedy for exploration)

### 8.26 Explain the concept of exploration vs. exploitation.

The **exploration-exploitation trade-off** is the dilemma of any learner that makes decisions under uncertainty: **exploit** what currently looks best to collect reward now, or **explore** less-known options to learn something that may pay off later. Too much exploitation locks in a suboptimal choice; too much exploration wastes reward.

**Where it appears**

- Reinforcement learning agents choosing actions (see 16.7).
- **Multi-armed bandits:** choosing which ad, headline, price or recommendation to show (see 16.74).
- Recommender systems, where showing only known favourites creates feedback loops and starves new items (see 32.19).
- Hyperparameter search and Bayesian optimisation (acquisition functions balance the two).

**Common strategies**

| Strategy | Idea | Notes |
| --- | --- | --- |
| Epsilon-greedy | With probability $\varepsilon$ pick a random action, otherwise the best | Simple; usually decay $\varepsilon$ over time |
| Softmax (Boltzmann) | Choose actions with probability increasing in their estimated value | Temperature controls exploration |
| Upper confidence bound (UCB) | Pick the action with the highest optimistic estimate: value plus an uncertainty bonus | Explores where uncertainty is high; strong guarantees |
| Thompson sampling | Sample a value for each action from its posterior and pick the best sample | Excellent in practice; natural for Bayesian models |
| Intrinsic motivation | Reward novelty or prediction error | For sparse-reward RL problems |

**Practical considerations**

- **Cost of exploration:** in production, exploring means showing users worse options; limit it to a small share of traffic or to low-risk decisions.
- **Non-stationarity:** when the environment changes, keep exploring a little forever, or use sliding windows.
- **Delayed rewards:** exploration decisions may be judged only days later, which slows learning.
- **Logging:** record the probability with which each action was chosen, so data from exploration can be used for off-policy evaluation later.

### 8.27 What is the curse of dimensionality, and how do you address it?

The curse of dimensionality is the set of problems that appear as the number of features grows: the space grows exponentially, so data becomes sparse, distances stop discriminating between near and far points, and models need far more data to avoid fitting noise.

**What goes wrong:**

- **Data becomes sparse:** to keep the same density in 10 dimensions as 10 points give in 1 dimension, you need $10^{10}$ points. Any fixed dataset covers a high-dimensional space ever more thinly.
- **Volume moves to the edges:** in a unit hypercube, the fraction of volume within 0.05 of the boundary is $1 - 0.9^d$, which is about 65 percent at $d = 10$ and over 99 percent at $d = 50$. "Typical" points are all near the corners.
- **Distances concentrate:** for many distributions the ratio of the farthest to the nearest neighbour distance tends to 1 as dimension grows, which breaks k-NN, k-means, kernel methods with Euclidean distance and density estimation.
- **Overfitting risk rises:** with many features and few samples ($p \gg n$), a model can always find spurious patterns that look predictive.
- **Compute and memory cost** grow, and visual inspection becomes impossible.

**How to address it:**

| Approach | Examples | Notes |
| --- | --- | --- |
| Feature selection | Filter methods (mutual information), wrapper methods, L1 regularisation | Keeps features interpretable |
| Dimensionality reduction | PCA, truncated SVD, autoencoders; UMAP and t-SNE for visualisation only | Loses direct interpretability |
| Regularisation and simpler models | Ridge, lasso, shallow trees | Controls variance when $p$ is large |
| Learned embeddings | Replace huge one-hot or sparse vectors with dense embeddings | Standard for categorical IDs and text |
| Better distance measures | Cosine similarity on normalised embeddings, learned metrics | Helps retrieval and clustering |
| More data or domain knowledge | Engineer fewer, more informative features | Often the most effective |

**Nuance a senior candidate should add:** real data usually lies near a low-dimensional manifold. Natural images have millions of pixels but far fewer degrees of freedom, which is why deep networks work on them. The curse bites hardest for methods that rely on raw Euclidean distance over many uninformative features. Tree ensembles and regularised linear models cope well with many irrelevant features; k-NN does not.

**Practical checks:** compare the number of features with the number of examples, look at the explained-variance curve of PCA to estimate intrinsic dimension, and test whether removing feature groups hurts validation performance.

### 8.28 How do you combine losses in multi-task learning? Explain per-task losses, focal loss and gradient blending.

In multi-task (or multi-modal) learning, one network optimises several objectives at once, and the main difficulty is balancing them: tasks have different loss scales, difficulty, noise and learning speed, so a naive sum lets one task dominate. Per-task losses, loss weighting schemes, focal loss and gradient blending are tools for that balancing.

**Per-task ("local") losses.** Each task head computes its own loss, and the shared trunk receives a weighted sum:

$$
L_{\text{total}} = \sum_{k=1}^{K} w_k \thinspace L_k
$$

"Local loss" is not a standard term; it usually means these per-head or per-branch losses, or auxiliary losses attached to intermediate layers (deep supervision) that give earlier layers a direct training signal. Keeping per-task losses separate also lets you monitor each task independently.

**Choosing the weights** is the real problem:

| Method | Idea |
| --- | --- |
| Manual or grid-searched weights | Simple; normalise losses to similar scales first |
| Uncertainty weighting (Kendall, Gal and Cipolla, 2018) | Learn a noise parameter per task; noisier tasks get lower weight automatically |
| GradNorm (Chen et al., 2018) | Adjust weights so each task's gradient norm on shared layers trains at a similar rate |
| Dynamic weight averaging | Weight tasks by how slowly their loss is falling |
| PCGrad and other gradient-surgery methods (Yu et al., 2020) | When two task gradients conflict, project one onto the normal plane of the other |

**Focal loss (Lin et al., 2017, RetinaNet).** A modified cross-entropy that down-weights easy, well-classified examples so training focuses on hard ones. With $p_t$ the predicted probability of the true class:

$$
\text{FL}(p_t) = -\alpha_t \thinspace (1 - p_t)^{\gamma} \log(p_t)
$$

- $\gamma = 0$ recovers weighted cross-entropy; $\gamma = 2$ is the common default.
- Designed for extreme foreground-background imbalance in dense object detection; used in multi-task settings when some heads have heavily imbalanced labels.
- It distorts probabilities, so recalibrate if you need calibrated outputs.

**Gradient blending (Wang, Tran and Feiszli, 2020, Facebook AI).** In multi-modal networks (for example audio plus video), the joint model often performs worse than the best single modality, because different modalities overfit at different rates. Gradient blending attaches a classification head to each modality and to the fused representation, estimates each head's overfitting-to-generalisation ratio on held-out data, and sets the loss weights to favour heads that generalise rather than overfit.

**Practical guidance:**

1. Start with normalised losses and equal weights; log each task's loss and metric separately.
2. If one task degrades when others are added (negative transfer), try uncertainty weighting or GradNorm, then gradient surgery, and finally separate trunks for unrelated tasks.
3. Use focal loss or class weights per head for imbalanced tasks.
4. Choose checkpoints by a weighted business metric across tasks, not by the total loss.

**What interviewers look for:** recognising that multi-task gains depend on task relatedness, that loss scale and gradient conflict are the failure modes, and that per-task monitoring is essential.

### 8.29 Explain Contrastive Learning

**Contrastive learning** learns representations by comparison: an encoder is trained so that **related pairs** (two views of the same image, an image and its caption, a question and its answer) have similar embeddings, while **unrelated pairs** have dissimilar ones. It is the main recipe behind modern self-supervised pretraining and embedding models.

**How it works**

1. Build **positive pairs:** two augmentations of one image (crop, colour change), a caption and its image, a query and a relevant passage.
2. Treat other items in the batch as **negatives**.
3. Encode everything, compute similarities, and train with a loss that makes each positive pair score higher than the negatives, usually **InfoNCE** (a softmax cross-entropy over similarities with a temperature, see 19.23).

**Key design choices**

- **Augmentations** decide what the model learns to ignore; they encode the notion of "sameness".
- **Negatives:** more and harder negatives help; methods use large batches (SimCLR), a queue of past embeddings (MoCo), or hard-negative mining. Some methods avoid explicit negatives altogether (BYOL, DINO, which use a teacher network as the target).
- **Temperature** controls how sharply the loss focuses on hard negatives.
- **False negatives:** items in the batch that are actually related get pushed apart, which hurts; deduplication and careful sampling help.

**Well-known examples**

- **SimCLR, MoCo:** self-supervised image representations.
- **CLIP:** aligns images and text from web image-caption pairs, enabling zero-shot classification and cross-modal search (see 15.28).
- **Sentence and retrieval embedding models:** trained contrastively on query-document pairs.

**Why it matters in practice:** contrastive pretraining produces general-purpose embeddings from unlabelled or weakly labelled data, so downstream tasks need far fewer labels. Chapter 15 (15.41) covers self-supervised learning for vision in more depth.

### 8.30 What is Generative AI?

**Generative AI** refers to models that **create new content** (text, images, audio, video, code, molecules) by learning the distribution of their training data and sampling from it, as opposed to discriminative models that predict a label or value for a given input (see 8.57 and 13.17).

**Main model families**

| Family | Idea | Typical use |
| --- | --- | --- |
| Autoregressive transformers | Predict the next token given previous ones | Text, code, increasingly images and audio (LLMs) |
| Diffusion models | Learn to reverse a gradual noising process | Images, video, audio |
| GANs | A generator and a discriminator trained against each other | Fast image synthesis; largely replaced by diffusion for quality |
| VAEs and normalising flows | Learn a latent space with explicit probabilistic structure | Representation learning, components of larger systems |

**What changed with modern generative AI**

- **Foundation models:** one large pretrained model adapts to many tasks through prompting, retrieval and fine-tuning, instead of training a model per task.
- **Instruction tuning and alignment** made models follow natural-language requests.
- **Multimodality:** models that read and generate across text, images and audio.

**Where it creates value**

- Drafting and summarising, coding assistance, customer support, search and question answering over company knowledge, document processing, content generation, synthetic data.

**Limitations a senior practitioner should name**

- **Hallucination:** fluent but false output; mitigated with retrieval, grounding and verification.
- **Non-determinism and evaluation difficulty:** quality is hard to measure automatically (see Chapter 24).
- **Cost and latency** of inference at scale (see Chapter 27).
- **Security and safety:** prompt injection, data leakage, harmful content (see Chapter 30).
- **Intellectual property and privacy** questions about training data and outputs.

Part IV of this book (Chapters 17 to 24) covers large language models and generative AI engineering in depth.

### 8.31 What are the training, validation and test sets, and how should you split data?

The training set is used to fit model parameters, the validation set to choose hyperparameters, features and models, and the test set to estimate final performance on unseen data exactly once. Keeping these roles separate is what makes an offline score a trustworthy estimate of production performance.

| | Training set | Validation set | Test set |
| --- | --- | --- | --- |
| Purpose | Learn parameters | Tune and select | Final, unbiased estimate |
| Seen during training? | Yes | Indirectly, through decisions | No |
| How often used | Every step | Many times | Once, at the end |
| Main risk | Overfitting | Overfitting to the validation set through repeated tuning | Leakage or reuse, which makes it optimistic |

**How to split:**

- **Typical ratios:** 60/20/20 or 70/15/15 for modest data; with millions of examples, 98/1/1 can be enough because 1 percent is already a large, stable test set. What matters is that the validation and test sets are large enough to detect the differences you care about.
- **With little data:** use k-fold cross-validation on the training portion instead of a fixed validation set, and still hold out a test set (see 8.20).
- **Match production:** split by time for anything temporal (train on the past, test on the future), by group for repeated entities (a customer, patient or document should appear in one split only), and stratify by class for imbalanced labels.
- **Make the validation and test sets come from the distribution you will serve.** If production traffic differs from your historical training data, build test data from recent or production-like samples.

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=customer_id))
```

**Pitfalls:**

- Fitting scalers, encoders or feature selection on all the data before splitting (see 8.39).
- Duplicates or near-duplicates across splits.
- Repeatedly "peeking" at the test set; once it guides decisions it becomes a second validation set.
- Using a random split on time series, which lets the model see the future.

See 10.11 for the mechanics of splitting and 12.23 for time-series validation.

### 8.32 Differentiate between Deep Learning and Machine Learning?

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

### 8.33 What is semi-supervised learning, and where is it applied?

Semi-supervised learning trains a model on a small amount of labelled data together with a large amount of unlabelled data, using the structure of the unlabelled data to improve on what the labels alone could achieve. It is valuable when labels are expensive (they need experts or time) but raw data is plentiful.

**Example:** 10,000 X-ray images, of which only 500 are labelled by radiologists. The model learns from the 500 and uses the other 9,500 to learn what normal variation looks like.

| Learning type | Labelled data | Unlabelled data |
| --- | --- | --- |
| Supervised | Large amount | Not used |
| Unsupervised | None | Large amount |
| Semi-supervised | Small amount | Large amount |

**Assumptions that make it work** (if they fail, unlabelled data can hurt):

- **Smoothness:** points close together in input space tend to share a label.
- **Cluster assumption:** the decision boundary should pass through low-density regions, between clusters.
- **Manifold assumption:** data lies on a low-dimensional manifold, which unlabelled data helps map out.

**Common techniques:**

- **Self-training and pseudo-labelling:** train on labelled data, predict on unlabelled data, add high-confidence predictions as labels, retrain.
- **Consistency regularisation:** the model should give the same prediction for different augmentations of an unlabelled input (Mean Teacher, FixMatch, which combines both ideas).
- **Graph-based label propagation:** spread labels along a similarity graph.
- **Self-supervised pretraining then fine-tuning:** pretrain on all unlabelled data (contrastive or masked modelling), then fine-tune on the labelled subset. This is now often the strongest practical approach.

**Applications:**

1. **Medical imaging:** X-rays, MRI and CT with limited expert labels.
2. **Speech recognition:** small transcribed sets with large untranscribed audio.
3. **Text classification and NLP:** sentiment, intent and document categorisation with few labelled examples.
4. **Fraud and cybersecurity:** few confirmed cases among many unlabelled transactions or network events.
5. **Web and content classification**, and **image classification** over large partially labelled collections.

**Risks and good practice:**

- **Confirmation bias:** wrong pseudo-labels reinforce themselves. Use confidence thresholds, class-balanced selection and keep a clean labelled validation set.
- **Distribution mismatch** between labelled and unlabelled data (for example unlabelled data containing new classes) degrades results.
- Always compare against a supervised baseline trained on the labelled data only; and consider whether labelling more data (with active learning, see 8.43) is cheaper than the added complexity.

### 8.34 What is Inductive Logic Programming?

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

### 8.35 How would you differentiate data mining from Machine learning?

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

### 8.36 What do you mean by the unreasonable effectiveness of data?

"**The Unreasonable Effectiveness of Data**" is a 2009 paper by Halevy, Norvig and Pereira (Google), echoing Wigner's "unreasonable effectiveness of mathematics". The thesis: for many real-world problems, **simple models trained on very large datasets outperform sophisticated models trained on less data**.

**Key ideas**
- Evidence from Banko & Brill (2001): on a word-disambiguation task, simple algorithms' accuracy kept rising with more data, and the ranking of algorithms changed — data volume mattered more than algorithm choice.
- Web-scale text (n-gram models for translation, spelling correction, search) captured the messy, long-tailed reality of language better than hand-crafted rules or elaborate grammars.
- "Follow the data": instead of building complex theories of the domain, memorize and generalize from huge amounts of examples.

**Modern relevance**
- It anticipated **scaling laws** and LLMs: performance improves predictably with more data and compute.
- Qualifications today: data **quality** and deduplication matter as much as quantity (Chinchilla, Phi); diminishing returns follow power laws; labeled data is expensive; and more data can amplify bias.

**Practical takeaway:** before spending weeks tuning a model, ask whether acquiring, labeling or cleaning more representative data would yield a bigger gain — it often does. Learning curves tell you which regime you are in (see 8.60).

### 8.37 What is the No Free Lunch theorem, and what does inductive bias mean in practice?

The No Free Lunch (NFL) theorem (Wolpert, 1996, for supervised learning; Wolpert and Macready, 1997, for optimisation) states that averaged over all possible problems (all possible data-generating functions), every learning algorithm has the same expected performance. No algorithm is universally best: any gain on one class of problems is paid for by losses on another. An algorithm does well only when its **inductive bias**, the assumptions it brings beyond the data, matches the structure of the real problem.

**Inductive bias** is everything that makes a learner prefer one hypothesis over another that fits the training data equally well. Without it, learning from a finite sample is impossible.

| Model | Inductive bias | Works well when |
| --- | --- | --- |
| Linear and logistic regression | Effects are additive and linear in the features | Relationships are roughly linear, or features are well engineered |
| Decision trees and gradient boosting | Axis-aligned splits, interactions via hierarchy | Tabular data with heterogeneous features and thresholds |
| k-NN, kernel methods | Smoothness: similar inputs have similar outputs | Low-dimensional data with a meaningful distance |
| CNNs | Locality and translation equivariance | Images, audio spectrograms |
| RNNs and state space models | Sequential order, recency | Time series, streams |
| Transformers | Weak bias; relations learned via attention | Very large data, where weak bias pays off |
| Graph neural networks | Permutation invariance over neighbours | Relational and network data |
| Regularisation, priors | Preference for small or sparse weights | Noisy, limited data |

**What NFL means in practice:**

- Real-world problems are not uniformly distributed over "all possible functions": they have structure, so some algorithms are reliably good in particular domains (gradient boosting on tabular data, Transformers on language).
- You must **compare candidate models empirically** on your data with proper validation rather than assuming a favourite always wins.
- **Encoding the right bias is a lever:** data augmentation, architecture choice, monotonic constraints, feature engineering and pretrained representations all inject prior knowledge. The less data you have, the more the right inductive bias matters; with huge data, weaker biases (Transformers) can catch up and surpass stronger ones.

**Interview framing:** "NFL is why I start with a baseline and a few model families suited to the data type, evaluate them with cross-validation on the metric that matters, and pick based on evidence, while also weighing interpretability, latency and maintenance cost."

**Common misreading:** NFL does not say all algorithms are equally good on your problem, only when averaged over all conceivable problems, most of which look like random noise.

### 8.38 What are the main differences between Structured and Unstructured Data?

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

### 8.39 What is data leakage, which forms are hard to detect, and how do you prevent it?

**Data leakage** happens when information that would not be available at prediction time gets into training or model selection. The result is optimistic offline metrics that collapse in production. It is one of the most common reasons a model "works in the notebook" and fails when deployed.

**Common forms:**

- **Target leakage:** features derived from the label or recorded after the outcome (for example `refund_issued` when predicting fraud).
- **Train-test contamination:** fitting scalers, imputers, encoders or feature selection on the full dataset before splitting.
- **Temporal leakage:** random splits on time-dependent data; features built with future windows.
- **Group leakage:** the same patient, user or document appears in both train and test.
- **Duplicates and near-duplicates** across splits.

**Examples that are difficult to detect:**

1. **Post-outcome features:** predicting loan default with `days_overdue`, which only takes non-zero values once the customer is already defaulting.
2. **Aggregates computed over the whole period:** predicting January behaviour with `customer_total_transactions` calculated over the whole year.
3. **Process-generated features:** predicting fraud with `fraud_investigation_status`, which exists because the transaction was already suspected.
4. **Feature windows that cross the label time:** `average_support_tickets_before_churn` that accidentally includes tickets raised after the churn date.
5. **Preprocessing before splitting:**

   ```python
   scaler.fit(X)                       # learns test-set statistics
   X = scaler.transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   ```

6. **Imputation or target encoding** computed on the full dataset rather than inside each training fold.
7. **Entity leakage:** splitting transactions randomly instead of by customer, so the model learns the customer rather than the pattern.
8. **Near-duplicates with different IDs:** the same image resized, the same document re-scanned, the same patient re-registered.
9. **Model-selection leakage:** repeatedly tuning against the test set, so it gradually influences decisions even though it is never trained on.
10. **Upstream leakage in data pipelines:** tables that are overwritten with corrected values, so the training snapshot contains information that was not available at the original decision time.

**How to prevent it:**

1. **Split first**, then do all preprocessing inside a `Pipeline` so it is fitted only on training folds.
2. Use **time-based splits** (`TimeSeriesSplit`, rolling origin) and **point-in-time correct** feature joins (feature stores with as-of joins).
3. Use **group-aware cross-validation** (`GroupKFold`) for entity-level data.
4. Audit features: for each one ask "would I know this, with this value, at prediction time?" Treat suspiciously high feature importance or near-perfect AUC as a red flag (see 12.33).
5. Deduplicate, including near-duplicates, before splitting.
6. Keep a truly untouched holdout set and validate with a shadow deployment before trusting offline numbers.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(pipe, X, y, cv=TimeSeriesSplit(n_splits=5))   # scaler refitted per fold
```

See 10.30 for leakage specific to feature engineering.

### 8.40 What are the different types of bias in Machine Learning?

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

### 8.41 Why does L1 regularisation give sparse coefficients while L2 does not? Explain geometrically.

L1 (lasso) produces exact zeros; L2 (ridge) only shrinks coefficients towards zero. (A common trick question asks why L2 gives sparse coefficients: the premise is wrong.) The reason can be seen three ways: from the gradient, from the geometry of the constraint region and from the Bayesian prior.

**Gradient view:**

- L2 penalty $\lambda \sum_j w_j^2$ has gradient $2\lambda w_j$, which shrinks in proportion to the weight. As $w_j$ gets small, the push gets small too, so weights become tiny but almost never exactly zero.
- L1 penalty $\lambda \sum_j \lvert w_j \rvert$ has a constant-magnitude subgradient $\lambda \thinspace \text{sign}(w_j)$, so it keeps pushing small weights at the same rate until they reach exactly zero. For an orthonormal design, the lasso solution is the soft-thresholding operator, which sets every coefficient whose least-squares value is below $\lambda$ in magnitude to zero:

$$
w_j = \text{sign}(z_j)\max(\lvert z_j \rvert - \lambda, 0)
$$

**Geometric view.** The penalised problem is equivalent to minimising the loss subject to a budget on the weights, $\lVert w \rVert_1 \le t$ or $\lVert w \rVert_2^2 \le t$. Picture the elliptical contours of the loss growing out from the unregularised optimum until they first touch the constraint region.

For two weights, the L1 region $\lvert w_1 \rvert + \lvert w_2 \rvert \le t$ is a diamond with corners on the axes:

```text
          w2
           |
          /\
         /  \
        /    \
-------+------+------- w1
        \    /
         \  /
          \/
```

The expanding ellipse usually touches the diamond at a corner, where one coordinate is exactly zero. In many dimensions the L1 ball has many corners and edges lying in coordinate subspaces, so many weights end up at zero.

The L2 region $w_1^2 + w_2^2 \le t$ is a circle:

```text
          w2
           |
        .-----.
      .'       '.
     /           \
----+-------------+---- w1
     \           /
      '.       .'
        '-----'
```

It has no corners, so the contact point is generically off the axes: weights are small but non-zero.

**Bayesian view:** L1 corresponds to a Laplace prior, sharply peaked at zero; L2 to a Gaussian prior, smooth at zero. The MAP estimate under a Laplace prior often sits exactly at the peak.

**Memory aid:** L1 = diamond = sparse; L2 = circle = small weights.

**Practical use:**

- **L1** for feature selection and interpretability with many candidate features. It is unstable with correlated features: it picks one of a correlated group somewhat arbitrarily.
- **L2** for stability under multicollinearity and when most features carry some signal.
- **Elastic net** for both: sparse, while keeping groups of correlated features together.
- Sparsity also needs scaled features; otherwise the penalty zeros out features measured in large units first.

### 8.42 List some ways using which you can improve a model’s performance.

Work systematically: diagnose first, then pick the lever.

**1. Diagnose:** compare train vs validation error. A high-bias model underfits; a high-variance model overfits. Do error analysis on misclassified examples and slices. Learning curves show whether more data would help (see 8.60).

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

### 8.43 What is active learning, and where is it useful?

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

### 8.44 What types of models tend to overfit?

Models with **high capacity and few built-in constraints** relative to the amount of data overfit most easily.

**Prone to overfitting:**
- **Deep, unpruned decision trees:** can create one leaf per sample. The classic high-variance model.
- **KNN with small k** (k = 1 memorizes the training set).
- **Kernel SVM with large $\gamma$ and large $C$.**
- **High-degree polynomial regression** and linear models with many features relative to samples ($p \gg n$) and no regularization.
- **Large neural networks** trained long on small datasets without dropout, weight decay, augmentation or early stopping.
- **Gradient boosting** with too many rounds, deep trees or a high learning rate. Boosting keeps fitting residuals, including the noise.
- **Non-parametric and flexible models** in general (Gaussian processes with short length scales, splines with many knots).
- Complex ensembles like stacking, if the meta-learner is fit on in-sample predictions.

**Less prone:** regularized linear models, Naive Bayes, shallow trees, random forests (bagging averages away variance), large-k KNN. These tend to underfit instead.

**Aggravating factors (model-independent):** small or noisy datasets, too many features, data leakage, heavy hyperparameter tuning on one validation set, and training on non-representative data.

**Controls:** regularization, pruning, early stopping, dropout, cross-validation, more data or augmentation, feature selection and ensembling. The bias–variance tradeoff says to match model capacity to data size and noise.

### 8.45 Can you tell the complete life cycle of a data science project?

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

### 8.46 What are the properties of a good ML model?

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

### 8.47 Given a dataset, how do you decide which machine learning algorithm to use?

There is no universally best algorithm (see 8.37), so the choice is driven by the problem, the data and the constraints, then confirmed empirically against a baseline.

**1. Define the problem:** supervised (classification, regression, ranking), unsupervised (clustering, anomaly detection) or sequential decision-making (reinforcement learning or bandits). Which metric matters, and what does each kind of error cost?

**2. Look at the data:**

- **Size:** a few hundred rows favours simple, regularised models (logistic regression, Naive Bayes, small trees, SVMs); hundreds of thousands or more allows gradient boosting and neural networks.
- **Type:** tabular data, use gradient boosting (XGBoost, LightGBM, CatBoost) as the usual strong default; text, a TF-IDF plus linear baseline and then fine-tuned Transformers; images and audio, CNNs or vision Transformers via transfer learning; time series, seasonal naive baselines, then ARIMA or boosting with lag features.
- **Structure:** is the boundary likely linear? Dimensionality, sparsity, missing values, high-cardinality categoricals, label noise, class imbalance. Trees handle missing values and mixed types well; k-NN suffers in high dimensions; Naive Bayes is robust with tiny data.

**3. Constraints:**

- **Interpretability or regulation:** logistic regression, shallow trees, or boosting with monotonic constraints and SHAP explanations.
- **Latency and memory at inference:** edge or high-QPS services favour small linear or tree models.
- **Calibrated probabilities:** logistic regression is naturally well calibrated; others may need Platt or isotonic calibration.
- **Training budget and retraining frequency**, and the team's ability to maintain the model.

**4. Empirical process:**

1. Start with a trivial baseline (majority class, mean, current rule) and a simple model.
2. Shortlist two to four model families and evaluate them with cross-validation that mirrors production (stratified, grouped or time-based) on the business metric, for example PR-AUC for rare fraud.
3. Tune the best one or two; inspect errors per segment; check calibration and latency.
4. Pick the simplest model within noise of the best, and confirm on the untouched test set.

| Situation | Sensible first choices |
| --- | --- |
| Small tabular data, explanation needed | Regularised logistic or linear regression |
| Medium to large tabular data | LightGBM, XGBoost or CatBoost |
| Text classification | TF-IDF plus logistic regression, then a fine-tuned encoder or LLM |
| Images, audio | Pretrained CNN or ViT, fine-tuned |
| Unlabelled data, find groups | k-means or GMM, HDBSCAN for irregular shapes |
| Rare-event detection without labels | Isolation Forest, autoencoders |

**Examples:**

- Churn prediction on 200k rows of tabular CRM data with a requirement to explain decisions: logistic regression baseline, then LightGBM with SHAP; pick LightGBM only if the uplift justifies the extra complexity.
- Loan default with 50k rows, mixed features and regulatory explainability: logistic regression versus LightGBM with monotonic constraints and SHAP.

**What interviewers look for:** a process, not a favourite algorithm: baseline first, a few candidates chosen for the data type, validation that mirrors production and a decision that weighs accuracy against cost, latency and explainability.

### 8.48 Is more data always better?

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

**How to decide in practice:** plot a **learning curve** (validation score vs. training-set size). If it's still rising, collecting data is worthwhile; if it has flattened, invest in features, model changes or label quality instead (see 8.60 and 8.59). Targeted collection via active learning or error analysis usually beats collecting more of the same.

### 8.49 How would you differentiate between multi-label and multi-class classification?

**Multi-class:** each sample belongs to **exactly one** of $K \gt 2$ mutually exclusive classes.
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

### 8.50 What is an AI ontology?

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

### 8.51 Can AI systems be conscious?

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

### 8.52 What is unsupervised learning through compression?

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

### 8.53 Why can a model with lower training loss have worse test performance?

A lower training loss does not guarantee better test performance, because **training loss measures fit to the training data**, while **test performance measures generalization to unseen data**.

#### Intuition

Suppose two models:

| Model | Training loss | Test loss |
| --- | --- | --- |
| Model A | 0.05 | 0.20 |
| Model B | 0.15 | 0.12 |

Model A fits the training data better, but its test performance is worse because it has **overfit**.

#### Mathematically

Training minimizes empirical risk:

$$
\hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n} L(f(x_i), y_i)
$$

But what we actually care about is the expected/test risk:

$$
R(f) = \mathbb{E}_{(x,y)\sim P}\left[L(f(x), y)\right]
$$

A model can have:

$$
\hat{R}(f_A) < \hat{R}(f_B)
$$

while:

$$
R(f_A) > R(f_B)
$$

because $f_A$ has learned **training-specific patterns/noise** rather than general patterns.

#### Why this happens

- **Overfitting:** Model memorizes training examples.
- **High variance:** Small changes in data can significantly change the model.
- **Noise fitting:** Model learns random patterns in labels/features.
- **Distribution differences:** Test data may differ from training data.
- **Insufficient regularization:** Model has too much freedom to fit the training set.

#### Key idea

> **Lower training loss ≠ better generalization.**

### 8.54 How do you detect overfitting beyond simply comparing train and validation accuracy?

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

### 8.55 What is the difference between model capacity and model complexity?

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

$$
y = w_0 + w_1 x + w_2 x^2 + \dots + w_{10} x^{10}
$$

A degree-10 polynomial has **higher capacity** than a degree-2 polynomial because it can represent more functions.

But if the learned degree-10 polynomial happens to have small higher-order coefficients:

$$
w_7 \approx 0,\quad w_8 \approx 0,\quad w_9 \approx 0,\quad w_{10} \approx 0
$$

the **effective complexity of the learned solution** may still be low.

### 8.56 Why does regularization reduce variance?

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

### 8.57 What is the difference between generative and discriminative models?

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

$$
P(X \mid Y)\thinspace P(Y)
$$

and then use Bayes' rule:

$$
P(Y \mid X) = \frac{P(X \mid Y)\thinspace P(Y)}{P(X)}
$$

**Example:** Naive Bayes learns what spam and non-spam emails look like, then estimates whether a new email is spam.

Generative models can potentially **create new samples** similar to the training data.

---

#### 2. Discriminative model

A discriminative model directly learns the relationship between input and target:

$$
P(Y \mid X)
$$

or directly learns a decision boundary:

$$
f(X) \rightarrow Y
$$

**Example:** Logistic regression learns:

$$
P(\text{spam} \mid \text{email})
$$

without explicitly modeling how spam emails themselves are generated.

### 8.58 How do you frame a business problem as a machine learning problem, and when should you not use machine learning?

Framing means translating a business goal into a precise prediction or decision task: what is predicted, for whom, when, with what data, how the prediction changes an action and how success is measured. Most failed ML projects fail here, not in modelling, so a senior engineer starts by asking whether ML is needed at all.

**A framing checklist:**

1. **Decision first:** what action will change because of the prediction, who takes it and how often? "Predict churn" is not a framing; "each Monday, rank active subscribers by 60-day churn risk so the retention team can call the top 500" is.
2. **Target definition:** the exact label, observation window and prediction horizon. What counts as churn: cancelled, inactive for 30 days or downgraded? When is the label known?
3. **Prediction point and available features:** what is known at the moment of prediction? This defines the point-in-time data you may use and prevents leakage (see 8.39).
4. **Task type:** classification, regression, ranking, forecasting, anomaly detection, recommendation, or a bandit or RL problem when actions influence future data.
5. **Success metrics at three levels:** business KPI (retained revenue), ML metric aligned to it (precision in the top 500), and guardrails (latency, fairness, cost per prediction, complaint rate).
6. **Baseline:** the current process or a simple rule. The model must beat it by enough to pay for itself.
7. **Value of being right and cost of being wrong:** the cost of a false positive versus a false negative sets the threshold and tells you whether the economics work.
8. **Constraints:** latency, batch versus real time, explainability, regulation, privacy, retraining cadence.
9. **Feedback loop:** how will labels arrive after deployment, and will the model's own actions bias them (treated customers stop churning, so future labels change)?

```text
Business goal  ->  Decision/action  ->  Prediction target + horizon  ->  Data at prediction time
      |                                                                       |
      +-- KPI and guardrails  <--  Offline metric  <--  Baseline  <--  Model  +
```

**When not to use machine learning:**

| Situation | Better alternative |
| --- | --- |
| The logic is simple, stable and known | Rules, SQL or a lookup table |
| A heuristic gets 90 percent of the value | Ship the heuristic; revisit later |
| Errors are unacceptable and must be guaranteed | Deterministic logic, formal verification, human decision |
| No representative data or labels, and no way to get them | Collect data first, or use rules with logging |
| Nobody will act on the prediction | Fix the process; a model adds nothing |
| Full explainability is legally required and a simple model is not good enough | Redesign the decision or keep a human in the loop |
| The environment changes faster than you can retrain | Rules owned by domain experts, or online learning with strong safeguards |
| The cost of building and running the system exceeds the value | Do nothing, or buy a product |

**A quick value estimate** often settles it: number of decisions per year times uplift per decision times the share that changes because of the model, minus build, run and maintenance cost.

**Rules of thumb:**

- Start with the simplest thing that could work and log everything, so you collect the data a model would need later.
- If you cannot describe how the prediction will be used in one sentence, the problem is not yet framed.
- Agree on the success criteria and the go/no-go threshold with stakeholders before modelling, not after seeing the results.

**What interviewers look for:** questions before solutions, a clear link from decision to label to metric, attention to leakage and feedback loops, and the confidence to recommend not using ML when it does not pay.

### 8.59 What is the difference between data-centric and model-centric improvement, and when should you focus on each?

Model-centric improvement keeps the data fixed and iterates on the algorithm, architecture, hyperparameters and training procedure. Data-centric improvement keeps the model fairly fixed and systematically improves the data: labels, coverage, consistency and features. On most real projects, once a reasonable model family is in place, data work delivers larger and more durable gains, which is the point of the "data-centric AI" movement popularised by Andrew Ng.

| | Model-centric | Data-centric |
| --- | --- | --- |
| What changes | Architecture, loss, hyperparameters, ensembles | Labels, label guidelines, coverage of rare cases, data cleaning, augmentation, features |
| Typical tools | Hyperparameter search, new architectures, distillation | Error analysis, label audits, confident learning, active learning, slice-based evaluation, data validation |
| Gains | Often small once a strong model family is used | Often large, especially for noisy or narrow datasets |
| Where it shines | Benchmarks with fixed, clean data; large-scale pretraining research | Industrial applications with small, messy or shifting data |
| Durability | Gains can disappear when data changes | Better data helps every future model |

**Signals that the data is the bottleneck:**

- Different strong model families plateau at the same error.
- Error analysis shows mistakes clustered on specific slices (a region, device type, rare class) or on ambiguous examples.
- Annotators disagree with each other at rates close to the model's error rate.
- Many "errors" turn out to be wrong labels on inspection.
- Learning curves are still rising, but only for some slices (see 8.60).

**Signals that the model is the bottleneck:**

- Training error is high even on clean, unambiguous examples (high bias).
- A clearly better inductive bias exists for the data type (pretrained encoders for text, CNNs or ViTs for images).
- The model ignores obviously useful signals because of its form (a linear model with strong interactions).

**A data-centric workflow:**

1. Fix the evaluation set first: clean, representative, with slices that matter to the business.
2. Run error analysis: sample errors, categorise them (label noise, ambiguity, missing coverage, missing feature) and count each category.
3. Fix the largest category: rewrite label guidelines, relabel, collect targeted examples, add augmentation or features.
4. Retrain the same model and measure, then repeat.

```python
# find likely label errors with out-of-fold predicted probabilities
import numpy as np
from sklearn.model_selection import cross_val_predict

proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")
self_confidence = proba[np.arange(len(y)), y]          # probability of the given label
suspects = np.argsort(self_confidence)[:200]           # send to human review
```

Libraries such as cleanlab implement this idea (confident learning) more rigorously.

**For a data or AI leader:** data-centric work needs investment in labelling tools, guidelines, annotator training, data versioning and quality monitoring, which are organisational capabilities rather than individual model tweaks. The same principle drives LLM work today: curated, deduplicated, high-quality training and fine-tuning data routinely beats more data or bigger models (see 8.36).

**Balanced answer:** both matter. Use a strong standard model to get to a reasonable baseline quickly, then let error analysis decide whether the next week is better spent on the data or on the model.

### 8.60 How do you use learning curves to diagnose whether more data, more capacity or better features will help?

A learning curve plots training and validation performance against the size of the training set (sometimes also against training time). Its shape tells you whether the model is limited by variance, in which case more data helps, or by bias, in which case more data will not help and you need more capacity or better features. It is the cheapest way to answer "should we pay to label more data?".

**How to read it:**

| Pattern | Diagnosis | What helps |
| --- | --- | --- |
| Large gap between training and validation; validation still improving as data grows | High variance | More data, augmentation, regularisation, simpler model |
| Both curves converge quickly at a poor level, small gap | High bias | More capacity, better features, less regularisation; more data will not help |
| Both curves converge at a good level | Well fitted | Stop or focus on cost and robustness |
| Validation curve noisy or non-monotonic | Small or unrepresentative validation set, or leakage | Larger validation set, repeated cross-validation, check splits |
| Training score falls as data grows, validation flat | Model capacity saturated | Bigger model or better features |

```text
 error
   |  validation \
   |              \______________   <- plateau: more data will not help
   |                ______________   <- gap = variance
   |  training  ___/
   +-------------------------------- training set size
```

**Computing one in Python:**

```python
import numpy as np
from sklearn.model_selection import learning_curve, StratifiedKFold

sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
    scoring="roc_auc", n_jobs=-1,
)
for n, tr, va in zip(sizes, train_scores.mean(1), val_scores.mean(1)):
    print(f"{n:>7d}  train={tr:.3f}  val={va:.3f}")
```

**Extrapolating the value of more data:** validation error often follows an approximate power law in the number of examples, $\text{error}(n) \approx a \thinspace n^{-b} + c$. Fitting this to the curve, ideally on a log-log plot, gives a rough forecast of the gain from doubling the data and of the asymptotic floor $c$. Treat it as an estimate, not a promise.

**Related diagnostic curves:**

- **Validation (complexity) curves:** performance against a hyperparameter such as tree depth or $\lambda$, which shows where underfitting turns into overfitting (`validation_curve` in scikit-learn).
- **Training curves:** loss against epochs, for spotting overfitting during training and choosing early stopping (see 11.41).
- **Per-slice learning curves:** the overall curve may be flat while a rare segment is still improving; targeted collection for that segment is then the best investment.

**Pitfalls:**

- Use the same validation set (or the same folds) at every training size, and subsample training data in a way that preserves class balance and time order.
- For time series, grow the training window forward in time rather than sampling randomly.
- Report variance across folds; with small data, differences between points can be noise.
- The curve tells you about more data **from the same distribution**. If production data differs, collecting more historical data may not help at all.

**What interviewers look for:** the ability to turn the plot into a decision with a cost attached ("labelling 20k more examples should gain about one point of AUC, worth X; feature work on the bias side is cheaper").

### 8.61 What is Bayes error, how do you estimate the best achievable performance, and how do you know when to stop improving a model?

Bayes error is the lowest error any model could achieve on a problem, because the inputs do not fully determine the target: noise, missing information and ambiguous labels put a floor under the error. Knowing roughly where that floor is tells you whether remaining error is fixable (avoidable bias or variance) or not, and therefore when further model work stops paying.

**Formally,** for classification the Bayes-optimal classifier predicts the most probable class given $x$, and its error is:

$$
\varepsilon_{\text{Bayes}} = \mathbb{E}_x\left[1 - \max_k P(y = k \mid x)\right]
$$

It is zero only if the label is a deterministic function of the features. It depends on the features you have: adding an informative feature lowers the Bayes error itself.

**Ways to estimate the floor in practice:**

- **Human-level performance** as a proxy, especially for perception tasks. Use the best achievable human performance (a panel of experts), not the average annotator.
- **Inter-annotator agreement:** have several people label the same sample. If experts disagree on 8 percent of cases, expecting a model to be 99 percent accurate against those labels is unrealistic.
- **Duplicate inputs with different labels:** identical feature vectors with conflicting labels give a direct lower bound on irreducible error for those features.
- **Repeated measurements:** in regression, the variance between repeated measurements of the same quantity estimates the noise term $\sigma^2$.
- **Strong-model plateau:** if very different high-capacity models, more data and more tuning all converge at the same error, you are probably near the floor for the current features.
- **Label audits:** relabel a random sample carefully and measure how often the original label was wrong.

**Using it to decide what to do next** (Andrew Ng's framing):

| Gap | Meaning | Action |
| --- | --- | --- |
| Training error minus Bayes estimate | Avoidable bias | Bigger model, better features, train longer |
| Validation error minus training error | Variance | More data, regularisation |
| Test or production error minus validation error | Mismatch or overfitting to validation | Fix splits, collect production-like data |

**Knowing when to stop:**

1. **The model is near the estimated floor:** further gains need new information (new features or data sources, cleaner labels), not more tuning.
2. **Marginal gains are below marginal value:** estimate the business value of one more point of the metric and compare it with engineering cost, compute and added complexity.
3. **Offline gains no longer translate online:** if the last improvements did not move the A/B test, the offline metric has stopped being a good proxy.
4. **Gains are within noise:** differences smaller than the confidence interval across folds or seeds are not real improvements (see 12.34).
5. **Other properties matter more:** at this point, calibration, robustness, latency, monitoring and fairness often deliver more value than another 0.2 percent of accuracy.

**Pitfalls:**

- Measuring "human level" with a single non-expert, which makes the floor look higher than it is.
- Assuming the floor is fixed: better sensors, features or problem reformulation can lower it.
- Super-human results against noisy labels usually mean the model has learned annotator biases or leakage, not that it is better than experts.

### 8.62 What is the i.i.d. assumption, and what happens to generalisation when it breaks?

Almost every guarantee and evaluation practice in ML assumes training and test examples are independent and identically distributed (i.i.d.): each example is drawn independently from the same fixed distribution. When this holds, a held-out score is an unbiased estimate of future performance. In production it rarely holds exactly, and most "the model worked offline but failed in production" stories are i.i.d. violations.

**The two parts:**

- **Independent:** knowing one example tells you nothing extra about another. Violated by repeated entities, time dependence, network effects and duplicates.
- **Identically distributed:** training, validation, test and production data come from the same distribution. Violated by drift, selection effects and feedback loops.

**Common violations and their consequences:**

| Violation | Example | Consequence | Mitigation |
| --- | --- | --- | --- |
| Temporal dependence | Daily sales, sensor readings | Random splits leak future into past; optimistic scores | Time-based splits, rolling-origin validation (see 12.23) |
| Grouped data | Many rows per customer or patient | Model memorises entities; scores inflated | Group-aware splits (`GroupKFold`) |
| Covariate shift | New market, new device, new customer mix | Inputs move to regions the model has rarely seen | Monitoring, importance weighting, adversarial validation (see 10.37) |
| Label shift | Fraud rate doubles during an attack | Probabilities miscalibrated, thresholds wrong | Recalibrate priors, monitor prediction distribution |
| Concept drift | Customer behaviour changes after a price change | The same input now means a different outcome | Retraining, online updates, drift detection (see 26.65) |
| Selection bias | Credit model trained only on approved applicants | Model never sees the population it will score | Reject inference, exploration, randomised holdouts |
| Feedback loops | Recommender trained on clicks it caused | Model reinforces its own past choices | Exploration traffic, logging propensities, counterfactual evaluation |
| Adversarial behaviour | Spammers adapting to the filter | Distribution moves in response to the model | Frequent retraining, robust features, red-teaming |

**Why it matters for generalisation theory:** the bounds in 8.4 hold only under the i.i.d. assumption. Under shift, low test error on old data says little about future error, and capacity control alone cannot fix it.

**Designing evaluation around the real deployment:**

1. Ask how the model will be used: on new customers, on future time periods, in new regions? Build the test set the same way (future time slice, unseen groups, new region).
2. Compare a random split with the deployment-like split. A large difference reveals how much your random-split score overstates performance.
3. Use adversarial validation to check whether training and recent production data are distinguishable.
4. Monitor input, prediction and outcome distributions after launch, and define what triggers retraining.

```python
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score

random_cv = cross_val_score(model, X, y, cv=5).mean()
group_cv = cross_val_score(model, X, y, cv=GroupKFold(5), groups=customer_id).mean()
time_cv = cross_val_score(model, X_sorted, y_sorted, cv=TimeSeriesSplit(5)).mean()
print(random_cv, group_cv, time_cv)   # a big drop exposes dependence the random split hid
```

**What interviewers look for:** recognising that the held-out score is only as good as the match between the split and deployment, naming the specific violation in a scenario, and proposing both an evaluation fix and a production safeguard.

---

← [Chapter 7. Pipelines and Orchestration](../part-2-data-engineering/07-pipelines-and-orchestration.md) · [Table of contents](../README.md) · [Chapter 9. Machine Learning Algorithms](09-machine-learning-algorithms.md) →
