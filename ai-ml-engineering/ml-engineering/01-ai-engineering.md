<!-- notion-id: 3d667418285480909f27ff3fadb2638b -->
# AI Engineering

AI engineering interview questions: A/B testing of models, ML foundations (induction, over/underfitting, KNN), LLM concepts (RAG, MCP, agents, fine-tuning, quantization, autoregression, function calling), and model types.

### Q1. A/B Testing in Machine Learning

A/B testing compares two versions of a model in a **live environment** to determine which performs better in practice.

---

#### How It Works

| Step | Description |
| --- | --- |
| **Version A (Control)** | The existing or baseline model currently in production |
| **Version B (Treatment)** | The new/updated model with different features, algorithms, or hyperparameters |
| **Traffic Split** | Incoming real-world user requests are randomly split between the two models |
| **Measurement** | Performance is compared using business or technical metrics |

**Common metrics used:**

- Conversion rate
- Latency
- Click-through rate (CTR)
- Prediction accuracy

---

#### Why It Matters

- **Real-World Validation** — Offline metrics (accuracy, F1 score) from lab testing don't always guarantee real-world success
- **Risk Mitigation** — Limits the blast radius of a bad update by exposing only a small % of users to the new model
- **Data-Driven Decisions** — Confirms whether a model change genuinely improves user experience or business outcomes before a full rollout

---

#### Quick Summary

> A/B testing = **Control vs. Treatment model** → **Random traffic split** → **Compare real metrics** → **Roll out the winner with confidence**

### Q2. Why do we need Machine Learning?

Many tasks are too complex to solve with explicitly hand-coded rules (e.g., recognizing handwriting, detecting objects, driving a car). Machine Learning lets a computer learn how to perform such tasks directly from training examples, instead of a programmer trying to anticipate every rule manually. This makes it possible to build systems for spam filtering, bioinformatics, medical image analysis, language processing, and game-playing agents like AlphaGo, where explicit programming is impractical.

### Q3. What is Machine Learning, formally?

A set of algorithms that allow a computer to learn a specific task from training examples. "Learning" here doesn't mean memorizing the seen examples — it means generalizing to previously unseen instances. Ideally, the computer should use the training examples to extract a general rule describing how the task should be performed correctly.

### Q4. Give some real-world applications of Machine Learning.

Handwritten letter recognition, object detection, self-driving cars, spam filtering, bioinformatics, medical image analysis, natural language processing, and game-playing systems like AlphaGo and chess engines.

### Q5. What is the difference between Deduction and Induction?

- **Deduction** reasons from one or more general premises to a logically certain conclusion. If the premises are true, the conclusion is guaranteed to be true.
- **Induction** reasons from specific examples to construct or evaluate a general proposition. We can never be fully certain the conclusion is correct — it's a generalization, not a guarantee.

### Q6. How does Machine Learning relate to induction?

Machine Learning can be viewed as an attempt to automate the process of inductive inference — using specific training examples to derive a general rule or model, without any certainty that the rule is perfectly correct.

### Q7. Why should Machine Learning work at all?

It only works if there is actually something to learn — i.e., if the data contains an underlying pattern or structure. If the examples were pure noise with no consistent relationship between inputs and outputs, no algorithm could extract a meaningful general rule.

### Q8. What is Overfitting?

> 🔁 _See also [01-turing-100-interview-questions.md → Q40](../resources/01-turing-100-interview-questions.md#q40-what-is-overfitting) (separate Notion notes on the same question)._

Overfitting happens when a model is too complex and fits the training data extremely well — including its noise — rather than the true underlying pattern. It has low training (approximation) error but high test (estimation) error, because it fails to generalize to unseen data.

### Q9. What is Underfitting?

> 🔁 _See also [01-turing-100-interview-questions.md → Q44](../resources/01-turing-100-interview-questions.md#q44-what-is-underfitting) (separate Notion notes on the same question)._

Underfitting happens when a model is too simplistic to capture the underlying pattern. It tends to be stable with respect to noise, so the training error is high, but paradoxically the "estimation gap" (variance) can be low — the model consistently fails, both on training and test data.

### Q10. How would you find a good balance between overfitting and underfitting?

By choosing model complexity that minimizes total error — balancing bias (underfitting) and variance (overfitting) — often through techniques like cross-validation, regularization, or pruning, rather than maximizing fit to the training set alone.

### Q11. What is the KNN (K-Nearest Neighbors) classifier?

A simple, non-parametric learning algorithm that classifies a new data point based on the majority class among its "k" closest training examples in feature space. It doesn't build an explicit general rule during training (it's "lazy learning") — instead, it stores the training data and makes decisions at prediction time by comparing distances to neighbors.

### Q12. Is KNN prone to overfitting or underfitting?

It depends on k. A small k (e.g., k=1) closely follows individual training points and is prone to overfitting (sensitive to noise). A large k smooths the decision boundary more, which can lead to underfitting if k is too large relative to the dataset.

### Q13. LLM (Large Language Model)?

A neural network (typically transformer-based) trained on massive text datasets to predict and generate language. It's the core model behind tools like Claude or GPT.

### Q14. RAG (Retrieval-Augmented Generation)?

A technique where the model retrieves relevant documents/data from an external knowledge source at query time and uses that as context to generate more accurate, grounded answers, instead of relying only on what it memorized during training.

### Q15. MCP (Model Context Protocol)?

An open protocol (introduced by Anthropic) that standardizes how AI models connect to external tools, data sources, and services — think of it as a universal adapter so an LLM can call a calendar, database, or file system without custom integration code for each one.

### Q16. Agent

An LLM-driven system that can plan, take actions (via tools/APIs), observe results, and iterate toward a goal with some autonomy, rather than just producing a single response to a prompt.

### Q17. Fine-tuning

> 🔁 _See also [01-llm-fundamentals.md → Q5](../ai-engineering/01-llm-fundamentals.md#q5-what-is-fine-tuning) (separate Notion notes on the same question)._

Further training a pre-trained model on a smaller, task-specific dataset to specialize its behavior (e.g., tone, domain knowledge, format) beyond what the base model does out of the box.

### Q18. Quantization

> 🔁 _See also [01-llm-fundamentals.md → Q6](../ai-engineering/01-llm-fundamentals.md#q6-what-is-quantization) (separate Notion notes on the same question)._

Reducing the numerical precision of a model's weights (e.g., from 32-bit to 8-bit or 4-bit) to shrink its size and speed up inference, usually with some trade-off in accuracy.

### Q19. Autoregressive

**Autoregressive** describes the way most large language models (GPT, Claude, Llama, etc.) generate text: one token at a time, where each new token is predicted based on all the tokens that came before it.

**The mechanism**

"Autoregressive" literally means "regressing on itself" — the model's own previous outputs become part of the input for predicting the next output.

At each step, the model computes a probability distribution over its entire vocabulary (every possible next token) conditioned on everything generated so far:

```txt
P(token_n | token_1, token_2, ..., token_n-1)
```

So generating a sentence is really a chain of predictions:

1. Given "The cat sat on the" → predict "mat" (highest probability)
2. Append "mat" → now input is "The cat sat on the mat"
3. Predict the next token → maybe "."
4. Repeat until done

This is why LLMs sometimes appear to "think out loud" or benefit from chain-of-thought prompting — each token they commit to becomes context that constrains and informs everything after it. It's also why LLMs generate text left-to-right and can't "go back" and revise earlier tokens without regenerating the sequence.

**How this connects to your JEPA question**

This is exactly the point LeCun critiques. His argument against autoregressive LLMs:

- **Error accumulation** — since each token depends on all prior tokens, small mistakes early in generation can compound, since the model can never revisit and fix them — it can only keep generating forward.
- **No real-world model** — the model is trained purely to predict the next token in text, not to build a compressed causal/physical understanding of reality. It may reproduce statistical patterns that look like a world model without possessing one that's robust, consistent, and useful for planning.
- **Discrete, one-token-at-a-time reasoning** — unlike JEPA's abstract-embedding-space prediction, autoregressive LLMs predict at the token level, which LeCun sees as too low-level and rigid for the kind of hierarchical planning and physical reasoning humans do effortlessly.

**Where autoregression genuinely excels**

Despite the critique, autoregressive training is what has powered essentially the entire current generation of frontier language models — it's a simple, scalable objective (just predict the next token) that turns out to teach models a huge amount of grammar, facts, reasoning patterns, and even some emergent world-knowledge, just from massive text corpora. The debate isn't whether it works — it obviously does — but whether it's sufficient on its own to reach something like human-level general intelligence, or whether it needs to be paired with (or replaced by) architectures like JEPA for tasks requiring physical reasoning, long-horizon planning, or robust world models.

### Q20. k-NN vs k-means

#### Core difference

|  | **k-NN** | **k-means** |
| --- | --- | --- |
| **Type** | Supervised (classification/regression) | Unsupervised (clustering) |
| **Goal** | Predict a label/value for a new point | Group unlabeled points into k clusters |
| **What "k" means** | Number of neighbors to consult | Number of clusters to form |
| **Needs labeled data?** | Yes | No |
| **Training phase** | None really — it's "lazy," just stores the data | Yes — iteratively fits cluster centroids |
| **Output** | A prediction per query point | A cluster assignment per point + centroids |

#### How each actually works

**k-NN:**

- Given a new point, find its k closest points in the *labeled* training set
- Classification → majority vote; regression → average
- No model is built ahead of time — all the "work" happens at prediction time (hence "lazy learner")

**k-means:**

- Given unlabeled data, initialize k centroids (randomly, or via k-means++)
- Repeat: assign each point to its nearest centroid → recompute centroids as the mean of assigned points
- Converges when assignments stop changing
- This is "eager" — it builds a model (centroids) upfront

#### Where confusion usually comes from

- Both rely on **distance metrics** (usually Euclidean) — that's the main surface similarity
- Both have a **k that needs tuning** — but tuning k in k-NN is about bias/variance on predictions, while tuning k in k-means is about how many natural groupings exist in the data (elbow method, silhouette score)
- **k-means uses k-NN-style distance calculations internally** during the assignment step, which can make them feel related — but the *purpose* of the k is completely different

#### Quick way to remember

- k-**NN**: k = how many neighbors *vote* on a label you don't know
- k-**means**: k = how many *groups* you're trying to discover

### Q21. Parametric models vs. Non-parametric models

**Parametric models:**

- Assume a fixed functional form with a **fixed, finite number of parameters**, decided *before* seeing the data
- Learning = estimating those parameters from data
- Once trained, the training data can be discarded — the parameters *are* the model
- Model complexity doesn't grow with more data

**Non-parametric models:**

- Don't assume a fixed functional form (or make much weaker assumptions)
- Number of "parameters" effectively **grows with the size of the data**
- Often need to keep the training data around at prediction time
- Model complexity can grow as you get more data

#### Examples

| Parametric | Non-parametric |
| --- | --- |
| Linear regression | k-NN |
| Logistic regression | Decision trees (can grow unbounded) |
| Naive Bayes (with assumed distribution) | Kernel density estimation |
| Neural nets with fixed architecture\* | k-means (arguably semi-parametric — see below) |

> ✍️ _Addition drafted during sync._

**About the two footnoted rows in the table:**

- **\*Neural nets with a fixed architecture** are technically parametric: the parameter count is set before training and does not grow with the data. In practice, very large networks behave a lot like non-parametric models. They can memorize their training set, and when there is more data we usually pick a bigger architecture, so their effective capacity grows with the data.
- **k-means as "semi-parametric":** once trained, the model is just k centroids (k × d numbers). That is a fixed, finite set of parameters, which points to parametric. But the number of clusters is often chosen from the data (elbow or silhouette), and k-means assumes no distributional form beyond spherical, similarly sized clusters. So it sits between the two. A Gaussian Mixture Model is the clearly parametric probabilistic counterpart.

**Trade-offs to mention in an interview:**

- **Parametric:** fast, small and easy to interpret. It works with little data but has high bias if the assumed form is wrong (for example, a linear model on a non-linear relationship).
- **Non-parametric:** flexible, with low bias. It needs more data and is prone to overfitting without controls such as k, tree depth or kernel bandwidth. Inference can be expensive because k-NN and KDE keep the whole training set, which matters for latency and memory in production.

### Q22. Function calling

Function calling (also called "tool calling") is a capability that lets a language model interact with external tools, APIs, or code — rather than just generating text.

#### The basic idea

Normally, an LLM only produces text. It can't check today's weather, run a calculation precisely, query a database, or send an email — it can only *talk about* doing those things. Function calling bridges that gap:

1. You give the model a list of available "functions" (tools), each with a name, description, and expected parameters
2. The model, based on the user's request, decides *if* and *which* function to call, and generates the right arguments for it — as structured data (usually JSON), not free text
3. Your code actually executes that function (the model itself doesn't run it)
4. The result gets passed back to the model, which uses it to continue the conversation or produce a final answer

#### Simple example

Say you define a tool:

json

```json
{  "name": "get_weather",  "description": "Get current weather for a location",  "parameters": {    "location": "string"  }}
```

User asks: *"What's the weather in Kathmandu?"*

The model doesn't know the weather — but it recognizes this matches the `get_weather` tool, and outputs something like:

json

```json
{"name": "get_weather", "arguments": {"location": "Kathmandu"}}
```

Your application code catches that, calls the real weather API, gets back `"22°C, partly cloudy"`, and feeds that back to the model — which then replies in natural language: *"It's 22°C and partly cloudy in Kathmandu right now."*

#### Why it matters

- **Grounding**: lets models fetch real, current, or precise data instead of hallucinating
- **Action**: lets models actually *do* things — send messages, update a spreadsheet, query a database, run code
- **Structured output**: forces the model's intent into a reliable, parseable format instead of ambiguous prose
- **Composability**: models can chain multiple tool calls together to complete multi-step tasks (this is the basis of most "agentic" AI systems)

### Q23. Online vs Offline Learning

| Aspect | **Online Learning** | **Offline Learning (Batch Learning)** |
| --- | --- | --- |
| **Definition** | Model learns continuously from new data | Model learns from a fixed dataset |
| **Data** | Data arrives incrementally/continuously | Data is collected first, then used for training |
| **Training** | Updates model frequently | Retrains periodically |
| **Memory** | Usually requires less memory | Can require large memory/storage |
| **Adaptability** | Quickly adapts to changing patterns | Does not automatically adapt to new patterns |
| **Computation** | Small, frequent updates | Large, less frequent computations |
| **Use case** | Streaming data, real-time systems | Static or periodically updated datasets |
| **Example** | Fraud detection adapting to new transactions | Training an image classifier on a fixed dataset |
