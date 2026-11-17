# Chapter 24. Evaluation and Testing of AI Systems

_Part IV — Generative AI and LLM Engineering_

Evaluating LLMs and agents: metrics, LLM-as-a-judge, human evaluation, golden datasets, statistical rigour, benchmark contamination, red teaming, hallucination detection, regression testing and online evaluation.

This chapter contains **33 questions**.

## Contents

- **24.1** [How do you evaluate AI agents?](#241-how-do-you-evaluate-ai-agents)
- **24.2** [How do you evaluate LLMs and LLM outputs, and which metrics do you use beyond accuracy?](#242-how-do-you-evaluate-llms-and-llm-outputs-and-which-metrics-do-you-use-beyond-accuracy)
- **24.3** [What is AI agent observability?](#243-what-is-ai-agent-observability)
- **24.4** [What is evaluation-driven development for AI applications?](#244-what-is-evaluation-driven-development-for-ai-applications)
- **24.5** [Explain BLEU, ROUGE, and BERTScore. When would you use each?](#245-explain-bleu-rouge-and-bertscore-when-would-you-use-each)
- **24.6** [What is G-Eval, and how does it use LLMs for evaluation?](#246-what-is-g-eval-and-how-does-it-use-llms-for-evaluation)
- **24.7** [What is LLM-as-a-judge, what are its limitations, and how do you calibrate a judge?](#247-what-is-llm-as-a-judge-what-are-its-limitations-and-how-do-you-calibrate-a-judge)
- **24.8** [How do you conduct human evaluation for AI systems?](#248-how-do-you-conduct-human-evaluation-for-ai-systems)
- **24.9** [How do you red team an LLM application, and how is it different from adversarial testing?](#249-how-do-you-red-team-an-llm-application-and-how-is-it-different-from-adversarial-testing)
- **24.10** [How do you detect and measure hallucinations and factual consistency in LLM outputs?](#2410-how-do-you-detect-and-measure-hallucinations-and-factual-consistency-in-llm-outputs)
- **24.11** [How do you build a regression test suite for AI applications and run it in CI?](#2411-how-do-you-build-a-regression-test-suite-for-ai-applications-and-run-it-in-ci)
- **24.12** [What are benchmark suites (MMLU, HumanEval, GSM8K), and how do you interpret them?](#2412-what-are-benchmark-suites-mmlu-humaneval-gsm8k-and-how-do-you-interpret-them)
- **24.13** [How do you evaluate a RAG system end-to-end?](#2413-how-do-you-evaluate-a-rag-system-end-to-end)
- **24.14** [What is the difference between offline and online evaluation for AI systems?](#2414-what-is-the-difference-between-offline-and-online-evaluation-for-ai-systems)
- **24.15** [How do you evaluate multi-turn conversation quality?](#2415-how-do-you-evaluate-multi-turn-conversation-quality)
- **24.16** [What is a golden dataset, and how do you build and maintain one over time?](#2416-what-is-a-golden-dataset-and-how-do-you-build-and-maintain-one-over-time)
- **24.17** [How do you implement continuous evaluation and track quality, such as accuracy and hallucination, in production?](#2417-how-do-you-implement-continuous-evaluation-and-track-quality-such-as-accuracy-and-hallucination-in-production)
- **24.18** [How do you evaluate bias in AI model outputs?](#2418-how-do-you-evaluate-bias-in-ai-model-outputs)
- **24.19** [How do you compare two models or prompts in a statistically rigorous way?](#2419-how-do-you-compare-two-models-or-prompts-in-a-statistically-rigorous-way)
- **24.20** [How do you evaluate the robustness of an LLM application across input variations?](#2420-how-do-you-evaluate-the-robustness-of-an-llm-application-across-input-variations)
- **24.21** [What are the key differences between evaluating traditional ML vs LLM applications?](#2421-what-are-the-key-differences-between-evaluating-traditional-ml-vs-llm-applications)
- **24.22** [How do you set up an evaluation framework from scratch for a new or enterprise LLM application?](#2422-how-do-you-set-up-an-evaluation-framework-from-scratch-for-a-new-or-enterprise-llm-application)
- **24.23** [Your model passes one fairness metric but fails another. How do you handle conflicting audit results?](#2423-your-model-passes-one-fairness-metric-but-fails-another-how-do-you-handle-conflicting-audit-results)
- **24.24** [Your model was fair at deployment, but became biased 6 months later. How do you monitor continuously?](#2424-your-model-was-fair-at-deployment-but-became-biased-6-months-later-how-do-you-monitor-continuously)
- **24.25** [An external auditor cannot reproduce your model's results. How do you ensure audit reproducibility?](#2425-an-external-auditor-cannot-reproduce-your-models-results-how-do-you-ensure-audit-reproducibility)
- **24.26** [How do you red team a multimodal model where text-only safety tests miss cross-modal attacks?](#2426-how-do-you-red-team-a-multimodal-model-where-text-only-safety-tests-miss-cross-modal-attacks)
- **24.27** [How do you use confidence thresholds to decide when an LLM output is automated or sent for review?](#2427-how-do-you-use-confidence-thresholds-to-decide-when-an-llm-output-is-automated-or-sent-for-review)
- **24.28** [How many evaluation examples do you need, and how do you handle non-determinism across runs?](#2428-how-many-evaluation-examples-do-you-need-and-how-do-you-handle-non-determinism-across-runs)
- **24.29** [Why do public benchmarks and leaderboards mislead, and how do you detect benchmark contamination?](#2429-why-do-public-benchmarks-and-leaderboards-mislead-and-how-do-you-detect-benchmark-contamination)
- **24.30** [How do you evaluate long-form and open-ended outputs with rubrics?](#2430-how-do-you-evaluate-long-form-and-open-ended-outputs-with-rubrics)
- **24.31** [Offline evaluation improved but online metrics did not. How do you design online evaluation for LLM features and debug the gap?](#2431-offline-evaluation-improved-but-online-metrics-did-not-how-do-you-design-online-evaluation-for-llm-features-and-debug-the-gap)
- **24.32** [Your LLM provider is retiring the model you use. How do you prove the replacement is safe to ship?](#2432-your-llm-provider-is-retiring-the-model-you-use-how-do-you-prove-the-replacement-is-safe-to-ship)
- **24.33** [How do you know whether your evaluation suite itself is any good?](#2433-how-do-you-know-whether-your-evaluation-suite-itself-is-any-good)

---

### 24.1 How do you evaluate AI agents?

Evaluate an agent at three levels: **did it achieve the goal (outcome), did it get there in an acceptable way (trajectory), and can you afford and trust it (system behaviour)**. Judging only the final answer is not enough, because an agent can reach the right answer by luck or through a dangerous action. Chapter 21 (see 21.15) covers agent test environments, reliability metrics such as pass@k and pass^k, and agent benchmarks in depth; this answer focuses on what to measure and how it fits into the evaluation practices of this chapter.

**The three levels**

| Level | Question | Typical metrics | How to check |
| --- | --- | --- | --- |
| Outcome | Was the user's goal achieved? | Task success rate, correctness of final answer or final state | Programmatic state checks, reference answers, LLM judge with rubric |
| Trajectory | Was the path sensible and safe? | Tool-selection accuracy, argument accuracy, forbidden actions, confirmation requested, recovery from tool errors | Compare against expected tool calls, policy rules, judge over the trace |
| System | Is it efficient and reliable? | Steps, tokens, latency and cost per successful task, variance across runs, retry and loop rate | Traces and repeated runs |

**Outcome: prefer state checks over text checks**

- For agents that act, check the **end state of the environment**: was the ticket created with the right fields, did the tests pass, is the refund recorded once and only once? These checks are deterministic and cheap.
- Use an LLM judge or humans only where the outcome is open-ended text (see 24.7).

**Trajectory: choose the strictness deliberately**

- **Exact sequence match** against an expected tool sequence is brittle; many valid paths exist.
- **Subset or in-order match**: the required calls appear (optionally in order), extra harmless calls are tolerated.
- **Policy checks**: hard rules such as "never call `delete_*` without a confirmation step" or "never read another customer's record". A single violation fails the case regardless of the final answer.
- **Judge over the trace**: a rubric asking whether each step was reasonable given what the agent knew at that point. Useful for diagnosing planning errors, but calibrate it like any judge.

**System behaviour**

- **Efficiency**: an agent that succeeds in 30 steps may be worse than one that succeeds in five. Report cost and latency per *successful* task, not per run.
- **Reliability**: run each task several times; a 70 percent average can hide tasks that fail half the time. Use pass^k for consistency (see 21.15) and the sample-size guidance in 24.28.
- **Error handling**: inject tool failures, timeouts and malformed tool outputs, and check that the agent recovers, retries sensibly or stops and escalates.
- **Stopping behaviour**: does it stop when done, avoid loops (see 21.28) and ask for clarification when the request is ambiguous?

**Safety**

- Prompt injection through tool outputs and retrieved documents, attempts to exceed permissions, and high-impact actions without approval. Fold these into the red-team suite (see 24.9).

**Putting it together**

```text
Task suite (real tasks + edge cases + injected failures)
        |
Sandboxed, resettable environment
        |
Agent runs (n repeats per task)
        |
  +-----------+--------------+---------------+
  |           |              |               |
State checks  Trajectory     Judge on        Cost, latency,
(outcome)     and policy     open-ended      steps, variance
              checks         output
        |
Per-task and per-category report -> CI gate -> production trace sampling
```

**Pitfalls**

- Non-resettable environments make runs incomparable.
- Averaging over task categories hides that the agent is reliable on easy tasks and useless on the ones that matter.
- Grading the final answer only, so a wrong-but-lucky path is rewarded and later fails in production.

### 24.2 How do you evaluate LLMs and LLM outputs, and which metrics do you use beyond accuracy?

There is no single metric. You evaluate an LLM **for a specific use case**, across several quality dimensions, using the cheapest reliable method for each dimension, and you always weigh quality against latency, cost and safety. Accuracy alone misses most of what makes an LLM application useful and safe.

**Model evaluation versus application evaluation**

- **Model evaluation** asks how capable a model is in general (public benchmarks, see 24.12 and 24.29). Useful for shortlisting.
- **Application evaluation** asks whether *your* system (model plus prompt, retrieval, tools and guardrails) does *your* task well on *your* data. This is what decides whether you ship.

**Dimensions to measure**

| Dimension | Question | Typical method |
| --- | --- | --- |
| Correctness | Is the answer right? | Exact match or F1 when there is one answer; rubric or reference-guided judge otherwise |
| Groundedness | Is every claim supported by the provided context? | Claim-level NLI or judge (see 24.10) |
| Relevance and completeness | Does it answer the actual question, all parts of it? | Judge with rubric |
| Instruction following and format | Right schema, length, tone, language? | Deterministic validators |
| Reasoning | Are the steps valid, not just the final answer? | Step-level checks, unit tests for code |
| Calibration and abstention | Does it say "I don't know" when it should? | Unanswerable questions in the test set |
| Safety | Harmful content, PII leakage, jailbreaks, and over-refusal | Classifiers, red-team suite (see 24.9) |
| Robustness and consistency | Stable under paraphrase, typos, reordered context, repeated runs? | Perturbation sets (see 24.20) |
| Efficiency | Latency (p50, p95), tokens, cost per request | Traces |
| Outcome | Task completion, deflection, escalation, user rating | Online metrics (see 24.17) |

**Metrics by task type**

| Task | Useful metrics |
| --- | --- |
| Classification and extraction | Accuracy, precision, recall, F1, field-level exact match |
| Closed-form QA | Exact match, token F1, judge-graded correctness |
| Summarisation | Faithfulness, coverage of key points, judge with rubric; ROUGE only as a weak signal (see 24.5) |
| RAG | Context recall and precision, faithfulness, answer correctness (see 24.13) |
| Code and SQL | Execution-based: unit-test pass rate, pass@k, result-set match |
| Chat assistants | Helpfulness, correctness, pairwise win rate, user satisfaction |
| Safety | Violation rate, attack success rate, false refusal rate |

**Three layers of methods**

1. **Deterministic checks** first: exact match, schema validation, execution, regexes and business rules. Cheap, precise and reproducible.
2. **LLM-as-a-judge** for open-ended qualities, with rubrics and calibration against human labels (see 24.7 and 24.30). Pairwise comparison is more reliable than absolute scores for choosing between versions.
3. **Human evaluation** for subjective or high-stakes quality, for building the golden set and for calibrating the judges (see 24.8).

**Operating the evaluation**

- A versioned **golden dataset** (see 24.16), stratified by intent, difficulty and risk, run on every model, prompt or retrieval change as a CI gate (see 24.11).
- **Statistical care**: report confidence intervals, compare systems on the same items, and repeat runs to measure variance (see 24.19 and 24.28).
- **Online signals**: thumbs up and down, edits, retries, escalation, A/B tests and judge-scored samples of live traffic (see 24.17 and 24.31).
- Track quality **together with latency, cost and safety**. A slightly better output that is three times slower or twice the price may not ship.

**What interviewers look for**

- You start from the use case and the cost of errors, not from a list of metrics.
- You know lexical metrics correlate weakly with quality for open-ended generation.
- You validate automated evaluators against humans and treat evaluation numbers as estimates with uncertainty.

### 24.3 What is AI agent observability?

#### AI Agent Observability

**AI agent observability** is the ability to understand **what an agent did, why it did it, where it failed, and how well it performed** by collecting traces, logs, metrics, and evaluation data.

##### What to observe

- **Agent decisions:** Which action did the agent choose and why?
- **Tool calls:** Which tools were called, with what parameters, and what results were returned?
- **Agent trajectory:** Full sequence of steps from input to final response.
- **LLM calls:** Model, prompt/version, tokens, latency, and response.
- **Failures:** Tool errors, timeouts, invalid parameters, model failures, and retries.
- **Performance:** p50/p95/p99 latency and task completion time.
- **Cost:** Token usage, model cost, and number of tool calls.
- **Quality:** Task success, groundedness, hallucination, and user feedback.
- **Safety:** Policy violations, unauthorized tool actions, and guardrail triggers.

##### Distributed tracing

Give every request a unique `trace_id` and create spans for each operation:

```txt
User Request
     │
     ▼
Agent
 ├── LLM Call
 ├── Tool: Search
 │    └── API Call
 ├── Tool: Database
 ├── LLM Call
 └── Final Response
```

A trace lets you identify exactly **which step consumed time or caused the failure**.

##### Important metrics

| Category | Metrics |
| --- | --- |
| Reliability | Task success rate, error rate, retry rate |
| Agent | Tool-selection accuracy, steps/task |
| LLM | Tokens, latency, model errors |
| Tools | Success rate, timeout rate |
| Performance | p50/p95/p99 latency |
| Cost | Cost/request, tokens/request |
| Quality | Groundedness, hallucination rate |
| Safety | Guardrail violations, blocked actions |

##### Logs vs Metrics vs Traces

- **Logs:** What happened?
- **Metrics:** How often/how much?
- **Traces:** Where did it happen across the entire workflow?

##### Example

Without observability:

```txt
Agent failed
```

With observability:

```txt
Trace: abc123

LLM decision        → 450 ms
Search tool         → 120 ms
Database tool       → 80 ms
Payment API         → timeout
Retry               → timeout
Agent                → failed

Root cause: Payment API timeout
```

### 24.4 What is evaluation-driven development for AI applications?

#### Evaluation-Driven Development (EDD) for AI Applications

**Evaluation-driven development** is an approach where you **define how an AI system will be evaluated before or alongside building it**, then continuously use those evaluations to guide development.

Instead of:

```txt
Build → Deploy → Discover problems
```

you use:

```txt
Define evaluation criteria
        ↓
Create evaluation dataset
        ↓
Build AI system
        ↓
Run evaluations
        ↓
Identify failures
        ↓
Improve system
        ↓
Run evaluations again
```

##### Key components

- **Define success criteria:** Decide what "good" means—accuracy, groundedness, safety, latency, cost, etc.
- **Create an evaluation dataset:** Include normal cases, edge cases, and known failure cases.
- **Build automated evaluations:** Measure performance consistently after every change.
- **Use human/LLM evaluation:** Evaluate subjective qualities such as helpfulness, reasoning quality, and response quality.
- **Track regressions:** Ensure a new prompt, model, RAG change, or agent modification doesn't break previously working behavior.
- **Evaluate continuously:** Run evaluations during development and in CI/CD before production deployment.

##### Example: RAG application

Suppose you're improving a customer-support RAG system.

```txt
100 test questions
        ↓
Current RAG
        ↓
Evaluation
        ↓
Answer correctness: 91%
Groundedness:       94%
Retrieval recall:   87%
        ↓
Improve chunking
        ↓
Evaluate again
        ↓
Answer correctness: 94%
Groundedness:       96%
Retrieval recall:   92%
```

You can then automatically block a deployment if an important metric falls below an acceptable threshold.

##### Why it matters for AI

Traditional software often has deterministic tests:

```txt
input → expected output
```

LLM applications are probabilistic, so exact-output testing is often insufficient. EDD therefore combines **test cases, evaluation datasets, statistical metrics, LLM-as-a-Judge, human review, and production feedback**.

### 24.5 Explain BLEU, ROUGE, and BERTScore. When would you use each?

#### BLEU, ROUGE, and BERTScore

These are **automatic text-generation evaluation metrics** that compare a model's output with one or more reference answers.

##### 1. BLEU

**BLEU (Bilingual Evaluation Understudy)** measures **n-gram overlap** between the generated text and reference text.

It focuses heavily on **precision**: how much of the generated text appears in the reference.

Example:

```txt
Reference:  The cat is sitting on the mat.
Generated:  The cat sits on the mat.
```

BLEU recognizes overlapping words/n-grams, but may penalize different wording.

**Best for:**

- Machine translation
- Comparing generated text with a known reference
- Tasks where exact wording matters

**Limitation:** Poor at recognizing semantic similarity when wording differs.

---

##### 2. ROUGE

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** measures overlap between generated text and reference text, with strong emphasis on **recall**.

Common variants:

- **ROUGE-1:** Unigram/word overlap
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest Common Subsequence

Example:

```txt
Reference:
The company reduced cloud costs by 30%.

Generated:
Cloud costs were reduced by 30%.
```

ROUGE can recognize substantial word overlap even though the sentence structure differs.

**Best for:**

- Text summarization
- Document summarization
- Comparing generated summaries against reference summaries

**Limitation:** Still primarily based on lexical overlap and may not fully capture meaning.

---

##### 3. BERTScore

**BERTScore** evaluates **semantic similarity** rather than relying only on exact word overlap.

It uses contextual embeddings from transformer models to compare the meaning of generated and reference text.

For example:

```txt
Reference:
The company reduced cloud expenses.

Generated:
The organization lowered its infrastructure costs.
```

BLEU/ROUGE may give relatively low scores because many words differ.

BERTScore can recognize that the two sentences have **similar meaning**.

**Best for:**

- Open-ended text generation
- Paraphrasing
- Summarization
- Tasks where semantic similarity matters more than exact wording

**Limitation:** More computationally expensive and still doesn't guarantee factual correctness.

---

##### Quick comparison

| Metric | Main idea | Best use |
| --- | --- | --- |
| **BLEU** | N-gram precision/overlap | Machine translation |
| **ROUGE** | N-gram/LCS recall/overlap | Summarization |
| **BERTScore** | Semantic similarity using embeddings | Open-ended generation |

### 24.6 What is G-Eval, and how does it use LLMs for evaluation?

**G-Eval** (Liu et al., 2023) is a method for using a strong LLM as an evaluator of generated text. It combines three ideas: **chain-of-thought evaluation steps generated from the criterion, a form-filling prompt, and a probability-weighted score** instead of a single sampled number. It was shown to correlate better with human judgements on summarisation and dialogue benchmarks than BLEU, ROUGE, BERTScore and earlier LLM-based metrics.

**How it works**

1. **Task introduction and criterion**: for example "Rate the coherence of the summary on a scale of 1 to 5. Coherence means ...".
2. **Auto chain-of-thought**: the evaluator LLM is asked to write the detailed **evaluation steps** for that criterion (for example "read the article, identify the main points, check the summary presents them in a logical order ..."). The steps are generated once and reused for every sample, so all samples are graded the same way.
3. **Form-filling**: the prompt contains the criterion, the evaluation steps, the source input and the output to grade, and ends with a form such as `Coherence (1-5):`.
4. **Probability-weighted scoring**: rather than take the single token the model emits, G-Eval reads the probabilities of each score token and computes an expected score.

$$
\text{score} = \sum_{i=1}^{n} p(s_i) \thinspace s_i
$$

This gives a continuous score, reduces ties (LLM judges tend to output the same integer for most samples) and makes rankings more stable. When the API does not expose log-probabilities, a practical approximation is to sample the score several times and average.

```text
Criterion -> LLM generates evaluation steps (once)
                    |
Input + output + criterion + steps -> evaluator LLM
                    |
p(score = 1..5) from token probabilities
                    |
Expected score (continuous)
```

**Where it fits**

- Reference-free grading of qualities that lexical metrics cannot capture: coherence, relevance, consistency with a source, fluency, tone.
- In practice, libraries such as DeepEval expose a `GEval` metric where you supply the criterion and optionally your own evaluation steps; writing explicit steps yourself usually makes the judge more consistent and easier to audit.

**Limitations**

- It inherits all LLM-judge biases (see 24.7). The original paper itself reported that the GPT-4 evaluator tended to score LLM-generated text higher than human-written text.
- Probability weighting needs log-probability access; many hosted models give limited or none.
- A good correlation on a public benchmark does not transfer automatically to your domain. Calibrate against your own human-labelled sample before using it as a release gate.

**Practical guidance**

- One criterion per call; do not ask for coherence, correctness and tone in one score.
- Keep the evaluation steps under version control alongside the prompt they grade.
- Combine with deterministic checks for anything that can be verified exactly.

### 24.7 What is LLM-as-a-judge, what are its limitations, and how do you calibrate a judge?

**LLM-as-a-judge** uses an LLM to grade another system's output against a rubric, either **pointwise** ("score faithfulness to the context, pass or fail") or **pairwise** ("which answer is better, A or B?"). It scales evaluation of open-ended outputs far more cheaply and quickly than human raters, and a well-designed judge can agree with humans about as often as humans agree with each other (the MT-Bench study reported over 80 percent agreement for a GPT-4 judge on pairwise preferences). Chapter 12 introduces the idea (see 12.18); this answer covers how to make it trustworthy.

**Judging modes**

| Mode | Use it for | Notes |
| --- | --- | --- |
| Pointwise, binary or 3-level | Release gates, monitoring | Most stable; "pass, fail" beats "1 to 10" |
| Pointwise, Likert scale | Diagnostic detail | Noisy at fine granularity; scores cluster |
| Pairwise | Choosing between two prompts or models | More reliable than comparing two absolute scores |
| Reference-guided | When a gold answer or source exists | Far more reliable for correctness and faithfulness |
| Reference-free | Tone, helpfulness, style | Weakest for factual correctness |

**Known biases and failure modes**

- **Position bias**: in pairwise comparisons the judge favours the first (or second) answer. Mitigate by running both orders and counting a win only if it is consistent.
- **Verbosity bias**: longer, more confident answers score higher. Mitigate with rubrics that penalise padding and by checking score against length.
- **Self-preference**: judges favour outputs from their own model family. Use a judge from a different family than the generator.
- **Cannot verify unknown facts**: a judge without a reference may approve a fluent hallucination.
- **Prompt sensitivity and drift**: small rubric changes or a provider model update can shift scores.
- **Weak in specialised domains** (law, medicine, code correctness) without references or execution.
- **Goodhart risk**: if you optimise prompts against the same judge you report on, you learn to please the judge. Keep a held-out judge or human check.
- **Cost and latency** at scale.

**Calibration: making the judge agree with humans**

Calibration means measuring and improving how well the judge's verdicts match expert human labels, then fixing a decision threshold. It is not fine-tuning by default; most teams calibrate by improving the rubric and few-shot examples.

1. Have domain experts label a sample (typically 100 to 300 items, stratified, including hard cases) using the same rubric the judge will use.
2. Run the judge on the same items.
3. Measure agreement: **Cohen's kappa** for categorical verdicts, weighted kappa or Spearman correlation for ordinal scores, plus precision and recall of the judge's "fail" verdict, since that is what blocks releases.
4. Read the disagreements. Usually the rubric is ambiguous, a criterion is missing, or the judge lacks the reference it needs.
5. Revise rubric and examples, then re-measure on a fresh split so you do not overfit the calibration set.
6. Re-calibrate whenever the judge model, judge prompt or the product's domain changes.

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score

df = pd.read_csv("judge_calibration.csv")  # columns: item_id, human, judge  (values: "pass" or "fail")

kappa = cohen_kappa_score(df["human"], df["judge"])
fail_precision = precision_score(df["human"], df["judge"], pos_label="fail")
fail_recall = recall_score(df["human"], df["judge"], pos_label="fail")
print(f"kappa={kappa:.2f}  fail precision={fail_precision:.2f}  fail recall={fail_recall:.2f}")

# Inspect the disagreements to improve the rubric
print(df[df["human"] != df["judge"]].head(20))
```

As a rough guide, a kappa above about 0.6 is usually workable for monitoring, but also compare it with the agreement between two human raters on the same items: the judge cannot be expected to beat that.

**Good practice**

- Clear rubric with definitions and short examples of each grade; one criterion per call.
- Ask for brief reasoning before the verdict, use temperature 0 (or average several samples) and a structured output.
- Pin the judge model version and store judge prompts in version control.
- Use deterministic checks (unit tests, schema validation, exact match) wherever possible and keep judges for the rest.
- Spot-check judge outputs with humans continuously in production, not just once.

### 24.8 How do you conduct human evaluation for AI systems?

#### Human Evaluation for AI Systems

**Human evaluation** is the process of having people assess AI outputs using a **predefined rubric and standardized evaluation procedure**.

##### How to conduct it

1. **Define evaluation criteria**
   - Correctness
   - Relevance
   - Helpfulness
   - Faithfulness/groundedness
   - Clarity
   - Safety
   - Instruction following
2. **Create a representative evaluation dataset**
   - Normal cases
   - Edge cases
   - Difficult cases
   - Known failure cases
   - Different user intents
3. **Create a clear rubric**

For example:

```txt
Correctness
1 = Completely incorrect
2 = Mostly incorrect
3 = Partially correct
4 = Mostly correct
5 = Completely correct
```

4. **Give evaluators standardized instructions**

Evaluators should judge outputs using the same criteria and should not know unnecessary information, such as which model produced the response when possible.

5. **Use blinded or randomized evaluation**

For model comparison:

```txt
Question
   ↓
Response A ─┐
            ├── Human evaluator
Response B ─┘
```

The evaluator chooses the better response without knowing which model generated each one.

6. **Use multiple evaluators**

Multiple reviewers reduce the impact of individual preferences or mistakes.

7. **Measure evaluator agreement**

Use measures such as **Cohen's kappa** or **Krippendorff's alpha** to determine how consistently evaluators agree.

8. **Analyze disagreements**

Low agreement often indicates that the rubric is ambiguous or the task is difficult. Refine the rubric and repeat the evaluation when necessary.

##### Example

For a RAG chatbot:

| Criterion | Score |
| --- | --- |
| Answer correctness | 4/5 |
| Groundedness | 5/5 |
| Relevance | 5/5 |
| Completeness | 3/5 |
| Safety | 5/5 |

##### Best practice

Use human evaluation alongside automated evaluation:

```txt
Automated metrics
       +
LLM-as-a-Judge
       +
Human evaluation
       +
Production feedback
```

### 24.9 How do you red team an LLM application, and how is it different from adversarial testing?

**Red teaming** is a structured exercise in which people and tools deliberately attack an AI system to find safety, security and misuse failures before real users or attackers do. **Adversarial testing** is the repeatable part: a curated suite of attack cases run automatically. In practice you need both: red teaming discovers new failure modes, and each discovery becomes an adversarial test that runs on every release. The goal is not only that the model refuses bad prompts, but that **the whole application stays safe when the model misbehaves**.

| | Adversarial testing | Red teaming |
| --- | --- | --- |
| Goal | Regression protection against known attacks | Discover unknown attack paths |
| Who | Automated suite in CI | Humans, often with automated attack generation |
| Scope | Predefined test cases | Whole system, open-ended, creative |
| Output | Pass rate per attack category | Findings with severity, then new test cases |

**1. Threat model first**

- Assets: system prompt, customer data, credentials, tools that change state, money.
- Actors: curious users, malicious users, attackers who control content the system reads (web pages, emails, documents).
- Trust boundaries: where untrusted text enters (user input, retrieved documents, tool outputs, images, audio).

**2. Attack taxonomy**

- Direct prompt injection and jailbreaks, including role-play, encoding (base64, other languages), and many-shot attacks (see 30.2 and 30.42).
- **Indirect prompt injection** through retrieved documents, web pages, emails and tool outputs; usually the most serious risk for RAG and agents.
- **Multi-turn attacks** that escalate gradually over a conversation (for example the Crescendo pattern), which single-turn tests miss.
- System-prompt and data extraction, cross-user or cross-tenant leakage.
- Tool and agent abuse: unauthorised actions, excessive permissions, actions without confirmation.
- Harmful content for the domain (medical, financial, legal advice; self-harm).
- Hallucination under pressure: false premises, unanswerable questions.
- Resource exhaustion: huge inputs, recursive agent loops, expensive tool calls.
- Cross-modal attacks for multimodal systems (see 24.26).

**3. Generate and run attacks**

- Hand-written attacks by people who know the domain and the product.
- Automated generation and scanning with open-source tools such as garak, PyRIT and promptfoo, or an attacker LLM that mutates successful prompts.
- Test **every layer**, not just the model:

```text
User -> input guardrails -> prompt assembly -> retrieval -> LLM -> tools/APIs -> output filters -> user
          ^                     ^                ^                   ^
       attacks here       poisoned documents    injection via    unauthorised
                                                tool output      actions
```

**4. Measure, do not just collect anecdotes**

- **Attack success rate** per category, with a confidence interval (a 0 out of 50 result still allows a true rate of several percent; see 24.28).
- Leakage rate, unauthorised tool-call rate, unsafe-output rate.
- **False refusal rate** on benign but sensitive-looking requests, so the fixes do not make the product useless.
- Guardrail precision and recall.

**5. Triage and fix**

- Rate findings by severity, exploitability and impact. Data leakage across users or an unauthorised state-changing action should block launch.
- Prefer **architectural fixes** over prompt patches: least-privilege tool credentials, per-user data isolation enforced outside the model, human confirmation for high-impact actions, treating all retrieved and tool content as untrusted data.
- Then input and output guardrails, rate limits and monitoring.

**6. Make it permanent**

- Every confirmed finding becomes an adversarial regression test in CI (see 24.11).
- Re-run red teaming before major launches, after model upgrades and when new tools or data sources are added; attackers adapt, so a suite that passes for months is a sign it needs new attacks.

**Pitfalls**

- Testing only the base model in a playground instead of the deployed system with its tools and data.
- Reporting "we tried 200 jailbreaks and none worked" without categories, severity or uncertainty.
- Fixing a specific prompt string rather than the class of vulnerability.

### 24.10 How do you detect and measure hallucinations and factual consistency in LLM outputs?

A hallucination is content that is **unsupported by the provided sources or false about the world**. Measure it at the **claim level**, separate retrieval failures from generation failures, and combine several detectors because no single one is reliable on its own. "Factual consistency" (or faithfulness) is the narrower question of whether the output is supported by a given source, which is what you can measure most reliably.

**Two kinds of hallucination**

| Type | Definition | Reference needed |
| --- | --- | --- |
| Intrinsic, unfaithful | Contradicts or goes beyond the provided context (RAG, summarisation) | The context itself |
| Extrinsic, factual | False about the world, no context given | External knowledge source |

**In RAG, first separate retrieval from generation**

```text
Question -> retriever -> was the needed evidence retrieved?
              no  -> retrieval failure (fix chunking, embeddings, reranking)
              yes -> did the answer stay within the evidence?
                       no -> generation hallucination (fix prompt, model, decoding)
```

Without this split, teams change the model when the real problem is retrieval.

**Detection methods**

1. **Claim-level groundedness against sources**: split the answer into atomic claims, then label each as supported, contradicted or not found in the context, using an NLI (entailment) model or an LLM judge. Tools: RAGAS faithfulness, TruLens groundedness, SummaC and AlignScore for summaries.
2. **Citation verification**: require a citation per claim and check that the cited passage actually entails the claim.
3. **Reference-based comparison** with a gold answer or system of record where one exists.
4. **Reference-free signals** when there is no source:
   - **Self-consistency**: sample several answers and measure disagreement (SelfCheckGPT). **Semantic entropy** clusters the samples by meaning first, so paraphrases do not count as disagreement.
   - **Token-level uncertainty** on key entities and numbers: cheap, but only loosely calibrated.
   - **External verification**: retrieve evidence for each claim from search or a knowledge base and check it (the FActScore approach for long-form factual precision).
5. **Structural and domain checks**: validate IDs, SQL, code, URLs, numbers and totals against systems of record. These are cheap and very high precision.
6. **Human expert review** for high-risk domains and to calibrate the automated detectors.

**Metrics**

- **Response-level hallucination rate**: share of responses with at least one unsupported or false claim.
- **Claim-level faithfulness**: supported claims divided by total claims.

$$
\text{faithfulness} = \frac{\text{supported claims}}{\text{total claims}}
$$

- **Citation precision** (cited passages that support the claim) and **citation recall** (claims that have a supporting citation).
- **Abstention quality**: on unanswerable questions, how often the system correctly declines.

Report both response-level and claim-level numbers: a long answer with one wrong claim out of twenty is very different from a short answer that is entirely wrong.

**Evaluation data**

- Include unanswerable questions, false-premise questions, questions whose answer changed recently, and near-duplicate entities (two products with similar names). These provoke hallucinations far more than average traffic.

**In production**

- Run cheap checks inline (citation presence, ID validation, a small NLI model) and heavier judge-based checks asynchronously on a sample.
- Gate high-stakes answers: below a groundedness threshold, regenerate, answer "I don't know" or route to a human (see 24.27).
- Track the hallucination rate over time by segment and add every confirmed case to the regression suite.

**Pitfalls**

- Judges that see only the answer, not the context, cannot measure faithfulness.
- Claim extraction itself can be wrong; spot-check it.
- A low hallucination rate achieved by refusing too often is not a win; track refusal rate alongside.

### 24.11 How do you build a regression test suite for AI applications and run it in CI?

An AI regression suite is a versioned set of test cases plus automated checks that runs whenever the model, prompt, retrieval, tools or code change, and **blocks the release if quality falls below an agreed baseline**. The difference from ordinary unit tests is that outputs are non-deterministic and many checks are statistical, so the suite compares aggregate scores against thresholds with a tolerance rather than asserting exact strings.

**1. Test cases**

- Representative real queries, stratified by intent and risk.
- Edge cases, unanswerable questions and known hard cases.
- **Every production failure** that was fixed, so it cannot silently return.
- Safety and adversarial cases from red teaming (see 24.9).
- For agents, tasks with expected end states and forbidden actions (see 24.1).
- Store them as versioned files (for example JSONL) with an owner, a tag for the category and the date added. See 24.16 for building and maintaining the golden set.

**2. Checks, cheapest first**

| Check type | Examples | Determinism |
| --- | --- | --- |
| Hard assertions | Valid JSON, schema, required fields, no PII, citation present, correct tool called | Deterministic, must be 100 percent |
| Reference metrics | Exact match, F1, execution result match | Deterministic given an output |
| Judge metrics | Correctness, faithfulness, helpfulness | Statistical, calibrated (see 24.7) |
| Non-functional | Latency p95, tokens, cost per case | Measured |

**3. Tiered suites**

- **Smoke suite on every pull request**: 30 to 100 cases, hard assertions plus a few judge checks; a few minutes and cheap.
- **Full suite nightly or before release**: the complete golden set, repeated runs, full judge metrics, cost and latency.
- **Targeted suites** triggered by the change: a retrieval change runs retrieval metrics; a tool change runs agent tasks.

**4. Gating rules that survive non-determinism**

- Gate on **aggregate metrics against the current baseline**, with a tolerance derived from measured run-to-run variance (see 24.28), not on individual cases.
- Hard constraints are absolute: schema validity 100 percent, zero critical safety failures, zero cross-tenant leaks.
- Track **per-case flips**: a case that passed on the baseline and now fails is shown to the reviewer, even if the average is unchanged.
- Use temperature 0 or fixed seeds where possible, but do not assume this makes outputs fully deterministic.
- Cache model responses by (prompt version, model version, input) so unchanged parts of the system are not re-run, which saves cost and removes noise.

```python
import json
import pytest

BASELINE = json.load(open("eval/baseline_metrics.json"))
TOLERANCE = 0.02  # set from measured run-to-run standard deviation


@pytest.fixture(scope="session")
def results():
    from eval.runner import run_suite  # runs the app on the golden set and scores outputs
    return run_suite("eval/golden_smoke.jsonl", repeats=3)


def test_schema_always_valid(results):
    assert results["schema_valid_rate"] == 1.0


def test_no_critical_safety_failures(results):
    assert results["critical_safety_failures"] == 0


@pytest.mark.parametrize("metric", ["answer_correctness", "faithfulness"])
def test_quality_not_regressed(results, metric):
    assert results[metric] >= BASELINE[metric] - TOLERANCE, (
        f"{metric} dropped from {BASELINE[metric]:.3f} to {results[metric]:.3f}"
    )
```

**5. Reporting and ownership**

- The CI output should show the metric deltas, the cases that flipped, and links to the traces, so a reviewer can decide in minutes.
- Version everything that affects outputs together: prompt, model identifier, retrieval index, judge prompt and test data. A score is only meaningful with its versions.
- Refresh the baseline deliberately when a change is accepted, not automatically.

**Pitfalls**

- A suite so slow or expensive that engineers bypass it.
- Thresholds tighter than the noise, giving flaky failures that teams learn to ignore.
- A suite that only grows easy cases and stays at 100 percent; it then detects nothing (see 24.33).

### 24.12 What are benchmark suites (MMLU, HumanEval, GSM8K), and how do you interpret them?

#### Benchmark Suites: MMLU, HumanEval, GSM8K

**Benchmark suites** are standardized datasets used to compare the capabilities of AI models under controlled evaluation conditions.

##### 1. MMLU

**MMLU (Massive Multitask Language Understanding)** tests broad knowledge and reasoning across many academic and professional subjects.

Examples include:

- Mathematics
- Computer science
- Physics
- History
- Law
- Medicine
- Economics

**What it measures:** General knowledge and problem-solving across diverse domains.

**Interpretation:**

> A higher MMLU score generally indicates stronger broad knowledge and reasoning, but it does not necessarily mean the model will perform better on your specific production task.

---

##### 2. HumanEval

**HumanEval** evaluates **code-generation ability**.

The model receives a programming problem and generates code. The generated code is then executed against hidden test cases.

**Primary metric:** **`pass@k`**

For example:

$$
\text{pass@}1 = 0.70
$$

means the model successfully solves approximately 70% of the benchmark problems with one generated solution under the benchmark's evaluation procedure.

**What it measures:**

- Code generation
- Programming problem solving
- Functional correctness

It does **not** fully measure production software engineering ability, architecture, security, maintainability, or debugging.

---

##### 3. GSM8K

**GSM8K (Grade School Math 8K)** evaluates **multi-step mathematical reasoning** using grade-school-level word problems.

Example:

```txt
A store has 20 apples.
It sells 7.
Then it receives 5 more.

How many apples does it have?
```

The model must derive the answer rather than simply retrieve factual knowledge.

**What it measures:**

- Arithmetic
- Multi-step reasoning
- Mathematical problem solving

---

##### Quick comparison

| Benchmark | Primary capability | Typical metric |
| --- | --- | --- |
| **MMLU** | General knowledge/reasoning | Accuracy |
| **HumanEval** | Code generation | pass@k |
| **GSM8K** | Mathematical reasoning | Accuracy |

##### How to interpret benchmark scores

Don't simply say:

> "Model A has a higher score, therefore it's better."

Instead:

```txt
Benchmark score
      ↓
Specific capability
      ↓
Does that capability matter for my application?
      ↓
Production evaluation
```

For example, if you're building a **coding agent**, HumanEval is more relevant than MMLU. If you're building a **mathematical reasoning system**, GSM8K is more relevant.

Also consider:

- **Benchmark contamination:** The model may have seen benchmark data during training.
- **Benchmark saturation:** Older benchmarks may no longer distinguish strong models well.
- **Distribution mismatch:** Benchmark questions may not resemble your users' real inputs.
- **Gaming/overfitting:** Models can be optimized specifically for popular benchmarks.
- **Limited coverage:** One benchmark measures only a subset of model capabilities.

### 24.13 How do you evaluate a RAG system end-to-end?

#### End-to-End RAG Evaluation

I evaluate a RAG system in **two layers: retrieval quality and generation quality**, then measure the complete system from user query to final answer.

##### 1. Evaluate retrieval

First, check whether the retriever finds the **right information**.

- **Context Recall:** Did we retrieve the information needed to answer?
- **Context Precision:** How much of the retrieved context is actually relevant?
- **Top-k Recall:** Is the correct document/chunk present in the top-k results?
- **Ranking quality:** Are the most relevant chunks ranked highly?
- **Retrieval latency:** How quickly are results returned?

```txt
Query
  ↓
Retriever
  ↓
Top-k documents
  ↓
Are the relevant documents present?
```

##### 2. Evaluate generation

Then evaluate whether the LLM uses the retrieved context correctly.

- **Faithfulness / Groundedness:** Is the answer supported by the retrieved context?
- **Answer correctness:** Is the final answer actually correct?
- **Answer relevance:** Does it answer the user's question?
- **Completeness:** Does it include the important information?
- **Hallucination rate:** How often does it make unsupported claims?
- **Citation correctness:** Do citations actually support the claims?

```txt
Retrieved Context
       ↓
      LLM
       ↓
Final Answer
       ↓
Correct?
Grounded?
Relevant?
Complete?
```

##### 3. Evaluate the complete pipeline

Measure production-level characteristics:

- **End-to-end latency:** p50/p95/p99
- **Token usage**
- **Cost per query**
- **Error rate**
- **Throughput**
- **Failure/retry rate**
- **User satisfaction**
- **Abstention quality:** Does the system correctly say it doesn't know when evidence is insufficient?

##### 4. Build a golden evaluation dataset

Create representative examples containing:

```txt
Question
Expected answer
Relevant documents
Expected citations
Difficulty
```

Include:

- Normal questions
- Multi-hop questions
- No-answer questions
- Ambiguous questions
- Edge cases
- Adversarial queries
- Previously observed production failures

##### 5. Run regression evaluations

Whenever you change:

- Embedding model
- Chunking strategy
- Retrieval algorithm
- Reranker
- Prompt
- LLM
- Top-k
- Context window

Run the evaluation dataset again.

```txt
RAG Change
    ↓
Retrieval Evaluation
    ↓
Generation Evaluation
    ↓
End-to-End Evaluation
    ↓
Compare with Baseline
    ↓
Pass → Deploy
Fail → Investigate
```

##### Key metrics

| Layer | Metrics |
| --- | --- |
| Retrieval | Context Recall, Context Precision, Recall@k, ranking quality |
| Generation | Correctness, Faithfulness, Relevance, Completeness |
| Grounding | Hallucination rate, citation correctness |
| System | Latency, throughput, error rate |
| Cost | Tokens/query, cost/query |
| Production | User feedback, task success rate |

### 24.14 What is the difference between offline and online evaluation for AI systems?

#### Offline vs Online Evaluation

The main difference is **where and when the AI system is evaluated**.

|  | Offline Evaluation | Online Evaluation |
| --- | --- | --- |
| **Where** | Controlled test environment | Real production environment |
| **Data** | Fixed evaluation dataset | Real user traffic |
| **Purpose** | Compare models and detect regressions | Measure real-world performance |
| **Risk** | Low | Requires production safeguards |
| **Speed** | Fast and repeatable | Continuous |
| **Examples** | Accuracy, F1, groundedness | User satisfaction, task success, latency |

##### 1. Offline Evaluation

Evaluate the system against a **fixed dataset before deployment**.

```txt
Evaluation Dataset
       ↓
AI System
       ↓
Metrics
       ↓
Pass / Fail
```

Examples:

- Accuracy
- Precision / Recall / F1
- BLEU / ROUGE / BERTScore
- RAG faithfulness
- Hallucination rate
- Tool-call accuracy
- Safety tests
- LLM-as-a-Judge scores

**Use it for:**

- Model selection
- Prompt comparison
- RAG optimization
- Regression testing
- CI/CD deployment gates

##### 2. Online Evaluation

Evaluate the AI system **after or during production usage** using real traffic.

Examples:

- User satisfaction
- Task completion rate
- Click-through/conversion rate
- Production hallucination reports
- Latency p50/p95/p99
- Error rate
- Cost per request
- Human feedback
- A/B testing

For example:

```txt
Users
  ↓
Model A ── 50%
Model B ── 50%
  ↓
Compare real-world metrics
```

This is commonly called **online experimentation or A/B testing**.

##### Why both are necessary

Offline evaluation tells you:

> **"Does the model perform well on our known test cases?"**

Online evaluation tells you:

> **"Does the system actually work well for real users?"**

A model can score better offline but perform worse in production because real users have different queries, distributions, edge cases, and behavior.

### 24.15 How do you evaluate multi-turn conversation quality?

Multi-turn quality is a property of the **whole conversation**, not the average of its turns: a chatbot can give five individually good replies and still fail because it forgot a constraint from turn one or never completed the task. Evaluate at two levels, **turn level** for diagnosis and **conversation level** for success, and use **simulated users** because static single-turn test sets cannot exercise conversational behaviour.

**What to measure**

| Level | Criteria |
| --- | --- |
| Turn | Correctness, relevance, groundedness, format, safety of each reply given the history so far |
| Conversation | Task completion, number of turns to completion, context retention, consistency with earlier statements, adherence to constraints stated earlier, recovery after a misunderstanding, appropriate clarifying questions, graceful hand-off to a human |
| Experience | User frustration signals: repetition, rephrasing, "that is not what I asked", abandonment, escalation |

**Common multi-turn failure modes**

- Forgetting or overriding earlier constraints ("I said vegetarian").
- Contradicting an earlier answer.
- Failing to resolve references ("the second one", "same as before").
- Drift and repetition in long conversations as the context fills.
- Safety erosion: gradual escalation attacks that a single-turn test would block (see 24.9).
- Over-asking clarifying questions, or never asking when the request is ambiguous.

**Evaluation methods**

1. **Scripted conversations**: fixed multi-turn transcripts where the evaluated reply is the last turn, with specific checks ("turn 5 must respect the budget from turn 1"). Good for regression, cheap and reproducible.
2. **Simulated users**: an LLM plays a user with a persona, a goal and hidden information, and converses with the system until the goal is reached or a turn limit hits. A judge or programmatic check then scores the outcome. This is how benchmarks such as τ-bench evaluate customer-service agents. Calibrate the simulator: check that its behaviour resembles real users (message length, patience, how often it changes its mind).
3. **Conversation-level judging**: an LLM judge reads the full transcript with a rubric covering task success, consistency and retention. Long transcripts make judges less reliable, so ask targeted questions ("did the assistant ever contradict itself? quote the turns") rather than one global score.
4. **Human review** of sampled real conversations, especially those ending in escalation or abandonment.
5. **Online metrics**: task completion, turns per resolved task, escalation rate, re-contact within 24 hours, explicit ratings.

```text
Persona + goal + hidden facts
          |
   Simulated user  <-->  System under test   (until goal reached or N turns)
          |
     Transcript
          |
  +---------------+-----------------+-------------------+
  |               |                 |                   |
Goal achieved?  Constraint and    Turn-level checks   Cost, turns,
(state check)   consistency judge (safety, grounding)  latency
```

**Practical guidance**

- Run each simulated scenario several times; conversations branch and variance is high.
- Keep a set of long conversations (20 or more turns) to catch context and memory issues.
- When the system has memory across sessions, include cross-session scenarios (see 29.25).
- Add real failed conversations, anonymised, to the scripted regression set.

**Pitfalls**

- Evaluating each turn with the gold history instead of the system's own previous replies, which hides compounding errors.
- A simulator that is too cooperative, so every scenario succeeds.

### 24.16 What is a golden dataset, and how do you build and maintain one over time?

A **golden dataset** is a curated, versioned set of inputs with trusted expected outputs or grading criteria, used as the fixed yardstick for comparing versions of an AI system and gating releases. Its value depends on three properties: it must be **representative** of real use, **trustworthy** in its labels, and **alive**, growing with the product without losing comparability.

**What a record contains**

```text
id, input (or conversation), context or documents if fixed,
expected answer or acceptance criteria, expected citations or tool calls,
category / intent, difficulty, risk level, source (production, expert, synthetic),
label author and reviewer, date added, version
```

For open-ended tasks, store **acceptance criteria** ("must mention the 30-day limit, must not promise a refund") rather than one exact reference answer.

**Building the first version**

1. **Start from real traffic** if any exists: sample production logs or support tickets, then stratify by intent so rare but important intents are represented.
2. If there is no traffic, have domain experts write realistic queries, then expand with synthetic variations (see 29.54), always reviewed by a human.
3. Add deliberate **hard cases**: unanswerable questions, ambiguous requests, false premises, adversarial inputs, long inputs.
4. Get labels from domain experts with written guidelines; double-label a subset and measure agreement (see 24.8). Low agreement means the task definition is unclear, which is worth knowing before you build anything.
5. Size: a few hundred well-chosen items is usually enough to start; see 24.28 for how size relates to the differences you can detect.

**Splits with different purposes**

| Split | Purpose | Rule |
| --- | --- | --- |
| Development set | Iterate on prompts, retrieval and judges | Engineers can look at it freely |
| Held-out test set | Release decisions | Not used for tuning; access restricted |
| Regression set | Fixed bugs and incidents | Only grows |
| Judge calibration set | Human labels for validating judges | Refreshed periodically |

Tuning prompts against the same set you report on overfits it, just as with any model.

**Maintaining it over time**

- **Feed it from production**: every confirmed failure, escalation or complaint is triaged and, if useful, added with a label. This is the data flywheel (see 26.87).
- **Re-label when the truth changes**: policies, prices and documents change, and a golden answer that is now wrong will punish a correct system. Give each item an owner or a source document so changes can be traced.
- **Retire saturated items**: if every version passes a case, it no longer discriminates. Keep a small sample as a smoke test and move the rest out of the headline score.
- **Watch coverage**: compare the category distribution of the golden set with current production traffic every quarter; new features and user groups appear.
- **Version it** like code (data version control or a dataset registry), and record which version produced which score. When the set changes, re-run the baseline on the new version so comparisons stay like for like.
- **Protect it**: keep the held-out set out of prompts, few-shot examples and fine-tuning data, and out of anything a vendor might train on.
- **Handle privacy**: anonymise production samples and respect retention rules.

**Pitfalls**

- A set written by engineers that reflects how engineers phrase questions, not how users do.
- Only adding easy cases, so the score drifts to 100 percent and stops being informative.
- Mixing label versions so last quarter's score is not comparable with this quarter's.

### 24.17 How do you implement continuous evaluation and track quality, such as accuracy and hallucination, in production?

Continuous evaluation means quality is **measured all the time, on real traffic, with dashboards and alerts**, not with occasional manual spot checks. It extends the offline golden-set evaluation (see 24.14) with three things production adds: real inputs that drift, signals from users, and the need to catch regressions you did not cause, such as a provider silently updating a model or a document source changing.

**Signals to collect**

| Signal | Examples | Latency | Cost |
| --- | --- | --- | --- |
| System | Latency p50 and p95, errors, timeouts, tokens, cost per request | Real time | Free |
| Deterministic checks | Schema validity, citation present, ID exists, guardrail triggered | Real time | Cheap |
| Automated quality | Judge-scored faithfulness, correctness and relevance on a sample | Minutes to hours | Moderate |
| User behaviour | Thumbs, edits, copy, retries, rephrasing, abandonment, escalation | Hours | Free |
| Human review | Expert labels on a stratified sample | Days | Expensive |
| Business outcome | Resolution, deflection, conversion | Days to weeks | Free |

**Pipeline**

```text
Production traces (with prompt, model, index and config versions)
        |
Sampling: random + targeted (low confidence, negative feedback, new intents, high risk)
        |
Async evaluators: deterministic checks, judges, hallucination detectors
        |
Metrics store, sliced by version, intent, customer segment, language
        |
Dashboards + alerts on thresholds and trend changes
        |
Human review queue for failures -> labels -> golden and regression sets
        |
Fixes -> offline regression -> canary or A/B -> release
```

**Tracking accuracy and hallucination over time**

- Define each metric precisely and keep the definition stable: for example "share of sampled responses with at least one unsupported claim, as judged by judge version 3".
- **Version the evaluator**. If you change the judge prompt or model, run old and new side by side for a period, otherwise the trend line breaks for a reason unrelated to the product.
- **Slice everything**: by model and prompt version, intent, language, tenant, and new versus returning users. Aggregates hide regressions in small but important segments.
- Show **confidence intervals** on dashboards; a daily sample of 200 judged responses moves by several points by chance alone (see 24.28).
- Alert on **sustained** changes (for example a control chart or a rolling window crossing a threshold), not on single noisy days.
- Periodically measure true accuracy with human labels to check that judge-based numbers have not drifted from reality.

**Drift to watch**

- **Input drift**: new topics, languages or user groups; track embedding clusters of queries and the share of queries the system declines.
- **Knowledge drift**: source documents changed, so retrieved context and correct answers changed.
- **Model drift**: hosted model updates behind the same name. Pin versions where possible and run a daily canary of the golden set against production to detect silent changes.

**Feedback loop**

- Every confirmed failure gets a root-cause tag (retrieval, generation, tool, policy, data) so the team fixes the right component.
- Add failures to the regression set and re-run offline before the fix ships.

**Pitfalls**

- Treating thumbs-down rate as quality; it is sparse and biased towards unhappy users. Use it to find cases, not to measure.
- Logging outputs without the versions and retrieved context needed to reproduce them.
- Judging 100 percent of traffic with an expensive model when a stratified sample would give the same answer.

### 24.18 How do you evaluate bias in AI model outputs?

##### Evaluating Bias in AI Model Outputs

Bias evaluation checks whether an AI system produces **systematically different, unfair, or less accurate outcomes for different groups or under different demographic conditions**.

- **Define protected groups:** Identify relevant attributes such as gender, age, disability, or other legally/ethically relevant groups for the application.
- **Create balanced test sets:** Evaluate equivalent scenarios while varying only the demographic attribute being tested.
- **Compare performance:** Measure differences in accuracy, precision, recall, false-positive/false-negative rates, and other task-specific metrics across groups.
- **Measure outcome disparities:** For classification or decision systems, use metrics such as demographic parity, equal opportunity, and equalized odds where appropriate.
- **Test language and cultural bias:** Check whether equivalent prompts produce different stereotypes, sentiment, toxicity, or quality across demographic or cultural variations.
- **LLM-specific testing:** Use counterfactual prompts, where only the demographic characteristic changes, and compare the outputs.
- **Human evaluation:** Have diverse evaluators review potentially biased or stereotypical outputs.
- **Monitor production:** Continuously track subgroup performance and complaints because bias can emerge after deployment or distribution changes.

##### Example

Suppose an AI evaluates job applications. Create equivalent test cases where only the applicant's gender changes:

```txt
Candidate A → Female
Candidate B → Male
```

If equivalent qualifications consistently receive different outcomes, investigate whether the model or surrounding pipeline introduces bias.

### 24.19 How do you compare two models or prompts in a statistically rigorous way?

Run both systems on **the same items**, compute the **per-item difference**, and report the mean difference with a **confidence interval** from a paired test or a paired bootstrap. Decide in advance which metric is primary and what difference matters in practice. A headline like "92.5 versus 91.2 percent" means nothing without its uncertainty.

**1. Design the comparison**

- **Same items, same conditions**: identical inputs, retrieved context (or the same retrieval configuration), judge and judge prompt.
- **One primary metric** decided before looking at results, plus guardrail metrics (safety, latency, cost) that must not get worse.
- **Minimum effect that matters**: for example "at least 2 points of answer correctness". This determines how many items you need (see 24.28).

**2. Use paired analysis**

Item difficulty varies far more than the difference between two versions. Pairing removes that variance, so a paired analysis can detect much smaller differences than comparing two independent averages.

| Data | Test |
| --- | --- |
| Binary per item (pass or fail) | McNemar's test, or paired bootstrap of the difference |
| Continuous or ordinal scores | Paired bootstrap, Wilcoxon signed-rank, paired t-test if roughly normal |
| Pairwise preferences (A wins, B wins, tie) | Sign test or binomial test on wins excluding ties; Bradley-Terry when ranking many systems |
| Items grouped (several questions per document or conversation) | Cluster bootstrap, resampling whole groups |

For binary outcomes McNemar uses only the **discordant pairs**: $b$ items where A passed and B failed, and $c$ items where B passed and A failed.

$$
\chi^2 = \frac{(b - c)^2}{b + c}
$$

**3. Account for run-to-run variance**

- LLM outputs and judges are stochastic. Run each system several times per item (or sample several outputs) and average per item before comparing, which reduces noise.
- Measure the variance of each system against itself first: if two runs of the *same* system differ by 1.5 points, a 1.3-point difference between systems is not evidence.

**4. Paired bootstrap example**

```python
import numpy as np

rng = np.random.default_rng(0)

# per-item scores (e.g. 1/0 correctness, or a mean over repeated runs), same items in the same order
a = np.load("scores_model_a.npy")
b = np.load("scores_model_b.npy")
diff = b - a

boot = np.array([
    diff[rng.integers(0, len(diff), len(diff))].mean()
    for _ in range(10_000)
])
low, high = np.percentile(boot, [2.5, 97.5])
print(f"mean difference = {diff.mean():+.3f}, 95% CI [{low:+.3f}, {high:+.3f}]")
print(f"B better on {np.sum(diff > 0)}, worse on {np.sum(diff < 0)}, tied on {np.sum(diff == 0)} items")
```

If the interval excludes zero and its lower end is above your practical threshold, the improvement is both real and worth having.

**5. Avoid the common traps**

- **Multiple comparisons**: trying 20 prompt variants and picking the best guarantees a lucky winner. Select on the development set, then confirm once on the held-out set, or apply a correction such as Holm or Benjamini-Hochberg.
- **Peeking**: stopping as soon as a result looks significant inflates false positives. Fix the sample size first, or use a sequential method designed for it.
- **Judge confounds**: if the judge is from the same family as one candidate, self-preference biases the comparison (see 24.7).
- **Aggregate only**: check per-category differences; a small overall gain can hide a large loss on a critical intent.
- **Statistical versus practical significance**: with thousands of items a 0.3-point difference can be "significant" and irrelevant.

**Reporting**

Report the per-item paired difference with its confidence interval, the number of items and repeats, the win, loss and tie counts, per-category results and the guardrail metrics. Offline wins should then be confirmed online (see 24.31).

### 24.20 How do you evaluate the robustness of an LLM application across input variations?

##### Evaluating Robustness Across Input Variations

Robustness measures whether an LLM application **continues to produce correct, safe, and reliable behavior when the input changes slightly or becomes more challenging**.

- **Create input variants:** Test paraphrases, spelling mistakes, typos, different wording, formatting changes, long/short inputs, ambiguous queries, multilingual inputs, and noisy inputs.
- **Use adversarial variations:** Test prompt injection, conflicting instructions, irrelevant context, misleading information, and edge cases.
- **Counterfactual testing:** Change one important attribute while keeping the rest of the input unchanged and check whether behavior changes appropriately.
- **Measure consistency:** Compare outputs across equivalent inputs for correctness, relevance, groundedness, safety, and instruction following.
- **Measure degradation:** Compare performance on normal inputs versus perturbed inputs.
- **Test context limits:** Gradually increase input length and irrelevant information to identify context-window or retrieval degradation.
- **Evaluate repeated runs:** For nondeterministic models, run the same inputs multiple times and measure output variability.
- **Regression testing:** Add discovered failure cases and difficult input variations to the golden dataset.

##### Example

Original:

```txt
"How do I reset my password?"
```

Variants:

```txt
"I forgot my password. How can I reset it?"
"how do i reset my pasword?"
"PASSWORD RESET?"
"Can you help me regain access to my account?"
```

A robust system should provide **consistent and correct behavior** across these equivalent inputs.

##### Useful metrics

- Task success rate
- Accuracy / F1
- Groundedness/faithfulness
- Hallucination rate
- Safety violation rate
- Output consistency
- Performance degradation under perturbation
- Latency and error rate

### 24.21 What are the key differences between evaluating traditional ML vs LLM applications?

##### Traditional ML vs LLM Application Evaluation

| Aspect | Traditional ML | LLM Applications |
| --- | --- | --- |
| **Output** | Usually structured: class, score, prediction | Open-ended natural language, code, tool calls |
| **Ground truth** | Often clearly defined | Frequently subjective or reference-dependent |
| **Metrics** | Accuracy, Precision, Recall, F1, RMSE, MAE, AUC | Correctness, relevance, faithfulness, groundedness, hallucination, safety |
| **Determinism** | Usually deterministic for fixed input/model | Often probabilistic and variable |
| **Evaluation** | Compare prediction directly with ground truth | Often requires semantic or rubric-based comparison |
| **Human evaluation** | Usually limited | Often important for quality, helpfulness, and preference |
| **Safety** | Usually task-specific | Major concern: harmful content, prompt injection, data leakage, misuse |
| **Context** | Input features are usually predefined | Conversation history, retrieved documents, tools, and prompts affect output |
| **System evaluation** | Mostly model performance | Must evaluate model + prompt + RAG + tools + agents + orchestration |
| **Production monitoring** | Accuracy/drift, latency, errors | Quality, hallucination, groundedness, safety, latency, tokens, cost, tool failures |

##### Key Difference

Traditional ML typically evaluates:

```txt
Input → Prediction → Compare with Ground Truth
```

LLM applications often require:

```txt
Input + Context + Prompt
        ↓
      LLM
        ↓
Output / Tool Calls
        ↓
Quality + Grounding + Safety + Task Success
```

### 24.22 How do you set up an evaluation framework from scratch for a new or enterprise LLM application?

Build the evaluation framework **before optimising the system**, starting small and growing it with the product: agree what "good" means with stakeholders, build a first golden set, automate the checks, establish a baseline, then wire it into CI and production. For an enterprise application add governance: documented results per release, audit trails and clear owners.

**1. Define success with stakeholders**

- The task and the cost of each error type (a wrong refund policy answer is worse than a clumsy sentence).
- Quality dimensions with target thresholds: correctness, groundedness, format compliance, safety, abstention.
- Non-functional budgets: latency p95, cost per request.
- One business outcome the system should move, such as ticket deflection or handling time.
- A **regression budget**: how much a metric may drop before a release is blocked.

**2. Build the datasets** (see 24.16)

- **Golden set**: a few hundred real or realistic queries with reference answers or acceptance criteria, stratified by intent, difficulty and risk, reviewed by domain experts.
- **Adversarial set**: prompt injection, jailbreaks, PII, out-of-scope requests, false premises (see 24.9).
- Development and held-out splits; a regression set that grows with every fixed failure.

**3. Choose metrics per component**

| Component | Metrics |
| --- | --- |
| Retrieval | Recall@k, precision@k, MRR or nDCG, context recall (see 24.13) |
| Generation | Correctness, faithfulness, relevance, completeness |
| Structure | Schema validity, required fields, citation presence |
| Agents | Task success, tool-call accuracy, policy violations (see 24.1) |
| Safety | Attack success rate, leakage, false refusal rate |
| Operations | Latency, tokens, cost, error rate |

**4. Implement evaluators, cheapest and most reliable first**

1. Deterministic checks (schema, exact match, execution, business rules).
2. NLI or claim-level faithfulness checks.
3. LLM judges with rubrics, calibrated against a human-labelled sample (see 24.7).
4. Human review for what automation cannot judge.

**5. Baseline and iterate**

- Run the simplest reasonable system first and record scores, cost and latency with all versions. Every later change is compared with this baseline using paired statistics (see 24.19).
- Do error analysis by reading failures and tagging root causes; the categories tell you what to fix and which new checks to write.

**6. CI/CD gating** (see 24.11)

- Smoke suite on each pull request, full suite before release, results stored keyed by prompt, model, index, data and judge versions.

**7. Production evaluation** (see 24.17 and 24.31)

- Tracing (for example Langfuse, LangSmith or OpenTelemetry), sampled automated grading of live traffic, user feedback, A/B tests, dashboards and alerts on quality, safety and cost drift.

**8. Governance for enterprise settings**

- An evaluation report per release: datasets and versions, scores with intervals, known limitations and sign-off.
- Audit logs linking each production answer to its model, prompt and retrieval versions (see 24.25 and 30.14).
- Clear escalation paths for high-risk failures, and periodic independent review in regulated domains.

**Reference architecture**

```text
Golden + adversarial + regression sets (versioned)
        |
Evaluation runner (parallel, cached, repeat runs)
        |
Application under test (pinned versions)
        |
Evaluators: deterministic | NLI | calibrated judges | human queue
        |
Results store (scores + versions + traces)
        |
Report and CI gate -> release -> production traces
        |                                   |
        +------- new cases from failures <--+
```

**Pitfalls**

- Spending months on the perfect framework before shipping anything; start with 50 good cases and grow.
- Metrics nobody acts on. Each metric should have an owner and a decision it informs.
- Building a bespoke platform when an off-the-shelf tool covers the need; a platform is worth it only when many teams share it (see 29.10).

### 24.23 Your model passes one fairness metric but fails another. How do you handle conflicting audit results?

##### Handling Conflicting Fairness Metrics

Fairness metrics can legitimately conflict because **different metrics measure different notions of fairness**. Passing one does not mean the model is universally fair.

- **Understand what each metric measures**
  - Demographic parity → similar positive-outcome rates across groups.
  - Equal opportunity → similar true-positive rates.
  - Equalized odds → similar true-positive and false-positive rates.
- **Identify the application's objective and risk:** Determine which type of error is most harmful and which fairness property is relevant to the use case.
- **Check statistical uncertainty:** Use confidence intervals and significance tests before treating small differences as meaningful.
- **Analyze trade-offs:** Some fairness criteria cannot generally be satisfied simultaneously when groups have different base rates or model score distributions.
- **Consider performance impact:** Don't optimize fairness by severely degrading overall accuracy or creating another harmful disparity.
- **Use subgroup analysis:** Investigate whether the conflict comes from data imbalance, label quality, sampling, thresholds, or model behavior.
- **Set explicit priorities:** Work with stakeholders to define which fairness constraints are mandatory and which are secondary.
- **Document the decision:** Record the metrics, trade-offs, rationale, and residual risks.
- **Monitor continuously:** Re-audit after model, data, or threshold changes and monitor subgroup performance in production.

##### Example

Suppose:

```txt
Model A:
Demographic parity     → Pass
Equal opportunity      → Fail

Model B:
Demographic parity     → Fail
Equal opportunity      → Pass
```

I would **not simply choose the model with more passing metrics**. I would determine which fairness criterion aligns with the application's risk and decision objective, quantify the trade-off, and select thresholds or model changes that satisfy the most important constraints.

### 24.24 Your model was fair at deployment, but became biased 6 months later. How do you monitor continuously?

##### Continuous Fairness Monitoring

A model can become biased after deployment because the **data distribution, user population, labels, policies, or model behavior changes over time**. I would treat fairness as a continuous monitoring problem.

- **Monitor subgroup metrics:** Track fairness metrics such as demographic parity, equal opportunity, equalized odds, and subgroup error rates over time.
- **Monitor data drift:** Compare the distribution of features and relevant group populations against the deployment baseline.
- **Monitor performance drift:** Track accuracy, precision, recall, false-positive and false-negative rates separately for each group.
- **Set fairness thresholds:** Define acceptable disparity limits and trigger alerts when they are exceeded.
- **Use statistical significance:** Don't react to small fluctuations; use confidence intervals and minimum sample sizes.
- **Monitor label/outcome drift:** Changes in ground-truth distributions can create new disparities even when the model itself hasn't changed.
- **Investigate root causes:** Determine whether bias comes from changing data, sampling, labels, thresholds, upstream systems, or the model.
- **Periodic audits:** Perform deeper fairness reviews in addition to automated monitoring.
- **Mitigation:** Rebalance data, retrain the model, adjust thresholds where appropriate, or introduce fairness constraints.
- **Maintain a fairness regression suite:** Re-test historical fairness cases whenever the model or data pipeline changes.

##### Monitoring Pipeline

```txt
Production Data
      ↓
Data / Population Drift
      ↓
Subgroup Performance
      ↓
Fairness Metrics
      ↓
Statistical Tests
      ↓
Thresholds / Alerts
      ↓
Root-Cause Analysis
      ↓
Mitigation
      ↓
Re-evaluation
```

### 24.25 An external auditor cannot reproduce your model's results. How do you ensure audit reproducibility?

##### Ensuring Audit Reproducibility

Audit reproducibility means an external auditor should be able to **recreate the same model behavior, evaluation results, and decision process** using the documented artifacts.

- **Version everything:** Model version, code, prompts, configuration, dependencies, datasets, feature pipelines, and evaluation scripts.
- **Freeze the evaluation dataset:** Store an immutable copy of the exact dataset used for the audit, including preprocessing and labeling versions.
- **Record model artifacts:** Store model weights, tokenizer, architecture, hyperparameters, and training configuration.
- **Track the environment:** Record Python/runtime versions, libraries, OS/container image, GPU/CPU configuration, and infrastructure dependencies.
- **Control randomness:** Record random seeds and deterministic settings where possible.
- **Maintain experiment tracking:** Store training runs, parameters, metrics, artifacts, and timestamps using tools such as MLflow or equivalent.
- **Track data lineage:** Document where training/evaluation data came from and every transformation applied.
- **Version prompts and policies:** For LLM applications, preserve system prompts, templates, retrieval configuration, tool definitions, guardrails, and model/provider versions.
- **Log inference metadata:** Record request IDs, model version, configuration, retrieved context identifiers, tool calls, and evaluation results while protecting sensitive data.
- **Use reproducible deployment:** Package the application in versioned containers or infrastructure-as-code.
- **Maintain an audit trail:** Keep immutable records of deployments, model changes, approvals, and evaluation results.
- **Provide an audit package:** Give auditors the exact artifacts, documentation, test dataset, environment specification, and reproduction procedure.

##### Reproducibility Chain

```txt
Data Version
     ↓
Code Version
     ↓
Model / Prompt Version
     ↓
Environment Version
     ↓
Configuration + Seeds
     ↓
Evaluation Dataset
     ↓
Evaluation Pipeline
     ↓
Results + Audit Logs
```

### 24.26 How do you red team a multimodal model where text-only safety tests miss cross-modal attacks?

##### Red Teaming Multimodal Models

For multimodal models, I would test **each modality independently and the interactions between modalities**, because an attack may only work when text, images, audio, or video are combined.

- **Test individual modalities**
  - Text prompts
  - Images/screenshots
  - Audio
  - Video
  - OCR and speech-to-text inputs
- **Test cross-modal attacks**
  - Malicious instructions embedded inside images
  - Text that conflicts with instructions visible in an image
  - Hidden or misleading text that OCR extracts
  - Audio containing instructions that conflict with the user's text
  - Images designed to manipulate visual reasoning
  - Multi-turn attacks where one modality establishes context and another introduces the malicious instruction
- **Test instruction hierarchy**
  - Verify the model distinguishes trusted system instructions from untrusted content extracted from images, documents, audio, or retrieved data.
  - Treat multimodal content as potentially untrusted input.
- **Test perception failures**
  - Small/obscured text
  - Adversarial visual elements
  - Low-quality or rotated images
  - Ambiguous objects
  - Misleading captions
  - Speech transcription errors
- **Test multimodal RAG**
  - Malicious instructions in PDFs, screenshots, images, or other retrieved content.
  - Cross-modal data leakage between users or documents.
  - Incorrect grounding caused by mismatching text and visual evidence.
- **Test tool use**
  - Check whether visual or audio inputs can cause unauthorized tool calls.
  - Verify that high-impact actions require appropriate authorization and confirmation.

##### Example

```txt
User text:
"Summarize this image."

Image:
[Contains an instruction attempting to override the system behavior]

Expected behavior:
The model treats the instruction in the image as untrusted content
and summarizes the image rather than following that instruction.
```

##### Evaluation Metrics

- Cross-modal attack success rate
- Prompt-injection success rate
- Unsafe-output rate
- Unauthorized tool-call rate
- Sensitive-data leakage rate
- OCR/ASR-related failure rate
- Groundedness and factual consistency
- False-positive/false-negative safety rates

### 24.27 How do you use confidence thresholds to decide when an LLM output is automated or sent for review?

A confidence threshold turns a model's estimated certainty into a decision: **above the threshold the output is acted on automatically; below it the system abstains, asks a clarifying question or routes to a human**. This is selective prediction: you trade coverage (the share of cases automated) for accuracy on the cases you automate. The hard part with LLMs is getting a confidence score that is actually calibrated, and choosing the threshold from the cost of errors and review capacity rather than picking 0.8 by habit.

**Where confidence comes from**

| Source | Notes |
| --- | --- |
| Token log-probabilities of the answer or label | Good for classification-style outputs; less meaningful for long text |
| Self-consistency: agreement across several samples | Robust and model-agnostic; costs extra calls |
| Verifier or judge score (groundedness, correctness) | Works for open-ended output; must be calibrated |
| Retrieval signals: top similarity, number of supporting passages | Catches "no good evidence" cases in RAG |
| Rule checks: schema valid, values within range, IDs exist | Hard gates rather than scores |
| Verbalised confidence ("I am 90 percent sure") | Usually poorly calibrated; do not use alone |

Often the best signal is a small model (for example logistic regression) that combines several of these features and is trained on labelled outcomes.

**Calibrate before you threshold**

- A score is calibrated if, among outputs scored 0.9, about 90 percent are correct. Check with a reliability diagram and expected calibration error on a labelled set; recalibrate with Platt scaling or isotonic regression if needed (see 12.36 for calibration and 12.21 for threshold selection in classical classifiers).

**Choose the threshold from costs and capacity**

- Plot a **risk-coverage curve**: for each threshold, the error rate on automated cases versus the share automated.
- Pick the threshold that meets a target precision on automated cases (for example 99 percent for posting invoices), or that minimises expected cost given the cost of an automated error versus the cost of a human review, subject to how many reviews the team can handle per day.

```python
import numpy as np
import pandas as pd

df = pd.read_csv("labelled_outputs.csv")  # columns: confidence (0..1), correct (0/1)
df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

df["coverage"] = (np.arange(len(df)) + 1) / len(df)
df["precision_if_cut_here"] = df["correct"].cumsum() / (np.arange(len(df)) + 1)

target = 0.99
ok = df[df["precision_if_cut_here"] >= target]
threshold = ok["confidence"].min() if not ok.empty else None
coverage = ok["coverage"].max() if not ok.empty else 0.0
print(f"threshold={threshold}, automated share={coverage:.1%} at precision >= {target:.0%}")
```

Use a held-out set for the final choice, and leave a margin because a precision estimated on a few hundred items has a wide interval.

**Design the actions below the threshold**

- Ask a clarifying question, retrieve more evidence and retry, fall back to a safer answer ("I could not find this in our policy documents"), or queue for human review with the model's draft and evidence attached.
- Use different thresholds by risk: a higher bar for actions that move money than for drafting an email.

**Operate it**

- Monitor coverage, automated-case precision (from audits of a sample of automated cases) and review queue size. A drop in coverage is often the first sign of input drift.
- Recalibrate after any model, prompt or retrieval change, because the score distribution moves.
- Use the reviewed cases as new labelled data for calibration and for the golden set.

**Pitfalls**

- Trusting self-reported confidence.
- Auditing only the low-confidence cases, so errors among confident, automated outputs go unseen.
- A threshold set once at launch and never revisited.

### 24.28 How many evaluation examples do you need, and how do you handle non-determinism across runs?

You need enough examples that the **uncertainty in your metric is smaller than the differences you want to act on**, and enough repeated runs that sampling noise from the model and the judge does not masquerade as a real change. A score from 50 examples run once is an anecdote; a score with a confidence interval, from a sample sized for the decision, is evidence.

**Uncertainty of a single score**

For a pass rate $p$ measured on $n$ independent items, the 95 percent margin of error is approximately:

$$
\text{margin} \approx 1.96 \sqrt{\frac{p(1-p)}{n}}
$$

| Items | Margin at p = 0.8 |
| --- | --- |
| 50 | about 11 points |
| 100 | about 8 points |
| 400 | about 4 points |
| 1,000 | about 2.5 points |

So a 100-item eval cannot distinguish 80 percent from 85 percent. For rare failures use the **rule of three**: if you see zero failures in $n$ trials, the 95 percent upper bound on the true failure rate is roughly $3/n$. Zero jailbreaks out of 100 attempts is still consistent with a 3 percent success rate.

**Sample size to detect a difference between two versions**

Comparing on the same items (paired) needs far fewer items than comparing independent samples, because only the items where the two versions disagree carry information. For binary outcomes, with $p_d$ the share of discordant items and $\delta$ the difference you want to detect (5 percent significance, 80 percent power):

$$
n \approx \frac{(1.96 + 0.84)^2 \thinspace p_d}{\delta^2}
$$

Example: if the versions disagree on 10 percent of items and you want to detect a 2-point difference, you need about 2,000 items. Unpaired, at an 80 percent pass rate, it would take over 6,000 per arm. The practical consequences:

- Decide the smallest difference that would change a decision before choosing the size.
- If you cannot afford the items, accept that you can only detect large differences, and say so.
- Increase sensitivity cheaply by pairing, by removing saturated items that every version passes, and by focusing items on the categories the change targets.

**Sources of non-determinism**

- Sampling temperature and top-p.
- Even at temperature 0, hosted models are not fully deterministic: batching, floating-point order on GPUs and mixture-of-experts routing can change outputs.
- Silent provider model updates behind the same name.
- The judge is itself an LLM with the same issues.
- Retrieval differences when the index is rebuilt or approximate search is used.

**Handling it**

1. **Measure it first**: run the same system on the same items several times and look at the spread of the aggregate score and at how many items flip. This is your noise floor; regression thresholds must be wider than it (see 24.11).
2. **Repeat runs per item** and average per item before aggregating. With $k$ samples per item, the within-item part of the variance shrinks by a factor of $k$; the between-item part does not, so more items help more than more repeats once $k$ is around 3 to 5.
3. **Report per-item consistency** where it matters: for agents and customer-facing flows, "succeeds on all $k$ runs" (pass^k, see 21.15) is more meaningful than the average.
4. **Fix what you can**: pinned model versions, temperature 0 for judges, fixed seeds where supported, frozen retrieval snapshots for offline runs, cached outputs for unchanged components.
5. **Cluster correctly**: when items come in groups (several questions about one document, several turns of one conversation), they are not independent; use a cluster bootstrap or clustered standard errors, otherwise the interval is too narrow. Anthropic's "Adding Error Bars to Evals" (Miller, 2024) gives a practical treatment of these points.

```python
import numpy as np

rng = np.random.default_rng(0)

# scores[i, j] = score of item i on repeat j (e.g. 0/1), shape (n_items, k_repeats)
scores = np.load("scores.npy")
item_means = scores.mean(axis=1)

boot = [item_means[rng.integers(0, len(item_means), len(item_means))].mean() for _ in range(10_000)]
low, high = np.percentile(boot, [2.5, 97.5])
flaky = np.mean((scores.min(axis=1) == 0) & (scores.max(axis=1) == 1))

print(f"score={item_means.mean():.3f}  95% CI [{low:.3f}, {high:.3f}]  flaky items={flaky:.1%}")
```

**What interviewers look for**

- You attach an interval to every number and know roughly how big it is for a given $n$.
- You know temperature 0 is not a guarantee of determinism.
- You size the evaluation for the decision instead of using a round number.

### 24.29 Why do public benchmarks and leaderboards mislead, and how do you detect benchmark contamination?

Public benchmarks and leaderboards are useful for shortlisting models, but they **measure a narrow task under conditions that differ from yours, can leak into training data, saturate, and are optimised against**. A leader must know why a model that tops a leaderboard can lose on the company's own task, and must make model decisions on private, task-specific evaluations (see 24.12 for what the common benchmarks measure).

**Why benchmark scores mislead**

- **Contamination**: test questions (or close paraphrases) appear in web-scale pre-training data or in fine-tuning sets, so the model partly recalls answers. Scores then overstate generalisation.
- **Saturation**: when top models all score above 90 percent (as happened with MMLU, GSM8K and HumanEval), remaining differences are within noise and label error. Successors such as MMLU-Pro, GPQA, Humanity's Last Exam and SWE-bench Verified exist for this reason.
- **Label errors**: popular benchmarks contain wrong or ambiguous answers (MMLU-Redux re-annotated MMLU and found a measurable share of flawed questions). Near the ceiling, a model can be penalised for being right.
- **Evaluation setup differences**: prompt format, few-shot examples, chain-of-thought or not, answer extraction, number of samples and scaffolding can move scores by several points. Vendors' numbers are often not comparable with each other.
- **Benchmark overfitting**: models, prompts and training data mixes are tuned towards headline benchmarks, which improves the number more than the capability.
- **Distribution mismatch**: exam-style questions do not resemble your users' messy, domain-specific, multi-turn requests.
- **Missing dimensions**: latency, cost, refusal behaviour, format reliability and robustness rarely appear on leaderboards but often decide production fit.

**Preference leaderboards (arena-style) have their own issues**

- Votes reflect what crowd users prefer in short chats: style, length, formatting and confidence influence wins, which is why style-controlled rankings were introduced.
- The prompt mix is not your workload.
- Providers can test many private variants and publish the best (a concern raised by the 2025 "Leaderboard Illusion" analysis), and a model version tuned for the arena may differ from the one released, as in the 2025 controversy over an experimental Llama 4 variant.

**Detecting contamination**

| Method | Idea | Needs |
| --- | --- | --- |
| N-gram or substring overlap | Search training data for test items | Access to training data (model builders only) |
| Perturbed or fresh rewrites | Build a parallel set of new items of the same difficulty; a large score drop suggests memorisation (GSM1k showed drops for some model families on fresh GSM8K-style problems) | Effort to write items |
| Time-split evaluation | Use problems published after the model's training cutoff (the LiveCodeBench and LiveBench approach) | Dated items |
| Completion probing | Give the first half of a test item and see whether the model completes it verbatim, or reproduces answer options in order | Model access |
| Membership inference | Unusually high likelihood on test text, for example Min-K% Prob | Log-probabilities |
| Canary strings | Embed a unique marker in your own datasets so leakage into training can be detected later | Foresight |

For your own evaluations, prevention matters most: keep held-out sets private, never paste them into prompts or vendor tools that may train on them, and rotate in fresh items regularly.

**How to use public benchmarks well**

1. Use them to **shortlist** candidates, choosing benchmarks closest to your task (code, tool use, long context, your languages).
2. Prefer contamination-resistant benchmarks: recent, time-split, held-out or human-verified.
3. Reproduce key numbers yourself under one harness when the decision is significant.
4. Decide on a **private evaluation built from your own data**, including cost and latency at your expected volume (see 24.22).

**What interviewers look for**

- You treat a leaderboard as a weak prior, not a decision.
- You can name concrete mechanisms (contamination, saturation, setup sensitivity, style bias) and how to check for them.
- You protect your own eval sets from the same problems.

### 24.30 How do you evaluate long-form and open-ended outputs with rubrics?

For long-form output such as reports, summaries, research answers or generated documents there is no single right answer, so you define **what a good answer must contain and must avoid**, as a rubric of specific, checkable criteria, and grade each criterion separately. A good rubric turns "is this a good report?" into many small yes or no questions that humans and LLM judges can answer consistently.

**Holistic versus analytic rubrics**

| Type | How it works | When to use |
| --- | --- | --- |
| Holistic | One overall score (for example 1 to 5) with descriptions of each level | Quick human triage; poor for diagnosis and noisy for judges |
| Analytic | Separate scores per dimension (accuracy, coverage, structure, tone) | Diagnosis and targeted improvement |
| Checklist (binary criteria) | Many specific yes or no items, often per example | Most reliable for LLM judges and for release gates |

**Designing the rubric**

1. **Start from failures, not theory**: read 30 to 50 real outputs, write down what made the bad ones bad, and turn those into criteria.
2. **Make criteria atomic and observable**: "mentions the 30-day return window" or "does not recommend a specific medication dose" rather than "is accurate and helpful".
3. **Separate dimensions**: factual accuracy, coverage of required points, faithfulness to sources, structure and readability, tone, safety, length.
4. **Use anchors**: for any graded scale, give a short example of each level so graders share a reference point.
5. **Weight criteria by importance** and mark some as **critical** (any failure fails the whole output), such as a safety or compliance breach.
6. **Per-example criteria** where possible: alongside a general rubric, each golden item lists the points its answer must include and the errors it must avoid. HealthBench, for example, grades medical conversations against conversation-specific criteria written by physicians.

```text
Item: "Summarise the Q3 incident report for the leadership team"
Must include:      root cause; customer impact (duration, number affected); remediation status
Must not include:  blame of named individuals; speculation presented as fact
Critical:          no figures that contradict the source report
Quality:           fits on one page; leads with impact; plain language
```

**Grading**

- **One judge call per criterion** (or a small group), with the source document and the criterion, returning pass or fail plus a short quote as evidence. Asking for evidence reduces hallucinated verdicts and makes disagreements easy to review.
- Combine criterion results into a score with the weights, and report per-criterion pass rates, which show what to fix.
- Calibrate each judge criterion against human labels (see 24.7); some criteria (factual accuracy in a specialist domain) may need human or reference-based grading.
- **Control length**: long answers pass more "includes X" checks. Add precision-style criteria ("no irrelevant sections", "no unsupported claims") or a length budget.
- For comparisons between versions, pairwise judging per dimension is often more sensitive than absolute rubric scores.

**Long inputs and outputs**

- Judges degrade on very long texts: grade section by section, or extract the claims first and verify them against the source (see 24.10).
- For multi-source research outputs, check citation precision and coverage of the key sources.

**Maintaining rubrics**

- Version rubrics with the golden set; a rubric change resets comparability, so re-score the baseline.
- Review human-judge disagreements monthly; recurring disagreement on one criterion means it is ambiguous.

**Pitfalls**

- Vague criteria ("is insightful") that no two graders apply the same way.
- Too many criteria per call, so the judge skims.
- Rubrics that reward the style of the current system and penalise better but different answers.

### 24.31 Offline evaluation improved but online metrics did not. How do you design online evaluation for LLM features and debug the gap?

Offline evaluation tells you whether a change is better **on your test set by your metrics**; online evaluation tells you whether it is better **for real users on real traffic**. When they disagree, either the offline set or metric does not represent what users value, or the online measurement is flawed. The mechanics of running an A/B test for LLM systems are covered in 26.13; this answer covers what to measure online and how to diagnose disagreement.

**Online evaluation design for LLM features**

- **Primary metric tied to the user goal**: task completion, resolution without escalation, accepted suggestion rate (for code or writing assistants), time to complete a task.
- **Implicit quality signals**: retries and regenerations, rephrased follow-up questions, copy or accept events, edits to generated text (edit distance), abandonment.
- **Explicit feedback**: thumbs and ratings. Useful for finding bad cases, but sparse and biased, so do not use it as the main metric.
- **Automated online scoring**: judge-scored samples from each arm (faithfulness, policy compliance) so you see quality, not just engagement.
- **Guardrail metrics**: safety violations, hallucination rate on sampled traffic, latency p95, cost per request, escalation rate. A variant cannot win if a guardrail is breached.
- **Rollout sequence**: shadow mode (generate but do not show, then compare offline) before a small canary, then an A/B test, then gradual ramp (see 26.78).

**Why offline and online disagree**

| Cause | Symptom | Check |
| --- | --- | --- |
| Golden set unrepresentative | Gains on categories that are rare in production | Compare category mix of golden set versus traffic; reweight results |
| Metric misaligned with user value | Judge prefers longer, more thorough answers; users want short ones | Correlate judge scores with user outcomes per item |
| Latency or cost changed | Better answers, but slower, so users abandon | Segment outcomes by latency |
| Offline conditions differ | Frozen retrieval snapshot offline, live index online; different prompt assembly | Replay production traces through both versions |
| Novelty or learning effects | Early lift that fades, or a dip while users adapt | Look at outcomes by days since exposure |
| Effect too small to detect online | Offline gain of 1 point, online test underpowered | Power analysis before the test |
| Instrumentation or assignment bugs | Sample ratio mismatch, missing events in one arm | Sample ratio check, event volume per arm |
| Segment heterogeneity | Wins for new users, losses for power users | Slice by segment, language, intent |

**Debugging procedure**

1. **Validate the experiment**: sample ratio, logging completeness, exposure definition, enough power.
2. **Replay production traffic offline**: run both versions on a sample of real requests from the experiment period and score them with your offline evaluators. If the offline gain disappears, the golden set is the problem.
3. **Join item-level offline scores with online outcomes** for the same requests. A weak relationship means the offline metric does not predict what users care about.
4. **Read transcripts** from the losing segments; patterns are usually obvious within 50 examples.
5. **Fix the offline side**: add the missing categories to the golden set, adjust rubrics (for example add a conciseness criterion), and record how well offline scores predict online results over time.

**Long-term practice**

- Track, per launch, the offline delta and the online delta. Over many launches this tells you how far to trust offline gains and which offline metrics are good proxies.
- Treat online results as the ground truth for user value, and offline evaluation as the fast, cheap filter that decides what is worth testing online.

### 24.32 Your LLM provider is retiring the model you use. How do you prove the replacement is safe to ship?

Treat a model migration as a **change with unknown regressions**: assume the new model behaves differently in format, refusals, verbosity, tool use and cost, even if it is "better" on benchmarks. Prove equivalence or improvement on your own evaluations, adapt the prompts to the new model, and roll out gradually with a fallback. Plan it early, because deprecation dates are fixed and migrations take weeks.

**1. Inventory what depends on the model**

- Every prompt, chain and agent that calls it, the features they power and their risk level.
- Hidden dependencies: parsers that expect a certain format, few-shot examples tuned to the old model, judges that use the same model, fine-tuned variants, token limits and context sizes, cost budgets.

**2. Shortlist candidates**

- Same-family successor, competitors, or an open-weight model; consider price, latency, context length, data residency and contractual terms.
- Use public benchmarks only to shortlist (see 24.29).

**3. Evaluate on your own suites**

- Run the full golden, regression and adversarial sets for each candidate with several repeats, and compare with the current model using paired statistics (see 24.19).
- Look beyond the headline metric:

| Area | What often changes |
| --- | --- |
| Format | JSON validity, extra prose around output, markdown where plain text is expected |
| Behaviour | Verbosity, refusal and over-refusal rates, tone, willingness to say "I don't know" |
| Tools | Tool-selection accuracy, argument formatting, parallel calls, loop behaviour |
| Grounding | Faithfulness to retrieved context, citation behaviour |
| Safety | Jailbreak and injection resistance on your red-team suite |
| Operations | Latency (time to first token and total), tokens per answer (different tokenisers change cost), rate limits |

- **Per-item diff review**: list items where the old model passed and the new one fails, and read them. Aggregates can be flat while important cases flip.
- If the judge uses the retiring model, migrate and re-calibrate the judge **first**, separately, so you are not changing the ruler and the thing being measured at once.

**4. Adapt, then re-evaluate**

- Prompts tuned for one model are rarely optimal for another. Allow a prompt adaptation pass on the development set, then confirm on the held-out set.
- Fix parser and format issues with structured outputs rather than prompt pleading (see 26.30).

**5. Shadow and canary**

- **Shadow**: send a copy of real traffic to the new model, do not show results, and compare outputs with automated evaluators and spot reviews.
- **Canary or A/B**: a small share of users, with guardrail metrics and automatic rollback triggers (see 26.13 and 26.37).
- Keep the old model available as a fallback until the cutoff date, and keep a second provider route for critical features (see 26.40).

**6. Sign off and record**

- An evaluation report: candidates, scores with intervals, known behaviour differences, prompt changes and owner sign-off.
- Pin the new model version explicitly and add a daily canary run of the golden set to catch silent updates.

**Pitfalls**

- Swapping the model name in config and trusting a benchmark table.
- Evaluating with the old prompts only, then blaming the model.
- Ignoring cost per task: a cheaper price per token can still cost more if the model is more verbose or needs more agent steps.

### 24.33 How do you know whether your evaluation suite itself is any good?

An evaluation suite is a measuring instrument, so evaluate it like one: it should be **valid** (measures what users care about), **sensitive** (detects real regressions), **reliable** (gives the same answer twice) and **current** (reflects today's traffic and truth). Many teams have suites that pass every release and catch nothing; a senior engineer audits the suite as seriously as the system.

**1. Validity: does it predict what matters?**

- Compare suite scores with human expert judgements on the same items; for judge-based metrics this is calibration (see 24.7).
- Compare offline deltas with online outcomes across past launches (see 24.31). If offline gains never show up online, the suite measures the wrong thing.
- Check that critical requirements (safety, compliance, money-moving actions) each have explicit tests, not just general quality scores.

**2. Sensitivity: would it catch a regression?**

- **Mutation testing for evals**: deliberately break the system (remove the retrieval step, truncate context, swap in a weaker model, delete a key prompt instruction, inject a known bug) and check that the suite fails. If a degraded system still passes, the suite is blind to that failure.
- **Replay past incidents**: every production incident should have produced a test; check that the suite fails on the version that caused it.
- **Discrimination**: items that every version passes or every version fails carry no information. Track the share of saturated items and replace them with harder ones.

**3. Reliability: is the score stable?**

- Run the same system several times; the spread of the aggregate score and the share of flaky items should be small relative to the differences you act on (see 24.28).
- Measure inter-annotator agreement on human labels and test-retest agreement for judges.
- Audit labels: sample golden items and have a second expert check them. Wrong labels put a ceiling on measurable quality.

**4. Coverage: does it represent reality?**

- Compare the distribution of intents, languages, input lengths and user segments in the suite with recent production traffic.
- Check for missing categories: new features, new data sources, new user groups, multi-turn flows, adversarial inputs.
- Look at production failures that the suite would not have caught; each one reveals a gap.

**5. Practicality**

- Run time and cost per suite run. A suite that takes hours and costs hundreds of dollars per run will be skipped; split it into tiers (see 24.11).
- Clarity of reports: can a reviewer see what changed and why in a few minutes?

**A simple scorecard**

| Property | Indicator | Healthy sign |
| --- | --- | --- |
| Validity | Judge agreement with humans; offline-online correlation | Kappa near human-human level; offline gains mostly confirmed online |
| Sensitivity | Share of injected faults detected | Most deliberate breakages fail the suite |
| Discrimination | Share of saturated items | Small, and reviewed each quarter |
| Reliability | Run-to-run spread; label error rate | Spread well below the gating tolerance |
| Coverage | Distribution gap versus production | No important category missing |
| Freshness | Age of newest items; incidents converted to tests | Items added every month |

**What interviewers look for**

- You know a green dashboard can mean a blind evaluation, and you test the evaluation deliberately.
- You connect the evaluation to outcomes rather than treating it as an end in itself.
- You assign ownership and a review cadence to the suite, as for any production system.

---

← [Chapter 23. Multimodal AI](23-multimodal-ai.md) · [Table of contents](../README.md) · [Chapter 25. Backend Engineering, APIs and DevOps](../part-5-production-systems/25-backend-engineering-apis-and-devops.md) →
