# Chapter 22. Fine-Tuning and Model Adaptation

_Part IV — Generative AI and LLM Engineering_

Supervised fine-tuning, parameter-efficient methods and LoRA variants, RLHF, DPO and related preference methods, RL for reasoning, training memory and debugging, serving adapters, safety, and when to fine-tune at all.

This chapter contains **41 questions**.

## Contents

- **22.1** [What is fine-tuning, and when should you fine-tune an LLM?](#221-what-is-fine-tuning-and-when-should-you-fine-tune-an-llm)
- **22.2** [Explain the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT).](#222-explain-the-difference-between-full-fine-tuning-and-parameter-efficient-fine-tuning-peft)
- **22.3** [What is LoRA (Low-Rank Adaptation), and how does it work?](#223-what-is-lora-low-rank-adaptation-and-how-does-it-work)
- **22.4** [What is QLoRA, and how does it enable fine-tuning on consumer hardware?](#224-what-is-qlora-and-how-does-it-enable-fine-tuning-on-consumer-hardware)
- **22.5** [How does fine-tuning work?](#225-how-does-fine-tuning-work)
- **22.6** [Explain Prefix Tuning and Prompt Tuning. How are they different from LoRA?](#226-explain-prefix-tuning-and-prompt-tuning-how-are-they-different-from-lora)
- **22.7** [What is adapter-based fine-tuning?](#227-what-is-adapter-based-fine-tuning)
- **22.8** [What is RLHF (Reinforcement Learning from Human Feedback), and how is it used to align LLMs?](#228-what-is-rlhf-reinforcement-learning-from-human-feedback-and-how-is-it-used-to-align-llms)
- **22.9** [What is instruction tuning, and why is it important for chat models?](#229-what-is-instruction-tuning-and-why-is-it-important-for-chat-models)
- **22.10** [How do you prepare a dataset for fine-tuning an LLM?](#2210-how-do-you-prepare-a-dataset-for-fine-tuning-an-llm)
- **22.11** [What is catastrophic forgetting, and how do you prevent it during fine-tuning?](#2211-what-is-catastrophic-forgetting-and-how-do-you-prevent-it-during-fine-tuning)
- **22.12** [How would you determine whether to fine-tune, RAG, prompt-engineer, or train from scratch?](#2212-how-would-you-determine-whether-to-fine-tune-rag-prompt-engineer-or-train-from-scratch)
- **22.13** [How do you evaluate a fine-tuned model's performance?](#2213-how-do-you-evaluate-a-fine-tuned-models-performance)
- **22.14** [What is synthetic data generation, and how do you use it for fine-tuning?](#2214-what-is-synthetic-data-generation-and-how-do-you-use-it-for-fine-tuning)
- **22.15** [What are the key hyperparameters for fine-tuning (learning rate, epochs, batch size, LoRA rank)?](#2215-what-are-the-key-hyperparameters-for-fine-tuning-learning-rate-epochs-batch-size-lora-rank)
- **22.16** [How do you fine-tune a model for a specific domain (legal, medical, finance)?](#2216-how-do-you-fine-tune-a-model-for-a-specific-domain-legal-medical-finance)
- **22.17** [What is continual pre-training, and when would you use it?](#2217-what-is-continual-pre-training-and-when-would-you-use-it)
- **22.18** [How do you merge multiple LoRA adapters?](#2218-how-do-you-merge-multiple-lora-adapters)
- **22.19** [What is the difference between SFT (Supervised Fine-Tuning) and alignment training?](#2219-what-is-the-difference-between-sft-supervised-fine-tuning-and-alignment-training)
- **22.20** [What is RLAIF (RL from AI Feedback), and how does it differ from RLHF?](#2220-what-is-rlaif-rl-from-ai-feedback-and-how-does-it-differ-from-rlhf)
- **22.21** [What is knowledge distillation for fine-tuning, and what are the legal considerations?](#2221-what-is-knowledge-distillation-for-fine-tuning-and-what-are-the-legal-considerations)
- **22.22** [Your fine-tuned LLM produces factually wrong outputs due to training data quality issues. How do you fix it?](#2222-your-fine-tuned-llm-produces-factually-wrong-outputs-due-to-training-data-quality-issues-how-do-you-fix-it)
- **22.23** [You must choose between LoRA and full fine-tuning for a domain-specific assistant. How do you decide?](#2223-you-must-choose-between-lora-and-full-fine-tuning-for-a-domain-specific-assistant-how-do-you-decide)
- **22.24** [Your fine-tuned model memorized training data verbatim instead of learning patterns. How do you fix overfitting?](#2224-your-fine-tuned-model-memorized-training-data-verbatim-instead-of-learning-patterns-how-do-you-fix-overfitting)
- **22.25** [Your fine-tuned LLM forgot its general capabilities after domain-specific fine-tuning. How do you fix catastrophic forgetting?](#2225-your-fine-tuned-llm-forgot-its-general-capabilities-after-domain-specific-fine-tuning-how-do-you-fix-catastrophic-forgetting)
- **22.26** [Your RLHF preference data has low annotator agreement. How do you ensure data quality?](#2226-your-rlhf-preference-data-has-low-annotator-agreement-how-do-you-ensure-data-quality)
- **22.27** [What is the difference between SFT and RL-based tuning (RLHF/DPO)?](#2227-what-is-the-difference-between-sft-and-rl-based-tuning-rlhfdpo)
- **22.28** [How would you fine-tune an LLM while preserving its general capabilities?](#2228-how-would-you-fine-tune-an-llm-while-preserving-its-general-capabilities)
- **22.29** [Explain RLHF, PPO, DPO, and reward hacking.](#2229-explain-rlhf-ppo-dpo-and-reward-hacking)
- **22.30** [What is RLVR (Reinforcement Learning with Verifiable Rewards)?](#2230-what-is-rlvr-reinforcement-learning-with-verifiable-rewards)
- **22.31** [What is RLCD (Reinforcement Learning from Contrastive Distillation)?](#2231-what-is-rlcd-reinforcement-learning-from-contrastive-distillation)
- **22.32** [How do you estimate GPU memory and compute for fine-tuning an LLM?](#2232-how-do-you-estimate-gpu-memory-and-compute-for-fine-tuning-an-llm)
- **22.33** [How do loss masking and sequence packing work in supervised fine-tuning?](#2233-how-do-loss-masking-and-sequence-packing-work-in-supervised-fine-tuning)
- **22.34** [Which LoRA variants and settings matter in practice: target modules, rank, alpha, DoRA and others?](#2234-which-lora-variants-and-settings-matter-in-practice-target-modules-rank-alpha-dora-and-others)
- **22.35** [How do you serve many fine-tuned LoRA adapters efficiently?](#2235-how-do-you-serve-many-fine-tuned-lora-adapters-efficiently)
- **22.36** [How do you fine-tune a model for reliable tool calling (function calling)?](#2236-how-do-you-fine-tune-a-model-for-reliable-tool-calling-function-calling)
- **22.37** [How do you choose among preference optimisation methods such as DPO, IPO, KTO, ORPO and SimPO?](#2237-how-do-you-choose-among-preference-optimisation-methods-such-as-dpo-ipo-kto-orpo-and-simpo)
- **22.38** [How would you train a model to reason better using reinforcement learning with verifiable rewards?](#2238-how-would-you-train-a-model-to-reason-better-using-reinforcement-learning-with-verifiable-rewards)
- **22.39** [How does fine-tuning affect a model's safety behaviour, and how do you preserve it?](#2239-how-does-fine-tuning-affect-a-models-safety-behaviour-and-how-do-you-preserve-it)
- **22.40** [Should you fine-tune through a provider's API or fine-tune open-weight models yourself?](#2240-should-you-fine-tune-through-a-providers-api-or-fine-tune-open-weight-models-yourself)
- **22.41** [How do you debug a fine-tuning run that is not working?](#2241-how-do-you-debug-a-fine-tuning-run-that-is-not-working)

---

### 22.1 What is fine-tuning, and when should you fine-tune an LLM?

#### What is Fine-Tuning?

- **Fine-tuning** is the process of taking a **pre-trained LLM** and training it further on a smaller, task-specific dataset.
- It adjusts the model's weights so it becomes better at a particular **task, domain, behavior, or style**.
- Examples:
  - Customer-support conversations
  - Medical or legal terminology
  - Code generation for a specific framework
  - Following a company's response format
  - Classification or extraction tasks

#### When should you fine-tune an LLM?

Fine-tune when you need **consistent behavior or domain-specific capabilities** that prompting or RAG cannot reliably provide.

- **Domain specialization:** The model needs strong knowledge of specialized language, terminology, or patterns.
- **Consistent output format:** You need reliable structured outputs or a specific response style.
- **Task specialization:** The model repeatedly performs one specific task, such as classification, extraction, or code generation.
- **Behavior/style adaptation:** You want the model to consistently follow a particular tone, workflow, or instruction style.
- **Reduce prompting:** Fine-tuning can encode repeated instructions into the model, reducing complex prompts.
- **Improve performance/cost:** A smaller fine-tuned model may perform a specialized task as well as a larger general-purpose model at lower inference cost.

#### When NOT to fine-tune

- **Changing or frequently updated knowledge:** Use **RAG** instead.
- **One-off tasks:** Use prompting.
- **Need factual knowledge from private documents:** Usually use RAG rather than embedding the knowledge into model weights.
- **Insufficient training data:** Fine-tuning generally needs enough high-quality examples.
- **You haven't established a baseline:** First try prompting/RAG and evaluate whether fine-tuning is actually necessary.

### 22.2 Explain the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT).

#### Full Fine-Tuning vs PEFT

| Aspect | Full Fine-Tuning | PEFT |
| --- | --- | --- |
| **Parameters updated** | All/most model parameters | Only a small subset of parameters |
| **GPU memory** | Very high | Much lower |
| **Training cost** | Expensive | Cheaper |
| **Training speed** | Slower | Faster |
| **Storage** | Requires a full model copy per task | Stores only small adapter weights |
| **Risk of catastrophic forgetting** | Higher | Generally lower |
| **Typical use** | Large-scale specialization with substantial resources | Most practical LLM customization |

#### Full Fine-Tuning

- Updates **the model's existing weights** using task-specific training data.
- Can significantly change the model's behavior and capabilities.
- Requires substantial **GPU memory, compute, and training data**.
- Each specialized model typically requires storing another full set of weights.

#### PEFT — Parameter-Efficient Fine-Tuning

- Keeps most of the original model **frozen** and trains only a small number of additional or selected parameters.
- Popular methods include:
  - **LoRA:** learns low-rank weight updates.
  - **QLoRA:** applies LoRA while keeping the base model quantized.
  - **Adapters:** add small trainable modules to the model.
  - **Prompt/Prefix Tuning:** learns trainable representations that influence the model's behavior.
- Multiple task-specific adapters can share the same base model.

### 22.3 What is LoRA (Low-Rank Adaptation), and how does it work?

#### What is LoRA?

- **LoRA (Low-Rank Adaptation)** is a **PEFT technique** for fine-tuning large language models efficiently.
- Instead of updating the model's original weights, LoRA **freezes the pre-trained weights** and learns a small set of additional parameters.
- It is especially useful when fine-tuning large LLMs with limited GPU memory.

#### How does LoRA work?

For an existing weight matrix **$W$**, LoRA keeps **$W$ frozen** and learns a low-rank update:

$$
W' = W + \Delta W
$$

Instead of directly learning the large **$\Delta W$**, LoRA represents it as:

$$
\Delta W = BA
$$

where **$A$** and **$B$** are much smaller matrices.

During training:

1. **Freeze the original LLM weights.**
2. Insert LoRA adapters into selected layers, commonly the **attention projections** such as Q, K, V, or O.
3. Train only the small **$A$ and $B$ matrices**.
4. During inference, the LoRA update is combined with the original weights.

#### Why is LoRA efficient?

- **Far fewer trainable parameters**
- **Lower GPU memory usage**
- **Faster training**
- **Smaller fine-tuning checkpoints**
- Multiple LoRA adapters can be trained for different tasks while sharing the same base model.

#### Simple example

Instead of fine-tuning a **7B-parameter model**, you might freeze all 7B parameters and train only a small fraction of additional LoRA parameters.

### 22.4 What is QLoRA, and how does it enable fine-tuning on consumer hardware?

#### What is QLoRA?

- **QLoRA (Quantized Low-Rank Adaptation)** combines **quantization** with **LoRA** to fine-tune large LLMs using much less GPU memory.
- The **base model is loaded in low-bit precision**, commonly **4-bit**, while LoRA adapters are trained in higher precision.
- The original quantized model weights remain **frozen**; only the small LoRA parameters are updated.

#### How does QLoRA work?

1. **Quantize the base LLM** to 4-bit precision.
2. **Freeze** the quantized base-model weights.
3. Add **LoRA adapters** to selected layers.
4. Train only the LoRA parameters.
5. During the forward/backward pass, the quantized weights are used while gradients update the LoRA adapters.

A key technique in QLoRA is **NF4 (NormalFloat 4-bit)** quantization, designed for neural-network weights and particularly effective for normally distributed weights.

#### Why can it run on consumer hardware?

The biggest memory savings come from storing the huge base model in **4-bit instead of 16-bit or 32-bit precision**.

For example, a **7B-parameter model** requires roughly:

- FP32: **\~28 GB** just for weights
- FP16/BF16: **\~14 GB**
- 4-bit: **\~3.5 GB**

Actual training memory is higher because of activations, gradients, optimizer states, and other overhead, but QLoRA avoids maintaining full-precision trainable copies of the entire model.

#### LoRA vs QLoRA

- **LoRA:** Base model stays in normal precision; train small LoRA adapters.
- **QLoRA:** Base model is **quantized**, then train small LoRA adapters.
- **Result:** QLoRA significantly reduces memory requirements while retaining much of the performance of full fine-tuning.

### 22.5 How does fine-tuning work?

Fine-tuning takes a **pre-trained LLM** and trains it further on a smaller, task-specific dataset.

1. **Start with a pre-trained model**
   - The model already understands language, patterns, and general knowledge.
2. **Prepare a fine-tuning dataset**
   - Provide examples of the desired behavior.
   - Example: instruction → ideal response.
3. **Tokenize the data**
   - Text is converted into tokens that the model can process.
4. **Forward pass**
   - The model generates predictions from the input.
5. **Calculate the loss**
   - Compare the model's prediction with the target output.
   - The loss measures how far the prediction is from the desired answer.
6. **Backpropagation**
   - Calculate how each trainable parameter contributed to the error.
7. **Update parameters**
   - An optimizer adjusts the model's trainable weights to reduce the loss.
8. **Repeat**
   - The process runs over many batches and epochs until the model learns the desired behavior.

#### Full Fine-Tuning vs PEFT

- **Full fine-tuning:** Updates most or all model weights.
- **LoRA/PEFT:** Keeps the base model frozen and updates only a small number of additional parameters.

#### Simple Flow

**Pre-trained LLM → Training Data → Forward Pass → Loss → Backpropagation → Parameter Updates → Fine-Tuned LLM**

### 22.6 Explain Prefix Tuning and Prompt Tuning. How are they different from LoRA?

All three are **Parameter-Efficient Fine-Tuning (PEFT)** methods. They keep the original LLM mostly or completely frozen and train a small number of parameters.

#### 1. Prompt Tuning

- Learns a small set of **trainable virtual token embeddings**.
- These embeddings are added to the input prompt.
- The LLM's original weights remain **frozen**.
- The learned tokens are not necessarily readable words; they are optimized vectors.
- Very parameter-efficient.

**Example:**

`[Learned tokens] + "Translate this sentence to French."`

The learned tokens steer the model toward the desired task.

#### 2. Prefix Tuning

- Similar to prompt tuning, but more powerful.
- Learns **trainable prefix vectors** that are injected into the Transformer layers, particularly the attention mechanism.
- The base model remains frozen.
- The learned prefix influences the model throughout its processing rather than only at the input.

#### 3. How are they different from LoRA?

| Method | What is trained? | Where is it added? |
| --- | --- | --- |
| **Prompt Tuning** | Virtual token embeddings | Input |
| **Prefix Tuning** | Trainable prefix vectors | Transformer attention layers |
| **LoRA** | Low-rank weight updates | Model weight matrices |

**Key distinction:**

- **Prompt Tuning:** "Learn how to modify the input."
- **Prefix Tuning:** "Learn information that guides attention throughout the model."
- **LoRA:** "Learn a small update to the model's weights."

### 22.7 What is adapter-based fine-tuning?

#### What is Adapter-Based Fine-Tuning?

- **Adapter-based fine-tuning** is a **Parameter-Efficient Fine-Tuning (PEFT)** technique.
- Instead of modifying the entire pre-trained LLM, you **freeze the original model weights** and insert small trainable **adapter modules** into selected Transformer layers.
- Only the adapter parameters are trained.

#### How does it work?

1. Start with a **pre-trained LLM**.
2. Freeze the model's original weights.
3. Insert small adapter layers into Transformer blocks.
4. Train only these adapters on task-specific data.
5. During inference, the adapters work together with the frozen base model.

A typical adapter uses a **bottleneck architecture**:

**Hidden State → Down-Projection → Nonlinearity → Up-Projection → Residual Connection**

The down-projection reduces the dimensionality, so the adapter contains far fewer parameters than the original layer.

#### Advantages

- Much lower **memory and compute requirements** than full fine-tuning.
- Small task-specific checkpoints.
- Multiple adapters can share one base model.
- Easy to switch between different tasks by loading different adapters.
- Reduces the risk of changing the original model's behavior.

#### Adapter vs LoRA

- **Adapters:** Add new trainable neural-network modules to the Transformer.
- **LoRA:** Learns low-rank updates to existing weight matrices.
- Both **freeze the base model** and train only a small number of parameters.

### 22.8 What is RLHF (Reinforcement Learning from Human Feedback), and how is it used to align LLMs?

#### What is RLHF?

- **RLHF (Reinforcement Learning from Human Feedback)** is a technique for aligning an LLM's behavior with **human preferences**.
- Instead of optimizing only for predicting the next token, the model is trained to produce responses that humans prefer: helpful, safe, relevant, and instruction-following.

#### How does RLHF work?

1. **Pre-training**
   - Train the LLM on large amounts of text using next-token prediction.
   - This gives the model general language and knowledge capabilities.
2. **Supervised Fine-Tuning (SFT)**
   - Train the model on high-quality instruction–response examples.
   - This teaches it to follow instructions.
3. **Collect human preferences**
   - Generate multiple responses to the same prompt.
   - Human annotators **rank the responses** from best to worst.
4. **Train a Reward Model**
   - Train a separate model using these human rankings.
   - The reward model learns to predict which responses humans would prefer.
5. **Reinforcement Learning**
   - Use an algorithm such as **PPO (Proximal Policy Optimization)** to optimize the LLM.
   - The LLM generates responses, the reward model scores them, and the LLM's parameters are updated to increase the expected reward.

#### Simple Flow

**Pre-training → SFT → Human Preferences → Reward Model → PPO → Aligned LLM**

#### Why RLHF is useful

It helps improve:

- **Instruction following**
- **Helpfulness**
- **Response quality**
- **Safety**
- **Human preference alignment**
- **Reducing undesirable behaviors**

### 22.9 What is instruction tuning, and why is it important for chat models?

> **See also:** [Question 17.111 — What is instruction tuning, and why is it important for chat models?](17-large-language-model-foundations.md#17111-what-is-instruction-tuning-and-why-is-it-important-for-chat-models), which covers the same question from another angle.

#### What is Instruction Tuning?

- **Instruction tuning** is a form of **supervised fine-tuning (SFT)** where an LLM is trained on many **instruction–response examples**.
- It teaches the model to understand natural-language instructions and produce appropriate responses.
- The base model is trained primarily to **predict the next token**, while instruction tuning teaches it to **follow user intent**.

#### How does it work?

1. Collect instruction–response pairs.
   - Example: `Instruction: Summarize this text`
   - `Response: A concise summary...`
2. Fine-tune the pre-trained LLM on these examples.
3. The model learns patterns for following different types of instructions.
4. The resulting model becomes better at conversational and task-oriented interactions.

#### Why is it important for chat models?

Instruction tuning improves:

- **Instruction following**
- **Conversational behavior**
- **Task generalization**
- **Response relevance**
- **Ability to follow formatting requirements**
- **Zero-shot and few-shot performance**

It is typically an important stage **before preference-alignment methods such as RLHF**.

#### Pre-training vs Instruction Tuning

- **Pre-training:** Learn general language patterns and knowledge from large-scale text.
- **Instruction tuning:** Learn how to **respond appropriately to instructions**.

### 22.10 How do you prepare a dataset for fine-tuning an LLM?

1. **Define the objective**
   - Decide what behavior or task you want to improve.
   - Examples: instruction following, classification, summarization, extraction, coding, or domain-specific responses.
2. **Collect high-quality examples**
   - Gather representative **instruction–input–output** examples.
   - Use reliable and relevant data.
   - Quality is generally more important than simply having a large dataset.
3. **Clean the data**
   - Remove duplicates, irrelevant samples, corrupted text, spam, and low-quality responses.
   - Normalize formatting where appropriate.
4. **Remove sensitive or unsafe information**
   - Remove unnecessary PII, credentials, confidential information, and problematic content.
   - Check licensing and data-use rights.
5. **Format the dataset**
   - Convert examples into the format expected by the training framework/model.
   - For chat models, a common structure is:
     - `system`
     - `user`
     - `assistant`
6. **Ensure consistency**
   - Use consistent instructions, response formats, terminology, and style.
   - Make sure examples demonstrate the behavior you actually want the model to learn.
7. **Tokenize and check sequence length**
   - Tokenize the examples using the model's tokenizer.
   - Identify samples that exceed the model's context window and handle them appropriately.
8. **Split the dataset**
   - Create **training**, **validation**, and optionally **test** sets.
   - Avoid leakage or near-duplicate examples across splits.
9. **Balance the dataset**
   - Ensure important tasks, classes, domains, and difficulty levels are adequately represented.
   - Avoid allowing one type of example to dominate unnecessarily.
10. **Validate before training**

- Manually inspect samples.
- Check formatting, label correctness, duplicates, token lengths, and data distribution.

#### Simple Pipeline

**Define objective → Collect → Clean → Deduplicate → Validate → Format → Tokenize → Split → Quality check → Fine-tune**

### 22.11 What is catastrophic forgetting, and how do you prevent it during fine-tuning?

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

### 22.12 How would you determine whether to fine-tune, RAG, prompt-engineer, or train from scratch?

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

### 22.13 How do you evaluate a fine-tuned model's performance?

- **Define evaluation metrics**
  - Choose metrics based on the task.
  - Classification: **Accuracy, Precision, Recall, F1**
  - Generation: **BLEU, ROUGE, BERTScore**
  - LLM/chat: **helpfulness, correctness, relevance, safety, instruction following**
- **Use a held-out test set**
  - Keep evaluation data separate from training data.
  - Avoid data leakage and duplicates.
- **Compare against a baseline**
  - Compare the fine-tuned model against:
    - Original base model
    - Prompt-engineered baseline
    - Existing production model
- **Evaluate task performance**
  - Measure whether fine-tuning actually improves the target task.
  - Use domain-specific test cases that represent real-world usage.
- **Perform human evaluation**
  - Have humans compare responses for qualities such as correctness, relevance, helpfulness, and style.
  - Pairwise comparison is often useful for LLM outputs.
- **Check general capabilities**
  - Make sure fine-tuning hasn't caused **catastrophic forgetting**.
  - Test general reasoning, instruction following, and other important capabilities.
- **Evaluate safety and robustness**
  - Test harmful or adversarial prompts, unexpected inputs, hallucination, and edge cases.
  - Check whether the model behaves consistently outside the training distribution.
- **Measure production metrics**
  - Monitor **latency, throughput, token usage, cost, failure rate**, and user feedback.

### 22.14 What is synthetic data generation, and how do you use it for fine-tuning?

#### What is Synthetic Data Generation?

- **Synthetic data generation** is the process of creating artificial training examples using an LLM, simulator, templates, or other generative methods instead of collecting every example from real-world data.
- For LLM fine-tuning, a stronger model can generate **instruction–response pairs** that are then used to train another model.

#### How is it used for fine-tuning?

1. **Define the target task**
   - Example: customer support, SQL generation, document extraction, or code generation.
2. **Create seed examples**
   - Provide a small set of high-quality examples or detailed instructions.
3. **Generate synthetic examples**
   - Use a capable LLM to generate diverse inputs and corresponding ideal responses.
4. **Filter and validate**
   - Remove incorrect, duplicated, low-quality, or unsafe examples.
   - Use rule-based checks, another model, or human review.
5. **Fine-tune the target model**
   - Train the model on the filtered synthetic dataset.
6. **Evaluate on real-world data**
   - Use a separate real test set to verify that synthetic-data training actually improves performance.

#### Why use synthetic data?

- **Scales data collection**
- Creates examples for **rare or difficult scenarios**
- Provides consistent **instruction-response formats**
- Can reduce the need for expensive manual annotation
- Helps create targeted examples for specific behaviors

#### Main risks

- **Model-generated errors** can become training data.
- **Biases** from the generating model can be transferred.
- Low diversity can cause **overfitting or reduced generalization**.
- Training repeatedly on model-generated content can cause quality degradation.

### 22.15 What are the key hyperparameters for fine-tuning (learning rate, epochs, batch size, LoRA rank)?

#### Key Hyperparameters for Fine-Tuning

| Hyperparameter | What it controls | Typical consideration |
| --- | --- | --- |
| **Learning Rate** | Size of each parameter update | Too high → unstable/forgetting; too low → slow learning |
| **Epochs** | Number of times the model sees the training data | Too many → overfitting; too few → underfitting |
| **Batch Size** | Number of samples processed before an update | Larger → more stable gradients but higher memory usage |
| **LoRA Rank (r)** | Capacity of the LoRA adapter | Higher → more expressive but more parameters/memory |

#### 1. Learning Rate

- One of the **most important hyperparameters**.
- Controls how aggressively model parameters are updated.
- For LLM fine-tuning, learning rates are generally much lower than those used when training a model from scratch.
- With LoRA, a somewhat higher learning rate can often be used because only the adapter parameters are being trained.

**Too high:** unstable training, degraded model behavior, catastrophic forgetting.

**Too low:** slow learning or insufficient adaptation.

#### 2. Epochs

- An **epoch** is one complete pass through the training dataset.
- More epochs allow the model to learn the task more deeply.
- Too many epochs can cause **overfitting**, especially with small datasets.
- Monitor validation performance and use **early stopping** when appropriate.

#### 3. Batch Size

- Determines how many training examples contribute to each gradient update.
- Larger batches generally provide more stable gradient estimates but require more memory.
- If GPU memory is limited, use **gradient accumulation** to achieve a larger effective batch size.

$$
\text{Effective batch size} \approx \text{batch size} \times \text{gradient accumulation steps} \times \text{number of GPUs}
$$

#### 4. LoRA Rank

- **LoRA rank ($r$)** controls the size/capacity of the low-rank update.
- **Lower rank:** fewer parameters, lower memory usage, but potentially less adaptation capacity.
- **Higher rank:** more parameters and capacity, but higher memory/compute requirements.
- Common values include **4, 8, 16, 32, or 64**, depending on the task and model.

#### Other Important Hyperparameters

- **LoRA alpha:** scales the LoRA update.
- **LoRA dropout:** regularizes the adapter.
- **Weight decay:** helps reduce overfitting.
- **Warmup ratio/steps:** gradually increases the learning rate at the beginning.
- **Optimizer:** commonly AdamW or variants.
- **Maximum sequence length:** determines how much context is used during training.

### 22.16 How do you fine-tune a model for a specific domain (legal, medical, finance)?

For domains such as **legal, medical, or finance**, I would use a controlled pipeline:

1. **Define the domain task**
   - Determine what the model should improve.
   - Examples: legal document classification, medical summarization, financial report extraction.
2. **Collect domain-specific data**
   - Use high-quality, representative domain documents and instruction–response examples.
   - Ensure the data is legally usable and appropriately licensed.
3. **Clean and preprocess**
   - Remove duplicates, corrupted data, irrelevant content, and unnecessary sensitive information.
   - Normalize terminology and formatting.
4. **Create instruction-tuning examples**
   - Convert domain knowledge into examples such as:
     - User question → expert answer
     - Document → summary
     - Document → structured extraction
     - Scenario → recommended analysis
5. **Use domain experts for validation**
   - Especially important for **medical and legal** applications.
   - Experts should verify correctness, terminology, and safety.
6. **Choose the adaptation method**
   - Start with **LoRA/QLoRA** when practical.
   - Use full fine-tuning only when there is a strong reason and sufficient data/compute.
7. **Fine-tune and validate**
   - Tune learning rate, epochs, batch size, LoRA rank, etc.
   - Monitor validation loss and domain-specific metrics.
8. **Evaluate against a baseline**
   - Compare the fine-tuned model with the original model and a RAG/prompting baseline.
   - Test factual accuracy, hallucination, robustness, and domain-specific performance.
9. **Use RAG for changing knowledge**
   - Fine-tuning teaches the model **how to behave in the domain**.
   - RAG provides **current or private domain knowledge**.
   - For example, a legal assistant might be fine-tuned for legal reasoning/output style while RAG retrieves the latest laws and case documents.
10. **Add domain-specific safety controls**

- Medical and legal systems require stronger validation, access controls, monitoring, and human oversight.
- The model should not be treated as a substitute for qualified professionals in high-stakes decisions.

#### Example: Finance LLM

**Financial documents → Clean & validate → Instruction dataset → QLoRA fine-tuning → Domain evaluation → RAG for current financial data → Production monitoring**

### 22.17 What is continual pre-training, and when would you use it?

#### What is Continual Pre-Training?

- **Continual pre-training (CPT)** is the process of taking an already pre-trained LLM and **continuing its pre-training on additional unlabeled data**.
- Unlike supervised fine-tuning, the data does not need instruction–response pairs.
- The model typically continues using the same **next-token prediction objective**.

#### How does it work?

**Pre-trained LLM → Domain-specific corpus → Continued next-token training → Domain-adapted LLM → Instruction tuning**

For example:

- Start with a general-purpose LLM.
- Continue training it on a large corpus of **legal documents**.
- The model becomes more familiar with legal terminology, writing patterns, and domain-specific language.
- You can then instruction-tune it for specific legal tasks.

#### When would you use it?

Use continual pre-training when:

- The model lacks familiarity with a **specialized domain's language**.
- You have a **large amount of high-quality unlabeled domain text**.
- You want deeper domain adaptation than instruction tuning alone provides.
- You are adapting a general model to areas such as **law, medicine, finance, science, or programming**.
- The domain has specialized terminology, syntax, or writing patterns.

#### Continual Pre-Training vs Fine-Tuning

- **Continual pre-training:** Learns from large amounts of **unlabeled text** and improves domain knowledge/language understanding.
- **Instruction fine-tuning:** Learns from **labeled instruction–response examples** and improves task following and behavior.

### 22.18 How do you merge multiple LoRA adapters?

#### How do you merge multiple LoRA adapters?

- **LoRA adapters** contain low-rank weight updates that can be combined with the same base model.
- Merging means combining the adapters' updates into a **single adapter or the base model**.

#### Common approaches

1. **Weighted adapter merging**
   - Assign each adapter a weight and combine their updates.
   - Conceptually:

   ```math
   \Delta W = \alpha_1 \Delta W_1 + \alpha_2 \Delta W_2 + \cdots + \alpha_n \Delta W_n
   ```

   - This allows you to control how strongly each adapter contributes.
2. **Merge into the base model**
   - Combine the LoRA updates with the original model weights:

   ```math
   W' = W + \Delta W
   ```

   - The resulting model no longer needs the separate LoRA adapter for inference.
3. **Keep adapters separate**
   - Instead of permanently merging them, some frameworks allow multiple adapters to remain separate and activate or combine them dynamically.
   - Useful when you want to switch between domains or tasks.

#### Example

Suppose you have:

- **Legal LoRA**
- **Finance LoRA**
- **Medical LoRA**

You could combine their updates with appropriate weights to create a model influenced by multiple domains.

#### Important considerations

- Adapters should generally be trained from the **same or compatible base model**.
- Combining unrelated adapters can cause **parameter conflicts** or degrade performance.
- The adapter weights should be tuned and evaluated rather than assuming equal weighting works.
- After merging, evaluate the model on **each target capability** and general capabilities.

### 22.19 What is the difference between SFT (Supervised Fine-Tuning) and alignment training?

#### SFT vs Alignment Training

| Aspect | SFT | Alignment Training |
| --- | --- | --- |
| **Goal** | Teach the model to follow instructions and perform tasks | Make model behavior match human preferences and values |
| **Training signal** | Demonstrations / correct answers | Preferences, rankings, rewards, or critiques |
| **Typical data** | Instruction → response | Response A preferred over Response B |
| **Examples** | Instruction tuning, supervised fine-tuning | RLHF, DPO, preference optimization |
| **Main focus** | Capability and instruction following | Helpfulness, safety, style, and preference alignment |

#### SFT — Supervised Fine-Tuning

- Trains the model using **high-quality input-output examples**.
- The model learns to reproduce desirable responses.
- Commonly used to teach:
  - Instruction following
  - Specific tasks
  - Output formats
  - Domain-specific behavior
- Example:

**Instruction:** "Summarize this document."

**Target:** "The document discusses..."

The model learns from the target response.

#### Alignment Training

- Goes beyond simply showing the model correct answers.
- Teaches the model **which behaviors or responses humans prefer**.
- Common approaches include:
  - **RLHF:** Uses human preferences and a reward model.
  - **DPO:** Directly optimizes preference pairs without a separate RL loop.
  - **RLAIF:** Uses AI-generated feedback instead of exclusively human feedback.

#### Typical LLM Training Pipeline

**Pre-training → SFT → Alignment Training**

- **Pre-training:** Learn general language and knowledge.
- **SFT:** Learn to follow instructions.
- **Alignment:** Learn to produce responses that better match desired human preferences and safety requirements.

### 22.20 What is RLAIF (RL from AI Feedback), and how does it differ from RLHF?

#### What is RLAIF?

- **RLAIF (Reinforcement Learning from AI Feedback)** is an alignment technique where an **AI model provides feedback** on LLM outputs instead of relying primarily on human annotators.
- The AI feedback can be used to create **preference data** or train a **reward model**, which is then used to optimize the target LLM.

#### How does RLAIF work?

1. Start with a pre-trained and usually instruction-tuned LLM.
2. Generate multiple responses to prompts.
3. A **judge/critic AI model** evaluates or ranks the responses.
4. Use those AI preferences to create a reward signal or preference dataset.
5. Optimize the target model using methods such as **RL or preference optimization**.
6. Evaluate the resulting model with human evaluation and safety tests.

#### RLAIF vs RLHF

| Aspect | RLHF | RLAIF |
| --- | --- | --- |
| **Feedback source** | Humans | AI model |
| **Cost** | Higher | Usually lower |
| **Scalability** | Limited by human annotation | Highly scalable |
| **Consistency** | Can vary between annotators | More consistent |
| **Potential issue** | Human bias | AI/judge bias and errors |

#### Example

Suppose an LLM generates two answers:

- **Response A:** Accurate but overly verbose
- **Response B:** Accurate, concise, and directly answers the question

In **RLHF**, human annotators choose B.

In **RLAIF**, an AI judge evaluates A and B and selects B based on predefined criteria.

#### Key challenge

RLAIF can **amplify the biases or mistakes of the feedback model**. Therefore, human evaluation is still important for validating the final model and the AI feedback process.

### 22.21 What is knowledge distillation for fine-tuning, and what are the legal considerations?

#### What is Knowledge Distillation?

- **Knowledge distillation** is a technique where a smaller **student model** learns from a larger, more capable **teacher model**.
- Instead of learning only from hard labels, the student learns to reproduce the teacher's **outputs or probability distribution**.
- For LLMs, the teacher can generate high-quality responses, reasoning traces where appropriate, rankings, or probability/logit information, which are then used to train the student.

#### How is it used for fine-tuning?

A common pipeline is:

**Large Teacher LLM → Generate/score domain examples → Filter & validate → Fine-tune Student LLM → Evaluate**

For example:

- Teacher: large general-purpose LLM
- Student: smaller 7B model
- Dataset: domain-specific questions and high-quality teacher answers
- Fine-tuning: train the student to produce responses similar to the teacher.

This can produce a **smaller, cheaper, lower-latency model** while retaining much of the teacher's useful behavior.

#### Knowledge Distillation vs Regular Fine-Tuning

- **Fine-tuning:** Usually learns from human-created target examples.
- **Distillation:** Learns from the behavior or outputs of a **teacher model**.
- They can also be combined: use teacher-generated synthetic data and then fine-tune the student.

#### Legal Considerations

This is particularly important when the teacher is a **commercial or third-party model**.

- **Terms of service:** Check whether the teacher provider permits using its outputs to train another model. Some providers restrict or prohibit this.
- **Copyright:** Generated outputs can potentially raise copyright issues depending on the jurisdiction, source material, and how the outputs are used.
- **Training-data rights:** If your distillation dataset contains copyrighted, licensed, confidential, or proprietary material, you need appropriate rights to use it.
- **Privacy:** Don't include personal or confidential information unnecessarily. Consider privacy obligations when processing user data.
- **Trade secrets/confidential information:** Avoid using confidential enterprise data or proprietary model outputs in ways that violate agreements.
- **Model licenses:** Check both the teacher's terms and the student's license, including restrictions on commercial use or derivative models.
- **Regulated domains:** Medical, financial, and legal applications may have additional regulatory and professional obligations.
- **Data provenance:** Keep records of where training examples came from and what permissions apply.

### 22.22 Your fine-tuned LLM produces factually wrong outputs due to training data quality issues. How do you fix it?

#### How do you fix factual errors caused by poor training data?

1. **Identify the root cause**
   - Analyze incorrect outputs and trace them back to specific training examples.
   - Determine whether the problem is incorrect labels, outdated information, noisy data, duplication, or insufficient coverage.
2. **Audit and clean the dataset**
   - Remove incorrect and contradictory examples.
   - Remove duplicates and low-quality synthetic data.
   - Verify important facts against **trusted sources**.
   - Remove outdated information where appropriate.
3. **Improve data quality**
   - Add high-quality, verified examples.
   - Include difficult and edge cases.
   - Use domain experts for validation in specialized domains.
4. **Retrain or fine-tune again**
   - Fine-tune on the corrected dataset.
   - Use appropriate learning rates and fewer epochs to avoid further degradation.
   - Consider **LoRA/PEFT** if appropriate.
5. **Use RAG for factual knowledge**
   - If the issue is missing or frequently changing knowledge, don't rely solely on fine-tuning.
   - Use **RAG** to retrieve authoritative information at inference time.
   - This is particularly useful for policies, regulations, documentation, and current information.
6. **Add evaluation and regression tests**
   - Build a curated factual evaluation set.
   - Measure factual accuracy before and after retraining.
   - Add previously failed cases as regression tests.
7. **Monitor production**
   - Track hallucinations, user feedback, and factual errors.
   - Periodically refresh the data and evaluation set.

#### Key principle

**Fine-tuning teaches the model behavior; it should not be treated as a reliable database.**

For frequently changing or authoritative facts, **RAG + source verification** is generally preferable to repeatedly fine-tuning the model.

### 22.23 You must choose between LoRA and full fine-tuning for a domain-specific assistant. How do you decide?

#### LoRA vs Full Fine-Tuning: How do you decide?

I would make the decision based on **data size, task complexity, compute budget, and how much of the model's behavior needs to change**.

| Factor | LoRA | Full Fine-Tuning |
| --- | --- | --- |
| **Compute** | Low | Very high |
| **GPU memory** | Low | High |
| **Trainable parameters** | Small fraction | Most/all parameters |
| **Training data** | Works well with modest datasets | Usually benefits from substantial data |
| **Model adaptation** | Targeted adaptation | Maximum adaptation |
| **Storage** | Small adapter | Full model copy |
| **Multiple domains** | Easy to maintain separate adapters | Separate model per domain |
| **Risk of forgetting** | Generally lower | Higher |
| **Best starting point** | Usually yes | Only when justified |

#### Choose LoRA when

- You have **limited GPU/compute resources**.
- You want to adapt a large model to a specific domain or task.
- You have a relatively modest amount of high-quality training data.
- You want separate adapters for different domains.
- You want lower training and storage costs.
- You want to preserve the base model's general capabilities.

**Example:**

A general LLM needs to become better at answering questions using a company's particular support style and workflows.

#### Choose Full Fine-Tuning when

- You have **substantial high-quality training data**.
- You have sufficient compute and memory.
- You need **deep changes to the model's behavior or capabilities**.
- LoRA has been evaluated and doesn't provide sufficient performance.
- You need the entire model to adapt rather than just targeted parameter updates.

#### Practical decision process

**1. Establish a baseline**

Compare the base model using prompting and, where appropriate, RAG.

**2. Try LoRA first**

Fine-tune with LoRA and evaluate against the baseline.

**3. Analyze the gap**

If LoRA meets the business requirements, use it because it is cheaper and easier to maintain.

**4. Consider full fine-tuning**

If LoRA consistently fails because the model requires deeper adaptation, and you have enough data and compute, evaluate full fine-tuning.

### 22.24 Your fine-tuned model memorized training data verbatim instead of learning patterns. How do you fix overfitting?

#### How do you fix overfitting during LLM fine-tuning?

If the model memorizes training examples verbatim, I would first confirm overfitting by comparing **training vs. validation loss** and testing on unseen examples.

1. **Reduce the number of epochs**
   - Too many epochs can cause the model to memorize the training set.
   - Use **early stopping** based on validation performance.
2. **Lower the learning rate**
   - Prevents the model from making overly aggressive updates.
3. **Increase dataset size and diversity**
   - Add more varied examples rather than repeatedly training on similar samples.
   - Include different phrasings, scenarios, and edge cases.
4. **Remove duplicates and near-duplicates**
   - Duplicate or highly similar examples make memorization much easier.
5. **Use regularization**
   - Techniques such as **weight decay** and **dropout** can reduce overfitting.
6. **Use data augmentation**
   - Generate diverse variations of training examples while preserving their correct meaning.
7. **Use PEFT such as LoRA**
   - Training a smaller number of parameters can reduce the tendency to over-specialize.
   - Tune the **LoRA rank** and dropout appropriately.
8. **Reduce model capacity if appropriate**
   - If a smaller model can solve the task, it may generalize better with a limited dataset.
9. **Check the data split**
   - Ensure validation/test examples are genuinely unseen.
   - Avoid leakage between training and evaluation sets.
10. **Test for memorization explicitly**

- Search for verbatim or near-verbatim reproduction of training examples.
- Evaluate on paraphrased and completely unseen inputs.

### 22.25 Your fine-tuned LLM forgot its general capabilities after domain-specific fine-tuning. How do you fix catastrophic forgetting?

#### How do you fix catastrophic forgetting?

If a domain-specific fine-tuned LLM loses its general capabilities, I would use a combination of **better data, gentler training, and PEFT**.

1. **Mix general-purpose data with domain data**
   - Combine domain-specific examples with representative general instruction data.
   - This helps preserve general language and reasoning capabilities.
2. **Reduce the learning rate**
   - A high learning rate can cause large changes to the pre-trained weights.
   - Use a smaller learning rate for full fine-tuning.
3. **Reduce training epochs**
   - Avoid excessive training on the narrow domain dataset.
   - Use validation performance and early stopping.
4. **Use LoRA/PEFT**
   - Keep the base model **frozen** and train only adapter parameters.
   - This preserves the original model while adding domain-specific behavior.
5. **Use regularization**
   - Weight decay and other regularization techniques can discourage excessive movement away from the pre-trained solution.
6. **Increase data diversity**
   - Avoid a narrow or repetitive domain dataset.
   - Include diverse tasks, instructions, and examples.
7. **Use replay data**
   - Periodically train on samples representing the model's original capabilities.
   - This is sometimes called **rehearsal** or **experience replay**.
8. **Evaluate general capabilities**
   - Maintain a general-purpose evaluation suite alongside the domain-specific test set.
   - Don't optimize solely for domain performance.

#### Practical approach

**Base LLM → Domain + General Data → LoRA/controlled fine-tuning → Domain + General Evaluation → Adjust training**

### 22.26 Your RLHF preference data has low annotator agreement. How do you ensure data quality?

#### How do you ensure RLHF preference data quality?

- **Clarify the labeling rubric:** Give annotators precise criteria for helpfulness, correctness, relevance, safety, and style, with clear examples and edge cases.
- **Train annotators:** Provide calibration exercises and feedback before they label production data.
- **Measure agreement:** Track metrics such as **Cohen’s $\kappa$**, **Fleiss’ $\kappa$**, or **Krippendorff’s $\alpha$** to identify ambiguous tasks or inconsistent annotators.
- **Use multiple annotators:** Have multiple independent annotators rank the same responses rather than relying on a single judgment.
- **Resolve disagreements:** Send low-agreement examples to expert reviewers or adjudication.
- **Remove ambiguous samples:** If reasonable annotators cannot agree because the prompt is inherently subjective, exclude or separately categorize the example.
- **Monitor annotator quality:** Use gold-standard questions, consistency checks, and agreement statistics to detect unreliable labeling.
- **Balance the dataset:** Ensure preference data covers different tasks, difficulty levels, domains, and failure modes.
- **Check for bias:** Look for systematic preferences caused by annotator demographics, wording, response length, or stylistic bias rather than actual quality.
- **Iterate on the rubric:** If disagreement is concentrated around specific cases, refine the labeling guidelines and retrain annotators.
- **Validate the reward model:** Evaluate whether the trained reward model agrees with expert judgments on a held-out dataset.

### 22.27 What is the difference between SFT and RL-based tuning (RLHF/DPO)?

**SFT (Supervised Fine-Tuning):** train on curated (prompt, ideal response) pairs with standard next-token cross-entropy on the response tokens. It is **imitation learning**: "write like this demonstration."

- Pros: simple, stable, cheap, and good for teaching format, style, domain and instruction-following.
- Cons: capped by demo quality, it teaches only *what to do*, never *what not to do*, can encourage confident hallucination (the model imitates confident answers even where it lacks the knowledge), and needs expensive expert-written demonstrations.

**RL-based / preference tuning:** optimize the model using *relative* feedback about which outputs are better.

- **RLHF (PPO):** collect human comparisons, train a **reward model**, then optimize the policy with PPO to maximize reward minus a **KL penalty** to the SFT model, which prevents drift and reward hacking. It is powerful, but uses four models in memory (policy, reference, reward, value), is unstable, and is sensitive to hyperparameters.
- **DPO:** a closed-form reformulation that trains directly on (chosen, rejected) pairs with a classification-style loss. There is no reward model and no sampling loop, so it is far simpler and more stable. It is offline, though, so it is limited to the data distribution it was given.
- **RLVR / GRPO:** RL with programmatic rewards (unit tests, math answer checks), the main driver of recent reasoning models.

| | SFT | RLHF / DPO |
| --- | --- | --- |
| Signal | Absolute demos | Relative preferences / rewards |
| Learns | Imitation | Ranking of good vs bad; can exceed demo quality |
| Stability / cost | High / low | Lower (PPO) to medium (DPO) / higher |
| Typical use | First stage | After SFT, for helpfulness, harmlessness, reasoning |

**Practice:** the standard pipeline is pretrain → SFT → preference tuning. SFT gets the model into the right distribution, and preference/RL tuning refines it. Comparisons are also easier for humans to give than perfect demonstrations.

### 22.28 How would you fine-tune an LLM while preserving its general capabilities?

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

### 22.29 Explain RLHF, PPO, DPO, and reward hacking.

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

### 22.30 What is RLVR (Reinforcement Learning with Verifiable Rewards)?

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

### 22.31 What is RLCD (Reinforcement Learning from Contrastive Distillation)?

**RLCD** (Yang et al., 2023) is a method for aligning a language model **without human preference labels**. It generates its own preference pairs by prompting the same model in two opposite ways, then trains a preference model and runs RL as in standard RLHF.

**How it works**

1. Take a prompt from the training set.
2. Create two versions of the context:
   - a **positive** prompt $p^+$ that encourages the desired behaviour (for example, "give a harmless, helpful response"),
   - a **negative** prompt $p^-$ that encourages the opposite (for example, "give a harmful or unhelpful response").
3. Generate one response from each: $o^+$ from $p^+$ and $o^-$ from $p^-$.
4. **Label the pair automatically:** $o^+$ is preferred to $o^-$ because of how it was generated. No human or AI judge scores the pair.
5. Train a preference (reward) model on these pairs, then optimise the policy with PPO (or another RL method) on the **original** prompt without the behavioural instruction.

**Why it helps**

- **Compared with RLAIF**, where an AI judge scores two samples from the same prompt: those two samples are often very similar, so the judge's labels are noisy. RLCD's contrasting prompts make the two responses differ clearly in the target attribute, so the labels are cleaner.
- **Compared with context distillation**, which trains only on outputs from the positive prompt: RLCD also learns from the contrast with negative examples, giving a stronger signal about what to avoid.
- In the paper's experiments on harmlessness, helpfulness and story-outline tasks with LLaMA models, RLCD outperformed both RLAIF and context distillation baselines.

**Limitations**

- The negative prompt must produce responses that differ mainly in the target attribute; otherwise the reward model learns unrelated differences such as length or style.
- The pair label can be wrong when the positive prompt fails to produce a good response or the negative prompt fails to produce a bad one.
- Like any learned reward, the preference model can be exploited by the policy, so a KL penalty and evaluation are still needed.

**In an interview:** place RLCD in the family of methods that reduce the cost of preference data (RLAIF, Constitutional AI, self-rewarding models, synthetic preference pairs), and explain the key idea: **generate contrast deliberately, so the preference label comes for free**.

### 22.32 How do you estimate GPU memory and compute for fine-tuning an LLM?

Interviewers expect quick, explicit arithmetic. Memory for training has four parts: **model weights, gradients, optimiser states and activations**.

**1. Full fine-tuning with mixed precision and Adam**

| Item | Bytes per parameter |
| --- | --- |
| Weights (bf16) | 2 |
| Gradients (bf16) | 2 |
| fp32 master copy of weights | 4 |
| Adam first and second moments (fp32) | 8 |
| **Total** | **about 16** |

For an 8B-parameter model: 8B × 16 bytes ≈ **128 GB** before activations. That needs several 80 GB GPUs with sharding (FSDP or DeepSpeed ZeRO), which split these states across GPUs.

**2. LoRA**

- The frozen base model only needs its weights: 8B × 2 bytes ≈ **16 GB** in bf16.
- Adapters are tiny (often 10 to 100M parameters), so their gradients and optimiser states add well under 2 GB.
- Fits on a single 24 to 48 GB GPU for moderate sequence lengths.

**3. QLoRA**

- The base model is stored in 4-bit: about 0.5 bytes per parameter plus quantisation constants, so roughly **5 to 6 GB** for 8B and about **35 to 40 GB** for 70B.
- This is why QLoRA can fine-tune a 65 to 70B model on a single 48 GB GPU, and an 8B model on a consumer 24 GB card.

**4. Activations**

- Grow with **batch size × sequence length × number of layers × hidden size**. For long sequences they can exceed the weights.
- Reduce them with **gradient checkpointing** (recompute activations in the backward pass, trading about 20% to 30% more compute for much less memory), smaller micro-batches with gradient accumulation, and memory-efficient attention kernels (FlashAttention).

**Compute estimate**

- Training costs about $6 \times N \times D$ FLOPs, where $N$ is parameters and $D$ is training tokens (2 for the forward pass, 4 for the backward pass).
- Example: 8B parameters on 10M tokens: 6 × 8×10⁹ × 10⁷ ≈ 4.8×10¹⁷ FLOPs. An H100 delivers roughly 10¹⁵ dense bf16 FLOP/s at peak; at a realistic 40% utilisation that is 4×10¹⁴ FLOP/s, so about 1,200 seconds, or **20 minutes per epoch** on one GPU.
- LoRA still runs the full forward and backward pass through the frozen model (it needs gradients with respect to activations), so it saves memory far more than it saves compute.

**Practical checklist:** pick the method by memory budget (full, LoRA, QLoRA), set sequence length to the real data distribution rather than the maximum, use bf16, enable gradient checkpointing, and run a short profiling job before launching the full run.

### 22.33 How do loss masking and sequence packing work in supervised fine-tuning?

Two data-loading details cause many silent SFT bugs: **which tokens the loss is computed on**, and **how examples are packed into fixed-length sequences**.

**Loss masking (train on responses only)**

- An SFT example is a formatted conversation: system prompt, user turns and assistant turns, rendered with the model's chat template (see the question on chat templates in Chapter 17).
- Usually the loss is computed **only on assistant tokens**. Prompt tokens are masked, for example by setting their labels to `-100` in PyTorch and Hugging Face, so they are ignored by the cross-entropy loss.
- Why: the model should learn to produce responses, not to predict user messages or memorise the system prompt. Training on prompts wastes capacity and can make the model imitate user text.
- In multi-turn data, include all assistant turns in the loss (or only the final turn, if earlier turns are lower quality).
- **Include the end-of-turn or EOS token in the labels.** If it is masked or missing, the model never learns to stop and keeps generating.

**Sequence packing**

- Examples have very different lengths. Padding each one to the maximum wastes most of the compute.
- **Packing** concatenates several examples into one sequence of the target length, which can speed training several times on short-example datasets.
- The risk is **cross-contamination**: with a normal causal mask, tokens in the second example can attend to the first. Correct packing resets attention at example boundaries, using variable-length attention kernels (for example FlashAttention's variable-length mode) with position IDs restarted for each example.
- Without boundary-aware attention, packing still often works acceptably for pretraining-style data, but for SFT it can leak context between unrelated conversations.

**Other common bugs**

- Training with a different chat template, or different special tokens, from the one used at inference. The model then sees an unfamiliar format in production and quality drops sharply.
- Tokenising the prompt and response separately and joining them, which can create different tokens at the boundary than tokenising the full text.
- Truncating long examples in a way that cuts off the response (and its EOS), teaching the model to produce incomplete answers.

**How to check:** decode a few batches back to text with the loss mask shown, confirm only the intended tokens are trained on, and confirm the EOS token is included.

### 22.34 Which LoRA variants and settings matter in practice: target modules, rank, alpha, DoRA and others?

Plain LoRA (see 22.3) has several choices that strongly affect quality, plus a family of variants.

**Target modules**

- Early LoRA work adapted only the attention query and value projections. Later results, including the QLoRA paper, found that **adapting all linear layers** (attention projections and the MLP or feed-forward projections) is needed to approach full fine-tuning quality. This is the common default now.

**Rank and alpha**

- **Rank $r$** sets adapter capacity. 8 to 16 is often enough for style or format tasks; 32 to 128 helps for harder tasks or more new knowledge. Higher rank costs more memory and can overfit small datasets.
- **Alpha $\alpha$** scales the update as $\frac{\alpha}{r} BA$. A common heuristic is $\alpha = r$ or $\alpha = 2r$.
- **rsLoRA** scales by $\frac{\alpha}{\sqrt{r}}$ instead. With the standard scaling, the update shrinks as rank grows, so high ranks learn slowly; rank-stabilised scaling makes higher ranks actually useful.
- **Learning rate** for LoRA is typically about 10 times higher than for full fine-tuning (for example 1e-4 to 2e-4).

**Variants**

| Variant | Idea | Benefit |
| --- | --- | --- |
| DoRA | Decompose each weight into magnitude and direction; apply LoRA to the direction and learn the magnitude separately | Closer to full fine-tuning behaviour, often better at low rank |
| LoRA+ | Use a higher learning rate for matrix $B$ than for $A$ | Faster, more stable convergence |
| PiSSA | Initialise the adapter from the principal singular vectors of the weight, and freeze the residual | Faster convergence than random initialisation |
| VeRA | Share frozen random matrices across layers and learn only small scaling vectors | Far fewer trainable parameters |
| QLoRA | LoRA on a 4-bit quantised frozen base model | Large memory savings |

**LoRA versus full fine-tuning**

- Research has found that LoRA **learns less and forgets less**: it tends to underperform full fine-tuning when the task requires a lot of new knowledge (for example, large-scale continued pretraining on code or mathematics), but better preserves the base model's general abilities.
- For most product use cases (format, style, domain behaviour, classification), LoRA with all linear layers targeted gets very close to full fine-tuning at a fraction of the cost.

**Practical recipe:** all linear layers, rank 16 to 64, alpha equal to rank or twice it (or rsLoRA for high ranks), learning rate around 1e-4 to 2e-4, and compare against a full fine-tuning run only if the evaluation shows a meaningful gap.

### 22.35 How do you serve many fine-tuned LoRA adapters efficiently?

A common product pattern is **one base model with many adapters**: one per customer, task or language. Serving each as a separate merged model wastes GPU memory and money. Multi-LoRA serving shares the base model.

**Options**

1. **Merge each adapter into the base weights** ($W' = W + BA$) and deploy a separate model per adapter.
   - No inference overhead, but N full copies of the model, so it only makes sense for a few high-traffic adapters.
2. **Serve unmerged adapters on a shared base model.**
   - The base weights are loaded once. For each request, the server adds the request's adapter computation ($xBA$) alongside the base computation ($xW$).
   - Requests for **different adapters can be batched together**, using specialised kernels that apply a different adapter to each row of the batch (the Punica SGMV kernels and S-LoRA are well-known designs). vLLM, SGLang, TGI and LoRAX support this.
   - Adapters live in GPU memory with an LRU cache, with the rest in CPU memory or storage, loaded on demand. Thousands of adapters can be served from one deployment.

**Design considerations**

- **Same base model and precision:** every adapter must be trained against the exact base model version (and quantisation) being served.
- **Rank limits:** servers usually pre-allocate for a maximum adapter rank; keep ranks standardised.
- **Overhead:** unmerged serving adds some latency and reduces throughput compared with a merged model, and loading a cold adapter adds latency to the first request. Keep hot adapters resident.
- **Routing:** pass the adapter ID with each request; route requests for the same adapter to replicas that already have it loaded.
- **Isolation:** for per-customer adapters, enforce that a tenant can only use its own adapter, and version adapters with rollback.
- **Base model upgrades** require retraining every adapter, so keep training data and pipelines reproducible.

**When to merge instead:** a single high-volume adapter with strict latency needs, or deployment targets that do not support dynamic adapters (some edge runtimes).

### 22.36 How do you fine-tune a model for reliable tool calling (function calling)?

Strong frontier models usually handle tool calling well with good prompts and tool descriptions. Fine-tuning helps when a **smaller or open model** must call your tools reliably, at lower cost or latency, or when your tools and workflows are unusual.

**What the model must learn**

- **Whether** to call a tool, including when no tool is needed or when to ask the user for missing information.
- **Which** tool to choose among similar ones.
- **How** to fill arguments correctly: types, formats, enums, and values taken from the conversation rather than invented.
- **Multi-step and parallel** calls, using earlier results to decide the next call.
- **Using the results**: answering from the tool output, and handling errors and empty results gracefully.

**Building the dataset**

- Use the **exact tool-calling format and chat template** of the model being trained (special tokens or structured fields for tool calls and tool results).
- Mix example types: single call, multiple sequential calls, parallel calls, **no-call** cases, clarification questions, tool errors and retries. Too few no-call examples produces a model that calls tools for everything.
- Generate synthetic trajectories with a strong model, then **validate them by executing the calls** against real or mocked tools, and keep only trajectories that succeed and produce correct final answers.
- **Vary tool names, descriptions and schemas** during training so the model learns to read tool definitions rather than memorising specific tools.
- Mix in general instruction data to avoid losing conversational ability.

**Evaluation**

- **Static checks:** correct tool selection, argument match against references (compare parsed structures, not raw strings), valid JSON.
- **Execution-based checks:** run the calls and check that the final task outcome is correct.
- Public benchmarks such as the Berkeley Function Calling Leaderboard give a general signal; your own task suite matters more.
- Track over-calling and under-calling rates, hallucinated arguments, and number of steps per task.

**Beyond SFT:** RL with verifiable rewards on agent tasks (reward when the task outcome is correct) can further improve multi-step tool use, but needs a sandboxed environment and careful reward design.

### 22.37 How do you choose among preference optimisation methods such as DPO, IPO, KTO, ORPO and SimPO?

DPO (covered in Chapter 17) made preference tuning simple and stable, and several variants followed. Each changes the loss or the data it needs. Choosing among them is mostly about **what data you have** and **what failure you are seeing**.

| Method | Data needed | Key idea | When to consider |
| --- | --- | --- | --- |
| DPO | Paired (chosen, rejected) | Increase the log-probability ratio of chosen over rejected, relative to a frozen reference model | Default starting point |
| IPO | Paired | A squared loss that targets a fixed margin instead of pushing the margin to infinity | DPO overfits, especially when preferences are nearly deterministic |
| KTO | **Unpaired** good or bad labels | A loss inspired by prospect theory, applied to single responses marked desirable or undesirable | You have thumbs-up and thumbs-down feedback rather than pairs |
| ORPO | Paired | Adds an odds-ratio preference term to the normal SFT loss; **no reference model** and no separate SFT stage | Want one training stage and less memory |
| SimPO | Paired | **Reference-free**; uses length-normalised average log-probability as the reward, plus a target margin | Want lower memory and less length bias |

**Offline versus online**

- Standard DPO is **offline**: it learns from a fixed dataset of responses, usually generated by another model. Its quality is limited by how well that data matches what the current model produces.
- **Online or iterative** preference optimisation samples new responses from the current model, labels them with a reward model, an LLM judge or a verifier, and trains again. This usually improves results and reduces the gap with PPO-style RLHF, at the cost of a generation loop.

**Common problems and fixes**

- **Length exploitation:** the model learns that longer answers win. Use length-normalised methods (SimPO), length-controlled evaluation, or balance lengths in the data.
- **Both likelihoods fall:** DPO can reduce the probability of the chosen response too, as long as the rejected one falls faster, which can damage fluency. Add an SFT term on chosen responses or use a smaller learning rate.
- **Noisy preferences:** low annotator agreement limits all methods; clean the data first (see 22.26).
- **Beta ($\beta$) tuning:** in DPO, $\beta$ controls how far the model may move from the reference. Too high learns little; too low overfits and drifts.

**Practical recipe:** start with SFT, then DPO with a modest $\beta$ (around 0.1) on high-quality pairs that the current model actually produces, evaluate on held-out preferences and task benchmarks, and move to iterative or online training if offline gains plateau.

### 22.38 How would you train a model to reason better using reinforcement learning with verifiable rewards?

Recent reasoning models were trained largely with **reinforcement learning on tasks whose answers can be checked automatically** (mathematics, code, logic, structured tasks). GRPO (Chapter 17) is the most widely used algorithm. This question is about the practical recipe.

**1. Start from a capable base**

- RL improves reasoning that the model can already sometimes do; it rarely creates abilities from nothing. Pick a strong base or instruction model.
- A short **cold-start SFT** on high-quality long reasoning traces gives a readable format and makes RL more stable. DeepSeek-R1 used this; its RL-only predecessor (R1-Zero) reasoned well but produced hard-to-read, mixed-language output.

**2. Build the task set and verifiers**

- Problems with checkable answers: mathematics with exact final answers, coding problems with unit tests, puzzles, SQL against a test database, extraction against ground truth.
- **Reward design:** correctness reward (for example 1 for correct, 0 otherwise), plus small format rewards (reasoning and answer in the expected tags). Make the answer extraction robust: equivalent forms of a mathematical answer should count as correct.
- **Difficulty matters:** in GRPO, if all sampled answers for a prompt are correct, or all are wrong, the group advantage is zero and the prompt teaches nothing. Filter for prompts the model solves sometimes, and refresh the pool as the model improves (a curriculum).

**3. Run the RL loop**

```text
sample batch of prompts ─► generate G answers per prompt with the current policy (fast inference engine, e.g. vLLM)
  ─► score each answer with the verifier ─► group-relative advantages
  ─► policy update (clipped objective, optional KL to reference) ─► sync new weights to the inference engine ─► repeat
```

- **Rollout generation dominates cost,** because answers are long. Most of the engineering effort goes into fast generation and keeping training and inference GPUs busy.
- Typical settings: 8 to 64 samples per prompt, low learning rate, long maximum generation length.

**4. Monitor**

- Reward and accuracy on held-out problems, average response length (it usually grows as the model learns to reason longer), and entropy of the policy (a collapse means exploration has stopped).
- Read samples regularly to catch **reward hacking**: exploiting answer-extraction bugs, writing code that special-cases the tests, or guessing formats.

**5. Finish for general use**

- Reasoning-only RL can hurt general helpfulness. Production recipes add rejection-sampled SFT on a broad mix of tasks and a final RL stage with preference rewards for helpfulness and safety.

**Known pitfalls:** verifier bugs become model behaviour; overly long responses waste inference cost (consider length penalties or budgets); and gains on benchmark-like problems may not transfer to real tasks, so evaluate on your own use cases.

### 22.39 How does fine-tuning affect a model's safety behaviour, and how do you preserve it?

Safety behaviour (refusing harmful requests, following policies) is learned during alignment and can be **easily weakened by later fine-tuning**. Research has shown that fine-tuning an aligned model on only a handful of harmful examples can largely remove its safety guardrails, and that even fine-tuning on **entirely benign** data can measurably reduce refusal of harmful requests.

**Why it happens**

- Safety behaviour is a relatively thin layer on top of the base model's capabilities. Any update that moves the model away from the aligned weights can disturb it.
- Fine-tuning data rarely contains refusal examples, so the model is trained only to comply, never to decline.
- Strong compliance training (always answer, always follow the format) generalises to harmful requests too.

**How to preserve safety**

- **Mix in safety data:** include examples of correct refusals and safe completions (a few percent of the dataset is often enough to recover most of the safety behaviour).
- **Prefer LoRA** and conservative hyperparameters (lower learning rate, fewer epochs), which limit drift.
- **Screen the training data** for harmful content, jailbreak prompts and poisoned examples. Data poisoning can also implant backdoors, where a trigger phrase activates hidden behaviour.
- **Evaluate safety before and after** with a fixed red-team and refusal benchmark (harmful request refusal rate, and over-refusal rate on benign but sensitive requests). Treat a safety regression like a failing test.
- **Add system-level defences:** input and output moderation, guardrail classifiers and policy checks, so safety does not rest on the model alone.
- **For open-weight models you release,** remember that anyone can fine-tune away safety; release decisions should account for that.

**Hosted fine-tuning services** usually screen training data and evaluate fine-tuned models for safety before allowing deployment, for exactly these reasons.

### 22.40 Should you fine-tune through a provider's API or fine-tune open-weight models yourself?

Both are valid; the choice depends on control, cost at scale, data governance and the team's skills.

| Factor | Provider fine-tuning API | Self-hosted open-weight model |
| --- | --- | --- |
| Starting quality | Often the strongest available base models | Strong and improving, but may trail the frontier on hard tasks |
| Effort | Upload data, start a job, call an endpoint | Training infrastructure, serving, monitoring and on-call |
| Methods and control | Limited methods and hyperparameters; no access to weights | Any method (full, LoRA, DPO, RL), full control of data, training and decoding |
| Serving cost | Per-token pricing, often higher for fine-tuned models | GPU cost; cheaper at high, steady volume; supports multi-LoRA serving |
| Data governance | Data leaves your environment (check retention, region and training terms) | Data and weights stay in your environment |
| Portability | Tied to the provider; base model deprecations can force retraining | Weights are yours and portable (check the model licence) |
| Latency and deployment | Provider regions and limits | Any region, on premises, edge |

**When the API is the better choice**

- Small team, moderate volume, need to move fast.
- The task benefits from the strongest base model.
- Data can be sent to the provider under acceptable terms.

**When self-hosting is the better choice**

- High, steady volume where GPU serving is cheaper per request.
- Strict data residency, privacy or sovereignty requirements.
- Need for methods or control the API does not offer, or many adapters for many customers.
- Long-term independence from a vendor's model life cycle.

**Practical approach:** prototype with prompting and a provider model to establish the quality bar and a labelled evaluation set; then decide whether a fine-tuned open model can match it at lower cost. Keep training data, evaluation sets and pipelines provider-independent, so the decision can be revisited as models and prices change.

### 22.41 How do you debug a fine-tuning run that is not working?

Most fine-tuning failures come from data and formatting, not from the optimiser. Debug systematically, starting with the cheapest checks.

| Symptom | Likely causes | What to do |
| --- | --- | --- |
| Loss does not decrease | Learning rate too low; all labels masked; adapters not attached or parameters frozen by mistake | Check trainable parameter count; decode a batch with its loss mask; try a higher learning rate on a tiny subset |
| Loss falls to near zero very fast | Duplicates or leakage; trivial or templated targets; very small dataset | Deduplicate; check that validation data is not in training; inspect targets |
| Loss spikes or becomes NaN | Learning rate too high; fp16 overflow; corrupted or extremely long examples | Lower learning rate, add warmup and gradient clipping, use bf16, find the batch that caused the spike |
| Validation loss rises while training loss falls | Overfitting | Fewer epochs, early stopping, more or more diverse data, lower rank |
| Model never stops generating | EOS or end-of-turn token missing from labels or masked | Include the end token in the labels; check the chat template |
| Outputs are repetitive or generic | Overfitting to a narrow style; low-diversity data | More diverse data, fewer epochs, check decoding settings |
| Good offline, poor in production | Different chat template or system prompt at inference; production inputs differ from training data | Use identical formatting code for training and serving; sample production inputs into the evaluation set |
| Task improves but general ability drops | Catastrophic forgetting | Mix in general data, use LoRA, lower learning rate (see 22.11) |
| Safety or refusal behaviour changes | No safety data in the mix | Add safety examples and evaluate refusal rates |

**Debugging workflow**

1. **Overfit a tiny dataset first** (10 to 50 examples). If the model cannot memorise them, the pipeline is broken (masking, template, frozen parameters), not the hyperparameters.
2. **Inspect the data as the model sees it:** decode tokenised batches with the loss mask, check special tokens, truncation and EOS.
3. **Evaluate the base model** on the same evaluation set, so you know whether fine-tuning helps at all.
4. **Change one thing at a time** and track every run (data version, hyperparameters, metrics) with an experiment tracker.
5. **Read outputs,** not just metrics. Loss can improve while the outputs get worse in ways the metric does not capture.

---

← [Chapter 21. AI Agents and Agentic Systems](21-ai-agents-and-agentic-systems.md) · [Table of contents](../README.md) · [Chapter 23. Multimodal AI](23-multimodal-ai.md) →
