# Chapter 22. Fine-Tuning and Model Adaptation

_Part IV — Generative AI and LLM Engineering_

Supervised fine-tuning, parameter-efficient methods, RLHF, DPO and related alignment methods, and when to fine-tune at all.

This chapter contains **38 questions**.

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
- **22.12** [When should you choose fine-tuning over RAG over prompt engineering?](#2212-when-should-you-choose-fine-tuning-over-rag-over-prompt-engineering)
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
- **22.27** [What is fine-tuning?](#2227-what-is-fine-tuning)
- **22.28** [What is the difference between SFT and RL-based tuning (RLHF/DPO)?](#2228-what-is-the-difference-between-sft-and-rl-based-tuning-rlhfdpo)
- **22.29** [What is the purpose of fine-tuning/distillation techniques like LoRA/PEFT?](#2229-what-is-the-purpose-of-fine-tuningdistillation-techniques-like-lorapeft)
- **22.30** [When is synthetic data generation used?](#2230-when-is-synthetic-data-generation-used)
- **22.31** [What is RLHF?](#2231-what-is-rlhf)
- **22.32** [What is catastrophic forgetting during fine-tuning?](#2232-what-is-catastrophic-forgetting-during-fine-tuning)
- **22.33** [How would you fine-tune an LLM while preserving its general capabilities?](#2233-how-would-you-fine-tune-an-llm-while-preserving-its-general-capabilities)
- **22.34** [How would you determine whether to fine-tune, RAG, prompt-engineer, or train from scratch?](#2234-how-would-you-determine-whether-to-fine-tune-rag-prompt-engineer-or-train-from-scratch)
- **22.35** [Explain LoRA and QLoRA.](#2235-explain-lora-and-qlora)
- **22.36** [Explain RLHF, PPO, DPO, and reward hacking.](#2236-explain-rlhf-ppo-dpo-and-reward-hacking)
- **22.37** [What is RLVR (Reinforcement Learning with Verifiable Rewards)?](#2237-what-is-rlvr-reinforcement-learning-with-verifiable-rewards)
- **22.38** [What is RLCD (Reinforcement Learning for Calibrated Decisions)?](#2238-what-is-rlcd-reinforcement-learning-for-calibrated-decisions)

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

For an existing weight matrix **W**, LoRA keeps **W frozen** and learns a low-rank update:

**W' = W + ΔW**

Instead of directly learning the large **ΔW**, LoRA represents it as:

**ΔW = B × A**

where **A** and **B** are much smaller matrices.

During training:

1. **Freeze the original LLM weights.**
2. Insert LoRA adapters into selected layers, commonly the **attention projections** such as Q, K, V, or O.
3. Train only the small **A and B matrices**.
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

#### What is Catastrophic Forgetting?

- **Catastrophic forgetting** occurs when an LLM is fine-tuned on a new dataset and **loses some of the knowledge or capabilities it learned during pre-training**.
- It happens because fine-tuning updates the model's weights toward the new task, which can overwrite previously learned representations.
- Example: Fine-tuning a general LLM heavily on a narrow coding dataset may improve that coding task but reduce its performance on general conversation or other domains.

#### How do you prevent it?

- **Use a smaller learning rate**
  - Reduces large changes to the pre-trained weights.
- **Use fewer training epochs**
  - Prevents excessive adaptation to a narrow dataset.
- **Use a representative dataset**
  - Include diverse examples rather than only highly repetitive task-specific data.
- **Mix general data with task-specific data**
  - Helps preserve the model's general capabilities.
- **Use PEFT methods such as LoRA**
  - Keep the base model frozen and train only small adapter parameters.
- **Regularization**
  - Penalize large deviations from the original model parameters.
- **Monitor evaluation performance**
  - Evaluate both the target task and general capabilities before, during, and after fine-tuning.

### 22.12 When should you choose fine-tuning over RAG over prompt engineering?

#### Fine-Tuning vs RAG vs Prompt Engineering

The choice depends mainly on **what you need to change**: the model's behavior, its knowledge, or simply how you ask it to perform a task.

| Approach | Choose it when | Best for |
| --- | --- | --- |
| **Prompt Engineering** | The model already has the capability but needs better instructions | Simple tasks, formatting, reasoning guidance |
| **RAG** | The model needs access to external or changing information | Private documents, current knowledge, factual Q&A |
| **Fine-Tuning** | You need the model to consistently learn a specific behavior or task | Style, format, domain behavior, specialized tasks |

#### 1. Choose Prompt Engineering when

- The task can be solved by giving better instructions.
- You need a quick and inexpensive solution.
- Requirements or behavior may change frequently.
- You need few-shot examples or structured output.

**Example:**

You want an LLM to return customer information in a specific JSON format.

#### 2. Choose RAG when

- The model needs **external knowledge**.
- Information changes frequently.
- You need answers grounded in private/company documents.
- You want to update the knowledge base without retraining the model.

**Example:**

A company chatbot answers questions using its latest HR policies and internal documentation.

#### 3. Choose Fine-Tuning when

- Prompting isn't producing **consistent behavior**.
- You have enough high-quality training examples.
- You need specialized task performance, style, or output behavior.
- The model repeatedly performs the same type of task.

**Example:**

Fine-tune a model to consistently extract specific fields from thousands of domain-specific documents.

#### Practical Decision Rule

**Need better instructions? → Prompt Engineering**

**Need new/current/private knowledge? → RAG**

**Need to change the model's behavior or specialize its capabilities? → Fine-Tuning**

In real systems, they are often **combined**:

**Fine-tuned model + RAG + good prompting**

For example, fine-tuning can teach an enterprise assistant **how to respond**, while RAG provides **what it should know**.

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

**Effective batch size ≈ batch size × gradient accumulation steps × number of GPUs**

#### 4. LoRA Rank

- **LoRA rank (`r`)** controls the size/capacity of the low-rank update.
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

   **ΔW = α₁ΔW₁ + α₂ΔW₂ + ... + αₙΔWₙ**

   - This allows you to control how strongly each adapter contributes.
2. **Merge into the base model**
   - Combine the LoRA updates with the original model weights:

   **W' = W + ΔW**

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
- **Measure agreement:** Track metrics such as **Cohen’s κ**, **Fleiss’ κ**, or **Krippendorff’s α** to identify ambiguous tasks or inconsistent annotators.
- **Use multiple annotators:** Have multiple independent annotators rank the same responses rather than relying on a single judgment.
- **Resolve disagreements:** Send low-agreement examples to expert reviewers or adjudication.
- **Remove ambiguous samples:** If reasonable annotators cannot agree because the prompt is inherently subjective, exclude or separately categorize the example.
- **Monitor annotator quality:** Use gold-standard questions, consistency checks, and agreement statistics to detect unreliable labeling.
- **Balance the dataset:** Ensure preference data covers different tasks, difficulty levels, domains, and failure modes.
- **Check for bias:** Look for systematic preferences caused by annotator demographics, wording, response length, or stylistic bias rather than actual quality.
- **Iterate on the rubric:** If disagreement is concentrated around specific cases, refine the labeling guidelines and retrain annotators.
- **Validate the reward model:** Evaluate whether the trained reward model agrees with expert judgments on a held-out dataset.

### 22.27 What is fine-tuning?

> **See also:** [Question 17.5 — What is Fine-tuning?](17-large-language-model-foundations.md#175-what-is-fine-tuning), which covers the same question from another angle.

Further training a pre-trained model on a smaller, task-specific dataset to specialize its behavior (e.g., tone, domain knowledge, format) beyond what the base model does out of the box.

### 22.28 What is the difference between SFT and RL-based tuning (RLHF/DPO)?

> ✍️ _Drafted answer (not from the original notes)._

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

### 22.29 What is the purpose of fine-tuning/distillation techniques like LoRA/PEFT?

To efficiently train smaller models for narrow, high-volume tasks without the cost of full fine-tuning.

### 22.30 When is synthetic data generation used?

When real training or evaluation data is scarce, an LLM can generate synthetic data to fill the gap.

### 22.31 What is RLHF?

**RLHF** is a technique for aligning an AI model with human preferences by using human feedback as the reward signal.

#### Typical pipeline

1. **Pretraining** — Train the model on large-scale data.
2. **Supervised Fine-Tuning (SFT)** — Fine-tune on high-quality human demonstrations.
3. **Preference collection** — Humans rank multiple model responses.
4. **Reward modeling** — Train a reward model to predict those preferences.
5. **RL optimization** — Optimize the language model against the learned reward, traditionally using PPO.

RLHF is useful when the desired behavior is difficult to specify with a deterministic reward, such as **helpfulness, harmlessness, instruction following, and response quality**.

### 22.32 What is catastrophic forgetting during fine-tuning?

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

### 22.33 How would you fine-tune an LLM while preserving its general capabilities?

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

### 22.34 How would you determine whether to fine-tune, RAG, prompt-engineer, or train from scratch?

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

### 22.35 Explain LoRA and QLoRA.

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

### 22.36 Explain RLHF, PPO, DPO, and reward hacking.

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

### 22.37 What is RLVR (Reinforcement Learning with Verifiable Rewards)?

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

### 22.38 What is RLCD (Reinforcement Learning for Calibrated Decisions)?

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

---

← [Chapter 21. AI Agents and Agentic Systems](21-ai-agents-and-agentic-systems.md) · [Table of contents](../README.md) · [Chapter 23. Multimodal AI](23-multimodal-ai.md) →
