<!-- notion-id: 3d6674182854802db793c5cb1240fdb3 -->
# Large Language Model

LLM interview questions: Transformer internals, attention variants, inference optimization (KV cache, PagedAttention, FlashAttention, batching, speculative decoding), tokenization, positional encodings, alignment (RLHF, PPO, DPO, GRPO) and related topics.

### Q1. What is a Large Language Model (LLM), and how does it work?

#### Large Language Model (LLM)

- **Core idea:** A deep learning model (typically Transformer-based) trained on massive amounts of text data to understand and generate human language, with billions to trillions of parameters enabling broad, general-purpose language capabilities.

---

##### How It Works

- **Architecture:** Built on the Transformer architecture (usually decoder-only, like GPT), using **self-attention** to process and relate all tokens in a sequence simultaneously, capturing context and relationships regardless of distance.
- **Tokenization:** Input text is broken into tokens (subword units via BPE or similar), converted to embeddings, combined with positional encodings to preserve order.
- **Pretraining:** Trained on massive, diverse text corpora (web text, books, code) using a **self-supervised objective** — typically **next-token prediction** (causal language modeling): predict the next token given all previous tokens, using the standard cross-entropy loss.
- **Scale:** "Large" refers to both the model size (billions/trillions of parameters) and training data volume (trillions of tokens) — scale is a key driver of emergent capabilities (reasoning, few-shot learning) not seen in smaller models.
- **Generation (inference):** Uses **autoregressive decoding** — generates one token at a time, feeding each generated token back as input for predicting the next, until an end-of-sequence token or length limit is reached (using decoding strategies like sampling, top-k, top-p, or greedy/beam search).

---

##### Post-Pretraining Refinement (Modern LLMs)

- **Instruction tuning:** Fine-tuned on (instruction, response) pairs to make the model follow natural language instructions rather than just complete arbitrary text.
- **RLHF (Reinforcement Learning from Human Feedback):** Further aligns model outputs with human preferences using a reward model trained on human comparisons, improving helpfulness, safety, and coherence of responses.

---

##### Key Emergent Capabilities

- **In-context learning:** Can perform new tasks based purely on examples/instructions given in the prompt, without updating model weights (zero-shot/few-shot learning).
- **Reasoning:** Can perform multi-step reasoning, especially with prompting techniques like chain-of-thought.
- **Broad knowledge:** Encodes substantial world knowledge learned from training data, usable for question-answering, summarization, coding, translation, and more — all within a single general-purpose model.

---

##### Limitations

- **Hallucination:** Can generate plausible-sounding but factually incorrect information.
- **Knowledge cutoff:** Limited to knowledge present in training data up to a certain date, unless augmented with retrieval (RAG) or tools.
- **Compute/cost:** Training and running inference on LLMs requires significant computational resources.
- **Context length limits:** Practical limits on how much text can be processed at once, though this has been growing significantly with newer architectures/techniques.

### Q2. What is the Transformer architecture, and how does it work?

> 🔁 _See also [01-llm-fundamentals.md → Q9](../ai-engineering/01-llm-fundamentals.md#q9-what-is-the-transformer-architecture-and-how-does-it-work) (separate Notion notes on the same question)._

#### Transformer Architecture

- **Core idea:** A neural network architecture that processes entire sequences in parallel using **self-attention**, instead of sequentially like RNNs — enabling faster training and better long-range dependency modeling.

---

##### How It Works

- **Input embedding + positional encoding:** Tokens are converted to embeddings; since attention has no inherent sense of order, positional encodings are added to inject sequence position information.
- **Self-attention:** For each token, computes **Query (Q)**, **Key (K)**, **Value (V)** vectors, then calculates attention weights showing how relevant every other token is to the current one:
  - `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
- **Multi-head attention:** Runs several attention operations in parallel (different "heads"), each capturing different types of relationships (syntactic, semantic, positional); outputs are concatenated and combined.
- **Feedforward network:** Each token's representation (post-attention) passes through a position-wise fully connected layer, applied identically at every position.
- **Residual connections + layer normalization:** Wrap each sub-layer (attention, feedforward) to stabilize training and enable deeper stacking.
- **Stacked layers:** Multiple identical encoder/decoder layers are stacked, each refining the representation further.

---

##### Encoder-Decoder Structure

- **Encoder:** Processes the input sequence, producing contextual representations (used in models like BERT).
- **Decoder:** Generates output tokens one at a time, using **masked self-attention** (can only see previous tokens) plus attention over encoder outputs — used in translation-style seq2seq tasks; decoder-only variants (GPT) drop the encoder entirely for pure text generation.

---

##### Why It Matters

- **Parallelizable:** Unlike RNNs, all tokens are processed simultaneously, enabling much faster training on GPUs/TPUs.
- **Long-range dependencies:** Any two tokens can interact directly in one step via attention, regardless of distance — no information decay like in RNN chains.
- **Foundation of modern AI:** Powers BERT, GPT, T5, and virtually all modern LLMs, as well as adaptations for vision (ViT) and multimodal models (CLIP).

### Q3. What are the key components of the Transformer architecture?

> 🔁 _See also [01-llm-fundamentals.md → Q10](../ai-engineering/01-llm-fundamentals.md#q10-what-are-the-key-components-of-the-transformer-architecture) (separate Notion notes on the same question)._

#### Key Components of the Transformer Architecture

- **Input Embeddings:** Converts input tokens (words/subwords) into dense vector representations.
- **Positional Encoding:** Added to embeddings to inject information about token order/position, since self-attention has no inherent sense of sequence order (uses sine/cosine functions or learned embeddings).
- **Self-Attention (Scaled Dot-Product Attention):** Computes Query (Q), Key (K), Value (V) vectors for each token; determines how much each token should attend to every other token:
  - `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
- **Multi-Head Attention:** Runs multiple attention operations in parallel (different heads), each learning different types of relationships (syntactic, semantic); outputs concatenated and linearly combined.
- **Masked Self-Attention (Decoder only):** Prevents a token from attending to future tokens, preserving the autoregressive property during generation.
- **Feedforward Network (FFN):** A position-wise fully connected network (two linear layers with a non-linearity like ReLU/GELU in between) applied independently to each token's representation.
- **Residual (Skip) Connections:** Add each sub-layer's input to its output, helping gradients flow through deep stacks and easing training.
- **Layer Normalization:** Normalizes activations within each layer, stabilizing and speeding up training.
- **Encoder Stack:** Multiple identical layers (self-attention + FFN, each with residuals/norm) that process the input sequence into contextual representations.
- **Decoder Stack:** Multiple identical layers (masked self-attention + cross-attention over encoder output + FFN) that generate the output sequence token by token.
- **Cross-Attention (Encoder-Decoder Attention):** In the decoder, allows each output token to attend to relevant parts of the encoder's output representations (used in encoder-decoder models like the original Transformer, T5).
- **Output Linear + Softmax Layer:** The final layer projects decoder output into vocabulary-sized logits, converted to probabilities via softmax to predict the next token.

### Q4. What Does Each Transformer Block Learn?

#### What Each Transformer Block Learns

- **Lower layers (early blocks):**
  - Capture **surface-level, local patterns** — basic syntax, word order, part-of-speech-like signals, local token relationships (e.g., adjacent word dependencies, simple grammatical structure).
  - Attention heads often focus on nearby tokens or simple positional patterns.
- **Middle layers:**
  - Capture **syntactic structure** — phrase boundaries, grammatical dependencies (subject-verb-object relationships), coreference-like patterns (linking pronouns to entities).
  - Attention becomes more selective, focusing on syntactically or semantically related tokens rather than just nearby ones.
- **Higher layers (deeper blocks):**
  - Capture **semantic and abstract relationships** — meaning, topic-level context, long-range dependencies, task-relevant high-level features.
  - Representations here are more tuned toward the specific pretraining objective (e.g., predicting masked words, predicting next tokens) and are most useful for downstream task performance after fine-tuning.

---

##### Attention Heads Specialization

- Different **attention heads within the same layer** often specialize in different relationship types simultaneously:
  - Some heads track **positional/local** relationships (attending to adjacent tokens).
  - Some heads track **syntactic dependencies** (e.g., verb-object, subject-verb links).
  - Some heads capture **coreference** (linking pronouns like "it," "she" to their referents).
  - Some heads act more like **global/broad context aggregators**, attending broadly across the whole sequence.

---

##### Empirical Findings (from interpretability research on models like BERT)

- Early layers behave somewhat like traditional NLP pipeline steps (POS tagging, basic parsing).
- Middle layers align well with syntactic parsing tasks.
- Later layers align well with semantic tasks (e.g., question answering, sentence similarity, NLI).
- This rough "hierarchy" isn't strict or perfectly clean — there's overlap and redundancy across layers, and different models/tasks can shift where certain features emerge.

---

##### Why This Matters Practically

- Explains why **feature-based transfer learning** (using intermediate layer representations) can work well for certain tasks — different layers suit different downstream applications depending on whether syntactic or semantic information is more relevant.
- Informs techniques like **layer-wise fine-tuning** or **probing classifiers**, where researchers analyze/exploit specific layers to understand or improve model behavior for a given task.
- Explains why **truncating or freezing certain layers** during fine-tuning (e.g., freezing early layers, only fine-tuning later ones) can be effective — early layers often capture generic patterns useful across many tasks, while later layers need more task-specific adaptation.

### Q5. Why do we scale the dot product attention by √dₖ in the Transformer architecture?

> 🔁 _See also [01-llm-fundamentals.md → Q19](../ai-engineering/01-llm-fundamentals.md#q19-why-do-we-scale-the-dot-product-attention-by-dₖ-in-the-transformer-architecture) (separate Notion notes on the same question)._

#### Why Scale Dot-Product Attention by √dₖ

- **The problem without scaling:** In dot-product attention, the score between a query `Q` and key `K` is computed as `Q · K^T`. As the dimensionality `d_k` of these vectors grows, the **variance of the dot product grows proportionally with `d_k`** (assuming components of Q and K are independent random variables with mean 0 and variance 1, the dot product's variance scales as `d_k`).
- **Consequence of large dot products:** When `d_k` is large, the raw dot-product scores can become very large in magnitude (either strongly positive or negative).
- **Effect on softmax:** These large-magnitude scores push the **softmax function into regions where it has extremely small gradients** — softmax saturates when inputs are very large/small, meaning it produces very peaked distributions (close to one-hot) with gradients near zero almost everywhere.
- **Why this is a problem:** Vanishing gradients through softmax make it very difficult for the model to learn effectively during backpropagation — the attention weights become overly confident too early/inappropriately, and the model struggles to adjust them via gradient descent.
- **The fix — scaling by √dₖ:** Dividing the dot product by `√d_k` normalizes the variance of the scores back down to approximately 1, regardless of `d_k`:
  - `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
  - This keeps the softmax inputs in a reasonable range, avoiding extreme saturation and preserving useful gradients for stable training.
- **Why √dₖ specifically (not `d_k` itself):** Since variance scales linearly with `d_k`, scaling by the square root of `d_k` (standard deviation) brings the variance of the dot product back down to approximately 1, matching the assumption of unit variance for Q and K components — this is the mathematically appropriate normalization factor.
- **Net effect:** Scaled dot-product attention leads to more stable gradients, better-behaved softmax outputs, and more effective training — this is why the original Transformer paper ("Attention Is All You Need") introduced this scaling as a key design choice over standard (unscaled) dot-product attention.

### Q6. KV Cache in LLMs

#### KV Cache in LLMs

- **Core idea:** An optimization technique used during autoregressive text generation that stores (caches) the **Key (K)** and **Value (V)** vectors computed for previous tokens, so they don't need to be recomputed at every new generation step — significantly speeding up inference.

---

##### The Problem It Solves

- In autoregressive generation, the model generates one token at a time, and each new token depends on all previous tokens (via self-attention/causal attention).
- **Without caching:** At every generation step, the model would need to recompute K and V vectors for **all** tokens in the sequence so far (including ones already processed in earlier steps) — this is highly redundant since those K/V values don't change once computed.
- This redundant recomputation makes generation slower and slower as the sequence grows, since attention computation scales with sequence length at every single step.

---

##### How KV Cache Works

- **First forward pass (prompt processing):** The model processes the full input prompt, computing K and V vectors for every token — these are stored in the cache.
- **Subsequent generation steps:** For each new token generated:
  - Only the **new token's** Query (Q), Key (K), and Value (V) vectors need to be computed.
  - The new K and V vectors are **appended** to the cached K/V vectors from all previous tokens.
  - Attention is computed using the new Query against **all cached Keys/Values** (previous + new), without recomputing K/V for earlier tokens.
- **Result:** Each generation step only requires computing attention for the **single new token** against the growing cache, rather than reprocessing the entire sequence from scratch.

---

##### Why It Matters (Impact)

- **Speed:** Dramatically reduces computation per generation step — turns an otherwise `O(n²)` recomputation pattern per step into a much cheaper incremental update, making autoregressive generation practical for real-time use.
- **Standard practice:** Virtually all production LLM inference systems (GPT, Claude, Llama, etc.) use KV caching as a default optimization for text generation.

---

##### Trade-offs / Challenges

- **Memory cost:** The cache grows linearly with sequence length and must be stored for every layer and every attention head — for long contexts and large models, this can consume significant GPU memory, sometimes becoming the main bottleneck for serving long-context requests.
- **Memory scales with:** Batch size × sequence length × number of layers × number of heads × head dimension × 2 (for both K and V) — this multiplies quickly for large models serving many concurrent users with long contexts.

---

##### Techniques to Reduce KV Cache Memory

- **Multi-Query Attention (MQA):** Shares a single Key/Value head across all query heads (instead of separate K/V per head), drastically reducing cache size at some cost to model quality.
- **Grouped-Query Attention (GQA):** A middle ground — groups multiple query heads to share a smaller number of K/V heads, balancing memory savings with quality (used in models like Llama 2/3).
- **KV cache quantization:** Storing cached K/V vectors in lower precision (e.g., INT8 instead of FP16) to reduce memory footprint.
- **PagedAttention (used in vLLM):** Manages KV cache memory more efficiently using techniques inspired by virtual memory paging, reducing fragmentation and enabling better GPU memory utilization for serving many requests simultaneously.

### Q7. Paged Attention in LLMs

#### PagedAttention in LLMs

- **Core idea:** A memory management technique (introduced by the **vLLM** project) for efficiently handling the KV cache during LLM inference, inspired by **virtual memory and paging** concepts from operating systems — solves the memory fragmentation and inefficiency problems of traditional KV cache management.

---

##### The Problem It Solves

- **Traditional KV cache allocation:** Typically allocates a large, **contiguous block of memory** for each sequence's KV cache, sized for the _maximum possible sequence length_ upfront.
- **Resulting inefficiencies:**
  - **Internal fragmentation:** Since actual generated sequence lengths vary (often much shorter than the max), most of the pre-allocated memory goes unused/wasted for shorter sequences.
  - **External fragmentation:** As sequences of different lengths are allocated and freed dynamically, memory becomes fragmented into scattered unusable chunks over time.
  - **No memory sharing:** Techniques like beam search or parallel sampling (generating multiple outputs from the same prompt) end up duplicating the KV cache for shared prefix tokens, wasting memory unnecessarily.

---

##### How PagedAttention Works

- **Block-based memory allocation:** Instead of one contiguous memory block per sequence, the KV cache is divided into fixed-size **blocks** (analogous to memory "pages" in OS virtual memory), each holding KV vectors for a fixed number of tokens.
- **Non-contiguous storage:** A sequence's KV cache can be stored across multiple **non-contiguous** blocks in GPU memory — a **block table** (like a page table) keeps track of which physical blocks belong to which logical sequence positions.
- **On-demand allocation:** Blocks are allocated only as needed (as the sequence grows token by token), rather than reserving the maximum possible length upfront — eliminating internal fragmentation.
- **Memory sharing via copy-on-write:** When multiple sequences share a common prefix (e.g., same prompt in beam search or parallel sampling), they can **share the same physical blocks** for that prefix instead of duplicating them, only creating separate copies when sequences diverge (copy-on-write) — significantly reducing memory usage in these scenarios.

---

##### Why It Matters (Impact)

- **Near-zero memory waste:** Reduces memory fragmentation from the traditional \~60-80% waste seen in naive KV cache allocation down to a small percentage (only the last, partially-filled block per sequence has minor waste).
- **Higher throughput:** More efficient memory usage means more sequences/requests can be processed concurrently on the same GPU, significantly increasing overall serving throughput.
- **Enables efficient batching:** Since memory is managed more flexibly, dynamic batching of requests with varying sequence lengths becomes much more efficient.
- **Better support for advanced decoding strategies:** Memory sharing makes techniques like beam search, parallel sampling, and prefix caching (for shared system prompts across requests) far more memory-efficient.

---

##### Significance

- PagedAttention is the core innovation behind **vLLM**, one of the most widely adopted high-throughput LLM inference/serving frameworks, and has influenced memory management designs in other inference engines as well.
- An analogous concept to OS-level **virtual memory paging** — the same fundamental idea (page tables, non-contiguous physical storage, on-demand allocation) applied to GPU memory management for LLM serving.

### Q8. What is Flash Attention?

> 🔁 _See also [01-llm-fundamentals.md → Q36](../ai-engineering/01-llm-fundamentals.md#q36-what-is-flash-attention) (separate Notion notes on the same question)._

#### FlashAttention

- **Core idea:** An exact (not approximate) attention algorithm that computes standard self-attention **faster and with much less memory** by being aware of GPU memory hierarchy — reorganizing computation to minimize slow memory reads/writes, without changing the actual attention output.

---

##### The Problem It Solves

- **Standard attention implementation:** Computes the full `Q·K^T` attention matrix (size `n × n` for sequence length `n`), applies softmax, then multiplies by `V` — this requires **materializing** the full `n × n` attention matrix in GPU memory.
- **Memory bottleneck:** GPUs have a small amount of very fast memory (**SRAM**) and a much larger amount of slower memory (**HBM** - High Bandwidth Memory). Standard attention repeatedly reads/writes the large `n × n` matrix to/from slow HBM, which becomes the main bottleneck — not the actual compute (FLOPs), but the **memory I/O**.
- **Quadratic memory usage:** The `n × n` attention matrix grows quadratically with sequence length, making very long sequences memory-prohibitive with standard implementations.

---

##### How FlashAttention Works

- **Tiling/block-wise computation:** Instead of computing the full attention matrix at once, FlashAttention splits Q, K, V into smaller **blocks/tiles** that fit into the GPU's fast SRAM.
- **Fused kernel operations:** Combines the matrix multiplication, softmax, and weighted sum steps into a single fused GPU kernel, avoiding writing intermediate results (like the full attention matrix) back to slow HBM.
- **Online softmax computation:** Uses a technique to compute softmax **incrementally** across blocks (keeping running max and sum statistics), avoiding the need to have the entire row of attention scores available at once to compute a numerically stable softmax.
- **Recomputation for backward pass:** Instead of storing the full attention matrix for use during backpropagation (which would require large memory), FlashAttention **recomputes** necessary values on-the-fly during the backward pass using stored statistics — trading a bit of extra compute for massive memory savings.
- **Exact computation:** Importantly, FlashAttention produces **mathematically identical results** to standard attention — it's an IO-aware reorganization of computation, not an approximation.

---

##### Why It Matters (Impact)

- **Memory efficiency:** Reduces memory usage from **quadratic** (`O(n²)`) to **linear** (`O(n)`) in sequence length, enabling much longer context windows to be processed.
- **Speed:** Significantly faster than standard attention implementations in practice (2-4x+ speedups reported), since it minimizes slow HBM memory access, which was the true bottleneck (not raw compute).
- **Enables longer context windows:** Made training and inference with much longer sequences (thousands to hundreds of thousands of tokens) practical, directly enabling the long-context capabilities seen in modern LLMs.

---

##### Versions

- **FlashAttention (v1):** Introduced the core IO-aware tiling and fused kernel approach.
- **FlashAttention-2:** Improved parallelization and work partitioning across GPU thread blocks, further increasing speed.
- **FlashAttention-3:** Further optimized specifically for newer GPU architectures (e.g., NVIDIA Hopper), leveraging new hardware features for even greater speedups.

---

##### Significance

- Now a **standard component** in most modern LLM training and inference pipelines (used in PyTorch, Hugging Face Transformers, vLLM, and most major LLM frameworks) due to its combination of exactness, speed, and memory efficiency.

### Q9. What is speculative decoding, and how does it speed up inference?

> 🔁 _See also [11-ai-infrastructure-and-scalability.md → Q8](../ai-engineering/11-ai-infrastructure-and-scalability.md#q8-what-is-speculative-decoding-and-how-does-it-speed-up-inference) (separate Notion notes on the same question)._

#### Speculative Decoding

- **Core idea:** A technique to speed up autoregressive LLM inference by using a small, fast **"draft" model** to quickly generate multiple candidate tokens ahead of time, which the large **target model** then verifies in parallel — reducing the number of expensive sequential forward passes needed from the large model.

---

##### The Problem It Solves

- Standard autoregressive generation requires the large LLM to perform **one full forward pass per token**, which is slow since each step is sequential and must wait for the previous token before proceeding.
- This makes generation latency scale directly with output length, and the large model's forward pass is computationally expensive per step.

---

##### How It Works

1. **Draft generation:** A smaller, much faster model (the "draft model") generates a short sequence of `k` candidate tokens autoregressively (e.g., 4-8 tokens), conditioned on the current context.
2. **Parallel verification:** The large target model then processes **all `k` draft tokens in a single forward pass** (since verifying multiple tokens in parallel, given they're already generated, is much cheaper than generating them one-by-one autoregressively) — computing what the target model's actual next-token probabilities would have been at each of those positions.
3. **Accept/reject mechanism:** Compare the draft model's tokens against the target model's probability distributions:
   - If the target model agrees that a draft token was likely (matches within an acceptance criterion, often using rejection sampling to preserve the target model's exact output distribution), it's **accepted**.
   - As soon as a token is **rejected** (the target model disagrees significantly), all draft tokens after that point are discarded, and the target model's own prediction is used instead for that position.
4. **Continue from there:** Generation resumes from the last accepted token (or the corrected token from the target model), and the process repeats — draft again, verify again.

---

##### Why This Speeds Up Inference

- **Fewer sequential large-model passes:** Instead of the large model doing one sequential forward pass per single token, it does **one parallel forward pass to verify multiple tokens at once** — since verification (a single parallel pass over k tokens) is much cheaper than k sequential autoregressive passes.
- **Best case:** If the draft model's predictions are frequently correct (high acceptance rate), many tokens get generated per large-model forward pass, providing significant speedup (commonly reported as 2-3x faster in practice, depending on task and draft model quality).
- **Worst case:** If draft tokens are frequently rejected, the speedup diminishes (falls back closer to standard one-token-at-a-time generation), but **correctness is never compromised** — output quality/distribution matches the target model exactly, regardless of acceptance rate.

---

##### Key Property — Lossless Speedup

- Speculative decoding is designed to produce **statistically identical output** to running the target model alone (via careful rejection sampling), meaning it's a **pure speed optimization** — not an approximation that trades off quality for speed.

---

##### Practical Considerations

- **Draft model choice:** Needs to be small/fast but reasonably aligned with the target model's behavior (e.g., a smaller version of the same model family, or a distilled version) — the closer the draft model mimics the target, the higher the acceptance rate and speedup.
- **Variants:**
  - **Medusa:** Uses multiple extra prediction heads attached to the target model itself to generate draft tokens, avoiding the need for a completely separate draft model.
  - **Self-speculative decoding:** Uses parts of the same large model (e.g., early-exit layers) as the draft mechanism.
- **Adoption:** Used in production inference systems (e.g., various LLM serving frameworks) to reduce latency, especially valuable for real-time/interactive applications.

### Q10. How does continuous batching improve LLM inference throughput?

> 🔁 _See also [11-ai-infrastructure-and-scalability.md → Q7](../ai-engineering/11-ai-infrastructure-and-scalability.md#q7-how-does-continuous-batching-improve-llm-inference-throughput) (separate Notion notes on the same question)._

#### Continuous Batching (Dynamic Batching)

- **Core idea:** A serving optimization that allows new requests to be added to a running batch **as soon as a slot frees up** (when another request finishes generating), instead of waiting for the entire batch to complete before starting new requests — maximizing GPU utilization.

---

##### The Problem It Solves — Static Batching

- **Traditional static batching:** Groups a fixed set of requests together and processes them as a batch until **all** requests in that batch finish generating (i.e., until the longest sequence completes).
- **Inefficiency:** Since different requests generate different numbers of tokens (some finish early, some run long), shorter requests finish, but their GPU slot sits **idle**, waiting for the longest request in the batch to complete — wasting compute capacity.
- New incoming requests must **wait** for the entire current batch to fully finish before being scheduled, increasing latency under load.

---

##### How Continuous Batching Works

- Instead of batching at the **request level**, batching happens at the **iteration/token level** — after every single decoding step (generating one token for all active sequences), the scheduler checks:
  - Which sequences have finished (hit EOS or max length) → removed from the batch.
  - Freed-up slots are immediately filled with **new incoming requests**, which join the batch and start generating from their first token.
- This means the batch composition can change **every single generation step**, keeping the GPU consistently busy processing a full batch, rather than idling on completed slots.

---

##### Why It Improves Throughput

- **Higher GPU utilization:** Eliminates idle time caused by waiting for the slowest sequence in a static batch — the GPU is always processing a full batch of active work.
- **Lower latency for new requests:** New requests don't have to wait for an entire batch cycle to complete; they can start almost immediately when any slot frees up.
- **Better handling of variable-length outputs:** Real-world LLM requests vary widely in output length, and continuous batching naturally adapts to this variability instead of being bottlenecked by the longest sequence.

---

##### Implementation Considerations

- Requires careful memory management since sequences of different lengths/stages are mixed in the same batch at different points in their generation — this pairs naturally with techniques like **PagedAttention** (used in vLLM), which efficiently manages KV cache memory for dynamically changing batches.
- **Adoption:** Core technique used in high-throughput LLM serving systems like **vLLM**, **TensorRT-LLM**, and **Hugging Face TGI (Text Generation Inference)** — considered a standard practice for production LLM serving today.

### Q11. What is pre-training vs fine-tuning in LLMs?

#### Pre-training vs Fine-tuning in LLMs

- **Pre-training:**
  - Trains the model from scratch (random weights) on massive, diverse, **unlabeled** text corpora using a self-supervised objective (typically next-token prediction).
  - Goal: learn general language patterns, grammar, facts, and reasoning ability.
  - Extremely compute/data-intensive (many GPUs/TPUs, weeks of training).
- **Fine-tuning:**
  - Takes the pretrained model and further trains it on a smaller, **task-specific or curated labeled dataset**.
  - Goal: specialize the model's behavior — e.g., follow instructions (instruction tuning), align with human preferences (RLHF), or adapt to a specific domain/task.
  - Much cheaper/faster than pretraining; often uses a smaller learning rate to avoid erasing pretrained knowledge.
- **Key relationship:** Pretraining builds broad general capability; fine-tuning shapes/directs that capability toward specific, desired behavior or tasks.

### Q12. What are some challenges in training LLMs?

#### Challenges in Training LLMs

- **Computational cost:** Requires massive GPU/TPU clusters running for weeks to months, costing millions of dollars — a major barrier to entry.
- **Data quality and curation:** Sourcing, cleaning, and deduplicating trillions of tokens of high-quality text at scale is difficult; low-quality or biased data degrades model performance and behavior.
- **Data contamination:** Ensuring evaluation/benchmark data hasn't leaked into training data, which would inflate perceived performance.
- **Memory constraints:** Storing model weights, optimizer states, gradients, and activations for billion/trillion-parameter models exceeds single-GPU memory, requiring distributed training techniques (model/tensor/pipeline parallelism, ZeRO optimization).
- **Training instability:** Loss spikes, divergence, or NaN losses can occur at scale, requiring careful learning rate scheduling, gradient clipping, and checkpointing/recovery strategies.
- **Hyperparameter tuning at scale:** Expensive to experiment with hyperparameters (learning rate, batch size) directly at full scale, so techniques like scaling laws are used to extrapolate from smaller experiments.
- **Bias and toxicity:** Models can learn and amplify harmful biases or toxic content present in training data, requiring careful data filtering and alignment techniques.
- **Hallucination:** Models can generate fluent but factually incorrect content, which is hard to fully eliminate through training alone.
- **Alignment challenges:** Ensuring the model is helpful, harmless, and honest requires additional stages (RLHF, instruction tuning) beyond pretraining, which are themselves complex and resource-intensive (requiring quality human feedback data).
- **Catastrophic forgetting:** During fine-tuning, the model can lose general capabilities learned during pretraining if not carefully managed (e.g., too high a learning rate, too narrow a fine-tuning dataset).
- **Evaluation difficulty:** Measuring true model quality (reasoning, safety, helpfulness) is harder than simple metrics like perplexity — often requires human evaluation or carefully designed benchmarks, which have their own limitations.
- **Environmental impact:** Large-scale training consumes significant energy, raising sustainability concerns.
- **Reproducibility:** Given the scale and cost, replicating results exactly (or at all) is difficult for most researchers/organizations outside large labs.

### Q13. What is zero-shot learning in the context of LLMs?

#### Zero-Shot Learning in LLMs

- **Core idea:** The ability of a pretrained LLM to perform a new task **without any task-specific training examples**, relying purely on the instructions/description given in the prompt at inference time.
- **How it works:** The model leverages the broad general knowledge and patterns learned during pretraining (and often instruction tuning) to interpret a natural language instruction and generate an appropriate response, even for tasks it wasn't explicitly trained on.
- **Example:** Prompting an LLM with _"Translate the following sentence to French: 'How are you?'"_ — with no example translations provided — and the model correctly outputs _"Comment allez-vous?"_ based on its general language understanding.

---

##### Zero-Shot vs. Few-Shot vs. Fine-Tuning

- **Zero-shot:** No examples provided in the prompt — just an instruction/task description.
- **Few-shot:** A handful of examples are included directly in the prompt (in-context learning) to demonstrate the desired input-output pattern, without updating model weights.
- **Fine-tuning:** Model weights are actually updated using labeled examples for the specific task.

---

##### Why It's Possible

- **Scale and diverse pretraining data:** LLMs trained on massive, diverse text corpora implicitly learn a very wide range of task-like patterns (Q&A, translation, summarization, classification, etc.) embedded within the natural text, enabling generalization to related but unseen task formulations.
- **Instruction tuning:** Fine-tuning on diverse (instruction, response) pairs significantly improves zero-shot performance, since the model becomes better at interpreting and following novel natural language instructions it wasn't explicitly trained on.

---

##### Limitations

- **Performance ceiling:** Zero-shot performance is generally lower than few-shot or fine-tuned performance on complex or highly specific/niche tasks, since the model has no direct examples to calibrate its response format/style.
- **Ambiguity sensitivity:** Without examples, the model may misinterpret ambiguous instructions or produce outputs in an unexpected format.
- **Task complexity limits:** Very specialized or unusual tasks (requiring precise domain knowledge or exact output formatting) often benefit significantly from at least few-shot examples or fine-tuning.

### Q14. How do you handle bias and fairness in LLMs?

#### Handling Bias and Fairness in LLMs

##### Sources of Bias

- **Training data bias:** LLMs learn from massive web-scraped text that reflects historical/societal biases (gender, racial, cultural stereotypes) present in the source data.
- **Representation imbalance:** Certain demographics, languages, dialects, or viewpoints may be over/under-represented in training data.
- **Annotator/labeler bias:** Human feedback used in RLHF/instruction tuning can introduce the biases/preferences of the specific annotators involved.
- **Algorithmic amplification:** Models can amplify subtle biases present in data due to how patterns are statistically reinforced during training.

---

##### Approaches to Mitigate Bias

- **Data curation and filtering:**
  - Carefully curate/filter training data to reduce toxic, biased, or harmful content.
  - Increase representation of underrepresented groups/perspectives through balanced sampling.
  - Deduplicate and remove low-quality sources that may contain concentrated bias.
- **Bias evaluation/benchmarking:**
  - Use fairness benchmarks (e.g., **StereoSet**, **BOLD**, **WinoBias**) to systematically measure biased associations (e.g., gender-occupation stereotypes) before and after mitigation efforts.
  - Regularly audit model outputs across demographic groups for disparities in quality, sentiment, or accuracy.
- **Alignment techniques (RLHF/instruction tuning):**
  - Use diverse human feedback (annotators from varied backgrounds) to reduce single-group bias in preference data.
  - Explicitly train the model to avoid generating biased/stereotypical content through targeted fine-tuning examples.
- **Prompt-level and inference-time mitigation:**
  - System prompts/guidelines that instruct the model toward neutral, balanced responses.
  - Output filtering/moderation layers to catch biased or harmful content before it reaches users.
- **Debiasing techniques (technical):**
  - Adjusting embeddings to reduce biased associations (e.g., reducing gender bias in word vectors via projection methods).
  - Counterfactual data augmentation — creating training examples with swapped demographic attributes (e.g., swapping gender pronouns) to reduce spurious correlations.
- **Red-teaming:**
  - Proactively testing models with adversarial prompts designed to surface biased, harmful, or unfair outputs before deployment.

---

##### Ongoing Challenges

- **Bias is hard to fully eliminate:** Even with mitigation, subtle biases can persist or shift form (e.g., reducing one type of bias may inadvertently introduce another).
- **Defining "fairness" is context-dependent:** Different fairness definitions (equal treatment vs. equal outcomes) can conflict, and what counts as "fair" can vary by culture, application, and stakeholder.
- **Trade-offs with other objectives:** Aggressive bias mitigation can sometimes reduce model helpfulness/accuracy or lead to overly cautious, evasive responses.
- **Evaluation limitations:** Current bias benchmarks are imperfect proxies and don't capture the full complexity of real-world fairness concerns.
- **Continuous monitoring needed:** Bias mitigation isn't a one-time fix — models need ongoing evaluation and updates as societal norms and use cases evolve.

### Q15. What are some real-world applications of LLMs in business and tech?

#### Real-World Applications of LLMs in Business and Tech

- **Customer support:** AI chatbots/virtual assistants handling queries, ticket triage, and 24/7 support (reducing human agent workload).
- **Content creation:** Drafting marketing copy, blog posts, product descriptions, social media content, and ad copy.
- **Code generation/assistance:** Tools like GitHub Copilot, Claude Code — code completion, debugging, code review, and documentation generation.
- **Summarization:** Condensing long documents, meeting notes, legal contracts, research papers, or emails into concise summaries.
- **Search and knowledge retrieval:** Powering enterprise search and Q&A systems (often via RAG) over internal documents/knowledge bases.
- **Translation and localization:** Real-time or batch translation for global business communication and product localization.
- **Sentiment analysis/market research:** Analyzing customer reviews, social media, and survey responses at scale for insights.
- **Personalized recommendations:** Enhancing recommendation systems with natural language understanding of user preferences/queries.
- **Data extraction/document processing:** Extracting structured information from invoices, contracts, resumes, forms (often paired with OCR).
- **Sales and marketing automation:** Drafting personalized outreach emails, lead qualification, CRM data enrichment.
- **Legal/compliance:** Contract review, clause extraction, compliance checking against regulations/playbooks.
- **Healthcare:** Clinical note summarization, medical literature review assistance, patient communication drafting (with human oversight).
- **Education:** Personalized tutoring, automated grading/feedback, content generation for курс materials.
- **HR/recruiting:** Resume screening, interview question generation, job description drafting.
- **Financial services:** Report generation, earnings call summarization, risk analysis assistance, fraud pattern detection support.
- **Virtual agents/copilots:** Embedded AI assistants within productivity tools (e.g., Microsoft Copilot, Notion AI) to draft, edit, and analyze content directly in workflows.

### Q16. How does the Transformer architecture improve LLM performance over RNNs?

#### How Transformers Improve LLM Performance Over RNNs

- **Parallelization:** RNNs process tokens sequentially (each step depends on the previous), preventing parallel training. Transformers process the entire sequence at once via self-attention, enabling massive parallelization on GPUs/TPUs — critical for training on today's huge datasets within a feasible time.
- **Long-range dependencies:** RNNs suffer from vanishing gradients across many time steps, weakening their ability to connect distant tokens even with LSTM/GRU improvements. Transformers connect any two tokens directly in a single step via self-attention, regardless of distance, capturing long-range context far more effectively.
- **No information bottleneck:** RNNs compress sequence history into a single hidden state that gets overwritten at each step, losing information over long sequences. Transformers give every token direct access to every other token's representation, avoiding this compression bottleneck.
- **Better scalability:** Transformers scale much more effectively with more data and parameters — this scalability is precisely why large language models (GPT, Claude, etc.) became possible, whereas RNN-based models hit performance/scaling walls much earlier.
- **Stable gradient flow:** Residual connections and layer normalization in Transformers support training very deep networks (dozens of layers) more stably than deep RNN chains, which suffer more from vanishing/exploding gradients.
- **Richer contextual representations:** Multi-head attention lets the model simultaneously capture different types of relationships (syntactic, semantic, positional) at each layer, versus a single sequential hidden-state pathway in RNNs.
- **Net result:** These properties combined are why Transformers replaced RNNs as the foundation for virtually all modern LLMs — enabling both the training scale and the modeling quality needed for today's language capabilities.

### Q17. Explain the Query(Q), Key(K), and Value(V) in attention.

> 🔁 _See also [01-llm-fundamentals.md → Q16](../ai-engineering/01-llm-fundamentals.md#q16-explain-the-queryq-keyk-and-valuev-in-attention) (separate Notion notes on the same question)._

#### Query, Key, Value in Attention

- **Analogy:** Think of it like a search/retrieval system — **Query** is what you're looking for, **Key** is like a label/index for each piece of information, **Value** is the actual content retrieved.
- **Query (Q):** A vector representing the current token's "question" — what information it's seeking from the rest of the sequence.
- **Key (K):** A vector representing each token's "label" — what kind of information that token offers, used to match against queries.
- **Value (V):** A vector representing the actual content/information of each token, which gets passed forward if that token is deemed relevant.
- **How they interact:**
  - Compute similarity between the current token's Query and every token's Key via dot product: `Q · K^T` — this gives a relevance score for each token pair.
  - Scale by `√d_k` and apply **softmax** to turn scores into attention weights (summing to 1).
  - Multiply these weights by the corresponding **Value** vectors and sum them up — producing a weighted combination of Values, weighted by how relevant each token was.
  - Formula: `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
- **Where Q, K, V come from:** Each is a learned linear projection of the same input embedding (via separate weight matrices `W_Q`, `W_K`, `W_V`) — so the model learns what aspects of a token to use as its query, key, and value.
- **Self-attention vs. cross-attention:**
  - **Self-attention:** Q, K, V all come from the same sequence (e.g., encoder attending to itself).
  - **Cross-attention:** Q comes from one sequence (e.g., decoder), while K and V come from another (e.g., encoder output) — used to relate two different sequences.

### Q18. What is self-attention, and how does it work in Transformers?

> 🔁 _See also [01-llm-fundamentals.md → Q17](../ai-engineering/01-llm-fundamentals.md#q17-what-is-self-attention-and-how-does-it-work-in-transformers) (separate Notion notes on the same question)._

#### Self-Attention in Transformers

- **Core idea:** A mechanism that lets each token in a sequence look at **every other token in the same sequence** and decide how much to "attend to" (weigh) each one when building its own representation — capturing context and relationships regardless of distance.
- **How it works (building on Q/K/V):**
  - Each token generates its own Query, Key, and Value vectors (via learned projections).
  - Each token's Query is compared against **every token's Key** (including itself) to compute relevance scores.
  - Scores are scaled and passed through softmax to get attention weights.
  - The token's new representation = weighted sum of all tokens' **Values**, using those attention weights.
- **"Self" part:** Q, K, and V all come from the **same sequence** — tokens attend to each other within one sequence, unlike cross-attention where Q comes from one sequence and K/V from another.
- **Result:** Every token's output representation is a context-aware blend of the entire sequence, so words can incorporate relevant information from anywhere else in the sentence in a single step (e.g., resolving "it" by attending back to the noun it refers to).
- **Multi-head extension:** Multiple self-attention operations run in parallel (multi-head attention), each learning to focus on different types of relationships (syntax, semantics, position), then combined.
- **Why it's powerful:** Unlike RNNs (sequential, information decays over distance) or CNNs (local receptive field), self-attention gives direct, parallel access between any two tokens regardless of position — enabling both long-range dependency modeling and full parallelization during training.

### Q19. What is cross-attention in Transformers?

> 🔁 _See also [01-llm-fundamentals.md → Q18](../ai-engineering/01-llm-fundamentals.md#q18-what-is-cross-attention-in-transformers) (separate Notion notes on the same question)._

#### Cross-Attention in Transformers

- **Core idea:** An attention mechanism where the **Query** comes from one sequence, while the **Key and Value** come from a **different** sequence — allowing one sequence to attend to and pull relevant information from another.
- **Contrast with self-attention:** In self-attention, Q, K, and V all come from the same sequence (a sequence attending to itself). In cross-attention, Q comes from sequence A, K and V come from sequence B — relating two distinct sequences.

---

##### How It Works

- The formula is the same as regular attention: `Attention(Q, K, V) = softmax(QK^T / √d_k) * V`
- The difference is only in the **source** of the vectors:
  - **Query (Q):** Derived from the target/decoder sequence (what it's currently generating/processing).
  - **Key (K) and Value (V):** Derived from the source/encoder sequence (the reference information being attended to).
- Each decoder token's query is compared against all encoder tokens' keys to determine relevance, then combines the corresponding encoder values weighted accordingly.

---

##### Where It's Used

- **Encoder-Decoder Transformers (e.g., original Transformer, T5, BART):** In the decoder, after masked self-attention, a **cross-attention layer** lets each decoder token attend to the full encoder output — this is how the decoder "looks back" at the source input while generating output (e.g., attending to relevant source words during translation).
- **Multimodal models:** Used to let one modality attend to another — e.g., in image captioning, text tokens (Q) attend to image patch features (K, V) to ground generated words in relevant visual regions.
- **Retrieval-augmented generation (RAG):** Generated text can cross-attend to retrieved document embeddings.

---

##### Example (Machine Translation)

- The encoder processes the source sentence (e.g., French) into contextual representations (K, V).
- Decoder generates the target sentence (e.g., English) token by token; at each step, the decoder's current token representation (Q) cross-attends to the encoder's K/V, pulling in relevant source-language context to decide what to generate next.

---

##### Key Takeaway

- Cross-attention is the mechanism that **connects two different sequences/modalities**, enabling one to be conditioned on / informed by the other — essential for tasks like translation, summarization (input document → output summary), image captioning, and multimodal fusion.

### Q20. What are multi-head attention mechanisms? Why use multiple attention heads?

> 🔁 _See also [01-llm-fundamentals.md → Q21](../ai-engineering/01-llm-fundamentals.md#q21-what-are-multi-head-attention-mechanisms-why-use-multiple-attention-heads) (separate Notion notes on the same question)._

##### Multi-Head Attention

Multi-head attention is the mechanism at the core of the Transformer architecture. Instead of computing a single attention function over the queries, keys, and values, it runs several attention computations ("heads") in parallel, each with its own learned projections, then combines the results.

**How it works, step by step:**

1. **Linear projections**: The input (queries Q, keys K, values V) is projected into multiple lower-dimensional subspaces using separate learned weight matrices for each head. If the model dimension is `d_model` and there are `h` heads, each head typically works in a space of dimension `d_model / h`.
2. **Parallel attention**: Each head independently computes standard scaled dot-product attention:<br />`Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`
3. **Concatenation**: The outputs of all heads are concatenated back together.
4. **Final projection**: The concatenated result is passed through one more learned linear layer to produce the final output, mixing information across heads.

##### Why Use Multiple Heads?

**1. Different representation subspaces**

A single attention head averages over all the ways tokens might relate to each other, which can wash out distinct patterns. Multiple heads let the model attend to information from different representation subspaces simultaneously — one head might specialize in tracking syntactic relationships (like subject-verb agreement), another in coreference (matching pronouns to nouns), another in positional/local patterns.

**2. Attending to different positions differently**

With one head, the attention weights for a given query are a single distribution — it's hard to focus sharply on more than one thing at once. Separate heads can each attend to different positions or context windows for the same word, capturing multiple relevant relationships in parallel rather than blending them into one averaged signal.

**3. Increased model capacity without much added compute**

Because each head operates in a reduced dimension (`d_model / h`), the total computation is similar to one full-dimensional attention operation, but the model gains more expressive power — each head learns its own query/key/value projections, effectively giving the model several "views" of the data for roughly the same cost.

**4. Empirical robustness and stability**

In practice, multi-head attention tends to produce more stable and higher-quality representations. Some heads end up somewhat redundant, but ablation studies (and interpretability work since) show many heads specialize in interpretable, distinct patterns — evidence that the diversity is functionally useful rather than incidental.

**A useful analogy**: imagine reading a sentence and simultaneously tracking grammar, meaning, tone, and references to earlier sentences. Doing all of that with one "lens" forces compromises. Multi-head attention is like having several specialized readers, each focused on one aspect, then pooling their notes together.

### Q21. How does attention help capture long-range dependencies?

##### Attention and Long-Range Dependencies

The core reason attention handles long-range dependencies well comes down to the **path length** between any two positions in a sequence.

**The path length problem in older architectures**

In RNNs, information from position 1 has to pass through every intermediate hidden state to influence position 1000. Each step risks losing or distorting that information (vanishing/exploding gradients), so the "signal" from distant tokens degrades the further it has to travel. Convolutional architectures reduce this somewhat but still need stacked layers (or dilation) to expand the receptive field across long distances.

**How attention changes this**

Attention connects every position to every other position directly, in a single step. The path length between any two tokens, no matter how far apart, is O(1) — constant — rather than O(n) as in RNNs. When computing the representation for a word near the end of a document, attention can directly weigh its relationship to a word at the very beginning, with no intermediate steps for information to pass through or degrade across.

Concretely: for each query, the model computes a similarity score against _every_ key in the sequence, then takes a weighted sum of _all_ values based on those scores. Distance in the sequence doesn't enter into the calculation at all — only the pattern-matching between the query and key vectors does. A relevant token 500 positions away is just as directly accessible as one right next door.

**What this enables in practice**

- Resolving coreference across long spans (a pronoun referring to a noun many sentences earlier).
- Maintaining thematic or logical consistency across a long document.
- Capturing dependencies like matching an opening clause to a distant closing clause, or connecting a question to relevant context buried deep in a passage.

**The trade-off**

This comes at a cost: computing all pairwise interactions is O(n²) in sequence length, since every token attends to every other token. That's the well-known scalability bottleneck of standard attention, and it's why techniques like sparse attention, linear attention approximations, and sliding-window methods exist — they try to preserve the "any-to-any" long-range benefit while cutting down the computation.

**One caveat**

Direct connectivity doesn't automatically mean the model _learns_ to use distant information well — it still depends on training, data, and what patterns are useful for the task. But architecturally, attention removes the structural bottleneck that made long-range dependencies hard for earlier models.

### Q22. What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?

> 🔁 _See also [01-llm-fundamentals.md → Q38](../ai-engineering/01-llm-fundamentals.md#q38-what-is-grouped-query-attention-gqa-and-how-does-it-differ-from-multi-head-attention-mha) (separate Notion notes on the same question)._

##### Grouped-Query Attention (GQA)

GQA is a variant of attention designed to reduce the memory and speed costs of inference (specifically the key-value cache) while trying to preserve most of the quality of standard multi-head attention.

**Quick recap: the problem GQA solves**

In standard Multi-Head Attention (MHA), each of the `h` heads has its own separate set of Query, Key, and Value projections. During autoregressive generation, the model caches the Keys and Values for all previous tokens (the "KV cache") so it doesn't have to recompute them at every new step. But since every head has its own K and V, the cache size scales directly with the number of heads — this becomes a major memory bottleneck for long sequences and large models, and it also limits how much you can batch during inference.

**How GQA works**

GQA sits between two extremes:

1. **Multi-Head Attention (MHA)**: `h` query heads, `h` key heads, `h` value heads. Maximum expressiveness, maximum KV cache size.
2. **Multi-Query Attention (MQA)**: `h` query heads, but only **1** shared key head and **1** shared value head. Minimal KV cache, but can lose some quality since all query heads are forced to share the same keys/values.
3. **Grouped-Query Attention (GQA)**: `h` query heads are split into `g` groups, and each group shares a single set of key/value heads. So instead of `h` KV heads (MHA) or 1 KV head (MQA), you have `g` KV heads, where `1 < g < h`.

Concretely, if you have 32 query heads and set `g = 8` groups, then every 4 query heads share one K/V head — reducing the KV cache size by 4x compared to full MHA, while still giving more diversity than MQA's single shared K/V.

**Key differences from MHA**

| Aspect | MHA | GQA |
| --- | --- | --- |
| Query heads | h | h |
| Key/Value heads | h (one per query head) | g (shared across groups of query heads) |
| KV cache size | Largest | Reduced by factor of h/g |
| Inference speed | Slower (more memory bandwidth needed) | Faster |
| Quality | Highest (most expressive) | Close to MHA, better than MQA |
| Training cost | Same as always | Can be obtained cheaply by "uptraining" an existing MHA checkpoint, converting it via mean-pooling the KV heads into groups |

**Why this trade-off works well in practice**

The intuition is that the _value_ of having many independent heads mostly comes from the queries doing different pattern-matching, not necessarily from each head needing entirely separate keys and values. Sharing K/V within a group forces some heads to compress their "view" of the sequence, but empirically this costs very little in quality while saving a lot in memory bandwidth — since attention inference at scale is often memory-bandwidth-bound (loading the KV cache) rather than compute-bound, shrinking the cache directly speeds up generation.

### Q23. What are Feed-Forward Networks in LLMs?

> 🔁 _See also [01-llm-fundamentals.md → Q22](../ai-engineering/01-llm-fundamentals.md#q22-what-are-feed-forward-networks-in-llms) (separate Notion notes on the same question)._

##### Feed-Forward Networks (FFNs) in LLMs

The Feed-Forward Network is the other major sublayer in a Transformer block, alongside attention. While attention handles _mixing information across positions_ (letting tokens look at each other), the FFN handles _processing information within each position_ independently.

**Basic structure**

Each Transformer layer typically looks like:

```txt
x → Self-Attention → Add & Norm → Feed-Forward → Add & Norm → next layer
```

The FFN itself is a simple two-layer fully connected network applied identically (with shared weights) to every position in the sequence:

`FFN(x) = W₂ · activation(W₁ · x + b₁) + b₂`

- **First linear layer (W₁)**: projects the input from `d_model` up to a much larger hidden dimension, typically `4 × d_model` (e.g., 512 → 2048).
- **Activation function**: a nonlinearity applied elementwise. Originally ReLU in the first Transformer paper; modern LLMs mostly use variants like GELU or, more commonly now, GLU-based activations such as SwiGLU (used in LLaMA, PaLM, and others).
- **Second linear layer (W₂)**: projects back down from the large hidden dimension to `d_model`, so the output can be added back into the residual stream.

Critically, this network is applied **position-wise** — the exact same weights are used at every token position, but each token passes through independently, with no interaction between positions inside the FFN. All the cross-token mixing already happened in the attention sublayer before it.

**Why the "expand then contract" shape?**

Projecting up to a wider hidden dimension gives the network more room to represent complex, nonlinear transformations of each token's representation before compressing back down. Conceptually, this is where the model does much of its "thinking" about a single token's representation — combining and transforming the features that attention just gathered from other positions.

**Why FFNs matter so much**

- **Capacity**: FFNs contain the majority of a Transformer's parameters — often roughly two-thirds of the total, since the hidden dimension is so much larger than `d_model`. Much of what a model "knows" (factual associations, learned patterns) is thought to be stored substantially in these weights.
- **Nonlinearity**: Attention itself is fundamentally about weighted averaging (largely linear/convex combinations of value vectors). Without the FFN's nonlinear activation function in between, stacking attention layers wouldn't add much representational power — the FFN is where the real nonlinear transformation happens.
- **Per-token computation**: It lets the model apply complex, learned transformations to each token's features independently, complementing attention's role of gathering context across tokens.

**Modern variations**

- **SwiGLU / GLU variants**: instead of a single activation, these use a gating mechanism — one linear projection is passed through an activation and multiplied elementwise with another linear projection, before the final down-projection. This tends to improve quality for a similar parameter/compute budget, and is now standard in models like LLaMA, PaLM, and Mistral.
- **Mixture-of-Experts (MoE)**: rather than one dense FFN per layer, a MoE layer contains many separate FFN "experts," and a small router network selects only a few experts to process each token. This lets the model scale total parameter count massively while keeping the compute cost per token roughly constant, since only a fraction of the experts are activated per token. Models like Mixtral and some versions of GPT-4 reportedly use this approach.

### Q24. Tokenization in Large Language Models (LLMs).

##### Tokenization in Large Language Models

Tokenization is the process of converting raw text into a sequence of discrete units (tokens) that a model can process numerically. It's the first step in any LLM pipeline — before any embedding, attention, or FFN computation happens, text has to be broken into tokens and mapped to integer IDs.

**Why not just use words or characters?**

- **Word-level tokenization**: Simple, but the vocabulary explodes (every inflection, typo, and rare word needs its own entry), and the model can't handle words it's never seen before (out-of-vocabulary problem).
- **Character-level tokenization**: Small vocabulary, no out-of-vocabulary issues, but sequences become very long (a sentence becomes hundreds of tokens), making training and inference more expensive since compute scales with sequence length.

Modern LLMs use a middle ground: **subword tokenization**.

##### Subword Tokenization Approaches

**1. Byte-Pair Encoding (BPE)**

Starts with individual characters (or bytes) as the base vocabulary, then iteratively merges the most frequently co-occurring pairs into new tokens, repeating until reaching a target vocabulary size (e.g., 50,000 tokens). Common words end up as single tokens, while rare words get split into meaningful subword pieces. GPT-2, GPT-3, and many others use variants of this.

**2. Byte-level BPE**

Operates on raw UTF-8 bytes rather than Unicode characters as the base units. This guarantees any input — including emoji, unusual symbols, or any language — can always be tokenized without an "unknown token" fallback, since every possible byte sequence is representable.

**3. WordPiece**

Similar to BPE, but merges are chosen based on maximizing the likelihood of the training data rather than raw frequency. Used in BERT and related models.

**4. SentencePiece / Unigram**

Treats tokenization as a probabilistic model — starts with a large set of candidate subwords and prunes them down based on which ones maximize the likelihood of the corpus. SentencePiece also treats the input as a raw stream (including spaces) rather than pre-splitting on whitespace, which makes it easier to apply consistently across languages, including ones without clear word boundaries like Chinese or Japanese.

##### How It Works in Practice

A sentence like:

`"Tokenization is fascinating"`

might get split into something like:

`["Token", "ization", "Ġis", "Ġfascin", "ating"]`

(the `Ġ` represents a leading space in some BPE implementations). Common words stay whole; rarer or morphologically complex words get broken into recognizable pieces, and truly novel strings can always be built up from smaller fragments — down to individual bytes/characters if necessary.

##### Why Tokenization Choices Matter

- **Vocabulary size trade-off**: A larger vocabulary means fewer tokens per sentence (shorter sequences, less compute per input) but a bigger embedding/output matrix and more parameters. A smaller vocabulary means smaller matrices but longer token sequences.
- **Efficiency and cost**: Since compute and context window limits are measured in tokens, tokenization efficiency directly affects how much text fits in a given context window and how expensive inference is per unit of actual text. This is also why API pricing is denominated in tokens rather than words or characters.
- **Multilingual fairness**: Tokenizers trained primarily on English text often split non-English text (especially non-Latin scripts) into far more tokens per word than English, meaning the same sentence in another language can cost significantly more tokens — a well-known limitation being addressed with more balanced, multilingual tokenizer training.
- **Arithmetic and character-level tasks**: Because tokens don't align with characters, models can struggle with tasks like counting letters in a word or precise character manipulation — the model often doesn't "see" individual characters directly, only the token chunks they were grouped into.
- **Numbers**: Some tokenizers split numbers inconsistently (e.g., "1234" might become "123" + "4" or "12" + "34" depending on the vocabulary), which can hurt arithmetic performance. Newer tokenizers often deliberately tokenize digits individually or in consistent groups to mitigate this.

### Q25. What is subword tokenization?

##### Subword Tokenization

- Splits text into units smaller than words but larger than characters — common words stay whole, rare/complex words break into meaningful chunks
- Example: `"unhappiness"` → `["un", "happi", "ness"]`
- **Solves whole-word tokenization's problem**: massive vocabulary, no way to handle unseen words
- **Solves character-level tokenization's problem**: extremely long sequences, weak per-token meaning
- **Vocabulary is learned statistically** from a text corpus via algorithms like:
  - **BPE**: merges most frequent adjacent character pairs iteratively
  - **WordPiece**: merges chosen to maximize training-data likelihood
  - **Unigram/SentencePiece**: starts large, prunes down to best-fitting subwords
- **Key benefits**:
  - No out-of-vocabulary tokens — any string can fall back to smaller fragments
  - Captures morphology — `play`, `playing`, `played` share the root `play`
  - Balances sequence length vs. vocabulary size
- Typical vocabulary size: 30,000–100,000+ tokens

### Q26. What is BPE (Byte Pair Encoding) in LLMs?

##### Byte Pair Encoding (BPE)

- A subword tokenization algorithm that builds a vocabulary by iteratively merging the most frequent pair of adjacent symbols

**How it's trained:**

- Start with a base vocabulary of individual characters (or bytes)
- Count all adjacent symbol pairs in the training corpus
- Merge the most frequent pair into a single new token
- Repeat until reaching a target vocabulary size (e.g., 50,000 tokens)

**Example walkthrough:**

- Start: `"low lower lowest"` → characters: `l o w l o w e r l o w e s t`
- Merge `l`+`o` → `lo` (most frequent pair)
- Merge `lo`+`w` → `low`
- Continue merging until common patterns like `low`, `er`, `est` become single tokens

**At inference time:**

- New text is tokenized by applying the learned merge rules in the same order they were learned
- Frequent words end up as single tokens; rare words get broken into smaller learned pieces

**Byte-level BPE variant:**

- Operates on raw UTF-8 bytes instead of Unicode characters
- Guarantees every possible input is representable — no unknown-token fallback needed
- Used in GPT-2, GPT-3, and many modern LLMs

**Why it's popular:**

- Simple, frequency-driven, no linguistic rules needed
- Automatically balances vocabulary size against sequence length
- Handles rare words, typos, and multiple languages without breaking

### Q27. What is positional embedding in LLMs?

##### Positional Embeddings

- Attention has no inherent sense of order — it computes weighted sums over all tokens simultaneously, so without extra information, `"dog bites man"` and `"man bites dog"` would look identical to the attention mechanism
- Positional embeddings inject information about each token's position in the sequence, so the model knows _where_ each token sits, not just _what_ it is

**How they're added:**

- A positional vector (same dimension as the token embedding) is added to each token's embedding before it enters the Transformer layers
- Final input = token embedding + positional embedding

##### Main Types

**1. Sinusoidal (fixed) positional encoding**

- Used in the original Transformer paper
- Each position gets a unique pattern of sine/cosine values at different frequencies
- Not learned — computed with a fixed formula
- Can theoretically generalize to sequence lengths longer than seen in training

**2. Learned positional embeddings**

- A trainable embedding vector for each position (like a lookup table), learned during training
- Used in models like BERT, GPT-2
- Limited to the maximum sequence length seen during training

**3. Relative positional encoding**

- Encodes the _distance_ between tokens rather than their _absolute_ position
- Helps the model generalize better to varying sequence lengths and focus on relationships rather than fixed positions

**4. Rotary Positional Embedding (RoPE)**

- Encodes position by rotating the query and key vectors in a way that depends on their position
- The dot product between rotated Q and K naturally captures relative position
- Widely used in modern LLMs (LLaMA, GPT-NeoX, Mistral) due to better length generalization and efficiency

**Why it matters:**

- Without positional information, word order — which fundamentally changes meaning — would be invisible to the model
- The choice of positional encoding affects how well a model generalizes to sequences longer than those seen during training

### Q28. What is temperature in the context of LLMs?

##### Temperature in LLMs

- A hyperparameter that controls the randomness/confidence of a model's output when sampling the next token
- Applied to the logits (raw output scores) before they're converted into probabilities via softmax

**How it works:**

`P(token) = softmax(logits / T)`

- Dividing logits by temperature `T` before softmax reshapes the probability distribution
- **T = 1**: standard softmax, no change — probabilities reflect the model's raw confidence
- **T \< 1 (e.g., 0.2)**: sharpens the distribution — high-probability tokens become even more dominant, output becomes more deterministic/focused
- **T > 1 (e.g., 1.5)**: flattens the distribution — probabilities become more uniform, output becomes more random/diverse
- **T → 0**: approaches greedy decoding (always picks the single highest-probability token)

**Practical effect:**

| **Temperature** | **Behavior** | **Use case** |
| --- | --- | --- |
| Low (0–0.3) | Focused, consistent, deterministic | Factual Q&A, code generation, math |
| Medium (0.5–0.8) | Balanced creativity and coherence | General conversation, writing assistance |
| High (0.9–1.5+) | Diverse, creative, sometimes incoherent | Brainstorming, creative writing, poetry |

**Why it exists:**

- Without any randomness, the model would always output the same response to the same prompt (greedy decoding), which can be repetitive or get stuck in loops
- Without any control, fully random sampling from raw probabilities could pick low-quality, incoherent tokens too often
- Temperature gives a tunable knob between deterministic and creative output

**Often used alongside other sampling controls:**

- **Top-k sampling**: only consider the k most probable tokens
- **Top-p (nucleus) sampling**: only consider the smallest set of tokens whose cumulative probability exceeds p
- These are typically combined with temperature to further shape which tokens are eligible before the randomness is applied

### Q29. What is causal masking?

> 🔁 _See also [01-llm-fundamentals.md → Q20](../ai-engineering/01-llm-fundamentals.md#q20-what-is-causal-masking) (separate Notion notes on the same question)._

##### Causal Masking

- A technique used in autoregressive Transformers (like GPT-style models) to prevent a token from attending to future tokens — it can only see itself and everything before it
- Enforces the constraint that predictions for position `i` can only depend on positions `1, 2, ..., i`, never on `i+1, i+2, ...`

**Why it's needed:**

- Standard self-attention lets every token attend to every other token, including ones that come later in the sequence
- For language modeling (predicting the next token), letting a model "see" the future answer while training would make the task trivial and useless — the model would just learn to copy the next token instead of actually learning to predict it
- Causal masking blocks this by hiding future positions during both training and generation

**How it works:**

- Before the softmax step in attention, a mask is applied to the raw attention scores (QKᵀ)
- Positions corresponding to future tokens are set to `∞` (negative infinity)
- After softmax, `∞` scores become 0 probability — so future tokens contribute nothing to the weighted sum, regardless of how relevant they might look
- Positions at or before the current one keep their normal computed scores

**Visual example (mask matrix for a 4-token sequence):**

```txt
        tok1  tok2  tok3  tok4
tok1  [  0    -∞    -∞    -∞  ]
tok2  [  0     0    -∞    -∞  ]
tok3  [  0     0     0    -∞  ]
tok4  [  0     0     0     0  ]
```

- Each row represents what that token is allowed to attend to (0 = allowed, -∞ = blocked)
- This creates a lower-triangular pattern — token `i` can only look left (at itself and earlier tokens)

**Why this matters for training efficiency:**

- Causal masking allows **parallel training** on the entire sequence at once — since every position's prediction is independently constrained to only use past context, the model can compute losses for all positions in a single forward pass, rather than needing to generate one token at a time during training
- This is a major reason Transformers train much faster than RNNs despite both being sequential in nature at inference time

**Causal vs. Bidirectional masking:**

| Type | Can see future? | Used in |
| --- | --- | --- |
| Causal (masked) | No | GPT, LLaMA, Claude — generative/autoregressive models |
| Bidirectional (unmasked) | Yes | BERT — encoder-only models for understanding tasks (classification, embeddings) |

### Q30. What are skip connections?

##### Skip Connections (Residual Connections)

- A shortcut that adds a layer's input directly to its output, bypassing ("skipping") the transformation the layer performs
- Formally: `output = x + F(x)`, where `F(x)` is the transformation (e.g., attention or FFN sublayer) and `x` is the original input

**Where they're used in Transformers:**

- Around every sublayer — both the attention block and the feed-forward block
- Combined with layer normalization, typically as: `x = LayerNorm(x + Sublayer(x))`

```txt
x → Self-Attention → (+x) → Add & Norm → Feed-Forward → (+x) → Add & Norm → next layer
      └──────────────┘                      └──────────────┘
       skip connection                       skip connection
```

##### Why They're Needed

**1. Solves the vanishing gradient problem**

- In deep networks, gradients can shrink toward zero as they backpropagate through many layers, making early layers train very slowly or not at all
- The `+x` term provides a direct path for gradients to flow backward unimpeded, since the derivative of `x + F(x)` with respect to `x` always includes a clean `1`, regardless of how complex `F(x)` is
- This makes it possible to train very deep networks (dozens or hundreds of layers) that would otherwise be untrainable

**2. Preserves information across layers**

- Without skip connections, each layer could potentially distort or lose information from earlier representations
- The skip path guarantees the original input is always available at the output, so layers only need to learn the _residual_ (the difference/adjustment), not reconstruct everything from scratch

**3. Makes learning the identity function easy**

- If a particular layer isn't useful for a given input, the network can effectively learn `F(x) ≈ 0`, making the layer act as a no-op that just passes `x` through
- This is much easier than forcing a layer without skip connections to learn to output exactly its input unchanged

**4. Smoother loss landscape**

- Empirically, networks with skip connections have a smoother, easier-to-optimize loss landscape, which helps with faster and more stable convergence during training

**Origin:**

- Introduced in ResNet (2015) for deep convolutional image networks, showing that even 100+ layer networks could be trained effectively
- Adopted directly into the Transformer architecture, where they're essential given how many layers modern LLMs stack (often 32, 64, or more)

### Q31. How does Rotary Position Embedding (RoPE) work, and why is it preferred over learned positional embeddings?

> 🔁 _See also [01-llm-fundamentals.md → Q39](../ai-engineering/01-llm-fundamentals.md#q39-how-does-rotary-position-embedding-rope-work-and-why-is-it-preferred-over-learned-positional-embeddings) (separate Notion notes on the same question)._

##### Rotary Position Embedding (RoPE)

- A method of injecting positional information by **rotating** the query and key vectors in multi-dimensional space, rather than adding a separate positional vector to the token embedding
- Position is encoded through the _angle_ of rotation applied to Q and K vectors before computing attention scores

##### How It Works

**Core mechanism:**

- Each Q and K vector is split into pairs of dimensions (treated like 2D coordinates)
- Each pair is rotated by an angle proportional to the token's position `m` and a frequency specific to that pair: `θ = m × ω_i`
- Different pairs of dimensions rotate at different frequencies — similar in spirit to the sinusoidal encoding's use of multiple frequencies, but applied as a rotation rather than an addition

**The key mathematical property:**

- When you take the dot product of a rotated query (at position `m`) and a rotated key (at position `n`), the result depends only on their **relative distance** `(m - n)`, not their absolute positions
- This falls out naturally from rotation math: rotating two vectors by their own angles and then computing their dot product is equivalent to rotating one vector by the _difference_ of the two angles

**Simplified 2D intuition:**

- Imagine Q and K as arrows on a clock face
- Position determines how far each arrow gets rotated
- The angle _between_ the two arrows after rotation — which is what the dot product effectively measures — depends only on how far apart their positions are, not where they started

##### Why RoPE Is Preferred Over Learned Positional Embeddings

**1. Naturally encodes relative position**

- Learned/absolute positional embeddings tell the model "this token is at position 5," but the model has to _learn_ to infer relative relationships (like "these two tokens are 3 apart") from that
- RoPE bakes relative position directly into the attention computation itself, which tends to align better with what language actually depends on — relationships between tokens, not their absolute index

**2. Better length generalization**

- Learned positional embeddings have a fixed lookup table sized to the max training sequence length — feeding a longer sequence at inference means positions with no trained embedding, so it fails or performs poorly
- RoPE is computed with a formula (not a lookup table), so it can be applied to positions beyond what was seen in training, generally generalizing better to longer sequences (though it still degrades at very long extrapolation without extra tricks)

**3. No added parameters**

- Learned positional embeddings add a full trainable embedding table (`max_seq_len × d_model` parameters)
- RoPE requires no extra learned parameters — it's a fixed mathematical transformation applied at attention time

**4. Compatible with efficient attention mechanisms**

- Because RoPE modifies Q and K directly rather than the input embedding, it integrates cleanly into optimized attention implementations (like those used with KV caching, GQA, and Flash Attention) without extra overhead

**5. Decays naturally with distance (empirically)**

- Even without explicit design for it, RoPE's rotational structure tends to produce attention scores that decay somewhat as relative distance increases, which loosely mirrors the intuition that closer tokens are often more relevant

##### Quick Comparison

| Aspect | Learned Positional Embeddings | RoPE |
| --- | --- | --- |
| Encodes | Absolute position | Relative position (via rotation) |
| Extra parameters | Yes (embedding table) | No |
| Length generalization | Poor beyond training length | Better |
| Applied to | Input embeddings | Q and K vectors, at attention time |
| Used in | BERT, GPT-2 | LLaMA, Mistral, GPT-NeoX, most modern LLMs |

### Q32. What is normalization?

> 🔁 _See also [06-deep-learning.md → Q49](06-deep-learning.md#q49-normalization-in-machine-learning) (separate Notion notes on the same question)._

##### Normalization (in Neural Networks / LLMs)

- A technique that rescales intermediate activations to have stable statistics (e.g., consistent mean and variance) as data flows through a deep network
- Helps keep training stable, gradients well-behaved, and convergence faster

##### Why It's Needed

- As data passes through many layers, the scale and distribution of activations can shift or grow/shrink unpredictably (sometimes called "internal covariate shift")
- Without normalization, this can lead to exploding/vanishing gradients, slow training, or instability, especially in very deep networks like Transformers with dozens of stacked layers

##### Layer Normalization (LayerNorm) — used in Transformers

- Normalizes across the **feature dimension** for each individual token (not across the batch, unlike BatchNorm)
- For each token's vector: subtract the mean and divide by the standard deviation across its features, then apply a learned scale (`γ`) and shift (`β`)

`LayerNorm(x) = γ × (x - μ) / σ + β`

- Works independently per token, which fits well with variable-length sequences and autoregressive generation (no dependency on other examples in a batch)

##### Where It's Applied in Transformers

**Post-LN (original Transformer):**

`x = LayerNorm(x + Sublayer(x))` — normalization applied _after_ the residual addition

**Pre-LN (most modern LLMs — GPT-2 onward, LLaMA, etc.):**

`x = x + Sublayer(LayerNorm(x))` — normalization applied _before_ the sublayer, inside the residual branch

- Generally trains more stably, especially at scale, because it keeps the residual stream's growth more controlled across many layers

##### RMSNorm (increasingly common — LLaMA, Mistral, etc.)

- A simplified variant of LayerNorm
- Skips the mean-centering step, only rescales by the root-mean-square of the activations: `RMSNorm(x) = γ × x / RMS(x)`
- Fewer computations, slightly faster, and works about as well in practice — one reason it's replaced standard LayerNorm in many recent models

##### Why It Matters

- **Training stability**: keeps activation scales consistent across layers, avoiding blow-ups or collapses
- **Faster convergence**: smoother loss landscape, allows higher learning rates
- **Enables very deep networks**: works together with skip connections to make training stacks of dozens/hundreds of layers feasible

### Q33. Explain RMSNorm (Root Mean Square Layer Normalization)

> 🔁 _See also [01-llm-fundamentals.md → Q41](../ai-engineering/01-llm-fundamentals.md#q41-explain-rmsnorm-root-mean-square-layer-normalization) (separate Notion notes on the same question)._

##### RMSNorm (Root Mean Square Layer Normalization)

- A simplified variant of LayerNorm that normalizes using only the **root mean square** of activations, skipping the mean-centering step

##### The Formula

`RMSNorm(x) = γ × x / RMS(x)`

where:

`RMS(x) = √( (1/n) × Σ xᵢ² )`

- `x` is the input vector (a token's activations)
- `n` is the number of features
- `γ` is a learned per-feature scale parameter (no learned shift/bias term, unlike LayerNorm's `β`)

##### How It Differs from LayerNorm

| Step | LayerNorm | RMSNorm |
| --- | --- | --- |
| Subtract mean (re-centering) | Yes | **No** |
| Divide by standard deviation | Yes | Divide by RMS instead |
| Learned scale (γ) | Yes | Yes |
| Learned shift (β) | Yes | **No** |

- LayerNorm assumes both the _mean_ and _variance_ of activations matter and need correcting
- RMSNorm assumes only the _scale_ (magnitude) of activations matters — it re-scales the vector so its RMS is 1, without worrying about centering it around zero

##### Why Skip Mean-Centering?

- Empirically, the mean-centering (re-centering) step in LayerNorm contributes relatively little to training stability compared to the re-scaling step
- Most of LayerNorm's benefit comes from controlling the _magnitude_ of activations, not their central tendency
- RMSNorm keeps the part that matters (rescaling) and drops the part that adds computation without much benefit (re-centering)

##### Why It's Preferred in Modern LLMs

**1. Computational efficiency**

- Skipping mean calculation and subtraction reduces the number of operations
- Faster forward and backward passes, which adds up significantly at LLM scale (billions of parameters, trillions of tokens)

**2. Comparable quality**

- Despite being simpler, RMSNorm performs about as well as LayerNorm in practice on large-scale language modeling tasks

**3. Simplicity**

- Fewer learned parameters (no `β` shift term) and a simpler computation graph

**4. Adopted in leading architectures**

- Used in LLaMA, Mistral, PaLM, and other modern open-weight LLMs, largely replacing standard LayerNorm

### Q34. What is dropout, and how is it applied in LLMs?

##### Dropout

- A regularization technique that randomly "drops" (zeros out) a fraction of neurons/activations during training, forcing the network not to rely too heavily on any single neuron or feature
- Helps prevent overfitting by making the network learn more robust, redundant representations

##### How It Works

- During each training step, every neuron in a dropout-applied layer is independently set to zero with some probability `p` (the dropout rate, e.g., 0.1 or 0.5)
- The remaining active neurons are typically scaled up by `1/(1-p)` so the expected total activation stays roughly the same (this is called "inverted dropout")
- At **inference time**, dropout is turned off entirely — all neurons are active, and no scaling is needed since it was already handled during training

**Simple analogy:**

- Like randomly disabling a fraction of a team's members each day — the team learns to function well regardless of which specific members happen to be missing, rather than depending on any one person

##### Why It Prevents Overfitting

- Without dropout, neurons can co-adapt — learning to rely heavily on very specific combinations of other neurons, which can memorize training data patterns that don't generalize
- Randomly dropping neurons forces each one to be useful somewhat independently, since it can't count on any particular partner neuron being present
- Effectively trains an ensemble of many different "thinned" subnetworks (sharing weights) that get averaged together at inference

##### Where It's Applied in Transformers/LLMs

- **Attention dropout**: applied to attention weights (after softmax), randomly zeroing some attention scores so the model doesn't over-rely on specific token-to-token connections
- **Feed-forward dropout**: applied after the FFN's activation function, before the second linear projection
- **Embedding dropout**: sometimes applied to token or positional embeddings
- **Residual/output dropout**: applied to sublayer outputs before adding them back into the residual stream

##### Important Nuance for Modern LLMs

- **Dropout is far less commonly used in large-scale pretraining of modern LLMs** (like GPT-3/4, LLaMA, etc.) compared to smaller models or earlier architectures like the original Transformer or BERT
- **Why**: with massive datasets (trillions of tokens) and models trained for a limited number of epochs (often less than one pass over the data), overfitting is much less of a concern — the model rarely even sees the same data twice, so there's less need to regularize against memorization
- Dropout is more commonly retained in **fine-tuning** stages, where a pretrained model is adapted on a smaller dataset and overfitting risk is higher again

##### Quick Comparison

| Setting | Dropout typically used? |
| --- | --- |
| Small models / limited data | Yes, often significant (0.1–0.5) |
| Large-scale LLM pretraining | Often minimal or none |
| Fine-tuning pretrained LLMs | Often reintroduced (e.g., 0.1) |

### Q35. Why does Attention use Softmax?

##### Why Attention Uses Softmax

Softmax converts raw attention scores (from `QKᵀ`) into a valid probability distribution that determines how much weight each value vector contributes to the output.

##### What Softmax Does

`softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)`

- Takes a vector of raw scores (which can be any real numbers — positive, negative, large, small) and converts them into values that:
  - Are all **positive** (between 0 and 1)
  - **Sum to 1** across the row
- This gives a proper probability distribution over "how much attention to pay to each token"

##### Why This Matters for Attention

**1. Produces a weighted average of values**

- The attention output is computed as a weighted sum: `Σ (attention_weight_i × value_i)`
- For this to make sense as an interpretable "blend" of information, the weights need to be non-negative and sum to 1 — otherwise the output could have unbounded scale or weights that partially "cancel out" information in unintuitive ways
- Softmax guarantees this property automatically

**2. Amplifies differences between scores (relative emphasis)**

- The exponential in softmax means larger raw scores get disproportionately more weight than smaller ones — small differences in similarity get emphasized
- This lets the model sharply focus on the most relevant tokens rather than spreading attention too evenly, while still allowing some weight on secondary relevant tokens
- Compare this to just normalizing scores by dividing by their sum (without exponentiation) — that wouldn't handle negative scores properly and wouldn't provide the same emphasis on the highest-scoring items

**3. Differentiability**

- Unlike a hard "argmax" (which would pick only the single highest-scoring token deterministically), softmax is smooth and fully differentiable
- This is essential for backpropagation — gradients can flow through softmax during training, allowing the model to learn which tokens _should_ receive more attention based on the loss signal
- A hard argmax selection would make the whole system non-differentiable and untrainable via gradient descent

**4. Handles arbitrary raw score ranges**

- `QKᵀ` dot products can range anywhere from large negative to large positive numbers depending on the vectors involved
- Softmax maps this unbounded range into a clean, bounded, and normalized distribution, regardless of the raw score magnitude

##### Connection to the Scaling Factor

- Attention is typically computed as `softmax(QKᵀ / √d_k)`, not just `softmax(QKᵀ)`
- As the dimension `d_k` grows, dot products tend to grow large in magnitude, which pushes softmax into regions where gradients become extremely small (saturation) — nearly all the probability mass concentrates on one token, and the exponential's derivative flattens out elsewhere
- Dividing by `√d_k` keeps the scores in a reasonable range before softmax, preventing this saturation and keeping gradients healthy during training

### Q36. What does a vector database (Vector DB) store for LLM usage?

##### What a Vector Database Stores for LLM Usage

- Stores **embeddings** — high-dimensional numerical vectors that represent the semantic meaning of pieces of content (text chunks, documents, images, etc.) — along with metadata linking each vector back to its original source content

##### Core Components Stored

**1. The embedding vector itself**

- A fixed-length array of floating-point numbers (e.g., 768, 1024, or 1536 dimensions, depending on the embedding model used)
- Generated by passing content through an embedding model, which maps semantically similar content to nearby points in vector space

**2. The original content (or a reference to it)**

- The raw text chunk, document snippet, or a pointer/ID back to where the full content lives (e.g., a database row, file path, or URL)
- Needed so that once a relevant vector is found, the actual human-readable content can be retrieved and used

**3. Metadata**

- Additional structured information attached to each vector, such as: source document name, chunk index, timestamps, author, category/tags, permissions, or any custom fields
- Enables **filtered search** — e.g., "find semantically similar chunks, but only from documents tagged 'legal' and created after 2023"

**4. An index structure for fast similarity search**

- Since comparing a query vector against millions of stored vectors one-by-one (brute force) is slow, vector DBs build specialized indexes — like **HNSW** (Hierarchical Navigable Small World graphs), **IVF** (Inverted File Index), or **LSH** (Locality-Sensitive Hashing) — that enable fast **approximate nearest neighbor (ANN)** search
- This trades a small amount of accuracy for massive speed gains at scale

##### How This Is Used in LLM Applications (mainly RAG)

1. Documents are split into chunks and each chunk is converted into an embedding vector, then stored in the vector DB along with its text and metadata
2. When a user asks a question, the query itself is converted into an embedding using the same embedding model
3. The vector DB performs a similarity search (commonly using **cosine similarity** or **dot product/Euclidean distance**) to find the stored vectors closest to the query vector
4. The corresponding original text chunks are retrieved and injected into the LLM's prompt as context (this is the core of **Retrieval-Augmented Generation**, or RAG)
5. The LLM then generates a response grounded in that retrieved context, rather than relying solely on its parametric/trained knowledge

##### Why Not Just Use a Regular Database?

- Traditional databases (SQL, keyword search) match on exact terms or structured queries — they can't capture _semantic_ similarity (e.g., "car" and "automobile" would be treated as unrelated by keyword search)
- Vector databases enable **semantic search**: finding content that is conceptually similar in meaning, even if it uses completely different words

##### Common Vector Databases

- Pinecone, Weaviate, Milvus, Qdrant, Chroma, FAISS (library, not full DB), pgvector (Postgres extension)

### Q37. How do you improve inference speed in production LLM deployments?

> 🔁 _See also [11-ai-infrastructure-and-scalability.md → Q1](../ai-engineering/11-ai-infrastructure-and-scalability.md#q1-how-do-you-improve-inference-speed-in-production-llm-deployments) (separate Notion notes on the same question)._

##### Improving Inference Speed in Production LLM Deployments

Inference speed optimization spans several layers: model-level changes, serving infrastructure, hardware utilization, and request-handling strategies.

##### 1. Model Architecture & Weight Optimizations

**Quantization**

- Reduce numerical precision of weights/activations (FP32 → FP16/BF16 → INT8 → INT4)
- Cuts memory footprint and memory-bandwidth bottlenecks, often with minimal quality loss
- Techniques: GPTQ, AWQ, bitsandbytes, GGUF-based quantization

**Knowledge Distillation**

- Train a smaller "student" model to mimic a larger "teacher" model's outputs
- Smaller model = faster inference, at some cost to capability

**Pruning**

- Remove less-important weights/neurons/heads (structured or unstructured)
- Reduces compute, though gains depend on hardware support for sparse operations

**Architecture choices for efficient attention**

- **GQA/MQA** instead of full MHA — shrinks KV cache size, reduces memory-bandwidth bottleneck
- **Sliding window / sparse attention** — avoids full O(n²) attention cost for long contexts

##### 2. KV Cache Optimization

- Cache computed Key/Value vectors from previous tokens so they aren't recomputed at every generation step (standard practice)
- **PagedAttention** (used in vLLM): manages KV cache memory like OS virtual memory pages, reducing fragmentation and allowing much higher batch sizes
- **KV cache quantization**: store cached K/V in lower precision to save memory, allowing larger batches or longer contexts

##### 3. Batching Strategies

**Continuous (dynamic) batching**

- Instead of waiting for a fixed batch of requests to all finish before starting new ones, continuously inserts new requests into the batch as older ones complete
- Dramatically improves GPU utilization compared to static batching, since generation lengths vary a lot per request

**Request scheduling**

- Prioritize based on latency requirements, group similar-length requests, etc.

##### 4. Speculative Decoding

- Use a small, fast "draft" model to generate several candidate tokens ahead of time, then have the large target model verify them all in a single forward pass
- If the draft matches what the large model would have generated, multiple tokens are accepted at once — turning several sequential steps into effectively one
- Falls back to normal generation when the draft diverges, so output quality is unaffected

##### 5. Optimized Kernels & Compilation

- **FlashAttention**: a fused, memory-efficient attention implementation that avoids materializing the full attention matrix, reducing memory reads/writes significantly
- **Kernel fusion**: combining multiple operations into a single GPU kernel to reduce overhead from launching many small kernels
- **Compilation frameworks**: TensorRT-LLM, torch.compile, ONNX Runtime — optimize the computation graph for the target hardware

##### 6. Serving Frameworks

Popular production-grade inference servers that bundle many of the above:

- **vLLM** — PagedAttention, continuous batching
- **TensorRT-LLM** (NVIDIA) — highly optimized kernels, quantization support
- **Text Generation Inference (TGI)** (Hugging Face)
- **SGLang** — optimized for complex prompting patterns and structured generation

##### 7. Hardware & Parallelism

- **Tensor parallelism**: split individual layers/matrices across multiple GPUs
- **Pipeline parallelism**: split different layers across different GPUs
- **Using appropriate hardware**: newer GPUs (H100, etc.) with higher memory bandwidth directly relieve the memory-bound nature of autoregressive decoding

##### 8. Reducing Unnecessary Work

- **Prompt caching**: cache KV states for repeated/shared prompt prefixes (e.g., system prompts) across requests, avoiding recomputation
- **Early exit / output length limits**: stop generation as soon as a stopping condition is met, avoid over-generating
- **Response caching**: for identical or near-identical repeated queries, skip inference entirely and serve from a cache

##### Key Insight: Memory-Bandwidth Bound, Not Compute Bound

- Autoregressive decoding (one token at a time) is typically **memory-bandwidth bound**: the GPU spends more time moving weights and KV cache data than doing actual math
- This is why techniques like **quantization**, **GQA/MQA**, and **PagedAttention** — all of which reduce memory movement — tend to give outsized speed gains compared to purely reducing FLOPs
- Speculative decoding also exploits this: since a forward pass is memory-bound regardless of how many tokens it verifies, verifying multiple draft tokens per pass is nearly as cheap as verifying one, giving a large speedup for "free"

### Q38. What is the context window in an LLM?

##### Context Window in LLMs

- The maximum number of tokens an LLM can process at once as input — encompassing the prompt, any system instructions, conversation history, and the output it generates, all combined
- Acts as the model's "working memory" — anything outside this window is invisible to the model during that particular forward pass

##### Why It Exists

- Transformers process the entire input sequence at once (unlike RNNs, which process token-by-token with a persistent hidden state)
- Self-attention computes relationships between every pair of tokens, which costs **O(n²)** in compute and memory relative to sequence length `n`
- This quadratic scaling makes arbitrarily long context computationally expensive, both during training and inference — so models are trained and deployed with a fixed maximum length

##### What Counts Toward the Context Window

- The system prompt / instructions
- All prior conversation turns (in a multi-turn chat)
- Any documents, retrieved context (e.g., from RAG), or attached files included in the prompt
- The model's own generated output (tokens produced also consume context budget as generation continues)

If the total exceeds the limit, older content typically has to be dropped, truncated, or summarized — the model simply can't "see" content that doesn't fit.

##### Why Context Window Size Matters

**Larger windows enable:**

- Longer conversations without losing earlier context
- Processing entire documents, codebases, or books in a single pass
- More retrieved context in RAG pipelines (more chunks can be included)
- Complex multi-step reasoning that references far-apart information

**Trade-offs of larger windows:**

- Higher compute and memory cost (quadratic attention cost, larger KV cache)
- Slower inference, since more tokens must be processed and cached
- **"Lost in the middle" effect**: research has shown models don't always use long context uniformly well — information in the middle of a very long context is sometimes attended to less reliably than information at the beginning or end

##### How Context Windows Have Grown

- Early Transformer models: \~512–1024 tokens
- GPT-3: 2048–4096 tokens
- Modern LLMs: context windows have expanded dramatically — many now support well beyond 100K tokens, with some models supporting over a million, achieved through techniques like:
  - **RoPE scaling / extrapolation** techniques to extend beyond training length
  - **Sparse/sliding-window attention** to avoid full quadratic cost
  - **Efficient KV cache management** to handle the memory burden of long sequences

##### Context Window vs. Model's "Knowledge"

- The context window is **not** the same as the model's trained/parametric knowledge
- Context window = what's available to the model _right now, in this specific input_
- Parametric knowledge = information baked into the model's weights during training, available regardless of what's in the current prompt
- This distinction is exactly why RAG exists — to bring relevant external information _into_ the context window at inference time, rather than relying solely on what the model memorized during training

### Q39. Why is the context window limited in LLMs?

> 🔁 _See also [01-llm-fundamentals.md → Q24](../ai-engineering/01-llm-fundamentals.md#q24-why-is-the-context-window-limited-in-llms) (separate Notion notes on the same question)._

##### Why the Context Window Is Limited

##### 1. Quadratic Compute Cost of Self-Attention

- Standard self-attention computes a relationship score between **every pair of tokens** in the sequence
- For a sequence of length `n`, this means `n × n` = **O(n²)** score computations
- Doubling the sequence length doesn't double the compute — it **quadruples** it
- This makes very long contexts disproportionately expensive to process, both in training and inference

##### 2. Quadratic Memory Cost

- The full attention matrix (`QKᵀ`) that gets computed has size `n × n`, which must be held in memory (at least in the standard, non-fused implementation)
- At long sequence lengths, this matrix alone can consume enormous amounts of GPU memory, even before accounting for the model's weights

##### 3. Growing KV Cache During Generation

- During autoregressive generation, the model caches Keys and Values for every previous token so it doesn't recompute them at each step
- Cache size grows **linearly** with sequence length, multiplied by the number of layers and attention heads
- For long contexts, this cache alone can require many gigabytes of GPU memory, competing directly with the memory needed for batching multiple requests — a major bottleneck in production serving

##### 4. Positional Encoding Limitations

- Models are trained with a specific maximum sequence length in mind
- **Learned positional embeddings** simply have no entry for positions beyond what was seen in training — the model literally has no representation for those positions
- Even position schemes designed to generalize better (like RoPE) tend to degrade in quality at lengths far beyond training, since the model never learned to handle relationships at those extreme distances during training

##### 5. Training Cost and Data Constraints

- Training on longer sequences multiplies compute cost per training step, due to the quadratic attention cost
- Longer-context training also requires long, coherent training examples — much of internet text data isn't naturally structured as extremely long, coherent documents, making it harder to fill and effectively utilize very long context windows during training

##### 6. Diminishing Effectiveness at Long Range ("Lost in the Middle")

- Even when a model technically supports a long context window, empirical studies show attention doesn't always utilize distant information as reliably as nearby information
- Information in the middle of a very long input is sometimes attended to less effectively than information near the beginning or end
- This means simply extending the context window doesn't guarantee proportionally better use of that extra context — there are practical returns-vs-cost trade-offs, not just hard technical limits

##### How These Limits Are Being Pushed

Despite these constraints, context windows have grown substantially through techniques designed to mitigate each bottleneck:

| Bottleneck | Mitigation Technique |
| --- | --- |
| Quadratic attention cost | Sparse attention, sliding-window attention, linear attention approximations |
| Memory for attention matrix | FlashAttention (fused, memory-efficient computation) |
| Growing KV cache | GQA/MQA, KV cache quantization, PagedAttention |
| Positional generalization | RoPE scaling/extrapolation techniques, ALiBi |
| Training data availability | Synthetic long-document construction, specialized long-context fine-tuning stages |

### Q40. Explain Prompting, Retrieval-Augmented Generation (RAG), and Fine-Tuning.

##### Prompting, RAG, and Fine-Tuning

Three distinct approaches to getting an LLM to produce better, more relevant, or more specialized outputs — differing in how they inject knowledge/behavior and at what cost.

---

##### 1. Prompting

- Providing instructions, examples, or context directly in the input text to guide the model's output, without changing the model's weights at all
- Everything happens at **inference time**, using the model exactly as trained

**Common techniques:**

- **Zero-shot prompting**: just ask directly, no examples given
- **Few-shot prompting**: include a handful of input/output examples in the prompt to demonstrate the desired pattern
- **Chain-of-thought (CoT) prompting**: ask the model to reason step-by-step before giving a final answer
- **System prompts**: persistent instructions defining role, tone, or constraints for the whole conversation
- **Prompt templates**: structured, reusable prompt formats for consistent behavior across many inputs

**Pros:**

- No training required — fastest and cheapest to implement
- Fully flexible, can be changed instantly
- No risk of degrading the base model's other capabilities

**Cons:**

- Limited by context window size
- Can't teach the model genuinely new knowledge it wasn't trained on — only _reminds_ or _guides_ it using what it already knows
- Quality can be inconsistent, sensitive to phrasing

---

##### 2. Retrieval-Augmented Generation (RAG)

- Combines a retrieval system (typically a vector database) with the LLM: relevant external documents/chunks are fetched at query time and inserted into the prompt as context before generation
- Still doesn't change the model's weights — it changes _what's in the context window_ dynamically, per query

**How it works:**

1. Documents are chunked and embedded, stored in a vector database
2. User query is embedded and used to retrieve the most semantically relevant chunks
3. Retrieved chunks are inserted into the prompt
4. The LLM generates a response grounded in that retrieved content

**Pros:**

- Gives the model access to up-to-date or proprietary information it was never trained on, without retraining
- Reduces hallucination by grounding answers in retrieved source material
- Easy to update — just update the underlying documents/database, no retraining needed
- Can cite/attribute sources, since retrieved chunks are traceable

**Cons:**

- Quality depends heavily on retrieval accuracy — irrelevant or poor retrieval leads to poor answers
- Adds system complexity (embedding pipeline, vector DB, retrieval logic)
- Still constrained by context window size for how much retrieved content can be included
- Slightly higher latency due to the retrieval step

---

##### 3. Fine-Tuning

- Continues training a pretrained model's weights on a smaller, task-specific or domain-specific dataset, actually updating the model's parameters
- Changes the model itself, not just what's fed into it at inference time

**Common approaches:**

- **Full fine-tuning**: updates all model parameters (expensive, requires significant compute/memory)
- **Parameter-efficient fine-tuning (PEFT)**: updates only a small subset of parameters or adds small trainable modules — e.g., **LoRA** (Low-Rank Adaptation), adapters — much cheaper while retaining most of the benefit
- **Instruction fine-tuning**: trains the model on instruction/response pairs to improve how it follows directions
- **RLHF (Reinforcement Learning from Human Feedback)**: fine-tunes based on human preference signals rather than direct supervised examples

**Pros:**

- Can teach the model new behaviors, styles, formats, or domain-specific patterns deeply, baked into the weights
- No added inference-time context cost (no retrieval step, no long prompt needed) — behavior is intrinsic
- Can improve consistency and reliability for narrow, well-defined tasks

**Cons:**

- Requires labeled training data, compute resources, and ML expertise
- Risk of **catastrophic forgetting** — degrading the model's general capabilities while specializing it
- Harder and slower to update — any new information requires retraining
- Doesn't easily provide traceable "sources" for its outputs, unlike RAG

---

##### Comparison Table

| Aspect | Prompting | RAG | Fine-Tuning |
| --- | --- | --- | --- |
| Changes model weights? | No | No | Yes |
| Adds new knowledge? | No (uses existing knowledge) | Yes (via retrieved context) | Yes (baked into weights) |
| Update speed | Instant | Fast (update documents) | Slow (requires retraining) |
| Cost | Lowest | Medium | Highest |
| Best for | Quick behavior guidance, formatting, reasoning style | Dynamic/up-to-date knowledge, large external knowledge bases | Deep behavioral/style changes, narrow specialized tasks |
| Traceability/citations | No | Yes (source chunks) | No |
| Context window dependency | High | High | Low (knowledge is internal) |

##### How They're Often Combined

- **Prompting + RAG**: nearly universal — prompting techniques (like clear instructions on how to use retrieved context) are used alongside RAG
- **Fine-tuning + RAG**: fine-tune a model to be better at _using_ retrieved context effectively, while RAG still supplies the actual up-to-date knowledge
- **Fine-tuning + Prompting**: fine-tune for a specific style/behavior, then still use prompting on top for per-request customization

### Q41. What is Mixture of Experts (MoE), and how does it work in models like Mixtral?

> 🔁 _See also [01-llm-fundamentals.md → Q34](../ai-engineering/01-llm-fundamentals.md#q34-what-is-mixture-of-experts-moe-and-how-does-it-work-in-models-like-mixtral) (separate Notion notes on the same question)._

##### Mixture of Experts (MoE)

- An architecture where a single dense Feed-Forward Network (FFN) is replaced with multiple parallel FFNs ("experts"), and a lightweight router network decides which experts process each token
- Key idea: **scale up total model parameters massively while keeping the compute cost per token roughly constant**, since only a small subset of experts is activated for any given token

##### How It Works

**Standard Transformer FFN vs. MoE layer:**

- Normal Transformer: every token passes through the _same single_ FFN
- MoE Transformer: the FFN is replaced by `N` separate expert FFNs (e.g., 8 experts), plus a router

**Step by step:**

1. Each token's representation is passed to a **router** (a small learned linear layer + softmax)
2. The router computes a score for each expert, indicating how well-suited that expert is for this token
3. Only the **top-k** highest-scoring experts are selected (commonly top-1 or top-2) — this is called **sparse activation**
4. The token is processed only by those selected experts (not all of them)
5. The outputs of the selected experts are combined, typically as a weighted sum based on the router's scores
6. This combined output replaces what a single dense FFN would have produced, and continues through the rest of the Transformer layer (residual connection, normalization, etc.)

##### How Mixtral Specifically Works

- Mixtral 8x7B has **8 experts per MoE layer**, with the router selecting the **top 2** experts for each token
- Despite having 8 experts (making total parameter count much larger than a comparable dense model), only 2 experts' worth of compute is used per token
- This means Mixtral has roughly **47B total parameters** but only uses about **13B parameters worth of compute** per forward pass — giving it capacity closer to a much larger dense model while keeping inference cost closer to a smaller one
- Only the FFN layers are replaced with MoE layers — the attention layers remain standard (shared across all tokens, not routed)

##### Why This Works: Capacity vs. Compute Trade-off

- Total _parameter count_ determines how much the model can potentially "know" or represent (more experts = more specialized capacity)
- _Active compute per token_ determines inference speed and training cost
- MoE decouples these two — you can keep adding more experts (more total capacity) without proportionally increasing the compute needed to process each token, since each token only touches a few experts

**Analogy**: Instead of one generalist doctor seeing every patient, you have a large panel of specialists, and a triage system directs each patient only to the 1–2 most relevant specialists. The hospital's total expertise (parameters) is huge, but each patient visit (token) still only involves seeing a couple of doctors, not the whole staff.

##### Benefits

- **Higher capacity for similar per-token compute**: can match or exceed dense models with far fewer active parameters
- **Specialization**: different experts often end up handling different types of tokens/patterns (though not always cleanly interpretable — specialization isn't always along obvious semantic lines)
- **Efficient scaling**: a common way to scale total model size without a proportional increase in serving cost

##### Challenges

**1. Load balancing**

- Without care, the router can collapse to favoring only a few experts, leaving others undertrained ("expert collapse")
- Solved with auxiliary load-balancing losses during training that encourage roughly even usage across experts

**2. Memory requirements**

- Even though only a few experts are _active_ per token, **all experts' weights must be loaded into memory**, since different tokens in a batch may route to different experts
- This means MoE models save on compute but not on memory — Mixtral's 47B total parameters all need to be held in memory even though only \~13B are used per token

**3. Training complexity**

- Router training can be unstable; requires careful tuning of load-balancing losses and routing strategies
- Communication overhead in distributed training, since tokens need to be routed to whichever device holds the relevant expert

**4. Batch efficiency**

- Since different tokens in a batch route to different experts, efficient batched computation is more complex than a simple dense matrix multiply — requires specialized kernels/infrastructure

##### Quick Comparison: Dense vs. MoE

| Aspect | Dense Model | MoE Model |
| --- | --- | --- |
| FFN per token | Same FFN for all tokens | Routed to top-k experts |
| Total parameters | Matches active parameters | Much larger than active parameters |
| Compute per token | Full model | Only top-k experts' worth |
| Memory footprint | Matches parameter count | High (all experts must be loaded) |
| Examples | LLaMA (non-MoE versions), GPT-3 | Mixtral, some GPT-4 variants (reported) |

### Q42. What is the difference between dense and sparse models?

> 🔁 _See also [01-llm-fundamentals.md → Q35](../ai-engineering/01-llm-fundamentals.md#q35-what-is-the-difference-between-dense-and-sparse-models) (separate Notion notes on the same question)._

##### Dense vs. Sparse Models

##### Core Definition

**Dense models:**

- Every parameter is used (activated) for every input/token during a forward pass
- The full network — every neuron, every weight — participates in processing each piece of input

**Sparse models:**

- Only a subset of parameters is activated for any given input/token
- Large portions of the network stay "inactive" for a particular input, even though the full model contains far more parameters overall

##### How This Applies to LLMs

**Dense LLM (e.g., LLaMA, GPT-3 style):**

- Every token passes through every layer, every attention head, and the full FFN
- Total parameters = active parameters (per token)
- If the model has 70B parameters, all 70B are used to process every single token

**Sparse LLM (e.g., Mixtral, MoE-based models):**

- Uses a Mixture of Experts (MoE) architecture — token routed to only a few "expert" FFNs out of many available
- Total parameters ≫ active parameters per token
- Mixtral has \~47B total parameters, but only \~13B are actually used per token (top-2 of 8 experts)

##### Key Differences

| Aspect | Dense | Sparse (MoE) |
| --- | --- | --- |
| Parameters used per token | All | Subset (e.g., top-k experts) |
| Total vs. active parameters | Equal | Total ≫ active |
| Compute per token | Fixed, matches full model size | Lower than total size implies |
| Memory footprint | Matches parameter count | High — all parameters must still be loaded in memory |
| Scaling capacity vs. cost | Capacity and compute scale together | Capacity can scale independently of per-token compute |
| Routing mechanism needed | No | Yes (a router/gating network) |
| Training/inference complexity | Simpler | More complex (load balancing, routing overhead) |

##### Why the Distinction Matters

**Dense models:**

- Simpler to train, serve, and reason about — no routing logic, straightforward compute scaling
- Increasing capacity (more parameters) directly increases compute cost proportionally — there's no way to "add more knowledge" without also making every forward pass more expensive

**Sparse models:**

- Let you scale total capacity (more specialized "knowledge" stored in more experts) without proportionally scaling the compute needed per token
- The trade-off: you still need enough memory/hardware to hold _all_ the experts, even though you're only computing with a few of them at a time — so sparsity saves compute, not memory

##### Broader Meaning of "Sparse" (Beyond MoE)

- MoE is the most common form of sparsity in LLMs, but "sparse" more generally refers to any technique where not all parameters/connections are used at once:
  - **Sparse attention**: only computing attention over a subset of token pairs, rather than full O(n²) attention
  - **Weight pruning**: permanently zeroing out unimportant individual weights (structured or unstructured sparsity), reducing effective parameter count altogether, not just per-input activation
- MoE is sometimes called **"conditional computation"** — sparsity that's dynamic and input-dependent, rather than a fixed pruned structure

### Q43. Transformers work on text, so can they also understand images?

> 🔁 _See also [01-llm-fundamentals.md → Q59](../ai-engineering/01-llm-fundamentals.md#q59-transformers-work-on-text-so-can-they-also-understand-images) (separate Notion notes on the same question)._

##### Can Transformers Understand Images?

Yes — Transformers were originally designed for text, but the core architecture (attention over a sequence of tokens) turns out to generalize well beyond text, as long as the input can be converted into a sequence of vectors.

##### How Images Get Turned Into "Tokens"

**Vision Transformer (ViT) approach:**

- An image is split into fixed-size patches (e.g., 16×16 pixels)
- Each patch is flattened and linearly projected into a vector — essentially treating each patch like a "word" in a sequence
- Positional embeddings are added (since patch order/location matters, just like word order in text)
- The sequence of patch embeddings is fed through standard Transformer encoder layers, using the same self-attention mechanism used for text
- So a 224×224 image split into 16×16 patches becomes a sequence of 196 "image tokens" — attention then lets every patch attend to every other patch, capturing relationships like "this patch is part of the same object as that patch," regardless of spatial distance

##### Multimodal Models (Text + Image Together)

For models like GPT-4V, Claude, or Gemini that understand both text and images in the same conversation:

- Images are encoded into a sequence of embeddings (often via a ViT-style encoder or a similar vision encoder)
- Those image embeddings are projected into the same embedding space as text token embeddings
- Text tokens and image tokens are combined into a single sequence and processed together through the same Transformer decoder
- This lets the model attend across modalities — a text token like "What color is the car?" can directly attend to the relevant image patch tokens

##### Why This Works

- Attention doesn't inherently "know" or care whether a token came from text or pixels — it just computes similarity between vector representations and produces weighted combinations
- As long as different modalities are mapped into a **shared embedding space**, the same attention mechanism can relate information across them
- This is part of why Transformers are described as a fairly general-purpose architecture — the same core mechanism has been applied to text, images, audio (spectrograms as patches), video (patches across time), and even protein structures

##### Key Differences From Text Transformers

| **Aspect** | **Text Transformer** | **Vision Transformer** |
| --- | --- | --- |
| Input unit | Subword tokens | Image patches |
| Sequence order | Inherent (word order) | Spatial (2D position, needs 2D-aware positional encoding) |
| Causal masking | Yes (for generation) | Often no (bidirectional — full image is visible at once, not generated left-to-right) |
| Typical use | Autoregressive generation | Classification, embeddings, or feeding into multimodal decoders |

### Q44. Small Language Models (SLMs)

> 🔁 _See also [01-llm-fundamentals.md → Q60](../ai-engineering/01-llm-fundamentals.md#q60-small-language-models-slms) (separate Notion notes on the same question)._

##### Small Language Models (SLMs)

- Language models with significantly fewer parameters than large-scale LLMs — typically ranging from a few million up to a few billion parameters (vs. tens/hundreds of billions for LLMs like GPT-4 or Claude)
- Designed to run efficiently on limited hardware while still performing well on targeted tasks

##### Key Characteristics

- **Smaller parameter count**: often 1B–13B parameters, sometimes even smaller (hundreds of millions)
- **Lower compute and memory requirements**: can run on consumer GPUs, edge devices, or even mobile phones/CPUs, unlike large LLMs that require multiple high-end GPUs
- **Faster inference**: fewer parameters mean less computation per token, lower latency
- **Narrower or more specialized capability**: generally can't match large LLMs on broad, open-ended reasoning, but can perform very well on focused tasks

##### How SLMs Achieve Good Performance Despite Size

**1. High-quality, curated training data**

- Many modern SLMs (e.g., Microsoft's Phi series) prioritize carefully filtered, high-quality, textbook-like training data over sheer data volume — showing that data quality can partly compensate for smaller scale

**2. Knowledge distillation**

- Training a small "student" model to mimic the outputs/behavior of a larger "teacher" LLM, transferring much of the teacher's capability into a much smaller model

**3. Efficient architectures**

- Incorporating the same efficiency techniques used in large models — GQA, RoPE, RMSNorm, quantization — scaled down

**4. Fine-tuning for narrow tasks**

- Rather than trying to be a generalist, SLMs are often fine-tuned for specific domains (customer support, code completion, summarization) where a smaller model can match larger-model performance on that narrow task

##### Why Use SLMs Instead of LLMs

| Reason | Benefit |
| --- | --- |
| Cost | Much cheaper to run at scale (less compute per request) |
| Latency | Faster response times, important for real-time applications |
| On-device/edge deployment | Can run offline on phones, IoT devices, laptops — no internet/API dependency |
| Privacy | Data never leaves the device, useful for sensitive applications |
| Lower infrastructure needs | Doesn't require GPU clusters, easier to self-host |

##### Trade-offs vs. Large LLMs

- **Reduced general reasoning ability**: struggle more with complex, multi-step, or open-ended reasoning tasks
- **Smaller knowledge base**: fewer parameters generally means less "memorized" world knowledge baked into weights
- **Narrower context handling**: often (though not always) support shorter context windows
- **Better suited to narrow, well-defined tasks** than broad, general-purpose assistance

##### Examples of SLMs

- Microsoft Phi-3 / Phi-4 (3.8B, 14B variants)
- Google Gemma (2B, 9B)
- Meta LLaMA 3.2 (1B, 3B variants)
- Mistral 7B (borderline — sometimes classified as "small" relative to larger models)

##### Common Use Cases

- On-device assistants (mobile apps, smart devices)
- Real-time applications where latency is critical
- Cost-sensitive, high-volume applications (customer service bots, autocomplete)
- Privacy-sensitive domains (healthcare, finance) where data can't leave a local environment
- Edge computing / IoT scenarios with no reliable internet connection

### Q45. Large Reasoning Models (LRMs)

> 🔁 _See also [01-llm-fundamentals.md → Q61](../ai-engineering/01-llm-fundamentals.md#q61-large-reasoning-models-lrms) (separate Notion notes on the same question)._

##### Large Reasoning Models (LRMs)

- A class of LLMs specifically trained/optimized to perform extended, deliberate multi-step reasoning before producing a final answer, rather than generating a response in a single fast pass
- Examples: OpenAI's o1/o3 series, DeepSeek-R1, Gemini's "Thinking" models, Claude's extended thinking mode

##### How They Differ From Standard LLMs

| Aspect | Standard LLM | LRM |
| --- | --- | --- |
| Generation style | Direct, fast response | Extended internal reasoning before final answer |
| Compute at inference | Fixed, low per response | Variable, can be much higher ("thinking longer") |
| Training focus | Next-token prediction + instruction tuning | Reinforcement learning on reasoning trajectories |
| Best suited for | General conversation, quick tasks | Math, coding, logic, multi-step planning |

##### Core Idea: Test-Time Compute

- Traditional scaling focused on making models bigger or training on more data (**train-time compute**)
- LRMs introduce a new scaling axis: **test-time (inference-time) compute** — letting the model spend more computation _at the moment of answering_ by generating longer chains of intermediate reasoning
- The insight: for many hard problems, a model doesn't need more _parameters_ — it needs more _steps of deliberation_, similar to how a person might scratch out working on paper before answering a hard question

##### How They're Trained

- Typically start from a strong base LLM, then apply **reinforcement learning** where the model is rewarded for producing reasoning chains that lead to correct final answers (especially on verifiable domains like math and code, where correctness can be automatically checked)
- The model learns to generate longer, more exploratory reasoning — including behaviors like backtracking, self-correction, and trying alternative approaches — because these behaviors are reinforced when they lead to better final answers
- This differs from simple prompted chain-of-thought, where a standard LLM is just _asked_ to "think step by step" without having been specifically trained to reason well through RL

##### Key Behaviors Observed

- **Self-correction**: recognizing a flawed reasoning path mid-way and backtracking to try a different approach
- **Decomposition**: breaking a complex problem into smaller sub-problems before solving
- **Verification**: checking intermediate results before committing to a final answer
- **Variable-length reasoning**: spending more reasoning tokens on harder problems, less on easy ones (the model implicitly learns how much "thinking" a problem needs)

##### Benefits

- **Significantly better performance on hard, structured problems**: competitive math (e.g., olympiad-level), coding, logic puzzles, scientific reasoning
- **Reduced hallucination on multi-step problems**: intermediate verification catches errors that a single-pass answer might miss
- **New scaling lever**: performance can be improved by allowing more inference-time compute, without necessarily needing a larger or retrained model

##### Trade-offs

- **Higher latency**: generating long reasoning chains takes much more time than a direct answer
- **Higher inference cost**: more tokens generated per query means more compute per response, even though the final answer might be short
- **Overkill for simple queries**: extended reasoning provides little benefit (and wastes time/cost) for straightforward factual or conversational requests
- **Reasoning traces aren't always fully transparent or faithful**: the visible "thinking" text doesn't always perfectly reflect the actual computation happening inside the model

##### Why This Matters for LLM Development

- Represents a shift from "scale the model" to "scale the _thinking_" — a complementary axis to traditional pretraining scale
- Suggests future LLM systems may dynamically choose _how much_ to reason based on task difficulty, rather than using a fixed amount of compute for every query, balancing cost/latency against accuracy

### Q46. Reinforcement Learning from Human Feedback (RLHF)

##### Reinforcement Learning from Human Feedback (RLHF)

- A training technique that fine-tunes a pretrained LLM to better align with human preferences — helpfulness, harmlessness, honesty, style — using human judgments as the training signal, rather than just predicting the next token from raw text

##### Why It's Needed

- Pretraining on raw internet text teaches a model to predict plausible next tokens, but "plausible" isn't the same as "helpful," "safe," or "aligned with what a person actually wants"
- A purely pretrained model can be inconsistent, produce harmful/biased content, ramble, or fail to follow instructions well
- RLHF steers the model toward outputs that humans actually prefer, using explicit human feedback rather than hoping good behavior emerges from raw text prediction alone

##### The Three-Stage RLHF Pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**

- Start with a pretrained base model
- Fine-tune it on a dataset of high-quality human-written examples (prompt + ideal response pairs)
- Gives the model a reasonable starting point for following instructions in a desired format/style

**Stage 2: Reward Model (RM) Training**

- Generate multiple responses to the same prompt using the SFT model
- Human labelers rank or compare these responses (e.g., "Response A is better than Response B")
- Train a separate **reward model** to predict human preference scores — given a prompt and a response, output a scalar score estimating how much a human would like it
- This effectively distills human judgment into a learnable, automatable scoring function

**Stage 3: Reinforcement Learning Optimization**

- Use the reward model as the reward signal in a reinforcement learning loop (commonly **PPO** — Proximal Policy Optimization)
- The LLM (now acting as a "policy") generates responses, the reward model scores them, and the LLM's weights are updated to increase the likelihood of high-reward outputs
- A **KL-divergence penalty** against the original SFT model is typically included, preventing the policy from drifting too far and "reward hacking" (exploiting quirks in the reward model rather than genuinely improving)

##### What RLHF Optimizes For

- **Helpfulness**: following instructions accurately, being relevant and useful
- **Harmlessness**: refusing dangerous, unethical, or policy-violating requests
- **Honesty**: reducing confidently stated falsehoods (though it doesn't fully solve hallucination)
- **Style/tone preferences**: conversational quality, formatting, appropriate length

##### Limitations and Challenges

- **Reward hacking**: the policy can find ways to score well on the reward model without genuinely improving (e.g., producing longer responses because length correlates with perceived quality in training data)
- **Reward model imperfection**: the RM is only an approximation of human preference, trained on a limited sample of comparisons — its blind spots become the policy's blind spots
- **Expensive human labeling**: requires large volumes of human comparison data, which is costly and slow to collect
- **Potential for over-caution or homogenized outputs**: overly aggressive alignment can make models overly hedgy, repetitive in refusals, or reluctant on borderline-legitimate requests
- **Doesn't fully solve alignment**: RLHF nudges behavior toward preferred patterns but doesn't guarantee robustness against adversarial prompts or novel situations

##### Alternatives and Extensions

- **DPO (Direct Preference Optimization)**: skips training a separate reward model and RL loop entirely — directly optimizes the policy on preference data using a simpler, more stable loss function, achieving similar alignment goals with less complexity
- **RLAIF (RL from AI Feedback)**: uses another AI model to generate preference judgments instead of (or alongside) human labelers, reducing the cost/bottleneck of human annotation
- **Constitutional AI**: a technique where a model critiques and revises its own outputs against a set of guiding principles, reducing reliance on large volumes of human-labeled comparisons

### Q47. Proximal Policy Optimization (PPO)

> 🔁 _See also [01-llm-fundamentals.md → Q64](../ai-engineering/01-llm-fundamentals.md#q64-proximal-policy-optimization-ppo) (separate Notion notes on the same question)._

##### Proximal Policy Optimization (PPO)

- **PPO** is a **policy-gradient reinforcement learning algorithm** used to optimize an agent’s policy while preventing excessively large policy updates.
- It improves training **stability and sample efficiency** by constraining how much the new policy can differ from the old policy.
- PPO uses a **clipped objective function** that limits the policy probability ratio to a small range, typically **\[1 − ε, 1 + ε\]**.
- It uses **advantage estimates** to determine whether an action performed better or worse than expected.
- PPO is **on-policy**, meaning it learns primarily from data collected using the current policy.
- It is widely used in **continuous-control tasks, robotics, games, and RLHF for language models**.
- Compared with **TRPO**, PPO is simpler to implement while providing a similar mechanism for preventing unstable policy updates.

### Q48. Direct Preference Optimization (DPO)

> 🔁 _See also [01-llm-fundamentals.md → Q65](../ai-engineering/01-llm-fundamentals.md#q65-direct-preference-optimization-dpo) (separate Notion notes on the same question)._

##### Direct Preference Optimization (DPO)

- **DPO** is a **preference-based fine-tuning method** that directly trains an LLM to prefer desired responses over undesired responses.
- It avoids the need to explicitly train a separate **reward model** and avoids the complex RL optimization used in methods such as PPO.
- DPO uses preference pairs: **chosen response** vs. **rejected response**.
- The model is trained to increase the probability of the **chosen response** while decreasing the probability of the **rejected response**, relative to a reference model.
- DPO is commonly used for **LLM alignment, instruction following, helpfulness, safety, and conversational quality**.
- Compared with **RLHF + PPO**, DPO is generally **simpler, more stable, and easier to implement**, but its performance depends heavily on the quality and diversity of preference data.

### Q49. Group Relative Policy Optimization (GRPO)

> 🔁 _See also [01-llm-fundamentals.md → Q66](../ai-engineering/01-llm-fundamentals.md#q66-group-relative-policy-optimization-grpo) (separate Notion notes on the same question)._

##### Group Relative Policy Optimization (GRPO)

- **GRPO** is a **policy optimization algorithm** designed to improve LLM reasoning using **relative rewards** among multiple responses generated for the same prompt.
- Instead of training a separate **value/critic model**, GRPO compares a **group of sampled responses** and uses their relative rewards to estimate advantages.
- For each prompt, the model generates multiple responses and evaluates them using a **reward function**.
- Responses that receive higher rewards are reinforced, while lower-reward responses are discouraged.
- This reduces memory and computational requirements compared with methods that require a separate critic/value model, such as PPO.
- GRPO became particularly notable for **reasoning-focused LLM training**, where rewards can come from verifiable signals such as **mathematical correctness or code execution**.
- Compared with **PPO**, GRPO is simpler for reasoning tasks because it eliminates the separate value model and derives advantages from **group-relative rewards**.

### Q50. Recursive Language Models (RLMs)

> 🔁 _See also [01-llm-fundamentals.md → Q67](../ai-engineering/01-llm-fundamentals.md#q67-recursive-language-models-rlms) (separate Notion notes on the same question)._

##### Recursive Language Models (RLMs)

- **RLMs** are language-model systems that handle **very large or complex inputs by recursively decomposing and processing them** rather than requiring the entire input to fit directly into one model context.
- The model can **break a large problem into smaller subproblems**, solve them separately, and combine the results.
- Intermediate results can be fed back into the model through **recursive calls**, enabling deeper reasoning over large inputs.
- RLMs can help with **long-document analysis, complex reasoning, codebases, and tasks that exceed the model’s practical context window**.
- They can reduce the need to process massive contexts in a single forward pass, potentially improving **scalability and reasoning reliability**.
- Compared with standard LLM prompting, RLMs introduce an explicit **recursive computation strategy** around the language model rather than relying solely on a single context window.

### Q51. Continual Learning in LLMs

> 🔁 _See also [01-llm-fundamentals.md → Q68](../ai-engineering/01-llm-fundamentals.md#q68-continual-learning-in-llms) (separate Notion notes on the same question)._

##### Continual Learning in LLMs

- **Continual learning** is the process of continuously updating an LLM with **new data, knowledge, or tasks** without retraining the model from scratch.
- It allows models to adapt to **changing information, evolving user needs, new domains, and new languages**.
- Common approaches include **continued pretraining, incremental fine-tuning, PEFT/LoRA adapters, and rehearsal using historical data**.
- A major challenge is **catastrophic forgetting**, where learning new information causes the model to lose previously learned knowledge or capabilities.
- Other challenges include **data distribution shift, knowledge conflicts, training stability, and preventing low-quality or malicious data from being learned**.
- Techniques such as **replay buffers, regularization, mixture-of-adapters, periodic evaluation, and knowledge distillation** can help preserve existing capabilities.
- For frequently changing factual knowledge, **RAG is often preferable to continually modifying the model's parameters**.

### Q52. How does Knowledge Distillation work?

##### Knowledge Distillation

- **Knowledge distillation** is a technique where a smaller **student model learns from a larger, more capable teacher model**.
- The teacher produces **soft probability distributions** over possible outputs, which contain more information than just the correct label.
- The student is trained to **match the teacher's behavior** while also learning from the original training targets.
- A **temperature parameter** is often used to soften the teacher's output distribution, making relationships between classes easier for the student to learn.
- The goal is to create a **smaller, faster, and cheaper model** that retains much of the teacher's performance.
- It is commonly used for **LLMs, model compression, deployment on resource-constrained devices, and transferring capabilities from large models to smaller models**.
- Example: a large 70B LLM acts as the **teacher**, generating high-quality responses, while a smaller 7B LLM learns from those responses and becomes the **student**.

### Q53. What is instruction tuning, and why is it important for chat models?

> 🔁 _See also [05-fine-tuning-and-model-adaptation.md → Q9](../ai-engineering/05-fine-tuning-and-model-adaptation.md#q9-what-is-instruction-tuning-and-why-is-it-important-for-chat-models) (separate Notion notes on the same question)._

##### Instruction Tuning

- **Instruction tuning** is the process of fine-tuning a pretrained LLM on **instruction–response pairs** so it learns to follow human instructions.
- The training data typically contains examples such as **user instruction → desired assistant response**.
- It teaches the model **what to do with a request**, rather than simply predicting the next token from general text.
- It improves **instruction following, task completion, formatting, helpfulness, and conversational behavior**.
- It is especially important for **chat models** because pretrained models are primarily trained for language modeling, not necessarily for acting as interactive assistants.
- Instruction tuning is often followed by **preference alignment** techniques such as **RLHF, DPO, or GRPO** to further improve response quality and alignment.
- Example: a pretrained model may complete `"Explain photosynthesis"` unpredictably, while an instruction-tuned model learns to respond with a clear, structured explanation.

### Q54. Prefill vs Decode

> 🔁 _See also [07-llmops-and-production-ai.md → Q2](../ai-engineering/07-llmops-and-production-ai.md#q2-prefill-vs-decode) (separate Notion notes on the same question)._

##### Prefill vs Decode

- **Prefill** is the phase where the LLM processes the **entire input prompt** in parallel and builds the **KV cache**.
- **Decode** is the phase where the LLM generates the response **one token at a time**, reusing the KV cache from prefill.

|  | Prefill | Decode |
| --- | --- | --- |
| Input | Entire prompt | One new token at a time |
| Computation | Highly parallel | Sequential |
| Main bottleneck | Usually **compute-bound** | Usually **memory-bandwidth / KV-cache-bound** |
| KV cache | Created | Read and extended |
| Purpose | Understand/process context | Generate output |

**Example:**

`Prompt → [Prefill] → KV Cache → [Decode] → Token 1 → Token 2 → Token 3 → ...`

- **Long prompts** mainly increase prefill cost.
- **Long generated responses** mainly increase decode cost.
- Optimizations such as **continuous batching, KV-cache optimization, speculative decoding, and paged attention** primarily improve inference efficiency, especially during decode.

### Q55. How does Sliding Window Attention work?

##### Sliding Window Attention

- **Sliding Window Attention (SWA)** limits each token's attention to a **fixed-size window of nearby tokens** instead of attending to the entire sequence.
- For example, with a window size of 4, each token attends only to the **4 most recent relevant tokens** rather than all previous tokens.
- As the model moves through the sequence, the attention window **slides** along the tokens.
- This reduces attention complexity from **O(n²)** to approximately **O(n × w)**, where `n` is the sequence length and `w` is the window size.
- It significantly reduces **memory and computation**, making it useful for long-context LLMs.
- Information can still propagate across distant tokens through **multiple layers**, even though each attention operation is local.
- A limitation is that tokens **outside the local window cannot directly interact** with each other in the same attention layer.
- Some models combine SWA with **global attention or special global tokens** to preserve long-range information.

**Example:**

```txt
Tokens:   1  2  3  4  5  6  7  8

Window:         [3  4  5  6]
                    ↓
                [4  5  6  7]
                    ↓
                [5  6  7  8]
```

The window continuously moves through the sequence, allowing the model to process long sequences without full quadratic attention.

### Q56. How do Attention Sinks work?

- Softmax forces attention weights to sum to 1, so even when no token is relevant, the model must still allocate mass somewhere — it learns to dump the excess onto a few "junk" tokens rather than distort meaningful ones.
- The first token(s) in a sequence tend to become this dumping ground, since causal attention lets every later token see them — a reliable, always-visible target.
- This is a learned, self-reinforcing behavior: once a head starts treating a token as a sink, it keeps doing so throughout training.
- The sink token itself usually carries little to no real semantic meaning — its job is structural, not content-based.
- Identified/named by the **StreamingLLM** paper: sliding-window KV caches broke badly once sink tokens got evicted, even though those tokens seemed unimportant.
- Fix: always keep the first few tokens in the cache alongside the sliding window — this restores stable long-context generation.
- Implication: high attention weight ≠ semantic importance, which complicates naive attention-based interpretability.
- Some newer architectures add an explicit learned sink token/bias so attention heads have a dedicated "no-op" slot instead of hijacking a real token.
