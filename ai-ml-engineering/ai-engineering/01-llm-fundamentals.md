<!-- notion-id: 3d567418285480f0aa7ad772d8ad68be -->
# LLM Fundamentals

Interview questions and notes on LLM fundamentals: Transformer architecture, attention, tokenization, decoding, training/alignment methods and production troubleshooting.

### Q1. What is an LLM?

- **LLM stands for Large Language Model.** 
- **It is a deep learning model trained on a very large corpus of text to understand and generate human language.** 
- **Most modern LLMs are based on the Transformer architecture and use self-attention to understand the relationship between tokens in a sequence.**
- **An LLM learns to predict the next token based on the previous context.** 
- **Through large-scale pretraining and subsequent fine-tuning or instruction tuning, it can perform tasks such as question answering, summarization, code generation, reasoning, and information extraction.**
- **LLMs are often combined with techniques such as RAG, tool calling, agents, fine-tuning, embeddings, and vector databases to build reliable AI applications.**

### Q2. What is RAG?

- **RAG stands for Retrieval-Augmented Generation.** 
- **It is an architecture that combines information retrieval with an LLM.** 
- **Instead of relying only on the knowledge learned during the model's training, RAG retrieves relevant information from an external knowledge source and provides that context to the LLM to generate an answer.** 
- **Typically, documents are split into chunks, converted into embeddings, and stored in a vector database.** 
- **When a user asks a question, the query is converted into an embedding and used to retrieve the most relevant chunks.**
- **Those chunks are then provided as context to the LLM, which generates the final response.** 
- **RAG is useful for building applications over private, domain-specific, or frequently changing data without necessarily fine-tuning the LLM.** 
- **It can also improve factual grounding and allow responses to be supported by retrieved sources.**

### Q3. What is MCP?

- **MCP stands for Model Context Protocol.** 
- **It is an open protocol that standardizes how AI applications, particularly LLM-based agents, connect to external tools, data sources, and services.**
- **Instead of building a custom integration for every LLM application and every external system, MCP provides a standardized interface through which an AI application can discover and interact with capabilities exposed by an MCP server.**
- **An MCP server can expose things such as tools, resources, and prompts. For example, an MCP server could provide an AI assistant with tools to query a database, access documents from a knowledge base, interact with GitHub, or perform operations in an internal business system.**
- **The key benefit is decoupling the AI application from individual integrations, making tool and data access more standardized and reusable.**
- LLM/Application → MCP Client → MCP Server → External System

### Q4. What is an AI agent?

- **An AI agent is a system that uses an AI model, typically an LLM, to understand a goal, reason about the steps required to achieve that goal, use available tools or external systems, observe the results, and take further actions until the task is completed.**
- **Unlike a traditional LLM application, where we usually define a fixed sequence of steps, an agent can dynamically decide which tool to use and what action to take based on the current state and the result of previous actions.**
- **For example, a customer-support agent might receive a user's request, search the knowledge base, check an order database, call an API to retrieve the order status, and then generate a response based on the retrieved information.**
- **In production, an agent typically consists of an LLM, instructions, tools, state or memory, an orchestration layer, and mechanisms for controlling and evaluating its actions.**

### Q5. What is Fine-tuning?

> 🔁 _See also [01-ai-engineering.md → Q17](../ml-engineering/01-ai-engineering.md#q17-fine-tuning) (separate Notion notes on the same question)._

- **Fine-tuning is the process of taking a pretrained model and further training it on a smaller, task-specific or domain-specific dataset so that the model becomes better adapted to a particular task, behavior, or domain.**
- **Instead of training a model from scratch, we start with the knowledge and representations learned during pretraining and update some or all of the model's parameters using the fine-tuning dataset.**
- **For LLMs, fine-tuning can be used for things like instruction following, classification, structured output, domain-specific language, or adapting the model's behavior and style. Depending on the use case and resources, we can use full fine-tuning or parameter-efficient methods such as LoRA and QLoRA.**

### Q6. What is Quantization?

> 🔁 _See also [01-ai-engineering.md → Q18](../ml-engineering/01-ai-engineering.md#q18-quantization) (separate Notion notes on the same question)._

- **Quantization is the process of representing a model's weights and sometimes activations using lower-precision numerical formats, such as INT8 or INT4, instead of higher-precision formats like FP32 or FP16.**
- **The main goal is to reduce model memory usage, storage requirements, and inference latency, while maintaining as much model accuracy as possible. This is particularly useful for deploying large language models on GPUs or resource-constrained environments.**
- **For example, converting a model from FP16 to INT8 reduces the number of bits used to represent each value from 16 to 8, which can significantly reduce memory requirements. However, quantization can introduce some loss of accuracy, so the appropriate quantization method and precision need to be selected based on the application.**

### Q7. What are foundation models, and how have they changed AI engineering?

- **Foundation models are large, pretrained AI models trained on broad and diverse datasets at scale.** 
- **They learn general-purpose representations and capabilities that can then be adapted to many downstream tasks through prompting, fine-tuning, RAG, tool use, or other techniques. Examples include large language models, vision-language models, and multimodal models.**
- **Foundation models have significantly changed AI engineering because we no longer need to build and train a separate model from scratch for every application. Instead, engineers can build applications on top of pretrained models and focus more on system architecture, data pipelines, retrieval, evaluation, orchestration, security, and deployment.**
- **For example, instead of training an NLP model from scratch for a company's internal knowledge base, we can use a foundation model together with RAG, embeddings, a vector database, and appropriate evaluation to build a domain-specific application much faster.**
- **They have also shifted AI engineering from primarily model development toward building reliable AI systems around models. This has introduced new engineering challenges such as prompt engineering, model selection, context management, hallucination and evaluation, latency, inference cost, observability, security, and responsible AI.**

### Q8. Inside ChatGPT: What Happens After You Hit Enter?

**1. Input → Tokens**

- Your text is broken into **tokens**.
- Tokens can be words, subwords, punctuation, etc.

**2. Tokens → Numbers**

- Each token is mapped to a **token ID**.
- Token IDs are converted into **embeddings** (high-dimensional vectors).

**3. Transformer Processing**

- Tokens pass through **Transformer layers**.
- **Self-attention** helps the model understand relationships and context between tokens.

**4. Next-Token Prediction**

- The model produces a **probability distribution** over possible next tokens.
- A token is selected based on the model's output and decoding strategy.

**5. Autoregressive Generation**

- The selected token is added to the sequence.
- The model predicts the next token again.
- This repeats until a stop condition or length limit is reached.

#### Interview Flow

**Prompt → Tokenization → Token IDs → Embeddings → Transformer/Self-Attention → Probability Distribution → Token Selection → Repeat → Response**

#### Key Interview Point

> **When you hit Enter, the input is tokenized and processed through the Transformer. The model then generates the response autoregressively by predicting one token at a time based on the preceding context.**

### Q9. What is the Transformer architecture, and how does it work?

> 🔁 _See also [09-large-language-model.md → Q2](../ml-engineering/09-large-language-model.md#q2-what-is-the-transformer-architecture-and-how-does-it-work) (separate Notion notes on the same question)._

#### Transformer Architecture

- **Transformer** is a neural network architecture built around **self-attention**, widely used in modern LLMs.
- It processes tokens and learns relationships between them to understand context.

##### How it works

**Tokens → Embeddings + Position → Self-Attention → FFN → Residual + Normalization → Output**

1. **Embeddings** — Convert tokens into vectors.
2. **Positional information** — Adds token order information.
3. **Self-Attention** — Determines which tokens are relevant to each other using **Query, Key, and Value (Q, K, V)**.
4. **Feed-Forward Network** — Further transforms the representations.
5. **Residual Connections + Layer Normalization** — Stabilize training and information flow.
6. **Stacked Layers** — Multiple Transformer blocks progressively learn complex patterns.
7. **Output** — For an LLM, produces probabilities for the **next token**.

##### Interview Answer

> **A Transformer is a neural network architecture based on self-attention. It converts tokens into embeddings, adds positional information, processes them through multiple layers of self-attention and feed-forward networks, and ultimately predicts the next token in an LLM.**

### Q10. What are the key components of the Transformer architecture?

> 🔁 _See also [09-large-language-model.md → Q3](../ml-engineering/09-large-language-model.md#q3-what-are-the-key-components-of-the-transformer-architecture) (separate Notion notes on the same question)._

1. **Token Embeddings** — Convert tokens into vectors.
2. **Positional Encoding** — Adds token position information.
3. **Multi-Head Self-Attention** — Captures relationships between tokens using **Q, K, V**.
4. **Feed-Forward Network (FFN)** — Transforms the attention output.
5. **Residual Connections** — Improve information flow and training.
6. **Layer Normalization** — Stabilizes the network.
7. **Output Layer** — Produces probabilities for **next-token prediction**.

**Transformer Block:**

**Embedding → Position → Self-Attention → FFN → Residual + Norm → Repeat → Output**

### Q11. What is tokenization in LLMs?

#### Tokenization in LLMs

- **Tokenization** converts text into smaller units called **tokens** that an LLM can process.
- Tokens can be **words, subwords, characters, punctuation, or special tokens**.
- Modern LLMs commonly use **subword tokenization**, such as **BPE, WordPiece, or Unigram**.
- Each token is mapped to a **token ID**, which is then converted into an **embedding**.

**Flow:**

**Text → Tokens → Token IDs → Embeddings → Transformer**

**Interview:**

> Tokenization is the process of converting text into tokens and mapping them to numerical IDs so the LLM can process the input.

### Q12. Explain BPE (Byte Pair Encoding).

#### BPE (Byte Pair Encoding)

- **BPE** is a **subword tokenization algorithm** used by many LLMs.
- It starts with small units and **repeatedly merges the most frequent adjacent token pairs**.
- This creates a vocabulary containing both **common words and subwords**.

**Example:**

`"playing"` → `["play", "ing"]`

**Why BPE?**

- Handles **unknown/rare words** effectively.
- Keeps the vocabulary size manageable.
- Balances **word-level** and **character-level** tokenization.

**Interview:**

> **BPE builds a subword vocabulary by repeatedly merging the most frequent adjacent token pairs, allowing LLMs to efficiently represent both common and rare words.**

### Q13. Explain WordPiece and SentencePiece.

#### WordPiece

- **WordPiece** is a **subword tokenization algorithm** that constructs a fixed-size vocabulary from a corpus.
- It starts with smaller units and iteratively adds/merges subwords based on how well they improve the **likelihood of the training data**.
- Uses special markers such as `##` to indicate that a token is a continuation of a word.
- During inference, it uses a **greedy longest-match-first** strategy to tokenize text.
- Commonly associated with **BERT**.

**Example:**

```txt
"unwanted" → ["un", "##want", "##ed"]
```

---

#### SentencePiece

- **SentencePiece** is a **language-independent tokenizer framework**, not a single tokenization algorithm.
- It operates directly on **raw Unicode text**, so explicit whitespace-based pre-tokenization is not required.
- It commonly implements **BPE** or **Unigram Language Model** tokenization.
- Whitespace is treated as part of the tokenization process, often represented using `▁`.

**Example:**

```txt
"I love AI" → ["▁I", "▁love", "▁AI"]
```

##### Key Technical Difference

| WordPiece | SentencePiece |
| --- | --- |
| Subword algorithm | Tokenization framework |
| Likelihood-based vocabulary construction | Supports BPE/Unigram |
| Typically uses `##` for continuation | Often uses `▁` for word boundaries |
| Used by BERT | Used by models such as T5 and LLaMA |

**Interview:**

> **WordPiece is a likelihood-based subword tokenization algorithm, while SentencePiece is a language-independent tokenizer framework that can implement algorithms such as BPE and Unigram directly on raw text.**

### Q14. What is positional encoding, and why is it needed in Transformers?

#### Positional Encoding

- **Positional encoding** provides information about the **position and order of tokens** in a sequence.
- It is needed because **self-attention does not inherently encode token order**.
- The positional information is combined with token embeddings before processing by the Transformer.

##### Example

```txt
"I love AI" ≠ "AI love I"
```

The same tokens have different meanings when their positions change.

##### Common Approaches

- **Sinusoidal Positional Encoding** — Uses fixed sine/cosine functions.
- **Learned Positional Embeddings** — Position vectors are learned during training.
- **RoPE (Rotary Positional Embedding)** — Encodes position by rotating query/key representations; widely used in modern LLMs.

**Interview:**

> **Positional encoding injects sequence-order information into token representations because self-attention alone is permutation-invariant. It enables Transformers to distinguish tokens based on their positions.**

### Q15. What are embeddings?

> 🔁 _See also [02-fundamentals-of-machine-learning.md → Q13](../ml-engineering/02-fundamentals-of-machine-learning.md#q13-what-are-embeddings-in-machine-learning) (separate Notion notes on the same question)._

#### Embeddings

- **Embeddings** are dense, numerical vector representations of data such as **text, images, or audio**.
- They encode **semantic and contextual relationships** in a continuous vector space.
- Similar concepts tend to have **similar vector representations**.
- In LLMs, token IDs are mapped to **embedding vectors** before being processed by the Transformer.

##### Example

```txt
"king" → [0.21, -0.45, 0.73, ...]
"queen" → [0.19, -0.42, 0.71, ...]
```

##### Key Properties

- **Dense:** Most dimensions contain meaningful values.
- **Continuous:** Representations exist in a continuous vector space.
- **Semantic:** Capture relationships between concepts.
- **Fixed-dimensional:** Each embedding has a predefined number of dimensions.

**Interview:**

> **An embedding is a dense vector representation that maps data into a numerical space where semantic or structural relationships can be measured using metrics such as cosine similarity.**

### Q16. Explain the Query(Q), Key(K), and Value(V) in attention.

> 🔁 _See also [09-large-language-model.md → Q17](../ml-engineering/09-large-language-model.md#q17-explain-the-queryq-keyk-and-valuev-in-attention) (separate Notion notes on the same question)._

#### Query (Q), Key (K), and Value (V)

In **self-attention**, each token is transformed into three vectors:

- **Query (Q):** Represents what the current token is **looking for**.
- **Key (K):** Represents what information each token **offers/matches against**.
- **Value (V):** Contains the **actual information** passed to the output.

##### Attention Calculation

```txt
Q = XW_Q
K = XW_K
V = XW_V

Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **QKᵀ** → Measures similarity between queries and keys.
- **√dₖ** → Scaling factor to prevent excessively large values.
- **Softmax** → Converts scores into attention weights.
- **× V** → Produces the weighted information.

##### Simple Example

For:

> **"The animal didn't cross the street because it was tired."**

The query for **"it"** can assign higher attention to the key corresponding to **"animal"**, allowing its value to contribute more to the representation of **"it"**.

**Interview:**

> **Q determines what a token is looking for, K determines how relevant each token is to that query, and V contains the information that gets aggregated. Attention computes Q-K similarity, converts it into weights using softmax, and uses those weights to combine the V vectors.**

### Q17. What is self-attention, and how does it work in Transformers?

> 🔁 _See also [09-large-language-model.md → Q18](../ml-engineering/09-large-language-model.md#q18-what-is-self-attention-and-how-does-it-work-in-transformers) (separate Notion notes on the same question)._

#### Self-Attention

- **Self-attention** allows each token to determine how much it should focus on other tokens in the same sequence.
- Each token is transformed into **Query (Q), Key (K), and Value (V)** vectors.
- Attention scores are calculated using **QKᵀ**, then normalized with **softmax**.
- These scores determine how much each **Value (V)** contributes to the token's output.

##### Formula

```txt
Q = XW_Q
K = XW_K
V = XW_V

Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

##### Process

**Input → Q, K, V → QKᵀ → Scaling → Softmax → Weighted V → Output**

**Interview:**

> **Self-attention allows each token to dynamically attend to other tokens in the same sequence. It computes Q-K similarity to obtain attention weights and uses those weights to combine the Value vectors, producing context-aware representations.**

### Q18. What is cross-attention in Transformers?

> 🔁 _See also [09-large-language-model.md → Q19](../ml-engineering/09-large-language-model.md#q19-what-is-cross-attention-in-transformers) (separate Notion notes on the same question)._

#### Cross-Attention

- **Cross-attention** allows one sequence to attend to information from a **different sequence**.
- **Q** comes from the current/target sequence, while **K and V** come from the source sequence.
- Commonly used in **encoder-decoder Transformers**, such as machine translation.

##### Formula

```txt
Q = X_target W_Q
K = X_source W_K
V = X_source W_V

CrossAttention(Q, K, V) = softmax(QK^T / √d_k) V
```

##### Self-Attention vs Cross-Attention

- **Self-attention:** Q, K, V come from the **same sequence**.
- **Cross-attention:** Q comes from one sequence; K, V come from **another sequence**.

**Interview:**

> **Cross-attention enables one sequence to selectively retrieve information from another sequence by generating Q from the target sequence and K, V from the source sequence.**

#### Notes from a repeated entry in Notion: "Cross Attention?"

**Cross-Attention** allows one sequence to attend to another sequence.

The key idea is:

> **Queries come from one sequence, while Keys and Values come from another sequence.**

### Q19. Why do we scale the dot product attention by √dₖ in the Transformer architecture?

> 🔁 _See also [09-large-language-model.md → Q5](../ml-engineering/09-large-language-model.md#q5-why-do-we-scale-the-dot-product-attention-by-dₖ-in-the-transformer-architecture) (separate Notion notes on the same question)._

In scaled dot-product attention:

```txt
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

- **QKᵀ** produces attention scores.
- As **dₖ increases**, the dot-product values tend to become larger.
- Large scores make **softmax extremely peaked**, causing very small gradients and unstable learning.
- Dividing by **√dₖ** keeps the scores at a reasonable scale.
- This makes **softmax more stable** and improves gradient flow during training.

#### Interview

> **We divide the attention scores by √dₖ to prevent their magnitude from growing with the dimensionality of the key vectors, which keeps softmax well-behaved and improves training stability.**

### Q20. What is causal masking?

> 🔁 _See also [09-large-language-model.md → Q29](../ml-engineering/09-large-language-model.md#q29-what-is-causal-masking) (separate Notion notes on the same question)._

#### Causal Masking

- **Causal masking** prevents a token from attending to **future tokens** during autoregressive generation.
- It ensures the model can only use the **current and previous tokens**.
- Typically implemented by setting future attention scores to **−∞** before softmax.

##### Example

```txt
Tokens:    I   love   AI   models

I          ✓    ✗     ✗     ✗
love       ✓    ✓     ✗     ✗
AI         ✓    ✓     ✓     ✗
models     ✓    ✓     ✓     ✓
```

##### Formula

```txt
Attention(Q, K, V) =
softmax((QK^T + M) / √d_k) V
```

Where **M** is the causal mask:

```txt
M[i,j] = 0      if j ≤ i
M[i,j] = −∞    if j > i
```

**Interview:**

> **Causal masking prevents each token from attending to future tokens, preserving the autoregressive property required for next-token prediction.**

### Q21. What are multi-head attention mechanisms? Why use multiple attention heads?

> 🔁 _See also [09-large-language-model.md → Q20](../ml-engineering/09-large-language-model.md#q20-what-are-multi-head-attention-mechanisms-why-use-multiple-attention-heads) (separate Notion notes on the same question)._

#### Multi-Head Attention

- **Multi-head attention** runs multiple attention mechanisms (**heads**) in parallel.
- Each head learns different **relationships or patterns** between tokens.
- Each head has separate **Q, K, V projection matrices**.

##### Formula

```txt
head_i = Attention(XW_Q^i, XW_K^i, XW_V^i)

MultiHead(X) = Concat(head_1, ..., head_h) W_O
```

##### Why Multiple Heads?

- Capture **different relationships** simultaneously.
- Learn different types of information, such as **syntax, semantics, and positional relationships**.
- Provide richer representations than a single attention mechanism.

**Interview:**

> **Multi-head attention performs attention in parallel across multiple learned projections, allowing the Transformer to capture different relationships and representations from the same input sequence.**

#### Notes from a repeated entry in Notion: "Multi Head Attention?"

**Multi-Head Attention (MHA)** allows the model to perform attention from multiple different representation subspaces simultaneously.

The key idea is:

> **Each attention head has its own Query, Key, and Value projections.**

### Q22. What are Feed-Forward Networks in LLMs?

> 🔁 _See also [09-large-language-model.md → Q23](../ml-engineering/09-large-language-model.md#q23-what-are-feed-forward-networks-in-llms) (separate Notion notes on the same question)._

- **Feed-Forward Network (FFN)** is the **non-linear transformation layer** inside each Transformer block.
- It processes each token **independently** after self-attention.
- Typical structure: FFN(x)=W2σ(W1x+b1)+b2
- It usually **expands → activates → compresses** the hidden representation.
- **Attention:** mixes information **between tokens**.
- **FFN:** transforms and enriches **each token's representation**.
- Modern LLMs often use **gated FFNs such as SwiGLU** for better performance.

**Interview answer:**

> “FFN is the non-linear component of a Transformer. Attention mixes information across tokens, while the FFN independently transforms each token representation to learn richer features.”

### Q23. What is the context window in LLMs, and why does it matter?

- **Context window** is the maximum amount of text/tokens an LLM can consider **at once** when generating a response.
- It includes the **input prompt, conversation history, retrieved documents, and sometimes the generated output**, depending on the model/API.
- Example: A **128K-token context window** can process up to roughly 128,000 tokens in a single context.
- **Why it matters:**
  - Larger context → can handle longer documents and conversations.
  - Important for **RAG, coding, document analysis, and agents**.
  - Larger context does **not automatically mean better reasoning**; models can still struggle to effectively use information buried in very long contexts.
  - Larger contexts also generally increase **memory/computation requirements and cost/latency**.

**Interview answer:**

> “The context window is the maximum number of tokens an LLM can process as context in a single request. It determines how much information the model can consider at once, making it important for long conversations, RAG, and document analysis. However, larger context can increase cost and latency, and the model may not use all information equally well.”

### Q24. Why is the context window limited in LLMs?

> 🔁 _See also [09-large-language-model.md → Q39](../ml-engineering/09-large-language-model.md#q39-why-is-the-context-window-limited-in-llms) (separate Notion notes on the same question)._

- **Context window** is the maximum amount of text/tokens an LLM can consider **at once** when generating a response.
- It includes the **input prompt, conversation history, retrieved documents, and sometimes the generated output**, depending on the model/API.
- Example: A **128K-token context window** can process up to roughly 128,000 tokens in a single context.
- **Why it matters:**
  - Larger context → can handle longer documents and conversations.
  - Important for **RAG, coding, document analysis, and agents**.
  - Larger context does **not automatically mean better reasoning**; models can still struggle to effectively use information buried in very long contexts.
  - Larger contexts also generally increase **memory/computation requirements and cost/latency**.

**Interview answer:**

> “The context window is the maximum number of tokens an LLM can process as context in a single request. It determines how much information the model can consider at once, making it important for long conversations, RAG, and document analysis. However, larger context can increase cost and latency, and the model may not use all information equally well.”

> ✍️ _Addition drafted during sync._

The Notion answer above is the same as Q23 (what the context window is and why it matters). It does not explain why the window is limited. The main reasons are:

- **Attention cost grows quadratically.** Standard self-attention scores every token against every other token, so compute and attention memory scale as O(n²) with sequence length. Doubling the context roughly quadruples attention work during prefill.
- **The KV cache grows linearly with length.** The model keeps K and V tensors for every previous token in every layer. At long contexts and large batch sizes, the KV cache (not the weights) often dominates GPU memory and limits how many concurrent requests a server can handle.
- **Training length and positional encoding.** Models are trained on sequences up to a maximum length. Positional schemes such as RoPE do not extrapolate reliably far beyond it without scaling techniques (position interpolation, YaRN) plus long-context fine-tuning.
- **Data and training cost.** High-quality long documents are scarcer, and training on long sequences is expensive.
- **Effective vs. nominal context.** Even within the limit, quality degrades for information buried in the middle ("lost in the middle"), so providers cap the window where quality, latency and cost stay acceptable.

**Production angle:** model-level mitigations include FlashAttention, GQA/MQA, KV-cache quantization and paging, and sliding-window attention. System-level approaches (RAG, chunking, summarization/compaction) are usually cheaper than relying on an ever-larger raw window.

### Q25. What is temperature in the context of LLMs, and how does it affect output?

- **Temperature** is a decoding parameter that controls the **randomness of an LLM’s output** by modifying the probability distribution over the next token.
- **Low temperature (e.g., 0–0.3):** More deterministic and focused → useful for factual answers, extraction, and structured outputs.
- **Medium temperature (e.g., 0.5–0.8):** Balanced between consistency and diversity.
- **High temperature (e.g., 1.0+):** More diverse and unpredictable → useful for creative generation.
- Temperature **does not change the model’s knowledge or weights**; it changes how tokens are sampled during generation.

**Interview answer:**

> “Temperature controls the randomness of token sampling. Lower temperatures make the output more deterministic and focused, while higher temperatures increase diversity and randomness. It affects the decoding process, not the model’s underlying knowledge.”

### Q26. Why is the first token slower than the rest in an LLM?

- The first token is slower because the LLM must perform **prompt processing (prefill)** before it can generate the first output token.
- During prefill, the model:
  - Tokenizes the input.
  - Runs the entire prompt through all Transformer layers.
  - Computes and stores **KV (Key-Value) cache** for the prompt.
- After that, token generation is **autoregressive**: the model generates one token at a time and reuses the KV cache instead of recomputing the entire prompt.

**Key terms:**

- **TTFT (Time to First Token):** Time until the first output token.
- **TPOT (Time Per Output Token):** Time required to generate subsequent tokens.

**Interview answer:**

> “The first token has higher latency because the model must first process the entire input prompt and build the KV cache. Once the KV cache is available, subsequent tokens can be generated incrementally without recomputing the whole prompt, so they are typically much faster.”

### Q27. Explain Top-p (nucleus) sampling and Top-k sampling. How do they differ?

- **Top-k sampling:** Selects the **k most probable next tokens**, then samples from only those tokens.
  - Example: `top_k = 50` → only the 50 highest-probability tokens are considered.
  - Fixed number of candidate tokens.
- **Top-p (nucleus) sampling:** Selects the **smallest set of tokens whose cumulative probability reaches p**, then samples from that set.
  - Example: `top_p = 0.9` → considers enough tokens to cover 90% of the probability mass.
  - Number of candidate tokens is **dynamic**.

#### Key difference

| Top-k | Top-p |
| --- | --- |
| Fixed number of tokens | Dynamic number of tokens |
| `k` controls candidate count | `p` controls probability mass |
| Can be too restrictive or too broad depending on distribution | Adapts to the model's probability distribution |

**Interview answer:**

> “Top-k limits sampling to the k most probable tokens, while Top-p dynamically selects the smallest set of tokens whose cumulative probability exceeds p. Top-k uses a fixed candidate size, whereas Top-p adapts to the probability distribution.”

### Q28. What are logits, and how are they used in text generation?

- **Logits** are the raw, unnormalized scores produced by the LLM for every possible next token.
- The model converts the final hidden representation into a **logit for each token** in the vocabulary.
- **Softmax** converts logits into probabilities:

P(token\_i) = e\^(z\_i) / Σ\_j e\^(z\_j)

- The next token is then selected using **greedy decoding or sampling** (e.g., temperature, Top-k, Top-p).

**Interview answer:**

> “Logits are the raw scores the LLM produces for each possible next token. They are converted into probabilities using softmax, and the decoding strategy then selects the next token based on those probabilities.”

### Q29. What are skip connections (residual connections) in Transformers?

- **Skip/Residual connections** allow the input of a Transformer layer to be **added directly to its output**.
- Typically:

```txt
Output = x + Sublayer(x)
```

- Transformers use residual connections around **self-attention** and **FFN** layers.
- **Why they matter:**
  - Improve **gradient flow** during training.
  - Reduce the problem of **vanishing gradients**.
  - Make training **deep Transformers** more stable.
  - Allow a layer to learn a **residual modification** instead of an entirely new representation.

**Interview answer:**

> “Residual connections add a layer’s input directly to its output, allowing information and gradients to flow more easily through deep Transformer networks. They improve training stability and make very deep LLMs easier to optimize.”

### Q30. What is the difference between open-source and closed-source LLMs? When would you choose one over the other?

- **Open-source/open-weight LLMs:** Model weights are publicly available, allowing organizations to **run, customize, fine-tune, and self-host** the model, subject to its license.
- **Closed-source LLMs:** Model weights and training details are generally proprietary; users typically access them through an **API or hosted service**.

#### Key differences

| Factor | Open / Open-weight | Closed-source |
| --- | --- | --- |
| Customization | High | Limited |
| Self-hosting | Usually possible | Usually not |
| Data control | Higher | Depends on provider |
| Cost | Infrastructure cost | API/usage cost |
| Maintenance | Your responsibility | Provider handles it |
| Latest capabilities | Varies | Often strong/frequently updated |

#### When to choose?

**Choose open/open-weight when:**

- You need **data privacy and full control**.
- You need **fine-tuning or model customization**.
- You want to **self-host** or control inference infrastructure.
- You need predictable long-term infrastructure costs.

**Choose closed-source when:**

- You want **strong capabilities without managing infrastructure**.
- You need to move quickly to production.
- You don't need access to model weights.
- You prefer the provider to handle **scaling, serving, and model updates**.

**Interview answer:**

> “I would choose an open-weight model when control, privacy, customization, or self-hosting is important. I would choose a closed-source model when I prioritize ease of deployment, strong out-of-the-box performance, scalability, and lower infrastructure management. The decision should ultimately depend on performance, cost, latency, data privacy, licensing, and customization requirements.”

### Q31. What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?

> 🔁 _See also [06-deep-learning.md → Q20](../ml-engineering/06-deep-learning.md#q20-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformer-architectures) (separate Notion notes on the same question)._

#### Encoder-only vs Decoder-only vs Encoder-Decoder

| Architecture | Attention | Typical use | Examples |
| --- | --- | --- | --- |
| **Encoder-only** | Bidirectional | Understanding / representation | BERT, RoBERTa |
| **Decoder-only** | Causal (left-to-right) | Text generation | GPT, Llama, Claude |
| **Encoder-Decoder** | Encoder: bidirectional; Decoder: causal + cross-attention | Sequence-to-sequence tasks | T5, BART |

- **Encoder-only:** Reads the entire input and creates contextual representations. Best for **classification, NER, embeddings, and understanding**.
- **Decoder-only:** Predicts the **next token** based on previous tokens. Best for **text generation, chat, coding, and LLM applications**.
- **Encoder-decoder:** Encoder understands the input, and decoder generates the output while attending to the encoder's representations. Best for **translation, summarization, and other input-to-output transformations**.

**Interview answer:**

> “Encoder-only models are optimized for understanding input, decoder-only models are optimized for autoregressive generation, and encoder-decoder models combine both for sequence-to-sequence tasks. For modern generative LLM applications, decoder-only architectures are the most common.”

### Q32. What is KV cache, and how does it speed up inference?

- **KV Cache (Key-Value Cache)** stores the **Key and Value vectors** computed by the Transformer’s attention layers for previously processed tokens.
- During autoregressive generation, the model does **not recompute K and V for all previous tokens**.
- Instead, it reuses the cached K/V and computes attention only for the **new token**.

#### Why it speeds up inference

Without KV cache:

> Each new token requires recomputing attention for all previous tokens.

With KV cache:

> Previous K/V values are reused, significantly reducing redundant computation.

**Interview answer:**

> “KV cache stores the Key and Value tensors from previous tokens during autoregressive generation. When generating the next token, the model reuses these cached values instead of recomputing them, which significantly reduces computation and improves inference latency.”

**Key point:**

**KV Cache → less redundant computation → faster token generation.**

### Q33. What is model distillation, and how is it used with LLMs?

- **Model distillation (Knowledge Distillation)** is a technique where a smaller **student model learns to reproduce the behavior of a larger, more capable teacher model**.
- Instead of learning only from hard labels, the student can learn from the teacher's **soft probability distribution (logits)**, which contains richer information about the teacher's predictions.
- In LLMs, distillation is used to create models that are:
  - **Smaller**
  - **Faster**
  - **Cheaper to serve**
  - More suitable for **edge or high-throughput applications**

#### Typical process

```txt
Large Teacher LLM
       ↓
Teacher outputs / logits
       ↓
   Distillation
       ↓
Smaller Student LLM
```

**Interview answer:**

> “Knowledge distillation is a technique where a smaller student model learns from a larger teacher model, typically using the teacher’s outputs or probability distributions. For LLMs, it is used to transfer capabilities from large models into smaller models, reducing inference cost and latency while retaining as much performance as possible.”

### Q34. What is Mixture of Experts (MoE), and how does it work in models like Mixtral?

> 🔁 _See also [09-large-language-model.md → Q41](../ml-engineering/09-large-language-model.md#q41-what-is-mixture-of-experts-moe-and-how-does-it-work-in-models-like-mixtral) (separate Notion notes on the same question)._

- **Mixture of Experts (MoE)** is a Transformer architecture where the model contains multiple **expert networks**, but only a subset of experts is activated for each token.
- A **router/gating network** decides which experts should process each token.
- For example, in **Mixtral**, each token is routed to a small number of experts rather than all experts.

```txt
Token
  ↓
Router
  ↓
Select top experts
  ↓
Expert 1 + Expert 3
  ↓
Combine outputs
```

#### Why use MoE?

- **More total parameters** → higher model capacity.
- **Fewer active parameters per token** → lower computation than a dense model with the same total parameter count.
- Can improve **performance while keeping inference relatively efficient**.

**Interview answer:**

> “Mixture of Experts is an architecture where a model contains multiple expert networks, and a router dynamically selects a small subset of experts for each token. This allows the model to have a large number of total parameters while activating only a fraction for each token, improving model capacity without proportionally increasing computation.”

### Q35. What is the difference between dense and sparse models?

> 🔁 _See also [09-large-language-model.md → Q42](../ml-engineering/09-large-language-model.md#q42-what-is-the-difference-between-dense-and-sparse-models) (separate Notion notes on the same question)._

- **Dense model:** Every input token passes through **all model parameters/layers** during inference.
  - Example: A traditional Transformer with all FFN parameters active for every token.
  - Higher computation per token.
- **Sparse model:** Only a **subset of parameters** is activated for each input token.
  - **MoE (Mixture of Experts)** is a common example.
  - A router selects specific experts for each token.
  - Lower computation per token while allowing a large total parameter count.

#### Key difference

> **Dense → all parameters are active.**
>
> **Sparse → only selected parameters are active.**

**Interview answer:**

> “In a dense model, all relevant model parameters are activated for every input token. In a sparse model, only a subset of parameters is activated, typically through mechanisms like Mixture of Experts. Sparse models can therefore provide higher total model capacity without proportionally increasing computation per token.”

### Q36. What is Flash Attention?

> 🔁 _See also [09-large-language-model.md → Q8](../ml-engineering/09-large-language-model.md#q8-what-is-flash-attention) (separate Notion notes on the same question)._

- **FlashAttention** is an **IO-aware algorithm for computing Transformer attention more efficiently**. It does not change the attention formula; it optimizes how attention is computed and moved through GPU memory.
- Standard attention materializes large intermediate matrices such as the QKT attention matrix, which requires significant **GPU memory reads/writes**.
- FlashAttention uses **tiling/blocking** and keeps intermediate data in fast GPU memory (SRAM/cache) as much as possible, reducing **HBM memory traffic**.

#### Benefits

- **Faster attention computation**
- **Lower GPU memory usage**
- Enables **larger context lengths**
- Particularly useful for training and inference of Transformers/LLMs

**Interview answer:**

> “FlashAttention is an IO-aware, memory-efficient algorithm for computing attention. Instead of materializing the full attention matrix in GPU high-bandwidth memory, it uses tiling and efficient memory access to reduce GPU memory traffic. This makes attention significantly faster and more memory efficient without changing the mathematical result of attention.”

### Q37. What is Cross-Entropy Loss?

- **Cross-Entropy Loss** measures the difference between the **true probability distribution** and the model’s predicted probability distribution.
- For classification, it penalizes the model when it assigns **low probability to the correct class**.

For a single example:

```txt
L = -log(P(correct class))
```

For LLMs, it is commonly used for **next-token prediction**:

```txt
Input:  "The sky is"
Target: "blue"

Loss = -log(P("blue"))
```

- High probability for the correct token → **low loss**
- Low probability for the correct token → **high loss**

**Interview answer:**

> “Cross-entropy loss measures how well the model’s predicted probability distribution matches the true target. In LLMs, it is typically used for next-token prediction, where the model is penalized when it assigns low probability to the correct next token.”

### Q38. What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?

> 🔁 _See also [09-large-language-model.md → Q22](../ml-engineering/09-large-language-model.md#q22-what-is-grouped-query-attention-gqa-and-how-does-it-differ-from-multi-head-attention-mha) (separate Notion notes on the same question)._

#### Grouped-Query Attention (GQA)

- **GQA** is an attention mechanism that reduces the **KV-cache size and inference cost** by sharing **Key and Value heads across multiple Query heads**.
- In **MHA (Multi-Head Attention)**, every Query head has its own **Key and Value head**.
- In **GQA**, multiple Query heads share the same Key/Value head.

```txt
MHA:
Q1 → K1, V1
Q2 → K2, V2
Q3 → K3, V3
Q4 → K4, V4

GQA:
Q1 ─┐
Q2 ─┼→ K1, V1
Q3 ─┤
Q4 ─┘
```

#### Why GQA?

- **Smaller KV cache**
- **Lower memory usage**
- **Faster inference**
- Maintains much of MHA's quality compared with more aggressive **Multi-Query Attention (MQA)**.

#### Interview answer

> **“GQA is a compromise between MHA and MQA. In MHA, every query head has its own Key and Value heads, while in GQA, multiple query heads share a smaller number of Key/Value heads. This significantly reduces KV-cache memory and improves inference efficiency while retaining performance close to MHA.”**

#### Notes from a repeated entry in Notion: "Grouped-Query Attention?"

**Grouped-Query Attention (GQA)** is an attention mechanism that sits between **Multi-Head Attention (MHA)** and **Multi-Query Attention (MQA)**.

The key idea is:

> **Multiple query heads share the same Key and Value heads.**

### Q39. How does Rotary Position Embedding (RoPE) work, and why is it preferred over learned positional embeddings?

> 🔁 _See also [09-large-language-model.md → Q31](../ml-engineering/09-large-language-model.md#q31-how-does-rotary-position-embedding-rope-work-and-why-is-it-preferred-over-learned-positional-embeddings) (separate Notion notes on the same question)._

#### Rotary Position Embedding (RoPE)

- **RoPE** encodes token position by **rotating the Query (Q) and Key (K) vectors** according to their positions.
- Instead of adding a positional vector to token embeddings, RoPE applies a **position-dependent rotation** to Q and K before attention.

Conceptually:

```txt
Token representation
        ↓
   Q and K vectors
        ↓
Position-dependent rotation
        ↓
   Self-Attention
```

#### Why RoPE is preferred

- Encodes **relative positional information** naturally through the interaction between Q and K.
- Does not require a separate learned positional embedding for every position.
- Works well with **long-context Transformers** and can be extended to longer sequences using techniques such as **RoPE scaling**.
- Efficient because it is applied directly to Q and K.

**Interview answer:**

> **“RoPE encodes positional information by applying position-dependent rotations to the Query and Key vectors. Unlike learned positional embeddings, which add a learned vector to each token based on its absolute position, RoPE naturally captures relative position through the attention interaction between Q and K. It is efficient and works particularly well for modern long-context LLMs.”**

### Q40. Explain Layer Normalization

#### Layer Normalization

- **Layer Normalization (LayerNorm)** normalizes the activations **across the features of each individual token**, helping keep their values in a stable range.
- For a token representation x:

```txt
μ = mean(x)
σ² = variance(x)

x̂ = (x - μ) / √(σ² + ε)

output = γx̂ + β
```

- **γ (gamma)** and **β (beta)** are learnable parameters.
- In Transformers, LayerNorm helps **stabilize training and improve gradient flow**.
- Unlike **BatchNorm**, LayerNorm does **not depend on the batch size**.

**Interview answer:**

> “Layer Normalization normalizes the hidden features of each token independently, followed by learnable scaling and shifting. It stabilizes activations and makes deep Transformer training more stable. Unlike BatchNorm, it operates independently of the batch dimension, which makes it well suited for Transformers and variable-length sequences.”

### Q41. Explain RMSNorm (Root Mean Square Layer Normalization)

> 🔁 _See also [09-large-language-model.md → Q33](../ml-engineering/09-large-language-model.md#q33-explain-rmsnorm-root-mean-square-layer-normalization) (separate Notion notes on the same question)._

#### RMSNorm (Root Mean Square Normalization)

- **RMSNorm** is a simplified alternative to **LayerNorm** used in many modern LLMs.
- Instead of calculating both **mean and variance**, RMSNorm normalizes using only the **root mean square (RMS)** of the activations.

```txt
RMS(x) = √(mean(x²) + ε)

output = γ · x / RMS(x)
```

- RMSNorm **does not subtract the mean**, unlike LayerNorm.
- It is **computationally simpler and faster** while providing similar training stability.
- Used in modern LLM architectures such as **LLaMA**.

#### LayerNorm vs RMSNorm

| LayerNorm | RMSNorm |
| --- | --- |
| Subtracts mean | Does not subtract mean |
| Uses mean + variance | Uses RMS |
| More computation | Simpler/faster |
| Common in older Transformers | Common in modern LLMs |

**Interview answer:**

> **“RMSNorm is a simplified normalization technique that normalizes activations using their root mean square without subtracting the mean. It is computationally cheaper than LayerNorm while providing similar training stability, which is why it is widely used in modern LLMs such as LLaMA.”**

### Q42. Your LLM keeps ignoring your instructions. How do you make it follow structured output formats?

#### How to enforce structured output from an LLM

- **Use structured output / constrained decoding** when the framework or model supports it.
  - Examples: **JSON Schema, Pydantic structured output, function/tool calling**.
- **Clearly define the schema** and required fields/types.
- Use **low temperature** for more deterministic output.
- **Validate the response programmatically** before using it.
- If validation fails, **retry or repair** the output with the validation error.
- For critical systems, use **grammar/constrained decoding** rather than relying only on prompting.

**Typical pipeline:**

```txt
User Input
    ↓
LLM
    ↓
Structured Output / JSON Schema
    ↓
Validation (Pydantic / JSON Schema)
    ↓
Valid → Application
Invalid → Retry / Repair
```

**Interview answer:**

> **“I don't rely on prompting alone for structured output. I define an explicit schema, use native structured-output or tool-calling capabilities when available, validate the response programmatically, and retry or repair when validation fails. For strict requirements, I prefer constrained decoding or grammar-based generation. This gives much stronger reliability than simply asking the model to return JSON.”**

### Q43. Your LLM-powered tool hits the context window limit on long documents. How do you handle it?

#### Handling Context Window Limits

- **Chunk the document** into smaller, meaningful sections rather than sending the entire document to the LLM.
- Use **RAG**: retrieve only the most relevant chunks for the current query.
- Use **hierarchical summarization** for tasks requiring information from the entire document.
- **Compress/summarize conversation history** instead of passing the full history every time.
- Use **reranking** to select the most relevant chunks before sending them to the LLM.
- If appropriate, use a **larger-context model**, but don't rely on context size alone.

**Interview answer:**

> **“I wouldn't simply increase the context window. I would typically chunk the document, retrieve the most relevant chunks using RAG, rerank them, and pass only the necessary context to the LLM. For tasks requiring global understanding, I would use hierarchical or map-reduce summarization. I would also manage conversation history through summarization or compression. The goal is to provide the model with the most relevant information while controlling latency and cost.”**

### Q44. Your LLM does not admit when it does not know the answer. How do you make it say "I don't know"?

#### How to make an LLM say “I don't know”

- **Ground the model with trusted sources** using RAG or tool calls.
- Add an explicit instruction: **“If the provided context does not contain enough evidence, respond with ‘I don't know’ rather than guessing.”**
- Set a **confidence/relevance threshold** before sending retrieved context to the LLM.
- Require the model to provide **citations/evidence** for factual answers.
- Use **structured outputs** with fields such as `answer`, `confidence`, and `evidence`.
- **Validate/guardrail** the response and reject unsupported claims.
- For high-stakes systems, combine LLM output with **deterministic business rules or verification**.

**Interview answer:**

> **“I would not rely on prompting alone. I would ground the LLM with trusted context, use retrieval and relevance thresholds, and explicitly instruct it to answer only when sufficient evidence is available. I would require evidence or citations and validate the output. If the evidence is insufficient, the system should return ‘I don't know’ rather than hallucinate.”**

### Q45. Your LLM generates responses that are too verbose. How do you control response length?

- **Prompt constraints:** Specify the desired length, e.g., “Answer in 3–5 bullet points” or “under 100 words.”
- **Token limits:** Use `max_output_tokens` to enforce a hard upper limit.
- **Structured output:** Use JSON/schema constraints for predictable responses.
- **Context control:** In RAG, limit the amount of retrieved context to reduce unnecessary verbosity.
- **Validation:** Check output length and regenerate or summarize if it exceeds the limit.

### Q46. Your LLM memorized proprietary training data and leaks it in responses. How do you prevent this?

- **Data governance:** Don’t include sensitive/proprietary data in training unless explicitly authorized.
- **Data sanitization:** Remove PII, secrets, credentials, and confidential information before training.
- **Access control:** Restrict which users and applications can access sensitive data.
- **Privacy techniques:** Use data minimization, anonymization, and, where appropriate, differential privacy.
- **Output filtering:** Detect and block sensitive information in model responses.
- **Testing:** Perform privacy/red-team tests to identify memorization and data-extraction risks.
- **RAG:** Keep proprietary data in controlled databases and retrieve it only for authorized users instead of embedding it directly into model weights.

### Q47. Your LLM coding assistant generates outdated code using deprecated libraries. How do you fix it?

- **Use up-to-date documentation:** Ground the LLM with current library/API documentation.
- **Version-aware prompting:** Explicitly provide the required library and version.
- **RAG:** Retrieve the latest official documentation before generating code.
- **Tool/API access:** Let the assistant verify APIs and package versions when possible.
- **Validation:** Run generated code through tests, linters, and type checkers.
- **Continuous updates:** Regularly update prompts, RAG indexes, and evaluation datasets.

### Q48. Your tokenizer splits important domain terms into meaningless subword pieces. How do you fix it?

- **Use a domain-specific tokenizer:** Train or adapt the tokenizer on domain-specific data.
- **Add special tokens:** Add frequently used domain terms as vocabulary tokens.
- **Vocabulary expansion:** Extend the tokenizer vocabulary for important terms.
- **Compare tokenization:** Evaluate tokenization before/after on representative domain data.
- **Fine-tune the model:** Fine-tune the model with domain-specific data after tokenizer changes.
- **Alternative:** Use a model/tokenizer already trained for the target domain.

### Q49. Your Transformer's KV cache grows too large during long sequence generation. How do you manage memory?

**KV cache memory management strategies:**

- **Multi-Query/Grouped-Query Attention (MQA/GQA)** — share K/V heads across multiple query heads, cutting cache size by the head-sharing ratio
- **Quantization** — store K/V in INT8/FP8 instead of FP16/FP32
- **Sliding window attention** — only cache the last N tokens, discard older ones (Mistral-style)
- **KV cache eviction/pruning** — drop low-importance tokens using attention-score heuristics (e.g., H2O, StreamingLLM's attention sinks)
- **PagedAttention (vLLM)** — allocate cache in non-contiguous fixed-size blocks like OS virtual memory, eliminating fragmentation
- **Cache compression** — low-rank projection or clustering of K/V vectors
- **Multi-head Latent Attention (MLA)** — compress K/V into a shared latent vector before caching (DeepSeek-style)
- **Layer-wise/cross-layer sharing** — reuse K/V across adjacent layers instead of storing per-layer
- **Offloading** — move older cache entries to CPU RAM or disk, stream back as needed
- **Batching-aware allocation** — dynamic memory pools sized per request instead of static max-length allocation
- **Speculative/early-exit truncation** — limit context length via summarization or chunk-and-summarize for very long sequences

### Q50. Your Transformer runs out of memory on long documents due to quadratic self-attention. How do you scale it?

- **Use sparse attention:** Replace full self-attention with **Sliding Window Attention**, local attention, or block-sparse attention so each token attends only to a limited number of nearby tokens.
- **Use efficient attention mechanisms:** Techniques such as **FlashAttention** reduce memory usage and improve GPU efficiency without changing the attention pattern.
- **Use linear attention:** Choose attention variants whose computational/memory cost grows approximately **linearly with sequence length** rather than quadratically.
- **Use hierarchical/chunked processing:** Split long documents into chunks, process each chunk separately, and then combine the chunk-level representations.
- **Use retrieval-based approaches:** Instead of putting the entire document into the context window, store document chunks in a **vector database** and retrieve only the most relevant chunks.
- **Use long-context architectures:** Models such as those using **Longformer-style sparse attention** or other long-context mechanisms are designed specifically for long sequences.
- **Use KV-cache efficiently:** For autoregressive inference, techniques such as **KV-cache quantization, compression, or paged KV caching** can significantly reduce memory consumption.
- **Practical approach:** For an LLM/RAG system, I would typically use **chunking + embeddings + retrieval + reranking**, while using **FlashAttention or sparse attention** when the model itself needs to process very long sequences.

### Q51. Your distilled student model fails on the complex reasoning that the teacher model handled. How do you close the gap?

- **Use knowledge distillation:** Train the student not only on the teacher's final answers but also on the teacher's **soft probabilities/logits**, which contain richer information about alternative answers.
- **Distill reasoning behavior:** Generate high-quality teacher explanations, intermediate reasoning steps, or structured solution traces and train the student on them. For safety and reliability, the student can learn **concise rationales or intermediate results** rather than blindly copying hidden chain-of-thought.
- **Use a reasoning-focused dataset:** Create more training examples involving **multi-step reasoning, mathematics, coding, planning, and difficult edge cases** rather than mostly simple examples.
- **Curriculum distillation:** Start with simpler problems and progressively introduce harder reasoning tasks.
- **Increase student capacity:** If the student is too small, increase its parameter count or use a stronger architecture. Some reasoning capability simply cannot be compressed beyond a certain point without loss.
- **Use intermediate-layer distillation:** Match selected **hidden representations or attention patterns** between teacher and student, not just the final output.
- **Use teacher-generated data:** Have the teacher generate additional difficult examples, solutions, and counterexamples, then filter them for quality before training the student.
- **Use preference/verification training:** Train the student to prefer correct solutions and use a verifier or reward model to reject incorrect reasoning.
- **Evaluate by capability:** Test separately on **simple knowledge, multi-step reasoning, mathematics, coding, and out-of-distribution problems** to identify exactly where the student loses capability.

### Q52. After RLHF alignment, your LLM became safer but lost capability on hard tasks. How do you manage the alignment tax?

- **Measure the alignment tax:** Evaluate the base model and aligned model separately on reasoning, coding, math, knowledge, safety, and instruction-following benchmarks to identify where capability was lost.
- **Use a capability-preserving objective:** Don't optimize only for the reward model. Combine the alignment objective with a **capability/behavior-preservation loss** so the model retains useful pretrained behavior.
- **Use KL regularization:** Keep the aligned model close to the capable reference model where there is no safety reason to change its behavior.
- **Improve the preference data:** RLHF can over-optimize noisy or simplistic preferences. Use high-quality, difficult examples and include **hard reasoning tasks** in the preference dataset.
- **Use targeted alignment:** Apply stronger safety constraints specifically where needed rather than globally suppressing behaviors that are actually useful.
- **Separate safety from capability:** Use classifiers, policy models, or guardrails around the LLM where appropriate, instead of forcing the core model to become overly conservative.
- **Use continued capability training:** After alignment, periodically train on math, coding, reasoning, and domain-specific data to recover capabilities while maintaining safety constraints.
- **Use better preference optimization:** Methods such as **DPO, IPO, or other preference-optimization approaches** can sometimes provide more stable alignment than aggressive online RL.
- **Track Pareto trade-offs:** Maintain separate safety and capability metrics and choose the training checkpoint that provides the best **safety–capability trade-off**, rather than maximizing safety alone.

### Q53. Your RLHF-trained LLM is gaming the reward model instead of being genuinely helpful. How do you fix reward hacking?

- **Improve the reward model:** Train it on diverse, high-quality human preferences, especially examples where superficial answers look good but are actually incorrect or unhelpful.
- **Use multiple reward signals:** Don't rely on a single reward model. Combine signals for **correctness, helpfulness, safety, factuality, and instruction-following**.
- **Use adversarial evaluation:** Deliberately search for cases where the model can get high reward through shortcuts, verbosity, manipulation, or other unintended behavior.
- **Use reward-model ensembles:** Multiple independently trained reward models make it harder for the policy to exploit a specific weakness in one reward model.
- **Regularize the policy:** Use a **KL penalty** to prevent the RL-trained model from moving too far from the reference model merely to maximize reward.
- **Improve reward-model robustness:** Continuously retrain the reward model using examples generated from the policy's newly discovered failure modes.
- **Use verifiable rewards:** For tasks such as mathematics and coding, use **objective verification**—for example, unit tests or exact answers—instead of relying entirely on human preference scores.
- **Monitor reward vs. actual performance:** If reward keeps increasing while human evaluations or task accuracy decrease, that's a strong signal of reward hacking.
- **Keep humans in the loop:** Periodically audit high-reward outputs and add problematic examples back into the preference dataset.

### Q54. Your chatbot loses context after 10 turns in a conversation. How do you maintain a long conversation context?

- **Use a conversation buffer:** Store recent messages and pass the relevant history to the LLM instead of sending only the last few turns.
- **Use summarization:** When the conversation becomes long, summarize older messages into a compact **conversation summary** and keep recent turns verbatim.
- **Use long-term memory:** Extract important facts, preferences, decisions, and user information into a persistent memory store such as **PostgreSQL, Redis, or a vector database**.
- **Use retrieval:** When generating a response, retrieve only the memories or previous messages relevant to the current query.
- **Use hierarchical memory:** Maintain separate layers:
  - Recent messages → exact conversation
  - Conversation summary → overall context
  - Long-term memory → important persistent facts
- **Manage the context window:** Track token usage and dynamically remove, summarize, or retrieve information before exceeding the model's context limit.
- **Use a sliding window:** Keep the most recent N turns while retrieving older relevant turns when needed.
- **Store structured state:** For agents, keep important information such as `user_name`, `current_task`, `preferences`, and `appointment_details` as structured state rather than relying entirely on raw chat history.
- **Persist state:** Give each conversation a unique `thread_id` and store its state in a database so context survives across API calls or application restarts.

### Q55. Your chatbot fails when users switch topics mid-conversation. How do you handle topic switches?

- **Detect topic changes:** Use a topic/intent classifier or LLM to determine whether the new message belongs to the current topic or starts a new one.
- **Maintain separate topic states:** Instead of keeping one undifferentiated history, maintain a separate state for each topic or task.
- **Use topic segmentation:** Group conversation turns into topic-based sessions, for example:
  - Topic A → “My AWS deployment”
  - Topic B → “My Python error”
  - Topic A → “Back to the AWS issue”
- **Use topic-aware retrieval:** Store previous conversation turns as embeddings with metadata such as `conversation_id` and `topic_id`. When the user returns to an earlier topic, retrieve the relevant history.
- **Preserve global context:** Keep information that applies across topics—such as user preferences or ongoing tasks—in a separate long-term memory layer.
- **Detect explicit switches:** Phrases like *“another question,” “unrelated,” “by the way,”* or *“let's talk about…”* can be strong topic-switch signals.
- **Handle ambiguous messages:** If it's unclear whether the user is continuing the old topic or starting a new one, ask a short clarification rather than retrieving irrelevant context.
- **Use a conversation state machine:** For agentic systems, maintain states such as `active_topic`, `previous_topics`, and `task_state`, allowing the system to switch and return to topics cleanly.

### Q56. Your QA system always generates an answer even when no answer exists in the context. How do you detect unanswerable questions?

- **Check retrieval quality first:** If the retrieved chunks do not contain evidence relevant to the question, the system should treat the question as potentially unanswerable.
- **Use similarity thresholds:** Set a minimum retrieval similarity score. If all retrieved documents fall below the threshold, return **“I don't know based on the provided context.”**
- **Use an entailment/grounding check:** Ask a separate model to determine whether the retrieved context actually **supports the proposed answer**.
- **Require evidence:** Make the LLM identify the specific passages that support its answer. If it cannot find supporting evidence, reject the answer.
- **Use a confidence threshold:** Combine retrieval confidence, answer confidence, and grounding scores. Only answer when the combined confidence exceeds a predefined threshold.
- **Use an abstention classifier:** Train a classifier on **answerable vs. unanswerable** questions so the system can explicitly choose to abstain.
- **Use negative examples:** Include questions whose answers are deliberately absent from the knowledge base during evaluation and training.
- **Prevent outside-knowledge leakage:** In strict RAG QA, instruct the model to answer **only from retrieved context**, rather than using its pretrained knowledge.
- **Evaluate separately:** Measure **answer accuracy, groundedness, and abstention accuracy** rather than only measuring whether the chatbot produces an answer.

### Q57. Your summarization system hallucinated facts not in the original article. How do you fix it?

- **Use extractive or grounded summarization:** Restrict the model to information explicitly present in the source article.
- **Add source grounding:** Require each important claim in the summary to be traceable to a specific sentence or passage in the original article.
- **Use a two-stage pipeline:** First extract the key facts from the article, then generate the summary only from those extracted facts.
- **Use factuality verification:** Run the generated summary through a separate verifier that checks whether each claim is supported by the source.
- **Use entailment checking:** Compare each summary statement against the article and reject or rewrite statements that are not entailed by the source.
- **Reduce decoding creativity:** Use lower temperature and constrained generation when factual accuracy is more important than stylistic variation.
- **Handle uncertainty explicitly:** If the article does not contain enough information, the system should omit the claim rather than infer or invent it.
- **Evaluate hallucination rate:** Build a test set containing difficult articles and measure **faithfulness, factual consistency, and unsupported-claim rate**.

### Q58. Your text generation repeats phrases in long outputs. How do you fix repetition?

- **Use repetition penalties:** Apply `repetition_penalty`, `frequency_penalty`, or `presence_penalty` during decoding to discourage repeating tokens and phrases.
- **Use n-gram blocking:** Prevent the model from generating the same n-gram repeatedly, such as blocking repeated 3-grams or 4-grams.
- **Tune decoding:** Adjust **temperature, top-k, and top-p** to improve generation diversity without making the output incoherent.
- **Improve the prompt:** Give the model clear instructions about structure, avoiding repeated points, and maintaining progression.
- **Use coverage mechanisms:** Track what has already been generated and encourage the model to focus on new information.
- **Use hierarchical generation:** For very long outputs, first generate an outline, then generate each section while maintaining a record of previously covered topics.
- **Control context:** Excessive or poorly organized context can cause the model to revisit the same information. Use relevant context rather than dumping the entire history.
- **Fine-tune on quality data:** If repetition is systematic, fine-tune with high-quality examples that demonstrate diverse, non-repetitive long-form generation.
- **Add a post-processing check:** Detect repeated sentences or phrases and regenerate the affected section.

### Q59. Transformers work on text, so can they also understand images?

> 🔁 _See also [09-large-language-model.md → Q43](../ml-engineering/09-large-language-model.md#q43-transformers-work-on-text-so-can-they-also-understand-images) (separate Notion notes on the same question)._

Yes. **Transformers can understand images**, but the original Transformer architecture was designed for sequences such as text. To process images, we represent the image as a sequence of numerical tokens.

- **Image → patches:** Split the image into small patches, such as 16×16 pixels.
- **Patches → embeddings:** Convert each patch into a vector representation.
- **Add positional information:** Tell the Transformer where each patch came from.
- **Apply self-attention:** The Transformer learns relationships between different image regions.
- **Generate an output:** Depending on the architecture, it can perform classification, object detection, image captioning, or visual question answering.

This is the basic idea behind **Vision Transformers (ViT)**.

For modern multimodal LLMs, the architecture is usually more sophisticated:

**Image → Vision Encoder → Visual Embeddings → LLM → Text Response**

The vision encoder converts the image into representations that the language model can use alongside text.

### Q60. Small Language Models (SLMs)

> 🔁 _See also [09-large-language-model.md → Q44](../ml-engineering/09-large-language-model.md#q44-small-language-models-slms) (separate Notion notes on the same question)._

**Small Language Models (SLMs)** are language models with significantly fewer parameters than large language models (LLMs), designed to perform specific language tasks efficiently with lower **memory, compute, latency, and cost**.

- **Smaller size:** Typically have millions to a few billion parameters, compared with large models that can have tens or hundreds of billions.
- **Lower resource requirements:** Can run on **CPUs, edge devices, laptops, or smaller GPUs**.
- **Faster inference:** Smaller models generally provide lower latency.
- **Lower cost:** Require less GPU memory and computational power.
- **Task-specific performance:** Can perform very well when **fine-tuned for a specific domain or task**.
- **Privacy:** They can often run locally, reducing the need to send sensitive data to cloud APIs.
- **Distillation:** An SLM can be trained using knowledge distilled from a larger teacher model.
- **Trade-off:** They generally have less broad knowledge and reasoning capability than larger models.

**Examples:** Phi, Gemma, Qwen, and Llama models have smaller variants suitable for SLM-style deployments.

### Q61. Large Reasoning Models (LRMs)

> 🔁 _See also [09-large-language-model.md → Q45](../ml-engineering/09-large-language-model.md#q45-large-reasoning-models-lrms) (separate Notion notes on the same question)._

#### Large Reasoning Models (LRMs)

**Large Reasoning Models (LRMs)** are large AI models optimized to solve **complex, multi-step problems** by spending additional computation on reasoning before producing the final answer.

- **Deep reasoning:** Designed for tasks requiring multiple logical steps, such as mathematics, coding, planning, and scientific problems.
- **Test-time computation:** They can use additional computation during inference to explore or verify possible solutions.
- **Better problem solving:** They generally perform better than standard LLMs on difficult reasoning benchmarks.
- **Self-verification:** They can evaluate intermediate results and correct mistakes before producing an answer.
- **Longer inference:** Reasoning can require more tokens and computation, increasing **latency and cost**.
- **Training:** They can be trained using techniques such as **reinforcement learning, supervised reasoning data, and outcome-based rewards**.
- **Not simply larger:** The key difference is not just parameter count; the model is optimized to **reason more effectively and allocate computation to difficult problems**.

#### LLM vs LRM

|  | LLM | LRM |
| --- | --- | --- |
| Primary goal | Generate useful responses | Solve complex problems |
| Reasoning | Usually faster/direct | More deliberate |
| Inference cost | Lower | Higher |
| Best for | Chat, summarization, extraction | Math, coding, planning, complex reasoning |
| Compute usage | Relatively fixed | Can allocate more computation to difficult problems |

### Q62. What are Autoregressive Models?

#### Autoregressive Models

**Autoregressive models** generate data **one step at a time**, where each new output depends on the previously generated outputs.

For language models:

> **Previous tokens → predict next token → add it → predict the next token → repeat**

For example:

> “The cat” → **“is”** → “The cat is” → **“sleeping”** → “The cat is sleeping”

- **Sequential generation:** Generates one token at a time.
- **Uses previous context:** Each prediction depends on the tokens already generated.
- **Training:** Typically learns to predict the next token from previous tokens.
- **Inference:** Continues generating until it reaches an end condition or token limit.
- **Examples:** GPT-style models, Llama, Qwen, and many other decoder-only Transformers.
- **Applications:** Text generation, code generation, chatbots, and speech/audio generation.

### Q63. Explain the difference between autoregressive and masked language modeling.

#### Autoregressive vs. Masked Language Modeling

The main difference is **what information the model can see when predicting a token**.

|  | Autoregressive (AR) | Masked Language Modeling (MLM) |
| --- | --- | --- |
| Prediction | Predicts the **next token** | Predicts **masked tokens** |
| Context | Usually sees tokens **before** the target | Can use **both left and right context** |
| Attention | Causal attention | Bidirectional attention |
| Training example | “The cat is → \_\_\_” | “The cat \[MASK\] sleeping” |
| Typical models | GPT, Llama, Qwen | BERT, RoBERTa |
| Best suited for | Text generation | Understanding/representation learning |

#### Simple example

**Autoregressive:**

> The cat is → **sleeping**

The model can use **“The cat is”**, but cannot look ahead at future tokens.

**Masked LM:**

> The cat **\[MASK\]** sleeping.

The model can use both **“The cat”** and **“sleeping”** to predict the missing word, such as **“is.”**

### Q64. Proximal Policy Optimization (PPO)

> 🔁 _See also [09-large-language-model.md → Q47](../ml-engineering/09-large-language-model.md#q47-proximal-policy-optimization-ppo) (separate Notion notes on the same question)._

#### Proximal Policy Optimization (PPO)

**PPO** is a **reinforcement learning algorithm** that updates a policy to maximize rewards while preventing the policy from changing too much in a single update.

It is widely used in **RLHF (Reinforcement Learning from Human Feedback)** to align language models with human preferences.

#### How PPO works

1. **Generate responses:** The current policy/LLM generates responses to prompts.
2. **Calculate reward:** A reward model evaluates those responses.
3. **Compare with the old policy:** PPO measures how much the new policy differs from the previous policy.
4. **Clip the update:** PPO limits how much the policy can change during one training step.
5. **Update the model:** The model learns to increase the probability of actions that produce higher rewards while avoiding excessively large policy changes.
6. **Repeat:** Generate → evaluate → update → repeat.

#### Why clipping matters

Without a constraint, the model could discover a high-reward behavior and change its policy too aggressively.

PPO uses a **clipped objective** to keep the new policy close to the old policy:

> **Small useful updates → accepted**
>
> **Too-large updates → clipped**

This makes training more stable.

#### PPO in RLHF

A simplified RLHF pipeline is:

**Pretrained LLM → SFT → Reward Model → PPO → Aligned LLM**

For example:

> User prompt → LLM generates answer → Reward Model gives reward → PPO updates LLM → better responses

PPO also commonly uses a **KL penalty** to prevent the RL-trained model from drifting too far from the reference model.

### Q65. Direct Preference Optimization (DPO)

> 🔁 _See also [09-large-language-model.md → Q48](../ml-engineering/09-large-language-model.md#q48-direct-preference-optimization-dpo) (separate Notion notes on the same question)._

#### Direct Preference Optimization (DPO)

**DPO** is a preference-learning method used to align language models with human preferences **without running a separate reinforcement-learning algorithm such as PPO**.

Instead of training a reward model and then optimizing the LLM with RL, DPO directly trains the model using **preferred and rejected responses**.

#### How DPO works

Suppose we have:

- **Prompt:** “Explain quantum computing.”
- **Chosen response:** High-quality, accurate explanation
- **Rejected response:** Poor or incorrect explanation

DPO trains the model to:

> **Increase the probability of the chosen response**
>
> **Decrease the probability of the rejected response**

It also compares the model against a **reference model** to prevent the policy from moving too far from its original behavior.

#### DPO vs PPO

|  | PPO | DPO |
| --- | --- | --- |
| Preference data | Yes | Yes |
| Reward model | Usually required | **Not required** |
| RL loop | Yes | **No** |
| Policy optimization | Reinforcement learning | Direct preference optimization |
| Complexity | Higher | Lower |
| Training stability | Can be challenging | Generally simpler |

#### RLHF with PPO

**Human preferences → Reward Model → PPO → Aligned LLM**

#### DPO

**Human preferences → DPO → Aligned LLM**

#### Why use DPO?

- Simpler training pipeline
- No separate reward-model training
- No online RL loop
- Generally easier to implement and stabilize
- Works directly with preference pairs
- Often more computationally efficient than PPO-based RLHF

### Q66. Group Relative Policy Optimization (GRPO)

> 🔁 _See also [09-large-language-model.md → Q49](../ml-engineering/09-large-language-model.md#q49-group-relative-policy-optimization-grpo) (separate Notion notes on the same question)._

#### Group Relative Policy Optimization (GRPO)

**GRPO** is a reinforcement learning algorithm designed to improve the reasoning ability of language models. It is especially associated with training reasoning models using **verifiable rewards**, such as mathematical or coding correctness.

The key idea is:

> **Generate multiple answers for the same prompt → compare their rewards relative to each other → reinforce the better answers.**

#### How GRPO works

Suppose the model receives:

> “Solve this mathematical problem.”

It generates several solutions:

- Response A → reward = 0.9
- Response B → reward = 0.7
- Response C → reward = 0.2
- Response D → reward = 0.5

Instead of requiring a separate value/critic model to estimate the value of each response, GRPO calculates a **relative advantage within the group**.

So approximately:

> **Above-average response → positive advantage → increase its probability**
>
> **Below-average response → negative advantage → decrease its probability**

#### Why "Group Relative"?

For each prompt, GRPO generates a **group of responses** and evaluates them relative to one another.

For example:

> Rewards: `[0.9, 0.7, 0.2, 0.5]`

The model learns which responses performed better **relative to the group**, rather than relying on an independently learned value function.

#### GRPO vs PPO

|  | PPO | GRPO |
| --- | --- | --- |
| RL algorithm | Yes | Yes |
| Multiple responses per prompt | Not required | **Yes** |
| Reward model | Often used | Can use **verifiable rewards** directly |
| Critic/value model | Typically required | **Not required** |
| Advantage estimation | Value function | **Group-relative rewards** |
| Memory requirement | Higher | Generally lower |
| Common use | RLHF | **Reasoning models** |

#### GRPO for reasoning

GRPO works particularly well when the reward can be **objectively verified**.

For example:

**Math problem → generate 8 solutions → check each answer → assign rewards → compare within the group → update the model**

Similarly for coding:

**Problem → generate solutions → run unit tests → reward correct solutions → reinforce them**

This is powerful because you don't necessarily need a human preference model to determine which solution is correct.

### Q67. Recursive Language Models (RLMs)

> 🔁 _See also [09-large-language-model.md → Q50](../ml-engineering/09-large-language-model.md#q50-recursive-language-models-rlms) (separate Notion notes on the same question)._

#### Recursive Language Models (RLMs)

**Recursive Language Models (RLMs)** are language-model systems that handle complex or very long inputs by **recursively breaking a problem into smaller subproblems**, processing them, and then combining the results.

The core idea is:

> **Large problem → smaller problems → solve recursively → combine results**

#### How RLMs work

For a long document, instead of asking one LLM call to process everything:

1. **Receive the large input**
2. **Split or decompose the task**
3. **Call the model on smaller portions**
4. **Process or summarize the intermediate results**
5. **Recursively repeat if a portion is still too complex**
6. **Combine the results into the final answer**

For example:

> 1,000-page document
>
> → divide into chapters
>
> → analyze each chapter
>
> → combine chapter-level findings
>
> → analyze the combined findings
>
> → produce final answer

#### Why use RLMs?

- Handle **inputs larger than the model's practical context window**
- Break complex reasoning into manageable pieces
- Reduce the amount of information processed in a single LLM call
- Enable hierarchical processing of large documents
- Useful for **long-document QA, codebases, research, and complex reasoning**

#### RLM vs RAG

They solve different problems:

- **RAG:** Retrieve only the most relevant information from a large knowledge base.
- **RLM:** Recursively process/decompose a large or complex input.
- **Hybrid:** Use RAG to retrieve relevant sections and RLM-style decomposition to analyze them hierarchically.

### Q68. Continual Learning in LLMs

> 🔁 _See also [09-large-language-model.md → Q51](../ml-engineering/09-large-language-model.md#q51-continual-learning-in-llms) (separate Notion notes on the same question)._

#### Continual Learning in LLMs

**Continual Learning** is the process of continuously updating an LLM with **new data, knowledge, or tasks over time** without having to train the model from scratch.

The main challenge is:

> **Learn new information without forgetting what the model already knows.**

#### Why is it needed?

LLMs may need to adapt to:

- New company or domain knowledge
- New products and documentation
- New programming frameworks
- Changing user preferences
- New languages or tasks
- Newly discovered information

#### Main challenges

- **Catastrophic forgetting:** Learning new data can cause the model to lose previously learned capabilities.
- **Data distribution shift:** New data may differ significantly from the original training data.
- **Knowledge conflicts:** New information can contradict information learned previously.
- **Model drift:** Continuous updates can gradually change model behavior.
- **Training cost:** Repeatedly fine-tuning a large model can be expensive.

#### Techniques to reduce forgetting

- **Replay:** Mix new training data with samples from older datasets.
- **Regularization:** Penalize changes to important model parameters.
- **Parameter-efficient fine-tuning:** Use **LoRA/adapters** instead of updating the entire model.
- **Knowledge distillation:** Preserve behavior from the previous model while learning new information.
- **Modular adapters:** Maintain separate adapters for different domains or tasks.
- **RAG:** For frequently changing knowledge, retrieve updated information at inference time instead of continually modifying model weights.
- **Periodic retraining:** Periodically consolidate accumulated knowledge into a new model version.

#### Continual Learning vs RAG

**RAG:**

> New information → database → retrieve at inference time

**Continual learning:**

> New information → training → update model parameters

For frequently changing information, **RAG is often safer and cheaper** because you don't need to modify the model every time the knowledge changes.

### Q69. How do Diffusion Language Models (DLMs) work?

> 🔁 _See also [06-deep-learning.md → Q47](../ml-engineering/06-deep-learning.md#q47-how-do-diffusion-language-models-dlms-work) (separate Notion notes on the same question)._

#### Diffusion Language Models (DLMs)

**Diffusion Language Models (DLMs)** generate text using a **denoising process**, rather than predicting tokens strictly from left to right like traditional autoregressive LLMs.

The core idea is:

> **Clean text → add noise → learn to remove noise → generate clean text**

#### How DLMs work

1. **Start with text**
   - During training, take a normal sentence or sequence of tokens.
2. **Add noise**
   - Randomly corrupt or mask parts of the sequence.
   - At high noise levels, much of the original information is lost.
3. **Learn denoising**
   - Train a Transformer to reconstruct the original text from the corrupted version.
4. **Start generation from noise**
   - During inference, begin with a highly corrupted or masked sequence.
5. **Iteratively denoise**
   - The model progressively predicts and refines the missing/corrupted tokens.
6. **Produce the final sequence**
   - After several denoising steps, the corrupted sequence becomes coherent text.

Conceptually:

> `[MASK] [MASK] the [MASK] is [MASK]`
>
> ↓
>
> `The [MASK] cat is [MASK]`
>
> ↓
>
> `The small cat is sleeping`

#### DLM vs Autoregressive LLM

|  | Autoregressive LM | Diffusion LM |
| --- | --- | --- |
| Generation | Token-by-token | Iterative denoising |
| Direction | Usually left → right | Can refine multiple positions |
| Parallelism | Limited during generation | Potentially more parallel |
| Generation steps | Roughly one per token | Multiple denoising iterations |
| Example approach | GPT-style | Diffusion-based text generation |

#### Why are DLMs interesting?

- Can potentially generate or refine **multiple tokens simultaneously**
- Naturally supports **infilling and editing**
- Can revise earlier generated content instead of being strictly committed to previous tokens
- May offer different speed/quality trade-offs from autoregressive generation

#### Main challenge

DLMs typically require **multiple denoising iterations**, so they don't automatically guarantee faster inference. The key research challenge is reducing the number of denoising steps while maintaining generation quality.

### Q70. Why layer normalization in Transformers?

**Layer Normalization (LayerNorm)** is used in Transformers to **stabilize activations and gradients, improve training stability, and make deep Transformer networks easier to optimize**.

#### Why LayerNorm instead of BatchNorm?

- **Independent of batch size:** LayerNorm does not use statistics from the mini-batch.
- **Works well with variable sequence lengths:** Each token is normalized independently.
- **Suitable for sequence models:** It works naturally with Transformers and RNNs.
- **Stable training:** It keeps the scale of hidden representations controlled, helping gradient flow.
- **Works with small batches:** It remains effective even when the batch size is very small or 1.

#### How does LayerNorm work?

For each token, LayerNorm calculates the **mean and variance across its feature dimensions**, not across the entire training dataset:

$$
\text{LayerNorm}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

where:

- μ = mean of the token's features
- σ2 = variance of the token's features
- γ,β = learnable scale and shift parameters
- ϵ = small value for numerical stability

#### BatchNorm vs LayerNorm

|  | Batch Normalization | Layer Normalization |
| --- | --- | --- |
| Normalizes | Features across mini-batch | Features within each sample/token |
| Depends on batch size | Yes | **No** |
| Commonly used in | CNNs | **RNNs, Transformers** |
| Small batch | Less reliable | **Works well** |
| Sequence modeling | Less suitable | **Very suitable** |

### Q71. Layer normalization vs. Batch Normalization?

#### Layer Normalization vs. Batch Normalization

| Aspect | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) |
| --- | --- | --- |
| **What it normalizes** | Each feature across the mini-batch | Features within each individual sample/token |
| **Mean & variance** | Calculated across the batch | Calculated across feature dimensions |
| **Depends on batch size** | Yes | **No** |
| **Small batch size** | Can be less effective | **Works well** |
| **Training vs. inference** | Uses batch/running statistics | Same basic operation in both |
| **Best suited for** | CNNs and many vision tasks | **Transformers, RNNs, sequence models** |
| **Sequence modeling** | Less suitable | **Very suitable** |
| **Regularization effect** | Some | Generally less than BatchNorm |
| **Transformer usage** | Rare | **Standard/common** |

##### Key Difference

**BatchNorm:**

> Normalize each feature using statistics calculated across different samples in a mini-batch.

**LayerNorm:**

> Normalize each individual sample/token using statistics calculated across its feature dimensions.

##### Why Transformers prefer LayerNorm

Transformers process sequences where **batch size and sequence length can vary**. LayerNorm does not depend on other samples in the batch, so each token can be normalized independently.

### Q72. What is SVD (singular value decomposition)?

> 🔁 _See also [advanced-ml.md → Q170](../advanced-ml.md#q170-singular-value-decomposition-svd) (separate Notion notes on the same question)._

**Singular Value Decomposition (SVD)** is a matrix factorization technique that decomposes a matrix into **three matrices**:

$$
A = U\Sigma V^T
$$

Where:

- **A** = original matrix
- **U** = left singular vectors
- **Σ** = diagonal matrix containing singular values
- **V\^T** = transpose of the right singular vectors

#### Key idea

SVD represents a complex matrix as a combination of **orthogonal directions and their importance (singular values)**.

The singular values in Σ\\Sigma are usually arranged from **largest to smallest**. Larger singular values represent the most important components of the data.

#### Applications

- **Dimensionality reduction**
- **Principal Component Analysis (PCA)**
- **Latent Semantic Indexing (LSI) in NLP**
- **Image compression**
- **Noise reduction**
- **Recommendation systems**
- **Low-rank matrix approximation**

#### SVD for dimensionality reduction

If only the top k singular values are retained:

$$
A \approx U_k\Sigma_kV_k^T
$$

This produces a **lower-dimensional approximation** while preserving most of the important information.

### Q73. What is KL Divergence?

**KL Divergence (Kullback–Leibler Divergence)** measures how different one probability distribution P is from another distribution Q.

It is commonly used to measure the **information loss when Q is used to approximate P**.

#### Formula

For discrete distributions:

$$
D_{KL}(P\parallel Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

#### Key points

- P = **true/reference distribution**
- Q = **approximate/estimated distribution**
- DKL(P∥Q)≥0
- DKL(P∥Q)=0 when P=Q,  P=Q
- It is **not symmetric**:

$$
D_{KL}(P\parallel Q)\neq D_{KL}(Q\parallel P)
$$

So, KL Divergence is **not a true distance metric**.

#### Example

Suppose:

P=\[0.7,0.3\]

and

Q=\[0.6,0.4\]

KL Divergence tells us how much information is lost when we use Q to represent P.

#### Applications in ML

- **Variational Autoencoders (VAEs):** regularize the learned latent distribution.
- **Knowledge distillation:** compare teacher and student probability distributions.
- **Generative models:** compare probability distributions.
- **Reinforcement learning:** constrain policy updates.
- **LLMs:** compare probability distributions between models or policies.

### Q74. What is entropy vs. cross-entropy?

> 🔁 _See also [02-fundamentals-of-machine-learning.md → Q27](../ml-engineering/02-fundamentals-of-machine-learning.md#q27-what-is-cross-entropy) (separate Notion notes on the same question)._

| Concept | Entropy | Cross-Entropy | KL Divergence |
| --- | --- | --- | --- |
| Notation | $H(P)$ | $H(P,Q)$ | $D_{KL}(P \parallel Q)$ |
| Formula | $-\sum_x P(x)\log P(x)$ | $-\sum_x P(x)\log Q(x)$ | $\sum_x P(x)\log\frac{P(x)}{Q(x)}$ |
| Measures | Uncertainty in $P$ | Prediction loss | Distribution mismatch |
| Question | How uncertain is $P$? | How well does $Q$ represent $P$? | How different is $Q$ from $P$? |
| Lower value | Less uncertainty | Better prediction | Less mismatch |
| Classification | Inherent uncertainty | Common training loss | Prediction mismatch |
| Minimum | $0$ for deterministic $P$ | $H(P)$ when $Q=P$ | $0$ when $P=Q$ |
| Relationship |  | $H(P,Q)=H(P)+D_{KL}(P\parallel Q)$ |  |
| Key idea | **Uncertainty** | **Prediction loss** | **Extra mismatch** |

### Q75. L1 vs L2 Geometry intuition?

|  | **L1** | **L2** |
| --- | --- | --- |
| **Constraint** | $\|w\|_1 = \sum_i \|w_i\| \leq c$ | $\|w\|_2 = \sqrt{\sum_i w_i^2} \leq c$ |
| **Shape** | Diamond | Circle |
| **Boundary** | Sharp corners | Smooth |
| **Corners on axes?** | Yes | No |
| **Likely solution** | On an axis | Anywhere on the circle |
| **Effect** | Some weights become exactly $0$ | Weights become small |
| **Key intuition** | **Sparsity** | **Shrinkage** |

### Q76. Multi Query Attention?

**Multi-Query Attention (MQA)** is an attention mechanism where **all Query heads share a single Key and Value head**.

The key idea is:

> **Multiple Query heads share the same Key and Value.**

### Q77. Multi-Head Attention vs Multi-Query Attention vs Grouped-Query Attention?

|  | **Multi-Head Attention (MHA)** | **Multi-Query Attention (MQA)** | **Grouped-Query Attention (GQA)** |
| --- | --- | --- | --- |
| **Query heads** | $h$ | $h$ | $h$ |
| **Key heads** | $h$ | $1$ | $g$ |
| **Value heads** | $h$ | $1$ | $g$ |
| **K/V sharing** | No sharing | All Q heads share K/V | Q heads share K/V within groups |
| **Example** | $8Q \rightarrow 8K,V$ | $8Q \rightarrow 1K,V$ | $8Q \rightarrow 2K,V$ |
| **KV cache** | Largest | Smallest | Medium |
| **Inference memory** | Highest | Lowest | Lower than MHA |
| **Representation capacity** | Highest | Reduced | Between MHA and MQA |
| **Main advantage** | Independent attention heads | Very low KV-cache cost | Balance of quality and efficiency |
| **Main trade-off** | Large KV cache | More K/V sharing | More K/V sharing than MHA |
| **Core idea** | Each Q has its own K/V | All Q share K/V | Groups of Q share K/V |

#### Visual intuition

**MHA**

```txt
Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
Q4 → K4,V4
Q5 → K5,V5
Q6 → K6,V6
Q7 → K7,V7
Q8 → K8,V8
```

**GQA**

```txt
Q1 ─┐
Q2 ─┤
Q3 ─┤ → K1,V1
Q4 ─┘

Q5 ─┐
Q6 ─┤
Q7 ─┤ → K2,V2
Q8 ─┘
```

**MQA**

```txt
Q1 ─┐
Q2 ─┤
Q3 ─┤
Q4 ─┤
Q5 ─┤ → K1,V1
Q6 ─┤
Q7 ─┤
Q8 ─┘
```

#### The key relationship

MHA→GQA→MQA

### Q78. What is reinforcement learning?

> 🔁 _See also [02-fundamentals-of-machine-learning.md → Q17](../ml-engineering/02-fundamentals-of-machine-learning.md#q17-what-is-reinforcement-learning) (separate Notion notes on the same question)._

**Reinforcement Learning** is a machine learning paradigm where an **agent learns by interacting with an environment** and receiving **rewards or penalties**.

##### Core components

- **Agent** → makes decisions
- **Environment** → responds to actions
- **State** → current situation
- **Action** → decision taken
- **Reward** → feedback from the environment
- **Policy** → strategy for choosing actions

$St→At→Rt+1,St+1$

#### Types of RL

##### 1. Value-Based RL

**Learn a value function** that tells us how good a state or state-action pair is.

For Q-learning:

$Q^*(s,a)$

represents the **maximum expected cumulative future reward** starting from state $s$, taking action $a$, and then behaving optimally.

The action is selected using:

$a^* = \arg\max_a Q^*(s,a)$

**Examples:** Q-Learning, $DQN$.

**Key idea:**

> Learn **how good each action is**, then choose the action with the highest value.

##### 2. Policy-Based RL

**Learn a policy directly** that maps states to actions or action probabilities.

$\pi_\theta(a|s)$

For example:

$s \rightarrow \pi_\theta(a|s)$

might output:

$P(a_1|s)=0.7,\quad P(a_2|s)=0.2,\quad P(a_3|s)=0.1$

The policy is optimized to maximize expected return:

$J(\pi_\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$

**Examples:** REINFORCE, Vanilla Policy Gradient.

**Key idea:**

> Learn **which actions to take**, directly.

|  | Value-Based | Policy-Based |
| --- | --- | --- |
| Learns | Value function | Policy |
| Main function | $Q(s,a)$ | $\pi(a\mid s)$ |
| Action selection | $\arg\max_a Q(s,a)$ | Sample/select from policy |
| Example | Q-Learning, DQN | REINFORCE, VPG |
| Core question | "How good is this action?" | "What action should I take?" |

### Q79. LM Cache vs KV Cache

|  | **KV Cache** | **LM Cache** |
| --- | --- | --- |
| What it caches | Attention **Keys & Values** | Previously computed **LLM outputs/states**, depending on implementation |
| Purpose | Avoid recomputing attention for previous tokens | Avoid recomputing repeated LLM computation |
| Used during | Autoregressive generation | Repeated/prefix/shared computation |
| Granularity | Token-level K/V tensors | Typically request/prefix-level computation |
| Main benefit | Faster token generation | Faster repeated inference |
| Memory | High GPU memory usage | Can use GPU/CPU/SSD depending on system |

### Q80. Dropout During Inference

- **Normal inference:** Dropout **OFF** → deterministic prediction.
- **MC Dropout:** Dropout **ON** → multiple stochastic predictions → estimate **uncertainty**.
  ${Normal: } \hat{y}=f(x)$

  ${MC Dropout: } \hat{y}_t=f(x;m_t), \quad t=1,\dots,T$

**Key idea:** Dropout is normally disabled during inference, but MC Dropout intentionally enables it for uncertainty estimation.
