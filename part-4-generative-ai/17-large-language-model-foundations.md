# Chapter 17. Large Language Model Foundations

_Part IV — Generative AI and LLM Engineering_

Transformers, attention, tokenization, pre-training, scaling laws, mixture of experts, reasoning models and hallucination.

This chapter contains **142 questions**.

## Contents

- **17.1** [What is an LLM?](#171-what-is-an-llm)
- **17.2** [What is RAG?](#172-what-is-rag)
- **17.3** [What is MCP?](#173-what-is-mcp)
- **17.4** [What is an AI agent?](#174-what-is-an-ai-agent)
- **17.5** [What is Fine-tuning?](#175-what-is-fine-tuning)
- **17.6** [What is Quantization?](#176-what-is-quantization)
- **17.7** [What are foundation models, and how have they changed AI engineering?](#177-what-are-foundation-models-and-how-have-they-changed-ai-engineering)
- **17.8** [Inside ChatGPT: What Happens After You Hit Enter?](#178-inside-chatgpt-what-happens-after-you-hit-enter)
- **17.9** [What is the Transformer architecture, and how does it work?](#179-what-is-the-transformer-architecture-and-how-does-it-work)
- **17.10** [What are the key components of the Transformer architecture?](#1710-what-are-the-key-components-of-the-transformer-architecture)
- **17.11** [What is tokenization in LLMs?](#1711-what-is-tokenization-in-llms)
- **17.12** [Explain BPE (Byte Pair Encoding).](#1712-explain-bpe-byte-pair-encoding)
- **17.13** [Explain WordPiece and SentencePiece.](#1713-explain-wordpiece-and-sentencepiece)
- **17.14** [What is positional encoding, and why is it needed in Transformers?](#1714-what-is-positional-encoding-and-why-is-it-needed-in-transformers)
- **17.15** [What are embeddings?](#1715-what-are-embeddings)
- **17.16** [Explain the Query(Q), Key(K), and Value(V) in attention.](#1716-explain-the-queryq-keyk-and-valuev-in-attention)
- **17.17** [What is self-attention, and how does it work in Transformers?](#1717-what-is-self-attention-and-how-does-it-work-in-transformers)
- **17.18** [What is cross-attention in Transformers?](#1718-what-is-cross-attention-in-transformers)
- **17.19** [Why do we scale the dot product attention by √dₖ in the Transformer architecture?](#1719-why-do-we-scale-the-dot-product-attention-by-dₖ-in-the-transformer-architecture)
- **17.20** [What is causal masking?](#1720-what-is-causal-masking)
- **17.21** [What are multi-head attention mechanisms? Why use multiple attention heads?](#1721-what-are-multi-head-attention-mechanisms-why-use-multiple-attention-heads)
- **17.22** [What are Feed-Forward Networks in LLMs?](#1722-what-are-feed-forward-networks-in-llms)
- **17.23** [What is the context window in LLMs, and why does it matter?](#1723-what-is-the-context-window-in-llms-and-why-does-it-matter)
- **17.24** [Why is the context window limited in LLMs?](#1724-why-is-the-context-window-limited-in-llms)
- **17.25** [What is temperature in the context of LLMs, and how does it affect output?](#1725-what-is-temperature-in-the-context-of-llms-and-how-does-it-affect-output)
- **17.26** [Why is the first token slower than the rest in an LLM?](#1726-why-is-the-first-token-slower-than-the-rest-in-an-llm)
- **17.27** [Explain Top-p (nucleus) sampling and Top-k sampling. How do they differ?](#1727-explain-top-p-nucleus-sampling-and-top-k-sampling-how-do-they-differ)
- **17.28** [What are logits, and how are they used in text generation?](#1728-what-are-logits-and-how-are-they-used-in-text-generation)
- **17.29** [What are skip connections (residual connections) in Transformers?](#1729-what-are-skip-connections-residual-connections-in-transformers)
- **17.30** [What is the difference between open-source and closed-source LLMs? When would you choose one over the other?](#1730-what-is-the-difference-between-open-source-and-closed-source-llms-when-would-you-choose-one-over-the-other)
- **17.31** [What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?](#1731-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformer-architectures)
- **17.32** [What is KV cache, and how does it speed up inference?](#1732-what-is-kv-cache-and-how-does-it-speed-up-inference)
- **17.33** [What is model distillation, and how is it used with LLMs?](#1733-what-is-model-distillation-and-how-is-it-used-with-llms)
- **17.34** [What is Mixture of Experts (MoE), and how does it work in models like Mixtral?](#1734-what-is-mixture-of-experts-moe-and-how-does-it-work-in-models-like-mixtral)
- **17.35** [What is the difference between dense and sparse models?](#1735-what-is-the-difference-between-dense-and-sparse-models)
- **17.36** [What is Flash Attention?](#1736-what-is-flash-attention)
- **17.37** [What is Cross-Entropy Loss?](#1737-what-is-cross-entropy-loss)
- **17.38** [What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?](#1738-what-is-grouped-query-attention-gqa-and-how-does-it-differ-from-multi-head-attention-mha)
- **17.39** [How does Rotary Position Embedding (RoPE) work, and why is it preferred over learned positional embeddings?](#1739-how-does-rotary-position-embedding-rope-work-and-why-is-it-preferred-over-learned-positional-embeddings)
- **17.40** [Explain Layer Normalization](#1740-explain-layer-normalization)
- **17.41** [Explain RMSNorm (Root Mean Square Layer Normalization)](#1741-explain-rmsnorm-root-mean-square-layer-normalization)
- **17.42** [Your LLM keeps ignoring your instructions. How do you make it follow structured output formats?](#1742-your-llm-keeps-ignoring-your-instructions-how-do-you-make-it-follow-structured-output-formats)
- **17.43** [Your LLM-powered tool hits the context window limit on long documents. How do you handle it?](#1743-your-llm-powered-tool-hits-the-context-window-limit-on-long-documents-how-do-you-handle-it)
- **17.44** [Your LLM does not admit when it does not know the answer. How do you make it say "I don't know"?](#1744-your-llm-does-not-admit-when-it-does-not-know-the-answer-how-do-you-make-it-say-i-dont-know)
- **17.45** [Your LLM generates responses that are too verbose. How do you control response length?](#1745-your-llm-generates-responses-that-are-too-verbose-how-do-you-control-response-length)
- **17.46** [Your LLM memorized proprietary training data and leaks it in responses. How do you prevent this?](#1746-your-llm-memorized-proprietary-training-data-and-leaks-it-in-responses-how-do-you-prevent-this)
- **17.47** [Your LLM coding assistant generates outdated code using deprecated libraries. How do you fix it?](#1747-your-llm-coding-assistant-generates-outdated-code-using-deprecated-libraries-how-do-you-fix-it)
- **17.48** [Your tokenizer splits important domain terms into meaningless subword pieces. How do you fix it?](#1748-your-tokenizer-splits-important-domain-terms-into-meaningless-subword-pieces-how-do-you-fix-it)
- **17.49** [Your Transformer's KV cache grows too large during long sequence generation. How do you manage memory?](#1749-your-transformers-kv-cache-grows-too-large-during-long-sequence-generation-how-do-you-manage-memory)
- **17.50** [Your Transformer runs out of memory on long documents due to quadratic self-attention. How do you scale it?](#1750-your-transformer-runs-out-of-memory-on-long-documents-due-to-quadratic-self-attention-how-do-you-scale-it)
- **17.51** [Your distilled student model fails on the complex reasoning that the teacher model handled. How do you close the gap?](#1751-your-distilled-student-model-fails-on-the-complex-reasoning-that-the-teacher-model-handled-how-do-you-close-the-gap)
- **17.52** [After RLHF alignment, your LLM became safer but lost capability on hard tasks. How do you manage the alignment tax?](#1752-after-rlhf-alignment-your-llm-became-safer-but-lost-capability-on-hard-tasks-how-do-you-manage-the-alignment-tax)
- **17.53** [Your RLHF-trained LLM is gaming the reward model instead of being genuinely helpful. How do you fix reward hacking?](#1753-your-rlhf-trained-llm-is-gaming-the-reward-model-instead-of-being-genuinely-helpful-how-do-you-fix-reward-hacking)
- **17.54** [Your chatbot loses context after 10 turns in a conversation. How do you maintain a long conversation context?](#1754-your-chatbot-loses-context-after-10-turns-in-a-conversation-how-do-you-maintain-a-long-conversation-context)
- **17.55** [Your chatbot fails when users switch topics mid-conversation. How do you handle topic switches?](#1755-your-chatbot-fails-when-users-switch-topics-mid-conversation-how-do-you-handle-topic-switches)
- **17.56** [Your QA system always generates an answer even when no answer exists in the context. How do you detect unanswerable questions?](#1756-your-qa-system-always-generates-an-answer-even-when-no-answer-exists-in-the-context-how-do-you-detect-unanswerable-questions)
- **17.57** [Your summarization system hallucinated facts not in the original article. How do you fix it?](#1757-your-summarization-system-hallucinated-facts-not-in-the-original-article-how-do-you-fix-it)
- **17.58** [Your text generation repeats phrases in long outputs. How do you fix repetition?](#1758-your-text-generation-repeats-phrases-in-long-outputs-how-do-you-fix-repetition)
- **17.59** [Transformers work on text, so can they also understand images?](#1759-transformers-work-on-text-so-can-they-also-understand-images)
- **17.60** [What are Small Language Models (SLMs)?](#1760-what-are-small-language-models-slms)
- **17.61** [What are Large Reasoning Models (LRMs)?](#1761-what-are-large-reasoning-models-lrms)
- **17.62** [What are Autoregressive Models?](#1762-what-are-autoregressive-models)
- **17.63** [Explain the difference between autoregressive and masked language modeling.](#1763-explain-the-difference-between-autoregressive-and-masked-language-modeling)
- **17.64** [What is Proximal Policy Optimization (PPO)?](#1764-what-is-proximal-policy-optimization-ppo)
- **17.65** [What is Direct Preference Optimization (DPO)?](#1765-what-is-direct-preference-optimization-dpo)
- **17.66** [What is Group Relative Policy Optimization (GRPO)?](#1766-what-is-group-relative-policy-optimization-grpo)
- **17.67** [What are Recursive Language Models (RLMs)?](#1767-what-are-recursive-language-models-rlms)
- **17.68** [What is continual learning in LLMs?](#1768-what-is-continual-learning-in-llms)
- **17.69** [How do Diffusion Language Models (DLMs) work?](#1769-how-do-diffusion-language-models-dlms-work)
- **17.70** [Why layer normalization in Transformers?](#1770-why-layer-normalization-in-transformers)
- **17.71** [Layer normalization vs. Batch Normalization?](#1771-layer-normalization-vs-batch-normalization)
- **17.72** [What is SVD (singular value decomposition)?](#1772-what-is-svd-singular-value-decomposition)
- **17.73** [What is KL Divergence?](#1773-what-is-kl-divergence)
- **17.74** [What is entropy vs. cross-entropy?](#1774-what-is-entropy-vs-cross-entropy)
- **17.75** [L1 vs L2 Geometry intuition?](#1775-l1-vs-l2-geometry-intuition)
- **17.76** [Multi Query Attention?](#1776-multi-query-attention)
- **17.77** [Multi-Head Attention vs Multi-Query Attention vs Grouped-Query Attention?](#1777-multi-head-attention-vs-multi-query-attention-vs-grouped-query-attention)
- **17.78** [What is reinforcement learning?](#1778-what-is-reinforcement-learning)
- **17.79** [What is the difference between an LM cache and a KV cache?](#1779-what-is-the-difference-between-an-lm-cache-and-a-kv-cache)
- **17.80** [How does dropout behave during inference?](#1780-how-does-dropout-behave-during-inference)
- **17.81** [What is a Large Language Model (LLM)?](#1781-what-is-a-large-language-model-llm)
- **17.82** [What does autoregressive mean?](#1782-what-does-autoregressive-mean)
- **17.83** [What is a Large Language Model (LLM), and how does it work?](#1783-what-is-a-large-language-model-llm-and-how-does-it-work)
- **17.84** [What Does Each Transformer Block Learn?](#1784-what-does-each-transformer-block-learn)
- **17.85** [What is the KV cache in LLMs?](#1785-what-is-the-kv-cache-in-llms)
- **17.86** [What is PagedAttention in LLMs?](#1786-what-is-pagedattention-in-llms)
- **17.87** [What is speculative decoding, and how does it speed up inference?](#1787-what-is-speculative-decoding-and-how-does-it-speed-up-inference)
- **17.88** [How does continuous batching improve LLM inference throughput?](#1788-how-does-continuous-batching-improve-llm-inference-throughput)
- **17.89** [What is pre-training vs fine-tuning in LLMs?](#1789-what-is-pre-training-vs-fine-tuning-in-llms)
- **17.90** [What are some challenges in training LLMs?](#1790-what-are-some-challenges-in-training-llms)
- **17.91** [What is zero-shot learning in the context of LLMs?](#1791-what-is-zero-shot-learning-in-the-context-of-llms)
- **17.92** [How do you handle bias and fairness in LLMs?](#1792-how-do-you-handle-bias-and-fairness-in-llms)
- **17.93** [What are some real-world applications of LLMs in business and tech?](#1793-what-are-some-real-world-applications-of-llms-in-business-and-tech)
- **17.94** [How does the Transformer architecture improve LLM performance over RNNs?](#1794-how-does-the-transformer-architecture-improve-llm-performance-over-rnns)
- **17.95** [How does attention help capture long-range dependencies?](#1795-how-does-attention-help-capture-long-range-dependencies)
- **17.96** [How does tokenization work in Large Language Models?](#1796-how-does-tokenization-work-in-large-language-models)
- **17.97** [What is subword tokenization?](#1797-what-is-subword-tokenization)
- **17.98** [What is BPE (Byte Pair Encoding) in LLMs?](#1798-what-is-bpe-byte-pair-encoding-in-llms)
- **17.99** [What is positional embedding in LLMs?](#1799-what-is-positional-embedding-in-llms)
- **17.100** [What is temperature in the context of LLMs?](#17100-what-is-temperature-in-the-context-of-llms)
- **17.101** [What are skip connections?](#17101-what-are-skip-connections)
- **17.102** [What is normalization?](#17102-what-is-normalization)
- **17.103** [What is dropout, and how is it applied in LLMs?](#17103-what-is-dropout-and-how-is-it-applied-in-llms)
- **17.104** [Why does Attention use Softmax?](#17104-why-does-attention-use-softmax)
- **17.105** [What does a vector database (Vector DB) store for LLM usage?](#17105-what-does-a-vector-database-vector-db-store-for-llm-usage)
- **17.106** [How do you improve inference speed in production LLM deployments?](#17106-how-do-you-improve-inference-speed-in-production-llm-deployments)
- **17.107** [What is the context window in an LLM?](#17107-what-is-the-context-window-in-an-llm)
- **17.108** [Explain Prompting, Retrieval-Augmented Generation (RAG), and Fine-Tuning.](#17108-explain-prompting-retrieval-augmented-generation-rag-and-fine-tuning)
- **17.109** [What is Reinforcement Learning from Human Feedback (RLHF)?](#17109-what-is-reinforcement-learning-from-human-feedback-rlhf)
- **17.110** [How does Knowledge Distillation work?](#17110-how-does-knowledge-distillation-work)
- **17.111** [What is instruction tuning, and why is it important for chat models?](#17111-what-is-instruction-tuning-and-why-is-it-important-for-chat-models)
- **17.112** [What is the difference between prefill and decode?](#17112-what-is-the-difference-between-prefill-and-decode)
- **17.113** [How does Sliding Window Attention work?](#17113-how-does-sliding-window-attention-work)
- **17.114** [How do Attention Sinks work?](#17114-how-do-attention-sinks-work)
- **17.115** [What is next-concept modeling (NCM), and how does it differ from an LLM?](#17115-what-is-next-concept-modeling-ncm-and-how-does-it-differ-from-an-llm)
- **17.116** [How does BPE tokenization work?](#17116-how-does-bpe-tokenization-work)
- **17.117** [How do you implement a tokenizer (training, encoding, decoding, special tokens)?](#17117-how-do-you-implement-a-tokenizer-training-encoding-decoding-special-tokens)
- **17.118** [What does the LM head do, and what is weight tying?](#17118-what-does-the-lm-head-do-and-what-is-weight-tying)
- **17.119** [Explain self-attention mathematically.](#17119-explain-self-attention-mathematically)
- **17.120** [Why divide by $\sqrt{d_k}$?](#17120-why-divide-by-sqrtd_k)
- **17.121** [What is the difference between self-attention and cross-attention?](#17121-what-is-the-difference-between-self-attention-and-cross-attention)
- **17.122** [Why do Transformers need positional information?](#17122-why-do-transformers-need-positional-information)
- **17.123** [Compare absolute positional embeddings, RoPE, ALiBi, and relative positional encoding.](#17123-compare-absolute-positional-embeddings-rope-alibi-and-relative-positional-encoding)
- **17.124** [Why does self-attention have $O(n^2)$ complexity?](#17124-why-does-self-attention-have-on2-complexity)
- **17.125** [What techniques reduce Transformer attention complexity?](#17125-what-techniques-reduce-transformer-attention-complexity)
- **17.126** [Explain Multi-Head Attention.](#17126-explain-multi-head-attention)
- **17.127** [Why are multiple attention heads useful?](#17127-why-are-multiple-attention-heads-useful)
- **17.128** [What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?](#17128-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformers)
- **17.129** [Why are decoder-only architectures dominant for modern LLMs?](#17129-why-are-decoder-only-architectures-dominant-for-modern-llms)
- **17.130** [What is the difference between pretraining, instruction tuning, and alignment?](#17130-what-is-the-difference-between-pretraining-instruction-tuning-and-alignment)
- **17.131** [Explain supervised fine-tuning (SFT).](#17131-explain-supervised-fine-tuning-sft)
- **17.132** [What is model collapse?](#17132-what-is-model-collapse)
- **17.133** [What causes LLM hallucinations?](#17133-what-causes-llm-hallucinations)
- **17.134** [Can hallucinations be completely eliminated?](#17134-can-hallucinations-be-completely-eliminated)
- **17.135** [What is the relationship between model scaling and emergent capabilities?](#17135-what-is-the-relationship-between-model-scaling-and-emergent-capabilities)
- **17.136** [Explain the scaling laws for LLMs.](#17136-explain-the-scaling-laws-for-llms)
- **17.137** [What is the Chinchilla scaling principle?](#17137-what-is-the-chinchilla-scaling-principle)
- **17.138** [Why can a smaller model outperform a larger model?](#17138-why-can-a-smaller-model-outperform-a-larger-model)
- **17.139** [What is mixture-of-experts (MoE)?](#17139-what-is-mixture-of-experts-moe)
- **17.140** [Why does MoE allow larger parameter counts without proportionally increasing compute?](#17140-why-does-moe-allow-larger-parameter-counts-without-proportionally-increasing-compute)
- **17.141** [Explain expert routing and load balancing in MoE.](#17141-explain-expert-routing-and-load-balancing-in-moe)
- **17.142** [What is the difference between System 1 and System 2 thinking?](#17142-what-is-the-difference-between-system-1-and-system-2-thinking)

---

### 17.1 What is an LLM?

- **LLM stands for Large Language Model.** 
- **It is a deep learning model trained on a very large corpus of text to understand and generate human language.** 
- **Most modern LLMs are based on the Transformer architecture and use self-attention to understand the relationship between tokens in a sequence.**
- **An LLM learns to predict the next token based on the previous context.** 
- **Through large-scale pretraining and subsequent fine-tuning or instruction tuning, it can perform tasks such as question answering, summarization, code generation, reasoning, and information extraction.**
- **LLMs are often combined with techniques such as RAG, tool calling, agents, fine-tuning, embeddings, and vector databases to build reliable AI applications.**

### 17.2 What is RAG?

- **RAG stands for Retrieval-Augmented Generation.** 
- **It is an architecture that combines information retrieval with an LLM.** 
- **Instead of relying only on the knowledge learned during the model's training, RAG retrieves relevant information from an external knowledge source and provides that context to the LLM to generate an answer.** 
- **Typically, documents are split into chunks, converted into embeddings, and stored in a vector database.** 
- **When a user asks a question, the query is converted into an embedding and used to retrieve the most relevant chunks.**
- **Those chunks are then provided as context to the LLM, which generates the final response.** 
- **RAG is useful for building applications over private, domain-specific, or frequently changing data without necessarily fine-tuning the LLM.** 
- **It can also improve factual grounding and allow responses to be supported by retrieved sources.**

### 17.3 What is MCP?

- **MCP stands for Model Context Protocol.** 
- **It is an open protocol that standardizes how AI applications, particularly LLM-based agents, connect to external tools, data sources, and services.**
- **Instead of building a custom integration for every LLM application and every external system, MCP provides a standardized interface through which an AI application can discover and interact with capabilities exposed by an MCP server.**
- **An MCP server can expose things such as tools, resources, and prompts. For example, an MCP server could provide an AI assistant with tools to query a database, access documents from a knowledge base, interact with GitHub, or perform operations in an internal business system.**
- **The key benefit is decoupling the AI application from individual integrations, making tool and data access more standardized and reusable.**
- LLM/Application → MCP Client → MCP Server → External System

### 17.4 What is an AI agent?

- **An AI agent is a system that uses an AI model, typically an LLM, to understand a goal, reason about the steps required to achieve that goal, use available tools or external systems, observe the results, and take further actions until the task is completed.**
- **Unlike a traditional LLM application, where we usually define a fixed sequence of steps, an agent can dynamically decide which tool to use and what action to take based on the current state and the result of previous actions.**
- **For example, a customer-support agent might receive a user's request, search the knowledge base, check an order database, call an API to retrieve the order status, and then generate a response based on the retrieved information.**
- **In production, an agent typically consists of an LLM, instructions, tools, state or memory, an orchestration layer, and mechanisms for controlling and evaluating its actions.**

### 17.5 What is Fine-tuning?

> **See also:** [Question 22.27 — What is fine-tuning?](22-fine-tuning-and-model-adaptation.md#2227-what-is-fine-tuning), which covers the same question from another angle.

- **Fine-tuning is the process of taking a pretrained model and further training it on a smaller, task-specific or domain-specific dataset so that the model becomes better adapted to a particular task, behavior, or domain.**
- **Instead of training a model from scratch, we start with the knowledge and representations learned during pretraining and update some or all of the model's parameters using the fine-tuning dataset.**
- **For LLMs, fine-tuning can be used for things like instruction following, classification, structured output, domain-specific language, or adapting the model's behavior and style. Depending on the use case and resources, we can use full fine-tuning or parameter-efficient methods such as LoRA and QLoRA.**

### 17.6 What is Quantization?

> **See also:** [Question 27.26 — What is quantization?](../part-5-production-systems/27-ai-infrastructure-and-inference-optimization.md#2726-what-is-quantization), which covers the same question from another angle.

- **Quantization is the process of representing a model's weights and sometimes activations using lower-precision numerical formats, such as INT8 or INT4, instead of higher-precision formats like FP32 or FP16.**
- **The main goal is to reduce model memory usage, storage requirements, and inference latency, while maintaining as much model accuracy as possible. This is particularly useful for deploying large language models on GPUs or resource-constrained environments.**
- **For example, converting a model from FP16 to INT8 reduces the number of bits used to represent each value from 16 to 8, which can significantly reduce memory requirements. However, quantization can introduce some loss of accuracy, so the appropriate quantization method and precision need to be selected based on the application.**

### 17.7 What are foundation models, and how have they changed AI engineering?

- **Foundation models are large, pretrained AI models trained on broad and diverse datasets at scale.** 
- **They learn general-purpose representations and capabilities that can then be adapted to many downstream tasks through prompting, fine-tuning, RAG, tool use, or other techniques. Examples include large language models, vision-language models, and multimodal models.**
- **Foundation models have significantly changed AI engineering because we no longer need to build and train a separate model from scratch for every application. Instead, engineers can build applications on top of pretrained models and focus more on system architecture, data pipelines, retrieval, evaluation, orchestration, security, and deployment.**
- **For example, instead of training an NLP model from scratch for a company's internal knowledge base, we can use a foundation model together with RAG, embeddings, a vector database, and appropriate evaluation to build a domain-specific application much faster.**
- **They have also shifted AI engineering from primarily model development toward building reliable AI systems around models. This has introduced new engineering challenges such as prompt engineering, model selection, context management, hallucination and evaluation, latency, inference cost, observability, security, and responsible AI.**

### 17.8 Inside ChatGPT: What Happens After You Hit Enter?

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

### 17.9 What is the Transformer architecture, and how does it work?

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

### 17.10 What are the key components of the Transformer architecture?

1. **Token Embeddings** — Convert tokens into vectors.
2. **Positional Encoding** — Adds token position information.
3. **Multi-Head Self-Attention** — Captures relationships between tokens using **Q, K, V**.
4. **Feed-Forward Network (FFN)** — Transforms the attention output.
5. **Residual Connections** — Improve information flow and training.
6. **Layer Normalization** — Stabilizes the network.
7. **Output Layer** — Produces probabilities for **next-token prediction**.

**Transformer Block:**

**Embedding → Position → Self-Attention → FFN → Residual + Norm → Repeat → Output**

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

### 17.11 What is tokenization in LLMs?

#### Tokenization in LLMs

- **Tokenization** converts text into smaller units called **tokens** that an LLM can process.
- Tokens can be **words, subwords, characters, punctuation, or special tokens**.
- Modern LLMs commonly use **subword tokenization**, such as **BPE, WordPiece, or Unigram**.
- Each token is mapped to a **token ID**, which is then converted into an **embedding**.

**Flow:**

**Text → Tokens → Token IDs → Embeddings → Transformer**

**Interview:**

> Tokenization is the process of converting text into tokens and mapping them to numerical IDs so the LLM can process the input.

### 17.12 Explain BPE (Byte Pair Encoding).

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

### 17.13 Explain WordPiece and SentencePiece.

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

### 17.14 What is positional encoding, and why is it needed in Transformers?

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

### 17.15 What are embeddings?

> **See also:** [Question 8.24 — What are embeddings in Machine Learning?](../part-3-machine-learning/08-machine-learning-fundamentals.md#824-what-are-embeddings-in-machine-learning), which covers the same question from another angle.

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

### 17.16 Explain the Query(Q), Key(K), and Value(V) in attention.

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

### 17.17 What is self-attention, and how does it work in Transformers?

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

### 17.18 What is cross-attention in Transformers?

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

#### Notes from a repeated entry: "Cross Attention?"

**Cross-Attention** allows one sequence to attend to another sequence.

The key idea is:

> **Queries come from one sequence, while Keys and Values come from another sequence.**

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

### 17.19 Why do we scale the dot product attention by √dₖ in the Transformer architecture?

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

### 17.20 What is causal masking?

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

### 17.21 What are multi-head attention mechanisms? Why use multiple attention heads?

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

#### Notes from a repeated entry: "Multi Head Attention?"

**Multi-Head Attention (MHA)** allows the model to perform attention from multiple different representation subspaces simultaneously.

The key idea is:

> **Each attention head has its own Query, Key, and Value projections.**

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

### 17.22 What are Feed-Forward Networks in LLMs?

- **Feed-Forward Network (FFN)** is the **non-linear transformation layer** inside each Transformer block.
- It processes each token **independently** after self-attention.
- Typical structure: FFN(x)=W2σ(W1x+b1)+b2
- It usually **expands → activates → compresses** the hidden representation.
- **Attention:** mixes information **between tokens**.
- **FFN:** transforms and enriches **each token's representation**.
- Modern LLMs often use **gated FFNs such as SwiGLU** for better performance.

**Interview answer:**

> “FFN is the non-linear component of a Transformer. Attention mixes information across tokens, while the FFN independently transforms each token representation to learn richer features.”

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

### 17.23 What is the context window in LLMs, and why does it matter?

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

### 17.24 Why is the context window limited in LLMs?

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

The answer above is the same as Q23 (what the context window is and why it matters). It does not explain why the window is limited. The main reasons are:

- **Attention cost grows quadratically.** Standard self-attention scores every token against every other token, so compute and attention memory scale as O(n²) with sequence length. Doubling the context roughly quadruples attention work during prefill.
- **The KV cache grows linearly with length.** The model keeps K and V tensors for every previous token in every layer. At long contexts and large batch sizes, the KV cache (not the weights) often dominates GPU memory and limits how many concurrent requests a server can handle.
- **Training length and positional encoding.** Models are trained on sequences up to a maximum length. Positional schemes such as RoPE do not extrapolate reliably far beyond it without scaling techniques (position interpolation, YaRN) plus long-context fine-tuning.
- **Data and training cost.** High-quality long documents are scarcer, and training on long sequences is expensive.
- **Effective vs. nominal context.** Even within the limit, quality degrades for information buried in the middle ("lost in the middle"), so providers cap the window where quality, latency and cost stay acceptable.

**Production angle:** model-level mitigations include FlashAttention, GQA/MQA, KV-cache quantization and paging, and sliding-window attention. System-level approaches (RAG, chunking, summarization/compaction) are usually cheaper than relying on an ever-larger raw window.

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

### 17.25 What is temperature in the context of LLMs, and how does it affect output?

- **Temperature** is a decoding parameter that controls the **randomness of an LLM’s output** by modifying the probability distribution over the next token.
- **Low temperature (e.g., 0–0.3):** More deterministic and focused → useful for factual answers, extraction, and structured outputs.
- **Medium temperature (e.g., 0.5–0.8):** Balanced between consistency and diversity.
- **High temperature (e.g., 1.0+):** More diverse and unpredictable → useful for creative generation.
- Temperature **does not change the model’s knowledge or weights**; it changes how tokens are sampled during generation.

**Interview answer:**

> “Temperature controls the randomness of token sampling. Lower temperatures make the output more deterministic and focused, while higher temperatures increase diversity and randomness. It affects the decoding process, not the model’s underlying knowledge.”

### 17.26 Why is the first token slower than the rest in an LLM?

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

### 17.27 Explain Top-p (nucleus) sampling and Top-k sampling. How do they differ?

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

### 17.28 What are logits, and how are they used in text generation?

- **Logits** are the raw, unnormalized scores produced by the LLM for every possible next token.
- The model converts the final hidden representation into a **logit for each token** in the vocabulary.
- **Softmax** converts logits into probabilities:

P(token\_i) = e\^(z\_i) / Σ\_j e\^(z\_j)

- The next token is then selected using **greedy decoding or sampling** (e.g., temperature, Top-k, Top-p).

**Interview answer:**

> “Logits are the raw scores the LLM produces for each possible next token. They are converted into probabilities using softmax, and the decoding strategy then selects the next token based on those probabilities.”

### 17.29 What are skip connections (residual connections) in Transformers?

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

### 17.30 What is the difference between open-source and closed-source LLMs? When would you choose one over the other?

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

### 17.31 What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?

> **See also:** [Question 13.20 — What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?](../part-3-machine-learning/13-deep-learning.md#1320-what-is-the-difference-between-encoder-only-decoder-only-and-encoder-decoder-transformer-architectures), which covers the same question from another angle.

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

### 17.32 What is KV cache, and how does it speed up inference?

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

### 17.33 What is model distillation, and how is it used with LLMs?

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

### 17.34 What is Mixture of Experts (MoE), and how does it work in models like Mixtral?

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

### 17.35 What is the difference between dense and sparse models?

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

### 17.36 What is Flash Attention?

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

### 17.37 What is Cross-Entropy Loss?

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

### 17.38 What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?

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

#### Notes from a repeated entry: "Grouped-Query Attention?"

**Grouped-Query Attention (GQA)** is an attention mechanism that sits between **Multi-Head Attention (MHA)** and **Multi-Query Attention (MQA)**.

The key idea is:

> **Multiple query heads share the same Key and Value heads.**

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

### 17.39 How does Rotary Position Embedding (RoPE) work, and why is it preferred over learned positional embeddings?

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

### 17.40 Explain Layer Normalization

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

### 17.41 Explain RMSNorm (Root Mean Square Layer Normalization)

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

### 17.42 Your LLM keeps ignoring your instructions. How do you make it follow structured output formats?

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

### 17.43 Your LLM-powered tool hits the context window limit on long documents. How do you handle it?

#### Handling Context Window Limits

- **Chunk the document** into smaller, meaningful sections rather than sending the entire document to the LLM.
- Use **RAG**: retrieve only the most relevant chunks for the current query.
- Use **hierarchical summarization** for tasks requiring information from the entire document.
- **Compress/summarize conversation history** instead of passing the full history every time.
- Use **reranking** to select the most relevant chunks before sending them to the LLM.
- If appropriate, use a **larger-context model**, but don't rely on context size alone.

**Interview answer:**

> **“I wouldn't simply increase the context window. I would typically chunk the document, retrieve the most relevant chunks using RAG, rerank them, and pass only the necessary context to the LLM. For tasks requiring global understanding, I would use hierarchical or map-reduce summarization. I would also manage conversation history through summarization or compression. The goal is to provide the model with the most relevant information while controlling latency and cost.”**

### 17.44 Your LLM does not admit when it does not know the answer. How do you make it say "I don't know"?

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

### 17.45 Your LLM generates responses that are too verbose. How do you control response length?

- **Prompt constraints:** Specify the desired length, e.g., “Answer in 3–5 bullet points” or “under 100 words.”
- **Token limits:** Use `max_output_tokens` to enforce a hard upper limit.
- **Structured output:** Use JSON/schema constraints for predictable responses.
- **Context control:** In RAG, limit the amount of retrieved context to reduce unnecessary verbosity.
- **Validation:** Check output length and regenerate or summarize if it exceeds the limit.

### 17.46 Your LLM memorized proprietary training data and leaks it in responses. How do you prevent this?

- **Data governance:** Don’t include sensitive/proprietary data in training unless explicitly authorized.
- **Data sanitization:** Remove PII, secrets, credentials, and confidential information before training.
- **Access control:** Restrict which users and applications can access sensitive data.
- **Privacy techniques:** Use data minimization, anonymization, and, where appropriate, differential privacy.
- **Output filtering:** Detect and block sensitive information in model responses.
- **Testing:** Perform privacy/red-team tests to identify memorization and data-extraction risks.
- **RAG:** Keep proprietary data in controlled databases and retrieve it only for authorized users instead of embedding it directly into model weights.

### 17.47 Your LLM coding assistant generates outdated code using deprecated libraries. How do you fix it?

- **Use up-to-date documentation:** Ground the LLM with current library/API documentation.
- **Version-aware prompting:** Explicitly provide the required library and version.
- **RAG:** Retrieve the latest official documentation before generating code.
- **Tool/API access:** Let the assistant verify APIs and package versions when possible.
- **Validation:** Run generated code through tests, linters, and type checkers.
- **Continuous updates:** Regularly update prompts, RAG indexes, and evaluation datasets.

### 17.48 Your tokenizer splits important domain terms into meaningless subword pieces. How do you fix it?

- **Use a domain-specific tokenizer:** Train or adapt the tokenizer on domain-specific data.
- **Add special tokens:** Add frequently used domain terms as vocabulary tokens.
- **Vocabulary expansion:** Extend the tokenizer vocabulary for important terms.
- **Compare tokenization:** Evaluate tokenization before/after on representative domain data.
- **Fine-tune the model:** Fine-tune the model with domain-specific data after tokenizer changes.
- **Alternative:** Use a model/tokenizer already trained for the target domain.

### 17.49 Your Transformer's KV cache grows too large during long sequence generation. How do you manage memory?

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

### 17.50 Your Transformer runs out of memory on long documents due to quadratic self-attention. How do you scale it?

- **Use sparse attention:** Replace full self-attention with **Sliding Window Attention**, local attention, or block-sparse attention so each token attends only to a limited number of nearby tokens.
- **Use efficient attention mechanisms:** Techniques such as **FlashAttention** reduce memory usage and improve GPU efficiency without changing the attention pattern.
- **Use linear attention:** Choose attention variants whose computational/memory cost grows approximately **linearly with sequence length** rather than quadratically.
- **Use hierarchical/chunked processing:** Split long documents into chunks, process each chunk separately, and then combine the chunk-level representations.
- **Use retrieval-based approaches:** Instead of putting the entire document into the context window, store document chunks in a **vector database** and retrieve only the most relevant chunks.
- **Use long-context architectures:** Models such as those using **Longformer-style sparse attention** or other long-context mechanisms are designed specifically for long sequences.
- **Use KV-cache efficiently:** For autoregressive inference, techniques such as **KV-cache quantization, compression, or paged KV caching** can significantly reduce memory consumption.
- **Practical approach:** For an LLM/RAG system, I would typically use **chunking + embeddings + retrieval + reranking**, while using **FlashAttention or sparse attention** when the model itself needs to process very long sequences.

### 17.51 Your distilled student model fails on the complex reasoning that the teacher model handled. How do you close the gap?

- **Use knowledge distillation:** Train the student not only on the teacher's final answers but also on the teacher's **soft probabilities/logits**, which contain richer information about alternative answers.
- **Distill reasoning behavior:** Generate high-quality teacher explanations, intermediate reasoning steps, or structured solution traces and train the student on them. For safety and reliability, the student can learn **concise rationales or intermediate results** rather than blindly copying hidden chain-of-thought.
- **Use a reasoning-focused dataset:** Create more training examples involving **multi-step reasoning, mathematics, coding, planning, and difficult edge cases** rather than mostly simple examples.
- **Curriculum distillation:** Start with simpler problems and progressively introduce harder reasoning tasks.
- **Increase student capacity:** If the student is too small, increase its parameter count or use a stronger architecture. Some reasoning capability simply cannot be compressed beyond a certain point without loss.
- **Use intermediate-layer distillation:** Match selected **hidden representations or attention patterns** between teacher and student, not just the final output.
- **Use teacher-generated data:** Have the teacher generate additional difficult examples, solutions, and counterexamples, then filter them for quality before training the student.
- **Use preference/verification training:** Train the student to prefer correct solutions and use a verifier or reward model to reject incorrect reasoning.
- **Evaluate by capability:** Test separately on **simple knowledge, multi-step reasoning, mathematics, coding, and out-of-distribution problems** to identify exactly where the student loses capability.

### 17.52 After RLHF alignment, your LLM became safer but lost capability on hard tasks. How do you manage the alignment tax?

- **Measure the alignment tax:** Evaluate the base model and aligned model separately on reasoning, coding, math, knowledge, safety, and instruction-following benchmarks to identify where capability was lost.
- **Use a capability-preserving objective:** Don't optimize only for the reward model. Combine the alignment objective with a **capability/behavior-preservation loss** so the model retains useful pretrained behavior.
- **Use KL regularization:** Keep the aligned model close to the capable reference model where there is no safety reason to change its behavior.
- **Improve the preference data:** RLHF can over-optimize noisy or simplistic preferences. Use high-quality, difficult examples and include **hard reasoning tasks** in the preference dataset.
- **Use targeted alignment:** Apply stronger safety constraints specifically where needed rather than globally suppressing behaviors that are actually useful.
- **Separate safety from capability:** Use classifiers, policy models, or guardrails around the LLM where appropriate, instead of forcing the core model to become overly conservative.
- **Use continued capability training:** After alignment, periodically train on math, coding, reasoning, and domain-specific data to recover capabilities while maintaining safety constraints.
- **Use better preference optimization:** Methods such as **DPO, IPO, or other preference-optimization approaches** can sometimes provide more stable alignment than aggressive online RL.
- **Track Pareto trade-offs:** Maintain separate safety and capability metrics and choose the training checkpoint that provides the best **safety–capability trade-off**, rather than maximizing safety alone.

### 17.53 Your RLHF-trained LLM is gaming the reward model instead of being genuinely helpful. How do you fix reward hacking?

- **Improve the reward model:** Train it on diverse, high-quality human preferences, especially examples where superficial answers look good but are actually incorrect or unhelpful.
- **Use multiple reward signals:** Don't rely on a single reward model. Combine signals for **correctness, helpfulness, safety, factuality, and instruction-following**.
- **Use adversarial evaluation:** Deliberately search for cases where the model can get high reward through shortcuts, verbosity, manipulation, or other unintended behavior.
- **Use reward-model ensembles:** Multiple independently trained reward models make it harder for the policy to exploit a specific weakness in one reward model.
- **Regularize the policy:** Use a **KL penalty** to prevent the RL-trained model from moving too far from the reference model merely to maximize reward.
- **Improve reward-model robustness:** Continuously retrain the reward model using examples generated from the policy's newly discovered failure modes.
- **Use verifiable rewards:** For tasks such as mathematics and coding, use **objective verification**—for example, unit tests or exact answers—instead of relying entirely on human preference scores.
- **Monitor reward vs. actual performance:** If reward keeps increasing while human evaluations or task accuracy decrease, that's a strong signal of reward hacking.
- **Keep humans in the loop:** Periodically audit high-reward outputs and add problematic examples back into the preference dataset.

### 17.54 Your chatbot loses context after 10 turns in a conversation. How do you maintain a long conversation context?

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

### 17.55 Your chatbot fails when users switch topics mid-conversation. How do you handle topic switches?

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

### 17.56 Your QA system always generates an answer even when no answer exists in the context. How do you detect unanswerable questions?

- **Check retrieval quality first:** If the retrieved chunks do not contain evidence relevant to the question, the system should treat the question as potentially unanswerable.
- **Use similarity thresholds:** Set a minimum retrieval similarity score. If all retrieved documents fall below the threshold, return **“I don't know based on the provided context.”**
- **Use an entailment/grounding check:** Ask a separate model to determine whether the retrieved context actually **supports the proposed answer**.
- **Require evidence:** Make the LLM identify the specific passages that support its answer. If it cannot find supporting evidence, reject the answer.
- **Use a confidence threshold:** Combine retrieval confidence, answer confidence, and grounding scores. Only answer when the combined confidence exceeds a predefined threshold.
- **Use an abstention classifier:** Train a classifier on **answerable vs. unanswerable** questions so the system can explicitly choose to abstain.
- **Use negative examples:** Include questions whose answers are deliberately absent from the knowledge base during evaluation and training.
- **Prevent outside-knowledge leakage:** In strict RAG QA, instruct the model to answer **only from retrieved context**, rather than using its pretrained knowledge.
- **Evaluate separately:** Measure **answer accuracy, groundedness, and abstention accuracy** rather than only measuring whether the chatbot produces an answer.

### 17.57 Your summarization system hallucinated facts not in the original article. How do you fix it?

- **Use extractive or grounded summarization:** Restrict the model to information explicitly present in the source article.
- **Add source grounding:** Require each important claim in the summary to be traceable to a specific sentence or passage in the original article.
- **Use a two-stage pipeline:** First extract the key facts from the article, then generate the summary only from those extracted facts.
- **Use factuality verification:** Run the generated summary through a separate verifier that checks whether each claim is supported by the source.
- **Use entailment checking:** Compare each summary statement against the article and reject or rewrite statements that are not entailed by the source.
- **Reduce decoding creativity:** Use lower temperature and constrained generation when factual accuracy is more important than stylistic variation.
- **Handle uncertainty explicitly:** If the article does not contain enough information, the system should omit the claim rather than infer or invent it.
- **Evaluate hallucination rate:** Build a test set containing difficult articles and measure **faithfulness, factual consistency, and unsupported-claim rate**.

### 17.58 Your text generation repeats phrases in long outputs. How do you fix repetition?

- **Use repetition penalties:** Apply `repetition_penalty`, `frequency_penalty`, or `presence_penalty` during decoding to discourage repeating tokens and phrases.
- **Use n-gram blocking:** Prevent the model from generating the same n-gram repeatedly, such as blocking repeated 3-grams or 4-grams.
- **Tune decoding:** Adjust **temperature, top-k, and top-p** to improve generation diversity without making the output incoherent.
- **Improve the prompt:** Give the model clear instructions about structure, avoiding repeated points, and maintaining progression.
- **Use coverage mechanisms:** Track what has already been generated and encourage the model to focus on new information.
- **Use hierarchical generation:** For very long outputs, first generate an outline, then generate each section while maintaining a record of previously covered topics.
- **Control context:** Excessive or poorly organized context can cause the model to revisit the same information. Use relevant context rather than dumping the entire history.
- **Fine-tune on quality data:** If repetition is systematic, fine-tune with high-quality examples that demonstrate diverse, non-repetitive long-form generation.
- **Add a post-processing check:** Detect repeated sentences or phrases and regenerate the affected section.

### 17.59 Transformers work on text, so can they also understand images?

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

### 17.60 What are Small Language Models (SLMs)?

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

### 17.61 What are Large Reasoning Models (LRMs)?

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

### 17.62 What are Autoregressive Models?

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

### 17.63 Explain the difference between autoregressive and masked language modeling.

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

### 17.64 What is Proximal Policy Optimization (PPO)?

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

##### Proximal Policy Optimization (PPO)

- **PPO** is a **policy-gradient reinforcement learning algorithm** used to optimize an agent’s policy while preventing excessively large policy updates.
- It improves training **stability and sample efficiency** by constraining how much the new policy can differ from the old policy.
- PPO uses a **clipped objective function** that limits the policy probability ratio to a small range, typically **\[1 − ε, 1 + ε\]**.
- It uses **advantage estimates** to determine whether an action performed better or worse than expected.
- PPO is **on-policy**, meaning it learns primarily from data collected using the current policy.
- It is widely used in **continuous-control tasks, robotics, games, and RLHF for language models**.
- Compared with **TRPO**, PPO is simpler to implement while providing a similar mechanism for preventing unstable policy updates.

### 17.65 What is Direct Preference Optimization (DPO)?

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

##### Direct Preference Optimization (DPO)

- **DPO** is a **preference-based fine-tuning method** that directly trains an LLM to prefer desired responses over undesired responses.
- It avoids the need to explicitly train a separate **reward model** and avoids the complex RL optimization used in methods such as PPO.
- DPO uses preference pairs: **chosen response** vs. **rejected response**.
- The model is trained to increase the probability of the **chosen response** while decreasing the probability of the **rejected response**, relative to a reference model.
- DPO is commonly used for **LLM alignment, instruction following, helpfulness, safety, and conversational quality**.
- Compared with **RLHF + PPO**, DPO is generally **simpler, more stable, and easier to implement**, but its performance depends heavily on the quality and diversity of preference data.

### 17.66 What is Group Relative Policy Optimization (GRPO)?

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

##### Group Relative Policy Optimization (GRPO)

- **GRPO** is a **policy optimization algorithm** designed to improve LLM reasoning using **relative rewards** among multiple responses generated for the same prompt.
- Instead of training a separate **value/critic model**, GRPO compares a **group of sampled responses** and uses their relative rewards to estimate advantages.
- For each prompt, the model generates multiple responses and evaluates them using a **reward function**.
- Responses that receive higher rewards are reinforced, while lower-reward responses are discouraged.
- This reduces memory and computational requirements compared with methods that require a separate critic/value model, such as PPO.
- GRPO became particularly notable for **reasoning-focused LLM training**, where rewards can come from verifiable signals such as **mathematical correctness or code execution**.
- Compared with **PPO**, GRPO is simpler for reasoning tasks because it eliminates the separate value model and derives advantages from **group-relative rewards**.

### 17.67 What are Recursive Language Models (RLMs)?

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

##### Recursive Language Models (RLMs)

- **RLMs** are language-model systems that handle **very large or complex inputs by recursively decomposing and processing them** rather than requiring the entire input to fit directly into one model context.
- The model can **break a large problem into smaller subproblems**, solve them separately, and combine the results.
- Intermediate results can be fed back into the model through **recursive calls**, enabling deeper reasoning over large inputs.
- RLMs can help with **long-document analysis, complex reasoning, codebases, and tasks that exceed the model’s practical context window**.
- They can reduce the need to process massive contexts in a single forward pass, potentially improving **scalability and reasoning reliability**.
- Compared with standard LLM prompting, RLMs introduce an explicit **recursive computation strategy** around the language model rather than relying solely on a single context window.

### 17.68 What is continual learning in LLMs?

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

##### Continual Learning in LLMs

- **Continual learning** is the process of continuously updating an LLM with **new data, knowledge, or tasks** without retraining the model from scratch.
- It allows models to adapt to **changing information, evolving user needs, new domains, and new languages**.
- Common approaches include **continued pretraining, incremental fine-tuning, PEFT/LoRA adapters, and rehearsal using historical data**.
- A major challenge is **catastrophic forgetting**, where learning new information causes the model to lose previously learned knowledge or capabilities.
- Other challenges include **data distribution shift, knowledge conflicts, training stability, and preventing low-quality or malicious data from being learned**.
- Techniques such as **replay buffers, regularization, mixture-of-adapters, periodic evaluation, and knowledge distillation** can help preserve existing capabilities.
- For frequently changing factual knowledge, **RAG is often preferable to continually modifying the model's parameters**.

### 17.69 How do Diffusion Language Models (DLMs) work?

> **See also:** [Question 13.47 — How do Diffusion Language Models (DLMs) work?](../part-3-machine-learning/13-deep-learning.md#1347-how-do-diffusion-language-models-dlms-work), which covers the same question from another angle.

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

### 17.70 Why layer normalization in Transformers?

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

### 17.71 Layer normalization vs. Batch Normalization?

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

### 17.72 What is SVD (singular value decomposition)?

> **See also:** [Question 1.53 — What is Singular Value Decomposition (SVD)?](../part-1-foundations/01-mathematics-probability-and-statistics.md#153-what-is-singular-value-decomposition-svd), which covers the same question from another angle.

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

### 17.73 What is KL Divergence?

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

### 17.74 What is entropy vs. cross-entropy?

> **See also:** [Question 8.38 — What is Cross-Entropy?](../part-3-machine-learning/08-machine-learning-fundamentals.md#838-what-is-cross-entropy), which covers the same question from another angle.

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

### 17.75 L1 vs L2 Geometry intuition?

|  | **L1** | **L2** |
| --- | --- | --- |
| **Constraint** | $\|w\|_1 = \sum_i \|w_i\| \leq c$ | $\|w\|_2 = \sqrt{\sum_i w_i^2} \leq c$ |
| **Shape** | Diamond | Circle |
| **Boundary** | Sharp corners | Smooth |
| **Corners on axes?** | Yes | No |
| **Likely solution** | On an axis | Anywhere on the circle |
| **Effect** | Some weights become exactly $0$ | Weights become small |
| **Key intuition** | **Sparsity** | **Shrinkage** |

### 17.76 Multi Query Attention?

**Multi-Query Attention (MQA)** is an attention mechanism where **all Query heads share a single Key and Value head**.

The key idea is:

> **Multiple Query heads share the same Key and Value.**

### 17.77 Multi-Head Attention vs Multi-Query Attention vs Grouped-Query Attention?

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

### 17.78 What is reinforcement learning?

> **See also:** [Question 8.28 — What is Reinforcement Learning?](../part-3-machine-learning/08-machine-learning-fundamentals.md#828-what-is-reinforcement-learning), which covers the same question from another angle.

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

### 17.79 What is the difference between an LM cache and a KV cache?

|  | **KV Cache** | **LM Cache** |
| --- | --- | --- |
| What it caches | Attention **Keys & Values** | Previously computed **LLM outputs/states**, depending on implementation |
| Purpose | Avoid recomputing attention for previous tokens | Avoid recomputing repeated LLM computation |
| Used during | Autoregressive generation | Repeated/prefix/shared computation |
| Granularity | Token-level K/V tensors | Typically request/prefix-level computation |
| Main benefit | Faster token generation | Faster repeated inference |
| Memory | High GPU memory usage | Can use GPU/CPU/SSD depending on system |

### 17.80 How does dropout behave during inference?

- **Normal inference:** Dropout **OFF** → deterministic prediction.
- **MC Dropout:** Dropout **ON** → multiple stochastic predictions → estimate **uncertainty**.
  ${Normal: } \hat{y}=f(x)$

  ${MC Dropout: } \hat{y}_t=f(x;m_t), \quad t=1,\dots,T$

**Key idea:** Dropout is normally disabled during inference, but MC Dropout intentionally enables it for uncertainty estimation.

### 17.81 What is a Large Language Model (LLM)?

A neural network (typically transformer-based) trained on massive text datasets to predict and generate language. It's the core model behind tools like Claude or GPT.

### 17.82 What does autoregressive mean?

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

### 17.83 What is a Large Language Model (LLM), and how does it work?

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

### 17.84 What Does Each Transformer Block Learn?

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

### 17.85 What is the KV cache in LLMs?

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

### 17.86 What is PagedAttention in LLMs?

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

### 17.87 What is speculative decoding, and how does it speed up inference?

> **See also:** [Question 27.8 — What is speculative decoding, and how does it speed up inference?](../part-5-production-systems/27-ai-infrastructure-and-inference-optimization.md#278-what-is-speculative-decoding-and-how-does-it-speed-up-inference), which covers the same question from another angle.

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

### 17.88 How does continuous batching improve LLM inference throughput?

> **See also:** [Question 27.7 — How does continuous batching improve LLM inference throughput?](../part-5-production-systems/27-ai-infrastructure-and-inference-optimization.md#277-how-does-continuous-batching-improve-llm-inference-throughput), which covers the same question from another angle.

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

### 17.89 What is pre-training vs fine-tuning in LLMs?

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

### 17.90 What are some challenges in training LLMs?

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

### 17.91 What is zero-shot learning in the context of LLMs?

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

### 17.92 How do you handle bias and fairness in LLMs?

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

### 17.93 What are some real-world applications of LLMs in business and tech?

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

### 17.94 How does the Transformer architecture improve LLM performance over RNNs?

#### How Transformers Improve LLM Performance Over RNNs

- **Parallelization:** RNNs process tokens sequentially (each step depends on the previous), preventing parallel training. Transformers process the entire sequence at once via self-attention, enabling massive parallelization on GPUs/TPUs — critical for training on today's huge datasets within a feasible time.
- **Long-range dependencies:** RNNs suffer from vanishing gradients across many time steps, weakening their ability to connect distant tokens even with LSTM/GRU improvements. Transformers connect any two tokens directly in a single step via self-attention, regardless of distance, capturing long-range context far more effectively.
- **No information bottleneck:** RNNs compress sequence history into a single hidden state that gets overwritten at each step, losing information over long sequences. Transformers give every token direct access to every other token's representation, avoiding this compression bottleneck.
- **Better scalability:** Transformers scale much more effectively with more data and parameters — this scalability is precisely why large language models (GPT, Claude, etc.) became possible, whereas RNN-based models hit performance/scaling walls much earlier.
- **Stable gradient flow:** Residual connections and layer normalization in Transformers support training very deep networks (dozens of layers) more stably than deep RNN chains, which suffer more from vanishing/exploding gradients.
- **Richer contextual representations:** Multi-head attention lets the model simultaneously capture different types of relationships (syntactic, semantic, positional) at each layer, versus a single sequential hidden-state pathway in RNNs.
- **Net result:** These properties combined are why Transformers replaced RNNs as the foundation for virtually all modern LLMs — enabling both the training scale and the modeling quality needed for today's language capabilities.

### 17.95 How does attention help capture long-range dependencies?

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

### 17.96 How does tokenization work in Large Language Models?

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

### 17.97 What is subword tokenization?

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

### 17.98 What is BPE (Byte Pair Encoding) in LLMs?

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

### 17.99 What is positional embedding in LLMs?

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

### 17.100 What is temperature in the context of LLMs?

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

### 17.101 What are skip connections?

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

### 17.102 What is normalization?

> **See also:** [Question 13.49 — Normalization in machine learning?](../part-3-machine-learning/13-deep-learning.md#1349-normalization-in-machine-learning), which covers the same question from another angle.

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

### 17.103 What is dropout, and how is it applied in LLMs?

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

### 17.104 Why does Attention use Softmax?

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

### 17.105 What does a vector database (Vector DB) store for LLM usage?

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

### 17.106 How do you improve inference speed in production LLM deployments?

> **See also:** [Question 27.1 — How do you improve inference speed in production LLM deployments?](../part-5-production-systems/27-ai-infrastructure-and-inference-optimization.md#271-how-do-you-improve-inference-speed-in-production-llm-deployments), which covers the same question from another angle.

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

### 17.107 What is the context window in an LLM?

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

### 17.108 Explain Prompting, Retrieval-Augmented Generation (RAG), and Fine-Tuning.

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

### 17.109 What is Reinforcement Learning from Human Feedback (RLHF)?

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

### 17.110 How does Knowledge Distillation work?

##### Knowledge Distillation

- **Knowledge distillation** is a technique where a smaller **student model learns from a larger, more capable teacher model**.
- The teacher produces **soft probability distributions** over possible outputs, which contain more information than just the correct label.
- The student is trained to **match the teacher's behavior** while also learning from the original training targets.
- A **temperature parameter** is often used to soften the teacher's output distribution, making relationships between classes easier for the student to learn.
- The goal is to create a **smaller, faster, and cheaper model** that retains much of the teacher's performance.
- It is commonly used for **LLMs, model compression, deployment on resource-constrained devices, and transferring capabilities from large models to smaller models**.
- Example: a large 70B LLM acts as the **teacher**, generating high-quality responses, while a smaller 7B LLM learns from those responses and becomes the **student**.

### 17.111 What is instruction tuning, and why is it important for chat models?

> **See also:** [Question 22.9 — What is instruction tuning, and why is it important for chat models?](22-fine-tuning-and-model-adaptation.md#229-what-is-instruction-tuning-and-why-is-it-important-for-chat-models), which covers the same question from another angle.

##### Instruction Tuning

- **Instruction tuning** is the process of fine-tuning a pretrained LLM on **instruction–response pairs** so it learns to follow human instructions.
- The training data typically contains examples such as **user instruction → desired assistant response**.
- It teaches the model **what to do with a request**, rather than simply predicting the next token from general text.
- It improves **instruction following, task completion, formatting, helpfulness, and conversational behavior**.
- It is especially important for **chat models** because pretrained models are primarily trained for language modeling, not necessarily for acting as interactive assistants.
- Instruction tuning is often followed by **preference alignment** techniques such as **RLHF, DPO, or GRPO** to further improve response quality and alignment.
- Example: a pretrained model may complete `"Explain photosynthesis"` unpredictably, while an instruction-tuned model learns to respond with a clear, structured explanation.

### 17.112 What is the difference between prefill and decode?

> **See also:** [Question 26.2 — What is the difference between prefill and decode?](../part-5-production-systems/26-mlops-and-llmops.md#262-what-is-the-difference-between-prefill-and-decode), which covers the same question from another angle.

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

### 17.113 How does Sliding Window Attention work?

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

### 17.114 How do Attention Sinks work?

- Softmax forces attention weights to sum to 1, so even when no token is relevant, the model must still allocate mass somewhere — it learns to dump the excess onto a few "junk" tokens rather than distort meaningful ones.
- The first token(s) in a sequence tend to become this dumping ground, since causal attention lets every later token see them — a reliable, always-visible target.
- This is a learned, self-reinforcing behavior: once a head starts treating a token as a sink, it keeps doing so throughout training.
- The sink token itself usually carries little to no real semantic meaning — its job is structural, not content-based.
- Identified/named by the **StreamingLLM** paper: sliding-window KV caches broke badly once sink tokens got evicted, even though those tokens seemed unimportant.
- Fix: always keep the first few tokens in the cache alongside the sliding window — this restores stable long-context generation.
- Implication: high attention weight ≠ semantic importance, which complicates naive attention-based interpretability.
- Some newer architectures add an explicit learned sink token/bias so attention heads have a dedicated "no-op" slot instead of hijacking a real token.

### 17.115 What is next-concept modeling (NCM), and how does it differ from an LLM?

**LLM (next-token modeling):** a Transformer trained to predict the next *token* (a subword). Reasoning, planning and abstraction emerge implicitly from that objective, and generation happens one token at a time.

**Next-concept modeling (NCM):** the idea of predicting the next *concept*, meaning a higher-level semantic unit such as a sentence or an idea embedding, instead of the next token. The best-known example is Meta's **Large Concept Model (LCM, 2024)**. It encodes each sentence into a language- and modality-agnostic embedding space (SONAR), trains a model (MSE-regression or diffusion-based) to predict the *next sentence embedding*, then decodes that embedding back to text in any supported language.

| | LLM (token-level) | NCM / LCM (concept-level) |
| --- | --- | --- |
| Unit of prediction | Subword token | Sentence/concept embedding |
| Sequence length | Long (many tokens) | Much shorter (one step per sentence), so attention is cheaper |
| Language dependence | Tied to the tokenizer/vocabulary | Concept space shared across languages and modalities |
| Planning | Implicit | Explicitly hierarchical: plan at the idea level, render the words later |
| Maturity | Production-grade, huge ecosystem | Research stage; quality depends on the encoder/decoder and lags strong LLMs on fluency |

**Why it matters:** humans plan at the level of ideas, not words. Concept-level modeling promises better long-form coherence, cheaper long-context processing, and zero-shot multilinguality.

**Caveats:** embedding-space prediction is lossy (fine detail such as numbers or exact wording can drift), the field has few tools and benchmarks yet, and in production today it is a research direction, not a replacement for LLMs.

### 17.116 How does BPE tokenization work?

**Byte-Pair Encoding** builds a subword vocabulary by repeatedly merging the most frequent adjacent symbol pair.

**Training**

1. Pre-tokenize the corpus (split on whitespace and punctuation, e.g. GPT-2's regex) and count word frequencies.
2. Start with a base vocabulary of characters, or the 256 **bytes** in byte-level BPE, so any string is encodable and there is never an `[UNK]`.
3. Count all adjacent symbol pairs (weighted by word frequency). Merge the most frequent pair into a new symbol and record the merge rule.
4. Repeat until the target vocabulary size is reached (e.g., 32k–200k).

```text
corpus: low×5 lower×2 newest×6 widest×3
merge 1: (e,s) → es     merge 2: (es,t) → est    merge 3: (l,o) → lo ...
```

**Encoding:** split the new text into base symbols, then apply the learned merges **in the order they were learned** (by merge rank) until none apply. Decoding just concatenates the tokens (bytes), then UTF-8 decodes.

**Why it works:** frequent words become single tokens, while rare words break into meaningful pieces (`unhappiness → un + happi + ness`). That balances vocabulary size against sequence length and handles open vocabularies.

**Trade-offs and production notes**

- A larger vocabulary means shorter sequences (cheaper attention, more context) but a bigger embedding/LM-head matrix and rare, undertrained tokens.
- Tokenization is greedy and frequency-driven, not linguistic. It causes quirks such as poor arithmetic on numbers split oddly, spelling and character-counting failures, and non-English text costing more tokens (a fairness and cost issue).
- Related schemes: **WordPiece** (BERT; merges by likelihood gain), **Unigram LM** (SentencePiece; prunes a large vocabulary probabilistically). Libraries: `tiktoken`, Hugging Face `tokenizers` (Rust), SentencePiece.

### 17.117 How do you implement a tokenizer (training, encoding, decoding, special tokens)?

A tokenizer has four parts: a **normalizer**, a **pre-tokenizer**, a **model** (BPE/Unigram/WordPiece) and a **post-processor** for special tokens. A minimal byte-level BPE:

```python
import re
from collections import Counter

PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+""")

class BPETokenizer:
    def __init__(self, special=("<|bos|>", "<|eos|>", "<|pad|>")):
        self.merges, self.special = {}, list(special)

    def train(self, text, vocab_size):
        words = Counter(tuple(w.encode()) for w in PAT.findall(text))   # bytes 0..255
        self.vocab = {i: bytes([i]) for i in range(256)}
        nxt = 256
        while nxt < vocab_size - len(self.special):
            pairs = Counter()
            for w, f in words.items():
                for p in zip(w, w[1:]): pairs[p] += f
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges[best] = nxt; self.vocab[nxt] = self.vocab[best[0]] + self.vocab[best[1]]
            words = Counter({self._merge(w, best, nxt): f for w, f in words.items()})
            nxt += 1
        self.special_ids = {s: nxt + i for i, s in enumerate(self.special)}

    @staticmethod
    def _merge(w, pair, new):
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair: out.append(new); i += 2
            else: out.append(w[i]); i += 1
        return tuple(out)

    def encode(self, text):
        ids = []
        for chunk in re.split("(" + "|".join(map(re.escape, self.special)) + ")", text):
            if chunk in self.special_ids: ids.append(self.special_ids[chunk]); continue
            for w in PAT.findall(chunk):
                t = tuple(w.encode())
                while len(t) > 1:          # apply the lowest-rank (earliest-learned) merge first
                    pair = min(zip(t, t[1:]), key=lambda p: self.merges.get(p, 1e18))
                    if pair not in self.merges: break
                    t = self._merge(t, pair, self.merges[pair])
                ids.extend(t)
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in self.special_ids.items()}
        return b"".join(inv[i].encode() if i in inv else self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

**Key design points to discuss**

- **Special tokens** (BOS/EOS/PAD, chat-role markers, tool-call tags) are matched *before* BPE and must never be produced from user text by accident, or you open a prompt-injection hole. Make "allowed special tokens" an explicit flag.
- **Round-trip guarantee:** `decode(encode(x)) == x`. Byte-level BPE guarantees it; test it with fuzzing on Unicode and emoji.
- **Performance:** the naive training loop is $O(\text{merges} \times \text{corpus})$. Real implementations update pair counts incrementally with a heap, and are written in Rust/C++ (HF `tokenizers`, `tiktoken`).
- **Streaming decode:** a multi-byte UTF-8 character can span several tokens, so buffer bytes before emitting text.
- **Versioning:** the tokenizer is part of the model contract. Changing it means retraining the embeddings.

### 17.118 What does the LM head do, and what is weight tying?

**LM head:** the final layer of a language model. It maps each position's final hidden state $h \in \mathbb{R}^{d}$ to **logits over the vocabulary**:

$$\text{logits} = h W_{\text{out}}^\top (+ b), \quad W_{\text{out}} \in \mathbb{R}^{|V| \times d}, \qquad p(\text{next token}) = \text{softmax}(\text{logits}/T)$$

During training these logits go into cross-entropy against the next token. At inference they feed sampling (greedy, temperature, top-k/top-p). It is usually preceded by a final LayerNorm/RMSNorm.

**Weight tying:** reuse the **input embedding matrix** $E \in \mathbb{R}^{|V| \times d}$ as the output projection, $W_{\text{out}} = E$ (Press & Wolf 2017; Inan et al. 2017).

```python
class TinyLM(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self.lm_head.weight = self.embed.weight   # tied: same Parameter object
```

**Why tie**

- **Fewer parameters:** $|V| \times d$ can be huge (128k × 4096 ≈ 0.5B params), which matters a lot for small models.
- **Regularization and better perplexity:** the input and output token representations live in one space, and rare tokens get gradient signal from both ends.
- Intuition: a token's output score is the dot product between the context vector and that token's embedding, a natural similarity.

**Why many large LLMs *untie* (e.g., LLaMA 2/3 at larger sizes)**

- At scale the parameter saving is relatively small, and separate matrices give extra capacity, because input semantics and output prediction need somewhat different geometry.
- It interacts with optimization (embedding vs. output gradient scales) and with sharding in tensor parallelism.

**Practical gotchas:** when you resize the vocabulary (adding special tokens) you must resize both matrices and keep them tied. LoRA or quantization configs must handle the shared parameter correctly. The LM head is also often the single biggest matmul at decode time for small models.

### 17.119 Explain self-attention mathematically.

The core operation is:

$Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

#### Transformer Self-Attention

Self-attention allows each token to determine **how much attention to give to every other token** in the sequence.

Given input embeddings:

$X \in \mathbb{R}^{n \times d_{\text{model}}}$

means **$X$ is a matrix of real numbers with $n$ rows and $d_{\text{model}}$ columns.**

we create **Query, Key, and Value** matrices:

$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V$

where $W_Q,W_K,W_V$ are learned projection matrices.

#### 1. Compute attention scores

Compare each query with every key using the dot product: $S = QK^T$

Scale the scores to prevent large values: $S = \frac{QK^T}{\sqrt{d_k}}$

where $d_k$ is the dimension of each key vector.

#### 2. Apply softmax

Convert scores into attention probabilities:

$A = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$

Each row of $A $ represents how much one token attends to all tokens.

#### 3. Weighted sum of values

$\boxed{ \operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V }$

So the overall flow is:

$X \rightarrow Q,K,V \rightarrow QK^T \rightarrow \text{Scale} \rightarrow \text{Softmax} \rightarrow AV$

#### Why Q,K,VQ,K,V?

- **Query:** What information is this token looking for?
- **Key:** What information does each token contain?
- **Value:** What information should actually be retrieved?

#### Multi-Head Attention

Instead of performing one attention operation, Transformers use multiple heads:

$head_i = \operatorname{Attention} (XW_Q^i,XW_K^i,XW_V^i)$

Then concatenate them: $\operatorname{MHA}(X) = \operatorname{Concat}(head_1,\ldots,head_h)W_O$

Different heads can learn different relationships, such as syntax, semantic relationships, or long-range dependencies.

### 17.120 Why divide by $\sqrt{d_k}$?

Scaled dot-product attention is $\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$. The $\sqrt{d_k}$ is there to keep the **variance of the logits** around 1.

**Derivation:** if the components of $q$ and $k$ are independent with mean 0 and variance 1, then

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \quad \text{Var}(q \cdot k) = d_k.$$

So the standard deviation grows like $\sqrt{d_k}$. With $d_k = 128$ the logits have std ≈ 11.

**Why that is a problem:** large logits push softmax into **saturation**. One position gets almost all the probability mass and the rest get ~0. The softmax Jacobian $\text{diag}(p) - pp^\top$ then becomes nearly zero, so **gradients vanish** and training is slow or unstable. Attention also becomes prematurely "hard".

Dividing by $\sqrt{d_k}$ restores unit variance, so the softmax temperature does not depend on head dimension.

**Related practice:** some models instead apply **QK-LayerNorm** or learn the temperature for extra stability at large scale. In mixed precision, the scaling also helps avoid FP16 overflow in the logits, which is why FlashAttention applies the scale inside the fused kernel.

### 17.121 What is the difference between self-attention and cross-attention?

Both use the same operation, $\text{softmax}(QK^\top/\sqrt{d_k})V$. The difference is **where Q, K and V come from**.

| | Self-attention | Cross-attention |
| --- | --- | --- |
| Q from | Sequence X | Sequence X (e.g. the decoder) |
| K, V from | The **same** sequence X | A **different** sequence Y (e.g. encoder output, image features) |
| Purpose | Relate tokens within one sequence | Condition one sequence on another |
| Shape | $n \times n$ attention map | $n \times m$ attention map |

**Examples:**

- **Self-attention** — every token in a sentence attends to the other tokens (BERT, GPT). In decoders it is **causally masked**.
- **Cross-attention** — in the original encoder-decoder Transformer for translation, the decoder queries attend over the encoder's source-sentence states. In **Stable Diffusion**, image latents (Q) attend to CLIP text embeddings (K, V). In Flamingo and some VLMs, text attends to image features.

**Production angle:** cross-attention K/V over a fixed context (the encoder output or image) can be **computed once and cached** for the whole generation. Decoder-only LLMs avoid cross-attention entirely by concatenating the context into the prompt and using self-attention, which simplifies the architecture but makes context length the cost driver.

### 17.122 Why do Transformers need positional information?

Self-attention is **permutation-equivariant**: it computes weighted sums over a *set* of tokens. If you shuffle the input tokens, the outputs shuffle in the same way, and nothing in $\text{softmax}(QK^\top)V$ depends on order. Without extra information, "dog bites man" and "man bites dog" look identical.

RNNs get order from sequential processing and CNNs from local kernels. Transformers process all tokens in parallel, so order must be **injected explicitly**.

**Ways to add it:**

- **Absolute positional encodings** — sinusoidal (original Transformer) or learned vectors added to token embeddings (BERT, GPT-2).
- **Relative encodings** — bias attention by the distance $i - j$ (T5, Transformer-XL).
- **RoPE** — rotate Q and K by position-dependent angles so the dot product depends on relative offset (LLaMA, most modern LLMs).
- **ALiBi** — add a linear distance penalty to attention logits.

**Nuance:** a causally masked decoder can infer some position implicitly (by counting how many tokens are visible), and "NoPE" models work at small scale. In practice explicit encodings train better and matter a lot for **long-context extrapolation**, which is why choices like RoPE scaling (NTK, YaRN) are central when extending context windows.

### 17.123 Compare absolute positional embeddings, RoPE, ALiBi, and relative positional encoding.

| Method | How it works | Pros | Cons | Used in |
| --- | --- | --- | --- | --- |
| **Absolute (sinusoidal / learned)** | Add a position vector $p_i$ to token embedding $x_i$ at the input | Simple, cheap | Learned: cannot go beyond max trained length. Both extrapolate poorly; position is mixed with content | Original Transformer, BERT, GPT-2 |
| **Relative positional encoding** | Add a learned bias or embedding based on distance $i-j$ inside attention | Captures what usually matters (relative offset); better generalisation | Extra compute/memory; harder with efficient kernels | Transformer-XL, T5 (bucketed bias) |
| **RoPE** (rotary) | Rotate pairs of dimensions of Q and K by angle $m\theta_k$ for position $m$; then $q_m^\top k_n$ depends only on $m-n$ | Relative by construction, no extra parameters, works with KV cache and FlashAttention, extendable via NTK/YaRN/position interpolation | Degrades beyond trained length without scaling tricks | LLaMA, Mistral, Qwen, GPT-NeoX |
| **ALiBi** | Add a fixed, head-specific linear penalty $-m\cdot\lvert i-j\rvert$ to attention logits | No parameters; extrapolates well to longer inputs ("train short, test long") | Strong recency bias can hurt long-range retrieval; somewhat lower quality than RoPE at scale | BLOOM, MPT |

**Interview summary:** absolute encodings say *where* a token is; relative, RoPE and ALiBi say *how far apart* two tokens are, which generalises better. RoPE is the current default because it is parameter-free, efficient and has well-understood context-extension methods.

### 17.124 Why does self-attention have $O(n^2)$ complexity?

For a sequence of $n$ tokens with head dimension $d$:

1. **Scores:** $QK^\top$ multiplies an $n \times d$ matrix by a $d \times n$ matrix, producing an **$n \times n$** matrix. Every token is compared with every other token, which is $O(n^2 d)$ compute.
2. **Softmax** over each row of the $n \times n$ matrix is $O(n^2)$.
3. **Weighted sum** with $V$ ($n \times n$ times $n \times d$) is $O(n^2 d)$.

So **time is $O(n^2 d)$** and naïvely **memory is $O(n^2)$** to store the attention matrix, per head and per layer. Compare the feed-forward layers at $O(n d^2)$: attention dominates once $n$ exceeds roughly $d$ (a few thousand tokens).

**Example:** going from 4k to 32k tokens is 8× the length but **64×** the attention FLOPs.

**What changes this:**

- **FlashAttention** keeps compute at $O(n^2)$ but cuts memory to $O(n)$ by never materialising the full matrix.
- **Sparse, sliding-window, linear and low-rank attention**, or state-space models (Mamba), reduce compute to ~$O(n)$ or $O(n \log n)$, usually with some quality tradeoff.
- **At inference with a KV cache**, each new token costs $O(n)$, but over the full generation it is still quadratic in total.

### 17.125 What techniques reduce Transformer attention complexity?

**Exact but more efficient (same result):**

- **FlashAttention** — tiled, fused kernel; $O(n)$ memory and much less HBM traffic. Still $O(n^2)$ FLOPs.
- **KV cache + GQA/MQA** — cut inference memory and bandwidth rather than asymptotic cost.

**Sparse patterns:**

- **Sliding-window / local attention** — each token attends to $w$ neighbours, giving $O(nw)$ (Longformer, Mistral).
- **Global + local + random tokens** — BigBird, Longformer. Stays expressive with near-linear cost.
- **Dilated / strided / block-sparse** attention (Sparse Transformer).

**Approximations:**

- **Low-rank** — project K and V to a fixed length $k$ (Linformer): $O(nk)$.
- **Kernel / linear attention** — replace softmax with a feature map so $(QK^\top)V = Q(K^\top V)$: $O(nd^2)$ (Performer, Linear Transformer).
- **Hashing / clustering** — attend only within LSH buckets (Reformer).

**Architectural alternatives:**

- **State-space models and RNN-like designs** — Mamba, RWKV, RetNet: linear time, constant-memory inference.
- **Hybrids** — mostly linear or SSM layers with a few full-attention layers (Jamba).
- **Context management** — retrieval, summarisation and memory tokens so you do not need long context at all.

**Tradeoff:** approximate and linear methods often lose quality on precise long-range retrieval ("needle in a haystack"). That is why frontier LLMs mostly keep exact attention plus FlashAttention, GQA, sliding windows and ring/sequence parallelism, rather than replacing softmax attention.

### 17.126 Explain Multi-Head Attention.

Instead of one attention over the full model width $d_{model}$, run $h$ attentions in parallel, each in a lower-dimensional subspace $d_k = d_{model}/h$.

$$\text{head}_i = \text{softmax}\left(\frac{(XW_i^Q)(XW_i^K)^\top}{\sqrt{d_k}}\right)XW_i^V$$

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\,W^O$$

**Steps:**

1. Project the input into Q, K and V (one big matmul in practice), then reshape to $(batch, h, n, d_k)$.
2. Compute scaled dot-product attention independently per head, with masks as needed.
3. Concatenate the heads and mix them with the output projection $W^O$.

```python
q, k, v = qkv(x).view(B, n, 3, h, dk).permute(2, 0, 3, 1, 4)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out = out.transpose(1, 2).reshape(B, n, h * dk)
y = proj(out)
```

**Cost:** roughly the same FLOPs and parameters as one full-width head, because the dimensions are split, not duplicated.

**Example:** GPT-3 175B uses 96 heads of size 128 ($d_{model} = 12288$). LLaMA-2 70B uses 64 query heads but only 8 KV heads (GQA) to shrink the KV cache.

### 17.127 Why are multiple attention heads useful?

A single head produces **one** softmax distribution per token, so it must average all the relationships it cares about into one weighted sum. Multiple heads let the model attend to **different things at the same time**.

- **Different relation types in parallel** — research has found heads specialised in previous-token, syntactic dependency (subject–verb), coreference, positional and **induction heads** (copying patterns, key for in-context learning).
- **Different representation subspaces** — each head has its own projections, so similarity is measured in different learned spaces.
- **Avoids averaging away information** — one head attending to two important tokens blurs them; two heads can each focus sharply.
- **Same cost** — heads split $d_{model}$, so you get this diversity at about the cost of one full-width head, and the computation parallelises well on GPUs.
- **Ensemble-like robustness.**

**Nuance:** many heads are redundant. Michel et al. (2019) pruned a large fraction of heads with little loss, and too many tiny heads ($d_k$ too small) hurts quality. That redundancy motivates **MQA/GQA** (share K/V across heads) and **head pruning** for cheaper inference.

### 17.128 What is the difference between encoder-only, decoder-only, and encoder-decoder Transformers?

| | **Encoder-only** | **Decoder-only** | **Encoder-decoder** |
| --- | --- | --- | --- |
| Attention | Bidirectional self-attention | **Causal** (masked) self-attention | Bidirectional encoder + causal decoder with **cross-attention** to the encoder |
| Pretraining | Masked language modelling (BERT), replaced-token detection (ELECTRA) | Next-token prediction | Span corruption / denoising (T5, BART) |
| Output | Contextual embeddings per token | Generated text, token by token | Generated text conditioned on an encoded input |
| Best for | Classification, NER, embeddings, retrieval, reranking | Open-ended generation, chat, code, in-context learning | Input→output transformation: translation, summarisation, speech-to-text |
| Examples | BERT, RoBERTa, DeBERTa, sentence-transformers | GPT, LLaMA, Claude, Mistral | T5, BART, mT5, Whisper, original Transformer |

**Production angle:**

- **Encoder-only** models are small, fast and cheap. They are still the right choice for embeddings, classifiers and cross-encoder rerankers in RAG.
- **Decoder-only** models are the general-purpose LLM default because one model handles any task via prompting.
- **Encoder-decoder** models are efficient when the input is long and the output short (the input is encoded once). They remain strong for translation and ASR.

### 17.129 Why are decoder-only architectures dominant for modern LLMs?

1. **Simplest objective that scales.** Next-token prediction on raw text needs no labels, uses every token as a training signal (unlike MLM, which trains on ~15%), and its loss follows smooth scaling laws.
2. **Generation is native.** Training (predict the next token) and inference (generate the next token) are the same task, so there is no train/inference gap.
3. **One model, every task.** Any task can be phrased as "text in, text out" via prompting. That enables **in-context learning** and instruction-following without task-specific heads, which emerged strongly in GPT-3.
4. **Engineering efficiency.** One homogeneous stack is easier to scale, shard and optimise. The KV cache makes autoregressive inference efficient, and an ecosystem (FlashAttention, vLLM, speculative decoding) is built around it.
5. **Empirical results.** At equal compute, decoder-only models have done as well as or better than encoder-decoder models for zero-shot generalisation (Wang et al., 2022). Most labs converged on the design, which compounded tooling advantages.
6. **Long-context and multimodal extension** is straightforward: just prepend tokens (images, documents, tool outputs).

**Caveats:** causal attention is less efficient for pure understanding tasks. Encoders still win for embeddings and classification per dollar, and encoder-decoder models remain strong for translation and ASR (Whisper). Decoder-only dominance is partly a scaling and ecosystem effect, not proof that it is strictly better.

### 17.130 What is the difference between pretraining, instruction tuning, and alignment?

| Stage | Data | Objective | What it gives you |
| --- | --- | --- | --- |
| **Pretraining** | Trillions of tokens of unlabeled text and code | Next-token prediction (self-supervised) | Knowledge, language and reasoning ability. A **base model** that continues text rather than follows instructions |
| **Instruction tuning (SFT)** | ~10k–1M curated (instruction, response) pairs, often multi-turn chats | Supervised cross-entropy on the response tokens | Follows instructions, adopts a chat format, uses tools, answers in the desired style |
| **Alignment / preference tuning** | Human or AI preference comparisons, verifiable rewards | RLHF (reward model + PPO), DPO, RLAIF / Constitutional AI, RLVR | Helpful, harmless and honest behaviour; refusing harmful requests; better reasoning when rewarded for verified correctness |

**Intuition:** pretraining builds **capability**, SFT teaches the **format and behaviour**, and alignment shapes **preferences and values** ("which of several valid answers is best").

**Practical points:**

- Compute is heavily skewed to pretraining (over 95%). Post-training is cheap but determines most of the user-perceived quality.
- SFT alone can only imitate demonstrations. Preference optimisation can push beyond them and reduce harmful or hallucinated outputs.
- Risks: the **alignment tax** (small capability loss), **reward hacking**, and over-refusal.
- In industry, teams rarely pretrain. They pick an aligned open model and do SFT, LoRA or DPO on domain data.

### 17.131 Explain supervised fine-tuning (SFT).

**Supervised fine-tuning (SFT)** continues training a pretrained model on **labeled input→output examples** with the standard next-token cross-entropy loss, usually computing the loss **only on the response tokens** (prompt tokens are masked).

$$\mathcal{L} = -\sum_{t \in \text{response}} \log p_\theta(y_t \mid x, y_{<t})$$

**Uses:** instruction-following and chat formatting (turning a base model into an assistant), domain adaptation (legal, medical, internal style), structured output, tool calling, and distilling a larger model's outputs into a smaller one.

**Practical recipe:**

- **Data quality beats quantity.** A few thousand clean, diverse, consistent examples often beat 100k noisy ones (the LIMA paper used 1k). Deduplicate and remove contradictions.
- Use the model's **chat template** exactly; template mismatch is a common silent failure.
- Low learning rate (~1e-5 to 2e-4 with LoRA), 1–3 epochs, early stopping on a held-out set; packing sequences for efficiency.
- **Parameter-efficient fine-tuning (LoRA/QLoRA)** for cost and to reduce forgetting.

**Limitations:** SFT only imitates the demonstrations. It can increase hallucination when it teaches facts the model does not know, and it can cause **catastrophic forgetting** of general skills. Mix in general data and evaluate on general benchmarks as well as the target task. Use preference tuning (DPO/RLHF) afterwards for quality beyond imitation.

### 17.132 What is model collapse?

**Model collapse** is the degradation that happens when generative models are **trained recursively on data generated by earlier models** instead of real human data (Shumailov et al., *Nature* 2024).

**Mechanism:**

- Each generation samples from an approximation of the previous distribution. Sampling and estimation errors compound, and the **tails of the distribution disappear first**: rare words, minority viewpoints and unusual cases.
- Over generations the distribution narrows and drifts. Outputs become bland and repetitive, and eventually nonsensical, with variance collapsing towards a few modes.
- The causes are statistical approximation error, function-approximation error and sampling choices (low temperature, top-p truncation), which all cut off the tails.

**Why it matters now:** more and more of the web is LLM-generated, so naïve future crawls get contaminated. Heavy use of synthetic data for SFT or pretraining can also reduce diversity.

**Mitigations:**

- **Keep real data in the mix** — accumulating synthetic data *alongside* the original data, rather than replacing it, largely avoids collapse (Gerstgrasser et al., 2024).
- **Provenance tracking and filtering** — watermarking, AI-text detection, data deduplication, and favouring pre-2022 or curated human sources.
- **Quality-controlled synthetic data** — verification (unit tests, math checkers), diversity-promoting generation, rejection sampling and human review.
- **Monitor diversity metrics** (entropy, distinct-n, tail-class coverage) across data and model versions.

Do not confuse it with **mode collapse** in GANs, which is a single training run failing to cover the modes, though both lose diversity.

### 17.133 What causes LLM hallucinations?

An LLM is trained to produce **plausible next tokens**, not true statements, so hallucination is a natural failure mode. Main causes:

- **Training objective:** next-token likelihood rewards fluent continuations; nothing forces "I don't know", and RLHF can reward confident, helpful-sounding answers.
- **Knowledge gaps and staleness:** rare, long-tail, post-cutoff or private facts were never (or weakly) learned, so the model interpolates.
- **Lossy parametric memory:** knowledge is compressed into weights; similar entities blur (wrong dates, merged people, invented but realistic-looking citations).
- **Decoding:** high temperature/top-p can pick wrong low-probability tokens; once emitted, autoregressive generation commits and builds on them (snowballing).
- **Prompt/context issues:** ambiguous questions, false premises, contradictory or irrelevant context, or key facts lost in a long context.
- **Poor calibration:** confidence doesn't track accuracy, so the model can't reliably abstain.
- **Noisy training data:** errors and fiction in the corpus get reproduced.

**In RAG** add: retrieval misses (answer not in context), the model overriding context with priors, and chunking that splits key facts.

**Production takeaway:** treat hallucination as a systems problem — grounding with citations, abstention instructions, low temperature for factual tasks, verification steps and continuous groundedness evaluation.

### 17.134 Can hallucinations be completely eliminated?

**No, not completely** — they can be reduced and contained, not eliminated.

**Why not:**
- Generation is probabilistic and the model has no built-in notion of truth.
- Its knowledge is incomplete and compressed; questions outside it invite interpolation.
- Even with perfect retrieval it can misread, over-generalise or combine facts wrongly.
- Theoretical results suggest any computable LLM used as a general question answerer will hallucinate on some inputs.

**What you can do:**
- **Ground** answers in authoritative retrieved sources and require citations.
- **Constrain** outputs: structured outputs, closed-set classification, tool calls to databases/calculators instead of free recall.
- **Allow abstention:** instruct and reward "I don't know"; apply confidence thresholds.
- **Verify:** groundedness checks (NLI or LLM judge vs context), self-consistency, rule checks.
- **Human-in-the-loop** for high-stakes decisions.
- **Monitor** hallucination rate on sampled production traffic.

**Interview framing:** aim for a hallucination rate that is *acceptable for the use case and detectable when it happens*. A creative-writing tool tolerates it; a tax or medical assistant needs grounding, verification and escalation so the residual risk is managed.

### 17.135 What is the relationship between model scaling and emergent capabilities?

**Emergent capabilities** are abilities that appear to be absent in smaller models and show up abruptly once models pass a certain scale (parameters, data, compute) — for example multi-step arithmetic, chain-of-thought reasoning, or in-context learning of new tasks (Wei et al., 2022).

**Relationship to scaling**
- **Loss scales smoothly:** pretraining loss decreases predictably as a power law in compute (the scaling laws).
- **Task metrics can look discontinuous:** a smooth gain in per-token accuracy can turn into a sharp jump on an all-or-nothing metric. If a 5-step problem needs every step right, accuracy ≈ $p^5$, which stays near zero and then rises quickly as $p$ improves.
- **The "mirage" critique (Schaeffer et al., 2023):** much apparent emergence comes from discontinuous metrics (exact match) and coarse model sampling; with continuous metrics (log-likelihood, partial credit), improvement often looks gradual and predictable.

**Practical view**
- Some capabilities genuinely need a threshold of competence across several sub-skills before the full task works.
- Capability gains also come from data quality, instruction tuning and RLHF, not just size.
- For planning: forecast with continuous proxy metrics on small models; don't assume a capability will appear at a given scale, and don't assume it won't.
- For safety: unpredictable capability jumps are why labs run dangerous-capability evals at each scale-up.

### 17.136 Explain the scaling laws for LLMs.

**Scaling laws** are empirical power-law relationships between a language model's pretraining loss and the resources used to train it.

**Kaplan et al. (OpenAI, 2020)**
$L(N) \approx (N_c/N)^{\alpha_N}$, $L(D) \approx (D_c/D)^{\alpha_D}$, $L(C) \approx (C_c/C)^{\alpha_C}$
where $N$ = non-embedding parameters, $D$ = tokens, $C$ = compute. Key findings:
- Loss falls smoothly as a power law over many orders of magnitude when the other factors aren't bottlenecks.
- Architecture details (depth vs width) matter much less than scale.
- They recommended putting most extra compute into **bigger models** (trained on relatively fewer tokens).

**Chinchilla (Hoffmann et al., DeepMind, 2022)** revised this: for a fixed compute budget $C \approx 6ND$, parameters and tokens should grow **roughly equally**, at about **20 tokens per parameter**. Many earlier models (GPT-3, Gopher) were undertrained.

**Common fitted form**
$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$ with $E$ the irreducible loss.

**How they're used**
- Predict the loss of a large run from small pilot runs; choose model size and token count for a budget.
- Since inference cost matters, modern practice **over-trains** smaller models far beyond 20 tokens/param (LLaMA 3 8B: ~15T tokens) to make serving cheaper.

**Limits:** they predict loss, not specific downstream abilities; they assume fixed data quality; and the supply of high-quality data is becoming the bottleneck (driving synthetic data and test-time compute scaling).

### 17.137 What is the Chinchilla scaling principle?

The **Chinchilla principle** (Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models*): for a fixed training compute budget, **model size and number of training tokens should be scaled in equal proportion**.

- Training compute ≈ $C \approx 6 N D$ (N = parameters, D = tokens).
- Compute-optimal: $N_{opt} \propto C^{0.5}$, $D_{opt} \propto C^{0.5}$.
- Rule of thumb: **~20 training tokens per parameter**.

**Evidence:** Chinchilla (70B params, 1.4T tokens) used the same compute as Gopher (280B params, 300B tokens) and outperformed it across most benchmarks — and was 4× cheaper to serve. It showed that GPT-3-era models were heavily **undertrained**, following Kaplan et al.'s advice to favor parameters over data.

**Nuances**
- "Compute-optimal" minimizes **training** compute for a target loss; it ignores inference cost. If a model will serve billions of requests, it is cheaper overall to train a **smaller model on far more tokens** (LLaMA, Mistral and Phi-style models go to 100–2000 tokens/param).
- It assumes unlimited, unique, high-quality data; data quality and repetition change the optimum.
- The exact coefficients depend on the data, tokenizer and architecture; re-fit them with pilot runs before a big training run.

**Interview one-liner:** "Chinchilla says double the data whenever you double the parameters — about 20 tokens per parameter — but production teams deliberately over-train smaller models because inference dominates lifetime cost."

### 17.138 Why can a smaller model outperform a larger model?

Parameter count is only one driver of quality. A smaller model can win because of:

- **Training data and tokens:** a smaller model trained on far more (or better) tokens beats an undertrained large one — Chinchilla 70B beat Gopher 280B; LLaMA-13B beat GPT-3 175B on many benchmarks.
- **Data quality:** curated, deduplicated, filtered or synthetic "textbook-quality" data (the Phi series) can be worth more than scale.
- **Task specialization:** a 7B model fine-tuned on a narrow domain (classification, extraction, SQL for one schema) often beats a general frontier model on that task.
- **Distillation:** learning from a stronger teacher's outputs/logits transfers capability into a compact student.
- **Post-training:** better instruction tuning, RLHF/DPO and reasoning RL can matter more than base size.
- **Architecture and context use:** better attention variants, longer effective context, and retrieval (RAG) give a small model access to knowledge it doesn't store.
- **Inference strategy:** test-time compute (self-consistency, verification, reasoning traces) lets a small model "think longer".
- **Evaluation artifacts:** benchmark contamination or metric quirks can also make a small model look better than it is — always validate on your own held-out data.

**Production angle:** smaller models bring lower latency and cost, on-prem deployment, and easier fine-tuning. A standard pattern is a **router**: a small specialized model for the bulk of traffic, escalating hard cases to a large model.

### 17.139 What is mixture-of-experts (MoE)?

A **Mixture-of-Experts (MoE)** layer replaces a single dense feed-forward block with **many parallel "expert" FFNs plus a learned router (gating network)** that sends each token to only a few experts.

```txt
token h ─► router: softmax(W_g · h) ─► top-k experts (e.g., 2 of 8)
                                        │
           y = Σ_{i∈topk} g_i(h) · Expert_i(h)
```

**Key properties**
- **Sparse activation:** each token uses only *k* experts, so compute per token ≈ that of a much smaller dense model, while total parameters (capacity/knowledge) are much larger.
- In Transformers, MoE usually replaces the FFN in some or all layers; attention stays dense.
- Examples: Switch Transformer (top-1), GShard, Mixtral 8×7B (top-2: ~47B total, ~13B active), DeepSeek-V3 (fine-grained + shared experts), and reportedly several frontier models.

**Benefits:** better quality per training FLOP; faster training to a target loss; experts can specialize (often by token type or syntax rather than clean "topics").

**Costs and challenges**
- **Memory:** all experts must be stored, so VRAM needs follow *total* params, not active ones.
- **Load balancing:** routers collapse onto a few experts without auxiliary balancing losses or capacity limits.
- **Communication:** expert parallelism needs all-to-all exchanges across GPUs.
- **Training instability and harder fine-tuning;** batching at inference is less efficient when tokens scatter across experts.

### 17.140 Why does MoE allow larger parameter counts without proportionally increasing compute?

Because **compute scales with the parameters a token actually touches, not with the parameters that exist**.

- In a dense Transformer, every token passes through every weight: FLOPs per token ≈ $2N$.
- In an MoE layer with $E$ experts and top-$k$ routing, each token passes through only $k$ experts. FFN FLOPs per token scale with $k$, not $E$. Total parameters grow roughly linearly with $E$, while active parameters stay about constant.

**Example — Mixtral 8×7B:** ~47B total parameters, but with top-2 routing only ~13B are **active** per token, so inference compute is roughly that of a 13B dense model while capacity is much larger. DeepSeek-V3: 671B total, ~37B active.

**Why extra parameters still help:** more experts = more total capacity to store knowledge and specialized features; the router conditionally picks the relevant subset per token (*conditional computation*).

**What does NOT scale down**
- **Memory:** all experts must sit in GPU memory (or be offloaded), so serving needs VRAM for total params.
- **Communication:** expert parallelism adds all-to-all traffic between devices.
- **Router overhead** (small), plus wasted capacity from load imbalance or dropped tokens.
- At small batch sizes decoding is memory-bandwidth bound, and touching different experts per token reduces weight reuse.

**Summary:** MoE decouples **capacity** (total params) from **cost per token** (active params); the price is memory footprint and systems complexity.

### 17.141 Explain expert routing and load balancing in MoE.

**Routing:** a gating network computes scores for each token, $g = \text{softmax}(W_g h)$, and the token is sent to the **top-k** experts (k=1 in Switch, k=2 in Mixtral). The outputs are combined with the (renormalized) gate weights: $y = \sum_{i \in \text{topk}} g_i E_i(h)$. Variants: noisy top-k (adds noise for exploration), expert-choice routing (experts pick tokens, which balances by construction), and fine-grained plus **shared experts** (DeepSeek), where a few experts process every token.

**Why load balancing is needed:** routers are self-reinforcing — popular experts get more gradient, get better and attract even more tokens. Results: **expert collapse** (a few experts do everything, others are wasted), GPU hotspots, and dropped tokens.

**Balancing techniques**
- **Auxiliary load-balancing loss** (Switch): $L_{aux} = \alpha \cdot E \sum_i f_i P_i$, where $f_i$ = fraction of tokens routed to expert *i* and $P_i$ = mean router probability for *i*; minimized when routing is uniform.
- **Router z-loss:** penalizes large router logits for numerical stability.
- **Capacity factor:** each expert accepts at most $C = \text{cf} \cdot \frac{\text{tokens}}{E}$ tokens per batch; overflow tokens are dropped (they pass through the residual) or rerouted. A higher factor drops fewer tokens but costs memory and padding.
- **Auxiliary-loss-free balancing** (DeepSeek-V3): a per-expert bias added to routing scores is adjusted dynamically to even out load, avoiding interference from the extra loss.
- **Expert-choice routing:** guarantees an even load.

**Production concerns:** monitor per-expert utilization and token-drop rate; at inference, balance experts across GPUs (expert parallelism) and replicate hot experts.

### 17.142 What is the difference between System 1 and System 2 thinking?

The distinction comes from the dual-process model of cognition popularized by Daniel Kahneman.

| System 1 | System 2 |
| --- | --- |
| Fast | Slow |
| Automatic | Deliberate |
| Intuitive | Analytical |
| Low cognitive effort | Higher cognitive effort |
| Pattern recognition | Step-by-step reasoning |
| Usually inexpensive | More computationally expensive |

#### AI analogy

**System 1-like AI:**

- Fast classification
- Simple prediction
- Direct response generation

**System 2-like AI:**

- Planning
- Multi-step reasoning
- Tool use
- Verification
- Iterative problem solving

Modern AI agents often combine fast generation with additional reasoning, planning, tool use, or verification loops rather than treating inference as a single forward pass.

---

← [Chapter 16. Reinforcement Learning](../part-3-machine-learning/16-reinforcement-learning.md) · [Table of contents](../README.md) · [Chapter 18. Prompt and Context Engineering](18-prompt-and-context-engineering.md) →
