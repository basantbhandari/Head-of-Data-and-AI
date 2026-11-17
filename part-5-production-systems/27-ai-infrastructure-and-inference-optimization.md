# Chapter 27. AI Infrastructure and Inference Optimization

_Part V — Production Systems and Architecture_

GPUs and accelerators, quantization, KV caching, batching, speculative decoding, serving engines, caching and cost control.

This chapter contains **62 questions**.

## Contents

- **27.1** [How do you improve inference speed in production LLM deployments?](#271-how-do-you-improve-inference-speed-in-production-llm-deployments)
- **27.2** [What are the main LLM optimization techniques?](#272-what-are-the-main-llm-optimization-techniques)
- **27.3** [How do you select GPUs for LLM inference?](#273-how-do-you-select-gpus-for-llm-inference)
- **27.4** [What is model parallelism vs data parallelism in distributed training?](#274-what-is-model-parallelism-vs-data-parallelism-in-distributed-training)
- **27.5** [What is tensor parallelism, and how does it help serve large models?](#275-what-is-tensor-parallelism-and-how-does-it-help-serve-large-models)
- **27.6** [What is pipeline parallelism?](#276-what-is-pipeline-parallelism)
- **27.7** [How does continuous batching improve LLM inference throughput?](#277-how-does-continuous-batching-improve-llm-inference-throughput)
- **27.8** [What is speculative decoding, and how does it speed up inference?](#278-what-is-speculative-decoding-and-how-does-it-speed-up-inference)
- **27.9** [What is KV cache, and how do you manage memory for it?](#279-what-is-kv-cache-and-how-do-you-manage-memory-for-it)
- **27.10** [What is Paged Attention?](#2710-what-is-paged-attention)
- **27.11** [How does GGUF work?](#2711-how-does-gguf-work)
- **27.12** [How do you optimize inference for edge and mobile deployment?](#2712-how-do-you-optimize-inference-for-edge-and-mobile-deployment)
- **27.13** [What is model quantization (INT8, INT4, FP16, BF16), and how does it affect quality?](#2713-what-is-model-quantization-int8-int4-fp16-bf16-and-how-does-it-affect-quality)
- **27.14** [How do you implement auto-scaling for AI workloads?](#2714-how-do-you-implement-auto-scaling-for-ai-workloads)
- **27.15** [What is the role of load balancing in AI serving infrastructure?](#2715-what-is-the-role-of-load-balancing-in-ai-serving-infrastructure)
- **27.16** [How do you manage GPU memory for serving multiple models?](#2716-how-do-you-manage-gpu-memory-for-serving-multiple-models)
- **27.17** [What is model sharding, and when would you use it?](#2717-what-is-model-sharding-and-when-would-you-use-it)
- **27.18** [How do you implement request queuing and priority scheduling for AI services?](#2718-how-do-you-implement-request-queuing-and-priority-scheduling-for-ai-services)
- **27.19** [What are the cost trade-offs between self-hosted and API-based AI inference?](#2719-what-are-the-cost-trade-offs-between-self-hosted-and-api-based-ai-inference)
- **27.20** [How do you handle cold start latency for serverless AI deployments?](#2720-how-do-you-handle-cold-start-latency-for-serverless-ai-deployments)
- **27.21** [How do you implement model caching to reduce redundant computations?](#2721-how-do-you-implement-model-caching-to-reduce-redundant-computations)
- **27.22** [What is the difference between synchronous and asynchronous inference, and when do you use each?](#2722-what-is-the-difference-between-synchronous-and-asynchronous-inference-and-when-do-you-use-each)
- **27.23** [What is FSDP (Fully Sharded Data Parallel), and how does it differ from DeepSpeed ZeRO?](#2723-what-is-fsdp-fully-sharded-data-parallel-and-how-does-it-differ-from-deepspeed-zero)
- **27.24** [How do you monitor and profile LLM inference in production (TTFT, inter-token latency, GPU utilization)?](#2724-how-do-you-monitor-and-profile-llm-inference-in-production-ttft-inter-token-latency-gpu-utilization)
- **27.25** [What is model routing at the infrastructure level, and how do you route requests based on complexity and cost?](#2725-what-is-model-routing-at-the-infrastructure-level-and-how-do-you-route-requests-based-on-complexity-and-cost)
- **27.26** [What is quantization?](#2726-what-is-quantization)
- **27.27** [What are the main model compression techniques?](#2727-what-are-the-main-model-compression-techniques)
- **27.28** [What are some methods to improve inference time?](#2728-what-are-some-methods-to-improve-inference-time)
- **27.29** [What is the common idea behind GPUs and TPUs?](#2729-what-is-the-common-idea-behind-gpus-and-tpus)
- **27.30** [What is the difference between static and continuous batching in LLMs?](#2730-what-is-the-difference-between-static-and-continuous-batching-in-llms)
- **27.31** [Why is AI infrastructure moving towards Rust?](#2731-why-is-ai-infrastructure-moving-towards-rust)
- **27.32** [How does prompt caching work, and when does it help?](#2732-how-does-prompt-caching-work-and-when-does-it-help)
- **27.33** [What does TensorRT (TensorRT-LLM) do?](#2733-what-does-tensorrt-tensorrt-llm-do)
- **27.34** [What are the main techniques for optimizing LLM inference overall (latency, throughput, cost)?](#2734-what-are-the-main-techniques-for-optimizing-llm-inference-overall-latency-throughput-cost)
- **27.35** [What is model routing?](#2735-what-is-model-routing)
- **27.36** [What is prompt caching?](#2736-what-is-prompt-caching)
- **27.37** [How does semantic caching differ from standard caching?](#2737-how-does-semantic-caching-differ-from-standard-caching)
- **27.38** [What is batching used for?](#2738-what-is-batching-used-for)
- **27.39** [What does working knowledge of Azure for AI/ML deployment typically involve?](#2739-what-does-working-knowledge-of-azure-for-aiml-deployment-typically-involve)
- **27.40** [How do you decide between using a managed cloud LLM/vector-DB service vs. self-hosting?](#2740-how-do-you-decide-between-using-a-managed-cloud-llmvector-db-service-vs-self-hosting)
- **27.41** [What is KV caching during inference?](#2741-what-is-kv-caching-during-inference)
- **27.42** [Why does KV cache significantly improve autoregressive generation?](#2742-why-does-kv-cache-significantly-improve-autoregressive-generation)
- **27.43** [What is Grouped Query Attention (GQA)?](#2743-what-is-grouped-query-attention-gqa)
- **27.44** [Compare MHA, MQA, and GQA.](#2744-compare-mha-mqa-and-gqa)
- **27.45** [What is FlashAttention?](#2745-what-is-flashattention)
- **27.46** [Why is FlashAttention faster without changing the attention result mathematically?](#2746-why-is-flashattention-faster-without-changing-the-attention-result-mathematically)
- **27.47** [What is PagedAttention?](#2747-what-is-pagedattention)
- **27.48** [How does PagedAttention improve LLM serving?](#2748-how-does-pagedattention-improve-llm-serving)
- **27.49** [Compare INT8, INT4, FP16, BF16, and FP8.](#2749-compare-int8-int4-fp16-bf16-and-fp8)
- **27.50** [What is quantization-aware training vs post-training quantization?](#2750-what-is-quantization-aware-training-vs-post-training-quantization)
- **27.51** [How would you reduce inference latency without significantly reducing accuracy?](#2751-how-would-you-reduce-inference-latency-without-significantly-reducing-accuracy)
- **27.52** [What is speculative decoding?](#2752-what-is-speculative-decoding)
- **27.53** [How can speculative decoding reduce LLM inference latency?](#2753-how-can-speculative-decoding-reduce-llm-inference-latency)
- **27.54** [What is continuous batching?](#2754-what-is-continuous-batching)
- **27.55** [Why is GPU memory often the bottleneck in LLM inference?](#2755-why-is-gpu-memory-often-the-bottleneck-in-llm-inference)
- **27.56** [Compare compute-bound vs memory-bandwidth-bound workloads.](#2756-compare-compute-bound-vs-memory-bandwidth-bound-workloads)
- **27.57** [What is the difference between training FLOPs and inference FLOPs?](#2757-what-is-the-difference-between-training-flops-and-inference-flops)
- **27.58** [How would you estimate the GPU requirements for serving an LLM?](#2758-how-would-you-estimate-the-gpu-requirements-for-serving-an-llm)
- **27.59** [What happens to model quality when you aggressively quantize a model?](#2759-what-happens-to-model-quality-when-you-aggressively-quantize-a-model)
- **27.60** [How would you design production LLM serving?](#2760-how-would-you-design-production-llm-serving)
- **27.61** [Explain KV cache, PagedAttention, FlashAttention, GQA and quantization.](#2761-explain-kv-cache-pagedattention-flashattention-gqa-and-quantization)
- **27.62** [What is LLM inference? Explain prefill vs. decode.](#2762-what-is-llm-inference-explain-prefill-vs-decode)

---

### 27.1 How do you improve inference speed in production LLM deployments?

> **See also:** [Question 17.106 — How do you improve inference speed in production LLM deployments?](../part-4-generative-ai/17-large-language-model-foundations.md#17106-how-do-you-improve-inference-speed-in-production-llm-deployments), which covers the same question from another angle.

To improve **LLM inference speed in production**, I would optimize the model, serving stack, request handling, and infrastructure.

- **Reduce computation**
  - Use smaller models when possible.
  - Apply **quantization** such as INT8/INT4.
  - Use knowledge distillation or specialized models for simpler workloads.
- **Optimize decoding**
  - Reduce unnecessary output tokens with concise prompts and output limits.
  - Use **speculative decoding**: a small draft model proposes tokens while the larger model verifies them.
  - Use optimized KV-cache management.
- **Optimize attention and memory**
  - Use **PagedAttention** or efficient KV-cache implementations.
  - Use **FlashAttention** or other optimized attention kernels.
  - Use **GQA/MQA** where supported to reduce KV-cache memory and bandwidth.
- **Improve batching**
  - Use **continuous/dynamic batching** to combine concurrent requests efficiently.
  - Avoid waiting for an entire batch before starting new requests.
- **Optimize prompts and context**
  - Reduce unnecessary input tokens.
  - Use RAG to retrieve only relevant context instead of sending entire documents.
  - Cache repeated prefixes or prompts using **prompt/prefix caching**.
- **Use optimized inference engines**
  - Deploy with systems such as **vLLM, TensorRT-LLM, or optimized model runtimes**.
  - Keep models resident in GPU memory and minimize CPU↔GPU transfers.
- **Scale intelligently**
  - Use GPU autoscaling and load balancing.
  - Route requests to different model sizes based on latency and quality requirements.
  - Use asynchronous queues for non-real-time workloads.
- **Measure the right metrics**
  - **TTFT (Time to First Token)**
  - **Inter-token latency**
  - **End-to-end latency**
  - **Tokens/second**
  - GPU utilization
  - Queue time
  - Cost per request

**Key idea:** Production LLM speed comes from optimizing **both prefill and decoding** through quantization, KV-cache optimization, continuous batching, speculative decoding, efficient attention, prompt caching, and optimized serving infrastructure.

### 27.2 What are the main LLM optimization techniques?

#### LLM Optimization Techniques

- **Quantization**
  - Reduce model precision: FP16/BF16 → INT8/INT4.
  - Reduces memory usage and can improve inference speed.
- **Knowledge Distillation**
  - Train a smaller **student model** to reproduce a larger teacher model.
  - Reduces inference cost and latency.
- **Pruning**
  - Remove less important weights, neurons, or attention components.
  - Reduces model size and computation.
- **LoRA / PEFT**
  - Fine-tune only a small number of parameters instead of the entire model.
  - Reduces training memory and compute.
- **KV Cache Optimization**
  - Cache previously computed key/value states during autoregressive generation.
  - Techniques such as **PagedAttention, GQA, and MQA** reduce memory pressure.
- **Efficient Attention**
  - Use optimized kernels such as **FlashAttention**.
  - Reduces memory traffic and improves attention performance.
- **Speculative Decoding**
  - A small draft model proposes tokens.
  - The larger model verifies them in batches.
  - Can significantly improve generation speed.
- **Continuous Batching**
  - Dynamically combine requests during inference.
  - Improves GPU utilization and throughput.
- **Prompt/Prefix Caching**
  - Reuse computation for repeated system prompts or shared context.
  - Reduces prefill latency.
- **Context Optimization**
  - Remove unnecessary instructions and history.
  - Use RAG to retrieve only relevant information.
  - Apply context compression/summarization for long conversations.
- **Model Routing**
  - Route simple requests to smaller/cheaper models and complex requests to larger models.
  - Balances quality, latency, and cost.
- **Inference Optimization**
  - Use optimized serving engines such as **vLLM** or TensorRT-LLM.
  - Use GPU acceleration, FP16/BF16, optimized kernels, and efficient memory management.
- **Architecture Optimization**
  - Use **Mixture-of-Experts (MoE)** to activate only a subset of parameters per token.
  - Use architectures with efficient attention and smaller KV caches.

##### Key idea

**LLM optimization = reduce computation + reduce memory usage + reduce tokens + maximize hardware utilization.**

The main production techniques are **quantization, distillation, KV-cache optimization, efficient attention, speculative decoding, continuous batching, prompt caching, context optimization, and model routing**.

### 27.3 How do you select GPUs for LLM inference?

I select GPUs for **LLM inference** based on **model size, memory requirements, latency, throughput, concurrency, and cost**.

- **1. Calculate GPU memory requirements**
  - Model weights are the primary requirement.
  - Consider:
    - Parameter count
    - Precision: FP16/BF16/INT8/INT4
    - KV-cache memory
    - Activations and runtime overhead
  - Ensure the model and KV cache fit comfortably in GPU VRAM.
- **2. Consider model size**
  - Small models may run efficiently on a single consumer/datacenter GPU.
  - Large models may require multiple GPUs with tensor or pipeline parallelism.
- **3. Consider latency requirements**
  - For low-latency interactive applications, prioritize:
    - High GPU compute
    - High memory bandwidth
    - Fast interconnects
  - Measure **TTFT and inter-token latency**.
- **4. Consider throughput and concurrency**
  - For high request volume, prioritize GPUs that provide strong **tokens/sec/\$**.
  - Continuous batching can significantly improve GPU utilization.
- **5. Consider GPU memory bandwidth**
  - LLM inference is often memory-bandwidth bound, particularly during autoregressive decoding.
  - Higher-bandwidth GPUs can therefore improve token generation speed.
- **6. Consider multi-GPU communication**
  - For models that don't fit on one GPU, high-speed interconnects such as **NVLink** can reduce communication overhead.
  - Avoid unnecessary cross-GPU transfers.
- **7. Consider cost and utilization**
  - Compare:
    - Cost per GPU hour
    - Tokens/sec
    - Requests/sec
    - Cost per million tokens
    - Expected utilization
  - The most powerful GPU isn't always the most economical choice.
- **8. Match GPU to workload**
  - **Low traffic / small models:** smaller, cheaper GPUs.
  - **High-throughput production:** high-memory datacenter GPUs.
  - **Large models:** high-VRAM GPUs or multi-GPU configurations.
  - **Latency-sensitive workloads:** GPUs with high compute, memory bandwidth, and fast interconnects.

**Key idea:** GPU selection is not simply about maximum TFLOPS. I would optimize for **VRAM capacity + memory bandwidth + latency + throughput + interconnect + cost per token**, based on the production workload.

### 27.4 What is model parallelism vs data parallelism in distributed training?

**Data parallelism** and **model parallelism** are techniques for distributing deep-learning training across multiple GPUs.

##### Data Parallelism

**Each GPU has a complete copy of the model but processes different batches of data.**

- GPU 1 → Batch A
- GPU 2 → Batch B
- GPU 3 → Batch C
- GPU 4 → Batch D
- Gradients are synchronized across GPUs, typically using **AllReduce**.
- Every GPU updates the same model parameters.

**Best when:** The model fits comfortably on a single GPU, but you need higher training throughput.

**Advantage:** Simple and highly scalable.

**Limitation:** Every GPU must hold the entire model, optimizer states, and activations.

##### Model Parallelism

**The model itself is split across multiple GPUs.**

Example:

- GPU 1 → Layers 1–10
- GPU 2 → Layers 11–20
- GPU 3 → Layers 21–30
- GPU 4 → Layers 31–40

Different GPUs hold different portions of the model and communicate intermediate activations during forward/backward passes.

**Best when:** The model is too large to fit on a single GPU.

**Advantage:** Enables training very large models.

**Limitation:** More communication overhead and potentially lower GPU utilization.

##### Key Difference

|  | Data Parallelism | Model Parallelism |
| --- | --- | --- |
| What is split? | Data | Model |
| Model copy | Full copy on each GPU | Model partitioned |
| Main goal | Increase throughput | Fit large models |
| Communication | Gradient synchronization | Activation/gradient transfer |
| Typical use | Smaller/medium models | Very large LLMs |

**Key idea:**

**Data parallelism = same model, different data.**

**Model parallelism = different GPUs hold different parts of the model.**

In large LLM training, these are often combined with **tensor parallelism, pipeline parallelism, and data parallelism** to scale across many GPUs.

### 27.5 What is tensor parallelism, and how does it help serve large models?

**Tensor parallelism (TP)** splits the mathematical operations of a neural network across multiple GPUs, so **one model layer is computed jointly by several GPUs**.

- **How it works**
  - Large weight matrices are partitioned across GPUs.
  - Each GPU computes part of the matrix operation.
  - GPUs communicate intermediate results using collective operations such as **AllReduce** or **AllGather**.
  - The results are combined to produce the layer output.
- **Example**
  - A large Transformer layer is split across 4 GPUs.
  - Each GPU stores approximately 25% of the relevant weights.
  - All 4 GPUs work together on the same input token.
- **Why it helps serve large LLMs**
  - Allows models that **don't fit in a single GPU's VRAM** to be served.
  - Reduces per-GPU memory requirements.
  - Enables larger models and higher throughput.
  - Can reduce latency because matrix computations are performed in parallel.
- **Trade-offs**
  - Requires fast GPU-to-GPU communication.
  - Communication overhead can become significant.
  - Works best with high-bandwidth interconnects such as **NVLink**.
  - More complex than single-GPU serving.

##### Tensor vs Data Parallelism

|  | Tensor Parallelism | Data Parallelism |
| --- | --- | --- |
| Splits | Model computations/weights | Requests/data |
| Model per GPU | Partial | Full |
| Purpose | Fit/accelerate large models | Increase throughput |
| Communication | Frequent during layers | Mainly gradient synchronization during training |

**Key idea:** **Tensor parallelism splits a single model's computation across GPUs**, allowing large LLMs to fit and execute across multiple GPUs. It is especially useful for **large-model inference**, where the model cannot fit into one GPU's memory.

### 27.6 What is pipeline parallelism?

**Pipeline parallelism** splits a neural network into **sequential groups of layers**, with each group placed on a different GPU.

##### How it works

For example, a 24-layer Transformer can be divided across 4 GPUs:

- **GPU 1:** Layers 1–6
- **GPU 2:** Layers 7–12
- **GPU 3:** Layers 13–18
- **GPU 4:** Layers 19–24

During inference:

`Input → GPU 1 → GPU 2 → GPU 3 → GPU 4 → Output`

Each GPU processes its assigned layers and passes the intermediate activations to the next GPU.

##### During training

Pipeline parallelism typically uses **micro-batches** to keep GPUs busy:

- GPU 1 processes micro-batch 2 while GPU 2 processes micro-batch 1.
- This creates a pipeline and improves GPU utilization.
- **Pipeline bubbles** can occur when GPUs are idle while the pipeline is being filled or drained.

##### Advantages

- Allows very large models to fit across multiple GPUs.
- Reduces memory requirements per GPU.
- Can scale models beyond a single machine.

##### Disadvantages

- Communication between pipeline stages adds latency.
- Pipeline bubbles can reduce utilization.
- More complex scheduling.
- Sequential layer dependencies make it less flexible than pure tensor parallelism.

##### Tensor vs Pipeline Parallelism

|  | Tensor Parallelism | Pipeline Parallelism |
| --- | --- | --- |
| Splits | Operations/weights | Layers |
| GPUs | Work on same layer | Work on different layers |
| Communication | Frequent within layers | Between layers |
| Main benefit | Faster computation | Fit larger models |
| Common use | Low-latency inference | Very large model training/serving |

**Key idea:** **Tensor parallelism splits a layer across GPUs, while pipeline parallelism splits the model's layers across GPUs.** In large LLM systems, they are often combined with **data parallelism** for distributed training.

### 27.7 How does continuous batching improve LLM inference throughput?

> **See also:** [Question 17.88 — How does continuous batching improve LLM inference throughput?](../part-4-generative-ai/17-large-language-model-foundations.md#1788-how-does-continuous-batching-improve-llm-inference-throughput), which covers the same question from another angle.

**Continuous batching** dynamically adds and removes requests from a batch while the LLM is generating tokens, instead of waiting for all requests in a batch to finish.

##### How it works

- Traditional batching waits for the **slowest request** in a batch to complete before starting new requests.
- Continuous batching schedules requests **token-by-token**.
- When one request finishes, a new request can immediately take its place.
- The GPU remains highly utilized instead of sitting idle.

Example:

```txt
Traditional batching:
Request A ────────────────
Request B ───────
Request C ────────────────
             ↓
       Wait for A/C

Continuous batching:
Request A ────────────────
Request B ───────
Request C ────────────────
Request D       ───────────
Request E          ────────
```

##### Why throughput improves

- **Higher GPU utilization:** fewer idle GPU cycles.
- **Dynamic scheduling:** new requests enter as others finish.
- **Handles variable output lengths:** short requests don't block long ones.
- **Higher requests/sec and tokens/sec:** more useful work is processed per unit of GPU time.
- **Better scalability:** particularly valuable for high-concurrency LLM serving.

##### Trade-off

Continuous batching requires sophisticated scheduling and KV-cache management. It can improve throughput substantially, but the scheduler must balance **throughput, TTFT, and per-request latency**.

**Key idea:** **Continuous batching keeps the GPU continuously busy by dynamically replacing completed requests with new ones during generation, significantly improving LLM serving throughput.**

### 27.8 What is speculative decoding, and how does it speed up inference?

> **See also:** [Question 17.87 — What is speculative decoding, and how does it speed up inference?](../part-4-generative-ai/17-large-language-model-foundations.md#1787-what-is-speculative-decoding-and-how-does-it-speed-up-inference), which covers the same question from another angle.

**Speculative decoding** is an LLM inference technique where a **small, fast draft model** generates several candidate tokens, and the **large target model** verifies them in parallel.

##### How it works

1. **The draft model** quickly predicts multiple tokens.
2. **The target model** evaluates those tokens in a single forward pass.
3. Correct predictions are **accepted**.
4. Incorrect predictions are **rejected**, and the target model generates the correct token.
5. The process repeats until generation is complete.

```txt
Prompt
  ↓
Small Draft Model
  ↓
Token1 → Token2 → Token3 → Token4
  ↓
Large Target Model
  ↓
Verify in parallel
  ↓
Accept valid tokens + generate correction
```

##### Why it speeds up inference

- Autoregressive LLMs normally require a target-model computation for **every generated token**.
- Speculative decoding allows the target model to verify **multiple tokens in parallel**.
- If the draft model is accurate, several tokens can be accepted per target-model step.
- This reduces the number of expensive target-model decoding iterations.
- It can significantly improve **tokens/sec and generation latency** without changing the final model distribution when using an exact speculative-decoding algorithm.

##### Trade-offs

- Requires an additional draft model.
- Speedup depends on how accurately the draft model predicts the target model.
- Poor draft-model agreement can reduce or eliminate the benefit.
- Additional memory and system complexity are required.

**Key idea:** **Use a cheap model to predict multiple tokens and a powerful model to verify them in parallel, reducing expensive sequential decoding steps.**

### 27.9 What is KV cache, and how do you manage memory for it?

**KV cache (Key-Value cache)** stores the **key and value tensors from previous tokens** during autoregressive LLM generation so the model doesn't recompute them for every new token.

##### Why it is needed

Without KV caching:

- Each new token would require recomputing attention for all previous tokens.
- This creates significant redundant computation.

With KV caching:

- Previous **K/V states are stored in GPU memory**.
- For each new token, the model computes only its new K/V states.
- The new query attends to the cached K/V states.

```txt
Without KV Cache:
Token 1 → recompute
Token 2 → recompute Token 1 + 2
Token 3 → recompute Token 1 + 2 + 3
...

With KV Cache:
Token 1 → store K/V
Token 2 → reuse K/V + add new K/V
Token 3 → reuse cached K/V + add new K/V
```

##### KV Cache Memory Management

- **PagedAttention**
  - Stores KV cache in fixed-size blocks/pages instead of requiring one large contiguous memory region.
  - Reduces memory fragmentation and enables efficient sharing.
- **Quantized KV cache**
  - Stores K/V values at lower precision such as INT8 or FP8 where supported.
  - Reduces memory usage.
- **GQA / MQA**
  - Use fewer key/value heads than query heads.
  - Significantly reduces KV-cache size.
- **Prefix caching**
  - Reuse KV cache for shared prefixes such as system prompts.
  - Particularly useful for applications with repeated context.
- **Eviction/context management**
  - Remove or compress older context when the conversation becomes very long.
  - Use sliding-window or other attention strategies when supported.
- **Dynamic allocation**
  - Allocate KV-cache blocks as requests grow rather than reserving maximum context length for every request.
  - Important for high-concurrency serving.
- **Continuous batching**
  - Efficiently allocate and release KV-cache memory as requests enter and leave the batch.

##### Key Trade-off

**KV cache reduces computation but consumes significant GPU memory.** Longer contexts and higher concurrency therefore increase KV-cache requirements substantially.

**Key idea:** Efficient LLM serving requires treating KV cache as a first-class resource using **paged memory, quantization, GQA/MQA, prefix caching, dynamic allocation, and context management**.

### 27.10 What is Paged Attention?

**PagedAttention** is a memory-management technique for efficiently storing and accessing the **KV cache** during LLM inference. It is famously used by **vLLM**.

##### How it works

Traditional KV-cache allocation often reserves a large contiguous memory region for each request, which can cause **memory fragmentation and wasted GPU memory**.

PagedAttention divides the KV cache into fixed-size **blocks/pages**:

```txt
Request
   ↓
KV Cache
┌─────┬─────┬─────┬─────┐
│ B1  │ B2  │ B3  │ B4  │
└─────┴─────┴─────┴─────┘
   ↓
GPU memory blocks
```

- Each request gets KV-cache blocks as needed.
- The blocks do **not** need to be physically contiguous.
- A block table maps a request's logical KV positions to physical GPU-memory blocks.
- Blocks can be allocated and freed dynamically as requests grow or finish.

##### Why it improves LLM serving

- **Reduces memory fragmentation**
- **Reduces wasted KV-cache memory**
- Allows **higher request concurrency**
- Enables more efficient **continuous batching**
- Makes dynamic sequence lengths easier to manage
- Can support **KV-cache sharing**, such as shared prefixes, when the serving system implements it.

##### Traditional vs PagedAttention

|  | Traditional KV Cache | PagedAttention |
| --- | --- | --- |
| Memory allocation | Large/contiguous | Fixed-size blocks |
| Fragmentation | Higher | Lower |
| Memory utilization | Lower | Higher |
| Dynamic sequences | Less efficient | Efficient |
| Concurrency | More limited | Higher |

**Key idea:** **PagedAttention treats the KV cache like virtual memory: it divides it into manageable blocks and maps logical token positions to physical GPU memory, allowing LLM serving systems to use GPU memory much more efficiently.**

### 27.11 How does GGUF work?

**GGUF** is a file format designed to store **LLM model weights, metadata, and configuration** efficiently for local inference. It is commonly used with **llama.cpp** and tools such as Ollama.

##### How GGUF works

- **Stores model weights**
  - Contains tensors representing the model parameters.
  - Supports multiple precisions and quantization formats such as **F16, Q8, Q6, Q5, Q4**, etc.
- **Quantization**
  - Converts higher-precision weights into lower-bit representations.
  - Example: FP16 → 4-bit quantization.
  - Reduces model size and memory requirements while attempting to preserve model quality.
- **Stores metadata**
  - Architecture information
  - Vocabulary/tokenizer information
  - Context-length configuration
  - Tensor information
  - Quantization/type information
- **Memory-mapped loading**
  - GGUF files can be efficiently memory-mapped by runtimes such as llama.cpp.
  - This allows models to be loaded without necessarily copying the entire file into RAM at once.
- **CPU/GPU inference**
  - Designed for efficient local inference.
  - Parts or all of the model can run on CPU, GPU, or a combination depending on the runtime and hardware.

##### Example

```txt
Original model
      ↓
FP16 weights
      ↓
Quantization
      ↓
GGUF
 ┌──────────────────┐
 │ Metadata         │
 │ Tokenizer        │
 │ Quantized tensors│
 │ Model config     │
 └──────────────────┘
      ↓
llama.cpp / Ollama
      ↓
CPU / GPU inference
```

##### GGUF vs Quantization

They are **not the same thing**:

- **GGUF** = file/container format.
- **Quantization** = technique for reducing numerical precision of model weights.
- A GGUF file can contain **quantized or non-quantized** tensors.

**Key idea:** GGUF provides a portable, efficient way to package LLM weights and metadata, while supporting quantized representations that make local inference much more memory-efficient.

### 27.12 How do you optimize inference for edge and mobile deployment?

To optimize **AI/LLM inference for edge and mobile devices**, I focus on reducing **model size, memory usage, computation, and power consumption** while maintaining acceptable accuracy.

- **Model compression**
  - Use **quantization** such as INT8 or INT4.
  - Apply **pruning** to remove unnecessary parameters.
  - Use **knowledge distillation** to create smaller student models.
- **Choose smaller models**
  - Use compact architectures or task-specific models instead of large foundation models.
  - Use model routing so complex requests can be sent to the cloud when necessary.
- **Optimize inference runtime**
  - Use hardware-accelerated runtimes such as **Core ML, TensorFlow Lite, ONNX Runtime, or TensorRT** where supported.
  - Take advantage of mobile **GPU, NPU, DSP, or Neural Engine** acceleration.
- **Optimize memory**
  - Minimize model weights and activation memory.
  - Use memory-efficient operators.
  - Avoid unnecessary CPU ↔ GPU/NPU data transfers.
  - Stream or process inputs incrementally when possible.
- **Optimize computation**
  - Use operator/kernel fusion.
  - Use optimized attention and matrix operations.
  - Reduce input resolution or sequence length when quality permits.
  - Cache reusable computations.
- **Optimize latency and power**
  - Benchmark **p50/p95 latency, throughput, memory usage, and energy per inference**.
  - Batch only when it actually improves efficiency; on-device workloads often prioritize low latency.
  - Avoid keeping high-power accelerators active unnecessarily.
- **Use hybrid edge-cloud inference**
  - Perform privacy-sensitive or latency-critical processing locally.
  - Offload computationally expensive tasks to the cloud when appropriate.
  - Use fallback mechanisms when connectivity is unavailable.

**Key idea:** Edge optimization is a trade-off between **accuracy, model size, latency, memory, and battery consumption**. The most common approach is **smaller/quantized models + hardware acceleration + memory-efficient runtimes + selective edge/cloud execution**.

### 27.13 What is model quantization (INT8, INT4, FP16, BF16), and how does it affect quality?

**Model quantization** reduces the numerical precision used to represent model weights and/or activations, making models **smaller, faster, and more memory-efficient**.

##### Precision levels

| Format | Typical bits | Memory | Quality impact | Common use |
| --- | --- | --- | --- | --- |
| FP32 | 32 | Very high | Baseline | Training |
| **BF16** | 16 | High | Very low | Training/inference |
| **FP16** | 16 | High | Very low | Training/inference |
| **INT8** | 8 | \~50% of FP16 | Usually small | Production inference |
| **INT4** | 4 | \~25% of FP16 | More noticeable | Efficient LLM inference |

##### How quantization works

```txt
FP16 / BF16 model
       ↓
Quantization
       ↓
INT8 / INT4 model
       ↓
Smaller memory footprint
       ↓
Faster / cheaper inference
```

For example, an LLM with FP16 weights can require roughly **2 bytes per parameter**, while INT4 uses roughly **0.5 bytes per parameter** for the raw quantized weights, before metadata/scales and runtime overhead.

##### Effect on quality

- **FP16 → BF16**
  - Usually negligible quality difference for many workloads.
  - BF16 has a wider numerical range than FP16.
- **FP16 → INT8**
  - Usually small quality degradation.
  - Often a strong quality/performance trade-off.
- **FP16 → INT4**
  - Greater potential quality degradation.
  - Can still work extremely well for many LLMs when using good quantization methods.
- **Very aggressive quantization**
  - Can hurt reasoning, instruction following, factual accuracy, or long-context performance.
  - The impact is model- and task-dependent.

##### Important techniques

- **PTQ (Post-Training Quantization):** Quantize an already-trained model.
- **QAT (Quantization-Aware Training):** Train/fine-tune while accounting for quantization effects.
- **Weight-only quantization:** Quantize weights while keeping activations at higher precision.
- **Activation-aware methods:** Consider which weights/activations are particularly important for preserving quality.

##### Key idea

**Quantization trades numerical precision for efficiency.** INT8 usually provides a strong balance between quality and efficiency, while INT4 provides much larger memory savings but requires more careful calibration and evaluation. FP16/BF16 remain common when maximum quality and numerical stability are priorities.

### 27.14 How do you implement auto-scaling for AI workloads?

**Auto-scaling for AI workloads** dynamically increases or decreases compute resources based on workload demand while maintaining latency, throughput, and cost targets.

- **1. Choose the right scaling strategy**
  - **Horizontal scaling:** Add/remove inference workers or GPU instances.
  - **Vertical scaling:** Increase CPU/GPU resources per instance.
  - For production AI serving, horizontal scaling is usually preferred.
- **2. Define scaling metrics**
  - CPU/GPU utilization
  - Request rate
  - Queue depth
  - Concurrent requests
  - **p95/p99 latency**
  - Tokens/sec
  - GPU memory utilization
  - TTFT for LLMs
- **3. Use queue-based scaling**
  - Put requests into a queue such as Kafka, SQS, or RabbitMQ.
  - Scale workers based on queue depth and request age.
  - This works particularly well for asynchronous AI workloads.
- **4. Use GPU-aware autoscaling**
  - GPU workloads require specialized node pools.
  - Add/remove GPU instances based on demand.
  - Keep minimum capacity for predictable traffic and scale to zero when appropriate for non-real-time workloads.
- **5. Use dynamic batching**
  - Combine concurrent inference requests into batches.
  - For LLMs, use **continuous batching** to improve GPU utilization.
- **6. Use model-aware routing**
  - Route simple requests to smaller models.
  - Route complex requests to larger models.
  - Use fallback models during capacity constraints.
- **7. Handle traffic spikes**
  - Configure **min/max replicas**.
  - Use predictive or scheduled scaling for known traffic patterns.
  - Maintain warm instances to avoid GPU startup/cold-start latency.
- **8. Protect the system**
  - Apply rate limiting, backpressure, request prioritization, timeouts, and circuit breakers.
  - Prevent overload when demand exceeds available GPU capacity.
- **9. Optimize cost**
  - Scale down during low traffic.
  - Use cheaper instances for non-critical workloads.
  - Use spot/preemptible GPUs for fault-tolerant batch jobs.
  - Monitor **cost per request/token**.

##### Example architecture

```txt
Users
  ↓
Load Balancer
  ↓
AI API
  ↓
Queue / Scheduler
  ↓
GPU Inference Workers
  ├── Model A
  ├── Model B
  └── Model C
       ↑
  Autoscaler
       ↑
GPU utilization + queue depth + latency
```

**Key idea:** AI autoscaling should be **workload-aware**, not based only on CPU utilization. For GPU/LLM workloads, combine **queue depth, concurrency, GPU utilization, TTFT, p95/p99 latency, and token throughput** to scale intelligently while controlling cost.

### 27.15 What is the role of load balancing in AI serving infrastructure?

**Load balancing in AI serving infrastructure** distributes incoming inference requests across available CPU/GPU workers or model replicas to improve **availability, latency, throughput, and resource utilization**.

- **Distribute traffic**
  - Spread requests across multiple inference servers.
  - Prevent one GPU or instance from becoming overloaded.
- **Improve latency**
  - Route requests to the least-loaded or fastest available worker.
  - Consider GPU utilization, queue depth, and current request load.
- **Increase availability**
  - If one model server fails, route requests to healthy replicas.
  - Use health checks, timeouts, and automatic failover.
- **Support horizontal scaling**
  - As autoscaling adds GPU replicas, the load balancer automatically distributes traffic across them.
  - When demand decreases, traffic is reduced before instances are removed.
- **AI-specific routing**
  - Route based on:
    - Model type/version
    - GPU availability
    - Model already loaded on a GPU
    - Request priority
    - Context length
    - Tenant/customer
  - Avoid sending a request to a GPU that must first load the required model.
- **LLM serving**
  - Consider **queue depth, KV-cache utilization, concurrent sequences, and token generation rate** rather than only CPU/GPU utilization.
  - Work with continuous batching to maximize GPU utilization.
- **Cost optimization**
  - Route requests to appropriate model sizes or GPU tiers.
  - Keep expensive GPUs focused on workloads that actually require them.

**Key idea:** AI load balancing is more than simple round-robin routing. It should be **GPU-aware and model-aware**, using workload, queue, memory, and latency information to route requests efficiently while maintaining reliability and cost targets.

### 27.16 How do you manage GPU memory for serving multiple models?

**Managing GPU memory for serving multiple AI models** requires controlling model weights, KV cache, activations, and runtime overhead so that multiple models can share GPU resources efficiently.

- **Model placement**
  - Keep frequently used models permanently loaded.
  - Load less frequently used models on demand.
  - Use **model routing** to decide which GPU should serve each request.
- **Quantization**
  - Use FP16/BF16, INT8, or INT4 depending on quality requirements.
  - Quantization significantly reduces model-weight memory.
- **GPU partitioning**
  - Use technologies such as **MIG** on supported NVIDIA GPUs to partition GPU resources into isolated instances.
  - Useful when serving multiple smaller models with predictable resource requirements.
- **KV-cache management**
  - KV cache can become a major memory consumer for LLMs.
  - Use **PagedAttention**, dynamic allocation, prefix caching, and appropriate context limits.
  - Use GQA/MQA or KV-cache quantization where supported.
- **Dynamic model loading/unloading**
  - Load models when needed and evict unused models.
  - Use **LRU-style policies** when GPU memory is constrained.
  - Keep commonly used models warm to avoid model-loading latency.
- **Memory-aware scheduling**
  - Track:
    - GPU VRAM usage
    - KV-cache usage
    - Model size
    - Active requests
    - Available memory
  - Reject, queue, or route requests when insufficient memory is available.
- **Multi-GPU strategies**
  - Use **tensor parallelism** or **pipeline parallelism** for models that cannot fit on one GPU.
  - Replicate smaller models across GPUs for higher throughput.
- **Continuous batching**
  - Dynamically batch requests while carefully managing KV-cache allocation.
  - Improves GPU utilization without unnecessarily reserving memory.
- **Monitoring**
  - Monitor GPU memory utilization, allocation failures/OOMs, model load time, utilization, latency, and throughput.
  - Set alerts before memory exhaustion causes production failures.

**Key idea:** Efficient multi-model serving combines **quantization + intelligent model placement + dynamic loading/eviction + KV-cache management + memory-aware scheduling** to maximize GPU utilization without causing OOM failures.

### 27.17 What is model sharding, and when would you use it?

**Model sharding** is the process of splitting a model's parameters, weights, or related states across **multiple files, GPUs, or machines** so that the entire model does not need to reside in one location.

##### How it works

For a large LLM:

```txt
Large Model
     ↓
 ┌───────┬───────┬───────┐
 │Shard 1│Shard 2│Shard 3│
 │ GPU 1 │ GPU 2 │ GPU 3 │
 └───────┴───────┴───────┘
```

Each GPU stores only part of the model, and the inference/training system coordinates computation between the shards.

##### Types

- **Tensor parallel sharding**
  - Splits individual weight matrices across GPUs.
  - GPUs cooperate on the same layer.
- **Pipeline parallel sharding**
  - Splits different layers across GPUs.
  - Activations move between pipeline stages.
- **Parameter/file sharding**
  - Stores a large model across multiple checkpoint files.
  - Primarily helps with storage and loading; it doesn't necessarily mean distributed computation.

##### When to use it

- The model **doesn't fit into a single GPU's VRAM**.
- You need to serve very large LLMs.
- You want to distribute memory requirements across multiple GPUs.
- Training requires more memory than one GPU can provide.
- Model checkpoints are too large to efficiently manage as a single file.

##### Trade-offs

- Additional GPU-to-GPU communication.
- More complex deployment and scheduling.
- Network/interconnect bandwidth becomes important.
- Poor sharding strategies can increase latency.

**Key idea:** **Model sharding breaks a large model into manageable pieces across GPUs or storage locations.** For distributed LLM serving, it is commonly combined with **tensor parallelism or pipeline parallelism** when the model cannot fit on a single GPU.

### 27.18 How do you implement request queuing and priority scheduling for AI services?

**Request queuing and priority scheduling** allow AI services to control traffic, protect GPUs from overload, and ensure important requests receive resources first.

- **1. Put requests behind a queue**
  - API receives the request and places it in a queue such as Kafka, RabbitMQ, SQS, or Redis-based queues.
  - Workers consume requests when GPU capacity is available.
- **2. Assign priorities**
  - Example:
    - **P0:** critical production requests
    - **P1:** interactive user requests
    - **P2:** batch/background jobs
  - Use priority queues so higher-priority requests are processed first.
- **3. Prevent starvation**
  - Don't allow low-priority requests to wait indefinitely.
  - Use **aging**, where a request's effective priority increases as its waiting time grows.
- **4. Use workload-aware scheduling**
  - Consider:
    - GPU memory requirements
    - Model required
    - Estimated token count
    - Context length
    - Current GPU utilization
    - Expected execution time
  - Route requests to GPUs that already have the required model loaded when possible.
- **5. Use dynamic/continuous batching**
  - Combine compatible requests into batches.
  - For LLMs, **continuous batching** allows new requests to join while existing requests are still generating.
- **6. Apply admission control**
  - Set queue-size limits and concurrency limits.
  - Reject, defer, or gracefully degrade requests when the system is overloaded.
- **7. Use backpressure**
  - Slow down request acceptance when workers or queues become saturated.
  - Prevent cascading failures and GPU OOM conditions.
- **8. Meet SLA requirements**
  - Track **queue time, TTFT, p95/p99 latency, throughput, and deadline misses**.
  - Reserve capacity for high-priority workloads when necessary.

##### Example

```txt
Clients
   ↓
API Gateway
   ↓
Priority Scheduler
   ├── P0 Critical
   ├── P1 Interactive
   └── P2 Batch
          ↓
     GPU Worker Pool
     ├── GPU 1
     ├── GPU 2
     └── GPU 3
```

**Key idea:** Build a **priority-aware, GPU-aware scheduler** with queues, admission control, aging, backpressure, and continuous batching to maximize throughput while protecting latency and critical workloads.

### 27.19 What are the cost trade-offs between self-hosted and API-based AI inference?

The main trade-off is **control and predictable high-volume cost vs. operational simplicity and pay-per-use economics**.

- **Self-hosted inference**
  - Higher upfront cost: GPUs, servers, storage, networking, and infrastructure.
  - Ongoing costs: electricity, cooling, maintenance, monitoring, model updates, and engineering staff.
  - Can become **cheaper per request at high, predictable traffic** because you amortize GPU costs over many requests.
  - Better control over model, data, latency, quantization, and infrastructure.
  - Risk of **GPU underutilization** during low-traffic periods.
  - Requires handling scaling, failover, capacity planning, and GPU availability.
- **API-based inference**
  - Very low infrastructure and operational overhead.
  - Usually pay based on tokens, requests, images, or compute consumed.
  - Excellent for **low or unpredictable traffic** because you don't pay for idle GPUs.
  - Easier to scale and handle traffic spikes.
  - Vendor manages GPUs, model serving, upgrades, and availability.
  - Costs can become significantly higher at **large, sustained request volumes**.
  - Creates vendor dependency and potentially less control over data, model behavior, and latency.

##### Cost comparison

| Factor | Self-hosted | API-based |
| --- | --- | --- |
| Initial cost | High | Very low |
| Fixed cost | High | Low |
| Variable cost | Relatively low | Per request/token |
| High traffic | Often cheaper | Can become expensive |
| Low traffic | Often inefficient | Usually cheaper |
| Engineering effort | High | Low |
| Scaling | Your responsibility | Provider-managed |
| GPU utilization | Critical | Provider-managed |
| Control | High | Lower |
| Vendor lock-in | Low | Higher |

**Key interview point:**

> Use **API inference** when traffic is low, variable, or you want minimal operational complexity. Use **self-hosted inference** when traffic is high and predictable, latency/control requirements are strict, and you can keep expensive GPU infrastructure highly utilized. Many production systems use a **hybrid approach** to balance cost, reliability, and control.

### 27.20 How do you handle cold start latency for serverless AI deployments?

Cold starts happen when a serverless function must **initialize the runtime, load dependencies, load the AI model, and allocate compute** before serving the first request.

I handle them with:

- **Provisioned concurrency / warm instances** — keep a minimum number of instances ready.
- **Keep-warm strategy** — send lightweight scheduled requests to reduce idle shutdowns where appropriate.
- **Smaller models** — use distilled or quantized models to reduce model-loading time.
- **Model caching** — keep model weights and dependencies in fast local storage or memory when possible.
- **Container optimization** — minimize image size, dependencies, and initialization work.
- **Lazy loading** — load non-critical components only when needed.
- **Pre-initialization** — initialize tokenizers, connections, and model components during startup rather than per request.
- **Separate inference from orchestration** — keep the serverless layer lightweight and route heavy inference to persistent GPU services.
- **Request routing/fallbacks** — route latency-sensitive requests to warm infrastructure and batch/background workloads to serverless workers.
- **Monitor cold-start metrics** — track initialization time, warm vs. cold requests, TTFT, p95/p99 latency, and cost.

**Key interview point:**

> For lightweight AI workloads, optimize the serverless function itself. For large LLMs or GPU inference, serverless cold starts can be expensive, so I typically keep the model-serving layer warm and use serverless components mainly for API orchestration and asynchronous tasks.

### 27.21 How do you implement model caching to reduce redundant computations?

Model caching stores **models or their intermediate results in memory/storage so repeated requests don't require the same expensive computation again**.

- **Keep models warm** — load frequently used models into GPU/CPU memory and avoid repeated model initialization.
- **Cache model weights** — store weights in fast local storage or a shared cache instead of downloading them for every instance.
- **Embedding cache** — cache embeddings for repeated documents, queries, or frequently accessed content.
- **Response cache** — cache deterministic LLM responses for identical or semantically equivalent requests.
- **Prefix/prompt caching** — reuse KV-cache or precomputed representations for shared prompt prefixes.
- **Feature/intermediate cache** — cache expensive preprocessing or intermediate model outputs when they can be safely reused.
- **Distributed cache** — use Redis or another shared cache when multiple inference workers need access to cached results.
- **Cache invalidation** — include model version, prompt/template version, preprocessing version, and relevant parameters in the cache key.
- **TTL and LRU eviction** — expire stale entries and remove less-used items when memory is limited.
- **Cache monitoring** — track hit rate, miss rate, latency reduction, memory usage, and cost savings.

**Example:**

`Request → Cache lookup → Hit: return cached result → Miss: run model → Store result → Return response`

**Key interview point:**

> I use caching at multiple levels—model weights, embeddings, prefixes/KV cache, and final responses—to avoid redundant computation while using versioned cache keys and appropriate TTL/eviction policies to prevent stale results.

### 27.22 What is the difference between synchronous and asynchronous inference, and when do you use each?

The difference is **how the client waits for the inference result**.

- **Synchronous inference**
  - The client sends a request and **waits for the model response**.
  - Best for interactive, real-time applications.
  - Examples: chatbots, real-time classification, recommendation APIs, fraud checks.
  - Requires low latency and typically has request timeouts.
  - Simpler request/response architecture.
- **Asynchronous inference**
  - Client submits a job and **does not wait for the model to finish**.
  - A queue or job system processes the request in the background.
  - The client can poll for the result or receive a callback/event.
  - Best for long-running or high-volume workloads.
  - Examples: video processing, document extraction, batch prediction, image generation.

##### Architecture

**Synchronous:**

`Client → API → Model → Response`

**Asynchronous:**

`Client → API → Queue → Inference Worker → Result Store → Client`

##### When to use

| Requirement | Synchronous | Asynchronous |
| --- | --- | --- |
| Real-time response | Best | Not ideal |
| Long-running inference | Not ideal | Best |
| Interactive chatbot | Best | Usually not |
| Batch processing | Not ideal | Best |
| Traffic spikes | More difficult | Easier with queues |
| Client waiting | Yes | No |

**Key interview point:**

> I use **synchronous inference for latency-sensitive, interactive workloads** and **asynchronous inference for long-running, high-volume, or bursty workloads**, typically using a queue and worker pool to provide backpressure and reliable processing.

### 27.23 What is FSDP (Fully Sharded Data Parallel), and how does it differ from DeepSpeed ZeRO?

**FSDP (Fully Sharded Data Parallel)** is a distributed training technique from PyTorch that **shards model parameters, gradients, and optimizer states across GPUs** instead of keeping a full copy on every GPU.

##### FSDP

- Each GPU stores only a **shard** of the model state.
- During computation, required parameters are temporarily **all-gathered**.
- After computation, parameters can be released/sharded again.
- Gradients and optimizer states remain distributed.
- Significantly reduces **per-GPU memory usage**.
- Particularly useful for training large Transformer/LLM models.

##### FSDP vs DeepSpeed ZeRO

DeepSpeed **ZeRO (Zero Redundancy Optimizer)** uses a similar sharding philosophy but is part of the DeepSpeed ecosystem.

| Feature | FSDP | DeepSpeed ZeRO |
| --- | --- | --- |
| Framework | PyTorch-native | DeepSpeed |
| Parameters sharded | Yes | Yes, depending on ZeRO stage |
| Gradients sharded | Yes | Yes, depending on stage |
| Optimizer states sharded | Yes | Yes |
| Ease with PyTorch | Very good | Requires DeepSpeed integration |
| Training optimizations | Strong | Very extensive |
| CPU/NVMe offload | Supported | Strong support |
| Ecosystem | PyTorch | DeepSpeed |

##### ZeRO stages

- **ZeRO-1:** shard optimizer states.
- **ZeRO-2:** shard optimizer states + gradients.
- **ZeRO-3:** shard optimizer states + gradients + model parameters.

**Important distinction:** FSDP is conceptually closest to **ZeRO-3**, because both fully shard model parameters, gradients, and optimizer states across data-parallel workers.

**Key interview point:**

> FSDP and ZeRO solve the same fundamental problem—**eliminating redundant model state across GPUs to make large-model training possible**. FSDP is tightly integrated into PyTorch, while DeepSpeed ZeRO provides a broader optimization ecosystem with features such as aggressive memory optimization and CPU/NVMe offloading.

### 27.24 How do you monitor and profile LLM inference in production (TTFT, inter-token latency, GPU utilization)?

I monitor LLM inference at **three levels: application metrics, model-serving metrics, and GPU/system metrics**.

##### 1. Latency metrics

- **TTFT (Time to First Token):** time from request arrival until the first generated token.
  - High TTFT usually indicates **queueing or slow prefill**.
- **Inter-token latency (ITL):** time between generated tokens.
  - High ITL usually indicates **decode bottlenecks, GPU saturation, or inefficient batching**.
- **End-to-end latency:** total request duration.
- Track **p50, p95, and p99**, not just averages.

##### 2. Throughput metrics

- Tokens/second per request.
- Aggregate output tokens/second.
- Requests/second.
- Input and output tokens per request.
- Batch size and active sequences.

##### 3. GPU/system metrics

Monitor:

- GPU utilization.
- GPU memory utilization.
- KV-cache utilization.
- GPU memory bandwidth.
- Power/temperature.
- CPU utilization and memory.
- Network I/O.
- Queue depth and request concurrency.

##### 4. Profiling

I use serving/runtime metrics plus tools such as:

- **Prometheus + Grafana** for metrics and dashboards.
- **OpenTelemetry** for distributed traces.
- **NVIDIA Nsight Systems/Compute** for GPU/kernel profiling.
- Serving frameworks such as **vLLM or TensorRT-LLM** for inference-specific metrics.

I break the request into:

`Request → Queue → Prefill → Decode → Post-processing → Response`

This helps determine whether latency comes from **queueing, prompt processing, token generation, GPU utilization, or network/application overhead**.

##### 5. Production monitoring

I create dashboards and alerts for:

- p95/p99 TTFT
- p95/p99 ITL
- End-to-end latency
- Tokens/sec
- GPU utilization/memory
- KV-cache usage
- Queue depth
- Error/timeouts
- Cost per request/token

**Key interview point:**

> I correlate **TTFT, inter-token latency, queue time, throughput, GPU utilization, and KV-cache usage** to identify whether the bottleneck is scheduling, prefill, decode, memory, or infrastructure, then use GPU profiling to optimize the specific bottleneck.

### 27.25 What is model routing at the infrastructure level, and how do you route requests based on complexity and cost?

**Model routing** is an infrastructure-level mechanism that **selects the most appropriate AI model or inference backend for each request** based on factors such as complexity, latency, cost, availability, and capacity.

##### How I implement it

- **Classify request complexity**
  - Simple query → smaller/cheaper model.
  - Complex reasoning or coding → larger/more capable model.
  - Specialized task → domain-specific model.
  - High-volume batch workload → optimized/quantized model.
- **Use routing signals**
  - Prompt length and token count.
  - Task type and complexity score.
  - Required accuracy/reasoning level.
  - User/tenant SLA.
  - Current GPU utilization and queue depth.
  - Model latency and cost per token.
  - Provider availability.
- **Apply routing policies**
  - `Simple → Small model`
  - `Medium → Mid-size model`
  - `Complex → Large model`
  - `High priority → Low-latency model`
  - `Batch → Cheapest suitable model`
- **Use fallback routing**
  - If the preferred model is unavailable, overloaded, or exceeds latency limits, route to another suitable model/provider.
- **Optimize cost dynamically**
  - Prefer the **cheapest model that meets the required quality/SLA**, rather than always using the most powerful model.

##### Example architecture

`Client → API Gateway → Routing Layer → Model A / Model B / Model C → Response`

The routing layer can maintain a policy such as:

`complexity + SLA + cost + capacity → selected model`

##### Production considerations

- Track quality, latency, cost/request, and failure rate **per model**.
- Use A/B testing and evaluation datasets to validate routing decisions.
- Add rate limits, circuit breakers, and fallbacks.
- Avoid routing purely by price; a cheaper model that produces poor results can increase overall cost through retries or human review.

**Key interview point:**

> I implement model routing as a policy-based or learned decision layer that selects the **lowest-cost model capable of satisfying the request's quality and latency requirements**, while considering real-time capacity, failures, and SLA constraints.

### 27.26 What is quantization?

> **See also:** [Question 17.6 — What is Quantization?](../part-4-generative-ai/17-large-language-model-foundations.md#176-what-is-quantization), which covers the same question from another angle.

Reducing the numerical precision of a model's weights (e.g., from 32-bit to 8-bit or 4-bit) to shrink its size and speed up inference, usually with some trade-off in accuracy.

### 27.27 What are the main model compression techniques?

Compression shrinks models to cut **latency, memory, cost and energy**, which matters for edge deployment and for cheap large-scale serving.

1. **Quantisation:** lower the numeric precision of weights and activations (FP32 → FP16/BF16 → INT8 → INT4/FP8).
   - *Post-training quantisation (PTQ):* fast, needs a small calibration set. LLM methods include GPTQ, AWQ, and bitsandbytes NF4.
   - *Quantisation-aware training (QAT):* simulates quantisation during training for better accuracy at low bit widths.
   - INT8 gives about 4× smaller weights and often 2–4× speedup on supported hardware, usually for under 1% accuracy loss.
2. **Pruning:** remove weights that matter little.
   - *Unstructured* (individual weights): high sparsity, but needs sparse kernels to get a speedup.
   - *Structured* (whole channels, heads, layers): real speedups on standard hardware. 2:4 semi-structured sparsity is supported on NVIDIA Ampere+.
3. **Knowledge distillation:** train a small *student* to mimic a large *teacher's* soft outputs or logits (and intermediate representations). Examples: DistilBERT (40% smaller, about 97% of BERT's performance), and distilling LLMs into small task models.
4. **Low-rank factorisation:** approximate weight matrices with low-rank products (SVD). LoRA uses the same idea for efficient fine-tuning.
5. **Weight sharing and clustering:** cluster weights into a small codebook (Deep Compression).
6. **Efficient architecture design / NAS:** MobileNet (depthwise separable convolutions), EfficientNet, SqueezeNet, and mixture-of-experts for sparse compute.
7. **Runtime and graph optimisation:** operator fusion, constant folding, compilation (TensorRT, ONNX Runtime, TVM, `torch.compile`). Not compression as such, but usually applied alongside it.

**Practical recipe:** distil or pick an efficient architecture, apply structured pruning if needed, then quantise (PTQ first, QAT if accuracy drops), then compile for the target hardware. Always benchmark accuracy *and* latency on the real target device, and check per-slice accuracy, because compression can hurt rare classes disproportionately.

```python
import torch
qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

### 27.28 What are some methods to improve inference time?

#### Methods to Improve Inference Time

Common methods to reduce **model inference latency** include:

- **Model Quantization:** Reduce model precision, such as FP32 to FP16 or INT8, to make computation faster.
- **Model Pruning:** Remove unnecessary weights or parameters from the model.
- **Knowledge Distillation:** Train a smaller model (student) to reproduce the behavior of a larger model (teacher).
- **Smaller Model Architecture:** Use a model with fewer layers or parameters.
- **Batching:** Process multiple requests together to improve hardware utilization.
- **Caching:** Cache frequently used results or intermediate computations to avoid repeated work.
- **Efficient Hardware:** Use GPUs, TPUs, or specialized inference accelerators.
- **Optimized Inference Engines:** Use tools such as TensorRT, ONNX Runtime, or optimized serving frameworks.
- **Parallelism:** Run independent computations simultaneously.
- **Reduce Input Size:** Use shorter sequences, smaller images, or fewer unnecessary features.

### 27.29 What is the common idea behind GPUs and TPUs?

There is processor and there is memory. The amount of time it take to move a floating point number from memory to processor, with in that time we can perform hundreds of floating point operations in processor. So, we can perform batch processing.

### 27.30 What is the difference between static and continuous batching in LLMs?

Batching groups requests so the GPU processes them together, which raises throughput because LLM decoding is memory-bandwidth-bound (weights are loaded once per step for the whole batch).

**Static batching**

- Collect a batch of requests, run them together until **every** sequence finishes, then start the next batch.
- Problems: sequences have different output lengths, so short requests finish early and their slots sit idle (padding waste) while the longest request runs; new requests wait for the whole batch, increasing latency. GPU utilization is poor for chat workloads with highly variable lengths.
- Fine for offline jobs with similar lengths (bulk embedding, batch scoring).
- **Dynamic batching** is a variant: wait up to a short timeout to fill a batch, but it still runs batch-to-completion.

**Continuous batching (iteration-level / in-flight batching)**

- Scheduling happens **every decoding step**: as soon as a sequence emits its end token, it is evicted and a waiting request is inserted into the free slot.
- The batch stays full, giving much higher throughput (vLLM reported up to ~10–20x over naive static batching) and lower queueing latency.
- Requires careful **KV-cache memory management**, because sequences of different lengths come and go. **PagedAttention** (vLLM) allocates KV cache in fixed-size blocks to avoid fragmentation. Prefill of new requests is often split into chunks (**chunked prefill**) so it doesn't stall ongoing decodes.
- Implemented in vLLM, TGI, TensorRT-LLM, SGLang.

**Tradeoffs to mention:** larger batches increase throughput but also per-token latency (inter-token latency), so production systems cap batch size / tokens per step to meet p95 latency SLOs; scheduling policies (priority, fairness, preemption with KV swapping or recomputation) matter under load.

### 27.31 Why is AI infrastructure moving towards Rust?

Python remains the *interface* for AI, but more of the *infrastructure underneath* is being written in Rust:

- **Performance without a GC:** C/C++-level speed and predictable latency (no garbage-collection pauses), which matters for tokenizers, data loaders, inference servers and vector databases.
- **Memory safety:** the ownership model removes whole classes of bugs (use-after-free, data races) that plague C++ infra. This matters for long-running, multi-tenant services where a segfault means an outage or a security hole.
- **Fearless concurrency:** the compiler enforces thread safety, so heavy parallel work (tokenizing millions of docs, async request handling) is easier to get right than in C++. It also sidesteps Python's GIL.
- **Great Python interop:** PyO3 and maturin make it easy to ship Rust-backed Python packages, so users keep a Python API while the hot path runs in Rust.
- **Tooling and deployment:** Cargo, single static binaries, small containers, and WebAssembly targets for edge or in-browser inference.

**Real examples:** Hugging Face `tokenizers` and `safetensors`, Hugging Face TGI's router, `candle` (an ML framework), Polars (a DataFrame library), vector DBs (Qdrant, LanceDB), `uv`/`ruff` (Python tooling), and parts of OpenAI's and other vendors' serving stacks.

**Trade-offs:** a steeper learning curve, slower development than Python, a smaller ML ecosystem, and CUDA kernels are still mostly C++/CUDA (or Triton). So the realistic pattern is **Python for research and orchestration, Rust for performance-critical, safety-critical plumbing**, rather than a full rewrite.

### 27.32 How does prompt caching work, and when does it help?

**What it is:** during prefill an LLM computes **KV-cache tensors** for every prompt token. Prompt caching stores the KV cache for a prompt **prefix**, so a later request with the *exact same prefix* skips recomputing it and only processes the new suffix.

**How it works**

- Matching is **prefix-based and exact** (token-identical, usually in blocks of tokens). Any change early in the prompt invalidates everything after it.
- Hosted APIs (Anthropic, OpenAI, Gemini) cache automatically above a minimum length, or via explicit cache breakpoints. Cached input tokens are billed at a steep discount (about 10–50% of the normal price) with a TTL of minutes, extendable on some providers. Anthropic charges a small premium to write to the cache.
- Self-hosted: vLLM *automatic prefix caching* and SGLang *RadixAttention* keep KV blocks in GPU memory and share them across requests using a hash or radix tree of prefixes.

**When it helps**

- Long, **static** system prompts, tool definitions, few-shot examples.
- RAG or document Q&A where many questions hit the same large document.
- Multi-turn chat and **agents**, where each turn resends the growing history and only the tail is new.
- Effects: **lower TTFT** (often 50–80% for long prefixes) and **lower cost**.

**When it doesn't:** short prompts, prompts where variable content (timestamps, user names, retrieved chunks) comes *first*, low request rates where entries expire before reuse, and highly diverse prompts.

**Design tips:** order the prompt **static → semi-static → dynamic** (system prompt, tools, documents, then history, then the user query). Keep tool definitions and serialization deterministic (stable JSON key order). Track the cache hit rate as a metric.

**Don't confuse with:** *semantic caching*, which returns a stored **response** for a semantically similar query. That skips the model entirely but risks stale or wrong answers. Prompt caching still runs the model, so the output is unaffected.

### 27.33 What does TensorRT (TensorRT-LLM) do?

**TensorRT** is NVIDIA's SDK for **optimizing and running trained neural networks for inference on NVIDIA GPUs**. It takes a model (typically via ONNX or framework integrations such as Torch-TensorRT) and compiles it into an optimized **engine** for a specific GPU:

- **Graph optimizations:** layer and kernel **fusion** (e.g., conv + bias + ReLU in one kernel), constant folding, removing no-op layers.
- **Precision calibration:** FP16, BF16, INT8 (with calibration data), FP8 on Hopper/Blackwell, and INT4 weight-only.
- **Kernel auto-tuning:** benchmarks candidate kernels and picks the fastest for the target GPU, batch size and shapes.
- **Memory optimization:** reuses tensor memory, and supports dynamic shapes through optimization profiles.

**TensorRT-LLM** is the open-source library built on it specifically for **LLM inference**:

- Optimized Transformer kernels, fused attention (FlashAttention-style), **paged KV cache**, KV-cache quantization and reuse (prefix caching).
- **In-flight (continuous) batching**, chunked prefill, **speculative decoding** (draft models, Medusa, EAGLE), and disaggregated prefill/decode.
- Quantization (FP8, INT8 SmoothQuant, INT4 AWQ/GPTQ) and **tensor/pipeline parallelism** across multiple GPUs and nodes.
- Served via **Triton Inference Server** or NVIDIA NIM, with an OpenAI-compatible API.

**Trade-offs**

- Often delivers the best throughput and latency on NVIDIA hardware, frequently ahead of generic PyTorch serving.
- But: NVIDIA lock-in, engines are tied to a GPU type and TensorRT version (rebuild when either changes), a longer build and compile step, more operational complexity, and support for new model architectures can lag behind.
- Alternatives: **vLLM** and **SGLang** (easier to use, broad model support, close performance), ONNX Runtime, and llama.cpp for CPU/edge.

**When to use it:** high-volume, latency-sensitive production on fixed NVIDIA hardware where the extra engineering effort pays off in GPU cost.

### 27.34 What are the main techniques for optimizing LLM inference overall (latency, throughput, cost)?

First identify what you are optimizing: **latency** (TTFT, inter-token latency), **throughput** (tokens/s per GPU), or **cost** (\$ per 1M tokens). Prefill is compute-bound; decode is **memory-bandwidth-bound** (reading weights and the KV cache for every token).

**Model-level**

- **Quantization:** FP8/INT8 weights and activations, INT4 weight-only (AWQ/GPTQ), KV-cache quantization. Gives 2–4× less memory and faster decode for a small quality loss.
- **Smaller or distilled models**, and **routing** (simple queries to a small model, hard ones to a large model).
- **Architecture choices:** GQA/MQA (smaller KV cache), MoE (fewer active parameters per token), sliding-window attention.

**Serving and system-level**

- **Continuous (in-flight) batching:** raises GPU utilization dramatically compared with static batching.
- **PagedAttention:** near-zero KV-cache fragmentation, so more concurrent sequences fit.
- **Optimized kernels:** FlashAttention, fused ops, CUDA graphs (vLLM, SGLang, TensorRT-LLM).
- **Prefix / prompt caching:** reuse the KV cache of shared system prompts and documents.
- **Speculative decoding:** a draft model, Medusa/EAGLE heads or n-gram lookup proposes tokens and the large model verifies several at once. Gives 2–3× lower latency with identical output distribution.
- **Chunked prefill** and **disaggregated prefill/decode** on separate GPU pools, so long prompts don't stall ongoing decodes.
- **Parallelism:** tensor parallelism for latency, replicas (data parallelism) for throughput, pipeline/expert parallelism for very large models.

**Application-level**

- Shorter prompts (context budgeting, compression, fewer retrieved chunks), **output length limits**, structured outputs, stop sequences.
- **Semantic caching** of full responses, **streaming** to improve perceived latency, **batch APIs** (about 50% cheaper) for offline jobs.
- Parallelize tool calls and retrieval, and avoid unnecessary agent steps.

**Trade-offs to discuss:** bigger batches raise throughput but hurt per-request latency, so tune to the SLO. Quantization and distillation can cost quality, so evaluate on *your* tasks. Every optimization should be measured on a load test (p50/p95/p99 TTFT, tokens/s, GPU utilization, cost per request).

### 27.35 What is model routing?

Sending simple tasks to cheaper models and reserving frontier models only for tasks that actually need that capability.

### 27.36 What is prompt caching?

Reusing the processed, static prefix of a prompt across multiple calls instead of reprocessing it every time.

### 27.37 How does semantic caching differ from standard caching?

It caches responses by the meaning of the request rather than requiring an exact text match, so semantically similar queries can hit the cache.

### 27.38 What is batching used for?

Grouping requests for efficiency in workloads that don't require real-time responses.

### 27.39 What does working knowledge of Azure for AI/ML deployment typically involve?

Azure ML for training/experiment tracking and managed endpoints; Azure Kubernetes Service (AKS) or Container Apps for containerized service deployment; Azure OpenAI Service for hosted LLM access with enterprise compliance guarantees; Azure Blob Storage for data/artifact storage; Azure Cognitive Search for hybrid retrieval-as-a-service; and Azure Monitor/Application Insights for observability. Familiarity with IAM/RBAC for securing access to client data, and networking (VNETs, private endpoints) for deployments requiring data isolation — a common requirement in consulting engagements with sensitive client data.

### 27.40 How do you decide between using a managed cloud LLM/vector-DB service vs. self-hosting?

Factors: data sensitivity/compliance requirements (some clients require data to never leave their tenant, pushing toward self-hosted or VNET-isolated managed services), cost at scale (self-hosting can be cheaper at high volume but adds ops burden), latency requirements, team's operational capacity to maintain infrastructure, and time-to-market pressure (managed services are faster to stand up, important under the "short delivery cycles" constraint typical in consulting).

### 27.41 What is KV caching during inference?

In autoregressive decoding, each new token must attend to all previous tokens. A **KV cache** stores the **Key and Value tensors** of every previous token, for every layer and head, so they are computed once and reused instead of reprocessing the whole prefix at each step.

**How it works:**

1. **Prefill** — run the prompt through the model in parallel, compute K and V for all prompt tokens and store them.
2. **Decode** — for each new token, compute only its Q, K and V, append K and V to the cache, and attend $q_t$ against all cached keys and values.

Per-step attention cost drops from reprocessing $t$ tokens to processing one token against $t$ cached entries.

**Memory cost:**

$$
\text{KV bytes} = 2 \times L \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{seq len} \times \text{batch} \times \text{bytes}
$$

**Example:** LLaMA-2 7B (32 layers, 32 heads × 128, FP16) uses ~0.5 MB per token, so ~2 GB for one 4k-token sequence. At serving scale **the KV cache, not the weights, often limits batch size and context length.**

**Optimisations:** GQA/MQA (fewer KV heads), **PagedAttention** (vLLM, no fragmentation), KV quantization (FP8/INT8), prefix/prompt caching across requests, sliding-window eviction and offloading to CPU memory.

### 27.42 Why does KV cache significantly improve autoregressive generation?

#### KV Cache

**KV caching improves LLM inference by avoiding recomputation of previous tokens during autoregressive decoding.**

In the Transformer attention mechanism, each token produces **Key (K)** and **Value (V)** vectors. During generation:

- Without KV cache → the model repeatedly computes K/V for **all previous tokens** at every new token.
- With KV cache → previously computed K/V vectors are **stored in GPU memory** and reused.
- Only the **new token's K/V** needs to be computed.

#### Example

Generating: `I → I am → I am learning → I am learning AI`

When generating **AI**, the model reuses the cached K/V for: `I am learning`

and computes K/V only for **AI**.

#### Benefits

- **Lower decoding latency**
- Avoids redundant computation
- Improves **tokens/second**
- Especially important for long contexts and autoregressive generation

**Trade-off:** KV caching consumes significant **GPU memory**, which can become a bottleneck for large models and long context windows.

### 27.43 What is Grouped Query Attention (GQA)?

**Grouped Query Attention (GQA)** is a middle ground between multi-head (MHA) and multi-query attention (MQA). Query heads are split into $G$ groups, and **each group shares one Key/Value head**.

- MHA: $h$ query heads, $h$ KV heads.
- **GQA: $h$ query heads, $G$ KV heads** ($1 \lt G \lt h$).
- MQA: $h$ query heads, 1 KV head.

**Why:** during decoding, the bottleneck is **reading the KV cache from memory**, not FLOPs. The KV cache scales with the number of KV heads, so GQA cuts it by $h/G$.

**Example:** LLaMA-2 70B has 64 query heads and 8 KV heads, an **8× smaller KV cache**. That allows bigger batches and longer contexts on the same GPU, with quality close to full MHA.

**Implementation:** the K and V tensors are broadcast/repeated across the query heads within each group (`repeat_kv`), or the kernel indexes them directly. FlashAttention and vLLM support GQA natively.

**Uptraining:** Ainslie et al. (2023) showed you can convert an MHA checkpoint to GQA by **mean-pooling the KV heads** within each group and continuing training on ~5% of the original compute.

**Tradeoff:** fewer KV heads reduce representational capacity slightly. GQA recovers most of the quality lost by MQA, which is why it is the default in LLaMA-2/3, Mistral, Qwen and Gemma.

### 27.44 Compare MHA, MQA, and GQA.

| | **MHA** (multi-head) | **GQA** (grouped-query) | **MQA** (multi-query) |
| --- | --- | --- | --- |
| Query heads | $h$ | $h$ | $h$ |
| KV heads | $h$ | $G$ (e.g. 8) | 1 |
| KV cache size | 1× (baseline) | $G/h$ (e.g. 1/8) | $1/h$ |
| Decode speed | Slowest (memory-bound) | Fast | Fastest |
| Quality | Best | Close to MHA | Noticeably lower; can be unstable |
| Tensor-parallel fit | Easy | Good if $G \ge$ number of TP shards | KV head must be replicated on every shard |
| Examples | GPT-3, original Transformer, BERT | LLaMA-2/3 70B, Mistral, Qwen | PaLM, Falcon-7B, StarCoder |

**Why it matters:** in autoregressive inference each generated token reads the full KV cache, so decode throughput is **memory-bandwidth-bound**. Fewer KV heads means less memory traffic, larger batches and longer contexts.

**How to choose:**

- **Training a small or encoder model where inference cost is not critical** → MHA.
- **Serving an LLM at scale** → GQA is the sweet spot; pick $G$ as a multiple of your tensor-parallel degree.
- **Extreme memory constraints** → MQA, accepting some quality loss.

A further step is **MLA (Multi-head Latent Attention, DeepSeek-V2/V3)**, which compresses K/V into a low-rank latent per token, shrinking the cache even more while keeping per-head expressivity.

### 27.45 What is FlashAttention?

**FlashAttention** (Dao et al., 2022) is an **exact**, IO-aware GPU kernel for attention. It computes the same output as standard attention but much faster and with far less memory.

**Problem with standard attention:** it materialises the $n \times n$ score and probability matrices in GPU **HBM** (slow, large memory). It writes and reads them several times (matmul → mask → softmax → dropout → matmul), so it is **memory-bandwidth-bound** and uses $O(n^2)$ memory.

**Key ideas:**

- **Tiling** — split Q, K and V into blocks that fit in on-chip **SRAM**. Compute attention block by block, never storing the full matrix.
- **Online softmax** — keep a running max and running sum per row so softmax can be computed incrementally across blocks and rescaled correctly.
- **Kernel fusion** — matmul, mask, softmax, dropout and matmul run in one kernel.
- **Recomputation** — in the backward pass, recompute attention blocks instead of storing the $n \times n$ matrix (trading cheap FLOPs for expensive memory).

**Impact:** memory is $O(n)$ instead of $O(n^2)$, there is a 2–4× wall-clock speed-up, and long-context training becomes feasible.

**Versions:** FA-2 (better parallelism and work partitioning), FA-3 (Hopper-specific: asynchrony, FP8).

**In practice:** use it via `torch.nn.functional.scaled_dot_product_attention`, `flash-attn`, or inference engines (vLLM, TensorRT-LLM). It is now the default in essentially every serious LLM stack.

### 27.46 Why is FlashAttention faster without changing the attention result mathematically?

Because it changes **how data moves**, not **what is computed**. Attention on modern GPUs is limited by **memory bandwidth**, not arithmetic.

**Where the time goes in standard attention:**

- An A100 does ~312 TFLOPs (BF16) but reads HBM at only ~2 TB/s. On-chip SRAM is ~10× faster but only ~20 MB in total.
- Standard attention writes $S = QK^\top$ ($n \times n$) to HBM, reads it back for softmax, writes $P$, and reads it again for $PV$. For long sequences that traffic dominates the runtime.

**What FlashAttention does:**

1. **Tiling** — load blocks of Q, K and V into SRAM and do all the work there.
2. **Online softmax** — softmax needs the row max and row sum over *all* keys. FlashAttention keeps running statistics $m$ (max) and $\ell$ (sum). When a new block arrives with a larger max, previous partial results are rescaled by $e^{m_{\text{old}} - m_{\text{new}}}$. This is algebraically **identical** to the full softmax.
3. **Fusion** — one kernel, so intermediates never touch HBM.
4. **Recompute in backward** — cheaper than reading a stored $n \times n$ matrix.

**Result:** HBM accesses drop from $\Theta(nd + n^2)$ to $\Theta(n^2 d^2 / M)$ ($M$ = SRAM size). FLOPs actually *increase* slightly because of recomputation, yet wall-clock time falls 2–4×.

The output is exact up to floating-point rounding: results are not bit-identical because summation order changes, but the method is not an approximation.

### 27.47 What is PagedAttention?

**PagedAttention** (Kwon et al., 2023, the core of **vLLM**) manages the KV cache like an operating system manages **virtual memory pages**.

**Problem:** naïve servers pre-allocate one **contiguous** KV buffer per request sized for the maximum length (e.g. 2k–32k tokens). Because output length is unknown:

- **Internal fragmentation** — reserved space that is never used.
- **External fragmentation** — free gaps between buffers that are too small to reuse.
- The vLLM paper measured only **20–40%** of KV memory holding real tokens in existing systems.

**How PagedAttention works:**

- The KV cache is split into fixed-size **blocks** (e.g. 16 tokens each), stored anywhere in GPU memory.
- Each sequence has a **block table** that maps logical blocks to physical blocks, like a page table.
- Blocks are allocated **on demand** as tokens are generated; waste is limited to the last partial block (under 4%).
- The attention kernel is modified to gather K and V from non-contiguous blocks through the block table.
- **Copy-on-write sharing** — sequences with a common prefix (same system prompt, parallel sampling, beam search) share physical blocks with reference counts.

**Result:** near-zero waste, so many more concurrent sequences fit on the GPU. vLLM reported 2–4× throughput over prior systems (up to ~24× vs naïve HF Transformers).

### 27.48 How does PagedAttention improve LLM serving?

LLM decoding is memory-bound, so **throughput scales with how many sequences you can batch at once**, and batch size is limited by KV-cache memory. PagedAttention frees that memory.

**How it improves serving:**

- **Higher batch size → higher throughput.** Blocks are allocated on demand rather than reserved for the max length, so KV utilisation goes from ~20–40% to over 96%. More concurrent requests per GPU means a lower cost per token.
- **Pairs with continuous batching.** Requests join and leave the running batch at token granularity. Because memory is paged, a finished sequence's blocks are returned immediately and reused by new requests, with no compaction or copying.
- **Prefix sharing.** Shared system prompts, few-shot examples and RAG templates are stored once and referenced by many requests (copy-on-write). This is the basis for automatic **prefix caching**.
- **Parallel sampling and beam search** share the prompt's blocks across candidates, reducing memory by up to ~55% for those workloads.
- **Preemption and swapping.** Under memory pressure, whole blocks can be swapped to CPU memory or recomputed, which degrades gracefully instead of failing with OOM.
- **Predictable capacity planning** — you can compute how many tokens fit in the cache and set admission control on that.

**Tradeoff:** the indirection adds some kernel overhead (gathering non-contiguous blocks), which is small compared with the batching gains. vLLM, TensorRT-LLM, TGI and SGLang all use paged KV caches today.

### 27.49 Compare INT8, INT4, FP16, BF16, and FP8.

| Format | Bits | Layout | Range / precision | Typical use |
| --- | --- | --- | --- | --- |
| **FP16** | 16 | 1 sign, 5 exponent, 10 mantissa | Max ~65,504; good precision, **narrow range** | Inference; mixed-precision training (needs loss scaling) |
| **BF16** | 16 | 1 sign, **8 exponent**, 7 mantissa | Same range as FP32, lower precision | Default for LLM training and inference on A100/H100/TPU; no loss scaling needed |
| **FP8** | 8 | E4M3 (precision) or E5M2 (range) | Very coarse; needs per-tensor or per-block scaling | Training and inference on H100/Blackwell (Transformer Engine); W8A8 serving at ~2× BF16 throughput |
| **INT8** | 8 | Integer + scale (+ zero-point) | 256 levels, uniform spacing | PTQ for CNNs and LLMs (LLM.int8(), SmoothQuant); broad hardware support including CPUs and edge |
| **INT4** | 4 | Integer + per-group scale | 16 levels | Weight-only LLM inference (GPTQ, AWQ, GGUF Q4); 4× smaller than FP16 |

**Key tradeoffs:**

- **Range vs precision:** FP16 overflows (hence NaNs in training) while BF16 rarely does, at the cost of mantissa precision.
- **Floating vs integer:** FP8 handles outliers better than INT8 thanks to its exponent. Integer formats need careful scale selection.
- **Hardware support decides speed:** FP8 needs Hopper or newer. INT4 is mostly a *memory* win (weights are dequantized to FP16 for matmul) unless kernels are specialised.

**Rule of thumb:** train in BF16 (or FP8 at frontier scale), serve in FP8 or INT8 when quality matters and hardware supports it, and use INT4 weight-only to fit large models on limited GPUs, always validating on task evals.

### 27.50 What is quantization-aware training vs post-training quantization?

| | **Post-training quantization (PTQ)** | **Quantization-aware training (QAT)** |
| --- | --- | --- |
| When | After training, on a frozen model | During training or fine-tuning |
| Data | Small calibration set (~100–1000 samples) or none | Full or substantial training data |
| Cost | Minutes to hours | A training run (expensive for LLMs) |
| Accuracy | Good at 8-bit; can degrade at 4-bit or below | Best, especially at low bit-widths |
| Examples | Min-max / percentile calibration, GPTQ, AWQ, SmoothQuant | Fake-quant nodes with the straight-through estimator; QLoRA-style low-bit fine-tuning (adjacent) |

**How QAT works:** insert **fake-quantize** ops (quantize → dequantize) in the forward pass so the model "feels" the rounding error. Backprop uses the **straight-through estimator** (treating round() as identity for gradients), so the weights learn to be robust to quantization.

**How PTQ works:** run calibration data to collect activation ranges and choose scales and zero-points per tensor, channel or group. Advanced LLM PTQ methods minimise layer-wise reconstruction error (GPTQ uses second-order information) or protect salient weight channels (AWQ).

**Decision rule:**

- Start with **PTQ**. It is cheap, and INT8/FP8 or INT4 weight-only with GPTQ/AWQ is usually within ~1% on LLMs.
- Use **QAT** when PTQ loses too much accuracy: very low bit-widths (INT4 activations, 2–3 bit), small models (mobile CNNs, where each layer matters) or edge targets with strict integer-only hardware. Examples include Google's QAT Gemma checkpoints and MobileNet on-device.

Always compare against the FP16 baseline on the task's own evaluation, not just perplexity.

### 27.51 How would you reduce inference latency without significantly reducing accuracy?

I start by **profiling** where time goes (preprocessing, network, feature fetch, model compute, postprocessing), then apply the cheapest fix with the least accuracy loss.

**Model-level:**
- **Graph optimisation/compilation:** ONNX Runtime, TensorRT, `torch.compile`, operator fusion — often 2–5× with no accuracy loss.
- **Lower precision:** FP16/BF16 (essentially lossless), INT8 quantisation with calibration or QAT (usually < 1% loss).
- **Distillation:** train a smaller student on the teacher's outputs.
- **Pruning** (structured) and efficient architectures (e.g. DistilBERT, MobileNet).
- **Early exit / cascades:** cheap model handles easy cases, expensive one only for hard ones.

**LLM-specific:** KV caching, continuous batching, FlashAttention, PagedAttention (vLLM), speculative decoding, prompt caching, shorter prompts, streaming, model routing to smaller models.

**Serving/system-level:**
- Dynamic batching (latency/throughput tradeoff), warm instances, GPU/accelerator choice.
- Cache results and precompute features/embeddings.
- Reduce network hops (co-locate feature store), use gRPC, async I/O.
- Parallelise independent steps (feature fetch, retrieval).

**Process:** define latency budget (p99), benchmark on representative inputs, verify accuracy on the eval set after each optimisation, and roll out with canary. The usual order of payoff: system fixes and compilation → FP16 → INT8 → distillation.

### 27.52 What is speculative decoding?

**Speculative decoding** speeds up autoregressive generation without changing the output distribution. A small, fast **draft model** proposes several tokens ahead; the large **target model** checks them all in **one forward pass** and accepts the longest correct prefix.

**Algorithm (per step)**
1. The draft model generates $k$ candidate tokens $x_1, \dots, x_k$ autoregressively (cheap).
2. The target model runs once over the prefix plus all $k$ candidates, producing its distributions at every position in parallel.
3. Each token is accepted with probability $\min\left(1, \frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)}\right)$; at the first rejection, a replacement is sampled from the normalized residual $\max(0, p_{\text{target}} - p_{\text{draft}})$ and the rest are discarded.
4. If all $k$ are accepted, the target's next-token distribution gives a bonus token for free.

This rejection-sampling scheme guarantees the output is **distributed exactly as if sampled from the target model alone** (with greedy decoding: identical output).

**Variants**
- **Self-speculative / Medusa:** extra decoding heads on the target model propose tokens; no separate draft model.
- **EAGLE:** drafts at the feature level for higher acceptance rates.
- **Lookahead decoding / n-gram (prompt lookup):** draft tokens are copied from the prompt — great for summarization, code edits and RAG.

**Requirements:** the draft must share the tokenizer and agree with the target often (acceptance rate $\alpha$ typically 0.6–0.8). Supported in vLLM, TensorRT-LLM, SGLang and TGI.

### 27.53 How can speculative decoding reduce LLM inference latency?

Decoding in a large LLM is **memory-bandwidth bound**: each new token requires streaming all the weights (and the KV cache) from HBM to compute just one token, so the GPU's arithmetic units sit mostly idle. Speculative decoding converts that idle compute into extra tokens.

**Why it's faster**
- Verifying $k$ draft tokens in one target forward pass costs about the same wall-clock time as generating **one** token (the weights are read once; the extra FLOPs are nearly free when memory bound).
- If on average $\bar{a}$ draft tokens are accepted, each expensive target pass yields $\bar{a}+1$ tokens instead of 1.
- Expected tokens per target pass with acceptance rate $\alpha$ and draft length $k$: $\frac{1-\alpha^{k+1}}{1-\alpha}$. With $\alpha = 0.8$ and $k = 4$ that is ~3.4 tokens per pass → typical **2–3× lower latency**.
- The output is unchanged (exact under rejection sampling), so there is no quality trade-off.

**When it helps / hurts**
- Best at **low batch sizes / latency-sensitive** serving, where the GPU has spare compute.
- Gains shrink at high batch sizes (the GPU is already compute-bound) and with low acceptance rates (creative, high-temperature sampling, or a mismatched draft model).
- Costs: memory and compute for the draft model, and engineering complexity in batching and KV-cache management.

**Production tuning:** choose a draft model ~10–50× smaller from the same family; tune $k$ (3–8); monitor acceptance rate per workload; use n-gram/prompt-lookup drafting for input-grounded tasks (RAG, code editing), where acceptance is very high.

### 27.54 What is continuous batching?

**Continuous (in-flight / iteration-level) batching** schedules LLM requests **per decoding step** instead of per batch.

**Problem with static batching:** a batch of requests starts together and the batch is held until the *longest* sequence finishes. Short requests sit idle, finished slots are padding, new requests wait in the queue → poor GPU utilization and high latency.

**Continuous batching**
- After every forward step (one token), the scheduler **removes finished sequences and inserts new requests** into the free slots.
- The batch composition changes every iteration, so the GPU always works on a full batch of *active* sequences.
- New requests' **prefill** can be interleaved or chunked alongside other requests' **decode** steps (chunked prefill) to avoid latency spikes.

**Enablers**
- **PagedAttention (vLLM):** the KV cache is allocated in fixed-size blocks, so sequences of varying lengths can join and leave without memory fragmentation.
- A scheduler with policies for admission, preemption (swap or recompute KV when memory runs out) and priorities.

**Impact:** throughput gains of up to ~10–20× over naive static batching (per the Orca/vLLM papers), lower queueing latency, and better time-to-first-token for short requests.

**Trade-offs:** larger batches raise throughput but also per-token latency (TPOT); tune max batch size and max batched tokens against your SLOs (p95 TTFT, tokens/sec). Implemented in vLLM, TGI, TensorRT-LLM and SGLang.

### 27.55 Why is GPU memory often the bottleneck in LLM inference?

Two things must live in GPU memory (HBM) during inference, and both are large:

**1. Model weights**
- $\text{Memory} \approx \text{params} \times \text{bytes per param}$: a 70B model needs ~140 GB in FP16 — more than one 80 GB GPU — so it needs tensor parallelism or quantization.

**2. KV cache** — grows with batch size × sequence length:

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{seq len} \times \text{batch} \times \text{bytes}
$$

- Llama-2-70B (80 layers, 8 KV heads with GQA, $d_{\text{head}} = 128$, FP16): ~0.33 MB per token → a 4k context is ~1.3 GB per sequence. Thirty concurrent requests consume tens of GB.
- The KV cache limits **how many concurrent requests (batch size)** you can serve, which directly caps throughput.

**3. Bandwidth, not just capacity**
- During decode, each token requires reading **all weights plus the KV cache** from HBM, while doing only ~2 FLOPs per parameter. Arithmetic intensity is tiny, so $\text{token speed} \approx \text{memory bandwidth} / \text{bytes read}$. An H100 has ~3.35 TB/s HBM but ~1,000 TFLOPS of BF16 compute — compute sits idle.

**Mitigations**
- **Quantization** (INT8/INT4/FP8) of weights and the KV cache.
- **GQA/MQA** to shrink the KV cache; **PagedAttention** to avoid fragmentation; prefix caching to share common prompts.
- **Batching** to amortize weight reads over many sequences; **speculative decoding** to get more tokens per weight read.
- Tensor/pipeline parallelism across GPUs; KV offloading for long contexts.

### 27.56 Compare compute-bound vs memory-bandwidth-bound workloads.

The **roofline model** classifies a workload by its **arithmetic intensity**: $\text{arithmetic intensity} = \text{FLOPs performed} / \text{bytes moved from memory}$.

| | Compute-bound | Memory-bandwidth-bound |
| --- | --- | --- |
| Bottleneck | ALU / tensor-core throughput | Moving data from HBM/DRAM |
| Arithmetic intensity | High (above the hardware's ridge point) | Low |
| Examples | Large matmuls, training with big batches, **LLM prefill**, convolutions | **LLM decode at small batch**, elementwise ops, LayerNorm, softmax, embedding lookups |
| Faster if you… | add FLOPS, use lower precision math (FP8 tensor cores), better kernels | reduce bytes (quantization, fusion), increase reuse (batching), more bandwidth (HBM3) |

**Ridge point:** $\text{peak FLOPS} / \text{bandwidth}$. An H100 has ~990 TFLOPS BF16 and ~3.35 TB/s, so $990 \times 10^{12} / 3.35 \times 10^{12} \approx 300$ FLOPs/byte. Below that, you are memory bound.

**LLM example**
- **Prefill** processes all prompt tokens at once: matrix-matrix products with high intensity → compute-bound.
- **Decode** at batch 1 is a matrix-vector product: each weight (2 bytes) is used for ~2 FLOPs → intensity $\approx 2 / 2 = 1$ → heavily memory-bound. Increasing the batch raises intensity linearly until it hits the compute roof (often at batch ~100–300).

**Practical implications**
- Optimizations must match the bottleneck: kernel fusion (FlashAttention) and quantization help memory-bound phases; FP8 and better parallelism help compute-bound ones.
- This is why **disaggregated prefill/decode** serving puts the two phases on differently tuned hardware or pools.
- Profile (Nsight, PyTorch profiler) before optimizing: check achieved bandwidth vs achieved FLOPS.

### 27.57 What is the difference between training FLOPs and inference FLOPs?

For a dense Transformer with $N$ parameters:

**Training FLOPs** $\approx 6ND$ ($D$ = training tokens)
- Forward pass $\approx 2N$ FLOPs per token (one multiply plus one add per parameter).
- Backward pass $\approx 4N$ per token (gradients with respect to both activations and weights).
- Example: a 7B model on 2T tokens: $6 \times 7 \times 10^{9} \times 2 \times 10^{12} \approx 8.4 \times 10^{22}$ FLOPs — a one-time cost, but enormous.

**Inference FLOPs** $\approx 2N$ **per generated token** (forward only)
- Plus attention cost that grows with context: $\approx 2 \cdot n_{\text{layers}} \cdot n_{\text{ctx}} \cdot d_{\text{model}}$ per token (more for long contexts).
- Generating 1,000 tokens with a 7B model: $2 \times 7 \times 10^{9} \times 1000 \approx 1.4 \times 10^{13}$ FLOPs.

**Key differences**
- **Scale vs frequency:** training is a one-off cost; inference is paid per request, forever. For a popular product, cumulative inference compute can exceed training compute — which is why teams over-train smaller models (beyond Chinchilla-optimal).
- **Bottleneck:** training is mostly compute-bound (big batches, high utilization, MFU 40–60%). Inference decode is usually **memory-bandwidth bound**, so FLOPs alone underestimate latency; prefill is compute-bound.
- **Memory:** training also stores optimizer states, gradients and activations (~16–20 bytes per param with Adam in mixed precision); inference stores weights plus the KV cache.
- **MoE:** use *active* parameters in both formulas.

**Use in practice:** estimate $\text{training time} = \frac{\text{FLOPs}}{\text{GPUs} \times \text{peak FLOPS} \times \text{MFU}}$; estimate inference cost from $\text{tokens/day} \times 2N$, then sanity-check against bandwidth limits.

### 27.58 How would you estimate the GPU requirements for serving an LLM?

Work through **memory first, then throughput**, then validate with a benchmark.

**1. Weights:** $\text{params} \times \text{bytes}$. A 70B model: FP16 ≈ 140 GB, FP8/INT8 ≈ 70 GB, INT4 ≈ 35–40 GB.

**2. KV cache per token:** $2 \times n_{layers} \times n_{\text{kv heads}} \times d_{head} \times \text{bytes}$.
Llama-3-70B (80 layers, 8 KV heads, $d_{\text{head}} = 128$, FP16): $2 \times 80 \times 8 \times 128 \times 2 \approx 320$ KB per token.
**Total KV:** $\text{KV}_{\text{total}} = \text{KV per token} \times \text{avg context length} \times \text{concurrent sequences}$. For 32 concurrent users at 8k tokens: $32 \times 8192 \times 320\ \text{KB} \approx 84\ \text{GB}$.

**3. Overhead:** activations, CUDA graphs, fragmentation → leave ~10–20% headroom (vLLM's `gpu_memory_utilization` ≈ 0.9).

**4. Memory total → GPU count:** $140 + 84 + 20 \approx 244$ GB → 4× H100 80 GB (tensor-parallel 4), or 2× H100 with FP8 weights and a smaller KV budget.

**5. Throughput check (decode is bandwidth-bound):**
- $\text{tokens/s per sequence} \approx \frac{\text{effective bandwidth}}{\text{bytes read per step}}$. With TP=4 (~13 TB/s aggregate) reading ~140 GB → ~60–90 tokens/s at batch 1 (in practice lower).
- Batching amortizes weight reads: aggregate throughput rises nearly linearly with batch size until compute-bound or out of KV memory.
- $\text{required replicas} = \frac{\text{peak requests/s} \times \text{avg output tokens}}{\text{tokens/s per replica}}$.

**6. Latency SLOs:** TTFT (prefill, compute-bound; scales with prompt length) and TPOT (decode). Long prompts may need chunked prefill or separate prefill nodes.

**7. Validate:** load test with real prompt/response length distributions (vLLM benchmark, GenAI-Perf), measure p95 TTFT/TPOT, then add autoscaling and ~30% headroom for peaks and failover.

**Levers:** quantization, GQA models, prefix caching, speculative decoding, smaller distilled models, routing easy traffic to cheaper models.

### 27.59 What happens to model quality when you aggressively quantize a model?

Quality degrades **gracefully down to ~8 bits, noticeably at 4 bits, and sharply below that** — but how much depends on the method, the model size and the task.

**Typical pattern (weight-only quantization)**
- **FP16/BF16 → INT8/FP8:** near-lossless (<1% on benchmarks) with decent calibration.
- **INT4 (GPTQ, AWQ, NF4):** small perplexity increase; usually acceptable for chat, a few points lost on reasoning/math/code.
- **3-bit and below:** perplexity rises quickly; models become repetitive, lose factual recall, follow instructions badly, and can produce incoherent output (unless you use specialized methods such as QuIP#/AQLM, or quantization-aware training).

**Why it hurts**
- **Outliers:** a few activation channels and weights have huge magnitudes; uniform low-bit grids clip them or waste resolution (hence LLM.int8(), SmoothQuant, AWQ).
- **Error accumulation** across many layers and long generations.
- **Activation and KV-cache quantization** is harder than weight quantization; KV quantization can hurt long-context retrieval.
- **Small models suffer more** than large ones (less redundancy).
- Benchmark averages hide damage on **tail capabilities**: multilingual, long context, precise arithmetic, tool-call formatting, and safety behavior.

**Production practice**
- Evaluate the quantized model on **your own task eval set**, not just perplexity; check structured-output validity and refusal behavior.
- Prefer calibrated methods (AWQ/GPTQ) with representative calibration data; keep sensitive layers (embeddings, LM head, first/last layers) in higher precision.
- Compare "bigger model at 4-bit" vs "smaller model at 16-bit" for the same memory — the former often wins.
- Consider FP8 on H100-class GPUs as a near-free default.

### 27.60 How would you design production LLM serving?

#### Production LLM Serving Architecture

**Client → API Gateway → LLM Router → Inference Servers → Model**

Key components:

- **API Gateway:** Authentication, rate limiting, request validation.
- **LLM Router:** Load balancing, model selection, fallback, and request prioritization.
- **Inference:** vLLM/TGI with **continuous batching, KV caching, PagedAttention, and quantization**.
- **GPU layer:** Multiple GPU replicas with autoscaling based on traffic and GPU utilization.
- **Streaming:** Return tokens progressively using SSE/WebSocket.
- **Caching:** Cache repeated prompts/responses where appropriate.
- **Observability:** Track latency (p50/p95/p99), tokens/sec, GPU utilization, errors, token usage, and cost.
- **Reliability:** Timeouts, retries, circuit breakers, health checks, and graceful degradation.
- **Security:** Authentication, authorization, PII protection, prompt-injection controls, and tenant isolation.

### 27.61 Explain KV cache, PagedAttention, FlashAttention, GQA and quantization.

- **KV Cache:** Stores past **Key/Value states** during generation so they don't need to be recomputed.
- **PagedAttention:** Organizes KV-cache memory into fixed-size **blocks/pages**, allowing efficient allocation and sharing.
- **FlashAttention:** Uses **tiling and fused GPU operations** to reduce memory I/O during attention computation.
- **GQA (Grouped Query Attention):** Multiple Q heads share fewer K/V heads, reducing KV-cache memory while retaining much of MHA's quality.
- **Quantization:** Converts weights/activations to lower precision such as **INT8 or INT4**, reducing memory and compute requirements.

### 27.62 What is LLM inference? Explain prefill vs. decode.

LLM inference is the process of generating tokens from a trained language model.

It is generally divided into **prefill** and **decode** phases.

#### Prefill

The model processes the entire input prompt in parallel.

```txt
Prompt → Transformer → KV Cache
```

Prefill is generally **compute-intensive** because many prompt tokens are processed together.

#### Decode

The model generates output **one token at a time**.

```txt
Token 1 → Token 2 → Token 3 → ...
```

The KV cache stores previously computed attention information, avoiding recomputing the entire sequence.

Decode is often **memory-bandwidth and latency sensitive**.

#### Key metrics

**Time to First Token (TTFT):** Measures how long it takes to generate the first token.

**Time Per Output Token (TPOT):** Measures generation latency per token.

**Throughput:** Often measured in tokens/second.

Optimizing LLM serving requires balancing **prefill throughput, decode latency, KV-cache memory, batching, GPU utilization, and concurrency**.

---

← [Chapter 26. MLOps and LLMOps](26-mlops-and-llmops.md) · [Table of contents](../README.md) · [Chapter 28. Machine Learning System Design](28-machine-learning-system-design.md) →
