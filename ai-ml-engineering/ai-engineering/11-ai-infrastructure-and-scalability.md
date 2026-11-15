# AI Infrastructure and Scalability

Interview Q&A on LLM inference optimization, GPUs, parallelism, batching, KV cache, quantization, scaling, scheduling, and model routing.

### Q1. How do you improve inference speed in production LLM deployments?

> 🔁 _See also [09-large-language-model.md → Q37](../ml-engineering/09-large-language-model.md#q37-how-do-you-improve-inference-speed-in-production-llm-deployments) (separate notes on the same question)._

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

### Q2. LLM optimization techniques

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

### Q3. How do you select GPUs for LLM inference?

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

### Q4. What is model parallelism vs data parallelism in distributed training?

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

### Q5. What is tensor parallelism, and how does it help serve large models?

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

### Q6. What is pipeline parallelism?

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

### Q7. How does continuous batching improve LLM inference throughput?

> 🔁 _See also [09-large-language-model.md → Q10](../ml-engineering/09-large-language-model.md#q10-how-does-continuous-batching-improve-llm-inference-throughput) (separate notes on the same question)._

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

### Q8. What is speculative decoding, and how does it speed up inference?

> 🔁 _See also [09-large-language-model.md → Q9](../ml-engineering/09-large-language-model.md#q9-what-is-speculative-decoding-and-how-does-it-speed-up-inference) (separate notes on the same question)._

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

### Q9. What is KV cache, and how do you manage memory for it?

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

### Q10. What is Paged Attention?

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

### Q11. How does GGUF work?

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

### Q12. How do you optimize inference for edge and mobile deployment?

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

### Q13. What is model quantization (INT8, INT4, FP16, BF16), and how does it affect quality?

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

### Q14. How do you implement auto-scaling for AI workloads?

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

### Q15. What is the role of load balancing in AI serving infrastructure?

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

### Q16. How do you manage GPU memory for serving multiple models?

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

### Q17. What is model sharding, and when would you use it?

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

### Q18. How do you implement request queuing and priority scheduling for AI services?

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

### Q19. What are the cost trade-offs between self-hosted and API-based AI inference?

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

### Q20. How do you handle cold start latency for serverless AI deployments?

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

### Q21. How do you implement model caching to reduce redundant computations?

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

### Q22. What is the difference between synchronous and asynchronous inference, and when do you use each?

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

### Q23. What is FSDP (Fully Sharded Data Parallel), and how does it differ from DeepSpeed ZeRO?

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

### Q24. How do you monitor and profile LLM inference in production (TTFT, inter-token latency, GPU utilization)?

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

### Q25. What is model routing at the infrastructure level, and how do you route requests based on complexity and cost?

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
