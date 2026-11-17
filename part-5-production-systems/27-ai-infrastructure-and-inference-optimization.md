# Chapter 27. AI Infrastructure and Inference Optimization

_Part V — Production Systems and Architecture_

GPUs and accelerators, quantization, KV-cache management, batching, speculative decoding, parallelism and MoE serving, prefill-decode disaggregation, inference engines, Kubernetes GPU operations, caching, reliability and cost.

This chapter contains **45 questions**.

## Contents

- **27.1** [How do you optimise LLM inference in production for latency, throughput and cost?](#271-how-do-you-optimise-llm-inference-in-production-for-latency-throughput-and-cost)
- **27.2** [How do you choose GPUs or accelerators for LLM inference?](#272-how-do-you-choose-gpus-or-accelerators-for-llm-inference)
- **27.3** [What is model parallelism vs data parallelism in distributed training?](#273-what-is-model-parallelism-vs-data-parallelism-in-distributed-training)
- **27.4** [How do you split a large model across GPUs for serving: tensor parallelism, pipeline parallelism and model sharding?](#274-how-do-you-split-a-large-model-across-gpus-for-serving-tensor-parallelism-pipeline-parallelism-and-model-sharding)
- **27.5** [What is the difference between static, dynamic and continuous batching for LLM serving?](#275-what-is-the-difference-between-static-dynamic-and-continuous-batching-for-llm-serving)
- **27.6** [How does speculative decoding work, and when does it help in production?](#276-how-does-speculative-decoding-work-and-when-does-it-help-in-production)
- **27.7** [How do you manage KV-cache memory when serving LLMs?](#277-how-do-you-manage-kv-cache-memory-when-serving-llms)
- **27.8** [What is PagedAttention, and how does it improve LLM serving?](#278-what-is-pagedattention-and-how-does-it-improve-llm-serving)
- **27.9** [How does GGUF work?](#279-how-does-gguf-work)
- **27.10** [How do you optimize inference for edge and mobile deployment?](#2710-how-do-you-optimize-inference-for-edge-and-mobile-deployment)
- **27.11** [What is quantization, and how does it affect model quality, especially when it is aggressive?](#2711-what-is-quantization-and-how-does-it-affect-model-quality-especially-when-it-is-aggressive)
- **27.12** [How do you implement auto-scaling for AI workloads?](#2712-how-do-you-implement-auto-scaling-for-ai-workloads)
- **27.13** [What is the role of load balancing in AI serving infrastructure?](#2713-what-is-the-role-of-load-balancing-in-ai-serving-infrastructure)
- **27.14** [How do you manage GPU memory for serving multiple models?](#2714-how-do-you-manage-gpu-memory-for-serving-multiple-models)
- **27.15** [How do you implement request queuing and priority scheduling for AI services?](#2715-how-do-you-implement-request-queuing-and-priority-scheduling-for-ai-services)
- **27.16** [How do you compare the cost of self-hosting LLM inference with using an API, and how do you plan capacity?](#2716-how-do-you-compare-the-cost-of-self-hosting-llm-inference-with-using-an-api-and-how-do-you-plan-capacity)
- **27.17** [How do you handle cold-start latency for serverless and autoscaled GPU inference?](#2717-how-do-you-handle-cold-start-latency-for-serverless-and-autoscaled-gpu-inference)
- **27.18** [How do you use caching to avoid redundant computation in AI serving, including semantic caching?](#2718-how-do-you-use-caching-to-avoid-redundant-computation-in-ai-serving-including-semantic-caching)
- **27.19** [What is the difference between synchronous and asynchronous inference, and when do you use each?](#2719-what-is-the-difference-between-synchronous-and-asynchronous-inference-and-when-do-you-use-each)
- **27.20** [What is FSDP (Fully Sharded Data Parallel), and how does it differ from DeepSpeed ZeRO?](#2720-what-is-fsdp-fully-sharded-data-parallel-and-how-does-it-differ-from-deepspeed-zero)
- **27.21** [How do you monitor and profile LLM inference in production (TTFT, inter-token latency, GPU utilization)?](#2721-how-do-you-monitor-and-profile-llm-inference-in-production-ttft-inter-token-latency-gpu-utilization)
- **27.22** [What is model routing at the infrastructure level, and how do you route requests based on complexity and cost?](#2722-what-is-model-routing-at-the-infrastructure-level-and-how-do-you-route-requests-based-on-complexity-and-cost)
- **27.23** [What are the main model compression techniques?](#2723-what-are-the-main-model-compression-techniques)
- **27.24** [How do GPUs, TPUs and other AI accelerators differ, and when would you use each?](#2724-how-do-gpus-tpus-and-other-ai-accelerators-differ-and-when-would-you-use-each)
- **27.25** [Why is AI infrastructure moving towards Rust?](#2725-why-is-ai-infrastructure-moving-towards-rust)
- **27.26** [How does prompt caching work, and when does it help?](#2726-how-does-prompt-caching-work-and-when-does-it-help)
- **27.27** [What does TensorRT (TensorRT-LLM) do?](#2727-what-does-tensorrt-tensorrt-llm-do)
- **27.28** [What does deploying AI and ML workloads on Azure typically involve?](#2728-what-does-deploying-ai-and-ml-workloads-on-azure-typically-involve)
- **27.29** [How do MHA, MQA, GQA and MLA compare, and why does the choice matter for serving?](#2729-how-do-mha-mqa-gqa-and-mla-compare-and-why-does-the-choice-matter-for-serving)
- **27.30** [What is FlashAttention, and why is it faster without changing the result?](#2730-what-is-flashattention-and-why-is-it-faster-without-changing-the-result)
- **27.31** [Compare INT8, INT4, FP16, BF16, and FP8.](#2731-compare-int8-int4-fp16-bf16-and-fp8)
- **27.32** [What is quantization-aware training vs post-training quantization?](#2732-what-is-quantization-aware-training-vs-post-training-quantization)
- **27.33** [How would you reduce inference latency without significantly reducing accuracy?](#2733-how-would-you-reduce-inference-latency-without-significantly-reducing-accuracy)
- **27.34** [Why is GPU memory often the bottleneck in LLM inference?](#2734-why-is-gpu-memory-often-the-bottleneck-in-llm-inference)
- **27.35** [Compare compute-bound vs memory-bandwidth-bound workloads.](#2735-compare-compute-bound-vs-memory-bandwidth-bound-workloads)
- **27.36** [What is the difference between training FLOPs and inference FLOPs?](#2736-what-is-the-difference-between-training-flops-and-inference-flops)
- **27.37** [How would you estimate the GPU requirements for serving an LLM?](#2737-how-would-you-estimate-the-gpu-requirements-for-serving-an-llm)
- **27.38** [How would you design a production LLM serving stack?](#2738-how-would-you-design-a-production-llm-serving-stack)
- **27.39** [What are the prefill and decode phases, and how do serving systems schedule them?](#2739-what-are-the-prefill-and-decode-phases-and-how-do-serving-systems-schedule-them)
- **27.40** [What is prefill-decode disaggregation, and when is it worth it?](#2740-what-is-prefill-decode-disaggregation-and-when-is-it-worth-it)
- **27.41** [How do you serve mixture-of-experts (MoE) models efficiently?](#2741-how-do-you-serve-mixture-of-experts-moe-models-efficiently)
- **27.42** [What does structured output (constrained decoding) cost in latency and quality, and how do you make it fast?](#2742-what-does-structured-output-constrained-decoding-cost-in-latency-and-quality-and-how-do-you-make-it-fast)
- **27.43** [How do you run GPU inference on Kubernetes and keep utilisation high?](#2743-how-do-you-run-gpu-inference-on-kubernetes-and-keep-utilisation-high)
- **27.44** [How do you choose an LLM inference engine, and how do you benchmark it fairly?](#2744-how-do-you-choose-an-llm-inference-engine-and-how-do-you-benchmark-it-fairly)
- **27.45** [What failure modes do you see in production GPU inference, and how do you design for them?](#2745-what-failure-modes-do-you-see-in-production-gpu-inference-and-how-do-you-design-for-them)

---

### 27.1 How do you optimise LLM inference in production for latency, throughput and cost?

Start by deciding which number you are optimising, because the techniques pull in different directions: **latency** (time to first token, inter-token latency), **throughput** (tokens per second per GPU) or **cost** (dollars per million tokens). Then find the bottleneck: prefill is compute-bound, decode is memory-bandwidth-bound (every generated token re-reads the weights and the KV cache), and queueing often dominates both under load. The model-level mechanics of these techniques are in Chapter 17 (see 17.106); this answer is the serving playbook.

**Metrics to agree first**

| Metric | What it tells you | Typical driver |
| --- | --- | --- |
| TTFT (time to first token) | Queueing plus prefill | Prompt length, queue depth, prefill scheduling |
| ITL / TPOT (inter-token latency) | Decode speed per stream | Batch size, memory bandwidth, model size |
| Output tokens/s per GPU | Fleet efficiency | Batching, KV-cache capacity, quantisation |
| Cost per 1M tokens | Unit economics | All of the above plus utilisation |
| Goodput | Requests/s that meet the SLO | The number to maximise, not raw throughput |

**1. Reduce the work (application level, usually the cheapest wins)**

- Send fewer tokens: trim system prompts, retrieve fewer and better chunks, summarise long histories, cap `max_tokens`, use stop sequences and structured outputs.
- Route easy requests to a small model and only hard ones to a large model (see 27.22).
- Cache: prefix/prompt caching for shared prefixes (see 27.26) and response or semantic caching where answers repeat (see 27.18).
- Stream tokens to improve perceived latency; use batch APIs or offline batch jobs for anything that is not interactive.
- In agents, parallelise tool calls and retrieval and remove unnecessary LLM steps.

**2. Make each token cheaper (model level)**

- **Quantisation:** FP8 or INT8 weights and activations on hardware that supports them, INT4 weight-only (AWQ, GPTQ) to fit large models, and KV-cache quantisation (see 27.11 and 27.7).
- **Smaller or distilled models**, fine-tuned for the task, often beat a larger general model on cost per correct answer.
- **Architecture:** GQA or MLA models have much smaller KV caches (see 27.29); mixture-of-experts models activate only a fraction of parameters per token (see 27.41 for serving them).

**3. Use the GPU fully (serving level)**

- **Continuous batching** (see 27.5) and **paged KV cache** (see 27.8) are the two biggest throughput wins over naive serving.
- **Optimised kernels:** FlashAttention (see 27.30), fused operators, CUDA graphs. You get these by using a modern engine (vLLM, SGLang, TensorRT-LLM) rather than hand-rolled PyTorch.
- **Speculative decoding** (see 27.6) for latency at low to moderate batch sizes.
- **Chunked prefill**, and at larger scale **prefill-decode disaggregation** (see 27.40), so long prompts do not stall ongoing decodes.
- **Parallelism:** tensor parallelism to cut per-token latency or fit a model, replicas for throughput, expert parallelism for large MoE models (see 27.4).

**4. Run the fleet well (infrastructure level)**

- Pick hardware by memory capacity and bandwidth, not peak FLOPS (see 27.2).
- Autoscale on queue depth and KV-cache usage, not CPU (see 27.12); route with cache and load awareness (see 27.13).
- Keep utilisation high: idle GPUs are the largest hidden cost in self-hosting (see 27.16).

**Trade-offs to state explicitly**

- Bigger batches raise throughput but raise inter-token latency, so tune `max_num_seqs` and max batched tokens against the p95 SLO.
- Quantisation and distillation can cost quality on tail tasks, so evaluate on your own task set, not only perplexity.
- Speculative decoding helps at low load and can hurt at high load.
- Each optimisation must be measured with a load test that replays realistic prompt and output length distributions (see 27.44).

**Typical order of payoff:** cut tokens and add caching, then move to a modern engine with continuous batching and paged KV cache, then FP8 or INT8, then right-size hardware and parallelism, then speculative decoding and disaggregation for the last stretch of latency or cost.

### 27.2 How do you choose GPUs or accelerators for LLM inference?

Choose by **memory capacity first, memory bandwidth second, interconnect third, and price per useful token last**, not by peak TFLOPS. Capacity decides whether the model and enough KV cache fit; bandwidth decides decode speed; interconnect decides whether multi-GPU serving scales; utilisation decides what you actually pay.

**Step 1: does it fit?**

- Weights: parameters times bytes per parameter (70B is about 140 GB in BF16, 70 GB in FP8, about 38 GB in INT4 including scales).
- KV cache: per-token size times context length times concurrent sequences (see 27.7 and 27.37 for the arithmetic).
- Leave 10 to 20 percent for activations, CUDA graphs and fragmentation.
- If weights plus a useful KV budget do not fit on one device, you need tensor parallelism across GPUs joined by NVLink (see 27.4), or a bigger-memory part.

**Step 2: how fast can it decode?**

At small batch sizes decode is memory-bound, so an upper bound for one stream is:

$$
\text{tokens per second per stream} \approx \frac{\text{memory bandwidth}}{\text{bytes read per token}}
$$

Example: a 70B model in FP8 (about 70 GB) on an H200 at 4.8 TB/s gives at most about 68 tokens/s for a single stream; real numbers are 60 to 80 percent of that. Batching amortises the weight reads across sequences, so aggregate throughput grows almost linearly with batch size until compute or KV memory runs out.

**Step 3: compare the realistic options (approximate public figures)**

| Accelerator | Memory | Bandwidth | Notes for inference |
| --- | --- | --- | --- |
| NVIDIA L4 | 24 GB | about 0.3 TB/s | Cheap, low power; small models, embeddings, rerankers |
| NVIDIA L40S | 48 GB | about 0.86 TB/s | No NVLink; good for 7B to 13B models and image generation |
| NVIDIA A100 | 40 or 80 GB | about 2.0 TB/s | No FP8; still cost-effective for mid-size models |
| NVIDIA H100 SXM | 80 GB | about 3.35 TB/s | FP8, NVLink; the 2024 to 2025 default |
| NVIDIA H200 | 141 GB | about 4.8 TB/s | Same compute as H100, much more memory; 70B FP8 fits on one GPU |
| NVIDIA B200 | about 180 to 192 GB | about 8 TB/s | FP4 support, large step in throughput per GPU |
| AMD MI300X / MI325X / MI355X | 192 / 256 / 288 GB | about 5.3 / 6 / 8 TB/s | Large memory per GPU; ROCm support in vLLM and SGLang is good but check your model |
| Google TPU (v5e, v6e, Ironwood) | 16 to 192 GB per chip | varies | Strong price-performance on Google Cloud; JAX or vLLM TPU backend |
| AWS Inferentia2 / Trainium | 32 GB and up | varies | Cheaper per token on AWS once the model is compiled with Neuron |

Treat the table as a starting point and check current specifications and prices; vendors revise them often.

**Step 4: interconnect and topology**

- Tensor parallelism all-reduces activations twice per layer, so it needs NVLink or an equivalent within a node; across nodes over InfiniBand or Ethernet it is usually too slow for latency-sensitive decode.
- Large MoE models need fast all-to-all for expert parallelism (see 27.41), which favours NVL72-class racks or well-provisioned InfiniBand.

**Step 5: economics**

- Compare **cost per million output tokens at your SLO and expected utilisation**, not hourly price (see 27.16). A GPU that costs 40 percent more per hour but serves 2 times the tokens is cheaper.
- Factor availability and lock-in: reserved capacity versus on-demand, the cost of porting to non-CUDA stacks, and engineering time.
- Smaller GPUs replicated many times are often better for small models than a few big GPUs, because they give more independent replicas and smoother autoscaling.

**Pitfalls**

- Choosing on FLOPS: decode rarely uses them.
- Forgetting the KV cache: a 70B model that "fits" with 5 GB to spare can only serve a handful of concurrent users.
- Ignoring FP8 support: Ada (L4, L40S), Hopper and Blackwell have FP8 tensor cores, Ampere (A100) does not.
- Not benchmarking: run your own prompt and output length distribution on a shortlist of two or three options before committing to reserved capacity.

### 27.3 What is model parallelism vs data parallelism in distributed training?

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

### 27.4 How do you split a large model across GPUs for serving: tensor parallelism, pipeline parallelism and model sharding?

When a model (plus its KV cache) does not fit on one GPU, or one GPU is too slow, you shard it. **Tensor parallelism (TP)** splits each layer's weight matrices across GPUs that compute the same layer together; **pipeline parallelism (PP)** puts different groups of layers on different GPUs; **expert parallelism (EP)** places different MoE experts on different GPUs. "Model sharding" is the umbrella term for all of these (and, more loosely, for splitting checkpoint files, which only helps storage and loading).

**Tensor parallelism**

- Each linear layer is split by columns or rows (Megatron-style), so every GPU holds about 1/TP of each layer and computes part of each matrix multiply.
- Results are combined with an all-reduce, roughly twice per Transformer layer, on every forward step.
- Reduces both memory per GPU and per-token latency, because every GPU works on every token.
- Needs very fast links: NVLink inside a node. TP across nodes over slower networks usually costs more in communication than it saves.
- Practical constraint: the number of KV heads should be divisible by the TP degree, otherwise KV heads are replicated (see 27.29).

**Pipeline parallelism**

```text
Input -> [GPU 1: layers 1-20] -> [GPU 2: layers 21-40] -> [GPU 3: layers 41-60] -> [GPU 4: layers 61-80] -> Output
```

- Only activations at stage boundaries are sent, so it tolerates slower links and works across nodes.
- Does not reduce single-request latency (a token still passes through every stage in turn) and creates **pipeline bubbles**: stages idle while the pipeline fills and drains. Micro-batching (many requests or micro-batches in flight) keeps stages busy.
- In training it is combined with micro-batch schedules such as 1F1B; in serving it is used mainly to span nodes.

**Comparison**

| | Tensor parallelism | Pipeline parallelism | Expert parallelism |
| --- | --- | --- | --- |
| Splits | Each layer's matrices | Layers into stages | MoE experts |
| Communication | All-reduce every layer | Point-to-point between stages | All-to-all token dispatch |
| Link needed | NVLink-class | Can cross nodes | Fast all-to-all |
| Latency effect | Lowers per-token latency | Adds a little latency | Neutral if balanced |
| Main use | Fit and speed up a model within a node | Span nodes | Very large MoE models |

**How to choose for serving**

1. If the model fits on one GPU with a healthy KV budget, do not shard; run replicas (data parallelism) for throughput. This gives the best throughput per GPU and simplest operations.
2. If not, use TP within one node (TP 2, 4 or 8) until it fits with enough KV cache.
3. If one node is not enough, use TP within nodes and PP across nodes (for example TP 8 and PP 2 for a very large dense model on two 8-GPU nodes).
4. For large MoE models, combine attention data parallelism with expert parallelism (see 27.41).
5. Prefer quantisation or a larger-memory GPU over more sharding when possible: every extra degree of TP adds communication and reduces the number of replicas you can run.

**Trade-offs and pitfalls**

- More TP lowers latency but often lowers throughput per GPU, because communication overhead rises and batches get smaller per replica.
- A single failed GPU takes down the whole sharded replica, so blast radius grows with shard count (see 27.45).
- Sharded checkpoints (safetensors shards) speed up loading by letting each GPU read only its slice, which matters for cold starts (see 27.17).
- For training-time sharding of parameters, gradients and optimiser state (FSDP and ZeRO), see 27.20.

### 27.5 What is the difference between static, dynamic and continuous batching for LLM serving?

Batching processes many requests in one forward pass so the GPU reads the weights once for all of them. Because decode is memory-bandwidth-bound, batching is the main lever on throughput and cost per token. The three strategies differ in **when** requests may join and leave the batch. For the model-side view see 17.88.

**Static batching**

- Collect a fixed batch, run it until every sequence finishes, then start the next batch.
- Short requests finish early and their slots sit idle as padding while the longest request runs; new arrivals wait for the whole batch.
- Fine for offline jobs with similar lengths (bulk embedding, scoring, classification).

**Dynamic batching**

- Wait up to a short timeout (for example 5 to 20 ms) or until a maximum size, then run the batch.
- Standard for fixed-cost models such as encoders, vision models and classic ML in Triton or TorchServe (see 26.82).
- For LLMs it still runs batch-to-completion, so it has the same idle-slot problem as static batching.

**Continuous batching (iteration-level or in-flight batching)**

- The scheduler runs **every decode step**: finished sequences leave immediately, waiting requests join in the free slots.
- The batch stays full of active sequences, so throughput is many times higher than static batching on chat workloads with variable output lengths (Orca and vLLM reported roughly an order of magnitude).
- New requests' prefill is mixed with ongoing decodes; **chunked prefill** splits long prompts into pieces so they do not stall everyone else's next token.

```text
Static:      A ================            Continuous:  A ================
             B =======.........                         B =======
             C ================                         C ================
             D waits ..........  D ======                D        ========
                                                        E           =======
```

**What makes continuous batching work**

- **Paged KV cache** (see 27.8): sequences of different lengths join and leave without fragmenting memory.
- A scheduler with admission control (only admit when KV blocks are available), preemption (swap or recompute KV for a low-priority request under memory pressure) and priorities (see 27.15).

**Tuning knobs and trade-offs**

| Knob (vLLM names) | Raising it | Lowering it |
| --- | --- | --- |
| `max_num_seqs` (max concurrent sequences) | More throughput, higher inter-token latency | Lower latency, lower throughput |
| `max_num_batched_tokens` (tokens per step) | Faster prefill, bigger latency spikes | Smoother ITL, slower TTFT for long prompts |
| `gpu_memory_utilization` | More KV cache, more concurrency | Headroom against OOM |

- Throughput rises with batch size until the step becomes compute-bound or the KV cache is full; beyond that, latency rises with no throughput gain.
- Choose the batch limits from a load test that plots throughput against p95 TTFT and ITL and pick the point that meets the SLO (goodput).

**When batching does not help:** single-user on-device inference, strict per-request latency with very low traffic, or when requests need different models or adapters (multi-LoRA serving solves the last case, see 22.35).

All modern engines (vLLM, SGLang, TensorRT-LLM, TGI) implement continuous batching; the question in practice is how you tune and protect it.

### 27.6 How does speculative decoding work, and when does it help in production?

**Speculative decoding** lets a cheap drafter propose several tokens and the large target model verify them all in one forward pass, accepting the longest correct prefix. Because decode is memory-bound, verifying k tokens costs about the same wall-clock time as generating one, so each expensive pass yields more than one token, with **no change to the output distribution**. The algorithm's derivation is in 17.87; this answer focuses on when it pays off.

**Algorithm per step**

1. The drafter proposes k tokens.
2. The target runs once over the prefix plus the k candidates and produces its distribution at every position.
3. Each draft token is accepted with probability equal to the minimum of 1 and the ratio of target to draft probability; at the first rejection a replacement is sampled from the normalised residual distribution and later drafts are discarded.
4. If all k are accepted, the target's last distribution gives one bonus token.

This rejection-sampling rule makes the output exactly distributed as the target's own samples; under greedy decoding the text is identical.

**Expected speed-up**

With acceptance rate $\alpha$ and draft length $k$, the expected tokens per target pass is:

$$
\frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

With $\alpha = 0.8$ and $k = 4$ that is about 3.4 tokens per pass, which after drafting overhead typically gives 2 to 3 times lower latency.

**Kinds of drafter**

| Drafter | How it drafts | Strengths | Weaknesses |
| --- | --- | --- | --- |
| Separate small model | Same family, 10 to 50 times smaller | Simple, works with any target | Extra memory; must share the tokenizer |
| Medusa heads | Extra heads on the target predict future tokens | No second model | Needs head training |
| EAGLE (1, 2, 3) | Light head drafts at the feature level | High acceptance, widely supported | Needs a trained head per target model |
| MTP heads | Multi-token prediction heads trained with the model (DeepSeek-V3 and others) | Free if the model ships them | Model-specific |
| N-gram / prompt lookup | Copies spans from the prompt | No training; excellent for RAG, summaries, code edits | Poor on open-ended generation |

**When it helps**

- Low to moderate batch sizes and latency-sensitive traffic, where GPU compute is idle during decode.
- Predictable, input-grounded output: code editing, extraction, RAG answers that quote the source, low temperature.

**When it hurts**

- **High load:** at large batch sizes the GPU is closer to compute-bound, so the extra verification FLOPs are no longer free and throughput can fall. Many engines let you switch speculation off above a batch-size threshold.
- **Low acceptance:** high-temperature creative sampling or a poorly matched drafter.
- **Memory:** draft model weights and KV cache take space that would otherwise hold more sequences.

**Production practice**

- Monitor acceptance rate per workload and per route; it is the leading indicator of whether speculation pays.
- Tune k (typically 3 to 8) on real traffic; longer drafts waste work when acceptance is low.
- Benchmark throughput and p95 latency at your real concurrency with speculation on and off, not only single-request latency.
- Supported in vLLM, SGLang and TensorRT-LLM; check that it composes with your quantisation, structured output and LoRA settings.

### 27.7 How do you manage KV-cache memory when serving LLMs?

The **KV cache** stores the key and value tensors of every previous token, for every layer, so decode computes K and V only for the new token instead of re-processing the whole prefix (the mechanism is covered in 17.32 and 17.85). The price is GPU memory: at serving scale the **KV cache, not the weights, usually limits concurrency, context length and therefore throughput**. Managing it is the central job of an LLM serving engine.

**How big it is**

$$
\text{KV bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{tokens} \times \text{bytes per element}
$$

| Model (BF16 cache) | KV per token | One 8k-token sequence | 64 concurrent 8k sequences |
| --- | --- | --- | --- |
| Llama-2-7B (32 layers, 32 KV heads, MHA) | about 0.5 MB | about 4 GB | about 270 GB |
| Llama-3-8B (32 layers, 8 KV heads, GQA) | about 128 KB | about 1 GB | about 69 GB |
| Llama-3-70B (80 layers, 8 KV heads, GQA) | about 320 KB | about 2.7 GB | about 172 GB |

The jump from MHA to GQA is why architecture choice matters so much for serving cost (see 27.29).

**Techniques, roughly in order of impact**

- **Paged allocation** (PagedAttention, see 27.8): allocate fixed-size blocks on demand instead of reserving the maximum length per request; waste drops from 60 to 80 percent to a few percent.
- **Prefix caching and sharing:** requests with the same system prompt or document reuse the same KV blocks (vLLM automatic prefix caching, SGLang RadixAttention, see 27.26).
- **KV-cache quantisation:** FP8 KV halves memory with little quality loss on most models; INT4 KV is riskier, especially for long-context retrieval.
- **Model choice:** GQA, MQA or MLA models have 4 to 30 times smaller caches than MHA models of similar size.
- **Admission control and context limits:** only admit a request when enough blocks are free; set per-tenant maximum context; reject or route very long requests to a dedicated pool.
- **Preemption:** when memory runs out, swap a low-priority sequence's blocks to CPU memory or drop and recompute them later, instead of failing with OOM.
- **Offloading and tiering:** keep cold prefixes in CPU memory or local SSD and bring them back on reuse (LMCache, NVIDIA Dynamo's KV block manager). Useful for agents and multi-turn chat where the same long history returns minutes later.
- **Sliding-window or sparse attention** where the model supports it, which caps per-sequence cache size.
- **Disaggregated prefill and decode** (see 27.40): KV is computed on prefill GPUs and transferred to decode GPUs, so decode pools can be sized for cache capacity.

**Capacity planning**

1. Compute KV per token for the model and chosen cache precision.
2. Free memory for KV = GPU memory times utilisation target, minus weights, minus activation overhead.
3. Token capacity = free memory divided by KV per token; concurrency is roughly token capacity divided by the average live context length.
4. If concurrency is below what the throughput target needs, change the lever: quantise weights or KV, choose a GQA/MLA model, add GPUs, or cap context.

**What to monitor:** KV-cache utilisation percent, number of preemptions or swaps, prefix-cache hit rate, queue time, and OOM events. Rising preemptions with flat traffic usually mean output lengths or context lengths have grown.

### 27.8 What is PagedAttention, and how does it improve LLM serving?

**PagedAttention** (Kwon et al., 2023, the core idea of **vLLM**) manages the KV cache the way an operating system manages virtual memory: the cache is split into fixed-size **blocks** that can live anywhere in GPU memory, and each sequence has a **block table** mapping its logical token positions to physical blocks. The model-level description is in 17.86; here the focus is why it matters for serving.

**The problem it solves**

Earlier servers reserved one contiguous KV buffer per request sized for the maximum possible length, because the output length is unknown in advance. That causes:

- **Internal fragmentation:** reserved space that is never filled.
- **External fragmentation:** free gaps between buffers too small to reuse.
- **Reservation waste:** space held for tokens that will not be generated for many steps.

The vLLM paper measured only about 20 to 40 percent of KV memory holding real tokens in existing systems.

**How it works**

```text
Sequence A logical blocks:  [A0][A1][A2]          Physical GPU memory
Sequence B logical blocks:  [B0][B1]              [A1][free][B0][A0][shared prefix][B1][A2]...
Block table (per sequence): logical block -> physical block id
```

- Blocks (for example 16 tokens each) are allocated **on demand** as tokens are generated; waste is limited to the last partly filled block of each sequence, typically under 4 percent.
- The attention kernel gathers K and V through the block table, so blocks need not be contiguous.
- Blocks are **reference-counted**: sequences that share a prefix (same system prompt, parallel sampling, beam search) point to the same physical blocks, with **copy-on-write** when they diverge.

**Why serving improves**

- **Bigger effective batch, higher throughput:** decode is memory-bound and throughput scales with how many sequences fit; near-zero waste means many more sequences per GPU. vLLM reported 2 to 4 times the throughput of earlier optimised systems, and up to about 24 times that of naive Hugging Face Transformers serving.
- **Makes continuous batching practical:** finished sequences return their blocks immediately, so new requests can join without compaction or copying (see 27.5).
- **Prefix sharing:** the basis of automatic prefix caching across requests (see 27.26).
- **Parallel sampling and beam search** share prompt blocks across candidates, cutting memory by up to about 55 percent for those workloads.
- **Graceful degradation:** under memory pressure, whole blocks can be swapped to CPU or recomputed rather than crashing with OOM.
- **Predictable capacity:** you know exactly how many tokens fit, so admission control and autoscaling can work on KV-block usage.

**Trade-offs**

- The indirection adds some kernel overhead; it is small compared with the batching gain.
- Block size is a tuning choice: larger blocks mean less table overhead but more waste per sequence and coarser prefix sharing.

Paged KV caches are now standard in vLLM, SGLang, TensorRT-LLM and TGI, so the interview question is less "what is it" than "how do you size, monitor and protect it" (see 27.7).

### 27.9 How does GGUF work?

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

### 27.10 How do you optimize inference for edge and mobile deployment?

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

### 27.11 What is quantization, and how does it affect model quality, especially when it is aggressive?

**Quantisation** stores weights (and optionally activations and the KV cache) in fewer bits, for example BF16 to FP8, INT8 or INT4, to reduce memory and memory traffic. For LLM serving this matters because decode is memory-bound: halving bytes per weight roughly halves the bytes read per token and frees memory for KV cache. Quality degrades **gracefully down to about 8 bits, noticeably at 4 bits and sharply below that**, but the amount depends on method, model size and task. For number formats see 27.31; for PTQ versus QAT see 27.32; for the model-level overview see 17.6.

**What to quantise**

| Scheme | Notation | Benefit | Typical use |
| --- | --- | --- | --- |
| Weight-only | W4A16, W8A16 | Less memory and faster memory-bound decode | Fitting large models; low-batch serving |
| Weights and activations | W8A8 (INT8 or FP8) | Also uses low-precision tensor cores, faster prefill and high-batch decode | High-throughput serving on Hopper, Ada, Blackwell |
| KV cache | FP8 or INT8 KV | Doubles KV capacity, more concurrency | Long context and high concurrency |
| 4-bit floating (FP4, MXFP4, NVFP4) | W4A4 | Large throughput gains on Blackwell | Newer models released with FP4 checkpoints |

Memory for a 70B model: about 140 GB in BF16, 70 GB in FP8 or INT8, and 35 to 40 GB in INT4 (scales and zero-points add a few percent).

**Typical quality pattern**

- **BF16 to FP8 or INT8:** near-lossless (often well under 1 percent on benchmarks) with reasonable calibration. FP8 is close to a free default on hardware that supports it.
- **INT4 weight-only (AWQ, GPTQ, NF4):** small perplexity increase; usually fine for chat and RAG, but a few points can be lost on maths, code and multi-step reasoning.
- **3 bits and below:** perplexity rises quickly; models become repetitive, lose factual recall, follow instructions poorly and break output formats, unless specialised methods (QuIP#, AQLM) or quantisation-aware training are used.

**Why aggressive quantisation hurts**

- **Outliers:** a few channels have very large magnitudes; uniform low-bit grids either clip them or waste resolution on everything else. LLM.int8(), SmoothQuant and AWQ exist to handle this.
- **Error accumulation** across many layers and long generations.
- **Activations and KV are harder** than weights: activations have outliers that change per token, and KV quantisation can hurt long-context retrieval.
- **Small models suffer more** than large ones because they have less redundancy.
- **Averages hide tail damage:** multilingual, long context, precise arithmetic, tool-call formatting, JSON validity and safety or refusal behaviour can regress while the headline benchmark stays flat.

**Production practice**

1. Start with the cheapest safe option for your hardware: FP8 W8A8 on Hopper or Blackwell, INT8 or INT4 weight-only elsewhere.
2. Use calibrated methods (AWQ, GPTQ, SmoothQuant) with calibration data that looks like your traffic, including your languages and domains.
3. Keep sensitive parts in higher precision: embeddings, the LM head, sometimes the first and last layers.
4. Evaluate on your own task set and guardrail tests, not only perplexity: check structured-output validity, tool-call accuracy and refusal behaviour.
5. Compare at equal memory: a larger model in 4-bit often beats a smaller model in 16-bit.
6. Measure the speed-up on your engine and GPU. INT4 weight-only gives less speed-up at high batch sizes, where the step is compute-bound and weights are dequantised to BF16 for the matmul.
7. Roll out behind a canary with quality monitoring; if accuracy drops, see 26.43 for recovery techniques.

### 27.12 How do you implement auto-scaling for AI workloads?

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

### 27.13 What is the role of load balancing in AI serving infrastructure?

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

### 27.14 How do you manage GPU memory for serving multiple models?

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

### 27.15 How do you implement request queuing and priority scheduling for AI services?

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

### 27.16 How do you compare the cost of self-hosting LLM inference with using an API, and how do you plan capacity?

The trade-off is **fixed cost and control (self-host) versus variable cost and simplicity (API)**. Self-hosting only wins when you can keep GPUs busy, so the honest comparison is **cost per million tokens at your real utilisation and SLO**, plus the engineering cost, against the API's per-token price.

**Cost per million tokens for self-hosting**

$$
\text{cost per 1M tokens} = \frac{\text{GPU price per hour} \times \text{GPUs per replica}}{\text{tokens per second per replica} \times 3600 \times \text{average utilisation}} \times 10^{6}
$$

Illustrative example (check current prices; they move fast):

- A 70B model in FP8 on 2 H100s (TP 2), rented at about 3 dollars per GPU-hour: 6 dollars per hour per replica.
- Load test at the p95 latency SLO gives about 2,000 output tokens/s per replica.
- At 100 percent utilisation: 6 divided by 7.2M tokens per hour is about 0.83 dollars per million output tokens.
- At a realistic 35 percent average utilisation (diurnal traffic, headroom for peaks): about 2.40 dollars per million.
- Add engineers, monitoring, storage, networking and redundancy, and compare against hosted prices for the same open model, which are often below 1 dollar per million tokens from serving specialists.

The example shows the usual result: **utilisation and engineering time decide the answer more than GPU price does.**

**Capacity planning steps**

1. **Characterise traffic:** peak and average requests per second, input and output token distributions (p50, p95), required TTFT and ITL, diurnal shape.
2. **Fit the model:** weights plus KV cache for the target concurrency (see 27.7 and 27.37) decides GPUs per replica.
3. **Measure throughput per replica** at the SLO with a realistic load test (see 27.44). Use goodput, not peak throughput.
4. **Replicas needed** = peak output tokens per second divided by tokens per second per replica, plus N+1 or 20 to 30 percent headroom for failures and spikes.
5. **Blend capacity types:** reserved or committed GPUs for the base load, on-demand or a hosted API for peaks, spot for batch jobs.
6. **Revisit monthly:** new models, quantisation and engine versions often change tokens per GPU by 1.5 to 3 times.

**Decision factors beyond price**

| Factor | Favours API | Favours self-hosting |
| --- | --- | --- |
| Volume | Low, spiky, uncertain | High, steady, predictable |
| Model | Need frontier closed models | Open-weight model is good enough, or fine-tuned |
| Data | Provider terms and regions acceptable | Data must stay in your tenant or on-premises |
| Latency | Standard | Very tight, or co-location with other services |
| Team | Small, no GPU operations skills | Platform team that already runs GPUs |
| Time to market | Weeks | Months are acceptable |
| Control | Provider may change or retire models | Need pinned versions, custom decoding, LoRA adapters |

**Managed middle grounds:** cloud-hosted models in your own tenant (Azure OpenAI, Bedrock, Vertex AI) give enterprise data terms and private networking without running GPUs; dedicated or provisioned-throughput endpoints give reserved capacity at a fixed price; managed open-model hosts give open weights without operations. The same logic applies to vector databases: start managed unless compliance, scale or cost at volume forces self-hosting.

**Pitfalls**

- Comparing API price with self-hosted cost at 100 percent utilisation.
- Ignoring the cost of the second region, on-call, upgrades and evaluation when models change.
- Committing to multi-year reservations before traffic is known; GPU generations turn over every 12 to 24 months.
- Optimising unit cost while total cost is driven by wasteful prompts; cutting tokens is usually the first saving (see 27.1 and 26.12).

**Common outcome:** use APIs to launch and learn, then move high-volume, stable, well-understood workloads (classification, extraction, embeddings, a fine-tuned small model) to self-hosting, and keep frontier reasoning on APIs. A gateway that can route to either keeps the choice reversible.

### 27.17 How do you handle cold-start latency for serverless and autoscaled GPU inference?

A cold start is the time from "a new replica is needed" to "it serves its first token": getting a GPU node, pulling the container image, loading weights into GPU memory, compiling or capturing kernels, and warming caches. For CPU functions this is seconds; for LLMs it can be **several minutes**, because a 70B model means moving 70 to 140 GB of weights. That gap is why naive scale-to-zero rarely works for interactive LLM traffic.

**Where the time goes (typical large-model replica)**

| Phase | Typical time | Fixes |
| --- | --- | --- |
| Provision GPU node | 1 to 10 minutes, longer if capacity is scarce | Warm node pools, reserved capacity, overprovisioned placeholder pods |
| Pull container image (often 10 to 20 GB with CUDA) | 1 to 5 minutes | Slim images, pre-pulled images on nodes, lazy-pulling snapshotters |
| Download weights | Minutes for 100 GB from object storage | Local NVMe cache, regional mirrors, streaming loaders |
| Load to GPU | Tens of seconds | safetensors with memory mapping, sharded parallel loading per GPU |
| Engine start-up (CUDA graphs, compilation, profiling) | Tens of seconds to minutes | Cache compiled artefacts; prebuild TensorRT-LLM engines |
| Warm-up | Seconds | Send synthetic requests before marking the pod ready |

**Strategies**

- **Keep a warm floor:** minimum replicas or provisioned concurrency for interactive traffic; scale-to-zero only for internal, batch or rarely used models.
- **Scale earlier:** autoscale on queue depth and KV-cache usage with headroom, and use scheduled scaling ahead of known peaks, so new replicas arrive before they are needed (see 27.12).
- **Make loading fast:** pre-pulled images, weights on local NVMe or a high-throughput shared cache, streaming weight loaders that load straight into GPU memory, and GPU memory or process snapshots where the platform supports them.
- **Smaller artefacts:** quantised or distilled models load proportionally faster.
- **Multi-LoRA instead of many models:** one warm base model with adapters swapped in milliseconds avoids cold-starting many fine-tuned models (see 22.35).
- **Graceful handling while cold:** queue with a timeout, route to a warm fallback model or a hosted API, and return a clear "busy" response rather than hanging.
- **Readiness probes that mean ready:** mark a pod ready only after the model is loaded and warmed, so the load balancer never sends traffic to a loading replica.
- **Separate layers:** keep serverless functions for orchestration and API logic, and keep GPU inference on persistent services.

**What to monitor:** cold-start duration broken down by phase, the share of requests that waited on a cold replica, time-to-ready for scale-out events, and the cost of the warm floor. The trade-off is explicit: a warm floor costs money every hour; cold starts cost latency and failed requests at the worst moment.

### 27.18 How do you use caching to avoid redundant computation in AI serving, including semantic caching?

Cache at every layer where work repeats: **model artefacts** (avoid reloading), **intermediate computation** (embeddings, KV prefixes), and **final responses** (exact or semantic). Each layer has a different hit rate, staleness risk and invalidation rule.

**Layers of caching**

| Layer | What is cached | Saves | Main risk |
| --- | --- | --- | --- |
| Model weights and images | Weights on local NVMe, pre-pulled images, warm replicas | Cold-start time (see 27.17) | Stale versions after rollout |
| Embedding cache | Vectors for documents and frequent queries | Embedding calls and latency | Embedding model change invalidates everything |
| Prefix / KV cache | KV tensors of shared prompt prefixes | Prefill compute and TTFT (see 27.26) | None for correctness; only hit rate |
| Tool and retrieval cache | Search results, API responses, feature lookups | External calls in agents and RAG | Freshness of the underlying data |
| Exact response cache | Output for an identical normalised request | The entire model call | Personalised or time-sensitive answers |
| Semantic cache | Output for a request with similar meaning | The entire model call | Returning the wrong answer to a similar-looking question |

**Exact versus semantic caching**

- **Exact cache:** key is a hash of the normalised prompt plus model, version and parameters. Hits are always correct (for deterministic settings) but rare in free-text traffic.
- **Semantic cache:** embed the query, search a vector index of past queries, and return the stored answer if similarity exceeds a threshold. Hit rates can be much higher for FAQ-style traffic, but "How do I cancel my order?" and "How do I cancel my subscription?" can be close in embedding space and need different answers.

**Making semantic caching safe**

- Tune the threshold on labelled pairs of "same intent" and "different intent" queries and measure the false-hit rate, not only the hit rate.
- Scope by tenant, user permissions, language and route; never share cached answers across access boundaries.
- Exclude personalised, time-sensitive or high-risk intents; cache only answers that passed guardrails.
- Optionally verify a hit with a cheap reranker or small model before returning it.
- Short TTLs for content tied to changing data, and invalidate on source-document updates.

**Cache keys and invalidation**

- Include model id and version, prompt-template version, retrieval index version, decoding parameters, and tenant in the key.
- Use TTL plus LRU eviction; invalidate on deployments of a new model or prompt.

**Implementation sketch**

```python
import hashlib, json

def cache_key(prompt: str, model: str, template_version: str, params: dict, tenant: str) -> str:
    payload = json.dumps(
        {"p": " ".join(prompt.split()).lower(), "m": model, "t": template_version,
         "params": params, "tenant": tenant},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
```

**What to monitor:** hit rate per layer, latency and cost saved, false-hit rate for the semantic cache (from sampled human or LLM-judge review), and staleness incidents. For application-level caching design see 29.27.

### 27.19 What is the difference between synchronous and asynchronous inference, and when do you use each?

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

### 27.20 What is FSDP (Fully Sharded Data Parallel), and how does it differ from DeepSpeed ZeRO?

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

### 27.21 How do you monitor and profile LLM inference in production (TTFT, inter-token latency, GPU utilization)?

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

### 27.22 What is model routing at the infrastructure level, and how do you route requests based on complexity and cost?

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

### 27.23 What are the main model compression techniques?

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

### 27.24 How do GPUs, TPUs and other AI accelerators differ, and when would you use each?

All AI accelerators rest on the same idea: neural networks are dominated by large matrix multiplies, and **moving data is far more expensive than computing on it**, so the chip packs in thousands of multiply-accumulate units, feeds them from high-bandwidth memory, and keeps data on-chip as long as possible to reuse it (batching, tiling). They differ in how general they are, how they connect, and which software stack they need.

**The shared principle**

- One read of a number from HBM costs the time of hundreds of arithmetic operations, so performance depends on **arithmetic intensity**: how many operations you do per byte moved (see 27.35).
- Hence batching (reuse each weight for many inputs), tiling and kernel fusion (keep data in on-chip SRAM), and low-precision formats (move fewer bytes).

**How the main families differ**

| | GPU (NVIDIA, AMD) | TPU (Google) | Inference ASICs (Inferentia, Groq, Cerebras and others) |
| --- | --- | --- | --- |
| Design | Many general SIMT cores plus tensor cores | Large systolic arrays for matrix multiply | Designed for specific inference patterns |
| Programmability | Very general: CUDA or ROCm, custom kernels | XLA compiler via JAX, PyTorch/XLA, vLLM TPU backend | Vendor compiler or hosted API only |
| Memory | Large HBM per device (80 to 288 GB) | HBM per chip, pods linked by fast inter-chip interconnect | Varies: HBM, or very large on-chip SRAM (Groq, Cerebras) |
| Scale-out | NVLink within node or rack, InfiniBand or Ethernet across | Pod-scale torus interconnect | Vendor-specific |
| Ecosystem | Largest: every model and engine supports it first | Strong for Google-scale training and serving | Narrow; best for supported models |
| Availability | Every cloud and on-premises | Google Cloud only | Specific clouds or vendor APIs |

**When to use each**

- **NVIDIA GPUs:** the default. Day-one support for new models, every engine (vLLM, SGLang, TensorRT-LLM), and the most operational know-how. Pay a premium for that flexibility.
- **AMD GPUs (MI300 series and later):** more memory per GPU, which helps large models and long context; competitive cost. Check that your model, quantisation format and engine features are supported on ROCm before committing.
- **TPUs:** strong price-performance on Google Cloud for large, steady workloads, especially JAX-based training and serving; less attractive if you depend on custom CUDA kernels or multi-cloud portability.
- **AWS Inferentia and Trainium:** lower cost per token on AWS for supported models once compiled with the Neuron SDK; good for stable, high-volume serving of a fixed model.
- **SRAM-heavy ASICs (Groq, Cerebras and similar):** very high tokens per second per user, usually consumed as a hosted API; useful when per-user output speed is the product (voice agents, interactive coding).
- **CPUs and NPUs:** small models, embeddings and on-device inference (see 27.10).

**Senior considerations**

- **Portability is a cost:** each non-CUDA platform adds a compile step, version matrix and support lag for new architectures. Keep an abstraction (an OpenAI-compatible endpoint behind a gateway) so you can move workloads.
- **Benchmark your own workload** at your SLO; vendor numbers use favourable batch sizes and sequence lengths.
- **Mix by workload:** frontier-scale serving on the newest GPUs, steady high-volume small models on cheaper accelerators, batch jobs on whatever is cheapest that week.

### 27.25 Why is AI infrastructure moving towards Rust?

Python remains the *interface* for AI, but more of the *infrastructure underneath* is being written in Rust:

- **Performance without a GC:** C/C++-level speed and predictable latency (no garbage-collection pauses), which matters for tokenizers, data loaders, inference servers and vector databases.
- **Memory safety:** the ownership model removes whole classes of bugs (use-after-free, data races) that plague C++ infra. This matters for long-running, multi-tenant services where a segfault means an outage or a security hole.
- **Fearless concurrency:** the compiler enforces thread safety, so heavy parallel work (tokenizing millions of docs, async request handling) is easier to get right than in C++. It also sidesteps Python's GIL.
- **Great Python interop:** PyO3 and maturin make it easy to ship Rust-backed Python packages, so users keep a Python API while the hot path runs in Rust.
- **Tooling and deployment:** Cargo, single static binaries, small containers, and WebAssembly targets for edge or in-browser inference.

**Real examples:** Hugging Face `tokenizers` and `safetensors`, Hugging Face TGI's router, `candle` (an ML framework), Polars (a DataFrame library), vector DBs (Qdrant, LanceDB), `uv`/`ruff` (Python tooling), and parts of OpenAI's and other vendors' serving stacks.

**Trade-offs:** a steeper learning curve, slower development than Python, a smaller ML ecosystem, and CUDA kernels are still mostly C++/CUDA (or Triton). So the realistic pattern is **Python for research and orchestration, Rust for performance-critical, safety-critical plumbing**, rather than a full rewrite.

### 27.26 How does prompt caching work, and when does it help?

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

### 27.27 What does TensorRT (TensorRT-LLM) do?

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

### 27.28 What does deploying AI and ML workloads on Azure typically involve?

Azure deployment for AI usually combines **managed model access (Azure OpenAI in Azure AI Foundry)**, **managed ML (Azure Machine Learning)** and **general compute (AKS or Container Apps)**, wrapped in enterprise networking, identity and monitoring. Interviewers want to hear that you can make it secure, private and cost-controlled, not just list services. Equivalent patterns exist on AWS (Bedrock, SageMaker, EKS) and Google Cloud (Vertex AI, GKE).

**Core building blocks**

| Need | Azure service | Notes |
| --- | --- | --- |
| Hosted frontier models | Azure OpenAI (in Azure AI Foundry) | Standard (pay per token), global or data-zone deployments, and provisioned throughput units (PTU) for reserved capacity |
| Other hosted models | Azure AI Foundry model catalogue (serverless and managed endpoints) | Open-weight and partner models without running GPUs |
| Training, experiments, registry | Azure Machine Learning | Workspaces, compute clusters, MLflow tracking, model registry, managed online and batch endpoints |
| Custom serving (vLLM, Triton) | AKS with GPU node pools, or Azure Container Apps with serverless GPUs | Full control of engine, autoscaling and multi-model serving |
| Retrieval | Azure AI Search | Hybrid keyword and vector search, semantic ranking |
| Data and artefacts | Blob Storage / ADLS Gen2, Cosmos DB, Fabric or Databricks | Lakehouse and application data |
| Secrets and identity | Entra ID, managed identities, Key Vault | No keys in code; role-based access per service |
| Observability | Azure Monitor, Application Insights, Log Analytics | Traces, token usage, latency, cost |
| API front door | Azure API Management | Auth, rate limiting, token quotas, load balancing across model deployments and regions |

**What "working knowledge" means in practice**

- **Private networking:** private endpoints for Azure OpenAI, AI Search and storage, VNet integration, disabled public access, so client data never crosses the public internet.
- **Identity:** managed identities and RBAC instead of API keys; least-privilege roles per service; per-tenant data isolation.
- **Capacity and quotas:** tokens-per-minute quotas per region and deployment; spreading load across regions or using PTU for predictable latency; API Management to fail over between deployments when one returns 429s.
- **Data residency:** choosing regional or data-zone deployments to meet residency requirements; understanding what is logged and for how long.
- **GPU operations on AKS:** GPU node pools, the NVIDIA device plugin or GPU operator, autoscaling on queue depth, spot node pools for batch (see 27.43).
- **Cost control:** budgets and alerts, tagging per team, PTU versus pay-as-you-go break-even, and shutting down idle compute.
- **CI/CD and IaC:** Bicep or Terraform for infrastructure, GitHub Actions or Azure DevOps pipelines, Azure ML pipelines for retraining.

**Typical enterprise RAG architecture on Azure**

```text
Users -> Front Door / API Management (auth, quotas)
      -> App on AKS or Container Apps (managed identity)
           -> Azure AI Search (private endpoint)  <- indexer from Blob / SharePoint
           -> Azure OpenAI deployment (private endpoint, PTU or standard)
      -> Application Insights + Log Analytics (traces, tokens, cost)
```

**Pitfalls:** hitting regional quota limits at launch, forgetting that some models or features are not available in every region, leaving public network access on, and not attributing token spend to teams.

### 27.29 How do MHA, MQA, GQA and MLA compare, and why does the choice matter for serving?

All four are ways of organising attention heads; they differ in **how many key and value heads are stored per token**, which sets the KV-cache size. Because decode reads the whole KV cache for every token, fewer KV bytes means more concurrent sequences, longer contexts and faster decode on the same GPU. The mechanism is covered in 17.38 and 17.77; this answer is about the serving consequences.

| | MHA (multi-head) | GQA (grouped-query) | MQA (multi-query) | MLA (multi-head latent) |
| --- | --- | --- | --- | --- |
| Query heads | h | h | h | h |
| Stored per token per layer | h K and V heads | G K and V heads | 1 K and V head | One compressed latent vector |
| KV cache relative to MHA | 1 | G/h (for example 1/8) | 1/h | Very small (DeepSeek reports over 90 percent reduction) |
| Quality | Baseline | Close to MHA | Noticeably lower, can be unstable | Close to MHA |
| Tensor-parallel fit | Easy | Good if G is a multiple of the TP degree | KV head replicated on every shard | Needs engine support |
| Examples | Original Transformer, GPT-3, Llama-2-7B | Llama-2-70B and Llama-3, Mistral, Qwen, Gemma | PaLM, Falcon-7B, StarCoder | DeepSeek-V2 and V3 |

**How GQA works in one line:** the h query heads are split into G groups and each group shares one K and V head; MHA is the case G = h and MQA is G = 1. Ainslie et al. (2023) showed an MHA checkpoint can be converted to GQA by mean-pooling KV heads within each group and continuing training on about 5 percent of the original compute.

**Serving consequences**

- **Capacity:** Llama-2-70B has 64 query heads and 8 KV heads, so its KV cache is 8 times smaller than an MHA equivalent. On the same GPU that means roughly 8 times more cached tokens: more users or longer contexts.
- **Speed:** decode is memory-bound, so reading fewer KV bytes per token raises tokens per second, especially at long context where KV reads exceed weight reads.
- **Tensor parallelism:** with TP degree larger than G, KV heads must be replicated across GPUs, wasting memory. With 8 KV heads, TP 8 splits cleanly; TP 16 does not.
- **Kernel support:** GQA is supported everywhere; MLA needs dedicated kernels (FlashMLA and engine support in vLLM and SGLang), and MLA models are often served with data-parallel attention plus expert parallelism (see 27.41).
- **Model choice is a cost decision:** when choosing between two open models of similar quality, compare KV bytes per token as well as parameter count; it can change the number of GPUs needed by a factor of several.

**How to choose when training or selecting a model**

- Serving at scale: GQA is the safe default; MLA if your engine supports it well.
- Extreme memory limits on small models: MQA, accepting some quality loss.
- Small encoders or models where inference cost does not matter: MHA.

### 27.30 What is FlashAttention, and why is it faster without changing the result?

**FlashAttention** (Dao et al., 2022) is an **exact, IO-aware attention kernel**: it computes the same output as standard attention but never writes the full score matrix to GPU main memory, so it is much faster and uses memory linear in sequence length. It changes **how data moves**, not **what is computed**. For the model-level introduction see 17.36.

**Why standard attention is slow**

- On an A100, compute is about 312 TFLOPS (BF16) but HBM bandwidth is only about 2 TB/s; on-chip SRAM is much faster but only about 20 MB in total.
- Standard attention writes the n-by-n score matrix to HBM, reads it back for the mask and softmax, writes the probability matrix, and reads it again to multiply by V. For long sequences this traffic, not arithmetic, dominates runtime, and memory grows quadratically.

**What FlashAttention does**

1. **Tiling:** split Q, K and V into blocks that fit in SRAM and do all the work for a block there.
2. **Online softmax:** softmax needs the row maximum and row sum over all keys. FlashAttention keeps running values $m$ (maximum) and $\ell$ (sum); when a new block has a larger maximum, the previous partial results are rescaled by $e^{m_{\text{old}} - m_{\text{new}}}$. This is algebraically identical to the full softmax.
3. **Kernel fusion:** matmul, mask, softmax and the second matmul run in one kernel, so intermediates never touch HBM.
4. **Recomputation in the backward pass:** recompute attention blocks instead of storing the n-by-n matrix, trading cheap FLOPs for expensive memory traffic.

**Result:** HBM accesses fall from order $nd + n^2$ to order $n^2 d^2 / M$, where $M$ is SRAM size. FLOPs rise slightly because of recomputation, yet wall-clock time falls by 2 to 4 times, and memory becomes linear in sequence length. The output is exact up to floating-point rounding: results are not bit-identical because summation order changes, but the method is not an approximation.

**Versions**

- **FlashAttention-2:** better parallelism and work partitioning across thread blocks.
- **FlashAttention-3:** Hopper-specific, using asynchronous memory copies and FP8.
- **Decode variants:** Flash-Decoding and paged variants split long KV caches across thread blocks so a single decoding query still uses the whole GPU; FlashInfer and engine-specific kernels add paged KV and GQA support.

**Why it matters for serving**

- **Prefill:** attention over long prompts becomes much faster, cutting TTFT for long-context and RAG requests.
- **Long context:** linear memory makes 128k-token contexts feasible.
- **Decode:** the gain is smaller, because decode is dominated by reading weights and the KV cache; paged decode kernels and KV quantisation matter more there.

**In practice** you rarely call it directly: `torch.nn.functional.scaled_dot_product_attention` selects a flash backend, and vLLM, SGLang and TensorRT-LLM ship their own fused attention kernels. The engineering question is whether your engine has fast kernels for your GPU generation, head dimension, GQA or MLA layout and KV-cache format.

### 27.31 Compare INT8, INT4, FP16, BF16, and FP8.

| Format | Bits | Layout | Range / precision | Typical use |
| --- | --- | --- | --- | --- |
| **FP16** | 16 | 1 sign, 5 exponent, 10 mantissa | Max ~65,504; good precision, **narrow range** | Inference; mixed-precision training (needs loss scaling) |
| **BF16** | 16 | 1 sign, **8 exponent**, 7 mantissa | Same range as FP32, lower precision | Default for LLM training and inference on A100/H100/TPU; no loss scaling needed |
| **FP8** | 8 | E4M3 (precision) or E5M2 (range) | Very coarse; needs per-tensor or per-block scaling | Training on Hopper and Blackwell (Transformer Engine); inference also on Ada (L4, L40S); W8A8 serving at up to ~2× BF16 throughput |
| **INT8** | 8 | Integer + scale (+ zero-point) | 256 levels, uniform spacing | PTQ for CNNs and LLMs (LLM.int8(), SmoothQuant); broad hardware support including CPUs and edge |
| **INT4** | 4 | Integer + per-group scale | 16 levels | Weight-only LLM inference (GPTQ, AWQ, GGUF Q4); 4× smaller than FP16 |

**Key tradeoffs:**

- **Range vs precision:** FP16 overflows (hence NaNs in training) while BF16 rarely does, at the cost of mantissa precision.
- **Floating vs integer:** FP8 handles outliers better than INT8 thanks to its exponent. Integer formats need careful scale selection.
- **Hardware support decides speed:** FP8 tensor cores need Ada (L4, L40S), Hopper or Blackwell; Ampere (A100) can store FP8 weights but computes in FP16 or BF16. INT4 is mostly a *memory* win (weights are dequantized to FP16 for matmul) unless kernels are specialised.

**Rule of thumb:** train in BF16 (or FP8 at frontier scale), serve in FP8 or INT8 when quality matters and hardware supports it, and use INT4 weight-only to fit large models on limited GPUs, always validating on task evals.

### 27.32 What is quantization-aware training vs post-training quantization?

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

### 27.33 How would you reduce inference latency without significantly reducing accuracy?

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

### 27.34 Why is GPU memory often the bottleneck in LLM inference?

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

### 27.35 Compare compute-bound vs memory-bandwidth-bound workloads.

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

### 27.36 What is the difference between training FLOPs and inference FLOPs?

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

### 27.37 How would you estimate the GPU requirements for serving an LLM?

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

### 27.38 How would you design a production LLM serving stack?

Design it as layers with clear jobs: a **gateway** that controls access and cost, a **router** that picks a model and replica, an **inference engine** that keeps GPUs busy, a **GPU platform** that scales and heals, and **observability and evaluation** across all of them. For a full system-design walk-through (multi-tenant inference platform) see 29.9; this answer gives the architecture and the key decisions.

```text
Clients
  -> API gateway: auth, tenant quotas, rate and token limits, request validation, PII filters
  -> Router: model selection (size, cost, capability), prefix/cache-aware replica choice, fallbacks
  -> Inference engine pools (vLLM / SGLang / TensorRT-LLM)
       continuous batching, paged KV cache, prefix caching, FP8, speculative decoding
       optional: separate prefill and decode pools
  -> GPU platform: Kubernetes GPU node pools, autoscaling on queue depth and KV usage,
       weight cache, health checks, multi-region capacity
  -> Side systems: response/semantic cache, guardrails, async batch queue, usage metering
  -> Observability: TTFT, ITL, tokens/s, queue time, KV usage, errors, cost per tenant, quality evals
```

**Key decisions**

1. **Models and routing:** which models (API, open-weight, fine-tuned), and a router that sends each request to the cheapest model that meets quality (see 27.22). Multi-LoRA serving for many fine-tunes on one base model (see 22.35).
2. **Engine:** choose by model support, hardware and features you need (see 27.44); pin versions and upgrade through benchmarks.
3. **Hardware and parallelism:** fit weights plus KV for the target concurrency; prefer replicas over deep sharding (see 27.2 and 27.4).
4. **SLOs and scheduling:** set p95 TTFT and ITL per tier; separate interactive and batch traffic into different priority classes or pools (see 27.15 and 27.19).
5. **Scaling:** autoscale on queue depth, waiting requests and KV-cache usage, keep a warm floor, and plan cold starts (see 27.12 and 27.17).
6. **Load balancing:** least-outstanding-requests or KV-aware routing, with prefix affinity so repeated prefixes hit the same replica's cache (see 27.13).
7. **Streaming:** SSE or WebSocket token streaming, with cancellation propagated to the engine so abandoned requests free GPU slots.
8. **Reliability:** timeouts, retries only on idempotent failure modes, circuit breakers, fallback to another model or provider, multi-region or multi-provider for critical paths (see 27.45).
9. **Security:** tenant isolation, no prompts in logs without redaction, prompt-injection and output guardrails, encryption, audit logs (see Chapter 30).
10. **Cost governance:** per-tenant token metering, budgets and alerts, cost per request dashboards, and a monthly review of utilisation (see 27.16).

**Rollout and change management**

- Treat a new model, quantisation or engine version as a release: offline evals, load test, shadow traffic, canary by percentage, and automatic rollback on quality or latency regressions (see Chapter 26).
- Version prompts, models and routing policies together so incidents are traceable.

**What interviewers look for:** clear SLOs, an explanation of where latency and cost come from (prefill, decode, queueing), protection against overload (admission control, backpressure), and a credible path from launch on APIs to self-hosting where it pays.

### 27.39 What are the prefill and decode phases, and how do serving systems schedule them?

LLM inference has two phases with opposite performance profiles. **Prefill** processes the whole prompt in one parallel pass and writes its KV cache; it is **compute-bound** and determines time to first token. **Decode** generates one token per step, reading all weights and the KV cache each time; it is **memory-bandwidth-bound** and determines inter-token latency. Most serving design is about scheduling these two without letting one harm the other. The model-level explanation is in 17.112.

| | Prefill | Decode |
| --- | --- | --- |
| Work per step | All prompt tokens at once (matrix-matrix) | One token per sequence (matrix-vector) |
| Bottleneck | Compute (tensor-core FLOPS) | Memory bandwidth |
| Latency metric | TTFT | ITL or TPOT |
| Scales with | Prompt length (attention grows with length squared) | Output length, model size, context length |
| Helped by | FP8 compute, FlashAttention, prefix caching | Batching, quantisation, GQA/MLA, speculative decoding |

**The scheduling problem**

With continuous batching, new requests' prefills share GPU steps with existing requests' decodes. A 30k-token prefill arriving in the middle of 100 active decodes makes every one of those users wait for that step, which shows up as an ITL spike. Conversely, prioritising decodes delays prefills and raises TTFT.

**Scheduling strategies**

- **Prefill-first (simple engines):** good TTFT, poor ITL stability under long prompts.
- **Chunked prefill:** split long prompts into chunks (for example 512 to 2,048 tokens) and mix one chunk with the decode tokens in each step, capped by a token budget. This smooths ITL at a small cost to TTFT for long prompts; it is the default in vLLM and supported in SGLang and TensorRT-LLM.
- **Prefix caching:** skip prefill for cached prefixes entirely, which is the cheapest TTFT win for repeated system prompts and documents (see 27.26).
- **Disaggregation:** run prefill and decode on separate GPU pools and transfer the KV cache between them, so each pool is tuned for its bottleneck (see 27.40).

**Diagnosing with the two phases**

- High TTFT with low GPU utilisation: queueing or admission limits; scale out or raise concurrency.
- High TTFT with long prompts: prefill-bound; use prefix caching, chunked prefill, more compute, or disaggregation.
- High ITL: decode-bound; batch is too large, KV cache is full and preempting, or long prefills are interfering.
- Good average but bad p99: look for long-prompt outliers and preemptions.

**Metrics to track:** TTFT and ITL percentiles split by prompt-length bucket, prefill tokens per second, decode tokens per second, and the ratio of input to output tokens, which tells you whether your traffic is prefill-heavy (RAG, summarisation) or decode-heavy (generation, reasoning).

### 27.40 What is prefill-decode disaggregation, and when is it worth it?

**Prefill-decode disaggregation** runs the two phases of LLM inference on **separate GPU pools**: prefill workers process prompts and build the KV cache, then hand that cache over a fast network to decode workers that generate the tokens. Because prefill is compute-bound and decode is memory-bandwidth-bound (see 27.39), separating them removes the interference between them and lets each pool be sized, parallelised and even placed on different hardware for its own bottleneck.

**The problem it solves**

- In a co-located engine, a long prefill shares GPU steps with ongoing decodes, so every active user's inter-token latency spikes when a big prompt arrives. Chunked prefill smooths this but does not remove it.
- One configuration must serve both phases: the batch size, tensor-parallel degree and GPU type that suit prefill rarely suit decode.
- When SLOs are tight on both TTFT and ITL, co-located engines must run at lower load to meet both, which wastes GPUs. DistServe (2024) framed the goal as maximising **goodput**: requests per second that meet both SLOs.

**How it works**

```text
Request -> Router
            |-> Prefill pool (compute-optimised, larger TP for fast TTFT)
            |        builds KV cache for the prompt
            |        KV blocks transferred over NVLink / RDMA (NIXL, UCX, Mooncake transfer engine)
            '-> Decode pool (memory-optimised, large batches, KV-cache capacity)
                     streams tokens to the client
```

- A router or orchestrator chooses a prefill worker and a decode worker per request, ideally using KV-cache locality (a replica that already holds the prefix).
- The KV cache moves layer by layer or block by block so transfer overlaps with compute.
- Implementations: NVIDIA Dynamo, llm-d on Kubernetes, vLLM and SGLang disaggregated serving with KV connectors (NIXL, LMCache), Mooncake (Moonshot AI's KV-centric architecture), Splitwise and DistServe in research.

**The cost: moving the KV cache**

For Llama-3-70B in BF16 the KV cache is about 320 KB per token (see 27.7), so an 8k-token prompt produces about 2.6 GB. Over a 400 Gb/s RDMA link (about 50 GB/s) that is around 50 ms; over a slow Ethernet path it can be seconds. Disaggregation therefore needs **RDMA-class networking**, KV quantisation (FP8 halves it), GQA or MLA models with small caches, and transfer overlapped with prefill.

**When it is worth it**

| Situation | Disaggregate? |
| --- | --- |
| Long prompts (RAG, agents, documents) with strict ITL SLOs | Yes, strongest case |
| Very large models served at scale (large MoE across many GPUs) | Yes; production DeepSeek-style deployments use it |
| Prefill-to-decode ratio that varies a lot by tenant or time | Yes; pools scale independently |
| Small models on a few GPUs | Usually no; chunked prefill is enough |
| Short prompts, decode-heavy chat | Usually no; little interference to remove |
| No fast interconnect between pools | No; transfer cost erases the gain |

**Operational decisions**

1. **Pool ratio:** set the number of prefill to decode workers from the input-to-output token ratio of your traffic, then autoscale each pool on its own signal (prefill queue length; decode KV usage and ITL).
2. **Parallelism per pool:** larger TP on prefill for TTFT; smaller TP and many replicas, or expert parallelism, on decode for throughput.
3. **Short-prompt bypass:** short prompts can prefill locally on the decode worker to avoid the transfer.
4. **Failure handling:** a lost transfer or a failed prefill worker must trigger a retry or a local prefill, not a hung request.

**Pitfalls:** adding disaggregation before basic tuning (prefix caching, chunked prefill, right batch sizes); underestimating network design; and running pools at a fixed ratio while traffic mix changes during the day.

### 27.41 How do you serve mixture-of-experts (MoE) models efficiently?

MoE models activate only a few experts per token, so **compute per token follows active parameters but memory follows total parameters**. DeepSeek-V3 has 671B total and 37B active parameters; Qwen3-235B-A22B has 235B total and 22B active; gpt-oss-120b has about 117B total and about 5B active. Serving them well means holding all the experts somewhere, keeping each expert's matmuls large enough to be efficient, and keeping the all-to-all communication and expert load balanced. The architecture and router training are covered in 17.139 to 17.141.

**Why MoE serving is different**

- **Memory:** all experts must be resident, so a 671B model needs about 700 GB in FP8, which is many GPUs, even though each token only uses about 5 percent of the weights.
- **Bandwidth at low batch:** at batch 1 each token touches a handful of experts and decode is fast. As batch grows, different tokens hit different experts, so a step soon reads most experts' weights. Throughput per GPU only becomes good when batches are large enough that each expert receives many tokens.
- **Communication:** with experts spread across GPUs, every MoE layer does an **all-to-all**: dispatch tokens to the GPUs holding their experts, then combine the results.
- **Load imbalance:** if some experts are popular, their GPUs become stragglers and everyone waits.

**Parallelism patterns**

| Pattern | What it does | When |
| --- | --- | --- |
| Tensor parallelism for everything | Split every expert across GPUs | Small MoE (Mixtral-class) on one node; simple |
| Expert parallelism (EP) | Each GPU holds whole experts; tokens are routed to them | Large MoE; avoids splitting small expert matmuls |
| Data-parallel attention plus EP | Attention replicated per GPU (each holds its own requests' KV), MoE layers expert-parallel | Large MoE with MLA, such as DeepSeek-V3; avoids duplicating KV across TP ranks |
| Wide EP across nodes | Experts spread across many nodes (for example 32 to 96 GPUs) | Very high throughput deployments; needs fast all-to-all |

**Techniques that make it efficient**

- **Optimised all-to-all kernels** (for example DeepEP) and overlapping communication with computation, such as splitting the batch into two micro-batches so one computes while the other communicates.
- **Expert load balancing at serving time:** replicate hot experts (redundant experts) and rebalance placement from observed routing statistics (EPLB-style).
- **Grouped GEMM kernels** that run many small expert matmuls efficiently.
- **Low precision:** FP8 or FP4 experts (gpt-oss ships MXFP4 weights, which lets the 120B model fit on a single 80 GB GPU).
- **Disaggregated prefill and decode** (see 27.40): prefill and decode want very different EP and batch settings for large MoE models.
- **Offloading** for low-traffic or local use: keep cold experts in CPU memory, at a large latency cost.

**Choosing between a dense and an MoE model for serving**

- MoE gives more quality per FLOP, so at high, steady traffic with large batches it usually wins on cost per token.
- At low traffic, a dense model of similar quality may be cheaper because it needs far fewer GPUs to hold its weights.
- MoE is operationally harder: multi-node deployments, interconnect-sensitive performance, and larger blast radius per failure (see 27.45).

**What to monitor:** per-expert token counts and imbalance, all-to-all time as a share of step time, per-GPU utilisation spread (stragglers), and tokens per second per GPU at your SLO.

### 27.42 What does structured output (constrained decoding) cost in latency and quality, and how do you make it fast?

**Constrained decoding** guarantees that output matches a JSON Schema, regex or grammar by masking, at every step, the tokens that would break the format. Done well, it costs almost nothing per token and saves retries and parsing failures; done badly, it adds CPU stalls to every step and can lower answer quality. Prompting techniques for structured output are covered in Chapter 18; this answer is about the serving side.

**How it works**

1. Compile the schema or grammar into an automaton: a finite-state machine for regex and simple schemas, a pushdown automaton for recursive grammars such as nested JSON.
2. At each decode step, compute the set of allowed next tokens from the automaton state and set the logits of all other tokens to minus infinity.
3. Advance the automaton with the sampled token.

The hard part is step 2: the vocabulary has 100k to 250k tokens, many spanning several characters, so checking each against the grammar every step is expensive.

**Where the latency comes from**

| Cost | Cause | Mitigation |
| --- | --- | --- |
| Schema compilation | Building the automaton and token tables on first use | Cache compiled grammars by schema hash; precompile common schemas |
| Per-step mask computation | Checking tokens against the grammar on CPU | Engines such as XGrammar and llguidance precompute masks for most tokens and check only the few context-dependent ones; overlap mask computation with the GPU forward pass |
| Batch stalls | One request's slow mask blocks the whole batch step | Asynchronous or parallel mask generation; per-request timeouts |
| Lost speculative decoding gains | Draft tokens must also satisfy the grammar | Use engines that apply the grammar to drafts |

Modern backends (XGrammar, now the default in vLLM and SGLang, and llguidance) bring overhead close to zero for typical JSON schemas. Older approaches could add milliseconds per token on large vocabularies.

**Speed-ups beyond "no overhead"**

- **Jump-forward decoding:** when the grammar allows only one continuation (field names, braces, fixed keys), insert those tokens directly instead of decoding them one by one (SGLang does this).
- **Shorter outputs:** compact schemas with short keys and no free-text padding reduce output tokens, the main driver of latency.
- **Fewer retries:** guaranteed-valid output removes the parse-fail-retry loop, which often matters more for p95 latency than the per-token cost.

**Quality pitfalls**

- **Format before thinking:** forcing `{"answer": ...}` first makes the model commit before reasoning. Put a reasoning or evidence field before the answer field, or let the model reason in free text first and then emit the structured block.
- **Token-boundary distortion:** masking can push the model onto unusual tokenisations of the same text, which slightly shifts its distribution; quality-check against unconstrained output on your eval set.
- **Over-tight schemas:** enums that miss a real case force a wrong value instead of an "other" or "unknown"; always include an escape value.
- **Valid is not correct:** the JSON will parse, but field values still need validation and evaluation.

**Choosing an approach**

- Hosted APIs: use the provider's strict structured-output or tool-calling mode rather than parsing free text.
- Self-hosted: use the engine's built-in guided decoding with a fast backend, cache compiled schemas, and load-test with your real schemas at your real concurrency, because grammar complexity and batch size interact.

### 27.43 How do you run GPU inference on Kubernetes and keep utilisation high?

Kubernetes is the usual control plane for self-hosted inference, but GPUs are expensive, indivisible by default and slow to start, so the defaults waste money. The goals are: **pack work onto GPUs tightly, share small GPUs safely, autoscale on signals that reflect LLM load, and start replicas quickly**.

**Basic plumbing**

- **NVIDIA GPU Operator** (driver, container toolkit, device plugin, DCGM exporter) so pods can request `nvidia.com/gpu`; the AMD GPU Operator plays the same role for AMD.
- **Dedicated GPU node pools** per GPU type, with taints and tolerations so CPU pods do not land on GPU nodes, and node labels for GPU model and memory.
- **Topology awareness:** tensor-parallel replicas must sit on GPUs joined by NVLink in one node; multi-node replicas need gang scheduling (all pods or none), for example with LeaderWorkerSet, Kueue or Volcano.
- **Dynamic Resource Allocation (DRA)**, generally available since Kubernetes 1.34, gives finer-grained device requests than the device-plugin counter model.

**Sharing GPUs between small models**

| Method | Isolation | Use for |
| --- | --- | --- |
| Whole GPU per pod | Full | LLMs that use the GPU fully |
| MIG (multi-instance GPU, A100, H100 and newer) | Hardware: separate memory and compute slices, up to 7 per GPU | Several small models or tenants with predictable needs |
| MPS (multi-process service) | Partial: shared memory space, compute limits | Many small processes on one GPU |
| Time-slicing | None: processes take turns, memory not limited | Development and low-traffic internal models only |
| One engine serving many adapters | Logical | Many fine-tunes of one base model (multi-LoRA, see 22.35) |

**Autoscaling on the right signals**

- GPU utilisation from DCGM (`DCGM_FI_DEV_GPU_UTIL`) only says that some kernel was running; a GPU can read 100 percent while doing little useful work. Prefer SM activity and tensor-core activity metrics for efficiency, and engine metrics for scaling.
- Scale replicas with **KEDA or HPA on custom metrics** from the engine: waiting requests (queue depth), KV-cache usage, and p95 TTFT. Scale down slowly, and drain in-flight streams before terminating a pod.
- Scale nodes with Cluster Autoscaler or Karpenter, and keep a small buffer of warm GPU nodes (placeholder pods with low priority) because GPU nodes take minutes to arrive (see 27.17).

**Routing and load balancing**

- Plain round-robin ignores KV-cache state. Use least-outstanding-requests at minimum, and ideally KV-aware and prefix-aware routing (the Kubernetes Gateway API Inference Extension, llm-d or NVIDIA Dynamo routers), so requests sharing a prefix land on the replica that already has it cached (see 27.13).

**Utilisation levers for a mixed fleet**

1. Put interactive and batch workloads on the same GPUs with priority classes: batch jobs (evaluation runs, embeddings, offline generation) soak up idle capacity and are preempted when interactive load rises. Kueue handles quotas and fair sharing between teams.
2. Right-size: do not give a 7B model an 80 GB GPU if an L4 or a MIG slice will meet the SLO.
3. Consolidate low-traffic models onto shared multi-model or multi-LoRA servers instead of one deployment each.
4. Use spot or preemptible GPUs for batch and for stateless replicas behind a warm on-demand floor.
5. Cache weights on node-local NVMe or a shared high-throughput cache and pre-pull images, to make scaling cheap enough to use.

**What to report to leadership:** GPU allocation versus actual SM activity per team, cost per million tokens per service, idle GPU-hours, and queue time for batch jobs. Allocated-but-idle GPUs are usually the largest line of waste in an AI platform.

### 27.44 How do you choose an LLM inference engine, and how do you benchmark it fairly?

Choose the engine on **model and hardware support, the features your workload depends on, and measured goodput at your SLO**, not on a published leaderboard. Engines leapfrog each other every few months, so the durable skill is a fair benchmarking method and an architecture that lets you switch. For a comparison of Ollama, vLLM and SGLang see 26.88; for TensorRT-LLM see 27.27.

**The main options (late 2026 landscape, check current status)**

| Engine | Strengths | Watch out for |
| --- | --- | --- |
| vLLM | Broadest model and hardware support (NVIDIA, AMD, TPU, Gaudi, Neuron), large community, OpenAI-compatible server, multi-LoRA, prefix caching, disaggregation connectors | Fast-moving; pin versions and re-benchmark on upgrade |
| SGLang | RadixAttention prefix reuse, strong structured-output and agentic workloads, strong large-MoE serving (DeepSeek-style EP and disaggregation) | Somewhat narrower hardware and model coverage than vLLM |
| TensorRT-LLM | Often highest peak performance on NVIDIA, FP8 and FP4 kernels, in-flight batching, speculative decoding | NVIDIA only; more build and operations effort; new architectures can lag |
| Hugging Face TGI | Simple, well integrated with the Hugging Face hub | In maintenance mode since late 2025; Hugging Face recommends vLLM or SGLang for new deployments |
| llama.cpp, Ollama, MLX | Local, CPU and Apple silicon, GGUF quantisation (see 27.9) | Not designed for high-concurrency server workloads |
| Orchestration layers (NVIDIA Dynamo, llm-d) | Routing, disaggregation and KV management across many engine instances | Add a layer to operate; worth it at multi-node scale |

**Selection questions**

1. Does it support your model architecture, quantisation format and GPU generation well today?
2. Does it have the features you need: multi-LoRA, structured output, tool-call parsing for your model, speculative decoding, prefix caching, long context, vision inputs?
3. How does it perform on your workload at your SLO (below)?
4. Can your team operate it: metrics, health checks, upgrade cadence, community or vendor support?
5. Is the API OpenAI-compatible, so you can switch engines behind a gateway?

**How to benchmark fairly**

1. **Use your traffic shape:** replay real or realistic distributions of input and output lengths, shared-prefix rates and concurrency. A benchmark of 128-token prompts says nothing about a RAG system with 6k-token prompts.
2. **Hold variables fixed:** same model weights and precision, same GPU type and count, same parallelism, same maximum context, same sampling parameters, same output lengths (fix or force them, otherwise one engine "wins" by stopping early).
3. **Sweep load:** measure at increasing request rates or concurrency, not one point. Plot throughput against p50 and p95 TTFT and ITL.
4. **Report goodput:** the maximum request rate at which the p95 TTFT and ITL SLOs are still met. That number maps directly to GPUs needed and cost per token.
5. **Warm up and repeat:** discard warm-up, run several times, report variance.
6. **Check quality too:** run a small task eval on each engine's output; differences in chat templates, tokenizer handling or kernels can silently change answers.
7. **Measure operational behaviour:** start-up time, memory headroom, behaviour under overload (graceful queueing or OOM), and recovery after a crash.

Tools: the engines' own serving benchmarks (for example `vllm bench serve`), NVIDIA GenAI-Perf or AIPerf, and GuideLLM. Whatever tool you use, the workload definition matters more than the tool.

**Pitfalls**

- Comparing a quantised setup on one engine with BF16 on another.
- Measuring only single-request latency or only peak throughput.
- Ignoring streaming overhead, tokenisation time and the API layer, which show up in real end-to-end latency.
- Locking in through engine-specific APIs instead of an OpenAI-compatible interface behind your own gateway.

### 27.45 What failure modes do you see in production GPU inference, and how do you design for them?

GPU inference fails differently from ordinary web services: hardware faults are more frequent, a single sharded replica spans many GPUs, overload shows up as memory exhaustion rather than CPU saturation, and quality can break silently while every health check stays green. Design for **detection, isolation, graceful degradation and safe change**.

**Common failure modes**

| Failure | How it shows up | Mitigation |
| --- | --- | --- |
| GPU hardware faults (Xid errors, uncorrectable ECC or HBM errors, fallen-off-the-bus GPUs) | Pod crashes, CUDA errors, hung kernels | DCGM health checks, node problem detection, automatic cordon and drain, replace node |
| NCCL or NVLink problems in multi-GPU replicas | Requests hang; one GPU at 100 percent, others idle | Collective timeouts with watchdog restart, liveness probes that run a tiny generation, smaller shard counts |
| KV-cache exhaustion | Preemption storms, rising ITL, OOM on long contexts | Admission control on KV blocks, per-tenant context caps, route long requests to a separate pool (see 27.7) |
| Overload and queue build-up | TTFT climbs, timeouts, client retries amplify load | Rate limits, load shedding with clear 429 or 503 responses, retry budgets with jitter, priority classes (see 27.15) |
| Slow scale-out | Traffic spike arrives before replicas are ready | Warm floor, predictive scaling, fast weight loading (see 27.17) |
| Cancelled requests still generating | GPU busy on abandoned streams | Propagate client disconnects to the engine to abort generation |
| Stragglers | One slow GPU or expert slows a whole TP or EP group | Per-GPU latency metrics, expert load balancing, replace slow nodes |
| Bad rollout of engine, driver or model | Throughput drops, crashes, or subtly worse answers | Pinned versions, canary with latency and quality gates, fast rollback |
| Silent quality failures | Wrong chat template, tokenizer mismatch, broken quantisation, truncated context | Golden-prompt checks after deploy, continuous sampled evaluation, structured-output validity metrics |
| Upstream API dependency | 429s, outages or model deprecation from a provider | Multi-provider routing, fallbacks, cached responses for critical paths |
| Capacity shortage | No GPUs available in a region at scale-out time | Reserved capacity for the floor, multi-region deployment, degrade to smaller models |

**Design principles**

1. **Limit blast radius:** prefer more, smaller replicas to few large sharded ones when the model allows it; spread replicas across nodes and zones.
2. **Health checks that test the real path:** a liveness probe that generates a few tokens catches hung GPUs that a TCP check misses; readiness only after warm-up.
3. **Degrade gracefully:** fall back to a smaller model, shorter maximum output, or cached answers rather than failing outright; tell the client when it is degraded.
4. **Protect against retry storms:** retries only on safe errors, with budgets and backoff, and idempotency keys for side-effecting agent actions.
5. **Treat every change as risky:** model weights, quantisation, engine, driver, CUDA and prompt template changes all go through benchmark, canary and rollback.
6. **Observe quality, not only uptime:** latency and error dashboards will not show a broken chat template; sampled evals and format-validity metrics will.

**Runbook signals to alert on:** p95 TTFT and ITL against SLO, error and timeout rates, KV-cache usage and preemptions, queue depth, GPU Xid or ECC events, replica restarts, and evaluation score drift. In an interview, walking through how you would detect and contain one of these (for example a hung NCCL collective taking down a TP-8 replica) shows real production experience.

---

← [Chapter 26. MLOps and LLMOps](26-mlops-and-llmops.md) · [Table of contents](../README.md) · [Chapter 28. Machine Learning System Design](28-machine-learning-system-design.md) →
