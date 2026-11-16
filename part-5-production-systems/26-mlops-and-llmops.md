# Chapter 26. MLOps and LLMOps

_Part V — Production Systems and Architecture_

Deploying, monitoring and operating ML and LLM systems: drift, versioning, rollout strategies, observability and the managed platforms that support them.

This chapter contains **88 questions**.

## Contents

- **26.1** [How does Prompt Caching work?](#261-how-does-prompt-caching-work)
- **26.2** [What is the difference between prefill and decode?](#262-what-is-the-difference-between-prefill-and-decode)
- **26.3** [Explain the AI product lifecycle from ideation to production.](#263-explain-the-ai-product-lifecycle-from-ideation-to-production)
- **26.4** [What is LLMOps, and how does it differ from traditional MLOps?](#264-what-is-llmops-and-how-does-it-differ-from-traditional-mlops)
- **26.5** [How do you serve LLMs in production?](#265-how-do-you-serve-llms-in-production)
- **26.6** [What is model quantization?](#266-what-is-model-quantization)
- **26.7** [How do you monitor LLM applications in production?](#267-how-do-you-monitor-llm-applications-in-production)
- **26.8** [What is LLM observability?](#268-what-is-llm-observability)
- **26.9** [What are guardrails for LLMs, and how do you implement them?](#269-what-are-guardrails-for-llms-and-how-do-you-implement-them)
- **26.10** [How do you implement content filtering for AI outputs?](#2610-how-do-you-implement-content-filtering-for-ai-outputs)
- **26.11** [How do you estimate the cost of running an AI-powered feature in production?](#2611-how-do-you-estimate-the-cost-of-running-an-ai-powered-feature-in-production)
- **26.12** [How do you optimize LLM inference costs in production?](#2612-how-do-you-optimize-llm-inference-costs-in-production)
- **26.13** [How do you implement A/B testing for LLM systems?](#2613-how-do-you-implement-ab-testing-for-llm-systems)
- **26.14** [What is CI/CD for AI applications, and how does it differ from traditional CI/CD?](#2614-what-is-cicd-for-ai-applications-and-how-does-it-differ-from-traditional-cicd)
- **26.15** [How do you version and manage prompts in production?](#2615-how-do-you-version-and-manage-prompts-in-production)
- **26.16** [What is model versioning, and how do you handle model rollbacks?](#2616-what-is-model-versioning-and-how-do-you-handle-model-rollbacks)
- **26.17** [How do you implement rate limiting and throttling for LLM APIs?](#2617-how-do-you-implement-rate-limiting-and-throttling-for-llm-apis)
- **26.18** [How do you handle model updates and migrations without downtime?](#2618-how-do-you-handle-model-updates-and-migrations-without-downtime)
- **26.19** [What is the role of feature flags in AI deployments?](#2619-what-is-the-role-of-feature-flags-in-ai-deployments)
- **26.20** [How do you implement logging and tracing for LLM applications?](#2620-how-do-you-implement-logging-and-tracing-for-llm-applications)
- **26.21** [How do you handle PII and sensitive data in LLM inputs and outputs?](#2621-how-do-you-handle-pii-and-sensitive-data-in-llm-inputs-and-outputs)
- **26.22** [What is a gateway pattern for LLM API management?](#2622-what-is-a-gateway-pattern-for-llm-api-management)
- **26.23** [How does Token Streaming work?](#2623-how-does-token-streaming-work)
- **26.24** [How do you implement streaming responses for real-time AI applications?](#2624-how-do-you-implement-streaming-responses-for-real-time-ai-applications)
- **26.25** [How does vLLM work?](#2625-how-does-vllm-work)
- **26.26** [How does SGLang work?](#2626-how-does-sglang-work)
- **26.27** [What are the key SLAs and metrics for production AI systems (latency, throughput, availability)?](#2627-what-are-the-key-slas-and-metrics-for-production-ai-systems-latency-throughput-availability)
- **26.28** [Cloud vs. on-device model deployment: how do you choose for AI applications?](#2628-cloud-vs-on-device-model-deployment-how-do-you-choose-for-ai-applications)
- **26.29** [How do you implement fallback strategies when the primary model is unavailable or rate-limited?](#2629-how-do-you-implement-fallback-strategies-when-the-primary-model-is-unavailable-or-rate-limited)
- **26.30** [How do you implement structured output from LLMs reliably in production?](#2630-how-do-you-implement-structured-output-from-llms-reliably-in-production)
- **26.31** [How do you handle long contexts efficiently in production (context compression, prefix caching)?](#2631-how-do-you-handle-long-contexts-efficiently-in-production-context-compression-prefix-caching)
- **26.32** [What is semantic routing, and how do you implement it in a multi-model system?](#2632-what-is-semantic-routing-and-how-do-you-implement-it-in-a-multi-model-system)
- **26.33** [How do you manage secrets and API keys securely in LLM applications?](#2633-how-do-you-manage-secrets-and-api-keys-securely-in-llm-applications)
- **26.34** [Your LLM API has latency spikes during peak hours. How do you stabilize it?](#2634-your-llm-api-has-latency-spikes-during-peak-hours-how-do-you-stabilize-it)
- **26.35** [Your LLM costs are too high in production. How do you reduce costs without degrading quality?](#2635-your-llm-costs-are-too-high-in-production-how-do-you-reduce-costs-without-degrading-quality)
- **26.36** [Your application is hitting LLM provider rate limits during peak hours. How do you handle it?](#2636-your-application-is-hitting-llm-provider-rate-limits-during-peak-hours-how-do-you-handle-it)
- **26.37** [Your application depends on one LLM provider. How do you switch providers without downtime?](#2637-your-application-depends-on-one-llm-provider-how-do-you-switch-providers-without-downtime)
- **26.38** [Your AI system handles 100 requests/sec but crashes at 5000. How do you scale for concurrent requests?](#2638-your-ai-system-handles-100-requestssec-but-crashes-at-5000-how-do-you-scale-for-concurrent-requests)
- **26.39** [A traffic spike brings down your AI system. How do you handle peak traffic?](#2639-a-traffic-spike-brings-down-your-ai-system-how-do-you-handle-peak-traffic)
- **26.40** [One LLM provider outage took down your entire system. How do you eliminate single points of failure?](#2640-one-llm-provider-outage-took-down-your-entire-system-how-do-you-eliminate-single-points-of-failure)
- **26.41** [Your multi-LLM pipeline fails when one model in the chain breaks. How do you handle orchestration failure?](#2641-your-multi-llm-pipeline-fails-when-one-model-in-the-chain-breaks-how-do-you-handle-orchestration-failure)
- **26.42** [Your AI pipeline has zero visibility into which step is failing. How do you add observability?](#2642-your-ai-pipeline-has-zero-visibility-into-which-step-is-failing-how-do-you-add-observability)
- **26.43** [You quantized your LLM, but accuracy dropped significantly. How do you minimize quantization loss?](#2643-you-quantized-your-llm-but-accuracy-dropped-significantly-how-do-you-minimize-quantization-loss)
- **26.44** [One failing AI component can take down your entire platform. How do you design graceful degradation?](#2644-one-failing-ai-component-can-take-down-your-entire-platform-how-do-you-design-graceful-degradation)
- **26.45** [How do you choose the right machine learning algorithm?](#2645-how-do-you-choose-the-right-machine-learning-algorithm)
- **26.46** [What is model drift, and how do you handle it?](#2646-what-is-model-drift-and-how-do-you-handle-it)
- **26.47** [How would you handle large-scale data for training?](#2647-how-would-you-handle-large-scale-data-for-training)
- **26.48** [How do you deploy an ML model in production?](#2648-how-do-you-deploy-an-ml-model-in-production)
- **26.49** [How do you monitor a model's performance in production?](#2649-how-do-you-monitor-a-models-performance-in-production)
- **26.50** [How would you deploy a model with low-latency requirements?](#2650-how-would-you-deploy-a-model-with-low-latency-requirements)
- **26.51** [What are the common challenges when deploying ML models?](#2651-what-are-the-common-challenges-when-deploying-ml-models)
- **26.52** [Discuss scalability and latency requirements for ML systems.](#2652-discuss-scalability-and-latency-requirements-for-ml-systems)
- **26.53** [How do you ensure your model is scalable and performs well with large datasets?](#2653-how-do-you-ensure-your-model-is-scalable-and-performs-well-with-large-datasets)
- **26.54** [Explain MLOps and its key components.](#2654-explain-mlops-and-its-key-components)
- **26.55** [What is a feature store, and why is it important?](#2655-what-is-a-feature-store-and-why-is-it-important)
- **26.56** [Cloud vs. on-device model deployment: what are the trade-offs?](#2656-cloud-vs-on-device-model-deployment-what-are-the-trade-offs)
- **26.57** [Why use tracing tools like Langfuse, LangSmith, or OpenTelemetry?](#2657-why-use-tracing-tools-like-langfuse-langsmith-or-opentelemetry)
- **26.58** [What does AWS Bedrock provide?](#2658-what-does-aws-bedrock-provide)
- **26.59** [Why do teams aim for a multi-provider experience?](#2659-why-do-teams-aim-for-a-multi-provider-experience)
- **26.60** [How do you containerize and deploy an AI/agentic service, and what does your Docker setup typically look like?](#2660-how-do-you-containerize-and-deploy-an-aiagentic-service-and-what-does-your-docker-setup-typically-look-like)
- **26.61** [What does "observability" mean for an LLM/agent system in production, and what would you actually instrument?](#2661-what-does-observability-mean-for-an-llmagent-system-in-production-and-what-would-you-actually-instrument)
- **26.62** [How do you monitor for and respond to model/data drift in a deployed system?](#2662-how-do-you-monitor-for-and-respond-to-modeldata-drift-in-a-deployed-system)
- **26.63** [Walk through how you'd take a PyTorch/TensorFlow model from training to production.](#2663-walk-through-how-youd-take-a-pytorchtensorflow-model-from-training-to-production)
- **26.64** [How would you detect distribution shift between training and production data?](#2664-how-would-you-detect-distribution-shift-between-training-and-production-data)
- **26.65** [What is the difference between covariate shift, label shift, and concept drift?](#2665-what-is-the-difference-between-covariate-shift-label-shift-and-concept-drift)
- **26.66** [How would you handle training-serving skew?](#2666-how-would-you-handle-training-serving-skew)
- **26.67** [How would you version models and datasets?](#2667-how-would-you-version-models-and-datasets)
- **26.68** [How would you perform online vs offline inference?](#2668-how-would-you-perform-online-vs-offline-inference)
- **26.69** [How would you deploy a model using blue-green or canary deployment?](#2669-how-would-you-deploy-a-model-using-blue-green-or-canary-deployment)
- **26.70** [How would you monitor model performance after deployment?](#2670-how-would-you-monitor-model-performance-after-deployment)
- **26.71** [What metrics would you monitor for an ML service?](#2671-what-metrics-would-you-monitor-for-an-ml-service)
- **26.72** [How would you detect data drift automatically?](#2672-how-would-you-detect-data-drift-automatically)
- **26.73** [How would you roll back a model that performs badly in production?](#2673-how-would-you-roll-back-a-model-that-performs-badly-in-production)
- **26.74** [How do you make ML experiments reproducible, and what should an experiment tracking system capture?](#2674-how-do-you-make-ml-experiments-reproducible-and-what-should-an-experiment-tracking-system-capture)
- **26.75** [What is a model registry, and how do you design the promotion workflow from experiment to production?](#2675-what-is-a-model-registry-and-how-do-you-design-the-promotion-workflow-from-experiment-to-production)
- **26.76** [What are the MLOps maturity levels, and what does continuous training (CT) look like?](#2676-what-are-the-mlops-maturity-levels-and-what-does-continuous-training-ct-look-like)
- **26.77** [When and how should you retrain models automatically, and how do champion/challenger evaluations gate a new model?](#2677-when-and-how-should-you-retrain-models-automatically-and-how-do-championchallenger-evaluations-gate-a-new-model)
- **26.78** [What is shadow deployment, and how does it compare with canary releases, interleaving and multi-armed bandits for online evaluation?](#2678-what-is-shadow-deployment-and-how-does-it-compare-with-canary-releases-interleaving-and-multi-armed-bandits-for-online-evaluation)
- **26.79** [What is point-in-time correctness in a feature store, and how do you build leakage-free training sets?](#2679-what-is-point-in-time-correctness-in-a-feature-store-and-how-do-you-build-leakage-free-training-sets)
- **26.80** [How do you test ML systems?](#2680-how-do-you-test-ml-systems)
- **26.81** [How do you monitor a model when ground-truth labels arrive late or never?](#2681-how-do-you-monitor-a-model-when-ground-truth-labels-arrive-late-or-never)
- **26.82** [How do model serving frameworks such as Triton, KServe, BentoML and Ray Serve compare, and how does dynamic batching improve throughput?](#2682-how-do-model-serving-frameworks-such-as-triton-kserve-bentoml-and-ray-serve-compare-and-how-does-dynamic-batching-improve-throughput)
- **26.83** [How do you choose an ML pipeline orchestrator, comparing Kubeflow Pipelines, Vertex AI or SageMaker Pipelines, Airflow and Metaflow?](#2683-how-do-you-choose-an-ml-pipeline-orchestrator-comparing-kubeflow-pipelines-vertex-ai-or-sagemaker-pipelines-airflow-and-metaflow)
- **26.84** [How do you design batch inference at scale?](#2684-how-do-you-design-batch-inference-at-scale)
- **26.85** [What is hidden technical debt in ML systems, and how do you prevent it?](#2685-what-is-hidden-technical-debt-in-ml-systems-and-how-do-you-prevent-it)
- **26.86** [How do you change the embedding model behind a production vector index without downtime?](#2686-how-do-you-change-the-embedding-model-behind-a-production-vector-index-without-downtime)
- **26.87** [How do you build a data flywheel that turns production feedback into better evaluations and models for an LLM application?](#2687-how-do-you-build-a-data-flywheel-that-turns-production-feedback-into-better-evaluations-and-models-for-an-llm-application)
- **26.88** [How do Ollama, vLLM and SGLang compare, and when would you use each?](#2688-how-do-ollama-vllm-and-sglang-compare-and-when-would-you-use-each)

---

### 26.1 How does Prompt Caching work?

##### What is Prompt Caching?

- **Prompt caching** stores reusable parts of a prompt so the system does not need to fully process the same input repeatedly.
- It is especially useful when requests repeatedly contain a **large, unchanged prefix**, such as:
  - System instructions
  - Long documents
  - Tool definitions
  - Few-shot examples
  - Conversation context

##### How does it work?

1. First request sends the full prompt.
2. The system processes the prompt and creates internal representations, such as **KV-cache states**.
3. The reusable portion is cached.
4. A later request contains the same prefix plus a new question.
5. The system reuses the cached portion instead of recomputing it from scratch.
6. Only the new portion needs significant additional processing.

Example:

```txt
Request 1:
[Large system prompt + 100-page document] + "What is X?"

Request 2:
[Same system prompt + same document] + "What is Y?"
                                      ↑
                              Reuse cached prefix
```

##### Benefits

- **Lower latency**
- **Lower inference cost** when the provider charges less for cached input tokens
- Better performance for **long-context applications**
- Particularly useful for **agents, RAG systems, and multi-turn conversations**

##### Important considerations

- The cached portion generally needs to be **identical or sufficiently matching** for reuse.
- Put stable content **before dynamic content** to maximize cache hits.
- Changing the beginning of the prompt can invalidate the cache for subsequent content.
- Prompt caching is different from **response caching**:
  - Prompt caching → avoids recomputing input/context.
  - Response caching → reuses a previously generated answer.

### 26.2 What is the difference between prefill and decode?

> **See also:** [Question 17.112 — What is the difference between prefill and decode?](../part-4-generative-ai/17-large-language-model-foundations.md#17112-what-is-the-difference-between-prefill-and-decode), which covers the same question from another angle.

##### Prefill vs Decode

In LLM inference, **prefill** and **decode** are the two main phases of generating a response.

|  | Prefill | Decode |
| --- | --- | --- |
| Purpose | Process the input prompt | Generate output tokens |
| Processing | Processes many/all input tokens in parallel | Generates one token at a time |
| Main bottleneck | Compute / GPU FLOPs | Memory bandwidth / KV cache |
| Parallelism | High | Low |
| KV cache | Created for prompt tokens | Continuously extended |
| Example | Process 1,000 prompt tokens | Generate 100 response tokens |

##### 1. Prefill

- The model receives the entire input prompt.
- Transformer layers process the prompt **in parallel**.
- The model computes attention and creates the **KV cache** for those tokens.
- Produces the first next-token prediction.
- Usually **compute-intensive**.

##### 2. Decode

- The model generates the response **one token at a time**.
- For each new token, it uses the previously generated tokens' **KV cache**.
- Only the new token needs to be processed through the model.
- Continues until an end condition is reached.
- Usually **memory-bandwidth and latency-sensitive**.

##### Example

```txt
Prompt:
"Explain how transformers work in simple terms."

        ↓
     PREFILL
Process all prompt tokens
        ↓
     KV Cache
        ↓
     DECODE
"Transformers" → "are" → "neural" → "networks" → ...
```

### 26.3 Explain the AI product lifecycle from ideation to production.

##### AI Product Lifecycle: Ideation to Production

1. **Ideation & Problem Definition**
   - Identify a real business/user problem.
   - Define the target users and expected outcome.
   - Establish success metrics: accuracy, revenue, time saved, latency, cost, etc.
   - Decide whether AI is actually needed.
2. **Data Collection & Preparation**
   - Collect relevant data from databases, APIs, documents, logs, or user interactions.
   - Clean, deduplicate, validate, and label data where necessary.
   - Address privacy, security, licensing, and data-quality requirements.
3. **Solution Design**
   - Choose the appropriate approach:
     - Traditional ML
     - LLM/API
     - RAG
     - Fine-tuning
     - Agentic workflow
     - Hybrid approach
   - Design the architecture, APIs, storage, model serving, and security.
4. **Prototype / PoC**
   - Build the smallest working version.
   - Test whether the AI approach solves the problem.
   - Use existing models/services initially rather than over-engineering.
   - Validate technical feasibility and business value.
5. **Model Development**
   - Select or develop the model.
   - Train, fine-tune, or configure it if necessary.
   - Optimize prompts, retrieval, tools, agents, or model parameters.
   - Establish reproducible experiments.
6. **Evaluation**
   - Create representative evaluation datasets.
   - Measure task-specific metrics such as accuracy, precision/recall, retrieval metrics, or LLM-specific quality metrics.
   - Evaluate hallucination, robustness, safety, latency, cost, and failure cases.
   - Compare against a baseline.
7. **MVP & User Testing**
   - Integrate the AI system into the actual product.
   - Release to a small group of users.
   - Collect feedback and real-world failure cases.
   - Iterate on the model, data, prompts, and UX.
8. **Production Engineering**
   - Containerize and deploy the system.
   - Implement CI/CD, model/version management, monitoring, logging, authentication, and access control.
   - Add caching, rate limiting, retries, fallbacks, and scalability mechanisms.
   - Optimize inference cost and latency.
9. **Production Deployment**
   - Use **canary, shadow, or gradual rollout** rather than immediately exposing the system to everyone.
   - Monitor technical and business metrics.
   - Maintain rollback capability.
10. **Monitoring & Continuous Improvement**

- Monitor:
  - Model quality
  - Data/model drift
  - Latency and throughput
  - Token usage and cost
  - Errors and hallucinations
  - User feedback
  - Safety/security incidents
- Feed production insights back into the data, evaluation, and development cycle.

##### Simple Flow

**Ideation → Data → Design → PoC → Development → Evaluation → MVP → Production → Monitoring → Continuous Improvement**

### 26.4 What is LLMOps, and how does it differ from traditional MLOps?

##### What is LLMOps?

**LLMOps (Large Language Model Operations)** is the set of practices and infrastructure used to **develop, evaluate, deploy, monitor, and continuously improve LLM-based applications in production**.

It extends MLOps with capabilities specific to LLMs, such as prompt management, RAG pipelines, token/cost monitoring, and LLM evaluation.

##### LLMOps vs Traditional MLOps

| Area | Traditional MLOps | LLMOps |
| --- | --- | --- |
| Models | ML models | LLMs and LLM-based systems |
| Data | Structured/tabular, images, etc. | Text, documents, conversations, multimodal data |
| Training | Often train/fine-tune models | Often use pretrained/foundation models + prompting/RAG/PEFT |
| Evaluation | Accuracy, F1, RMSE, AUC | Quality, relevance, faithfulness, hallucination, safety, instruction following |
| Deployment | Model serving | Model/API serving, agents, RAG pipelines, tool calling |
| Versioning | Model + dataset | Model + prompts + datasets + embeddings + indexes + configurations |
| Monitoring | Accuracy, drift, latency | Token usage, cost, latency, hallucinations, safety, retrieval quality, drift |
| Optimization | Model performance/inference | Prompt, model, retrieval, context, caching, routing, quantization |
| Main cost | Training + inference | Often inference/API token cost |
| Security | Model/data security | Also prompt injection, data leakage, jailbreaks, unsafe tool use |

##### Key LLMOps Components

- **Model management:** Foundation models, fine-tuned models, model versions.
- **Prompt management:** Version, test, and deploy prompts systematically.
- **RAGOps:** Manage document ingestion, chunking, embeddings, vector indexes, retrieval, and reranking.
- **Evaluation:** Automated and human evaluation of LLM outputs.
- **Observability:** Track prompts, responses, latency, token usage, errors, and costs.
- **Guardrails:** Control unsafe outputs, sensitive information, and undesirable behavior.
- **Agent monitoring:** Track tool calls, workflows, failures, and execution traces.
- **Cost optimization:** Caching, model routing, smaller models, batching, and context optimization.

### 26.5 How do you serve LLMs in production?

##### How do you serve LLMs in production?

1. **Choose the serving approach**
   - Managed API: OpenAI, Anthropic, etc.
   - Self-hosted open-source model: Llama, Qwen, Mistral, etc.
   - Choose based on latency, cost, privacy, model control, and traffic.
2. **Use an optimized inference server**
   - Common options include **vLLM**, TensorRT-LLM, and Hugging Face TGI.
   - These provide batching, KV-cache management, streaming, and efficient GPU utilization.
3. **Optimize the model**
   - Quantization: FP16/BF16 → INT8/4-bit where appropriate.
   - Use smaller models when possible.
   - Apply speculative decoding where supported.
   - Optimize context length and prompts.
4. **Optimize request handling**
   - **Continuous batching** to process multiple requests efficiently.
   - **KV caching** to avoid recomputing previous tokens.
   - Prompt caching for repeated prefixes.
   - Streaming tokens to reduce perceived latency.
5. **Build an API layer**
   - Expose the model through REST/gRPC.
   - Add authentication, authorization, rate limiting, request validation, timeouts, and retries.
   - Handle streaming responses when required.
6. **Scale horizontally**
   - Run multiple inference replicas across GPUs/nodes.
   - Use load balancing.
   - Scale based on traffic, GPU utilization, queue depth, and latency.
   - For very large models, use tensor/pipeline parallelism.
7. **Add observability**
   - Monitor:
     - TTFT (Time to First Token)
     - Tokens/second
     - End-to-end latency
     - Throughput
     - GPU utilization/memory
     - Token usage and cost
     - Error and timeout rates
     - Model quality and hallucinations
8. **Add reliability and safety**
   - Timeouts and circuit breakers.
   - Fallback models/providers.
   - Input/output validation and guardrails.
   - Prompt-injection protection for RAG/agents.
   - Logging with appropriate privacy controls.
9. **Version and evaluate**
   - Version models, prompts, adapters, tokenizers, and configurations.
   - Run offline evaluations before deployment.
   - Use **canary or gradual releases** and maintain rollback capability.

##### Production architecture

**Client → API Gateway → Auth/Rate Limit → LLM Router → Inference Server → GPU Model**

For RAG:

**Client → API → Retrieval → Reranking → Prompt Construction → LLM → Guardrails → Response**

### 26.6 What is model quantization?

##### What is Model Quantization?

- **Model quantization** is the process of representing a model's weights and sometimes activations using **lower numerical precision**.
- For example:
  - FP32 → FP16/BF16
  - FP16/BF16 → INT8
  - FP16/BF16 → INT4
- The goal is to **reduce memory usage and improve inference efficiency** while keeping model quality acceptable.

##### Why use quantization?

- **Lower GPU/CPU memory usage**
- **Faster inference** on hardware that supports low-precision operations
- **Higher throughput**
- **Lower serving cost**
- Makes larger models possible on limited hardware

##### Example

A 7B-parameter model's raw weight storage is approximately:

- FP32: **28 GB**
- FP16/BF16: **14 GB**
- INT8: **7 GB**
- INT4: **3.5 GB**

Actual runtime memory is higher because of the **KV cache, activations, and other overhead**.

##### Types

- **Post-Training Quantization (PTQ):**
  - Quantize an already-trained model.
  - Simple and commonly used for deployment.
- **Quantization-Aware Training (QAT):**
  - Simulates quantization during training/fine-tuning.
  - Usually provides better quality when aggressive quantization is required.

##### Important trade-off

**Lower precision → lower memory/cost → potentially lower model quality.**

Therefore, quantization should be evaluated on the actual workload rather than assuming that INT4 or INT8 will always be better.

### 26.7 How do you monitor LLM applications in production?

##### How do you monitor LLM applications in production?

Monitor **four main areas: system performance, cost, LLM quality, and safety**.

1. **System performance**
   - Track:
     - TTFT (Time to First Token)
     - End-to-end latency
     - Tokens/second
     - Throughput/QPS
     - Error and timeout rates
     - GPU/CPU/memory utilization
   - Monitor p50, p95, and p99 latency.
2. **LLM usage and cost**
   - Input/output token usage
   - Cost per request/user/task
   - Model usage by application
   - Cache hit rate
   - Track expensive prompts and unusually large contexts.
3. **Response quality**
   - Track:
     - Answer correctness
     - Relevance
     - Faithfulness
     - Hallucination rate
     - Instruction following
   - Use automated evaluators plus human feedback.
   - Maintain a production evaluation dataset and regression tests.
4. **RAG monitoring**
   - Retrieval latency
   - Recall@K / Precision@K
   - Context relevance
   - Reranker performance
   - Missing or irrelevant retrieved documents
   - Embedding/index version and drift
5. **Agent monitoring**
   - Trace the complete agent workflow.
   - Monitor:
     - Tool-selection accuracy
     - Tool-call failures
     - Number of steps
     - Loop detection
     - Execution time
     - Failed workflows
   - Log structured traces rather than only the final response.
6. **Safety and security**
   - Detect prompt injection and jailbreak attempts.
   - Monitor sensitive-data leakage.
   - Validate tool calls and permissions.
   - Track unsafe outputs and policy violations.
   - Apply guardrails and access controls.
7. **Drift and regression**
   - Monitor changes in input distributions and output quality.
   - Detect degradation after changing the model, prompt, embedding model, RAG pipeline, or system configuration.
   - Compare new versions against a fixed evaluation baseline.
8. **Observability**
   - Use centralized logs, metrics, and distributed traces.
   - Correlate a request across: **API → retrieval → reranking → prompt → LLM → tools → response**.
   - Keep enough information for debugging while respecting privacy.

##### Key dashboard

**Latency + Throughput + Errors + Token Cost + Quality + Retrieval Quality + Safety + User Feedback**

### 26.8 What is LLM observability?

##### What is LLM Observability?

- **LLM observability** is the practice of monitoring and tracing an LLM application to understand **what happened, why it happened, and how well it performed** in production.
- It goes beyond traditional application monitoring because LLM behavior is probabilistic and depends heavily on prompts, context, retrieved data, and model outputs.

##### What do you monitor?

- **Performance**
  - TTFT
  - End-to-end latency
  - Tokens/second
  - Throughput
  - Errors and timeouts
- **LLM behavior**
  - Prompt and response quality
  - Hallucinations
  - Instruction following
  - Output relevance and correctness
  - Model failures
- **Token & cost usage**
  - Input/output tokens
  - Cost per request
  - Context size
  - Cache hit rate
- **RAG**
  - Retrieved documents
  - Retrieval scores
  - Context relevance
  - Retrieval latency
  - Reranker performance
- **Agents**
  - Agent reasoning/workflow traces
  - Tool calls
  - Tool-call failures
  - Number of steps
  - Loops and unexpected behavior
- **Safety**
  - Prompt injection
  - Sensitive-data leakage
  - Unsafe outputs
  - Unauthorized tool usage

##### Why is tracing important?

Instead of only seeing:

```txt
User → Bad Answer
```

Observability lets you inspect:

```txt
User Query
   ↓
Query Rewrite
   ↓
Retriever
   ↓
Retrieved Documents
   ↓
Reranker
   ↓
Prompt
   ↓
LLM
   ↓
Tool Calls
   ↓
Final Response
```

This makes it possible to identify **where the failure occurred**.

##### LLM Monitoring vs LLM Observability

- **Monitoring:** Tells you **that something is wrong**.
  - Example: latency increased from 2s to 8s.
- **Observability:** Helps you understand **why it is wrong**.
  - Example: latency increased because retrieval returned too many documents, increasing the LLM context size.

### 26.9 What are guardrails for LLMs, and how do you implement them?

##### What are LLM Guardrails?

- **Guardrails** are rules, filters, and controls that constrain an LLM application to behave **safely, reliably, and within defined boundaries**.
- They can be applied to the **input, model process, tool calls, and output**.

##### How do you implement them?

1. **Input guardrails**
   - Detect prompt injection and jailbreak attempts.
   - Validate input format and length.
   - Detect sensitive or restricted information.
   - Block or sanitize malicious inputs.
2. **Prompt/context guardrails**
   - Use clear system instructions.
   - Separate trusted instructions from untrusted user/RAG content.
   - Restrict what retrieved content can instruct the model to do.
   - Limit context size and permitted sources.
3. **Tool-use guardrails**
   - Use allowlists for tools.
   - Validate tool arguments against schemas.
   - Enforce authentication and authorization.
   - Apply least-privilege permissions.
   - Require human approval for high-impact actions.
   - Never allow the LLM to directly bypass application-level authorization.
4. **Output guardrails**
   - Validate structured outputs against schemas.
   - Detect unsafe, irrelevant, or policy-violating responses.
   - Filter sensitive information.
   - Ground factual responses against trusted sources when required.
   - Use a second model/classifier for additional validation when appropriate.
5. **RAG guardrails**
   - Restrict retrieval by user permissions and tenant.
   - Validate document sources.
   - Detect prompt injection inside retrieved documents.
   - Require citations or evidence for factual answers where appropriate.
6. **Operational guardrails**
   - Rate limiting and quotas.
   - Token and cost limits.
   - Timeouts and circuit breakers.
   - Logging and monitoring.
   - Model fallbacks.
   - Human-in-the-loop escalation.

##### Defense-in-depth

A production system should not rely only on the system prompt:

**Input validation → Retrieval controls → Prompt constraints → Tool authorization → Output validation → Monitoring → Human escalation**

###### Input and Output Guardrails

Guardrails are **controls around an AI system that detect, block, or modify unsafe, invalid, or unauthorized inputs and outputs**.

###### 1. Input Guardrails

Validate the request **before sending it to the model**.

- **Content safety:** Detect prohibited or harmful requests.
- **Prompt-injection detection:** Identify attempts to override system instructions.
- **PII detection:** Detect sensitive personal information when the application should not process it.
- **Input validation:** Enforce length, format, schema, and file-type limits.
- **Authentication and authorization:** Verify the user can access the requested data or functionality.
- **Rate limiting:** Prevent abuse and resource exhaustion.
- **Topic/domain restrictions:** Reject requests outside the application's intended scope.

```txt
User Input
    ↓
Authentication / Authorization
    ↓
Input Validation
    ↓
Safety + Injection Detection
    ↓
LLM / Agent
```

###### 2. Output Guardrails

Validate the model's response **before returning it to the user or executing an action**.

- **Safety filtering:** Detect unsafe or policy-violating content.
- **PII / secret detection:** Prevent sensitive information from being exposed.
- **Grounding checks:** Verify important claims against trusted sources.
- **Schema validation:** Ensure structured outputs match the required format.
- **Hallucination checks:** Detect unsupported claims.
- **Tool-call validation:** Validate tool name, parameters, authorization, and action scope before execution.
- **Business-rule validation:** Ensure the response follows application-specific rules.
- **Redaction:** Remove sensitive information when appropriate.

```txt
LLM Output
    ↓
Schema Validation
    ↓
Safety / PII Checks
    ↓
Grounding / Factuality Checks
    ↓
Tool Authorization / Business Rules
    ↓
Return or Block / Regenerate
```

###### Best Practices

- **Defense in depth:** Don't depend on a single classifier or prompt.
- **Fail closed for high-risk actions:** Block when validation fails.
- **Keep authorization outside the LLM:** The model should never be the final authority for permissions.
- **Use deterministic validation where possible:** Schema, permissions, limits, and business rules should be code-enforced.
- **Log guardrail decisions:** Track blocks, violations, false positives, and false negatives.
- **Test continuously:** Include guardrail tests in red-team and regression suites.

### 26.10 How do you implement content filtering for AI outputs?

##### How do you implement content filtering for AI outputs?

- **Define content policies**
  - Decide what outputs are allowed, restricted, or blocked.
  - Define categories such as harmful content, sensitive information, harassment, or policy violations.
- **Use an output moderation layer**
  - Send the generated response through a **moderation model, classifier, or rule-based filter**.
  - Classify the output against predefined safety categories.
- **Apply deterministic checks**
  - Regex/pattern matching for known sensitive information.
  - PII detection for emails, phone numbers, credentials, etc.
  - Schema validation for structured outputs.
  - Blocklists/allowlists where appropriate.
- **Use confidence thresholds**
  - High-risk/high-confidence → block or replace.
  - Medium confidence → send for additional validation or human review.
  - Low risk → allow.
- **Regenerate or transform**
  - If the output violates policy, don't return it directly.
  - Regenerate with safer constraints or return a predefined safe response.
- **Ground and validate factual content**
  - For RAG applications, verify important claims against retrieved trusted sources.
  - Reject or flag responses that aren't adequately supported.
- **Monitor continuously**
  - Log filtering decisions and failure categories.
  - Track false positives and false negatives.
  - Regularly update policies, classifiers, and test datasets.

##### Production pipeline

**LLM → Output Validation → Moderation/Classification → PII/Security Checks → Policy Decision → Response**

###### Implementing Content Safety Filters

I use **defense in depth**, with safety checks both before and after the LLM.

###### 1. Input Filtering

Before sending the request to the model:

- Detect harmful or prohibited content.
- Detect jailbreak and prompt-injection attempts.
- Detect sensitive information or PII when relevant.
- Apply topic/domain restrictions.
- Enforce input length and file-size limits.
- Rate-limit abusive requests.

###### 2. Model-Level Safety

- Use a model with appropriate safety training.
- Provide clear system-level safety policies.
- Restrict the model's capabilities to the application's intended use.
- Apply least-privilege access to tools and external systems.

###### 3. Output Filtering

After generation:

- Run the response through a **content safety classifier**.
- Detect categories such as violence, sexual content, hate, harassment, self-harm, illegal activity, or sensitive information, depending on the application's policy.
- Check for PII or secrets.
- Validate structured output against a schema.
- For RAG, verify that important claims are grounded in trusted sources.
- Block, redact, regenerate, or escalate responses that violate policy.

```txt
User Input
    ↓
Input Safety Filter
    ↓
Prompt / LLM
    ↓
Output Safety Filter
    ↓
Grounding / PII / Schema Checks
    ↓
 ┌──────────────┐
 │ Safe         │ → Return
 │ Unsafe       │ → Block / Regenerate / Escalate
 └──────────────┘
```

###### 4. Continuous Evaluation

- Build a safety evaluation dataset.
- Include normal, edge-case, and adversarial inputs.
- Measure false-positive and false-negative rates.
- Red-team the filters regularly.
- Monitor production violations.
- Add discovered failures to the regression suite.

###### Important Design Principle

**Don't rely on the LLM's system prompt as the only safety mechanism.** Critical controls such as authorization, tool permissions, schema validation, and business rules should be enforced by deterministic application code.

### 26.11 How do you estimate the cost of running an AI-powered feature in production?

##### How do you estimate AI feature cost in production?

Break the cost into **usage, infrastructure, and operational costs**.

1. **Estimate traffic**
   - Requests per day/month
   - Peak QPS
   - Average requests per user
   - Expected growth
2. **Estimate token usage**
   - Average input tokens/request
   - Average output tokens/request
   - Context size for RAG/agents
   - Number of LLM calls per request

   For agentic systems, count **all model calls**, not just the final response.
3. **Calculate model/API cost**
   - Input-token cost
   - Output-token cost
   - Cached-token cost, if applicable
   - Multiply by expected monthly usage.
4. **Calculate RAG costs**
   - Embedding generation
   - Vector database storage
   - Vector queries
   - Reranking
   - Document processing and re-indexing
5. **Calculate infrastructure**
   - GPU/CPU compute
   - Model serving
   - Kubernetes/container infrastructure
   - Database and object storage
   - Network/egress
   - Monitoring and logging
6. **Include operational costs**
   - Data pipelines
   - Evaluation
   - Fine-tuning/training
   - Engineering and maintenance
   - Backup and disaster recovery
7. **Account for caching and optimization**
   - Prompt caching
   - Response caching
   - Smaller/cheaper models
   - Quantization
   - Batching
   - Model routing
   - Context reduction
8. **Calculate unit economics**

$$
\begin{aligned}
\text{Monthly Cost} = {} & \text{LLM Cost} \\
& + \text{Embedding/Reranking Cost} \\
& + \text{Compute} \\
& + \text{Storage} \\
& + \text{Network} \\
& + \text{Monitoring} \\
& + \text{Other Operational Costs}
\end{aligned}
$$

Then calculate:

$$
\text{Cost per request} = \frac{\text{Monthly Cost}}{\text{Monthly Requests}}
$$

and, where useful:

$$
\text{Cost per user} = \frac{\text{Monthly Cost}}{\text{Monthly Active Users}}
$$

##### Example

Suppose:

- 1M requests/month
- 2,000 input tokens/request
- 500 output tokens/request
- Average 2 LLM calls/request

You first calculate total input/output tokens across **all model calls**, apply the provider's pricing, then add embeddings, vector DB, compute, networking, and observability costs.

Finally, compare the result against the **business value per request/user**.

### 26.12 How do you optimize LLM inference costs in production?

##### How do you optimize LLM inference costs in production?

1. **Use the smallest model that meets quality requirements**
   - Route simple tasks to smaller/cheaper models.
   - Use larger models only for complex requests.
   - Implement **model routing** based on task complexity.
2. **Reduce token usage**
   - Remove unnecessary prompt instructions.
   - Reduce conversation history.
   - Retrieve only relevant RAG chunks.
   - Limit maximum output tokens.
   - Summarize or compact long context.
3. **Use prompt caching**
   - Cache repeated system prompts, tool definitions, and static context.
   - Reduces repeated processing and potentially input-token costs.
4. **Cache responses**
   - Cache identical or semantically equivalent requests when responses can safely be reused.
   - Particularly effective for frequently repeated queries.
5. **Use batching**
   - Batch multiple inference requests when latency requirements allow.
   - Improves GPU utilization and reduces cost per request.
6. **Quantize models**
   - Use FP16/BF16, INT8, or INT4 depending on hardware and quality requirements.
   - Reduces GPU memory and can improve inference efficiency.
7. **Optimize serving**
   - Use efficient inference engines such as **vLLM**.
   - Enable continuous batching and efficient KV-cache management.
   - Use tensor/pipeline parallelism for large models when necessary.
8. **Use speculative decoding**
   - A smaller draft model proposes tokens.
   - The larger model verifies them.
   - Can increase generation speed and improve GPU utilization.
9. **Optimize RAG**
   - Better chunking and retrieval reduce unnecessary context.
   - Use reranking to retrieve fewer but more relevant documents.
   - Avoid sending entire documents to the LLM.
10. **Optimize agent workflows**

- Reduce unnecessary agent iterations.
- Avoid repeated tool calls and redundant LLM calls.
- Use deterministic code for simple operations instead of asking the LLM.

1. **Monitor cost continuously**

- Track:
  - Cost/request
  - Input/output tokens
  - Cache hit rate
  - Model usage
  - GPU utilization
  - Cost per user/task
- Set budgets and alerts.

##### Optimization strategy

**Measure → Identify the biggest cost driver → Optimize → Evaluate quality → Deploy gradually → Monitor**

### 26.13 How do you implement A/B testing for LLM systems?

##### How do you implement A/B testing for LLM systems?

1. **Define the hypothesis**
   - Example: “Model B should improve answer quality without increasing latency or cost significantly.”
2. **Define evaluation metrics**
   - Quality: correctness, relevance, faithfulness, helpfulness.
   - RAG: Recall@K, MRR, context relevance.
   - System: latency, TTFT, error rate.
   - Business: conversion, retention, task completion.
   - Cost: tokens/request and cost/request.
3. **Create two variants**
   - **A:** Current production model/system.
   - **B:** New model, prompt, RAG pipeline, or agent configuration.
   - Change only the component being tested when possible.
4. **Randomly split traffic**
   - For example, 90% → A and 10% → B initially.
   - Use consistent user/request assignment so the same user doesn't frequently switch variants.
5. **Collect results**
   - Log variant ID, inputs, outputs, latency, token usage, errors, and evaluation signals.
   - Protect sensitive data in logs.
6. **Evaluate statistical significance**
   - Compare A and B using appropriate statistical tests or confidence intervals.
   - Make sure the sample size is sufficient before declaring a winner.
7. **Monitor guardrails**
   - Check hallucination rate, safety violations, failures, unexpected costs, and latency regressions.
   - A variant shouldn't win on quality if it creates unacceptable safety or reliability problems.
8. **Gradual rollout**
   - Start with a small percentage.
   - If B performs better and remains stable, gradually increase traffic.
   - Keep the ability to roll back quickly.

##### Example

```txt
                 Traffic
                    |
              Random Assignment
                /           \
             A /             \ B
        Current Model      New Model
             |                 |
             +-------+---------+
                     |
              Compare Metrics
                     |
                Statistical
                 Evaluation
                     |
               Rollout / Rollback
```

##### Important for LLMs

LLM outputs are **non-deterministic**, so evaluating only exact string matches is usually insufficient. Use a combination of **automated evaluators, deterministic task metrics, human evaluation, and production/user metrics**.

### 26.14 What is CI/CD for AI applications, and how does it differ from traditional CI/CD?

##### CI/CD for AI applications

**CI/CD for AI applications** extends traditional Continuous Integration and Continuous Delivery to cover not only application code, but also **data, models, prompts, evaluation, and AI-specific behavior**.

- **CI (Continuous Integration):**
  - Automatically test application and AI code.
  - Validate data schemas and data quality.
  - Test preprocessing and feature pipelines.
  - Run model evaluation and regression tests.
  - For GenAI, evaluate things such as response quality, hallucination, safety, and prompt changes.
- **CD (Continuous Delivery/Deployment):**
  - Build and deploy the application and AI components.
  - Package and version models or other AI artifacts.
  - Validate model performance before promotion.
  - Deploy using staging, canary, or A/B testing.
  - Automatically roll back if production metrics degrade.
- **Continuous Training (CT):**
  - Automatically retrain models when new data becomes available or when model performance deteriorates.

##### Traditional CI/CD vs AI CI/CD

| Traditional CI/CD | AI CI/CD |
| --- | --- |
| Primarily code | Code + data + models |
| Unit/integration tests | Code + data + model/evaluation tests |
| Deterministic behavior is common | AI outputs can be probabilistic |
| Build application artifacts | Build applications + model/AI artifacts |
| Deploy application | Deploy application + models/AI pipelines |
| Monitor uptime, errors, latency | Also monitor accuracy, drift, quality, cost, safety |
| Code changes trigger pipeline | Code, data, model, prompt, or configuration changes can trigger pipelines |
| Usually no retraining | May include continuous training/evaluation |

The key difference is that **traditional CI/CD asks, “Does the new code work?”**, while AI CI/CD must also ask **“Is the data valid, is the model still good, and is the AI system behaving correctly?”**

### 26.15 How do you version and manage prompts in production?

##### How do you version and manage prompts in production?

- **Store prompts as code/configuration** in Git rather than hard-coding them inside application logic.
- **Assign explicit versions** such as : `customer-support-v1.2`and record the exact prompt version used for every production request.
- **Separate prompt templates from variables**, keeping user data, retrieved context, and system instructions separate.
- **Use a prompt registry** or centralized prompt-management system for teams to discover, review, and manage prompts.
- **Review changes through pull requests** with code review, testing, and approval before deployment.
- **Run automated evaluations** against a fixed test dataset to compare the new prompt with the current production version.
- **Use staging and canary/A/B deployments** to safely test new prompts with a small percentage of traffic.
- **Track prompt + model + parameters together**, including model version, temperature, tools, retrieval configuration, and other relevant settings.
- **Monitor production metrics** such as task success, quality scores, hallucination rate, latency, token usage, and cost.
- **Maintain rollback capability** so a problematic prompt can immediately be switched back to the previous version.

### 26.16 What is model versioning, and how do you handle model rollbacks?

##### What is model versioning?

**Model versioning** is the practice of assigning and tracking unique versions of ML models and their associated artifacts so that a model can be reproduced, evaluated, deployed, and rolled back reliably.

- Store each model version in a **model registry**.
- Track the **model artifact, training code, dataset/version, features, hyperparameters, and dependencies**.
- Record evaluation metrics such as accuracy, F1, latency, and business metrics.
- Promote models through stages such as **development → staging → production**.
- Keep the previous production model available for rollback.

##### How do you handle model rollbacks?

- **Detect the problem** through monitoring, such as performance degradation, data drift, increased latency, or errors.
- **Identify the last known-good model version** from the model registry.
- **Switch traffic back** to that version using deployment mechanisms such as blue-green, canary, or feature flags.
- **Verify production metrics** after rollback.
- Investigate the failed model and dataset/pipeline changes before attempting redeployment.
- Keep deployment and model metadata so the rollback is **reproducible and auditable**.

### 26.17 How do you implement rate limiting and throttling for LLM APIs?

##### How do you implement rate limiting and throttling for LLM APIs?

- **Rate limiting:** Restrict how many requests a user, API key, or tenant can make within a time window.
- **Token-based limiting:** Limit **input/output tokens per minute (TPM)** in addition to requests per minute (RPM), since LLM cost depends heavily on tokens.
- **Use a distributed rate limiter:** Implement it with Redis or an API gateway so limits work consistently across multiple application instances.
- **Use algorithms such as:**
  - Token bucket
  - Leaky bucket
  - Sliding window
  - Fixed window
- **Apply different limits by tenant/user:** For example, premium customers can have higher quotas than free users.
- **Concurrency limits:** Restrict the number of simultaneous LLM requests to prevent resource exhaustion.
- **Queue requests:** When traffic exceeds the immediate capacity, queue lower-priority requests rather than sending everything directly to the LLM provider.
- **Backoff and retries:** Handle provider `429` responses using exponential backoff with jitter.
- **Set timeouts and maximum token limits:** Prevent individual requests from consuming excessive resources.
- **Monitor usage:** Track RPM, TPM, concurrency, latency, `429` errors, token consumption, and cost per tenant.
- **Use graceful degradation:** When limits are reached, return a clear response, queue the request, or route to a cheaper/smaller model when appropriate.

### 26.18 How do you handle model updates and migrations without downtime?

- **Deploy the new model alongside the existing model** instead of replacing it directly.
- **Version everything:** model artifact, code, dependencies, preprocessing, tokenizer, prompts, and configuration.
- **Validate in staging** using offline evaluation and representative test data.
- **Use shadow deployment:** send copies of production requests to the new model without using its responses for users.
- **Use canary deployment:** gradually route traffic, e.g., 5% → 25% → 50% → 100%, while monitoring performance.
- **Use blue-green deployment:** keep the old model (blue) running while the new model (green) is deployed and validated; switch traffic when ready.
- **Maintain backward compatibility** for APIs, input/output schemas, feature formats, and preprocessing pipelines.
- **Monitor both versions** for accuracy/quality, latency, errors, token usage, cost, and business metrics.
- **Keep the previous model warm** so traffic can immediately be redirected if the new model fails.
- **Roll back quickly** using traffic routing or feature flags rather than redeploying the old model from scratch.

### 26.19 What is the role of feature flags in AI deployments?

**Feature flags** allow you to enable, disable, or gradually roll out AI features or model versions **without redeploying the application**.

- **Gradual model rollout:** Send a small percentage of traffic to a new model before full deployment.
- **A/B testing:** Compare two models, prompts, or AI pipelines using real traffic.
- **Instant rollback:** Disable a problematic model or feature immediately.
- **Risk control:** Release new AI capabilities to selected users or tenants first.
- **Model routing:** Dynamically choose between models based on user, workload, cost, or availability.
- **Prompt testing:** Switch between different prompt versions without changing application code.
- **Emergency controls:** Quickly disable an AI feature if it produces unsafe, incorrect, or unexpectedly expensive outputs.
- **Experimentation:** Test different configurations such as model parameters, RAG strategies, or agent behaviors.

### 26.20 How do you implement logging and tracing for LLM applications?

- **Structured logging:** Log each request as structured JSON with fields such as request ID, user/tenant ID, model, prompt version, latency, token usage, status, and cost.
- **Distributed tracing:** Use OpenTelemetry or similar tools to trace the complete request across API → retrieval → reranking → LLM → tool calls → response.
- **Trace LLM calls:** Capture model name, parameters, input/output token counts, latency, and provider errors.
- **Trace RAG pipelines:** Record embedding generation, vector search, retrieved document IDs, reranking, and context size.
- **Trace agents:** Track each agent step, tool selected, tool parameters, tool response, retries, and execution time.
- **Correlation IDs:** Use a single trace/request ID across all services so one user request can be followed end-to-end.
- **Monitor key metrics:** Track latency, throughput, errors, token usage, cost, cache hit rate, retrieval quality, and model response quality.
- **Protect sensitive data:** Avoid logging passwords, API keys, personal information, or sensitive prompts/responses unnecessarily. Redact or mask sensitive content.
- **Centralize observability:** Send logs and traces to systems such as OpenTelemetry-compatible collectors, CloudWatch, Datadog, Grafana, or an LLM observability platform.
- **Set alerts:** Trigger alerts for increased latency, error rates, token consumption, cost spikes, or degraded AI quality.

### 26.21 How do you handle PII and sensitive data in LLM inputs and outputs?

- **Identify sensitive data:** Detect PII such as names, emails, phone numbers, addresses, financial information, credentials, and other confidential data.
- **Minimize data:** Send only the information the LLM actually needs; avoid sending unnecessary PII.
- **Redact or mask PII:** Use a PII-detection layer before sending data to the model.
- **Tokenize sensitive values:** Replace sensitive information with placeholders such as `<CUSTOMER_ID>` and restore it afterward when necessary.
- **Encrypt data:** Use encryption in transit and at rest, including secure storage for prompts, responses, and logs.
- **Control access:** Apply RBAC/ABAC and least-privilege access to prompts, outputs, logs, and model infrastructure.
- **Protect logs:** Never blindly log raw prompts and responses. Redact sensitive fields before storing them.
- **Use secure model providers:** Understand their data-retention, training, and privacy policies and configure enterprise privacy controls where available.
- **Prevent data leakage:** Validate model outputs and prevent the model from exposing confidential information from prompts, retrieved documents, or tools.
- **Secure RAG:** Enforce document-level authorization before retrieving content so users cannot retrieve documents they are not permitted to access.
- **Retention controls:** Define how long prompts, outputs, traces, and user data are retained and delete them when no longer required.
- **Audit and monitor:** Track access and detect unusual patterns or potential data leakage.

###### Handling PII in LLM Inputs and Outputs

I use a **data-minimization and defense-in-depth approach** so sensitive information is protected before, during, and after LLM processing.

###### Input Protection

- **Detect PII:** Use a PII detection/NER service to identify names, emails, phone numbers, addresses, IDs, etc.
- **Redact or tokenize:** Replace PII with placeholders before sending data to the LLM.

```txt
"John's email is john@example.com"
            ↓
"[PERSON]'s email is [EMAIL]"
```

- **Data minimization:** Send only the information required for the task.
- **Authorization:** Verify the user is allowed to access the underlying data.
- **Encryption:** Protect data in transit and at rest.
- **Provider controls:** Understand whether the LLM provider stores inputs or uses them for training.
- **Prompt/log hygiene:** Avoid storing raw PII in prompts, traces, debugging logs, or evaluation datasets.

###### Output Protection

- **Scan generated output for PII** before returning it.
- **Redact sensitive information** when it should not be exposed.
- **Validate access permissions:** Don't allow the model to reveal information simply because it generated it.
- **Prevent cross-user leakage:** Enforce tenant/user-level authorization at the application and retrieval layers.
- **Audit and monitor:** Log security events without unnecessarily logging the sensitive data itself.

###### For RAG Applications

Authorization should happen **before retrieval**, not only after generation:

```txt
User
 ↓
Authentication + Authorization
 ↓
Authorized Retrieval
 ↓
LLM
 ↓
PII / Security Validation
 ↓
Response
```

The LLM should **never be the final access-control mechanism**.

### 26.22 What is a gateway pattern for LLM API management?

An **LLM gateway** is a centralized service that sits between your application and one or more LLM providers. It provides a single interface for managing, securing, routing, and monitoring LLM requests.

**Typical flow:**

`Application → LLM Gateway → Model Provider(s)`

- **Authentication & authorization:** Validate API keys, users, and tenant permissions.
- **Rate limiting:** Control requests, tokens, and concurrency.
- **Model routing:** Route requests to OpenAI, Anthropic, Gemini, or self-hosted models based on cost, latency, capability, or availability.
- **Fallbacks:** Automatically switch to another model/provider when one fails or becomes unavailable.
- **Caching:** Cache suitable responses to reduce latency and API costs.
- **Logging & tracing:** Centralize usage, latency, token consumption, errors, and costs.
- **Security:** Detect/redact PII and enforce content and access policies.
- **Cost management:** Track usage by application, user, team, or tenant and enforce budgets.
- **Model abstraction:** Applications use one consistent API even when the underlying provider or model changes.
- **Prompt/model governance:** Manage approved models, prompt versions, and deployment policies.

### 26.23 How does Token Streaming work?

##### How does Token Streaming work?

**Token streaming** is a technique where an LLM sends generated tokens to the client **incrementally as they are produced**, instead of waiting for the entire response to be generated.

**Flow:**

`User → API → LLM → Token 1 → Token 2 → Token 3 → ... → Final response`

- The client sends a request to the LLM.
- The model starts generating tokens.
- Instead of buffering the complete response, the server sends each generated token or small chunk immediately.
- The frontend progressively displays the response.
- The connection remains open until generation finishes or an error occurs.
- Common implementations use **Server-Sent Events (SSE)** or **WebSockets**.

##### Why use token streaming?

- **Lower perceived latency:** Users see the response quickly instead of waiting for the complete generation.
- **Better UX:** The response appears progressively, similar to ChatGPT.
- **Long responses feel faster:** Users can start reading while the model is still generating.
- **Early termination:** Users can stop generation before the complete response is produced.

##### Example

Without streaming:

`Request → [wait 5 seconds] → Complete response`

With streaming:

`Request → "The" → " model" → " generates" → " tokens" → ...`

### 26.24 How do you implement streaming responses for real-time AI applications?

##### How do you implement streaming responses for real-time AI applications?

- **Use a streaming protocol:** Typically **Server-Sent Events (SSE)** for one-way token streaming or **WebSockets** for bidirectional real-time communication.
- **Stream tokens from the LLM:** Configure the LLM provider to return generated tokens/chunks incrementally.
- **Stream through the backend:** The backend receives each chunk and immediately forwards it to the client instead of buffering the full response.
- **Progressive rendering:** The frontend displays tokens as they arrive, creating a real-time response experience.
- **Handle cancellation:** Allow users to stop generation and cancel the underlying LLM request to save tokens and cost.
- **Handle errors and disconnects:** Detect provider failures, network disconnects, timeouts, and reconnect when appropriate.
- **Track usage:** Record token usage, latency, model version, and cost even when the response is streamed.
- **Apply backpressure:** Prevent a fast producer from overwhelming slower clients or downstream services.
- **Secure the stream:** Authenticate the connection and enforce authorization, rate limits, and tenant quotas.
- **Use heartbeats/timeouts:** Keep long-lived connections healthy and detect dead connections.

**Typical architecture:**

`Frontend → SSE/WebSocket → API Server → LLM → Token Stream → API Server → Frontend`

### 26.25 How does vLLM work?

##### How does vLLM work?

**vLLM** is a high-performance inference and serving engine for LLMs. It is designed to serve models efficiently, especially under high-concurrency workloads.

- **Loads the model:** vLLM loads the LLM weights into GPU memory and prepares the model for inference.
- **Continuous batching:** Instead of waiting for a fixed batch of requests, vLLM dynamically adds new requests to ongoing batches. This keeps the GPU highly utilized.
- **PagedAttention:** vLLM manages the **KV cache** in non-contiguous memory blocks, similar to virtual memory in operating systems. This reduces memory fragmentation and allows more concurrent requests.
- **Efficient KV-cache management:** Previously computed attention keys and values are stored and reused during autoregressive generation rather than recomputed for every token.
- **Token scheduling:** vLLM continuously schedules requests and allocates GPU resources as requests progress through generation.
- **Streaming:** Generated tokens can be streamed to clients as soon as they become available.
- **OpenAI-compatible API:** vLLM provides an API compatible with the OpenAI API, making it relatively easy to replace an external LLM endpoint with a self-hosted model.
- **Distributed inference:** It supports techniques such as tensor and pipeline parallelism for serving large models across multiple GPUs.

##### Why is vLLM fast?

The key idea is **efficient GPU and KV-cache utilization**.

Traditional serving can waste GPU memory because each request may reserve large contiguous KV-cache regions. vLLM's **PagedAttention** divides KV-cache memory into blocks and allocates them dynamically, allowing many more requests to share GPU memory efficiently.

### 26.26 How does SGLang work?

##### How does SGLang work?

**SGLang (Structured Generation Language)** is an LLM serving and programming framework designed to make **structured generation and high-throughput inference** efficient.

- **Request processing:** An application sends a prompt or structured generation request to the SGLang runtime.
- **RadixAttention:** SGLang uses a **radix-tree-based KV-cache** to reuse previously computed attention states for shared prompt prefixes.
- **Prefix caching:** If multiple requests share the same system prompt, few-shot examples, or conversation history, SGLang can reuse the cached computation instead of recomputing it.
- **Continuous batching:** New requests can be dynamically added to batches while other requests are generating.
- **Structured generation:** SGLang provides primitives for constrained generation, JSON/regex formats, tool calls, and multi-step LLM programs.
- **Efficient scheduling:** The runtime schedules requests and manages GPU memory to maximize throughput.
- **Multi-model serving:** It supports serving different models and distributed inference across GPUs.
- **Streaming:** Generated tokens can be streamed to clients as they become available.

##### SGLang vs vLLM

| SGLang | vLLM |
| --- | --- |
| Strong focus on structured generation and LLM programs | Strong focus on high-performance general LLM serving |
| **RadixAttention** for prefix/KV-cache reuse | **PagedAttention** for efficient KV-cache management |
| Excellent for repeated/shared prompts | Excellent general-purpose serving |
| Useful for complex multi-step LLM workflows | Simple and scalable model serving |
| Continuous batching | Continuous batching |
| Streaming and OpenAI-compatible APIs | Streaming and OpenAI-compatible APIs |

### 26.27 What are the key SLAs and metrics for production AI systems (latency, throughput, availability)?

##### Key SLAs (Service Level Agreements) and metrics for production AI systems

Production AI systems should be measured across **latency, throughput, availability, reliability, quality, and cost**.

- **Latency**
  - **TTFT (Time to First Token):** How quickly the first token is returned.
  - **E2E latency:** Total time to complete the request.
  - **TPOT (Time Per Output Token):** Time required to generate each additional token.
  - Track **P50, P95, and P99**, not just averages.
- **Throughput**
  - Requests per second (**RPS**).
  - Tokens per second (**TPS**).
  - Concurrent requests/users.
  - GPU utilization and batch utilization.
- **Availability**
  - Percentage of time the AI service is operational.
  - Example SLA: **99.9% availability**.
  - Track downtime, failed requests, and dependency/provider outages.
- **Reliability**
  - Error rate and timeout rate.
  - HTTP 5xx/4xx rates.
  - LLM provider `429` rate.
  - Retry and fallback rate.
- **AI quality**
  - Task success rate.
  - Accuracy/F1 for traditional ML.
  - Groundedness and faithfulness for RAG.
  - Hallucination rate.
  - Tool-call success rate for agents.
  - Human/evaluator quality scores.
- **Cost efficiency**
  - Cost per request.
  - Cost per 1K/1M tokens.
  - Input vs. output token consumption.
  - GPU utilization and infrastructure cost.
- **Safety**
  - Policy-violation rate.
  - PII leakage incidents.
  - Prompt-injection detection rate.
  - Unsafe-output rate.

##### Example production SLOs

| Metric | Example SLO |
| --- | --- |
| Availability | ≥ 99.9% |
| P95 TTFT | \< 1.5 sec |
| P95 E2E latency | \< 8 sec |
| Error rate | \< 0.5% |
| Successful requests | ≥ 99.5% |
| GPU utilization | 60–85% |
| Hallucination rate | \< 2% |
| Task success rate | ≥ 95% |

The exact targets depend on the application. A real-time chatbot may prioritize **TTFT and streaming latency**, while an offline batch-processing system may prioritize **throughput and cost**.

### 26.28 Cloud vs. on-device model deployment: how do you choose for AI applications?

##### Cloud vs. On-Device Model Deployment

The main trade-off is between **scalability and model capability** in the cloud versus **privacy, latency, and offline operation** on the device.

| Cloud Deployment | On-Device Deployment |
| --- | --- |
| Model runs on cloud GPUs/TPUs | Model runs locally on phone, laptop, edge device, etc. |
| Can support very large models | Usually uses smaller/quantized models |
| Easy to scale horizontally | Limited by device CPU/GPU/NPU and memory |
| Requires network connectivity | Can work offline |
| Higher network latency | Very low local latency |
| Data leaves the device | Data can remain on-device |
| Easier centralized monitoring | Monitoring is more difficult |
| Infrastructure/API cost | Lower per-request cloud cost, but device hardware matters |
| Easier model updates | Model updates must be distributed to devices |
| Strong choice for large LLMs and complex workloads | Strong choice for privacy-sensitive and real-time workloads |

##### Cloud deployment

Best when:

- You need **large foundation models**.
- Workload is highly variable and requires autoscaling.
- Centralized monitoring and model management are important.
- You need powerful GPUs that devices cannot provide.
- You want frequent model updates without updating every device.

##### On-device deployment

Best when:

- **Privacy** is critical.
- The application must work offline.
- Very low latency is required.
- Network connectivity is unreliable.
- You want to reduce cloud inference costs.
- The model can be sufficiently compressed using **quantization, pruning, or distillation**.

##### Hybrid deployment

A common production architecture is:

`Device → Small On-Device Model → Cloud LLM when needed`

For example, a device could handle simple classification, wake-word detection, or basic summarization locally, while sending complex reasoning tasks to a cloud model.

### 26.29 How do you implement fallback strategies when the primary model is unavailable or rate-limited?

##### How do you implement fallback strategies when the primary model is unavailable or rate-limited?

- **Use multiple models/providers:** Define a primary model and one or more compatible fallback models.
- **Detect failures:** Trigger fallback for `429` rate limits, `5xx` errors, timeouts, connection failures, or provider outages.
- **Retry transient failures:** Use limited retries with **exponential backoff + jitter** before switching models.
- **Use a circuit breaker:** Temporarily stop sending traffic to an unhealthy primary model and periodically check whether it has recovered.
- **Use an LLM gateway:** Centralize model routing, retries, rate limits, health checks, and fallback logic.
- **Check model compatibility:** Ensure the fallback supports required context length, tool calling, structured output, and other capabilities.
- **Graceful degradation:** If the primary is unavailable, use a smaller/cheaper model when acceptable rather than failing the entire request.
- **Prevent retry storms:** Use bounded retries, concurrency limits, and separate quotas for fallback providers.
- **Monitor fallback rate:** A sudden increase in fallback traffic can indicate a provider or infrastructure problem.
- **Keep the previous model available:** For self-hosted models, maintain a healthy previous version so traffic can be switched back quickly.

##### Example

```txt
Request
   ↓
LLM Gateway
   ↓
Primary Model
   ↓
429 / Timeout / 5xx?
   ├── No → Response
   │
   └── Yes
        ↓
   Retry + Backoff
        ↓
   Still failing?
        ↓
   Fallback Model
        ↓
   Response
```

### 26.30 How do you implement structured output from LLMs reliably in production?

##### How do you implement structured output from LLMs reliably in production?

- **Define a strict schema:** Use JSON Schema, Pydantic models, or equivalent schemas to define the expected output.
- **Use native structured-output support:** Prefer the model/provider's **JSON Schema or constrained-generation** capabilities over asking the model to “return JSON.”
- **Validate every response:** Parse and validate the output against the schema before passing it downstream.
- **Use constrained decoding:** Where supported, restrict generation to valid JSON/schema-compatible outputs.
- **Handle validation failures:** Retry with the validation error or use a fallback model rather than accepting malformed data.
- **Keep schemas simple:** Avoid unnecessarily complex nested structures that increase generation errors.
- **Separate reasoning from output:** Don't depend on free-form reasoning being embedded in the structured response; request only the fields required by the application.
- **Use type and business validation:** Schema validation checks structure, while business rules check things like valid ranges, required relationships, and allowed values.
- **Version schemas:** Track schema versions alongside prompt and model versions so changes remain backward compatible.
- **Monitor failures:** Track parsing errors, schema-validation failures, retries, latency, and fallback rates.
- **Never blindly trust output:** Treat LLM-generated structured data as untrusted input and sanitize it before database writes, API calls, or tool execution.

##### Production flow

```txt
User Request
     ↓
Prompt + Schema
     ↓
LLM
     ↓
Structured Response
     ↓
JSON Parsing
     ↓
Schema Validation
     ↓
Business Validation
     ↓
Application / Tool
```

If validation fails:

```txt
Invalid Output
     ↓
Retry / Repair
     ↓
Validate Again
     ↓
Fallback Model
     ↓
Fail Safely
```

### 26.31 How do you handle long contexts efficiently in production (context compression, prefix caching)?

##### How do you handle long contexts efficiently in production?

The goal is to **reduce the amount of context processed while preserving the information needed for the task**.

- **Context compression:** Summarize older conversation history instead of repeatedly sending the entire history.
- **Semantic retrieval:** Store historical information in a vector database and retrieve only the most relevant chunks.
- **Hierarchical summarization:** Maintain summaries at different levels, such as conversation → session → key facts.
- **Token budgeting:** Reserve tokens for the model's output and dynamically limit how much context is included.
- **Chunking:** Split large documents into meaningful chunks rather than putting entire documents into the prompt.
- **Reranking:** Retrieve a larger candidate set and rerank it before sending only the best results to the LLM.
- **Deduplication:** Remove repeated instructions, documents, and conversation content.
- **Prefix caching:** Cache frequently repeated prefixes such as system prompts, instructions, few-shot examples, and shared conversation context.
- **Prompt caching:** Reuse provider-side cached prompt prefixes when supported, reducing latency and input-token costs.
- **KV-cache reuse:** In self-hosted inference engines such as vLLM or SGLang, reuse previously computed KV-cache states where possible.
- **Sliding-window context:** Keep recent messages while removing or summarizing older low-value messages.
- **Model selection:** Use a model with a larger context window only when the task actually requires it.

##### Context compression vs. Prefix caching

| Technique | Main purpose |
| --- | --- |
| **Context compression** | Reduce the amount of information sent to the model |
| **Retrieval** | Select only relevant information |
| **Prefix caching** | Avoid recomputing repeated context |
| **KV-cache reuse** | Reuse attention computation during inference |
| **Summarization** | Preserve important information in fewer tokens |

### 26.32 What is semantic routing, and how do you implement it in a multi-model system?

##### What is semantic routing?

**Semantic routing** is the process of analyzing the **meaning or intent of a request** and dynamically routing it to the most appropriate model, workflow, or service.

Instead of:

`All requests → Same LLM`

you use:

`User Request → Semantic Router → Best Model / Workflow`

##### How do you implement it?

- **Define routing categories:** For example:
  - Simple/general questions → small, cheap model
  - Complex reasoning → powerful model
  - Coding → coding-specialized model
  - Summarization → optimized model
  - Sensitive tasks → approved/private model
- **Create embeddings or classifiers:** Convert the incoming request into an embedding or use an intent classifier.
- **Compare against route descriptions:** Calculate semantic similarity between the request and predefined route examples.
- **Apply routing rules:** Select the model when the similarity score exceeds a threshold.
- **Use metadata:** Combine semantic similarity with user tier, latency requirements, cost limits, privacy requirements, and model availability.
- **Add a fallback:** If confidence is low, route to a general-purpose model or a stronger model.
- **Monitor routing quality:** Track route accuracy, model performance, latency, cost, and fallback frequency.

##### Example

```txt
User Request
     ↓
Semantic Router
     ↓
 ┌──────────────┬───────────────┬────────────────┐
 ↓              ↓               ↓
Simple        Coding          Complex Reasoning
 ↓              ↓               ↓
Small LLM    Code LLM        Large LLM
```

For example:

- `"What is HTTP?"` → small/cheap model
- `"Optimize this Python algorithm"` → coding model
- `"Prove this mathematical theorem"` → stronger reasoning model

### 26.33 How do you manage secrets and API keys securely in LLM applications?

- **Never hard-code secrets:** Don't put API keys, passwords, or tokens in source code or Git repositories.
- **Use a secrets manager:** Store credentials in services such as AWS Secrets Manager, HashiCorp Vault, or cloud-native secret managers.
- **Inject secrets at runtime:** Applications retrieve secrets through environment variables, workload identity, or a secrets-management API.
- **Use least privilege:** Give each service only the permissions it actually needs.
- **Rotate credentials regularly:** Automatically rotate API keys and immediately revoke compromised credentials.
- **Separate environments:** Use different credentials and configurations for development, staging, and production.
- **Restrict API keys:** Where supported, limit keys by API scope, service, environment, IP, or application.
- **Never expose secrets to clients:** API keys should remain on the backend; browsers and mobile clients should call your backend rather than directly exposing provider keys.
- **Prevent secret leakage in logs:** Redact API keys, authorization headers, tokens, and sensitive request data from logs and traces.
- **Scan for leaked secrets:** Use tools such as secret scanners in Git hooks and CI/CD pipelines.
- **Encrypt secrets:** Protect secrets both at rest and in transit.
- **Audit access:** Log who or which workload accessed a secret and alert on unusual access.
- **Use workload identity when possible:** Prefer short-lived credentials and cloud IAM roles over long-lived static keys.

### 26.34 Your LLM API has latency spikes during peak hours. How do you stabilize it?

##### How do you stabilize an LLM API with latency spikes during peak hours?

- **Measure the bottleneck first:** Monitor P50/P95/P99 latency, TTFT, tokens/sec, queue time, concurrency, provider `429`s, and GPU utilization.
- **Control concurrency:** Put a limit on simultaneous LLM requests so traffic spikes don't overwhelm the provider or your infrastructure.
- **Use rate limiting and backpressure:** Throttle excess traffic and queue requests instead of allowing unlimited concurrent calls.
- **Use continuous batching:** For self-hosted models, engines such as vLLM or SGLang can improve GPU utilization and throughput.
- **Reduce context size:** Compress conversation history, retrieve only relevant documents, and remove unnecessary prompt tokens.
- **Use prompt/prefix caching:** Reuse frequently repeated system prompts and prefixes to reduce computation and latency.
- **Stream responses:** Stream tokens to reduce **perceived latency**, even when total generation time remains similar.
- **Use model routing:** Send simple requests to smaller/faster models and reserve larger models for complex tasks.
- **Autoscale:** Scale inference workers based on queue depth, concurrency, GPU utilization, or request rate.
- **Use multiple providers/models:** Route traffic to another provider when the primary reaches capacity or becomes slow.
- **Cache safe/repeatable responses:** Cache deterministic or frequently requested results where appropriate.
- **Set timeouts and graceful fallbacks:** Don't let slow requests consume resources indefinitely.
- **Load test before peak traffic:** Simulate expected concurrency and identify the saturation point.

##### Production architecture

```txt
                 ┌─→ Fast/Small Model
                 │
Requests → Gateway → Primary LLM
                 │
                 ├─→ Queue → Autoscaled Workers
                 │
                 └─→ Fallback Provider

        Rate Limit + Concurrency Control
                    ↓
          Monitoring / Alerting
```

### 26.35 Your LLM costs are too high in production. How do you reduce costs without degrading quality?

##### How do you reduce LLM costs without degrading quality?

- **Measure cost first:** Track cost per request, input/output tokens, model usage, cache hit rate, and cost by user/feature.
- **Use model routing:** Send simple tasks to smaller, cheaper models and reserve expensive models for complex reasoning.
- **Reduce unnecessary context:** Use RAG, context compression, deduplication, and reranking instead of sending entire documents or conversation history.
- **Use prompt caching:** Cache repeated system prompts, instructions, and prefixes where supported.
- **Reduce output tokens:** Set appropriate `max_tokens` limits and design prompts to request concise structured outputs.
- **Cache responses:** Cache identical or highly reusable requests when correctness permits.
- **Use batching:** Batch suitable offline workloads to improve inference efficiency.
- **Optimize RAG:** Retrieve fewer, higher-quality chunks rather than increasing the context size.
- **Use quantization for self-hosted models:** Techniques such as INT8 or INT4 can reduce GPU memory and infrastructure costs with acceptable quality loss.
- **Use efficient inference engines:** vLLM or SGLang can improve GPU utilization and throughput.
- **Fine-tune smaller models:** If a task is repetitive and well-defined, a smaller specialized model may achieve similar quality at much lower inference cost.
- **Use asynchronous processing:** Move non-real-time workloads to queues/batch processing instead of expensive synchronous inference.
- **Set budgets and quotas:** Apply per-user, tenant, or feature spending limits.
- **A/B test optimizations:** Never assume a cheaper configuration is equivalent—compare it against the production baseline using automated evaluations and real business metrics.

##### Production strategy

```txt
Request
   ↓
Semantic Router
   ├── Simple → Small/Cheap Model
   ├── Moderate → Mid-size Model
   └── Complex → Large Model
                 ↓
        Prompt/Context Optimization
                 ↓
           Cache if possible
                 ↓
             LLM
                 ↓
       Quality + Cost Monitoring
```

### 26.36 Your application is hitting LLM provider rate limits during peak hours. How do you handle it?

##### How do you handle LLM provider rate limits during peak hours?

- **Implement rate limiting at your gateway:** Control requests per second/minute per user, tenant, and application before they reach the provider.
- **Track both RPM and TPM:** LLM providers often limit both **requests per minute** and **tokens per minute**.
- **Use a distributed limiter:** Redis or an API gateway ensures limits are enforced consistently across multiple application instances.
- **Queue requests:** Instead of sending every request immediately, place excess traffic into a queue and process it according to priority.
- **Use exponential backoff + jitter:** When receiving `429 Too Many Requests`, retry after an increasing delay with randomness to avoid synchronized retry storms.
- **Limit concurrency:** Control the number of simultaneous requests sent to the provider.
- **Use adaptive throttling:** Dynamically reduce request rates when provider capacity or remaining quota decreases.
- **Use model/provider fallback:** Route requests to another model or provider when the primary reaches its limit.
- **Prioritize requests:** Give critical/interactive requests higher priority than batch or background workloads.
- **Use caching:** Avoid making repeated LLM calls for identical or cacheable requests.
- **Reduce token consumption:** Compress context, retrieve fewer relevant documents, and limit unnecessary output tokens.
- **Monitor provider quotas:** Track RPM, TPM, 429 rates, queue depth, latency, and remaining quota.
- **Graceful degradation:** If all providers are saturated, queue the request or return a controlled response rather than allowing the entire application to fail.

##### Production flow

```txt
User Requests
      ↓
   LLM Gateway
      ↓
Rate Limit + Concurrency Control
      ↓
    Queue
      ↓
 ┌───────────────┐
 │ Primary Model │
 └───────┬───────┘
         │ 429 / unavailable
         ↓
 ┌───────────────┐
 │ Fallback Model│
 └───────────────┘
```

### 26.37 Your application depends on one LLM provider. How do you switch providers without downtime?

##### How do you switch LLM providers without downtime?

The key is to **decouple the application from the provider** by introducing an LLM abstraction or gateway.

- **Create a provider-agnostic interface:** The application calls your internal `LLMService` rather than directly calling OpenAI, Anthropic, Gemini, etc.
- **Normalize APIs:** Convert provider-specific request/response formats into a common internal schema.
- **Deploy the new provider alongside the existing one:** Don't remove the old provider immediately.
- **Test the new provider in staging and shadow traffic** using production-like requests.
- **Use feature flags or traffic routing:** Start with 1–5% traffic and gradually increase it.
- **Use canary or blue-green deployment:** Keep the old provider available while validating the new one.
- **Maintain prompt compatibility:** Test system prompts, tool calling, structured outputs, context limits, and tokenization with the new provider.
- **Monitor quality and performance:** Compare latency, error rate, output quality, token usage, and cost.
- **Keep automatic fallback:** If the new provider fails, immediately route traffic back to the old provider.
- **Migrate gradually:** `0% → 5% → 25% → 50% → 100%`.
- **Keep provider-specific configuration isolated:** API keys, model names, endpoints, and parameters should live in configuration rather than application code.

##### Architecture

```txt
                    ┌──→ Provider A
Application → LLM Gateway ──→ Provider B
                    └──→ Provider C
                         ↓
                  Routing / Fallback
```

### 26.38 Your AI system handles 100 requests/sec but crashes at 5000. How do you scale for concurrent requests?

##### How do you scale an AI system for high concurrent requests?

If the system handles **100 RPS but crashes at 5,000 RPS**, I would first identify the bottleneck, then scale horizontally and control concurrency rather than simply adding more servers.

- **Load test first:** Measure CPU/GPU utilization, memory, queue depth, connection limits, latency, throughput, and error rates at increasing concurrency.
- **Horizontal scaling:** Run multiple stateless API/inference workers behind a load balancer.
- **Autoscaling:** Scale workers based on RPS, queue depth, concurrency, GPU utilization, or latency.
- **Use a request queue:** Absorb traffic spikes with Kafka, SQS, Redis Streams, or similar infrastructure instead of processing every request immediately.
- **Control concurrency:** Set maximum in-flight LLM requests per worker and prevent resource exhaustion.
- **Continuous batching:** For self-hosted LLMs, use engines such as **vLLM or SGLang** to efficiently batch concurrent generation requests.
- **Use async I/O:** Avoid blocking application threads while waiting for LLM APIs, databases, vector stores, or other services.
- **Connection pooling:** Reuse HTTP/database connections instead of creating new connections for every request.
- **Caching:** Cache embeddings, retrieval results, or safe/repeatable LLM responses where appropriate.
- **Rate limiting and backpressure:** Protect downstream LLM providers and infrastructure from overload.
- **Load balancing:** Distribute traffic across model replicas, GPUs, or providers.
- **Optimize the workload:** Reduce unnecessary context, tokens, expensive tool calls, and synchronous processing.
- **Graceful degradation:** Under extreme load, prioritize critical requests, use smaller models, or queue non-critical work.
- **Monitor continuously:** Track P50/P95/P99 latency, throughput, concurrency, queue depth, GPU utilization, 429s, errors, and cost.

##### Architecture

```txt
                ┌── API Worker 1 ──┐
                ├── API Worker 2 ──┤
Clients → Load Balancer ────────────┤
                ├── API Worker N ──┘
                ↓
           Request Queue
                ↓
       Inference / LLM Workers
          ↓       ↓       ↓
        GPU 1   GPU 2   GPU N
```

### 26.39 A traffic spike brings down your AI system. How do you handle peak traffic?

##### How do you handle peak traffic in an AI system?

The goal is to **absorb traffic spikes, protect downstream services, and degrade gracefully instead of allowing the entire system to crash**.

- **Autoscaling:** Scale API and inference workers horizontally based on RPS, concurrency, queue depth, latency, or GPU utilization.
- **Load balancing:** Distribute traffic across healthy application and model instances.
- **Rate limiting:** Limit requests per user, tenant, and API key to prevent a traffic surge from exhausting resources.
- **Backpressure:** Reject, delay, or queue excess requests instead of allowing unlimited requests into the system.
- **Request queues:** Use Kafka, SQS, Redis Streams, or similar queues for workloads that don't require immediate responses.
- **Concurrency limits:** Restrict the number of simultaneous LLM requests each worker can process.
- **Continuous batching:** For self-hosted models, use vLLM or SGLang to efficiently serve many concurrent requests.
- **Caching:** Cache embeddings, retrieval results, and safe/repeatable responses to reduce unnecessary model calls.
- **Semantic/model routing:** Send simple requests to smaller, faster models and reserve expensive models for complex requests.
- **Provider rate-limit protection:** Track RPM/TPM and use adaptive throttling, retries with exponential backoff, and provider fallbacks.
- **Prioritize traffic:** Give interactive or business-critical requests higher priority than background jobs.
- **Graceful degradation:** Temporarily disable non-critical AI features, reduce model size, or queue low-priority requests during extreme load.
- **Circuit breakers:** Stop sending traffic to unhealthy providers or services until they recover.
- **Observability:** Monitor P95/P99 latency, RPS, concurrency, queue depth, GPU utilization, error rates, 429s, and cost.

##### Peak-traffic architecture

```txt
                    ┌── API Worker ──┐
                    ├── API Worker ──┤
Users → CDN/LB → Gateway ────────────┤
                    └── API Worker ──┘
                           ↓
                Rate Limit + Backpressure
                           ↓
                     Request Queue
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
           Model 1      Model 2      Model 3
              ↓            ↓            ↓
          Cache / Provider Fallback
```

### 26.40 One LLM provider outage took down your entire system. How do you eliminate single points of failure?

##### Eliminating Single Points of Failure in an LLM System

- **Use multiple LLM providers:** Support providers such as OpenAI, Anthropic, Google, or self-hosted models.
- **Use an LLM gateway:** Put a provider-agnostic gateway between the application and providers.
- **Implement automatic failover:** If the primary provider returns `5xx`, `429`, timeout, or health-check failure, route traffic to a healthy fallback.
- **Use health checks:** Continuously monitor provider availability, latency, error rate, and rate limits.
- **Circuit breaker:** Stop sending requests to an unhealthy provider temporarily instead of repeatedly hitting it.
- **Keep model compatibility:** Standardize prompts, tool schemas, structured outputs, and response formats across providers.
- **Avoid shared dependencies:** Ensure the gateway, queues, caches, databases, and authentication systems also have redundancy.
- **Deploy across multiple availability zones/regions:** Prevent a single infrastructure failure from taking down the service.
- **Use queues and backpressure:** Absorb temporary traffic spikes and prevent cascading failures.
- **Graceful degradation:** If all LLM providers fail, return cached results, use a smaller local model, or provide a controlled fallback response.
- **Test failure scenarios:** Regularly perform provider-failure and disaster-recovery tests.

##### Architecture

```txt
                    ┌── Provider A
                    │
Users → API Gateway → LLM Router ── Provider B
                    │
                    └── Self-hosted Model
                           │
                    Health Checks
                    Circuit Breaker
                    Failover
```

### 26.41 Your multi-LLM pipeline fails when one model in the chain breaks. How do you handle orchestration failure?

##### Handling Orchestration Failure in a Multi-LLM Pipeline

- **Isolate each step:** Treat every model call as an independent task with clear inputs, outputs, timeouts, and error states.
- **Retry transient failures:** Use bounded retries with exponential backoff and jitter for `429`, `5xx`, and temporary network failures.
- **Use model-level fallback:** If Model A fails, route the same task to a compatible backup model.
- **Checkpoint intermediate results:** Persist successful outputs so the pipeline doesn't restart from the beginning.
- **Use circuit breakers:** Temporarily stop calling a consistently failing model.
- **Define failure policies per step:** Decide whether a failed step should **retry, fallback, skip, compensate, or terminate** the workflow.
- **Make steps idempotent:** Retrying a step should not create duplicate side effects.
- **Use timeouts and cancellation:** Prevent one slow model from blocking the entire pipeline.
- **Support partial completion:** Return completed results when non-critical steps fail rather than failing the whole workflow.
- **Validate outputs between steps:** Don't pass malformed or hallucinated output to the next model.
- **Trace the entire workflow:** Track workflow ID, step ID, model, latency, retries, errors, and fallback usage.
- **Use durable orchestration:** For complex workflows, use systems such as LangGraph or Temporal to manage state, retries, checkpoints, and recovery.

##### Example

```txt
Request
   ↓
Planner LLM
   ↓
Research LLM ──X──→ Retry → Backup LLM
   ↓
Synthesis LLM
   ↓
Final Response
```

If the Research LLM fails:

```txt
Failure
   ↓
Retry with backoff
   ↓
Still failing?
   ↓
Circuit breaker + Backup model
   ↓
Continue from checkpoint
```

### 26.42 Your AI pipeline has zero visibility into which step is failing. How do you add observability?

- **Add end-to-end tracing** across the entire AI pipeline so every request gets a unique `trace_id`.
- **Create a span for each pipeline step**, such as:
  - API request
  - Prompt construction
  - Retrieval
  - Reranking
  - Tool/API calls
  - LLM inference
  - Post-processing
  - Final response
- **Capture structured logs** for each step:
  - `trace_id`
  - `span_id`
  - step name
  - status
  - latency
  - error type/message
  - model/provider
  - token usage
  - retrieved document count
- **Track key metrics** such as:
  - p50/p95/p99 latency
  - error rate
  - timeout rate
  - token usage
  - LLM cost
  - retrieval quality
  - throughput
- **Propagate correlation IDs** between microservices so a single request can be followed across the entire system.
- **Use distributed tracing** with tools such as OpenTelemetry and visualize traces in platforms such as Jaeger, Grafana, or Datadog.
- **Add alerts** for abnormal error rates, latency, token consumption, or specific pipeline-step failures.
- **Track AI-specific signals**, including hallucination/evaluation scores, retrieval failures, tool-call failures, model fallbacks, and guardrail violations.

##### Example

```txt
Request
  │
  ├── API ───────────── 20 ms
  ├── Retrieval ─────── 80 ms
  ├── Reranking ─────── 40 ms
  ├── LLM ───────────── 1,200 ms  ← timeout
  ├── Post-processing ── skipped
  └── Response ───────── failed
```

Now instead of seeing only **"AI request failed"**, you can immediately identify that the **LLM inference step timed out**.

### 26.43 You quantized your LLM, but accuracy dropped significantly. How do you minimize quantization loss?

- **Use a less aggressive quantization level:** If INT4 causes a large accuracy drop, try **INT8**, **W8A8**, or mixed precision instead of forcing everything to INT4.
- **Use calibration data:** Calibrate activation ranges using a representative dataset that matches real production inputs.
- **Apply per-channel or per-group quantization:** These usually preserve accuracy better than a single scale for the entire tensor.
- **Keep sensitive layers in higher precision:** Leave embeddings, attention projections, normalization layers, or output heads in FP16/BF16 when they are particularly sensitive.
- **Use GPTQ/AWQ or similar weight-only methods:** These can provide better accuracy than naive post-training quantization.
- **Use Quantization-Aware Training (QAT):** Simulate quantization during training so the model learns to compensate for quantization errors.
- **Use mixed precision:** Quantize less-sensitive layers more aggressively while keeping sensitive layers at FP16/BF16.
- **Evaluate layer-by-layer:** Identify which layers contribute most to the accuracy degradation and selectively increase their precision.
- **Validate on representative benchmarks:** Compare the quantized model against the original model on task accuracy, perplexity, reasoning, and production-specific evaluation sets.

### 26.44 One failing AI component can take down your entire platform. How do you design graceful degradation?

- **Isolate AI components:** Use separate services for LLM inference, RAG, embeddings, reranking, and tools so one failure does not crash the entire application.
- **Use timeouts and circuit breakers:** Stop calling an unhealthy component after repeated failures and fail fast instead of allowing requests to pile up.
- **Provide fallbacks:**
  - Primary LLM → secondary LLM/provider
  - RAG failure → direct LLM response with a warning/limited capability
  - Reranker failure → return initial retrieval results
  - Advanced model → smaller/cheaper model
- **Make non-critical features optional:** If memory, analytics, or personalization fails, the core AI request should still work.
- **Use queues and backpressure:** Prevent downstream failures from propagating upstream during traffic spikes.
- **Return partial results:** If only one stage fails, return whatever reliable result is available rather than failing the entire request.
- **Use health checks and monitoring:** Detect unhealthy services quickly and automatically route traffic away from them.
- **Design idempotent retries:** Retry transient failures with exponential backoff and jitter, while avoiding retry storms.
- **Use feature flags:** Disable problematic AI features without deploying the entire application.
- **Define degraded modes:** Explicitly decide what the system should do when each dependency fails.

##### Example

```txt
User Request
     │
     ▼
 API Gateway
     │
     ├── RAG ──────── X
     │                 ↓
     │            Skip RAG
     │                 ↓
     └──────────► LLM
                    │
                    ├── Primary LLM ── X
                    │
                    └── Fallback LLM ──► Response
```

### 26.45 How do you choose the right machine learning algorithm?

Start from the problem and constraints, not the algorithm.

1. **Problem type:** classification, regression, ranking, clustering, forecasting, generation. Is ML needed at all, or will a rule or heuristic do?
2. **Data shape and size:**
   - *Tabular, small to medium:* gradient-boosted trees (XGBoost/LightGBM/CatBoost) are the default winner. Logistic or linear regression is the baseline.
   - *Images, audio, text:* deep learning, almost always **transfer learning** from pretrained models or foundation models.
   - *Very little labelled data:* pretrained embeddings plus a simple classifier, few-shot LLMs, or semi-supervised methods.
   - *Sequences and time series:* seasonal baselines, GBDT on lags, or sequence models.
3. **Constraints:** latency and memory (on-device means small models), interpretability and regulation (credit scoring favours monotonic GBDT or scorecards), training budget, and team skills.
4. **Empirical process:** build a trivial baseline, then a simple model, then iterate toward complexity only when validation metrics justify it. Use proper cross-validation (time-split when time matters) and compare on the *business metric*.
5. **Operational fit:** retraining cost, monitoring, explainability for stakeholders, and the cost of serving.

**Rule of thumb:** "No free lunch". Pick 2–3 candidates suited to the data type, benchmark them fairly, and prefer the simplest model within noise of the best. For a Head of Data and AI, the deciding factor is often maintainability and time-to-value, not +0.5% AUC.

### 26.46 What is model drift, and how do you handle it?

**Model drift** means a model's performance degrades in production because the world has moved away from its training data.

| Type | What changes | Example |
| --- | --- | --- |
| **Data / covariate drift** | $P(X)$: input distribution | New user demographics, a new app version changes feature values |
| **Concept drift** | $P(Y \mid X)$: the relationship itself | Fraudsters change tactics; "normal" spending changes after inflation |
| **Label / prior drift** | $P(Y)$: base rates | Churn rate jumps during a recession |
| **Upstream / data-quality issues** | Pipelines | A feature becomes null, units change from cents to dollars |

**Detection:**
- Monitor feature distributions against a training reference: PSI (PSI above 0.2 means significant), KS test, Jensen-Shannon/Wasserstein distance, and null rates.
- Monitor prediction distribution and confidence (a proxy when labels are delayed).
- Once labels arrive, track the true performance metrics by segment.
- Tools: Evidently, WhyLabs, Arize, or custom dashboards.

**Handling:**
- **Scheduled retraining** (daily or weekly) on a rolling window, or **triggered retraining** when drift or performance alerts fire.
- Online or incremental learning for fast-moving domains (ads, fraud).
- Weight recent data more heavily. Add robust features that are less sensitive to drift.
- Shadow-test or canary the retrained models before promotion. Keep the ability to roll back.
- Investigate the root cause first: many "drifts" are pipeline bugs, not changes in the world.

```python
import numpy as np
def psi(expected, actual, bins=10):
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    e, a = np.histogram(expected, q)[0] / len(expected), np.histogram(actual, q)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))
```

### 26.47 How would you handle large-scale data for training?

**First ask whether all of it is needed.** Learning curves often flatten. Smart sampling (stratified, hard-example mining, negative downsampling with reweighting) can cut cost 10× with no loss.

**Storage and formats:** columnar formats (Parquet) on object storage with a table format (Delta/Iceberg) for versioning and time travel. Partition by date. For deep learning, stream sharded files (WebDataset, TFRecord, MosaicML StreamingDataset) instead of millions of small files.

**Processing:** distributed feature engineering with Spark, Dask, Ray Data or BigQuery. Precompute features once into a feature store or offline tables with point-in-time-correct joins.

**Data loading:** keep GPUs fed with parallel workers, prefetching, pinned memory, and caching preprocessed tensors. Profile: a GPU idle waiting on I/O is the most common waste.

**Distributed training:**
- *Data parallelism* (PyTorch DDP, Horovod): replicate the model, shard the batches.
- *Model and sharded parallelism* for big models: FSDP/ZeRO, tensor and pipeline parallelism (DeepSpeed, Megatron).
- For tabular data: distributed XGBoost/LightGBM on Spark or Dask, or out-of-core training.

**Incremental and online learning:** for streaming data, use `partial_fit` models, warm-starting from previous checkpoints, or continuous training pipelines.

**Operational concerns:** checkpoint often (spot instances), keep runs reproducible (data versioning, seeds), track experiments (MLflow/W&B), and control cost (spot or preemptible compute, autoscaling clusters).

### 26.48 How do you deploy an ML model in production?

**1. Choose the serving pattern from the requirements:**
- *Batch scoring* (nightly Spark or Airflow job writing predictions to a table): churn scores, recommendations. Cheapest and simplest.
- *Online, synchronous* (REST/gRPC microservice): fraud scoring, search ranking.
- *Streaming* (score events from Kafka/Flink): real-time alerting.
- *Edge or on-device:* privacy, offline use, latency.

**2. Package:** serialise the model plus the *exact* preprocessing (a sklearn Pipeline, a TorchScript/ONNX graph) and pin dependencies in a Docker image. Register it in a **model registry** (MLflow, SageMaker, Vertex) with lineage: data version, code commit, metrics.

**3. Serve:** FastAPI/BentoML for simple models; Triton, TorchServe, KServe or vLLM for GPU and LLM serving with dynamic batching. Run on Kubernetes or managed endpoints with autoscaling.

**4. Release safely:** CI/CD runs tests (unit, data validation, model quality gates against the current champion). Then **shadow deployment** (score live traffic with no impact), **canary** (1–5% of traffic), an A/B test on business metrics, and full rollout, with one-click rollback.

**5. Operate:** monitor latency, errors, throughput, input drift, prediction distribution and delayed-label performance. Log features and predictions for retraining and debugging. Retrain on a schedule or trigger.

```text
train ─► evaluate ─► registry (staging) ─► CI tests ─► shadow ─► canary ─► prod ─► monitor ─► retrain
```

**Key point:** the model is a small part of the system. Feature parity between training and serving (a feature store), versioning and monitoring are what make deployment reliable.

### 26.49 How do you monitor a model's performance in production?

Monitor in layers, from cheapest and fastest to most meaningful:

1. **Service health:** latency (p50/p95/p99), throughput, error rate, timeouts, CPU/GPU/memory. Standard SRE dashboards (Prometheus/Grafana) and alerts.
2. **Data quality:** schema violations, null and default-value rates, out-of-range values, missing upstream features, and freshness of streaming features. These catch most incidents, which are pipeline bugs.
3. **Data drift:** distribution of each important feature against the training baseline (PSI, KS, JS divergence). Weight alerts by feature importance.
4. **Prediction drift:** distribution of scores, predicted-class mix, and confidence. Shifts here are an early proxy when labels are delayed.
5. **Model quality (ground truth):** once labels arrive (minutes for clicks, weeks for chargebacks), compute AUC, precision/recall, calibration and RMSE, **sliced by segment** (region, device, new versus existing users) to catch hidden regressions and fairness issues.
6. **Business KPIs:** conversion, revenue, fraud losses, tickets. This is the outcome that matters, tracked against a holdout or control group where possible.

**Practices:**
- Log every request: features, model version, prediction, and later the label (an "inference log" joined with outcomes).
- Use human review samples for tasks without natural labels (moderation, LLM outputs).
- Alert with sensible thresholds and owners. Avoid alert fatigue.
- Tools: Evidently, Arize, WhyLabs, Fiddler, or SageMaker/Vertex Model Monitor.
- Connect monitoring to action: runbooks, automatic rollback, and retraining triggers.

### 26.50 How would you deploy a model with low-latency requirements?

Set the budget first (for example p99 under 20 ms end to end) and break it down: network, feature fetch, inference and post-processing. Then attack each part.

**Model side:**
- Pick or design a smaller model: GBDT or shallow networks, distillation from a big teacher.
- **Quantisation** (INT8/FP8), pruning, and operator fusion. Compile with ONNX Runtime, TensorRT or OpenVINO.
- Cascades: a cheap model handles most traffic, and only uncertain cases go to the heavy model.

**Serving side:**
- Keep the model loaded in memory and warm (no cold starts). Pre-allocate.
- Use an efficient runtime and protocol: gRPC instead of REST/JSON, Triton with **dynamic batching** tuned for latency (small max queue delay).
- Co-locate the model with the caller (sidecar or in-process) or deploy at the edge or in-region to cut network hops.
- Choose hardware to match: CPU with AVX-512 for small models, GPU or Inferentia for large models with batching.

**Features:**
- Precompute heavy features in batch or streaming jobs into a low-latency online store (Redis, DynamoDB, Feast online). Avoid live joins and SQL queries.
- Precompute embeddings and use ANN indexes.

**Caching and precomputation:** cache predictions for repeated inputs; fully precompute when the input space is enumerable (per-user recommendations refreshed hourly).

**Reliability:** timeouts with a fallback (last known good score, rule-based default), autoscaling on latency metrics, and load testing at peak QPS against p99, not the average.

### 26.51 What are the common challenges when deploying ML models?

- **Training/serving skew:** features computed differently offline (Spark SQL) and online (Java service), or preprocessing that doesn't match. *Fix:* shared feature definitions or a feature store, logging served features for training, and parity tests.
- **Data leakage discovered late:** a great offline AUC collapses online because a feature isn't available at prediction time. *Fix:* point-in-time-correct joins and time-based validation.
- **Data quality and pipeline fragility:** upstream schema changes, nulls, late data. *Fix:* data contracts, validation (Great Expectations), and alerting.
- **Drift and decay:** the world changes. *Fix:* monitoring and retraining pipelines.
- **Delayed or missing ground truth:** you can't measure accuracy quickly. *Fix:* proxy metrics, human labelling samples.
- **Latency, scalability and cost:** a big model is too slow or expensive. *Fix:* compression, batching, caching, right-sized hardware.
- **Reproducibility and versioning:** unknown data, code or hyperparameters behind the production model. *Fix:* model registry, data versioning, experiment tracking.
- **Dependency and environment issues:** "works on my notebook". *Fix:* containers, pinned environments, CI.
- **Offline metrics not matching business impact.** *Fix:* A/B tests and guardrail metrics.
- **Feedback loops:** the model's own outputs shape future training data. *Fix:* exploration and holdouts.
- **Governance:** explainability, fairness, privacy (GDPR), and audit trails, especially in regulated domains.
- **Organisational:** unclear ownership between data science and engineering, no on-call for models. *Fix:* MLOps platform and clear SLAs. This one is often the biggest blocker.

### 26.52 Discuss scalability and latency requirements for ML systems.

**Define the requirements up front:** peak QPS (not the average), p95/p99 latency SLOs, availability (99.9%?), data volume growth, freshness (how stale may features or models be), and cost per prediction. These choose the architecture.

**Latency tiers drive design:**
| Requirement | Pattern |
| --- | --- |
| Hours/days | Batch scoring, precomputed results in a table or KV store |
| Seconds | Async queue + workers, streaming |
| < 100 ms | Online service, precomputed features, optimized model |
| < 10 ms | In-process or edge inference, tiny model / lookup, heavy caching |

**Scalability levers:**
- *Horizontal scaling:* stateless inference services behind a load balancer, with Kubernetes HPA scaling on QPS, GPU utilisation or queue depth.
- *Batching:* dynamic batching raises GPU throughput at a small latency cost. This is the core throughput/latency trade-off.
- *Multi-stage funnels:* cheap retrieval over millions of items, expensive ranking over hundreds. Cost scales with the funnel, not the catalogue.
- *Caching:* results, embeddings and features. Use semantic caching for LLMs.
- *Sharding:* embedding tables, ANN indexes, feature stores.
- *Model efficiency:* quantisation, distillation, efficient serving runtimes (TensorRT, vLLM with continuous batching).

**Training scalability:** distributed data processing, distributed training, incremental retraining.

**Trade-offs to discuss:** accuracy against latency (bigger models are slower), freshness against cost (streaming against batch features), consistency against availability (serve a stale model or features instead of failing), and graceful degradation (fall back to a simpler model or popular items under load).

**Validate:** load tests at 2–3× expected peak, and capacity planning with cost per 1k predictions.

### 26.53 How do you ensure your model is scalable and performs well with large datasets?

**Training on large data:**
- Choose algorithms that scale: mini-batch SGD-trained models, GBDT with histogram binning (LightGBM), distributed implementations (XGBoost on Spark/Dask, PyTorch DDP/FSDP). Avoid $O(n^2)$ methods (kernel SVMs, exact kNN) at scale. Use approximate versions (ANN, random features).
- Efficient data handling: columnar formats, partitioning, streaming data loaders, out-of-core and incremental learning (`partial_fit`), sampling where learning curves show diminishing returns.
- Cache and reuse feature computation; vectorise; push heavy joins to Spark or the warehouse.

**Generalisation as the data grows:**
- Larger data usually supports higher-capacity models; re-tune regularisation and check learning curves.
- Validate on large, representative, time-split holdouts and check performance per segment, not just overall.
- Watch for label noise and class imbalance at scale.

**Inference at scale:**
- Make the model cheap: compression, efficient runtime, batching.
- Choose batch versus online serving deliberately. Precompute when possible.
- Stateless, horizontally scalable services with autoscaling; sharded embedding and ANN indexes.

**Engineering discipline:**
- Benchmark training time and inference latency/throughput as data grows (scaling tests on 10× synthetic data).
- Profile bottlenecks: I/O, CPU and GPU utilisation, memory.
- Monitor cost per training run and per prediction.
- Automate retraining pipelines so the scale stays manageable operationally.

**Example:** moving a churn model from pandas plus sklearn on a single node to Spark feature pipelines with distributed LightGBM cut training from 6 h to 40 min and allowed daily retraining on the full 2B-row history.

### 26.54 Explain MLOps and its key components.

**MLOps** applies DevOps and data-engineering practice to machine learning, so that models are built, deployed, monitored and improved **reliably, reproducibly and quickly**. The difference from DevOps is that an ML system depends on *code + data + model*, and all three change.

**Key components:**
| Component | Purpose | Example tools |
| --- | --- | --- |
| Data management & versioning | Reproducible datasets, lineage, quality checks | Delta/Iceberg, DVC, lakeFS, Great Expectations |
| Feature store | Consistent offline/online features, reuse | Feast, Tecton, Databricks/Vertex/SageMaker FS |
| Experiment tracking | Log params, metrics, artifacts; compare runs | MLflow, Weights & Biases |
| Training pipelines / orchestration | Automated, repeatable training workflows | Airflow, Kubeflow Pipelines, Vertex/SageMaker Pipelines, Dagster |
| Model registry | Versioned models, stage transitions, lineage, approvals | MLflow Registry, SageMaker/Vertex registry |
| CI/CD/CT | Test code, data and models; automated deploy; continuous training | GitHub Actions, Argo CD |
| Model serving | Batch, online, streaming, edge inference | KServe, Triton, BentoML, Seldon, vLLM |
| Monitoring & observability | Service health, drift, performance, business KPIs | Prometheus/Grafana, Evidently, Arize |
| Governance | Access control, audit, model cards, fairness, compliance | Unity Catalog, model cards |

**Maturity levels (Google):** level 0, manual notebooks and hand-offs. Level 1, automated training pipelines with continuous training. Level 2, full CI/CD of the pipelines themselves.

**Head-of perspective:** MLOps is a platform and team investment. Standardise paved roads (templates, a shared feature store and registry) so data scientists ship in days, not months. Measure time-to-production, the number of models in production, and incident rates.

### 26.55 What is a feature store, and why is it important?

A **feature store** is a central system for defining, computing, storing and serving ML features consistently for **training (offline)** and **inference (online)**.

**Core parts:**
- **Feature definitions / registry:** features defined once as code (entity, transformation, owner, freshness), discoverable and versioned.
- **Offline store** (warehouse or lake: BigQuery, Snowflake, Delta): full history for training, with **point-in-time-correct joins**, meaning each training row gets feature values as they were at that event's timestamp. This prevents leakage.
- **Online store** (Redis, DynamoDB, Cassandra): the latest feature values per entity with millisecond lookups for real-time inference.
- **Materialisation and ingestion:** batch jobs and streaming pipelines (Kafka/Flink) that compute features and keep both stores in sync.
- **Monitoring:** freshness, null rates, drift.

**Why it matters:**
1. **Eliminates training/serving skew:** the same logic and values serve both paths. This is the top cause of silent production failures.
2. **Prevents leakage** with time-travel joins.
3. **Reuse:** a "customer 30-day spend" feature built once is used by fraud, churn and marketing models, reducing duplicated pipelines.
4. **Low-latency serving** of precomputed and streaming features.
5. **Governance and discovery:** lineage, ownership, access control.

**Trade-offs:** operational complexity and cost. A small team with only batch models may not need one: well-modelled warehouse tables plus dbt may be enough. It pays off with many models, real-time features and several teams.

```python
from feast import FeatureStore
store = FeatureStore(repo_path=".")
training_df = store.get_historical_features(entity_df=events, features=["user_stats:spend_30d"]).to_df()
online = store.get_online_features(features=["user_stats:spend_30d"], entity_rows=[{"user_id": 42}]).to_dict()
```

### 26.56 Cloud vs. on-device model deployment: what are the trade-offs?

| Aspect | Cloud deployment | On-device (edge) deployment |
| --- | --- | --- |
| Compute | Effectively unlimited; large models, GPUs | Constrained CPU/NPU, memory, battery, thermals |
| Model size | Any (incl. large LLMs) | Small: compressed, quantized (e.g. ≤ few hundred MB) |
| Latency | Network round-trip (50–300 ms+) | Very low, no network |
| Offline use | Requires connectivity | Works offline |
| Privacy | Data leaves device; needs strong governance | Data stays local, which helps privacy and regulation |
| Cost | Pay per inference/server; scales with users | Inference cost borne by user hardware; near-zero marginal server cost |
| Updates | Instant, centralized, easy A/B tests and rollback | Requires app/model rollout; version fragmentation across devices |
| Monitoring | Full logging and observability | Limited telemetry; harder to debug and measure drift |
| Security | Model weights protected on server | Model can be extracted / reverse-engineered |
| Tooling | Triton, KServe, SageMaker, vLLM | TFLite/LiteRT, Core ML, ONNX Runtime Mobile, ExecuTorch, MediaPipe |

**When to choose:**
- *On-device:* real-time camera or AR, keyboard prediction, wake-word detection, privacy-sensitive health data, offline use, or a very high request volume where server costs would be large.
- *Cloud:* large or frequently updated models, heavy compute (big LLMs), need for central data and fast iteration.

**Hybrid is common:** a small on-device model handles common or simple requests (and privacy filtering), and hard requests escalate to a cloud model (Apple Intelligence's on-device plus Private Cloud Compute is one example). **Federated learning** can improve on-device models without centralising raw data.

### 26.57 Why use tracing tools like Langfuse, LangSmith, or OpenTelemetry?

They give full visibility into prompts, tool calls, and retrievals, which is necessary for debugging complex multi-step LLM systems.

### 26.58 What does AWS Bedrock provide?

Unified API access to multiple foundation models through a single platform.

### 26.59 Why do teams aim for a multi-provider experience?

To avoid vendor lock-in by keeping the system able to work across providers like OpenAI, Anthropic, and Bedrock.

### 26.60 How do you containerize and deploy an AI/agentic service, and what does your Docker setup typically look like?

Multi-stage Dockerfile: a build stage installing dependencies (pinned versions, using a lockfile) and a slim runtime stage copying only what's needed — keeps images small and reduces attack surface. Environment variables/secrets injected at runtime (never baked into the image). Health-check endpoints for orchestration. For GPU workloads, use a CUDA-compatible base image matching the framework version. Orchestration via Docker Compose for simpler multi-service setups (agent service + vector DB + API gateway) or Kubernetes/Azure Container Apps for production-scale, auto-scaling deployments.

### 26.61 What does "observability" mean for an LLM/agent system in production, and what would you actually instrument?

Three pillars, adapted to LLM systems:

- **Tracing**: capture every step of an agent's execution — prompts, tool calls and their arguments/results, intermediate reasoning, latency per step (tools like LangSmith, Arize Phoenix, or OpenTelemetry-based custom tracing). Critical for debugging *why* an agent made a decision.
- **Metrics**: latency (p50/p95), token usage/cost per request, tool-call success/failure rates, retrieval hit rate, error rates.
- **Evaluation pipelines**: continuous/offline eval against labeled test sets (accuracy, groundedness, task success rate) plus online monitoring for drift — e.g., tracking answer quality or user feedback (thumbs up/down) over time to catch degradation from data drift, prompt regressions, or upstream model changes.

### 26.62 How do you monitor for and respond to model/data drift in a deployed system?

Track input distribution shifts (e.g., new types of queries/documents appearing that weren't in the original eval set), monitor output quality metrics over time (not just at launch), and set up alerting on metric degradation (e.g., retrieval recall dropping, hallucination-rate estimates rising). Periodically re-run the eval suite against production samples. For RAG specifically, monitor whether the knowledge base is stale relative to the domain it serves and set up a refresh/re-indexing cadence.

### 26.63 Walk through how you'd take a PyTorch/TensorFlow model from training to production.

Train and validate with proper train/val/test splits and cross-validation where appropriate; track experiments (e.g., MLflow/W&B) for reproducibility. Export to an optimized inference format (ONNX, TorchScript, or TensorRT for GPU speedups) to decouple serving from the training framework and reduce latency. Containerize with a serving layer (FastAPI, TorchServe, Triton Inference Server). Deploy behind a versioned API with canary/blue-green rollout to catch regressions before full traffic cutover. Set up monitoring (above) and a rollback plan.

### 26.64 How would you detect distribution shift between training and production data?

#### Detecting Distribution Shift

**Distribution shift** occurs when production data differs from training data.

I would check:

- **Feature distributions:** Mean, variance, quantiles, missing values, category frequencies.
- **Statistical tests:** KS test, PSI, Wasserstein distance, Jensen-Shannon divergence.
- **Prediction distribution:** Compare model outputs between training and production.
- **Performance:** Monitor precision, recall, F1, PR-AUC when production labels become available.
- **Train-vs-production classifier:** If a classifier can easily distinguish them, there is likely a shift.

**Key types:**

$$
\text{Covariate shift: } P(X) \text{ changes}
$$

$$
\text{Label shift: } P(Y) \text{ changes}
$$

$$
\text{Concept drift: } P(Y \mid X) \text{ changes}
$$

### 26.65 What is the difference between covariate shift, label shift, and concept drift?

#### Covariate Shift vs Label Shift vs Concept Drift

| Type | What changes? | Example |
| --- | --- | --- |
| **Covariate shift** | $P(X)$ changes | Customer demographics change, but the relationship to fraud stays the same |
| **Label shift** | $P(Y)$ changes | Fraud rate increases from 1% → 5%, but $P(X \mid Y)$ stays similar |
| **Concept drift** | $P(Y \mid X)$ changes | The same customer behavior that previously indicated fraud no longer does |

### 26.66 How would you handle training-serving skew?

**Training-serving skew** is any difference between how data/features look at training time and at inference time, causing a model that looked good offline to underperform in production.

**Common causes:**
- **Different code paths:** features computed in Spark/SQL for training and re-implemented in Java/Python for serving.
- **Time leakage:** training features computed with information not available at prediction time (e.g. daily aggregates including later events).
- **Data freshness:** online features stale or computed over different windows.
- **Preprocessing mismatch:** different tokenizer, scaler, vocabulary or missing-value handling.
- **Distribution changes** between training data and live traffic.

**How to prevent it:**
- **Single source of feature definitions** via a feature store (Feast, Tecton, Databricks) serving the same logic offline and online.
- **Point-in-time correct joins** for training data.
- **Package preprocessing with the model** (sklearn Pipeline, TF Transform, model graph includes tokenisation) so serving can't diverge.
- **Log features at serving time** and train on those logged features ("log and wait").
- Shared libraries and unit tests asserting offline and online feature values match for the same entity/time.

**How to detect it:** compare feature distributions between training data and serving logs (PSI, KS tests), compare online predictions with offline predictions on the same inputs (shadow scoring), and monitor per-feature null rates and ranges. Alert on divergence before it shows up as a business metric drop.

### 26.67 How would you version models and datasets?

Goal: any prediction must be **reproducible and traceable** to the exact code, data, config and model that produced it.

**Datasets:**
- Store immutable snapshots; never overwrite. Use table formats with time travel (Delta Lake, Iceberg, Hudi) or tools like DVC/lakeFS for files.
- Record dataset version (snapshot ID/commit hash), schema, row counts, statistics and lineage (source tables, transformation code version).
- Version labels and train/val/test split definitions too.

**Models:**
- **Model registry** (MLflow, SageMaker, Vertex, W&B) storing artifact, semantic version, stage (staging/production/archived), metrics and links to training run.
- Each training run logs: git commit, dataset version, hyperparameters, environment (Docker image, library versions), random seeds, evaluation results.
- Package the model with its preprocessing and a signature (input/output schema).

**Prompts and LLM apps:** version prompts, retrieval configs, embedding model and index version together as a deployable unit.

**Serving and audit:** each prediction log contains model version (and feature/data versions where relevant), enabling rollback and debugging.

**Practices:** promotion via CI/CD with automated evaluation gates, approvals for regulated domains, retention policies for old versions, and model cards documenting intended use and limitations.

### 26.68 How would you perform online vs offline inference?

**Offline (batch) inference:** predictions computed on a schedule for many records at once and stored for later lookup.
- Examples: nightly churn scores, weekly demand forecasts, precomputed recommendations.
- Implementation: Spark/Ray/Beam jobs or warehouse SQL UDFs orchestrated by Airflow/Dagster; results written to a table, cache or key-value store.
- **Pros:** high throughput, cheap (spot instances, large batches), simple, no strict latency requirements, easy to validate before publishing.
- **Cons:** stale predictions; wasted compute on entities never queried; can't react to fresh context.

**Online (real-time) inference:** predictions computed per request, synchronously.
- Examples: fraud scoring at payment time, search ranking, chatbots.
- Implementation: model behind a REST/gRPC service (FastAPI, Triton, KServe), online feature store, autoscaling.
- **Pros:** fresh, uses request context; only computes what's needed.
- **Cons:** latency SLAs, high-availability ops, more expensive per prediction, training-serving skew risk.

**Middle grounds:**
- **Streaming/near-real-time:** score events from Kafka with Flink within seconds; write to a store.
- **Hybrid:** precompute heavy parts in batch (user embeddings, candidates) and do lightweight real-time ranking with fresh features.
- **Async online:** queue requests for heavy models (e.g. document processing) and notify when done.

**Decision drivers:** freshness requirement, latency, request volume and pattern, cost, and how far ahead inputs are known.

### 26.69 How would you deploy a model using blue-green or canary deployment?

**Blue-green deployment:** run two identical environments. **Blue** serves production with model v1; deploy model v2 to **green**, run smoke tests and validation against it, then switch the router/load balancer to send 100% of traffic to green. Blue remains on standby for instant rollback.
- **Pros:** simple, near-zero downtime, instant rollback.
- **Cons:** double infrastructure during the switch; all users exposed to v2 at once, so issues only visible under real traffic hit everyone.

**Canary deployment:** route a small slice of traffic to v2 (e.g. 1% → 5% → 25% → 100%) while monitoring.
- Compare v2 vs v1 on technical metrics (latency, errors, memory) and model metrics (prediction distribution, confidence, business KPIs such as conversion or fraud catch rate).
- Automated promotion/rollback based on thresholds (Argo Rollouts, Flagger, Istio/Service Mesh traffic splitting, SageMaker/Vertex traffic splitting).
- **Pros:** limits blast radius, real-traffic validation. **Cons:** slower, needs good monitoring and enough traffic for statistical confidence; sticky routing may be needed for consistent user experience.

**ML-specific additions:**
- **Shadow deployment first:** v2 receives mirrored traffic, predictions are logged but not used — validates latency and prediction agreement with zero user risk.
- Beyond canary, use **A/B tests** for business impact with proper statistics.
- Version model + preprocessing + features together; ensure the feature pipeline is compatible with both versions during the rollout; keep v1 warm until v2 is proven.

### 26.70 How would you monitor model performance after deployment?

Monitoring has to cover the **system**, the **data** and the **model's quality**, because models fail silently.

**1. Operational health:** latency (p50/p95/p99), throughput, error/timeout rates, resource utilisation (CPU/GPU/memory), queue depth, cost.

**2. Data quality and drift:**
- Schema checks, null rates, ranges, category cardinality (Great Expectations, Evidently, WhyLabs).
- Input drift vs training baseline per feature (PSI, KS test, Jensen-Shannon).
- Training-serving skew checks.

**3. Prediction monitoring:** distribution of scores/classes, confidence, rate of positive predictions — changes are an early proxy when labels are delayed.

**4. Model performance with ground truth:** join predictions with labels when they arrive (immediately for clicks, weeks later for chargebacks/churn); track accuracy, precision/recall, AUC, calibration, RMSE by segment (region, device, customer type) to catch localised degradation and fairness issues.

**5. Business KPIs:** conversion, revenue, fraud losses, user feedback — the ultimate measure.

**Process:**
- Log every prediction with model version, features and request ID.
- Dashboards and alerts with sensible thresholds; on-call runbooks.
- Scheduled evaluation on fresh labelled samples; human review for LLM outputs.
- Triggers for retraining or rollback; champion/challenger comparisons.

### 26.71 What metrics would you monitor for an ML service?

I group them into four layers:

**Service (SRE golden signals):**
- Latency p50/p95/p99 (and time-to-first-token for LLMs)
- Throughput (RPS, tokens/s)
- Error rate (5xx, timeouts, fallbacks triggered)
- Saturation: CPU/GPU utilisation, GPU memory, queue length, autoscaling events
- Availability/SLO burn rate; cost per 1k predictions

**Data/input:**
- Schema violations, missing values, out-of-range values
- Feature drift (PSI/KS per feature), new unseen categories
- Feature freshness/staleness from the feature store
- Input volume anomalies

**Model/output:**
- Prediction distribution, class balance, mean score, confidence/entropy
- Online performance when labels arrive: accuracy, precision/recall, AUC, RMSE, calibration (by segment)
- For LLMs: groundedness/hallucination rate, refusal rate, toxicity, token usage, tool-call failures, user thumbs up/down

**Business:**
- The KPI the model drives: CTR, conversion, fraud loss, approval rate, deflection rate, revenue

Alert on SLO breaches and significant drift; review business metrics alongside model metrics because a model can be "accurate" yet hurt the business (or vice versa). Always slice metrics by model version and key segments.

### 26.72 How would you detect data drift automatically?

**Approach:** compare the distribution of recent production data against a reference (training data or a stable recent window) on a schedule or streaming, and alert when the difference is significant and meaningful.

**Statistical tests / distances per feature:**
- **Numerical:** Kolmogorov–Smirnov test, Wasserstein distance, PSI (Population Stability Index; rule of thumb: < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant).
- **Categorical:** chi-squared test, PSI, Jensen–Shannon divergence; flag new categories.
- **Multivariate:** train a **domain classifier** to distinguish reference vs production rows — AUC well above 0.5 means drift, and its feature importances show what drifted. Or monitor drift in embedding space (MMD, centroid distance) for text/images.
- **Prediction drift:** distribution of model outputs — a cheap early signal when labels are delayed.
- **Concept drift:** when labels arrive, track performance over time (and detectors like DDM/ADWIN on error rates).

**Implementation:**
```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(train_df["amount"], prod_df["amount"])
```
Tools: Evidently, WhyLabs, NannyML (estimates performance without labels), Arize, SageMaker Model Monitor.

**Practicalities:**
- With large samples, p-values flag tiny, harmless shifts → use effect-size thresholds (PSI, distance) and weight by feature importance.
- Handle seasonality with appropriate reference windows.
- Alert → investigate (pipeline bug vs real-world change) → retrain or fix. Many "drift" alerts are actually upstream data bugs.

### 26.73 How would you roll back a model that performs badly in production?

**Preparation (before anything goes wrong):**
- Every model version is immutable in a **model registry**, together with its preprocessing, feature definitions and container image.
- Deploy via blue-green/canary so the previous version stays deployed (warm) for a while.
- Routing is configuration-driven (traffic split / alias like `production` → v12), so rollback is a config change, not a rebuild.
- Automated rollback triggers on canary metrics (error rate, latency, prediction distribution, business KPI guardrails).
- A runbook with owners and a decision threshold.

**During the incident:**
1. **Detect and confirm:** alerts on metrics, verify it's the model and not upstream data (a bad feature pipeline will also break the old model).
2. **Roll back:** shift traffic to the last known good version (or to a fallback such as a rules engine/simpler model) — ideally one command or automatic.
3. **Check compatibility:** old model must still match the current feature schema; if the feature pipeline changed too, roll back that as well (version them together).
4. **Communicate** to stakeholders; record the incident.

**After:**
- Root-cause analysis: data drift, bug in training data, offline/online skew, insufficient evaluation coverage.
- Add the failing cases to the evaluation/regression suite and strengthen release gates (shadow testing, longer canary).
- Reprocess or correct decisions made by the bad model if needed (important in finance/healthcare), using prediction logs keyed by model version.

### 26.74 How do you make ML experiments reproducible, and what should an experiment tracking system capture?

An experiment is reproducible when someone else can rebuild the same model, or one that is statistically the same, from a run ID alone. That means pinning **code, data, environment, configuration and randomness**, and having every run record them automatically, not from memory.

**The five things to pin:**
- **Code:** the git commit SHA, with a flag or a refusal to run when the working tree has uncommitted changes. Notebooks are converted to scripts or run in a pipeline before results count.
- **Data:** an immutable dataset reference, such as a Delta/Iceberg snapshot ID, a DVC or lakeFS commit, or a content hash of the training files. Record the split definitions and the label snapshot too. "s3://bucket/train/latest" is not a version.
- **Environment:** a lock file (uv/poetry/conda-lock) and the container image digest, plus the CUDA, cuDNN and driver versions and the GPU type. A different GPU or library version can change results.
- **Configuration:** every hyperparameter and feature list in a single config object (Hydra, a YAML file, a dataclass) that is logged in full, not just the values someone changed.
- **Randomness:** seeds for Python, NumPy and the framework, data loader worker seeds, and deterministic kernels where needed (`torch.use_deterministic_algorithms(True)`). Distributed training and some cuDNN kernels stay slightly non-deterministic, so the target is "within noise", which you check by rerunning with three to five seeds.

**What the tracking system should capture per run:**

| Category | Examples |
|---|---|
| Lineage | git SHA, dataset version, parent run, pipeline run ID, who launched it |
| Inputs | full config, hyperparameters, feature set, data statistics (rows, class balance) |
| Environment | image digest, package versions, hardware, number of workers |
| Metrics | training/validation curves per step, final metrics overall and per slice, seed variance |
| Artifacts | model weights, preprocessing objects, tokenizer, model signature, evaluation reports, confusion matrices |
| Cost and time | wall-clock time, GPU hours, peak memory |
| Context | tags (ticket, hypothesis), notes on what was learned, including failed runs |

```python
import mlflow, subprocess
with mlflow.start_run(run_name="ranker-v7-lr-sweep"):
    mlflow.set_tags({"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                     "dataset_snapshot": "orders_features@v4182"})
    mlflow.log_params(cfg)
    # ... train ...
    mlflow.log_metrics({"val_auc": auc, "val_auc_new_users": auc_new})
    mlflow.sklearn.log_model(pipeline, "model", signature=signature)
```

**Tooling:** MLflow (open source, sits next to the Databricks registry), Weights & Biases (strong for deep learning dashboards, sweeps and artifact lineage), Vertex AI Experiments, SageMaker Experiments, Neptune or Comet. Pick one per organisation so that runs can be compared across teams.

**What goes wrong in practice:**
- Logging is optional and done by hand, so only the "good" runs are recorded. Fix this by logging inside the shared training template or pipeline component.
- Data is referenced by a mutable path, and three months later the table has been backfilled.
- Preprocessing lives in a notebook and is not saved with the model.
- Results come from a single seed, so a 0.3-point "improvement" is just noise.

**Senior test:** pick a production model at random and rebuild it from its registry entry in a clean environment. If that takes more than one command and an hour of compute, the reproducibility story is incomplete.

### 26.75 What is a model registry, and how do you design the promotion workflow from experiment to production?

A **model registry** is the system of record for deployable models. It holds named models, immutable numbered versions, the artifacts and signatures for each version, lineage back to the training run and data, evaluation results, and pointers that say which version serves where. Deployment tools read from the registry, not from someone's laptop or a bucket path.

**What a registry entry should hold:**
- Artifact plus preprocessing, the input/output signature and the serving image or runtime.
- Link to the experiment run: code SHA, dataset snapshot, config.
- Evaluation report: global and per-slice metrics, fairness checks, latency and size benchmarks.
- Governance metadata: owner, intended use, model card, approval records, risk tier.
- Deployment pointers: aliases or stages such as `candidate`, `challenger`, `champion`. MLflow has deprecated fixed stages in favour of free-form **aliases** (for example `models:/fraud@champion`), Unity Catalog uses aliases with access control, and SageMaker uses model package groups with an approval status.

**Promotion workflow:**

1. **Register:** the training pipeline, not a human, registers a new version once training finishes, tagged `candidate`, with all lineage attached.
2. **Automated gates (CI for models):**
   - Schema and signature compatible with the current feature pipeline.
   - Offline metrics beat the champion on a frozen holdout and on a fresh recent window, with no regression on critical slices beyond a set tolerance.
   - Behavioural tests pass: invariance, directional checks, known-bad regression cases.
   - Latency, memory and model size within the serving budget. The package loads and serves in a staging container.
   - Bias and fairness checks for regulated use cases.
3. **Human approval when risk calls for it:** a model risk or domain reviewer signs off in the registry, which records who approved what and when. Low-risk models can be promoted automatically.
4. **Staged rollout:** set alias `challenger`, deploy to shadow or canary, compare online against the champion on guardrail and business metrics.
5. **Promote:** move the `champion` alias to the new version. Serving resolves the alias, so promotion and rollback are both a metadata change.
6. **Archive:** keep the previous champion warm for fast rollback, then archive old versions under a retention policy. Never delete versions that served production decisions in audited domains.

```python
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("fraud_model", "challenger", version=42)
# after canary passes
client.set_registered_model_alias("fraud_model", "champion", version=42)
```

**Design choices to discuss:**
- **One registry across environments, or one per environment?** A common pattern is a single registry with access controlled by environment, where the exact same artifact moves from dev to staging to prod. Rebuilding the model per environment breaks lineage.
- **Promote the artifact, not the code.** Retraining in production "with the same code" gives a different model. If you promote code instead (common at level 2 maturity), the production pipeline retrains and the gates run again on its output.
- **Couple versions:** a model version should name the feature definitions and preprocessing it needs, so deployment can refuse an incompatible combination.

**Failure modes:** aliases moved by hand outside CI, so nobody knows why production changed; gates that compare only global AUC, hiding a regression on a small but important segment; and no link from a prediction log back to the version that produced it.

### 26.76 What are the MLOps maturity levels, and what does continuous training (CT) look like?

The best-known model is Google Cloud's three levels. They describe how much of the path from data to deployed model is automated: **level 0** is manual, **level 1** automates the training pipeline so models retrain continuously, and **level 2** also automates building, testing and deploying the pipeline itself through CI/CD. Microsoft's Azure model uses five levels, but the progression is the same.

| Level | What is automated | Typical symptoms |
|---|---|---|
| 0: Manual | Nothing. Notebooks, hand-run scripts, model file handed to engineers | Releases a few times a year; training-serving skew; no monitoring; "only one person knows how to retrain it" |
| 1: ML pipeline automation | The training pipeline runs end to end on triggers: data validation, feature engineering, training, evaluation, validation, registration | Continuous training; the same pipeline runs in dev and prod; feature store and metadata store in place; models deployed automatically or after approval |
| 2: CI/CD pipeline automation | Pipeline code is tested, built and deployed automatically; many pipelines run in parallel | Data scientists ship new pipeline versions several times a week; full lineage; monitoring drives triggers |

**Key shift at level 1:** you stop deploying a *model* and start deploying a *training pipeline* that produces models. That is what makes continuous training possible.

**What continuous training looks like:**

1. **Trigger:** a schedule (nightly or weekly), new data arriving (a partition lands), a monitoring signal (drift or performance decay), or a manual run.
2. **Data extraction and validation:** pull a snapshot and check schema, volumes, null rates and distributions against expectations. The pipeline **stops** on anomalies rather than training on broken data.
3. **Feature computation:** reuse the same feature definitions that serving uses, with point-in-time correct joins.
4. **Train and tune:** a fixed or bounded hyperparameter search. Warm-start where it is safe to do so.
5. **Evaluate:** against the current champion on the same fresh holdout, overall and per slice.
6. **Validate and register:** infrastructure compatibility, latency, behavioural tests. Register as a candidate with lineage.
7. **Deploy:** automatically through shadow or canary for low-risk models, or wait for an approval step.
8. **Monitor:** production metrics feed back into the trigger.

**Level 2 adds** unit tests for feature logic and components, integration tests of the whole pipeline on a sample, building component images, deploying the pipeline definition to the orchestrator, and promoting it through environments, all from a pull request.

**How to use this in a role:**
- Don't aim for level 2 everywhere. A quarterly-updated model whose input data barely changes is fine at level 1 with a scheduled run. Invest where data changes quickly and models matter commercially: fraud, ranking, pricing.
- The order of investment that usually pays off is: reproducible training script, then a pipeline orchestrator, then data validation, then a model registry with gates, then monitoring, then triggers driven by monitoring.
- LLM applications follow a similar ladder, but the artefacts are prompts, retrieval configs and evaluation sets instead of trained weights. "Continuous training" becomes continuous evaluation, plus occasional fine-tuning.

**Failure modes of CT:** an automated pipeline that faithfully retrains on corrupted data, feedback loops where the model learns from its own decisions, and silent promotion of a slightly worse model every week. Validation gates and champion/challenger comparisons are what make CT safe.

### 26.77 When and how should you retrain models automatically, and how do champion/challenger evaluations gate a new model?

Retrain when doing so is expected to improve the business metric enough to justify the cost and risk, and let a **champion/challenger** comparison decide whether the new model actually replaces the old one. Retraining is cheap to trigger. Promotion is where discipline matters.

**When to retrain:**

| Trigger | Good for | Watch out for |
|---|---|---|
| Schedule (daily, weekly) | Stable, predictable decay; simple to operate | Wasted compute when nothing changed; too slow for sudden shifts |
| New data volume | Data arrives in batches, such as new labelled cases | Label delay means "new" data may be weeks old |
| Performance decay | Labels arrive quickly (clicks, short-horizon outcomes) | Needs reliable labels and a threshold set with statistical care |
| Drift alert | Labels late or absent | Many drift alerts are upstream data bugs, and retraining on a bug locks it in |
| Upstream change | New features, a schema change, a new product line | Usually needs a human, not only a retrain |

**Find the right cadence empirically:** train models on data ending at different dates, evaluate each on later periods, and plot performance against model age. If the curve drops 2% after a week, retrain weekly. If it is flat for six months, a quarterly schedule is fine.

**How to retrain safely:**
- Use a fixed pipeline (see continuous training), not ad hoc scripts.
- Choose the window deliberately: a sliding window forgets old regimes, an expanding window keeps them, and time-decay sample weights are a middle ground.
- Validate data before training and halt on anomalies.
- Watch for feedback loops: if the model's decisions shape the labels (credit approvals, recommendations), keep a small random exploration slice or use counterfactual correction so that training data is not only what the old model chose.

**Champion/challenger gating:**

1. **Same data, same metrics:** score both the champion and the challenger on an identical, recent, out-of-time holdout that neither model trained on.
2. **Primary metric with a margin:** the challenger must beat the champion by a minimum effect, and the improvement must hold across bootstrap resamples or seeds, not appear from a single number.
3. **Guardrail slices:** no regression beyond a tolerance on critical segments (new users, high-value customers, each region, protected groups).
4. **Operational checks:** latency, memory, calibration, prediction distribution not shifted unexpectedly.
5. **Online stage:** shadow mode to compare predictions on live traffic, then a canary or A/B test against the champion for the business metric.
6. **Promote or reject automatically,** and record the decision and evidence in the registry.

```python
def gate(champ, chall, min_lift=0.003, slice_tol=0.005):
    if chall["auc"] - champ["auc"] < min_lift:
        return False, "insufficient lift"
    for s in CRITICAL_SLICES:
        if champ[s] - chall[s] > slice_tol:
            return False, f"regression on {s}"
    if chall["p99_ms"] > SLA_MS:
        return False, "latency over SLA"
    return True, "promote"
```

**Failure modes:** comparing the challenger on fresh data against the champion's old evaluation numbers; small improvements that look significant because of one lucky split; slowly ratcheting worse because each step was "within tolerance" (compare against a fixed reference model as well as the current champion); and alert fatigue from retraining that fails noisily every night.

### 26.78 What is shadow deployment, and how does it compare with canary releases, interleaving and multi-armed bandits for online evaluation?

In **shadow deployment**, a new model receives a copy of live production traffic and makes predictions, but its outputs are only logged and never shown to users or used in decisions. It tests the model on real inputs with zero user risk, but it cannot measure how users react, because users never see its outputs.

**How shadow mode works:**
- Mirror requests at the gateway or service mesh (Istio mirroring, SageMaker shadow variants, a dual call in application code), usually asynchronously so that shadow latency cannot slow the main path.
- Log both predictions with a shared request ID, then compare: agreement rate, score distributions, latency, errors, and, once labels arrive, accuracy of both.
- Good for catching training-serving skew, feature bugs, latency regressions and unexpected prediction shifts before any user is exposed.
- Costs extra compute. Avoid shadow calls with side effects (sending emails, calling paid APIs, writing to shared state).

**The online evaluation options:**

| Method | What users see | What it measures | Strengths | Limits |
|---|---|---|---|---|
| Shadow | Only the old model | Technical behaviour, agreement, offline-style accuracy on live data | Zero user risk; catches pipeline bugs | No causal measure of user impact; doesn't work when the model's output changes later inputs |
| Canary | Small, growing share sees the new model | Errors, latency, guardrails, early KPI signal | Limits blast radius; standard rollout safety | Small slices give weak statistics; mainly a safety mechanism |
| A/B test | Users randomised to old or new | Causal effect on business metrics | Gold standard for impact; supports long-term metrics | Needs a lot of traffic and time; one comparison at a time |
| Interleaving | One mixed ranked list built from both rankers | Which ranker's items users prefer (clicks attributed to each) | Far more sensitive, often reported as 10 to 100 times fewer users needed, because each user compares both | Only for ranking and recommendation; measures preference, not retention or revenue |
| Multi-armed bandit | Traffic shifts towards the better arm as evidence grows | Cumulative reward | Less regret during the test; good for many variants or short-lived content | Biased effect estimates; poor for slow-arriving or long-term metrics; harder to analyse |

**Interleaving in practice:** team-draft interleaving alternates picks from rankers A and B into one list and credits each click to the ranker that supplied the item. Netflix, Spotify and DoorDash use it as a fast first filter: many candidate rankers are interleaved, and only the winners go to a full A/B test that measures long-term metrics.

**Bandits in practice:** Thompson sampling or UCB fit headline selection, promotions, or choosing among several prompts when the reward is immediate (a click or a thumbs up). Contextual bandits personalise the choice. Don't use them when the success metric arrives weeks later or when you need a clean, unbiased effect estimate for a launch decision.

**How I sequence them for a new ranking model:** offline evaluation, then shadow for a few days (skew, latency, agreement), then interleaving against the champion to filter candidates, then a canary for safety, then an A/B test for the business decision, then a full rollout with a small long-term holdback group.

**Failure modes:** treating shadow agreement as proof of improvement (a better model is supposed to disagree); calling a canary "a test" and launching on 1% of traffic over two hours; and running bandits on metrics with delayed rewards, so traffic chases noise.

### 26.79 What is point-in-time correctness in a feature store, and how do you build leakage-free training sets?

**Point-in-time correctness** means each training example contains only the feature values that were actually known at the moment the prediction would have been made. If a training row for a transaction at 10:05 uses a "transactions in the last hour" feature computed at 11:00, the model learns from the future, looks great offline and disappoints in production. This is **data leakage**, and a time-unaware join is the most common cause.

**How a point-in-time join works:**
- Start from a **spine** (Feast calls it the entity DataFrame): one row per labelled event with an entity key, an event timestamp and the label.
- For each row and each feature view, pick the **latest feature value whose timestamp is at or before the event timestamp**. This is an "as-of" join, not an equality join.
- Apply a **TTL** or maximum age: if the latest value is older than the TTL, return null rather than a stale value. This mirrors what the online store would have served.
- Feast's `get_historical_features`, Tecton, Databricks Feature Engineering and SageMaker Feature Store all implement this. You can also write it yourself in SQL.

```sql
SELECT s.order_id, s.event_ts, s.label, f.avg_basket_30d
FROM spine s
LEFT JOIN LATERAL (
  SELECT avg_basket_30d
  FROM customer_features f
  WHERE f.customer_id = s.customer_id
    AND f.feature_ts <= s.event_ts
    AND f.feature_ts > s.event_ts - INTERVAL '2 days'   -- TTL
  ORDER BY f.feature_ts DESC
  LIMIT 1
) f ON TRUE;
```
Spark, Snowflake and DuckDB offer `ASOF JOIN` or window-function patterns that do the same thing at scale.

**Subtle sources of leakage to design against:**
- **Event time vs availability time:** a feature computed for day D might only land at 03:00 on D+1. Use the time the value became *available* (Feast's created timestamp, Tecton's processing time) or add an explicit availability lag, not just the logical date.
- **Overwritten tables:** a dimension table updated in place (current customer segment, current address) leaks future state. Keep history as SCD2 or append-only snapshots.
- **Backfilled aggregates:** windows recomputed with late data contain events that were not visible at serving time.
- **Labels inside features:** features derived from the outcome, such as "number of chargebacks" when predicting a chargeback, or a status field updated after the event.
- **Random splits on time series:** use out-of-time validation, and group splits by entity when the same user appears many times.
- **Global preprocessing:** scalers, target encodings or vocabularies fitted on the full dataset before splitting. Fit them on training folds only.

**Making it structural, not heroic:**
- Define features once and use the same definition for offline backfills and online serving.
- Log features at serving time and train on those logs where possible ("log and wait"). This is leakage-free by construction and removes skew.
- Automate leakage checks: features with suspiciously high importance, offline/online metric gaps, and feature timestamps later than event timestamps should fail the pipeline.
- Handle duplicate timestamps deterministically. Ties in the source can make the join pick a different row on each run.

**Interview example:** a fraud model reached 0.97 AUC offline and 0.81 online. The cause was a merchant risk score table refreshed daily and joined on merchant ID only, so training rows saw the score computed after the fraud was reported. Rebuilding the training set with as-of joins and a one-day availability lag brought offline AUC to 0.84, which matched production.

### 26.80 How do you test ML systems?

ML systems need ordinary software tests **plus** tests for data and for learned behaviour, because a model can pass every unit test and still be wrong. Google's "ML Test Score" rubric groups these into four areas: **data tests, model tests, ML infrastructure tests and monitoring tests**. I use that structure and add behavioural testing from NLP practice.

**1. Data tests (before training and on every batch):**
- Schema: types, required columns, allowed categories.
- Expectations: ranges, null rates, uniqueness, referential integrity, volume within normal bounds, freshness.
- Distribution checks against a reference (PSI, KS) with alert thresholds.
- Feature value checks: every feature in the model is justified, costs what it is worth, and complies with privacy rules.
- Tools: Great Expectations, Pandera, Soda, dbt tests, TFX Data Validation.

**2. Model tests:**
- **Quality gates:** beats a simple baseline and the current champion on an out-of-time holdout, overall and per important slice.
- **Behavioural tests** (the CheckList approach):
  - **Minimum functionality tests:** simple cases the model must get right ("I love this" is positive).
  - **Invariance tests:** changes that should *not* change the prediction, such as swapping a person's name or gender in a support ticket, adding whitespace, or using a different currency format.
  - **Directional expectation tests:** changes that should move the prediction in a known direction. Higher income should not lower a credit score; adding "refund not received" should raise complaint severity.
- **Regression suite:** every past production incident becomes a test case.
- **Robustness and fairness:** noise, missing features, adversarial inputs; metric parity across protected groups.
- **Calibration and staleness:** predicted probabilities match observed rates; measured decay of an older model shows how often to retrain.

```python
def test_income_monotonic(model, base_applicant):
    low  = model.predict_proba(base_applicant | {"income": 40_000})[1]
    high = model.predict_proba(base_applicant | {"income": 90_000})[1]
    assert high >= low - 1e-3

def test_name_invariance(clf):
    a = clf("Order 123 from Emily is late")
    b = clf("Order 123 from Jamal is late")
    assert a.label == b.label
```

**3. Infrastructure and pipeline tests:**
- Unit tests for feature transformations, with fixed inputs and expected outputs.
- **Training-serving parity test:** compute features for the same entity and time through the offline and online paths and assert they are equal.
- Pipeline integration test: run the full pipeline on a small sample in CI, including registration and serving a test request.
- Training reproducibility (two runs land within tolerance), model loading, signature validation, and a quick training sanity check (the model can overfit a tiny batch, the loss decreases).
- Serving tests: latency and throughput under load, memory, graceful degradation, rollback works.

**4. Monitoring tests (production):** schema and invariants on live inputs, skew between training and serving features, prediction distribution, staleness of the deployed model, and the latency, errors and throughput of the serving system.

**Where each test runs:**

| Stage | Tests |
|---|---|
| Pull request | Unit, feature parity on fixtures, tiny-batch training |
| Training pipeline | Data validation, quality gates, behavioural suite, fairness |
| Pre-deploy | Load test, shadow comparison, signature checks |
| Production | Monitoring checks and alerts |

**Senior points:** tests are probabilistic, so set tolerances and run behavioural suites as pass rates (for example, 98% of invariance cases must hold) rather than single asserts. Keep evaluation datasets versioned like code. And remember the trap in ML testing: the dangerous failures are silent, so most of the value is in data and monitoring tests, not unit tests.

### 26.81 How do you monitor a model when ground-truth labels arrive late or never?

When labels are delayed (loan defaults after months, chargebacks after weeks) or never arrive (a declined transaction never shows whether it was fraud), you cannot measure accuracy directly. Instead you combine **proxy signals**, **performance estimation** and **targeted labelling**, and you still join real labels whenever they do arrive.

**1. Monitor the inputs.**
- Data quality: schema violations, null rates, out-of-range values, upstream volume drops. Many "model problems" are broken pipelines.
- Feature drift against the training reference (PSI, KS, Jensen-Shannon), weighted by feature importance, plus multivariate drift with a domain classifier.
- Embedding drift for text and images.

**2. Monitor the outputs.**
- Prediction distribution, mean score, positive rate, class mix, confidence or entropy. A fraud model whose flag rate jumps from 0.8% to 2.5% overnight needs investigating, whatever the labels later say.
- Calibration against the historical base rate.
- Agreement with a reference model or a rules baseline running in shadow.

**3. Estimate performance without labels.**
- **Confidence-based performance estimation (CBPE):** for a well-calibrated classifier, expected accuracy, precision or AUC can be computed from predicted probabilities. NannyML implements this. It fails when the model is poorly calibrated or under concept drift, because the model's confidence is then wrong in exactly the cases that matter.
- **Direct loss estimation:** train a second model to predict the primary model's error from its inputs and outputs. Useful for regression.
- **Importance weighting:** reweight the labelled reference set to resemble current inputs and recompute metrics. Works for covariate shift only.

**4. Use early and proxy labels.**
- Short-horizon proxies that correlate with the final label: 30-day delinquency as an early read on 12-month default, early dispute filings ahead of chargebacks, a user re-asking a question as a signal of a bad chatbot answer.
- Validate the proxy against the true label on historical data before trusting it.

**5. Create labels deliberately.**
- **Human review** of a stratified sample, with extra samples where the model is uncertain or where drift is highest.
- **Exploration or holdout slices** to deal with the selection problem: let a small random share of borderline cases through (or route them to manual review) so you learn the outcomes the model would otherwise hide. Approve-or-decline systems need this, otherwise you only ever see labels for what you approved.
- For LLM applications, use LLM-as-judge on sampled traffic, calibrated against human ratings, plus implicit feedback (thumbs, edits, retries).

**6. Close the loop when labels arrive.**
- Store predictions with request ID, model version and features, and join labels as they arrive. Report performance by **prediction cohort** (all predictions from week 32), noting how complete each cohort's labels are.

**Alerting design:**

| Signal | Latency | Reliability | Role |
|---|---|---|---|
| Data quality | Minutes | High for pipeline bugs | Page on-call |
| Prediction drift | Hours | Medium | Investigate |
| Estimated performance | Hours to days | Depends on calibration | Trigger review or retraining |
| Proxy labels | Days to weeks | Validated correlation | Early warning |
| True labels | Weeks to months | Ground truth | Confirm, recalibrate thresholds |

**Pitfall:** retraining automatically on drift alone. Drift without a performance effect is common (a feature the model barely uses), and an upstream bug that looks like drift gets baked into the new model.

### 26.82 How do model serving frameworks such as Triton, KServe, BentoML and Ray Serve compare, and how does dynamic batching improve throughput?

They solve different layers of the problem. **Triton** is a high-performance inference *server* that runs models efficiently on GPUs. **KServe** is a Kubernetes *control plane* for deploying, scaling and routing model servers. **BentoML** is a Python-first *packaging and serving* framework. **Ray Serve** is a Python-native *distributed serving library* that excels at composing many models and business logic. They are often combined, for example KServe deploying Triton, or Ray Serve calling vLLM.

| | Triton Inference Server | KServe | BentoML | Ray Serve |
|---|---|---|---|---|
| What it is | C++ inference server with multiple backends (TensorRT, ONNX Runtime, PyTorch, TensorFlow, Python, TensorRT-LLM, vLLM) | Kubernetes CRD (`InferenceService`) that runs model servers, including Triton, vLLM, sklearn, XGBoost and custom ones | Python framework to define a service, build a "Bento" image and deploy it anywhere or on BentoCloud | Library on the Ray cluster; deployments are Python classes with autoscaling replicas |
| Strengths | Maximum GPU throughput, dynamic batching, concurrent model instances, ensembles, metrics | Standard deployment API, scale-to-zero with Knative, canary traffic split, transformers/explainers, inference graphs, multi-model serving | Very fast path from a notebook to an API; adaptive batching; good developer experience | Composition of many models and logic in one Python app; fractional GPUs; shares a cluster with Ray Data and Ray Train |
| Weak spots | Configuration-heavy; Python logic is less natural | Needs a Kubernetes and Knative/Istio skill set; it is not a fast runtime by itself | Peak performance below a tuned Triton setup | Operating a Ray cluster; debugging distributed Python |
| Choose when | Latency and cost per GPU matter most | You run a platform on Kubernetes for many teams | A small team needs to ship custom Python models quickly | Pipelines of models (retrieval, ranking, LLM) and Python business logic |

For LLMs specifically, the runtime is usually vLLM, SGLang or TensorRT-LLM, wrapped by one of these for deployment and routing.

**Dynamic batching:**

GPUs are throughput machines. A single request uses a fraction of the compute, while the cost of reading the weights from memory is almost the same for one input as for thirty-two. **Dynamic (server-side) batching** collects individual requests that arrive close together and runs them as one batch.

- Triton waits up to `max_queue_delay_microseconds` to fill a batch, prefers certain sizes, and sends what it has when the delay expires.
- BentoML's adaptive batching tunes the batch window automatically from observed latency.
- Ray Serve's `@serve.batch` decorator batches calls to a method using `max_batch_size` and `batch_wait_timeout_s`.

```
# Triton config.pbtxt
max_batch_size: 64
dynamic_batching {
  preferred_batch_size: [16, 32]
  max_queue_delay_microseconds: 2000
}
instance_group [{ count: 2, kind: KIND_GPU }]
```

**The trade-off:** batching adds queueing delay to every request in exchange for much higher throughput. Under light load the delay is pure cost. Under heavy load, batching *reduces* tail latency because queues drain faster. Tune using measured load: sweep the queue delay and batch size with a load generator (Triton's `perf_analyzer`) and choose the setting that maximises throughput while meeting the p99 SLO.

**Continuous batching for LLMs:** request-level batching wastes capacity because sequences finish at different times. LLM servers batch at the token level: finished sequences leave and new ones join at every decode step. That is why vLLM and TensorRT-LLM achieve much higher throughput than a plain batch window.

**Pitfalls:** padding waste when inputs vary in length (bucket by length); a batch window longer than the latency budget; one slow request holding up a batch; and autoscaling on CPU when the real signal is queue depth or GPU utilisation.

### 26.83 How do you choose an ML pipeline orchestrator, comparing Kubeflow Pipelines, Vertex AI or SageMaker Pipelines, Airflow and Metaflow?

Choose on **who writes the pipelines, where compute runs, and what the organisation already operates**, not on feature checklists. Most orchestrators can run a training DAG. They differ in ML-specific features (artifact tracking, caching, GPU steps), developer experience, and operational burden.

| | Kubeflow Pipelines (KFP) | Vertex AI Pipelines / SageMaker Pipelines | Airflow | Metaflow |
|---|---|---|---|---|
| Model | Containerised steps on Kubernetes, defined with the KFP Python SDK | Managed services. Vertex runs KFP-format pipelines serverlessly; SageMaker has its own SDK | General-purpose scheduler of tasks defined in Python DAGs | Python flows of steps, created at Netflix; runs locally, on AWS Batch or Kubernetes, and deploys to Step Functions or Argo |
| ML features | Artifact lineage, step caching, metadata store, per-step GPUs | Native integration with each cloud's training, registry, endpoints, experiments and model monitoring | Few built in; relies on operators and external tracking | Artifact versioning of every step variable, `@resources` for GPUs, resume from a failed step, cards |
| Ops burden | High: you run Kubernetes and Kubeflow | Low: fully managed, pay per run | Medium; low with managed Airflow (MWAA, Cloud Composer, Astronomer) | Low to medium |
| Developer experience | Container-centric; slower local loop | Good inside the cloud console; tied to the cloud | Familiar to data engineers; DAGs are less natural for experimentation | Excellent for data scientists; local and cloud runs use the same code |
| Lock-in | Portable across Kubernetes clusters | High | Low | Low |
| Best fit | Platform teams standardised on Kubernetes, multi-cloud or on-premises | Teams committed to GCP or AWS who want the least operations | Organisations already running Airflow for data pipelines; batch scoring | Data-science-led teams who want fast iteration on many models |

**Questions I ask:**
1. **Who are the authors?** Data scientists favour Metaflow or managed SDKs. Data engineers already know Airflow or Dagster.
2. **Where is compute and data?** If everything is in one cloud and the team is small, the managed service saves months. Hybrid or on-premises GPUs push towards KFP or Metaflow on Kubernetes.
3. **Do we need ML lineage and caching?** KFP, Vertex, SageMaker and Metaflow record artifacts per step. With Airflow you add MLflow or W&B.
4. **Event triggers and scale:** hundreds of models retrained daily, dynamic fan-out per country or customer, and triggering on data arrival.
5. **Who will be on call?** An open-source Kubernetes stack without a platform team becomes a liability.

**Common architecture:** Airflow or Dagster stays the *data* orchestrator (ingestion, dbt, feature tables) and triggers an ML pipeline (Vertex, SageMaker, KFP or Metaflow) when inputs are ready. This keeps scheduling in one place and lets ML steps use specialised features. Avoid running heavy training inside Airflow workers; Airflow should launch the work elsewhere and wait.

**Other options worth naming:** Dagster (asset-centric, strong lineage, popular for combined data and ML), Prefect, Flyte (strongly typed, Kubernetes-native, used at Lyft and Spotify), ZenML (an abstraction layer over several orchestrators), and Databricks Workflows when the lakehouse is the platform.

**Failure modes:** choosing Kubeflow for a three-person team and spending a year operating it; writing business logic inside orchestrator-specific code so steps cannot be tested locally (keep steps as plain, tested Python functions or containers); and running two parallel orchestration stacks with no shared lineage.

### 26.84 How do you design batch inference at scale?

Batch inference scores large numbers of records on a schedule and publishes the results for later lookup. At scale, the design goal is **high throughput at low cost with correctness guarantees**: every eligible entity scored exactly once, with the right model and features, and nothing published until it has been validated.

**Reference architecture:**

1. **Trigger:** the orchestrator starts when upstream feature tables for the partition are complete, not just at a clock time.
2. **Resolve versions:** read the `champion` model version from the registry and pin it for the run, together with the feature snapshot. Record both in the output.
3. **Build the scoring set:** select the entities to score (all active customers, or only those whose features changed) with point-in-time features as of the run time.
4. **Distributed scoring:**
   - CPU models (trees, linear): Spark with a `mapInPandas` or pandas UDF, broadcasting the model once per executor, or warehouse-native scoring (Snowpark, BigQuery ML, Databricks `ai_query`).
   - GPU and deep learning models: Ray Data `map_batches` with an actor pool that loads the model once per GPU and streams data, so CPU preprocessing and GPU inference overlap. For LLM batch work, run vLLM offline or use provider batch APIs, which are typically about half price with a 24-hour turnaround.
5. **Validate output:** row counts against input, null rates, score distribution against the previous run, share of entities scored. Fail if the checks fail.
6. **Publish atomically:** write to a new partition or table version and swap a view or pointer, then sync to the online key-value store (Redis, DynamoDB, Cassandra) if applications read it in real time.
7. **Monitor:** run duration, cost, skipped entities, drift of predictions over time.

```python
import ray
class Scorer:
    def __init__(self):
        self.model = load_model("models:/ranker@champion").to("cuda")
    def __call__(self, batch):
        batch["score"] = self.model.predict(batch["features"])
        return batch

ds = ray.data.read_parquet("s3://features/dt=2026-11-15/")
ds.map_batches(Scorer, batch_size=2048, num_gpus=1, concurrency=8) \
  .write_parquet("s3://scores/dt=2026-11-15/run=8841/")
```

**Throughput and cost levers:**
- Load the model once per worker, never per row. Use large batches and vectorised inference.
- Right-size partitions so that work is spread evenly; deal with skewed keys.
- Use spot or preemptible instances with **idempotent, checkpointed** work units, so a lost node only reruns its shard.
- Score only changed entities (incremental scoring) when features change slowly.
- Use a lighter export (ONNX, TensorRT, quantised) for the batch path if quality holds.
- Prune columns and push filters into the reader.

**Correctness concerns:**
- **Idempotency:** a rerun overwrites the same partition, never appends duplicates.
- **Consistency:** one model version per run. Don't let the alias move halfway through.
- **Same transforms as training:** package preprocessing with the model to avoid skew.
- **Freshness contract:** consumers must know the score's as-of time and what happens when the job is late (serve yesterday's scores, fall back to a default).
- **Backfills:** design the job to score any historical date with the model and features of that date, for audits and for evaluating new models.

**Failure modes:** a model loaded inside a UDF for every row; an out-of-memory error from collecting results to the driver; partial results published after a mid-run failure; an upstream table landing late, so the job scores stale features "successfully"; and the online store holding scores from two different model versions after a partial sync.

### 26.85 What is hidden technical debt in ML systems, and how do you prevent it?

The idea comes from Sculley et al.'s 2015 paper "Hidden Technical Debt in Machine Learning Systems". ML gives fast wins, but it carries long-term maintenance costs that ordinary code-level tools don't reveal. The model code is a small box in a large system of data collection, feature extraction, configuration, serving and monitoring, and most of the debt sits in that surrounding system.

**Main forms of debt and how to counter each:**

| Debt | What it looks like | Prevention |
|---|---|---|
| **Entanglement (CACE: "changing anything changes everything")** | Adding, removing or rescaling one feature changes how the model uses all the others; no input is truly independent | Treat any change as a new model; full evaluation and gating for every change; ablation studies; isolated models where it makes sense |
| **Correction cascades** | Model B is trained to patch model A's output for a new domain, then C patches B; improving A breaks B and C | Add features or train a combined model instead of stacking corrections; document model dependencies |
| **Undeclared consumers** | Other teams quietly read a model's scores or a feature table; changes break them or create hidden feedback | Access control on prediction and feature outputs; registered consumers and data contracts; lineage tooling |
| **Unstable data dependencies** | Inputs come from another system whose semantics change (a new upstream model version, a recoded category) | Versioned, contract-bound inputs; freeze copies of upstream signals; data validation on every run |
| **Underutilised data dependencies** | Legacy, bundled or epsilon features that barely help but add risk and cost | Leave-one-feature-out evaluations; regular feature pruning; a cost for each feature |
| **Feedback loops** | The model influences its own future training data (recommendations, credit approvals, pricing); also hidden loops through two systems that affect each other | Exploration or holdout traffic; log propensities; counterfactual evaluation; watch for interacting systems |
| **Glue code** | 95% of the code adapts data to and from generic ML packages | Wrap libraries in clean, shared APIs; use standard pipeline components |
| **Pipeline jungles** | Data preparation grown incrementally into a tangle of scrapes, joins and intermediate files | Design feature pipelines holistically; feature store or a declarative transformation layer; end-to-end tests |
| **Dead experimental code paths** | Flags and branches left from old experiments | Remove experiments on a schedule; track flags with owners and expiry dates |
| **Configuration debt** | Thousands of lines of config, poorly reviewed, with hidden dependencies between settings | Config as code: validated schemas, code review, diffs between runs, sensible defaults |
| **Changes in the external world** | Fixed thresholds, prediction bias, drift | Monitoring of prediction bias, action limits, alerts on drift and upstream producers |

**Abstraction debt and smells:** few good abstractions for ML systems (compare relational algebra for databases), plus "smells" such as plain-old-data types (raw floats and strings with no meaning attached), multi-language stacks and prototypes promoted to production.

**How I prevent it as a leader:**
- **Make debt visible:** score systems with the ML Test Score rubric or a similar production-readiness checklist and review it quarterly.
- **Platform over heroics:** shared feature store, pipeline templates, registry, monitoring and evaluation harness, so each team does not rebuild glue.
- **Ownership:** every model, feature and dataset has an owner, consumers are registered, and deprecation is planned.
- **Budget for it:** reserve time each quarter for pruning features, deleting dead paths, upgrading dependencies and retiring unused models. Count retired models as an achievement.
- **Hiring and culture:** reward simplifications and deletions, not just new models with a 0.5% lift that double maintenance costs.

**LLM-era equivalents:** prompt sprawl without versions, chains of prompts correcting each other (a modern correction cascade), dependencies on provider model versions that change underneath you, unversioned evaluation sets and retrieval indexes built with an embedding model nobody recorded.

### 26.86 How do you change the embedding model behind a production vector index without downtime?

Vectors from different embedding models live in **incompatible spaces**: a query embedded by the new model cannot be compared with documents embedded by the old one, even when the dimensions match. So you cannot swap the model in place. You build a second index with the new embeddings, keep it current, validate it, and switch traffic atomically. It is a blue-green deployment for an index.

**Step-by-step migration:**

1. **Evaluate offline first.** Build a sample index with the new model and measure recall@k, nDCG and end-to-end answer quality on your own labelled queries, not on public benchmark scores. Check per segment: languages, document types, long-tail queries. Also measure the effect on latency, dimensions and storage.
2. **Create a new index or collection** sized and tuned for the new model (dimension, distance metric, HNSW parameters, quantisation). Version its name, such as `docs_v3_bge_m3`.
3. **Dual write:** from now on, every document insert, update and delete goes to both indexes, embedded with the respective model. Without dual writes, the new index is stale by the time the backfill finishes.
4. **Backfill:** re-embed the existing corpus into the new index with a batch job that is idempotent, checkpointed and throttled so it does not starve production. Track progress by document ID and version. For a provider API, batch endpoints are cheaper. For self-hosted models, GPU batch jobs.
5. **Verify completeness:** document counts and ID sets match, spot-check that metadata and access-control fields came across, and that deletes during the backfill were applied.
6. **Shadow and compare:** send live queries to both indexes, embedding each query with the model that matches the index. Log overlap of results, retrieval scores and, if possible, LLM-judge quality. Then canary a small share of real traffic on the new index.
7. **Cut over atomically:** flip an alias (Qdrant, Elasticsearch/OpenSearch and Weaviate support aliases) or a config pointer so that the query embedding model and the index switch **together**. Changing one without the other is the classic outage.
8. **Keep rollback open:** keep dual writes and the old index for a soak period, then stop writing to the old index and delete it.

```python
# Config that versions query encoder and index as one unit
RETRIEVER = {"index_alias": "docs_live", "encoder": "bge-m3@2026-09", "version": 3}
# cutover = repoint alias docs_live -> docs_v3 and deploy matching encoder together
```

**Costs and trade-offs:**
- Storage and write cost roughly double during the migration window, which can last days or weeks for large multi-tenant corpora.
- Re-embedding cost scales with corpus size. Estimate tokens times price, or GPU hours, before committing.
- **Alternatives** when a full re-embed is too expensive: a learned adapter that maps new query vectors into the old space (useful as a temporary bridge, with some quality loss), or lazy re-embedding, which is hard to do without mixing spaces. For quality-critical retrieval, plan the full re-embed.

**Related details people miss:**
- Chunking changes often come with model changes. Treat chunking, embedding model and index as one versioned artifact.
- Rerankers, similarity thresholds and hybrid-search weights were tuned for the old score distribution, so retune them.
- Cached query embeddings and semantic caches must be invalidated or namespaced by model version.
- Store the embedding model version as metadata on every vector, so mixed indexes can be detected.
- Pin hosted embedding models to dated versions. A provider silently updating a model has the same effect as an unplanned migration.

### 26.87 How do you build a data flywheel that turns production feedback into better evaluations and models for an LLM application?

A **data flywheel** is a loop in which production usage generates signals, those signals are turned into curated evaluation sets and training data, and the improved system attracts more usage and better signals. For LLM applications the first and most valuable output is usually **better evaluations**, then better prompts and retrieval, and only later fine-tuned models.

**The loop:**

1. **Capture everything, with consent and privacy controls.** Log traces with inputs, retrieved context, prompts, tool calls, outputs, model and prompt versions, latency and cost (Langfuse, LangSmith, Arize Phoenix, OpenTelemetry-based tracing). Redact PII and honour retention and opt-out rules; enterprise customers often forbid training on their data.
2. **Collect feedback signals.**
   - Explicit: thumbs up or down, ratings, reported issues, corrections.
   - Implicit: copying or accepting an answer, edits to a draft (the edit itself is gold for training), retries and rephrasing, escalation to a human agent, abandoning the session, task completion.
   - Automatic: LLM-as-judge scores for groundedness and relevance on sampled traffic, guardrail hits, tool errors.
3. **Triage and mine failures.** Cluster low-scoring traces by topic or embedding, find failure categories (retrieval misses, wrong tool, format errors, refusals, hallucinations), and size them by frequency and impact. This is error analysis, and it tells you what to fix.
4. **Curate evaluation sets.** Turn representative failures and important successes into versioned test cases with reference answers or grading rubrics. Keep a stable "golden" set for regression testing and a rotating set from recent traffic. Human experts label or review, and judge prompts are calibrated against them.
5. **Improve the cheapest lever first:** prompt changes, retrieval fixes (chunking, hybrid search, reranking, missing documents), tool descriptions, guardrails, routing. Each change must pass the evaluation suite in CI.
6. **Fine-tune when the data justifies it:** use accepted outputs and human corrections for supervised fine-tuning, pairs of preferred and rejected answers for preference tuning (DPO), or distil a large model's good outputs into a smaller, cheaper model for a narrow task.
7. **Ship safely:** offline evaluation, shadow or canary, A/B test on product metrics, then monitor. New traffic restarts the loop.

**Making the flywheel actually spin:**
- **Design the product for feedback:** edit-in-place, accept or reject buttons and citations that users can click give much richer signals than a rarely used thumbs icon.
- **Close the loop operationally:** a weekly review of failure clusters with owners, and a target such as "every production incident becomes an evaluation case within 48 hours".
- **Version everything together:** dataset version, evaluation suite version, prompt, model and index, so that improvements are attributable.
- **Measure the flywheel:** growth in evaluation coverage, time from failure to fix, the share of traffic covered by tests, quality and cost trends.

**Risks to manage:**
- **Biased feedback:** thumbs come mostly from unhappy or highly engaged users; correct for this with sampling, not raw counts.
- **Feedback loops and reward hacking:** optimising for thumbs up can produce flattering or overly long answers. Keep human-validated quality metrics as guardrails.
- **Judge drift:** periodically re-check the LLM judge against human labels.
- **Data leakage between training and evaluation:** keep evaluation cases out of fine-tuning data, or the scores become meaningless.
- **Privacy and licensing:** training on user content needs a lawful basis and contractual permission. Keep a lineage record of which data trained which model.

**Concrete example:** a support assistant logs that 18% of conversations end in a human handoff. Clustering shows a third concern refund policies for a new region whose documents were never indexed. Fixing ingestion and adding 60 evaluation cases cuts handoffs by four points. Months later, 20,000 agent-corrected answers are used to fine-tune a smaller model that matches quality at a third of the cost.

### 26.88 How do Ollama, vLLM and SGLang compare, and when would you use each?

They target different jobs. **Ollama** is a local model runner: one install, `ollama run`, and a model is working on a laptop. **vLLM** and **SGLang** are data-center serving engines built to maximise throughput on GPUs under many concurrent users. The real question is rarely "which is best" but "single developer or production fleet?"

**Architecture:**

- **Ollama:** a Go server around the GGML/llama.cpp family of runtimes, with its own engine for newer and multimodal models, and an MLX-based runner on Apple Silicon since 2026. Models are pulled from a registry as quantized GGUF files and configured with a `Modelfile`. It optimises for low memory footprint and zero configuration rather than batch throughput.
- **vLLM:** **PagedAttention** stores the KV cache in fixed-size blocks, like virtual memory pages, so memory fragmentation is near zero and many more sequences fit. **Continuous batching** adds and removes requests at every decode step. Automatic prefix caching, chunked prefill and speculative decoding are built in.
- **SGLang:** also continuous batching and paged memory, but the KV cache is indexed in a **radix tree** (RadixAttention), so any shared prefix (system prompt, few-shot examples, multi-turn history, agent loops) is reused automatically. A low-overhead scheduler and fast grammar-constrained decoding make it strong for structured outputs and multi-call LLM programs.

**Comparison:**

| | Ollama | vLLM | SGLang |
| --- | --- | --- | --- |
| Primary target | Laptop, workstation, Apple Silicon, CPU, single GPU | NVIDIA and AMD data-center GPUs; also TPU, Intel, Gaudi and others via plugins | NVIDIA and AMD data-center GPUs; also TPU, Intel, Ascend |
| Model format | GGUF (Q4_K_M, Q5_K_M, Q8_0 and so on) | Hugging Face weights; FP8, FP4 (NVFP4, MXFP4), INT8, INT4, GPTQ, AWQ; GGUF supported but not the fast path | Hugging Face weights; FP8, FP4, INT4, AWQ, GPTQ |
| Concurrency | Low: parallel slots per model (default 1) with a queue | High: continuous batching, hundreds of sequences | High: continuous batching plus prefix reuse |
| Multi-GPU | Splits layers across GPUs only if the model does not fit on one | Tensor, pipeline, data, expert and context parallelism | Tensor, pipeline, data and expert parallelism; prefill/decode disaggregation |
| Prefix caching | Basic, per slot | Automatic, block hash based | RadixAttention, the strongest for shared prefixes |
| Structured output | JSON schema via `format` | JSON schema, regex and grammar (xgrammar, guidance) | JSON schema, regex and EBNF (xgrammar by default) |
| API | Native REST plus OpenAI-compatible `/v1` | OpenAI-compatible server, plus Anthropic Messages and gRPC | OpenAI-compatible server and a native Python frontend |
| Ease of use | Easiest: install and run | Moderate: GPU drivers, memory tuning | Moderate: similar to vLLM, more knobs |

**Throughput in practice:** on a single GPU with one user, the gap is small. With dozens of concurrent users, vLLM and SGLang deliver many times Ollama's aggregate tokens per second, because one forward pass serves the whole batch. SGLang tends to lead when prompts share long prefixes (RAG templates, agents, few-shot classification) and on large mixture-of-experts models with expert parallelism. vLLM has the broadest model and hardware coverage and the largest ecosystem (KServe, llm-d, Ray Serve). Benchmarks swing between releases, so test on your own model, prompt mix and SLOs.

**All three expose OpenAI-compatible endpoints,** so application code can move between them by changing the base URL:

```python
from openai import OpenAI
# Default ports: Ollama 11434, vLLM 8000, SGLang 30000 (all serve under /v1)
schema = {"type": "object", "properties": {"label": {"type": "string"}}}
client = OpenAI(base_url=BASE_URL, api_key="unused")  # e.g. the vLLM server on port 8000
resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Classify this ticket"}],
    response_format={"type": "json_schema", "json_schema": {"name": "t", "schema": schema}},
)
```

**How I decide:**

- **Ollama:** local development, prototyping, offline or privacy-sensitive desktop use, demos, edge boxes, Apple Silicon, or a small internal tool with a handful of users.
- **vLLM:** the default for production self-hosting on GPUs: a wide model catalogue, many concurrent users, mature Kubernetes integrations and the broadest hardware support.
- **SGLang:** production workloads dominated by shared prefixes, agentic multi-call pipelines, heavy structured-output use, or very large MoE models where disaggregated prefill/decode and expert parallelism pay off.

**Common mistake:** prototyping on Ollama, then load-testing the same setup with 50 users and concluding that self-hosting is too slow. Ollama's defaults (one parallel slot, 4K context, a 4-bit GGUF) are not a production serving configuration. Another trap is comparing quality across engines without fixing the quantization: a Q4 GGUF and an FP8 checkpoint are different models.

---

← [Chapter 25. Backend Engineering, APIs and DevOps](25-backend-engineering-apis-and-devops.md) · [Table of contents](../README.md) · [Chapter 27. AI Infrastructure and Inference Optimization](27-ai-infrastructure-and-inference-optimization.md) →
