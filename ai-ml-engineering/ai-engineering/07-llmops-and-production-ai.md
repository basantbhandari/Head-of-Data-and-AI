<!-- notion-id: 3d567418285480528485d7c0d8618659 -->
# LLMOps and Production AI

Interview Q&A on running LLM applications in production: serving, monitoring, guardrails, cost, deployment, routing, and reliability scenarios.

### Q1. How does Prompt Caching work?

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

### Q2. Prefill vs Decode

> 🔁 _See also [09-large-language-model.md → Q54](../ml-engineering/09-large-language-model.md#q54-prefill-vs-decode) (separate Notion notes on the same question)._

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

### Q3. Explain the AI product lifecycle from ideation to production.

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

### Q4. What is LLMOps, and how does it differ from traditional MLOps?

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

### Q5. How do you serve LLMs in production?

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

### Q6. What is model quantization?

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

### Q7. How do you monitor LLM applications in production?

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

### Q8. What is LLM observability?

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

### Q9. What are guardrails for LLMs, and how do you implement them?

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

#### Additional notes (merged from "AI Safety, Ethics, and Responsible AI": How do you implement input and output guardrails for AI systems?)

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

### Q10. How do you implement content filtering for AI outputs?

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

#### Additional notes (merged from "AI Safety, Ethics, and Responsible AI": How do you implement content safety filters for AI-generated content?)

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

### Q11. How do you estimate the cost of running an AI-powered feature in production?

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

```txt
Monthly Cost =
LLM Cost
+ Embedding/Reranking Cost
+ Compute
+ Storage
+ Network
+ Monitoring
+ Other Operational Costs
```

Then calculate:

```txt
Cost per request = Monthly Cost / Monthly Requests
```

and, where useful:

```txt
Cost per user = Monthly Cost / Monthly Active Users
```

##### Example

Suppose:

- 1M requests/month
- 2,000 input tokens/request
- 500 output tokens/request
- Average 2 LLM calls/request

You first calculate total input/output tokens across **all model calls**, apply the provider's pricing, then add embeddings, vector DB, compute, networking, and observability costs.

Finally, compare the result against the **business value per request/user**.

### Q12. How do you optimize LLM inference costs in production?

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

### Q13. How do you implement A/B testing for LLM systems?

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

### Q14. What is CI/CD for AI applications, and how does it differ from traditional CI/CD?

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

### Q15. How do you version and manage prompts in production?

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

### Q16. What is model versioning, and how do you handle model rollbacks?

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

### Q17. How do you implement rate limiting and throttling for LLM APIs?

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

### Q18. How do you handle model updates and migrations without downtime?

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

### Q19. What is the role of feature flags in AI deployments?

**Feature flags** allow you to enable, disable, or gradually roll out AI features or model versions **without redeploying the application**.

- **Gradual model rollout:** Send a small percentage of traffic to a new model before full deployment.
- **A/B testing:** Compare two models, prompts, or AI pipelines using real traffic.
- **Instant rollback:** Disable a problematic model or feature immediately.
- **Risk control:** Release new AI capabilities to selected users or tenants first.
- **Model routing:** Dynamically choose between models based on user, workload, cost, or availability.
- **Prompt testing:** Switch between different prompt versions without changing application code.
- **Emergency controls:** Quickly disable an AI feature if it produces unsafe, incorrect, or unexpectedly expensive outputs.
- **Experimentation:** Test different configurations such as model parameters, RAG strategies, or agent behaviors.

### Q20. How do you implement logging and tracing for LLM applications?

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

### Q21. How do you handle PII and sensitive data in LLM inputs and outputs?

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

#### Additional notes (merged from "AI Safety, Ethics, and Responsible AI": How do you handle PII in LLM inputs and outputs?)

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

### Q22. What is a gateway pattern for LLM API management?

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

### Q23. How does Token Streaming work?

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

### Q24. How do you implement streaming responses for real-time AI applications?

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

### Q25. How does vLLM work?

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

### Q26. How does SGLang work?

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

### Q27. What are the key SLAs and metrics for production AI systems (latency, throughput, availability)?

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

### Q28. Cloud vs. on-device Model Deployment for AI applications.

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

### Q29. How do you implement fallback strategies when the primary model is unavailable or rate-limited?

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

### Q30. How do you implement structured output from LLMs reliably in production?

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

### Q31. How do you handle long contexts efficiently in production (context compression, prefix caching)?

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

### Q32. What is semantic routing, and how do you implement it in a multi-model system?

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

### Q33. How do you manage secrets and API keys securely in LLM applications?

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

### Q34. Your LLM API has latency spikes during peak hours. How do you stabilize it?

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

### Q35. Your LLM costs are too high in production. How do you reduce costs without degrading quality?

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

### Q36. Your application is hitting LLM provider rate limits during peak hours. How do you handle it?

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

### Q37. Your application depends on one LLM provider. How do you switch providers without downtime?

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

### Q38. Your AI system handles 100 requests/sec but crashes at 5000. How do you scale for concurrent requests?

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

### Q39. A traffic spike brings down your AI system. How do you handle peak traffic?

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

### Q40. One LLM provider outage took down your entire system. How do you eliminate single points of failure?

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

### Q41. Your multi-LLM pipeline fails when one model in the chain breaks. How do you handle orchestration failure?

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

### Q42. Your AI pipeline has zero visibility into which step is failing. How do you add observability?

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

### Q43. You quantized your LLM, but accuracy dropped significantly. How do you minimize quantization loss?

- **Use a less aggressive quantization level:** If INT4 causes a large accuracy drop, try **INT8**, **W8A8**, or mixed precision instead of forcing everything to INT4.
- **Use calibration data:** Calibrate activation ranges using a representative dataset that matches real production inputs.
- **Apply per-channel or per-group quantization:** These usually preserve accuracy better than a single scale for the entire tensor.
- **Keep sensitive layers in higher precision:** Leave embeddings, attention projections, normalization layers, or output heads in FP16/BF16 when they are particularly sensitive.
- **Use GPTQ/AWQ or similar weight-only methods:** These can provide better accuracy than naive post-training quantization.
- **Use Quantization-Aware Training (QAT):** Simulate quantization during training so the model learns to compensate for quantization errors.
- **Use mixed precision:** Quantize less-sensitive layers more aggressively while keeping sensitive layers at FP16/BF16.
- **Evaluate layer-by-layer:** Identify which layers contribute most to the accuracy degradation and selectively increase their precision.
- **Validate on representative benchmarks:** Compare the quantized model against the original model on task accuracy, perplexity, reasoning, and production-specific evaluation sets.

### Q44. One failing AI component can take down your entire platform. How do you design graceful degradation?

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
