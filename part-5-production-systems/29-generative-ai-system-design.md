# Chapter 29. Generative AI System Design

_Part V — Production Systems and Architecture_

A framework for answering generative AI design questions, then end-to-end designs of LLM, RAG, agentic, analytics and generative-media products and platforms, plus cross-cutting concerns such as latency, caching, guardrails, failover and capacity planning.

This chapter contains **54 questions**.

## Contents

- **29.1** [What framework do you use to answer a generative AI system design interview question?](#291-what-framework-do-you-use-to-answer-a-generative-ai-system-design-interview-question)
- **29.2** [Design ChatGPT: Training to Serving (End to End)](#292-design-chatgpt-training-to-serving-end-to-end)
- **29.3** [Design a RAG System (Chat with Your Documents)](#293-design-a-rag-system-chat-with-your-documents)
- **29.4** [Design Memory for a Personal AI Assistant](#294-design-memory-for-a-personal-ai-assistant)
- **29.5** [Design a Deep Research Agent](#295-design-a-deep-research-agent)
- **29.6** [Design a Multi-Agent Customer Support System](#296-design-a-multi-agent-customer-support-system)
- **29.7** [Design an On-Device AI Assistant](#297-design-an-on-device-ai-assistant)
- **29.8** [Design a Multimodal Search System (Text, Image, Video)](#298-design-a-multimodal-search-system-text-image-video)
- **29.9** [Design an LLM Inference Platform (vLLM-as-a-Service)](#299-design-an-llm-inference-platform-vllm-as-a-service)
- **29.10** [Design an LLM Evaluation Platform](#2910-design-an-llm-evaluation-platform)
- **29.11** [Design a Text-to-Image Generation Service (Midjourney-like)](#2911-design-a-text-to-image-generation-service-midjourney-like)
- **29.12** [Design a Music Generation Service (Suno-like)](#2912-design-a-music-generation-service-suno-like)
- **29.13** [Design a Video Generation Service (Sora-like)](#2913-design-a-video-generation-service-sora-like)
- **29.14** [Design an AI coding agent.](#2914-design-an-ai-coding-agent)
- **29.15** [Design a code generation and review system.](#2915-design-a-code-generation-and-review-system)
- **29.16** [Design a content moderation system using AI.](#2916-design-a-content-moderation-system-using-ai)
- **29.17** [Design a real-time AI recommendation system.](#2917-design-a-real-time-ai-recommendation-system)
- **29.18** [Design an AI-powered email assistant.](#2918-design-an-ai-powered-email-assistant)
- **29.19** [Design a medical diagnosis assistant using AI.](#2919-design-a-medical-diagnosis-assistant-using-ai)
- **29.20** [Design a fraud detection system powered by LLMs.](#2920-design-a-fraud-detection-system-powered-by-llms)
- **29.21** [Design an AI-powered data extraction pipeline from unstructured documents.](#2921-design-an-ai-powered-data-extraction-pipeline-from-unstructured-documents)
- **29.22** [Design a personalized learning assistant.](#2922-design-a-personalized-learning-assistant)
- **29.23** [Design an AI system for automated code migration.](#2923-design-an-ai-system-for-automated-code-migration)
- **29.24** [Design an AI-powered legal document review system.](#2924-design-an-ai-powered-legal-document-review-system)
- **29.25** [Design a conversational AI system with memory across sessions.](#2925-design-a-conversational-ai-system-with-memory-across-sessions)
- **29.26** [How do you design for latency vs quality trade-offs in AI systems?](#2926-how-do-you-design-for-latency-vs-quality-trade-offs-in-ai-systems)
- **29.27** [How do you implement caching strategies for LLM applications?](#2927-how-do-you-implement-caching-strategies-for-llm-applications)
- **29.28** [How do you design rate limiting and cost management for AI APIs?](#2928-how-do-you-design-rate-limiting-and-cost-management-for-ai-apis)
- **29.29** [How do you handle failover and fallback strategies for AI systems?](#2929-how-do-you-handle-failover-and-fallback-strategies-for-ai-systems)
- **29.30** [How do you design an AI system for high availability and fault tolerance?](#2930-how-do-you-design-an-ai-system-for-high-availability-and-fault-tolerance)
- **29.31** [How do you design an AI system that gracefully degrades when the model is unavailable?](#2931-how-do-you-design-an-ai-system-that-gracefully-degrades-when-the-model-is-unavailable)
- **29.32** [What are the key considerations for multi-region deployment of AI systems?](#2932-what-are-the-key-considerations-for-multi-region-deployment-of-ai-systems)
- **29.33** [Design an AI-powered search engine for an e-commerce platform.](#2933-design-an-ai-powered-search-engine-for-an-e-commerce-platform)
- **29.34** [Design an AI gateway/proxy for managing LLM access across an organization.](#2934-design-an-ai-gatewayproxy-for-managing-llm-access-across-an-organization)
- **29.35** [How do you design a RAG system that handles conflicting information across sources?](#2935-how-do-you-design-a-rag-system-that-handles-conflicting-information-across-sources)
- **29.36** [How do you approach capacity planning for an AI system?](#2936-how-do-you-approach-capacity-planning-for-an-ai-system)
- **29.37** [Design a multi-tenant AI chatbot platform where each business gets a custom chatbot.](#2937-design-a-multi-tenant-ai-chatbot-platform-where-each-business-gets-a-custom-chatbot)
- **29.38** [Design an AI meeting summarizer system for thousands of meetings daily.](#2938-design-an-ai-meeting-summarizer-system-for-thousands-of-meetings-daily)
- **29.39** [Design an AI notification system that prioritizes instead of broadcasting.](#2939-design-an-ai-notification-system-that-prioritizes-instead-of-broadcasting)
- **29.40** [Design an AI-powered anomaly detection system for cloud infrastructure.](#2940-design-an-ai-powered-anomaly-detection-system-for-cloud-infrastructure)
- **29.41** [Design an AI-powered document processing pipeline for financial institutions.](#2941-design-an-ai-powered-document-processing-pipeline-for-financial-institutions)
- **29.42** [Design an AI dynamic pricing engine.](#2942-design-an-ai-dynamic-pricing-engine)
- **29.43** [Design an AI resume screening system that handles 100K applications per week.](#2943-design-an-ai-resume-screening-system-that-handles-100k-applications-per-week)
- **29.44** [Design an AI voice assistant architecture.](#2944-design-an-ai-voice-assistant-architecture)
- **29.45** [Design a multi-agent workflow system where agents collaborate on complex tasks.](#2945-design-a-multi-agent-workflow-system-where-agents-collaborate-on-complex-tasks)
- **29.46** [Design a real-time AI transcription system for concurrent audio streams.](#2946-design-a-real-time-ai-transcription-system-for-concurrent-audio-streams)
- **29.47** [Design an AI-powered live streaming content moderation system.](#2947-design-an-ai-powered-live-streaming-content-moderation-system)
- **29.48** [Design a conversational assistant (memory, context management, safety, latency).](#2948-design-a-conversational-assistant-memory-context-management-safety-latency)
- **29.49** [Design an enterprise RAG platform.](#2949-design-an-enterprise-rag-platform)
- **29.50** [Design a natural-language analytics assistant (text-to-SQL) over a company's data warehouse.](#2950-design-a-natural-language-analytics-assistant-text-to-sql-over-a-companys-data-warehouse)
- **29.51** [Design a guardrails service for LLM applications across an organisation.](#2951-design-a-guardrails-service-for-llm-applications-across-an-organisation)
- **29.52** [Design a self-serve fine-tuning platform for internal teams.](#2952-design-a-self-serve-fine-tuning-platform-for-internal-teams)
- **29.53** [Design an AI answer engine over the web (search plus generated answers with citations).](#2953-design-an-ai-answer-engine-over-the-web-search-plus-generated-answers-with-citations)
- **29.54** [Design a synthetic data generation pipeline for training and evaluating models.](#2954-design-a-synthetic-data-generation-pipeline-for-training-and-evaluating-models)

---

### 29.1 What framework do you use to answer a generative AI system design interview question?

GenAI system design interviews share the structure of classic ML design (see 28.1), but the hard parts are different: there is usually no model to train, quality is hard to measure, outputs are non-deterministic, and cost and latency are dominated by inference. A dedicated framework keeps these in view.

| Step | Time | What to cover |
| --- | --- | --- |
| 1. Clarify | 5 min | Users and use case, scale (requests per second, documents, users), latency targets (time to first token, total), quality bar, risk level, data sensitivity and residency, budget |
| 2. Decide the shape | 3 min | Single call, workflow, RAG, agent or fine-tuned model; justify the simplest shape that meets the requirement |
| 3. Data and context | 8 min | Sources, ingestion, parsing, chunking, indexing, permissions, freshness; what goes into the context window and how it is budgeted |
| 4. Models | 5 min | Hosted or self-hosted, model routing (small model by default, large on demand), fine-tuning only if justified |
| 5. Request flow | 8 min | Gateway, guardrails, orchestration, retrieval, tool calls, generation, streaming, caching |
| 6. Evaluation | 6 min | Golden sets, automatic metrics, LLM judges calibrated against humans, online feedback, regression testing in CI |
| 7. Safety and security | 4 min | Prompt injection, data leakage, access control, harmful content, human approval for actions |
| 8. Operations | 5 min | Cost per request and unit economics, latency budget per stage, scaling, fallbacks, observability, rollout |

**Habits that score well**

- **Put numbers on it:** "10,000 daily users, 5 questions each, 3,000 input tokens and 400 output tokens per question" lets you estimate cost and capacity.
- **Start with the simplest architecture** and add retrieval, tools or agents only when the requirement needs them.
- **Treat evaluation as a first-class component,** not an afterthought; explain how you would know the system works and how you would catch regressions.
- **Show the latency budget** per stage (guardrails, retrieval, reranking, generation) and how streaming and parallelism hide it.
- **Address failure modes:** hallucination, stale data, permission leaks, provider outages, cost spikes.
- **Plan for model change:** a gateway, versioned prompts and an evaluation suite make model upgrades routine.

**Common mistakes:** jumping to multi-agent architectures; ignoring permissions on retrieved data; no evaluation plan; no cost estimate; treating the LLM as deterministic.

### 29.2 Design ChatGPT: Training to Serving (End to End)

**Requirements and scale.** A general chat assistant with roughly 100M DAU, streaming responses (time to first token under 1s), multi-turn context, safety, and many model sizes.

```text
Data → Pretrain → SFT → Preference tuning (RLHF/DPO) → Eval gates → Registry
                                                                 ↓
Client → API gateway (auth, rate limit) → Orchestrator (context, tools, safety)
       → Router (model tier) → Inference fleet (vLLM/TensorRT-LLM) → stream back
```

#### Training
- **Data:** web crawl, code, books and licensed data. Steps are dedup (MinHash), quality classifiers, PII/toxicity filtering, and a tokenizer (BPE).
- **Pretraining:** a decoder-only transformer trained with 3D parallelism (data + tensor + pipeline, FSDP/ZeRO), bf16, checkpointing every N steps, and automatic restart on node failure.
- **Post-training:** SFT on curated dialogues, then RLHF (reward model + PPO) or DPO, then safety tuning and red-teaming.
- **Eval gates:** benchmarks (MMLU, HumanEval), internal task suites, safety evals and human side-by-side comparisons. A model is promoted only if it doesn't regress.

#### Serving
- **Throughput:** continuous batching, PagedAttention KV cache, prefix caching for system prompts, speculative decoding, FP8/INT8 quantization, and tensor parallelism for large models.
- **Routing:** cheap queries go to a small model and hard ones go to a frontier model.
- **Conversation state:** stored in a DB, with long histories truncated or summarized to fit the context window.
- **Safety:** input/output moderation classifiers and a streaming output filter.

#### Ops
- **Capacity:** autoscale GPU pools on queue depth and tokens/s. Run multiple regions.
- **Metrics:** TTFT, inter-token latency, cost per 1K tokens, refusal rate, thumbs-up rate.
- **Feedback loop:** user feedback goes into the next round of preference data.

**Main tradeoffs:** model size vs latency/cost, batch size vs TTFT, and safety strictness vs over-refusal.

### 29.3 Design a RAG System (Chat with Your Documents)

**Requirements.** Users upload PDFs, docs and wikis (from thousands up to millions of documents) and ask questions. Answers must be grounded, cite their sources, and respect per-user permissions.

```text
Ingest: connectors → parse (OCR/layout) → chunk → embed → vector DB + BM25 index (+ ACL metadata)
Query:  question → rewrite → hybrid retrieve (dense + BM25) → ACL filter → rerank → LLM w/ citations
```

#### Key choices
- **Chunking:** structure-aware, about 300–800 tokens with overlap, keeping heading paths as metadata. Parent-child chunking lets you retrieve small chunks but feed the larger parent.
- **Retrieval:** hybrid retrieval (BM25 catches exact terms and IDs, dense catches paraphrases) fused with RRF, then a cross-encoder reranker to pick the top 5–10. Use an HNSW index with metadata pre-filtering for tenant and ACL.
- **Query handling:** rewrite multi-turn queries into standalone ones. HyDE or multi-query retrieval improves recall.
- **Generation:** the prompt says "answer only from the context, cite [doc:page], say I don't know otherwise".
- **Freshness:** incremental re-indexing on change events, with the embedding-model version stored per vector.

#### Evaluation
- **Retrieval:** recall@k and MRR on a labelled question set.
- **Generation:** faithfulness and answer relevance, scored by LLM-as-judge (RAGAS-style), plus online thumbs-up/down and citation-click rate.

#### Failure modes
- **Bad parsing:** tables and scanned PDFs often break extraction.
- **Retrieval misses:** fix by tuning chunk size and using hybrid retrieval.
- **Stale index:** keep re-indexing on change events.
- **Prompt injection:** documents can carry injected instructions, so treat retrieved text as data.
- **Permission leaks:** apply ACLs at retrieval time, never only in the UI.

**Cost:** embed once and cache. Use a small LLM for rewriting and a large one only for the final answer.

### 29.4 Design Memory for a Personal AI Assistant

**Goal.** The assistant remembers user preferences, facts, and past tasks across sessions without bloating context or storing things it shouldn't.

#### Memory tiers
1. **Working memory:** the current conversation, kept in the context window. Older turns are summarized as a rolling summary.
2. **Episodic memory:** past sessions stored as summaries with embeddings and timestamps, retrieved by similarity plus recency.
3. **Semantic memory (profile):** distilled facts such as "vegetarian", "works in Berlin", "prefers concise answers", stored as structured key-value or triples with confidence and source.
4. **Procedural memory:** learned workflows or tool preferences.

```text
Turn → LLM response
     ↘ async memory writer: extract candidate facts → dedupe/merge/contradict-check → store
Next turn → memory retriever (profile + top-k episodic) → inject into prompt
```

#### Design decisions
- **What to write:** an extraction LLM decides what is worth saving, scoring importance. On conflict, newer information wins but history is kept (versioned facts).
- **Retrieval:** $\text{score} = \text{similarity} \times \text{recency decay} \times \text{importance}$. Cap the injected memory tokens.
- **User control:** users can view, edit and delete memories ("forget this"). Respect GDPR deletion, encrypt per user, and never store sensitive categories without consent.
- **Hygiene:** a periodic consolidation job merges duplicates and expires stale items.

#### Evaluation
- **Offline:** memory precision (is what's recalled relevant?) and recall on scripted multi-session tests.
- **Online:** the rate of corrections ("I already told you...").

**Risks:** wrong memories compounding, privacy leaks across users (keep a strict user_id partition), and prompt injection that writes false memories.

### 29.5 Design a Deep Research Agent

**Goal.** Given a question, the agent autonomously searches the web and internal sources for minutes, reads dozens of sources, and writes a cited report.

```text
Planner (decompose into sub-questions) → parallel Researcher agents
   each: search → fetch → extract → note (with source URL)
→ Reflection: gaps? contradictions? → more searches (bounded loop)
→ Writer: outline → sections → citation check → final report
```

#### Components
- **Planner LLM:** produces a research plan and a todo list kept as explicit state.
- **Tools:** web search API, page fetcher/reader (HTML to text, PDF parsing), internal RAG, and a code sandbox for analysis.
- **Researchers:** sub-agents run in parallel (fan-out). Each writes compressed notes so the orchestrator's context stays small.
- **Memory:** a scratchpad store of notes and sources, deduplicated by URL and content hash.
- **Writer:** a long-context model synthesises the report. A verifier pass checks that each claim maps to a cited source.

#### Production concerns
- **Budgets:** limits on tokens, tool calls and wall time, plus loop detection and a max depth.
- **Long-running jobs:** run the job asynchronously with progress streaming, and checkpoint state so it can resume.
- **Source quality:** domain reputation scoring, recency, and conflict resolution that surfaces disagreements.
- **Safety:** prompt injection from web pages is the big one. Sanitise fetched content and never let it trigger privileged tools.

#### Evaluation
- **Rubric grading:** coverage, accuracy, citation precision, and LLM-as-judge with human calibration.
- **Benchmarks:** suites in the style of BrowseComp or GAIA.

**Cost:** a cheap model for search and extraction, a strong model for planning and writing, and caching of fetched pages.

### 29.6 Design a Multi-Agent Customer Support System

**Requirements.** Handle chat and email tickets for millions of customers, resolve 50–70% automatically, escalate cleanly to humans, and take actions such as refunds and order lookups safely.

```text
Customer → Triage/Router agent (intent, language, sentiment, risk)
   ├─ FAQ/Knowledge agent (RAG over help center)
   ├─ Orders agent (tools: order API, shipping API)
   ├─ Billing agent (tools: refund API w/ limits)
   └─ Human handoff (summary + context to Zendesk)
Supervisor enforces policy, budgets, and final response QA
```

#### Design choices
- **Topology:** a supervisor/router pattern, where each specialist has a narrow prompt and a minimal toolset (least privilege). It is easier to evaluate and secure than one mega-agent.
- **Tool safety:** tools are typed functions with server-side validation. Refunds above a threshold need human approval. Every action is idempotent and logged.
- **Shared state:** ticket state lives in a session store (customer profile, order history, conversation summary).
- **Escalation triggers:** low confidence, angry sentiment, policy-sensitive topics, repeated failures, or the user asking for a human. The handoff carries a summary so the customer doesn't repeat themselves.
- **Guardrails:** PII redaction in logs and output checks for policy (no promises outside policy).

#### Evaluation
- **Offline:** replay historical tickets and compare against human resolutions, with LLM-as-judge on correctness and tone.
- **Online:** containment rate, CSAT, reopen rate, average handle time and escalation accuracy.

**Production concerns:** latency (stream responses, run tool calls in parallel), multilingual support, rollout by intent category, and A/B tests against human baselines.

### 29.7 Design an On-Device AI Assistant

**Requirements.** The assistant runs on a phone or laptop with private, offline-capable, low-latency features: summarise notifications, rewrite text, answer questions over personal data, and control apps.

#### Architecture
```text
On-device: small LLM (1–4B, 4-bit quantized) + NPU runtime (Core ML / ONNX / ExecuTorch / llama.cpp)
           + local vector index over personal data + intent/tool router
Cloud (optional): larger model via private compute for hard queries, with user consent
```

#### Key choices
- **Model:** distilled small model, quantized to INT4/INT8 (GPTQ/AWQ), LoRA adapters per feature hot-swapped on a shared base, and a KV-cache size tuned for RAM limits.
- **Hybrid routing:** an on-device classifier decides local vs cloud based on complexity, connectivity and battery. Fall back gracefully when offline.
- **Personal RAG:** embeddings are computed locally, incrementally, while the device is idle or charging.
- **Privacy:** data never leaves the device by default. Cloud requests are stateless and encrypted.

#### Constraints and tradeoffs
- **Memory:** 2–4 GB budget, so smaller models mean lower quality.
- **Thermal and battery:** limits on sustained throughput.
- **Latency:** first-token latency on the NPU vs the CPU.
- **Size:** model download size and updates (delta updates).

#### Evaluation
- **Device lab:** quality evals per feature against the cloud model, with latency and energy measured on a device matrix.
- **Telemetry:** privacy-preserving, opt-in, aggregated or using differential privacy.

**Risks:** fragmentation across hardware, and keeping models updated without large downloads.

### 29.8 Design a Multimodal Search System (Text, Image, Video)

**Requirements.** Search a catalogue of billions of images and videos with text, image or video queries in under 200 ms at p95.

```text
Ingest: media → (images) CLIP-style encoder; (video) shot detection → keyframes + ASR transcript + captions
       → embeddings + metadata → ANN index (sharded) + inverted index
Query:  text/image → shared embedding space → ANN top-K → hybrid merge with keyword → reranker → results
```

#### Key choices
- **Embeddings:** a joint embedding space (CLIP/SigLIP) makes text-to-image and image-to-image search work with one index. For video, embed keyframes and segments and aggregate to the video level, plus ASR text for spoken content.
- **Index:** ANN with HNSW or IVF-PQ for memory efficiency at billions of vectors, sharded by ID with replicas. Product quantization trades a little recall for 10–30x less memory.
- **Hybrid retrieval:** combine with BM25 over titles, captions and OCR text for exact matches.
- **Reranking:** a heavier cross-modal model or learning-to-rank on top-K using engagement features.
- **Filters:** safe search, licence and region filters, applied as pre-filters on metadata.

#### Evaluation
- **Offline:** recall@K and NDCG on labelled query sets.
- **Online:** CTR, zero-result rate and dwell time.

**Ops:** batch GPU embedding pipelines with backfill when the encoder version changes (dual-index during migration). Monitor embedding drift and keep a cache for popular queries.

### 29.9 Design an LLM Inference Platform (vLLM-as-a-Service)

**Requirements.** An internal platform serving many open models (7B–70B+, LoRA fine-tunes) to many teams through an OpenAI-compatible API, with SLOs on TTFT and throughput, multi-tenancy, and cost efficiency.

```text
Clients → Gateway (auth, quotas, token-bucket rate limits, routing)
        → Scheduler/Router (model, LoRA, prefix-cache affinity, load)
        → GPU node pools running vLLM (continuous batching, PagedAttention)
        ← streaming SSE
Control plane: model registry, deployment (K8s + GPU operator), autoscaler, metrics
```

#### Key techniques
- **vLLM core:** continuous batching and a PagedAttention KV cache give high utilisation. Prefix caching handles shared system prompts. Chunked prefill keeps long prompts from stalling decode.
- **LoRA:** multi-LoRA serving (S-LoRA style) runs many fine-tunes on one base model.
- **Parallelism:** tensor parallelism within a node for large models, pipeline parallelism across nodes. Quantization (FP8, AWQ) increases density.
- **Disaggregation:** separate prefill and decode pools at scale.
- **Routing:** KV/prefix-cache-aware routing to the same replica improves hit rate. Priority queues separate interactive from batch traffic.
- **Autoscaling:** scale on queue depth, KV-cache utilisation and tokens/s rather than CPU. Model weights load slowly, so keep warm pools and use fast weight loading (safetensors, local NVMe).

#### Multi-tenancy
- **Limits:** per-team quotas and token budgets.
- **Isolation:** separate pools for noisy neighbours.
- **Chargeback:** charge by GPU-seconds or tokens.

#### Observability
- **Metrics:** TTFT, TPOT, p99 latency, GPU utilisation, KV-cache evictions/preemptions, error rates.
- **Releases:** canary deployments for new model versions.

**Tradeoffs:** larger batches raise throughput but hurt latency, and dedicated pools give isolation at the cost of utilisation.

### 29.10 Design an LLM Evaluation Platform

**Goal.** Teams can evaluate prompts, models, RAG pipelines and agents offline and online, gate releases, and catch regressions.

#### Components
```text
Dataset registry (versioned golden sets, production samples, adversarial sets)
→ Eval runner (parallel, cached model calls, reproducible configs)
→ Scorers: exact/regex, programmatic (unit tests for code), embedding sim,
           LLM-as-judge (rubric, pairwise), human review queue
→ Results store + dashboards (per-slice metrics, diffs vs baseline)
→ CI integration: block PR/deploy on regression
Online: tracing (OpenTelemetry), sampling of prod traffic, user feedback, A/B tests
```

#### Design points
- **Versioning:** version everything (dataset, prompt, model, judge prompt, scorer) so results are reproducible.
- **LLM-as-judge:** calibrate against human labels (measure agreement or Cohen's kappa). Use pairwise comparison with position swapping to reduce bias, and a different model family as judge where possible.
- **Slicing:** break results down by intent, language and difficulty, since aggregate scores hide regressions.
- **Statistics:** confidence intervals and bootstrap so 1% noise isn't treated as a win.
- **Agents:** trajectory-level evals (task success, tool-call correctness, steps, cost).
- **Safety suites:** jailbreaks, PII leakage, toxicity.
- **Closing the loop:** production failures flagged by users or monitors are triaged into the golden set.

#### Ops
- **Cost and speed:** cache model responses keyed by (model, prompt, params) and rate-limit judge calls.
- **Access:** RBAC on sensitive datasets.

**Metric examples:** faithfulness, answer relevance, pass@k, win rate, latency, cost per query.

### 29.11 Design a Text-to-Image Generation Service (Midjourney-like)

**Requirements.** Millions of generations per day, about 10 s per job, variations and upscales, a community gallery, and safety.

```text
Client → API (auth, credits) → Prompt safety filter → Job queue (priority tiers)
      → GPU workers (diffusion model, e.g. latent diffusion/DiT) → output safety classifier
      → object storage + CDN → notify client (websocket)
```

#### Key choices
- **Model:** latent diffusion with a text encoder (CLIP/T5). Fewer sampling steps come from distillation (LCM, consistency or turbo models), and fp16 or quantization reduce cost. Upscaling runs as a separate model.
- **Async jobs:** a queue with priority (paid/fast vs relaxed), workers batch same-resolution requests, and progress previews are shown.
- **GPU efficiency:** keep models resident, use compiled graphs (TensorRT), autoscale on queue depth, and use spot instances for the relaxed tier.
- **Prompt enhancement:** an LLM rewrites short prompts.
- **Safety:** prompt classifiers (CSAM, violence, real-person deepfakes), output NSFW image classifiers, watermarking or C2PA provenance, and rate limits against abuse.
- **Storage:** images go to S3 with CDN delivery. Metadata (prompt, seed, model version) is stored so variations are reproducible.

#### Evaluation
- **Offline:** FID/CLIP score, human preference ratings and aesthetic models.
- **Online:** upscale/download rate as an implicit quality signal.

**Cost:** GPU-seconds per image dominate, so step reduction, batching and right-sized resolutions are the main levers. Credits enforce per-user budgets.

### 29.12 Design a Music Generation Service (Suno-like)

**Requirements.** Text prompt plus optional lyrics in, 2–4 minute song with vocals out, in about 30–60 s. Users can extend, remix and create stems.

#### Pipeline
```text
Prompt → LLM: lyrics generation/structuring (verse/chorus tags) + style tags
       → Music model: audio tokens via neural codec (EnCodec/DAC) generated by
         autoregressive transformer or diffusion conditioned on text/lyrics/style
       → Codec decoder → vocoder/upsampling → mastering → stem separation (optional)
       → storage + CDN streaming
```

#### Key choices
- **Representation:** discrete audio tokens from a neural codec (multi-codebook RVQ) let a transformer model long sequences. Generate a coarse version first, then refine.
- **Long-form coherence:** condition on song structure tokens and generate in segments with overlap/continuation.
- **Latency:** stream the first clip while the rest generates.
- **Serving:** GPU job queue similar to image generation, with batching by duration.

#### Product and safety
- **Copyright:** the critical issue. Train on licensed data, add similarity detection against copyrighted catalogues (audio fingerprinting) to block near-copies, and filter lyrics for copyrighted text and artist-name imitation.
- **Provenance:** watermark generated audio.
- **Moderation:** apply content moderation to lyrics.

#### Evaluation
- **Human preference:** MOS and pairwise comparisons.
- **Automatic:** FAD, text-audio alignment (CLAP score), and lyric intelligibility measured by ASR word error rate on vocals.

**Cost:** long audio sequences are expensive, so use distillation, step reduction and caching of popular styles.

### 29.13 Design a Video Generation Service (Sora-like)

**Requirements.** Text or image prompt in, 5–20 s video at 720p–1080p out. It is very compute heavy (minutes of GPU time per clip), so the system is async with queues and credits.

```text
Prompt → safety + prompt rewrite (LLM expands to detailed caption)
→ Scheduler (priority, cost estimate, quota) → multi-GPU job
→ Diffusion transformer over spacetime latent patches (video VAE compresses frames)
→ decode → super-resolution / frame interpolation → output safety → watermark/C2PA → CDN
```

#### Key choices
- **Model:** a DiT on spatiotemporal latent patches. Text conditioning via T5 or an LLM encoder. Train with variable resolutions and durations.
- **Inference:** a single job spans multiple GPUs (sequence/context parallelism over patches), with step distillation to cut sampling steps. Generate low-resolution first and upscale.
- **Scheduling:** gang scheduling for multi-GPU jobs, preemption for priority tiers, checkpointing long jobs, and cost estimation before acceptance.
- **Storage:** large outputs go to object storage with a CDN.

#### Safety
- **Inputs:** strict prompt filters (real people, minors, violence).
- **Outputs:** frame-level classifiers on the output.
- **Provenance:** invisible watermarking and C2PA metadata.
- **Likeness:** detection of public figures' likeness.

#### Evaluation
- **Human ratings:** motion quality, prompt adherence, temporal consistency.
- **Automatic:** FVD and VBench-style metrics.

**Tradeoffs:** quality vs duration and resolution vs cost. A draft/preview mode gives fast iteration before committing expensive renders.

### 29.14 Design an AI coding agent.

**Goal:** given a task ("fix this failing test", "add an endpoint", an issue description), the agent reads a real repository, plans, edits code, runs tests and opens a pull request for human review, in the style of Claude Code, Devin or SWE-agent. See also 21.51 for how coding agents work in general.

```text
Task ─► agent loop (LLM): plan ─► call tool ─► observe ─► repeat until tests pass or budget ends
Tools: repo navigation (ls, ripgrep, symbol index via tree-sitter or LSP, optional embeddings)
       read and edit files (search-and-replace or patch edits), shell in a sandbox (tests, linters,
       type checks, build), git, docs or web lookup, issue tracker
State: plan or todo list, scratchpad, context manager (summarise or clear old tool outputs)
Output: patch + test results ─► pull request with description ─► human review ─► CI ─► merge
```

**Key design points**

- **Context engineering:** repositories do not fit in context. Use agentic search (grep, file tree, "go to definition") plus a repo map of files and key signatures, read only relevant files, summarise long tool outputs, and compact the history on long tasks. Sub-agents (explorer, implementer, reviewer) keep each context clean on large tasks.
- **Edits:** structured search-and-replace or diff edits are more reliable and cheaper than rewriting whole files; edits should fail loudly when the target text is not found.
- **Verification is the key differentiator:** reproduce the bug with a failing test first, make the change, then run tests, linters and type checks. Execution feedback drives the loop; retry with the error output as context.
- **Sandbox:** ephemeral containers or VMs, no production credentials, restricted network egress, resource and time limits.
- **Human in the loop:** permission prompts for destructive or external commands (force push, delete, deploy), and pull request review before merge.
- **Budgets:** caps on steps, tokens and time, with stuck-loop detection.
- **Model routing:** a strong model for planning and editing; cheaper models for summarising and search.

**Evaluation**

- **Offline:** SWE-bench Verified style suites (resolved rate) and internal golden tasks from your own repositories.
- **Online:** pull request acceptance rate, revert rate, review comments per pull request, iterations and cost per task, regressions caught in CI.
- Trace every step to debug failure modes: wrong file edited, hallucinated test results, loops.

**Risks:** prompt injection through repository files, issues or web pages; secrets leakage; plausible but wrong code; tests modified to pass instead of fixing the code. Mitigate with sandboxing, untrusted-content handling, test integrity checks and review.

### 29.15 Design a code generation and review system.

**Scope.** IDE code completion plus automated PR review across a large engineering organisation.

#### Code generation (completion and chat)
- **Latency:** completions need a p50 under 300 ms, so use a small fill-in-the-middle (FIM) model for inline completion and a larger model for chat/edits.
- **Context:** the current file plus cursor prefix/suffix, open tabs, and retrieved snippets from the repo (symbol graph or embedding index), assembled into a token budget.
- **Serving:** speculative decoding, prefix caching, request cancellation on keystroke, and debouncing.
- **Metrics:** acceptance rate, characters retained after 30 s, latency.

#### Code review
```text
PR webhook → diff + related files + repo conventions → static analyzers (lint, SAST)
→ LLM reviewer (per-file + cross-file pass) → dedupe/rank comments by confidence
→ post inline comments; developers thumbs-up/thumbs-down → feedback dataset
```
- **Precision over recall:** noisy reviewers get ignored. Filter low-confidence or style nits and combine LLM output with deterministic tools.
- **Repo rules:** a CLAUDE.md, CODEOWNERS or style guide feeds the prompt.

#### Governance
- **Data handling:** code stays in the tenant (self-hosted or zero data retention) and secrets are filtered.
- **Licensing:** licence and copyleft snippet detection.
- **Measurement:** track impact on cycle time and defect rate through experiments, not anecdotes.

### 29.16 Design a content moderation system using AI.

**See also:** 28.5 designs harmful content detection with classical ML; this answer focuses on LLM-based moderation.

**Requirements.** A social platform with millions of text, image and video posts per day. Policies cover hate, harassment, CSAM, spam, self-harm and more. Decisions must be fast (pre-publish for high-risk content), accurate, appealable and auditable.

```text
Content → hash matching (PhotoDNA/PDQ for known CSAM/terror) → fast classifiers (per policy, multimodal)
        → risk score → auto-action (high confidence) | human review queue (uncertain) | allow
        → LLM policy reasoner for borderline cases (policy text in prompt)
Appeals → human review → labels → retraining
```

#### Key choices
- **Tiered cascade:** cheap models first and expensive ones (an LLM or vision-language model) only for uncertain cases, which controls cost and latency.
- **Per-policy thresholds:** tuned on precision/recall to match harm severity. CSAM gets high recall; borderline political speech gets high precision plus human review.
- **LLM-based moderation:** policy as a prompt means quick policy updates without retraining. It can also generate a rationale for the decision.
- **Context features:** user history, account age, virality, and conversation context (sarcasm, reclaimed slurs).
- **Human review:** queues prioritised by $\text{reach} \times \text{severity}$, reviewer wellness tooling (blurring), and inter-rater agreement tracking.

#### Monitoring
- **Metrics:** prevalence (sampled), precision/recall per policy and language, appeal overturn rate, time to action.
- **Adversarial drift:** leetspeak and image text. Keep active-learning loops running.

**Concerns:** bias across dialects and languages, transparency (notices), and regulatory reporting (DSA).

### 29.17 Design a real-time AI recommendation system.

**See also:** 28.2 and 28.13 cover recommendation system design in depth, and Chapter 32 covers recommender techniques; this answer focuses on the real-time and generative AI aspects.

**Requirements.** Feed or product recommendations for 100M users, about 100 ms end to end, reacting to in-session behaviour within seconds.

```text
Events (clicks, views) → Kafka → stream processing (Flink) → real-time features (online store: Redis)
                                                     ↘ offline lake → training
Request → Candidate generation (two-tower ANN, item-to-item, trending, followed)  ~1000s
        → Ranking (GBDT/DNN, multi-task: click, dwell, purchase) ~100s
        → Re-ranking (diversity, freshness, business rules, dedupe) → top 20
```

#### Key choices
- **Two-tower retrieval:** user and item embeddings in a vector index (HNSW/ScaNN). The user embedding is computed at request time from recent session events for real-time freshness.
- **Ranking:** a feature store guarantees training/serving consistency (point-in-time joins). A multi-task model blends objectives.
- **LLM role (optional):** content understanding (item embeddings, tags), cold-start descriptions, and generating explanations, but rarely in the hot ranking path because of latency.
- **Exploration:** bandits or epsilon-exploration for new items, against feedback loops and popularity bias.
- **Caching:** precompute candidates for heavy users and fall back to popular items on timeouts.

#### Evaluation
- **Offline:** NDCG, recall@K.
- **Online:** A/B tests on engagement, retention and revenue, plus guardrail metrics (diversity, creator fairness).

**Ops:** daily retraining plus online fine-tuning, monitoring of feature freshness and drift, and latency budgets per stage.

### 29.18 Design an AI-powered email assistant.

**Features.** Triage and prioritise, summarise threads, draft replies in the user's tone, extract action items and meetings, and search the inbox with natural language.

```text
Mail provider (Gmail/Graph API, webhooks) → ingestion → per-user encrypted store + vector index
→ classifiers (priority, category, spam/phishing) → UI labels
User action (summarize/draft) → context builder (thread + relevant past emails + calendar + style profile)
→ LLM → draft (never auto-send by default) → user edits → feedback
```

#### Key choices
- **Personalisation:** a style profile learned from sent emails (few-shot examples retrieved by similarity) and contact importance from interaction history.
- **Triage:** a small, cheap classifier runs on every email. The LLM runs only on demand or on important threads, for cost reasons.
- **Agentic actions:** scheduling a meeting, for example, is calendar tool use that needs user confirmation.
- **NL search:** hybrid retrieval over the inbox with filters parsed from the query ("from Sarah last month about invoice").

#### Security and privacy
- **Least privilege:** OAuth with minimal scopes and per-user encryption.
- **Data use:** no training on user mail without consent, plus data residency controls.
- **Prompt injection via email:** the critical risk, because emails are attacker-controlled input. Never let email content trigger sends, forwards or data exfiltration without explicit user confirmation, and isolate tool permissions.

#### Metrics
- **Usage:** draft acceptance and edit distance, time saved.
- **Quality:** triage precision.

### 29.19 Design a medical diagnosis assistant using AI.

**Framing.** This is clinical decision *support* for clinicians, not autonomous diagnosis. It is regulated (possibly FDA SaMD or EU MDR), safety-critical, and handles PHI (HIPAA/GDPR).

```text
EHR (FHIR API): history, labs, meds, notes, imaging → de-identification where possible
→ structured feature extraction (NLP on notes) + imaging models
→ Differential diagnosis engine: LLM grounded via RAG on guidelines/literature (UpToDate, PubMed)
  + validated specialist models (e.g., radiology classifier)
→ Output: ranked differentials with evidence + citations + uncertainty + suggested tests
→ Clinician reviews, accepts/rejects → feedback, audit log
```

#### Design principles
- **Grounding:** every suggestion cites guideline or literature sources and patient data points. No uncited claims.
- **Calibrated uncertainty:** abstain or flag low confidence, and always surface red-flag conditions (high recall for dangerous diagnoses).
- **Human in the loop:** the clinician makes the final decision, and the UI avoids automation bias by showing evidence rather than one answer.
- **Model choices:** domain-tuned LLMs plus specialised validated models. Version-lock models for regulatory traceability.
- **Security:** HIPAA-compliant hosting with a BAA, encryption, access controls and full audit trails.

#### Validation
- **Retrospective:** test on historical cases (top-k accuracy vs final diagnosis), sliced by demographics for bias.
- **Prospective:** silent-mode trials before live use, then clinical studies.
- **Post-market:** drift monitoring as populations, coding practices and guidelines change.

**Risks:** hallucination, bias, liability and data leakage. Mitigations are grounding, guardrails, scope limits and governance boards.

### 29.20 Design a fraud detection system powered by LLMs.

**See also:** 28.14 designs a classical fraud detection system; this answer focuses on where LLMs add value.

**Key point.** LLMs should not replace low-latency tabular fraud models (GBDTs on transaction features are better and run in under 50 ms). LLMs add value around them.

```text
Transaction → real-time features (velocity, device, geo, graph) → GBDT/DNN risk score (<50ms)
   → rules engine → approve / decline / step-up auth / review
Review queue → LLM investigator copilot: summarizes case, pulls related accounts/graph,
   explains signals, drafts SAR narrative → analyst decides
Unstructured signals → LLM extraction: chat/email scams, merchant descriptions, documents (KYC forgery cues)
```

#### Where LLMs help
- **Analyst copilot:** case summarisation and explanation cuts investigation time.
- **Unstructured fraud:** detecting scam or social-engineering patterns in messages, merchant onboarding docs, and claim narratives (insurance).
- **Feature generation:** LLM embeddings of text fields feed the tabular model.
- **Synthetic data:** generating rare fraud scenarios for testing rules.
- **Rule authoring:** natural language to rule drafts, which humans review.

#### Production concerns
- **Constraints:** latency and cost keep the LLM off the authorisation hot path or make it asynchronous.
- **Explainability for regulators:** SHAP on the core model. LLM explanations must be grounded in the actual features.
- **Adversarial adaptation:** continuous retraining and graph-based fraud ring detection (GNNs).
- **Labels:** chargebacks arrive weeks later, so handle delayed labels.
- **Metrics:** fraud loss, false-positive rate (customer friction), analyst throughput.
- **Privacy:** PII handling.

### 29.21 Design an AI-powered data extraction pipeline from unstructured documents.

**Goal.** Extract structured fields (invoices, contracts, forms) from PDFs, scans and emails at scale into a typed schema with high accuracy.

```text
Sources (email, S3, upload) → queue → classify doc type → parse
  (native PDF text | OCR + layout model for scans, tables) → chunk/page selection
→ LLM/VLM extraction with JSON schema (structured outputs / function calling)
→ validation (types, regex, cross-field rules e.g. sum(line items)=total, master data lookup)
→ confidence scoring → auto-accept | human-in-the-loop review UI → corrections → eval/fine-tune set
→ warehouse/ERP
```

#### Key choices
- **Constrained output:** schema-enforced JSON (structured outputs, JSON mode, constrained decoding) plus Pydantic validation, with retry on failure.
- **Vision-language models:** send page images directly for complex layouts. Use OCR text plus layout for cost at scale.
- **Grounding:** ask for the source span or page and bounding box per field, which enables review UI highlighting and catches hallucinations.
- **Confidence:** combine validation results, logprobs, and self-consistency (two models or two runs agree) to route items to humans.
- **Cost:** a small model for classification and simple docs, a large one only for hard cases. Batch APIs work for offline volume.

#### Evaluation
- **Accuracy:** field-level precision/recall per doc type on a labelled set.
- **Operations:** straight-through-processing rate and human correction rate.

**Ops:** idempotent processing, a dead-letter queue, PII handling, audit trail of extraction version, and drift monitoring as new templates appear.

### 29.22 Design a personalized learning assistant.

**Goal.** An AI tutor that adapts to each learner's level, explains concepts, generates practice, and tracks mastery.

```text
Learner model (knowledge state per skill: Bayesian Knowledge Tracing / IRT / DKT)
Content: curriculum graph (skills + prerequisites) + vetted content index (RAG)
Session: Tutor LLM (Socratic prompting, grounded in curriculum) ↔ learner
→ exercise generator (LLM + validation/auto-grader) → answer evaluation
→ update mastery → next-best-activity policy (spaced repetition, difficulty targeting ~70–85% success)
```

#### Key design points
- **Pedagogy:** the tutor guides rather than giving answers (scaffolded hints), adapts explanations (analogies, level), and checks understanding.
- **Grounding:** RAG on approved material avoids hallucinated facts. Math and code answers are checked by tools (CAS, code execution) rather than trusting the LLM.
- **Generated practice:** validate questions automatically (solve them, check for ambiguity) and calibrate difficulty with IRT from response data.
- **Personalisation memory:** goals, misconceptions and preferred style.
- **Safety:** age-appropriate content, guardrails, COPPA/FERPA for minors, and teacher dashboards and oversight.

#### Evaluation
- **Learning outcomes:** pre/post test gains and retention, ideally from A/B tests, rather than engagement alone.
- **Tutor quality:** hint helpfulness and factual accuracy, using LLM-as-judge with an expert rubric.

**Cost:** cache explanations for common concepts and use small models for grading routine items.

### 29.23 Design an AI system for automated code migration.

**Examples.** Python 2→3, Java 8→17, framework upgrades (AngularJS→React), or moving a monolith's SQL dialect from Oracle to Postgres, across thousands of repos or millions of lines.

```text
Inventory & analysis (dependency graph, AST, call graph) → plan: order by dependency leaves first
→ for each unit: deterministic codemods first (OpenRewrite, jscodeshift, libcst)
  → LLM agent for remaining cases (context: file + deps + migration guide + examples)
→ build + tests + static checks in sandbox → agent iterates on failures
→ diff review (auto-approve low-risk patterns, human review others) → PR batches → rollout
```

#### Key choices
- **Hybrid approach:** rules and codemods are cheap and deterministic for the common 80%. The LLM handles the long tail and idiomatic rewrites.
- **Verification is everything:** existing tests, and generating characterisation tests *before* migration to capture behaviour. Differential testing (old vs new on the same inputs), type checkers, and shadow traffic for services.
- **Learned patterns:** successful fixes become few-shot examples or new codemod rules (a flywheel).
- **Scale orchestration:** a job queue over files or repos, with progress dashboards, retry and idempotency.

#### Metrics
- **Effectiveness:** % compiled, % tests passing, human acceptance rate, cost per kLOC, and post-merge incident or revert rate.

**Risks:** silent semantic changes (numeric precision, encoding, time zones) and low test coverage. Mitigate with risk scoring and stricter review for low-coverage areas.

### 29.24 Design an AI-powered legal document review system.

**Use cases.** Contract review (clause extraction, deviation from playbook, risk flags), due diligence over thousands of documents, and e-discovery relevance and privilege review.

```text
Ingest (DMS, data rooms) → OCR/layout → clause segmentation → clause classifier/embedding
→ Playbook comparison: LLM checks each clause vs firm's standard positions (RAG over playbooks/precedents)
→ outputs: extracted terms (parties, dates, liability caps, governing law), risk flags w/ citations,
  suggested redlines → lawyer review UI (accept/reject) → feedback loop
```

#### Key design points
- **Grounded outputs:** every flag links to the exact clause text. Prefer extractive answers. Never invent case law: citations must be verified against a legal database.
- **Precision/recall by use case:** high recall for e-discovery and privilege (missing a privileged document is costly), and calibrated confidence with human review for anything uncertain.
- **Long documents:** hierarchical processing (clause → section → document), with cross-references resolved (definitions sections).
- **Security:** client confidentiality, matter-level access controls (ethical walls), no training on client data, and private deployment or zero-retention APIs. Audit logs.
- **Evaluation:** a lawyer-labelled gold set, clause-level F1 and reviewer agreement. Measure time saved and error rates against fully manual review.

**Risks:** hallucinated legal reasoning, jurisdictional nuance and liability, mitigated by keeping lawyers as the decision makers and positioning the tool as triage and acceleration.

### 29.25 Design a conversational AI system with memory across sessions.

See also 29.4.

**Focus.** A multi-user product (consumer chatbot or enterprise assistant) where conversations persist and later sessions benefit from earlier ones.

```text
Client → Session service (session_id, user_id, tenant_id) → Conversation store (Postgres/DynamoDB: messages)
→ Context assembler: system prompt + user profile memory + retrieved past-session snippets
                     + rolling summary of current session + recent turns (token budget)
→ LLM → response stream
Async: session-end job → summarize session, extract durable facts → memory store (vector + KV)
```

#### Design points
- **Data model:** users → sessions → messages. Memories link to source session IDs for traceability and deletion.
- **Context budget:** a fixed allocation (for example 10% profile, 20% retrieved memory, rest recent turns). Summarise rather than truncating blindly.
- **Retrieval:** hybrid similarity plus recency, scoped strictly by user_id/tenant_id, enforced in the query layer, not in the prompt.
- **Consistency:** handle concurrent sessions (two devices) with optimistic locking or append-only events.
- **Controls:** a memory on/off toggle, visible memories, deletion that cascades to derived summaries and embeddings, and retention policies.
- **Scale:** hot recent sessions in cache, cold history in cheaper storage, with prompt/prefix caching for stable system prompt plus profile.

#### Evaluation
- **Offline:** multi-session test scripts (does it recall preferences? does it avoid surfacing irrelevant old details?).
- **Online:** a "repeat yourself" metric, satisfaction and retention.

### 29.26 How do you design for latency vs quality trade-offs in AI systems?

Start from the product SLO. Inline autocomplete needs under 300 ms, chat needs a TTFT under 1 s, and batch analysis can take minutes. Then pick levers per stage.

- **Model routing/cascades:** a small fast model handles most requests. Escalate to a larger model on low confidence, complexity classification or user request. This often gives 80% of the quality at 20% of the latency and cost.
- **Streaming:** perceived latency depends on TTFT, not total time.
- **Inference optimisation:** quantization (FP8/INT4), speculative decoding, continuous batching, prefix/KV caching, and smaller context (better retrieval means fewer tokens).
- **Parallelism:** run retrieval, tool calls and guardrails concurrently, and speculatively start generation while slower checks finish.
- **Caching:** exact and semantic caches for repeated queries, and precomputing for predictable requests.
- **Pipeline depth:** every agent step adds latency. Reduce the reranker's top-k, skip query rewriting for simple queries, and cap agent iterations.
- **Asynchronous UX:** for high-quality slow paths (deep research), make it a background job with progress updates.

Measure the Pareto frontier: plot quality (eval score) against p95 latency and cost for each configuration, and choose by segment (paid vs free, interactive vs batch). Always A/B test, because users often prefer fast and good enough over slow and perfect.

### 29.27 How do you implement caching strategies for LLM applications?

There are several layers, from cheapest to most complex.

1. **Exact-match response cache:** key = hash(model, version, normalised prompt, params) and value = response, in Redis with a TTL. Works for deterministic (temperature 0) and repeated queries such as FAQs.
2. **Semantic cache:** embed the query and return a cached answer if cosine similarity is above a threshold (for example 0.95). It has a big hit rate on support bots, but risks wrong answers for similar-but-different questions ("cancel order" vs "cancel subscription"). Tune thresholds per domain, add metadata constraints (tenant, user context), and validate with evals.
3. **Prompt/prefix caching (provider or server side):** reuse the KV cache for shared prefixes (system prompts, few-shot examples, long docs). Put static content first and variable content last. It cuts TTFT and cost a lot (Anthropic and OpenAI prompt caching, vLLM automatic prefix caching).
4. **Intermediate caches:** embeddings (by content hash), retrieval results, tool/API results, and parsed documents.
5. **KV cache within a request:** handled by the inference engine (PagedAttention).

#### Production concerns
- **Invalidation:** on model or prompt version change and source document updates (include versions in keys).
- **Personalisation and privacy:** never share cached personalised responses across users or tenants (scope keys).
- **Monitoring:** hit rate, staleness complaints, and cost savings.

### 29.28 How do you design rate limiting and cost management for AI APIs?

#### Rate limiting
- **Units:** limit on tokens, not just requests. Use RPM *and* TPM (input + output tokens), since one request can cost 100x another.
- **Algorithm:** token bucket (allows bursts) or sliding window, enforced at the gateway with distributed counters (Redis with Lua or atomic increments). Estimate output tokens up front (using max_tokens) and reconcile after completion.
- **Hierarchy:** organisation → team → user → API key, with per-model limits (frontier models are scarcer).
- **Priority classes:** interactive over batch. Shed load by priority under pressure and return 429 with Retry-After. Clients retry with exponential backoff and jitter.
- **Provider awareness:** track upstream provider limits and spread load across keys, regions and providers.

#### Cost management
- **Attribution:** tag every call (team, feature, user) and log tokens and cost, giving chargeback dashboards.
- **Budgets:** soft alerts at 80% and hard caps. Anomaly detection catches runaway agents or loops.
- **Optimisation levers:** model routing (small model by default), prompt caching, response caching, trimming context/RAG chunks, max_tokens limits, batch APIs (about 50% cheaper) for offline work, and distilling or fine-tuning small models for high-volume tasks.
- **Unit economics:** track cost per resolved ticket or per active user, not just total spend.

### 29.29 How do you handle failover and fallback strategies for AI systems?

#### Failure types
Provider outages, 429 rate limits, timeouts and slow tails, model errors or invalid output, and quality failures such as hallucination or refusal.

#### Strategies
- **Retries:** exponential backoff with jitter for transient errors (429, 5xx), capped at a few attempts within the latency budget. Use idempotency keys for tool actions.
- **Timeouts and hedging:** set per-stage timeouts. For tail latency, send a hedged request to a second replica or provider after the p95 time.
- **Circuit breakers:** when an endpoint's error rate passes a threshold, open the circuit and route elsewhere, then probe half-open to recover.
- **Model fallback chain:** primary model → same model in another region → alternative provider (Bedrock, Vertex, Azure) → smaller self-hosted model. An AI gateway (LiteLLM-style) keeps a unified API.
- **Prompt portability:** fallback models behave differently, so maintain and evaluate prompts per model and keep eval suites running on fallbacks too.
- **Output validation fallback:** schema validation fails → retry with a repair prompt → fall back to a deterministic path.
- **Graceful degradation:** cached answers, search results without generation, or "try again later" with queued processing (see 29.31).

#### Ops
- **Synthetic probes:** health checks per provider and region.
- **Chaos testing:** test the failover paths themselves.
- **Visibility:** dashboards for fallback rate, because silent fallback to a weaker model is a quality incident.

### 29.30 How do you design an AI system for high availability and fault tolerance?

**Targets.** For example 99.9% to 99.95% availability, with an RTO and RPO defined per component.

#### Principles
- **Redundancy at every layer:** stateless API and orchestrator services behind load balancers across multiple AZs, GPU inference replicas spread across AZs (N+1 or N+2 capacity headroom), and replicated data stores (vector DB replicas, Postgres with failover, Redis cluster).
- **Multi-provider/model redundancy:** for hosted LLMs, an AI gateway with fallback providers and circuit breakers (see 29.29).
- **Decoupling:** async queues (Kafka/SQS) for non-interactive work, so spikes and outages buffer instead of failing. Use dead-letter queues and idempotent consumers.
- **Bulkheads:** separate pools for different tenants and features so one runaway workload can't take down everything.
- **Load shedding and backpressure:** admission control based on queue depth and KV-cache usage, with priority to critical traffic.
- **GPU realities:** nodes fail more often and are scarce. Use health checks (ECC errors, NCCL hangs), automatic cordoning, warm standby capacity, and fast model loading (local cache of weights).
- **Deployment safety:** canary and blue-green releases for models and prompts, automatic rollback on SLO breach, and versioned model registry.
- **State:** keep conversation and agent state in durable stores, and checkpoint long agent jobs so they resume after crashes.

#### Observability
- **SLOs:** on availability, TTFT and error rate, with burn-rate alerts.
- **Tracing and drills:** distributed tracing across retrieval, LLM and tools, plus regular game days.

### 29.31 How do you design an AI system that gracefully degrades when the model is unavailable?

Define degradation tiers ahead of time, from full to minimal functionality, and switch automatically using health signals and circuit breakers.

1. **Primary:** the best model with the full pipeline.
2. **Fallback model:** a cheaper or self-hosted model with the same API, possibly with simpler prompts. Shorter context and fewer agent steps.
3. **Cached or precomputed results:** semantic cache hits, precomputed summaries and recommendations, last-known-good outputs.
4. **Non-generative path:** classic search or retrieval results ("here are the most relevant docs"), rule-based or template responses, and traditional ML models (for example a GBDT classifier instead of LLM classification).
5. **Async promise:** "We'll email you the result" by queueing the request for later processing.
6. **Transparent failure:** a clear message, with the core product still working without the AI feature (feature flag off).

#### Design requirements
- **Optional AI:** AI features must not be on the critical path of core flows. Checkout must work if the recommendation LLM is down.
- **Kill switches:** feature flags per AI feature.
- **Timeouts:** so a slow model doesn't hang the UI.
- **UX:** degraded modes should be signalled honestly ("limited mode").
- **Testing:** chaos tests that disable model endpoints, and monitoring of time spent in each tier.

### 29.32 What are the key considerations for multi-region deployment of AI systems?

- **Data residency and compliance:** GDPR and sector rules may require EU data, including prompts, logs, embeddings and vector indexes, to stay in-region. Route by user or tenant home region and check the provider's regional endpoints and data-processing terms.
- **GPU capacity:** H100s and similar are unevenly available across regions. Plan reservations per region and accept that some regions may serve smaller models or fall back to others (if residency allows).
- **Latency:** GeoDNS or anycast routes to the nearest healthy region. TTFT includes network RTT, so streaming helps.
- **Model and version consistency:** deploy the same model, prompt and guardrail versions everywhere through a central registry and CI/CD, with region-by-region canary rollouts.
- **State replication:** conversation history and user memory need either a home-region model (simpler, compliant) or multi-region replication (conflict handling). Vector indexes need re-indexing or replication pipelines per region.
- **Failover:** active-active or active-passive. Cross-region failover must respect residency, so define which tenants can fail over where.
- **Cost:** cross-region egress, duplicate GPU fleets, and duplicate indexes. Use regional autoscaling and follow-the-sun batch scheduling to use idle capacity.
- **Observability:** per-region SLOs, a global view, and per-region quotas with upstream providers.

### 29.33 Design an AI-powered search engine for an e-commerce platform.

**Requirements.** 10M+ SKUs, natural-language queries ("waterproof hiking boots under \$150 for wide feet"), latency under 200 ms, and the goal of maximising conversion.

```text
Query → query understanding (spell-correct, NER: brand/size/price, category classifier,
        LLM rewrite for long-tail — cached) → structured filters + rewritten text
→ Retrieval: BM25 (Elasticsearch/OpenSearch) + dense two-tower embeddings (ANN) → merge (~1000)
→ Ranking: learning-to-rank (LambdaMART/DNN) with features: relevance, CTR, conversion, price,
           inventory, personalization, freshness → top 50
→ Business re-rank (sponsored, diversity, out-of-stock demotion) → results + facets
```

#### Key choices
- **Hybrid retrieval:** keyword search is essential for SKUs, brands and model numbers. Dense embeddings handle semantic and long-tail queries. Fine-tune embeddings on query→purchase pairs.
- **LLM usage:** offline enrichment of the catalogue (attribute extraction, normalised categories, generated synonyms), plus query rewriting for rare queries, cached because the head is precomputed. Optionally a conversational shopping assistant (RAG over the catalogue) as a separate experience.
- **Personalisation:** user embeddings and recent session behaviour as ranking features.
- **Cold start:** content features for new products.

#### Evaluation
- **Offline:** NDCG on human-judged relevance plus click logs, corrected for position bias.
- **Online:** A/B tests on conversion, revenue per search, zero-result rate and search exits.

**Ops:** near-real-time index updates for price and stock, caching of head queries, and monitoring of query-category drift (seasonality).

### 29.34 Design an AI gateway/proxy for managing LLM access across an organization.

**Goal.** A single control point for every team's LLM usage: one API, many providers and models, with governance, cost control and observability.

```text
Apps/SDKs → AI Gateway (OpenAI-compatible API)
  ├─ AuthN/Z (SSO, API keys per team/app, RBAC per model)
  ├─ Policy: allowed models per data classification, PII redaction/DLP, prompt-injection & content guardrails
  ├─ Rate limits & budgets (token buckets per team; hard/soft caps)
  ├─ Routing: model aliases ("default-chat" → provider X), load balancing, fallbacks, circuit breakers
  ├─ Caching: exact + semantic, prompt-cache aware
  └─ Logging/metering → cost attribution, traces, eval sampling
→ Providers (OpenAI, Anthropic, Bedrock, Vertex, Azure) + self-hosted vLLM fleet
```

#### Design points
- **Abstraction:** model aliases let the platform team swap providers or versions without app changes. Normalise streaming, tool calling and errors across providers.
- **Security:** central secret management, so apps never hold provider keys. Zero-retention agreements. Route data by classification (for example PII only to approved or self-hosted models). Audit logs with configurable prompt logging for privacy.
- **Performance:** the gateway sits on the hot path, so keep added latency under 10 ms, stream pass-through, keep it horizontally scalable and stateless, and use Redis for counters.
- **Self-service:** onboarding portal, usage dashboards, budget alerts.
- **Build vs buy:** LiteLLM, Portkey, Kong AI Gateway or cloud-native gateways, depending on control needs.

**Risks:** single point of failure (deploy it highly available, multi-AZ), and lowest-common-denominator APIs hiding provider-specific features (allow passthrough).

### 29.35 How do you design a RAG system that handles conflicting information across sources?

Conflicts come from stale versions, different scopes (regional policies), different authorities, or genuine disagreement.

#### At ingestion
- **Metadata:** capture source, authority tier (official policy > wiki > Slack), effective date, version, region or product scope, and owner.
- **Deduplication and versioning:** keep only the latest version active, or mark superseded docs. Many "conflicts" are really staleness.

#### At retrieval
- **Filtering:** by applicable scope (user region, product version) before ranking.
- **Ranking:** $\text{relevance} \times \text{authority} \times \text{recency}$ boosts.
- **Diversity:** retrieve with diversity (MMR) so conflicting viewpoints are visible rather than accidentally hidden.

#### At generation
- **Conflict prompting:** instruct the model to detect conflicts explicitly. Prefer authoritative and recent sources, and when conflict remains, present both with citations and dates ("Policy doc (2025) says X; older FAQ (2023) says Y").
- **Conflict-detection step:** optionally run a pairwise claim comparison (NLI or LLM) over the top chunks and flag contradictions.
- **Abstention:** for high-stakes domains, escalate rather than guess.

#### Feedback loop
- **Content governance:** log detected conflicts to a queue for content owners. Fixing the source is the real fix.

#### Evaluation
- **Test sets:** planted contradictions, measuring whether the system picks the correct authoritative answer or surfaces the conflict.

### 29.36 How do you approach capacity planning for an AI system?

#### 1. Model the demand
- **Usage profile:** $\text{DAU} \times \text{requests per user} \times \text{peak factor}$ (peak-to-average is often 2–4x). Measure input and output tokens per request, which are the real units.
- **Example:** $10^{6}\ \text{DAU} \times 10\ \text{req} = 10^{7}$ req/day. Average is about $10^{7} / 86{,}400 \approx 116$ req/s and peak about 400 req/s. At 1,500 input and 300 output tokens, peak is about $400 \times 1{,}500 = 600\text{K}$ input tokens/s and $400 \times 300 = 120\text{K}$ output tokens/s.

#### 2. Benchmark supply
- **Load test:** test the actual model, hardware and engine (for example 70B FP8 on H100 with vLLM) to find sustainable tokens/s per GPU at your SLO (p95 TTFT and TPOT), not maximum throughput. Prefill and decode have different costs, and long contexts consume KV cache, which limits concurrency.

#### 3. Compute the fleet
- **Fleet size:** $\text{GPUs} = \frac{\text{peak tokens/s}}{\text{tokens/s per GPU at SLO}}$, plus headroom (20–30%), N+1 per AZ, and capacity for failover regions.
- **Other components:** include embedding models, rerankers, vector DB memory ($\text{vectors} \times \text{dims} \times \text{bytes}$ plus index overhead), and the gateway.

#### 4. Optimise and decide
- **Caching:** prefix and response caching lower the effective load.
- **Mix:** model routing (share of traffic on small models), batch vs interactive split (batch fills off-peak), reserved vs on-demand vs spot, and self-hosting vs API (break-even at sustained utilisation).

#### 5. Iterate
- **Forecasting:** tie it to product roadmap and growth. Monitor utilisation, queue depth and SLO burn, and re-plan quarterly given GPU lead times.

### 29.37 Design a multi-tenant AI chatbot platform where each business gets a custom chatbot.

**Requirements.** Thousands of businesses each upload their knowledge (docs, website, FAQs), configure persona and tools, and embed the bot on their site or WhatsApp. Tenant isolation is critical.

```text
Tenant admin portal → config store (persona, prompts, allowed tools, branding, model tier)
Ingestion workers → per-tenant crawl/parse/chunk/embed → vector DB (namespace/partition per tenant)
Runtime: widget/API → gateway (tenant auth, rate limits, quotas) → orchestrator
  → load tenant config → RAG (tenant-scoped filter) → shared LLM pool → guardrails → response
Analytics: per-tenant conversations, deflection, unanswered questions → content gap reports
```

#### Key design points
- **Isolation:** tenant_id on every row, vector and log. Vector DB namespaces or partitions (or a separate index for large or enterprise tenants), enforced server side. Never rely on the prompt for isolation. Per-tenant encryption keys for enterprise customers.
- **Shared compute:** a shared LLM fleet with per-tenant quotas and fair scheduling against noisy neighbours. Per-tenant LoRA adapters only if needed, since prompt plus RAG customisation is usually enough.
- **Configuration as data:** prompt templates with tenant variables, versioned so tenants can preview and roll back.
- **Tools:** tenants connect APIs (order status), with OAuth credentials in a vault and sandboxed execution.
- **Billing:** metering on messages and tokens per tenant, with plan tiers mapped to model tiers.
- **Safety:** tenant-level guardrails (topics to avoid) plus platform-level ones. Defend against prompt injection that tries to read other tenants' data.

#### Evaluation
- **Per-tenant auto-generated test sets:** questions from their docs, re-run on model or prompt changes. Platform-wide regression gates.

### 29.38 Design an AI meeting summarizer system for thousands of meetings daily.

**See also:** 23.38 covers the speech recognition, diarisation and summarisation pipeline in detail; this answer focuses on running it for thousands of meetings a day.

**Requirements.** Thousands of meetings per day (30–60 min each) from Zoom, Meet and Teams. Deliver a summary, decisions, action items with owners, and searchable transcripts within minutes of the meeting ending.

```text
Meeting bot / platform API → audio stream → object storage
→ queue → ASR (Whisper-class, GPU batch) + speaker diarization (pyannote) + speaker identification (calendar attendees)
→ transcript store (with timestamps)
→ Summarization: chunk by topic/time → map (LLM per chunk) → reduce (meeting-level summary,
  decisions, action items as structured JSON) → validation
→ deliver (email/Slack/CRM sync) + index for search (RAG over all meetings)
```

#### Key choices
- **Batch vs streaming:** post-meeting batch processing is cheaper and more accurate. Optional live captions use streaming ASR.
- **Long transcripts:** a 60-minute meeting is about 10K words and fits modern context windows, so single-pass summarisation is often fine. Map-reduce handles very long meetings and reduces cost with smaller models.
- **Structured output:** action items as JSON (owner, task, due date) with transcript timestamps for verification.
- **Templates:** per meeting type (sales call → CRM fields, standup → blockers).
- **Throughput:** ASR dominates GPU cost. Batch inference with autoscaling, end-of-hour spike handling, and priority queues.

#### Privacy and compliance
- **Consent and access:** recording consent notices, per-organisation retention policies, encryption, access limited to attendees, and redaction options.

#### Evaluation
- **ASR:** word error rate.
- **Summary quality:** faithfulness checks (claims supported by the transcript) via LLM-as-judge, action-item precision/recall against human labels, and user edit rate.

### 29.39 Design an AI notification system that prioritizes instead of broadcasting.

**See also:** 28.32 covers notification ranking with classical ML; this answer focuses on LLM-based prioritisation.

**Goal.** Decide *whether*, *when*, *how* and *which* notifications to send each user, maximising long-term value (engagement, retention) while minimising fatigue and opt-outs.

```text
Event producers → candidate notifications (Kafka)
→ Enrichment: user features (activity, timezone, past opens), content features (urgency, type)
→ Scoring model: P(open), P(valuable action), P(disable/unsubscribe) → utility = value − fatigue cost
→ Policy: per-user budget/day, dedupe & bundling (digest), quiet hours, send-time optimization
→ Channel selection (push / email / in-app / SMS) → delivery → feedback events
```

#### Key design points
- **Must-send vs optional:** transactional and security alerts bypass ranking. Marketing and social ones are ranked.
- **Model:** a multi-objective ranker (GBDT/DNN). Use uplift modelling to measure the *incremental* effect of sending vs not sending, not just open probability.
- **LLM roles:** summarise bundles into a digest ("3 updates from your team"), personalise copy, and classify the urgency of free-text events.
- **Send-time optimisation:** a per-user propensity by hour or bandit.
- **Fatigue control:** adaptive caps, and back off after ignored notifications.
- **Real-time:** a stream processor with a delay buffer, which allows aggregation within windows.

#### Evaluation
- **Long-term holdouts:** test on retention and opt-out rates, not just CTR, since short-term CTR optimisation burns users out.
- **Monitoring:** delivery latency for urgent items.

### 29.40 Design an AI-powered anomaly detection system for cloud infrastructure.

**Requirements.** Millions of time series (CPU, latency, error rates, costs), logs and traces across thousands of services. Detect incidents early, reduce alert noise, and help with root cause.

```text
Telemetry (Prometheus/OTel, logs, traces, cost data) → stream (Kafka) → feature/rollup service
→ Detection layer:
   - per-metric: seasonal baselines (STL / Prophet-like), robust z-score, EWMA; ML (isolation forest, autoencoders) for multivariate
   - logs: template mining (Drain) → rare/new pattern detection; embeddings for clustering
→ Correlation: group anomalies by time + service dependency graph (topology) → single incident
→ LLM copilot: summarize incident, correlate deploys/config changes, suggest likely root cause & runbook
→ Alerting (PagerDuty/Slack) + feedback (true/false positive)
```

#### Key choices
- **Statistical first:** cheap, explainable per-series models at scale, with ML for multivariate and complex patterns. Handle seasonality (daily and weekly), trends and deploy-induced level shifts.
- **Noise reduction:** dynamic thresholds, minimum duration, and topology-aware grouping. Alert fatigue is the main failure mode.
- **Change correlation:** join anomalies with the deploy and config change feed, the most common root cause.
- **LLM role:** a triage assistant querying metrics and logs through tools (with read-only permissions), not the detector itself.

#### Evaluation
- **Labels:** past incidents. Measure precision/recall, time to detect, and alerts per incident.
- **Feedback:** on-call ratings tune sensitivity per service.

### 29.41 Design an AI-powered document processing pipeline for financial institutions.

**Documents.** Loan applications, KYC IDs, bank statements, tax forms, trade confirmations. Regulated (SOX, GDPR, model risk management under SR 11-7), auditable, with high accuracy requirements.

```text
Secure intake (portal, email, SFTP) → malware scan → classification
→ OCR/layout + VLM/LLM extraction to schema (structured outputs) per doc type
→ validation: checksums, cross-document consistency (income on payslip vs statement),
  master-data/KYC checks, fraud signals (tampering detection, font/metadata anomalies)
→ confidence routing: STP | human review (maker-checker) → downstream core systems
→ immutable audit log (input hash, model version, outputs, reviewer actions)
```

#### Key design points
- **Deployment:** private VPC or on-prem models, or zero-retention providers. Encryption at rest and in transit, data residency, and PII tokenisation for downstream analytics.
- **Explainability and traceability:** field-level provenance (page and bbox), versioned models and prompts, reproducible processing. Model risk management documentation and validation.
- **Human in the loop:** thresholds tuned to the cost of error. Four-eyes review for high-value decisions.
- **Fraud and tampering:** image forensics, template matching, and cross-checking against external data (credit bureaus, registries).
- **Reliability:** idempotent jobs, exactly-once handoff to core banking via an outbox pattern, and SLAs per document type.

#### Evaluation
- **Field-level accuracy:** per document type on labelled data, plus STP rate, review time and error escapes found in QA sampling.
- **Bias:** checks where outputs feed credit decisions (fair lending).

### 29.42 Design an AI dynamic pricing engine.

**See also:** 28.28 covers the demand modelling and causal side of dynamic pricing in depth.

**Use case.** E-commerce, ride-hailing or travel. Set prices per product, segment and time to maximise revenue or margin subject to business constraints.

```text
Data: sales history, inventory, competitor prices (scraped), demand signals (traffic, seasonality, events)
→ Demand model: estimate demand as function of price & features (elasticity) — GBDT/causal ML/Bayesian
→ Optimizer: maximize expected profit s.t. constraints (min margin, max change/day, price parity,
  MAP rules, fairness) — LP/MIP or per-item search
→ Experimentation: bandits / randomized price tests for elasticity estimation
→ Guardrails & approval → publish to pricing service (low-latency cache) → monitor
```

#### Key design points
- **Causality matters:** observational data is confounded (prices rose *because* demand was high). Use randomised experiments, instrumental variables, or double ML to estimate elasticity.
- **Exploration vs exploitation:** Thompson sampling or contextual bandits for continuous learning, with constrained exploration ranges.
- **Real-time vs batch:** ride-hailing surge needs streaming supply/demand within seconds. Retail repricing can be hourly or daily batch.
- **Guardrails:** price floors and ceilings, rate-of-change limits, kill switches, and anomaly detection (competitor-scrape glitches can cause price wars or \$0.01 errors).
- **Ethics and legal:** avoid discriminatory personalised pricing on protected attributes, comply with price-gouging laws during emergencies, and maintain transparency.
- **LLM role:** minor, such as competitor product matching (entity resolution) and explaining price changes to merchandisers.

#### Evaluation
- **Online:** A/B or switchback tests on revenue, margin and conversion, plus long-term customer trust metrics.

### 29.43 Design an AI resume screening system that handles 100K applications per week.

**Requirements.** About 100K resumes per week (about 15K/day, bursty) across many job postings. Rank and shortlist candidates for recruiters, fairly, explainably and legally. This is a high-risk use case under the EU AI Act and NYC Local Law 144.

```text
ATS → resume intake → parse (PDF/DOCX → text; LLM extraction to schema: skills, experience, education)
→ PII/protected-attribute masking (name, photo, age, gender cues) before scoring
→ Job requirements structured (LLM from JD, recruiter-confirmed must-haves)
→ Matching: embedding similarity (retrieval) + rubric scoring per requirement (LLM with evidence quotes)
→ Rank + explanation ("meets 4/5 must-haves: ...") → recruiter review → decisions → feedback
```

#### Key design points
- **Human in the loop:** the system recommends and humans decide. No automatic rejections without review, which may be legally required.
- **Fairness:** don't train on historical hiring decisions naively (they encode bias, as with the Amazon case). Run regular disparate-impact audits (four-fifths rule) across demographics where lawfully available, and use counterfactual tests (swap names or gender cues and check the score is stable).
- **Explainability:** requirement-level scores with quoted evidence, and candidate notice or appeal where required.
- **Scale and cost:** 100K/week is modest. Batch LLM calls, a small model for extraction, and a larger model only for the top candidates.
- **Security:** PII encryption, retention limits, GDPR rights.

#### Evaluation
- **Agreement:** with expert recruiter rankings and quality of hire downstream, plus adverse-impact ratios and auditor reviews.

**Risks:** adversarial resumes (hidden text with prompt injection), so sanitise and detect invisible text.

### 29.44 Design an AI voice assistant architecture.

**See also:** 23.33 covers real-time voice agent latency budgets, turn-taking and speech-to-speech models in more depth.

**Requirements.** Natural spoken conversation (support line or Alexa-like). Under 800 ms voice-to-voice latency, barge-in (interruptions), and tool use.

```text
Mic/phone (WebRTC/SIP) → VAD + endpointing → streaming ASR (partial transcripts)
→ Dialogue orchestrator: LLM (streaming) with tools/RAG, conversation state
→ streaming TTS (start speaking on first sentence) → audio out
Alternative: speech-to-speech native model (realtime APIs) for lower latency & prosody
```

#### Latency budget (cascade)
VAD and endpointing take about 200 ms, ASR finalisation about 100 ms, LLM TTFT about 300 ms, and TTS first audio about 150 ms.

#### Techniques
- **Overlap stages:** start the LLM on partial transcripts or speculatively, and stream LLM tokens into TTS sentence by sentence.
- **Engines:** small or fast LLMs for chit-chat routing, larger models for complex tasks. Filler responses ("let me check that") during tool calls.
- **Turn-taking:** semantic endpointing (an end-of-turn model) beats fixed silence timeouts. On barge-in, stop TTS immediately, truncate the assistant message in history to what was actually spoken, and listen.
- **Cascade vs speech-to-speech:** a cascade is modular, debuggable and allows any LLM and tools. Speech-to-speech gives lower latency and better emotion, but is less controllable.
- **Telephony:** 8 kHz audio, noise, DTMF, call transfer to humans.

#### Production
- **Scale:** WebRTC media servers and regional deployment for latency.
- **Logging:** recordings with consent.
- **Evaluation:** WER, task success, latency percentiles, interruption handling, and MOS for TTS.

### 29.45 Design a multi-agent workflow system where agents collaborate on complex tasks.

**Goal.** A platform where specialised agents (researcher, analyst, coder, reviewer) collaborate on long, complex tasks such as producing a market analysis or implementing a feature.

```text
User task → Orchestrator/Planner → task graph (DAG) with dependencies
→ Workflow engine (durable: Temporal/queue-based) dispatches steps to agents
   Agent = LLM + role prompt + scoped tools + memory
→ Shared state: blackboard/artifact store (documents, code, notes) with versioning
→ Reviewer/critic agents validate outputs → loops bounded by budgets
→ Human checkpoints for approvals → final synthesis
```

#### Key design points
- **Topology:** orchestrator-worker (hierarchical) is easiest to control. Peer-to-peer chat between agents tends to drift and cost more. Use a deterministic workflow where the process is known, and agentic planning only where needed.
- **Context isolation:** sub-agents get focused context and return compressed results, which avoids context explosion in the orchestrator.
- **Communication:** structured messages and artifacts (JSON schemas) rather than free-form chat.
- **Durability:** long tasks need checkpointing, retries, idempotent tools, and resumability after crashes (durable execution).
- **Control:** budgets on tokens, cost, time and step count. Loop and deadlock detection. Escalation to a human.
- **Security:** least-privilege tools per agent, sandboxed code execution, and treating inter-agent and tool content as untrusted (prompt-injection propagation).
- **Observability:** full traces of each agent's reasoning, tool calls and costs (OpenTelemetry), with replayability.

#### Evaluation
- **Outcome:** end-to-end task success on benchmark tasks, cost per task, human intervention rate.
- **Ablations:** single-agent vs multi-agent ablations, because multi-agent isn't automatically better.

### 29.46 Design a real-time AI transcription system for concurrent audio streams.

**Requirements.** 10K+ concurrent live streams (calls, meetings, broadcasts). Partial transcripts within about 300–500 ms, final segments within about 1–2 s, plus diarization, punctuation and multiple languages.

```text
Clients (WebSocket/gRPC/WebRTC) → edge ingest (regional) → session router (sticky)
→ streaming ASR workers on GPU: VAD → chunked encoder (streaming Conformer/RNN-T or
  chunked Whisper-style with overlap) → partial hypotheses + finalized segments
→ post-processing: punctuation/casing, ITN (numbers), diarization, custom vocabulary boosting
→ pub/sub (Kafka) → subscribers (UI captions, storage, downstream LLM analytics)
```

#### Key design points
- **Model choice:** streaming-native models (RNN-T, CTC with chunked attention) give stable low-latency partials. Whisper-style models need chunking with overlap and local agreement to stabilise output, at higher latency.
- **GPU efficiency:** batch many streams' chunks per inference step (dynamic batching across sessions). Concurrency per GPU is the key capacity metric.
- **Statefulness:** encoder and decoder state per stream means sticky routing. On worker failure, reconnect with a short audio buffer replay.
- **Backpressure:** if a worker lags, drop to a smaller model or reduce partial frequency instead of growing latency without bound.
- **Language and domain:** language ID at stream start, contextual biasing (names, jargon), and per-customer custom vocabularies.
- **Scaling:** autoscale on active streams per GPU, with regional deployment to cut network latency.

#### Evaluation
- **Accuracy:** WER (overall and per domain), and partial-transcript stability (how often words change).
- **Latency:** p50/p95 latency of partials and finals, plus diarization error rate.

**Compliance:** consent, encryption, retention, and PII redaction in transcripts.

### 29.47 Design an AI-powered live streaming content moderation system.

**Requirements.** 100K+ concurrent live streams plus live chat. Detect violations (nudity, violence, self-harm, hate, dangerous acts) within seconds, because damage happens live.

```text
Stream ingest → sampler: keyframes every 1–2s (adaptive: more for high-risk/high-viewer streams)
             + audio chunks → streaming ASR → text
→ Fast classifiers (vision NSFW/violence, audio toxicity, text) on GPU, batched across streams
→ Risk aggregator: temporal smoothing over window + context (creator history, viewer count, reports)
→ Actions: auto (blur/mute/end stream) for high-confidence severe; priority human review queue otherwise
→ Live chat: per-message text classifier + LLM for nuanced cases, rate limits, slow mode
→ Evidence capture (clip buffer) for appeals & law-enforcement reporting
```

#### Key design points
- **Cost control:** analysing every frame of 100K streams is unaffordable. Use adaptive sampling, cheap first-pass models, and escalate to heavier VLM or LLM analysis only for flagged segments. Prioritise by reach ($\text{viewer count} \times \text{risk}$).
- **Latency:** the end-to-end detect-to-action target is under 10 s for severe categories. Use stream processing (Kafka/Flink) and colocate inference with ingest.
- **Temporal reasoning:** single frames mislead, so aggregate over windows and use video-clip models for actions (fights, self-harm).
- **Human reviewers:** live queues with stream playback from the flagged timestamp, plus SLAs and reviewer-wellness tooling.
- **Known-bad content:** hash matching for re-streamed known illegal content (PDQ/video hashing).
- **Feedback loop:** reviewer decisions and appeals produce labels for retraining. Monitor per-category precision/recall and language coverage.

**Tradeoffs:** false positives (ending a legitimate stream hurts creators) vs false negatives (harm and regulatory risk). Use graduated actions (warning → blur → end) and severity-specific thresholds.

### 29.48 Design a conversational assistant (memory, context management, safety, latency).

**Clarify:** domain (customer support vs. general chat), scale (QPS, users), latency SLO (e.g., first token < 1s), modalities, compliance (PII, regulated data).

**Architecture**

```text
Client ⇄ (WebSocket/SSE streaming) ⇄ API Gateway (auth, rate limit)
   → Orchestrator ── Input guardrails (PII redaction, injection/jailbreak, moderation)
        ├─ Context builder: system prompt + user profile + memory + retrieved docs + recent turns
        ├─ Router: small model for simple intents, frontier model for hard ones
        ├─ Tools (search, CRM, booking) via function calling / MCP
        └─ LLM inference (vLLM/hosted API) → Output guardrails → stream to user
   → Async: memory writer, summarizer, eval/logging pipeline
```

**Memory**

- **Short-term:** recent turns kept verbatim inside a token budget.
- **Rolling summary:** older turns compressed by a cheap model as the conversation grows.
- **Long-term:** extracted facts and preferences stored per user (a key-value store plus a vector store), retrieved by relevance, with user-visible controls to view and delete (GDPR) and TTLs.
- **Episodic:** past sessions stored as summaries, so the assistant can say "last time you asked about…".

**Context management:** a fixed token budget per section (system prompt, memory, retrieval, history), reranking to keep only the top-N chunks, the most relevant content placed at the start or end (lost-in-the-middle), and prompt caching for the static prefix.

**Safety:** layered input and output classifiers, treat retrieved and tool content as untrusted data (injection), least-privilege tools with confirmation for side-effecting actions, grounded answers with citations, refusal and escalation-to-human paths, and an audit log.

**Latency:** streaming tokens, a parallel pipeline (retrieval, memory lookup and moderation run concurrently), small-model routing, semantic and prompt caching, speculative decoding / continuous batching on self-hosted models, and asynchronous memory writes off the critical path. Measure TTFT and p95/p99 latency.

**Evaluation and operations:** golden conversation sets and LLM-as-judge (calibrated to human ratings), regression tests in CI for prompt changes, online metrics (CSAT, resolution rate, escalation rate, thumbs up/down), tracing of every turn (Langfuse/OTel), and cost per conversation dashboards.

### 29.49 Design an enterprise RAG platform.

**Goal:** a shared platform letting many teams build grounded Q&A/assistants over internal content (Confluence, SharePoint, Drive, tickets, databases) with security, quality and cost control.

**Components:**
1. **Connectors and ingestion:** scheduled + event-driven sync; parsing (PDF, Office, HTML, OCR); deduplication; structure-aware chunking; metadata (source, owner, timestamps, **ACLs**); incremental re-indexing and deletion propagation.
2. **Indexing:** embeddings (versioned model) in a vector store plus BM25/keyword index; per-tenant or per-domain namespaces.
3. **Retrieval service:** query rewriting, hybrid search, **permission filtering at query time** (user's groups must match document ACLs), cross-encoder reranking, MMR.
4. **Generation service:** model gateway/router (multiple providers, fallback, model routing by complexity), prompt templates under version control, grounded answers with citations, abstention when evidence is weak, streaming.
5. **Guardrails:** PII redaction, prompt-injection defences (treat retrieved content as data), output filtering, rate limiting.
6. **Observability and evaluation:** end-to-end tracing, golden sets per use case, retrieval and faithfulness metrics in CI, sampled online evaluation, user feedback, cost per query dashboards.
7. **Platform APIs/SDK:** self-service onboarding of sources, configurable pipelines per team, chat UI and API.

**Non-functional:** data residency and tenant isolation, audit logs, SSO, latency targets (first token < 1–2 s), caching (embedding, retrieval, semantic/prompt caching), autoscaling.

**Key tradeoffs:** central platform vs team autonomy (offer defaults with overridable pipelines); freshness vs indexing cost; managed vector DB vs self-hosted; stricter grounding vs refusal rate.

### 29.50 Design a natural-language analytics assistant (text-to-SQL) over a company's data warehouse.

**Clarify:** business users ask questions such as "What was revenue by region last quarter compared with the year before?" and get correct numbers, a chart and an explanation. Hundreds of tables, several thousand users, answers in under about 15 seconds. **Wrong numbers are worse than no answer**, because they drive decisions.

**Architecture**

```text
question ─► intent and ambiguity check (clarify if needed)
  ─► context retrieval: semantic layer (certified metrics, dimensions, joins), table and column docs,
     sample values, similar past question-to-query pairs, business glossary
  ─► SQL generation (LLM, constrained to allowed tables)
  ─► validation: parse, allowlisted tables and columns, row-level security, cost estimate (dry run), limits
  ─► execute as the user (read-only, with their permissions) ─► results
  ─► answer: table or chart + plain-language summary + the SQL and the definitions used
  ─► feedback (correct / wrong) ─► examples store and evaluation set
```

**Key design decisions**

- **Ground on a semantic layer, not raw tables:** generating queries against certified metrics ("net revenue" defined once) avoids the most common error: correct SQL with the wrong business definition. Where a semantic layer exists, the model can generate semantic-layer queries rather than raw SQL.
- **Schema retrieval:** large warehouses do not fit in context; retrieve relevant tables, columns, descriptions and example values for each question.
- **Few-shot examples:** retrieve verified question-to-query pairs similar to the new question; this is one of the strongest accuracy levers.
- **Ambiguity handling:** ask a clarifying question when terms are ambiguous ("customers": active or all?) instead of guessing.
- **Security:** execute with the user's own permissions (row-level security, column masking), read-only credentials, query timeouts and cost limits; never let the model bypass access control.
- **Transparency:** always show the definitions and filters used and the SQL for analysts, so users can verify.
- **Agentic correction:** if the query fails or returns suspicious results (empty, implausible totals), let the model inspect the error and retry a limited number of times.

**Evaluation**

- A golden set of questions with verified answers, scored by **result equivalence** (comparing result sets), not SQL string match.
- Track accuracy by difficulty (joins, time comparisons, ratios), clarification rate, user-flagged errors, and latency and warehouse cost per question.

```python
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"analytics.fct_orders", "analytics.dim_customer", "analytics.dim_date"}

def validate_sql(sql: str, dialect="bigquery"):
    tree = sqlglot.parse_one(sql, read=dialect)
    if not isinstance(tree, exp.Select) and not tree.find(exp.Select):
        raise ValueError("only SELECT queries are allowed")
    if tree.find(exp.Insert) or tree.find(exp.Update) or tree.find(exp.Delete) or tree.find(exp.Drop):
        raise ValueError("data modification is not allowed")
    tables = {f"{t.db}.{t.name}" if t.db else t.name for t in tree.find_all(exp.Table)}
    if not tables <= ALLOWED_TABLES:
        raise ValueError(f"tables not allowed: {tables - ALLOWED_TABLES}")
    if not tree.args.get("limit"):
        tree = tree.limit(10_000)                       # cap result size
    return tree.sql(dialect=dialect)
```

### 29.51 Design a guardrails service for LLM applications across an organisation.

**Clarify:** dozens of LLM applications (chat assistants, RAG, agents) built by different teams. The organisation needs consistent protection against harmful content, data leakage and prompt injection, configurable per application, adding little latency.

**Architecture**

```text
application ─► AI gateway ─► input guardrails ─► LLM / agent ─► output guardrails ─► application
                               │                                    │
                               ▼                                    ▼
                    policy engine (per app and tenant config) ◄── decisions log, metrics, review queue
```

**Checks**

| Stage | Checks |
| --- | --- |
| Input | Personal data detection and redaction, secrets detection, prompt injection and jailbreak classifiers, topic and scope restrictions, content moderation, size limits |
| Retrieved content and tool outputs | Injection detection in documents and web pages, marking content as untrusted data |
| Tool calls (agents) | Argument validation, allowlists, permission checks, human approval for high-risk actions |
| Output | Content moderation, personal data and secret leakage, system prompt leakage, groundedness or citation checks for RAG, format and schema validation, brand and policy rules |

**Design decisions**

- **Layered and cheap first:** regular expressions and deterministic rules (secrets, card numbers), then small fast classifiers, then LLM-based judges only for uncertain cases or high-risk applications.
- **Latency:** run input checks in parallel with retrieval; stream output while running checks on chunks, with the ability to stop or retract; set a latency budget per check.
- **Policy as configuration:** each application declares its policies (which checks, thresholds, actions: block, redact, warn, log) in versioned configuration, reviewed by security and legal.
- **Actions:** block with a safe message, redact and continue, ask for confirmation, or route to human review.
- **Fail-open or fail-closed:** decide per check and application; safety-critical checks fail closed.
- **Observability:** log every decision with the policy version; dashboards of trigger rates; sampled review of blocks and passes to measure false positives and false negatives.

**Evaluation:** maintain attack and benign test sets per check (injection, jailbreak, personal data, harmful content); measure precision and recall, latency overhead, and the false-block rate on normal traffic. Red-team regularly and add successful attacks to the test sets.

**Build or buy:** cloud and open-source guardrail tools exist (moderation APIs, open classifiers such as Llama Guard-style models, guardrail frameworks); most organisations combine bought components with their own policy engine and domain rules.

### 29.52 Design a self-serve fine-tuning platform for internal teams.

**Clarify:** product teams want to fine-tune open-weight models (and sometimes hosted models) for their tasks: classification, extraction, domain assistants. The platform should make this safe, reproducible and cost-controlled, without each team building its own training stack.

**Workflow**

```text
dataset upload / connector ─► validation (format, chat template, duplicates, personal data scan, licence)
  ─► dataset versioning ─► job config (base model, method: LoRA / QLoRA / full / DPO, hyperparameters)
  ─► scheduler ─► GPU training cluster (checkpointing, preemption-safe) ─► artefact registry (adapters, configs)
  ─► automatic evaluation: task suite + general capability + safety regression vs base model
  ─► approval gate ─► deployment: multi-LoRA serving on shared base models ─► monitoring and feedback
```

**Key design decisions**

- **Opinionated defaults:** LoRA with tested hyperparameters for each supported base model, so most teams never tune by hand; advanced options for experts.
- **Data quality gates:** validation of formatting and chat templates, deduplication, train and evaluation leakage checks, personal data scanning, and licence checks for data sources.
- **Evaluation built in:** every job is compared with the base model and with a prompting baseline on the team's task suite, plus general capability and safety benchmarks to catch forgetting and weakened refusals (see 22.39).
- **Serving efficiency:** adapters are served on shared base models with multi-LoRA serving (see 22.35), so thousands of adapters do not need thousands of GPUs.
- **Reproducibility:** dataset version, code version, base model version, configuration and random seeds are recorded for every artefact.
- **Cost control:** quotas per team, cost estimates before launch, spot or preemptible GPUs with checkpointing, and automatic cleanup of unused artefacts.
- **Governance:** approval before production deployment, model cards, lineage from data to deployed adapter, and deletion workflows when source data must be removed.

**Metrics:** time from dataset to deployed model, share of jobs that beat the prompting baseline, GPU utilisation and cost per job, number of models in production and their quality over time.

### 29.53 Design an AI answer engine over the web (search plus generated answers with citations).

**Clarify:** users ask questions and get a direct answer with citations to web sources, plus follow-up questions. High query volume, latency target of a few seconds to a complete answer (first token within about a second), freshness for news, and high trust: answers must be faithful to the sources.

**Architecture**

```text
query ─► query understanding (intent, freshness need, rewrite into several search queries)
  ─► search: web search index or search API (parallel queries) ─► top results
  ─► fetch and extract page content (cached; respect robots rules and rate limits)
  ─► chunk and rerank passages against the query ─► select top passages within a token budget
  ─► answer generation with inline citations (streaming) ─► citation verification
  ─► answer + sources + suggested follow-ups
```

**Key design decisions**

- **Retrieval quality is the ceiling:** multiple query rewrites, a strong reranker, source quality signals (authority, recency, spam filtering) and diversity across sources.
- **Freshness:** route time-sensitive queries to recent indexes or news sources; show dates on sources.
- **Faithful citations:** instruct the model to cite only from provided passages; verify that each cited sentence is supported by its source (an entailment check or LLM judge on samples) and remove or flag unsupported claims.
- **Latency:** parallelise searches and page fetches, cache popular pages and queries, stream the answer while fetching continues for follow-ups, and use a fast model for simple factual queries with a stronger model for complex research questions.
- **Modes:** quick answers versus deeper research modes that run multiple search and read iterations (an agentic loop) and take longer (see 29.5 for a deep research agent).
- **Safety and trust:** handle conflicting sources by presenting both sides, prefer authoritative sources for medical, legal and financial topics, and refuse or add caution where needed.
- **Publisher considerations:** link prominently to sources, respect access restrictions, and follow licensing agreements for content use.

**Evaluation:** answer correctness on a labelled question set, citation precision (cited source supports the claim) and recall (claims that should be cited are cited), freshness on time-sensitive queries, latency percentiles, user feedback, follow-up and abandonment rates.

### 29.54 Design a synthetic data generation pipeline for training and evaluating models.

**Clarify:** a team needs training and evaluation data for an LLM-based task (for example, extracting fields from support tickets, or a domain assistant), but real labelled data is scarce, sensitive or expensive. Goal: high-quality synthetic data at scale, without contaminating evaluations or amplifying errors.

**Pipeline**

```text
seed data (small real sample, taxonomy of intents and edge cases, documents)
  ─► generation (strong model, diverse prompts: personas, difficulty levels, styles, edge cases)
  ─► filtering: schema validation, deduplication (exact and near-duplicate), quality scoring by judge models,
     verification (execute code, check answers against sources), safety and personal data screening
  ─► human review of a sample (and of all evaluation data) ─► versioned dataset with provenance
  ─► training ─► evaluation on REAL held-out data ─► error analysis ─► targeted generation for weak areas
```

**Key design decisions**

- **Diversity by design:** generate from a taxonomy (intents, difficulty, languages, formats, edge cases) and from varied personas and contexts, rather than repeating one prompt; measure diversity (embedding spread, n-gram overlap) and coverage of the taxonomy.
- **Grounding:** generate questions and answers from real documents so facts are correct; verify answers against the source passages.
- **Verification beats generation:** for code, math or structured outputs, execute or check results and keep only verified examples (rejection sampling).
- **Keep evaluation real:** evaluate primarily on real, human-labelled data; synthetic evaluation sets are useful for coverage but must be reviewed, and generated with a different model or prompt family than training data to avoid shared blind spots.
- **Avoid contamination:** deduplicate synthetic training data against evaluation sets.
- **Model collapse risk:** training repeatedly on model-generated data narrows diversity and can amplify errors; mix with real data and refresh seeds.
- **Licensing and terms:** check whether the generating model's terms allow using outputs to train other models (see 22.21).

**Metrics:** acceptance rate after filtering, diversity and coverage, downstream improvement on real held-out data per thousand synthetic examples, generation and verification cost.

---

← [Chapter 28. Machine Learning System Design](28-machine-learning-system-design.md) · [Table of contents](../README.md) · [Chapter 30. AI Safety, Security and Responsible AI](30-ai-safety-security-and-responsible-ai.md) →
