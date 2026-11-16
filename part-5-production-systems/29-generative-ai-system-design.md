# Chapter 29. Generative AI System Design

_Part V — Production Systems and Architecture_

End-to-end design of LLM, RAG, agentic and generative-media products, plus cross-cutting concerns such as latency, caching, failover and capacity planning.

This chapter contains **49 questions**.

## Contents

- **29.1** [Design ChatGPT: Training to Serving (End to End)](#291-design-chatgpt-training-to-serving-end-to-end)
- **29.2** [Design a RAG System (Chat with Your Documents)](#292-design-a-rag-system-chat-with-your-documents)
- **29.3** [Design Memory for a Personal AI Assistant](#293-design-memory-for-a-personal-ai-assistant)
- **29.4** [Design a Deep Research Agent](#294-design-a-deep-research-agent)
- **29.5** [Design a Multi-Agent Customer Support System](#295-design-a-multi-agent-customer-support-system)
- **29.6** [Design an On-Device AI Assistant](#296-design-an-on-device-ai-assistant)
- **29.7** [Design a Multimodal Search System (Text, Image, Video)](#297-design-a-multimodal-search-system-text-image-video)
- **29.8** [Design an LLM Inference Platform (vLLM-as-a-Service)](#298-design-an-llm-inference-platform-vllm-as-a-service)
- **29.9** [Design an LLM Evaluation Platform](#299-design-an-llm-evaluation-platform)
- **29.10** [Design a Text-to-Image Generation Service (Midjourney-like)](#2910-design-a-text-to-image-generation-service-midjourney-like)
- **29.11** [Design a Music Generation Service (Suno-like)](#2911-design-a-music-generation-service-suno-like)
- **29.12** [Design a Video Generation Service (Sora-like)](#2912-design-a-video-generation-service-sora-like)
- **29.13** [Design an AI Coding Agent.](#2913-design-an-ai-coding-agent)
- **29.14** [Design a code generation and review system.](#2914-design-a-code-generation-and-review-system)
- **29.15** [Design a content moderation system using AI.](#2915-design-a-content-moderation-system-using-ai)
- **29.16** [Design a real-time AI recommendation system.](#2916-design-a-real-time-ai-recommendation-system)
- **29.17** [Design an AI-powered email assistant.](#2917-design-an-ai-powered-email-assistant)
- **29.18** [Design a medical diagnosis assistant using AI.](#2918-design-a-medical-diagnosis-assistant-using-ai)
- **29.19** [Design a fraud detection system powered by LLMs.](#2919-design-a-fraud-detection-system-powered-by-llms)
- **29.20** [Design an AI-powered data extraction pipeline from unstructured documents.](#2920-design-an-ai-powered-data-extraction-pipeline-from-unstructured-documents)
- **29.21** [Design a personalized learning assistant.](#2921-design-a-personalized-learning-assistant)
- **29.22** [Design an AI system for automated code migration.](#2922-design-an-ai-system-for-automated-code-migration)
- **29.23** [Design an AI-powered legal document review system.](#2923-design-an-ai-powered-legal-document-review-system)
- **29.24** [Design a conversational AI system with memory across sessions.](#2924-design-a-conversational-ai-system-with-memory-across-sessions)
- **29.25** [How do you design for latency vs quality trade-offs in AI systems?](#2925-how-do-you-design-for-latency-vs-quality-trade-offs-in-ai-systems)
- **29.26** [How do you implement caching strategies for LLM applications?](#2926-how-do-you-implement-caching-strategies-for-llm-applications)
- **29.27** [How do you design rate limiting and cost management for AI APIs?](#2927-how-do-you-design-rate-limiting-and-cost-management-for-ai-apis)
- **29.28** [How do you handle failover and fallback strategies for AI systems?](#2928-how-do-you-handle-failover-and-fallback-strategies-for-ai-systems)
- **29.29** [How do you design an AI system for high availability and fault tolerance?](#2929-how-do-you-design-an-ai-system-for-high-availability-and-fault-tolerance)
- **29.30** [How do you design an AI system that gracefully degrades when the model is unavailable?](#2930-how-do-you-design-an-ai-system-that-gracefully-degrades-when-the-model-is-unavailable)
- **29.31** [What are the key considerations for multi-region deployment of AI systems?](#2931-what-are-the-key-considerations-for-multi-region-deployment-of-ai-systems)
- **29.32** [Design an AI-powered search engine for an e-commerce platform.](#2932-design-an-ai-powered-search-engine-for-an-e-commerce-platform)
- **29.33** [Design an AI gateway/proxy for managing LLM access across an organization.](#2933-design-an-ai-gatewayproxy-for-managing-llm-access-across-an-organization)
- **29.34** [How do you design a RAG system that handles conflicting information across sources?](#2934-how-do-you-design-a-rag-system-that-handles-conflicting-information-across-sources)
- **29.35** [How do you approach capacity planning for an AI system?](#2935-how-do-you-approach-capacity-planning-for-an-ai-system)
- **29.36** [Design a multi-tenant AI chatbot platform where each business gets a custom chatbot.](#2936-design-a-multi-tenant-ai-chatbot-platform-where-each-business-gets-a-custom-chatbot)
- **29.37** [Design an AI meeting summarizer system for thousands of meetings daily.](#2937-design-an-ai-meeting-summarizer-system-for-thousands-of-meetings-daily)
- **29.38** [Design an AI notification system that prioritizes instead of broadcasting.](#2938-design-an-ai-notification-system-that-prioritizes-instead-of-broadcasting)
- **29.39** [Design an AI-powered anomaly detection system for cloud infrastructure.](#2939-design-an-ai-powered-anomaly-detection-system-for-cloud-infrastructure)
- **29.40** [Design an AI-powered document processing pipeline for financial institutions.](#2940-design-an-ai-powered-document-processing-pipeline-for-financial-institutions)
- **29.41** [Design an AI dynamic pricing engine.](#2941-design-an-ai-dynamic-pricing-engine)
- **29.42** [Design an AI resume screening system that handles 100K applications per week.](#2942-design-an-ai-resume-screening-system-that-handles-100k-applications-per-week)
- **29.43** [Design an AI voice assistant architecture.](#2943-design-an-ai-voice-assistant-architecture)
- **29.44** [Design a multi-agent workflow system where agents collaborate on complex tasks.](#2944-design-a-multi-agent-workflow-system-where-agents-collaborate-on-complex-tasks)
- **29.45** [Design a real-time AI transcription system for concurrent audio streams.](#2945-design-a-real-time-ai-transcription-system-for-concurrent-audio-streams)
- **29.46** [Design an AI-powered live streaming content moderation system.](#2946-design-an-ai-powered-live-streaming-content-moderation-system)
- **29.47** [Design a conversational assistant (memory, context management, safety, latency).](#2947-design-a-conversational-assistant-memory-context-management-safety-latency)
- **29.48** [Design a coding agent.](#2948-design-a-coding-agent)
- **29.49** [Design an enterprise RAG platform.](#2949-design-an-enterprise-rag-platform)

---

### 29.1 Design ChatGPT: Training to Serving (End to End)

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

### 29.2 Design a RAG System (Chat with Your Documents)

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

### 29.3 Design Memory for a Personal AI Assistant

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
- **Retrieval:** score = similarity × recency decay × importance. Cap the injected memory tokens.
- **User control:** users can view, edit and delete memories ("forget this"). Respect GDPR deletion, encrypt per user, and never store sensitive categories without consent.
- **Hygiene:** a periodic consolidation job merges duplicates and expires stale items.

#### Evaluation
- **Offline:** memory precision (is what's recalled relevant?) and recall on scripted multi-session tests.
- **Online:** the rate of corrections ("I already told you...").

**Risks:** wrong memories compounding, privacy leaks across users (keep a strict user_id partition), and prompt injection that writes false memories.

### 29.4 Design a Deep Research Agent

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

### 29.5 Design a Multi-Agent Customer Support System

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

### 29.6 Design an On-Device AI Assistant

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

### 29.7 Design a Multimodal Search System (Text, Image, Video)

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

### 29.8 Design an LLM Inference Platform (vLLM-as-a-Service)

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

### 29.9 Design an LLM Evaluation Platform

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

### 29.10 Design a Text-to-Image Generation Service (Midjourney-like)

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

### 29.11 Design a Music Generation Service (Suno-like)

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

### 29.12 Design a Video Generation Service (Sora-like)

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

### 29.13 Design an AI Coding Agent.

**Goal.** Given an issue or task, the agent reads a repo, plans, edits code, runs tests, and opens a PR (Claude Code / Devin / SWE-agent style).

```text
Task → Agent loop (LLM): think → call tool → observe → repeat
Tools: file search (grep/ripgrep, symbol index), read/edit file (diff-based), run shell
       in sandbox (tests, linters, build), git, web/docs lookup
State: plan/todo list, context manager (summarize old observations)
Output: patch + test results → PR with description → human review
```

#### Key design points
- **Context engineering:** repos don't fit in context. Use agentic search (grep, tree, LSP "go to definition") over pure embedding RAG, and compress or summarise old tool outputs.
- **Edits:** structured search/replace or diff edits are more reliable than rewriting whole files.
- **Verification:** run tests, type checks and linters. The agent iterates on failures, and sometimes writes a reproduction test first.
- **Sandbox:** ephemeral containers with no production credentials, network egress controls, and resource and time limits.
- **Human in the loop:** permission prompts for destructive commands, and PR review before merge.
- **Budgets:** caps on steps and tokens, with stuck-loop detection.

#### Evaluation
- **Benchmarks:** SWE-bench Verified style task success, plus internal task suites.
- **Online:** PR acceptance rate, revert rate, review comments per PR, cost per task.

**Risks:** prompt injection via repo files or issues, secrets leakage, and plausible but wrong code (mitigated by tests and review).

### 29.14 Design a code generation and review system.

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
→ post inline comments; developers 👍/👎 → feedback dataset
```
- **Precision over recall:** noisy reviewers get ignored. Filter low-confidence or style nits and combine LLM output with deterministic tools.
- **Repo rules:** a CLAUDE.md, CODEOWNERS or style guide feeds the prompt.

#### Governance
- **Data handling:** code stays in the tenant (self-hosted or zero data retention) and secrets are filtered.
- **Licensing:** licence and copyleft snippet detection.
- **Measurement:** track impact on cycle time and defect rate through experiments, not anecdotes.

### 29.15 Design a content moderation system using AI.

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
- **Human review:** queues prioritised by reach × severity, reviewer wellness tooling (blurring), and inter-rater agreement tracking.

#### Monitoring
- **Metrics:** prevalence (sampled), precision/recall per policy and language, appeal overturn rate, time to action.
- **Adversarial drift:** leetspeak and image text. Keep active-learning loops running.

**Concerns:** bias across dialects and languages, transparency (notices), and regulatory reporting (DSA).

### 29.16 Design a real-time AI recommendation system.

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

### 29.17 Design an AI-powered email assistant.

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

### 29.18 Design a medical diagnosis assistant using AI.

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

### 29.19 Design a fraud detection system powered by LLMs.

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

### 29.20 Design an AI-powered data extraction pipeline from unstructured documents.

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

### 29.21 Design a personalized learning assistant.

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

### 29.22 Design an AI system for automated code migration.

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

### 29.23 Design an AI-powered legal document review system.

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

### 29.24 Design a conversational AI system with memory across sessions.

See also Q3.

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

### 29.25 How do you design for latency vs quality trade-offs in AI systems?

Start from the product SLO. Inline autocomplete needs under 300 ms, chat needs a TTFT under 1 s, and batch analysis can take minutes. Then pick levers per stage.

- **Model routing/cascades:** a small fast model handles most requests. Escalate to a larger model on low confidence, complexity classification or user request. This often gives 80% of the quality at 20% of the latency and cost.
- **Streaming:** perceived latency depends on TTFT, not total time.
- **Inference optimisation:** quantization (FP8/INT4), speculative decoding, continuous batching, prefix/KV caching, and smaller context (better retrieval means fewer tokens).
- **Parallelism:** run retrieval, tool calls and guardrails concurrently, and speculatively start generation while slower checks finish.
- **Caching:** exact and semantic caches for repeated queries, and precomputing for predictable requests.
- **Pipeline depth:** every agent step adds latency. Reduce the reranker's top-k, skip query rewriting for simple queries, and cap agent iterations.
- **Asynchronous UX:** for high-quality slow paths (deep research), make it a background job with progress updates.

Measure the Pareto frontier: plot quality (eval score) against p95 latency and cost for each configuration, and choose by segment (paid vs free, interactive vs batch). Always A/B test, because users often prefer fast and good enough over slow and perfect.

### 29.26 How do you implement caching strategies for LLM applications?

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

### 29.27 How do you design rate limiting and cost management for AI APIs?

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

### 29.28 How do you handle failover and fallback strategies for AI systems?

#### Failure types
Provider outages, 429 rate limits, timeouts and slow tails, model errors or invalid output, and quality failures such as hallucination or refusal.

#### Strategies
- **Retries:** exponential backoff with jitter for transient errors (429, 5xx), capped at a few attempts within the latency budget. Use idempotency keys for tool actions.
- **Timeouts and hedging:** set per-stage timeouts. For tail latency, send a hedged request to a second replica or provider after the p95 time.
- **Circuit breakers:** when an endpoint's error rate passes a threshold, open the circuit and route elsewhere, then probe half-open to recover.
- **Model fallback chain:** primary model → same model in another region → alternative provider (Bedrock, Vertex, Azure) → smaller self-hosted model. An AI gateway (LiteLLM-style) keeps a unified API.
- **Prompt portability:** fallback models behave differently, so maintain and evaluate prompts per model and keep eval suites running on fallbacks too.
- **Output validation fallback:** schema validation fails → retry with a repair prompt → fall back to a deterministic path.
- **Graceful degradation:** cached answers, search results without generation, or "try again later" with queued processing (see Q30).

#### Ops
- **Synthetic probes:** health checks per provider and region.
- **Chaos testing:** test the failover paths themselves.
- **Visibility:** dashboards for fallback rate, because silent fallback to a weaker model is a quality incident.

### 29.29 How do you design an AI system for high availability and fault tolerance?

**Targets.** For example 99.9% to 99.95% availability, with an RTO and RPO defined per component.

#### Principles
- **Redundancy at every layer:** stateless API and orchestrator services behind load balancers across multiple AZs, GPU inference replicas spread across AZs (N+1 or N+2 capacity headroom), and replicated data stores (vector DB replicas, Postgres with failover, Redis cluster).
- **Multi-provider/model redundancy:** for hosted LLMs, an AI gateway with fallback providers and circuit breakers (see Q28).
- **Decoupling:** async queues (Kafka/SQS) for non-interactive work, so spikes and outages buffer instead of failing. Use dead-letter queues and idempotent consumers.
- **Bulkheads:** separate pools for different tenants and features so one runaway workload can't take down everything.
- **Load shedding and backpressure:** admission control based on queue depth and KV-cache usage, with priority to critical traffic.
- **GPU realities:** nodes fail more often and are scarce. Use health checks (ECC errors, NCCL hangs), automatic cordoning, warm standby capacity, and fast model loading (local cache of weights).
- **Deployment safety:** canary and blue-green releases for models and prompts, automatic rollback on SLO breach, and versioned model registry.
- **State:** keep conversation and agent state in durable stores, and checkpoint long agent jobs so they resume after crashes.

#### Observability
- **SLOs:** on availability, TTFT and error rate, with burn-rate alerts.
- **Tracing and drills:** distributed tracing across retrieval, LLM and tools, plus regular game days.

### 29.30 How do you design an AI system that gracefully degrades when the model is unavailable?

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

### 29.31 What are the key considerations for multi-region deployment of AI systems?

- **Data residency and compliance:** GDPR and sector rules may require EU data, including prompts, logs, embeddings and vector indexes, to stay in-region. Route by user or tenant home region and check the provider's regional endpoints and data-processing terms.
- **GPU capacity:** H100s and similar are unevenly available across regions. Plan reservations per region and accept that some regions may serve smaller models or fall back to others (if residency allows).
- **Latency:** GeoDNS or anycast routes to the nearest healthy region. TTFT includes network RTT, so streaming helps.
- **Model and version consistency:** deploy the same model, prompt and guardrail versions everywhere through a central registry and CI/CD, with region-by-region canary rollouts.
- **State replication:** conversation history and user memory need either a home-region model (simpler, compliant) or multi-region replication (conflict handling). Vector indexes need re-indexing or replication pipelines per region.
- **Failover:** active-active or active-passive. Cross-region failover must respect residency, so define which tenants can fail over where.
- **Cost:** cross-region egress, duplicate GPU fleets, and duplicate indexes. Use regional autoscaling and follow-the-sun batch scheduling to use idle capacity.
- **Observability:** per-region SLOs, a global view, and per-region quotas with upstream providers.

### 29.32 Design an AI-powered search engine for an e-commerce platform.

**Requirements.** 10M+ SKUs, natural-language queries ("waterproof hiking boots under $150 for wide feet"), latency under 200 ms, and the goal of maximising conversion.

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

### 29.33 Design an AI gateway/proxy for managing LLM access across an organization.

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

### 29.34 How do you design a RAG system that handles conflicting information across sources?

Conflicts come from stale versions, different scopes (regional policies), different authorities, or genuine disagreement.

#### At ingestion
- **Metadata:** capture source, authority tier (official policy > wiki > Slack), effective date, version, region or product scope, and owner.
- **Deduplication and versioning:** keep only the latest version active, or mark superseded docs. Many "conflicts" are really staleness.

#### At retrieval
- **Filtering:** by applicable scope (user region, product version) before ranking.
- **Ranking:** relevance × authority × recency boosts.
- **Diversity:** retrieve with diversity (MMR) so conflicting viewpoints are visible rather than accidentally hidden.

#### At generation
- **Conflict prompting:** instruct the model to detect conflicts explicitly. Prefer authoritative and recent sources, and when conflict remains, present both with citations and dates ("Policy doc (2025) says X; older FAQ (2023) says Y").
- **Conflict-detection step:** optionally run a pairwise claim comparison (NLI or LLM) over the top chunks and flag contradictions.
- **Abstention:** for high-stakes domains, escalate rather than guess.

#### Feedback loop
- **Content governance:** log detected conflicts to a queue for content owners. Fixing the source is the real fix.

#### Evaluation
- **Test sets:** planted contradictions, measuring whether the system picks the correct authoritative answer or surfaces the conflict.

### 29.35 How do you approach capacity planning for an AI system?

#### 1. Model the demand
- **Usage profile:** DAU × requests per user × peak factor (peak-to-average is often 2–4x). Measure input and output tokens per request, which are the real units.
- **Example:** 1M DAU × 10 req = 10M req/day. Average is about 116 req/s and peak about 400 req/s. At 1,500 input and 300 output tokens, peak is about 600K input tokens/s and 120K output tokens/s.

#### 2. Benchmark supply
- **Load test:** test the actual model, hardware and engine (for example 70B FP8 on H100 with vLLM) to find sustainable tokens/s per GPU at your SLO (p95 TTFT and TPOT), not maximum throughput. Prefill and decode have different costs, and long contexts consume KV cache, which limits concurrency.

#### 3. Compute the fleet
- **Fleet size:** GPUs = peak tokens/s ÷ tokens/s per GPU at SLO, plus headroom (20–30%), N+1 per AZ, and capacity for failover regions.
- **Other components:** include embedding models, rerankers, vector DB memory (vectors × dims × bytes plus index overhead), and the gateway.

#### 4. Optimise and decide
- **Caching:** prefix and response caching lower the effective load.
- **Mix:** model routing (share of traffic on small models), batch vs interactive split (batch fills off-peak), reserved vs on-demand vs spot, and self-hosting vs API (break-even at sustained utilisation).

#### 5. Iterate
- **Forecasting:** tie it to product roadmap and growth. Monitor utilisation, queue depth and SLO burn, and re-plan quarterly given GPU lead times.

### 29.36 Design a multi-tenant AI chatbot platform where each business gets a custom chatbot.

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

### 29.37 Design an AI meeting summarizer system for thousands of meetings daily.

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

### 29.38 Design an AI notification system that prioritizes instead of broadcasting.

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

### 29.39 Design an AI-powered anomaly detection system for cloud infrastructure.

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

### 29.40 Design an AI-powered document processing pipeline for financial institutions.

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

### 29.41 Design an AI dynamic pricing engine.

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
- **Guardrails:** price floors and ceilings, rate-of-change limits, kill switches, and anomaly detection (competitor-scrape glitches can cause price wars or $0.01 errors).
- **Ethics and legal:** avoid discriminatory personalised pricing on protected attributes, comply with price-gouging laws during emergencies, and maintain transparency.
- **LLM role:** minor, such as competitor product matching (entity resolution) and explaining price changes to merchandisers.

#### Evaluation
- **Online:** A/B or switchback tests on revenue, margin and conversion, plus long-term customer trust metrics.

### 29.42 Design an AI resume screening system that handles 100K applications per week.

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

### 29.43 Design an AI voice assistant architecture.

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

### 29.44 Design a multi-agent workflow system where agents collaborate on complex tasks.

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

### 29.45 Design a real-time AI transcription system for concurrent audio streams.

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

### 29.46 Design an AI-powered live streaming content moderation system.

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
- **Cost control:** analysing every frame of 100K streams is unaffordable. Use adaptive sampling, cheap first-pass models, and escalate to heavier VLM or LLM analysis only for flagged segments. Prioritise by reach (viewer count × risk).
- **Latency:** the end-to-end detect-to-action target is under 10 s for severe categories. Use stream processing (Kafka/Flink) and colocate inference with ingest.
- **Temporal reasoning:** single frames mislead, so aggregate over windows and use video-clip models for actions (fights, self-harm).
- **Human reviewers:** live queues with stream playback from the flagged timestamp, plus SLAs and reviewer-wellness tooling.
- **Known-bad content:** hash matching for re-streamed known illegal content (PDQ/video hashing).
- **Feedback loop:** reviewer decisions and appeals produce labels for retraining. Monitor per-category precision/recall and language coverage.

**Tradeoffs:** false positives (ending a legitimate stream hurts creators) vs false negatives (harm and regulatory risk). Use graduated actions (warning → blur → end) and severity-specific thresholds.

### 29.47 Design a conversational assistant (memory, context management, safety, latency).

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

### 29.48 Design a coding agent.

**Goal:** an agent that takes a task ("fix this failing test", "add an endpoint") and produces a verified code change in a real repository.

**Core loop (ReAct-style):** plan → act with tools → observe the results → repeat until tests pass or budget is exhausted → present a diff or PR.

**Tools**

- Repo navigation: `ls`, `grep`/ripgrep, code search (symbol index via tree-sitter / LSP, plus embeddings for semantic search).
- File read and **edit** (search-and-replace or patch edits rather than whole-file rewrites, which saves tokens and reduces clobbering).
- **Shell execution in a sandbox:** run tests, linters, type-checkers, build.
- Optional: git operations, web or documentation lookup, issue tracker access.

**Context management:** repos don't fit in context. Retrieve just enough: a repo map (file tree plus key signatures), files relevant to the task, and test output. Summarize long tool outputs, keep a scratchpad or plan file, and compact the history on long tasks.

**Verification is the key differentiator:** the agent writes or reproduces a failing test first, then makes the change, then runs the tests, lint and type checks. Execution feedback is the reward signal. It retries with the error output as context, and caps iterations and tokens.

**Safety and sandboxing:** run in an isolated container or VM with no production credentials, network egress restricted, resource limits and timeouts. Any destructive action (force push, delete, deploy) requires human approval. Treat repository content and web pages as untrusted (prompt injection via README or comments). Human review of the final PR.

**Architecture options:** a single agent with good tools is often best. Sub-agents (explorer, implementer, reviewer) help for large tasks and keep each one's context clean. Use a strong model for planning and editing, and cheaper models for summarizing and search.

**Evaluation:** SWE-bench-style task suites (resolved rate), internal golden tasks, metrics for test pass rate, diff size, number of iterations, cost per task, human acceptance rate of PRs, and regressions caught in CI. Trace every step to debug failure modes such as wrong file edited, test hallucination, or loops.

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

---

← [Chapter 28. Machine Learning System Design](28-machine-learning-system-design.md) · [Table of contents](../README.md) · [Chapter 30. AI Safety, Security and Responsible AI](30-ai-safety-security-and-responsible-ai.md) →
