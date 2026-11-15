<!-- notion-id: 3d46741828548064a95ad6313aa2da08 -->
# AI Engineer – Senior Level

Prep for a senior AI engineer role, grouped by area: agentic systems, agent protocols, RAG, computer vision, deployment and MLOps, cloud, foundations, process and responsible AI.

## Agentic Systems (LangChain, LangGraph, Python/Pydantic)

### Q1. What's the difference between building an agent "from scratch" in Python vs. using a framework like LangChain or LangGraph? When would you choose one over the other?
Building from scratch means you own the control loop yourself — you define how the LLM decides which tool to call, how you parse and validate its output (typically with Pydantic models), how you manage state/memory, and how you handle retries and errors. This gives full control and avoids framework overhead/abstraction leakage, which matters in latency- or cost-sensitive production systems.

LangChain gives you pre-built abstractions for chains, tool-calling, memory, and retrievers — good for fast prototyping and standard patterns. LangGraph goes further: it models the agent as an explicit state graph (nodes = steps/agents, edges = transitions, conditional routing), which is far better for multi-agent orchestration, cyclic reasoning (ReAct-style loops), human-in-the-loop checkpoints, and persistence of long-running workflows.

In practice: prototype fast with LangGraph, but for a narrow, high-reliability production agent with tight latency/cost budgets, a hand-rolled orchestration loop with Pydantic-validated I/O is often more maintainable and debuggable.

### Q2. Why is Pydantic important for structured outputs from an LLM, and how do you enforce it?
LLMs return free text by default, which is unreliable for programmatic use. Pydantic models let you define an explicit schema (types, required fields, validators) that the LLM's output must conform to. You enforce this via:

- Function/tool-calling APIs (e.g., OpenAI/Anthropic tool-use) where the schema is passed as a JSON schema and the model returns structured arguments.
- `instructor`style libraries or LangChain's `with_structured_output`, which wrap the call and auto-parse/validate the response into a Pydantic object, retrying on validation failure.
- Custom validators (`@field_validator`) to catch domain-specific constraints (e.g., a date must be in the future, an ID must exist).

This gives you type safety, reduces hallucinated formats, and makes the LLM output a first-class object you can pass downstream without brittle regex parsing.

### Q3. Describe how you'd design a multi-agent system for a real task (e.g., a research-and-report agent).
Typical pattern:

- **Orchestrator/supervisor agent**: decomposes the task, routes sub-tasks to specialist agents, and merges results.
- **Specialist agents**: e.g., a retrieval agent, a web-search agent, a writer agent, a critic/reviewer agent — each with a narrow tool set and system prompt.
- **Shared state**: in LangGraph, this is an explicit `State` object (Pydantic/TypedDict) passed between nodes; conditional edges route based on state (e.g., "if critic rejects, loop back to writer").
- **Termination/guardrails**: max iteration counts, confidence thresholds, human-in-the-loop approval for high-stakes actions.
- **Observability**: every agent call is traced (see tracing question below) so you can debug which agent made which decision.

Key design principle: keep each agent's responsibility narrow — broad, do-everything agents are harder to prompt reliably and harder to evaluate.

### Q4. How do you handle an agent that gets stuck in a loop or calls a tool incorrectly?
Guardrails: max-turn/max-tool-call limits, timeout budgets, and explicit "stop conditions" in the graph. Validate tool arguments with Pydantic before execution and return structured error messages back to the model so it can self-correct (rather than crashing). Add a critic/reflection step that checks progress against the goal every N steps. Log everything via tracing so failure patterns can be caught in eval, not just in production.

## Agent Protocols: MCP and A2A

### Q5. What problem does MCP (Model Context Protocol) solve, and how does it differ from a custom tool integration?
Before MCP, every agent framework needed bespoke, one-off integration code for every tool/data source it connected to — an N×M problem (N agent frameworks × M tools). MCP standardizes this via a client-server protocol: an MCP **server** exposes tools, resources, and prompts in a uniform schema; any MCP-compatible **client** (agent) can discover and call them without custom glue code. It's essentially "USB-C for AI tools" — write the integration once, and any compliant agent can use it. This matters for maintainability at scale, especially in a consulting context where you're plugging agents into many different client systems.

### Q6. What is A2A (Agent-to-Agent) protocol, and how is it different from MCP?
MCP standardizes how an agent talks to *tools and data sources*. A2A standardizes how *agents talk to each other* — enabling agents built on different frameworks or by different vendors to discover each other's capabilities (via an "agent card"), negotiate a task, and exchange messages/results in a common format. Think of MCP as vertical (agent → tool) and A2A as horizontal (agent → agent). Together they let you build heterogeneous multi-agent ecosystems rather than a single monolithic framework.

### Q7. What security/reliability considerations come up when connecting an agent to external tools via MCP?
Authentication/authorization scoping (least-privilege access per tool), input validation on tool arguments before execution (prevent injection), sandboxing tool execution, rate-limiting, and auditing/logging every tool call for traceability. Also: treating any content returned from a tool (e.g., a fetched webpage) as untrusted data, not as instructions — to avoid prompt injection through tool outputs.

## RAG Pipelines & Vector Databases

### Q8. Walk through the architecture of a production-grade RAG pipeline using hybrid retrieval and reranking.
1. **Ingestion**: chunk documents (semantic or fixed-size with overlap), generate dense embeddings (e.g., via an embedding model), and index into a vector DB; also build a keyword/sparse index (BM25) over the same chunks.
2. **Query time — hybrid retrieval**: run the query against both the dense (vector similarity) and sparse (keyword/BM25) indexes in parallel, then fuse results (e.g., Reciprocal Rank Fusion). This captures both semantic similarity and exact-term matches (important for names, IDs, jargon that embeddings can miss).
3. **Reranking**: pass the top-K fused candidates through a cross-encoder reranker (which jointly scores query+passage pairs, unlike bi-encoder embeddings) to reorder by true relevance — this is more accurate but more expensive, hence applied only to the shortlist.
4. **Context assembly**: select the top-N reranked chunks within a token budget, optionally with source metadata for citation.
5. **Generation**: pass context + query to the LLM with a prompt that instructs it to ground answers strictly in the provided context and cite sources.
6. **Grounding/hallucination checks**: optionally a post-hoc verification step (e.g., a smaller model or rule-based check) confirming claims are supported by retrieved context.

### Q9. Why combine dense vector search with keyword (sparse) search instead of using just one?
Dense embeddings capture semantic/conceptual similarity but can miss exact matches on rare terms, product codes, acronyms, or numbers, since these get "smoothed over" in embedding space. Keyword search (BM25) is precise on exact terms but misses paraphrases and semantic relationships. Hybrid search gets the best of both — semantic recall plus lexical precision — and is especially important in technical or domain-specific corpora (legal, medical, enterprise data) where terminology precision matters.

### Q10. Why do you need a reranker if you already have a good retriever?
Bi-encoder embedding search encodes the query and each document independently, so it can't model fine-grained interactions between them — it's fast but approximate. A cross-encoder reranker feeds the query and candidate together into one model, letting it attend to their interaction directly, which produces a much more accurate relevance score. It's too slow to run over the whole corpus, so the standard pattern is: cheap retrieval (bi-encoder + BM25) to get a shortlist of \~50-100 candidates, then expensive-but-accurate reranking to pick the true top-N for the LLM's context window.

### Q11. How do you choose between Pinecone, Weaviate, Qdrant, and pgvector for a given client?
Depends on scale, existing infra, and operational requirements:

- **pgvector**: best when the client already runs Postgres and wants to avoid a new piece of infrastructure — good for small/medium scale, keeps data consistency (transactions) with the rest of the app, but scales less well for very high-dimensional/high-volume ANN search.
- **Pinecone**: fully managed, minimal ops overhead, strong at scale, good if the client wants a SaaS solution and doesn't want to self-host.
- **Weaviate**: open-source, supports hybrid search natively, good schema/class modeling, can self-host or use managed cloud — good middle ground.
- **Qdrant**: open-source, high performance, strong filtering support, good for self-hosted deployments with fine-grained metadata filtering needs.

Decision drivers: data volume, latency SLAs, need for hybrid search out-of-the-box, self-hosted vs. managed preference, existing infra, and cost.

### Q12. How do you evaluate and reduce hallucination in a RAG system?
Evaluation: build a labeled eval set (question, gold answer, gold source chunks) and measure retrieval metrics (recall@k, MRR) separately from generation metrics (faithfulness/groundedness — does the answer only use retrieved facts — and answer relevance). Use LLM-as-judge or frameworks like RAGAS to automate this at scale. Reduction techniques: hybrid retrieval + reranking (better context = fewer hallucinations), explicit prompting to say "I don't know" when context is insufficient, citation-forcing (require the model to point to source chunks), and post-generation verification against retrieved context.

## Computer Vision

### Q13. How would you approach building an object detection pipeline for a client, and what tools would you use?
Clarify the requirement first: real-time vs. batch, accuracy vs. latency tradeoffs, classes needed (pretrained-sufficient vs. needs fine-tuning), and deployment target (edge vs. cloud). Typical stack: YOLO (v8/v11) for real-time detection given its strong speed/accuracy tradeoff, OpenCV for pre/post-processing (resizing, NMS, drawing boxes), and PyTorch for any custom training/fine-tuning on client-specific data. If classes are non-standard, fine-tune on a labeled dataset (possibly bootstrapped via a foundation vision model or semi-supervised labeling). Package the trained model into a Docker container with a serving layer (e.g., TorchServe, FastAPI + ONNX Runtime, or Triton) for deployment.

### Q14. What's the difference between object detection, image classification, and segmentation, and when would you use each?
Classification assigns a single label to the whole image (e.g., "this image contains a defect"). Object detection localizes and classifies multiple objects with bounding boxes (e.g., "3 defects at these coordinates"). Segmentation goes further — pixel-level masks, either semantic (label each pixel's class) or instance (separate each object instance). Use classification when only presence/absence matters, detection when you need location/counting, and segmentation when precise shape/area matters (e.g., measuring defect size, medical imaging).

### Q15. How do vision-language models (VLMs) fit into a computer vision pipeline alongside traditional CV models?
Traditional CV models (YOLO, classifiers, segmentation nets) are fast, cheap, and precise for well-defined, narrow tasks with labeled training data. VLMs (e.g., GPT-4V-class, LLaVA, Claude with vision) add flexible, multi-modal *understanding* — reasoning about an image in natural language, answering open-ended questions, zero-shot classification of novel categories, or combining visual input with text context (e.g., "does this invoice image match the PO description?"). A common production pattern: use traditional CV for high-volume, well-defined detection/extraction (e.g., OCR, bounding boxes) and a VLM for the reasoning/decision layer on top, or for tasks where labeled training data doesn't exist yet.

### Q16. Describe your approach to an OCR pipeline for messy, real-world documents (e.g., scanned forms or invoices).
Pre-process (deskew, denoise, binarize) with OpenCV/PIL, run detection to localize text regions if layout is complex, then OCR (e.g., Tesseract, or a VLM/document-AI model for higher accuracy on messy layouts). For structured documents, combine OCR with a layout-aware model or a VLM prompted to extract fields into a Pydantic schema, directly bridging CV output into structured data validation. Post-process with confidence thresholds and human-in-the-loop review for low-confidence extractions in production.

## Deployment, MLOps & Observability

### Q17. How do you containerize and deploy an AI/agentic service, and what does your Docker setup typically look like?
Multi-stage Dockerfile: a build stage installing dependencies (pinned versions, using a lockfile) and a slim runtime stage copying only what's needed — keeps images small and reduces attack surface. Environment variables/secrets injected at runtime (never baked into the image). Health-check endpoints for orchestration. For GPU workloads, use a CUDA-compatible base image matching the framework version. Orchestration via Docker Compose for simpler multi-service setups (agent service + vector DB + API gateway) or Kubernetes/Azure Container Apps for production-scale, auto-scaling deployments.

### Q18. What does "observability" mean for an LLM/agent system in production, and what would you actually instrument?
Three pillars, adapted to LLM systems:

- **Tracing**: capture every step of an agent's execution — prompts, tool calls and their arguments/results, intermediate reasoning, latency per step (tools like LangSmith, Arize Phoenix, or OpenTelemetry-based custom tracing). Critical for debugging *why* an agent made a decision.
- **Metrics**: latency (p50/p95), token usage/cost per request, tool-call success/failure rates, retrieval hit rate, error rates.
- **Evaluation pipelines**: continuous/offline eval against labeled test sets (accuracy, groundedness, task success rate) plus online monitoring for drift — e.g., tracking answer quality or user feedback (thumbs up/down) over time to catch degradation from data drift, prompt regressions, or upstream model changes.

### Q19. How do you monitor for and respond to model/data drift in a deployed system?
Track input distribution shifts (e.g., new types of queries/documents appearing that weren't in the original eval set), monitor output quality metrics over time (not just at launch), and set up alerting on metric degradation (e.g., retrieval recall dropping, hallucination-rate estimates rising). Periodically re-run the eval suite against production samples. For RAG specifically, monitor whether the knowledge base is stale relative to the domain it serves and set up a refresh/re-indexing cadence.

### Q20. Walk through how you'd take a PyTorch/TensorFlow model from training to production.
Train and validate with proper train/val/test splits and cross-validation where appropriate; track experiments (e.g., MLflow/W&B) for reproducibility. Export to an optimized inference format (ONNX, TorchScript, or TensorRT for GPU speedups) to decouple serving from the training framework and reduce latency. Containerize with a serving layer (FastAPI, TorchServe, Triton Inference Server). Deploy behind a versioned API with canary/blue-green rollout to catch regressions before full traffic cutover. Set up monitoring (above) and a rollback plan.

## Cloud & Infrastructure

### Q21. What does working knowledge of Azure for AI/ML deployment typically involve?
Azure ML for training/experiment tracking and managed endpoints; Azure Kubernetes Service (AKS) or Container Apps for containerized service deployment; Azure OpenAI Service for hosted LLM access with enterprise compliance guarantees; Azure Blob Storage for data/artifact storage; Azure Cognitive Search for hybrid retrieval-as-a-service; and Azure Monitor/Application Insights for observability. Familiarity with IAM/RBAC for securing access to client data, and networking (VNETs, private endpoints) for deployments requiring data isolation — a common requirement in consulting engagements with sensitive client data.

### Q22. How do you decide between using a managed cloud LLM/vector-DB service vs. self-hosting?
Factors: data sensitivity/compliance requirements (some clients require data to never leave their tenant, pushing toward self-hosted or VNET-isolated managed services), cost at scale (self-hosting can be cheaper at high volume but adds ops burden), latency requirements, team's operational capacity to maintain infrastructure, and time-to-market pressure (managed services are faster to stand up, important under the "short delivery cycles" constraint typical in consulting).

## ML/DL Foundations & Math

### Q23. What foundational ML/DL concepts should you be able to explain clearly to a non-technical stakeholder, and how would you do it?
E.g., explaining "why did the model get this wrong" in terms of training data coverage, overfitting vs. underfitting, or explaining evaluation metrics (precision/recall tradeoffs) in business terms ("precision = how often we're right when we flag something; recall = how much we catch"). Being able to translate concepts like bias-variance tradeoff, embeddings ("numbers that capture meaning, so similar concepts end up close together"), and confidence/uncertainty into plain language is core to the client-facing part of this role.

### Q24. Where does linear algebra/statistics actually show up in day-to-day LLM/CV engineering work (not just theory)?
Linear algebra: embeddings are vectors, similarity search relies on dot product/cosine similarity, attention mechanisms are matrix multiplications, and dimensionality reduction (PCA) is used for visualizing/analyzing embeddings. Statistics/probability: evaluating model outputs (confidence calibration, sampling temperature/top-p in generation), A/B testing changes to a RAG pipeline or prompt, understanding precision/recall/F1 and ROC curves for CV classifiers, and reasoning about uncertainty in retrieval relevance scores. Calculus underlies backpropagation/gradient descent when fine-tuning models — understanding learning rate effects, loss landscapes, and why training might diverge.

## Data Pipelines, Code Quality & Process

### Q25. How do you build reliable data pipelines for training/inference, and what makes data "clean and reliable" in this context?
Validate schema and types at ingestion (Pydantic again is useful here), handle missing/malformed records explicitly rather than silently dropping them, version datasets so experiments are reproducible, and separate raw/staging/curated layers so transformations are traceable and reversible. For inference pipelines specifically, ensure feature/preprocessing parity between training and serving (a common source of production bugs — "training/serving skew").

### Q26. What does "maintainable Python code with proper testing" look like for you in an AI/agent context, given how non-deterministic LLM outputs are?
Standard practices (type hints, docstrings, linting, CI, version control with meaningful PR/commit history) still apply, plus AI-specific testing: unit tests for deterministic components (parsing, validation, retrieval logic) and separate evaluation suites (not pass/fail unit tests) for the non-deterministic LLM-generation parts — using metrics/thresholds and regression tracking rather than exact-match assertions. Mocking LLM/tool calls in unit tests to keep them fast and deterministic, reserving live-model tests for a smaller, slower integration/eval suite.

### Q27. How do you scope and deliver an AI project under tight client timelines?
Start with a narrow, well-defined MVP scope tied to a measurable success criterion, agreed with the client upfront. Deliver iteratively — a working (even if limited) demo early, then expand capability/robustness in subsequent iterations, incorporating client feedback. Make technical tradeoffs explicit to stakeholders (e.g., "we can ship a rule-based version this week, or a fine-tuned model in three weeks with better accuracy") so scope decisions are informed, not just imposed.

## Ethics, Privacy & Responsible AI

### Q28. What privacy/security considerations are specific to building GenAI systems with client data?
Data minimization (only send necessary context to third-party LLM APIs), understanding and respecting data residency/compliance requirements (e.g., GDPR, HIPAA depending on client domain), using enterprise/private endpoints (e.g., Azure OpenAI with no data retention) rather than consumer APIs for sensitive data, redacting/anonymizing PII before it enters prompts or logs where possible, and being deliberate about what gets stored in vector DBs (embeddings can sometimes leak information about underlying text) and for how long.

### Q29. How do you think about responsible AI / ethical considerations when deploying an agent that can take autonomous actions (e.g., via tool calls)?
Apply least-privilege scoping to what tools/actions an agent can take autonomously vs. what requires human approval (especially for irreversible or high-stakes actions — sending emails, making purchases, modifying records). Build in human-in-the-loop checkpoints for high-risk steps. Log and audit all agent actions for traceability/accountability. Evaluate for and monitor bias in model outputs, especially in client-facing decisions. Be transparent with clients and end-users about when they're interacting with an AI system and its limitations.

## Quick-Reference Summary Table

| Area | Key Tools/Concepts |
| --- | --- |
| Agent frameworks | LangChain, LangGraph, custom Python + Pydantic |
| Agent protocols | MCP (agent↔tool), A2A (agent↔agent) |
| RAG | Hybrid retrieval (dense + BM25), cross-encoder reranking, groundedness eval |
| Vector DBs | Pinecone, Weaviate, Qdrant, pgvector |
| Computer vision | OpenCV, YOLO, PIL, TensorFlow/PyTorch, VLMs, OCR |
| Deployment | Docker, Kubernetes/Azure Container Apps, ONNX/TorchScript |
| Observability | Tracing (LangSmith/Phoenix), metrics, eval pipelines, drift monitoring |
| Cloud | Azure ML, Azure OpenAI, AKS, Azure Cognitive Search |
| Foundations | Bias-variance, linear algebra (embeddings/attention), probability/stats, backprop |
| Process | Data pipeline hygiene, testing (unit + eval suites), iterative delivery, responsible AI |
