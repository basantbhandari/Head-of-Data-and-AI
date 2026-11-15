<!-- notion-id: 3cd674182854800ea6ead9f987aba504 -->
# Senior Software Engineer - Applied AI

Short-answer prep for an applied-AI engineering role: agents, tool calling, RAG, extraction, evaluation, cost, safety, observability and frameworks.

### Q1. What distinguishes an agent workflow from a single-turn Q&A system?
An agent workflow loops through plan → act → observe → decide across multiple steps, using intermediate results to inform the next action, rather than producing one response to one input.

### Q2. How does tool/function calling work in an LLM application?
The LLM is given a set of predefined actions described by a JSON schema. It requests an action with structured arguments; the application executes that action outside the model and returns the result to the LLM to continue reasoning.

### Q3. What problem do MCP servers solve?
They provide a standardized protocol for exposing tools and data to LLMs, removing the need to build a custom integration for every individual tool.

### Q4. What is a RAG pipeline, at a high level?
It retrieves relevant documents for a query, injects them into the prompt as context, and grounds the model's answer in those real sources rather than relying purely on parametric knowledge.

### Q5. Why does chunking strategy matter so much in RAG?
How documents are split — fixed-size, semantic, or header-based — directly determines what gets retrieved; poor chunking breaks context and degrades retrieval quality regardless of how good the embedding model is.

### Q6. Why can't you use one embedding model for every domain?
Embedding models encode different notions of similarity; a model tuned for general text may miss domain-specific meaning (e.g., legal language), so the model must be selected to fit the domain.

### Q7. What is hybrid search and why combine methods?
It combines semantic (vector) search with keyword search (BM25) to get the best coverage — semantic search catches conceptual matches, keyword search catches exact terms vectors can miss.

### Q8. What role does reranking play in a retrieval pipeline?
A second-pass model re-scores the initial candidate documents for relevance, refining the ranking beyond what the first-pass retriever produced.

### Q9. How should retrieval quality be judged?
By whether the right documents were actually found, not by how confident the final model output sounds — confidence is not evidence of correctness.

### Q10. What is an extraction pipeline?
A system that turns unstructured documents — PDFs, filings, etc. — into structured JSON or database rows.

### Q11. What does schema enforcement mean, and how is it achieved?
Forcing model output into a predefined structure, using tools like Instructor, Outlines, or constrained decoding, so downstream systems can rely on consistent output shape.

### Q12. What is a validation/repair loop?
An automated process that detects failed validation and retries or corrects the output, instead of letting the extraction fail silently.

### Q13. What is a golden dataset used for?
It's a set of ground-truth input/output pairs used to measure system performance objectively.

### Q14. What is a regression suite in CI for an LLM system?
Automated tests that run in the CI pipeline to catch regressions caused by prompt or code changes before they reach production.

### Q15. What does "LLM-as-judge calibration" mean?
Tuning one LLM to grade the output of another, adjusting it so its judgments align with human judgment.

### Q16. How should accuracy and hallucination be tracked over time?
Through dashboarded, ongoing metrics — not manual spot-checks — so trends and regressions are visible systematically.

### Q17. What is model routing?
Sending simple tasks to cheaper models and reserving frontier models only for tasks that actually need that capability.

### Q18. What is prompt caching?
Reusing the processed, static prefix of a prompt across multiple calls instead of reprocessing it every time.

### Q19. How does semantic caching differ from standard caching?
It caches responses by the meaning of the request rather than requiring an exact text match, so semantically similar queries can hit the cache.

### Q20. What is batching used for?
Grouping requests for efficiency in workloads that don't require real-time responses.

### Q21. What is context window budgeting?
Deliberately managing what content fits inside the context window to balance cost against accuracy.

### Q22. What is PII detection/redaction, and when should it happen?
Identifying and masking sensitive personal data before it's logged or sent to external APIs.

### Q23. What is prompt injection, and why is it a risk?
Hidden malicious instructions embedded in inputs or documents that attempt to override the system's intended behavior; defending against it is essential for safety.

### Q24. What are grounded citations?
Forcing the model to produce traceable citations back to source material, so claims can be verified.

### Q25. What is a confidence threshold used for?
Deciding whether to proceed automatically or flag an output for review, based on the system's estimated certainty.

### Q26. When should outputs be escalated to a human?
When confidence is low, or the stakes of the decision are high — those cases should route to human reviewers rather than auto-resolving.

### Q27. Why use tracing tools like Langfuse, LangSmith, or OpenTelemetry?
They give full visibility into prompts, tool calls, and retrievals, which is necessary for debugging complex multi-step LLM systems.

### Q28. What is multi-step planning in an agent?
Breaking a complex task down into an ordered sequence of sub-steps before or while executing it.

### Q29. Why do agents need state and memory?
To maintain context across multiple turns or steps within a session, so later actions can build on earlier ones.

### Q30. What does good error recovery look like in an agent?
The agent retries, replans, or falls back to an alternative approach instead of simply breaking when something fails.

### Q31. What is a human-in-the-loop checkpoint?
A deliberate pause in the agent's flow that requires human approval before it takes a high-stakes action.

### Q32. What does "prompt versioning" mean?
Treating prompts like code — tracked in version control and rollback-able — rather than editing them ad hoc in production.

### Q33. What is dynamic few-shot selection?
Picking the best few-shot examples for each specific input, rather than using a fixed static set for every request.

### Q34. What does AWS Bedrock provide?
Unified API access to multiple foundation models through a single platform.

### Q35. Why do teams aim for a multi-provider experience?
To avoid vendor lock-in by keeping the system able to work across providers like OpenAI, Anthropic, and Bedrock.

### Q36. What is LangGraph designed for?
Building stateful, graph-based agent workflows with explicit branching logic.

### Q37. What is LlamaIndex primarily focused on?
Data ingestion and constructing RAG pipelines.

### Q38. What does Pydantic AI offer?
A lightweight, type-safe way to define structured outputs and tools.

### Q39. What is the purpose of fine-tuning/distillation techniques like LoRA/PEFT?
To efficiently train smaller models for narrow, high-volume tasks without the cost of full fine-tuning.

### Q40. What are vision-language models (VLMs) used for in this context?
Processing images and text together, such as scanned documents or forms, where layout and visual content matter.

### Q41. What is an OCR pipeline?
A process that converts scanned images into machine-readable text.

### Q42. When is synthetic data generation used?
When real training or evaluation data is scarce, an LLM can generate synthetic data to fill the gap.

### Q43. How do knowledge graphs/ontologies differ from flat text chunks for retrieval?
They capture relationships between entities, enabling relationship-aware retrieval instead of treating documents as isolated flat text.

### Q44. Why do regulated domains like fintech, payroll, tax, and HR require special handling?
Because accuracy, auditability, and compliance are non-negotiable requirements, not optional quality goals
