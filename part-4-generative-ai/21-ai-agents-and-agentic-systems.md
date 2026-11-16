# Chapter 21. AI Agents and Agentic Systems

_Part IV — Generative AI and LLM Engineering_

Tool use, planning, memory, multi-agent orchestration, MCP and A2A, frameworks and agent reliability.

This chapter contains **64 questions**.

## Contents

- **21.1** [What is an AI agent, and how does it differ from a simple LLM call?](#211-what-is-an-ai-agent-and-how-does-it-differ-from-a-simple-llm-call)
- **21.2** [How does memory work in AI agents?](#212-how-does-memory-work-in-ai-agents)
- **21.3** [What is harness engineering in AI?](#213-what-is-harness-engineering-in-ai)
- **21.4** [Explain the ReAct (Reasoning + Acting) agent architecture.](#214-explain-the-react-reasoning--acting-agent-architecture)
- **21.5** [What is the Plan-and-Execute agent pattern?](#215-what-is-the-plan-and-execute-agent-pattern)
- **21.6** [What is tool use (function calling) in LLMs, and how does it enable agents?](#216-what-is-tool-use-function-calling-in-llms-and-how-does-it-enable-agents)
- **21.7** [How do you design and define tools for an AI agent?](#217-how-do-you-design-and-define-tools-for-an-ai-agent)
- **21.8** [What is the difference between single-agent and multi-agent systems?](#218-what-is-the-difference-between-single-agent-and-multi-agent-systems)
- **21.9** [What is Model Context Protocol (MCP), and how does it standardize tool integration?](#219-what-is-model-context-protocol-mcp-and-how-does-it-standardize-tool-integration)
- **21.10** [What are AI SubAgents?](#2110-what-are-ai-subagents)
- **21.11** [What are the different types of agent memory (short-term, long-term, episodic)?](#2111-what-are-the-different-types-of-agent-memory-short-term-long-term-episodic)
- **21.12** [How do you handle agent failures and implement error recovery?](#2112-how-do-you-handle-agent-failures-and-implement-error-recovery)
- **21.13** [What is an agent loop, and how does it decide when to stop?](#2113-what-is-an-agent-loop-and-how-does-it-decide-when-to-stop)
- **21.14** [What is context engineering?](#2114-what-is-context-engineering)
- **21.15** [How does context compaction work?](#2115-how-does-context-compaction-work)
- **21.16** [How do AI Agents Communicate?](#2116-how-do-ai-agents-communicate)
- **21.17** [What are Agent Skills?](#2117-what-are-agent-skills)
- **21.18** [How do you evaluate and test AI agents?](#2118-how-do-you-evaluate-and-test-ai-agents)
- **21.19** [What are the security risks of agentic systems, and how do you mitigate them?](#2119-what-are-the-security-risks-of-agentic-systems-and-how-do-you-mitigate-them)
- **21.20** [What is the difference between reactive and proactive agents?](#2120-what-is-the-difference-between-reactive-and-proactive-agents)
- **21.21** [How do you manage token consumption and cost in long-running agent workflows?](#2121-how-do-you-manage-token-consumption-and-cost-in-long-running-agent-workflows)
- **21.22** [What is the human-in-the-loop pattern for agents, and when is it needed?](#2122-what-is-the-human-in-the-loop-pattern-for-agents-and-when-is-it-needed)
- **21.23** [How do you implement guardrails for AI agents to prevent harmful actions?](#2123-how-do-you-implement-guardrails-for-ai-agents-to-prevent-harmful-actions)
- **21.24** [What is agent reflection, and how does it improve agent performance?](#2124-what-is-agent-reflection-and-how-does-it-improve-agent-performance)
- **21.25** [What is the difference between code-generating agents and tool-calling agents?](#2125-what-is-the-difference-between-code-generating-agents-and-tool-calling-agents)
- **21.26** [How do you handle multi-modal inputs and outputs in agentic systems?](#2126-how-do-you-handle-multi-modal-inputs-and-outputs-in-agentic-systems)
- **21.27** [How do you implement state management in complex agent workflows?](#2127-how-do-you-implement-state-management-in-complex-agent-workflows)
- **21.28** [How do you build a customer support agent with escalation logic?](#2128-how-do-you-build-a-customer-support-agent-with-escalation-logic)
- **21.29** [What is agent orchestration, and how do you implement it?](#2129-what-is-agent-orchestration-and-how-do-you-implement-it)
- **21.30** [How do you build a code execution agent safely using sandboxed environments?](#2130-how-do-you-build-a-code-execution-agent-safely-using-sandboxed-environments)
- **21.31** [Your AI agent is stuck in an infinite loop. How do you detect and break the cycle?](#2131-your-ai-agent-is-stuck-in-an-infinite-loop-how-do-you-detect-and-break-the-cycle)
- **21.32** [Your AI agent gets conflicting answers from different tools. How does it reconcile them?](#2132-your-ai-agent-gets-conflicting-answers-from-different-tools-how-does-it-reconcile-them)
- **21.33** [Your AI agent burns too many tokens per task. How do you reduce token consumption?](#2133-your-ai-agent-burns-too-many-tokens-per-task-how-do-you-reduce-token-consumption)
- **21.34** [Your AI agent keeps exceeding its budget per task. How do you enforce budget limits?](#2134-your-ai-agent-keeps-exceeding-its-budget-per-task-how-do-you-enforce-budget-limits)
- **21.35** [Your AI agent hallucinates tool capabilities and passes wrong inputs. How do you fix it?](#2135-your-ai-agent-hallucinates-tool-capabilities-and-passes-wrong-inputs-how-do-you-fix-it)
- **21.36** [Your AI agent deleted a production database. How do you prevent irreversible actions?](#2136-your-ai-agent-deleted-a-production-database-how-do-you-prevent-irreversible-actions)
- **21.37** [Your AI agent has many tools, but keeps picking the wrong one. How do you improve tool selection?](#2137-your-ai-agent-has-many-tools-but-keeps-picking-the-wrong-one-how-do-you-improve-tool-selection)
- **21.38** [Your AI agent takes too long to complete a task. How do you speed it up?](#2138-your-ai-agent-takes-too-long-to-complete-a-task-how-do-you-speed-it-up)
- **21.39** [Your LLM selects the right tool but extracts the wrong parameters. How do you fix parameter extraction?](#2139-your-llm-selects-the-right-tool-but-extracts-the-wrong-parameters-how-do-you-fix-parameter-extraction)
- **21.40** [How do Computer-Use Agents work?](#2140-how-do-computer-use-agents-work)
- **21.41** [How does LangChain work?](#2141-how-does-langchain-work)
- **21.42** [How does LangGraph work?](#2142-how-does-langgraph-work)
- **21.43** [What is OKF (Open Knowledge Format)?](#2143-what-is-okf-open-knowledge-format)
- **21.44** [What is the Model Context Protocol (MCP)?](#2144-what-is-the-model-context-protocol-mcp)
- **21.45** [What is an AI agent?](#2145-what-is-an-ai-agent)
- **21.46** [What is function calling?](#2146-what-is-function-calling)
- **21.47** [What are hooks, coroutines, plugins, rules and skills in agent tooling?](#2147-what-are-hooks-coroutines-plugins-rules-and-skills-in-agent-tooling)
- **21.48** [What distinguishes an agent workflow from a single-turn Q&A system?](#2148-what-distinguishes-an-agent-workflow-from-a-single-turn-qa-system)
- **21.49** [How does tool/function calling work in an LLM application?](#2149-how-does-toolfunction-calling-work-in-an-llm-application)
- **21.50** [What problem do MCP servers solve?](#2150-what-problem-do-mcp-servers-solve)
- **21.51** [When should outputs be escalated to a human?](#2151-when-should-outputs-be-escalated-to-a-human)
- **21.52** [What is multi-step planning in an agent?](#2152-what-is-multi-step-planning-in-an-agent)
- **21.53** [Why do agents need state and memory?](#2153-why-do-agents-need-state-and-memory)
- **21.54** [What does good error recovery look like in an agent?](#2154-what-does-good-error-recovery-look-like-in-an-agent)
- **21.55** [What is a human-in-the-loop checkpoint?](#2155-what-is-a-human-in-the-loop-checkpoint)
- **21.56** [What is LangGraph designed for?](#2156-what-is-langgraph-designed-for)
- **21.57** [What does Pydantic AI offer?](#2157-what-does-pydantic-ai-offer)
- **21.58** [What's the difference between building an agent "from scratch" in Python vs. using a framework like LangChain or LangGraph? When would you choose one over the other?](#2158-whats-the-difference-between-building-an-agent-from-scratch-in-python-vs-using-a-framework-like-langchain-or-langgraph-when-would-you-choose-one-over-the-other)
- **21.59** [Why is Pydantic important for structured outputs from an LLM, and how do you enforce it?](#2159-why-is-pydantic-important-for-structured-outputs-from-an-llm-and-how-do-you-enforce-it)
- **21.60** [Describe how you'd design a multi-agent system for a real task (e.g., a research-and-report agent).](#2160-describe-how-youd-design-a-multi-agent-system-for-a-real-task-eg-a-research-and-report-agent)
- **21.61** [How do you handle an agent that gets stuck in a loop or calls a tool incorrectly?](#2161-how-do-you-handle-an-agent-that-gets-stuck-in-a-loop-or-calls-a-tool-incorrectly)
- **21.62** [What problem does MCP (Model Context Protocol) solve, and how does it differ from a custom tool integration?](#2162-what-problem-does-mcp-model-context-protocol-solve-and-how-does-it-differ-from-a-custom-tool-integration)
- **21.63** [What is A2A (Agent-to-Agent) protocol, and how is it different from MCP?](#2163-what-is-a2a-agent-to-agent-protocol-and-how-is-it-different-from-mcp)
- **21.64** [What security/reliability considerations come up when connecting an agent to external tools via MCP?](#2164-what-securityreliability-considerations-come-up-when-connecting-an-agent-to-external-tools-via-mcp)

---

### 21.1 What is an AI agent, and how does it differ from a simple LLM call?

#### What is an AI Agent?

- An **AI agent** is an LLM-based system that can **reason, make decisions, use tools, maintain state, and take actions** to achieve a goal.
- It typically follows a **goal → plan → act → observe → adjust** loop.
- Agents can call tools such as APIs, databases, search engines, code interpreters, or other agents.
- They can make **multiple LLM calls** and decide dynamically what to do next.

#### Simple LLM Call vs AI Agent

| Simple LLM Call | AI Agent |
| --- | --- |
| One request → one response | Goal → multiple steps → result |
| Usually stateless | Can maintain state/memory |
| Generates text | Can take actions |
| No autonomous decision-making | Decides what action to take |
| Usually no tools | Can use tools/APIs |
| Fixed workflow | Dynamic workflow |
| Example: "Summarize this text" | Example: "Research competitors and prepare a report" |

#### Example

**Simple LLM call:**

> User: "What is the weather in Kathmandu?"

The LLM generates an answer based on its knowledge or provided context.

**AI Agent:**

> User: "Plan my trip to Kathmandu."

The agent might:

1. Search for current weather.
2. Search flights and hotels.
3. Check availability.
4. Build an itinerary.
5. Adjust the plan based on budget and preferences.
6. Return the final plan.

**In short:**

> **LLM = generates intelligence/content.**
>
> **AI Agent = uses an LLM as a reasoning engine to decide and execute actions toward a goal.**

### 21.2 How does memory work in AI agents?

- **AI agent memory** allows an agent to retain and use information from previous interactions.
- It helps agents **maintain context, personalize responses, and perform long-running tasks**.
- **Short-term memory:** Stores information during the current task or conversation.
- **Long-term memory:** Stores useful information across different sessions.
- Common implementations include **vector databases, SQL/NoSQL databases, and knowledge graphs**.
- A good memory system **retrieves relevant information when needed and updates or removes outdated information**.

### 21.3 What is harness engineering in AI?

- **Harness Engineering** is the practice of building the **tools, environment, workflows, and guardrails around an AI model/agent** so it can perform tasks reliably.
- It focuses on the **system surrounding the model**, rather than improving the model itself.
- A harness can include **tool calling, memory, context management, testing, monitoring, error handling, and security controls**.
- Example: Giving a coding agent access to a **codebase, terminal, tests, Git, documentation, and automated validation** forms its engineering harness.
- **Goal:** Make AI agents more **reliable, controllable, observable, and effective** in real-world tasks.

### 21.4 Explain the ReAct (Reasoning + Acting) agent architecture.

#### ReAct (Reasoning + Acting) Agent Architecture

- **ReAct** combines **reasoning and actions** so an AI agent can solve tasks step by step.
- The agent **reasons about what to do**, performs an **action/tool call**, observes the result, and then reasons again.
- This creates an iterative **Reason → Act → Observe → Repeat** loop.
- **Reasoning:** Determines the next step or tool needed.
- **Action:** Uses tools such as search, APIs, databases, or code execution.
- **Observation:** Processes the tool's result and decides the next step.
- **Final Answer:** Stops when enough information has been gathered to complete the task.

**Example:**

Question → Reason → Search Web → Observe results → Reason → Search again → Final Answer.

### 21.5 What is the Plan-and-Execute agent pattern?

#### Plan-and-Execute Agent Pattern

- **Plan-and-Execute** separates **planning** from **execution**.
- A **planner** first creates a step-by-step plan for completing the task.
- An **executor** then performs each step using tools or actions.
- The executor can **report results back to the planner**, which may update the plan if needed.
- It is useful for **complex, multi-step tasks** where upfront planning improves reliability.

**Flow:**

**User Task → Planner → Step-by-Step Plan → Executor → Actions/Tools → Results → Final Answer**

### 21.6 What is tool use (function calling) in LLMs, and how does it enable agents?

#### Tool Use (Function Calling) in LLMs

- **Tool use** allows an LLM to call external tools such as APIs, databases, search engines, or code execution.
- The LLM generates a **structured function call** with the required arguments.
- The tool executes the request and returns the **result** to the LLM.
- The LLM uses the result to decide the **next action or final response**.
- It enables AI agents to **interact with the real world**, rather than only generating text.

**Flow:**

**User → LLM → Tool/Function Call → Tool Result → LLM → Response/Next Action**

**Example:**

An agent receives *“What is today's weather?”* → calls a weather API → receives current weather data → provides the answer.

### 21.7 How do you design and define tools for an AI agent?

#### Designing and Defining Tools for an AI Agent

- Define each tool with a **clear, specific purpose**.
- Provide a **descriptive name and description** so the LLM knows when to use it.
- Define **structured input parameters** using a schema, including types and required fields.
- Return **structured, predictable outputs** that the LLM can easily interpret.
- Add **validation, error handling, authentication, and permission controls**.
- Keep tools **small and focused** rather than creating one tool that performs many unrelated tasks.
- Give the agent only the **minimum tools and permissions** required for the task.

**Example:**

`get_weather(city: string, date: string)` → returns temperature, conditions, and forecast.

### 21.8 What is the difference between single-agent and multi-agent systems?

#### Single-Agent vs Multi-Agent Systems

- **Single-Agent:** One AI agent handles the entire task using its available tools and memory.
- **Multi-Agent:** Multiple specialized AI agents **collaborate**, each handling a specific role or part of the task.

| Single-Agent | Multi-Agent |
| --- | --- |
| One agent | Multiple agents |
| Simpler architecture | More complex architecture |
| Easier to manage | Requires agent coordination |
| Good for focused tasks | Good for complex tasks |

**Example:**

- Single-agent: One agent researches, analyzes, and writes a report.
- Multi-agent: **Research Agent → Analysis Agent → Writing Agent → Review Agent**.

### 21.9 What is Model Context Protocol (MCP), and how does it standardize tool integration?

#### Model Context Protocol (MCP)

- **MCP** is an open protocol that provides a **standard way for AI applications to connect with external tools, data, and services**.
- It separates the **AI agent from individual tool implementations**, so tools can be integrated consistently.
- MCP defines standardized interfaces for **tools, resources, and prompts**.
- An MCP server exposes capabilities, and the AI client/agent can **discover and invoke them**.
- This reduces the need for **custom integrations** for every tool or data source.

**Flow:**

**AI Agent → MCP Client → MCP Server → Tool/Data → Result → AI Agent**

**Example:** An agent can use an MCP server to access GitHub, databases, files, or APIs through a consistent protocol.

### 21.10 What are AI SubAgents?

#### AI SubAgents

- **AI subagents** are specialized agents that work as **subordinate agents under a main (orchestrator) agent**.
- Each subagent is designed for a **specific task or role**, such as research, coding, data analysis, or validation.
- The main agent **delegates tasks** to subagents and combines their results.
- Subagents can have their own **tools, memory, prompts, and reasoning processes**.
- They improve **task decomposition, specialization, and scalability**.

**Example:**

**Main Agent → Research Subagent → Coding Subagent → Review Subagent → Main Agent → Final Answer**

### 21.11 What are the different types of agent memory (short-term, long-term, episodic)?

#### Types of Agent Memory

- **Short-Term Memory:** Stores information needed during the **current conversation or task**, such as recent messages and intermediate results.
- **Long-Term Memory:** Stores **persistent information across sessions**, such as user preferences, facts, or learned knowledge.
- **Episodic Memory:** Stores **past experiences or events**, allowing an agent to recall what happened in previous tasks or interactions.

**In short:**

**Short-term = current context**

**Long-term = persistent knowledge**

**Episodic = past experiences**

### 21.12 How do you handle agent failures and implement error recovery?

#### Handling Agent Failures and Error Recovery

- **Detect failures:** Monitor tool errors, timeouts, invalid outputs, and unexpected agent behavior.
- **Retry:** Retry temporary failures with **exponential backoff** and a limited number of attempts.
- **Fallback:** Use an alternative tool, model, or strategy when the primary approach fails.
- **Validate outputs:** Check tool responses and agent outputs before using them.
- **Maintain state:** Save checkpoints so the agent can **resume from the last successful step**.
- **Set limits:** Use timeouts, retry limits, and maximum iterations to prevent infinite loops.
- **Human escalation:** Escalate to a human when the agent cannot safely recover.
- **Log and monitor:** Record failures and recovery actions for debugging and improvement.

**In short:**

**Detect → Retry → Validate → Fallback → Recover from checkpoint → Escalate if needed**

### 21.13 What is an agent loop, and how does it decide when to stop?

#### Agent Loop

- An **agent loop** is the repeated cycle an AI agent follows to complete a task.
- The agent **observes the current state, reasons, chooses an action, executes it, and evaluates the result**.
- It continues the loop until the task is completed or a stopping condition is reached.

**Flow:**

**Observe → Reason → Act → Observe → Repeat → Stop**

#### When does it stop?

- The **goal is achieved,** and a final answer can be provided.
- The agent determines **no further action is needed**.
- A **maximum number of iterations** is reached.
- A **timeout, error, or resource limit** occurs.
- A **safety or permission constraint** prevents further execution.

### 21.14 What is context engineering?

#### Context Engineering

- **Context engineering** is the practice of **designing and managing the information provided to an LLM** so it can produce reliable results.
- It involves selecting the **right instructions, conversation history, retrieved knowledge, tools, memory, and examples**.
- The goal is to provide **relevant information while avoiding unnecessary context**.
- It is especially important for **AI agents and RAG systems** with limited context windows.

**In short:**

**Context Engineering = Selecting, structuring, and managing the right context for the LLM at the right time.**

### 21.15 How does context compaction work?

#### Context Compaction

- **Context compaction** reduces a long conversation or agent history into a **shorter, useful representation**.
- It summarizes important information while removing **redundant or less relevant details**.
- The compacted context is then provided back to the LLM, freeing space in the **context window**.
- It helps agents handle **long-running conversations and tasks** without exceeding token limits.
- Important state such as **goals, decisions, tool results, and unfinished tasks** should be preserved.

**In short:**

**Long Context → Summarize & Remove Redundancy → Compact Context → Continue Task**

### 21.16 How do AI Agents Communicate?

#### How AI Agents Communicate

- AI agents communicate by **exchanging structured messages, instructions, results, and context**.
- In a **multi-agent system**, a main agent can delegate tasks to specialized agents and receive their results.
- Communication can happen through **direct messages, APIs, shared state, message queues, or protocols such as MCP**.
- Agents may share **task status, tool results, observations, and required actions**.
- A central **orchestrator** can coordinate the communication and combine agent outputs.

**In short:**

**Agent → Message/Task → Another Agent → Result → Agent/Orchestrator**

### 21.17 What are Agent Skills?

#### Agent Skills

- **Agent skills** are reusable **capabilities or instructions** that teach an AI agent how to perform a specific type of task.
- A skill can define **procedures, tools, rules, and knowledge** needed to complete that task.
- Skills allow agents to be **specialized and reusable** without changing the underlying LLM.
- Examples include **web research, coding, data analysis, file processing, or database operations**.

**In short:**

**Agent Skills = Reusable capabilities that give an AI agent specialized abilities to perform tasks.**

### 21.18 How do you evaluate and test AI agents?

#### Evaluating and Testing AI Agents

- **Task Success:** Check whether the agent completes the requested task correctly.
- **Accuracy:** Evaluate the correctness and relevance of its outputs.
- **Tool Usage:** Check whether it selects and uses the right tools correctly.
- **Reliability:** Test performance across repeated runs and different inputs.
- **Safety:** Test for harmful actions, prompt injection, and unauthorized tool use.
- **Efficiency:** Measure **latency, token usage, number of tool calls, and cost**.
- **Robustness:** Test edge cases, failures, ambiguous inputs, and unexpected tool results.
- **Human Evaluation:** Use human feedback to assess quality, usefulness, and overall behavior.
- **Observability:** Log agent steps, tool calls, errors, and outcomes for debugging.

**In short:**

**Test → Measure → Analyze failures → Improve → Re-test**

### 21.19 What are the security risks of agentic systems, and how do you mitigate them?

#### Security Risks of Agentic Systems and Mitigations

- **Prompt Injection**
  - Malicious instructions in user input, web pages, documents, or retrieved content can manipulate the agent.
  - **Mitigation:** Input/output validation, content isolation, instruction hierarchy, trusted/untrusted context separation, and prompt-injection detection.
- **Excessive Agent Permissions**
  - An agent with broad access to databases, files, APIs, or cloud resources can cause significant damage if compromised.
  - **Mitigation:** Apply **least privilege**, scoped API keys, RBAC/ABAC, sandboxing, and short-lived credentials.
- **Tool/API Abuse**
  - Agents can misuse tools or invoke dangerous operations incorrectly.
  - **Mitigation:** Tool allowlists, parameter validation, schema enforcement, rate limits, and human approval for high-impact actions.
- **Data Leakage**
  - Agents may expose confidential information through responses, tool calls, logs, or retrieved documents.
  - **Mitigation:** Data classification, access controls, PII/secret filtering, encryption, and preventing sensitive data from entering prompts unnecessarily.
- **Memory Poisoning**
  - Malicious or incorrect information can be stored in long-term agent memory and influence future decisions.
  - **Mitigation:** Validate memory before persistence, separate trusted from untrusted memories, apply provenance, expiration, and user-controlled deletion.
- **Agent-to-Agent Attacks**
  - In multi-agent systems, one compromised agent can manipulate other agents or spread malicious instructions.
  - **Mitigation:** Authenticate agents, enforce capability-based permissions, validate inter-agent messages, and isolate agent roles.
- **Autonomous Action Risks**
  - Agents may take unintended actions such as modifying data, sending messages, or deploying code.
  - **Mitigation:** Human-in-the-loop approval, action limits, dry-run modes, transaction boundaries, and reversible operations.
- **RAG/Data Poisoning**
  - Attackers can inject malicious or misleading content into the knowledge base.
  - **Mitigation:** Trusted ingestion pipelines, document provenance, source validation, content scanning, versioning, and retrieval filtering.
- **Unauthorized Privilege Escalation**
  - An agent may attempt to use one tool or identity to gain access to resources beyond its intended scope.
  - **Mitigation:** Strong authorization at every tool/API boundary; never rely solely on the LLM to enforce permissions.
- **Denial of Service / Resource Abuse**
  - Agents can enter loops, make excessive tool calls, or consume large amounts of tokens and compute.
  - **Mitigation:** Maximum iteration limits, timeouts, token budgets, rate limits, cost controls, and circuit breakers.
- **Supply-Chain Risks**
  - Third-party tools, libraries, models, MCP servers, or plugins can introduce vulnerabilities.
  - **Mitigation:** Dependency scanning, signed/trusted components, sandboxing, security reviews, and strict tool permissions.

### 21.20 What is the difference between reactive and proactive agents?

#### Reactive vs Proactive Agents

- **Reactive Agents**
  - Respond only when they receive an input, event, or change in the environment.
  - They focus on the **current situation** rather than anticipating future needs.
  - Usually simpler and easier to control.
  - Example: A customer-support agent that answers when a user asks a question.
- **Proactive Agents**
  - Can initiate actions without waiting for a direct user request.
  - Monitor context, goals, schedules, or events and **anticipate what should happen next**.
  - Can plan and execute multi-step tasks autonomously.
  - Example: An agent that monitors a data pipeline and automatically investigates and alerts the team when it detects an anomaly.

#### Key Difference

**Reactive = "Something happened → respond."**

**Proactive = "Something may need to happen → anticipate, plan, and act."**

### 21.21 How do you manage token consumption and cost in long-running agent workflows?

#### Managing Token Consumption and Cost in Long-Running Agent Workflows

- **Set token budgets**
  - Define maximum input/output tokens per agent run, step, and task.
  - Stop or downgrade the workflow when the budget is exceeded.
- **Use context compaction**
  - Summarize older conversation history and retain only relevant information.
  - Keep the active context small instead of repeatedly sending the entire history.
- **Use short-term and long-term memory**
  - Store durable information separately from the active context.
  - Retrieve only the memories relevant to the current task.
- **Choose models dynamically**
  - Use smaller/cheaper models for simple tasks such as classification, routing, and extraction.
  - Use stronger models only for complex reasoning or critical decisions.
- **Control agent loops**
  - Set maximum iterations, tool calls, retries, and execution time.
  - Use termination conditions to prevent infinite reasoning/tool-call loops.
- **Optimize tool usage**
  - Cache repeated tool/API results.
  - Avoid redundant searches and duplicate function calls.
  - Batch independent operations when possible.
- **Reduce prompt size**
  - Use concise system prompts and structured tool outputs.
  - Send only relevant documents, retrieved chunks, and conversation history.
- **Optimize RAG**
  - Retrieve fewer but higher-quality chunks.
  - Use reranking and metadata filtering before passing context to the LLM.
- **Use caching**
  - Cache embeddings, retrieval results, repeated prompts, and deterministic tool results where appropriate.
- **Monitor cost and usage**
  - Track tokens, model usage, tool calls, latency, and cost per workflow.
  - Set alerts and hard spending limits.

### 21.22 What is the human-in-the-loop pattern for agents, and when is it needed?

#### Human-in-the-Loop (HITL) Pattern

- **Human-in-the-loop** means an AI agent can perform tasks autonomously but **pauses at important decision points and asks a human for approval, review, or intervention** before continuing.
- **Typical workflow:**
  1. Agent receives a goal.
  2. Agent plans and performs low-risk actions.
  3. Agent reaches a high-risk or irreversible action.
  4. Agent pauses and requests human approval.
  5. Human approves, rejects, or modifies the action.
  6. Agent continues based on the decision.

#### When Is It Needed?

- **High-impact decisions** — financial, legal, medical, or security-related actions.
- **Irreversible actions** — deleting data, deploying production code, sending important communications.
- **External side effects** — making purchases, modifying cloud infrastructure, changing permissions.
- **Low-confidence situations** — when the agent is uncertain or conflicting information exists.
- **Sensitive data** — when handling confidential or personally identifiable information.
- **Security-sensitive operations** — privilege changes, credential access, or suspicious activities.

#### Example

A coding agent can automatically write and test code, but before **deploying to production**, it pauses:

> Agent: "Tests passed. Production deployment is ready. Approve deployment?"

The human approves or rejects it.

#### Key Principle

**Let agents automate low-risk tasks, but require human approval for high-risk, sensitive, uncertain, or irreversible actions.**

### 21.23 How do you implement guardrails for AI agents to prevent harmful actions?

#### Implementing Guardrails for AI Agents

- **Define allowed actions**
  - Create explicit allowlists for tools, APIs, resources, and operations.
  - Deny anything outside the agent's defined capabilities.
- **Apply least-privilege access**
  - Give each agent only the permissions it needs.
  - Use scoped, short-lived credentials and RBAC/ABAC.
- **Validate tool calls**
  - Validate arguments against strict schemas before execution.
  - Check authorization, parameters, and target resources independently of the LLM.
- **Use human approval for high-risk actions**
  - Require approval before actions such as deleting data, making financial transactions, changing permissions, or deploying to production.
- **Separate trusted and untrusted data**
  - Treat user input, retrieved documents, websites, and tool outputs as potentially untrusted.
  - Prevent external content from overriding system-level instructions.
- **Add input/output safety checks**
  - Detect malicious prompts, sensitive-data exposure, unsafe outputs, and policy violations before execution or delivery.
- **Limit agent autonomy**
  - Set maximum iterations, tool calls, execution time, token usage, and resource consumption.
  - Use circuit breakers to stop abnormal behavior.
- **Sandbox dangerous operations**
  - Execute code and file operations in isolated environments with restricted network and filesystem access.
- **Monitor and audit**
  - Log decisions, tool calls, permissions, failures, and important state changes.
  - Use anomaly detection and alerts for suspicious behavior.
- **Make actions reversible**
  - Prefer transactions, dry runs, checkpoints, backups, and rollback mechanisms for operations that can cause damage.

#### Key Principle

**Never rely on the LLM itself as the security boundary.** Guardrails should be enforced externally at the **input, planning, tool-execution, authorization, and output layers**, with human approval for high-impact actions.

### 21.24 What is agent reflection, and how does it improve agent performance?

#### Agent Reflection

- **Agent reflection** is when an AI agent **reviews its own previous actions, reasoning, or results**, identifies mistakes or weaknesses, and then improves its approach before continuing.

#### How It Improves Performance

- **Detects errors** — checks whether the previous result is incorrect or incomplete.
- **Improves reasoning** — revisits assumptions and considers alternative approaches.
- **Validates results** — verifies whether the output satisfies the original goal.
- **Enables self-correction** — fixes mistakes instead of repeating them.
- **Improves planning** — uses feedback from previous steps to choose better next actions.
- **Handles failures** — analyzes failed tool calls or intermediate results and retries appropriately.

#### Example

An agent is asked to analyze a dataset:

1. Agent generates an analysis.
2. It reviews the result and notices missing data was not handled.
3. It identifies the problem.
4. It modifies its approach.
5. It reruns the analysis and produces an improved result.

**In short:**

**Act → Evaluate → Identify problems → Correct → Continue.**

Reflection is particularly useful for **complex, multi-step tasks**, but it should be bounded because excessive reflection increases **latency and token cost**.

### 21.25 What is the difference between code-generating agents and tool-calling agents?

#### Code-Generating Agents vs Tool-Calling Agents

- **Code-Generating Agents**
  - Generate code dynamically to solve a task.
  - Execute the generated code in a runtime or sandbox.
  - Useful for **data analysis, calculations, automation, and complex transformations**.
  - Example: An agent writes Python code to analyze a CSV file and generate statistics.
- **Tool-Calling Agents**
  - Select and invoke **predefined tools or APIs** using structured arguments.
  - The agent does not need to generate the implementation of the operation.
  - Useful for **database queries, API calls, web search, file operations, and business workflows**.
  - Example: An agent calls a `get_customer_orders(customer_id)` API.

#### Key Difference

|  | Code-Generating Agent | Tool-Calling Agent |
| --- | --- | --- |
| Action | Generates and executes code | Invokes predefined tools |
| Flexibility | Higher | More controlled |
| Safety | Requires sandboxing | Easier to restrict |
| Best for | Dynamic computation | Known operations/APIs |
| Example | Generate Python for analysis | Call a database tool |

**In practice, they can be combined:** an agent might call a data-analysis tool, generate Python for a custom calculation, and then call another tool to store the result.

### 21.26 How do you handle multi-modal inputs and outputs in agentic systems?

#### Handling Multimodal Inputs and Outputs in Agentic Systems

- **Normalize inputs**
  - Accept text, images, audio, video, and documents.
  - Convert each modality into a representation the agent can reason over, such as tokens, embeddings, transcripts, or structured metadata.
- **Use modality-specific models**
  - Vision models for images/video.
  - Speech-to-text models for audio.
  - OCR/document parsers for PDFs and scanned documents.
  - LLMs for reasoning and orchestration.
- **Create a unified context**
  - Combine information from different modalities into a structured context.
  - Preserve relationships between modalities, such as an image and its corresponding text description.
- **Route tasks intelligently**
  - Use an orchestrator to determine which model or tool should process each input.
  - Example: image → vision model → extracted information → LLM → database tool.
- **Use multimodal tools**
  - Agents can call tools for image generation, speech synthesis, OCR, video processing, or document analysis.
- **Generate multimodal outputs**
  - Depending on the task, the agent can produce text, images, audio, structured data, or a combination.
  - Example: analyze a chart image → explain the findings in text → generate a summary report.
- **Validate outputs**
  - Check generated content for correctness, format, safety, and consistency across modalities.
  - For critical applications, use deterministic validation or human review.

#### Example

A customer-support agent receives a **photo of a damaged product + text description**:

**Image + Text → Vision Model → Agent Reasoning → Order/CRM Tool → Response**

The agent can identify the visible issue, retrieve the customer's order, determine the appropriate workflow, and respond with a resolution.

**Key principle:** Build a **modality-aware orchestration layer** that routes each input to the right model/tool, converts results into a shared context, and validates the final multimodal output.

### 21.27 How do you implement state management in complex agent workflows?

#### State Management in Complex Agent Workflows

- **Define a shared state**
  - Maintain a structured state object containing things like user input, task status, intermediate results, tool outputs, errors, and agent decisions.
  - Example: `task → plan → tool_results → validation → final_output`.
- **Use explicit state transitions**
  - Model the workflow as states/nodes with well-defined transitions.
  - Example: `Planning → Execution → Validation → Retry → Completion`.
- **Separate state types**
  - **Working state:** temporary information needed during the current task.
  - **Conversation state:** relevant interaction history.
  - **Long-term state:** durable user preferences or knowledge.
  - **Workflow state:** progress, checkpoints, retries, and completed steps.
- **Persist important state**
  - Store workflow state in a database or durable checkpoint store so the agent can resume after failures or restarts.
- **Use checkpoints**
  - Save state after important steps.
  - If a tool call fails, resume from the latest valid checkpoint instead of restarting the entire workflow.
- **Control concurrent updates**
  - Use transactions, versioning, optimistic locking, or state reducers to prevent multiple agents from corrupting shared state.
- **Manage state size**
  - Don't keep the entire history in the active context.
  - Summarize or compact old information and retrieve relevant state when needed.
- **Track provenance**
  - Record where important state came from—user input, tool output, retrieved document, or agent-generated information.
- **Handle failures explicitly**
  - Store error state, retry counts, and recovery information.
  - Define clear transitions such as : `retry → fallback → human_review → failure`.

#### Example

A research agent might maintain:

```txt
State:
  goal
  plan
  completed_steps
  current_step
  retrieved_documents
  tool_results
  errors
  retry_count
  final_answer
```

After each step, the state is updated and checkpointed, allowing the workflow to **pause, resume, retry, or recover** without losing progress.

**Key principle:** Treat agent state as a **durable, structured state machine**, not simply as an ever-growing conversation history.

### 21.28 How do you build a customer support agent with escalation logic?

#### Building a Customer Support Agent with Escalation Logic

- **Define the workflow**
  - `Receive Request → Understand Intent → Retrieve Customer Data → Resolve → Validate → Escalate or Close`
- **Classify the request**
  - Identify intents such as billing, technical issue, refund, account access, or general questions.
  - Assign a confidence score to the classification.
- **Use RAG and tools**
  - Retrieve answers from the knowledge base.
  - Use tools for order status, account information, tickets, and other authorized operations.
- **Define escalation rules**
  - Escalate when:
    - Agent confidence is below a threshold.
    - The customer explicitly requests a human.
    - The issue involves sensitive or high-impact decisions.
    - Multiple automated attempts fail.
    - The knowledge base contains conflicting or insufficient information.
    - The issue requires an action outside the agent's permissions.
- **Implement human handoff**
  - Create or update a support ticket.
  - Transfer the conversation to a human agent.
  - Provide the human with a concise summary, relevant conversation history, retrieved documents, attempted actions, and error information.
- **Use state management**
  - Maintain states such as:
    - `NEW`
    - `INVESTIGATING`
    - `WAITING_FOR_CUSTOMER`
    - `RESOLVED`
    - `ESCALATED`
    - `CLOSED`
- **Prevent escalation loops**
  - Track escalation attempts and retry counts.
  - Define a maximum number of automated retries before mandatory human escalation.
- **Monitor performance**
  - Measure resolution rate, escalation rate, customer satisfaction, response time, and incorrect-resolution rate.
  - Analyze escalated conversations to improve the knowledge base and agent behavior.

#### Example

```txt
Customer: "I was charged twice."

        ↓

Intent Classification
        ↓
Billing Issue
        ↓
Check billing system
        ↓
Duplicate charge confirmed
        ↓
Can agent issue refund?
     /             \
   Yes              No
    ↓                ↓
Process refund    Escalate
    ↓                ↓
Validate          Human Agent
    ↓
Resolve
```

**Key principle:** The agent should **automate routine, low-risk issues while recognizing its limits and escalating uncertain, sensitive, or high-impact cases to humans with enough context to act quickly.**

### 21.29 What is agent orchestration, and how do you implement it?

#### Agent Orchestration

- **Agent orchestration** is the process of **coordinating one or more AI agents, tools, models, and workflows** to accomplish a complex task.
- The orchestrator decides:
  - **Which agent** should handle a task.
  - **Which tool** should be called.
  - **What order** tasks should run in.
  - **What information/state** should be passed between steps.
  - **When to retry, delegate, or escalate**.
  - **When the overall task is complete**.

#### How to Implement It

1. **Define the workflow**
   - Break the objective into clear tasks and dependencies.
   - Example: `Plan → Research → Analyze → Validate → Respond`.
2. **Create specialized agents**
   - Use agents with focused responsibilities.
   - Example: Research Agent, Data Agent, Coding Agent, Review Agent.
3. **Build a state layer**
   - Maintain shared workflow state containing the goal, current step, intermediate results, tool outputs, errors, and completed tasks.
4. **Implement routing**
   - Use rules, classifiers, or an LLM router to decide which agent/tool should handle each task.
5. **Implement communication**
   - Pass structured messages/results between agents rather than entire conversation histories.
6. **Add control logic**
   - Handle retries, timeouts, failures, iteration limits, fallbacks, and human escalation.
7. **Add guardrails**
   - Enforce authentication, authorization, tool permissions, input/output validation, and approval for high-risk actions.
8. **Monitor the workflow**
   - Track agent decisions, tool calls, latency, token usage, failures, and final-task success.

#### Example

```txt
User Request
     ↓
Orchestrator
     ↓
┌──────────────┬──────────────┐
↓              ↓              ↓
Research     Data Agent    Coding Agent
Agent           ↓              ↓
└──────────────┴──────────────┘
             ↓
        Review Agent
             ↓
          Validator
             ↓
        Final Response
```

**Key principle:** The orchestrator acts as the **control plane** of the agentic system—it manages **routing, state, coordination, tool execution, error recovery, and termination**, while specialized agents focus on individual tasks.

### 21.30 How do you build a code execution agent safely using sandboxed environments?

#### Building a Code Execution Agent Safely

- **Use an isolated sandbox**
  - Execute generated code inside a container or VM rather than on the host system.
  - Give each execution an isolated environment.
- **Restrict filesystem access**
  - Provide only a temporary working directory.
  - Prevent access to host files, credentials, SSH keys, environment secrets, and system directories.
- **Disable or restrict networking**
  - Block outbound network access by default.
  - Allow only explicitly required endpoints when necessary.
- **Apply resource limits**
  - Set limits for CPU, memory, disk, processes, execution time, and output size.
  - Prevent infinite loops and resource-exhaustion attacks.
- **Use non-privileged execution**
  - Never run generated code as root or with unnecessary capabilities.
  - Use isolated users and restricted Linux capabilities.
- **Control dependencies**
  - Prefer prebuilt, trusted environments.
  - Don't allow arbitrary package installation or execution of untrusted binaries without additional controls.
- **Validate the code**
  - Perform basic static analysis before execution.
  - Detect obviously dangerous operations and reject or require additional review.
- **Separate agent permissions from sandbox permissions**
  - The LLM may request an operation, but an external policy layer decides whether the sandbox is actually allowed to perform it.
- **Sanitize inputs and outputs**
  - Treat generated code, user-provided files, and tool results as untrusted.
  - Limit returned output and prevent secrets or sensitive data from being exposed.
- **Destroy the environment after execution**
  - Use ephemeral sandboxes and clean them up after each task to prevent persistence between executions.
- **Monitor and audit**
  - Log code execution, resource usage, failures, and security events.
  - Use timeouts and circuit breakers to terminate abnormal execution.

#### Secure Architecture

```txt
User
  ↓
Agent
  ↓
Code Generation
  ↓
Policy / Security Validator
  ↓
Sandbox
  ├── Restricted Filesystem
  ├── Limited CPU / Memory
  ├── Network Disabled
  ├── Non-Privileged User
  └── Execution Timeout
  ↓
Validate Output
  ↓
Agent
  ↓
Final Response
```

**Key principle:** Treat generated code as **untrusted code**. The sandbox should enforce security independently of the agent, using **isolation, least privilege, resource limits, network restrictions, validation, monitoring, and ephemeral environments**.

### 21.31 Your AI agent is stuck in an infinite loop. How do you detect and break the cycle?

#### Detecting and Breaking Agent Infinite Loops

- **Set a maximum iteration limit**
  - Limit the number of agent reasoning/tool-call cycles.
  - Example: stop after 10–20 iterations.
- **Set time and resource limits**
  - Use execution timeouts, token budgets, API-call limits, and cost limits.
  - Terminate the workflow when any limit is exceeded.
- **Detect repeated states**
  - Store hashes or fingerprints of the agent's state, tool calls, and outputs.
  - If the same state appears repeatedly, the agent is likely stuck.
- **Detect repeated actions**
  - Track recent tool calls.
  - Example: if the agent repeatedly calls `search → same result → search → same result,`trigger loop detection.
- **Track progress**
  - Define measurable progress criteria.
  - If several iterations produce no new information or state change, terminate or switch strategies.
- **Use cycle detection**
  - Model the workflow as a state graph and detect cycles such as:`A → B → C → A`.
- **Use retry limits**
  - Don't retry the same failed operation indefinitely.
  - After a small number of retries, use a fallback or escalate.
- **Force a recovery strategy**
  - When a loop is detected:
    1. Stop the current loop.
    2. Summarize the current state.
    3. Try an alternative approach, or
    4. Escalate to a human/failure state.

#### Example

```txt
Agent
  ↓
Tool Call
  ↓
Result unchanged
  ↓
Same Tool Call
  ↓
Result unchanged
  ↓
Loop Detector
  ↓
STOP
  ↓
Alternative Strategy / Human Escalation
```

**Key principle:** Combine **iteration limits + state/repetition detection + progress tracking + time/token budgets**. Never allow an autonomous agent to run indefinitely.

### 21.32 Your AI agent gets conflicting answers from different tools. How does it reconcile them?

#### Reconciling Conflicting Tool Results

- **Check source reliability**
  - Rank tools based on authority and trustworthiness.
  - Prefer an official database over an unverified external source, for example.
- **Check timestamps**
  - A newer result may be more accurate for time-sensitive information.
  - Consider whether one source contains stale data.
- **Compare the evidence**
  - Check whether the tools used different parameters, scopes, or assumptions.
  - Identify exactly which facts conflict rather than treating the entire results as contradictory.
- **Cross-validate**
  - Call another trusted source when the conflict is important.
  - Look for independent evidence that can confirm one result.
- **Use confidence and provenance**
  - Track where each result came from, when it was retrieved, and how reliable it is.
  - Assign confidence to competing claims.
- **Apply domain-specific rules**
  - Define deterministic precedence rules where possible.
  - Example: an internal transaction database may take precedence over a search result for account balance.
- **Don't blindly choose**
  - If the conflict cannot be resolved, the agent should explicitly represent the uncertainty rather than inventing an answer.
- **Escalate when necessary**
  - For high-impact decisions, unresolved conflicts should trigger human review.

#### Example

```txt
Tool A → Order status: Delivered
Tool B → Order status: In Transit

        ↓
Compare timestamps + source authority
        ↓
Tool A updated 2 minutes ago
Tool B updated 2 hours ago
        ↓
Use Tool A
        ↓
Record provenance
        ↓
Respond with confidence
```

**Key principle: Compare → verify → rank sources → resolve → record provenance.** If the conflict remains unresolved, preserve the uncertainty or escalate rather than guessing.

### 21.33 Your AI agent burns too many tokens per task. How do you reduce token consumption?

#### Reducing Token Consumption in AI Agents

- **Limit context size**
  - Send only relevant conversation history, documents, and tool results.
  - Remove redundant or obsolete context.
- **Use context compaction**
  - Summarize older interactions instead of repeatedly passing the full history.
  - Keep important facts and decisions while discarding unnecessary details.
- **Use smaller models for simple tasks**
  - Use cheaper models for routing, classification, extraction, and simple tool selection.
  - Reserve larger models for complex reasoning.
- **Reduce agent iterations**
  - Set maximum reasoning/tool-call steps.
  - Add clear stopping conditions to prevent unnecessary loops.
- **Optimize tool calls**
  - Avoid duplicate calls.
  - Cache repeated results.
  - Batch independent operations where possible.
- **Optimize RAG**
  - Retrieve fewer, higher-quality chunks.
  - Use metadata filtering and reranking before sending documents to the LLM.
- **Control output length**
  - Use structured outputs and concise response requirements.
  - Avoid asking the model to generate unnecessary explanations.
- **Use memory strategically**
  - Store durable information outside the active context.
  - Retrieve only the memories relevant to the current task.
- **Track token usage**
  - Measure input/output tokens per agent step.
  - Identify expensive prompts, tools, and loops and optimize them.

#### Key Principle

**Reduce unnecessary context + reduce iterations + use the cheapest capable model + cache repeated work + control output length.**

The goal is not simply to use fewer tokens, but to **maximize useful work per token**.

### 21.34 Your AI agent keeps exceeding its budget per task. How do you enforce budget limits?

#### Enforcing Budget Limits in AI Agent Workflows

- **Set a tight budget**
  - Define a maximum cost per task, such as `$0.10` or `$1.00`.
  - Track cumulative cost after every LLM and tool call.
- **Set token limits**
  - Limit input and output tokens per request.
  - Also maintain a total token budget for the entire workflow.
- **Limit agent iterations**
  - Set a maximum number of reasoning/tool-call cycles.
  - Example: stop after 15 iterations even if the task is incomplete.
- **Limit tool usage**
  - Set maximum calls per tool.
  - Prevent expensive APIs or searches from being called repeatedly.
- **Use model routing**
  - Start with a cheaper model for simple tasks.
  - Escalate to a more capable model only when necessary.
- **Implement a budget-aware controller**
  - Before every action, calculate the estimated remaining cost.
  - If the action would exceed the budget, block it or choose a cheaper alternative.
- **Use progressive degradation**
  - When the budget gets low:
    - Reduce context size.
    - Reduce output length.
    - Use a cheaper model.
    - Skip non-essential tools.
    - Return the best available result.
- **Add hard termination**
  - When the budget reaches its limit, transition the workflow to `BUDGET_EXCEEDED` rather than allowing the agent to continue.
- **Monitor and alert**
  - Track cost per task, model, agent, and tool.
  - Set alerts for unusual spending or runaway workflows.

#### Example

```txt
Task starts
    ↓
Budget = $1.00
    ↓
LLM call → $0.20
Tool call → $0.10
LLM call → $0.30
    ↓
Remaining = $0.40
    ↓
Estimate next action = $0.50
    ↓
BLOCK
    ↓
Cheaper alternative / Stop / Escalate
```

**Key principle:** Enforce budgets **outside the LLM**. The agent can request an action, but a budget controller decides whether that action is allowed based on **tokens, cost, iterations, tool calls, and remaining budget**.

### 21.35 Your AI agent hallucinates tool capabilities and passes wrong inputs. How do you fix it?

#### Fixing Tool Hallucination and Wrong Tool Inputs

- **Use explicit tool schemas**
  - Give the agent strict definitions for each tool, including name, purpose, parameters, types, required fields, and allowed values.
  - Avoid vague tool descriptions.
- **Validate tool calls externally**
  - Validate every tool call against its schema before execution.
  - Reject missing, extra, incorrectly typed, or invalid parameters.
- **Use structured tool calling**
  - Use the model's native function/tool-calling interface instead of asking it to generate arbitrary JSON or function names.
- **Restrict available tools**
  - Expose only the tools relevant to the current task.
  - Use tool allowlists and capability-based permissions.
- **Provide clear tool documentation**
  - Include concise examples of correct usage and clearly state what the tool **cannot** do.
  - Keep tool descriptions synchronized with the actual API.
- **Add a tool-validation layer**

```txt
User Request
     ↓
Agent
     ↓
Tool Selection
     ↓
Schema + Permission Validation
     ↓
Valid? ── No → Reject / Ask Agent to Correct
     ↓ Yes
Tool Execution
     ↓
Validate Result
     ↓
Agent Continues
```

- **Handle failures explicitly**
  - Return structured errors such as `INVALID_PARAMETER`, `UNAUTHORIZED`, or `TOOL_UNAVAILABLE`.
  - Let the agent correct the call rather than silently modifying incorrect inputs.
- **Prevent capability hallucination**
  - If no available tool supports an operation, the agent should state that it cannot perform it rather than inventing a tool or capability.
- **Test tool usage**
  - Create test cases for incorrect parameters, unavailable tools, ambiguous requests, permission violations, and malformed tool responses.

### 21.36 Your AI agent deleted a production database. How do you prevent irreversible actions?

#### Preventing Irreversible Actions in AI Agents

- **Require human approval**
  - Put a human approval step before destructive or high-impact operations such as deleting databases, production deployments, or changing permissions.
- **Use least-privilege permissions**
  - The agent should not have production-admin credentials by default.
  - Give it only the minimum permissions required for the task.
- **Separate environments**
  - Keep development, staging, and production credentials and resources strictly separated.
  - Agents should operate in staging by default.
- **Use deny-by-default policies**
  - Explicitly block dangerous operations such as `DROP DATABASE`bulk deletion, or production infrastructure changes unless specifically authorized.
- **Add an external policy layer**
  - Don't let the LLM decide whether an action is safe.
  - A deterministic authorization service should inspect every tool call.

```txt
Agent
  ↓
Tool Request: DELETE DATABASE
  ↓
Policy Engine
  ↓
Dangerous operation?
  ↓ Yes
Human Approval
  ↓
Approved?
 ┌───────┴───────┐
 No              Yes
 ↓                ↓
Block          Execute
```

- **Prefer reversible operations**
  - Use soft deletes, transactions, backups, snapshots, and point-in-time recovery where possible.
  - Provide rollback mechanisms.
- **Use scoped credentials**
  - Production credentials should be short-lived and restricted to specific resources and operations.
- **Add confirmation and dry-run modes**
  - Before execution, show exactly what will change.
  - Allow the agent to generate a plan or dry run without modifying production.
- **Use rate and blast-radius limits**
  - Limit how many records/resources an agent can modify in one operation.
  - Require additional approval for bulk operations.
- **Audit everything**
  - Log who/what requested the action, the tool call, authorization decision, affected resources, and result.

#### Key Principle

**The more irreversible the action, the stronger the control should be.**

For production systems, use **least privilege + deny-by-default + external authorization + human approval + backups/rollback**. An agent should never have unrestricted permission to perform destructive production operations.

### 21.37 Your AI agent has many tools, but keeps picking the wrong one. How do you improve tool selection?

#### Improving Tool Selection in AI Agents

- **Use clear tool descriptions**
  - Define each tool's purpose, inputs, outputs, limitations, and when it should be used.
  - Explicitly describe what the tool **cannot** do.
- **Reduce the number of available tools**
  - Don't expose every tool to every agent.
  - Dynamically provide only tools relevant to the current task.
- **Use a tool router**
  - First classify the user's request and select a small set of candidate tools.
  - Then let the main agent choose among those candidates.
- **Use strict schemas**
  - Define typed parameters, required fields, enums, and validation rules.
  - This reduces both tool-selection and argument errors.
- **Add tool-selection examples**
  - Provide few-shot examples showing which tool should be used for common requests and similar-looking alternatives.
- **Use hierarchical routing**
  - First choose a category, then a specific tool.

```txt
User Request
     ↓
Intent / Task Router
     ↓
Tool Category
     ↓
Candidate Tools
     ↓
Agent Selects Tool
     ↓
Schema + Permission Validation
     ↓
Execute
```

- **Use tool-use feedback**
  - If a tool fails because it was inappropriate, return a structured error and allow the agent to reconsider.
  - Track successful and failed tool selections to improve routing.
- **Add deterministic rules for critical tools**
  - For high-risk operations, use rules rather than relying entirely on LLM judgment.
  - Example: production database modifications always go through a dedicated approval workflow.
- **Monitor tool-selection accuracy**
  - Measure correct-tool rate, unnecessary tool calls, failed calls, retries, and task success.
  - Build an evaluation dataset of representative tasks and tool-selection decisions.

#### Key Principle

**Better tool selection comes from fewer, clearer tools + strong schemas + intelligent routing + validation + feedback.**

For large toolsets, a **hierarchical router that narrows dozens of tools to a few relevant candidates** is usually more reliable than exposing every tool to the LLM at once.

### 21.38 Your AI agent takes too long to complete a task. How do you speed it up?

#### Speeding Up a Slow AI Agent

- **Reduce the number of agent steps**
  - Simplify the workflow and eliminate unnecessary reasoning or tool calls.
  - Set clear stopping conditions.
- **Run independent tasks in parallel**
  - Execute independent API calls, searches, or sub-agents concurrently instead of sequentially.
- **Use faster models**
  - Use smaller/low-latency models for simple tasks such as routing, classification, and extraction.
  - Reserve larger models for complex reasoning.
- **Reduce context size**
  - Use context compaction and send only relevant history and retrieved information.
  - Smaller prompts generally mean faster inference.
- **Optimize tool calls**
  - Cache repeated results.
  - Batch independent requests.
  - Avoid unnecessary retries and duplicate calls.
- **Improve RAG performance**
  - Filter documents using metadata before semantic retrieval.
  - Retrieve fewer, higher-quality chunks.
  - Use efficient vector indexes and reranking only when needed.
- **Use streaming**
  - Stream intermediate/final output so users see results immediately instead of waiting for the entire response.
- **Use asynchronous execution**
  - Run long-running jobs asynchronously and allow the agent to continue with other independent work.
- **Use specialized agents**
  - Delegate well-defined tasks to focused agents instead of using one large agent for everything.
- **Profile the workflow**
  - Measure latency for every LLM call, tool call, retrieval operation, and external API.
  - Optimize the actual bottleneck rather than guessing.

#### Example

```txt
Slow:
Agent → Search A → Search B → Search C → Analyze → Validate

Faster:
Agent
  ├── Search A ─┐
  ├── Search B ─┼→ Analyze → Validate
  └── Search C ─┘
```

**Key principle:** Reduce **sequential steps and unnecessary context**, parallelize independent work, use faster models/tools, and profile each component to identify the real latency bottleneck.

### 21.39 Your LLM selects the right tool but extracts the wrong parameters. How do you fix parameter extraction?

#### Fixing Parameter Extraction in Tool Calling

- **Use strict tool schemas**
  - Define parameter names, types, required fields, enums, ranges, and defaults.
  - Avoid ambiguous parameter names.
- **Use structured function calling**
  - Let the LLM generate structured arguments rather than parsing free-form text or JSON manually.
- **Provide clear parameter descriptions**
  - Explain exactly what each field means and give examples.
  - Specify units and expected formats.
- **Normalize user input**
  - Convert dates, currencies, units, IDs, and other values into canonical formats before tool execution.
- **Validate parameters externally**
  - Never trust the LLM's arguments directly.
  - Validate type, range, format, authorization, and business rules before executing the tool.
- **Ask for clarification when information is missing**
  - Don't let the model guess critical values.
  - Example: if the user says "book it tomorrow" but multiple bookings exist, ask which booking.
- **Use constrained outputs**
  - Use enums and structured schemas wherever possible.
  - For example, `priority = ["low", "medium", "high"]` rather than accepting arbitrary text.
- **Add a correction/retry loop**
  - If validation fails, return a structured error to the agent and allow it to correct the parameters.
  - Limit retries to prevent loops.
- **Use deterministic extraction for simple fields**
  - For highly structured information such as IDs, dates, or numbers, combine LLM extraction with regex/parsers or deterministic validation.
- **Evaluate parameter accuracy**
  - Create test cases covering ambiguous requests, missing fields, incorrect types, boundary values, and similar-looking parameters.
  - Track parameter error rate and tool-call success rate.

#### Example

```txt
User: "Transfer $500 to account 12345"

        ↓
LLM Tool Call
        ↓
amount = 500
currency = USD
account_id = 12345
        ↓
Schema Validation
        ↓
Business Rule Validation
        ↓
Authorization Check
        ↓
Execute
```

If the model produces:

```txt
amount = 5000
```

the **validation layer must reject it before execution**.

**Key principle:** Use **structured schemas + external validation + normalization + clarification + bounded retries**. The LLM extracts the parameters, but a deterministic validation layer decides whether those parameters are actually valid.

### 21.40 How do Computer-Use Agents work?

#### Computer-Use Agents

**Computer-use agents** are AI agents that can interact with a computer through a graphical user interface (GUI), similar to how a human uses a mouse and keyboard.

#### How They Work

1. **Observe**
   - The agent receives a screenshot or structured representation of the current screen.
   - It identifies buttons, text fields, menus, windows, and other UI elements.
2. **Understand the task**
   - The LLM interprets the user's goal and determines what needs to be done.
3. **Plan the next action**
   - It decides whether to click, type, scroll, press a key, open an application, or perform another supported action.
4. **Execute**
   - A computer-control interface executes the selected action.
5. **Observe again**
   - The agent receives the updated screen state and determines whether the action succeeded.
6. **Repeat**
   - It continues the **observe → reason → act → observe** loop until the task is completed or it reaches a stopping condition.

#### Example

```txt
User: "Open the spreadsheet and find the total sales."

        ↓
Observe screen
        ↓
Identify spreadsheet application
        ↓
Click spreadsheet
        ↓
Observe
        ↓
Find sales column
        ↓
Read / calculate total
        ↓
Return result
```

#### Main Components

- **Vision model** — understands screenshots and UI elements.
- **LLM/agent** — plans actions and decides what to do next.
- **Computer-control tools** — provide actions such as click, type, scroll, and keyboard input.
- **State/memory** — tracks progress and previous actions.
- **Guardrails** — restrict dangerous operations and require confirmation when appropriate.

#### Security Considerations

Computer-use agents are particularly sensitive because they can interact with **real applications and data**. Use:

- Sandboxed environments where possible.
- Least-privilege accounts.
- Restricted filesystem and network access.
- Confirmation for destructive or irreversible actions.
- Action and tool-call validation.
- Maximum steps and execution time.
- Monitoring and audit logs.

**Key principle:** A computer-use agent combines **visual understanding + LLM reasoning + computer-control actions** in a feedback loop: **observe → reason → act → observe**.

### 21.41 How does LangChain work?

#### How LangChain Works

**LangChain is a framework for building applications and agents around LLMs.** It provides abstractions for connecting an LLM with prompts, tools, memory/state, retrievers, and workflows.

#### Basic Flow

```txt
User Request
     ↓
Prompt / Context
     ↓
LLM
     ↓
Decision
     ↓
Tool / Retriever / Function
     ↓
Tool Result
     ↓
LLM
     ↓
Final Response
```

#### Main Components

- **Models**
  - Provides a common interface for different LLMs such as OpenAI, Anthropic, Gemini, or local models.
- **Prompts**
  - Templates that combine system instructions, user input, retrieved context, and other state.
- **Tools**
  - Functions the LLM can call, such as APIs, database queries, web search, calculators, or custom Python functions.
- **Retrievers**
  - Fetch relevant information from vector databases, search engines, or other knowledge sources for RAG.
- **Chains**
  - Connect multiple operations in a predefined sequence.
  - Example: `Question → Retrieve → Generate Answer`.
- **Agents**
  - Allow the LLM to dynamically decide which tools to use and in what order.
- **Memory / State**
  - Maintains conversation or workflow state so the application can continue across multiple steps.
- **LangGraph**
  - Used for more complex, stateful agent workflows.
  - You define **nodes** for operations and **edges** for transitions, enabling loops, branching, retries, checkpoints, and human-in-the-loop workflows.

#### Simple Agent Example

```txt
User: "What is the weather in Kathmandu?"

        ↓
LangChain Agent
        ↓
LLM decides → weather_tool
        ↓
weather_tool("Kathmandu")
        ↓
Tool result
        ↓
LLM
        ↓
Final answer
```

#### LangChain vs LangGraph

- **LangChain** → building blocks for LLM applications, tools, retrieval, prompts, and agents.
- **LangGraph** → orchestration framework for **complex, stateful, long-running agent workflows**.

**Key principle:** LangChain doesn't make the LLM intelligent by itself. It provides the **integration layer** that connects the LLM to **prompts, tools, data, retrieval, state, and external systems** to build useful AI applications.

### 21.42 How does LangGraph work?

#### How LangGraph Works

**LangGraph is a framework for building stateful, multi-step agent workflows as graphs.** It gives you explicit control over **state, nodes, edges, loops, branching, retries, and human-in-the-loop steps**.

#### Core Components

- **State**
  - Shared data passed through the workflow.
  - Contains things like messages, tool results, intermediate outputs, and task status.
- **Nodes**
  - Individual steps that perform work.
  - A node can call an LLM, execute a tool, retrieve documents, validate results, or run custom code.
- **Edges**
  - Define what happens after a node.
  - They can be:
    - **Static:** always go to the same node.
    - **Conditional:** choose the next node based on the current state.
- **Graph**
  - Connects nodes and edges into a workflow.

#### Example Agent

```txt
                 ┌─────────────┐
                 │   Planner   │
                 └──────┬──────┘
                        ↓
                 ┌─────────────┐
                 │   LLM/Agent │
                 └──────┬──────┘
                        ↓
                 Need a tool?
                  /          \
                Yes           No
                 ↓             ↓
             Tool Call     Final Answer
                 ↓
              Result
                 ↓
              LLM/Agent
                 ↓
              Validate
              /      \
           Retry    Complete
```

#### How Execution Works

1. **Initialize state**
   - The workflow starts with an initial state.
2. **Execute a node**
   - A node reads the current state and performs its operation.
3. **Update state**
   - The node returns changes to the state.
4. **Follow an edge**
   - LangGraph determines the next node based on the workflow logic.
5. **Loop or branch**
   - The graph can repeat nodes, take different paths, or run multiple branches.
6. **Terminate**
   - The workflow reaches an explicit completion state.

#### Why LangGraph Is Useful for Agents

- **Stateful workflows**
- **Conditional routing**
- **Agent loops**
- **Tool calling**
- **Retries and error recovery**
- **Human-in-the-loop**
- **Checkpointing and persistence**
- **Long-running workflows**
- **Multi-agent coordination**

#### LangChain vs LangGraph

| LangChain | LangGraph |
| --- | --- |
| LLM application building blocks | Workflow/agent orchestration |
| Chains and tools | Graphs and state |
| Good for simpler flows | Good for complex workflows |
| Often sequential | Supports loops and branching |
| Agent abstractions | Explicit execution control |

**Key principle:** Think of **LangGraph as a state machine for AI agents**. You define **state + nodes + edges**, and LangGraph manages how the agent moves through the workflow, including **loops, decisions, persistence, and recovery**.

### 21.43 What is OKF (Open Knowledge Format)?

#### OKF (Open Knowledge Format)

**OKF usually refers to an Open Knowledge Format: a standardized, machine-readable way of representing and sharing knowledge so that it can be easily accessed, processed, reused, and exchanged between systems.**

In an AI/agent context, the idea is to represent knowledge in a **structured and interoperable format** rather than keeping it only as unstructured text.

- **Structured:** Knowledge is represented using consistent fields, relationships, or schemas.
- **Machine-readable:** AI systems and software can parse and process it.
- **Interoperable:** Different applications or agents can exchange the knowledge.
- **Reusable:** The same knowledge can be consumed by different systems.
- **Traceable:** Metadata such as source, version, and provenance can be attached.

#### In Agentic Systems

An open knowledge format can help agents exchange structured information:

```txt
Agent A
   ↓
Structured Knowledge
   ↓
OKF
   ↓
Agent B
   ↓
Reasoning / Action
```

For example, instead of passing a long paragraph:

```txt
"Customer John has an unpaid invoice from August..."
```

a structured representation could contain:

```txt
customer: John
invoice_month: August
status: unpaid
```

**Key idea:** OKF is about making knowledge **structured, machine-readable, shareable, and reusable across systems**.

### 21.44 What is the Model Context Protocol (MCP)?

An open protocol (introduced by Anthropic) that standardizes how AI models connect to external tools, data sources, and services — think of it as a universal adapter so an LLM can call a calendar, database, or file system without custom integration code for each one.

### 21.45 What is an AI agent?

An LLM-driven system that can plan, take actions (via tools/APIs), observe results, and iterate toward a goal with some autonomy, rather than just producing a single response to a prompt.

### 21.46 What is function calling?

Function calling (also called "tool calling") is a capability that lets a language model interact with external tools, APIs, or code — rather than just generating text.

#### The basic idea

Normally, an LLM only produces text. It can't check today's weather, run a calculation precisely, query a database, or send an email — it can only *talk about* doing those things. Function calling bridges that gap:

1. You give the model a list of available "functions" (tools), each with a name, description, and expected parameters
2. The model, based on the user's request, decides *if* and *which* function to call, and generates the right arguments for it — as structured data (usually JSON), not free text
3. Your code actually executes that function (the model itself doesn't run it)
4. The result gets passed back to the model, which uses it to continue the conversation or produce a final answer

#### Simple example

Say you define a tool:

json

```json
{  "name": "get_weather",  "description": "Get current weather for a location",  "parameters": {    "location": "string"  }}
```

User asks: *"What's the weather in Kathmandu?"*

The model doesn't know the weather — but it recognizes this matches the `get_weather` tool, and outputs something like:

json

```json
{"name": "get_weather", "arguments": {"location": "Kathmandu"}}
```

Your application code catches that, calls the real weather API, gets back `"22°C, partly cloudy"`, and feeds that back to the model — which then replies in natural language: *"It's 22°C and partly cloudy in Kathmandu right now."*

#### Why it matters

- **Grounding**: lets models fetch real, current, or precise data instead of hallucinating
- **Action**: lets models actually *do* things — send messages, update a spreadsheet, query a database, run code
- **Structured output**: forces the model's intent into a reliable, parseable format instead of ambiguous prose
- **Composability**: models can chain multiple tool calls together to complete multi-step tasks (this is the basis of most "agentic" AI systems)

### 21.47 What are hooks, coroutines, plugins, rules and skills in agent tooling?

These are building blocks for **extending and controlling AI agents / coding assistants** (Claude Code, Cursor, agent frameworks), plus coroutines as the async primitive under them.

- **Hooks:** user-defined code that the agent **harness** runs automatically at lifecycle events, e.g. `PreToolUse` (before a tool call), `PostToolUse`, session start, on stop. Because they're executed deterministically by the harness, not by the model, they enforce policy reliably: block `rm -rf`, run a formatter/linter after every edit, require tests before finishing, log tool calls for audit. Think git hooks or web middleware.
- **Coroutines:** functions that can pause and resume (`async def` / `await` in Python). Agent runtimes use them to run many LLM calls, tool calls and streams concurrently on one thread without blocking: e.g. `asyncio.gather` over parallel tool calls, streaming tokens while executing tools. Key for latency and throughput in I/O-bound agent systems.
- **Plugins:** packaged extensions that add capabilities to the agent or app: new tools, MCP servers, commands, UI, integrations. They are installable and distributable bundles (often containing skills, hooks and commands together).
- **Rules:** persistent instructions loaded into the agent's context (e.g. `CLAUDE.md`, `.cursor/rules`, system prompts): coding conventions, project facts, "always use pnpm". They are **soft guidance**, interpreted by the model, so they can be ignored; anything that must always happen belongs in a hook.
- **Skills:** packaged, on-demand know-how: a folder with instructions (e.g. `SKILL.md`), scripts and resources that the agent loads **only when relevant** (progressive disclosure), keeping the context window small. E.g. a "pdf" skill with extraction scripts.

**Design principle:** rules and skills shape *what the model decides*; hooks enforce *what the harness guarantees*; plugins *distribute* capabilities; coroutines make it all *run concurrently*.

```python
async def run_tools(calls):
    return await asyncio.gather(*(execute(c) for c in calls))  # parallel tool calls
```

### 21.48 What distinguishes an agent workflow from a single-turn Q&A system?

An agent workflow loops through plan → act → observe → decide across multiple steps, using intermediate results to inform the next action, rather than producing one response to one input.

### 21.49 How does tool/function calling work in an LLM application?

The LLM is given a set of predefined actions described by a JSON schema. It requests an action with structured arguments; the application executes that action outside the model and returns the result to the LLM to continue reasoning.

### 21.50 What problem do MCP servers solve?

They provide a standardized protocol for exposing tools and data to LLMs, removing the need to build a custom integration for every individual tool.

### 21.51 When should outputs be escalated to a human?

When confidence is low, or the stakes of the decision are high — those cases should route to human reviewers rather than auto-resolving.

### 21.52 What is multi-step planning in an agent?

Breaking a complex task down into an ordered sequence of sub-steps before or while executing it.

### 21.53 Why do agents need state and memory?

To maintain context across multiple turns or steps within a session, so later actions can build on earlier ones.

### 21.54 What does good error recovery look like in an agent?

The agent retries, replans, or falls back to an alternative approach instead of simply breaking when something fails.

### 21.55 What is a human-in-the-loop checkpoint?

A deliberate pause in the agent's flow that requires human approval before it takes a high-stakes action.

### 21.56 What is LangGraph designed for?

Building stateful, graph-based agent workflows with explicit branching logic.

### 21.57 What does Pydantic AI offer?

A lightweight, type-safe way to define structured outputs and tools.

### 21.58 What's the difference between building an agent "from scratch" in Python vs. using a framework like LangChain or LangGraph? When would you choose one over the other?

Building from scratch means you own the control loop yourself — you define how the LLM decides which tool to call, how you parse and validate its output (typically with Pydantic models), how you manage state/memory, and how you handle retries and errors. This gives full control and avoids framework overhead/abstraction leakage, which matters in latency- or cost-sensitive production systems.

LangChain gives you pre-built abstractions for chains, tool-calling, memory, and retrievers — good for fast prototyping and standard patterns. LangGraph goes further: it models the agent as an explicit state graph (nodes = steps/agents, edges = transitions, conditional routing), which is far better for multi-agent orchestration, cyclic reasoning (ReAct-style loops), human-in-the-loop checkpoints, and persistence of long-running workflows.

In practice: prototype fast with LangGraph, but for a narrow, high-reliability production agent with tight latency/cost budgets, a hand-rolled orchestration loop with Pydantic-validated I/O is often more maintainable and debuggable.

### 21.59 Why is Pydantic important for structured outputs from an LLM, and how do you enforce it?

LLMs return free text by default, which is unreliable for programmatic use. Pydantic models let you define an explicit schema (types, required fields, validators) that the LLM's output must conform to. You enforce this via:

- Function/tool-calling APIs (e.g., OpenAI/Anthropic tool-use) where the schema is passed as a JSON schema and the model returns structured arguments.
- `instructor`style libraries or LangChain's `with_structured_output`, which wrap the call and auto-parse/validate the response into a Pydantic object, retrying on validation failure.
- Custom validators (`@field_validator`) to catch domain-specific constraints (e.g., a date must be in the future, an ID must exist).

This gives you type safety, reduces hallucinated formats, and makes the LLM output a first-class object you can pass downstream without brittle regex parsing.

### 21.60 Describe how you'd design a multi-agent system for a real task (e.g., a research-and-report agent).

Typical pattern:

- **Orchestrator/supervisor agent**: decomposes the task, routes sub-tasks to specialist agents, and merges results.
- **Specialist agents**: e.g., a retrieval agent, a web-search agent, a writer agent, a critic/reviewer agent — each with a narrow tool set and system prompt.
- **Shared state**: in LangGraph, this is an explicit `State` object (Pydantic/TypedDict) passed between nodes; conditional edges route based on state (e.g., "if critic rejects, loop back to writer").
- **Termination/guardrails**: max iteration counts, confidence thresholds, human-in-the-loop approval for high-stakes actions.
- **Observability**: every agent call is traced (see tracing question below) so you can debug which agent made which decision.

Key design principle: keep each agent's responsibility narrow — broad, do-everything agents are harder to prompt reliably and harder to evaluate.

### 21.61 How do you handle an agent that gets stuck in a loop or calls a tool incorrectly?

Guardrails: max-turn/max-tool-call limits, timeout budgets, and explicit "stop conditions" in the graph. Validate tool arguments with Pydantic before execution and return structured error messages back to the model so it can self-correct (rather than crashing). Add a critic/reflection step that checks progress against the goal every N steps. Log everything via tracing so failure patterns can be caught in eval, not just in production.

### 21.62 What problem does MCP (Model Context Protocol) solve, and how does it differ from a custom tool integration?

Before MCP, every agent framework needed bespoke, one-off integration code for every tool/data source it connected to — an $N \times M$ problem ($N$ agent frameworks × $M$ tools). MCP standardizes this via a client-server protocol: an MCP **server** exposes tools, resources, and prompts in a uniform schema; any MCP-compatible **client** (agent) can discover and call them without custom glue code. It's essentially "USB-C for AI tools" — write the integration once, and any compliant agent can use it. This matters for maintainability at scale, especially in a consulting context where you're plugging agents into many different client systems.

### 21.63 What is A2A (Agent-to-Agent) protocol, and how is it different from MCP?

MCP standardizes how an agent talks to *tools and data sources*. A2A standardizes how *agents talk to each other* — enabling agents built on different frameworks or by different vendors to discover each other's capabilities (via an "agent card"), negotiate a task, and exchange messages/results in a common format. Think of MCP as vertical (agent → tool) and A2A as horizontal (agent → agent). Together they let you build heterogeneous multi-agent ecosystems rather than a single monolithic framework.

### 21.64 What security/reliability considerations come up when connecting an agent to external tools via MCP?

Authentication/authorization scoping (least-privilege access per tool), input validation on tool arguments before execution (prevent injection), sandboxing tool execution, rate-limiting, and auditing/logging every tool call for traceability. Also: treating any content returned from a tool (e.g., a fetched webpage) as untrusted data, not as instructions — to avoid prompt injection through tool outputs.

---

← [Chapter 20. Retrieval-Augmented Generation](20-retrieval-augmented-generation.md) · [Table of contents](../README.md) · [Chapter 22. Fine-Tuning and Model Adaptation](22-fine-tuning-and-model-adaptation.md) →
