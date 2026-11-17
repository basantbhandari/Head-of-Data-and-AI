# Chapter 21. AI Agents and Agentic Systems

_Part IV — Generative AI and LLM Engineering_

Tool use, planning, memory, sub-agents and multi-agent orchestration, MCP, A2A, skills and OKF, frameworks, evaluation, observability, identity and permissions, durable execution, coding agents, production operations and agentic training.

This chapter contains **53 questions**.

## Contents

- **21.1** [What is an AI agent, and how does it differ from a simple LLM call?](#211-what-is-an-ai-agent-and-how-does-it-differ-from-a-simple-llm-call)
- **21.2** [How does memory work in AI agents, and what types of memory are there?](#212-how-does-memory-work-in-ai-agents-and-what-types-of-memory-are-there)
- **21.3** [What is harness engineering in AI?](#213-what-is-harness-engineering-in-ai)
- **21.4** [Explain the ReAct (Reasoning + Acting) agent architecture.](#214-explain-the-react-reasoning--acting-agent-architecture)
- **21.5** [What is the Plan-and-Execute agent pattern?](#215-what-is-the-plan-and-execute-agent-pattern)
- **21.6** [What is tool use (function calling) in LLMs, and how does it enable agents?](#216-what-is-tool-use-function-calling-in-llms-and-how-does-it-enable-agents)
- **21.7** [How do you design and define tools for an AI agent?](#217-how-do-you-design-and-define-tools-for-an-ai-agent)
- **21.8** [What is the difference between single-agent and multi-agent systems, and when is multi-agent worth it?](#218-what-is-the-difference-between-single-agent-and-multi-agent-systems-and-when-is-multi-agent-worth-it)
- **21.9** [What is the Model Context Protocol (MCP), what problem does it solve, and what are its security considerations?](#219-what-is-the-model-context-protocol-mcp-what-problem-does-it-solve-and-what-are-its-security-considerations)
- **21.10** [What are sub-agents, and when should an agent delegate to them?](#2110-what-are-sub-agents-and-when-should-an-agent-delegate-to-them)
- **21.11** [How do you handle agent failures and implement error recovery?](#2111-how-do-you-handle-agent-failures-and-implement-error-recovery)
- **21.12** [What is an agent loop, and how does it decide when to stop?](#2112-what-is-an-agent-loop-and-how-does-it-decide-when-to-stop)
- **21.13** [How do AI Agents Communicate?](#2113-how-do-ai-agents-communicate)
- **21.14** [What are Agent Skills?](#2114-what-are-agent-skills)
- **21.15** [How do you evaluate and test AI agents?](#2115-how-do-you-evaluate-and-test-ai-agents)
- **21.16** [What are the security risks of agentic systems, and how do you mitigate them?](#2116-what-are-the-security-risks-of-agentic-systems-and-how-do-you-mitigate-them)
- **21.17** [What is the difference between reactive and proactive agents?](#2117-what-is-the-difference-between-reactive-and-proactive-agents)
- **21.18** [How do you manage token consumption and cost in long-running agent workflows?](#2118-how-do-you-manage-token-consumption-and-cost-in-long-running-agent-workflows)
- **21.19** [What is the human-in-the-loop pattern for agents, and when is it needed?](#2119-what-is-the-human-in-the-loop-pattern-for-agents-and-when-is-it-needed)
- **21.20** [How do you implement guardrails for AI agents to prevent harmful actions?](#2120-how-do-you-implement-guardrails-for-ai-agents-to-prevent-harmful-actions)
- **21.21** [What is agent reflection, and how does it improve agent performance?](#2121-what-is-agent-reflection-and-how-does-it-improve-agent-performance)
- **21.22** [What is the difference between code-generating agents and tool-calling agents?](#2122-what-is-the-difference-between-code-generating-agents-and-tool-calling-agents)
- **21.23** [How do you handle multi-modal inputs and outputs in agentic systems?](#2123-how-do-you-handle-multi-modal-inputs-and-outputs-in-agentic-systems)
- **21.24** [How do you implement state management in complex agent workflows?](#2124-how-do-you-implement-state-management-in-complex-agent-workflows)
- **21.25** [How do you build a customer support agent with escalation logic?](#2125-how-do-you-build-a-customer-support-agent-with-escalation-logic)
- **21.26** [What is agent orchestration, and how do you implement it?](#2126-what-is-agent-orchestration-and-how-do-you-implement-it)
- **21.27** [How do you build a code execution agent safely using sandboxed environments?](#2127-how-do-you-build-a-code-execution-agent-safely-using-sandboxed-environments)
- **21.28** [Your AI agent is stuck in an infinite loop. How do you detect and break the cycle?](#2128-your-ai-agent-is-stuck-in-an-infinite-loop-how-do-you-detect-and-break-the-cycle)
- **21.29** [Your AI agent gets conflicting answers from different tools. How does it reconcile them?](#2129-your-ai-agent-gets-conflicting-answers-from-different-tools-how-does-it-reconcile-them)
- **21.30** [Your AI agent burns too many tokens per task. How do you reduce token consumption?](#2130-your-ai-agent-burns-too-many-tokens-per-task-how-do-you-reduce-token-consumption)
- **21.31** [Your AI agent keeps exceeding its budget per task. How do you enforce budget limits?](#2131-your-ai-agent-keeps-exceeding-its-budget-per-task-how-do-you-enforce-budget-limits)
- **21.32** [Your AI agent hallucinates tool capabilities and passes wrong inputs. How do you fix it?](#2132-your-ai-agent-hallucinates-tool-capabilities-and-passes-wrong-inputs-how-do-you-fix-it)
- **21.33** [Your AI agent deleted a production database. How do you prevent irreversible actions?](#2133-your-ai-agent-deleted-a-production-database-how-do-you-prevent-irreversible-actions)
- **21.34** [Your AI agent has many tools, but keeps picking the wrong one. How do you improve tool selection?](#2134-your-ai-agent-has-many-tools-but-keeps-picking-the-wrong-one-how-do-you-improve-tool-selection)
- **21.35** [Your AI agent takes too long to complete a task. How do you speed it up?](#2135-your-ai-agent-takes-too-long-to-complete-a-task-how-do-you-speed-it-up)
- **21.36** [Your LLM selects the right tool but extracts the wrong parameters. How do you fix parameter extraction?](#2136-your-llm-selects-the-right-tool-but-extracts-the-wrong-parameters-how-do-you-fix-parameter-extraction)
- **21.37** [How do Computer-Use Agents work?](#2137-how-do-computer-use-agents-work)
- **21.38** [How does LangChain work?](#2138-how-does-langchain-work)
- **21.39** [How does LangGraph work?](#2139-how-does-langgraph-work)
- **21.40** [What is OKF (Open Knowledge Format)?](#2140-what-is-okf-open-knowledge-format)
- **21.41** [What are hooks, coroutines, plugins, rules and skills in agent tooling?](#2141-what-are-hooks-coroutines-plugins-rules-and-skills-in-agent-tooling)
- **21.42** [What does Pydantic AI offer?](#2142-what-does-pydantic-ai-offer)
- **21.43** [What's the difference between building an agent "from scratch" in Python vs. using a framework like LangChain or LangGraph? When would you choose one over the other?](#2143-whats-the-difference-between-building-an-agent-from-scratch-in-python-vs-using-a-framework-like-langchain-or-langgraph-when-would-you-choose-one-over-the-other)
- **21.44** [Why is Pydantic important for structured outputs from an LLM, and how do you enforce it?](#2144-why-is-pydantic-important-for-structured-outputs-from-an-llm-and-how-do-you-enforce-it)
- **21.45** [Describe how you'd design a multi-agent system for a real task (e.g., a research-and-report agent).](#2145-describe-how-youd-design-a-multi-agent-system-for-a-real-task-eg-a-research-and-report-agent)
- **21.46** [What is A2A (Agent-to-Agent) protocol, and how is it different from MCP?](#2146-what-is-a2a-agent-to-agent-protocol-and-how-is-it-different-from-mcp)
- **21.47** [When should you not build an agent?](#2147-when-should-you-not-build-an-agent)
- **21.48** [How do you build observability and tracing for agents?](#2148-how-do-you-build-observability-and-tracing-for-agents)
- **21.49** [How do you handle identity, authentication and permissions for agents?](#2149-how-do-you-handle-identity-authentication-and-permissions-for-agents)
- **21.50** [What is durable execution, and why do long-running agents need it?](#2150-what-is-durable-execution-and-why-do-long-running-agents-need-it)
- **21.51** [How do coding agents work, and what makes them effective?](#2151-how-do-coding-agents-work-and-what-makes-them-effective)
- **21.52** [How do you deploy and operate agents in production?](#2152-how-do-you-deploy-and-operate-agents-in-production)
- **21.53** [How are models trained to be better agents?](#2153-how-are-models-trained-to-be-better-agents)

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

### 21.2 How does memory work in AI agents, and what types of memory are there?

An LLM has no memory between calls: everything it "knows" about the current task must be in its context window. **Agent memory** is the set of mechanisms that decide what to store, where, and what to bring back into context at each step, so the agent can work over many steps and sessions.

**Types of memory**

| Type | What it holds | Typical implementation |
| --- | --- | --- |
| Working (short-term) memory | The current task: recent messages, tool results, the plan | The context window itself, managed with trimming and summarisation |
| Episodic memory | Past interactions and experiences: what happened, what worked, what failed | Logs or summaries of past sessions, retrieved by similarity or recency |
| Semantic memory | Durable facts: user preferences, project facts, domain knowledge | Key-value stores, documents, vector databases, knowledge graphs |
| Procedural memory | How to do things: instructions, workflows, learned rules | System prompts, skill files, playbooks the agent can load |

**The memory loop**

1. **Write:** decide what is worth remembering. Good candidates are user preferences, decisions, facts that will be needed again, and lessons from failures. Storing everything creates noise.
2. **Store:** with metadata (user, time, source, confidence) and in a form that can be updated or deleted.
3. **Retrieve:** bring back only what is relevant to the current step, by semantic similarity, recency, importance or explicit keys.
4. **Update and forget:** merge duplicates, correct outdated facts, and expire memories; stale memory is worse than none.

**Design choices**

- **Who decides what to remember:** the application (rules, extraction after each session) or the agent itself through memory tools (`save_note`, `search_memory`). Agent-managed memory is flexible but needs guardrails.
- **Scope and isolation:** memories belong to a user, team or tenant, and must never leak across them.
- **Within a long task,** working memory is managed with compaction, external notes and sub-agents (see 18.35 and 18.37 for context engineering and compaction).

**Risks**

- **Memory poisoning:** malicious or wrong content saved to memory influences every future session. Validate what is written, and keep provenance.
- **Privacy:** users must be able to see and delete what the agent remembers; retention must follow data policies.
- **Over-personalisation:** old preferences applied where they no longer fit.

**Evaluation:** test whether the agent recalls the right facts at the right time across sessions, does not recall irrelevant or deleted information, and handles contradictions (a preference that changed).

### 21.3 What is harness engineering in AI?

The **harness** is everything around the model that turns it into a working agent: the loop that calls the model, the tools it can use, how context is assembled, how results are checked, and the limits and permissions it runs under. **Harness engineering** is the discipline of designing that system. Two products built on the same model can differ enormously in quality because of their harnesses.

**Components of a harness**

- **The agent loop:** how the model is called, how tool calls are executed, and when the loop stops.
- **Tools and their interfaces:** which tools exist, how they are described, what they return (see 18.38).
- **Context management:** system prompt, retrieved knowledge, memory, compaction, sub-agents.
- **Verification:** tests, linters, type checkers, validators and critics that check the agent's work automatically.
- **Environment:** sandboxes, file systems, browsers, credentials and network access.
- **Guardrails and permissions:** what the agent may do without asking, and what needs approval.
- **Observability:** traces of every step for debugging and evaluation.

**Why it matters**

- **Feedback is the biggest lever.** A coding agent that can run the test suite and see failures fixes its own mistakes; one that cannot will confidently ship broken code. Giving the agent fast, reliable signals about whether it is succeeding often improves results more than a better prompt.
- **Context quality decides behaviour.** What the harness puts in the window at each step shapes every decision.
- **Model upgrades:** a good harness lets you swap in a newer model and benefit immediately; tight coupling to one model's quirks makes upgrades painful.

**Example:** a coding agent's harness includes repository search and file editing tools, a terminal in a sandbox, the ability to run tests and see results, instructions about project conventions, permission rules (ask before pushing or deleting), and a trace of every action.

**Evaluation:** benchmark the whole harness and model together on realistic tasks; the same model can score very differently in different harnesses.

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

Function calling (also called "tool calling") is a capability that lets a language model interact with external tools, APIs or code, rather than only generating text. It is the mechanism that turns a chat model into an agent.

**The basic idea**

Normally, an LLM only produces text. It cannot check today's weather, compute something exactly, query a database or send an email; it can only *talk about* doing those things. Function calling bridges that gap:

1. You give the model a list of available tools, each with a name, a description and a parameter schema (usually JSON Schema).
2. Based on the request, the model decides **whether** to call a tool, **which** one, and generates the arguments as structured data.
3. **Your code executes the call.** The model never runs anything itself.
4. The result is passed back to the model, which uses it to continue: answer, or call another tool.

```text
user request ─► model ─► tool call {name, arguments} ─► application validates and executes
             ◄── tool result ◄───────────────────────────┘
             ─► model ─► next tool call or final answer
```

**How it enables agents:** an agent is essentially this loop repeated: the model chooses actions, observes results and decides what to do next until the task is done.

**Practical details**

- **Validate arguments** against the schema and business rules before executing; never execute model-generated arguments blindly.
- **Return useful errors** so the model can correct itself.
- **Parallel calls:** many models can request several independent tool calls at once, which reduces latency.
- **Forcing or restricting tool use:** APIs usually let you require a tool call, forbid it, or force a specific tool.
- **Structured outputs:** a tool with a schema is also a reliable way to get structured data from a model, even when nothing is executed.

```python
import json

tools = [{
    "name": "get_order_status",
    "description": "Look up the current status of a customer's order by order ID.",
    "input_schema": {"type": "object",
                     "properties": {"order_id": {"type": "string", "description": "Order ID such as A-12345"}},
                     "required": ["order_id"]},
}]

def run_tool(name, args):
    if name == "get_order_status":
        if not args.get("order_id", "").startswith("A-"):
            return {"error": "order_id must look like A-12345"}          # helpful error for self-correction
        return {"order_id": args["order_id"], "status": "shipped"}
    return {"error": f"unknown tool {name}"}

# inside the agent loop: for each tool call the model returns
result = run_tool(call["name"], call["input"])
messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
```

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

### 21.8 What is the difference between single-agent and multi-agent systems, and when is multi-agent worth it?

**Single-agent systems** use one model loop with its tools and memory. **Multi-agent systems** split work across several agents, each with its own instructions, tools and context, coordinated by an orchestrator or by message passing.

**Common multi-agent patterns**

- **Orchestrator and workers:** a lead agent plans, delegates sub-tasks to worker agents (often in parallel), and combines the results.
- **Pipeline:** agents in sequence (research, then draft, then review).
- **Critic or reviewer:** one agent produces, another checks against criteria.
- **Debate or ensemble:** several agents propose answers and a judge selects or merges them.
- **Hand-off:** a routing agent passes the conversation to a specialist (billing, technical support).

**When multi-agent helps**

- **Parallelisable work** that exceeds one context window: broad research over many sources, analysing many documents, searching a large codebase. Each sub-agent explores in its own clean context and returns a condensed result.
- **Clearly separable specialisations** with different tools or permissions (a read-only research agent and a separate agent allowed to take actions).
- **Independent checking,** where a fresh context catches errors the producing agent is blind to.

**When it hurts**

- **Cost:** multi-agent systems use many more tokens, often several times a single agent. Worth it only when the task value justifies it.
- **Coordination failures:** duplicated work, conflicting decisions, and lost information in hand-offs, because sub-agents do not share the full context.
- **Tightly coupled tasks** (for example, most coding changes) where every step depends on detailed shared context; a single agent usually does better.
- **Harder debugging and evaluation:** failures spread across several traces.

| Factor | Prefer single agent | Prefer multi-agent |
| --- | --- | --- |
| Task structure | Sequential, tightly coupled | Parallel, decomposable |
| Context needed | Shared, detailed | Separable per sub-task |
| Budget | Cost-sensitive | High-value tasks |
| Permissions | Uniform | Different trust levels per role |

**Practical advice:** start with a single agent and good tools. Add sub-agents for clearly parallel or separable work, give each a precise task description, expected output format and limits, and evaluate the whole system against the single-agent baseline.

### 21.9 What is the Model Context Protocol (MCP), what problem does it solve, and what are its security considerations?

**MCP** is an open protocol, introduced by Anthropic in November 2024 and since adopted widely across AI tools and platforms, that standardises how AI applications connect to external tools, data and services.

**The problem it solves**

- Without a standard, every AI application needs custom integration code for every tool or data source: an $N \times M$ problem for $N$ applications and $M$ tools.
- With MCP, a tool provider writes one **MCP server**, and any MCP-compatible application (the **host**, using an **MCP client**) can discover and use it. The problem becomes $N + M$. It is often described as "USB-C for AI tools".

**Architecture**

```text
host application (IDE, chat app, agent) ─► MCP client ─► MCP server ─► tool, database, API or files
                                         ◄─ results ◄──────────────┘
```

- **Server primitives:**
  - **Tools:** functions the model can call (search tickets, run a query, create an issue).
  - **Resources:** data the application can read into context (files, records, documents).
  - **Prompts:** reusable prompt templates offered by the server.
- **Client features** that servers can use: sampling (asking the host's model to generate text), roots (which directories or resources are in scope) and elicitation (asking the user for more input).
- **Protocol:** JSON-RPC 2.0 messages, with capability negotiation at connection time.
- **Transports:** standard input and output for local servers, and streamable HTTP for remote servers.

**MCP compared with plain function calling:** function calling is how a model *requests* an action inside one application. MCP is how applications *discover and connect to* tools in a standard, reusable way. MCP tools are ultimately exposed to the model through function calling.

**Security considerations**

- **Authentication and authorisation:** remote servers should use the protocol's OAuth-based authorisation, with scoped, least-privilege access per user and per tool.
- **Untrusted servers:** a malicious or compromised server can return content with injected instructions, or describe tools misleadingly (tool poisoning). Install servers only from trusted sources, pin versions, and review tool descriptions.
- **Untrusted tool outputs:** treat everything a tool returns (web pages, emails, documents) as data, never as instructions.
- **Validate arguments and sandbox execution** for servers that run code or touch files.
- **Human approval** for consequential actions; rate limits; and an audit log of every tool call.
- **Too many tools** degrade tool selection and consume context; enable only the servers a task needs.

### 21.10 What are sub-agents, and when should an agent delegate to them?

A **sub-agent** is a separate agent instance that a main (orchestrating) agent creates or calls to handle a specific sub-task. The key property is that a sub-agent runs in its **own context window**, with its own instructions and often a restricted set of tools, and returns only a condensed result to the main agent.

**Why delegate**

- **Context isolation:** exploring a large codebase or reading dozens of web pages produces a lot of intermediate material. If a sub-agent does it, the main agent's context stays clean and focused; it receives only the conclusion.
- **Parallelism:** several sub-agents can investigate independent questions at the same time, reducing wall-clock time.
- **Specialisation:** each sub-agent gets instructions and tools suited to its job (a read-only researcher, a test runner, a reviewer).
- **Least privilege:** sub-agents can have fewer permissions than the main agent.

**Costs and risks**

- More total tokens and latency for coordination.
- **Information loss at hand-off:** the sub-agent lacks the main agent's full context, and its summary may omit what turns out to matter.
- Sub-agents can report confidently wrong conclusions; the main agent should verify important claims.

**Writing a good delegation**

- State the goal, the relevant background the sub-agent needs, what has already been ruled out, and the exact output format expected.
- Set clear scope and limits (time, number of steps, tools).
- Ask for evidence (file paths, sources) along with conclusions, so the main agent can check them.

**When not to delegate:** small tasks, tightly coupled work where the main agent's detailed context matters, or lookups that take a few tool calls. Delegation pays off for broad, separable investigation.

### 21.11 How do you handle agent failures and implement error recovery?

Agents fail in many ways: tools time out, APIs return errors, the model produces invalid arguments, plans go wrong, or the agent gets stuck. Good error recovery means the agent **retries, re-plans or falls back** rather than breaking or, worse, continuing confidently on a bad path.

**Types of failure and responses**

| Failure | Response |
| --- | --- |
| Transient tool error (timeout, rate limit, network) | Retry with exponential backoff and jitter; respect rate-limit headers |
| Invalid arguments | Validate before execution; return a precise error message so the model can correct itself |
| Tool permanently unavailable | Fall back to an alternative tool or a degraded mode, and tell the user |
| Wrong or empty results | Let the model reformulate (different query, different tool); verify results against expectations |
| Plan not working | Re-plan from the current state instead of repeating the same action |
| Repeated loops | Detect repeated actions or states, cap iterations, force a different strategy (see 21.28) |
| Unrecoverable or high-risk situation | Stop safely, preserve state, and escalate to a human with a clear summary |

**Engineering practices**

- **Checkpoint state** so a crashed or interrupted run can resume rather than restart (see the durable execution question in this chapter).
- **Make tools idempotent** where possible (for example with idempotency keys), so retries do not repeat side effects such as double payments.
- **Compensating actions** for multi-step side effects: if step 3 fails after steps 1 and 2 changed something, know how to undo them.
- **Budgets:** limits on steps, time, tokens and cost per task.
- **Structured errors:** a consistent error format (type, message, whether retrying might help) makes the model's recovery decisions better.
- **Trace everything,** and turn recurring failures into evaluation cases.

```python
import random, time

class RetryableError(Exception): ...

def call_with_retry(fn, *args, max_attempts=4, base=0.5, **kwargs):
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except RetryableError as e:
            if attempt == max_attempts - 1:
                return {"error": "tool_unavailable", "detail": str(e), "retryable": False}
            time.sleep(base * 2 ** attempt + random.uniform(0, base))       # exponential backoff with jitter
```

### 21.12 What is an agent loop, and how does it decide when to stop?

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

### 21.13 How do AI Agents Communicate?

#### How AI Agents Communicate

- AI agents communicate by **exchanging structured messages, instructions, results, and context**.
- In a **multi-agent system**, a main agent can delegate tasks to specialized agents and receive their results.
- Communication can happen through **direct messages, APIs, shared state, message queues, or protocols such as MCP**.
- Agents may share **task status, tool results, observations, and required actions**.
- A central **orchestrator** can coordinate the communication and combine agent outputs.

**In short:**

**Agent → Message/Task → Another Agent → Result → Agent/Orchestrator**

### 21.14 What are Agent Skills?

**Agent Skills** are packaged, reusable instructions and resources that teach an agent how to perform a specific kind of task, loaded only when the task needs them. Anthropic introduced the format in October 2025 for Claude, and it has since been published as an open format that other agent tools can use.

**What a skill contains**

- A folder with a `SKILL.md` file: YAML front matter with a **name** and a **description** of when to use the skill, followed by the detailed instructions.
- Optional supporting files: scripts the agent can run, templates, reference documents and examples.

**Progressive disclosure: why skills scale**

- At the start, the agent sees only each skill's short name and description, which costs very few tokens.
- When a task matches a description, the agent loads that skill's full instructions.
- Further files are read only if the instructions point to them.
- This lets an agent have access to many skills without filling its context window, unlike putting every instruction in the system prompt.

**What skills are good for**

- Organisation-specific procedures: how to write a report in the house style, how to run the release checklist, how to analyse data with the company's conventions.
- Tasks that need reliable, deterministic steps: a bundled script does the precise work (for example filling a PDF form or validating a file), and the instructions tell the agent when to run it.
- Sharing know-how across teams: skills are files that can be versioned, reviewed and distributed.

**Skills compared with other mechanisms**

| Mechanism | Purpose |
| --- | --- |
| System prompt | Always-on instructions for every task |
| Tools and MCP servers | Actions and data access |
| Skills | Task-specific procedures and resources, loaded on demand |
| Sub-agents | Separate contexts for delegated work |

**Cautions:** skills can contain executable scripts, so install them only from trusted sources and review them like code; keep descriptions precise so the agent picks the right skill.

### 21.15 How do you evaluate and test AI agents?

Agents are harder to evaluate than single model calls: they take many steps, interact with changing environments, and can reach correct outcomes by different paths, or reach wrong outcomes through plausible-looking steps. A serious evaluation combines **outcome checks, trajectory checks and reliability measurement** (see Chapter 24 for evaluation methods in general).

**1. Build realistic tasks and environments**

- Tasks drawn from real usage, with clear success criteria, including edge cases and failure-prone scenarios.
- **Sandboxed environments** that reproduce the tools: mock or test instances of APIs and databases, a repository with tests, a simulated user. The environment must be resettable so runs are repeatable.

**2. Outcome evaluation**

- **Programmatic checks** wherever possible: did the tests pass, is the database in the expected final state, was the ticket created with the right fields?
- **LLM or human judges** for open-ended outputs, with rubrics and calibration against human ratings.

**3. Trajectory evaluation**

- Did the agent use the right tools with valid arguments, avoid unnecessary or forbidden actions, ask for confirmation where required, and respect policies?
- Efficiency: number of steps, tokens, time and cost per successful task.

**4. Reliability, not just average success**

- Agents are stochastic; one successful run proves little. Run each task several times.
- **pass@k** measures whether *at least one* of $k$ attempts succeeds (useful when a human can pick the best result).
- **pass^k** (introduced with the τ-bench benchmark) measures whether *all* $k$ attempts succeed, which matters for production consistency. With $n$ trials per task and $c$ successes:

$$
\text{pass}^k = \mathbb{E}_{\text{task}} \left[ \frac{\binom{c}{k}}{\binom{n}{k}} \right]
$$

- An agent with 60% single-run success can have a much lower pass^8: it fails somewhere for most users.

**5. Safety and robustness testing**

- Prompt injection through tool outputs, requests to exceed permissions, ambiguous instructions, tool failures and unexpected responses.

**6. Public benchmarks for orientation**

- SWE-bench Verified (software engineering), τ-bench (tool use with simulated users and policies), OSWorld (computer use), WebArena (web tasks), GAIA (general assistant tasks). Use them to compare models; rely on your own task suite for decisions.

**7. Continuous evaluation**

- Run the suite on every change to prompts, tools, models or harness; sample production traces for review; and add every production failure as a new test case.

### 21.16 What are the security risks of agentic systems, and how do you mitigate them?

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

### 21.17 What is the difference between reactive and proactive agents?

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

### 21.18 How do you manage token consumption and cost in long-running agent workflows?

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

### 21.19 What is the human-in-the-loop pattern for agents, and when is it needed?

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

### 21.20 How do you implement guardrails for AI agents to prevent harmful actions?

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

### 21.21 What is agent reflection, and how does it improve agent performance?

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

### 21.22 What is the difference between code-generating agents and tool-calling agents?

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

### 21.23 How do you handle multi-modal inputs and outputs in agentic systems?

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

### 21.24 How do you implement state management in complex agent workflows?

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

### 21.25 How do you build a customer support agent with escalation logic?

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

### 21.26 What is agent orchestration, and how do you implement it?

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

### 21.27 How do you build a code execution agent safely using sandboxed environments?

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

### 21.28 Your AI agent is stuck in an infinite loop. How do you detect and break the cycle?

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

**Also:** validate tool arguments (for example with Pydantic) before execution and return structured errors so the model can self-correct; add a critic or reflection step that checks progress against the goal every few steps; and trace every run so loop patterns are caught in evaluation, not only in production.

### 21.29 Your AI agent gets conflicting answers from different tools. How does it reconcile them?

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

### 21.30 Your AI agent burns too many tokens per task. How do you reduce token consumption?

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

### 21.31 Your AI agent keeps exceeding its budget per task. How do you enforce budget limits?

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

### 21.32 Your AI agent hallucinates tool capabilities and passes wrong inputs. How do you fix it?

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

### 21.33 Your AI agent deleted a production database. How do you prevent irreversible actions?

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

### 21.34 Your AI agent has many tools, but keeps picking the wrong one. How do you improve tool selection?

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

### 21.35 Your AI agent takes too long to complete a task. How do you speed it up?

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

### 21.36 Your LLM selects the right tool but extracts the wrong parameters. How do you fix parameter extraction?

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

### 21.37 How do Computer-Use Agents work?

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

### 21.38 How does LangChain work?

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

### 21.39 How does LangGraph work?

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

### 21.40 What is OKF (Open Knowledge Format)?

**OKF (Open Knowledge Format)** is a specification published by Google Cloud engineers in June 2026 for packaging an organisation's knowledge so that both AI agents and humans can use it. Version 0.1 was released as an early open specification, and the ecosystem around it is still forming.

**What it is**

- An OKF **bundle** is a directory of **Markdown files with YAML front matter**: human-readable documents plus structured metadata (for example title, description, ownership, relationships and provenance).
- It is deliberately a **format, not a platform**: no proprietary runtime, account or SDK is required, and it is not tied to a particular cloud, database, model provider or agent framework.
- It builds on the "LLM wiki" idea: curated, linked Markdown pages that agents can navigate and read, instead of knowledge scattered across wikis, data catalogues and people's heads.

**Why it matters for agents**

- Agents perform better with **curated, well-described context** than with raw document dumps. A standard format makes that context portable across tools and teams.
- Plain Markdown with metadata can be version-controlled, reviewed and governed like code, which helps with ownership, freshness and audit.
- Google's Knowledge Catalog can ingest OKF bundles and serve them to agents, which is an early sign of platform support.

**How it relates to other standards**

| Standard | Role |
| --- | --- |
| OKF | Stored, version-controlled knowledge that agents read |
| MCP | Live connections between agents and tools or data sources |
| Agent Skills | Task procedures and resources loaded on demand |
| A2A | Communication between agents |

**A balanced view for interviews:** OKF is new, and adoption patterns are not settled. The durable lesson is the underlying practice: treat knowledge for agents as a governed product, in open, human-readable formats with clear metadata and ownership, whatever specific standard wins.

### 21.41 What are hooks, coroutines, plugins, rules and skills in agent tooling?

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

### 21.42 What does Pydantic AI offer?

A lightweight, type-safe way to define structured outputs and tools.

### 21.43 What's the difference between building an agent "from scratch" in Python vs. using a framework like LangChain or LangGraph? When would you choose one over the other?

Building from scratch means you own the control loop yourself — you define how the LLM decides which tool to call, how you parse and validate its output (typically with Pydantic models), how you manage state/memory, and how you handle retries and errors. This gives full control and avoids framework overhead/abstraction leakage, which matters in latency- or cost-sensitive production systems.

LangChain gives you pre-built abstractions for chains, tool-calling, memory, and retrievers — good for fast prototyping and standard patterns. LangGraph goes further: it models the agent as an explicit state graph (nodes = steps/agents, edges = transitions, conditional routing), which is far better for multi-agent orchestration, cyclic reasoning (ReAct-style loops), human-in-the-loop checkpoints, and persistence of long-running workflows.

In practice: prototype fast with LangGraph, but for a narrow, high-reliability production agent with tight latency/cost budgets, a hand-rolled orchestration loop with Pydantic-validated I/O is often more maintainable and debuggable.

### 21.44 Why is Pydantic important for structured outputs from an LLM, and how do you enforce it?

LLMs return free text by default, which is unreliable for programmatic use. Pydantic models let you define an explicit schema (types, required fields, validators) that the LLM's output must conform to. You enforce this via:

- Function/tool-calling APIs (e.g., OpenAI/Anthropic tool-use) where the schema is passed as a JSON schema and the model returns structured arguments.
- `instructor`style libraries or LangChain's `with_structured_output`, which wrap the call and auto-parse/validate the response into a Pydantic object, retrying on validation failure.
- Custom validators (`@field_validator`) to catch domain-specific constraints (e.g., a date must be in the future, an ID must exist).

This gives you type safety, reduces hallucinated formats, and makes the LLM output a first-class object you can pass downstream without brittle regex parsing.

### 21.45 Describe how you'd design a multi-agent system for a real task (e.g., a research-and-report agent).

Typical pattern:

- **Orchestrator/supervisor agent**: decomposes the task, routes sub-tasks to specialist agents, and merges results.
- **Specialist agents**: e.g., a retrieval agent, a web-search agent, a writer agent, a critic/reviewer agent — each with a narrow tool set and system prompt.
- **Shared state**: in LangGraph, this is an explicit `State` object (Pydantic/TypedDict) passed between nodes; conditional edges route based on state (e.g., "if critic rejects, loop back to writer").
- **Termination/guardrails**: max iteration counts, confidence thresholds, human-in-the-loop approval for high-stakes actions.
- **Observability**: every agent call is traced (see tracing question below) so you can debug which agent made which decision.

Key design principle: keep each agent's responsibility narrow — broad, do-everything agents are harder to prompt reliably and harder to evaluate.

### 21.46 What is A2A (Agent-to-Agent) protocol, and how is it different from MCP?

MCP standardizes how an agent talks to *tools and data sources*. A2A standardizes how *agents talk to each other* — enabling agents built on different frameworks or by different vendors to discover each other's capabilities (via an "agent card"), negotiate a task, and exchange messages/results in a common format. Think of MCP as vertical (agent → tool) and A2A as horizontal (agent → agent). Together they let you build heterogeneous multi-agent ecosystems rather than a single monolithic framework.

### 21.47 When should you not build an agent?

"Agent" has become a default answer, but many problems are better solved with a fixed workflow or a single model call. Knowing when *not* to use an agent is a sign of seniority.

**Workflows compared with agents**

- A **workflow** follows a path defined in code: call the model to classify, then retrieve, then generate, then validate. The developer decides the steps.
- An **agent** decides its own steps dynamically, choosing tools and how many iterations to run.

| Factor | Prefer a workflow or single call | Prefer an agent |
| --- | --- | --- |
| Steps | Known in advance and stable | Unknown; depend on what is discovered |
| Predictability | Needed (compliance, auditing, strict SLAs) | Some variation acceptable |
| Latency and cost | Tight budgets | Higher cost justified by task value |
| Error tolerance | Low; failures are costly | Errors can be caught and corrected |
| Verification | Hard to check results automatically | Results can be checked (tests, validators) |

**Signs an agent is the wrong choice**

- The task is a single transformation: classification, extraction, summarisation, translation.
- The steps are the same every time; an agent only adds variability.
- There is no way to verify progress, so the agent cannot tell when it is going wrong.
- Every action is high-risk and would need human approval anyway.
- Latency must be a few hundred milliseconds.

**Good uses of agents**

- Open-ended tasks with an unpredictable number of steps: debugging, research, data investigation, complex customer cases.
- Environments that give feedback the agent can act on: tests, query results, error messages.
- Tasks where a human would also need to explore and adapt.

**A pragmatic progression**

1. Single model call with good context.
2. A fixed workflow (prompt chaining, routing, parallel calls).
3. An agent with a small set of well-designed tools, limits and evaluation.
4. Multi-agent systems only where clearly justified.

Move to the next level only when evaluation shows the simpler approach cannot meet the requirement.

### 21.48 How do you build observability and tracing for agents?

An agent run is a tree of model calls, tool calls, retrievals and sub-agents. Without tracing, failures are nearly impossible to debug: you see a bad final answer but not the step where it went wrong. Observability is a prerequisite for evaluation, cost control and safety.

**What to capture**

- **A trace per task,** with nested **spans** for each step: model calls, tool calls, retrieval, sub-agent runs, guardrail checks and human approvals.
- For model calls: model and version, prompt version, input and output (or references to them), tokens, latency, cost, stop reason.
- For tool calls: tool name and version, arguments, result or error, latency, retries.
- Task-level data: user or tenant, outcome (success, failure, escalation), total cost and duration, feedback.
- Identifiers that link traces to logs, application requests and evaluation results.

**Standards and tools**

- **OpenTelemetry** has semantic conventions for generative AI, so traces from different frameworks share a common schema and can go to existing observability backends.
- LLM-focused platforms (for example Langfuse, LangSmith, Arize Phoenix) add trace visualisation, prompt management, evaluation and dataset building from traces.

**What to monitor**

- Task success rate, escalation rate, and user feedback.
- Steps, tokens and cost per task, with alerts on outliers (a runaway loop shows up as a cost spike).
- Tool error rates and latency per tool.
- Guardrail triggers, refusals and permission denials.
- Drift after any change to model, prompt or tools.

**Using traces**

- **Debugging:** replay a failed run step by step.
- **Evaluation:** turn real traces into test cases; score sampled production traces with judges or humans.
- **Audit:** for regulated actions, keep an immutable record of what the agent saw and did.

**Privacy:** traces contain user data and tool outputs. Redact sensitive fields, control access, and set retention periods.

```python
from opentelemetry import trace

tracer = trace.get_tracer("support-agent")

def run_tool_traced(name, args):
    with tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.args_size", len(str(args)))
        try:
            result = TOOLS[name](**args)
            span.set_attribute("tool.status", "ok")
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_attribute("tool.status", "error")
            raise

with tracer.start_as_current_span("agent.task") as task_span:
    task_span.set_attribute("tenant.id", tenant_id)
    answer = agent.run(user_request, tool_runner=run_tool_traced)
```

### 21.49 How do you handle identity, authentication and permissions for agents?

When an agent acts, whose authority is it using? Getting this wrong leads to agents that can read data the user should not see, or take actions nobody approved. Identity and permissions are central to running agents safely in an enterprise.

**Key principles**

- **Act on behalf of the user, not as a super-user.** The agent should have at most the permissions of the person it is acting for, and usually fewer.
- **Least privilege per task:** grant only the scopes the current task needs (read calendar, not delete email).
- **Short-lived, scoped credentials:** tokens that expire and are limited to specific resources, rather than long-lived API keys with broad access.
- **Separate agent identity:** give the agent (or each agent type) its own identity, so its actions are distinguishable from the user's in audit logs, and its permissions can be managed and revoked.

**Mechanisms**

- **OAuth delegated authorisation:** the user grants the agent limited scopes to act on their behalf; the agent uses access tokens, never the user's password. MCP's authorisation for remote servers is based on OAuth.
- **Token exchange and on-behalf-of flows:** exchange the user's token for a narrower token for a specific downstream service.
- **Policy enforcement outside the model:** the tool layer checks every call against policy (who, what action, which resource, under what conditions). The model can request an action; the system decides whether it is allowed.
- **Step-up approval:** high-risk actions (payments, deletions, external messages) require explicit user confirmation, possibly with re-authentication.

**Risks to design against**

- **Confused deputy:** an agent with broad permissions is tricked (for example by prompt injection in a document) into using them for someone else's goal. Mitigate with per-user permissions and policy checks at the tool layer.
- **Credential leakage:** never put secrets in prompts or tool outputs that reach the model.
- **Permission creep:** agents accumulate scopes over time; review and expire grants.

**Audit:** log every action with the user, the agent identity, the tool, the arguments, the authorisation decision and the outcome.

### 21.50 What is durable execution, and why do long-running agents need it?

Agents increasingly run for minutes, hours or even days: researching, processing many records, or waiting for human approvals. Long runs fail for ordinary reasons: a server restarts, a deployment happens, an API is down for an hour. Without durability, the agent loses its progress, or worse, repeats side effects when restarted.

**Durable execution** means the agent's progress is persisted step by step, so a run can resume exactly where it left off after a crash or a long pause.

**Core ideas**

- **Checkpointing:** after each step, save the agent's state (messages, plan, results so far, pending actions) to durable storage.
- **Deterministic replay:** on restart, the workflow engine replays recorded results of completed steps instead of executing them again, then continues from the first unfinished step.
- **Idempotent side effects:** each external action carries an idempotency key, so a retried call (for example creating an order or sending an email) does not happen twice.
- **Compensation (sagas):** for multi-step processes, define how to undo earlier steps if a later step fails.
- **Durable waits:** pause for hours or days for a human approval or an external event without holding a process open.

**Tools**

- Workflow engines built for durable execution, such as Temporal, and cloud workflow services.
- Agent frameworks with persistence, for example LangGraph checkpointers that save graph state and support resuming and human-in-the-loop interruptions.

**Design guidance**

- Separate **decisions** (model calls, which are non-deterministic) from **actions** (side effects). Record each model output once and reuse it during replay, so the run does not change course on restart.
- Version workflows carefully: a run started on an old version must be able to finish after a deployment.
- Set timeouts and budgets for each step and for the whole run.

```python
import hashlib, json

def idempotency_key(run_id, step, args):
    return hashlib.sha256(f"{run_id}:{step}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()

def execute_step(store, run_id, step, tool, args):
    key = idempotency_key(run_id, step, args)
    if (done := store.get(key)) is not None:      # already executed before a crash: reuse the result
        return done
    result = tool(**args, idempotency_key=key)    # downstream service also deduplicates on the key
    store.put(key, result)                        # checkpoint before moving to the next step
    return result
```

### 21.51 How do coding agents work, and what makes them effective?

Coding agents (in IDEs, terminals and cloud services) are among the most widely deployed agents. They show clearly what makes agents work: good tools, fast feedback and careful context management.

**The loop**

1. **Understand the task:** read the request, any linked issue, and relevant project instructions (conventions, how to build and test).
2. **Explore the codebase:** search for files and symbols, read relevant code, find similar patterns. Large repositories do not fit in context, so exploration tools (search, file listing, reading line ranges) and sub-agents matter.
3. **Plan:** decide which files to change; for larger tasks, write down a plan.
4. **Edit:** make targeted changes through edit tools (search-and-replace or diff-based edits rather than rewriting whole files).
5. **Verify:** run the build, tests, linters and type checkers; read the errors; fix and repeat.
6. **Finish:** summarise the changes, and optionally commit or open a pull request for human review.

**What makes them effective**

- **Executable feedback.** Tests and compilers provide fast, objective signals; an agent that can run them corrects most of its own mistakes.
- **Good tool design:** precise search, reliable edits that fail loudly when the target text is not found, and readable command output.
- **Project instructions:** files that describe build commands, conventions and constraints save exploration and prevent common errors.
- **Context management:** reading only what is needed, summarising long outputs, delegating broad searches to sub-agents.
- **Safe execution:** sandboxes or containers, permission prompts for risky commands (deleting files, pushing code, network access), and no production credentials.

**Evaluation**

- **SWE-bench Verified:** real GitHub issues with hidden tests; success means the agent's patch makes the failing tests pass without breaking others. Scores rose quickly through 2024 to 2026, though benchmark scores overstate real-world reliability on messy codebases.
- In practice, teams measure accepted changes, review time, test pass rates, and the rate of reverted changes.

**Risks**

- Plausible but wrong changes that pass weak tests; tests modified to pass instead of fixing the code (a form of reward hacking).
- Security issues in generated code and in dependencies the agent adds.
- Destructive commands; mitigate with sandboxing, permissions and version control.

**Human role:** coding agents work best with clear task descriptions, strong test suites and code review. They change the engineer's job towards specifying, reviewing and verifying.

### 21.52 How do you deploy and operate agents in production?

Shipping an agent is closer to operating a service with a non-deterministic core than to deploying a model. Operations need the usual software discipline plus controls specific to autonomous behaviour.

**Versioning and release**

- Version the whole configuration together: model and settings, system prompt, tool definitions and implementations, retrieval settings, guardrails.
- **Evaluation gates:** every change must pass the agent evaluation suite (success rate, reliability, safety tests, cost per task) before release.
- **Gradual rollout:** shadow mode (agent proposes, humans act), then a small percentage of traffic, then wider, with automatic rollback when metrics degrade.
- **Pin model versions** and plan migrations when providers deprecate models.

**Runtime controls**

- **Budgets** per task: maximum steps, tokens, cost and wall-clock time.
- **Rate limits** per user and per tenant, and on expensive or risky tools.
- **Kill switch:** the ability to disable the agent, a specific tool, or a type of action immediately, without a deployment.
- **Approval policies** for consequential actions, configurable per tenant.
- **Fallbacks:** when the agent fails or is disabled, route to a simpler workflow or a human.

**Service levels and monitoring**

- Define SLOs that reflect agent behaviour: task success rate, escalation rate, p95 time to completion, cost per successful task, safety incident rate.
- Monitor traces, tool error rates, guardrail triggers, and cost anomalies; alert on loops and runaway spending.

**Incident response**

- Playbooks for agent-specific incidents: harmful or wrong actions, data exposure, prompt-injection exploitation, cost explosions.
- Traces provide the evidence; every incident becomes a regression test and often a new guardrail.

**Organisation**

- Clear ownership of each agent (product, engineering, risk), a change review process for prompts and tools, and regular review of permissions and costs.

### 21.53 How are models trained to be better agents?

Agentic ability (planning over many steps, using tools correctly, recovering from errors, knowing when to stop) is now a major training target for frontier and open models, not just a matter of prompting.

**1. Supervised fine-tuning on trajectories**

- Collect or generate examples of complete agent runs: the task, each reasoning step, tool calls and results, and the final answer.
- Sources: human demonstrations, strong models acting in environments, filtered to keep only successful trajectories (rejection sampling).
- Teaches the format of tool use and good habits (checking results, asking for clarification), but only imitates; it does not learn from failures directly.

**2. Reinforcement learning in environments**

- The model acts in an **environment** (a code repository with tests, a simulated website, a database, a simulated customer with a policy) and receives a **reward** for task success.
- **Verifiable rewards** are preferred: tests pass, the database ends in the correct state, the answer matches. This extends RL with verifiable rewards from single-answer problems (Chapter 22) to multi-step tool use.
- Algorithms such as GRPO and PPO variants are applied to whole trajectories.

**Challenges**

- **Long-horizon credit assignment:** a reward at the end of 50 steps says little about which step mattered; shaping and intermediate checks help.
- **Environment engineering:** building many diverse, realistic, resettable environments at scale is a large share of the work.
- **Reward hacking:** agents find shortcuts, such as editing tests so they pass, special-casing evaluation checks, or claiming success without doing the work. Environments and verifiers must be robust, and trajectories must be audited.
- **Cost:** each rollout involves many model calls and tool executions.

**3. Preference and safety training**

- Training against unsafe actions, excessive autonomy, and ignoring user instructions; and towards asking for confirmation when appropriate.

**Why this matters to practitioners**

- Model choice for agents should be based on agentic evaluations (task success and reliability in environments similar to yours), not on general chat benchmarks.
- Fine-tuning smaller models on your own successful trajectories can cut cost for narrow, high-volume agent tasks, provided you have a solid evaluation suite and environment.

---

← [Chapter 20. Retrieval-Augmented Generation](20-retrieval-augmented-generation.md) · [Table of contents](../README.md) · [Chapter 22. Fine-Tuning and Model Adaptation](22-fine-tuning-and-model-adaptation.md) →
