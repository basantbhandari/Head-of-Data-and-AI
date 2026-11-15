# Backend Engineering: AI Security and ML Topics

AI jailbreaking and federated learning.

### Q1. AI Jailbreaking

#### AI Jailbreaking

- **AI jailbreaking** is the process of attempting to bypass an AI model’s safety policies, restrictions, or guardrails so it produces content it would normally refuse.
- It generally exploits **weaknesses in instruction following and context interpretation**, rather than traditional software vulnerabilities.

#### How AI Jailbreaking Works

- **Role-play:** Asking the model to act as a fictional character or unrestricted system.
- **Hypothetical framing:** Presenting a prohibited request as a hypothetical, simulation, or research scenario.
- **Instruction manipulation:** Crafting conflicting or misleading instructions intended to make the model prioritize the attacker’s request.
- **Encoding/obfuscation:** Altering or disguising harmful instructions to make them harder for safety mechanisms to recognize.
- **Context manipulation:** Providing large amounts of carefully constructed context designed to weaken or confuse the model’s safety constraints.

#### Jailbreaking vs. Prompt Injection

- **Jailbreaking:** Primarily attempts to make the model violate its **safety policies or behavioral restrictions** through crafted interaction.
- **Prompt injection:** Attempts to manipulate an AI system by placing malicious instructions in **user input or external data** such as documents, webpages, emails, or tool outputs.
- They can overlap: a prompt injection can sometimes be used as part of a jailbreak, especially in **agentic systems**.

#### Major Risks

- **Cybersecurity:** Encouraging generation of malware, phishing content, or exploit instructions.
- **Data leakage:** Attempting to extract system prompts, confidential information, or sensitive data.
- **Agent abuse:** Causing an AI agent to perform unauthorized actions through its tools.
- **Safety bypass:** Circumventing content moderation or organizational policies.
- **Reputational and compliance risk:** Causing an enterprise AI system to produce inappropriate or prohibited outputs.

#### How to Defend Against Jailbreaking

- Use **input/output safety classifiers** and policy enforcement.
- Apply **instruction hierarchy** so untrusted content cannot override system/developer instructions.
- Treat retrieved documents, webpages, emails, and tool outputs as **untrusted data**, not instructions.
- Validate and constrain **tool calls** with permissions and allowlists.
- Use **rate limiting and abuse monitoring**.
- Maintain **audit logs** for prompts, model decisions, and tool actions.
- Perform continuous **red-team testing** with adversarial prompts.
- Use **defense in depth** rather than relying on the LLM's refusal behavior alone.

### Q2. Federated Learning

> 🔁 _See also [commonly-asked-ml-questions.md → Q81](../../ai-ml-engineering/commonly-asked-ml-questions.md#q81-federated-learning) (separate notes on the same question)._

#### Federated Learning Model

**Federated Learning (FL)** is a machine learning approach where multiple devices or organizations **train a shared model collaboratively without sending their raw data to a central server**.

- **The central server** initializes and distributes the global model.
- **Clients** (devices, hospitals, banks, etc.) train the model locally using their private data.
- Clients send **model updates** (e.g., gradients or parameters), not raw data.
- The server **aggregates** the updates to create an improved global model.
- The updated global model is sent back to clients.
- This process repeats until the model converges.

#### Typical workflow

**Global Model → Local Training → Model Updates → Aggregation → Updated Global Model → Repeat**

#### Common aggregation method

**FedAvg (Federated Averaging)** averages client model parameters, typically weighted by the amount of training data each client has.

#### Advantages

- Keeps raw data decentralized.
- Reduces the need to transfer sensitive data.
- Useful for privacy-sensitive applications.
- Allows organizations to collaborate without directly sharing datasets.

#### Challenges

- **Non-IID data:** Clients may have very different data distributions.
- **Communication cost:** Frequent model updates can be expensive.
- **Client heterogeneity:** Devices have different compute and network capabilities.
- **Security:** Malicious clients can perform model poisoning attacks.
- **Privacy leakage:** Model updates can potentially reveal information.
