# Chapter 25. Backend Engineering, APIs and DevOps

_Part V — Production Systems and Architecture_

HTTP and APIs, latency and reliability engineering, scalability, CI/CD, testing and containerization.

This chapter contains **20 questions**.

## Contents

- **25.1** [What are the HTTP response status code classes?](#251-what-are-the-http-response-status-code-classes)
- **25.2** [What is an API, and what are its types?](#252-what-is-an-api-and-what-are-its-types)
- **25.3** [Define the purpose of the back-end.](#253-define-the-purpose-of-the-back-end)
- **25.4** [What is a web server?](#254-what-is-a-web-server)
- **25.5** [What's the difference between GET and POST?](#255-whats-the-difference-between-get-and-post)
- **25.6** [Mention some advantages of web services.](#256-mention-some-advantages-of-web-services)
- **25.7** [What does REST stand for?](#257-what-does-rest-stand-for)
- **25.8** [What is the difference between static and dynamic websites?](#258-what-is-the-difference-between-static-and-dynamic-websites)
- **25.9** [What do p50, p75, p90, p95 and p99 latency metrics mean?](#259-what-do-p50-p75-p90-p95-and-p99-latency-metrics-mean)
- **25.10** [What is exponential backoff with jitter?](#2510-what-is-exponential-backoff-with-jitter)
- **25.11** [How can you build scalability into a software program?](#2511-how-can-you-build-scalability-into-a-software-program)
- **25.12** [What is the CAP theorem?](#2512-what-is-the-cap-theorem)
- **25.13** [What does High Availability (HA) mean?](#2513-what-does-high-availability-ha-mean)
- **25.14** [What is the typical workflow to implement a new feature on the back-end?](#2514-what-is-the-typical-workflow-to-implement-a-new-feature-on-the-back-end)
- **25.15** [Talk about continuous integration and continuous delivery.](#2515-talk-about-continuous-integration-and-continuous-delivery)
- **25.16** [Differentiate between acceptance and functional tests.](#2516-differentiate-between-acceptance-and-functional-tests)
- **25.17** [Mention some performance testing steps.](#2517-mention-some-performance-testing-steps)
- **25.18** [What is Containerization?](#2518-what-is-containerization)
- **25.19** [What is Database Black Box Testing?](#2519-what-is-database-black-box-testing)
- **25.20** [Design a distributed rate limiter.](#2520-design-a-distributed-rate-limiter)

---

### 25.1 What are the HTTP response status code classes?

#### HTTP Response Status Codes

HTTP status codes are **three-digit codes** a server returns to indicate the result of an HTTP request.

- **1xx – Informational:** Request received; processing continues.
- **2xx – Success:** Request was successfully processed.
- **3xx – Redirection:** Client needs to follow a redirect or take additional action.
- **4xx – Client Error:** The request is invalid, or the client is not authorized.
- **5xx – Server Error:** The server encountered an error while processing a valid request.

#### Common Status Codes

- **200 OK** — Request succeeded.
- **201 Created** — A new resource was successfully created.
- **204 No Content** — Request succeeded with no response body.
- **301 Moved Permanently** — Resource has permanently moved to another URL.
- **302 Found** — Temporary redirect.
- **400 Bad Request** — Invalid or malformed request.
- **401 Unauthorized** — Authentication is missing or invalid.
- **403 Forbidden** — Client is authenticated but not allowed to access the resource.
- **404 Not Found** — Requested resource does not exist.
- **409 Conflict** — Request conflicts with the current state of the resource.
- **429 Too Many Requests** — Client has exceeded the allowed request rate.
- **500 Internal Server Error** — Unexpected server-side error.
- **502 Bad Gateway** — Gateway/proxy received an invalid response from an upstream server.
- **503 Service Unavailable** — Server is temporarily unavailable or overloaded.
- **504 Gateway Timeout** — Gateway/proxy did not receive a timely response from an upstream server.

### 25.2 What is an API, and what are its types?

An **API (Application Programming Interface)** is a defined way for two software systems to communicate. It specifies what requests can be made, what data should be supplied, and what responses come back.

Common **types of APIs** include:

- **REST API** — Uses HTTP methods such as `GET`, `POST`, `PUT`, `PATCH`, and `DELETE`. Data is commonly exchanged as JSON.
- **SOAP API** — Uses a stricter XML-based protocol with standardized messaging and contracts.
- **GraphQL API** — Lets clients request exactly the fields they need, usually through a single endpoint.
- **RPC APIs** — Treat communication like calling functions remotely. Examples include **gRPC** and JSON-RPC.
- **WebSocket API** — Maintains a persistent two-way connection, useful for live chats, games, notifications, and real-time market data.
- **Web APIs** — A broad category for APIs accessed over a network using HTTP/HTTPS; REST and GraphQL are examples.
- **Library/Framework APIs** — Functions, classes, and methods exposed by a programming library, such as a Python package.
- **Operating System APIs** — Allow programs to interact with OS features such as files, networking, processes, cameras, or notifications.

For example, a weather REST API might work conceptually as:

```txt
GET /weather?city=London
```

and return:

```txt
{
  "city":"London",
  "temperature":18,
  "condition":"Cloudy"
}
```

If by **“API and its types”** you mean this as a computer-science/exam topic, the simplest classification to remember is **REST, SOAP, GraphQL, RPC, and WebSocket APIs**.

### 25.3 Define the purpose of the back-end.

#### Purpose of the Back-End

- The **back-end** is the server-side part of an application that handles **business logic, data processing, security, and communication with databases and external services**.
- It receives requests from the front-end, processes them, and sends appropriate responses.
- Main responsibilities include:
  - **Business logic** — implements application rules and workflows.
  - **Database management** — stores, retrieves, updates, and deletes data.
  - **API handling** — provides APIs for the front-end and other services.
  - **Authentication & authorization** — verifies users and controls access.
  - **Data processing** — validates, transforms, and processes information.
  - **Integration** — communicates with external APIs, services, queues, and cloud systems.
  - **Security & reliability** — protects data and handles errors, logging, and monitoring.

**In short:** The back-end is the **engine behind an application** that processes requests, manages data and logic, and provides services to the front-end.

### 25.4 What is a web server?

A **web server** is a software application and/or computer system that **receives HTTP/HTTPS requests from clients and sends back responses**, such as web pages, files, or API data.

- **Client:** Browser, mobile app, or another service
- **Web server:** Receives and processes HTTP/HTTPS requests
- **Response:** HTML, JSON, images, files, etc.
- **Common web servers:** Nginx, Apache HTTP Server, Microsoft IIS
- **Example:** A browser requests `GET /users`  the web server receives the request and returns the appropriate response.

In a modern back-end system, a web server often sits in front of an **application server/API**, handling incoming traffic and forwarding requests to the appropriate application service.

### 25.5 What's the difference between GET and POST?

- **GET:** Used to **retrieve data** from the server.
  - Parameters are usually sent in the **URL/query string**.
  - Should not modify server data.
  - Can be cached and bookmarked.
  - Example: `GET /users/123`
- **POST:** Used to **send data to the server**, typically to create a resource or trigger an operation.
  - Data is usually sent in the **request body**.
  - Not normally cached or bookmarked.
  - Example: `POST /users` with user information in the body.

**In short:**

**GET = retrieve data**

**POST = submit/send data**

### 25.6 Mention some advantages of web services.

- **Interoperability:** Allows different applications, platforms, and programming languages to communicate with each other.
- **Reusability:** A web service can be reused by multiple applications and clients.
- **Loose coupling:** Services can operate independently, making systems easier to modify and maintain.
- **Scalability:** Individual services can be scaled based on demand.
- **Platform independence:** Clients can consume services regardless of their underlying operating system or technology.
- **Easy integration:** Makes it easier to integrate different systems and third-party applications.
- **Remote access:** Services can be accessed over a network using standard protocols such as HTTP/HTTPS.
- **Maintainability:** Changes to one service can often be made without changing the entire application.

**In short:** Web services provide **interoperability, reusability, flexibility, scalability, and easier system integration**.

### 25.7 What does REST stand for?

**REST** stands for **Representational State Transfer**.

It is an **architectural style for designing web APIs** that use standard HTTP methods to interact with resources.

- **GET** → Retrieve a resource
- **POST** → Create a resource
- **PUT** → Replace/update a resource
- **PATCH** → Partially update a resource
- **DELETE** → Delete a resource

**In short:** REST is a set of principles for building **simple, scalable, stateless, and resource-oriented web APIs**.

### 25.8 What is the difference between static and dynamic websites?

#### Static vs Dynamic Websites

- **Static Website**
  - Content is **pre-built and served as stored files**.
  - The content generally remains the same for all users unless the files are changed.
  - Usually faster, simpler, and easier to host.
  - Common technologies: **HTML, CSS, JavaScript**.
  - Example: A portfolio or company information website.
- **Dynamic Website**
  - Content is **generated or updated based on user requests, data, or application logic**.
  - Can provide personalized and interactive experiences.
  - Usually involves a **back-end application and database**.
  - Common technologies: **Python, Java, Node.js, PHP, databases**, etc.
  - Example: An e-commerce site showing products and prices from a database.

**In short:**

**Static = pre-built, mostly fixed content.**

**Dynamic = content generated or updated based on data and user interaction.**

### 25.9 What do p50, p75, p90, p95 and p99 latency metrics mean?

#### P50, P75, P90, P95, and P99 Latency

These are **percentile latency metrics** used to understand how fast or slow a system is across different requests.

- **P50 (Median):** 50% of requests are completed within this latency.
- **P75:** 75% of requests are completed within this latency.
- **P90:** 90% of requests are completed within this latency.
- **P95:** 95% of requests are completed within this latency.
- **P99:** 99% of requests are completed within this latency.

##### Example

Suppose an API has:

- P50 = **100 ms**
- P90 = **200 ms**
- P95 = **300 ms**
- P99 = **800 ms**

This means:

- 50% of requests finish in **≤100 ms**
- 90% finish in **≤200 ms**
- 95% finish in **≤300 ms**
- 99% finish in **≤800 ms**
- The slowest 1% take **more than 800 ms**

##### Why P99 matters

**Average latency can hide slow requests.** Percentiles show the experience of users at the slower end.

For production AI/LLM systems, you might monitor:

- **P50:** Typical user experience
- **P95:** Near-worst-case experience
- **P99:** Tail latency and extreme slowdowns

A large gap between P50 and P99 often indicates **tail-latency problems**, such as overloaded servers, database contention, network delays, cold starts, long LLM generation, or rate limiting.

### 25.10 What is exponential backoff with jitter?

**Exponential backoff with jitter** is a retry strategy used in distributed systems when an operation fails temporarily, such as an API call, database request, or LLM request.

#### How it works

Instead of retrying immediately, the system waits longer after each failure:

- 1st retry → wait \~1 second
- 2nd retry → wait \~2 seconds
- 3rd retry → wait \~4 seconds
- 4th retry → wait \~8 seconds
- 5th retry → wait \~16 seconds

This is **exponential backoff**.

#### What is jitter?

**Jitter adds a random amount of variation to the delay.**

For example, instead of every server retrying exactly after 4 seconds:

```txt
Server A → 3.2 seconds
Server B → 4.7 seconds
Server C → 3.8 seconds
Server D → 4.3 seconds
```

This prevents thousands of clients from retrying simultaneously.

#### Why is jitter important?

Imagine an LLM API goes down, and **10,000 clients** retry after exactly 8 seconds:

```txt
Failure
   ↓
Wait 8 seconds
   ↓
10,000 retries at the same time
   ↓
API overloaded again
   ↓
Failure
   ↓
Wait 16 seconds
   ↓
10,000 retries again
```

This creates a **thundering herd problem**.

With jitter:

```txt
Failure
   ↓
Exponential backoff
   +
Random jitter
   ↓
Retries spread over time
   ↓
Lower load
   ↓
Higher recovery probability
```

#### Common implementation

```txt
delay=min(max_delay,base_delay*2**retry)delay+=random.uniform(0,jitter)
```

For production systems, you would also typically use:

- **Maximum retry limit**
- **Maximum backoff delay**
- **Jitter**
- **Retry only transient failures** such as `429`, `502`, `503`, `504`
- **Do not retry permanent errors** such as `400` or invalid authentication

### 25.11 How can you build scalability into a software program?

- **Use stateless services:** Keep application servers stateless so requests can be distributed across multiple instances.
- **Horizontal scaling:** Add more application instances behind a **load balancer** instead of relying only on a larger server.
- **Database optimization:** Use proper indexing, query optimization, connection pooling, and read replicas when needed.
- **Caching:** Cache frequently accessed data using systems such as Redis to reduce database load.
- **Asynchronous processing:** Use message queues such as Kafka or RabbitMQ for long-running or resource-intensive tasks.
- **Efficient APIs:** Design lightweight APIs with pagination, filtering, rate limiting, and appropriate response sizes.
- **Microservices when appropriate:** Separate independently scalable components when the system's complexity justifies it.
- **CDN and object storage:** Offload static files and large objects to services designed to handle high traffic.
- **Monitoring and autoscaling:** Track CPU, memory, latency, throughput, and errors, and automatically scale based on demand.
- **Load testing:** Test the system under expected and peak traffic before production.

**In short:** Build scalability by designing for **horizontal scaling, efficient data access, caching, asynchronous processing, stateless services, and automated monitoring and scaling**.

### 25.12 What is the CAP theorem?

- **CAP theorem** states that a distributed system can guarantee at most **two of three properties** simultaneously:

1. **Consistency (C):** Every read receives the most recent write or an error.
2. **Availability (A):** Every request receives a response, even if some nodes are unavailable.
3. **Partition Tolerance (P):** The system continues operating despite network communication failures between nodes.

#### Important point

In a distributed system, **network partitions can happen**, so partition tolerance is generally required. Therefore, the practical trade-off is usually between:

- **CP:** Consistency + Partition Tolerance
  - Prioritizes correct, consistent data.
  - Example: **etcd, ZooKeeper**
- **AP:** Availability + Partition Tolerance
  - Prioritizes availability and can temporarily allow inconsistent data.
  - Example: **Cassandra, DynamoDB** (depending on configuration/operation)

**In short:** CAP is about the trade-off between **Consistency, Availability, and Partition Tolerance during network partitions**.

### 25.13 What does High Availability (HA) mean?

**High Availability (HA)** means designing a system so it remains **available and operational with minimal downtime**, even when failures occur.

- **Redundancy:** Multiple servers or components prevent a single point of failure.
- **Failover:** Traffic automatically moves to a healthy instance when one fails.
- **Load balancing:** Distributes traffic across multiple instances.
- **Health monitoring:** Detects failures and unhealthy services.
- **Auto-scaling:** Adds or removes resources based on demand.
- **Reliable infrastructure:** Uses backups, replicated databases, and multiple availability zones.

**Example:** If one API server fails, a load balancer routes requests to other healthy servers, so users continue using the application.

**In short:** HA = **keeping a system running and accessible despite failures.**

### 25.14 What is the typical workflow to implement a new feature on the back-end?

The typical workflow I follow to implement a new back-end feature is:

1. **Understand the requirements** — clarify the business goal, expected behavior, edge cases, and acceptance criteria.
2. **Design the solution** — decide the API contract, data model, service architecture, and dependencies.
3. **Create a development branch** — keep the feature isolated from the main codebase.
4. **Implement the feature** — develop the API, business logic, database changes, validations, and error handling.
5. **Write tests** — add unit and integration tests and cover important edge cases.
6. **Run code quality checks** — perform linting, formatting, security checks, and review the implementation.
7. **Test locally** — verify the API and its integration with databases and other services.
8. **Create a pull request** — document the changes and get feedback through code review.
9. **Deploy through CI/CD** — merge and deploy to staging for further validation, followed by production after approval.
10. **Monitor after deployment** — check logs, metrics, errors, latency, and user impact to ensure the feature works reliably.

Overall, I follow a cycle of **requirements → design → implementation → testing → code review → deployment → monitoring**.

### 25.15 Talk about continuous integration and continuous delivery.

- **Continuous Integration (CI):** The practice of frequently merging code changes into a shared repository. Each change automatically triggers **builds, automated tests, linting, and code-quality checks** to detect problems early.
- **Continuous Delivery (CD):** Extends CI by automatically preparing validated code for **production release**. The application is kept in a deployable state, and production deployment may require manual approval.

#### Typical CI/CD workflow

**Code → Git → Build → Test → Quality/Security Checks → Staging → Production**

- **CI:** Automates build and testing whenever code is pushed or a pull request is created.
- **Continuous Delivery:** Automatically deploys validated changes to staging and keeps them ready for production.
- **Continuous Deployment:** Goes one step further by automatically deploying validated changes directly to production.

**In short:**

**CI = continuously integrate and validate code.**

**Continuous Delivery = continuously prepare validated code for release.**

**Continuous Deployment = automatically release validated code to production.**

### 25.16 Differentiate between acceptance and functional tests.

- **Acceptance Testing:** Verifies whether the **complete system meets the business requirements and is acceptable to the customer or end user**.
  - Focus: **“Does the system meet the user's/business needs?”**
  - Usually performed toward the end of development.
  - Example: A customer verifies that an e-commerce system supports the required checkout workflow.
- **Functional Testing:** Verifies whether **specific features and functions work according to their requirements**.
  - Focus: **“Does each feature work correctly?”**
  - Usually performed throughout development and testing.
  - Example: Testing whether the checkout API correctly calculates the total price and processes an order.

**In short:**

**Functional testing = verifies individual functions.**

**Acceptance testing = verifies the complete system against business/user requirements.**

#### Repeated later as: "What is the difference between Acceptance Test and Functional Test?"

- **Functional Testing:** Verifies that a **specific feature or function** works according to its requirements.
  - Example: Testing whether a login API correctly validates username and password.
  - Usually performed throughout development/testing.
- **Acceptance Testing:** Verifies that the **complete system meets business and user requirements** and is acceptable for release.
  - Example: Testing the entire login and registration workflow from the user's perspective.
  - Usually performed near the end of development.

**In short:**

**Functional test = Does the feature work correctly?**

**Acceptance test = Does the complete system meet the user's/business needs?**

### 25.17 Mention some performance testing steps.

- **Define performance requirements:** Establish expected response time, throughput, concurrent users, and resource limits.
- **Identify critical scenarios:** Select important APIs, workflows, and operations to test.
- **Prepare the test environment:** Configure an environment similar to production and use representative test data.
- **Create test scenarios:** Define realistic workloads and user behavior.
- **Run baseline tests:** Measure normal performance under expected traffic.
- **Run load tests:** Gradually increase traffic to determine how the system performs under expected load.
- **Run stress tests:** Push the system beyond normal capacity to identify its breaking point.
- **Monitor metrics:** Track latency, throughput, CPU, memory, database performance, and error rates.
- **Analyze bottlenecks:** Identify slow queries, inefficient code, network issues, or resource constraints.
- **Optimize and retest:** Apply improvements and repeat the tests to verify performance gains.

**In short:**

**Define requirements → prepare environment → create workload → test → monitor → identify bottlenecks → optimize → retest.**

### 25.18 What is Containerization?

**Containerization** is the process of packaging an application together with its **code, dependencies, libraries, and configuration** into a lightweight, isolated container so it can run consistently across different environments.

- **Consistency:** Runs the same way across development, testing, and production.
- **Isolation:** Applications and their dependencies are separated from each other.
- **Portability:** Containers can run on different machines and cloud platforms.
- **Scalability:** Containers can be easily replicated and scaled.
- **Fast startup:** Containers are generally faster and lighter than virtual machines.
- **Common tool:** **Docker** is widely used for containerization.

**Example:** Package a FastAPI application with Python dependencies into a Docker container and run the same container locally, in staging, and in production.

**In short:** Containerization = **packaging an application and its dependencies into an isolated, portable environment.**

### 25.19 What is Database Black Box Testing?

**Database Black Box Testing** is a testing approach where the **database is tested from the user's or application's perspective without examining its internal implementation**.

- Focuses on **inputs and outputs**, not internal database code.
- Tests whether data is correctly **inserted, updated, retrieved, and deleted**.
- Verifies **data integrity and business rules**.
- Checks constraints such as `PRIMARY KEY`, `FOREIGN KEY`, `UNIQUE`, and `NOT NULL`.
- Tests database responses to valid and invalid inputs.
- Does not require knowledge of the database's internal implementation.

**Example:** Insert a customer through an application and verify that the correct customer data is stored and retrieved.

**In short:** Database black box testing verifies **database behavior and results without looking at its internal structure or implementation**.

### 25.20 Design a distributed rate limiter.

**Requirements:** limit requests per key (user, API key, IP, tenant) across many stateless API servers, e.g. 100 req/min. Low added latency (< a few ms), high availability, roughly accurate globally, and support for different limits per tier.

**Algorithms**

| Algorithm | Notes |
| --- | --- |
| Fixed window counter | Simple `INCR key:minute`; allows 2× bursts at window edges |
| Sliding window log | Exact; stores every timestamp, so memory-heavy |
| Sliding window counter | Weighted mix of current and previous window; good accuracy, O(1) memory |
| **Token bucket** | Refill rate + burst capacity; the industry default (allows controlled bursts) |
| Leaky bucket | Smooths output rate; good for shaping traffic to downstreams |

**Architecture**

```text
Client → LB → API servers (rate-limit middleware) → Redis Cluster (sharded by key)
                                   ↘ config service (limits per tier, cached locally)
```

- State lives in a **central fast store (Redis)**, sharded by rate-limit key so one key lives on one shard.
- **Atomicity:** do check-and-decrement in a single **Lua script** (or `INCR` + `EXPIRE`) to avoid races between servers:

```lua
-- token bucket: KEYS[1]=bucket, ARGV: capacity, refill_per_ms, now_ms, cost
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1]) or tonumber(ARGV[1])
local ts = tonumber(b[2]) or tonumber(ARGV[3])
tokens = math.min(tonumber(ARGV[1]), tokens + (tonumber(ARGV[3]) - ts) * tonumber(ARGV[2]))
local ok = tokens >= tonumber(ARGV[4])
if ok then tokens = tokens - tonumber(ARGV[4]) end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', ARGV[3])
redis.call('PEXPIRE', KEYS[1], 60000)
return ok and 1 or 0
```

**Scaling and trade-offs**

- **Latency vs accuracy:** for very high QPS, keep **local in-memory buckets** and sync to Redis asynchronously, or give each node a slice of the global quota. This trades a small over-admission for no network hop.
- **Failure mode:** if Redis is down, *fail open* (availability) for most APIs, or *fail closed* for expensive or abusive endpoints such as LLM calls or login. Use circuit breakers.
- **Multi-region:** per-region limits with approximate global aggregation; strict global consistency is too costly.
- **Hot keys:** a single tenant flooding one shard needs local pre-limiting.
- **Clock:** use Redis server time, or tolerate skew.
- **Responses:** return `429` with `Retry-After` and `X-RateLimit-Remaining` headers. Clients back off with jitter.
- **LLM-specific:** limit **tokens per minute** as well as requests, using the estimated cost of the request.

---

← [Chapter 24. Evaluation and Testing of AI Systems](../part-4-generative-ai/24-evaluation-and-testing-of-ai-systems.md) · [Table of contents](../README.md) · [Chapter 26. MLOps and LLMOps](26-mlops-and-llmops.md) →
