# Chapter 25. Backend Engineering, APIs and DevOps

_Part V — Production Systems and Architecture_

HTTP and APIs, latency and reliability engineering, scalability, CI/CD, testing and containerization.

This chapter contains **37 questions**.

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
- **25.21** [What makes a well-designed REST API?](#2521-what-makes-a-well-designed-rest-api)
- **25.22** [Compare REST, gRPC and GraphQL. When would you use each?](#2522-compare-rest-grpc-and-graphql-when-would-you-use-each)
- **25.23** [How do you make API requests idempotent and safe to retry?](#2523-how-do-you-make-api-requests-idempotent-and-safe-to-retry)
- **25.24** [How do you implement pagination for large result sets, and why is cursor (keyset) pagination usually better than offset?](#2524-how-do-you-implement-pagination-for-large-result-sets-and-why-is-cursor-keyset-pagination-usually-better-than-offset)
- **25.25** [How do authentication and authorization work for APIs? Explain OAuth 2.0, OpenID Connect, JWTs vs opaque tokens, and API keys.](#2525-how-do-authentication-and-authorization-work-for-apis-explain-oauth-20-openid-connect-jwts-vs-opaque-tokens-and-api-keys)
- **25.26** [What caching strategies do you use in backend systems, and how do you handle cache invalidation and cache stampedes?](#2526-what-caching-strategies-do-you-use-in-backend-systems-and-how-do-you-handle-cache-invalidation-and-cache-stampedes)
- **25.27** [How does load balancing work? Compare L4 and L7 load balancers, balancing algorithms, health checks and sticky sessions.](#2527-how-does-load-balancing-work-compare-l4-and-l7-load-balancers-balancing-algorithms-health-checks-and-sticky-sessions)
- **25.28** [How do you make service-to-service calls resilient?](#2528-how-do-you-make-service-to-service-calls-resilient)
- **25.29** [When should you use synchronous calls, a message queue or pub/sub, and how do you design APIs for long-running jobs?](#2529-when-should-you-use-synchronous-calls-a-message-queue-or-pubsub-and-how-do-you-design-apis-for-long-running-jobs)
- **25.30** [How do you stream responses to clients? Compare Server-Sent Events, WebSockets and long polling, for example for streaming LLM tokens.](#2530-how-do-you-stream-responses-to-clients-compare-server-sent-events-websockets-and-long-polling-for-example-for-streaming-llm-tokens)
- **25.31** [How do you scale a relational database behind a backend service?](#2531-how-do-you-scale-a-relational-database-behind-a-backend-service)
- **25.32** [How do you handle transactions that span multiple microservices? Compare two-phase commit and the saga pattern.](#2532-how-do-you-handle-transactions-that-span-multiple-microservices-compare-two-phase-commit-and-the-saga-pattern)
- **25.33** [How does async work in FastAPI, and how do you avoid blocking the event loop?](#2533-how-does-async-work-in-fastapi-and-how-do-you-avoid-blocking-the-event-loop)
- **25.34** [What are the core Kubernetes concepts for running a backend service?](#2534-what-are-the-core-kubernetes-concepts-for-running-a-backend-service)
- **25.35** [How do you deploy with zero downtime, including database schema migrations?](#2535-how-do-you-deploy-with-zero-downtime-including-database-schema-migrations)
- **25.36** [How do you implement observability for backend services?](#2536-how-do-you-implement-observability-for-backend-services)
- **25.37** [What are the most common API security risks and how do you mitigate them?](#2537-what-are-the-most-common-api-security-risks-and-how-do-you-mitigate-them)

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
| Sliding window counter | Weighted mix of current and previous window; good accuracy, $O(1)$ memory |
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

### 25.21 What makes a well-designed REST API?

A well-designed REST API is predictable: resources are nouns, HTTP methods and status codes mean what the HTTP specification says they mean, errors look the same everywhere, and the contract evolves without breaking existing clients. Consistency matters more than any single convention, because clients learn your API once and generalize.

**Resource naming**

- Plural nouns and hierarchy for ownership: `/datasets`, `/datasets/{id}/versions/{version_id}`. Keep nesting to one or two levels.
- No verbs in paths (`/getDataset`). For true actions that do not map to CRUD, use a sub-resource: `POST /jobs/{id}/cancel` or `POST /models/{id}:deploy`.
- Stable, opaque IDs (UUIDs or prefixed IDs such as `ds_8f2a`), never database auto-increment values that leak volume and invite enumeration.

**Methods and status codes**

| Method | Use | Safe | Idempotent | Typical success |
| --- | --- | --- | --- | --- |
| GET | Read | Yes | Yes | 200, 304 |
| POST | Create or trigger action | No | No | 201 + `Location`, 202 |
| PUT | Full replace | No | Yes | 200, 204 |
| PATCH | Partial update | No | Not by definition | 200 |
| DELETE | Remove | No | Yes | 204 |

Use 400 for malformed input, 401 for missing or invalid credentials, 403 for authenticated but not allowed, 404 for not found (also to hide existence from unauthorized callers), 409 for state conflicts, 412 for failed `If-Match` preconditions, 422 for semantically invalid payloads, 429 with `Retry-After` for throttling, and 503 for temporary unavailability.

**Filtering, sorting, field selection**

- Query parameters on collections: `GET /runs?status=failed&created_after=2026-01-01&sort=-created_at&fields=id,status&limit=50`.
- Allow-list filterable and sortable fields so callers cannot trigger unindexed full scans.
- Always paginate collections, with a server-enforced maximum page size.

**Consistent error format**

Adopt one envelope everywhere, ideally RFC 9457 Problem Details (`application/problem+json`):

```json
{
  "type": "/problems/validation",
  "title": "Invalid request",
  "status": 422,
  "detail": "limit must be between 1 and 500",
  "errors": [{"field": "limit", "code": "out_of_range"}],
  "trace_id": "4bf92f3577b34da6"
}
```

A machine-readable `code` lets clients branch without parsing English; the `trace_id` lets support find the request in your traces.

**Versioning and backward compatibility**

- Additive changes are safe: new optional fields, new endpoints, new enum values (only if clients were told to tolerate unknown values).
- Breaking changes: removing or renaming fields, changing types, tightening validation, changing defaults or semantics.
- Version in the path (`/v1/`) for simplicity and cacheability, or by header or date (Stripe-style pinned versions per account) when you want fine-grained evolution.
- Deprecate deliberately: announce, emit `Deprecation` and `Sunset` headers, track per-client usage of old versions, and only remove when traffic is near zero.

**Senior signals**: design the contract first in OpenAPI, generate clients and contract tests from it, run a breaking-change linter in CI, use `ETag` with `If-Match` for optimistic concurrency, and support idempotency keys on POST. In data platforms the most common failure is not REST purity but an undocumented change in a field's meaning that silently corrupts downstream pipelines.

### 25.22 Compare REST, gRPC and GraphQL. When would you use each?

REST is the default for public and resource-oriented APIs, gRPC is the best fit for low-latency internal service-to-service calls and streaming, and GraphQL suits client-driven aggregation where many front ends need different shapes of the same graph of data. Many mature platforms use all three: gRPC inside, REST for partners, GraphQL as a backend-for-frontend.

| Aspect | REST | gRPC | GraphQL |
| --- | --- | --- | --- |
| Transport | HTTP/1.1 or HTTP/2 | HTTP/2 (HTTP/3 emerging) | Usually HTTP POST to one endpoint |
| Payload | JSON (text) | Protocol Buffers (binary) | JSON |
| Contract | OpenAPI (optional) | `.proto` file (mandatory, strongly typed) | Schema (mandatory, typed) |
| Streaming | Via SSE or WebSockets | Native unary, server, client and bidirectional streaming | Subscriptions (usually over WebSockets) |
| Caching | Excellent HTTP and CDN caching on GET | No HTTP caching | Hard; needs persisted queries or client caches |
| Browser support | Native | Needs gRPC-Web or a proxy | Native |
| Over/under-fetching | Common | Fixed per RPC | Client selects exact fields |
| Tooling and debuggability | curl, everywhere | Needs grpcurl, reflection | GraphiQL, strong introspection |

**When to use REST**

- Public or partner APIs, webhooks, simple CRUD, anything where CDN caching, ubiquity and easy debugging matter.

**When to use gRPC**

- Internal microservices with high call volume, where binary encoding and HTTP/2 multiplexing cut latency and CPU.
- Polyglot teams that benefit from generated clients and compile-time contract checking.
- Streaming: feature-store lookups, embedding services, model inference servers (Triton and many serving stacks expose gRPC), bidirectional telemetry.
- Built-in deadlines and cancellation that propagate across calls, which is excellent for latency budgets.

Watch out for: L4 load balancers pin long-lived HTTP/2 connections to one backend, so you need L7 or client-side load balancing (or a service mesh); and `.proto` evolution rules (never reuse field numbers, add fields as optional).

**When to use GraphQL**

- Product front ends (web, mobile) aggregating data from many services, where each screen needs a different shape and round trips are expensive.
- Rapidly evolving UIs that should not require a new backend endpoint per screen.

Watch out for: N+1 resolver queries (solve with DataLoader-style batching), unbounded query cost (enforce depth and complexity limits, persisted queries), field-level authorization in every resolver, and weaker HTTP caching. GraphQL moves complexity from the client to the server.

**How I decide**

1. Who is the consumer? External developers point to REST; internal services to gRPC; your own UIs with diverse needs to GraphQL.
2. What is the latency and volume profile? Tens of thousands of internal calls per second favor gRPC.
3. Do you need streaming? gRPC natively; for browsers, SSE over REST.
4. What can the team operate? GraphQL in particular needs investment in query cost control and observability.

For an AI platform, a typical answer: model servers and feature lookups over gRPC, a REST gateway (OpenAI-compatible endpoints with SSE streaming) for consumers, and GraphQL only if a product UI genuinely needs flexible aggregation.

### 25.23 How do you make API requests idempotent and safe to retry?

An operation is idempotent if performing it once or many times leaves the server in the same state. Networks fail after the server has done the work but before the client sees the response, so any operation that clients, proxies or SDKs might retry must either be naturally idempotent or be made idempotent with a client-supplied key whose result the server stores and replays.

**Method semantics (RFC 9110)**

- GET, HEAD, OPTIONS are safe (no state change) and idempotent.
- PUT and DELETE are idempotent: `PUT /users/42 {...}` twice yields the same state; a second DELETE may return 404, but state is unchanged.
- POST and PATCH are not idempotent by definition. These are the ones that create duplicate orders, double charges or duplicate training jobs.

**Making non-idempotent operations safe: idempotency keys**

The client generates a high-entropy key (UUID v4) per logical operation and sends it on every retry, for example `Idempotency-Key: 9c1d...` (an IETF draft header that Stripe and others popularized).

Server algorithm:

1. Look up `(tenant_id, key)` in a durable store.
2. **Not found**: atomically insert a record with status `in_progress`, a hash of the request body, and a lock. Use a unique constraint so two concurrent requests cannot both win.
3. **Found, completed**: if the request hash matches, replay the stored status code and body without re-executing. If it differs, reject (422 is common) because the client reused a key for a different operation.
4. **Found, in progress**: return 409 so the client retries later, rather than executing twice.
5. On completion, store the response and mark `completed`, ideally in the same database transaction as the business write.
6. Expire keys after a retention window (Stripe uses about 24 hours).

```sql
CREATE TABLE idempotency_keys (
  tenant_id     text,
  key           text,
  request_hash  text NOT NULL,
  status        text NOT NULL,          -- in_progress | completed
  response_code int,
  response_body jsonb,
  created_at    timestamptz DEFAULT now(),
  PRIMARY KEY (tenant_id, key)
);
```

**Design details that matter**

- **Same transaction**: if the side effect and the key record commit separately, a crash between them breaks the guarantee. When the side effect is an external call (payment provider, LLM, email), split the work into recoverable steps with checkpoints, or pass your idempotency key downstream so the provider deduplicates too.
- **Which errors to cache**: cache final outcomes (success and deterministic 4xx). For 5xx caused before any side effect, allow re-execution.
- **Natural keys**: often you can avoid a separate store with a unique business key, e.g. `UNIQUE(source_file_checksum)` for ingestion or `INSERT ... ON CONFLICT DO NOTHING`.
- **Conditional requests**: `If-Match` with an ETag makes updates safe against lost updates, and `PUT` to a client-chosen ID (`PUT /jobs/{client_uuid}`) gives create-once semantics without a key table.

**Consumers and pipelines**

The same idea applies to message consumers with at-least-once delivery: deduplicate on a message ID, use upserts or MERGE on a primary key, and make each pipeline step overwrite a deterministic partition rather than append. Combined with retries (backoff and jitter), idempotency is what turns at-least-once delivery into effectively-once outcomes.

### 25.24 How do you implement pagination for large result sets, and why is cursor (keyset) pagination usually better than offset?

Offset pagination (`LIMIT 50 OFFSET 100000`) is simple but makes the database read and discard every skipped row and returns duplicates or gaps when data changes between pages. Cursor (keyset) pagination remembers the last row's sort key and asks for rows after it, so every page costs the same index seek and results stay stable under concurrent inserts.

**Offset pagination**

```sql
SELECT id, created_at, status FROM runs
ORDER BY created_at DESC, id DESC
LIMIT 50 OFFSET 100000;
```

- Cost grows linearly with the offset: page 2,000 scans about 100,000 rows to return 50.
- If a row is inserted on page 1 while a client reads page 2, every row shifts and the client sees a duplicate; deletions cause skipped rows.
- Advantages: random access ("jump to page 37"), easy total counts. Fine for small, admin-style tables.

**Keyset (cursor) pagination**

```sql
SELECT id, created_at, status FROM runs
WHERE (created_at, id) < (:last_created_at, :last_id)
ORDER BY created_at DESC, id DESC
LIMIT 51;            -- fetch one extra to know if there is a next page
```

- Requires a **deterministic, unique sort order**: add a tiebreaker (the primary key) because timestamps collide.
- Requires a matching composite index, here `(created_at DESC, id DESC)`, so each page is an index range seek.
- Row-value comparison `(a, b) < (x, y)` is supported in PostgreSQL and MySQL; otherwise expand it to `a < x OR (a = x AND b < y)`.

**API shape**

```json
{
  "data": [ ... ],
  "next_cursor": "eyJjIjoiMjAyNi0xMS0xNlQxMDowMDowMFoiLCJpIjoicnVuXzk4MSJ9",
  "has_more": true
}
```

- Make the cursor **opaque** (base64 of the sort values, optionally signed or encrypted) so clients cannot craft it and you can change the implementation later.
- Bind filters and sort order into the cursor, or reject a cursor used with different filters.
- Enforce a maximum `limit`.

**Comparison**

| Aspect | Offset | Keyset / cursor |
| --- | --- | --- |
| Deep page cost | Grows with offset | Constant |
| Consistency under writes | Duplicates and gaps | Stable |
| Jump to arbitrary page | Yes | No (next/previous only) |
| Total count | Easy but expensive on big tables | Usually omitted or approximate |
| Sort flexibility | Any column | Needs index per sort order |

**Trade-offs and production notes**

- Exact `COUNT(*)` on large filtered sets is often more expensive than the page itself; return `has_more` or an estimate.
- Arbitrary user-chosen sort columns each need an index for keyset to be efficient; allow-list sort options.
- Search engines have the same issue: Elasticsearch and OpenSearch cap deep `from + size` and offer `search_after` with a point-in-time for keyset-style paging.
- For bulk export (a data team asking for 50 million rows), do not paginate a request-response API at all: run an async export job that writes Parquet to object storage and returns a link. Pagination is for interactive consumption, not for moving datasets.

### 25.25 How do authentication and authorization work for APIs? Explain OAuth 2.0, OpenID Connect, JWTs vs opaque tokens, and API keys.

Authentication proves who the caller is; authorization decides what they may do. OAuth 2.0 is a delegated authorization framework that issues access tokens; OpenID Connect adds an identity layer on top so clients can also learn who the user is; JWTs and opaque tokens are two formats for those tokens; and API keys are simple shared secrets that identify a calling application.

**OAuth 2.0 roles and flows**

- Roles: resource owner (user), client (app), authorization server (issues tokens), resource server (your API).
- **Authorization Code with PKCE**: the flow for anything with a user, including single-page and mobile apps. The client redirects the user to log in, receives a short-lived code, and exchanges it (plus the PKCE verifier) for tokens.
- **Client Credentials**: machine-to-machine, no user. A batch job or microservice authenticates as itself.
- **Refresh tokens**: long-lived, used to get new short-lived access tokens. For public clients, rotate them on each use or bind them to the sender.
- The OAuth 2.0 Security BCP (RFC 9700, 2025) says: do not use the implicit grant or resource owner password grant, use PKCE (with S256) for all clients, use exact redirect URI matching, keep access tokens short-lived and narrowly scoped (audience-restricted), and prefer sender-constrained tokens (mTLS or DPoP) where possible.

**OpenID Connect**

OIDC adds the `openid` scope, an **ID token** (always a JWT) describing the authenticated user (`sub`, `iss`, `aud`, `exp`, `nonce`), a UserInfo endpoint and discovery metadata. Rule of thumb: the ID token is for the client to know who logged in; the access token is for calling APIs. Never send an ID token to an API as authorization.

**JWTs vs opaque tokens**

| Aspect | JWT (self-contained) | Opaque token |
| --- | --- | --- |
| Validation | Locally, verify signature with cached JWKS public keys | Introspection call to auth server, or cache lookup |
| Latency | No network hop | Extra hop (mitigate with caching) |
| Revocation | Hard until `exp`; needs short TTLs or a deny list | Immediate |
| Privacy | Claims are readable by anyone (signed, not encrypted) | Reveals nothing |
| Size | Larger headers | Small |

When validating a JWT, check signature with an allow-listed algorithm (never accept `alg: none` or let the token choose the key), `iss`, `aud`, `exp`, `nbf`, and required scopes. A common pattern is opaque tokens at the edge with the gateway exchanging them for short-lived internal JWTs.

**API keys**

- Identify an application or tenant, not a user. Good for server-to-server integrations, metering and rate limiting (for example, LLM gateway keys per team).
- Treat them as passwords: generate with high entropy, show once, store only a hash, prefix them (`sk_live_...`) so secret scanners detect leaks, support rotation with two active keys, scope them, and set expiry.
- They are bearer secrets with no built-in expiry or delegation, so for anything user-facing use OAuth.

**Authorization models**

- Scopes in tokens give coarse permissions (`datasets:read`).
- Fine-grained decisions use RBAC (roles), ABAC (attributes such as region or data classification), or relationship-based models (Zanzibar-style: "can user X read document Y").
- Enforce object-level checks on every request in the service, not only at the gateway; most real breaches are a valid token accessing someone else's object.

**Service-to-service**: mTLS (often via a mesh such as Istio or Linkerd with SPIFFE identities) for workload identity, plus client-credentials or workload-identity-federated tokens instead of static secrets.

### 25.26 What caching strategies do you use in backend systems, and how do you handle cache invalidation and cache stampedes?

Caching trades freshness for latency and load reduction. The core decisions are where to cache (client, CDN, application memory, distributed cache), which read and write pattern to use (cache-aside, read-through, write-through, write-behind), how stale data is allowed to get (TTL and invalidation), and how to protect the origin when a hot key expires (stampede control).

**Cache layers**

- **HTTP and CDN**: `Cache-Control`, `ETag` and `304 Not Modified` for public or per-user GET responses.
- **In-process (local)**: fastest, but each instance has its own copy, so invalidation is per node; best for small, rarely changing data such as config or feature flags.
- **Distributed (Redis, Memcached)**: shared across instances; adds a network hop of roughly a millisecond.
- **Database-level**: materialized views, buffer cache, query result caches in warehouses.
- **AI-specific**: embedding caches keyed by content hash, exact-match LLM response caches, semantic caches (vector similarity on prompts, risky for correctness), and provider-side prompt caching.

**Patterns**

| Pattern | How | Trade-off |
| --- | --- | --- |
| Cache-aside | App reads cache, on miss reads DB and fills cache | Most common, simple; first read is slow; app owns logic |
| Read-through | Cache library loads from DB on miss | Cleaner code; tied to cache provider |
| Write-through | Write to cache and DB synchronously | Fresh cache; slower writes, caches unread data |
| Write-behind | Write to cache, flush to DB async | Fast writes; risk of data loss |
| Refresh-ahead | Refresh popular keys before expiry | Avoids miss latency; wasted work for cold keys |

**Invalidation**

- **TTL** on everything as a safety net, with jitter (for example 300 s plus or minus 10%) so keys created together do not expire together.
- **Delete on write** (not update): after committing the DB write, delete the cache key. Updating the cache directly races with concurrent writers and can leave stale values indefinitely.
- **Event-driven**: consume CDC events (Debezium from the database log) to invalidate keys, which catches writes from any service, including backfills and admin scripts.
- **Versioned keys**: embed a version or content hash (`user:42:v17`, `model:fraud:2026-11-01`) so new data uses new keys and old ones simply age out.
- Even delete-on-write has a race: a reader misses, reads the old value, a writer commits and deletes, and the reader then writes the stale value. Short TTLs, versioned keys or a delayed second delete reduce the window.

**Cache stampede (thundering herd / dogpile)**

When a hot key expires, thousands of concurrent requests miss and hit the database at once.

- **Request coalescing (single flight)**: only one request per key recomputes; others wait for its result. Per process with an in-memory future, across processes with a short Redis lock (`SET key:lock NX PX 5000`).
- **Stale-while-revalidate**: keep serving the stale value while one worker refreshes in the background.
- **Probabilistic early expiration**: each reader recomputes early with a probability that rises as expiry nears, so refresh is spread out.
- **TTL jitter** and pre-warming after deploys or cache flushes.

```python
async def get_with_single_flight(key, loader, ttl=300):
    val = await redis.get(key)
    if val is not None:
        return val
    if await redis.set(f"{key}:lock", "1", nx=True, px=5000):
        try:
            val = await loader()
            await redis.set(key, val, ex=ttl + random.randint(0, ttl // 10))
            return val
        finally:
            await redis.delete(f"{key}:lock")
    await asyncio.sleep(0.05)
    return await get_with_single_flight(key, loader, ttl)
```

**Other failure modes**: cache penetration (requests for keys that do not exist, so cache negative results briefly or use a Bloom filter), hot keys (replicate to local caches), memory eviction policy surprises, and a cold cache after a Redis failover becoming a database outage. Always ask what happens to the origin if the cache disappears.

### 25.27 How does load balancing work? Compare L4 and L7 load balancers, balancing algorithms, health checks and sticky sessions.

A load balancer spreads incoming traffic across a pool of healthy backends so no single instance is overloaded and failed instances stop receiving requests. Layer 4 balancers route TCP or UDP connections using IP and port only; Layer 7 balancers terminate HTTP and route individual requests using paths, headers and cookies.

**L4 vs L7**

| Aspect | L4 (transport) | L7 (application) |
| --- | --- | --- |
| Sees | IP, port, TCP/UDP | Full HTTP: method, path, headers, body |
| Unit balanced | Connection | Request |
| Performance | Very high throughput, low latency | More CPU (TLS termination, parsing) |
| Features | Simple forwarding, preserves client TCP | Path and host routing, header rewrites, retries, auth, WAF, compression, canary splits |
| TLS | Usually passes through | Terminates (can re-encrypt) |
| Examples | AWS NLB, Google TCP/UDP LB, IPVS | AWS ALB, Envoy, NGINX, HAProxy (HTTP mode), Kubernetes Ingress or Gateway |

A key subtlety: with HTTP/2 and gRPC, many requests share one long-lived connection, so an L4 balancer pins all of a client's traffic to one backend. You need L7 (Envoy, a mesh) or client-side load balancing for even distribution.

**Algorithms**

- **Round robin / weighted round robin**: simple; good when requests are uniform.
- **Least connections / least outstanding requests**: better when request durations vary widely, which is the norm for LLM inference where a request can take 200 ms or 60 s.
- **Power of two choices**: pick two backends at random, send to the less loaded; near-optimal with little coordination (used by Envoy and many proxies).
- **Consistent hashing** (ring, Maglev): same key goes to the same backend, minimal remapping when nodes change; useful for cache locality, sharded state, or routing a conversation to the replica holding its KV cache.
- **Latency- or load-aware (EWMA)**: route by observed response time; for model serving, routing by queue depth or GPU KV-cache usage beats naive round robin.

**Health checks**

- **Active**: the balancer probes `/healthz` periodically; a backend is removed after N consecutive failures and re-added after M successes.
- **Passive (outlier detection)**: eject backends that return errors or time out on real traffic.
- Keep health checks cheap and honest: check that the process can serve, not every dependency. A deep check that fails when the shared database is slow will eject every instance at once and turn a partial degradation into a total outage.
- **Connection draining**: when removing an instance, stop new requests but let in-flight ones finish.

**Sticky sessions (session affinity)**

- Implemented via cookie or client-IP hashing so a user keeps hitting the same backend.
- Cons: uneven load, lost sessions when an instance dies, harder scaling and deploys.
- Prefer stateless services with session state in Redis or in signed tokens. Use affinity only for genuine locality benefits (WebSocket connections, in-memory caches, KV-cache reuse), and design for the instance disappearing anyway.

**Layered in practice**: DNS or anycast global balancing across regions, an L4 balancer at the edge, L7 proxies or ingress for routing, and client-side or mesh balancing between services. Each layer needs its own health checks and timeouts aligned with the layers above it.

### 25.28 How do you make service-to-service calls resilient?

Assume every remote call can be slow, fail, or fail partially. Resilience means bounding how long you wait (timeouts and deadlines), retrying only when it helps and only within a budget, stopping calls to a dependency that is clearly down (circuit breakers), isolating resources so one bad dependency cannot exhaust everything (bulkheads), and degrading gracefully (fallbacks and load shedding).

**Timeouts and deadline propagation**

- Every network call needs a connect timeout and a request timeout. Library defaults are often infinite.
- Set timeouts from the dependency's observed latency (slightly above its p99.9) and from your own SLO, not arbitrary round numbers.
- **Propagate deadlines**: if the user request has 2 s left, downstream calls get the remaining budget minus overhead. gRPC does this natively; with HTTP pass a header. This prevents servers from doing work whose result nobody is waiting for.
- Timeouts at each layer must shrink as you go deeper; an outer timeout shorter than an inner one causes wasted work and retries of work that is still running.

**Retries without amplification**

- Retry only transient errors (connection errors, 503, 429 honoring `Retry-After`) and only idempotent operations or ones with idempotency keys.
- Exponential backoff with jitter (see 25.10), and a small per-request cap (for example 3 attempts).
- **Retry budget**: limit retries to a fraction of total traffic per client (the Google SRE book's example is about 10%). Without it, a stack five services deep where each layer retries 3 times can multiply load up to $3^5 = 243$ times on the failing service.
- Retry at one layer only, ideally closest to the client or in the mesh, not at every hop.
- Hedged requests (send a second request if the first is slower than p95) cut tail latency for idempotent reads, but cost extra load.

**Circuit breaker**

States: **closed** (normal, counting failures), **open** (fail fast for a cool-down period after the error rate crosses a threshold over a minimum request volume), **half-open** (allow a few trial calls; close on success, reopen on failure). It saves your threads and the struggling dependency, and turns slow failure into fast failure that your fallback can handle.

**Bulkheads**

- Separate connection pools, thread pools or concurrency semaphores per dependency, so a slow recommendation service cannot consume all workers needed for checkout.
- In Kubernetes: separate deployments for critical and best-effort traffic, per-tenant quotas, and priority classes.

**Fallbacks and load shedding**

- Serve cached or stale data, a default (popular items instead of personalized ones), a smaller or cheaper model, or a partial response with a flag. Fallbacks must be simpler and more reliable than the primary path, and tested regularly.
- Shed load early when overloaded: reject with 503 or 429 at the edge based on concurrency or queue depth, prioritizing critical requests. Rejecting quickly is better than accepting everything and timing out everything.

**Example: an LLM-backed endpoint**

```text
client deadline 10 s
  -> gateway: per-tenant rate limit + concurrency bulkhead
  -> LLM call: timeout 8 s, 1 retry on 429/503 with jitter, retry budget 10%
       circuit breaker per provider; when open -> fallback provider or smaller model
  -> if all fail: cached answer or graceful "try again" message, never a hung request
```

Libraries: Resilience4j (JVM), Polly (.NET), tenacity and purgatory-style breakers (Python), or a service mesh (Envoy and Istio) for timeouts, retries, outlier detection and circuit breaking without code changes. Validate it all with fault injection and chaos experiments; untested resilience logic usually fails the first time it runs.

### 25.29 When should you use synchronous calls, a message queue or pub/sub, and how do you design APIs for long-running jobs?

Use synchronous request-response when the caller needs the result immediately to continue and the work finishes within a request timeout. Use a queue when work can happen later, needs buffering, retries or rate control, and exactly one worker should process each item. Use pub/sub when one event should fan out to many independent consumers. Long-running work should never hold an HTTP connection open; accept it, return a job resource, and let the client poll or receive a callback.

**Choosing the interaction style**

| Style | Use when | Trade-offs |
| --- | --- | --- |
| Synchronous (REST, gRPC) | User waits for the answer, short work (under a few seconds), strong consistency needed | Temporal coupling: if the callee is down, so are you; latency adds up along the chain |
| Point-to-point queue (SQS, RabbitMQ, Celery, Kafka consumer group) | Background tasks, smoothing bursts, rate-limited downstreams (LLM APIs), retries with dead-letter queues | Eventual completion, need idempotent workers, monitoring of queue depth and age |
| Pub/sub and event streams (Kafka, SNS, Pub/Sub, EventBridge) | "Order placed" consumed by billing, analytics, search indexing; decoupled teams | Eventual consistency, schema evolution, harder end-to-end debugging |

Signals for async: work exceeds a few seconds, load is bursty, the downstream has strict rate limits, or failure should not be visible to the user. A useful rule: commands go to queues ("resize this image"), facts go to topics ("image uploaded").

**Reliable publishing**: writing to the database and publishing to a broker are two systems; a crash between them loses or duplicates events. Use the **transactional outbox** (write the event to an outbox table in the same transaction, relay it via a poller or CDC) and make consumers idempotent.

**Long-running job API: 202 + polling**

```http
POST /v1/batch-inference
Idempotency-Key: 7f3e...
-> 202 Accepted
   Location: /v1/jobs/job_123
   {"id": "job_123", "status": "queued"}

GET /v1/jobs/job_123
-> 200 {"id": "job_123", "status": "running", "progress": 0.42}
-> 200 {"id": "job_123", "status": "succeeded", "result_url": "s3://.../output.parquet"}
```

- Model the job as a resource with a state machine: `queued -> running -> succeeded | failed | cancelled`, plus `created_at`, `progress`, `error`.
- Return `Retry-After` to guide polling intervals; support `POST /jobs/{id}/cancel`.
- Put large results in object storage with a pre-signed URL rather than in the response.
- Require an idempotency key so retrying the submit does not launch two expensive jobs.

**Webhooks (push)**

- Client registers a callback URL; you POST an event on completion.
- Sign payloads (HMAC with a per-subscriber secret plus timestamp) so receivers can verify authenticity and reject replays.
- Delivery is at-least-once: retry with backoff for hours or days, include an event ID for deduplication, and offer a way to list missed events.
- Receivers should acknowledge fast (2xx) and process asynchronously.
- Guard against SSRF: validate callback URLs, block internal address ranges.

Offer both: webhooks for efficiency, polling as the fallback source of truth. OpenAI's and Anthropic's batch APIs follow exactly this model: submit, get an ID, poll status, download results.

### 25.30 How do you stream responses to clients? Compare Server-Sent Events, WebSockets and long polling, for example for streaming LLM tokens.

For streaming LLM tokens to a browser, Server-Sent Events are usually the right default: a one-way server-to-client stream over ordinary HTTP that works through proxies and load balancers and is what the major LLM APIs use. Choose WebSockets when you need true bidirectional, low-latency messaging (voice agents, collaborative editing, interrupting mid-generation over the same channel), and long polling only as a fallback for very restrictive environments.

| Aspect | Long polling | Server-Sent Events | WebSockets |
| --- | --- | --- | --- |
| Direction | Server to client (simulated) | Server to client | Bidirectional |
| Protocol | Repeated HTTP requests | HTTP response with `text/event-stream` | Upgrade to `ws://`/`wss://` framing |
| Data | Any | UTF-8 text events | Text and binary frames |
| Reconnect | Client re-requests | Built-in, with `Last-Event-ID` | Implement yourself |
| Infrastructure fit | Excellent | Good (plain HTTP; disable proxy buffering) | Needs upgrade support, sticky long-lived connections |
| Overhead per message | High (full request each time) | Low | Lowest |
| Typical use | Legacy notifications | LLM tokens, progress, live feeds | Chat with presence, realtime voice, games, collaboration |

**SSE for LLM token streaming**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/chat")
async def chat(req: ChatRequest):
    async def events():
        async for chunk in llm.stream(req.messages):
            yield f"data: {json.dumps({'delta': chunk.text})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
```

The browser's native `EventSource` only supports GET without custom headers, so LLM front ends typically POST with `fetch` and read the stream with a reader, parsing SSE frames themselves.

**Production concerns**

- **Buffering**: proxies, ingress controllers and compression middleware may buffer the response and destroy the streaming effect. Disable buffering (NGINX `proxy_buffering off`), avoid gzip on event streams.
- **Timeouts**: load balancer idle timeouts (often 60 s) kill silent streams; send heartbeat comments (`: ping`) every 15 to 30 seconds and raise idle timeouts for streaming routes.
- **Cancellation**: when the client disconnects, stop the upstream generation to avoid paying for tokens nobody reads. Check for disconnects in the generator and cancel the upstream task.
- **Errors mid-stream**: the 200 status is already sent, so errors must be in-band events (`event: error`). Clients must handle partial output.
- **Concurrency**: each stream holds a connection for its full duration, so use async servers; capacity planning is about concurrent connections, not requests per second.
- **Observability**: measure time to first token, inter-token latency and total duration separately; a single request latency metric hides the user experience.
- **HTTP/1.1 limits**: browsers allow about six connections per origin over HTTP/1.1, so many parallel SSE streams need HTTP/2.

**WebSockets specifics**: need sticky routing or a pub/sub backplane (Redis) to reach the right connection, explicit ping/pong, authentication at handshake time (cookies or a short-lived token, since browsers cannot set headers on the upgrade), backpressure handling, and careful draining on deploys because connections are long-lived.

### 25.31 How do you scale a relational database behind a backend service?

Scale in order of increasing complexity: fix queries and indexes first, then pool connections, add caching and read replicas for read-heavy load, partition large tables for manageability, scale vertically, and only shard horizontally when writes or data size exceed what one primary can handle. Each step adds operational cost and consistency trade-offs.

**1. Query and schema hygiene**

Most database scaling problems are a few bad queries: missing indexes, N+1 ORM patterns, `SELECT *` on wide tables, long transactions holding locks. Use `EXPLAIN ANALYZE`, `pg_stat_statements` and slow-query logs before buying hardware.

**2. Connection pooling**

- PostgreSQL uses a process per connection, so thousands of connections waste memory and degrade throughput. Autoscaled pods times per-pod pool size can easily exceed `max_connections`.
- Pool in the app (SQLAlchemy, HikariCP) and with an external pooler (PgBouncer, RDS Proxy) in **transaction mode**, which multiplexes many clients onto few server connections.
- Transaction mode breaks session state: server-side prepared statements (unless the pooler supports them), `SET` session variables, advisory locks and `LISTEN`.
- Size pools from database capacity: a good starting point is roughly a few connections per CPU core on the database, not hundreds.

**3. Read replicas and replication lag**

- Asynchronous streaming replication sends WAL to replicas; route reads there to offload the primary.
- **Lag** means a replica can be milliseconds to minutes behind. Classic bug: a user updates their profile, the next read goes to a replica and shows the old value.
- Mitigations: read-your-writes routing (send a user's reads to the primary for a few seconds after they write, or until the replica has replayed past the write's log position), route consistency-critical reads to the primary, monitor lag and remove lagging replicas from rotation.
- Synchronous replication eliminates lag for durability but adds write latency; usually used for one standby for failover, not for read scaling.

**4. Partitioning (within one database)**

- Declarative partitioning by range (time), list (region) or hash. For event or log data, monthly or daily range partitions make retention a cheap `DROP PARTITION` instead of a massive `DELETE`, and partition pruning speeds time-bounded queries.
- Queries must include the partition key to benefit; unique constraints must include it.

**5. Sharding (across databases)**

- Split data across independent primaries by a shard key: `tenant_id` is ideal for B2B SaaS because most queries are single-tenant.
- Choose a key with high cardinality, even distribution, and that matches the dominant access pattern. Avoid monotonically increasing keys with range sharding (all writes hit the last shard).
- Use many logical shards mapped to fewer physical nodes (or consistent hashing) so rebalancing moves logical shards rather than re-hashing everything.
- Costs: cross-shard joins, transactions and unique constraints become application problems; resharding is a major project; hot tenants may need dedicated shards.
- Before hand-rolling, consider Citus, Vitess, or distributed SQL (CockroachDB, YugabyteDB, Spanner, Aurora Limitless) which handle sharding with SQL semantics, at their own cost and latency trade-offs.

**6. Offload what does not belong**

Move analytics to a warehouse via CDC rather than running heavy reports on the OLTP primary; move full-text and vector search to dedicated engines or at least separate replicas; put blobs in object storage. A data engineer's most common contribution here is getting analytical load off the production database.

### 25.32 How do you handle transactions that span multiple microservices? Compare two-phase commit and the saga pattern.

You usually avoid distributed ACID transactions across microservices and use sagas instead: a sequence of local transactions, each publishing an event or command that triggers the next, with compensating actions to semantically undo earlier steps on failure. Two-phase commit gives atomicity but blocks on coordinator failure and couples availability of all participants, so it is mostly reserved for databases and brokers that support it natively within one trust boundary.

**Two-phase commit (2PC)**

1. **Prepare**: the coordinator asks every participant to prepare; each durably locks its resources and votes yes or no.
2. **Commit or abort**: if all vote yes, the coordinator logs the decision and tells all to commit; otherwise abort.

Problems:
- **Blocking**: if the coordinator crashes after participants voted yes, they hold locks in an uncertain state until it recovers.
- **Availability**: the transaction succeeds only if every participant is up, so availability is the product of all participants' availability.
- **Latency and throughput**: multiple round trips while holding locks.
- **Support**: needs XA or similar in every participant; most HTTP services, SaaS APIs and many NoSQL stores cannot participate.

Where it works: within a single distributed database (Spanner, CockroachDB use consensus-replicated commit protocols internally), or Kafka's transactions for exactly-once read-process-write within Kafka.

**Saga pattern**

Example, ordering: `CreateOrder (pending) -> ReservePayment -> ReserveInventory -> ConfirmOrder`. If inventory fails: `RefundPayment -> CancelOrder`.

| Style | How | Pros | Cons |
| --- | --- | --- | --- |
| Choreography | Each service reacts to events and emits new ones | Loose coupling, no central component | Flow is implicit and hard to trace; cyclic dependencies |
| Orchestration | A central orchestrator sends commands and tracks state | Explicit, testable flow, easy monitoring and timeouts | Orchestrator is extra infrastructure; risk of centralizing logic |

Orchestration engines such as Temporal, AWS Step Functions or Camunda persist saga state and handle retries and timers; for more than three or four steps, orchestration is usually easier to reason about.

**What sagas give up and how to cope**

- **No isolation**: other transactions can see intermediate states (an order that is "pending"). Use semantic locks (status fields), design UIs around pending states, and order steps so the most likely-to-fail or hard-to-compensate steps come early.
- **Compensations are business logic**, not rollbacks: you cannot unsend an email, so send it last; a refund is not the same as a never-made charge.
- **Pivot transaction**: the step after which the saga must complete forward (retry until success) rather than compensate.
- **Every step and compensation must be idempotent and retryable**, because messages are delivered at least once.
- **Reliable messaging**: use the transactional outbox so the local commit and the event emission cannot diverge.

**Comparison**

| Aspect | 2PC | Saga |
| --- | --- | --- |
| Consistency | Atomic, isolated | Eventual, no isolation |
| Availability | Lowest of participants | High, steps proceed independently |
| Failure handling | Automatic rollback | Explicit compensating actions |
| Lock duration | Across the whole transaction | Only within each local transaction |
| Complexity | In the protocol and infrastructure | In the application design |

Often the best answer is to revisit service boundaries: if two services constantly need atomic updates together, they may belong in one service with one database.

### 25.33 How does async work in FastAPI, and how do you avoid blocking the event loop?

FastAPI runs on an ASGI server (Uvicorn) with a single-threaded asyncio event loop per process. `async def` endpoints run directly on that loop and are extremely efficient for I/O as long as everything inside them awaits non-blocking calls; plain `def` endpoints are run in a thread pool so blocking code does not freeze the loop. Scale across CPU cores with multiple worker processes, and move CPU-heavy work out of the request path entirely.

**The rules**

| Endpoint | Runs on | Use when |
| --- | --- | --- |
| `async def` | The event loop | All I/O uses async libraries (httpx, asyncpg, async SQLAlchemy, aioboto3, async LLM SDKs) |
| `def` | Thread pool (AnyIO, default limit of 40 threads per process) | Blocking libraries (requests, psycopg2, boto3, most SDKs) |

The most damaging mistake is a blocking call inside `async def`:

```python
@app.get("/bad")
async def bad():
    r = requests.get(URL)          # blocks the loop: every other request on this worker stalls
    time.sleep(1)                   # same
    return r.json()

@app.get("/good")
async def good():
    async with httpx.AsyncClient() as client:
        r = await client.get(URL)
    await asyncio.sleep(1)
    return r.json()
```

If you must call blocking code from an async endpoint, offload it explicitly: `await run_in_threadpool(fn, ...)` (Starlette) or `await asyncio.to_thread(fn, ...)`. The same applies to dependencies: a sync dependency also uses the thread pool.

**Thread pool limits**

- Sync endpoints and sync dependencies share the default 40-token limiter per process; under load, requests queue for a thread even if the CPU is idle. You can raise it at startup (adjusting AnyIO's default thread limiter), but threads are still constrained by the GIL for CPU work.
- Share clients: create one `httpx.AsyncClient` or DB pool in the lifespan handler, not per request, to reuse connections.

**Processes and workers**

- One Uvicorn process uses one core for Python execution. Use `uvicorn --workers N` (or Gunicorn with Uvicorn workers) on a VM.
- In Kubernetes, prefer one process per container and scale with replicas; the orchestrator handles restarts and balancing.
- Each worker has its own memory, so large ML models are loaded once per process; that is a common reason for out-of-memory errors when increasing workers.

**CPU-bound work**

CPU-heavy Python (feature engineering with pure Python loops, tokenization of huge documents, PDF parsing, model inference on CPU) blocks the loop if done in `async def` and saturates the GIL in threads. Options:

- `ProcessPoolExecutor` via `loop.run_in_executor` for moderate work.
- Libraries that release the GIL (NumPy, PyTorch, many C extensions) can run in threads effectively.
- For heavy or long tasks, a task queue (Celery, RQ, Arq, Dramatiq) or a dedicated inference server (Triton, vLLM, Ray Serve) and return 202 with a job ID.
- `BackgroundTasks` only for small fire-and-forget work; it runs in the same process and is lost if the pod dies.

**Detecting problems**

Enable asyncio debug mode (warns on slow callbacks), watch event loop lag, and look for a pattern where p99 latency rises for all endpoints at once when one slow endpoint is hit. Load test with realistic concurrency, since blocking bugs are invisible at a handful of requests.

### 25.34 What are the core Kubernetes concepts for running a backend service?

Kubernetes runs your containers as Pods, keeps the desired number of replicas running via a Deployment, gives them a stable virtual IP and DNS name via a Service, exposes HTTP routes externally via Ingress or the Gateway API, uses probes to decide when pods are ready or need restarting, schedules them by resource requests, and scales them with the Horizontal Pod Autoscaler.

**Building blocks**

- **Pod**: smallest unit; one or more containers sharing network and storage. Ephemeral and replaceable, so never store state on the pod's filesystem.
- **Deployment**: manages ReplicaSets for stateless apps; handles rolling updates and rollbacks. Use **StatefulSet** for stable identity and storage (databases, Kafka), **Job/CronJob** for batch, **DaemonSet** for per-node agents.
- **Service**: stable ClusterIP and DNS (`api.prod.svc.cluster.local`) load-balancing across ready pods selected by labels. Types: ClusterIP (internal), NodePort, LoadBalancer (cloud L4).
- **Ingress / Gateway API**: L7 HTTP routing (host, path, TLS) to Services via a controller (NGINX, Envoy, cloud ALB). The Gateway API is the newer, more expressive successor.
- **ConfigMap / Secret**: configuration and credentials injected as env vars or files; for real secrets, integrate an external secrets manager.

**Probes**

| Probe | Question | On failure |
| --- | --- | --- |
| Startup | Has the app finished starting? | Delays the other probes; kills the container if startup never completes |
| Readiness | Can it serve traffic now? | Removed from Service endpoints, not restarted |
| Liveness | Is it stuck beyond recovery? | Container restarted |

Pitfalls: a liveness probe that checks the database restarts every pod when the database blips, causing a cascading outage; keep liveness shallow (process responsive). Use a startup probe for slow starts such as loading a large model, rather than a huge liveness initial delay.

**Requests and limits**

- **Requests**: what the scheduler reserves; determines placement and the basis for HPA CPU utilization.
- **Limits**: hard caps. Exceeding memory limit kills the container (OOMKilled); exceeding CPU limit throttles it.
- Common practice: set memory request equal to limit (memory is incompressible); set CPU requests accurately and be cautious with CPU limits, which can cause throttling and latency spikes.
- QoS classes (Guaranteed, Burstable, BestEffort) determine eviction order under node pressure.
- GPUs are requested as extended resources (`nvidia.com/gpu: 1`) and are not shared by default.

**Horizontal Pod Autoscaler**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: api}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: api}
  minReplicas: 3
  maxReplicas: 30
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 65}}
  behavior:
    scaleDown: {stabilizationWindowSeconds: 300}
```

- CPU is a poor signal for I/O-bound or GPU services; scale on custom metrics (requests in flight, queue depth, GPU utilization) via the metrics adapter, or use KEDA for event-driven scaling, including scale to zero.
- HPA reacts in tens of seconds to minutes; node provisioning (Cluster Autoscaler, Karpenter) adds more. Keep headroom (min replicas) for sudden spikes.

**Production must-haves**: at least 3 replicas spread across zones (topology spread constraints, anti-affinity), a PodDisruptionBudget so node drains do not take all replicas, graceful shutdown handling (see 25.35), namespaces with resource quotas, NetworkPolicies, and GitOps (Argo CD, Flux) with Helm or Kustomize for declarative deploys.

### 25.35 How do you deploy with zero downtime, including database schema migrations?

Zero-downtime deployment requires three things: replacing instances gradually while there is always enough healthy capacity (rolling updates gated by readiness), letting old instances finish in-flight work before they exit (graceful shutdown), and making every schema change compatible with both the old and new code running at the same time (expand-and-contract migrations).

**Rolling updates**

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 25%        # extra pods allowed during rollout
    maxUnavailable: 0    # never drop below desired capacity
minReadySeconds: 10
```

- New pods receive traffic only after the readiness probe passes; with `maxUnavailable: 0` capacity never dips.
- Gate rollouts on health: automatic rollback if error rates or latency regress (progressive delivery tools do this).
- During the rollout, old and new versions serve simultaneously, so API and message contracts must be backward and forward compatible.

**Graceful shutdown**

When a pod is deleted, Kubernetes concurrently removes it from Service endpoints and starts termination: run the `preStop` hook, send SIGTERM, then SIGKILL after `terminationGracePeriodSeconds` (default 30 s). Because endpoint removal propagates asynchronously to proxies and load balancers, a pod that exits immediately on SIGTERM drops requests still being routed to it.

- Add a short `preStop` sleep (5 to 15 s) so routing updates propagate before shutdown begins.
- On SIGTERM: stop accepting new connections, fail readiness, finish in-flight requests, close DB pools, flush telemetry, then exit.
- Set the grace period longer than your longest expected request (LLM streams and WebSockets may need minutes); queue workers should stop pulling new messages and finish or requeue current ones.
- Make sure the app process actually receives SIGTERM (PID 1 issues with shell-form entrypoints are a classic bug).

**Expand-and-contract (parallel change) migrations**

Example: rename `users.name` to `users.full_name`.

1. **Expand**: add nullable `full_name` column. Deploy schema only; old code ignores it.
2. **Dual write**: deploy code that writes both columns and reads `name`.
3. **Backfill**: copy existing data in small batches (throttled, resumable), not one giant `UPDATE` that locks the table and bloats WAL.
4. **Switch reads**: deploy code that reads `full_name` (still writing both).
5. **Stop writing old**: deploy code that only uses `full_name`.
6. **Contract**: drop `name` after confirming nothing reads it, including BI tools, dbt models and CDC consumers.

Each step is independently deployable and reversible, and code at version N works with schema at N and N+1.

**Migration safety rules (PostgreSQL examples)**

- `CREATE INDEX CONCURRENTLY` instead of a locking index build.
- Add `NOT NULL` via a `CHECK ... NOT VALID` constraint, validate separately, then set NOT NULL.
- Adding a column with a constant default is cheap in modern PostgreSQL; changing a column type often rewrites the table, so create a new column instead.
- Set `lock_timeout` for migrations so a DDL waiting behind a long query does not block all traffic queued behind it.
- Run migrations as a separate step before the deploy (CI job or pre-deploy Job), not on application startup across many replicas.

**Data-engineering angle**: column renames and type changes break downstream pipelines and CDC consumers even when the application is fine. Treat the schema as a contract, publish changes through a schema registry or data contract, and include the contract phase only once downstream owners have migrated.

### 25.36 How do you implement observability for backend services?

Observability is the ability to explain why a system is behaving the way it is from its outputs. In practice: metrics tell you something is wrong and how big it is, traces show where in a request path time and errors occur, and logs give detailed context for a specific event. OpenTelemetry provides vendor-neutral instrumentation for all three, and SLOs with error budgets turn the data into decisions.

**The three signals**

| Signal | Best for | Watch out for |
| --- | --- | --- |
| Metrics | Dashboards, alerting, trends; cheap aggregates | High-cardinality labels (user_id, request_id) explode cost |
| Traces | Latency breakdown across services, dependency maps | Sampling strategy; context propagation gaps |
| Logs | Detailed debugging, audit | Volume and cost, unstructured text, PII leakage |

- **Structured logs** (JSON) with consistent fields: `service`, `env`, `version`, `trace_id`, `span_id`, `tenant_id`, `route`, `status`, `duration_ms`. Including `trace_id` lets you jump from a log line to its trace.
- **Metrics** as histograms for latency (so you can compute p95 and p99 and aggregate across instances), counters for requests and errors, gauges for saturation.
- **Traces**: context propagation via W3C `traceparent` headers across HTTP, gRPC and message queues (inject trace context into message headers).

**RED and USE**

- **RED** for request-driven services: **R**ate (requests per second), **E**rrors (failed requests per second or ratio), **D**uration (latency distribution). One dashboard row per service and endpoint.
- **USE** for resources: **U**tilization, **S**aturation (queue length, throttling), **E**rrors, for CPU, memory, disk, connection pools, thread pools, GPU.
- Google's four golden signals (latency, traffic, errors, saturation) combine both views.

**OpenTelemetry**

- SDKs and auto-instrumentation for frameworks (FastAPI, requests, SQLAlchemy, gRPC, Kafka clients) produce spans, metrics and logs with shared resource attributes.
- Export via OTLP to the **OpenTelemetry Collector**, which batches, enriches, filters PII, samples, and routes to any backend (Prometheus, Grafana Tempo and Loki, Jaeger, Datadog, Honeycomb), avoiding vendor lock-in.
- Sampling: head-based (decide at the root, cheap) or tail-based in the Collector (keep all errors and slow traces, sample the rest).
- GenAI semantic conventions cover LLM spans: model, token counts, latency, and optionally prompts with care.

```python
from opentelemetry import trace
tracer = trace.get_tracer("rag-service")

with tracer.start_as_current_span("retrieve") as span:
    span.set_attribute("retrieval.top_k", 8)
    docs = await vector_store.search(query, k=8)
    span.set_attribute("retrieval.results", len(docs))
```

**SLOs and error budgets**

- **SLI**: a measurable ratio of good events, e.g. proportion of requests served successfully in under 500 ms.
- **SLO**: target over a window, e.g. 99.9% over 30 days.
- **Error budget**: $1 - \text{SLO}$; 99.9% over 30 days allows about 43 minutes of full unavailability. While budget remains, ship features; when it is exhausted, prioritize reliability work and slow risky deploys.
- **Alert on burn rate**, not raw thresholds: page when the budget is being consumed fast (for example a 14.4x burn rate over a 1-hour window, confirmed by a 5-minute window), ticket on slow burns. This catches real user impact and cuts noisy alerts.
- Alert on symptoms users feel (SLO burn), use causes (CPU, disk) for dashboards and diagnosis.

**For data and AI services**, add domain signals: freshness and completeness of data, queue lag, model latency split into time to first token and tokens per second, token cost per request, cache hit rate, retrieval quality and guardrail trigger rates. A pipeline that runs successfully on stale data is still an incident.

### 25.37 What are the most common API security risks and how do you mitigate them?

The most common API breaches are not exotic exploits but authorization gaps: an authenticated caller changes an ID in the URL and reads someone else's data. The OWASP API Security Top 10 (2023) catalogs these risks; mitigation centers on enforcing authorization on every object and function, validating all inputs against strict schemas, limiting resource consumption, keeping an inventory of every API, and managing secrets properly.

**OWASP API Security Top 10 (2023)**

| # | Risk | Example | Mitigation |
| --- | --- | --- | --- |
| API1 | Broken Object Level Authorization (BOLA) | `GET /invoices/1043` returns another tenant's invoice | Check ownership of every object on every request; scope queries by tenant; random IDs as defense in depth |
| API2 | Broken Authentication | No rate limit on login, weak token validation | Standard protocols (OAuth/OIDC), strict JWT validation, MFA, lockout and throttling |
| API3 | Broken Object Property Level Authorization | Mass assignment of `is_admin`, responses exposing internal fields | Explicit input and output schemas (allow-lists), never bind request bodies directly to ORM models |
| API4 | Unrestricted Resource Consumption | Huge page sizes, giant uploads, unbounded LLM token usage | Rate limits, quotas, max payload and page size, timeouts, cost budgets per tenant |
| API5 | Broken Function Level Authorization | Regular user calls `DELETE /admin/users/7` | Deny by default, role checks on every route, separate admin APIs |
| API6 | Unrestricted Access to Sensitive Business Flows | Bots mass-buying stock or creating accounts | Flow-level rate limits, bot detection, step-up verification |
| API7 | Server-Side Request Forgery (SSRF) | Webhook or "fetch URL" feature hits cloud metadata endpoint | Allow-list destinations, block private IP ranges, isolate egress |
| API8 | Security Misconfiguration | Verbose stack traces, permissive CORS, missing TLS | Hardened defaults, automated config scanning, generic error messages |
| API9 | Improper Inventory Management | Forgotten `/v1` or staging API without auth | API catalog, gateway-only exposure, retire old versions |
| API10 | Unsafe Consumption of APIs | Trusting third-party API responses without validation | Validate and sanitize upstream data, timeouts, TLS |

**BOLA in code**

```python
@app.get("/v1/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, user: User = Depends(current_user)):
    ds = await repo.get(dataset_id, tenant_id=user.tenant_id)   # scope by tenant in the query
    if ds is None or not authz.can_read(user, ds):
        raise HTTPException(404)                                  # do not reveal existence
    return DatasetOut.model_validate(ds)                          # explicit output schema
```

Centralize authorization in a policy layer (OPA, Cedar, or an internal library) and test it: automated tests that call every endpoint as a user from another tenant catch most BOLA regressions.

**Input validation**

- Validate type, length, range, format and enums at the edge with schemas (Pydantic, JSON Schema, OpenAPI request validation); reject unknown fields.
- Parameterized queries everywhere (injection dropped from the API list but not from reality); never build SQL, shell commands or file paths from user input.
- Limit request size, JSON depth and array lengths to prevent parser-based denial of service.
- For LLM-backed APIs, treat model output as untrusted input too: validate before executing tools, rendering HTML or writing to databases, and defend against prompt injection reaching privileged actions.

**Secrets management**

- No secrets in code, images, Git history or plain environment files. Use a secrets manager (Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault) with short-lived, dynamically issued credentials where possible.
- Prefer workload identity (IAM roles for service accounts, federated identity) over static keys.
- Rotate regularly and automatically, scan repositories and CI logs for leaked secrets, and never log tokens or full request bodies with credentials.

**Defense in depth**: TLS everywhere (mTLS internally), an API gateway for authentication, rate limiting and schema validation, least-privilege database users, audit logs of sensitive access, security tests (DAST, fuzzing) in CI, and dependency scanning. The gateway helps, but object-level authorization can only be enforced by the service that understands the data.

---

← [Chapter 24. Evaluation and Testing of AI Systems](../part-4-generative-ai/24-evaluation-and-testing-of-ai-systems.md) · [Table of contents](../README.md) · [Chapter 26. MLOps and LLMOps](26-mlops-and-llmops.md) →
