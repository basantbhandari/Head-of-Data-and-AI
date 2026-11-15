<!-- notion-id: 3bd67418285480739651cd80ce66c26a -->
# Backend Engineering: Performance, Scalability and Reliability

Latency percentiles, retries with backoff and jitter, scalability, the CAP theorem and high availability.

### Q1. Understanding p50 p75 p90 p95 and p99 latency metrics

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

### Q2. What is exponential backoff with jitter?

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

### Q3. How can you build scalability into a software program?

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

### Q4. What is the CAP theorem?

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

### Q5. What does High Availability (HA) mean?

**High Availability (HA)** means designing a system so it remains **available and operational with minimal downtime**, even when failures occur.

- **Redundancy:** Multiple servers or components prevent a single point of failure.
- **Failover:** Traffic automatically moves to a healthy instance when one fails.
- **Load balancing:** Distributes traffic across multiple instances.
- **Health monitoring:** Detects failures and unhealthy services.
- **Auto-scaling:** Adds or removes resources based on demand.
- **Reliable infrastructure:** Uses backups, replicated databases, and multiple availability zones.

**Example:** If one API server fails, a load balancer routes requests to other healthy servers, so users continue using the application.

**In short:** HA = **keeping a system running and accessible despite failures.**
