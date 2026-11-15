# Chapter 6. Distributed Processing and Streaming

_Part II — Data Engineering_

Apache Spark internals, partitioning and performance tuning, Kafka, real-time streaming and NoSQL stores.

This chapter contains **35 questions**.

## Contents

- **6.1** [What is Apache Spark and why is it preferred over Hadoop MapReduce?](#61-what-is-apache-spark-and-why-is-it-preferred-over-hadoop-mapreduce)
- **6.2** [Explain the Spark architecture (Driver, Executor, Cluster Manager).](#62-explain-the-spark-architecture-driver-executor-cluster-manager)
- **6.3** [What is an RDD, and why do we use DataFrames instead?](#63-what-is-an-rdd-and-why-do-we-use-dataframes-instead)
- **6.4** [Explain "Shuffling" and why you should avoid it.](#64-explain-shuffling-and-why-you-should-avoid-it)
- **6.5** [What is a "Wide Transformation" vs. a "Narrow Transformation"?](#65-what-is-a-wide-transformation-vs-a-narrow-transformation)
- **6.6** [What is "Lazy Evaluation" in Spark?](#66-what-is-lazy-evaluation-in-spark)
- **6.7** [What is "Data Skew" and how do you fix it?](#67-what-is-data-skew-and-how-do-you-fix-it)
- **6.8** [How does Spark handle fault tolerance?](#68-how-does-spark-handle-fault-tolerance)
- **6.9** [What is the difference between repartition() and coalesce()?](#69-what-is-the-difference-between-repartition-and-coalesce)
- **6.10** [What are Broadcast Variables?](#610-what-are-broadcast-variables)
- **6.11** [Explain Spark's "Catalyst Optimizer."](#611-explain-sparks-catalyst-optimizer)
- **6.12** [What is "Speculative Execution" in Spark?](#612-what-is-speculative-execution-in-spark)
- **6.13** [What is the difference between persist() and cache()?](#613-what-is-the-difference-between-persist-and-cache)
- **6.14** [How do you handle "Out of Memory" (OOM) errors in Spark?](#614-how-do-you-handle-out-of-memory-oom-errors-in-spark)
- **6.15** [What is a "Checkpoint" in Spark?](#615-what-is-a-checkpoint-in-spark)
- **6.16** [What is the purpose of the SparkSession?](#616-what-is-the-purpose-of-the-sparksession)
- **6.17** [What is "Predicate Pushdown"?](#617-what-is-predicate-pushdown)
- **6.18** [Explain the difference between reduceByKey and groupByKey.](#618-explain-the-difference-between-reducebykey-and-groupbykey)
- **6.19** [What is "Z-Ordering"?](#619-what-is-z-ordering)
- **6.20** [What is the Hadoop "NameNode"?](#620-what-is-the-hadoop-namenode)
- **6.21** [What is Apache Kafka?](#621-what-isapache-kafka)
- **6.22** [Explain Kafka "Topics," "Partitions," and "Offsets."](#622-explain-kafka-topics-partitions-and-offsets)
- **6.23** [What is a "Consumer Group" in Kafka?](#623-what-is-a-consumer-group-in-kafka)
- **6.24** [What is "Exactly-Once Semantics" (EOS)?](#624-what-is-exactly-once-semantics-eos)
- **6.25** [What is a "NoSQL" database and name 4 types?](#625-what-is-a-nosql-database-and-name-4-types)
- **6.26** [When would you use Cassandra instead of MySQL?](#626-when-would-you-use-cassandra-instead-of-mysql)
- **6.27** [What is "Eventual Consistency"?](#627-what-is-eventual-consistency)
- **6.28** [What is a "Time-Series Database" (TSDB)?](#628-what-is-a-time-series-database-tsdb)
- **6.29** [What is "Windowing" in streaming (Tumbling, Sliding, Session)?](#629-what-is-windowing-in-streaming-tumbling-sliding-session)
- **6.30** [What is a "Dead Letter Topic" in Kafka?](#630-what-is-a-dead-letter-topic-in-kafka)
- **6.31** [What is "Schema Registry" in Kafka?](#631-what-is-schema-registry-in-kafka)
- **6.32** [What is "Stream-Table Duality"?](#632-what-is-stream-table-duality)
- **6.33** [What is Apache Flink?](#633-what-is-apache-flink)
- **6.34** [What is "Backpressure" in a streaming system?](#634-what-is-backpressure-in-a-streaming-system)
- **6.35** [What is a "Vector Database"?](#635-what-is-a-vector-database)

---

### 6.1 What is Apache Spark and why is it preferred over Hadoop MapReduce?

**Spark is a distributed computing engine that processes data "in-memory" (RAM), whereas MapReduce writes intermediate results to disk. This makes Spark significantly faster, especially for iterative algorithms.**

### 6.2 Explain the Spark architecture (Driver, Executor, Cluster Manager).

**Apache Spark** follows a **master-worker (distributed) architecture** designed to process large-scale data in parallel across a cluster of machines. The three core components are the **Driver**, **Executors**, and the **Cluster Manager**.

**1. Driver Program**

What it is:

- The **"brain"** of a Spark application — runs the `main()` function and creates the **SparkContext** (or SparkSession)
- Coordinates the entire execution of the Spark job

Responsibilities:

- **Converts user code** into a logical execution plan (DAG — Directed Acyclic Graph)
- **Splits the job into stages and tasks**
- **Schedules tasks** to run on executors
- **Tracks the status** of tasks and executors
- **Collects results** back from executors when needed (e.g., `.collect()`)

Key point:

- The driver runs on a **single node** (either the client machine or a cluster node, depending on deploy mode)
- If the driver fails, the **entire application fails** (it's a potential single point of failure for the job)

---

**2. Cluster Manager**

What it is:

- The component responsible for **allocating resources** (CPU, memory) across the cluster so Spark applications can run
- Acts as a **resource negotiator** between the Driver and the physical cluster

Responsibilities:

- **Launches executors** on worker nodes based on what the driver requests
- **Monitors resource availability** across the cluster
- Manages **multiple applications** sharing the same cluster (in shared cluster environments)

Common Cluster Manager options:

- **Standalone** — Spark's own built-in cluster manager (simple, for dedicated Spark clusters)
- **YARN** — Hadoop's resource manager (common in Hadoop ecosystems)
- **Mesos** — general-purpose cluster manager (less common now)
- **Kubernetes** — container-based orchestration (increasingly popular for cloud-native deployments)

---

**3. Executors**

What they are:

- **Worker processes** launched on cluster nodes that actually **execute the tasks** assigned by the driver
- Each Spark application gets its **own set of executors** for the duration of the application

Responsibilities:

- **Run the actual computation** (tasks) on partitions of data
- **Store data in memory/disk** for caching (`.cache()`/`.persist()`) and intermediate shuffle data
- **Report task status and results** back to the driver

Key point:

- Executors run **in parallel** across multiple nodes, which is what gives Spark its distributed processing power
- Each executor has a fixed number of **cores** and **memory**, determining how many tasks it can run simultaneously

---

**How it all fits together (execution flow):**

1. You submit a Spark application → the **Driver** starts and creates a SparkSession
2. The Driver contacts the **Cluster Manager** to request resources
3. The Cluster Manager allocates worker nodes and launches **Executors** on them
4. The Driver analyzes your code, builds a **DAG** of operations, and breaks it into **stages** and **tasks**
5. Tasks are sent to the **Executors** to run in parallel on different partitions of data
6. Executors process the data and send results/status back to the **Driver**
7. Once the job completes, the Driver collects final results (if needed) and the application ends

---

**Visual summary:**

```txt
                    ┌─────────────────┐
                    │  Driver Program  │
                    │ (SparkContext)   │
                    └────────┬─────────┘
                             │ requests resources
                             ▼
                    ┌─────────────────┐
                    │ Cluster Manager  │
                    │ (YARN/Mesos/K8s/ │
                    │  Standalone)     │
                    └────────┬─────────┘
                             │ allocates & launches
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
           ┌──────────┐┌──────────┐┌──────────┐
           │ Executor 1││ Executor 2││ Executor 3│
           │ (Tasks)   ││ (Tasks)   ││ (Tasks)   │
           └──────────┘└──────────┘└──────────┘
```

---

**Key architectural concepts:**

| Concept | Description |
| --- | --- |
| **Job** | Triggered by an action (e.g., `.collect()`, `.count()`) |
| **Stage** | A job is split into stages, divided by shuffle boundaries |
| **Task** | The smallest unit of work — one task per data partition |
| **Partition** | A chunk of the dataset distributed across executors |

---

**Why this architecture matters:**

- **Parallelism** — data is partitioned and processed simultaneously across many executors, dramatically speeding up large-scale computation
- **Fault tolerance** — if an executor fails, the driver can reschedule its tasks on another executor (thanks to Spark's lineage/DAG tracking via RDDs)
- **Scalability** — you can scale out by adding more worker nodes/executors as data volume grows

**In short:** the **Driver** plans and coordinates the job, the **Cluster Manager** allocates resources, and the **Executors** do the actual distributed computation — together enabling Spark's fast, fault-tolerant, large-scale data processing.

### 6.3 What is an RDD, and why do we use DataFrames instead?

**RDDs (Resilient Distributed Datasets) are low-level collections without a schema. DataFrames are higher-level, organized into named columns, allowing Spark's Catalyst Optimizer to create more efficient execution plans.**

### 6.4 Explain "Shuffling" and why you should avoid it.

**What is Shuffling?**

- The process of **redistributing data across partitions/nodes** in a distributed system (like Spark or Hadoop)
- Happens when an operation requires data with the **same key** to be grouped together on the same node — but that data is currently scattered across different partitions
- Involves **writing data to disk, transferring it over the network**, and **reading it back** on new nodes

**When it happens:**

- Operations like `groupBy()`, `join()`, `distinct()`, `repartition()`, `reduceByKey()`

**Why avoid it (or minimize it):**

- **Expensive network I/O** — data moves across nodes, which is slow compared to in-memory operations
- **Disk I/O overhead** — intermediate shuffle data is often written to disk
- **Serialization/deserialization cost** — data must be packaged for transfer and unpacked on arrival
- **Can cause memory pressure** — large shuffles may spill to disk or cause out-of-memory errors
- **Creates a stage boundary** — shuffling splits a Spark job into separate stages, adding scheduling overhead

**How to minimize shuffling:**

- Use `reduceByKey()` instead of `groupByKey()` (combines data locally before shuffling, less data moved)
- **Broadcast small tables** in joins instead of shuffling both sides (`broadcast join`)
- **Partition data wisely** upfront so related data is already co-located
- Avoid unnecessary `repartition()` calls
- Filter/reduce data **before** operations that trigger a shuffle, so less data needs to move

**In short:** Shuffling is costly because it moves data across the network and disk — minimizing it (via smarter operations, broadcasting, and pre-filtering) is one of the biggest levers for improving distributed processing performance.

### 6.5 What is a "Wide Transformation" vs. a "Narrow Transformation"?

**Narrow Transformation**

What it is:

- A transformation where each **input partition contributes to only one output partition**
- Data **doesn't need to move** across nodes — it can be computed **locally** within the same partition

Characteristics:

- No shuffling required
- Fast and efficient (no network/disk I/O for data movement)
- Can be **pipelined** — Spark can chain multiple narrow transformations together into a single stage

Examples:

- `map()`
- `filter()`
- `flatMap()`
- `union()`

---

**Wide Transformation**

What it is:

- A transformation where **input partitions contribute to multiple output partitions**
- Requires **shuffling** — data must be redistributed across nodes so that related data (e.g., same key) ends up together

Characteristics:

- Involves **network and disk I/O** (expensive)
- Creates a **new stage boundary** in the Spark execution plan
- Slower compared to narrow transformations

Examples:

- `groupByKey()`
- `reduceByKey()`
- `join()`
- `distinct()`
- `repartition()`

---

**Visual difference:**

**Narrow (1-to-1 partition mapping):**

```txt
Partition 1 ──▶ Partition 1
Partition 2 ──▶ Partition 2
Partition 3 ──▶ Partition 3
```

**Wide (data moves across partitions):**

```txt
Partition 1 ──┐
Partition 2 ──┼──▶ Partition 1, 2, 3 (mixed/shuffled)
Partition 3 ──┘
```

---

**Quick comparison**

| Aspect | Narrow Transformation | Wide Transformation |
| --- | --- | --- |
| Data movement | None (stays in same partition) | Requires shuffle across partitions/nodes |
| Performance | Fast | Slower (network + disk I/O) |
| Stage boundary | No new stage created | Creates a new stage |
| Fault recovery | Cheap (recompute single partition) | Expensive (may need to recompute multiple partitions) |
| Examples | `map`, `filter`, `flatMap`, `union` | `groupByKey`, `reduceByKey`, `join`, `distinct` |

---

**Why this distinction matters:**

- Spark's optimizer tries to **chain narrow transformations together** into a single stage (pipelining) since they don't require moving data
- **Wide transformations** are the main cause of **shuffling**, which is why minimizing their use (or optimizing how they're used — e.g., `reduceByKey` over `groupByKey`) is a key performance strategy in Spark

**In short:** Narrow transformations keep data within the same partition (fast, no shuffle), while wide transformations require redistributing data across partitions (slow, triggers a shuffle) — understanding this helps you write more efficient Spark jobs.

### 6.6 What is "Lazy Evaluation" in Spark?

**Lazy Evaluation** means Spark **doesn't execute transformations immediately** when they're called. Instead, it builds up a **logical execution plan (DAG)** of all the transformations, and only actually runs the computation when an **action** is triggered.

---

**How it works:**

- **Transformations** (e.g., `map()`, `filter()`, `select()`) are **lazy** — Spark just records what needs to be done, without doing it yet
- **Actions** (e.g., `.collect()`, `.count()`, `.show()`, `.write()`) trigger the actual execution — this is when Spark looks at the full chain of transformations and runs them

```py
df2 = df.filter(df.age > 30)      # Lazy — nothing happens yet
df3 = df2.select("name", "age")    # Lazy — still nothing happens
df3.show()                         # ACTION — now Spark executes everything
```

Until `.show()` is called, Spark hasn't actually touched the data — it's just built a plan.

---

**Why Spark does this:**

1. **Query Optimization** — Since Spark sees the _entire_ chain of transformations before running anything, it can optimize the execution plan as a whole (e.g., combining filters, reordering operations, skipping unnecessary computations) via the **Catalyst optimizer**
2. **Avoids Unnecessary Computation** — If part of the data or a transformation isn't actually needed for the final result, Spark can skip it entirely
3. **Efficient Pipelining** — Narrow transformations can be **chained together** into a single stage, minimizing overhead, since Spark knows the full sequence in advance
4. **Fault Tolerance** — Since Spark keeps track of the **lineage** (the sequence of transformations, via the DAG), if a partition is lost, Spark can **recompute just that piece** from the original data rather than needing to store everything redundantly

---

**Example showing optimization benefit:**

```py
df2 = df.filter(df.country == "US")
df3 = df2.filter(df.age > 30)
df3.show()
```

Instead of running two separate filter passes over the data, Spark's optimizer can **combine both filters into one pass**, since it saw the full plan before execution — this is only possible _because_ of lazy evaluation.

---

**Transformations vs. Actions (quick reference)**

| Type | Behavior | Examples |
| --- | --- | --- |
| **Transformation** | Lazy — builds the plan, doesn't execute | `map()`, `filter()`, `select()`, `groupBy()`, `join()` |
| **Action** | Triggers execution of the full plan | `.collect()`, `.count()`, `.show()`, `.write()`, `.take()` |

---

**In short:** Spark delays actual computation until an action is called, building a logical plan (DAG) along the way — this allows for query optimization, efficient execution, and better fault tolerance, rather than executing each transformation immediately and inefficiently.

### 6.7 What is "Data Skew" and how do you fix it?

**Data skew occurs when one partition has significantly more data than others, causing a single node to slow down the entire job. It can be fixed by "salting" keys or using broadcast joins.**

### 6.8 How does Spark handle fault tolerance?

**Spark uses its DAG to track the lineage of data. If a node fails and a partition is lost, Spark re-runs the specific transformations from the original source to reconstruct that partition.**

### 6.9 What is the difference between repartition() and coalesce()?

**`repartition()` can increase or decrease partitions and triggers a full shuffle. `coalesce()` can only decrease partitions and avoids a full shuffle, making it more efficient for reducing file count.**

### 6.10 What are Broadcast Variables?

**They allow you to cache a small, read-only variable on every worker node once, rather than sending it with every task. This is used in "Broadcast Joins" to avoid shuffling a small table.**

### 6.11 Explain Spark's "Catalyst Optimizer."

**The Catalyst Optimizer is the engine that optimizes Spark SQL queries. It performs logical plan optimization and physical planning to find the fastest way to execute a query.**

### 6.12 What is "Speculative Execution" in Spark?

**If a task is running much slower than others (a "straggler"), Spark launches a duplicate of that task on another node. Whichever finishes first is kept, and the other is killed to save time.**

### 6.13 What is the difference between persist() and cache()?

**`cache()` uses the default storage level (Memory only). `persist()` allows the user to specify the storage level, such as "Memory and Disk" or "Serialized."**

### 6.14 How do you handle "Out of Memory" (OOM) errors in Spark?

**I check for high concurrency (too many tasks), giant partitions that need repartitioning, or attempts to broadcast a table that is too large for the executor's memory.**

### 6.15 What is a "Checkpoint" in Spark?

**Checkpointing saves the state of a DataFrame to reliable storage and truncates its lineage. This prevents the DAG from becoming too long and complex in iterative workloads.**

### 6.16 What is the purpose of the SparkSession?

**Since Spark 2.0, `SparkSession` is the unified entry point for reading data, using SQL, and managing configurations, replacing the older `SparkContext`.**

### 6.17 What is "Predicate Pushdown"?

**An optimization where data filtering (the `WHERE` clause) is pushed down to the storage layer (like Parquet), so only the relevant data is read into memory.**

### 6.18 Explain the difference between reduceByKey and groupByKey.

**`reduceByKey` performs a local merge on each node before shuffling, drastically reducing network traffic. `groupByKey` shuffles all data first, which often leads to OOM errors.**

### 6.19 What is "Z-Ordering"?

**A technique used in Delta Lake to co-locate related information in the same files, allowing the engine to skip large amounts of irrelevant data during queries.**

### 6.20 What is the Hadoop "NameNode"?

**The NameNode is the master server in HDFS that manages the file system namespace and knows the location of every block of data stored in the cluster.**

### 6.21 What is Apache Kafka?

**Kafka is a distributed event streaming platform used to build high-throughput, fault-tolerant data pipelines that can buffer data between producers and consumers.**

### 6.22 Explain Kafka "Topics," "Partitions," and "Offsets."

**A Topic is a category of data. A Partition is a subset of a topic used for parallel processing. An Offset is a unique ID for a message, allowing consumers to track their progress.**

### 6.23 What is a "Consumer Group" in Kafka?

**A group of consumers that work together to read from a topic. Each partition is assigned to only one member of the group, ensuring parallel processing without duplicating messages.**

### 6.24 What is "Exactly-Once Semantics" (EOS)?

**The guarantee that a message is processed exactly one time, even in the event of a system or network failure, preventing both data loss and duplicates.**

### 6.25 What is a "NoSQL" database and name 4 types?

**A non-relational database. The four types are: Key-Value (Redis), Document (MongoDB), Wide-column (Cassandra), and Graph (Neo4j).**

### 6.26 When would you use Cassandra instead of MySQL?

**When you need massive write scalability across multiple data centers and high availability, and you don't require complex SQL joins or strict multi-table ACID transactions.**

### 6.27 What is "Eventual Consistency"?

**A model where data updates will eventually propagate to all nodes, but for a short time, different users might see different versions of the data.**

### 6.28 What is a "Time-Series Database" (TSDB)?

**A database optimized for time-stamped data, such as IoT sensor readings or stock market prices, allowing for very fast time-based aggregations.**

### 6.29 What is "Windowing" in streaming (Tumbling, Sliding, Session)?

**Tumbling windows are fixed-size and non-overlapping. Sliding windows overlap. Session windows are defined by periods of activity followed by a gap of inactivity.**

### 6.30 What is a "Dead Letter Topic" in Kafka?

**A specific topic where messages that fail to process are sent for later manual review, ensuring the main stream continues to flow.**

### 6.31 What is "Schema Registry" in Kafka?

**A central service that stores and manages schemas for Kafka messages, ensuring that producers and consumers remain compatible as data formats change.**

### 6.32 What is "Stream-Table Duality"?

**The concept that a stream (changes over time) can be turned into a table (current state), and a table can be turned into a stream (a feed of updates).**

### 6.33 What is Apache Flink?

**A streaming-first engine that processes every event individually (true streaming), offering lower latency than Spark's micro-batching approach.**

### 6.34 What is "Backpressure" in a streaming system?

**A situation where the producer sends data faster than the consumer can process it. The system must have a way to signal the producer to slow down to prevent crashes.**

### 6.35 What is a "Vector Database"?

**A database designed to store data as high-dimensional vectors, which is the core technology behind similarity searches in AI and LLM applications.**

---

← [Chapter 5. SQL, Databases and Data Modeling](05-sql-databases-and-data-modeling.md) · [Table of contents](../README.md) · [Chapter 7. Pipelines and Orchestration](07-pipelines-and-orchestration.md) →
