# Chapter 6. Distributed Processing and Streaming

_Part II — Data Engineering_

Apache Spark internals, partitioning and performance tuning, Kafka, real-time streaming and NoSQL stores.

This chapter contains **47 questions**.

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
- **6.36** [What join strategies does Spark use (broadcast hash, shuffle hash, sort-merge, broadcast nested loop) and how does it choose between them?](#636-what-join-strategies-does-spark-use-broadcast-hash-shuffle-hash-sort-merge-broadcast-nested-loop-and-how-does-it-choose-between-them)
- **6.37** [What is Adaptive Query Execution (AQE) in Spark, and what problems does it solve?](#637-what-is-adaptive-query-execution-aqe-in-spark-and-what-problems-does-it-solve)
- **6.38** [How do you choose the number of shuffle partitions and size Spark executors (cores, memory, overhead)?](#638-how-do-you-choose-the-number-of-shuffle-partitions-and-size-spark-executors-cores-memory-overhead)
- **6.39** [How does Spark's unified memory model work (execution vs storage memory, off-heap, overhead)?](#639-how-does-sparks-unified-memory-model-work-execution-vs-storage-memory-off-heap-overhead)
- **6.40** [What is the difference between event time and processing time, and how do watermarks handle late data?](#640-what-is-the-difference-between-event-time-and-processing-time-and-how-do-watermarks-handle-late-data)
- **6.41** [How does Spark Structured Streaming achieve fault tolerance and end-to-end exactly-once (checkpoints, offsets, idempotent/transactional sinks)?](#641-how-does-spark-structured-streaming-achieve-fault-tolerance-and-end-to-end-exactly-once-checkpoints-offsets-idempotenttransactional-sinks)
- **6.42** [How do stream-stream joins work, and why do they need watermarks and state bounds?](#642-how-do-stream-stream-joins-work-and-why-do-they-need-watermarks-and-state-bounds)
- **6.43** [How does Kafka guarantee ordering and durability? Explain acks, min.insync.replicas, idempotent producers and transactions.](#643-how-does-kafka-guarantee-ordering-and-durability-explain-acks-mininsyncreplicas-idempotent-producers-and-transactions)
- **6.44** [What is Kafka log compaction and how does it differ from time-based retention? When would you use it?](#644-what-is-kafka-log-compaction-and-how-does-it-differ-from-time-based-retention-when-would-you-use-it)
- **6.45** [What causes consumer group rebalancing and consumer lag in Kafka, and how do you deal with them?](#645-what-causes-consumer-group-rebalancing-and-consumer-lag-in-kafka-and-how-do-you-deal-with-them)
- **6.46** [Compare Lambda and Kappa architectures. When does each make sense today?](#646-compare-lambda-and-kappa-architectures-when-does-each-make-sense-today)
- **6.47** [How do you choose between Spark Structured Streaming, Flink and Kafka Streams?](#647-how-do-you-choose-between-spark-structured-streaming-flink-and-kafka-streams)

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

### 6.36 What join strategies does Spark use (broadcast hash, shuffle hash, sort-merge, broadcast nested loop) and how does it choose between them?

Spark picks a physical join strategy based on whether the join has an equality condition, the estimated size of each side, the join type, and any hints. For equi-joins the preference is broadcast hash join, then sort-merge join (the default for large-to-large), with shuffle hash join used when enabled or hinted. Non-equi joins fall back to broadcast nested loop or a cartesian product.

**The five strategies**

| Strategy | Shuffle? | Needs equi-join? | When it wins | Risk |
| --- | --- | --- | --- | --- |
| Broadcast hash join (BHJ) | No (small side is broadcast) | Yes | One side fits comfortably in executor and driver memory | Driver/executor OOM if the "small" side is actually large |
| Shuffle hash join (SHJ) | Yes, both sides | Yes | One side is much smaller per partition but too big to broadcast | Builds an in-memory hash map per partition; skew can OOM a task |
| Sort-merge join (SMJ) | Yes, plus sort | Yes, sortable keys | Two large tables; robust because it can spill | Sort cost; skewed keys create stragglers |
| Broadcast nested loop join (BNLJ) | No | No | Non-equi conditions (ranges, `LIKE`) with one small side | Quadratic comparisons |
| Cartesian product | Yes | No | Explicit cross joins | Output explodes |

**How the planner decides**

1. **Hints first.** `BROADCAST` > `MERGE` > `SHUFFLE_HASH` > `SHUFFLE_REPLICATE_NL`. A hint is honored only if the strategy supports the join type (for example, you cannot broadcast the preserved side of a left outer join).
2. **Equi-join without hints:** broadcast if one side's estimated size is below `spark.sql.autoBroadcastJoinThreshold` (10 MB by default); otherwise shuffle hash if `spark.sql.join.preferSortMergeJoin=false` and one side is clearly smaller; otherwise sort-merge.
3. **Non-equi join:** broadcast nested loop if one side can be broadcast, otherwise cartesian (for inner joins).
4. **At runtime with AQE:** statistics from completed shuffle stages can convert a planned SMJ into a BHJ, or into a SHJ when partitions are small enough (`spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold`).

**Senior considerations**

- Static size estimates come from file sizes or table statistics and are often wrong after filters or for computed DataFrames. Run `ANALYZE TABLE ... COMPUTE STATISTICS` or rely on AQE rather than raising the broadcast threshold blindly.
- A broadcast table is collected to the driver and then sent to every executor, so a 1 GB "small" table costs driver memory plus one copy per executor. Above a few hundred MB, broadcasting is usually a bad idea.
- Bucketing both tables on the join key with the same bucket count lets SMJ skip the shuffle entirely, which pays off for joins that repeat often.
- Always confirm the choice with `explain()`. A surprise `BroadcastNestedLoopJoin` usually means someone wrote an `OR` or a range condition in the join and the query has become quadratic.

```python
from pyspark.sql.functions import broadcast
orders.join(broadcast(dim_country), "country_code").explain()
```

### 6.37 What is Adaptive Query Execution (AQE) in Spark, and what problems does it solve?

AQE re-optimizes a Spark SQL query while it runs, using real statistics collected at shuffle and broadcast boundaries instead of compile-time estimates. It has been on by default since Spark 3.2. After each query stage finishes, Spark knows the actual size of every shuffle partition and can re-plan the rest of the query.

**The problems it addresses**

- **Wrong partition count.** A single `spark.sql.shuffle.partitions` value is too high for small stages (thousands of tiny tasks) and too low for big ones (spills). AQE **coalesces** adjacent small shuffle partitions toward `spark.sql.adaptive.advisoryPartitionSizeInBytes` (64 MB by default).
- **Bad join choice.** Estimates after filters and UDFs are often wrong. If a side turns out to be small at runtime, AQE **converts sort-merge join to broadcast hash join**, and the local shuffle reader avoids re-shuffling the other side. It can also switch to shuffled hash join when every partition is small enough to hash.
- **Skewed joins.** A partition is treated as skewed when it is larger than `skewedPartitionFactor` (5x) times the median *and* larger than `skewedPartitionThresholdInBytes` (256 MB). AQE splits it into sub-partitions and duplicates the matching partition from the other side, so one key no longer creates a straggler task.
- **Empty relations.** If a stage produces no rows, AQE can short-circuit joins and aggregations that depend on it.
- **Skew in `REBALANCE` writes.** Skewed partitions are split so output files come out evenly sized.

**How it works in practice**

The plan is cut into query stages at exchanges. Spark runs the leaf stages, materializes their shuffle output, reads the map output statistics, re-runs the optimizer on what is left, and repeats. The Spark UI shows `AdaptiveSparkPlan isFinalPlan=true` along with `AQEShuffleRead` nodes once re-planning has happened.

**What AQE does not fix**

- Skew inside **aggregations** or window functions. Skew-join handling covers sort-merge joins only, so salting or two-phase aggregation is still needed there.
- Skew on a side that cannot be split for that join type (for example the non-preserved side of an outer join).
- Problems before the first shuffle, such as a few huge input files or poor file layout.
- Streaming queries. AQE is not applied to Structured Streaming queries.

**Tuning tips**

- Set the initial partition count high (for example `spark.sql.adaptive.coalescePartitions.initialPartitionNum`) and let AQE coalesce down. Over-partitioning is cheap to fix at runtime; under-partitioning is not.
- On shared, busy clusters, set `coalescePartitions.parallelismFirst=false` so AQE honors the advisory size instead of maximizing task count.
- Lower the skew threshold for jobs where 100 MB partitions already hurt.

### 6.38 How do you choose the number of shuffle partitions and size Spark executors (cores, memory, overhead)?

Size shuffle partitions by data volume, aiming for roughly 100–200 MB of shuffled data per partition, and size executors around a moderate number of cores (about 4–5) with enough memory per core to hold a few partitions in flight. Then let AQE adjust partition counts at runtime and confirm the result with spill and GC metrics in the Spark UI.

**Shuffle partitions**

The default `spark.sql.shuffle.partitions=200` fits almost nobody. A practical starting point:

$$
N_{partitions} \approx \max\left(\frac{\text{shuffle bytes}}{\text{target partition size}},\ 2 \times \text{total cores}\right)
$$

Example: a 500 GB shuffle at a 200 MB target needs about 2,500 partitions. With 400 cores, that is around six waves of tasks, which is healthy. Keeping the count a multiple of total cores avoids a final wave that leaves most cores idle.

- **Too few partitions:** large tasks, spill to disk, OOM, poor parallelism.
- **Too many partitions:** scheduler overhead, tiny tasks, and many small output files.
- With AQE, set a generous initial number and let coalescing reduce it.

**Executor sizing**

| Knob | Typical guidance | Why |
| --- | --- | --- |
| `spark.executor.cores` | 4–5 | Higher counts hurt HDFS/S3 client throughput and GC; 1 core wastes per-JVM overhead and broadcast copies |
| `spark.executor.memory` | 4–8 GB per core, depending on workload | Each concurrent task gets roughly (unified memory / cores) |
| `spark.executor.memoryOverhead` | Default max(384 MB, 10% of heap); raise for PySpark, Arrow, native libraries | Off-heap usage: thread stacks, netty buffers, Python workers |
| Executors per node | (node cores − 1) / cores per executor | Leave capacity for the OS and node daemons |

**Worked example**

A node has 16 cores and 128 GB of RAM. Reserve 1 core and about 8 GB. That leaves 15 cores, so run 3 executors with 5 cores each. About 40 GB per executor, minus 10% overhead, gives `executor.memory` of roughly 36 GB and `memoryOverhead` of roughly 4 GB.

**Containers that get killed on YARN or Kubernetes** (exit code 137, "container killed for exceeding memory limits") usually point to overhead being too small, not the heap. This is common in PySpark, where Python worker memory lives outside the JVM. Raise `memoryOverhead` or set `spark.executor.pyspark.memory`.

**Driver sizing.** The driver needs more memory when you `collect()`, broadcast large tables, or plan queries with many files or partitions. Set `spark.driver.maxResultSize` as a guardrail.

**Dynamic allocation.** On shared clusters, enable `spark.dynamicAllocation.enabled` (with shuffle tracking or an external shuffle service) so executors scale with stage parallelism instead of being fixed for the whole job.

**Validate, do not guess.** In the Spark UI, look at spill (memory and disk), GC time (keep it under about 10% of task time), task duration distribution (skew), and peak execution memory. Tune one knob at a time against a representative input.

### 6.39 How does Spark's unified memory model work (execution vs storage memory, off-heap, overhead)?

Since Spark 1.6, execution memory (shuffles, joins, sorts, aggregations) and storage memory (cached data, broadcast blocks) share one unified pool inside the executor heap. Each side can borrow from the other, but execution can evict cached blocks while storage can never evict execution. Outside the heap sit overhead memory and, optionally, an off-heap region.

**Layout of an executor container**

```txt
Container = executor heap + memoryOverhead (+ offHeap.size if enabled)

Executor heap (spark.executor.memory)
├── Reserved memory: 300 MB (fixed)
├── User memory: (heap - 300 MB) * (1 - spark.memory.fraction)
│     UDF objects, user data structures, internal metadata
└── Unified memory M: (heap - 300 MB) * spark.memory.fraction (0.6)
      ├── Storage region R: M * spark.memory.storageFraction (0.5), protected from eviction
      └── Execution: the rest, can borrow all unused storage
```

For a 10 GB heap: usable memory is $10240 - 300 = 9940$ MB, and $M = 0.6 \times 9940 \approx 5964$ MB, with about 2982 MB of that protected for storage.

**Borrowing rules**

- If storage is idle, execution can use all of M, and the reverse is also true.
- Execution needs memory and storage has borrowed some: execution **evicts cached blocks** until storage drops back to R.
- Storage needs memory and execution is using it: storage **cannot evict** execution. The cache block is dropped, or written to disk if the storage level allows it.
- Execution memory is shared among the running tasks: each of N concurrent tasks gets between $\frac{1}{2N}$ and $\frac{1}{N}$ of the execution pool. When a task cannot get enough, it **spills** to disk.

**Off-heap memory**

- `spark.memory.offHeap.enabled=true` with `spark.memory.offHeap.size` adds a separate unified pool outside the JVM. Tungsten stores data there in binary form, which reduces GC pressure for large heaps.
- This size is added to the container request on YARN and Kubernetes, so account for it.

**Overhead memory**

- `spark.executor.memoryOverhead` defaults to max(384 MB, 10% of executor memory). It covers JVM internals, thread stacks, netty network buffers, and native libraries.
- PySpark Python workers, Arrow buffers, and native readers live outside the heap too. When they are not budgeted, the cluster manager kills the container even though the heap itself looks fine.

**How this shows up in production**

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `java.lang.OutOfMemoryError: Java heap space` in a task | One huge partition, skewed key, or a large object in user memory | More partitions, salting, avoid collecting into lists |
| Container killed (exit code 137 / 143) | Overhead too small | Raise `memoryOverhead` or `pyspark.memory` |
| Cache keeps getting evicted | Execution-heavy job borrowing storage | Cache less, persist with `MEMORY_AND_DISK`, or raise `storageFraction` |
| Long GC pauses | Huge heaps with many objects | Smaller executors, off-heap, G1GC |

Most teams should leave `memory.fraction` and `storageFraction` at their defaults and fix partition sizes and overhead first.

### 6.40 What is the difference between event time and processing time, and how do watermarks handle late data?

Event time is when something actually happened, stamped by the producer. Processing time is when the streaming engine sees it. Correct business results (revenue per minute, sessions per user) need event time, but event-time data arrives out of order and late. A watermark is the engine's moving estimate of how far event time has progressed, and it tells the engine when it is safe to finalize a window and drop its state.

**Why they diverge**

- Mobile devices go offline and upload hours later.
- Producer retries, partition lag, and backfills reorder events.
- An outage in the pipeline makes processing time jump while event time stays put.

Aggregating by processing time is simple and always "complete", but the numbers shift with every incident. Aggregating by event time is correct but requires deciding how long to wait.

**How watermarks work**

In Spark Structured Streaming:

$$
\text{watermark} = \max(\text{event time seen}) - \text{allowed lateness}
$$

```python
(events
  .withWatermark("event_time", "15 minutes")
  .groupBy(window("event_time", "5 minutes"), "store_id")
  .agg(sum("amount").alias("revenue")))
```

- A window ending at 10:05 is finalized once the watermark passes 10:05, which happens when an event with time 10:20 or later has been seen.
- Spark guarantees data **within** the threshold is included. Data older than the watermark **may** be dropped (it is not guaranteed to be dropped right away).
- In **append** mode, results are emitted only once a window closes. In **update** mode, partial results are emitted and refined.
- With several input streams, the global watermark defaults to the **minimum** across inputs, so one slow stream holds everything back. `spark.sql.streaming.multipleWatermarkPolicy=max` trades correctness for latency.

Flink generates watermarks at the source with a `WatermarkStrategy` (for example bounded out-of-orderness) and adds **idleness** detection so an empty partition does not stall the watermark. It also supports `allowedLateness` (which re-fires windows) and **side outputs** for records that arrive too late.

**Choosing the lateness bound**

- Measure the distribution of `processing_time - event_time` and choose a high percentile (for example p99.9), not the maximum.
- A longer bound gives more complete results, but adds latency and state size, because every open window is held in memory or RocksDB.
- A shorter bound gives faster, cheaper results but drops more data.

**Handling data later than the watermark**

- Route it to a late-data table or side output rather than silently losing it.
- Run a periodic batch reconciliation that recomputes affected windows from the raw event log in the lakehouse. This is the common pattern for financial numbers.
- Make sinks upsert-friendly (`MERGE` on window key) so corrections overwrite earlier results.

**Common pitfalls:** event timestamps in mixed time zones, producers that send clock-skewed future timestamps (they push the watermark ahead and cause valid data to be dropped), and forgetting the watermark entirely, which makes state grow without limit.

### 6.41 How does Spark Structured Streaming achieve fault tolerance and end-to-end exactly-once (checkpoints, offsets, idempotent/transactional sinks)?

Structured Streaming gets end-to-end exactly-once from three parts working together: a **replayable source** (such as Kafka offsets or file listings), a **checkpoint** that records which offsets each micro-batch covers along with the operator state, and an **idempotent or transactional sink** so that replaying a batch does not duplicate output. If any one of these is missing, the guarantee drops to at-least-once.

**What lives in the checkpoint directory**

| Folder | Purpose |
| --- | --- |
| `offsets/` | Write-ahead log: the planned source offsets for batch N, written **before** the batch runs |
| `commits/` | Marker written **after** the sink successfully finishes batch N |
| `state/` | Versioned state store snapshots and deltas (HDFS-backed or RocksDB) for aggregations, dedup, joins |
| `sources/`, `metadata` | Query ID and source-specific metadata |

**Recovery flow**

1. On restart, Spark reads the latest entry in `offsets/` and checks `commits/`.
2. If batch N has an offsets entry but no commit, it **re-runs batch N with exactly the same offsets**, so the input is deterministic.
3. State is restored to version N−1, so aggregations do not double count.
4. The sink receives batch N again, and must handle that safely.

**Sink side**

- **File and Delta/Iceberg sinks:** the file sink writes a `_spark_metadata` log, and Delta records the `(queryId, batchId)` pair in its transaction log, so a replayed batch is skipped. This gives exactly-once.
- **`foreachBatch`:** you receive `batchId`. Make writes idempotent with `MERGE` on a business key, or use Delta's `txnAppId`/`txnVersion` options. A plain `append` inside `foreachBatch` is at-least-once.
- **Kafka sink:** at-least-once. Downstream consumers must deduplicate on a key, or you put a dedup stage after it.
- **JDBC / external APIs:** use upserts keyed on a natural ID, or store the last committed `batchId` in the same transaction as the data.

```python
def upsert(batch_df, batch_id):
    (DeltaTable.forName(spark, "silver.orders").alias("t")
       .merge(batch_df.alias("s"), "t.order_id = s.order_id")
       .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

stream.writeStream.foreachBatch(upsert) \
      .option("checkpointLocation", "s3://chk/orders_silver") \
      .start()
```

**Production gotchas**

- **The checkpoint is tied to the query plan.** Changing aggregation keys, adding stateful operators, or changing the number of shuffle partitions can make the checkpoint incompatible. Plan how you will migrate state, or reprocess from the source.
- **Never share a checkpoint** between two queries, and never delete it to "fix" a bug without understanding the replay consequences.
- **Source retention must exceed your recovery time.** If Kafka deletes offsets before a failed query restarts, data is lost (`failOnDataLoss`).
- **Non-deterministic logic** (calling `current_timestamp()` or random IDs per row) breaks the idea that a replay produces identical output.
- Put checkpoints on durable object storage, and monitor batch duration against the trigger interval.

### 6.42 How do stream-stream joins work, and why do they need watermarks and state bounds?

In a stream-stream join, either side's matching row can arrive at any time, so the engine must buffer past rows from **both** streams in state and match each new row against the buffered rows of the other side. Without a watermark and a time constraint, that buffer grows forever. Watermarks plus an event-time range condition tell the engine when a buffered row can no longer match anything, so its state can be dropped.

**Mechanics (Spark Structured Streaming)**

- Each micro-batch, new rows on the left are joined with all buffered right rows, and the reverse, then the new rows are added to state.
- State is a keyed store (often RocksDB) partitioned by join key.
- To clean up, Spark combines each side's watermark with the time-range condition to compute a cutoff per side, and evicts rows older than it.

```python
impressions = impr.withWatermark("impr_time", "2 hours")
clicks = clk.withWatermark("click_time", "3 hours")

impressions.join(
    clicks,
    expr("""
      click_ad_id = impr_ad_id AND
      click_time >= impr_time AND
      click_time <= impr_time + interval 1 hour
    """),
    "leftOuter")
```

Here an impression can only match a click within one hour, and clicks can be up to 3 hours late, so an impression can be evicted once the click watermark passes `impr_time + 1 hour`.

**Requirements by join type**

| Join type | Watermark + time constraint | Notes |
| --- | --- | --- |
| Inner | Optional but strongly recommended | Without them, state is unbounded |
| Left/right outer | Required on the non-preserved side (the right side of a left join), plus the time constraint; a watermark on the other side enables full state cleanup | NULL-padded results are emitted only after the watermark proves no match can come, so they are **delayed** |
| Full outer | Required on at least one side plus the time constraint; set both for full state cleanup | Same delayed emission on both sides |
| Left semi | Same as left outer | Supported since Spark 3.1 |

**Why outer joins are delayed.** The engine cannot emit "impression with no click" until it knows no click can still arrive. That happens only when the watermark passes the end of the match window. Expect output delayed by the lateness bound plus the join range.

**Flink's options**

- **Interval join:** similar to the above, bounded by a time range.
- **Temporal (versioned) join:** joins a stream to the version of a changelog table valid at the event's time, for example converting each order with the FX rate in effect at order time. State holds only versions of the table.
- **Lookup join:** queries an external store per event. There is no stream state, but it adds latency and external load.
- Regular unbounded joins in Flink SQL need a **state TTL** to stay bounded.

**Production concerns**

- **State size** grows with throughput multiplied by retention. Estimate it: 50k events/s × 3 hours × 200 bytes is about 100 GB of state. Use RocksDB, and size local disks and checkpoint storage accordingly.
- **Skewed join keys** concentrate state on one task.
- **Clock-skewed events** push the watermark forward and silently drop matches.
- Often the better design is not a stream-stream join at all: enrich against a compacted dimension topic or a lookup table (stream-table join), which keeps state small.

### 6.43 How does Kafka guarantee ordering and durability? Explain acks, min.insync.replicas, idempotent producers and transactions.

Kafka guarantees ordering **only within a partition**: records with the same key go to the same partition and are appended in order. Durability comes from replicating each partition to several brokers and acknowledging writes only once enough in-sync replicas have them. Idempotent producers remove duplicates caused by retries, and transactions make writes across partitions, plus consumer offset commits, atomic.

**Ordering**

- The partition is the unit of order. There is no global order across a topic.
- Use a key that represents the entity that needs ordering (`account_id`, `order_id`). Choose it carefully: a hot key becomes a hot partition.
- Increasing a topic's partition count changes the key-to-partition mapping, which breaks per-key order for in-flight data.
- Retries could reorder records in the past. With idempotence enabled, Kafka keeps order even with up to 5 in-flight requests per connection.

**Durability: replication factor, acks, ISR**

| Setting | Meaning | Production default |
| --- | --- | --- |
| `replication.factor` | Copies of each partition | 3 |
| `acks=0` | Fire and forget | Only for metrics you can lose |
| `acks=1` | Leader wrote it; lost if the leader dies before followers copy it | Rarely appropriate |
| `acks=all` | All current in-sync replicas (ISR) have it | Default since Kafka 3.0 |
| `min.insync.replicas` | Minimum ISR size for an `acks=all` write to succeed | 2 with RF=3 |
| `unclean.leader.election.enable` | Allow an out-of-sync replica to become leader | `false` (keep it that way) |

`acks=all` alone is not enough: if the ISR shrinks to just the leader, "all" means one copy. With RF=3 and `min.insync.replicas=2`, every acknowledged write exists on at least two brokers. You can lose one broker without losing data or availability. If you lose two, producers get `NotEnoughReplicas` errors instead of writes being silently under-replicated.

Kafka acknowledges after the write reaches the page cache, not after `fsync`. It relies on replication across failure domains (racks or availability zones, using `broker.rack`) rather than on disk flushes.

**Idempotent producer**

- Enabled by default since Kafka 3.0 (`enable.idempotence=true`).
- The broker tracks a producer ID and per-partition sequence number, and discards duplicates caused by retries. This gives exactly-once **writes per partition for a single producer session**, not end-to-end exactly-once.

**Transactions**

- Set `transactional.id`. The producer can then write to several partitions and commit consumer offsets (`sendOffsetsToTransaction`) atomically.
- Consumers with `isolation.level=read_committed` see only committed records. Aborted records are filtered out.
- This is how Kafka Streams and Flink's Kafka sink achieve exactly-once for consume-transform-produce pipelines.
- **Zombie fencing:** a restarted instance with the same `transactional.id` bumps the epoch, so an old instance that hung and came back cannot commit.

```properties
acks=all
enable.idempotence=true
transactional.id=payments-enricher-0
max.in.flight.requests.per.connection=5
```

**Limits:** transactions add latency (commit markers, coordinator round trips) and reduce throughput with very frequent commits. They also only cover Kafka. Writing to an external database still needs idempotent upserts or an outbox pattern.

### 6.44 What is Kafka log compaction and how does it differ from time-based retention? When would you use it?

Time-based (or size-based) retention deletes whole old log segments after a period, regardless of content. Log compaction keeps **at least the latest value for every key** and removes older values for the same key, so the topic behaves like a durable, replayable snapshot of current state. A record with a null value (a tombstone) deletes its key.

**Comparison**

| Aspect | Delete retention (`cleanup.policy=delete`) | Compaction (`cleanup.policy=compact`) |
| --- | --- | --- |
| What is kept | Everything newer than `retention.ms` / `retention.bytes` | Latest record per key (plus the recent uncompacted head) |
| Unit of deletion | Entire closed segments | Individual superseded records |
| Requires keys | No | Yes, every record needs a key |
| Topic size | Bounded by time or bytes | Bounded by number of distinct keys |
| Replay gives | Recent history of events | Full current state, with some recent history |
| Typical use | Clickstream, logs, events | Changelogs, CDC tables, configuration, reference data |

You can combine them with `cleanup.policy=compact,delete`: compact per key, and also drop anything older than the retention period.

**How compaction works**

- The log has a **tail** (already compacted) and a **head** (recent records). The log cleaner thread scans the head, builds a map of latest offset per key, and rewrites older segments, dropping superseded records.
- The active segment is never compacted. `segment.ms` and `segment.bytes` control when segments close and become eligible.
- `min.compaction.lag.ms` guarantees a record stays uncompacted for a minimum time, so consumers can see every update. `max.compaction.lag.ms` forces compaction within a bounded time, which is useful for meeting deletion deadlines.
- Tombstones are kept for `delete.retention.ms` (24 hours by default) so that slow consumers see the delete before it disappears.
- **Offsets never change** and order per key is preserved. Compaction only creates gaps in offsets.

**When to use it**

- **Changelogs for stateful stream processing.** Kafka Streams state stores and KTables are backed by compacted topics. Restoring state replays only the latest value per key.
- **CDC topics** representing a table (Debezium), where a new consumer can bootstrap the full table from offset 0.
- **Reference data for enrichment** (customers, products, FX rates), consumed as a table by stream jobs.
- **Kafka's own internal topics**, such as `__consumer_offsets`.

**Gotchas**

- Compaction is **not immediate or exact**. Consumers can still see several values for a key, so they must handle duplicates and treat the topic as upserts.
- Producing without keys to a compacted topic fails.
- For GDPR erasure, a tombstone deletes data only after compaction runs and `delete.retention.ms` passes. Tune `max.compaction.lag.ms` if you have a legal deadline.
- High-cardinality keys (for example a UUID per event) make compaction useless; the topic grows like a normal log.
- Cleaner threads use broker CPU and I/O. Monitor `max-dirty-ratio` and cleaner lag.

### 6.45 What causes consumer group rebalancing and consumer lag in Kafka, and how do you deal with them?

A rebalance is the group coordinator reassigning partitions among consumers. It happens when membership or subscriptions change: a consumer joins, leaves, crashes, misses heartbeats, or takes too long between `poll()` calls. Consumer lag is the gap between the latest offset in a partition and the group's committed offset. It grows when consumers process more slowly than producers write, which is often made worse by rebalances that pause consumption.

**Rebalance triggers**

- Scaling out or in, deployments, and pods restarting.
- **`session.timeout.ms` exceeded** (45 s by default): no heartbeat because of a crash, long GC pause, or network issue.
- **`max.poll.interval.ms` exceeded** (5 min by default): the processing loop took too long between polls (slow database writes, a large batch), so the consumer is considered stuck and is removed.
- Subscription changes, such as new topics matching a regex pattern or partitions added to the topic.

**Why rebalances hurt and how to make them cheap**

| Technique | What it does |
| --- | --- |
| New consumer group protocol (KIP-848, `group.protocol=consumer`, GA in Kafka 4.0) | Server-side, incremental assignment: only affected partitions move, with no group-wide stop-the-world |
| Cooperative sticky assignor (classic protocol) | Incremental rebalancing; consumers keep partitions that do not move |
| Static membership (`group.instance.id`) | A restarting pod with the same ID gets its partitions back with no rebalance, as long as it returns within the session timeout |
| Smaller `max.poll.records` or async processing | Keeps each poll loop well under `max.poll.interval.ms` |
| Commit offsets in `onPartitionsRevoked` | Avoids reprocessing after partitions move |

**Diagnosing lag**

Measure lag per partition (`kafka-consumer-groups --describe`, Burrow, or exported metrics) and in **time** ("how many minutes behind"), not only messages. Then work out which pattern you have:

- **Lag rising on all partitions:** not enough processing capacity overall. Add consumers (up to the partition count), batch writes to downstream systems, or speed up processing.
- **Lag on one or a few partitions:** a hot key or skewed partitioner, or a single slow consumer. Fix the key design, or split heavy keys.
- **Saw-tooth lag that drops to zero:** periodic rebalances or GC pauses. Fix the timeouts or use static membership.
- **Lag stuck at a constant value:** a poison message causing a retry loop. Send it to a dead-letter topic.

**Scaling limits**

- In a classic consumer group, parallelism cannot exceed the number of partitions. Extra consumers sit idle, so plan the partition count for peak throughput.
- For work-queue style processing where per-key order does not matter, **share groups** (Kafka queues, KIP-932, production-ready in Kafka 4.2) let many consumers read the same partition, with per-record acknowledgement.
- When a downstream system is the bottleneck, process records in parallel within a consumer (for example with a parallel consumer library that keeps per-key ordering) instead of adding partitions.

**Alerting:** alert on lag in time against the SLA, and on lag growth rate, rather than on a raw message count that means different things for different topics.

### 6.46 Compare Lambda and Kappa architectures. When does each make sense today?

Lambda architecture runs two pipelines: a **batch layer** that recomputes accurate results over all history, and a **speed layer** that produces approximate real-time results. A serving layer merges the two. Kappa architecture uses **one streaming pipeline** over a replayable log and handles reprocessing by replaying the log through a new version of the job. Today, most teams build something closer to Kappa on a lakehouse, while keeping batch reconciliation where correctness rules demand it.

**Comparison**

| Aspect | Lambda | Kappa |
| --- | --- | --- |
| Pipelines | Two (batch + streaming) | One (streaming) |
| Code paths | Two implementations of the same logic, which drift apart | One |
| Reprocessing | Rerun the batch layer | Replay the log into a new job version, then switch over |
| Correctness | Batch layer is the source of truth and corrects the speed layer | Depends on event-time handling, state, and exactly-once |
| Requirements | Durable raw storage, batch engine, stream engine | Long log retention (or tiered storage), a strong stream engine |
| Operational cost | High: two systems to run and reconcile | Lower, but stateful streaming is harder to operate |

**Why Lambda existed.** In the early 2010s, stream processors could not provide exactly-once results, handle late data, or keep large state reliably. Batch was the only way to get correct numbers, so a second, lossy speed layer was added for freshness.

**What changed**

- Engines such as Flink and Spark Structured Streaming handle event time, watermarks, large RocksDB state, and exactly-once sinks.
- Kafka tiered storage and lakehouse tables (Delta, Iceberg, Hudi) make long retention cheap, so replaying a year of history is realistic.
- Lakehouse tables can be both a streaming sink and a batch source. The same transformation code (Spark, or Flink SQL) can run in streaming and batch mode, which removes the "two code bases" problem.

**When Kappa makes sense**

- Event-driven products where data is naturally a stream: fraud scoring, personalization, IoT, operational metrics.
- Logic that is expressible as incremental, stateful computation.
- When you can keep raw events replayable for as long as you might need to reprocess.

**When Lambda-style elements are still justified**

- **Regulated or financial reporting**, where the authoritative number must come from a full, auditable batch recompute and the stream provides a provisional figure.
- **Very late or corrected data** (chargebacks arriving weeks later, upstream restatements) that is cheaper to fix with a nightly recompute than with huge streaming state.
- **Complex logic** such as large ML feature backfills or heavy multi-way joins, which are much cheaper in batch.
- Teams without the skills to operate stateful streaming systems.

**A pragmatic modern pattern**

Stream raw events into bronze lakehouse tables. Build silver and gold tables incrementally with the same code in streaming or triggered mode. Run a scheduled batch job that recomputes recent partitions (for example the last 7 days) to absorb late data and correct streaming results. This is "Kappa by default, batch reconciliation where needed". In an interview, frame the choice around freshness SLA, correctness requirements, team skills, and total cost, rather than the labels.

### 6.47 How do you choose between Spark Structured Streaming, Flink and Kafka Streams?

Choose by latency requirement, state complexity, where the data lives, and who will operate it. Spark Structured Streaming fits teams already on Spark or Databricks who need seconds-level latency and lakehouse integration. Flink fits low-latency, complex stateful processing at large scale. Kafka Streams fits Kafka-to-Kafka microservices that want a library, not a cluster.

**Comparison**

| Dimension | Spark Structured Streaming | Apache Flink | Kafka Streams |
| --- | --- | --- | --- |
| Model | Micro-batch by default; real-time mode (Spark 4.1+) for low-latency stateless pipelines | Record-at-a-time, continuous | Record-at-a-time library |
| Typical latency | About 100 ms to seconds (lower in real-time mode) | Milliseconds | Milliseconds |
| Deployment | Spark cluster (Databricks, EMR, Kubernetes) | Flink cluster (Kubernetes operator, managed services) | Embedded in your Java/Kotlin application; no separate cluster |
| Sources/sinks | Very broad, strong with Delta/Iceberg | Very broad, Kafka, CDC, lakehouse | Kafka in and Kafka out only |
| State | RocksDB state store, `transformWithState` for custom logic | Most mature: RocksDB, large state, timers, savepoints, state TTL | RocksDB stores backed by compacted changelog topics |
| Event time / windows | Watermarks, standard windows | Richest: custom triggers, late-data side outputs, CEP | Good windowing with grace periods |
| Exactly-once | Replayable source + checkpoint + idempotent sink | Checkpoint barriers + two-phase commit sinks | Kafka transactions (`processing.guarantee=exactly_once_v2`) |
| Batch + streaming | Same DataFrame API and engine | Unified API (batch execution mode, Flink SQL) | Streaming only |
| Languages | Python, SQL, Scala, Java | Java, SQL, Python (PyFlink) | Java/Scala |

**Choose Spark Structured Streaming when**

- The team and platform are already on Spark or Databricks, and you want one engine and skill set for batch and streaming.
- The workload is mainly ingestion and incremental ETL into lakehouse tables (bronze to silver to gold), where seconds of latency are fine.
- Python is the team's main language.

**Choose Flink when**

- You need sub-second latency with complex state: fraud detection, real-time features, CEP, large joins with long windows.
- State is very large (hundreds of GB to TB). Flink's incremental checkpoints, savepoints for upgrades, and disaggregated state in Flink 2.x matter here.
- You want streaming SQL for many teams, for example through a managed Flink service.

**Choose Kafka Streams when**

- Input and output are both Kafka, and the logic belongs inside a microservice (enrich, filter, aggregate per entity).
- You want to deploy as a normal application with your existing CI/CD, without running a processing cluster.
- Scale is moderate. Parallelism is bounded by input partitions, and state restoration from changelogs can be slow for very large stores (use standby replicas).

**Decision questions to ask**

1. What latency does the business actually need: milliseconds, seconds, or minutes? Minutes often means a scheduled incremental batch is enough.
2. How large and long-lived is the state, and how often will the logic change? Think about state migration on upgrades.
3. Where do the data come from and go to: Kafka only, or the lakehouse and databases?
4. What can the team operate on call? A managed service often matters more than engine features.
5. Do we need one engine for both batch and streaming?

A common mature setup uses more than one: Kafka Streams in domain microservices, Flink for low-latency shared stream processing, and Spark for lakehouse ETL and backfills.

---

← [Chapter 5. SQL, Databases and Data Modeling](05-sql-databases-and-data-modeling.md) · [Table of contents](../README.md) · [Chapter 7. Pipelines and Orchestration](07-pipelines-and-orchestration.md) →
