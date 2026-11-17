# Chapter 6. Distributed Processing and Streaming

_Part II — Data Engineering_

Apache Spark internals, partitioning and performance tuning, PySpark UDFs, Kafka, Flink, exactly-once stream processing, windowing and watermarks, and NoSQL stores.

This chapter contains **46 questions**.

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
- **6.21** [What is Apache Kafka, and how do topics, partitions and offsets work?](#621-what-is-apache-kafka-and-how-do-topics-partitions-and-offsets-work)
- **6.22** [What is a "Consumer Group" in Kafka?](#622-what-is-a-consumer-group-in-kafka)
- **6.23** [What is "Exactly-Once Semantics" (EOS), and how do you achieve it end to end?](#623-what-is-exactly-once-semantics-eos-and-how-do-you-achieve-it-end-to-end)
- **6.24** [When would you use Cassandra (or another NoSQL store) instead of a relational database like MySQL?](#624-when-would-you-use-cassandra-or-another-nosql-store-instead-of-a-relational-database-like-mysql)
- **6.25** [What is "Eventual Consistency"?](#625-what-is-eventual-consistency)
- **6.26** [What is a "Time-Series Database" (TSDB)?](#626-what-is-a-time-series-database-tsdb)
- **6.27** [What is "Windowing" in streaming (Tumbling, Sliding, Session)?](#627-what-is-windowing-in-streaming-tumbling-sliding-session)
- **6.28** [What is a "Dead Letter Topic" in Kafka?](#628-what-is-a-dead-letter-topic-in-kafka)
- **6.29** [What is "Schema Registry" in Kafka?](#629-what-is-schema-registry-in-kafka)
- **6.30** [What is "Stream-Table Duality"?](#630-what-is-stream-table-duality)
- **6.31** [What is Apache Flink?](#631-what-is-apache-flink)
- **6.32** [What is "Backpressure" in a streaming system?](#632-what-is-backpressure-in-a-streaming-system)
- **6.33** [What join strategies does Spark use (broadcast hash, shuffle hash, sort-merge, broadcast nested loop) and how does it choose between them?](#633-what-join-strategies-does-spark-use-broadcast-hash-shuffle-hash-sort-merge-broadcast-nested-loop-and-how-does-it-choose-between-them)
- **6.34** [What is Adaptive Query Execution (AQE) in Spark, and what problems does it solve?](#634-what-is-adaptive-query-execution-aqe-in-spark-and-what-problems-does-it-solve)
- **6.35** [How do you choose the number of shuffle partitions and size Spark executors (cores, memory, overhead)?](#635-how-do-you-choose-the-number-of-shuffle-partitions-and-size-spark-executors-cores-memory-overhead)
- **6.36** [How does Spark's unified memory model work (execution vs storage memory, off-heap, overhead)?](#636-how-does-sparks-unified-memory-model-work-execution-vs-storage-memory-off-heap-overhead)
- **6.37** [What is the difference between event time and processing time, and how do watermarks handle late data?](#637-what-is-the-difference-between-event-time-and-processing-time-and-how-do-watermarks-handle-late-data)
- **6.38** [How does Spark Structured Streaming achieve fault tolerance and end-to-end exactly-once (checkpoints, offsets, idempotent/transactional sinks)?](#638-how-does-spark-structured-streaming-achieve-fault-tolerance-and-end-to-end-exactly-once-checkpoints-offsets-idempotenttransactional-sinks)
- **6.39** [How do stream-stream joins work, and why do they need watermarks and state bounds?](#639-how-do-stream-stream-joins-work-and-why-do-they-need-watermarks-and-state-bounds)
- **6.40** [How does Kafka guarantee ordering and durability? Explain acks, min.insync.replicas, idempotent producers and transactions.](#640-how-does-kafka-guarantee-ordering-and-durability-explain-acks-mininsyncreplicas-idempotent-producers-and-transactions)
- **6.41** [What is Kafka log compaction and how does it differ from time-based retention? When would you use it?](#641-what-is-kafka-log-compaction-and-how-does-it-differ-from-time-based-retention-when-would-you-use-it)
- **6.42** [What causes consumer group rebalancing and consumer lag in Kafka, and how do you deal with them?](#642-what-causes-consumer-group-rebalancing-and-consumer-lag-in-kafka-and-how-do-you-deal-with-them)
- **6.43** [Compare Lambda and Kappa architectures. When does each make sense today?](#643-compare-lambda-and-kappa-architectures-when-does-each-make-sense-today)
- **6.44** [How do you choose between Spark Structured Streaming, Flink and Kafka Streams?](#644-how-do-you-choose-between-spark-structured-streaming-flink-and-kafka-streams)
- **6.45** [A Spark job that used to take 20 minutes now takes 3 hours. How do you diagnose and tune it?](#645-a-spark-job-that-used-to-take-20-minutes-now-takes-3-hours-how-do-you-diagnose-and-tune-it)
- **6.46** [How do Python UDFs, pandas UDFs and built-in functions compare in PySpark, and when should you use each?](#646-how-do-python-udfs-pandas-udfs-and-built-in-functions-compare-in-pyspark-and-when-should-you-use-each)

---

### 6.1 What is Apache Spark and why is it preferred over Hadoop MapReduce?

Spark is a general-purpose distributed compute engine that runs a whole job as one optimised DAG of stages, keeps intermediate data in memory where it can, and exposes high-level APIs (DataFrames, SQL, Structured Streaming, MLlib). MapReduce forces every job into a rigid map-then-reduce pair and writes results to HDFS between every pair, so multi-step and iterative work pays repeated disk and replication costs.

**Why Spark replaced MapReduce**

| Aspect | Hadoop MapReduce | Apache Spark |
| --- | --- | --- |
| Execution model | One map and one reduce phase per job; pipelines are chains of jobs | One DAG per action, split into stages at shuffle boundaries |
| Intermediate data | Written to HDFS (replicated) between jobs | Pipelined in memory inside a stage; shuffle files on local disk |
| Iterative work (ML, graph) | Re-reads input from disk every iteration | Cache the working set once, iterate in memory |
| Optimisation | None beyond what you hand-code | Catalyst optimiser, Tungsten code generation, Adaptive Query Execution |
| APIs | Java map/reduce functions | SQL, DataFrames in Python, Scala, Java, R |
| Workloads | Batch only | Batch, streaming, SQL, ML in one engine |
| Start-up overhead | A new JVM per task | Long-lived executors reuse JVMs |

**Nuances a senior candidate should add**

- "In-memory" is often oversold. Spark still writes shuffle data to local disk and spills when memory runs out. The bigger wins are the DAG scheduler (no HDFS write between steps), JVM reuse, and the optimiser working on a declarative plan.
- For a single scan-and-aggregate job over data far larger than memory, the gap is smaller (often 2 to 5 times rather than the marketing figure of 100 times).
- MapReduce's strength was robustness on cheap hardware with huge datasets. Spark matches that through lineage-based recomputation (see 6.8) rather than replicating every intermediate result.
- Today the comparison has moved on: the real alternatives to Spark are cloud warehouses (Snowflake, BigQuery), vectorised engines (Photon, DuckDB, Polars for single-node work) and Flink for streaming. Spark is chosen for large-scale transformation in Python or SQL, ML feature pipelines and lakehouse ETL.

**When not to use Spark:** data that fits on one machine (a few tens of GB) is usually faster and cheaper with DuckDB or Polars, because Spark's scheduling and shuffle overhead dominates at small scale.

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

An RDD (Resilient Distributed Dataset) is Spark's original low-level abstraction: an immutable, partitioned collection of arbitrary objects, with a lineage of the transformations that produced it. A DataFrame is a distributed table with a schema. Because Spark knows the columns and types and sees declarative operations, it can optimise the plan and store data in a compact binary format, which an RDD of opaque Python or Java objects does not allow.

**What each gives you**

| Aspect | RDD | DataFrame / Dataset |
| --- | --- | --- |
| Schema | None; Spark sees opaque objects | Named, typed columns |
| Optimisation | Executes your functions exactly as written | Catalyst rewrites the plan (pushdown, pruning, join selection) |
| Memory format | Serialised JVM or pickled Python objects | Tungsten binary rows and columnar batches, off-heap capable |
| Python performance | Every row is pickled between the JVM and Python workers | Built-in functions run entirely in the JVM |
| API | Functional (`map`, `reduceByKey`) | SQL-like (`select`, `groupBy`, `join`) |
| Type safety | Compile-time in Scala | Datasets give it in Scala/Java; PySpark DataFrames are untyped |

**Why DataFrames win in practice**

- **The optimiser can help you.** A filter written after a join is pushed below it; unused columns are never read from Parquet.
- **Python stays out of the hot path.** `df.groupBy("k").agg(F.sum("x"))` runs as generated JVM code. The RDD equivalent `rdd.map(...).reduceByKey(...)` sends every row through Python.
- **Spark Connect** (the client-server mode added in Spark 3.4 and used by serverless and shared-cluster platforms) supports only the DataFrame API, so RDD code does not run there.

```python
# RDD style: every row goes through Python
totals = (sc.textFile("s3://raw/sales.csv")
            .map(lambda line: line.split(","))
            .map(lambda f: (f[0], float(f[2])))
            .reduceByKey(lambda a, b: a + b))

# DataFrame style: optimised, columnar, runs in the JVM
totals = (spark.read.option("header", True).csv("s3://raw/sales.csv")
            .groupBy("store_id").agg(F.sum("amount").alias("total")))
```

**When RDDs are still justified:** fine-grained control over partitioning for unusual algorithms, processing truly unstructured records, or maintaining legacy code. Even then, a common pattern is to drop to `mapPartitions` briefly and return to DataFrames. In an interview, say that you default to DataFrames and SQL and treat RDD code as a smell to justify.

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

Data skew means a few partitions hold far more data than the rest, usually because a handful of key values (a null key, a default customer, a viral product) dominate a join or aggregation. A stage finishes only when its slowest task finishes, so one 40 GB partition among two thousand 100 MB partitions turns a ten-minute stage into a two-hour one, or kills it with an out-of-memory error.

**How to recognise it**

- In the Spark UI stage view, the task duration and shuffle read size have a max far above the median (for example max 30 minutes, median 20 seconds).
- The job sits at "199 of 200 tasks complete" for a long time.
- One executor spills heavily or dies while others are idle.
- Confirm the cause by counting rows per key: `df.groupBy("key").count().orderBy(F.desc("count")).show(20)`.

**Fixes, from cheapest to most invasive**

1. **Remove the junk key.** Null or placeholder keys (`-1`, `"unknown"`) are the most common cause. Filter them out, or handle them separately and union the result back.
2. **Let AQE split skewed joins.** With `spark.sql.adaptive.skewJoin.enabled=true` (on by default), Spark splits oversized sort-merge join partitions automatically (see 6.34). Lower the thresholds if needed.
3. **Broadcast the small side** so no shuffle by key happens at all (see 6.33).
4. **Salt the hot keys.** Add a random suffix to the skewed side and replicate the other side once per suffix, so one hot key spreads across N tasks.
5. **Two-phase aggregation** for skewed `groupBy`: aggregate on `(key, salt)` first, then on `key`. AQE does not fix aggregation skew.
6. **Isolate the heavy keys.** Process the top few keys with a broadcast join and the rest with a normal join, then union.

```python
from pyspark.sql import functions as F

N = 16  # salt buckets for hot keys
facts_s = facts.withColumn("salt", (F.rand() * N).cast("int"))
dim_s = dim.crossJoin(spark.range(N).withColumnRenamed("id", "salt"))
joined = facts_s.join(dim_s, ["customer_id", "salt"]).drop("salt")

# Two-phase aggregation for a skewed groupBy
partial = (facts_s.groupBy("customer_id", "salt")
                  .agg(F.sum("amount").alias("part")))
totals = partial.groupBy("customer_id").agg(F.sum("part").alias("amount"))
```

**Trade-offs and pitfalls**

- Salting multiplies the dimension side by N. Salt only the known hot keys if the dimension is large.
- Skew also appears **before** any shuffle: one 20 GB gzip file is read by one task because gzip is not splittable. Fix the input layout.
- Skewed writes (partitioning output by a skewed column) produce one giant file. Use `repartition` on a higher-cardinality expression or AQE's `REBALANCE` hint.
- Window functions partitioned by a hot key cannot be salted easily; consider pre-aggregation or a different partitioning of the problem.

### 6.8 How does Spark handle fault tolerance?

Spark's main mechanism is **lineage**: every DataFrame or RDD knows the deterministic chain of transformations that produced each partition, so a lost partition is recomputed from its parents instead of being replicated in advance. Around that sit task retries, stage resubmission for lost shuffle output, and checkpointing to cut long lineages. Streaming adds write-ahead offset logs and state checkpoints (see 6.38).

**Failure scenarios and what Spark does**

| Failure | What happens |
| --- | --- |
| A task throws or its executor is slow to respond | The task is retried on another executor, up to `spark.task.maxFailures` (4 by default) |
| An executor dies | Its running tasks are rescheduled. Cached partitions it held are recomputed from lineage when next needed |
| Shuffle output is lost (executor or node gone) | The reducer gets a `FetchFailedException`; Spark resubmits the parent map stage, but only the missing map tasks |
| A node is repeatedly bad | Excludes it (`spark.excludeOnFailure.enabled`) so tasks stop landing there |
| The driver dies | The application fails. Batch jobs are rerun by the orchestrator; streaming queries restart from the checkpoint. Cluster mode with `--supervise` (standalone) or a Kubernetes restart policy can restart the driver |

**Why this design works**

- Narrow dependencies (see 6.5) make recovery cheap: recompute one parent partition.
- Wide dependencies are expensive to recompute from scratch, so Spark keeps shuffle files on local disk (or in an external shuffle service or remote shuffle storage) and only reruns lost map outputs.
- Lineage assumes **deterministic** transformations. A UDF that calls an API, uses `rand()` without a seed, or reads "latest" data can produce different results on recomputation, which can create duplicates or inconsistencies after a retry. Spark marks some stages as indeterminate and reruns all their downstream tasks, but your own side effects are not protected.

**Practical implications**

- Long iterative lineages (graph algorithms, loops that build on previous DataFrames) make recovery and planning slow. Use `checkpoint()` or write intermediate results to a table (see 6.15).
- Losing executors on spot or preemptible nodes causes shuffle refetch storms. Use decommissioning (`spark.decommission.enabled`, which migrates shuffle and cached blocks), an external or remote shuffle service, and keep the driver on on-demand capacity.
- Writes to external systems are not rolled back by Spark. Output must be atomic (table formats such as Delta and Iceberg commit only on success) or idempotent, or a retried task can write twice.
- Fault tolerance is not the same as data correctness: Spark will happily recompute a wrong answer. Data quality checks still belong in the pipeline.

### 6.9 What is the difference between repartition() and coalesce()?

`repartition(n)` or `repartition(n, cols)` performs a full shuffle to produce exactly n partitions, evenly sized, optionally hash-partitioned by columns. It can increase or decrease the count. `coalesce(n)` only reduces the count, by merging existing partitions on the same executor without a shuffle. It is cheap, but the partitions can end up uneven, and it can quietly reduce the parallelism of the work that comes before it.

**Comparison**

| Aspect | `repartition` | `coalesce` |
| --- | --- | --- |
| Shuffle | Always (full exchange) | No (narrow dependency) |
| Direction | Up or down | Down only (asking for more is a no-op) |
| Partition sizes | Balanced (round-robin or hash) | Possibly skewed; merges whatever is local |
| By column | Yes: `repartition(200, "customer_id")` | No |
| Effect on upstream work | None; a stage boundary separates them | Can collapse the parallelism of the whole preceding stage |

**The trap with coalesce**

Because `coalesce` adds no stage boundary, Spark pipelines it with the narrow work before it. `df.filter(...).withColumn(...).coalesce(1).write...` runs the filter and all the transformations in a single task. A `repartition(1)` would keep the upstream stage parallel and only shuffle the final, smaller result. The rule: use `coalesce` after a big reduction in data (for example after a selective filter), and `repartition` when the upstream work is heavy.

**Common uses**

- **Controlling output file count and size.** Too many small files hurt readers (see 4.10). Aim for files of roughly 128 MB to 1 GB.
- **Repartitioning by the write partition column** (`repartition("event_date")`) so each task writes one file per date rather than one per date per task.
- **Pre-partitioning by a join key** before repeated joins, or by a key for `foreachPartition` writes to an external system.
- **Increasing parallelism** after reading a few large unsplittable files.

```python
# Even-sized output: about 1 file per 256 MB
n_files = max(1, int(estimated_bytes / (256 * 1024 * 1024)))
df.repartition(n_files).write.mode("overwrite").parquet(path)

# One file per date partition, files spread across tasks
df.repartition("event_date").write.partitionBy("event_date").parquet(path)

# Prefer letting Spark size it (AQE coalesces shuffle partitions)
df.hint("rebalance", "event_date").write.partitionBy("event_date").parquet(path)
```

**Senior point:** with AQE enabled, explicit `repartition` calls are needed less often. The `REBALANCE` hint asks AQE to produce evenly sized output partitions and split skewed ones, and table formats offer optimised writes and auto-compaction. Hard-coded `coalesce(1)` in production jobs is a common cause of slow writes and driver-like bottlenecks on a single executor.

### 6.10 What are Broadcast Variables?

A broadcast variable ships a read-only value (a lookup dictionary, a small model, a set of IDs) to each executor **once**, where it is cached and shared by all tasks on that executor. Without it, a value captured in a closure is serialised into every task, so a 50 MB dictionary used by 10,000 tasks becomes 500 GB of network traffic. Broadcast joins (see 6.33) use the same mechanism for a small table.

**How it works**

1. The driver serialises the value and splits it into chunks.
2. Executors fetch chunks with a BitTorrent-like protocol (from the driver and from each other), so the driver is not the only source.
3. Each executor deserialises it once and keeps it in its block manager (storage memory). Tasks read the local copy through `.value`.

```python
from pyspark.sql import functions as F, types as T

country_names = {"NP": "Nepal", "GB": "United Kingdom", "DE": "Germany"}
bc = spark.sparkContext.broadcast(country_names)

@F.udf(T.StringType())
def to_name(code):
    return bc.value.get(code, "Unknown")

df = df.withColumn("country", to_name("country_code"))

# Usually better: express the lookup as a broadcast join, no Python UDF needed
lookup = spark.createDataFrame(list(country_names.items()), ["country_code", "country"])
df = df.join(F.broadcast(lookup), "country_code", "left")
```

**When to use which**

- For tabular lookups in DataFrame code, prefer a **broadcast join**: it stays in the JVM and the optimiser understands it.
- Use an explicit broadcast variable for non-tabular objects used inside UDFs, `mapPartitions` or pandas UDFs: a trie, a tokenizer vocabulary, a scikit-learn model for batch scoring.

**Limits and pitfalls**

- The value must fit in driver memory (it is built there) and in each executor's memory, multiplied by the number of concurrent jobs using it. Above a few hundred MB, broadcasting becomes a liability.
- It is **read-only**. Changing the Python object on the driver after broadcasting does not update executors. To refresh it (for example in a long-running streaming job), `unpersist()` and broadcast a new one.
- Call `bc.unpersist()` or `bc.destroy()` when finished in long-running applications, or executor storage fills up.
- In PySpark, each Python worker process deserialises its own copy, so a large broadcast multiplies by the number of Python workers per executor. That memory sits outside the JVM heap and counts against `memoryOverhead`.

**Accumulators** are the write-only counterpart: tasks add to them and only the driver reads the total. They are useful for counters such as "malformed rows", but updates made inside transformations can be applied more than once if tasks are retried, so treat them as approximate diagnostics, not business metrics.

### 6.11 Explain Spark's "Catalyst Optimizer."

Catalyst is the query optimiser behind Spark SQL and DataFrames. It turns your code into a tree, rewrites that tree with rule-based and cost-based optimisations, picks a physical plan, and hands it to Tungsten for whole-stage code generation. That is why the same logic written in SQL, Python or Scala produces the same plan and the same performance.

**The four phases**

```text
DataFrame / SQL
   -> Unresolved logical plan   (parsed, names not yet checked)
   -> Analyzed logical plan     (columns and tables resolved against the catalog)
   -> Optimized logical plan    (rule-based rewrites)
   -> Physical plans            (strategies, e.g. which join) -> cost model picks one
   -> Whole-stage codegen (Tungsten) -> RDD of binary rows
```

**Typical rule-based optimisations**

- **Predicate pushdown:** filters move below joins and into the data source (see 6.17).
- **Column pruning:** only the columns used are read and carried through the plan.
- **Constant folding and simplification:** `col + (1 + 2)` becomes `col + 3`, always-false filters remove whole branches.
- **Combining operators:** adjacent filters and projections merge into one.
- **Join reordering** and selection of join strategy (with statistics, see 6.33).

**Cost-based optimisation (CBO)** uses table and column statistics (`ANALYZE TABLE ... COMPUTE STATISTICS FOR ALL COLUMNS`) to estimate cardinalities, reorder multi-way joins and choose broadcast versus sort-merge. Statistics are often missing or stale, which is one reason **Adaptive Query Execution** (see 6.34) re-optimises at runtime with real sizes.

**Reading the plan**

```python
df = (orders.filter("status = 'PAID'")
            .join(customers, "customer_id")
            .groupBy("country").agg(F.sum("amount")))
df.explain(mode="formatted")   # also "extended", "cost", "codegen"
```

Look for `PushedFilters` and `ReadSchema` in the scan (proof that pushdown and pruning happened), the join type, the number of `Exchange` nodes (shuffles), and `*(n)` markers showing which operators run inside generated code.

**What blocks Catalyst**

- **Python UDFs** are black boxes. Filters on their output cannot be pushed down, and data must be serialised to Python (see 6.46).
- **RDD operations** sit outside Catalyst entirely.
- **Very long plans** built in loops (hundreds of `withColumn` calls) make analysis slow; use one `select` with many expressions instead.
- **Non-deterministic expressions** limit reordering.

**Interview angle:** knowing Catalyst lets you explain why "rewrite the UDF with built-in functions" or "add statistics" fixes a slow job, and how to prove it with `explain()` rather than guessing.

### 6.12 What is "Speculative Execution" in Spark?

Speculative execution launches a duplicate copy of a task that is running much slower than its peers in the same stage. Whichever copy finishes first wins and the other is killed. It targets **stragglers caused by the machine** (a slow disk, a noisy neighbour, a degraded node), not slowness caused by the data.

**How it is configured**

| Setting | Meaning |
| --- | --- |
| `spark.speculation` | Turn it on (off by default in open-source Spark) |
| `spark.speculation.quantile` | Fraction of tasks in the stage that must finish before Spark looks for stragglers |
| `spark.speculation.multiplier` | A task counts as slow if it has run this many times longer than the median |
| `spark.speculation.minTaskRuntime` | Ignore tasks shorter than this, so tiny tasks are never duplicated |

Defaults for the thresholds have changed between Spark versions and platforms, so check them for your runtime rather than quoting numbers.

**When it helps**

- Large clusters on shared or cheap hardware where a few nodes are periodically slow.
- Stages with many uniform tasks, where one outlier is clearly abnormal.

**When it does not help, or hurts**

- **Data skew.** The duplicate processes the same oversized partition and is just as slow, while doubling resource use. Fix skew instead (see 6.7).
- **Non-idempotent side effects.** If a task writes to an external database, calls an API or sends messages, both copies may perform the side effect. Spark's own committers (and Delta or Iceberg commits) handle duplicate task attempts safely, but custom `foreachPartition` writers often do not.
- **Busy clusters.** Duplicates compete for slots with real work.
- **Very short tasks**, where the scheduling overhead outweighs the benefit.

**Senior view:** treat speculation as a mitigation for infrastructure noise, not a performance feature. If a job only meets its SLA with speculation on, investigate why the stragglers exist: look at the executor and host of the slow tasks in the Spark UI, and whether the same hosts are slow repeatedly (then exclude them or fix the node pool).

### 6.13 What is the difference between persist() and cache()?

Both mark a DataFrame or RDD to be kept after it is first computed, so later actions reuse it instead of recomputing the lineage. `cache()` is `persist()` with the default storage level. `persist(level)` lets you choose where and how the data is kept. Neither does anything until an action runs, because caching is lazy.

**Defaults differ by API**

- **DataFrame / Dataset `cache()`:** `MEMORY_AND_DISK` (partitions that do not fit in memory go to local disk). In recent PySpark versions the default is `MEMORY_AND_DISK_DESER`.
- **RDD `cache()`:** `MEMORY_ONLY` (partitions that do not fit are recomputed when needed).

**Storage levels**

| Level | Behaviour | Use when |
| --- | --- | --- |
| `MEMORY_ONLY` | Deserialised in memory; recompute what does not fit | Small, hot data and cheap recomputation |
| `MEMORY_AND_DISK` | Spill partitions that do not fit to local disk | Default choice for expensive DataFrames |
| `MEMORY_ONLY_SER` / `_SER` variants (Scala/Java) | Serialised, smaller, more CPU to read | Memory-constrained JVM jobs |
| `DISK_ONLY` | Local disk only | Large, expensive results reused a few times |
| `..._2` variants | Replicated on two executors | Expensive recomputation and unstable executors (spot nodes) |
| `OFF_HEAP` | Off-heap memory | With off-heap memory configured |

DataFrames are cached in Spark's compressed columnar in-memory format, so they are usually much smaller than the equivalent RDD of objects.

```python
from pyspark import StorageLevel

features = expensive_joins(df).persist(StorageLevel.MEMORY_AND_DISK)
features.count()                      # materialise once
train = features.filter("split = 'train'")
valid = features.filter("split = 'valid'")
# ... several actions reuse the cached data ...
features.unpersist()
```

**When caching helps:** a DataFrame is reused by two or more actions, and it is expensive to recompute (wide joins, reads from a slow source, iterative algorithms).

**When it hurts**

- Caching something used once just adds a write.
- Cached data competes for the unified memory pool with execution memory; execution evicts it, so heavy caching can cause spills and recomputation (see 6.36).
- Caching a huge DataFrame "just in case" is a common cause of memory pressure and slow jobs.
- Forgetting `unpersist()` in notebooks and long-running applications leaks storage memory.
- Caching does not cut lineage. If an executor dies, the lost partitions are recomputed from source. Use checkpointing for that (see 6.15).

**Alternative:** for results reused across jobs or by other teams, write them to a Delta or Iceberg table instead. It survives application restarts and is visible to other engines. Check the Storage tab in the Spark UI to confirm what is cached and what fraction actually fits in memory.

### 6.14 How do you handle "Out of Memory" (OOM) errors in Spark?

First find **where** the OOM happens (driver, executor JVM heap, or the container's off-heap overhead), because each has different causes and fixes. Then fix the cause, usually oversized partitions, skew, a large broadcast or collect, or too little overhead for Python. Adding memory is the last step, not the first.

**1. Locate it**

| Symptom | Where | Typical cause |
| --- | --- | --- |
| `OutOfMemoryError` in the driver log, or `spark.driver.maxResultSize` exceeded | Driver | `collect()`, `toPandas()`, a large broadcast being built, plans over millions of files |
| `java.lang.OutOfMemoryError: Java heap space` in a task | Executor heap | Huge or skewed partitions, `collect_list` on a hot key, wide rows, large user objects |
| Container killed, exit code 137 or 143, "exceeding memory limits" | Container overhead | PySpark workers, Arrow buffers, native libraries, off-heap not budgeted |
| `GC overhead limit exceeded`, very long GC times | Executor heap | Too many objects in memory; heap too large or too full |

**2. Fix the cause**

- **Oversized partitions:** increase `spark.sql.shuffle.partitions` or let AQE size them; for input, lower `spark.sql.files.maxPartitionBytes`. Target roughly 100 to 200 MB per partition (see 6.35).
- **Skew:** one task OOMs while others are fine. Use AQE skew joins, salting or two-phase aggregation (see 6.7).
- **Broadcasts:** a "small" table turned out to be large. Lower `autoBroadcastJoinThreshold`, remove the broadcast hint, refresh table statistics.
- **Collecting to the driver:** replace `collect()` with writes to storage; use `toLocalIterator()` or limits; set `maxResultSize` as a guardrail.
- **Unbounded aggregations:** `collect_list`, `collect_set` or pivot with many values build huge rows. Aggregate differently or cap them.
- **Too many concurrent tasks per executor:** each task gets a share of execution memory. Fewer cores per executor gives each task more memory.
- **PySpark memory:** raise `spark.executor.memoryOverhead` or set `spark.executor.pyspark.memory`; reduce the Arrow batch size (`spark.sql.execution.arrow.maxRecordsPerBatch`) for pandas UDFs.
- **Caching too much:** unpersist what you do not reuse.

**3. Then tune memory**

Increase executor memory or overhead in measured steps, consider off-heap memory for very large heaps, and check that the node type actually has the memory you request (see 6.36 for the memory layout).

```python
spark.conf.set("spark.sql.shuffle.partitions", "2000")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(50 * 1024 * 1024))

# Instead of: rows = big_df.collect()
big_df.write.mode("overwrite").parquet("s3://tmp/export/")
```

**Habits that prevent OOMs:** keep partitions sized by data, avoid driver-side operations on large data, check the Spark UI's stage page for spill and the max-versus-median task size, and test jobs on production-like volumes before release. An OOM that "goes away" with double the memory usually comes back when data grows by 50 percent.

### 6.15 What is a "Checkpoint" in Spark?

Checkpointing writes a DataFrame or RDD to reliable storage and **truncates its lineage**, so later stages and recovery start from the saved data rather than recomputing the whole chain. In Structured Streaming, "checkpoint" also means something different: the directory holding offsets, commits and operator state for fault-tolerant recovery (see 6.38).

**Batch checkpointing: two variants**

| Variant | Where data goes | Survives executor loss | Use |
| --- | --- | --- | --- |
| `df.checkpoint()` (reliable) | `spark.sparkContext.setCheckpointDir(...)`, e.g. HDFS or S3 | Yes | Long iterative lineages, recovery-critical intermediates |
| `df.localCheckpoint()` | Executor local storage | No | Cheaper lineage truncation when recomputation on failure is acceptable |

**Why you would use it**

- **Iterative algorithms.** A loop that updates a DataFrame 50 times builds a plan that grows each iteration. Planning time explodes and can hit stack overflows. Checkpoint every N iterations.
- **Expensive, non-deterministic, or flaky upstream work.** Freeze the result so a retry does not recompute it (and get a different answer).
- **Very long query plans** generated by code that chains hundreds of transformations.

```python
spark.sparkContext.setCheckpointDir("s3://my-bucket/spark-checkpoints/")

ranks = initial_ranks
for i in range(30):
    ranks = update(ranks, edges)
    if i % 5 == 4:
        ranks = ranks.checkpoint(eager=True)   # cut lineage every 5 iterations
```

**Checkpoint versus cache**

- `cache` / `persist` keeps the data but keeps the lineage. If cached blocks are lost, Spark recomputes them from the original source.
- `checkpoint` writes to durable storage and forgets the lineage. It costs a full write (and a job, if eager), so it is slower than caching.
- A common combination is `persist()` followed by `checkpoint()`, so the checkpoint job does not recompute the DataFrame twice.

**Practical alternative:** in production pipelines, writing an intermediate result to a Delta or Iceberg table often beats `checkpoint()`. You get the same lineage cut, plus an inspectable, versioned dataset that other jobs and reruns can use, and the orchestrator can restart from that step.

**Pitfalls:** checkpoint directories are not cleaned up automatically unless `spark.cleaner.referenceTracking.cleanCheckpoints` is set; checkpointing every iteration is wasteful; and confusing batch checkpoints with streaming checkpoint locations, which must never be deleted or shared between queries.

### 6.16 What is the purpose of the SparkSession?

`SparkSession` is the single entry point to Spark since 2.0. It creates or attaches to the application's driver context, reads and writes data, runs SQL, holds runtime configuration, and gives access to the catalog. It wraps the older `SparkContext` (low-level RDD and cluster access), `SQLContext` and `HiveContext`, which you should no longer create directly.

**What it gives you**

- **Data access:** `spark.read` / `df.write` for files, tables, JDBC, Kafka; `spark.readStream` for streaming.
- **SQL:** `spark.sql("...")`, temporary views, UDF registration.
- **Configuration:** `spark.conf.set("spark.sql.shuffle.partitions", "400")` for session-level SQL settings.
- **Catalog:** `spark.catalog.listTables()`, current database, caching tables; integration with Hive Metastore, Unity Catalog, Glue or Iceberg REST catalogs.
- **Lower-level access:** `spark.sparkContext` for broadcast variables, accumulators and RDDs.

```python
from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .appName("orders_daily")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog")
         .getOrCreate())

orders = spark.read.table("lake.silver.orders")
```

**Things that trip people up**

- **`getOrCreate()` returns the existing session** if one exists. Static settings such as executor memory or extra JARs passed to a builder after the context has started are ignored, which is a classic "my config does nothing" bug in notebooks.
- **Static versus runtime config.** Cluster resources (`spark.executor.memory`) must be set at submit time; SQL settings can change per session.
- **One SparkContext per JVM, many sessions.** `spark.newSession()` creates a session with isolated SQL config, temporary views and UDFs but the same cluster resources, which is useful for multi-tenant servers.
- **Spark Connect** (Spark 3.4+) separates the client from the driver: `SparkSession.builder.remote("sc://host:15002")` creates a thin client that sends unresolved plans over gRPC. It allows lightweight Python clients, better isolation and independent upgrades, but has no RDD API or direct `sparkContext` access. Databricks serverless and shared clusters use this model.

**In production**, build the session once in a small factory function, pass it into transformation functions (which keeps them testable with a local session in pytest), and keep environment-specific configuration outside the code.

### 6.17 What is "Predicate Pushdown"?

Predicate pushdown moves filter conditions as close to the data as possible, ideally into the storage layer or source system, so less data is read, transferred and processed. Combined with column pruning (projection pushdown) and partition pruning, it is often the single biggest performance lever for queries over Parquet, ORC, Delta, Iceberg or a JDBC source.

**Levels at which data is skipped**

| Level | Mechanism | What is skipped |
| --- | --- | --- |
| Partition pruning | Filter on a partition column (`event_date = '2026-11-01'`) | Whole directories or table partitions, without opening files |
| File skipping | Table-format statistics (Delta, Iceberg manifests hold min/max per column per file) | Entire files |
| Row-group skipping | Parquet footer min/max and optional Bloom filters per row group | Row groups inside a file |
| Page-level filtering | Parquet column indexes | Pages within a column chunk |
| Source pushdown | Filter sent as `WHERE` to JDBC databases, or to Kafka / external catalogs | Rows never leave the source |
| Dynamic partition pruning | Filter on a dimension is applied at runtime to the fact table's partitions | Fact partitions not matching the joined dimension rows |

**Verify it, do not assume it**

```python
df = (spark.read.parquet("s3://lake/events/")
        .filter((F.col("event_date") == "2026-11-01") & (F.col("country") == "NP"))
        .select("user_id", "event_type"))
df.explain(mode="formatted")
# Look for: PartitionFilters: [event_date = 2026-11-01]
#           PushedFilters: [IsNotNull(country), EqualTo(country,NP)]
#           ReadSchema: struct<user_id:string,event_type:string>
```

**What prevents pushdown**

- **Functions on the column:** `F.to_date("ts") == "2026-11-01"` or `upper(country) = 'NP'` usually cannot use statistics. Filter on the raw column or on a derived partition column.
- **Python UDFs** in the filter.
- **Type mismatches** that force a cast on the column side (comparing a string column to an integer).
- **Data layout.** Min/max statistics only help when values are clustered. If every file contains every country, nothing is skipped. That is why sorting, Z-ordering and liquid clustering exist (see 6.19).
- **Formats without statistics** such as CSV and JSON.
- **Filters applied after a non-deterministic or opaque step** (some UDFs and `mapPartitions`).

**Senior point:** pushdown only pays off when the physical layout matches the query patterns. Design partitioning and clustering around the most common filters, check `explain()` and the "files read" and "bytes read" metrics in the Spark UI, and remember that on object storage the cost saved is not just CPU but also GET requests and egress.

### 6.18 Explain the difference between reduceByKey and groupByKey.

Both are RDD operations that bring together values for the same key, but `reduceByKey` combines values **on the map side** before the shuffle (like a MapReduce combiner), so only one partial result per key per partition crosses the network. `groupByKey` shuffles every single value and builds the full list of values for each key in memory on the reducer, which is slow and a common cause of out-of-memory errors on skewed keys.

**What happens in each**

```text
reduceByKey(add):   partition 1: (a,1)(a,1)(b,1) -> (a,2)(b,1) --shuffle--> a: 2+3=5
                    partition 2: (a,1)(a,1)(a,1) -> (a,3)
groupByKey():       every (a,1) crosses the network -> a: [1,1,1,1,1] -> sum
```

| Aspect | `reduceByKey` | `groupByKey` |
| --- | --- | --- |
| Map-side combine | Yes | No |
| Shuffle volume | One record per key per partition | Every record |
| Memory on reducer | One running value per key | Entire value list per key |
| Function | Must be associative and commutative | Anything (you get the full list) |
| Skew risk | Low | High: a hot key's list must fit in one task's memory |

**Related operators**

- `aggregateByKey(zero, seqOp, combOp)` when the result type differs from the value type (for example computing a sum and a count together for an average).
- `combineByKey` is the general form underneath both.
- `groupByKey` is justified only when you genuinely need all values together and cannot express the logic incrementally (for example a per-key model fit), and even then keys must be bounded in size.

```python
pairs = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
counts = pairs.reduceByKey(lambda x, y: x + y)

# Average with aggregateByKey: (sum, count) per key
sum_count = pairs.aggregateByKey((0, 0),
                                 lambda acc, v: (acc[0] + v, acc[1] + 1),
                                 lambda a, b: (a[0] + b[0], a[1] + b[1]))
avg = sum_count.mapValues(lambda t: t[0] / t[1])
```

**The DataFrame equivalent**

In DataFrame code you do not choose: `df.groupBy("k").agg(F.sum("v"))` is planned as a partial `HashAggregate` before the exchange and a final `HashAggregate` after it, exactly like `reduceByKey`. The groupByKey trap reappears with `collect_list` / `collect_set`, `applyInPandas` grouped by a hot key, and `Dataset.groupByKey` with `mapGroups` in Scala, all of which materialise every value of a key in one place. Prefer built-in aggregations, and for per-group Python logic make sure group sizes are bounded.

### 6.19 What is "Z-Ordering"?

Z-ordering is a way of sorting data files by several columns at once using a space-filling curve (the Z-order or Morton curve), so rows with similar values in all of those columns end up in the same files. Per-file min/max statistics then become tight for each column, and queries filtering on any of the Z-ordered columns can skip most files. A plain sort helps the first sort column a lot and the others hardly at all.

**Why it is needed**

Data skipping (see 6.17) works only if the values in each file cover a narrow range. If you partition by date but filter by `customer_id`, every file contains customers from the whole range, so no file can be skipped. Sorting by `customer_id` fixes that one column. Z-ordering interleaves the bits of the column values so that locality is preserved, approximately, across two to four columns.

**Usage (Delta Lake)**

```python
from delta.tables import DeltaTable

(DeltaTable.forName(spark, "silver.events")
   .optimize()
   .where("event_date >= '2026-11-01'")
   .executeZOrderBy("customer_id", "product_id"))
```

Iceberg offers the same idea through `rewrite_data_files` with a `zorder(...)` sort order; Hudi supports Z-order and Hilbert-curve clustering.

**Limits and trade-offs**

- **Effectiveness drops with every extra column.** Two to three high-cardinality columns that appear in filters is the sweet spot; beyond four it approaches random.
- **It is a rewrite.** `OPTIMIZE ZORDER BY` rewrites the files it touches, which costs compute, and new data arrives unclustered until the next run.
- **Not incremental in classic Delta:** re-running it can rewrite already clustered data. Scope it to recent partitions.
- **Columns must have min/max statistics** collected (by default Delta collects stats on the first 32 columns).
- Do not Z-order by the partition column, and do not Z-order by low-cardinality columns that should be partitions instead.

**Liquid clustering and its successors**

Delta's **liquid clustering** (`CLUSTER BY`) replaces both Hive-style partitioning and Z-ordering for new tables: clustering keys can be changed without rewriting the table, clustering is incremental (only unclustered data is rewritten), and it handles skew and small tables better. On Databricks, automatic liquid clustering can even choose keys from query history. Iceberg uses sort orders plus compaction, and Snowflake and BigQuery have their own automatic clustering (see 5.48).

**How to choose:** partition only by a low-cardinality column that almost every query filters on (usually a date), and cluster within it by the two or three columns most used in selective filters and joins. Measure the result with "files pruned" or bytes read before and after, since the cost of clustering must be repaid by query savings.

### 6.20 What is the Hadoop "NameNode"?

The NameNode is the master process of HDFS. It holds the whole file system namespace (directories, files, permissions) and the mapping from each file to its blocks and from blocks to DataNodes, entirely in memory. DataNodes store the actual block replicas (128 MB blocks, replicated three times by default) and send heartbeats and block reports to it. Clients ask the NameNode where blocks are, then read and write directly to DataNodes.

**How it persists and recovers state**

- **FsImage:** a checkpoint of the namespace on disk.
- **EditLog:** a journal of every change since the last FsImage.
- Block locations are **not** persisted; they are rebuilt from DataNode block reports at start-up, which is why a large cluster's NameNode takes a while to leave safe mode.
- A **Secondary NameNode** (misleading name) only merges the EditLog into the FsImage. It is not a hot standby.

**High availability**

Classic HDFS had a single NameNode, a single point of failure. HDFS HA adds an active and one or more standby NameNodes sharing the EditLog through a quorum of JournalNodes, with ZooKeeper-based automatic failover. **Federation** splits the namespace across several independent NameNodes to scale metadata.

**Its limits explain the small file problem**

Every file, directory and block costs roughly 150 bytes of NameNode heap. 100 million small files means tens of GB of heap, long GC pauses and slow restarts. This is the original "small file problem" (see 4.10).

**Why it matters less today**

Most new platforms store data in cloud object storage (S3, ADLS, GCS) instead of HDFS:

| Aspect | HDFS | Object storage |
| --- | --- | --- |
| Metadata | Central NameNode, memory-bound | Distributed service, effectively unlimited objects |
| Compute and storage | Co-located; scaling one scales the other | Separated; scale and pay independently |
| Durability | 3 replicas you operate | Managed, very high durability, cross-zone |
| Rename and listing | Fast, atomic directory rename | No real directories; rename is copy plus delete; listing is slow and paid |
| Data locality | Tasks scheduled where blocks live | Network reads, offset by fast networks and caching |

The object-store weaknesses (no atomic rename, slow listing) are exactly what table formats like Delta and Iceberg solve with a transaction log and manifests (see 4.32). In an interview, explaining the NameNode is mostly about showing you understand **metadata as a scaling bottleneck**, which still applies to Hive Metastores, table-format metadata and catalogs today.

### 6.21 What is Apache Kafka, and how do topics, partitions and offsets work?

Kafka is a distributed, append-only commit log used as an event streaming platform. Producers write records to **topics**, Kafka stores them durably and in order per **partition** for a configured retention period, and any number of consumers read them independently by tracking **offsets**. Because reading does not delete data, the same stream can feed real-time services, a lakehouse and a replay of history.

**Core concepts**

| Concept | Meaning |
| --- | --- |
| Record | Key, value, headers and timestamp |
| Topic | A named stream of records, such as `orders.v1` |
| Partition | An ordered, immutable log; a topic is split into partitions for parallelism. Order is guaranteed only within a partition |
| Offset | The position of a record in a partition (0, 1, 2, ...). Consumers store the next offset to read, per partition, in `__consumer_offsets` |
| Broker | A server that stores partitions; a cluster has many |
| Replica and leader | Each partition is replicated (typically three times); one leader handles reads and writes, followers copy it |
| Controller | Manages metadata and leader election. KRaft (Kafka's own Raft quorum) replaced ZooKeeper, which was removed in Kafka 4.0 |

```text
Topic orders (3 partitions, key = order_id)
P0: [0][1][2][3][4][5] ->      consumer group "billing"  committed offset P0=4
P1: [0][1][2][3] ->            consumer group "lakehouse" committed offset P0=6
P2: [0][1][2][3][4] ->
```

**How it achieves high throughput**

- Sequential appends to segment files and reliance on the OS page cache.
- Batching and compression on the producer (`linger.ms`, `batch.size`, `compression.type=zstd` or `lz4`).
- Zero-copy transfer from page cache to network socket for consumers.
- Horizontal scaling by partition: more partitions allow more parallel consumers.

**Design decisions that matter**

- **Partition key:** choose the entity that needs ordering (`customer_id`, `order_id`). A hot key creates a hot partition; a null key spreads records but gives up per-key ordering.
- **Partition count:** sets the maximum parallelism of a consumer group. Size for peak throughput (per-partition throughput is typically a few to tens of MB per second), but avoid thousands of tiny partitions. Increasing the count later remaps keys to partitions and breaks per-key order for in-flight data.
- **Retention:** time or size based, or compaction for changelog topics (see 6.41). Tiered storage moves old segments to object storage so long retention becomes cheap.
- **Schemas:** use Avro or Protobuf with a schema registry (see 6.29), not ad hoc JSON.

**What Kafka is not:** a database for ad hoc queries, or a task queue with per-message acknowledgement in classic consumer groups (Kafka 4.x share groups add queue semantics). Durability and delivery guarantees are covered in 6.40, and consumer groups in 6.22 and 6.42.

### 6.22 What is a "Consumer Group" in Kafka?

A consumer group is a set of consumers sharing one `group.id` that divide a topic's partitions between them: each partition is read by exactly one member of the group at a time, so the group processes the topic in parallel while preserving per-partition order. Different groups are independent, each with its own committed offsets, so the same topic can feed billing, analytics and a lakehouse sink without interfering.

**How assignment works**

```text
Topic with 6 partitions
Group "fraud" (3 consumers):  C1 -> P0,P1   C2 -> P2,P3   C3 -> P4,P5
Group "lake"  (1 consumer):   C1 -> P0..P5          (independent offsets)
Group "fraud" with 8 consumers: 6 busy, 2 idle (parallelism capped by partitions)
```

- The **group coordinator** (a broker) tracks membership and offsets. Assignment is computed by an assignor (range, round-robin, cooperative sticky) or, with the new consumer group protocol in Kafka 4.0, by the broker itself.
- When members join or leave, partitions are **rebalanced** (causes and mitigation in 6.42).
- Committed offsets live in the internal compacted topic `__consumer_offsets`, keyed by group, topic and partition.

**Offset commits decide your delivery guarantee**

| Pattern | Guarantee | Risk |
| --- | --- | --- |
| Commit before processing | At-most-once | A crash after commit loses the records |
| Process, then commit | At-least-once (the usual default) | A crash before commit reprocesses records, so processing must be idempotent |
| Commit offsets in the same transaction as the output (Kafka transactions, or offsets stored in the sink's own transaction) | Exactly-once | More complexity; see 6.23 and 6.40 |

Auto-commit (`enable.auto.commit=true`) commits on a timer during `poll()`, which can commit records that have not been processed yet if processing is asynchronous. Production consumers usually disable it and commit explicitly after a batch is safely written.

```python
from confluent_kafka import Consumer

c = Consumer({"bootstrap.servers": "kafka:9092", "group.id": "orders-sink",
              "enable.auto.commit": False, "auto.offset.reset": "earliest"})
c.subscribe(["orders.v1"])
while True:
    msgs = c.consume(num_messages=500, timeout=1.0)
    if not msgs:
        continue
    rows = [parse(m.value()) for m in msgs if m.error() is None]
    upsert_into_warehouse(rows)          # idempotent write keyed on order_id
    c.commit(asynchronous=False)         # commit only after the write succeeded
```

**Design points**

- `auto.offset.reset` (`earliest` or `latest`) only applies when the group has no committed offset or the offset has expired. A new group with `latest` silently skips history.
- Plan partition counts for the maximum consumer parallelism you will need.
- Monitor **lag per partition** for every group, ideally in time units (see 6.42).
- Use one group per independent application. Sharing a group id across two different applications means each sees only part of the data, which is a common and confusing bug.
- Kafka 4.x **share groups** relax the one-consumer-per-partition rule for queue-style work where ordering does not matter.

### 6.23 What is "Exactly-Once Semantics" (EOS), and how do you achieve it end to end?

Exactly-once semantics means that each input event affects the final output **exactly once**, even when producers retry, consumers crash and jobs restart. Networks and machines fail, so messages are still delivered more than once inside the system. What engineers build is **effectively-once results**: at-least-once delivery combined with deduplication, idempotent writes or atomic commits of output together with input progress.

**The three delivery guarantees**

| Guarantee | Mechanism | Failure behaviour |
| --- | --- | --- |
| At-most-once | Commit progress before processing | Loss on crash |
| At-least-once | Process, then commit progress | Duplicates on crash; the default for most systems |
| Exactly-once (effectively) | Replayable source, plus output and progress committed atomically, or idempotent output | No loss, no visible duplicates |

**End to end means every hop**

```text
Producer -> Kafka -> Processor (state) -> Sink (lakehouse, DB, Kafka) -> Consumers
```

Each hop needs its own answer:

1. **Producer to Kafka:** idempotent producers (default since Kafka 3.0) remove duplicates from retries within a producer session. Application-level retries after a crash still create duplicates, so give every event a unique `event_id` at source.
2. **Inside the processor:** state and input offsets are checkpointed together (Flink checkpoint barriers, Spark Structured Streaming offset log plus state store) so a restart resumes from a consistent point (see 6.38).
3. **Processor to sink**, the hardest part:
   - **Transactional sinks:** Kafka transactions (consume-transform-produce with `read_committed` consumers), Flink two-phase-commit sinks, Delta or Iceberg commits that record the batch id.
   - **Idempotent sinks:** `MERGE` or upsert on a deterministic key, so replays overwrite rather than append.
   - **Store progress with the data:** write the last processed offset or batch id in the same database transaction as the results.
4. **Downstream consumers:** must read only committed data (`isolation.level=read_committed`, table snapshots) and must themselves be idempotent if they have side effects.

**Where the guarantee breaks**

- **External side effects** such as sending an email, charging a card, or calling an API. A replay repeats them unless the call carries an idempotency key that the receiver deduplicates (see 25.23).
- **Non-deterministic processing:** using processing time, random values or lookups against changing data means a replay produces different output, so "the same result" is not well defined.
- **Kafka transactions cover Kafka only.** Writing to Postgres inside a Kafka transaction is not atomic with it.
- **Source retention:** if the source data expires before recovery, nothing can be replayed.
- **Upstream duplicates** from producers that resend after a crash are new records to Kafka; only dedup on a business key removes them.

**Cost**

Transactions add latency (commits are tied to checkpoint intervals, so output becomes visible every few seconds to minutes) and reduce throughput with very frequent commits. Many systems choose at-least-once delivery plus idempotent `MERGE` sinks, which is simpler to operate and gives the same final tables.

**What interviewers look for:** that you say "exactly-once is a property of the whole pipeline, not of Kafka", name the mechanism at each hop, and point out side effects as the classic place it fails.

### 6.24 When would you use Cassandra (or another NoSQL store) instead of a relational database like MySQL?

Use Cassandra when the workload is dominated by very high write throughput and simple, known access patterns by key, at a scale or availability level a single relational primary cannot reach: multi-region active-active writes, millions of writes per second, time-ordered data per entity. Keep a relational database when you need ad hoc queries, joins, multi-row transactions and strong integrity constraints, which is most business applications. NoSQL families and their basics are covered in 5.1; this answer is about choosing.

**Cassandra's design in brief**

- **Leaderless, peer-to-peer ring.** Data is partitioned by a hash of the partition key and replicated to N nodes (often three per data centre). Any node can coordinate a request, so there is no failover step.
- **LSM-tree storage.** Writes go to a commit log and an in-memory memtable, then flush to immutable SSTables that are compacted later. Writes are cheap; reads may touch several SSTables.
- **Tunable consistency** per query (`ONE`, `QUORUM`, `LOCAL_QUORUM`, `ALL`), see 6.25.
- **Query-first data modelling.** You design one table per query pattern, denormalise, and choose the partition key and clustering columns to serve that query from one partition.

**Comparison**

| Need | Relational (MySQL, PostgreSQL) | Cassandra / wide-column |
| --- | --- | --- |
| Ad hoc queries, joins, aggregations | Strong | Poor (no joins; queries must hit a partition key) |
| Transactions across rows and tables | ACID | Lightweight transactions per partition only, and slow |
| Write scale | Vertical, plus sharding by hand | Linear horizontal scaling |
| Multi-region writes | Hard (conflicts, failover) | Built in, active-active |
| Schema changes and new query patterns | Easy | Often a new table and a backfill |
| Consistency | Strong by default | Tunable, eventual by default |

**Typical good fits:** IoT and telemetry ingestion, messaging inboxes, activity feeds, user profiles and session stores at internet scale, fraud or feature lookups by key, time series per device.

**Choosing among NoSQL families**

- **Key-value** (Redis, DynamoDB): caching, sessions, rate limits, online feature serving.
- **Document** (MongoDB, Firestore): flexible nested entities read as a whole.
- **Wide-column** (Cassandra, ScyllaDB, Bigtable, HBase): huge write volumes, time-ordered data per key.
- **Graph** (Neo4j): multi-hop relationship queries.
- **Search** (Elasticsearch, OpenSearch): full-text and faceted search.

**Pitfalls**

- **Modelling it like SQL.** Using `ALLOW FILTERING` or secondary indexes to answer new questions leads to cluster-wide scans.
- **Large or hot partitions** (all events for one big customer in a single partition) cause latency spikes; bucket by time (`device_id, day`).
- **Tombstones** from deletes and TTLs slow reads until compaction removes them.
- **Analytics:** do not run reports on Cassandra. Stream changes (CDC) to a lakehouse or warehouse for analytics.
- **Operational load:** repair, compaction tuning and capacity planning need expertise; managed services (Astra, Keyspaces) or DynamoDB often win on total cost.

**Senior framing:** "Start with PostgreSQL unless the access pattern and scale clearly demand otherwise. Modern relational options (read replicas, partitioning, distributed SQL such as Spanner, CockroachDB or Aurora) cover much more ground than they did ten years ago."

### 6.25 What is "Eventual Consistency"?

Eventual consistency guarantees only that if no new writes are made to an item, all replicas will **eventually** return the same value. In the meantime, different readers can see different, stale or out-of-order values. It is the price of staying available and fast when replicas are spread across nodes or regions, as explained by CAP (see 4.3) and, more usefully day to day, by PACELC: even without a partition you trade latency against consistency.

**Why systems choose it**

- Writes can be acknowledged by one or a few replicas, so latency stays low and the system stays writable when some nodes or a whole region are unreachable.
- Replication happens asynchronously in the background.

**The consistency spectrum**

| Model | Guarantee | Example |
| --- | --- | --- |
| Linearizable (strong) | Every read sees the latest committed write | Spanner, etcd, a single-primary database read from the primary |
| Sequential / causal | Operations respect causality ("reply after the post") | Some geo-distributed databases |
| Read-your-writes, monotonic reads | A client sees its own writes and never goes back in time | Session guarantees in Cosmos DB, sticky sessions |
| Eventual | Replicas converge at some point | DNS, Cassandra at `ONE`, DynamoDB default reads, async read replicas |

**Tunable consistency with quorums**

In Dynamo-style systems with N replicas, a write waits for W acknowledgements and a read queries R replicas. If R + W is greater than N, every read set overlaps every write set, so reads see the latest acknowledged write. With N = 3, `QUORUM` writes and reads (W = 2, R = 2) give this; `ONE` and `ONE` do not but are faster and more available.

**How replicas converge**

- **Last-write-wins** by timestamp: simple, but clock skew can silently drop a write.
- **Version vectors or vector clocks:** detect concurrent writes and keep both for the application to resolve.
- **CRDTs** (counters, sets that merge deterministically): used for collaborative editing and counters.
- **Anti-entropy:** read repair, hinted handoff and Merkle-tree repair fix divergent replicas.

**Where data engineers meet it**

- **Reading from replicas:** a CDC or extract job reading from an async read replica can miss recent rows; incremental loads need an overlap window (see 7.14).
- **Object storage:** S3 has been strongly consistent for reads after writes and listings since late 2020, but other stores and caches may not be; table formats avoid relying on listings at all.
- **Microservices and event-driven systems:** a downstream read model (search index, cache, warehouse) lags behind the source of truth.
- **Search and vector indexes** become visible after a refresh interval.

**Designing for it:** make consumers idempotent, include versions or timestamps so stale updates are rejected, show freshness ("updated 2 minutes ago") in user-facing data, and use strong consistency for the few operations that truly need it (payments, inventory reservation, uniqueness constraints).

### 6.26 What is a "Time-Series Database" (TSDB)?

A time-series database is optimised for data points indexed by time, usually with a series identity (metric name plus tags such as host or device) and one or more values: metrics, IoT sensor readings, prices, application telemetry. Workloads are append-heavy, queries are mostly over recent time ranges with aggregation (average per minute, percentiles, rates), and old data is downsampled or deleted. A TSDB builds its storage, compression and query engine around exactly those patterns.

**What makes them fast**

- **Time-partitioned storage:** data is stored in time chunks, so a query for "last 6 hours" touches only recent chunks, and retention means dropping whole chunks rather than deleting rows.
- **Columnar compression designed for time series:** delta-of-delta encoding for timestamps and XOR (Gorilla-style) compression for float values, often reaching 1 to 2 bytes per point.
- **Time-aware functions:** time bucketing, gap filling, interpolation, rates, moving windows, as-of joins.
- **Downsampling and continuous aggregates:** pre-computed rollups (1 minute to 1 hour to 1 day) with different retention for each resolution.

**Landscape**

| Type | Examples | Typical use |
| --- | --- | --- |
| Metrics and monitoring | Prometheus, VictoriaMetrics, Mimir, InfluxDB | Infrastructure and application metrics, alerting |
| SQL on time series | TimescaleDB (PostgreSQL extension), QuestDB | IoT, finance, when you also need relational joins |
| Real-time OLAP | ClickHouse, Apache Druid, Apache Pinot | Event analytics with high cardinality and sub-second queries |
| Financial tick data | kdb+ | Very high-frequency trading data |
| Lakehouse | Delta or Iceberg partitioned by time | Long-term history, ML training, cheap storage |

**The cardinality problem**

Cost in metrics TSDBs grows with the number of unique series. Putting a `user_id` or `request_id` in a Prometheus label creates millions of series, explodes memory and can take down the monitoring system. Keep high-cardinality identifiers in logs, traces or an OLAP store, not in metric labels.

**When to use one, and when not**

- Use a TSDB when you need fast recent-window queries and alerting on metrics or sensor data at high ingest rates.
- For analytical questions across years of data, joined with business entities, a lakehouse or warehouse is cheaper and more flexible. A common architecture keeps 15 to 90 days in the TSDB and everything else in Parquet tables.
- For ML on time series (see Chapter 32), the training data usually comes from the lakehouse copy.

**Interview pitfalls to mention:** late and out-of-order points (some TSDBs reject or penalise them), clock skew between devices, choosing retention and downsampling policies with the business, and cardinality limits.

### 6.27 What is "Windowing" in streaming (Tumbling, Sliding, Session)?

A stream is unbounded, so aggregations such as "count per user" need a finite scope. A window groups events by time (normally event time, see 6.37) into finite buckets that can be aggregated, emitted and then discarded. The three standard types are tumbling, sliding (hopping) and session windows; the right one depends on the business question.

**The window types**

| Type | Definition | Overlap | Example |
| --- | --- | --- | --- |
| Tumbling | Fixed size, back to back | None; each event in exactly one window | Revenue per 5-minute bucket |
| Sliding / hopping | Fixed size, advancing by a smaller slide | Yes; each event in size / slide windows | 10-minute average updated every minute |
| Session | Dynamic; closes after a gap of inactivity per key | None per key | User sessions with a 30-minute timeout |
| Global (with triggers) | One window per key, emitted on a custom trigger | n/a | Flink custom logic, counts every N events |

```text
time ->      0    5    10   15   20
Tumbling 5:  [----)[----)[----)[----)
Sliding 10/5:[---------)
                  [---------)
                       [---------)
Session gap 3 (one key):  x x  x         x x      (gap > 3 starts a new session)
                          [------]       [---]
```

**PySpark examples**

```python
from pyspark.sql import functions as F

ev = events.withWatermark("event_time", "10 minutes")

tumbling = ev.groupBy(F.window("event_time", "5 minutes"), "store_id").agg(F.sum("amount"))
sliding = ev.groupBy(F.window("event_time", "10 minutes", "1 minute"), "store_id").count()
sessions = (ev.groupBy(F.session_window("event_time", "30 minutes"), "user_id")
              .agg(F.count("*").alias("events"), F.min("event_time").alias("start")))
```

**Trade-offs and pitfalls**

- **Sliding windows multiply state and output.** A 1-hour window sliding every 1 minute puts each event into 60 windows. For long windows with fine slides, consider pre-aggregating into small tumbling buckets and summing them.
- **Session windows** can merge when a late event bridges two sessions, so earlier results may be revised. They also grow without limit for always-active keys (bots), so cap session length.
- **When results are emitted** matters as much as the window: append mode waits for the watermark to pass the window end, update mode emits partial results that change. Flink adds triggers, allowed lateness and side outputs for late data.
- **Event time versus processing time:** windows over processing time are simple but change when there is an outage or backlog; business metrics should use event time.
- **Time zones:** "daily" windows must specify the business time zone, otherwise days are UTC days.
- **Window alignment:** tumbling windows align to the epoch by default; use offsets if business periods start at other times.

**Interview angle:** say which window type answers which question, how lateness and watermarks interact with window closing, and how state size grows with window length, slide and key cardinality.

### 6.28 What is a "Dead Letter Topic" in Kafka?

A dead letter topic (or dead letter queue, DLQ) is where a consumer sends records it cannot process after a defined number of attempts, together with error metadata, so one bad record does not block the partition or crash the pipeline. Kafka itself has no built-in DLQ; it is a pattern implemented by the consumer, Kafka Connect (`errors.deadletterqueue.topic.name`), Kafka Streams handlers or frameworks such as Spring Kafka.

**Why it is needed**

Kafka delivers records in order per partition. If record 1,000 always throws (bad schema, invalid value, a bug), a consumer that retries forever stops the whole partition behind it, so lag grows without limit. This is the "poison pill" problem. A DLQ lets the consumer skip it safely and keep going.

**Classify errors before routing**

| Error type | Examples | Handling |
| --- | --- | --- |
| Transient | Timeouts, downstream 503, throttling | Retry with backoff, possibly via retry topics; do not DLQ immediately |
| Permanent, data | Deserialization failure, schema violation, missing required field | Send to DLQ straight away |
| Permanent, code | A bug that fails every record | Stop and alert; a DLQ would just fill with good data |

**What to put in a DLQ record**

- The original key and value bytes, unchanged.
- Headers: source topic, partition, offset, error class and message, stack trace summary, consumer application and version, attempt count, timestamp.

```python
def handle(msg):
    try:
        process(deserialize(msg.value()))
    except (SchemaError, ValueError) as exc:          # permanent data errors
        producer.produce("orders.v1.dlq", key=msg.key(), value=msg.value(),
                         headers={"src_topic": msg.topic(),
                                  "src_partition": str(msg.partition()),
                                  "src_offset": str(msg.offset()),
                                  "error": type(exc).__name__,
                                  "detail": str(exc)[:500]})
    # transient errors raise and are retried by the caller with backoff
```

**Retry topics**

For transient failures without blocking the main partition, use tiered retry topics (`orders.retry.1m`, `orders.retry.10m`) with delayed reprocessing, and the DLQ as the final stop. This gives up strict ordering for those records, so do not use it when per-key order is essential (instead, pause the partition or park all later records for the same key).

**Operating a DLQ**

- **Monitor and alert** on DLQ rate and age. A DLQ nobody reads is silent data loss.
- **Own the replay path:** a tool or job that fixes or transforms records and republishes them to the main topic, idempotently.
- **Retention and privacy:** DLQ records contain the same PII as the main topic; apply the same access controls and deletion policies.
- **Set a circuit breaker:** if more than a small fraction of records go to the DLQ, stop the consumer, because a systemic bug or bad deployment is more likely than bad data.

In batch pipelines the same idea appears as quarantine or "rejected rows" tables with a reason column.

### 6.29 What is "Schema Registry" in Kafka?

A schema registry is a service that stores versioned schemas (Avro, Protobuf or JSON Schema) for the records on Kafka topics and enforces **compatibility rules** when a schema changes. Producers register or look up a schema and send only a small schema ID with each record; consumers fetch the schema by ID to deserialise. It turns the topic into a contract, so producers cannot silently break consumers.

**How it works on the wire**

```text
Producer: serialise with schema v3 -> registry returns id 42
Record bytes: [magic byte 0][schema id 42 (4 bytes)][Avro/Protobuf payload]
Consumer: reads id 42 -> fetches and caches schema -> deserialises
```

Schemas are grouped into **subjects**, usually one per topic value (`orders-value`) under the default `TopicNameStrategy`. `RecordNameStrategy` allows several event types on one topic.

**Compatibility modes**

| Mode | Rule | Allowed change (Avro) | Upgrade order |
| --- | --- | --- | --- |
| BACKWARD (default) | New schema can read data written with the previous one | Delete fields, add fields with defaults | Upgrade consumers first |
| FORWARD | Old schema can read data written with the new one | Add fields, delete fields that had defaults | Upgrade producers first |
| FULL | Both directions | Add or remove only fields with defaults | Any order |
| `_TRANSITIVE` variants | Check against all previous versions, not only the last | Same rules over all history | Needed when consumers may replay old data |
| NONE | No checks | Anything | Avoid in production |

**Practical rules for evolving schemas**

- Add new fields as optional with defaults; never reuse a field name or Protobuf field number with a different meaning.
- Renames are breaking: add a new field, migrate consumers, deprecate the old one (Avro aliases help on the reader side).
- Changing a type (string to int) is breaking; create a new field or a new topic version (`orders.v2`).
- Use `BACKWARD_TRANSITIVE` or `FULL_TRANSITIVE` for topics that are replayed from the beginning or feed a lakehouse.

**Operational points**

- **Register schemas in CI**, not on first produce: disable `auto.register.schemas` in production producers and check compatibility in the pull request, so breaking changes fail the build rather than the pipeline.
- The registry is on the critical path for new schema IDs. Run it highly available; clients cache schemas, so short outages affect only new schemas.
- Options: Confluent Schema Registry, Apicurio, AWS Glue Schema Registry, Redpanda's built-in registry.
- Use **data contracts** on top (owners, semantic rules, PII tags) for richer guarantees (see 4.27).

**Why it matters to a data platform:** downstream lakehouse tables can evolve their schema automatically for compatible changes, and breaking changes are stopped at the producer, which is far cheaper than repairing corrupted tables (see 4.30 on schema drift).

### 6.30 What is "Stream-Table Duality"?

Stream-table duality says that a stream and a table are two views of the same data. A **table** is the state at a point in time; a **stream** is the sequence of changes that produced it. Replaying a changelog stream from the beginning rebuilds the table, and capturing every change to a table produces a stream. This idea underpins Kafka Streams, Flink's dynamic tables, CDC and lakehouse change feeds.

**Both directions**

```text
Stream (changelog)                   Table (latest per key)
(alice, 10)                          alice -> 25
(bob, 5)            -- aggregate ->  bob   -> 5
(alice, 25)
               <-- capture changes --
```

- **Stream to table:** aggregate or upsert by key. A Kafka Streams `KTable` or a Flink SQL table over an upsert-kafka source is a materialised view of a changelog.
- **Table to stream:** emit every insert, update and delete. Database CDC (Debezium), Delta Change Data Feed and Iceberg changelog scans all turn tables back into streams.

**Event streams versus changelog streams**

| Kind | Each record means | Example | Natural table view |
| --- | --- | --- | --- |
| Event stream (`KStream`) | Something happened; records are independent facts | Page views, payments | Aggregations (counts, sums) |
| Changelog stream (`KTable`) | The new value for a key replaces the old one | Customer profile updates, CDC rows | Latest value per key; compacted topic |

Getting this wrong causes real bugs: summing a changelog of account balances double counts, while treating payment events as upserts keeps only the last payment per customer.

**Why it matters in practice**

- **Fault-tolerant state:** Kafka Streams backs local state stores with compacted changelog topics, so a restarted instance rebuilds its table from the stream (see 6.41).
- **Stream-table joins:** enrich each event with the current (or time-versioned) row from a table, which keeps state small compared with stream-stream joins (see 6.39).
- **Incremental view maintenance:** streaming SQL engines keep query results up to date as the underlying tables change, which is what "materialised views over streams" means in Flink, Materialize or RisingWave.
- **Lakehouse pipelines:** bronze append-only change logs are the stream; silver tables built with `MERGE` are the table; change data feeds turn silver back into a stream for downstream incremental processing.

**Subtleties:** updates and retractions (a changed aggregate must retract its earlier result, which sinks must handle as upserts), ordering per key (only within a partition), and deletes represented as tombstones.

### 6.31 What is Apache Flink?

Apache Flink is a distributed engine for stateful computation over unbounded and bounded data streams. It processes records continuously one at a time, keeps large keyed state locally (RocksDB or, in Flink 2.x, disaggregated state on remote storage), and recovers with consistent distributed snapshots, which together give millisecond latency with exactly-once state consistency. It is the main choice for low-latency, complex, stateful streaming. For choosing among Flink, Spark and Kafka Streams see 6.44.

**Architecture**

```text
Client -> JobManager (scheduling, checkpoint coordination, recovery)
              |
     TaskManagers (task slots run operator subtasks; local state per key group)
Source -> keyBy -> window / process function -> Sink
```

- **JobManager** turns the program into a dataflow graph, schedules it and coordinates checkpoints.
- **TaskManagers** run the parallel operator instances in task slots and exchange data over network buffers.
- **Keyed state** is partitioned by key groups, so state redistributes when you change parallelism.

**How exactly-once state works: checkpoint barriers**

1. The JobManager injects a **barrier** into every source at a checkpoint.
2. Barriers flow with the data. When an operator has received the barrier from all its inputs, it snapshots its state asynchronously and forwards the barrier.
3. When all operators and sinks acknowledge, the checkpoint is complete. Two-phase-commit sinks (Kafka transactions, file sinks) commit their pending output at this point.
4. On failure, every operator restores state from the last completed checkpoint and sources rewind to the recorded offsets.

Unaligned checkpoints let barriers overtake buffered data, which keeps checkpoints fast under backpressure. **Savepoints** are user-triggered, portable snapshots used for upgrades, rescaling and migrating jobs.

**Key features**

- **Event time and watermarks** with idleness handling, allowed lateness and side outputs for late data (see 6.37).
- **Rich windowing and timers** in the `KeyedProcessFunction` API, plus **CEP** for pattern detection.
- **Flink SQL and the Table API** for streaming and batch with the same semantics, including changelog (retraction) handling, temporal joins and lookup joins.
- **Batch execution mode:** the same programs run on bounded data.
- **Connectors:** Kafka, CDC sources (Flink CDC), Iceberg, Paimon, Delta, Hudi, JDBC, Elasticsearch.

**Typical use cases:** fraud detection, real-time feature computation for ML, streaming ETL into a lakehouse, alerting and anomaly detection, real-time analytics feeding Pinot or Druid.

**What makes it hard to run**

- **State management:** state size, RocksDB tuning, checkpoint duration and storage cost. Use incremental checkpoints and state TTL.
- **Upgrades:** changing operator logic must stay compatible with saved state; assign stable operator UIDs.
- **Backpressure and skew** (see 6.32): one slow sink or hot key throttles the whole job.
- **Operations:** the Kubernetes operator, autoscaling and managed offerings (Amazon Managed Service for Apache Flink, Confluent, Ververica) reduce but do not remove this load. Python support (PyFlink) exists, but most production Flink is Java or SQL.

### 6.32 What is "Backpressure" in a streaming system?

Backpressure is the mechanism by which a slow downstream stage pushes back on faster upstream stages so they slow down, instead of the system buffering without limit, running out of memory, or dropping data. It is also the name of the symptom: a pipeline running at the speed of its slowest operator, with lag building at the source.

**How different systems handle it**

| System | Mechanism |
| --- | --- |
| Kafka (pull-based) | Consumers pull at their own pace; the broker simply retains data, so backpressure shows up as **consumer lag** rather than failure |
| Flink | Credit-based flow control between tasks: a sender only sends when the receiver has buffer credits. Backpressure propagates up to the sources, which then read Kafka more slowly |
| Spark Structured Streaming | Micro-batches: rate limits such as `maxOffsetsPerTrigger` and `maxFilesPerTrigger` bound each batch; when processing is slower than the trigger interval, batches simply run back to back and lag grows |
| Reactive libraries, gRPC, HTTP | Bounded queues, request-n semantics, 429 responses telling clients to back off |

**Diagnosing it**

- **Flink:** the web UI shows backpressured (busy waiting on output) versus busy (doing work) per subtask. The first operator that is busy but **not** backpressured is the bottleneck.
- **Spark:** batch duration consistently above the trigger interval, input rate above processing rate in the streaming query progress metrics.
- **Kafka:** lag growing per partition. If only some partitions lag, suspect a hot key or a skewed partition.

**Common root causes**

- A slow sink: a database doing row-by-row inserts, an API with rate limits, a lakehouse table with commit contention.
- Data skew sending most records to one subtask.
- Expensive per-record logic (Python UDFs, synchronous external lookups).
- Large state with slow checkpoints, or long GC pauses.
- Under-provisioned parallelism or too few Kafka partitions.

**Fixes**

1. **Fix the bottleneck operator:** batch writes, use async I/O for lookups (Flink `AsyncDataStream`), cache reference data, replace UDFs with built-ins.
2. **Scale out** the slow stage (more parallelism or partitions) and fix skew.
3. **Absorb bursts with a durable buffer.** Kafka between stages turns a temporary overload into temporary lag instead of a failure.
4. **Bound intake** with rate limits so batches stay predictable.
5. **Shed or degrade load** when freshness is less important than survival: sample, drop low-priority events, or route to a slower path.
6. **Autoscale on lag** (KEDA for consumers, Flink autoscaler), with limits so that scaling does not overload the downstream database.

**Design principle:** never use unbounded in-memory queues between stages. Every buffer should be bounded or durable, and every pipeline should have a lag-in-time SLO with alerting, because backpressure that is silently absorbed becomes a freshness incident hours later.

### 6.33 What join strategies does Spark use (broadcast hash, shuffle hash, sort-merge, broadcast nested loop) and how does it choose between them?

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

### 6.34 What is Adaptive Query Execution (AQE) in Spark, and what problems does it solve?

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

### 6.35 How do you choose the number of shuffle partitions and size Spark executors (cores, memory, overhead)?

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

### 6.36 How does Spark's unified memory model work (execution vs storage memory, off-heap, overhead)?

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

### 6.37 What is the difference between event time and processing time, and how do watermarks handle late data?

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

### 6.38 How does Spark Structured Streaming achieve fault tolerance and end-to-end exactly-once (checkpoints, offsets, idempotent/transactional sinks)?

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

### 6.39 How do stream-stream joins work, and why do they need watermarks and state bounds?

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

### 6.40 How does Kafka guarantee ordering and durability? Explain acks, min.insync.replicas, idempotent producers and transactions.

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

```text
acks=all
enable.idempotence=true
transactional.id=payments-enricher-0
max.in.flight.requests.per.connection=5
```

**Limits:** transactions add latency (commit markers, coordinator round trips) and reduce throughput with very frequent commits. They also only cover Kafka. Writing to an external database still needs idempotent upserts or an outbox pattern.

### 6.41 What is Kafka log compaction and how does it differ from time-based retention? When would you use it?

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

### 6.42 What causes consumer group rebalancing and consumer lag in Kafka, and how do you deal with them?

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

### 6.43 Compare Lambda and Kappa architectures. When does each make sense today?

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

### 6.44 How do you choose between Spark Structured Streaming, Flink and Kafka Streams?

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

### 6.45 A Spark job that used to take 20 minutes now takes 3 hours. How do you diagnose and tune it?

Work from evidence, not from configuration folklore. Find what changed, find the stage that dominates the runtime in the Spark UI, classify the bottleneck (skew, spill, shuffle volume, scan volume, bad join, too little parallelism, Python overhead, or infrastructure), fix the cause in the code or data layout first, and only then adjust configuration, one change at a time against a representative input.

**1. Establish what changed**

- Input volume or distribution (a new large customer, a backfill, a key that suddenly became hot).
- Code or dependency changes, a Spark or runtime upgrade, a cluster or instance type change.
- Table layout: compaction stopped running, so the job now reads 200,000 small files; statistics are stale.
- Contention: the job now shares a cluster or warehouse with other workloads.

Compare the event logs or Spark UI of a good and a bad run side by side. The difference usually points straight at the cause.

**2. Find the expensive stage**

In the Spark UI (or the History Server), sort stages by duration. For the worst one, look at:

| Metric | What it tells you |
| --- | --- |
| Task duration: max versus median | Skew if max is many times the median (see 6.7) |
| Shuffle read and write size | Shuffle volume; compare with the input size |
| Spill (memory and disk) | Partitions too large for execution memory |
| GC time | Above roughly 10 percent of task time means memory pressure or too many objects |
| Input size and number of files | Scan problems: no pruning, small files, unsplittable files |
| Number of tasks | Too few (idle cores) or tens of thousands of tiny tasks |
| SQL tab plan | Join strategy, missing pushdown, a surprise `BroadcastNestedLoopJoin` or `CartesianProduct` |

**3. Map the symptom to a fix**

| Bottleneck | Typical fix |
| --- | --- |
| Skewed tasks | Drop or isolate null keys, AQE skew join, salting, two-phase aggregation |
| Spill and OOM | More shuffle partitions or AQE with a high initial count; fewer cores per executor; avoid `collect_list` on hot keys (see 6.14) |
| Huge shuffle | Filter and project earlier, broadcast the small side, pre-aggregate, bucket tables that are joined repeatedly |
| Reading too much | Partition and cluster for the filter columns, compact small files, fix filters that block pushdown (see 6.17 and 6.19) |
| Wrong join strategy | Refresh statistics, broadcast hints, rely on AQE (see 6.33 and 6.34) |
| Python UDFs | Replace with built-in functions or vectorised pandas UDFs (see 6.46) |
| Too few tasks | Repartition after reading few large files; remove a stray `coalesce(1)` |
| Repeated recomputation | Persist a DataFrame used by several actions, or write it to a table |
| Slow writes | Too many output files per partition, or one giant file; use `REBALANCE` or `repartition` by the partition column |
| Infrastructure | Lost executors on spot nodes, slow object-store throughput, driver bottleneck from listing files |

**4. Then configuration**

- Enable AQE and set `spark.sql.adaptive.coalescePartitions.initialPartitionNum` high, so Spark coalesces down.
- Size `spark.sql.shuffle.partitions` and executors from data volume (see 6.35).
- Check `spark.sql.files.maxPartitionBytes` for input split size.
- Use a vectorised engine (Photon, Gluten with Velox or Comet) where available; they speed up scans, joins and aggregations without code changes.

**5. Verify and prevent regressions**

Change one thing at a time, measure wall time **and** cost (core-hours), and record the result. Then add guardrails: track job duration and input size per run, alert on step changes, keep table maintenance (compaction, statistics) scheduled, and test with production-like data volumes before releasing.

```python
# Quick checks from a notebook before opening the UI
df = spark.table("silver.orders")
print(df.rdd.getNumPartitions())
(df.groupBy("customer_id").count()
   .orderBy(F.desc("count")).show(10))          # hot keys?
result.explain(mode="formatted")                # join types, pushed filters, exchanges
spark.conf.get("spark.sql.adaptive.enabled")
```

**What interviewers look for:** a calm, systematic method; reading the Spark UI fluently; knowing that most slow jobs come from skew, small files, missing pruning or Python UDFs rather than from memory settings; and caring about cost as well as runtime.

### 6.46 How do Python UDFs, pandas UDFs and built-in functions compare in PySpark, and when should you use each?

Built-in Spark SQL functions run inside the JVM as generated code and are always the fastest choice. Classic Python UDFs move every row from the JVM to a Python worker process and back, one row at a time with pickling, and are opaque to the optimiser, so they are often ten or more times slower. Pandas UDFs (vectorised UDFs) use Apache Arrow to move data in columnar batches and run vectorised pandas or NumPy code, which closes most of that gap when Python is genuinely needed.

**How the execution differs**

```text
Built-in:     JVM executor -> generated code over binary rows (no Python at all)
Python UDF:   JVM -> pickle row by row -> Python worker -> pickle back -> JVM
pandas UDF:   JVM -> Arrow record batch (e.g. 10,000 rows) -> pandas Series -> Arrow -> JVM
```

**Comparison**

| Option | Speed | Optimiser can see inside | Use for |
| --- | --- | --- | --- |
| Built-in functions (`pyspark.sql.functions`, SQL expressions) | Fastest | Yes (pushdown, codegen) | Almost everything: string, date, JSON, arrays, maps, window functions, `transform` and `aggregate` on arrays |
| Classic Python UDF (`@udf`) | Slowest | No | Rare, small datasets, or logic that cannot be vectorised |
| Arrow-optimised Python UDF (`useArrow=True`, Spark 3.5+) | Better than classic | No | Row-wise Python logic where you want cheaper serialisation |
| Scalar pandas UDF (`@pandas_udf`, Series to Series) | Close to native for numeric work | No | Vectorised maths, scikit-learn or model scoring, library functions |
| Grouped map (`groupBy().applyInPandas`) | Good, but one whole group in memory | No | Per-group logic: fitting a model per store, custom resampling |
| `mapInPandas` / `mapInArrow` | Good | No | Batch inference over partitions, loading a model once per worker |

```python
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf

# Avoid: row-at-a-time Python UDF
@F.udf("double")
def with_tax_slow(amount):
    return amount * 1.2 if amount is not None else None

# Better: vectorised pandas UDF
@pandas_udf("double")
def with_tax(amount: pd.Series) -> pd.Series:
    return amount * 1.2

# Best: no UDF at all
df = df.withColumn("amount_gross", F.col("amount") * 1.2)

# Batch model scoring: load the model once per batch iterator, not per row
def score(batches):
    model = load_model("s3://models/churn/v7")
    for pdf in batches:
        pdf["churn_score"] = model.predict_proba(pdf[FEATURES])[:, 1]
        yield pdf

scored = features.mapInPandas(score, schema=features.schema.add("churn_score", "double"))
```

**Pitfalls**

- **Hidden UDF cost:** a filter on a UDF result cannot be pushed into the scan, and the UDF can be evaluated more than once when its output column is referenced several times.
- **Memory outside the JVM:** Python workers and Arrow buffers count against `memoryOverhead`. Tune `spark.sql.execution.arrow.maxRecordsPerBatch` and overhead together (see 6.36).
- **`applyInPandas` loads each group entirely into one Python process,** so a hot key causes an OOM. Bound group sizes or pre-aggregate.
- **Type and null handling:** pandas converts integer columns with nulls to floats, and timestamps are converted to the session time zone. Declare the return schema precisely and test edge cases.
- **Model loading per call:** load models once per partition or iterator, and broadcast small models (see 6.10).
- **Version alignment:** pandas, PyArrow and Python versions on the driver and executors must match what the Spark version supports.

**Rule of thumb for code review:** every Python UDF in a production job needs a reason. First look for a built-in function or higher-order array function; then a pandas UDF; a row-wise Python UDF is the last resort.

---

← [Chapter 5. SQL, Databases and Data Modeling](05-sql-databases-and-data-modeling.md) · [Table of contents](../README.md) · [Chapter 7. Pipelines and Orchestration](07-pipelines-and-orchestration.md) →
