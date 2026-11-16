# Chapter 7. Pipelines and Orchestration

_Part II — Data Engineering_

ETL and ELT design, orchestration with Airflow, idempotency, backfills and building reliable pipelines for analytics and ML.

This chapter contains **29 questions**.

## Contents

- **7.1** [What is the difference between ETL and ELT?](#71-what-is-the-difference-between-etl-and-elt)
- **7.2** [Explain the concept of a DAG in Apache Airflow.](#72-explain-the-concept-of-a-dag-in-apache-airflow)
- **7.3** [How do you handle pipeline failures in production?](#73-how-do-you-handle-pipeline-failures-in-production)
- **7.4** [What are Airflow "Hooks" and "Operators"?](#74-what-are-airflow-hooks-and-operators)
- **7.5** [What is "Backfilling" in Airflow?](#75-what-is-backfilling-in-airflow)
- **7.6** [How do you pass data between Airflow tasks?](#76-how-do-you-pass-data-between-airflow-tasks)
- **7.7** [What is "Functional Programming" and why is it used in Data Engineering?](#77-what-is-functional-programming-and-why-is-it-used-in-data-engineering)
- **7.8** [Explain the Python yield keyword.](#78-explain-the-python-yield-keyword)
- **7.9** [What is pandas and when would you NOT use it?](#79-what-is-pandas-and-when-would-you-not-use-it)
- **7.10** [What is a "Decorator" in Python?](#710-what-is-a-decorator-in-python)
- **7.11** [How do you handle API Rate Limiting in a Python script?](#711-how-do-you-handle-api-rate-limiting-in-a-python-script)
- **7.12** [What is "Schema Evolution"?](#712-what-is-schema-evolution)
- **7.13** [What is the purpose of a "Staging Area"?](#713-what-is-the-purpose-of-a-staging-area)
- **7.14** [What is "Change Data Capture" (CDC)?](#714-what-is-change-data-capture-cdc)
- **7.15** [What is a "SLA" in data engineering?](#715-what-is-a-sla-in-data-engineering)
- **7.16** [What is Airflow?](#716-what-is-airflow)
- **7.17** [What issues does Airflow resolve?](#717-what-issues-does-airflow-resolve)
- **7.18** [Explain how a workflow is designed in Airflow?](#718-explain-how-a-workflow-is-designed-in-airflow)
- **7.19** [How do you build reliable data pipelines for training/inference, and what makes data "clean and reliable" in this context?](#719-how-do-you-build-reliable-data-pipelines-for-traininginference-and-what-makes-data-clean-and-reliable-in-this-context)
- **7.20** [How do you design an idempotent incremental load?](#720-how-do-you-design-an-idempotent-incremental-load)
- **7.21** [How do you handle late-arriving facts and late-arriving dimensions?](#721-how-do-you-handle-late-arriving-facts-and-late-arriving-dimensions)
- **7.22** [What is the Write-Audit-Publish pattern and how do you implement it?](#722-what-is-the-write-audit-publish-pattern-and-how-do-you-implement-it)
- **7.23** [What are Airflow executors, and how do you choose one?](#723-what-are-airflow-executors-and-how-do-you-choose-one)
- **7.24** [What is the difference between sensors and deferrable operators in Airflow?](#724-what-is-the-difference-between-sensors-and-deferrable-operators-in-airflow)
- **7.25** [What are the most important Airflow best practices for production DAGs?](#725-what-are-the-most-important-airflow-best-practices-for-production-dags)
- **7.26** [Compare task-based orchestration (Airflow) with asset-based orchestration (Dagster, Airflow assets).](#726-compare-task-based-orchestration-airflow-with-asset-based-orchestration-dagster-airflow-assets)
- **7.27** [How do dbt incremental models and snapshots work, and what are their pitfalls?](#727-how-do-dbt-incremental-models-and-snapshots-work-and-what-are-their-pitfalls)
- **7.28** [How do you test data pipelines and set up CI/CD for them?](#728-how-do-you-test-data-pipelines-and-set-up-cicd-for-them)
- **7.29** [How would you build a log-based CDC pipeline from an OLTP database into a lakehouse, and what is the transactional outbox pattern?](#729-how-would-you-build-a-log-based-cdc-pipeline-from-an-oltp-database-into-a-lakehouse-and-what-is-the-transactional-outbox-pattern)

---

### 7.1 What is the difference between ETL and ELT?

**ETL (Extract, Transform, Load)** and **ELT (Extract, Load, Transform)** describe the order in which data moves through a pipeline, and that ordering has real consequences.

**ETL**

1. Extract data from source systems
2. Transform it (clean, join, aggregate, reshape) on a separate processing engine
3. Load the finished, transformed data into the target system (usually a data warehouse)

The transformation happens _before_ the data lands in its final home. This was the standard approach when warehouses were expensive and had limited compute, so you did the heavy lifting elsewhere first.

**ELT**

1. Extract data from source systems
2. Load the raw data directly into the target system
3. Transform it _inside_ the target system using its own compute

This became popular once cloud data warehouses (like Snowflake, BigQuery, Redshift) got cheap and powerful enough to handle transformation workloads themselves.

**Key differences:**

|  | ETL | ELT |
| --- | --- | --- |
| Where transformation happens | External engine | Inside the target warehouse |
| Raw data stored? | Usually not | Yes, raw data is kept |
| Speed to load | Slower (transform first) | Faster (load first, transform later) |
| Flexibility | Rigid — must know schema/needs upfront | Flexible — can re-transform raw data anytime |
| Infrastructure cost | Separate transformation servers | Relies on warehouse compute (can get pricey) |
| Common tools | Informatica, Talend, SSIS | dbt, Fivetran + warehouse-native SQL |

**In short:** ETL transforms data _before_ loading it, which is more controlled but less flexible. ELT loads raw data first and transforms it _inside_ the destination, which is more flexible and scales better with modern cloud warehouses, but means you're storing and processing raw (sometimes messy) data at scale.

### 7.2 Explain the concept of a DAG in Apache Airflow.

**A Directed Acyclic Graph (DAG) is a collection of tasks with defined dependencies. "Directed" means there is a clear flow, and "Acyclic" means tasks cannot loop back to themselves.**

### 7.3 How do you handle pipeline failures in production?

**I implement automated alerting (Slack/Email), configure retries with exponential backoff, and use Dead Letter Queues to isolate bad data without stopping the entire pipeline.**

### 7.4 What are Airflow "Hooks" and "Operators"?

**Operators define a single task (like running a Python script). Hooks are interfaces that handle the connection logic to external platforms like Snowflake or Postgres.**

### 7.5 What is "Backfilling" in Airflow?

**The process of running a pipeline for historical dates. This is used when a new pipeline is deployed and needs to process data from the past to populate a warehouse.**

### 7.6 How do you pass data between Airflow tasks?

**Small amounts of metadata are passed using XComs. For large datasets, the first task writes the data to S3, and the second task reads from that S3 path.**

### 7.7 What is "Functional Programming" and why is it used in Data Engineering?

**Functional programming treats data as immutable and uses functions like `map` and `filter`. This is ideal for distributed systems because it prevents "side effects" when code runs on multiple nodes.**

### 7.8 Explain the Python yield keyword.

**`yield` creates a generator, which returns one item at a time instead of loading an entire list into memory. This is essential for processing multi-gigabyte files efficiently.**

### 7.9 What is pandas and when would you NOT use it?

**Pandas is a library for data manipulation on a single machine. You should not use it for "Big Data" that exceeds a single machine's RAM; for that, use PySpark or Dask.**

### 7.10 What is a "Decorator" in Python?

**A decorator is a function that wraps another function to extend its behavior (like adding logging or timing) without modifying the original function's code.**

### 7.11 How do you handle API Rate Limiting in a Python script?

**I use `time.sleep()` combined with "try-except" blocks to catch `429 Too Many Requests` errors, implementing an exponential backoff strategy for retries.**

### 7.12 What is "Schema Evolution"?

**The ability of a data format (like Avro or Parquet) to handle changes in structure, such as adding or renaming columns, over time without breaking the pipeline.**

### 7.13 What is the purpose of a "Staging Area"?

**A temporary storage zone where data is cleaned and validated before being loaded into the final production warehouse to prevent corrupting the "source of truth."**

### 7.14 What is "Change Data Capture" (CDC)?

**Change Data Capture (CDC)** is a technique for identifying and capturing changes (inserts, updates, deletes) made to data in a source system, so those changes can be propagated elsewhere — typically to a data warehouse, another database, or a downstream application — without having to reprocess the entire dataset every time.

**Why it matters:** Instead of re-extracting a whole table every hour (which is slow and wasteful), CDC lets you capture just the _delta_ — what actually changed — and stream or batch only that.

**Common ways CDC is implemented:**

1. **Log-based CDC** — Reads the database's transaction log (e.g., MySQL binlog, PostgreSQL WAL) to detect changes. This is the most efficient and least invasive method since it doesn't add load to the source database with extra queries.
2. **Trigger-based CDC** — Database triggers fire on insert/update/delete and write the change to a separate tracking table.
3. **Timestamp/column-based CDC** — Relies on a "last modified" column and periodically queries for rows changed since the last check. Simple, but can miss deletes and adds query load.
4. **Snapshot diffing** — Compares full snapshots of data over time to infer what changed. Least efficient, used when nothing else is available.

**Common use cases:**

- Keeping a data warehouse in near real-time sync with an operational database
- Feeding event streams (e.g., into Kafka) for real-time analytics
- Data replication between databases
- Powering microservices with up-to-date data without tight coupling to the source database

**Popular tools:** Debezium (log-based, open source), Fivetran, AWS DMS, Qlik Replicate, Striim.

**How it connects to ETL/ELT:** CDC is often the _extraction_ mechanism feeding an ETL or ELT pipeline — instead of doing a full extract each run, the pipeline just picks up the changes CDC has captured.

### 7.15 What is a "SLA" in data engineering?

**A Service Level Agreement, which in data engineering usually refers to "Data Freshness", the guaranteed time within which data must be available in the dashboard.**

### 7.16 What is Airflow?

Apache Airflow is an open-source workflow management platform. It began in October 2014 at Airbnb as a solution for managing the company's increasingly complex workflows. Airbnb's creation of Airflow enabled them to programmatically author, schedule, and monitor their workflows via the built-in Airflow user interface. Airflow is a data transformation pipeline ETL (Extract, Transform, Load) workflow orchestration tool.

### 7.17 What issues does Airflow resolve?

Crons are an old technique of task scheduling. Scalable Cron requires external assistance to log, track, and manage tasks. The Airflow UI is used to track and monitor the workflow's execution. Creating and maintaining a relationship between tasks in cron is a challenge, whereas it is as simple as writing Python code in Airflow. Cron jobs are not reproducible until they are configured externally. Airflow maintains an audit trail of all tasks completed.

### 7.18 Explain how a workflow is designed in Airflow?

A directed acyclic graph (DAG) is used to design an Airflow workflow. That is to say, when creating a workflow, consider how it can be divided into tasks that can be completed independently. The tasks can then be combined into a graph to form a logical whole. The overall logic of your workflow is based on the shape of the graph. An Airflow DAG can have multiple branches, and you can choose which ones to follow and which to skip during workflow execution. Airflow Pipeline DAG Airflow could be completely stopped, and able to run workflows would then resume through restarting the last unfinished task. It is important to remember that airflow operators can be run more than once when designing airflow operators. Each task should be idempotent, or capable of being performed multiple times without causing unintended consequences.

### 7.19 How do you build reliable data pipelines for training/inference, and what makes data "clean and reliable" in this context?

Validate schema and types at ingestion (Pydantic again is useful here), handle missing/malformed records explicitly rather than silently dropping them, version datasets so experiments are reproducible, and separate raw/staging/curated layers so transformations are traceable and reversible. For inference pipelines specifically, ensure feature/preprocessing parity between training and serving (a common source of production bugs — "training/serving skew").

### 7.20 How do you design an idempotent incremental load?

An incremental load processes only new or changed data rather than the whole source. It is idempotent if running it once, twice or ten times for the same window leaves the target in the same state. To get there, three choices have to fit together: how you pick the rows (the window), how you write them (overwrite or merge), and how far back you reach to catch corrections (the reprocessing window).

**Full vs incremental**

| | Full load | Incremental load |
| --- | --- | --- |
| Logic | Truncate and reload, or swap in a new table | Process only a window or a delta |
| Idempotency | Trivially idempotent | Must be designed in |
| Cost | Grows with table size | Grows with change volume |
| Handles deletes | Yes, automatically | Only with CDC or soft-delete flags |
| Use when | Small dimensions, sources with no change tracking | Large facts, frequent runs |

**Choosing the window**

- **High-watermark column.** Store the maximum `updated_at` (or a monotonic ID or log sequence number) that was loaded successfully, then extract `WHERE updated_at > :last_watermark`. Commit the new watermark only after the load has been written, in the same transaction or the step right after it. Pitfalls: long-running source transactions commit rows with timestamps older than the watermark you already saved; clock skew between replicas causes the same problem; and `updated_at` columns maintained by application code are sometimes not updated at all. To guard against all three, subtract a safety overlap (for example 15 minutes) from the watermark and rely on an idempotent write to absorb the duplicates.
- **Logical-date window.** In an orchestrated batch, derive the window from the run's data interval (`data_interval_start` to `data_interval_end`) rather than from `now()`. A rerun of the 3 March run then always processes 3 March.

**Choosing the write**

- **Partition overwrite.** Recompute whole partitions and replace them atomically, for example with `INSERT OVERWRITE`, Delta's `replaceWhere`, or Spark dynamic partition overwrite. This is idempotent by construction and needs no key. It is the right choice for append-mostly event facts that are partitioned by event date.
- **MERGE (upsert) on a business key.** Use it when rows change after they are first written (orders, accounts) or when changes cut across partitions. It is idempotent only if the source batch is deduplicated to one row per key first; otherwise the engine raises a "multiple source rows matched" error or the result depends on which duplicate it processes.
- **Plain append** is not idempotent. Use it only with a downstream dedup step or exactly-once sink semantics.

```sql
MERGE INTO silver.orders t
USING (
  SELECT * FROM staging.orders_batch
  QUALIFY ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updated_at DESC) = 1
) s
ON t.order_id = s.order_id
WHEN MATCHED AND s.updated_at > t.updated_at THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
```

The `s.updated_at > t.updated_at` guard keeps a replayed older batch from overwriting newer data.

**Reprocessing windows**

Real data arrives late and gets corrected, so each run should rebuild a trailing window, for example the last 3 days of partitions, rather than only the newest one. Base the window size on measured lateness: if 99.9% of events arrive within 48 hours, reprocess 3 days, and handle anything older with an explicit backfill. The trade-off is compute cost against correctness, and that trade-off belongs in the dataset's SLA.

**Checklist:** deterministic inputs (the data interval, not `now()`), atomic writes, dedup before merge, the watermark committed after the data, and a tested rerun that produces identical row counts and checksums.

### 7.21 How do you handle late-arriving facts and late-arriving dimensions?

The two problems run in opposite directions. A **late-arriving fact** is an event that shows up after the period it belongs to has already been processed. A **late-arriving dimension** (also called an early-arriving fact) is a fact that arrives before the dimension row it references.

**Late-arriving facts**

Example: a mobile app buffers events offline and uploads them two days later.

- **Partition by event time, not load time,** and keep both timestamps (`event_ts`, `ingested_at`).
- **Reprocess a trailing window** of partitions on each run (see 7.20), sized from the observed lateness distribution.
- **For older stragglers,** either route them through a targeted backfill of the affected partitions or accept them into a "late" partition and document the policy.
- **Use point-in-time dimension lookups.** When joining a late fact to an SCD Type 2 dimension, look up the version that was valid at `event_ts`, not the current version:

```sql
JOIN dim_customer d
  ON f.customer_id = d.customer_id
 AND f.event_ts >= d.valid_from
 AND f.event_ts <  d.valid_to
```

- **Downstream aggregates** must be recomputed for every affected period. This is where asset- or partition-aware orchestration helps, because it knows which downstream partitions a late partition invalidates.
- **In streaming,** watermarks and allowed lateness decide how long a window stays open. Events that arrive after the watermark are dropped or sent to a side output, and you reconcile them in batch later.

**Late-arriving dimensions**

Example: an order arrives referencing `customer_id = 991`, but the CRM extract containing that customer lands six hours later.

| Strategy | How it works | Trade-off |
| --- | --- | --- |
| Inferred member (placeholder) | Insert a dimension row with the natural key and "Unknown" attributes, flagged `is_inferred = true`. The fact gets a real surrogate key. When the real row arrives, update the placeholder in place (a Type 1 overwrite even on a Type 2 dimension) and clear the flag. | Best general answer. Facts never wait and joins never drop rows. |
| Default "Unknown" key (-1) | Point the fact at a single unknown member. | Simple, but needs a later re-keying pass, which is easy to forget. |
| Hold and retry | Park the fact in a pending table and retry the lookup on each run. | Exact, but delays data and needs a timeout policy. |
| Natural keys in facts | Store the natural key and resolve it at query time. | Common in lakehouse designs that skip surrogate keys. Pushes cost to queries. |

**Production concerns**

- Monitor the count and age of inferred members. A rising count usually means an upstream feed is broken, not late.
- Make the placeholder insert idempotent (`MERGE` on the natural key) so that parallel loads do not create duplicate placeholders.
- Agree with stakeholders on how much restatement of past numbers is acceptable. Late data changes published history, so dashboards should say "data for the last 72 hours is provisional".

### 7.22 What is the Write-Audit-Publish pattern and how do you implement it?

Write-Audit-Publish (WAP) is a pattern in which a pipeline writes new data somewhere consumers cannot see it, runs quality checks against it, and only then makes it visible in a single atomic step. Bad data never reaches dashboards or models, because a failed audit simply stops the publish. It is the data equivalent of running CI before you merge.

**The three steps**

1. **Write.** Produce the output into an isolated location: a staging table, a branch, or an unpublished snapshot.
2. **Audit.** Run checks against that isolated output: schema, row counts against the source, nulls and uniqueness on keys, referential integrity, distribution drift against recent runs, and business rules (revenue is not negative).
3. **Publish.** Atomically expose the audited data: a pointer swap, a branch fast-forward, or a metadata commit. If the audit fails, alert, keep the staged data for debugging, and leave production untouched.

**Implementation options**

| Platform | Write | Publish |
| --- | --- | --- |
| Apache Iceberg (branches) | Write to a branch such as `audit_0315` (Spark: `spark.wap.branch`) | `fast_forward` the `main` branch to the audit branch |
| Apache Iceberg (WAP id) | Set `write.wap.enabled=true` and write with a `spark.wap.id`. The snapshot is staged but not current. | `cherrypick_snapshot` |
| Delta Lake / generic warehouse | Write to `orders__staging` | `CREATE OR REPLACE TABLE ... CLONE`, a table rename or swap, or `INSERT OVERWRITE` from staging inside a transaction |
| Snowflake | Build in a staging schema | `ALTER TABLE ... SWAP WITH` or zero-copy clone |
| lakeFS / Nessie | Commit on a data branch | Merge the branch into main |
| dbt | Build models into a staging schema with `dbt build` (tests run inside the build) | Swap or clone into production, or deploy views over the audited tables |

```sql
-- Iceberg example (Spark SQL)
ALTER TABLE db.orders CREATE BRANCH audit_20260315;
SET spark.wap.branch = audit_20260315;
INSERT INTO db.orders SELECT * FROM staging_orders;
-- run checks against db.orders.branch_audit_20260315 ...
CALL catalog.system.fast_forward('db.orders', 'main', 'audit_20260315');
```

**Design considerations**

- **Atomicity across tables.** Publishing one table is easy. Publishing a consistent set of ten related tables needs either catalog-level branching (Nessie, lakeFS) or publishing views that are all switched in one transaction.
- **Audit cost.** Full-table checks on very large outputs are expensive. Audit the new partitions plus a few aggregate reconciliations.
- **Blocking vs warning.** Classify checks by severity. A broken primary key blocks the publish; a 5% volume dip warns. If too many checks block, people learn to bypass the gate.
- **Freshness trade-off.** WAP adds latency and a new failure mode, data that is stale because it was held back. Alert on held publishes as urgently as on failed jobs.
- **Cleanup.** Expire abandoned branches and staging snapshots so they do not pile up storage.

### 7.23 What are Airflow executors, and how do you choose one?

The executor is the part of Airflow that decides where and how queued task instances actually run. The scheduler decides when a task should run; the executor launches it. The executor sets the deployment's isolation, scaling behaviour and operating cost. In Airflow 3, tasks talk to the API server through the Task Execution Interface instead of connecting to the metadata database directly, which makes remote and edge execution much cleaner.

**The main executors**

| Executor | How tasks run | Strengths | Weaknesses |
| --- | --- | --- | --- |
| LocalExecutor | Subprocesses on the scheduler host | Zero extra infrastructure, low latency. The default. | Limited to one machine. A noisy task affects the scheduler. |
| CeleryExecutor | A pool of long-running workers that pull from a broker (Redis or RabbitMQ) | Fast task start, horizontal scaling, queues for routing to specialised workers | Workers idle when nothing runs. Shared Python environment across tasks. Broker to operate. |
| KubernetesExecutor | One pod per task instance | Strong isolation, per-task images and resources (`executor_config` / pod override), scales to zero | Pod start-up adds seconds of overhead per task. Needs Kubernetes expertise. Poor fit for thousands of tiny tasks. |
| EdgeExecutor (Edge3 provider) | Lightweight workers in remote sites that poll the API server over HTTPS | Runs tasks in other networks, on-prem or other clouds, without opening inbound ports | Newer and less mature. You manage remote workers. |
| AWS ECS / Batch executors (Amazon provider) | One container or job per task on managed AWS compute | Serverless-style isolation without running Kubernetes | Tied to a cloud. Container start latency. |

**Hybrid execution**

The old hybrid executors (CeleryKubernetesExecutor and LocalKubernetesExecutor) were removed in Airflow 3.0. Instead, Airflow supports **multiple executors at once**: list them in `[core] executor` (the first one is the default), then choose per DAG or per task.

```ini
[core]
executor = CeleryExecutor,KubernetesExecutor
```

```python
@task(executor="KubernetesExecutor")
def train_model(): ...
```

A typical setup sends many short SQL-triggering tasks to Celery for low latency, and sends heavy, dependency-conflicting or GPU tasks to Kubernetes for isolation.

**How to choose**

- **Small team, modest load:** LocalExecutor on a decent VM, or a managed service (MWAA, Cloud Composer, Astronomer), where the choice is mostly made for you.
- **High task volume, short tasks, homogeneous dependencies:** Celery, with autoscaled workers (KEDA on queue depth) and queues per workload class.
- **Heterogeneous dependencies, strict isolation, bursty load, existing Kubernetes platform:** KubernetesExecutor.
- **Tasks that must run near data in another network:** add EdgeExecutor alongside your main executor.

**A senior point:** most production Airflow tasks should be thin. They trigger work in Spark, a warehouse or dbt and wait. In that case the executor matters less than the waiting pattern, which is why deferrable operators (7.24) often save more money than tuning the executor. Also watch the controls that apply to every executor: `parallelism`, `max_active_tasks_per_dag`, pools, and worker concurrency.

### 7.24 What is the difference between sensors and deferrable operators in Airflow?

A sensor is a task that waits for a condition (a file lands, a partition exists, an upstream job finishes) by checking it repeatedly. A deferrable operator hands that waiting to an asynchronous **trigger** running in the separate **triggerer** process, and while it waits it holds no worker slot at all. They are not competing concepts: most modern sensors and long-running operators can run in deferrable mode.

**Three ways to wait**

| Mode | What happens while waiting | Cost | When to use |
| --- | --- | --- | --- |
| Sensor, `mode="poke"` | The task holds a worker slot and sleeps between pokes | One worker slot per waiting task, for the whole wait | Very short waits (seconds to a minute) |
| Sensor, `mode="reschedule"` | The task exits between pokes, enters `up_for_reschedule`, and the scheduler runs it again later | No slot between pokes, but a full task start every interval, plus scheduler load | Legacy pattern for long waits when no async trigger exists |
| Deferrable (`deferrable=True`) | The task raises `TaskDeferred`, frees its slot, and the triggerer runs an `async` trigger. When the trigger fires, the task resumes on a worker at a named method. | Thousands of waits share one triggerer event loop | Long waits: external jobs, files, time delays |

**How deferral works**

1. The operator starts work (for example, submits a Databricks run) and calls `self.defer(trigger=..., method_name="execute_complete")`.
2. The task instance moves to the `deferred` state and releases its worker or pod.
3. The triggerer process runs the trigger's `async def run()` coroutine, which polls or listens without blocking other triggers.
4. When the trigger yields a `TriggerEvent`, the scheduler requeues the task, and a worker calls `execute_complete` with the event payload.

**Why it matters in production**

- **Cost and capacity.** With poke sensors, 300 DAGs each waiting two hours on an S3 file tie up 300 worker slots, and on KubernetesExecutor that means 300 idle pods. Deferred, they cost a few coroutines.
- **Deadlock avoidance.** Sensors in poke mode can fill every pool slot, so the very tasks that would satisfy them never get scheduled. Deferral removes that failure mode.
- **Timeouts still apply.** Set `timeout` on sensors and `execution_timeout` on tasks so that a wait which never ends fails loudly instead of running forever.

**Pitfalls**

- The triggerer must be deployed and highly available (run at least two). If it is down, deferred tasks never resume.
- Trigger code must be truly async. A blocking call such as `requests.get` or `time.sleep` inside `run()` stalls every trigger in that process.
- Triggers must be serializable and are re-created after a triggerer restart, so they cannot hold in-memory state.
- **Often the best sensor is no sensor.** If the upstream is another DAG or a dataset you own, use asset-based scheduling (7.26) instead of polling for it.

```python
wait = S3KeySensor(
    task_id="wait_for_export",
    bucket_key="s3://raw/orders/{{ ds }}/_SUCCESS",
    deferrable=True,
    timeout=6 * 60 * 60,
)
```

### 7.25 What are the most important Airflow best practices for production DAGs?

Treat a DAG file as configuration that is parsed constantly and a task as a transaction that will be retried. Most production Airflow incidents trace back to breaking one of those two assumptions.

**1. Idempotent, deterministic tasks**

- A rerun of a task for the same data interval must produce the same result. Use partition overwrite or `MERGE`, never a blind `INSERT` (see 7.20).
- Read inputs for the run's interval, never "latest", and never use `datetime.now()` in business logic.

**2. Template with logical dates**

- Use `{{ data_interval_start }}`, `{{ data_interval_end }}` and `{{ ds }}` in templated fields so that backfills and reruns process the right slice.
- In Airflow 3, manually triggered or asset-triggered runs may have no logical date. Design those DAGs to read the triggering event or parameters instead.

**3. Keep top-level code trivial**

- The DAG processor re-parses every file continually. No database queries, API calls, `Variable.get()` or heavy imports (pandas, ML libraries) at module level. Move them inside task callables or use Jinja such as `{{ var.value.x }}`.
- Measure parse time (`python dag.py` and the DAG processor metrics). Generating DAGs dynamically from a config file is fine; generating them from a live database query is not.

**4. Be deliberate about scheduling and catchup**

- Airflow 3 changed the default of `catchup` to `False`. Set it explicitly on every DAG anyway, and use `max_active_runs` to limit how many historical runs execute in parallel.
- Use a static `start_date`, never a dynamic one such as `datetime.now()`.

**5. Use XCom for metadata only**

- Pass paths, IDs and row counts, not dataframes. Large XComs bloat the metadata database or need a custom object-storage backend. Write data to S3 or a table and pass the URI.

**6. Retries, timeouts and alerts**

- Set `retries`, `retry_delay` and `retry_exponential_backoff` in `default_args`, and `execution_timeout` on every task.
- Retries help with transient errors only. A non-idempotent task that retries becomes a data-corruption bug.
- Use `on_failure_callback` for alerting, and track dataset freshness as well, since the absence of a run is also a failure.

**7. Control concurrency with pools**

- Create pools per constrained resource (for example, `snowflake_pool` with 10 slots, or an API rate limit) so that a backfill cannot overwhelm a shared system. Use `priority_weight` to protect critical paths.

**8. Thin orchestration, heavy work elsewhere**

- Airflow coordinates; Spark, the warehouse or dbt compute. Use deferrable operators for long waits (7.24). Run conflicting dependencies with `KubernetesPodOperator`, `@task.virtualenv` or `@task.docker`.

**9. Secrets and configuration**

- Use Connections and a secrets backend (Vault, AWS Secrets Manager), never hard-coded credentials. Parameterize environments with variables, not code forks.

**10. Test and lint in CI**

- A DagBag import test (no import errors, required tags and owners, no cycles), unit tests for task logic, `ruff --select AIR` for deprecated patterns, and a staging deployment before production.

```python
@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=3,
    default_args={"retries": 3, "retry_exponential_backoff": True,
                  "execution_timeout": timedelta(hours=1)},
    tags=["orders", "tier-1"],
)
def orders_daily(): ...
```

### 7.26 Compare task-based orchestration (Airflow) with asset-based orchestration (Dagster, Airflow assets).

Task-based orchestration models a pipeline as **steps to run**: "run extract, then transform, then load". Asset-based orchestration models it as **data products that should exist and be up to date**: "the `daily_revenue` table depends on `orders` and `fx_rates`". The orchestrator works out what to run from the graph of assets. Both can produce the same jobs. The difference is what the system understands and can therefore reason about.

**Comparison**

| Dimension | Task-based (classic Airflow DAGs) | Asset-based (Dagster, Airflow 3 assets) |
| --- | --- | --- |
| Primary unit | Task / operator | Asset (table, file, model) with its upstream assets |
| Lineage | Implicit, inferred from task names or external tools | Explicit and built in, so a catalog comes for free |
| Scheduling | Cron or time-based per DAG | Can also be driven by upstream data changes or declarative freshness policies |
| Cross-pipeline dependencies | Sensors or cross-DAG triggers, often brittle | Native: a downstream asset knows its upstream assets regardless of which pipeline owns them |
| Backfills | By DAG run and logical date | By asset partition, including the downstream partitions that become stale |
| Observability | "Did the task succeed?" | "Is this table fresh, and did its checks pass?" |
| Fit | Heterogeneous operational workflows: API calls, ML jobs, infrastructure tasks | Analytics and ML data platforms built around tables and models |

**How each tool does it**

- **Dagster** is asset-first. You declare `@asset` functions whose parameters name their upstream assets. You get partitioning (time, static, dynamic, multi-dimensional), asset checks, IO managers that separate storage from logic, and declarative automation (automation conditions such as "materialize when upstream updates" or "on cron if missing"). Its dbt integration maps each dbt model to an asset.
- **Airflow** added data-aware scheduling with **Datasets** in 2.4. These were renamed **Assets** in 3.0 (`airflow.sdk.Asset`). A task declares `outlets=[Asset("s3://.../orders")]`, and a DAG with `schedule=[Asset(...)]` runs when the asset is updated. Airflow 3 also adds the `@asset` decorator, asset aliases, conditional logic (`asset_a & (asset_b | asset_c)`), and event-driven scheduling from external sources through asset watchers (for example, a message queue). In Airflow, though, the asset is still mainly a scheduling signal: tasks stay the unit of execution, and partition-level asset backfills are less mature than in Dagster.

**How to decide**

- Choose **asset-based** when most of the work produces tables and models, many teams depend on one another's outputs, and you want lineage, freshness and partition backfills to be built in.
- Choose **task-based** when the work is procedural (file transfers, triggering SaaS jobs, operational automation) or when you have a large Airflow estate. In that case, adopt Airflow assets incrementally to replace cross-DAG sensors.
- A mixed pattern is common: Airflow orchestrates ingestion and operational work, while dbt or Dagster manages the asset graph for transformations.

**The senior point:** asset-based orchestration pushes teams to think in terms of data contracts and SLAs per data product, and that habit matters more than the tool. The cost is that it fits poorly with work that has no meaningful output asset, and moving a large task-based codebase over is a real investment.

### 7.27 How do dbt incremental models and snapshots work, and what are their pitfalls?

An **incremental model** builds the full table on its first run (or with `--full-refresh`). On later runs it processes only new rows, filtered by an `is_incremental()` block, and folds them into the existing table using an **incremental strategy**. A **snapshot** records how the rows of a mutable source change over time, producing an SCD Type 2 table with `dbt_valid_from` and `dbt_valid_to`. Use incremental models for performance. Use snapshots to capture history that the source does not keep.

**Incremental models**

```sql
{{ config(materialized='incremental', unique_key='order_id',
          incremental_strategy='merge', on_schema_change='append_new_columns') }}
select * from {{ ref('stg_orders') }}
{% if is_incremental() %}
  where updated_at > (select max(updated_at) from {{ this }}) - interval '3 days'
{% endif %}
```

| Strategy | Behaviour | Good for |
| --- | --- | --- |
| `append` | Insert only | Immutable logs where duplicates are handled downstream |
| `merge` | Upsert on `unique_key` | Mutable entities |
| `delete+insert` | Delete matching keys, then insert | Warehouses where merge is slow or keys are not strictly unique |
| `insert_overwrite` | Replace whole partitions | Large date-partitioned facts (BigQuery, Spark, Databricks) |
| `microbatch` | dbt splits the run into time batches by `event_time`, with `batch_size`, `begin` and `lookback` | Very large time series. Batches can be retried and backfilled individually, and can run in parallel. |

**Incremental pitfalls**

- **Late data is missed** if the filter is a strict `> max(updated_at)`. Add a lookback window and make sure the strategy is idempotent for the overlap.
- **Duplicates.** `append` with an overlap duplicates rows, and `merge` fails or misbehaves when the source batch has two rows for one key. Deduplicate inside the model.
- **Logic drift.** If you change the SQL, history is not recomputed. Old rows keep the old logic until a full refresh, which can be very expensive on big tables, so plan for it (or use microbatch backfills of selected ranges).
- **Schema changes.** The default `on_schema_change='ignore'` silently drops new columns. Choose `append_new_columns`, `sync_all_columns` or `fail` deliberately.
- **Merge cost.** A merge without a partition predicate scans the whole target. Use `incremental_predicates` or `insert_overwrite` for large tables.
- **Deletes** do not propagate unless the source exposes them (soft-delete flags or CDC).

**Snapshots**

In dbt 1.9 and later, snapshots are defined in YAML (`relation`, `unique_key`, `strategy`), and `dbt snapshot` runs them.

- **`timestamp` strategy** (preferred): a row has changed if its `updated_at` is newer.
- **`check` strategy:** compares the listed `check_cols`. Use it when there is no reliable timestamp.
- **`hard_deletes`:** `ignore` (default), `invalidate` (close the row), or `new_record` (add a row with `dbt_is_deleted = true`).
- **`dbt_valid_to_current`:** set an end date such as `9999-12-31` for current rows instead of NULL, which simplifies range joins.

**Snapshot pitfalls**

- **History is only as fine as your schedule.** Snapshotting daily misses a row that changed three times in a day. For full fidelity, use CDC instead.
- **Snapshots cannot be rebuilt.** `--full-refresh` does not apply to them, and if you lose the table, the history is gone. Keep snapshots in a protected schema with restricted grants and backups.
- **A non-unique `unique_key`** corrupts history with overlapping validity ranges. Test the source's uniqueness.
- **The `check` strategy with `all` columns** creates spurious versions when unimportant columns churn, and becomes fragile as columns are added.
- **Do not transform heavily inside a snapshot.** Snapshot the raw-ish source and model on top of it, because a logic bug would be baked permanently into history.
- **Removed or retyped source columns** are not dropped or altered in the snapshot table.

### 7.28 How do you test data pipelines and set up CI/CD for them?

Pipelines need two kinds of tests. **Code tests** check that the logic is correct for known inputs, and they run in CI before merge. **Data tests** check that real data meets expectations, and they run in CI and on every production run. Mature teams combine both with isolated environments and a deployment pipeline that builds only what changed.

**The testing layers**

| Layer | What it verifies | Tools | Where it runs |
| --- | --- | --- | --- |
| Unit tests | Transformation functions on small fixture data, including edge cases (nulls, duplicates, time zones, late records) | pytest with local Spark or DuckDB, dbt unit tests (dbt 1.8+) | Every commit |
| DAG integrity | DAGs import, have no cycles, and carry owners, tags, retries and timeouts | pytest over DagBag, `ruff --select AIR` | Every commit |
| Integration tests | Components work together against real engines, for example ingest to staging to model on a sample | Docker Compose, Testcontainers, a CI warehouse schema | Pull request |
| Data tests | Uniqueness, not-null, accepted values, referential integrity, freshness, volume and distribution | dbt data tests, Great Expectations, Soda, dbt-expectations | CI and every production run |
| Contracts | Column names and types match what consumers expect | dbt model contracts, schema registry compatibility rules | CI (breaking changes fail the build) |
| Regression / diff | Output did not change unexpectedly compared with production | Data-diff tooling, row-count and aggregate comparisons | Pull request |

**dbt specifics**

- **Unit tests** mock the inputs to a model and assert the expected output, which makes them ideal for complex `CASE` logic and incremental logic.
- **Data tests** (`unique`, `not_null`, `relationships`, `accepted_values`, plus custom singular tests) run inside `dbt build`, so a failing test stops dependent models from building.
- **Slim CI:** build only modified models and their downstream dependents, and defer unchanged upstream references to production artifacts.

```bash
dbt build --select state:modified+ --defer --state ./prod-artifacts --target ci
```

Add `dbt clone` (zero-copy clones of production tables) to create realistic CI inputs cheaply. Write each PR to its own schema (`ci_pr_1234`) and drop it afterwards.

**Environments**

- **dev:** each engineer works in a personal schema or branch, with sampled or limited data.
- **CI:** an ephemeral schema per pull request, built automatically.
- **staging / pre-production:** production-like data volume, used to run the full DAG before release. Catalog branching (Iceberg, Nessie, lakeFS) or zero-copy clones make this affordable.
- **prod:** deploys only from the main branch, through the pipeline, never by hand.

**A CI/CD pipeline in practice**

1. On pull request: lint (SQLFluff, ruff), unit tests, DAG integrity tests, then slim CI `dbt build` with tests and contract checks, and a data diff against production posted as a PR comment.
2. On merge: build versioned artifacts (a Docker image, the dbt manifest), deploy to staging, run smoke tests, then promote to production.
3. In production: data tests and freshness checks gate the publish (Write-Audit-Publish, 7.22), and anomaly monitoring catches what the fixed tests missed.

**Pitfalls:** tests that only ever run on clean fixture data, CI that builds the whole project (slow and expensive, so people skip it), production credentials in CI, and treating test failures as noise. Every test should either block or have a named owner who responds to it.

### 7.29 How would you build a log-based CDC pipeline from an OLTP database into a lakehouse, and what is the transactional outbox pattern?

The standard design is: **source database transaction log, then Debezium (on Kafka Connect), then Kafka, then a stream or micro-batch job that applies changes with `MERGE` into Iceberg or Delta tables.** The hard parts are not moving bytes. They are the initial snapshot, ordering, deletes, schema changes, and applying changes idempotently.

**Architecture**

1. **Source setup.** Enable logical replication: for PostgreSQL, `wal_level=logical`, a replication slot and a publication; for MySQL, row-based binlog with full row images. Monitor slot lag, because an inactive Debezium connector makes the PostgreSQL WAL grow until the disk fills.
2. **Debezium connector.** Reads the log and emits one event per row change, with `before`, `after`, `op` (`c`, `u`, `d`, `r` for snapshot reads) and source metadata (LSN or binlog position, transaction ID, timestamp). Topics are typically one per table, keyed by primary key.
3. **Kafka.** Serialize with Avro or Protobuf through a schema registry. Partition by primary key so that all changes to a row stay in order. Ordering is guaranteed only within a partition, which is enough for per-row correctness.
4. **Apply job.** Spark Structured Streaming, Flink or Kafka Connect sinks (for example the Iceberg sink) deduplicate each micro-batch to the latest event per key, then `MERGE` into the target.

```sql
MERGE INTO lake.customers t
USING (latest event per id in this batch, ordered by source LSN) s
ON t.id = s.id
WHEN MATCHED AND s.op = 'd' THEN DELETE
WHEN MATCHED AND s.lsn > t.lsn THEN UPDATE SET *
WHEN NOT MATCHED AND s.op <> 'd' THEN INSERT *;
```

**The hard parts**

- **Initial snapshot.** Debezium first takes a consistent snapshot of existing rows, then streams changes from the log position recorded at snapshot time. For large tables, use **incremental snapshots** (triggered through a signalling table or Kafka signal topic). They read in chunks, interleave with live streaming, can be resumed, and can re-snapshot one table without stopping the connector.
- **Ordering and idempotency.** Delivery is at-least-once, so expect duplicates after restarts. Order by a source position (LSN or binlog offset), not by wall-clock timestamp, and make the merge reject older positions. Keeping the position on the target row makes replays safe.
- **Deletes and tombstones.** A delete emits a `d` event (with the `before` image) followed by a Kafka **tombstone** (the key with a null value), which lets log compaction eventually purge the key. Your apply job must turn `d` into a `DELETE` or a soft-delete flag and ignore the tombstones. If you flatten events with the `ExtractNewRecordState` transform, configure how deletes are rewritten so they are not silently lost.
- **Schema changes.** Additive changes (a new nullable column) flow through the schema registry with backward compatibility and can map to Iceberg or Delta schema evolution. Renames, type narrowing and dropped columns need a coordinated plan: registry compatibility rules that reject the change, an alert, and a controlled migration. Do not let the sink auto-evolve into a mess.
- **Lakehouse table health.** Frequent merges create many small files and delete files. Schedule compaction, snapshot expiry and (on Iceberg) rewriting of position deletes. Consider merge-on-read for write-heavy tables. A common pattern keeps an append-only bronze change log (useful for audit and replay) and builds a merged silver table from it.
- **Transactions.** Changes from one source transaction can land in different topics. If consumers need transactional consistency, use Debezium's transaction metadata topic or accept per-row eventual consistency.

**The transactional outbox pattern**

The problem: a service that writes to its database and then publishes an event to Kafka performs a **dual write**. If the service crashes between the two, the database and the event stream disagree.

The outbox fixes this. Within the **same local transaction** as the business change, the service inserts an event row into an `outbox` table (`id`, `aggregate_type`, `aggregate_id`, `type`, `payload`). Debezium tails the outbox table, and its **Outbox Event Router** transform routes each row to a topic based on `aggregate_type`, keyed by `aggregate_id`. The outbox rows can be deleted right after insert, because the event is already in the log.

Why it matters for data platforms: CDC on internal tables couples the warehouse to the service's private schema, so any refactor breaks the pipelines. Outbox events are a **published contract**: domain events with stable, versioned schemas that the service owns. Use raw table CDC for replication and analytics on legacy systems. Prefer the outbox for events between services and for data products whose producers can commit to a contract. Consumers must still be idempotent, because delivery remains at-least-once.

---

← [Chapter 6. Distributed Processing and Streaming](06-distributed-processing-and-streaming.md) · [Table of contents](../README.md) · [Chapter 8. Machine Learning Fundamentals](../part-3-machine-learning/08-machine-learning-fundamentals.md) →
