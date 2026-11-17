# Chapter 7. Pipelines and Orchestration

_Part II — Data Engineering_

ETL and ELT design, orchestration with Airflow, API ingestion, idempotency and incremental loads, backfills, data SLAs, testing and CDC, building reliable pipelines for analytics and ML.

This chapter contains **23 questions**.

## Contents

- **7.1** [What is the difference between ETL and ELT?](#71-what-is-the-difference-between-etl-and-elt)
- **7.2** [What is Apache Airflow, what problems does it solve, and how do you design a workflow (DAG) in it?](#72-what-is-apache-airflow-what-problems-does-it-solve-and-how-do-you-design-a-workflow-dag-in-it)
- **7.3** [How do you handle pipeline failures in production?](#73-how-do-you-handle-pipeline-failures-in-production)
- **7.4** [What are Airflow "Hooks" and "Operators"?](#74-what-are-airflow-hooks-and-operators)
- **7.5** [What is "Backfilling" in Airflow, and how do you run backfills safely?](#75-what-is-backfilling-in-airflow-and-how-do-you-run-backfills-safely)
- **7.6** [How do you pass data between Airflow tasks?](#76-how-do-you-pass-data-between-airflow-tasks)
- **7.7** [What is "Functional Data Engineering", and why do functional programming ideas matter in pipelines?](#77-what-is-functional-data-engineering-and-why-do-functional-programming-ideas-matter-in-pipelines)
- **7.8** [How do you choose between pandas, Polars, DuckDB and Spark for a pipeline step, and how do you process data larger than memory in Python?](#78-how-do-you-choose-between-pandas-polars-duckdb-and-spark-for-a-pipeline-step-and-how-do-you-process-data-larger-than-memory-in-python)
- **7.9** [How do you ingest data from external APIs reliably, including rate limiting, retries and pagination?](#79-how-do-you-ingest-data-from-external-apis-reliably-including-rate-limiting-retries-and-pagination)
- **7.10** [What is the purpose of a "Staging Area"?](#710-what-is-the-purpose-of-a-staging-area)
- **7.11** [What is "Change Data Capture" (CDC)?](#711-what-is-change-data-capture-cdc)
- **7.12** [What is an "SLA" in data engineering, and how do you define and monitor SLAs, SLOs and SLIs for data?](#712-what-is-an-sla-in-data-engineering-and-how-do-you-define-and-monitor-slas-slos-and-slis-for-data)
- **7.13** [How do you build reliable data pipelines for ML training and inference, and what makes data "clean and reliable" in this context?](#713-how-do-you-build-reliable-data-pipelines-for-ml-training-and-inference-and-what-makes-data-clean-and-reliable-in-this-context)
- **7.14** [How do you design an idempotent incremental load?](#714-how-do-you-design-an-idempotent-incremental-load)
- **7.15** [How do you handle late-arriving facts and late-arriving dimensions?](#715-how-do-you-handle-late-arriving-facts-and-late-arriving-dimensions)
- **7.16** [What is the Write-Audit-Publish pattern and how do you implement it?](#716-what-is-the-write-audit-publish-pattern-and-how-do-you-implement-it)
- **7.17** [What are Airflow executors, and how do you choose one?](#717-what-are-airflow-executors-and-how-do-you-choose-one)
- **7.18** [What is the difference between sensors and deferrable operators in Airflow?](#718-what-is-the-difference-between-sensors-and-deferrable-operators-in-airflow)
- **7.19** [What are the most important Airflow best practices for production DAGs?](#719-what-are-the-most-important-airflow-best-practices-for-production-dags)
- **7.20** [Compare task-based orchestration (Airflow) with asset-based orchestration (Dagster, Airflow assets).](#720-compare-task-based-orchestration-airflow-with-asset-based-orchestration-dagster-airflow-assets)
- **7.21** [How do dbt incremental models and snapshots work, and what are their pitfalls?](#721-how-do-dbt-incremental-models-and-snapshots-work-and-what-are-their-pitfalls)
- **7.22** [How do you test data pipelines and set up CI/CD for them?](#722-how-do-you-test-data-pipelines-and-set-up-cicd-for-them)
- **7.23** [How would you build a log-based CDC pipeline from an OLTP database into a lakehouse, and what is the transactional outbox pattern?](#723-how-would-you-build-a-log-based-cdc-pipeline-from-an-oltp-database-into-a-lakehouse-and-what-is-the-transactional-outbox-pattern)

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

### 7.2 What is Apache Airflow, what problems does it solve, and how do you design a workflow (DAG) in it?

Apache Airflow is an open-source platform for authoring, scheduling and monitoring batch workflows as code. Started at Airbnb in 2014 and now an Apache project, it represents each workflow as a **DAG** (directed acyclic graph) of tasks written in Python: "directed" because dependencies have a direction, "acyclic" because no task can depend on itself, directly or indirectly. Airflow is an **orchestrator**: it decides what runs when, in which order, with which retries, and records what happened. The heavy data processing should happen in the systems it calls (Spark, the warehouse, dbt).

**What it solves compared with cron and scripts**

| Problem with cron and scripts | What Airflow provides |
| --- | --- |
| No dependencies between jobs, only guessed start times | Explicit task dependencies, cross-DAG triggers and asset-based scheduling |
| Failures noticed only when someone complains | Retries with backoff, timeouts, failure callbacks and alerting |
| No history of what ran for which period | A run per data interval, with state, logs and duration kept in a metadata database |
| Reprocessing a past day is manual and risky | Backfills and clearing tasks for specific intervals (see 7.5) |
| Credentials scattered in scripts | Connections and secrets backends |
| No visibility | A UI with graph and grid views, logs, and run history |
| Resource contention | Pools, concurrency limits and priorities |

**Architecture (Airflow 3)**

```text
DAG files (Python) -> DAG processor parses them -> serialised DAGs in the metadata DB
Scheduler: creates DAG runs, queues task instances when dependencies are met
Executor (Local, Celery, Kubernetes, ...): launches tasks on workers (see 7.17)
Workers: run tasks and talk to the API server (Task Execution Interface)
Triggerer: runs async triggers for deferred tasks (see 7.18)
API server + UI: REST API, web UI
Metadata DB (PostgreSQL): state of every DAG run and task instance
```

**Designing a workflow**

1. **Decide the unit of scheduling:** what data interval each run covers (hourly, daily) or which asset update triggers it.
2. **Split work into tasks that are idempotent, retryable and meaningful to rerun on their own:** extract, load, transform, validate, publish. Too coarse and a retry repeats hours of work; too fine and scheduling overhead and UI noise grow.
3. **Define dependencies** and branching (`@task.branch`), and use task groups for readability.
4. **Pass only metadata between tasks** (paths, IDs, counts), not data (see 7.6).
5. **Template inputs with the run's data interval**, so reruns and backfills process exactly the right slice.
6. **Add quality gates** before publishing (see 7.16), and set retries, timeouts, owners and alerts.

```python
import pendulum
from airflow.sdk import dag, task

@dag(schedule="@daily", start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
     catchup=False, max_active_runs=1,
     default_args={"retries": 3, "retry_exponential_backoff": True},
     tags=["orders", "tier-1"])
def orders_pipeline():

    @task
    def extract(data_interval_start=None, data_interval_end=None) -> str:
        path = f"s3://raw/orders/{data_interval_start:%Y-%m-%d}/"
        export_orders(data_interval_start, data_interval_end, path)  # idempotent overwrite
        return path

    @task
    def load(path: str) -> int:
        return merge_into_silver(path)             # MERGE on order_id, returns row count

    @task
    def validate(rows: int) -> None:
        if rows == 0:
            raise ValueError("No orders loaded for this interval")

    validate(load(extract()))

orders_pipeline()
```

**Where Airflow is a weak fit**

- Low-latency or streaming work (use Kafka, Flink or Structured Streaming; Airflow can deploy and monitor them).
- Thousands of very short tasks per minute; per-task scheduling overhead is seconds.
- Passing large data between tasks.

Alternatives such as Dagster (asset-first, see 7.20), Prefect, and platform-native schedulers (Databricks Workflows, dbt Cloud) are worth considering; for ML pipelines see 26.83.

**What interviewers look for:** understanding that Airflow orchestrates rather than processes, that tasks must be idempotent and parameterised by the data interval, and awareness of the production concerns covered in 7.17 to 7.20.

### 7.3 How do you handle pipeline failures in production?

Design for failure in four layers: **prevent** what you can (tests, contracts, idempotent steps), **contain** the blast radius when something fails (retries, quarantine, quality gates that stop bad data before it is published), **detect** quickly (alerts on failures, freshness and data anomalies, routed to owners), and **recover and learn** (runbooks, safe reruns and backfills, blameless postmortems). The aim is that a failure produces late data, never wrong data.

**1. Classify the failure, because handling differs**

| Failure type | Examples | Handling |
| --- | --- | --- |
| Transient infrastructure | Network timeouts, throttling, spot node loss | Automatic retries with exponential backoff and jitter |
| Upstream late or missing | Source file not delivered, API down | Wait with deferrable sensors and a timeout; alert the source owner; mark data stale |
| Bad data | Schema change, nulls in keys, duplicates, impossible values | Quarantine bad records or block publication; do not retry blindly |
| Code or logic bug | Exception on every run, wrong join | Fail fast, alert, fix and redeploy; retries waste time |
| Resource | OOM, disk full, quota exceeded | Fix sizing or data skew (see 6.14); retries rarely help |
| Silent failure | Job succeeds but loads half the rows | Only caught by data checks and anomaly detection (see 4.28) |

**2. Contain**

- **Idempotent tasks** so any retry or rerun is safe (see 4.7 and 7.14).
- **Write-Audit-Publish:** consumers only see data that passed checks (see 7.16).
- **Dead letter queues and quarantine tables** for record-level errors, with a threshold that fails the run if too many records are rejected (see 6.28).
- **Circuit breakers:** stop downstream jobs when an upstream quality check fails, rather than propagating bad data.
- **Last good data with a staleness flag** on dashboards, rather than empty or wrong numbers.

**3. Detect**

- Task failure callbacks to the owning team's channel and paging for tier-1 pipelines.
- **Freshness SLO monitoring** per dataset: a job that never started does not fail, so you must alert on missing data, not just failed tasks (see 7.12).
- Volume, schema and distribution anomaly detection.
- Alert hygiene: deduplicate, include links to logs, the run, the runbook and affected downstream assets.

**4. Recover**

- Runbooks for each tier-1 pipeline: common failures, how to rerun, whom to contact.
- Safe reprocessing: clear and rerun specific intervals, or run a backfill with concurrency limits (see 7.5).
- Communicate with consumers: an incident note with impact, expected recovery time and whether published numbers changed.

**5. Learn**

Blameless postmortems for significant incidents, tracking time to detect and time to resolve, and fixing classes of problems (a contract with the source team, a new check, a template change) rather than single occurrences.

```python
from datetime import timedelta

default_args = {
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=1),
    "on_failure_callback": notify_owner,     # posts run URL, log link, runbook
}
```

**Pitfalls:** retrying non-idempotent tasks (retries become duplicates), retrying deterministic failures for hours, alert channels so noisy that people mute them, and treating "the DAG is green" as "the data is correct".

### 7.4 What are Airflow "Hooks" and "Operators"?

An **operator** is a template for one task: it defines what the task does when it executes (run a SQL statement, submit a Spark job, copy a file, call Python). A **hook** is a reusable client for an external system: it reads credentials from an Airflow **connection** and exposes methods to talk to that system (`PostgresHook.get_records`, `S3Hook.load_file`). Operators usually use hooks internally. Both are distributed in **provider packages** (`apache-airflow-providers-amazon`, `-google`, `-databricks`, `-snowflake`), which are versioned separately from Airflow core.

**How they fit together**

```text
Connection (host, login, secret from a secrets backend)
      -> Hook (authenticated client, e.g. SnowflakeHook)
            -> Operator (task logic, e.g. SQLExecuteQueryOperator) -> Task instance in a DAG run
```

**Operator families**

| Kind | Examples | Notes |
| --- | --- | --- |
| Action operators | `SQLExecuteQueryOperator`, `DatabricksSubmitRunOperator`, `GlueJobOperator` | Trigger work in external systems |
| Transfer operators | `S3ToSnowflakeOperator`, `GCSToBigQueryOperator` | Move data between systems; fine for modest volumes |
| Sensors | `S3KeySensor`, `ExternalTaskSensor` | Wait for a condition; prefer deferrable mode (see 7.18) |
| Python | `@task` (TaskFlow), `@task.virtualenv`, `@task.docker`, `@task.kubernetes` | Custom logic, isolated dependencies |
| Container | `KubernetesPodOperator` | Any language or dependency set in its own pod |

**Writing a custom operator with a hook**

```python
from airflow.models import BaseOperator
from airflow.providers.http.hooks.http import HttpHook

class ExportCrmAccountsOperator(BaseOperator):
    template_fields = ("target_path", "since")

    def __init__(self, target_path: str, since: str, http_conn_id: str = "crm_api", **kw):
        super().__init__(**kw)
        self.target_path, self.since, self.http_conn_id = target_path, since, http_conn_id

    def execute(self, context):
        hook = HttpHook(method="GET", http_conn_id=self.http_conn_id)
        resp = hook.run("/v2/accounts", data={"updated_since": self.since})
        write_json_to_s3(resp.json(), self.target_path)     # idempotent overwrite
        return self.target_path                             # pushed to XCom
```

**Design guidance**

- **Prefer provider operators** to hand-written API calls: they handle authentication, retries, deferrable modes and logging.
- **Keep operators thin.** Business logic in a plain Python package that the operator calls is easier to unit test than logic inside `execute()`.
- **Use `template_fields`** so values like `{{ ds }}` are rendered per run.
- **Never create hooks or connections at DAG-file top level;** they run on every parse (see 7.19).
- **Pin provider versions** and test upgrades; provider changes are a common source of breakage.
- **Avoid using transfer operators for large volumes;** they move data through the worker. Use the platform's bulk load (`COPY INTO`, Spark) instead.

### 7.5 What is "Backfilling" in Airflow, and how do you run backfills safely?

Backfilling means running a pipeline for past data intervals: to populate history for a new pipeline, to reprocess after a bug fix or a logic change, or to recover after an outage. In Airflow a backfill creates DAG runs for each logical date in a range; because well-designed tasks read only their run's data interval and write idempotently, each run reprocesses exactly its slice.

**Ways to backfill in Airflow**

- **Backfill command or UI / API (Airflow 3):** backfills are first-class objects run by the scheduler, with options such as `max_active_runs`, reprocessing behaviour (only missing runs, failed runs, or all) and run order (newest first is often useful).

```bash
airflow backfill create --dag-id orders_pipeline \
  --from-date 2026-01-01 --to-date 2026-03-31 \
  --max-active-runs 4 --reprocess-behavior failed
```

- **`catchup=True`:** the scheduler automatically creates runs for all intervals since `start_date` that have not run. Convenient for new pipelines, dangerous when turned on by accident (hundreds of runs at once). Airflow 3 defaults it to `False`.
- **Clearing tasks** for specific past runs reruns them, including downstream tasks if selected.

**What makes a backfill safe**

1. **Idempotent writes:** partition overwrite or `MERGE` (see 7.14). A backfill of an append-only task duplicates data.
2. **Inputs bound to the data interval,** never `now()` or "latest". Watch for dimension lookups that use current values when history needs point-in-time values (see 7.15).
3. **Concurrency limits:** `max_active_runs`, pools and warehouse or cluster sizing so a backfill does not starve production runs or hit source API limits.
4. **Source availability:** retention in the source system or Kafka may not cover the whole range; bronze tables in the lakehouse are often the better replay source.
5. **Downstream impact:** reprocessing silver invalidates gold tables, ML features and reports for the same periods. Plan the cascade, or use asset-based orchestration that tracks stale partitions (see 7.20).
6. **Communication:** tell consumers which periods will change and when.

**Large backfills: often better outside the daily DAG**

Running 1,000 daily DAG runs for three years of history is slow and creates scheduler load. For big reprocessing, a single bulk job (one Spark job over the whole range, or a dbt full refresh or microbatch backfill) is usually faster and cheaper, followed by normal incremental runs.

```python
# Bulk reprocessing in one Spark job, partition-overwrite semantics
(spark.table("bronze.orders")
     .filter("ingest_date BETWEEN '2024-01-01' AND '2026-03-31'")
     .transform(build_silver_orders)
     .write.mode("overwrite").option("partitionOverwriteMode", "dynamic")
     .insertInto("silver.orders"))
```

**Pitfalls:** backfilling with new code that silently changes historical numbers without anyone agreeing to the restatement; forgetting that a changed `start_date` or schedule creates unexpected runs; and backfills competing with production for the same warehouse.

### 7.6 How do you pass data between Airflow tasks?

Pass **metadata** through XCom (cross-communication): file paths, table names, partition identifiers, row counts, job IDs. Pass the **data itself** through external storage: the upstream task writes to object storage or a table, and the downstream task reads from the location it receives. Airflow's metadata database is not designed to hold datasets.

**How XCom works**

- A task pushes a value (implicitly by returning it from a `@task` function or an operator's `execute`, or explicitly with `xcom_push`), stored with the DAG run, task and key.
- Downstream tasks pull it (`ti.xcom_pull`, templates like `{{ ti.xcom_pull(task_ids='extract') }}`, or automatically in TaskFlow when you pass one task's output to another).
- Values must be JSON-serialisable (or handled by a custom serialiser).

```python
@task
def extract(data_interval_start=None) -> dict:
    path = f"s3://staging/orders/{data_interval_start:%Y-%m-%d}/part.parquet"
    df = fetch_orders(data_interval_start)
    df.to_parquet(path)                       # data goes to storage
    return {"path": path, "rows": len(df)}    # metadata goes to XCom

@task
def load(meta: dict) -> None:
    copy_into_warehouse(meta["path"])         # downstream reads from storage
    assert meta["rows"] > 0
```

**Why not put data in XCom**

- XComs live in the metadata database by default; large values bloat it and slow the scheduler and UI for everyone.
- Tasks may run on different machines or pods, so passing in-memory objects is impossible anyway.
- Data in storage is inspectable, reusable for reruns and debugging, and covered by normal access controls and lineage.

**Custom XCom backends and object storage**

You can configure an XCom backend that stores values in S3, GCS or Azure Blob storage and keeps only a reference in the database (the common object-storage XCom backend in the Common IO provider stores values above a size threshold externally). This helps for medium-sized intermediate results, but it is still better to design tasks around explicit storage locations and tables.

**Alternatives and related features**

- **Assets (formerly datasets):** a task declares the asset it updates; downstream DAGs are triggered by that update (see 7.20).
- **Dynamic task mapping:** an upstream task returns a list (of files, partitions or tenants) and Airflow creates one mapped task instance per item. Keep the list to IDs or paths.
- **Staging tables** in the warehouse when the next step is SQL anyway.

**Pitfalls:** returning a DataFrame from a TaskFlow task by accident (it gets serialised into XCom), writing intermediate files to the worker's local disk (the next task may run elsewhere), and non-deterministic paths that break reruns. Derive paths from the data interval, not from the current time.

### 7.7 What is "Functional Data Engineering", and why do functional programming ideas matter in pipelines?

Functional programming favours **pure functions** (output depends only on inputs, no side effects) and **immutable data**. Applied to data engineering, the idea (popularised by Maxime Beauchemin as "functional data engineering") is to treat each pipeline task like a pure function: it reads immutable inputs for a defined period and **deterministically overwrites** an output partition, so any run can be repeated, reproduced or backfilled safely. The same ideas explain why Spark and similar engines are built the way they are.

**Core principles applied to pipelines**

| Functional idea | Pipeline practice |
| --- | --- |
| Pure functions | Tasks depend only on their inputs and the data interval, not on `now()`, global state or "latest" data |
| Immutability | Raw data is append-only and never edited; corrections create new versions |
| No side effects | A task's only effect is overwriting its own output partition; no in-place `UPDATE` of shared state |
| Referential transparency | Rerunning a task for 2026-03-01 always produces the same partition |
| Composition | Small transformations chained into larger ones |

**Why it helps**

- **Idempotency and safe retries** come for free (see 4.7).
- **Backfills and reproducibility:** rebuilding history means rerunning functions over immutable inputs.
- **Testability:** pure transformation functions can be unit tested with small inputs (see 4.29).
- **Parallelism:** independent partitions can be processed concurrently without coordination.
- **Distributed engines rely on it:** Spark recomputes lost partitions from lineage, which only works if transformations are deterministic (see 6.8). DataFrames are immutable; every transformation returns a new one.

**In code: pure transformations, I/O at the edges**

```python
from pyspark.sql import DataFrame, functions as F

def clean_orders(orders: DataFrame) -> DataFrame:              # pure
    return (orders.filter(F.col("order_id").isNotNull())
                  .withColumn("amount", F.col("amount").cast("decimal(18,2)")))

def daily_revenue(orders: DataFrame) -> DataFrame:             # pure
    return orders.groupBy("order_date", "country").agg(F.sum("amount").alias("revenue"))

def run(spark, ds: str) -> None:                               # I/O only here
    src = spark.table("bronze.orders").filter(F.col("order_date") == ds)
    out = src.transform(clean_orders).transform(daily_revenue)
    (out.write.mode("overwrite")
        .option("replaceWhere", f"order_date = '{ds}'")
        .saveAsTable("gold.daily_revenue"))
```

**Where the purity has to bend**

- **Slowly changing dimensions and snapshots** depend on history; model them as functions of the full change log, or isolate them clearly (see 7.21).
- **Mutable entities** (CDC `MERGE`) are not pure overwrites; make them idempotent with ordering guards (see 7.14 and 7.23).
- **Storage cost of immutability:** keeping all versions costs money; use retention policies.

**Interview angle:** the value is not functional programming syntax (`map`, `lambda`), which is covered in Chapter 2, but the discipline: deterministic tasks, immutable inputs, partition overwrites, and side effects only at clearly defined edges.

### 7.8 How do you choose between pandas, Polars, DuckDB and Spark for a pipeline step, and how do you process data larger than memory in Python?

Choose the simplest engine that comfortably handles the data volume and the team can operate. pandas is ideal for small data and the ecosystem around it; Polars and DuckDB are fast, multi-threaded, single-node engines that handle tens to hundreds of GB efficiently, including out-of-core processing; Spark is for data that genuinely needs a cluster or for platforms already standardised on it. Most pipeline steps are smaller than people assume.

**Comparison**

| Engine | Model | Sweet spot | Limits |
| --- | --- | --- | --- |
| pandas | Eager, mostly single-threaded, in memory | Up to a few GB; rich ecosystem; ML preprocessing | Memory often 5 to 10 times the file size; slow on wide joins and group-bys |
| Polars | Columnar (Arrow), multi-threaded, lazy query optimiser, streaming engine | Single machine, GB to hundreds of GB | Single node; API differs from pandas |
| DuckDB | In-process analytical SQL database, vectorised, spills to disk | SQL over Parquet, CSV and lakehouse tables on one machine | Single node; one writer at a time to a database file |
| Spark (PySpark) | Distributed, lazy, Catalyst optimiser | Terabytes, cluster-scale joins, existing Spark platforms | Cluster cost and overhead; slow for small data |
| Dask, Ray | Distributed Python | Scaling Python and pandas-like code, ML workloads | More operational complexity |

**Processing data larger than memory in Python**

1. **Read only what you need:** select columns and filter at read time (Parquet makes this cheap).
2. **Stream instead of loading:** generators and chunked readers process records in constant memory (generator mechanics are covered in 2.45).
3. **Use a lazy, out-of-core engine:** Polars lazy queries and DuckDB plan the whole query and stream or spill.
4. **Push work into the database or warehouse** when the data already lives there.
5. **Distribute** only when single-node approaches are exhausted.

```python
import polars as pl, duckdb, pandas as pd

# Polars: lazy scan, only needed columns and rows are read, streaming execution
daily = (pl.scan_parquet("s3://lake/events/*.parquet")
           .filter(pl.col("event_type") == "purchase")
           .group_by("country")
           .agg(pl.col("amount").sum())
           .collect(engine="streaming"))

# DuckDB: SQL over Parquet, spills to disk if needed
top = duckdb.sql("""SELECT customer_id, sum(amount) AS total
                    FROM 'orders/*.parquet' GROUP BY 1 ORDER BY 2 DESC LIMIT 100""").df()

# pandas with chunks: constant memory for row-wise cleaning of a huge CSV
for i, chunk in enumerate(pd.read_csv("big.csv", chunksize=500_000, dtype={"id": "string"})):
    clean(chunk).to_parquet(f"out/part-{i:05d}.parquet", index=False)
```

**When not to use pandas**

- Data that does not fit comfortably in memory, or grows towards it.
- Heavy joins and aggregations over many millions of rows when Polars or DuckDB would be several times faster.
- Long-running production pipelines where dtype inference surprises (mixed types, integer columns turning into floats on nulls) cause bugs. Declare dtypes or use Arrow-backed dtypes.

**Senior guidance:** standardise on a small set of engines, use Arrow and Parquet to move between them without conversion cost, and right-size infrastructure. A single large VM running DuckDB or Polars often replaces a Spark cluster for mid-sized data at a fraction of the cost and complexity. For general Python performance techniques see 2.68.

### 7.9 How do you ingest data from external APIs reliably, including rate limiting, retries and pagination?

Treat an external API as an unreliable, rate-limited dependency. A reliable ingestion job **respects the provider's limits proactively** (throttling below the quota), **retries only transient errors** with exponential backoff and jitter while honouring `Retry-After`, **paginates and extracts incrementally** using cursors or updated-since filters, **writes idempotently** so reruns are safe, and **records progress** so it can resume after failure.

**Error handling by status**

| Response | Meaning | Action |
| --- | --- | --- |
| 429 Too Many Requests | Rate limited | Wait for `Retry-After` if given, else exponential backoff with jitter |
| 500, 502, 503, 504, timeouts, connection resets | Transient server or network problem | Retry with backoff, bounded attempts |
| 400, 404, 422 | Bad request or missing resource | Do not retry; log, quarantine or fail |
| 401, 403 | Authentication or permission | Refresh token once; otherwise fail and alert |

**A retry decorator (decorator mechanics are covered in 2.6)**

```python
import random, time, functools, requests

TRANSIENT = {429, 500, 502, 503, 504}

def retry(max_attempts=6, base=1.0, cap=60.0):
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = fn(*args, **kwargs)
                    if resp.status_code not in TRANSIENT:
                        resp.raise_for_status()
                        return resp
                    delay = float(resp.headers.get("Retry-After", 0)) or min(cap, base * 2 ** attempt)
                except (requests.ConnectionError, requests.Timeout):
                    delay = min(cap, base * 2 ** attempt)
                if attempt == max_attempts:
                    raise RuntimeError(f"giving up after {attempt} attempts")
                time.sleep(delay * random.uniform(0.5, 1.0))     # jitter
        return inner
    return wrap

@retry()
def get_page(session, url, params):
    return session.get(url, params=params, timeout=30)
```

Libraries such as `tenacity` (retries) and `pyrate-limiter` or a token bucket (proactive throttling) do the same with less code.

**Pagination and incremental extraction**

- Prefer **cursor-based pagination** (a `next` token) over offset pagination, which skips or duplicates records when data changes during extraction.
- Extract incrementally with `updated_since` and an overlap window, then deduplicate on the record ID (see 7.14).
- Save the cursor or high-watermark **after** each page or batch is durably written, so a crash resumes rather than restarts.

```python
def extract_all(session, since):
    params, url = {"updated_since": since, "limit": 500}, "https://api.example.com/v2/orders"
    while url:
        page = get_page(session, url, params).json()
        yield page["data"]
        url, params = page.get("next"), None          # next URL carries the cursor
```

**Throughput and politeness**

- Use a `requests.Session` (connection reuse) or an async client with bounded concurrency.
- Throttle proactively with a token bucket set below the published limit, shared across workers (for example in Redis or via an Airflow pool) so parallel tasks do not collectively exceed it.
- Prefer bulk export endpoints, webhooks or change feeds over polling when the provider offers them.

**Landing the data**

Write raw responses to bronze storage unchanged (with request metadata), then parse into typed tables. This makes reprocessing possible without calling the API again, which matters when API history is limited or calls are expensive.

**Pitfalls:** retrying non-idempotent `POST` calls without idempotency keys, infinite retry loops, ignoring `Retry-After`, not handling token expiry during long extractions, and building custom connectors for common SaaS sources that managed tools (Fivetran, Airbyte, dlt) already maintain.

### 7.10 What is the purpose of a "Staging Area"?

A staging area is an intermediate storage zone where data lands after extraction and before it is integrated into curated, consumer-facing tables. It decouples extraction from transformation, keeps a copy of what was received, and gives the pipeline a place to validate, deduplicate and reconcile data before it can affect the "source of truth" that dashboards and models read.

**What it is used for**

- **Decoupling:** extraction from source systems can run on the source's schedule and within its load windows; transformation runs later, independently, and can be retried without hitting the source again.
- **Validation and quality gates:** schema checks, row counts against source, deduplication, type conversion, before publishing (see 7.16).
- **Reconciliation and audit:** compare staged data with source control totals; keep evidence of what was received and when.
- **Change detection:** compare the staged snapshot with the current target to derive inserts, updates and deletes when the source has no CDC.
- **Performance:** bulk-load into staging with fast loaders (`COPY INTO`), then apply set-based `MERGE` into targets.

**Transient versus persistent staging**

| Type | Behaviour | Pros | Cons |
| --- | --- | --- | --- |
| Transient | Truncated and reloaded every run | Simple, cheap | No history; cannot replay past loads |
| Persistent (persistent staging area, PSA) | Keeps every load, append-only, with load timestamps | Full replay and audit; rebuild downstream models anytime | Storage cost, retention and privacy management |

In lakehouse terms, a persistent staging area is essentially the **bronze layer** (see 4.4), and in dbt projects the `staging` models are a light cleaning layer (renaming, casting) on top of raw sources.

**Design guidance**

- **Keep data close to source form** in staging: type casting and renaming are fine, business logic belongs downstream.
- **Add metadata columns:** load timestamp, batch or run ID, source file name, source system.
- **Restrict access:** staging often contains raw PII and unvalidated data; consumers should not query it directly.
- **Make loads idempotent:** reloading the same batch replaces it (partition by load date or batch ID).
- **Apply retention** to transient and persistent staging, aligned with privacy rules.

**Pitfalls:** staging areas that become an undocumented dumping ground used by analysts; unbounded growth of persistent staging without retention; and "staging" tables that contain business logic, which then has to be reverse-engineered when it breaks.

### 7.11 What is "Change Data Capture" (CDC)?

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

The end-to-end design of a log-based CDC pipeline into a lakehouse (snapshots, ordering, deletes, schema changes) is covered in 7.23.

### 7.12 What is an "SLA" in data engineering, and how do you define and monitor SLAs, SLOs and SLIs for data?

A data SLA is a commitment to the consumers of a dataset about its service level, most often **freshness** ("orders for yesterday are available by 07:00 UTC"), but also completeness, quality and availability. Borrowing from site reliability engineering: **SLIs** are the measurements (hours since last successful update), **SLOs** are internal targets for them (fresh by 07:00 on 99 percent of days per quarter), and the **SLA** is the agreement with consumers, sometimes with consequences, which should be looser than the SLO.

**Dimensions to cover**

| Dimension | Example SLI | Example SLO |
| --- | --- | --- |
| Freshness / timeliness | Time between event and availability, or time of last successful load | Daily table ready by 07:00 UTC on 99 percent of days |
| Completeness | Row count versus source, late-arriving share | At least 99.5 percent of source rows present by publication |
| Quality / validity | Share of quality checks passing | No critical check failures in published data |
| Availability | Query success rate for a serving table or API | 99.9 percent monthly |
| Latency (streaming) | End-to-end lag p95 | Under 60 seconds |

**How to set them**

1. **Start from the consumer's decision:** when does the finance team actually need the numbers? Many "real-time" requirements become "by 9 am" after this conversation.
2. **Tier datasets:** tier 1 (executive, regulatory, customer-facing) gets strict SLOs, on-call and redundancy; tier 3 gets best effort.
3. **Derive pipeline deadlines backwards** from the SLA through the dependency chain, including source delivery times, which you do not control; reflect upstream SLAs in your own.
4. **Write them down** in the catalog or data contract, with owner and escalation path (see 4.27).

**Monitoring**

- Freshness checks on the **data** (latest partition or `max(updated_at)`), not just on job success, because a job that never started produces no failure.
- Alert **before** a breach, when the expected completion time is at risk (an upstream delay predicts a miss), so there is time to act or warn consumers.
- Publish an SLO dashboard per tier, and review breaches in a regular operations meeting.

```python
from datetime import datetime, timezone

def freshness_hours(spark, table: str, ts_col: str) -> float:
    latest = spark.table(table).agg({ts_col: "max"}).collect()[0][0]
    return (datetime.now(timezone.utc) - latest).total_seconds() / 3600

if freshness_hours(spark, "gold.orders_daily", "loaded_at") > 26:
    page("orders_daily freshness SLO breached")
```

In Airflow, deadline alerts (Airflow 3.1+, replacing the old SLA miss feature) or external freshness monitors, dbt source freshness checks, and observability tools all provide this.

**Error budgets:** if a dataset meets its SLO comfortably, the team can take more risk with changes; if it repeatedly breaches, reliability work takes priority over new features.

**Pitfalls:** SLAs promised without measuring current performance; one SLA for everything; ignoring upstream dependencies; and measuring job success instead of data availability.

### 7.13 How do you build reliable data pipelines for ML training and inference, and what makes data "clean and reliable" in this context?

ML pipelines need everything a normal analytics pipeline needs (idempotency, quality checks, monitoring) plus three ML-specific guarantees: **point-in-time correctness** (training features use only information available at prediction time), **training-serving consistency** (the same feature logic produces the same values offline and online), and **reproducibility** (every model can be traced to the exact data snapshot and code that trained it). "Clean and reliable" data is data that is valid, complete, unbiased by leakage, consistent between training and serving, and versioned.

**What "clean and reliable" means for ML**

| Property | What it means | How to enforce |
| --- | --- | --- |
| Valid | Types, ranges, allowed values, no corrupt records | Schema validation at ingestion (pydantic, pandera, Great Expectations) |
| Complete and representative | Missing data handled explicitly; no silent drops that bias the sample | Track drop rates per reason; check class and segment distributions |
| Leakage-free | No future information in features or labels | Point-in-time joins, label cut-offs (see 26.79 and 10.30) |
| Consistent | Same definitions in training and serving | Shared feature code or a feature store (see 26.55 and 26.66) |
| Versioned | Exact training data can be recreated | Table snapshots or time travel, dataset versions logged with the model (see 26.67) |
| Fresh | Features updated within their expected window | Freshness SLOs on feature tables (see 7.12) |
| Monitored | Drift and quality issues caught after deployment | Feature and prediction drift monitoring (see 26.62) |

**Training pipeline**

1. **Ingest and validate** raw data into bronze and silver with quality checks and quarantine of malformed records (counted and reported, not silently dropped).
2. **Build features** with point-in-time correct joins against the label timestamp.
3. **Create the training set** from a fixed snapshot (Delta or Iceberg version, or a materialised dataset), record its version and statistics.
4. **Split** by time or entity to avoid leakage between train and test.
5. **Log** data version, feature code version and data statistics in the experiment tracker with the model.

```python
# Point-in-time join: latest feature value at or before each label time
labels = labels.sort_values("label_time")
features = features.sort_values("feature_time")
training = pd.merge_asof(labels, features,
                         left_on="label_time", right_on="feature_time",
                         by="customer_id", direction="backward",
                         tolerance=pd.Timedelta("30D"))
```

**Inference pipeline**

- **Batch inference:** read features for the scoring date, validate them against training statistics (schema, ranges, null rates), score, write predictions idempotently with model version and timestamp (see 26.84).
- **Online inference:** serve features from an online store populated by the same transformation code; log the features actually used for each prediction so training data for the next model matches serving exactly.
- **Fallbacks:** default values or a fallback model when features are missing or stale, with monitoring of how often fallbacks are used.

**Common failure modes**

- Training-serving skew from re-implementing feature logic in a different language or service.
- Leakage from joining on current dimension values instead of historical ones (see 7.15).
- Silent upstream changes (a unit change, a new category) that do not break schemas but shift distributions.
- Training on data that no longer exists, making the model irreproducible.

**Senior framing:** treat features and training sets as data products with owners, contracts and SLOs; most ML incidents in production are data incidents.

### 7.14 How do you design an idempotent incremental load?

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

### 7.15 How do you handle late-arriving facts and late-arriving dimensions?

The two problems run in opposite directions. A **late-arriving fact** is an event that shows up after the period it belongs to has already been processed. A **late-arriving dimension** (also called an early-arriving fact) is a fact that arrives before the dimension row it references.

**Late-arriving facts**

Example: a mobile app buffers events offline and uploads them two days later.

- **Partition by event time, not load time,** and keep both timestamps (`event_ts`, `ingested_at`).
- **Reprocess a trailing window** of partitions on each run (see 7.14), sized from the observed lateness distribution.
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

### 7.16 What is the Write-Audit-Publish pattern and how do you implement it?

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

### 7.17 What are Airflow executors, and how do you choose one?

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

```text
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

**A senior point:** most production Airflow tasks should be thin. They trigger work in Spark, a warehouse or dbt and wait. In that case the executor matters less than the waiting pattern, which is why deferrable operators (7.18) often save more money than tuning the executor. Also watch the controls that apply to every executor: `parallelism`, `max_active_tasks_per_dag`, pools, and worker concurrency.

### 7.18 What is the difference between sensors and deferrable operators in Airflow?

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
- **Often the best sensor is no sensor.** If the upstream is another DAG or a dataset you own, use asset-based scheduling (7.20) instead of polling for it.

```python
wait = S3KeySensor(
    task_id="wait_for_export",
    bucket_key="s3://raw/orders/{{ ds }}/_SUCCESS",
    deferrable=True,
    timeout=6 * 60 * 60,
)
```

### 7.19 What are the most important Airflow best practices for production DAGs?

Treat a DAG file as configuration that is parsed constantly and a task as a transaction that will be retried. Most production Airflow incidents trace back to breaking one of those two assumptions.

**1. Idempotent, deterministic tasks**

- A rerun of a task for the same data interval must produce the same result. Use partition overwrite or `MERGE`, never a blind `INSERT` (see 7.14).
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

- Airflow coordinates; Spark, the warehouse or dbt compute. Use deferrable operators for long waits (7.18). Run conflicting dependencies with `KubernetesPodOperator`, `@task.virtualenv` or `@task.docker`.

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

### 7.20 Compare task-based orchestration (Airflow) with asset-based orchestration (Dagster, Airflow assets).

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

### 7.21 How do dbt incremental models and snapshots work, and what are their pitfalls?

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

### 7.22 How do you test data pipelines and set up CI/CD for them?

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
3. In production: data tests and freshness checks gate the publish (Write-Audit-Publish, 7.16), and anomaly monitoring catches what the fixed tests missed.

**Pitfalls:** tests that only ever run on clean fixture data, CI that builds the whole project (slow and expensive, so people skip it), production credentials in CI, and treating test failures as noise. Every test should either block or have a named owner who responds to it.

### 7.23 How would you build a log-based CDC pipeline from an OLTP database into a lakehouse, and what is the transactional outbox pattern?

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
