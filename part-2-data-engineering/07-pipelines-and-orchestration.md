# Chapter 7. Pipelines and Orchestration

_Part II — Data Engineering_

ETL and ELT design, orchestration with Airflow, idempotency, backfills and building reliable pipelines for analytics and ML.

This chapter contains **19 questions**.

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

---

← [Chapter 6. Distributed Processing and Streaming](06-distributed-processing-and-streaming.md) · [Table of contents](../README.md) · [Chapter 8. Machine Learning Fundamentals](../part-3-machine-learning/08-machine-learning-fundamentals.md) →
