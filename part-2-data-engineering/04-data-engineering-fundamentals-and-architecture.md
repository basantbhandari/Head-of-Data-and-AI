# Chapter 4. Data Engineering Fundamentals and Architecture

_Part II — Data Engineering_

Warehouses, lakes and lakehouses, storage formats, cloud data platforms and how to choose and migrate to them, data quality, contracts, observability, security and governance. These questions test whether a candidate can design a data platform, not just operate one.

This chapter contains **44 questions**.

## Contents

- **4.1** [What is the fundamental goal of Data Engineering?](#41-what-is-the-fundamental-goal-of-data-engineering)
- **4.2** [Explain the difference between a Data Warehouse, a Data Lake and a Data Lakehouse.](#42-explain-the-difference-between-a-data-warehouse-a-data-lake-and-a-data-lakehouse)
- **4.3** [What is the CAP Theorem, and how does it affect your choice of database?](#43-what-is-the-cap-theorem-and-how-does-it-affect-your-choice-of-database)
- **4.4** [Describe the "Medallion Architecture."](#44-describe-the-medallion-architecture)
- **4.5** [What is "Data Lineage" and why is it critical?](#45-what-is-data-lineage-and-why-is-it-critical)
- **4.6** [Explain Batch vs. Stream processing.](#46-explain-batch-vs-stream-processing)
- **4.7** [What is "Idempotency" in a data pipeline?](#47-what-is-idempotency-in-a-data-pipeline)
- **4.8** [What is the difference between Structured, Semi-Structured, and Unstructured data?](#48-what-is-the-difference-between-structured-semi-structured-and-unstructured-data)
- **4.9** [What is a "Single Point of Failure" (SPOF) in a data system?](#49-what-is-a-single-point-of-failure-spof-in-a-data-system)
- **4.10** [What is the "Small File Problem" in distributed storage, and how does compaction fix it?](#410-what-is-the-small-file-problem-in-distributed-storage-and-how-does-compaction-fix-it)
- **4.11** [Why is Parquet preferred over CSV for data engineering?](#411-why-is-parquet-preferred-over-csv-for-data-engineering)
- **4.12** [What is the difference between Avro and Parquet?](#412-what-is-the-difference-between-avro-and-parquet)
- **4.13** [Explain "Object Storage" (S3) vs. "Block Storage" (EBS).](#413-explain-object-storage-s3-vs-block-storage-ebs)
- **4.14** [What is Snowflake's "Unique Architecture"?](#414-what-is-snowflakes-unique-architecture)
- **4.15** [What is a "Partition Key" in S3 or HDFS, and how do you choose one?](#415-what-is-a-partition-key-in-s3-or-hdfs-and-how-do-you-choose-one)
- **4.16** [What are managed and serverless services in the context of data pipelines, and how do you choose?](#416-what-are-managed-and-serverless-services-in-the-context-of-data-pipelines-and-how-do-you-choose)
- **4.17** [What is the "Modern Data Stack"?](#417-what-is-the-modern-data-stack)
- **4.18** [What is a "Data Mesh"?](#418-what-is-a-data-mesh)
- **4.19** [What is Infrastructure as Code (IaC), and how do you apply it to a data platform?](#419-what-is-infrastructure-as-code-iac-and-how-do-you-apply-it-to-a-data-platform)
- **4.20** [What is "Data Tiering"?](#420-what-is-data-tiering)
- **4.21** [Explain the difference between "Vertical" and "Horizontal" Scaling.](#421-explain-the-difference-between-vertical-and-horizontal-scaling)
- **4.22** [What is a "VPC" and why do data engineers care?](#422-what-is-a-vpc-and-why-do-data-engineers-care)
- **4.23** [What is "Data Governance"?](#423-what-is-data-governance)
- **4.24** [How do you handle PII (Personally Identifiable Information)?](#424-how-do-you-handle-pii-personally-identifiable-information)
- **4.25** [What is "Data Quality Profiling", and how do you build data quality into pipelines?](#425-what-is-data-quality-profiling-and-how-do-you-build-data-quality-into-pipelines)
- **4.26** [What is "dbt" and why is it so popular?](#426-what-is-dbt-and-why-is-it-so-popular)
- **4.27** [Explain the concept of "Data Contracts."](#427-explain-the-concept-of-data-contracts)
- **4.28** [What is the difference between "Data Observability" and "Monitoring"?](#428-what-is-the-difference-between-data-observability-and-monitoring)
- **4.29** [How do you implement "Unit Testing" for a data transformation?](#429-how-do-you-implement-unit-testing-for-a-data-transformation)
- **4.30** [How do you handle "Schema Drift" and schema evolution automatically?](#430-how-do-you-handle-schema-drift-and-schema-evolution-automatically)
- **4.31** [Where do you see the future of Data Engineering?](#431-where-do-you-see-the-future-of-data-engineering)
- **4.32** [What are open table formats (Apache Iceberg, Delta Lake, Apache Hudi), and what problems do they solve on top of Parquet files in object storage?](#432-what-are-open-table-formats-apache-iceberg-delta-lake-apache-hudi-and-what-problems-do-they-solve-on-top-of-parquet-files-in-object-storage)
- **4.33** [How do snapshots, time travel and optimistic concurrency work in Iceberg or Delta Lake?](#433-how-do-snapshots-time-travel-and-optimistic-concurrency-work-in-iceberg-or-delta-lake)
- **4.34** [What is the difference between copy-on-write and merge-on-read tables, and when would you pick each?](#434-what-is-the-difference-between-copy-on-write-and-merge-on-read-tables-and-when-would-you-pick-each)
- **4.35** [Compare Apache Iceberg, Delta Lake and Apache Hudi. How would you choose between them?](#435-compare-apache-iceberg-delta-lake-and-apache-hudi-how-would-you-choose-between-them)
- **4.36** [How do you implement the GDPR "right to be forgotten" in a data lake or lakehouse?](#436-how-do-you-implement-the-gdpr-right-to-be-forgotten-in-a-data-lake-or-lakehouse)
- **4.37** [How do you implement fine-grained access control (row-level security, column masking, tag-based policies) on a data platform?](#437-how-do-you-implement-fine-grained-access-control-row-level-security-column-masking-tag-based-policies-on-a-data-platform)
- **4.38** [What is a semantic (metrics) layer and why do organisations adopt one?](#438-what-is-a-semantic-metrics-layer-and-why-do-organisations-adopt-one)
- **4.39** [How do you control and optimise cost on a cloud data platform (warehouse and lakehouse FinOps)?](#439-how-do-you-control-and-optimise-cost-on-a-cloud-data-platform-warehouse-and-lakehouse-finops)
- **4.40** [How would you design a real-time clickstream analytics pipeline for a large e-commerce site?](#440-how-would-you-design-a-real-time-clickstream-analytics-pipeline-for-a-large-e-commerce-site)
- **4.41** [How would you design a data platform for a mid-size company from scratch, and what would you build vs buy?](#441-how-would-you-design-a-data-platform-for-a-mid-size-company-from-scratch-and-what-would-you-build-vs-buy)
- **4.42** [How would you design an enterprise ontology, and how do data platforms and AI agents use it?](#442-how-would-you-design-an-enterprise-ontology-and-how-do-data-platforms-and-ai-agents-use-it)
- **4.43** [How do you choose between Snowflake, BigQuery, Databricks and Redshift (or Microsoft Fabric) as the core data platform?](#443-how-do-you-choose-between-snowflake-bigquery-databricks-and-redshift-or-microsoft-fabric-as-the-core-data-platform)
- **4.44** [How would you migrate a legacy on-premises data warehouse or Hadoop platform to the cloud?](#444-how-would-you-migrate-a-legacy-on-premises-data-warehouse-or-hadoop-platform-to-the-cloud)

---

### 4.1 What is the fundamental goal of Data Engineering?

The goal of data engineering is to deliver **trustworthy, timely, well-modelled data to the people and systems that make decisions with it**, at a cost the business can justify. Pipelines are the means, not the goal. A data engineering team is judged by whether analysts, data scientists, ML models, AI applications and operational systems can rely on the data, not by how many pipelines it runs.

**What that means in practice**

- **Availability and freshness:** the data is there when it is needed, inside an agreed SLA (see 7.12).
- **Correctness and quality:** numbers reconcile with source systems, duplicates and nulls are handled, and definitions are consistent.
- **Usability:** data is modelled for its consumers (dimensional models, feature tables, semantic layer), documented and discoverable.
- **Security and compliance:** access is controlled, PII is protected, retention and deletion obligations are met.
- **Efficiency:** compute and storage costs scale sensibly with data volume and value.
- **Changeability:** new sources, columns and use cases can be added quickly without breaking what exists.

**The data engineering lifecycle**

```text
Generation (apps, SaaS, devices)
  -> Ingestion (batch, CDC, streaming)
  -> Storage (lake, warehouse, lakehouse)
  -> Transformation and modelling (bronze -> silver -> gold, dbt, Spark)
  -> Serving (BI, semantic layer, ML features, reverse ETL, APIs, RAG)
Undercurrents: security, data management and governance, DataOps, orchestration, architecture, software engineering
```

**How the role has changed**

- From hand-written ETL on on-premises clusters to assembling managed services and writing the parts that carry business logic.
- From "move the data" to **data products**: owned, documented, tested datasets with SLAs.
- From serving dashboards only to serving ML and generative AI: feature pipelines, training data, document ingestion for retrieval, and evaluation datasets.

**A senior answer adds the business lens:** good data engineering shortens the time from question to trusted answer, reduces incidents where leaders see wrong numbers, and makes the cost per insight visible. A useful one-line definition for interviews: "Data engineering builds and runs the systems that turn raw data into reliable data products, with quality, security and cost treated as engineering requirements."

### 4.2 Explain the difference between a Data Warehouse, a Data Lake and a Data Lakehouse.

A **data warehouse** stores cleaned, structured data in managed tables optimised for SQL analytics, with schema enforced on write. A **data lake** stores raw data of any format as files in cheap object storage, with schema applied on read. A **lakehouse** keeps the lake's open files and cheap storage but adds a transactional table format and catalog on top, so the same data supports warehouse-style SQL, BI, data science and ML without a second copy.

**Comparison**

| Aspect | Data warehouse | Data lake | Lakehouse |
| --- | --- | --- | --- |
| Data | Structured, curated | Raw, any format (JSON, CSV, Parquet, images, logs) | All of these; tables in open formats |
| Schema | On write, enforced | On read, often unenforced | Enforced on tables, with evolution |
| Storage | Proprietary, managed by the vendor | Object storage (S3, ADLS, GCS) | Object storage with Delta, Iceberg or Hudi |
| Transactions | ACID | None (files only) | ACID through the table format (see 4.32) |
| Engines | The warehouse's own | Many (Spark, Presto, Python) | Many engines over the same tables |
| Performance | Excellent for SQL out of the box | Depends on file layout; often poor | Near warehouse with good layout and a modern engine |
| ML and unstructured data | Weak | Strong | Strong |
| Typical risk | Cost and lock-in | Becomes a "data swamp" without governance | Table maintenance and catalog choice |
| Examples | Snowflake, BigQuery, Redshift, Synapse | Raw S3 or ADLS zones | Databricks, Iceberg with Trino, Snowflake or BigQuery on Iceberg |

**Why the lakehouse emerged**

The classic two-tier design loaded raw data into a lake, then copied a transformed subset into a warehouse for BI. That meant two copies, two pipelines, staleness between them, and ML teams working on different data from BI teams. Table formats made object storage transactional, so one copy can serve both.

**The lines are blurring**

- Warehouses now read and write open table formats (Snowflake and BigQuery support Iceberg tables; Redshift queries data in S3).
- Lakehouse platforms now offer warehouse features (serverless SQL, materialised views, fine-grained access control, semantic layers).
- So the real decision is less "warehouse or lake" and more: **which engines, which open format, which catalog, and who operates it.**

**How to choose**

- **Mostly BI and SQL users, modest data science, small platform team:** a cloud warehouse is the fastest path to value.
- **Heavy ML, large semi-structured or unstructured data, several engines, strong desire to avoid lock-in:** a lakehouse on Iceberg or Delta.
- **Common in practice:** raw and bronze data in the lake (cheap, replayable), curated gold tables served by a warehouse or lakehouse SQL engine, all on open formats where possible.

**Pitfalls:** a lake without catalog, ownership and quality checks turns into a swamp; a warehouse used to store everything raw becomes very expensive; and "lakehouse" without table maintenance (compaction, snapshot expiry) slowly degrades (see 4.10).

### 4.3 What is the CAP Theorem, and how does it affect your choice of database?

The CAP theorem says that when a **network partition** separates the nodes of a distributed system, the system must choose between **consistency** (every read sees the latest write, or gets an error) and **availability** (every request to a non-failed node gets a non-error response). Partitions cannot be ruled out in real networks, so the practical choice is CP or AP behaviour during a partition. The PACELC extension adds the more frequent everyday trade-off: **else**, when there is no partition, you trade **latency** against consistency.

**Precise definitions matter**

- **C** in CAP is linearizability, a strong, specific guarantee. It is not the C in ACID (which is about integrity constraints).
- **A** means every non-failed node responds, which is stricter than "high availability" in the SLA sense.
- **P** is not optional. A system that "chooses CA" is a single-node system or one that assumes the network never fails.

**PACELC classification (simplified)**

| System | During a partition | Normal operation | Notes |
| --- | --- | --- | --- |
| Single-primary PostgreSQL / MySQL | Writes stop on the minority side (CP-like) | Consistency (reads on primary) | Async replicas give stale reads |
| Spanner, CockroachDB, etcd, ZooKeeper | CP | Consistency, paying latency for consensus | Strong guarantees, cross-region writes cost round trips |
| Cassandra, DynamoDB (default), Riak | AP | Latency | Tunable per request (see 6.25) |
| MongoDB (majority write concern) | CP | Configurable | Read and write concerns move it along the spectrum |

**How it affects database choice**

- **Money, inventory, uniqueness, anything with invariants** ("never oversell", "no double payment"): prefer CP and strong consistency, and accept that some requests fail or wait during a fault.
- **User-facing reads where staleness is tolerable** (feeds, recommendations, product catalogues, telemetry): prefer AP and low latency, and design for convergence.
- **Multi-region active-active writes:** either pay consensus latency (Spanner-style) or accept conflicts and resolve them (last-write-wins, CRDTs).
- **Mixed systems** are normal: a CP store for orders and payments, AP stores and caches for read-heavy views, connected by CDC or events.

**What data engineers should take from it**

- Analytical stores fed from operational systems are eventually consistent with the source by design. Freshness SLAs and reconciliation checks make that explicit.
- Reading from replicas in extraction jobs can miss recent writes, so use overlap windows (see 7.14).
- Most incidents are not from partitions but from latency and replication lag, which is why PACELC is the more useful lens in design reviews.

**Interview tip:** avoid "pick two of three". Say: "Partitions happen, so the question is what the system does during one, and what latency it pays for consistency the rest of the time", then tie it to the business invariant at stake.

### 4.4 Describe the "Medallion Architecture."

The medallion architecture organises lakehouse data into layers of increasing quality and business readiness: **bronze** (raw, as ingested), **silver** (cleaned, conformed, deduplicated) and **gold** (business-level models and aggregates for consumption). Its value is not the names but the contract each layer gives: replayability in bronze, trustworthy entities in silver, and fit-for-purpose data products in gold.

**What each layer contains and guarantees**

| Layer | Contents | Typical operations | Guarantee to consumers |
| --- | --- | --- | --- |
| Bronze | Source data as received, plus ingestion metadata (`_ingested_at`, source file, offset) | Append-only loads, CDC change logs, schema captured or rescued | Complete and replayable history; no business logic |
| Silver | Cleaned, typed, deduplicated entities and events; conformed keys; PII handled | Parsing, validation, dedup, `MERGE` of CDC, SCD handling, joins to reference data | One trustworthy, documented version of each entity |
| Gold | Dimensional models, aggregates, feature tables, data products per use case | Business rules, metrics, star schemas, denormalisation | Ready for BI, semantic layer, ML and applications, with SLAs |

```text
Sources -> [Bronze: raw, append-only] -> [Silver: clean entities] -> [Gold: marts, features, metrics]
                 ^ replay from here when logic changes            -> BI, semantic layer, ML, reverse ETL
```

**A PySpark sketch**

```python
from pyspark.sql import functions as F, Window

bronze = spark.readStream.table("bronze.orders_raw")

w = Window.partitionBy("order_id").orderBy(F.col("updated_at").desc())
silver_batch = (bronze
    .withColumn("amount", F.col("amount").cast("decimal(18,2)"))
    .filter(F.col("order_id").isNotNull()))
# foreachBatch: dedup to latest per order_id, then MERGE into silver.orders (see 6.38)
```

**Design decisions that matter**

- **Keep bronze immutable and long-lived** (subject to privacy rules). It is what lets you fix a bug and rebuild silver and gold.
- **Quality gates between layers:** checks that block or quarantine bad data before it reaches silver and gold (see 7.16).
- **Who owns what:** platform teams often own bronze and silver ingestion; domain teams own gold data products.
- **Silver is the hardest layer.** Conforming keys across systems, handling late data and deletes, and choosing SCD types is where most of the work and most bugs are.

**Pitfalls**

- **Layers as folders only:** three copies of the same data with no extra guarantees, just more cost.
- **Too many layers** ("silver_v2_clean_final") or skipping silver so every gold table re-implements cleaning.
- **Treating it as a mandate for every source:** small reference tables may go straight to a curated table.
- **Storage and compute cost** of copying very large raw data three times; use views or incremental processing where appropriate.

**Relation to other models:** it maps naturally onto classic staging, integration and presentation layers (see 7.10), onto dbt's staging, intermediate and marts folders, and onto data products in a data mesh (see 4.18).

### 4.5 What is "Data Lineage" and why is it critical?

Data lineage records where data comes from, how it is transformed and where it goes: which source tables, jobs and columns feed a given dashboard field, model feature or file. It is critical because without it you cannot answer three questions quickly: **what breaks if I change this** (impact analysis), **why is this number wrong** (root cause), and **where does this personal data live** (compliance).

**Levels of lineage**

| Level | Example | Use |
| --- | --- | --- |
| System / dataset | CRM -> `bronze.accounts` -> `gold.revenue` -> dashboard | Ownership, high-level impact |
| Job / run | Which DAG run and code version produced partition `2026-11-01` | Debugging, reproducibility, audit |
| Column | `gold.revenue.net_amount` is derived from `orders.amount` minus `refunds.amount` | Precise impact analysis, PII propagation |

**How lineage is captured**

- **Parsing code and queries:** dbt manifests, SQL parsers (for example `sqlglot`), query history in Snowflake, BigQuery or Databricks system tables.
- **Runtime instrumentation:** the OpenLineage standard emits run, job and dataset events from Airflow, Spark, dbt and Flink to a backend (Marquez, DataHub, OpenMetadata, vendor catalogs).
- **Platform-native:** Unity Catalog, Dataplex and Purview capture lineage automatically for workloads that run on the platform.
- **Asset-based orchestrators** (Dagster, Airflow assets) make lineage explicit in the pipeline definition (see 7.20).

Manual documentation goes stale within weeks; lineage has to be generated from the systems themselves.

**Why it matters, with examples**

- **Impact analysis:** before dropping a column, list every downstream model, dashboard and ML feature that uses it, and their owners.
- **Root cause analysis:** a KPI changed overnight; lineage shows which upstream job ran with new code or late data.
- **Compliance:** GDPR erasure and access requests need every location of a person's data (see 4.36); column-level lineage shows how PII propagates.
- **Trust:** users can see where a metric comes from and when it was last refreshed.
- **ML reproducibility:** link a model version to the exact training data snapshot and feature code (see 26.67).
- **Cost:** find expensive tables that nobody reads.

**Pitfalls**

- Lineage that stops at tool boundaries (notebooks, Python scripts, exports to spreadsheets, reverse ETL).
- Coarse table-level lineage where column-level is needed for PII.
- No ownership metadata attached, so you know what breaks but not whom to tell.

**Senior view:** treat lineage as part of the platform. Emit OpenLineage events by default from every orchestrated job, combine it with a catalog, ownership and quality results, and use it in change management (a pull request that changes a model lists its downstream consumers automatically).

### 4.6 Explain Batch vs. Stream processing.

Batch processing runs over a bounded dataset on a schedule or trigger (every night, every hour) and produces results for that whole slice. Stream processing runs continuously over unbounded data and updates results as each event or small batch arrives. The choice is about **how fresh the result must be** versus the cost and complexity you will pay, and many real systems use both.

**Comparison**

| Aspect | Batch | Stream |
| --- | --- | --- |
| Input | Bounded (a day, a file, a table snapshot) | Unbounded event stream |
| Latency | Minutes to hours | Milliseconds to seconds |
| Correctness model | Sees all data for the window at once; easy to recompute | Must handle out-of-order and late data with event time and watermarks |
| State | Recomputed per run | Long-lived state (windows, joins, dedup) that must be checkpointed |
| Failure handling | Rerun the job for the interval | Checkpoints, replay, exactly-once sinks |
| Cost profile | Compute only while running; easy to use spot capacity | Always-on resources |
| Operational complexity | Lower | Higher (on-call, state upgrades, lag monitoring) |
| Tools | Spark, dbt, warehouse SQL, Airflow | Kafka, Flink, Spark Structured Streaming, Kafka Streams |

**The middle ground is often the right answer**

- **Micro-batch and incremental processing:** run every 1 to 15 minutes over only new data (Spark Structured Streaming with `availableNow` triggers, dbt incremental models, Delta Live Tables, Snowflake dynamic tables). Most "real-time dashboard" requirements are satisfied this way at a fraction of the cost of true streaming.
- **Unified engines:** Spark and Flink run the same logic in batch and streaming modes, which removes the two-codebase problem (see 6.43 on Lambda and Kappa).

**How to choose**

1. **Ask what decision the data drives and how fast it must happen.** Fraud blocking, real-time pricing, operational alerting and online features need streaming. Daily reporting does not.
2. **Check the source.** If data arrives in nightly files, streaming adds nothing.
3. **Consider correctness needs.** Financial reporting usually needs a batch reconciliation even when a streaming figure exists.
4. **Weigh team skills and on-call capacity.** Stateful streaming is harder to operate.
5. **Estimate cost.** Always-on clusters versus scheduled jobs.

**Pitfalls**

- Building streaming because it sounds modern, for a dashboard nobody checks more than once a day.
- Assuming batch is free of time problems: late-arriving data affects batch too (see 7.15).
- Running batch every 5 minutes with full refreshes, which costs more than incremental or streaming.

**Interview framing:** "I start from the freshness SLA. If it is hours, batch; if it is minutes, incremental micro-batch; if it is seconds and drives an automated decision, streaming with event-time semantics and a batch reconciliation where correctness matters."

### 4.7 What is "Idempotency" in a data pipeline?

A pipeline step is idempotent if running it once or several times with the same input leaves the target in the same state. It matters because in production every step **will** be retried: orchestrator retries, manual reruns, backfills, at-least-once message delivery, and Spark task retries. If a retry appends the same rows again or sends the same email twice, retries become data corruption.

**Where non-idempotency hides**

- `INSERT INTO ... SELECT` or `mode("append")` writes that are rerun after a partial failure.
- Logic based on `now()` or "latest" data, so a rerun processes a different window.
- Auto-increment surrogate keys generated on each run.
- Side effects: API calls, notifications, payments, messages to Kafka.
- Accumulating updates such as `balance = balance + x`.

**Patterns that make steps idempotent**

| Pattern | How it works | Best for |
| --- | --- | --- |
| Partition overwrite | Recompute and atomically replace a whole partition (`replaceWhere`, dynamic partition overwrite, `INSERT OVERWRITE`) | Event facts partitioned by date |
| `MERGE` / upsert on a business key | Insert new keys, update existing ones; dedup the batch first | Mutable entities, CDC |
| Deterministic windows | Derive the window from the run's logical date, not the clock | Scheduled batch |
| Deterministic keys | Hash of natural key columns instead of random or sequence IDs | Surrogate keys, event IDs |
| Dedup on event ID | Keep a key per event and drop repeats | Streaming, at-least-once sources |
| Idempotency keys for side effects | The receiver remembers processed keys | APIs, payments, notifications (see 25.23) |
| Write-then-swap | Write to staging, validate, then atomically publish (see 7.16) | Whole-table rebuilds |

```python
# Idempotent daily load: overwrite exactly the partition for this run's date
(daily_df.write.format("delta")
    .mode("overwrite")
    .option("replaceWhere", f"event_date = '{run_date}'")
    .saveAsTable("silver.events"))
```

**Testing it**

Run the step twice for the same interval in CI or staging and assert that row counts and checksums are identical. It is one of the cheapest and most valuable pipeline tests.

**Related concepts**

- **Determinism:** same inputs, same outputs. Idempotency is about the effect on the target; you usually need both.
- **Exactly-once:** in streaming, idempotent sinks are the most common way to get effectively-once results (see 6.23).
- Detailed incremental-load design (watermarks, lookback windows, `MERGE` guards) is in 7.14.

**Senior point:** make idempotency a code review rule and a platform default (templates that only offer overwrite or merge writes), because the failure mode is silent: duplicated revenue that nobody notices for weeks.

### 4.8 What is the difference between Structured, Semi-Structured, and Unstructured data?

The difference is how much of the schema is fixed and known in advance. **Structured** data fits a fixed schema of typed columns. **Semi-structured** data carries its own structure in tags or keys (JSON, XML, Avro), but the fields can vary between records and nest. **Unstructured** data (text, PDFs, images, audio, video) has no data model that a query engine can use directly; meaning must be extracted first.

**Comparison**

| Aspect | Structured | Semi-structured | Unstructured |
| --- | --- | --- | --- |
| Examples | Orders table, ledger, CRM records | JSON events, API responses, logs, XML, Parquet with nested types | Contracts, email bodies, call recordings, images, scans |
| Schema | Fixed, enforced on write | Self-describing, flexible, often evolving | None usable directly |
| Storage | Relational tables, warehouse tables | Object storage, document databases, `VARIANT` or JSON columns, nested Parquet | Object storage, with metadata in tables |
| Querying | SQL | SQL with JSON functions or after flattening | Requires extraction: OCR, speech-to-text, NLP, embeddings, vision models |
| Share of enterprise data | Minority | Large and growing | Often estimated as the majority |

**How a data platform handles each**

- **Structured:** model it (dimensional or normalised), enforce types and constraints, apply SCD logic.
- **Semi-structured:** land raw JSON in bronze (keep the original payload), then parse and flatten the stable fields into typed silver columns while keeping the rest in a `VARIANT` or map column. Watch for schema drift (see 4.30), deeply nested arrays that explode row counts, and inconsistent types for the same key.
- **Unstructured:** store the files in object storage, catalogue them in a table with metadata (source, owner, hash, access rights, timestamps), and run extraction pipelines that produce structured outputs: extracted fields, text chunks and embeddings for retrieval (see Chapter 20), labels for ML.

```python
from pyspark.sql import functions as F

raw = spark.read.json("s3://landing/events/2026/11/17/")
silver = raw.select(
    F.col("event_id"),
    F.to_timestamp("ts").alias("event_time"),
    F.col("user.id").alias("user_id"),
    F.col("context.device.os").alias("os"),
    F.to_json("properties").alias("properties_json"))   # keep the variable part
```

**Senior points**

- Generative AI has made unstructured data a first-class input: document pipelines now need the same lineage, access control, versioning and quality checks as tables.
- Governance is harder for unstructured data: PII hides in free text and images, so classification needs ML-based scanning.
- Semi-structured flexibility pushes the schema problem downstream; data contracts with producers reduce it.

### 4.9 What is a "Single Point of Failure" (SPOF) in a data system?

A single point of failure is any component whose failure stops the whole system or a critical function, because nothing takes over. In data platforms SPOFs are rarely only servers: they are also a single orchestrator scheduler, one Kafka broker for a topic with replication factor 1, a shared service account, one region, one engineer who understands a pipeline, or one upstream file drop that everything depends on.

**Typical SPOFs in a data platform**

| Layer | SPOF example | Mitigation |
| --- | --- | --- |
| Storage | Database with no replica; Kafka topic with RF=1; HDFS NameNode without HA | Replication, RF=3 with `min.insync.replicas=2`, HA pairs |
| Compute and orchestration | One Airflow scheduler or triggerer; one metastore database | Multiple schedulers, managed Airflow, HA metastore and backups |
| Spark jobs | The driver (job dies with it) | Orchestrator retries, idempotent steps, checkpoints |
| Network and cloud | One availability zone or region; one NAT gateway | Multi-AZ by default; multi-region only where the business case justifies it |
| Identity and secrets | One expiring service principal or API key used everywhere | Per-workload identities, rotation, expiry alerts |
| External dependencies | One vendor connector or SaaS API | Retries, buffering, a fallback path, contract SLAs |
| Data | One source table feeding hundreds of models | Contracts, quality gates, fast detection |
| People | One engineer who knows the pipeline | Documentation, runbooks, on-call rotation, pairing |

**How to find them**

- Draw the dependency graph of a critical data product end to end and ask "what if this disappears for an hour, a day, for good?".
- Review past incidents: SPOFs show up as the same root cause repeating.
- Run game days: kill a broker, expire a credential in staging, and see what happens.

**How to address them, proportionately**

- **Redundancy** for components whose outage is costly (replicas, multi-AZ, HA schedulers).
- **Graceful degradation:** dashboards show the last good data with a staleness banner instead of failing; streaming consumers buffer in Kafka.
- **Fast recovery** instead of redundancy where outages are tolerable: backups that are tested by restoring, infrastructure as code to rebuild, documented runbooks.
- **Blast-radius reduction:** separate critical and experimental workloads, use separate credentials and quotas.

**Senior framing:** removing every SPOF is expensive. Tie the investment to the data product's tier and its recovery time and recovery point objectives (RTO and RPO). A tier-1 regulatory feed deserves multi-AZ, HA and on-call; an internal exploratory dataset can tolerate a day of downtime.

### 4.10 What is the "Small File Problem" in distributed storage, and how does compaction fix it?

The small file problem is the performance and cost penalty of storing a dataset as very many small files (kilobytes to a few MB) instead of fewer large ones (roughly 128 MB to 1 GB). Every file carries fixed costs: metadata entries, open and list calls, a task to schedule, footer reads. With millions of small files, those fixed costs dominate the useful work. **Compaction** is the maintenance process that rewrites small files into larger, well-organised ones.

**Why small files hurt**

- **Metadata pressure:** in HDFS each file costs NameNode memory (see 6.20); in table formats, file counts inflate transaction logs and manifests, slowing query planning.
- **Object storage overhead:** each file needs separate GET requests, and listing large prefixes is slow and billed.
- **Scheduling overhead:** Spark creates at least one task per small file, so thousands of tasks each process a few KB.
- **Worse compression and skipping:** small Parquet files have small row groups and less effective encoding and statistics.
- **Streaming and CDC writes** create many tiny files and delete files, which merge-on-read tables then have to reconcile at read time (see 4.34).

**Common causes**

- Streaming or micro-batch jobs committing every few seconds.
- Over-partitioning (by hour and customer, for example), so each partition gets a tiny file per write.
- Many Spark tasks each writing one file per output partition (200 tasks times 365 dates).
- Frequent small `MERGE` or `UPDATE` operations.

**Preventing them at write time**

- Partition only by low-cardinality columns that queries filter on (see 4.15).
- `repartition` by the partition column or use the `REBALANCE` hint before writing (see 6.9).
- Optimised writes and auto-compaction in Delta, `write.distribution-mode` and target file size in Iceberg.
- Longer trigger intervals for streaming sinks where latency allows.

**Compaction**

- **Bin-packing:** combine small files into target-sized files without changing order. Cheap and the usual default.
- **Sort or clustering compaction:** rewrite while sorting or Z-ordering by query columns, which also improves data skipping (see 6.19).
- **Merge-on-read compaction:** fold delete files, deletion vectors or log files back into base files.

```python
# Delta Lake: compact recent partitions (bin-packing), optionally with clustering
spark.sql("OPTIMIZE silver.events WHERE event_date >= current_date() - INTERVAL 2 DAYS")

# Iceberg: rewrite small files and position deletes
spark.sql("""CALL lake.system.rewrite_data_files(
               table => 'silver.events',
               options => map('target-file-size-bytes', '536870912',
                              'min-input-files', '5'))""")
spark.sql("CALL lake.system.rewrite_position_delete_files('silver.events')")
```

**Operating compaction**

- **Schedule it per table** based on write patterns: hourly for streaming tables, daily for batch tables; only on recently written partitions.
- **Avoid conflicts:** compaction commits can conflict with concurrent writers (see 4.33). Compact partitions that are no longer being written, and use platform features that let compaction yield to writers.
- **Pair it with cleanup:** old files remain until snapshots are expired and `VACUUM` or orphan-file removal runs, so storage does not drop until then.
- **Monitor** file counts and average file size per partition, delete-file counts, and query planning time; alert when they drift.

**Kafka log compaction is a different thing:** it keeps the latest record per key in a topic (see 6.41). The shared name causes confusion in interviews, so say which one you mean.

### 4.11 Why is Parquet preferred over CSV for data engineering?

Parquet is a binary, columnar, compressed, self-describing format with embedded statistics. Analytical queries usually read a few columns from many rows, and Parquet lets an engine read only those columns, decode them efficiently, and skip whole blocks whose statistics rule them out. CSV is row-oriented text with no types, no schema, no statistics and poor compression, so every query must read and parse every byte.

**How Parquet is laid out**

```text
File
 ├── Row group 1 (e.g. 128 MB of rows)
 │     ├── Column chunk "user_id"  -> pages (dictionary + data pages)
 │     ├── Column chunk "amount"   -> pages
 │     └── ...
 ├── Row group 2 ...
 └── Footer: schema, row group locations, per-column min/max/null counts,
             optional Bloom filters and page indexes
```

**Why it is faster and cheaper**

| Feature | Effect |
| --- | --- |
| Columnar storage | Read 3 of 100 columns and read roughly 3 percent of the bytes (projection pushdown) |
| Encodings: dictionary, run-length, delta, bit-packing | Repetitive column values shrink dramatically before compression |
| Compression per column (Snappy, ZSTD, GZIP) | Similar values compress well; commonly 5 to 10 times smaller than CSV |
| Min/max statistics and Bloom filters | Skip row groups and pages that cannot match a filter (predicate pushdown, see 6.17) |
| Typed schema in the footer | No parsing ambiguity for dates, decimals, nulls; no type inference pass |
| Nested types | Structs, lists and maps stored natively |
| Splittable | Engines parallelise across row groups |

**Where CSV still makes sense**

- Exchanging small files with humans, spreadsheets or legacy systems.
- Very simple one-off exports. Even then, quoting, delimiters inside values, encodings and header drift cause frequent bugs.

**Parquet pitfalls**

- **Small files** destroy most benefits (see 4.10).
- **Row group size:** too small weakens statistics and compression; too large increases memory needed to write.
- **Statistics only help when data is clustered** by the filter column (see 6.19).
- **Immutable files:** updating rows means rewriting files, which is why table formats like Delta and Iceberg sit on top (see 4.32).
- **Schema evolution across files** must be managed (column renames, type changes).
- **Not ideal for row-at-a-time access** or high-frequency small appends; use a database or Avro for those.

```python
import pyarrow.parquet as pq
meta = pq.ParquetFile("events.parquet").metadata
print(meta.num_row_groups, meta.row_group(0).column(2).statistics)
```

**Related formats:** ORC is similar and strong in the Hive ecosystem; Avro is row-oriented and suited to streaming and messages (see 4.12); Arrow is the in-memory columnar format that engines use to exchange data without serialisation.

### 4.12 What is the difference between Avro and Parquet?

Avro is a **row-oriented** binary format with a schema stored alongside the data, designed for writing and exchanging whole records and for evolving schemas safely. Parquet is a **column-oriented** format designed for analytical reads of a few columns across many rows. They are complementary: Avro is typical on the wire (Kafka messages, CDC events, landing files), Parquet at rest in the lake or lakehouse.

**Comparison**

| Aspect | Avro | Parquet |
| --- | --- | --- |
| Layout | Row by row | Column chunks within row groups |
| Best access pattern | Write and read whole records; streaming | Scan selected columns over many rows |
| Write cost | Low; append record by record | Higher; buffers row groups in memory, encodes columns |
| Read cost for analytics | Must read every field | Reads only needed columns, skips by statistics |
| Compression | Good (block compression) | Better (column encodings plus compression) |
| Schema | JSON schema in the file header or a registry ID per message | Schema in the footer |
| Schema evolution | First-class: reader and writer schemas resolved with defaults and aliases | Supported by engines and table formats, but less formal |
| Splittable | Yes (sync markers between blocks) | Yes (row groups) |
| Typical use | Kafka with a schema registry, CDC (Debezium), raw landing, RPC | Lakehouse tables, warehouse external tables, ML training data |

**Schema evolution is Avro's main strength**

The reader supplies its own schema; Avro resolves differences with the writer's schema: new fields take default values, removed fields are ignored, aliases handle renames. Together with schema registry compatibility checks (see 6.29), this lets producers and consumers upgrade independently.

**Choosing a format**

- **Streaming and messaging:** Avro or Protobuf (Protobuf is popular for service-to-service APIs and has similar evolution rules).
- **Analytical storage:** Parquet, usually wrapped in Delta or Iceberg tables.
- **Hive-centric legacy platforms:** ORC is comparable to Parquet.
- **Interchange with humans:** CSV or JSON, kept at the edges of the platform.
- **In-memory and between processes:** Arrow, which moves columnar data between engines (Spark, pandas, DuckDB, Polars) without serialisation.

**A common pipeline:** services publish Avro to Kafka with a registry; a streaming job lands the records in a bronze Delta or Iceberg table (Parquet files); silver and gold stay in Parquet-based tables. Converting at the boundary from row format to columnar format is a normal, healthy pattern.

### 4.13 Explain "Object Storage" (S3) vs. "Block Storage" (EBS).

Block storage presents a raw disk (fixed-size blocks) attached to one machine, with a file system on top, giving low-latency random reads and writes and in-place updates. Object storage stores whole immutable objects (bytes plus metadata) in a flat namespace accessed over an HTTP API, scaling to effectively unlimited capacity and very high durability at low cost, but with higher per-request latency and no in-place updates. File storage (EFS, Azure Files, NFS) sits between them, offering a shared POSIX file system.

**Comparison**

| Aspect | Block (EBS, Azure Managed Disks, Persistent Disk) | Object (S3, ADLS Gen2, GCS) | File (EFS, Filestore) |
| --- | --- | --- | --- |
| Access | Mounted volume, POSIX via file system | HTTP API (GET, PUT, LIST) | NFS / SMB mount, shared |
| Latency | Sub-millisecond | Tens of milliseconds per request | Milliseconds |
| Updates | In place, random writes | Whole-object replace; no appends (except special tiers) | In place |
| Sharing | Usually one instance at a time | Any number of clients | Many instances |
| Scale | Per-volume limits (TBs) | Effectively unlimited | Large |
| Durability | Replicated within one zone | Very high, across zones | Regional options |
| Cost per GB | Higher; pay for provisioned size | Lowest; pay for used storage plus requests | Highest |
| Typical use | Databases, OS disks, Kafka broker logs, Spark local shuffle | Data lakes and lakehouses, backups, ML datasets, model artefacts | Shared home directories, legacy apps |

**Why lakes and lakehouses live on object storage**

- Separation of storage and compute: any number of clusters and engines read the same data, and compute can scale to zero.
- Cost and durability at petabyte scale, with lifecycle tiering (see 4.20).

**Object storage behaviours data engineers must design around**

- **No real directories or atomic rename.** Renaming a "folder" is copy plus delete per object, which is why committing Spark output by renaming is slow and unsafe; table formats commit through metadata instead (see 4.32).
- **Listing is slow and costs money**; avoid designs that list millions of keys per query.
- **Request costs and throughput limits per prefix**: millions of tiny files mean millions of requests.
- **Consistency:** S3 has provided strong read-after-write and list consistency since 2020; older advice about eventual consistency is out of date for S3, but caches and other systems may still lag.
- **Egress and cross-region transfer** can cost more than storage.

**Where block storage still matters:** databases (including the warehouse's own local SSD caches), streaming brokers, and local disks for Spark shuffle and spill, where fast NVMe makes a large difference to shuffle-heavy jobs.

### 4.14 What is Snowflake's "Unique Architecture"?

Snowflake separates the platform into three independently scaled layers: **centralised storage** in cloud object storage, **compute** as independent virtual warehouses (MPP clusters) that all read the same data, and a **cloud services** layer that handles metadata, query optimisation, transactions, security and caching. This "multi-cluster, shared data" design lets many workloads run concurrently on one copy of data without competing for compute.

**The three layers**

| Layer | What it does | Why it matters |
| --- | --- | --- |
| Storage | Tables stored as immutable, compressed columnar **micro-partitions** (roughly 50 to 500 MB uncompressed) with min/max metadata per column | Automatic pruning, time travel, zero-copy cloning, cheap storage |
| Compute (virtual warehouses) | Independent clusters, sized XS to 6XL, with local SSD caches; can auto-suspend and auto-resume; multi-cluster warehouses scale out for concurrency | Workload isolation: ELT, BI and data science each get their own warehouse; pay per second while running |
| Cloud services | Metadata, optimiser, transaction management, access control, result cache | Fast metadata operations, ACID transactions, no indexes or vacuuming to manage |

**Features that follow from the design**

- **Zero-copy cloning:** a clone copies metadata only, so a full production copy for testing is instant and costs storage only for changed data.
- **Time travel and fail-safe:** old micro-partitions are retained for a configurable period, so you can query or restore past states.
- **Secure data sharing:** another account reads your data in place, with no copies or pipelines.
- **Three caches:** the result cache in cloud services (identical query, unchanged data), the local disk cache on a warehouse, and metadata used for pruning and some aggregates (`COUNT`, `MIN`, `MAX`).
- **Open formats:** Iceberg tables let Snowflake and other engines share data in your own storage.

**Performance tuning in Snowflake**

- **Pruning is everything.** Data is co-located by load order; for very large tables with selective filters, define **clustering keys** (automatic clustering costs credits) or use search optimisation for point lookups.
- **Scale up for complex queries** that spill to local or remote storage; **scale out (multi-cluster)** for many concurrent queries.
- **Query profile:** check partitions scanned versus total, spill bytes, and exploding joins.

**Cost model and pitfalls**

- Compute is billed in credits per second (60-second minimum) while a warehouse runs. Idle warehouses with long auto-suspend times are the classic waste.
- Serverless features (Snowpipe, automatic clustering, materialised views, search optimisation) bill in the background.
- Cloud services usage above a daily threshold of compute is billed too.
- Use resource monitors, per-workload warehouses and query tags for cost control (see 4.39).

**Interview angle:** explain that separation of storage and compute is now common (BigQuery, Databricks SQL, Redshift RA3 all have it in some form). What made Snowflake distinctive was making it fully managed: no tuning knobs like indexes, distribution keys or vacuuming, plus instant elasticity and data sharing.

### 4.15 What is a "Partition Key" in S3 or HDFS, and how do you choose one?

A partition key is a column whose values determine the directory (or table partition) each row is stored in, for example `s3://lake/events/event_date=2026-11-17/`. When a query filters on that column, the engine reads only the matching partitions and skips the rest without opening any files (partition pruning). Choosing the key well is one of the highest-impact physical design decisions on a lake; choosing it badly creates the small file problem.

**Good partition keys**

- **Appear in the filters of most queries**, typically a date or month.
- **Low to moderate cardinality**: hundreds to a few thousand partitions per table, not millions.
- **Each partition holds enough data** for files of at least about 128 MB to 1 GB.
- **Stable and immutable for a row**: event date, not "status".
- **Align with how data is written and reprocessed**, so a daily job overwrites exactly one partition idempotently (see 4.7).

**Common mistakes**

| Mistake | Result |
| --- | --- |
| Partitioning by a high-cardinality column (`user_id`, `order_id`) | Millions of tiny partitions and files, slow listing and planning |
| Over-nesting (`year/month/day/hour/country`) | Tiny files; queries filtering only by country scan everything anyway |
| Partitioning by a column queries rarely filter on | All cost, no pruning |
| Filtering on a derived expression (`to_date(ts)`) instead of the partition column | No pruning (see 6.17) |
| Skewed key (one country holds 80 percent of data) | Uneven partition sizes and slow tasks |

```python
(events.withColumn("event_date", F.to_date("event_time"))
       .repartition("event_date")                 # one task per date, fewer files
       .write.format("delta")
       .partitionBy("event_date")
       .mode("overwrite")
       .option("partitionOverwriteMode", "dynamic")
       .saveAsTable("silver.events"))
```

**Partitioning versus clustering**

- Use **partitioning** for the coarse, near-universal filter (usually time).
- Use **clustering** within partitions for other selective columns: sort order, Z-order or liquid clustering (see 6.19), or warehouse clustering keys (see 5.48).
- Many modern table designs on Delta use **liquid clustering only** with no Hive-style partitions, and Iceberg uses **hidden partitioning** (`days(event_time)`), so users filter on the natural timestamp and the engine still prunes. Iceberg partition evolution lets you change the scheme without rewriting old data.

**Rule of thumb:** do not partition tables under about 1 TB at all unless retention or overwrite patterns require it; rely on file statistics and clustering. For larger tables, partition by date (day or month depending on volume), then cluster.

### 4.16 What are managed and serverless services in the context of data pipelines, and how do you choose?

They are points on a spectrum of how much infrastructure you operate. With a **managed service** (Amazon RDS, MSK, Cloud Composer, EMR) the provider runs the software, patching, backups and failover, but you still choose and pay for capacity (instance sizes, node counts). With **serverless** (Lambda, BigQuery, Athena, Snowflake serverless features, Databricks serverless, AWS Glue, Cloud Run) you provision no capacity at all: the service scales automatically, often to zero, and you pay per use (per query, per byte scanned, per second of compute).

**The spectrum**

```text
Self-managed on VMs/Kubernetes -> Managed (you size it) -> Serverless (no capacity planning)
More control, more ops work    ---------------------->   Less control, less ops work
```

**Trade-offs**

| Aspect | Self-managed | Managed | Serverless |
| --- | --- | --- | --- |
| Operational effort | Highest | Medium | Lowest |
| Cost at low or spiky usage | Pays for idle | Pays for provisioned size | Cheapest (scale to zero) |
| Cost at high steady usage | Often cheapest | Moderate | Can become most expensive |
| Control and tuning | Full | Partial | Limited |
| Start-up latency | None if running | None if running | Cold starts possible |
| Limits | Your hardware | Provider quotas | Execution time, memory, concurrency limits |
| Lock-in | Lowest | Medium | Often highest (proprietary APIs) |

**Where serverless fits well in data pipelines**

- Event-driven glue: a file lands in S3 and a function validates it or triggers a pipeline.
- Spiky or infrequent workloads: ad hoc queries (Athena, BigQuery on demand), small scheduled jobs.
- Teams without platform engineers who need reliability out of the box.

**Where it fits poorly**

- Long-running or heavy jobs that exceed limits (Lambda's 15-minute maximum, memory caps).
- Steady, high-volume workloads where reserved capacity is far cheaper.
- Workloads needing special hardware, libraries or network setups the service does not allow.

**Decision criteria**

1. **Total cost of ownership,** including engineer time. One avoided on-call rotation often outweighs a higher unit price.
2. **Usage profile:** spiky favours serverless, steady favours provisioned.
3. **Performance needs:** cold starts and concurrency limits.
4. **Security and networking:** private connectivity, data residency, compliance certifications.
5. **Exit cost:** open formats and standard interfaces (SQL, Spark, Iceberg) reduce lock-in.

**Pitfalls:** serverless bills that grow invisibly (a query pattern that scans terabytes, a function retry loop), lack of cost guardrails, and assuming "managed" means "no responsibility": you still own data modelling, access control, backups of what matters, and incident response.

### 4.17 What is the "Modern Data Stack"?

The modern data stack (MDS) is the cloud-native, mostly SaaS set of tools that became the default analytics architecture from about 2016 to 2022: managed **ingestion** (Fivetran, Airbyte), a cloud **warehouse** (Snowflake, BigQuery, Redshift, Databricks SQL), **transformation in SQL** inside the warehouse with dbt (ELT), **orchestration**, a **BI and semantic layer**, plus satellites for observability, cataloguing and reverse ETL. Its core idea is to load raw data first and transform it with the warehouse's elastic compute.

**Typical components**

| Layer | Examples |
| --- | --- |
| Ingestion / ELT | Fivetran, Airbyte, Stitch, cloud-native connectors |
| Event collection | Segment, RudderStack, Snowplow |
| Storage and compute | Snowflake, BigQuery, Redshift, Databricks |
| Transformation | dbt, SQLMesh |
| Orchestration | Airflow, Dagster, Prefect, dbt Cloud jobs |
| Semantic layer and BI | Looker, dbt Semantic Layer, Cube, Power BI, Tableau |
| Quality and observability | dbt tests, Great Expectations, Soda, Monte Carlo |
| Catalog and governance | DataHub, Atlan, Unity Catalog, Snowflake Horizon |
| Reverse ETL / activation | Hightouch, Census |

**What it got right**

- Days instead of months to a first dashboard; connectors became commodities.
- Analytics engineering: SQL users adopted software practices (version control, testing, CI) through dbt.
- Elastic, usage-based compute removed most capacity planning.

**What went wrong, and why the term faded**

- **Tool sprawl:** ten vendors, ten contracts, ten integration points, and nobody owning the whole.
- **Cost surprises:** usage-based pricing on ingestion (monthly active rows) and warehouse compute grew faster than value; many teams spent heavily on rebuilding unused models.
- **Gaps:** weak support for streaming, ML and unstructured data; governance bolted on.
- **Consolidation:** platforms now bundle most layers (Databricks and Snowflake offer ingestion, transformation, orchestration, catalog, ML and AI features), and vendors merged.

**Where it stands now**

The pattern (ELT into a cloud platform, SQL-based modelling, tests, orchestration, semantic layer) remains standard. The direction of travel is fewer tools, open table formats so storage is not tied to one engine, and adding AI workloads (document pipelines, vector search, LLM features) to the same governed platform.

**Interview framing:** be able to draw it, name the trade-offs, and explain when you would choose an integrated platform over best-of-breed tools: smaller teams usually benefit from consolidation; larger organisations with specific needs may justify specialised tools, with clear ownership and cost tracking for each.

### 4.18 What is a "Data Mesh"?

Data mesh is a socio-technical approach, defined by Zhamak Dehghani, that decentralises analytical data ownership to the business domains that produce and understand the data, instead of funnelling everything through a central data team. It rests on four principles, and it is primarily an **organisational** change; the technology supports it but does not create it.

**The four principles**

| Principle | Meaning | What it looks like in practice |
| --- | --- | --- |
| Domain ownership | Domains (payments, logistics, marketing) own their analytical data end to end | Domain teams include data engineers and are on call for their data |
| Data as a product | Datasets are products with users, owners, documentation, SLAs and quality guarantees | Discoverable in a catalog, versioned, with contracts and a support channel |
| Self-serve data platform | A central platform team provides tooling so domains do not each rebuild infrastructure | Templates for ingestion, storage, pipelines, quality checks, access control |
| Federated computational governance | Global standards (security, interoperability, naming, PII handling) defined jointly and enforced automatically | Policies as code, tag-based access control, shared identifiers and contracts |

**Problems it addresses**

- The central data team becomes a bottleneck with a backlog of months.
- Central engineers lack domain context, so data is technically correct but semantically wrong.
- Producers change source systems without considering analytics consumers.

**When it fits**

- Large organisations with many distinct domains, many data producers and consumers, and enough data talent to staff domain teams.
- A mature platform that makes the "self-serve" part real.

**When it does not**

- Small or mid-size companies: a central team with clear domain-aligned ownership inside it is usually more efficient.
- Organisations without the budget to put data engineers into domains.

**Common failure modes**

- **Rebranding without change:** calling existing tables "data products" with no owners, SLAs or contracts.
- **Data silos with new names:** domains choose incompatible tools and identifiers, and cross-domain analysis becomes harder.
- **Governance too weak or too heavy:** either chaos, or a central committee that recreates the bottleneck.
- **Duplicated platform effort** when the self-serve platform is not good enough.

**How to adopt it pragmatically**

Start with two or three high-value domains, define what a data product must have (owner, contract, SLA, documentation, quality checks, access policy), build platform templates that make meeting that standard easy, and measure time-to-data and incident rates. Data contracts (see 4.27) and a semantic layer (see 4.38) are the usual interoperability tools.

**Interview point:** data mesh and lakehouse are not alternatives. The mesh is about ownership and operating model; a lakehouse or warehouse is often the shared platform that the domains build their products on.

### 4.19 What is Infrastructure as Code (IaC), and how do you apply it to a data platform?

Infrastructure as Code means defining cloud resources (storage buckets, networks, warehouses, clusters, IAM roles, catalogs, schedules) in version-controlled code and applying it through an automated pipeline, instead of clicking in consoles. The result is reproducible environments, reviewable changes, an audit trail, and the ability to rebuild the platform after a disaster or in a new region.

**Tools**

| Tool | Style | Notes |
| --- | --- | --- |
| Terraform / OpenTofu | Declarative HCL, provider ecosystem | The de facto standard; providers for AWS, Azure, GCP, Snowflake, Databricks, Confluent |
| Pulumi | General-purpose languages (Python, TypeScript) | Useful when teams prefer Python and need loops and abstractions |
| AWS CDK, Bicep, CloudFormation | Cloud-specific | Good inside a single cloud |
| Crossplane | Kubernetes-native | For platform teams already on Kubernetes |

**What belongs in IaC on a data platform**

- Storage (buckets with encryption, versioning, lifecycle rules), networks and private endpoints.
- Compute: warehouse sizes and auto-suspend settings, cluster policies, job definitions.
- Identity and access: roles, grants, service principals, catalog permissions and row or column policies (see 4.37).
- Streaming: Kafka topics with partitions, retention and ACLs; schema registry compatibility settings.
- Orchestration deployment (managed Airflow environments) and monitoring alerts.

**A small Pulumi example in Python**

```python
import pulumi_aws as aws

raw = aws.s3.BucketV2("raw-landing", force_destroy=False,
                      tags={"team": "data-platform", "tier": "bronze"})

aws.s3.BucketLifecycleConfigurationV2("raw-lifecycle",
    bucket=raw.id,
    rules=[aws.s3.BucketLifecycleConfigurationV2RuleArgs(
        id="tier-old-raw", status="Enabled",
        transitions=[aws.s3.BucketLifecycleConfigurationV2RuleTransitionArgs(
            days=90, storage_class="GLACIER_IR")])])
```

**Practices that make it work**

- **Pipeline-only changes:** plan on pull request (the plan output is posted for review), apply on merge. Console changes are blocked or detected as drift.
- **Remote state with locking** (an S3 or GCS backend, Terraform Cloud), never local state files.
- **Modules for repeated patterns** ("a data product": bucket, schema, roles, tags), so domain teams self-serve within guardrails.
- **Separate state per environment and blast radius**, so a mistake in one module cannot destroy production storage.
- **Policy as code** (OPA, Sentinel, Checkov) to block public buckets, untagged resources or missing encryption.
- **Drift detection** on a schedule.
- **Secrets** come from a secrets manager, never from code or state in plain text.

**Pitfalls:** treating data as infrastructure (a `terraform destroy` must never delete a production table or bucket; use prevent-destroy and deletion protection), huge monolithic states that take 20 minutes to plan, and managing fast-changing objects (individual tables, dbt models) in Terraform when they belong in the transformation tool.

### 4.20 What is "Data Tiering"?

Data tiering places data on storage classes with different cost and access characteristics according to how often and how quickly it is needed: **hot** data on fast, expensive storage, **warm** data on cheaper storage, and **cold** or archive data on the cheapest storage with slower or more expensive retrieval. Done well, it cuts storage costs substantially without hurting the queries that matter.

**Typical tiers (AWS names; Azure and GCP have equivalents)**

| Tier | Example | Access pattern | Trade-off |
| --- | --- | --- | --- |
| Hot | S3 Standard, warehouse active storage, SSD caches | Daily queries, recent partitions | Highest storage price, no retrieval fee |
| Warm | S3 Standard-IA, Azure Cool | Monthly access | Cheaper storage, per-GB retrieval fee, minimum storage duration |
| Cold | S3 Glacier Instant or Flexible Retrieval, Azure Cold | Rare access, audits | Much cheaper; retrieval fees or minutes to hours to restore |
| Archive | S3 Glacier Deep Archive, Azure Archive | Compliance retention | Cheapest; hours to restore |
| Automatic | S3 Intelligent-Tiering | Unknown or changing patterns | Small monitoring fee per object; moves objects automatically |

**How to implement it**

- **Lifecycle policies** on buckets and prefixes: move raw landing files to infrequent access after 30 days and to archive after a year; expire temporary data.
- **Tiering by partition:** time-partitioned tables naturally separate hot recent partitions from old ones.
- **Kafka tiered storage:** recent segments on broker disks, older segments in object storage, making long retention affordable.
- **Warehouses:** many manage tiering internally (BigQuery long-term storage pricing for unmodified tables); others benefit from archiving old data to lake tables queried externally.
- **Downsampling:** keep raw detail for a short period and aggregates for longer, common for metrics and IoT (see 6.26).

**Pitfalls**

- **Retrieval and request fees** can exceed savings if "cold" data is actually queried, for example by a monthly backfill or an ML training job over all history.
- **Minimum storage durations and per-object overheads** make tiering tiny files counter-productive; compact first (see 4.10).
- **Table formats:** moving data files of a Delta or Iceberg table to archive classes breaks queries and time travel over them. Archive by partition with care, or copy old partitions to a separate archive table.
- **Retention and deletion obligations:** tiering is not deletion. GDPR and retention policies still apply to archived data (see 4.36).

**Senior approach:** classify datasets by access pattern using storage access logs or query history, agree retention with legal and the business, automate with lifecycle rules defined in IaC, and report savings. Storage is often a smaller line item than compute, so do not let tiering work crowd out compute optimisation (see 4.39).

### 4.21 Explain the difference between "Vertical" and "Horizontal" Scaling.

Vertical scaling (scaling up) gives one machine more resources: more CPU, memory, faster disks. Horizontal scaling (scaling out) adds more machines and spreads the work across them. Vertical is simpler but hits a ceiling and remains a single point of failure; horizontal scales much further and adds resilience, but requires the workload to be partitioned and brings distributed-systems problems such as shuffles, coordination and consistency.

**Comparison**

| Aspect | Vertical (up) | Horizontal (out) |
| --- | --- | --- |
| How | Bigger instance | More instances |
| Limit | Largest available machine | Practically very high |
| Complexity | Low; no code changes | Partitioning, coordination, network costs |
| Fault tolerance | Single node remains a SPOF | Survives node loss with replication |
| Cost curve | Grows steeply at the top end | Roughly linear, plus coordination overhead |
| Downtime to scale | Often a restart | Usually online |
| Data examples | Bigger PostgreSQL instance, larger Snowflake warehouse size, bigger single-node DuckDB machine | More Spark executors, Kafka brokers and partitions, multi-cluster warehouses, Cassandra nodes |

**What data engineers should know**

- **Not all work scales out well.** Operations that need global coordination (a global sort, a single hot key, a transaction across shards) limit horizontal scaling. Amdahl's law applies: the serial part caps the speed-up.
- **Shuffle costs** grow with cluster size; a join that fits in one large machine's memory may be faster there than across 50 small nodes.
- **Single-node engines are surprisingly capable.** A machine with hundreds of GB of RAM running DuckDB or Polars handles datasets that once needed a Hadoop cluster, with no distributed overhead. Scale up first when it is enough.
- **Warehouses offer both:** a larger warehouse size helps a single complex query (more memory, less spilling); more clusters help many concurrent queries.
- **Stateful systems are hardest to scale out:** databases need sharding or distributed SQL; streaming jobs need state redistribution.

**Practical approach:** scale vertically until cost, limits or availability make it unreasonable, design data and keys so that horizontal scaling is possible later (good partition keys, no global ordering assumptions), and use autoscaling for horizontally scaled components with limits so a runaway job cannot scale costs without bound.

### 4.22 What is a "VPC" and why do data engineers care?

A Virtual Private Cloud is a logically isolated private network inside a cloud provider, with your own IP ranges, subnets, route tables, firewall rules and gateways. Data engineers care because every pipeline moves data between networks: source databases, SaaS tools, warehouses, Spark clusters and object storage. Network design decides whether that traffic is private, whether it works at all, and how much it costs.

**Building blocks**

- **Subnets:** public (with a route to the internet gateway) and private (no direct inbound access). Data workloads belong in private subnets.
- **Security groups and network ACLs:** stateful and stateless firewall rules.
- **NAT gateway:** lets private resources reach the internet outbound.
- **Private endpoints / PrivateLink:** private access to cloud services (S3, Snowflake, Databricks, Kafka services) without traversing the public internet.
- **Gateway endpoints** for S3 and DynamoDB on AWS: private and free of data-processing charges.
- **Peering, Transit Gateway, VPN and Direct Connect / ExpressRoute:** connecting VPCs to each other and to on-premises networks.

**Why it matters for data work**

| Concern | Example |
| --- | --- |
| Security | Warehouses and databases reachable only through private endpoints; no public IPs on clusters |
| Compliance | Data must not traverse the public internet; network logs for audit |
| Connectivity | A managed ingestion tool needs to reach an on-premises database through a VPN or private link |
| Cost | Pulling terabytes from S3 through a NAT gateway incurs per-GB processing charges; a gateway endpoint avoids them. Cross-AZ and cross-region traffic is billed too |
| Reliability | NAT gateway or VPN as single points of failure; IP address exhaustion in small subnets when Spark or Kubernetes autoscaling adds many nodes |

**Common data platform pattern**

```text
On-prem DB --(Direct Connect/VPN)--> VPC private subnets
   Spark / Kubernetes workers ----(S3 gateway endpoint)----> S3 lake
   Orchestrator ----(PrivateLink)----> Snowflake / Databricks control plane
   Outbound to SaaS APIs ----(NAT gateway, allow-listed IPs)
```

**Pitfalls**

- Undersized subnets that run out of IP addresses during cluster autoscaling.
- Hidden NAT and cross-AZ data transfer costs on large data movement (see 4.39).
- SaaS vendors that need allow-listed static IPs from your NAT gateway.
- DNS configuration for private endpoints, which is the most common reason "it works from my laptop but not from the cluster".

**Senior view:** you do not need to be a network engineer, but you should design data flows with the network team: know where data crosses boundaries, keep it private by default, and include network costs in architecture decisions.

### 4.23 What is "Data Governance"?

Data governance is the set of decision rights, policies, roles and processes that ensure data is **owned, understood, secure, compliant and of known quality** across the organisation. It answers: who owns this data, who may use it and for what, what does it mean, how good is it, how long do we keep it, and how do we prove all of that to auditors. Good governance enables use of data; bad governance only blocks it.

**Pillars**

| Pillar | What it covers | Typical mechanisms |
| --- | --- | --- |
| Ownership and stewardship | Named owners and stewards per dataset and domain | Catalog entries with owners, RACI, escalation paths |
| Metadata and discovery | What exists, what it means, where it came from | Data catalog, business glossary, lineage (see 4.5), semantic layer |
| Access and security | Who can see what | RBAC/ABAC, row and column policies (see 4.37), audit logs |
| Privacy and compliance | GDPR, CCPA, HIPAA, sector rules, data residency | PII classification, consent, retention and deletion (see 4.36) |
| Quality | Fitness for use, measured | Quality rules, SLAs, observability, issue management (see 4.25) |
| Lifecycle | Retention, archiving, deletion | Retention schedules, tiering, legal hold |
| AI governance | Training data rights, model and prompt data use | Data use policies for ML and LLMs, documentation of training sets |

**Operating model**

- **Central** governance gives consistency but becomes a bottleneck.
- **Federated** governance (as in data mesh, see 4.18) sets global policies centrally and lets domains apply them, with automation ("computational governance") doing most of the enforcement.
- A **data governance council** with business and technical leaders agrees policies and resolves conflicts; stewards in domains do the daily work.

**Making it work in practice**

1. Start from risks and use cases (a regulatory requirement, a metric dispute, an AI initiative), not from a framework document.
2. Classify data automatically and tag it in the catalog; drive access policies from tags.
3. Put governance into the platform: templates, policies as code, CI checks for ownership and documentation, so the compliant path is the easy path.
4. Measure: percentage of critical datasets with owners, documentation and quality checks; access request turnaround time; policy violations; audit findings.

**Tools:** Unity Catalog, Snowflake Horizon, Microsoft Purview, Google Dataplex, Collibra, Alation, Atlan, DataHub, OpenMetadata.

**Pitfalls:** governance as a committee that produces policies nobody implements; a catalog nobody updates; access processes so slow that people copy data to spreadsheets; and treating AI use of data as out of scope.

### 4.24 How do you handle PII (Personally Identifiable Information)?

Handle PII with **data minimisation and layered controls across its whole lifecycle**: collect only what is needed, find and classify it, pseudonymise it early, restrict and mask access, encrypt it, keep it only as long as allowed, delete it reliably, and audit who used it. Encryption and RBAC alone are not enough, because most PII leaks come from copies, logs, exports and over-broad access.

**Lifecycle controls**

| Stage | Controls |
| --- | --- |
| Collection | Minimise fields; record purpose and legal basis; respect consent flags |
| Ingestion | Detect and classify PII automatically (pattern and ML scanners); tag columns in the catalog |
| Storage | Encryption at rest with managed keys; separate restricted zones or schemas for raw PII |
| Transformation | Pseudonymise early: replace direct identifiers with tokens or keyed hashes; keep the mapping table tightly restricted |
| Access | Least privilege; tag-based column masking and row filters (see 4.37); just-in-time access for raw PII |
| Use in analytics and ML | Aggregate, generalise or apply k-anonymity or differential privacy for shared outputs; avoid PII in features unless necessary |
| Use with LLMs | Redact before sending to external models; do not log raw prompts with PII; control what retrieval indexes contain (see 26.21) |
| Retention and deletion | Retention schedules, erasure pipelines, crypto-shredding (see 4.36) |
| Monitoring | Audit logs of access to sensitive columns, alerts on unusual queries or exports |

**Techniques compared**

| Technique | Reversible | Keeps joinability | Use |
| --- | --- | --- | --- |
| Masking (`****1234`) | No (for the viewer) | No | Display to most users |
| Tokenisation (vault-based) | Yes, with vault access | Yes | Payments, identifiers needed later |
| Keyed hashing (HMAC with a secret key) | No | Yes | Joining datasets without exposing raw IDs |
| Plain hashing of emails or phone numbers | Effectively reversible by dictionary attack | Yes | Avoid for low-entropy values |
| Encryption (field level) | Yes, with key | No (unless deterministic) | Highly sensitive fields |
| Generalisation and aggregation | No | Partially | Shared reports, research datasets |

```python
import hmac, hashlib

def pseudonymise(value: str, key: bytes) -> str:
    return hmac.new(key, value.strip().lower().encode(), hashlib.sha256).hexdigest()
```

**Pitfalls**

- PII in places nobody scans: free-text fields, log files, file names, Kafka topics with long retention, notebooks and CSV exports.
- Test and development environments filled with production PII; use masked or synthetic data.
- Unsalted hashes treated as anonymisation; under GDPR, pseudonymised data is still personal data.
- Quasi-identifiers (postcode, birth date, gender) that re-identify people when combined.

**Senior point:** make the safe path the default. Platform templates that tag, pseudonymise and mask automatically protect far more data than training people to remember the rules.

### 4.25 What is "Data Quality Profiling", and how do you build data quality into pipelines?

Data profiling is the systematic analysis of a dataset's actual content and structure: column types, null rates, distinct counts, value distributions, ranges, patterns and relationships between columns. It is how you discover what the data really looks like before trusting it, and it provides the baselines from which quality rules and anomaly checks are derived.

**What a profile includes**

| Area | Examples |
| --- | --- |
| Structure | Inferred versus declared types, schema consistency across files |
| Completeness | Null and empty rates per column |
| Uniqueness | Distinct counts, duplicate keys |
| Validity | Values within ranges or allowed sets; format patterns (emails, postcodes, ISO dates) |
| Distribution | Min, max, mean, percentiles, top values, skew |
| Relationships | Foreign keys that do not resolve, functional dependencies, correlations |
| Timeliness | Latest timestamp, gaps in time series |

**Quality dimensions to turn profiles into rules**

Completeness, uniqueness, validity, consistency (across systems), accuracy (against a trusted source), and timeliness. Each important dataset should have explicit rules for the dimensions that matter to its consumers.

**From profiling to production checks**

1. **Profile** a new source on arrival (and periodically) to learn its shape.
2. **Codify rules** as tests: `order_id` unique and not null, `amount` at least zero, `country` in the reference list, row count within 20 percent of the 7-day average.
3. **Run them in the pipeline** with severity levels: block publication on critical failures (Write-Audit-Publish, see 7.16), warn on others.
4. **Monitor trends** with anomaly detection on volume, freshness and distributions to catch problems no fixed rule anticipated (see 4.28).
5. **Route failures to owners** with context, and track recurring issues back to source systems.

```python
import pandas as pd

def profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_pct": df.isna().mean().round(4) * 100,
        "distinct": df.nunique(),
        "min": df.min(numeric_only=True),
        "max": df.max(numeric_only=True),
    })

orders = pd.read_parquet("orders.parquet")
print(profile(orders))
assert orders["order_id"].is_unique, "duplicate order_id"
assert (orders["amount"] >= 0).all(), "negative amounts"
```

**Tools:** ydata-profiling for exploratory profiles; Great Expectations, Soda, dbt tests and dbt-expectations, Deequ (Spark) and pandera (pandas, Polars) for rule-based checks; observability platforms and lakehouse monitoring for anomaly detection.

**Pitfalls:** profiling once and never again; rules with thresholds nobody understands; hundreds of low-value checks producing alert fatigue; and checking only the final tables, when catching bad data at ingestion is far cheaper. For ML-specific feature validation see 10.38.

### 4.26 What is "dbt" and why is it so popular?

dbt (data build tool) is a transformation framework that lets people write data models as `SELECT` statements (plus Python models on some platforms) and turns them into tables and views inside the warehouse or lakehouse. It handles dependency ordering, materialisation, testing, documentation and lineage. Its popularity comes from bringing software engineering practices to SQL-based analytics work, which created the analytics engineering role.

**What dbt does**

- **Models and dependencies:** each model references others with `ref()`, so dbt builds a DAG and runs models in the right order, in parallel where possible.
- **Materialisations:** view, table, incremental (process only new data, see 7.21), ephemeral, materialised view, snapshot (SCD Type 2).
- **Tests:** generic data tests (`unique`, `not_null`, `relationships`, `accepted_values`), custom tests, unit tests on mocked inputs, and model contracts that enforce column names and types.
- **Documentation and lineage:** generated docs site and a manifest that catalogs and lineage tools consume.
- **Jinja and macros:** reusable logic, environment-specific behaviour.
- **Environments:** the same code deploys to dev, CI and production targets.
- **Packages:** shared macros and tests (dbt-utils, dbt-expectations).

**Why teams adopted it**

- SQL is the language analysts already know; dbt adds version control, code review and CI without requiring Python or Spark skills.
- The ELT pattern on cloud warehouses made in-warehouse transformation the default (see 7.1).
- Tests and documentation live next to the code, which improves trust in data.
- Slim CI builds only changed models and their dependents (see 7.22).
- The semantic layer (MetricFlow) defines metrics on top of models (see 4.38).

**Limitations and trade-offs**

- **Not an orchestrator for everything:** ingestion, ML and operational tasks need Airflow, Dagster or platform jobs; dbt is usually called from them.
- **Large projects get hard to manage:** thousands of models, slow parsing and compilation, unclear ownership. Use groups, access modifiers, model versions and multi-project setups (dbt Mesh) for large organisations.
- **Jinja-heavy SQL** becomes hard to read and test.
- **Costs move to the warehouse:** careless full refreshes and rebuilding everything on every run are a common source of spend (see 4.39).
- **Alternatives:** SQLMesh (state-aware, virtual environments, column-level lineage), Dataform (BigQuery), Spark or Python-based transformations for complex logic.

**What a senior candidate adds:** dbt is a tool for a practice. The value comes from conventions: a layered project structure (staging, intermediate, marts), naming, required tests for keys, contracts on public models, ownership metadata, and CI that blocks breaking changes.

### 4.27 Explain the concept of "Data Contracts."

A data contract is an explicit, versioned agreement between a data producer and its consumers that specifies **what the data looks like and what can be relied on**: schema, semantics, quality guarantees, freshness and availability SLAs, ownership, and how changes will be made. Its purpose is to stop upstream changes (a renamed column, a new meaning for a status code) from silently breaking downstream dashboards, pipelines and models.

**What a contract typically contains**

| Section | Examples |
| --- | --- |
| Identity and ownership | Dataset or topic name, owning team, contact, version |
| Schema | Fields, types, nullability, keys, allowed values |
| Semantics | Business meaning of each field, units, time zones, how deletes are represented |
| Quality | Uniqueness, ranges, referential integrity, maximum null rates |
| Service levels | Freshness (available by 06:00 UTC), latency for streams, support hours |
| Classification and policy | PII tags, retention, allowed uses |
| Change management | Compatibility rules, deprecation notice period, versioning |

**How contracts are enforced**

- **At the producer, in CI:** schema changes are checked for compatibility before merge (schema registry checks for events, see 6.29; contract tests for database-backed exports).
- **At publication:** dbt model contracts, Delta or Iceberg schema enforcement, and quality checks gate the release.
- **At runtime:** validation on ingestion, with violations quarantined and alerted to the producer.

```python
from datetime import datetime
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field

class OrderV1(BaseModel):
    order_id: str = Field(min_length=1)
    customer_id: str
    status: Literal["PLACED", "PAID", "SHIPPED", "CANCELLED"]
    amount: Decimal = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    updated_at: datetime            # UTC

def validate(records: list[dict]) -> tuple[list[OrderV1], list[dict]]:
    good, bad = [], []
    for r in records:
        try:
            good.append(OrderV1.model_validate(r))
        except Exception as exc:
            bad.append({"record": r, "error": str(exc)})
    return good, bad
```

Specifications such as the Open Data Contract Standard (ODCS) describe contracts in YAML so tools can generate checks and documentation from them.

**Making them work organisationally**

- Producers must accept ownership: the contract is part of their service's interface, like an API.
- Start with the most critical interfaces (the source tables behind revenue reporting, the events feeding ML).
- Prefer published events or outbox tables with contracts over CDC on private tables (see 7.23).
- Version breaking changes (`orders_v2`) and run both versions in parallel for a deprecation period.

**Pitfalls:** contracts written as documents that nothing enforces; contracts imposed on producers without their buy-in; over-specifying so every harmless change needs a negotiation; and ignoring semantics, which cause more incidents than schema changes.

### 4.28 What is the difference between "Data Observability" and "Monitoring"?

Monitoring checks **known failure modes** against predefined thresholds: did the job succeed, is CPU high, did the task exceed its timeout. Data observability is the broader ability to understand the health of the **data itself** and of the system end to end, including failures nobody predicted: a table that loaded on time but with half the usual rows, a column whose distribution shifted, a silent schema change upstream. Monitoring tells you that something is broken; observability helps you find what, where and why.

**Comparison**

| Aspect | Monitoring | Data observability |
| --- | --- | --- |
| Focus | Infrastructure and jobs | Data content, plus pipelines and their lineage |
| Questions | Is it up? Did it run? | Is the data fresh, complete, valid, as expected? What changed? Who is affected? |
| Method | Static thresholds and health checks | Learned baselines, anomaly detection, lineage, metadata |
| Failure types caught | Known unknowns | Unknown unknowns ("silent" data failures) |
| Output | Alerts | Alerts with context: impact, root cause candidates, owners |

**The common pillars of data observability**

1. **Freshness:** when was the table last updated, compared with its normal cadence and SLA?
2. **Volume:** row counts and bytes against expected ranges.
3. **Schema:** added, removed or retyped columns.
4. **Distribution:** null rates, value ranges, category frequencies, uniqueness.
5. **Lineage:** which upstream sources and downstream consumers are connected, for impact and root-cause analysis (see 4.5).

Add **pipeline metrics** (run duration, cost, lag) and **usage** (who queries what) for a complete picture.

**A simple implementation pattern**

```python
def volume_anomaly(today: int, history: list[int], k: float = 3.0) -> bool:
    import statistics as st
    mu, sd = st.mean(history), st.pstdev(history) or 1.0
    return abs(today - mu) > k * sd

counts = load_daily_row_counts("gold.orders", days=28)   # from table metadata
if volume_anomaly(counts[-1], counts[:-1]):
    alert(owner="commerce-data", table="gold.orders", metric="row_count")
```

Real systems account for weekly seasonality and trends, and use metadata (table history, information schemas) instead of scanning data where possible.

**Tools:** Monte Carlo, Bigeye, Anomalo, Metaplane, Soda, Elementary (dbt-native), Databricks lakehouse monitoring, plus OpenLineage-based lineage.

**Making it useful rather than noisy**

- Prioritise by data product tier; not every table deserves anomaly alerts.
- Route alerts to owners with lineage context, not to one shared channel.
- Combine with explicit tests: fixed rules for known invariants, anomaly detection for the rest.
- Track data incidents like service incidents: time to detect, time to resolve, and the number of incidents found by consumers before the data team (the key metric to drive down).

### 4.29 How do you implement "Unit Testing" for a data transformation?

Structure transformations as **pure functions** (DataFrame in, DataFrame out, no reads or writes inside), then test them with pytest on small, hand-built inputs that cover normal cases and edge cases, asserting on the exact expected output. Unit tests check that the logic is right; data tests on real data (see 4.25) check that the data is right. You need both.

**Make the code testable first**

- Separate I/O (reading sources, writing tables) from logic.
- Pass configuration and run dates as parameters instead of calling `now()` inside.
- Keep functions small enough to test one rule at a time.

**A PySpark example**

```python
# transforms.py
from pyspark.sql import DataFrame, functions as F, Window

def latest_orders(df: DataFrame) -> DataFrame:
    w = Window.partitionBy("order_id").orderBy(F.col("updated_at").desc())
    return (df.withColumn("rn", F.row_number().over(w))
              .filter("rn = 1").drop("rn"))
```

```python
# test_transforms.py
import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.testing import assertDataFrameEqual
from transforms import latest_orders

@pytest.fixture(scope="session")
def spark():
    return (SparkSession.builder.master("local[2]")
            .config("spark.sql.shuffle.partitions", "2").getOrCreate())

def test_keeps_latest_version_per_order(spark):
    src = spark.createDataFrame(
        [("o1", "PLACED", datetime(2026, 1, 1)),
         ("o1", "PAID",   datetime(2026, 1, 2)),
         ("o2", "PLACED", datetime(2026, 1, 1))],
        "order_id string, status string, updated_at timestamp")
    expected = spark.createDataFrame(
        [("o1", "PAID", datetime(2026, 1, 2)),
         ("o2", "PLACED", datetime(2026, 1, 1))], src.schema)
    assertDataFrameEqual(latest_orders(src), expected)
```

**Edge cases worth a test each**

Nulls in keys and values, duplicates and ties, empty input, late or out-of-order records, time zones and daylight-saving boundaries, schema variations (extra or missing columns), very large or negative numbers, and Unicode in strings.

**Other testing levels**

- **SQL and dbt models:** dbt unit tests mock the inputs of a model and assert its output (see 7.22).
- **Integration tests:** run the pipeline end to end on a sample in an isolated schema.
- **Property-based tests** (Hypothesis) generate inputs to check invariants such as "output row count never exceeds input".
- **Idempotency tests:** run twice, compare results (see 4.7).

**Practical tips:** keep fixtures tiny and readable; a session-scoped Spark fixture keeps tests fast; consider DuckDB or Polars for logic that does not need Spark; run tests in CI on every pull request; and do not mock so much that the test only checks the mocks.

### 4.30 How do you handle "Schema Drift" and schema evolution automatically?

Schema drift is an unplanned change in the structure of incoming data: new columns, removed columns, renamed fields, changed types, or different nesting. Schema evolution is the managed ability of formats and tables to change schema over time without breaking readers. The aim is to **absorb safe changes automatically, stop or quarantine breaking ones, and tell the right people**, rather than either failing every pipeline on any change or silently corrupting tables.

**Classify the change**

| Change | Usually safe? | Typical handling |
| --- | --- | --- |
| New nullable column | Yes | Auto-add to bronze and silver; notify owners |
| Column removed | No | Keep it nullable downstream, alert; consumers may break |
| Rename | No | Treat as remove plus add; needs a mapping and a migration |
| Type widening (int to bigint, float to double) | Mostly | Allow where the format supports it |
| Type narrowing or change (string to int) | No | Quarantine and alert |
| New nested field in JSON | Usually | Capture in a semi-structured column; promote later |
| Semantic change (same name, new meaning) | No, and invisible to schema checks | Only contracts and data tests catch it |

**Mechanisms by layer**

- **Events and streams:** a schema registry with compatibility rules rejects breaking changes at the producer (see 6.29). Avro and Protobuf resolve compatible changes with defaults.
- **Ingestion of files and JSON:** Databricks Auto Loader with schema inference and evolution modes, a `_rescued_data` column for unexpected fields or types, or landing raw payloads as JSON or `VARIANT` in bronze so nothing is lost.
- **Table formats:** Delta (`mergeSchema`, column mapping for renames and drops) and Iceberg (column IDs make add, drop, rename and reorder safe; type promotion) support evolution as metadata changes.
- **Transformation layer:** explicit column selection in silver, so new upstream columns do not leak into gold uncontrolled; dbt `on_schema_change` set deliberately for incremental models (see 7.21).

```python
# Bronze: accept additive changes, capture anything unexpected.
# With addNewColumns the stream stops once when a new column appears and
# picks it up on restart, so run it under an orchestrator that restarts it.
(spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", "s3://chk/orders_schema")
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
    .option("rescuedDataColumn", "_rescued_data")
    .load("s3://landing/orders/")
  .writeStream.option("mergeSchema", "true")
    .option("checkpointLocation", "s3://chk/orders_bronze")
    .toTable("bronze.orders"))

# Silver: an explicit contract; fail on missing required columns
REQUIRED = {"order_id", "customer_id", "amount", "updated_at"}
missing = REQUIRED - set(spark.table("bronze.orders").columns)
if missing:
    raise ValueError(f"Breaking schema change, missing columns: {missing}")
```

**Policy by layer**

- **Bronze:** be permissive; never drop data, record the schema version with each batch.
- **Silver:** controlled evolution; additive changes allowed through a reviewed process or automatically with notification; breaking changes block.
- **Gold:** stable, contract-backed interfaces; changes are versioned and announced.

**Operational practice**

- Alert on every schema change with a diff, routed to the dataset owner.
- Monitor the volume of rescued or quarantined data.
- Prevent drift upstream with data contracts and CI checks on producers (see 4.27), which is far cheaper than handling it downstream.
- Remember that automatic evolution can leak PII: a new column `customer_email` should be classified before it reaches broad audiences.

### 4.31 Where do you see the future of Data Engineering?

Data engineering is moving from building pipelines by hand towards **operating governed data platforms that serve analytics, ML and AI agents**, with much of the routine pipeline code generated or automated. The durable skills are data modelling, platform architecture, quality and governance, cost management, and understanding the business; the tool layer will keep changing.

**Trends worth discussing**

1. **Open table formats and catalogs as the foundation.** Iceberg and Delta make storage engine-neutral; the competition shifts to catalogs, governance and engines. Expect multi-engine platforms where Spark, SQL warehouses, DuckDB and streaming engines share the same tables.
2. **Platform consolidation.** Fewer, broader platforms instead of a dozen point tools, driven by cost and governance pressures (see 4.17).
3. **AI as a consumer of data.** Retrieval pipelines over unstructured documents, embeddings and vector indexes, evaluation datasets, feature pipelines for models, and semantic layers and ontologies that let LLM agents query data correctly (see 4.38 and 4.42). Unstructured data needs the same lineage, access control and quality as tables.
4. **AI as a tool for data engineers.** Code assistants generate SQL, dbt models, tests and documentation; agents triage pipeline failures and propose fixes. The engineer's role moves towards review, design, and defining guardrails and evaluation.
5. **Streaming and incremental by default.** Unified batch and streaming engines and incremental materialisations (dynamic tables, materialised views, streaming tables) make freshness cheaper; full nightly rebuilds decline.
6. **Data products and contracts.** Ownership, SLAs and contracts become standard, whether or not organisations call it data mesh.
7. **Governance and regulation.** Privacy laws and AI regulation (such as the EU AI Act) make lineage, data provenance, consent and documentation of training data mandatory rather than nice to have.
8. **Cost discipline.** FinOps for data platforms becomes a core competency as usage-based billing grows with AI workloads (see 4.39).
9. **Small data on single nodes.** Fast single-node engines (DuckDB, Polars) handle more workloads, reducing the need for clusters for mid-sized data.

**How to answer as a senior candidate**

Pick two or three trends and connect them to concrete actions: "I would standardise on Iceberg with a REST catalog to keep engine choice open, invest in a semantic layer and contracts so AI assistants can query certified data, and set up cost observability before AI workloads scale." Avoid hype; show that you separate durable shifts from vendor marketing.

### 4.32 What are open table formats (Apache Iceberg, Delta Lake, Apache Hudi), and what problems do they solve on top of Parquet files in object storage?

An open table format is a metadata layer that turns a pile of Parquet files in S3, ADLS or GCS into a real table. It records which files make up the table at each point in time, the schema, partitioning and file statistics. That brings atomic commits, row-level updates, schema evolution and fast planning to object storage, while the data stays in open files that any engine can read.

**What goes wrong with "just Parquet in a folder"**

- **No atomicity.** A job that fails halfway leaves partial files behind, and readers see a half-written table. Object stores have no multi-file transaction.
- **No isolation.** A reader that lists a directory while a writer is overwriting it can see old and new files mixed together.
- **Expensive planning.** Engines have to `LIST` thousands of prefixes to find files. On object storage this is slow and costs money per call.
- **No row-level changes.** You cannot `UPDATE`, `DELETE` or `MERGE` one customer. You rewrite the whole partition by hand.
- **Fragile schema changes.** Renaming or reordering a column breaks readers that match columns by name or position.
- **Partitioning leaks into queries.** Hive-style `dt=2026-11-16` folders mean users have to filter on the partition column explicitly, and changing the partition scheme means rewriting the table.

**What the table format adds**

| Capability | How it is achieved |
| --- | --- |
| ACID commits | A new table version becomes visible through one atomic step: a log entry (Delta, Hudi timeline) or a catalog pointer swap (Iceberg) |
| Snapshot isolation | Readers pin a version and only see files listed in it |
| Fast planning | Manifests or checkpoints list files with min/max stats, so engines prune files without listing directories |
| Row-level DML | Copy-on-write rewrites, or merge-on-read delete files, deletion vectors and log files |
| Schema evolution | Columns tracked by ID (Iceberg, Delta column mapping), so renames and drops are safe |
| Time travel | Old snapshots stay readable until they are expired |
| Hidden or evolving partitioning | Iceberg derives partitions from transforms like `day(ts)`; Delta liquid clustering and Iceberg partition evolution avoid rewrites |

**Senior-level points**

- **The catalog matters as much as the format.** Something has to own the "current version" pointer and enforce access. That is why REST catalogs (Polaris, Unity Catalog, Glue, Nessie) are now central to lakehouse design.
- **Maintenance is your job.** Table formats create small files, delete files and old snapshots. Without scheduled compaction, snapshot expiry and orphan-file cleanup, query performance and storage cost both degrade.
- **The format does not replace the engine.** Concurrency limits, merge performance and feature support (for example Iceberg v3 deletion vectors) depend on which engine and version you run.

A typical example: a nightly CDC `MERGE` into a 5 TB customer table used to be a full partition rewrite with a "do not query between 2 and 4 am" warning. On Iceberg or Delta it becomes a transactional `MERGE` that touches only the affected files, and readers never see a half-applied state.

### 4.33 How do snapshots, time travel and optimistic concurrency work in Iceberg or Delta Lake?

Both formats treat a table as a sequence of immutable versions. Each commit writes new data files plus new metadata describing the full set of live files, then publishes that version atomically. Readers query whichever version is current, or an older one for time travel. Writers never lock the table. They assume no conflict, then check for one at commit time and retry if needed.

**Delta Lake**

- The `_delta_log/` directory holds numbered JSON commits (`00000000000000000042.json`). Each commit lists `add` and `remove` file actions, metadata changes and commit info.
- Every N commits a **checkpoint** (Parquet) summarises the whole state, so readers load the latest checkpoint plus a few JSON files instead of replaying the full history.
- A commit is atomic because writing version `N+1` uses put-if-absent semantics. Only one writer can create that file. On S3 this historically needed a coordination service such as DynamoDB, or the platform's own commit service.

**Apache Iceberg**

- A three-level tree: `metadata.json` (schema, partition specs, snapshot list) points to a **manifest list** per snapshot, which points to **manifests**, which list data and delete files with column stats.
- The catalog stores a pointer to the current `metadata.json`. A commit is a **compare-and-swap** on that pointer: "move from v41 to v42 only if it is still v41."

**Optimistic concurrency, step by step**

1. The writer reads version $N$ and records what it read (files, predicates).
2. It writes new data files. This is the slow part and it involves no locks.
3. It tries to commit $N+1$. If another writer committed first, it re-reads the new state and checks whether those changes **conflict** with its own: did they delete or rewrite files it read or plans to remove? Did they add rows matching its `MERGE` or `DELETE` predicate?
4. If there is no logical conflict, it rebases and retries the commit. Only metadata is rewritten; the data files are reused. If there is a conflict, it fails, for example with Delta's `ConcurrentAppendException` or Iceberg's `CommitFailedException`.

Blind appends almost never conflict. Two `MERGE` jobs on the same partitions often do. Delta's row-level concurrency (with deletion vectors and row tracking) and Iceberg's per-operation isolation levels (`serializable` vs `snapshot`) reduce false conflicts.

**Time travel**

```sql
-- Delta
SELECT * FROM sales VERSION AS OF 1520;
SELECT * FROM sales TIMESTAMP AS OF '2026-11-01';
RESTORE TABLE sales TO VERSION AS OF 1520;

-- Iceberg (Spark)
SELECT * FROM sales FOR SYSTEM_VERSION AS OF 849230111;
SELECT * FROM sales FOR SYSTEM_TIME AS OF '2026-11-01 00:00:00';
```

Time travel works only while the old files still exist. Delta `VACUUM` and Iceberg `expire_snapshots` plus `remove_orphan_files` delete them, so retention is a policy decision that trades auditability against storage cost and GDPR obligations.

**What goes wrong in production**

- **Commit storms.** Many small streaming writers, or a long compaction job racing with ingestion, cause repeated retries. Fix it by batching commits, separating partitions, or scheduling compaction for quiet partitions.
- **Metadata bloat.** Thousands of snapshots and manifests slow planning. Expire snapshots and rewrite manifests regularly.
- **Vacuuming too aggressively.** A retention below the longest-running query or stream lag deletes files that are still in use and breaks readers.

### 4.34 What is the difference between copy-on-write and merge-on-read tables, and when would you pick each?

They are two strategies for applying updates and deletes to immutable files. **Copy-on-write (CoW)** rewrites every data file that contains a changed row at write time, so reads stay simple and fast. **Merge-on-read (MoR)** writes only the change, as delete files, deletion vectors or row-based log files, and reconciles it with the base files at query time until compaction folds it in.

**How each works**

- **CoW:** to update 10 rows spread across 3 Parquet files of 512 MB each, the engine rewrites all 3 files (about 1.5 GB) and the commit swaps old files for new ones.
- **MoR in Iceberg:** writes position delete files (in v3, deletion vectors stored as bitmaps in Puffin files) or equality delete files, plus new data files for the updated rows.
- **MoR in Delta:** deletion vectors mark removed rows in existing files, and updated rows go into new files.
- **MoR in Hudi:** each file group has a base Parquet file plus Avro log files. A compaction service periodically merges them. Hudi also offers a *read-optimized* query that reads only base files: fast, but slightly stale.

**Trade-offs**

| Dimension | Copy-on-write | Merge-on-read |
| --- | --- | --- |
| Write cost | High (write amplification) | Low (write only the delta) |
| Write latency | Slower, can't keep up with high-rate upserts | Fast, suits streaming and CDC |
| Read cost | Lowest, plain columnar scans | Higher until compaction (merge or delete-apply) |
| Operational burden | Low | Needs compaction scheduling and monitoring |
| Concurrency conflicts | More (rewrites large files) | Fewer (smaller footprint) |
| Freshness | Commit = fully optimised | Fresh data, but read-optimised views lag |

**How I decide**

- **Pick CoW** for read-heavy tables with infrequent, batched changes: daily-loaded fact tables, gold aggregates, dimension tables refreshed once a day, anything that many BI dashboards hit.
- **Pick MoR** for write-heavy or low-latency upsert workloads: CDC from OLTP databases every minute, streaming dedup or upserts, GDPR deletes scattered across many files, or tables where updates touch a tiny fraction of rows in many files.
- **Hybrid is normal.** Ingest bronze/silver CDC tables as MoR with aggressive compaction, and build gold tables as CoW. In Iceberg you can even set the mode per operation (`write.delete.mode`, `write.update.mode`, `write.merge.mode`).

```sql
ALTER TABLE silver.orders SET TBLPROPERTIES (
  'write.merge.mode'  = 'merge-on-read',
  'write.delete.mode' = 'merge-on-read',
  'write.update.mode' = 'copy-on-write'
);
```

**Failure modes**

- **MoR without compaction** is the most common lakehouse performance incident. Delete files and log files pile up, every query merges them, and latency degrades slowly until someone notices. Monitor delete-file counts per partition and alert on them.
- **CoW on high-frequency CDC** leads to runaway compute and storage churn, plus constant commit conflicts with readers' compaction.
- Engine support varies. Some readers ignore delete files or don't support v3 deletion vectors yet, so check every engine that reads the table before switching modes.

### 4.35 Compare Apache Iceberg, Delta Lake and Apache Hudi. How would you choose between them?

All three give you ACID tables on object storage, and their core features have largely converged: row-level deletes, time travel, schema evolution, and deletion vectors with a shared encoding in Iceberg v3 and Delta. In 2026 the choice depends less on features and more on **which engines and catalog you standardise on**, and **what your write pattern looks like**.

**Comparison**

| Dimension | Apache Iceberg | Delta Lake | Apache Hudi |
| --- | --- | --- | --- |
| Origin | Netflix, Apache project | Databricks, Linux Foundation | Uber, Apache project |
| Metadata model | metadata.json → manifest lists → manifests, pointer in a catalog | Ordered JSON log plus Parquet checkpoints | Timeline of instants plus a metadata table |
| Engine breadth | Widest: Spark, Flink, Trino, Snowflake, BigQuery, Athena, Redshift, DuckDB, Dremio | Strongest in Spark/Databricks, good elsewhere via delta-rs and connectors | Spark and Flink first, read support elsewhere |
| Catalog story | Iceberg REST spec (Polaris, Unity, Glue, Nessie, Snowflake Horizon) | Unity Catalog; Iceberg REST via UniForm | Hive/Glue metastore; less standardised |
| Partitioning | Hidden partitioning, partition evolution | Liquid clustering, generated columns | Partition paths plus record-level indexes |
| Updates | CoW or MoR (delete files, v3 deletion vectors) | CoW or deletion vectors | CoW or MoR with log files and async compaction |
| Distinct strengths | Engine neutrality, spec-driven, row lineage in v3 | Tight Databricks integration, Change Data Feed, mature tooling | Record keys, indexes, incremental pulls, non-blocking concurrency for streaming upserts |
| Watch-outs | Catalog choice is now where lock-in lives; maintenance is yours | Best features often arrive on Databricks first | Smaller ecosystem, more tuning knobs |

**How I would choose**

1. **Start from the engines.** If Databricks is your main platform, Delta is the natural default. If you run several engines (Snowflake for BI, Spark for ML, Trino for ad hoc, Flink for streaming), Iceberg is the safest neutral choice because nearly every engine and catalog supports it natively.
2. **Look at the write pattern.** For very high-rate keyed upserts and incremental pipelines (CDC from many OLTP tables, Flink-based ingestion), Hudi's record index and MoR design are purpose-built. Iceberg with Flink and Delta with streaming MERGE also handle this well at moderate volume.
3. **Pick the catalog, not just the format.** Governance, credential vending and cross-engine access come from the catalog. Choosing Unity, Polaris, Glue or Horizon is a longer-term commitment than the file format.
4. **Plan for interoperability.** Delta UniForm writes Iceberg-compatible metadata, and Apache XTable translates metadata between all three. These help with reads across ecosystems, but treat them as bridges, not as a way to avoid choosing a primary write format.
5. **Check what your managed services support.** Feature support differs by engine version (for example v3 deletion vectors, row lineage, MoR reads). A format is only as good as the oldest engine that has to read it.

**A typical answer in an interview:** "On a Databricks-centred platform I'd use Delta with Unity Catalog and turn on UniForm so Snowflake and Trino can read gold tables. On a multi-engine or multi-cloud platform I'd use Iceberg with a REST catalog. I'd use Hudi only for a write-heavy CDC layer where its indexing clearly pays off, and I'd still expose curated data as Iceberg or Delta."

### 4.36 How do you implement the GDPR "right to be forgotten" in a data lake or lakehouse?

Article 17 requires you to erase a person's personal data without undue delay (generally within one month of the request) from every place it lives: raw files, curated tables, derived features, backups, logs and ML training sets. In a lakehouse the hard part is that storage is immutable and copies are everywhere. A `DELETE` statement alone is not enough, because the old files and snapshots still contain the data.

**1. Know where the data is**

- Maintain a PII inventory: tag columns as personal data in the catalog, and record the subject identifier (customer_id, email hash) for each table.
- Use lineage to find every downstream table, feature store, extract and model derived from those sources. If you cannot list where a person's data flows, you cannot prove you erased it.

**2. Design to make deletion cheap**

- **Pseudonymise early.** Replace direct identifiers in bronze with a surrogate key, and keep the mapping (key ↔ email, name, address) in one small, tightly controlled lookup table. Most tables then hold only the surrogate.
- **Crypto-shredding.** Encrypt each person's PII with a per-subject key held in a KMS or vault. Erasure means destroying the key, which makes every copy, including backups and old snapshots, unreadable. It is powerful but complex: key management at scale, performance, and legal sign-off that shredded ciphertext counts as erased.
- **Cluster or partition by subject where possible**, so a delete rewrites few files.
- **Avoid PII in places that are hard to clean**: Kafka topics with infinite retention, log files, free-text fields, and file names.

**3. Run a deletion pipeline**

```sql
-- requests table filled by the privacy/CRM system
MERGE INTO silver.customers t
USING (SELECT customer_id FROM privacy.erasure_requests
       WHERE status = 'APPROVED') r
ON t.customer_id = r.customer_id
WHEN MATCHED THEN DELETE;

-- Delta: physically remove rows hidden by deletion vectors, then old files
REORG TABLE silver.customers APPLY (PURGE);
VACUUM silver.customers RETAIN 168 HOURS;

-- Iceberg (Spark procedures): rewrite affected files, drop old snapshots
CALL cat.system.rewrite_data_files(table => 'silver.customers',
     where => 'customer_id IN (...)');
CALL cat.system.expire_snapshots(table => 'silver.customers',
     older_than => TIMESTAMP '2026-11-09 00:00:00');
```

- Process requests in **batches** (daily or weekly). One MERGE per request rewrites the same files hundreds of times.
- Propagate **bronze → silver → gold → features → exports**, and delete from source systems, queues and landing buckets too.
- **Snapshots and time travel**: the data is not gone until old versions are expired and unreferenced files removed. Set retention so the full cycle (delete, purge, vacuum or expire) finishes inside your legal deadline.
- **Backups and replicas**: either keep them short-lived, or re-apply deletions on restore by replaying the erasure log.
- **ML models**: record which training snapshot each model used. Policy usually requires excluding the person from future retraining. Whether a trained model must be retrained immediately is a legal question, so document the decision.

**4. Prove it**

Keep an audit log of each request with timestamps for every system it was applied to. Run a verification query afterwards to confirm zero matching rows. Regulators and auditors ask for evidence, not intent.

**Common mistakes:** stopping at `DELETE` without purge or vacuum, forgetting CDC streams that re-insert the deleted customer the next day (keep a suppression list), and leaving the data in exports sent to third parties.

### 4.37 How do you implement fine-grained access control (row-level security, column masking, tag-based policies) on a data platform?

Fine-grained access control means deciding access at the level of rows and columns, not just whole tables. A sales rep in Germany sees only German customers, an analyst sees `email` masked, and a privacy officer sees everything. At scale the only maintainable way to do this is **attribute-based access control (ABAC)**: classify data with tags, define a few central policies against those tags and user attributes, and let the platform enforce them in every engine.

**Building blocks**

- **RBAC baseline:** roles or groups get privileges on catalogs, schemas and tables. Grant to groups, never to individuals.
- **Row-level security:** a row filter or row access policy is a function evaluated per query that returns true or false for each row, based on the user's groups or a mapping table.
- **Column masking:** a masking function returns the real value, a partial mask (`****1234`), a hash or NULL, depending on who is asking.
- **Tags and classification:** columns tagged `pii=email` or `sensitivity=high`, ideally set by automated classification scanners plus human review.
- **Tag-based policies:** the policy is attached to the tag, not to each column, so any new column tagged `pii=email` is masked automatically.

**Example (Snowflake syntax)**

```sql
CREATE MASKING POLICY mask_email AS (val STRING) RETURNS STRING ->
  CASE WHEN IS_ROLE_IN_SESSION('PII_READER') THEN val
       ELSE REGEXP_REPLACE(val, '.+@', '***@') END;

ALTER TAG governance.pii_email SET MASKING POLICY mask_email;

CREATE ROW ACCESS POLICY region_rls AS (region STRING) RETURNS BOOLEAN ->
  EXISTS (SELECT 1 FROM governance.user_region m
          WHERE m.role_name = CURRENT_ROLE() AND m.region = region);

ALTER TABLE sales.orders ADD ROW ACCESS POLICY region_rls ON (region);
```

Databricks Unity Catalog does the same with row filters and column masks (SQL UDFs), plus ABAC policies defined at catalog or schema level against governed tags. BigQuery has row access policies and policy tags for column-level security.

**Design principles**

- **Enforce in the platform, not in the BI tool.** Security that lives only in Tableau or Power BI disappears the moment someone connects a notebook. Policies should sit in the catalog so every engine and path gets them.
- **Mapping tables over hardcoded logic.** Keep user-to-region or user-to-cost-centre entitlements in a governed table fed from HR or IAM, so access changes without code deploys.
- **Classify automatically, review by humans.** Scan new tables for PII patterns, propose tags, require owner confirmation.
- **Keep policy functions simple.** Each one runs on every query. A row filter with a heavy subquery can slow dashboards dramatically. Prefer lookups against small, cached mapping tables.
- **Test policies like code.** Write policies in Terraform or dbt, review them in pull requests, and add tests that run queries as specific roles and assert the expected rows and masked values.

**Pitfalls**

- **Inference attacks:** masking a column does not stop someone from filtering or joining on it. Restrict filtering on sensitive columns, or use aggregation policies and minimum group sizes.
- **Conflicting policies:** some platforms block the query when two masks apply to the same column for the same user. Design a single policy per tag.
- **Feature gaps:** filters and masks can limit time travel, cloning, materialized views, or external engines reading the raw files directly. Make sure no one can bypass policies by reading the underlying storage path. Lock down bucket access and use credential vending.
- **Audit:** log who queried which sensitive columns. Access reviews and incident investigations depend on it.

### 4.38 What is a semantic (metrics) layer and why do organisations adopt one?

A semantic layer is a governed definition of business concepts (entities, dimensions, measures and metrics such as "net revenue" or "active customer") that sits between the warehouse and every consumer. Tools ask for "net revenue by region, last quarter" and the layer generates the correct SQL. The goal is **one definition, computed the same way everywhere**: dashboards, notebooks, spreadsheets, reverse ETL and increasingly AI agents.

**The problem it solves**

- Five dashboards show five different revenue numbers because each analyst wrote the joins, filters (refunds? test accounts? currency?) and time logic slightly differently.
- Metric logic is buried in BI tools, so switching tools or adding a new consumer means re-implementing everything.
- LLM-based "chat with your data" tools hallucinate joins and definitions when they only see raw tables. A semantic layer gives them a vetted vocabulary to query.

**What it contains**

- **Semantic models / entities:** tables, their grain, primary and foreign keys, so the layer knows how to join safely and avoid fan-out double counting.
- **Dimensions:** categorical and time attributes, including time grains (day, week, fiscal quarter).
- **Measures and metrics:** simple aggregations, ratios, cumulative and derived metrics, with filters baked in.
- **Access:** query APIs (SQL, JDBC, GraphQL or REST), caching, and governance (who can see which metric).

```yaml
semantic_models:
  - name: orders
    model: ref('fct_orders')
    entities:
      - { name: order_id, type: primary }
      - { name: customer, type: foreign, expr: customer_id }
    dimensions:
      - { name: ordered_at, type: time, type_params: { time_granularity: day } }
      - { name: region, type: categorical }
    measures:
      - { name: gross_amount, agg: sum, expr: amount }
      - { name: refund_amount, agg: sum, expr: refunded_amount }
metrics:
  - name: net_revenue
    type: derived
    type_params:
      expr: gross - refunds
      metrics: [ { name: gross_amount, alias: gross }, { name: refund_amount, alias: refunds } ]
```

**Main options**

| Type | Examples | Notes |
| --- | --- | --- |
| Transformation-layer | dbt Semantic Layer (MetricFlow, open source) | Metrics versioned next to models |
| Standalone / headless | Cube, AtScale | Caching and APIs for many tools |
| BI-native | Looker LookML, Power BI semantic models | Excellent inside one tool, weaker outside it |
| Warehouse-native | Snowflake semantic views, Databricks metric views | Definitions close to the data and governance |

An industry effort to make definitions portable between tools (the Open Semantic Interchange specification, now an Apache incubator project) is emerging, which reduces the lock-in risk of picking one.

**Adoption advice**

- **Start with the 10 to 20 metrics the executive team argues about,** each with a named owner and a written definition. Do not try to model everything.
- **Model clean marts first.** A semantic layer on top of messy tables just centralises the mess.
- **Treat metrics as code:** pull requests, tests comparing results against known values, and change logs when a definition changes.
- **Watch performance and cost.** Generated SQL can be inefficient. Use caching or pre-aggregations for hot dashboards.
- **Failure mode:** buying a semantic layer while every team keeps its own BI logic. Adoption needs a mandate ("certified metrics come only from the layer") and BI tools wired into it.

### 4.39 How do you control and optimise cost on a cloud data platform (warehouse and lakehouse FinOps)?

Cloud data cost is driven mostly by **compute** (warehouses, clusters, serverless queries), then storage and data movement. You control it with three loops: **visibility** (who spends what, and on which workload), **guardrails** (limits that stop runaway spend), and **optimisation** (making each workload cheaper). Treat cost as an engineering metric with owners, like latency.

**1. Visibility and accountability**

- Tag every warehouse, cluster, job and query with team, product and environment. Use query tags, cluster tags and separate warehouses per workload.
- Build a cost model from billing and usage views (Snowflake `ACCOUNT_USAGE`, Databricks system billing tables, BigQuery `INFORMATION_SCHEMA.JOBS`) and publish showback or chargeback dashboards per team.
- Track unit costs that the business understands: cost per dashboard, per pipeline run, per 1,000 orders processed, per model training run.

**2. Guardrails**

- **Budgets and alerts:** Snowflake resource monitors, Databricks budgets, BigQuery custom quotas per user or project.
- **Timeouts:** statement timeouts on warehouses, and maximum bytes billed on BigQuery queries.
- **Policies:** cluster policies that cap node types and sizes, force auto-termination, and require tags. Forbid all-purpose clusters for scheduled jobs.
- **Separate dev from prod**, with smaller defaults and stricter limits in dev.

**3. Compute optimisation**

- **Right-size and auto-suspend.** Idle warehouses are the most common waste. Use short auto-suspend (around 60 seconds) for bursty workloads, longer where the cache pays off. Downsize until runtime starts to grow meaningfully.
- **Scale out for concurrency, up for complexity.** Multi-cluster warehouses handle many small BI queries. A larger size helps only if queries spill to disk or scan a lot.
- **Use job clusters and serverless** for scheduled pipelines, and spot or preemptible instances for fault-tolerant Spark work.
- **Make pipelines incremental.** Replacing full refreshes with incremental models or CDC merges often cuts transformation cost by 5 to 10 times.
- **Fix the top queries.** Usually 20 queries drive most of the spend. Look for full scans, exploding joins, `SELECT *` in BI, and dashboards refreshing every 5 minutes for nobody.
- **Prune data.** Partitioning, clustering or liquid clustering, and file compaction so queries read less data.
- **Materialise wisely.** Pre-aggregate hot dashboards, but audit materialised views and serverless features (auto-clustering, search optimisation, Snowpipe), which bill continuously in the background.

**4. Storage and data movement**

- Lifecycle rules for raw landing zones, tiering cold data to infrequent-access or archive classes.
- Shorten time travel and fail-safe retention on transient or staging tables. Vacuum or expire snapshots on lakehouse tables.
- Drop unused tables. Query-history access logs reveal tables nobody has read in 90 days.
- Watch egress: cross-region or cross-cloud reads and replication can quietly dominate the bill.

**5. Commercial levers**

Committed-use discounts or capacity contracts once usage is stable. Re-benchmark engines for heavy workloads. Moving one large transformation from a warehouse to Spark on a lakehouse, or the reverse, can change costs dramatically.

**Pitfalls:** chasing small storage savings while compute dominates, optimising without unit metrics (so growth looks like waste), and guardrails so strict they break production. Pair hard limits on dev with alert-first limits on prod.

### 4.40 How would you design a real-time clickstream analytics pipeline for a large e-commerce site?

A good answer goes **requirements → ingestion → processing → storage → serving → late data → scaling → monitoring**. The standard design: SDKs send events to a collector, the collector writes to Kafka, Flink (or Spark Structured Streaming) enriches and aggregates the stream, a real-time OLAP store serves fresh metrics, and an Iceberg or Delta lakehouse holds the full history for batch analytics and ML.

**1. Requirements (clarify first)**

- **Scale:** for example 50 million daily users, about 2 billion events per day, peaks of 200k events per second during sales, 1 to 2 KB per event, so roughly 3 to 4 TB per day raw.
- **Use cases and freshness:** live dashboards (conversion, revenue, funnel) under 10 seconds; personalisation features under 1 second; daily reporting and ML on full history.
- **Correctness:** at-least-once delivery plus deduplication; exact revenue numbers must reconcile with the orders database.
- **Privacy:** consent flags, PII minimisation, deletion support.

**2. Ingestion**

- Web and mobile SDKs batch events and send them to a stateless **collector** behind a load balancer. The collector validates against a schema registry (Avro or Protobuf), adds server timestamp and IP-derived geo, and drops or anonymises fields if consent is missing.
- Events go to **Kafka** (or Kinesis or Pub/Sub), keyed by `session_id` or `user_id` so each session stays in order within a partition. Partitions are sized for peak (for example 256+), with replication factor 3 and 7-day retention for replay.
- Invalid events go to a **dead-letter topic**, not into the void.
- Each event carries a client-generated `event_id` for deduplication.

**3. Stream processing (Flink)**

- **Dedup** on `event_id` with keyed state and TTL (for example 24 hours).
- **Enrich**: bot filtering, user-agent parsing, joining product and campaign dimensions from compacted Kafka topics or broadcast state.
- **Sessionise** with session windows (30-minute inactivity gap).
- **Aggregate** with event-time tumbling windows (1 minute) for KPIs, and compute real-time features (views in last 5 minutes per user).
- Checkpoint to object storage with exactly-once sinks (transactional Kafka producers, idempotent upserts).

**4. Storage and serving**

| Layer | Technology | Purpose |
| --- | --- | --- |
| Raw event log | Iceberg/Delta bronze (via Flink or Kafka Connect sink, commits every 1 to 5 min) | Full history, replay, compliance |
| Curated | Silver sessions, gold funnels (batch or incremental) | BI, experimentation, ML training |
| Real-time OLAP | Apache Pinot, Druid or ClickHouse | Sub-second dashboard queries on fresh data |
| Low-latency features | Redis or online feature store | Personalisation and recommendations |

**5. Late and out-of-order data**

- Use **event time** with watermarks (for example 2 minutes of allowed lateness). Mobile clients that were offline can send events hours late.
- Moderately late events update window results via upserts into the OLAP store. Very late events go to a side output that is written to the lakehouse, and nightly batch jobs recompute the affected partitions. Batch is the source of truth for reporting, streaming for freshness.
- Watch for client clock skew: keep both client and server timestamps, and clamp obviously wrong values.

**6. Scaling and resilience**

- Collectors and Flink scale horizontally. Kafka partitions cap parallelism, so provision for peak.
- Use hot-key mitigation (for example a viral product page) with key salting and two-stage aggregation.
- Backpressure: Kafka absorbs bursts, so consumer lag grows instead of losing data. Autoscale Flink on lag.
- Replay: reprocess from Kafka retention or the bronze table after a bug fix.

**7. Monitoring**

- Pipeline health: consumer lag, checkpoint duration and failures, end-to-end latency (event time to queryable), DLQ rate.
- Data health: event volume against expected baseline by platform and event type, schema violations, null rates, and a daily reconciliation of streamed revenue against the orders database.
- Alerts that page on lag or freshness SLO breaches, not on every spike.

**Trade-offs to mention:** Kappa (one streaming codebase plus replay) versus Lambda (streaming for speed, batch for correctness). Paying for a real-time OLAP store only for genuinely live use cases. And stating explicitly that sub-second personalisation and 10-second dashboards are different SLOs with different paths.

### 4.41 How would you design a data platform for a mid-size company from scratch, and what would you build vs buy?

For a mid-size company (say 200 to 2,000 employees, a data team of 5 to 20, terabytes rather than petabytes), the right platform is mostly **bought and assembled**, with engineering time spent on data modelling, quality and business value. Build only what differentiates the business or what no product does well. Start with use cases and constraints, not tools.

**1. Start with requirements**

- **Use cases for the first 12 months:** management reporting, product analytics, finance reconciliation, maybe one ML use case such as churn or demand forecasting.
- **Sources:** SaaS tools (CRM, ads, payments), one or two OLTP databases, event tracking.
- **Constraints:** cloud provider already in use, data residency, compliance (GDPR, SOC 2), budget, and team skills (SQL-heavy or Python-heavy).
- **Non-functional needs:** freshness (daily vs hourly vs real-time), number of users, uptime expectations.

**2. Reference architecture**

| Layer | Default choice | Build or buy |
| --- | --- | --- |
| Ingestion (SaaS) | Fivetran, Airbyte or cloud-native connectors | Buy; connectors are commodity |
| Ingestion (databases) | Managed CDC (Debezium on managed Kafka, or vendor CDC) | Buy or configure |
| Events | Segment, RudderStack or Snowplow | Buy; build only for unusual scale or privacy needs |
| Storage and compute | Snowflake, BigQuery or Databricks (lakehouse on Iceberg/Delta) | Buy |
| Transformation | dbt with medallion-style layers | Adopt open source; build the models |
| Orchestration | Airflow (managed), Dagster or Prefect, or platform-native jobs | Buy managed |
| Quality and observability | dbt tests, plus an observability tool if needed | Mostly buy or configure |
| Catalog and governance | Platform catalog (Unity, Horizon, Dataplex) | Buy; build classification rules |
| BI and semantic layer | Looker, Power BI, or a semantic layer plus BI tool | Buy; build certified metrics |
| ML | Platform ML tools, MLflow, a feature store only if needed | Buy or adopt; build models |

**3. What to build**

- The **data models** themselves: conformed dimensions, core facts, and certified metrics. This is where the business value is.
- **Conventions and templates:** naming, layering, a dbt project structure, CI checks, and pipeline templates so new sources follow one pattern.
- **Glue and automation:** Terraform for all infrastructure, CI/CD, cost dashboards, and access provisioning workflows.
- **Domain-specific logic** no vendor understands, such as a pricing engine or proprietary scoring.

**4. Decision criteria for build vs buy**

- Is it a **differentiator**? If not, buy.
- **Total cost of ownership:** a "free" open-source tool needs on-call engineers. One engineer at about \$150k a year is often more expensive than the SaaS licence it replaces.
- **Lock-in and exit cost:** prefer open formats (Parquet, Iceberg or Delta), SQL-based transformations and open orchestration, so the warehouse or vendor can be swapped later.
- **Team skills and hiring:** pick tools the market knows.
- **Security and compliance:** SSO, audit logs, data residency, and the vendor's certifications.

**5. Phased rollout**

1. **First 90 days:** ingestion for the top sources, warehouse or lakehouse, dbt, orchestration, and one high-value dashboard trusted by leadership. Set up IaC, CI and access control from day one.
2. **Months 3 to 9:** data quality tests and observability, a catalog, cost monitoring, self-service for analysts, and a semantic layer for key metrics.
3. **Months 9 to 18:** streaming only where a use case needs it, ML platform capabilities, data contracts with source teams, and domain ownership if the organisation is large enough.

**Common mistakes:** building a Kubernetes-plus-Spark platform when a managed warehouse would do; buying ten tools before having one trusted metric; skipping governance until an audit forces it; and having no owner for each dataset. A senior candidate frames success in business terms: time to answer a new question, trust in the numbers, and cost per insight, not in the number of tools deployed.

### 4.42 How would you design an enterprise ontology, and how do data platforms and AI agents use it?

An **ontology** is a formal model of a domain: its entity types, their attributes, the relationships between them and the rules that constrain them (see 8.50 for the definition and 20.13 for its role in retrieval). An **enterprise ontology** applies this to the business itself (customers, accounts, products, orders, suppliers, assets) so that every system, analyst and AI application shares one meaning for each concept.

**How it differs from related artefacts**

| Artefact | What it defines | Example |
| --- | --- | --- |
| Business glossary | Words and definitions | "Active customer: purchased in the last 90 days" |
| Semantic (metrics) layer | Certified metrics and dimensions for analytics | Revenue, churn rate by region |
| Data catalog | Where data lives, owners, lineage | Table `sales.orders`, owned by the commerce team |
| Ontology | Entity types, relationships and rules across systems | A Customer *owns* Accounts; an Order *contains* Products and *is fulfilled by* a Warehouse |
| Knowledge graph | The actual instances and facts organised by the ontology | Customer 123 owns Account 456 |

**Design process**

1. **Start from use cases, not from modelling the whole world.** Write **competency questions** the ontology must answer, for example "Which customers are exposed to supplier X's delayed shipments?" or "Which contracts reference a sanctioned entity?". They define the scope and later serve as tests.
2. **Reuse existing standards** where they fit: schema.org for general entities, FIBO for financial services, SNOMED CT and HL7 FHIR in healthcare, GS1 for products and supply chains. Extend rather than reinvent.
3. **Model the core:** a small set of entity types with clear identifiers, the key relationships, cardinalities and constraints. Keep it to what the use cases need; add detail when a new use case requires it.
4. **Choose a representation:**
   - **RDF and OWL** with SHACL constraints: standards-based, strong for reasoning, integration and publishing linked data; a steeper learning curve.
   - **Labelled property graph** (for example Neo4j) with a documented schema: easier for engineers, good for graph analytics and applications.
   - **Object model in a data platform** (the approach popularised by Palantir's ontology): business objects backed by datasets, with typed links and **actions** that applications and agents can invoke.
5. **Map sources to the ontology:** link each system's tables and fields to ontology types and properties. Resolve identities across systems (entity resolution, see 28.34) so that "customer" in the CRM, billing and support systems becomes one entity. Decide whether the knowledge graph is materialised (copied into a graph store) or virtual (queries translated to source systems).
6. **Govern it like a product:** named owners per domain, versioning, a change process with review, documentation, and deprecation rules. An unowned ontology drifts out of date quickly.

**How data platforms and AI use it**

- **Data integration and quality:** one canonical model to map sources into; constraints become automated validation rules.
- **Discovery and governance:** the catalog links datasets to business concepts; access policies can be defined on concepts (for example, all attributes typed as personal data).
- **GraphRAG and question answering:** the ontology defines which entities and relations to extract from documents and how to traverse them for multi-hop questions (see 20.13).
- **Structured extraction with LLMs:** ontology types become the output schemas for extraction, so LLM outputs are typed and validated against the ontology's constraints before loading.
- **Text-to-SQL and analytics assistants:** describing entities, relationships and join paths in ontology terms gives the model a far better map of the data than raw table names.
- **AI agents:** the ontology gives agents a typed world model: objects they can look up, relationships they can follow, and permitted actions with defined inputs and authorisation. This makes agent behaviour more reliable, auditable and safer than giving them raw database access.

**Common pitfalls**

- Trying to model the entire enterprise before delivering a use case ("boiling the ocean").
- Academic over-modelling that engineers and analysts cannot use.
- No adoption path: if applications and pipelines do not depend on it, it becomes documentation that nobody updates.
- Ignoring identity resolution, so the graph contains several unconnected versions of the same customer.

**Measuring success:** competency questions answered correctly, number of sources and applications mapped to it, time to integrate a new source, data quality issues caught by ontology constraints, and accuracy of AI applications grounded in it.

```python
from rdflib import Graph, Namespace, Literal, RDF, RDFS

EX = Namespace("https://example.com/ontology/")
g = Graph()
g.bind("ex", EX)

# Schema: classes and relationships
for cls in ["Customer", "Account", "Order", "Product"]:
    g.add((EX[cls], RDF.type, RDFS.Class))
g.add((EX.owns, RDFS.domain, EX.Customer))
g.add((EX.owns, RDFS.range, EX.Account))
g.add((EX.placedBy, RDFS.domain, EX.Order))
g.add((EX.placedBy, RDFS.range, EX.Customer))

# Instances (the knowledge graph)
g.add((EX.cust_123, RDF.type, EX.Customer))
g.add((EX.cust_123, RDFS.label, Literal("Asha Shrestha")))
g.add((EX.acct_456, RDF.type, EX.Account))
g.add((EX.cust_123, EX.owns, EX.acct_456))
g.add((EX.order_789, RDF.type, EX.Order))
g.add((EX.order_789, EX.placedBy, EX.cust_123))

# Traverse: which accounts belong to the customer who placed order_789?
customer = g.value(EX.order_789, EX.placedBy)
print([str(a) for a in g.objects(customer, EX.owns)])
```

### 4.43 How do you choose between Snowflake, BigQuery, Databricks and Redshift (or Microsoft Fabric) as the core data platform?

All of them can run a modern analytics platform well, so the decision rarely turns on raw performance. It turns on **workload mix** (SQL and BI versus heavy data science, ML and streaming), **the cloud you are already on**, **the team's skills**, **the pricing model against your usage pattern**, and **how much openness you need** through open table formats. Run a proof of concept on your own representative workloads before committing.

**How they differ**

| Dimension | Snowflake | BigQuery | Databricks | Redshift | Microsoft Fabric |
| --- | --- | --- | --- | --- | --- |
| Heritage | SQL warehouse, now broader | Serverless SQL warehouse | Spark and lakehouse, now strong SQL | AWS warehouse | Unified SaaS analytics on Azure |
| Clouds | AWS, Azure, GCP | GCP (with Omni for others) | AWS, Azure, GCP | AWS | Azure |
| Compute model | Virtual warehouses billed per second | Serverless slots: on-demand per TB scanned or reserved capacity | Clusters and serverless SQL warehouses billed in DBUs plus cloud VMs | Provisioned RA3 clusters or Redshift Serverless | Capacity units shared across workloads |
| Open formats | Iceberg tables, Polaris / Horizon catalog | BigLake and Iceberg tables | Delta native, Iceberg via UniForm, Unity Catalog | Reads Iceberg and data in S3 | OneLake on Delta |
| Strengths | Simplicity, concurrency, data sharing, governance features | Zero operations, massive scans, GCP ML integration | ML and AI, Python, streaming, notebooks, large-scale ETL | Deep AWS integration, predictable provisioned cost | Microsoft ecosystem, Power BI integration |
| Watch-outs | Credit spend on idle or oversized warehouses; serverless features bill in the background | On-demand bills on careless queries; slot contention | More knobs to manage; cost visibility across DBUs and cloud bills | More tuning (sort and distribution keys, WLM) | Maturity varies by component; capacity sizing |

**Decision criteria**

1. **Workloads for the next three years.** Mostly BI and SQL transformations favours Snowflake or BigQuery for simplicity. Significant ML, unstructured data, streaming or Python-heavy engineering favours Databricks (or a lakehouse with several engines).
2. **Cloud alignment and data gravity.** Egress, networking, identity and procurement are easier on your main cloud. BigQuery on GCP, Redshift on AWS and Fabric in Microsoft-centred organisations have natural advantages.
3. **Team skills.** SQL-first analytics teams adopt warehouses faster; engineering-heavy teams make better use of Databricks.
4. **Cost model fit.** Model your actual usage: steady versus spiky load, scan-heavy versus compute-heavy queries, concurrency. Compare reserved capacity and commitments, and include the people cost of tuning.
5. **Openness and exit cost.** Prefer storage in Iceberg or Delta on your own buckets with a standard catalog, so another engine can read the same data later.
6. **Governance and security.** Fine-grained access control, lineage, audit, data residency, and certifications your regulators require.
7. **Ecosystem.** Connectors, BI tools, dbt support, and the managed services you depend on.

**Running the evaluation**

- Pick five to ten real workloads: the heaviest transformation, a high-concurrency dashboard, an ad hoc analyst query, an ML feature pipeline, a streaming ingest.
- Measure runtime, concurrency behaviour **and** cost under each platform's realistic configuration, with vendor engineers allowed to tune; vendor-chosen benchmarks are not evidence.
- Score operational effort, security fit and developer experience with the people who will use it.
- Negotiate commercial terms after the technical decision is clear, with exit and data-portability terms.

**Senior framing:** many large organisations end up with two platforms (for example Databricks for engineering and ML, Snowflake or BigQuery for BI). That can be reasonable if both read the same open tables through one catalog; it is expensive if data is copied between them. The decision is reversible only if storage is open, which is why the format and catalog choice matters as much as the engine (see 4.35).

### 4.44 How would you migrate a legacy on-premises data warehouse or Hadoop platform to the cloud?

Treat it as a business programme, not a lift of tables. Inventory what exists and what is actually used, choose a target architecture, migrate in **waves by business domain or data product** with old and new systems running in parallel, prove equivalence with automated reconciliation, move consumers deliberately, and only then decommission. Most failed migrations underestimate the hidden logic (stored procedures, ETL tool mappings, reports with embedded SQL) and the effort of validating that numbers match.

**1. Discovery and scoping**

- Inventory tables, ETL jobs, stored procedures, schedules, reports and users, with **usage data** from query logs. Typically a large share of objects is unused; do not migrate them.
- Map dependencies with lineage: which sources feed which tables, which reports depend on which tables.
- Classify data sensitivity and residency requirements early; they constrain the target design.
- Agree success criteria: cost, performance, decommission date, and which business processes must not be disrupted (month-end close, regulatory reports).

**2. Choose the migration approach per workload**

| Approach | What it means | When |
| --- | --- | --- |
| Rehost (lift and shift) | Same engine on cloud VMs, or Hadoop to a managed Hadoop service | Deadline-driven exit from a data centre; low value in the long run |
| Replatform | Move SQL and data to a cloud warehouse or lakehouse with automated code conversion | Most warehouse workloads |
| Refactor / rebuild | Redesign pipelines (for example, legacy ETL tool mappings to dbt or Spark, nightly loads to CDC) | High-value, badly designed or frequently changing areas |
| Retire | Do not migrate | Unused or duplicated objects |

**3. Target architecture and foundations first**

Landing zone, networking and private connectivity, identity and access, IaC, a catalog, an orchestration standard, CI/CD and cost monitoring. Building these before wave one avoids re-migrating early workloads.

**4. Migrate in waves**

1. **Historical data:** bulk transfer (network transfer services, or physical appliances for very large volumes), converted to Parquet-based tables, partitioned and clustered for the new engine.
2. **Ongoing changes:** CDC from source systems into the new platform, so both platforms stay current during the parallel run (see 7.23).
3. **Code:** convert SQL dialects and procedures with automated translation tools (cloud migration services, `sqlglot`-based transpilers), then fix what does not convert by hand. Rewrite procedural logic as set-based models.
4. **Consumers:** repoint reports, extracts and applications, ideally through a semantic layer or views so future changes are cheaper.

**5. Validate equivalence automatically**

```python
def reconcile(legacy_df, cloud_df, keys, measures, tol=1e-6):
    legacy = legacy_df.groupby(keys)[measures].sum()
    cloud = cloud_df.groupby(keys)[measures].sum()
    joined = legacy.join(cloud, lsuffix="_old", rsuffix="_new", how="outer").fillna(0)
    diffs = {m: (joined[f"{m}_old"] - joined[f"{m}_new"]).abs() > tol for m in measures}
    return joined[sum(diffs.values()).astype(bool)]
```

Compare row counts, aggregates by key business dimensions, and sample row-level hashes for every migrated table, every day during the parallel run. Expect differences from collation, rounding, time zones and NULL handling, and document each explained difference.

**6. Cut over and decommission**

Switch consumers per wave with a rollback plan, freeze changes on the legacy side, and set a firm decommission date. Running two platforms indefinitely is the most expensive outcome.

**Risks to name in an interview**

- Underestimating hidden business logic and the long tail of reports.
- Cloud costs exceeding expectations because legacy workloads were never optimised; set budgets and FinOps practices from day one (see 4.39).
- Performance regressions from different engine characteristics (no indexes, different join behaviour).
- Organisational change: training, new on-call processes, and stakeholder trust built through visible reconciliation results.

**What interviewers look for:** a phased plan with measurable checkpoints, usage-driven scoping, automated validation, parallel running with clear cut-over criteria, and attention to people and cost, not only to the technology.

---

← [Chapter 3. Computer Science and Software Design](../part-1-foundations/03-computer-science-and-software-design.md) · [Table of contents](../README.md) · [Chapter 5. SQL, Databases and Data Modeling](05-sql-databases-and-data-modeling.md) →
