# Chapter 4. Data Engineering Fundamentals and Architecture

_Part II — Data Engineering_

Warehouses, lakes and lakehouses, storage formats, cloud data platforms, data quality, security and governance. These questions test whether a candidate can design a data platform, not just operate one.

This chapter contains **46 questions**.

## Contents

- **4.1** [What is the fundamental goal of Data Engineering?](#41-what-is-the-fundamental-goal-of-data-engineering)
- **4.2** [Explain the difference between a Data Warehouse and a Data Lake.](#42-explain-the-difference-between-a-data-warehouse-and-a-data-lake)
- **4.3** [What is the CAP Theorem, and how does it affect your choice of database?](#43-what-is-the-cap-theorem-and-how-does-it-affect-your-choice-of-database)
- **4.4** [Describe the "Medallion Architecture."](#44-describe-the-medallion-architecture)
- **4.5** [What is "Data Lineage" and why is it critical?](#45-what-is-data-lineage-and-why-is-it-critical)
- **4.6** [Explain Batch vs. Stream processing.](#46-explain-batch-vs-stream-processing)
- **4.7** [What is "Idempotency" in a data pipeline?](#47-what-is-idempotency-in-a-data-pipeline)
- **4.8** [What is the difference between Structured, Semi-Structured, and Unstructured data?](#48-what-is-the-difference-between-structured-semi-structured-and-unstructured-data)
- **4.9** [What is a "Single Point of Failure" (SPOF) in a data system?](#49-what-is-a-single-point-of-failure-spof-in-a-data-system)
- **4.10** [What is the "Small File Problem" in distributed storage?](#410-what-is-the-small-file-problem-in-distributed-storage)
- **4.11** [Why is Parquet preferred over CSV for data engineering?](#411-why-is-parquet-preferred-over-csv-for-data-engineering)
- **4.12** [What is the difference between Avro and Parquet?](#412-what-is-the-difference-between-avro-and-parquet)
- **4.13** [Explain "Object Storage" (S3) vs. "Block Storage" (EBS).](#413-explain-object-storage-s3-vs-block-storage-ebs)
- **4.14** [What is Snowflake's "Unique Architecture"?](#414-what-is-snowflakes-unique-architecture)
- **4.15** [What is a "Partition Key" in S3 or HDFS?](#415-what-is-a-partition-key-in-s3-or-hdfs)
- **4.16** [What is "Serverless" in the context of data pipelines?](#416-what-is-serverless-in-the-context-of-data-pipelines)
- **4.17** [What is the "Modern Data Stack"?](#417-what-is-the-modern-data-stack)
- **4.18** [What is a "Data Mesh"?](#418-what-is-a-data-mesh)
- **4.19** [What is "Data Lakehouse"?](#419-what-is-data-lakehouse)
- **4.20** [What is Infrastructure as Code (IaC)?](#420-what-is-infrastructure-as-code-iac)
- **4.21** [What is "Data Compaction"?](#421-what-is-data-compaction)
- **4.22** [What is "Data Tiering"?](#422-what-is-data-tiering)
- **4.23** [Explain the difference between "Vertical" and "Horizontal" Scaling.](#423-explain-the-difference-between-vertical-and-horizontal-scaling)
- **4.24** [What is a "VPC" and why do data engineers care?](#424-what-is-a-vpc-and-why-do-data-engineers-care)
- **4.25** [What is a "Managed Service"?](#425-what-is-a-managed-service)
- **4.26** [What is "Data Governance"?](#426-what-is-data-governance)
- **4.27** [How do you handle PII (Personally Identifiable Information)?](#427-how-do-you-handle-pii-personally-identifiable-information)
- **4.28** [What is "Data Quality Profiling"?](#428-what-is-data-quality-profiling)
- **4.29** [What is "dbt" and why is it so popular?](#429-what-is-dbt-and-why-is-it-so-popular)
- **4.30** [Explain the concept of "Data Contracts."](#430-explain-the-concept-of-data-contracts)
- **4.31** [What is the difference between "Data Observability" and "Monitoring"?](#431-what-is-the-difference-between-data-observability-and-monitoring)
- **4.32** [How do you implement "Unit Testing" for a data transformation?](#432-how-do-you-implement-unit-testing-for-a-data-transformation)
- **4.33** [What is "Data Compaction" in a streaming context?](#433-what-is-data-compaction-in-a-streaming-context)
- **4.34** [How do you handle "Schema Drift" automatically?](#434-how-do-you-handle-schema-drift-automatically)
- **4.35** [Where do you see the future of Data Engineering?](#435-where-do-you-see-the-future-of-data-engineering)
- **4.36** [What are open table formats (Apache Iceberg, Delta Lake, Apache Hudi), and what problems do they solve on top of Parquet files in object storage?](#436-what-are-open-table-formats-apache-iceberg-delta-lake-apache-hudi-and-what-problems-do-they-solve-on-top-of-parquet-files-in-object-storage)
- **4.37** [How do snapshots, time travel and optimistic concurrency work in Iceberg or Delta Lake?](#437-how-do-snapshots-time-travel-and-optimistic-concurrency-work-in-iceberg-or-delta-lake)
- **4.38** [What is the difference between copy-on-write and merge-on-read tables, and when would you pick each?](#438-what-is-the-difference-between-copy-on-write-and-merge-on-read-tables-and-when-would-you-pick-each)
- **4.39** [Compare Apache Iceberg, Delta Lake and Apache Hudi. How would you choose between them?](#439-compare-apache-iceberg-delta-lake-and-apache-hudi-how-would-you-choose-between-them)
- **4.40** [How do you implement the GDPR "right to be forgotten" in a data lake or lakehouse?](#440-how-do-you-implement-the-gdpr-right-to-be-forgotten-in-a-data-lake-or-lakehouse)
- **4.41** [How do you implement fine-grained access control (row-level security, column masking, tag-based policies) on a data platform?](#441-how-do-you-implement-fine-grained-access-control-row-level-security-column-masking-tag-based-policies-on-a-data-platform)
- **4.42** [What is a semantic (metrics) layer and why do organisations adopt one?](#442-what-is-a-semantic-metrics-layer-and-why-do-organisations-adopt-one)
- **4.43** [How do you control and optimise cost on a cloud data platform (warehouse and lakehouse FinOps)?](#443-how-do-you-control-and-optimise-cost-on-a-cloud-data-platform-warehouse-and-lakehouse-finops)
- **4.44** [How would you design a real-time clickstream analytics pipeline for a large e-commerce site?](#444-how-would-you-design-a-real-time-clickstream-analytics-pipeline-for-a-large-e-commerce-site)
- **4.45** [How would you design a data platform for a mid-size company from scratch, and what would you build vs buy?](#445-how-would-you-design-a-data-platform-for-a-mid-size-company-from-scratch-and-what-would-you-build-vs-buy)
- **4.46** [How would you design an enterprise ontology, and how do data platforms and AI agents use it?](#446-how-would-you-design-an-enterprise-ontology-and-how-do-data-platforms-and-ai-agents-use-it)

---

### 4.1 What is the fundamental goal of Data Engineering?

**Data engineering aims to transform raw, messy data into a high-quality, reliable asset for analysis. The primary goal is to build and maintain the "plumbing", the data pipelines, that ensure data is accurate, available, and performant for data scientists and analysts.**

### 4.2 Explain the difference between a Data Warehouse and a Data Lake.

- **Data Warehouse:** A centralized system that stores **structured, processed, and cleaned data** for reporting, analytics, and business intelligence.
- **Data Lake:** A centralized repository that stores **raw structured, semi-structured, and unstructured data** in its original form.

| Feature | Data Warehouse | Data Lake |
| --- | --- | --- |
| Data | Structured/processed | Raw, structured, semi-structured, unstructured |
| Schema | Schema-on-write | Schema-on-read |
| Purpose | BI, reporting, analytics | Data science, ML, big-data processing |
| Data processing | Before storage | Often after storage |
| Users | Analysts, BI teams | Data scientists, engineers, ML teams |
| Example | Snowflake, BigQuery | Amazon S3, Azure Data Lake |

**In short:**

**Data Warehouse = structured, processed data for analytics.**

**Data Lake = raw data in various formats for flexible analysis and ML.**

### 4.3 What is the CAP Theorem, and how does it affect your choice of database?

#### CAP Theorem

The **CAP Theorem** states that a distributed database system cannot simultaneously guarantee all three of these properties during a network partition:

- **Consistency (C):** Every read receives the most recent write or an error.
- **Availability (A):** Every request receives a response, even if some nodes fail.
- **Partition Tolerance (P):** The system continues operating despite communication failures between nodes.

When a **network partition** occurs, you generally must choose between:

- **CP:** Prioritize consistency over availability.
- **AP:** Prioritize availability over strict consistency.

#### How it affects database choice

The choice depends on application requirements:

- Choose a more **consistency-focused (CP)** database when incorrect or stale data is unacceptable, such as for critical financial or strongly consistent workflows.
- Choose a more **availability-focused (AP)** database when the system must remain responsive during failures and temporary inconsistency is acceptable.

**In short:** CAP helps you choose a distributed database based on the trade-off between **consistency and availability during network partitions**.

### 4.4 Describe the "Medallion Architecture."

**Medallion Architecture** is a data architecture pattern that organizes data into **three progressive layers**—**Bronze, Silver, and Gold**—to improve data quality and make analytics easier.

- **Bronze Layer:** Stores **raw data** in its original form with minimal transformation.
- **Silver Layer:** Contains **cleaned, validated, deduplicated, and transformed data**.
- **Gold Layer:** Contains **business-ready, aggregated, and optimized data** for reporting, dashboards, and analytics.

#### Example

**Raw APIs/Logs → Bronze → Silver → Gold → BI/ML**

| Layer | Purpose |
| --- | --- |
| Bronze | Raw data |
| Silver | Cleaned and standardized data |
| Gold | Business-ready/aggregated data |

**In short:** **Bronze = Raw → Silver = Cleaned → Gold = Business-ready.**

### 4.5 What is "Data Lineage" and why is it critical?

**Data lineage is a map that shows how data travels from source to destination, including all transformations. It is critical for troubleshooting bugs, ensuring regulatory compliance, and performing impact analysis when a source table changes.**

### 4.6 Explain Batch vs. Stream processing.

- **Batch Processing:** Processes a **large volume of data at scheduled intervals** rather than immediately.
  - Suitable for large historical datasets.
  - Higher latency but efficient for bulk processing.
  - Examples: daily payroll, monthly reports, ETL jobs.
  - Tools: **Apache Spark, Airflow**.
- **Stream Processing:** Processes data **continuously as it arrives**, enabling near-real-time results.
  - Suitable for real-time monitoring and event-driven applications.
  - Low latency but requires continuous infrastructure.
  - Examples: fraud detection, real-time analytics, IoT data.
  - Tools: **Apache Kafka, Apache Flink, Spark Streaming**.

| Feature | Batch | Stream |
| --- | --- | --- |
| Processing | In groups | Continuously |
| Latency | Higher | Low |
| Data | Large accumulated datasets | Incoming events |
| Use case | Reports, ETL | Real-time analytics, alerts |

**In short:** **Batch = process data periodically; Stream = process data continuously as it arrives.**

### 4.7 What is "Idempotency" in a data pipeline?

**Idempotency** in a data pipeline means that **processing the same data multiple times produces the same final result as processing it once**.

- Prevents **duplicate records** when a pipeline retries a failed job.
- Makes pipelines **safe to retry**.
- Commonly implemented using **unique IDs, deduplication, upserts, checkpoints, or merge operations**.
- Especially important in distributed and streaming systems.

**Example:** If an order with `order_id = 123` is processed twice, an idempotent pipeline ensures the database still contains **one correct order**, not two duplicates.

**In short:** **Idempotency = repeated processing does not create unintended side effects or duplicate results.**

### 4.8 What is the difference between Structured, Semi-Structured, and Unstructured data?

**Structured data**

- Organized into a fixed schema (rows/columns, like a database table)
- Every record has the same fields and data types
- Easy to query with SQL, sort, and search
- Examples: spreadsheets, customer databases, financial records

**Semi-structured data**

- No rigid table format, but has tags, labels, or key-value pairs for partial organization
- More flexible than structured data, less chaotic than unstructured
- Requires some parsing, but structure is discoverable
- Examples: JSON, XML, emails (structured fields + free-text body), HTML

**Unstructured data**

- No predefined format or schema
- Doesn't fit into rows/columns
- Harder to search/analyze directly; often needs NLP or computer vision to extract meaning
- Examples: plain text, images, audio, video, social media posts

**Quick way to remember it:**

- Structured = filled-out form
- Semi-structured = email with subject line + free-text body
- Unstructured = photo or voice memo

### 4.9 What is a "Single Point of Failure" (SPOF) in a data system?

A **Single Point of Failure (SPOF)** is any component in a system that, if it fails, causes the entire system (or a critical part of it) to stop working — because there's no backup or redundancy to take over.

**In a data system context:**

- If one server, database, network link, or piece of hardware handles a critical function alone, and it goes down, the whole system can become unavailable, corrupted, or inaccessible.
- The danger isn't just downtime — it can also mean data loss if that single component was the only place data was stored.

**Common examples of SPOFs in data systems:**

- A single database server with no replica — if it crashes, all reads/writes stop
- One network switch or router connecting critical infrastructure
- A single power supply to a server rack
- One person who's the only one who knows how to maintain a system (a "human SPOF")
- A single authentication service that all applications depend on

**How SPOFs are typically addressed:**

- **Redundancy** — having backup servers, databases, or network paths ready to take over
- **Replication** — keeping copies of data across multiple locations or machines
- **Failover systems** — automatic switching to a backup component when the primary fails
- **Load balancing** — distributing work across multiple systems so no single one is critical
- **Regular backups** — ensuring data isn't lost even if a system fails

**Why it matters:** SPOFs are a major concern in system design because they represent fragility — the more critical the system (e.g., financial transactions, healthcare records, cloud infrastructure), the more important it is to eliminate or minimize single points of failure to ensure **high availability** and **fault tolerance**.

### 4.10 What is the "Small File Problem" in distributed storage?

**What it is:**

- A performance issue in distributed storage systems (e.g., HDFS, S3) caused by having millions of small files instead of fewer, large ones

**Why it happens:**

- These systems are designed and optimized for large files, often split into large fixed-size blocks (e.g., 128MB/256MB in HDFS)
- Millions of tiny files break this design assumption and create overhead the system wasn't built for

**Core problems it causes:**

- **Metadata overload** — every file needs metadata tracked (name, location, permissions); in HDFS, this is held in memory by the **NameNode**, which can become overwhelmed or even crash under too many small files
- **Wasted storage** — small files still consume block-level tracking overhead, even if they don't fill a block
- **Slower processing** — frameworks like MapReduce/Spark create a task per file/block; millions of tiny tasks mean scheduling overhead that outweighs the actual work
- **High I/O overhead** — many small reads mean many disk seeks/network/API calls instead of fast sequential reads
- **Slower listing** — even in object stores like S3 (no NameNode equivalent), listing and querying many small files is slower and costlier

**Common causes:**

- Streaming systems writing frequent small batches
- Log aggregation generating many small log files
- Poorly designed pipelines that don't consolidate output

**Common solutions:**

- **Compaction** — periodically merging small files into larger ones (often sized to match the block size, e.g., 128MB+)
- **Container/columnar formats** — using Parquet, ORC, or Avro to bundle records efficiently
- **Batch writes** — accumulating data before writing instead of writing frequently
- **Hadoop Archives (HAR)** — packing small files together to reduce NameNode load

**Bottom line:**

- Distributed storage is built for scale via large files — too many small files create bottlenecks in both metadata management and processing/IO efficiency, and the standard fix is scheduled compaction into larger, well-formatted files

### 4.11 Why is Parquet preferred over CSV for data engineering?

**Parquet is a columnar format, meaning it only reads the specific columns requested. It also features built-in compression and metadata, making it much faster and cheaper for large-scale queries.**

### 4.12 What is the difference between Avro and Parquet?

**Parquet is columnar and optimized for heavy read/analytical workloads. Avro is row-based and optimized for write-heavy streaming and handling complex schema evolution.**

### 4.13 Explain "Object Storage" (S3) vs. "Block Storage" (EBS).

**Object Storage (S3) is highly scalable, web-accessible storage for any file type. Block Storage (EBS) acts like a local hard drive for a specific server, offering lower latency but less flexibility.**

### 4.14 What is Snowflake's "Unique Architecture"?

**Snowflake separates Storage, Compute, and Services. This allows users to scale processing power (compute) up or down instantly without affecting the underlying data (storage).**

### 4.15 What is a "Partition Key" in S3 or HDFS?

**It is a column (like `date`) used to organize files into folders. This enables "Partition Pruning," allowing a query to skip entire folders of irrelevant data.**

### 4.16 What is "Serverless" in the context of data pipelines?

**Services like AWS Lambda where the cloud provider manages all infrastructure. You only pay for the time your code is actually running, with no servers to maintain.**

### 4.17 What is the "Modern Data Stack"?

**A suite of cloud-native tools centered around a warehouse. It typically includes tools for ingestion (Fivetran), storage (Snowflake), transformation (dbt), and visualization (Tableau).**

### 4.18 What is a "Data Mesh"?

**A decentralized architecture where data is treated as a product. Individual business units (like Marketing) own and manage their own pipelines rather than relying on a central data team.**

### 4.19 What is "Data Lakehouse"?

A **Data Lakehouse** is exactly the architecture I just described — I can go a bit deeper here.

**The problem it solves:**

Traditionally, companies ran two separate systems:

- A **data lake** (S3, ADLS, GCS) — cheap storage for raw, unstructured/semi-structured data, great for data science and ML, but poor at fast SQL queries, no transactions, easy to turn into a "data swamp"
- A **data warehouse** (Snowflake, Redshift, BigQuery, or older systems like Teradata) — great for fast, reliable SQL analytics with ACID guarantees and indexing, but expensive, proprietary storage formats, and not great for unstructured data or ML workloads

This meant duplicating data: raw data lived in the lake, then got copied/transformed into the warehouse for BI — two copies, two pipelines, more cost, more staleness.

**The Lakehouse fix:**

Keep data in one place — cheap, open-format object storage — but add a **transactional metadata layer** on top that gives it warehouse-like guarantees:

- **ACID transactions** — reliable concurrent reads/writes, no corrupted data from failed jobs
- **Schema enforcement + evolution** — structure like a warehouse table, but able to change over time
- **Time travel** — query previous versions of a table via the transaction log
- **Data skipping/indexing via metadata** — fast queries without needing a separate proprietary engine
- **Open file formats** (usually Parquet) — so any compute engine (Spark, Trino, Flink, DuckDB, Snowflake) can read the same data, no vendor lock-in
- **One copy of data** serving both BI/SQL analytics _and_ data science/ML workloads

**The enabling technologies (table formats):**

| Format | Backed by | Notes |
| --- | --- | --- |
| **Delta Lake** | Databricks | Most mature ecosystem, `_delta_log` transaction log |
| **Apache Iceberg** | Netflix (open governance, Apache) | Strong multi-engine interoperability, widely adopted (Snowflake, Trino, Spark, Flink) |
| **Apache Hudi** | Uber | Strong for streaming upserts/incremental ingestion |

**Why it's become popular:** It reduces duplicate infrastructure, keeps costs closer to a data lake, and gives near-warehouse performance and reliability — which is why platforms like Databricks (Delta Lake) and increasingly Snowflake/Snowflake+Iceberg have leaned hard into this model.

Since I answered this same question just above — is there a specific part you'd like me to go deeper on, like how the transaction log actually works, or how Iceberg/Delta/Hudi differ from each other?

### 4.20 What is Infrastructure as Code (IaC)?

**The practice of managing cloud resources (like S3 buckets or clusters) using code files (e.g., Terraform), ensuring environments are reproducible and version-controlled.**

### 4.21 What is "Data Compaction"?

**The process of merging many small, fragmented data files into a few large ones to maintain high query performance in a Data Lake.**

### 4.22 What is "Data Tiering"?

**Moving older, rarely-accessed data to cheaper, slower storage tiers (like S3 Glacier) to optimize costs while keeping current data in fast "hot" storage.**

### 4.23 Explain the difference between "Vertical" and "Horizontal" Scaling.

**Vertical scaling means adding more RAM or CPU to a single server. Horizontal scaling means adding more servers to a cluster, which is the foundation of distributed big data systems.**

### 4.24 What is a "VPC" and why do data engineers care?

**A Virtual Private Cloud is a private network in the cloud. We use it to isolate data warehouses and pipelines from the public internet for security.**

### 4.25 What is a "Managed Service"?

**A service where the cloud provider handles maintenance and updates (like Amazon RDS), allowing engineers to focus on building rather than server administration.**

### 4.26 What is "Data Governance"?

**The set of processes and policies that define who owns the data, how it is secured, and how its quality is maintained across the organization.**

### 4.27 How do you handle PII (Personally Identifiable Information)?

**I use encryption at rest and in transit, implement data masking (hiding sensitive parts of a string), and use Role-Based Access Control (RBAC) to limit who can see the data.**

### 4.28 What is "Data Quality Profiling"?

**The process of checking data for common issues such as unexpected nulls, duplicate records, or values that fall outside of a logical range (e.g., negative prices).**

### 4.29 What is "dbt" and why is it so popular?

**dbt (data build tool) allows engineers to write transformations in SQL while providing software engineering features like version control, automated testing, and documentation.**

### 4.30 Explain the concept of "Data Contracts."

**An agreement between data providers and consumers that defines the schema and quality of the data, ensuring that changes at the source don't silently break downstream apps.**

### 4.31 What is the difference between "Data Observability" and "Monitoring"?

**Monitoring focuses on binary states (up/down) and predefined thresholds. Observability uses logs, metrics, and traces to understand the internal state of a complex system and troubleshoot unknown "silent" failures, like data drift.**

### 4.32 How do you implement "Unit Testing" for a data transformation?

**I use a framework like `pytest`. I create a small, "mock" dataset with known values, pass it through the transformation function, and assert that the output matches the expected "golden" result.**

### 4.33 What is "Data Compaction" in a streaming context?

**In Kafka, log compaction ensures that the topic retains at least the last known value for each record key. This is useful for restoring state in downstream databases after a crash.**

### 4.34 How do you handle "Schema Drift" automatically?

**I use a "Schema Evolution" strategy where the pipeline detects new columns in the incoming JSON/Parquet and automatically updates the destination table schema or routes the record to an "alert" queue for review.**

### 4.35 Where do you see the future of Data Engineering?

**The field is moving toward Data Observability (automated monitoring), the rise of AI-augmented pipelines, and the unification of batch and stream processing into a single "Lakehouse" architecture.**

### 4.36 What are open table formats (Apache Iceberg, Delta Lake, Apache Hudi), and what problems do they solve on top of Parquet files in object storage?

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

### 4.37 How do snapshots, time travel and optimistic concurrency work in Iceberg or Delta Lake?

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

### 4.38 What is the difference between copy-on-write and merge-on-read tables, and when would you pick each?

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

### 4.39 Compare Apache Iceberg, Delta Lake and Apache Hudi. How would you choose between them?

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

### 4.40 How do you implement the GDPR "right to be forgotten" in a data lake or lakehouse?

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

### 4.41 How do you implement fine-grained access control (row-level security, column masking, tag-based policies) on a data platform?

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

### 4.42 What is a semantic (metrics) layer and why do organisations adopt one?

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

### 4.43 How do you control and optimise cost on a cloud data platform (warehouse and lakehouse FinOps)?

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

### 4.44 How would you design a real-time clickstream analytics pipeline for a large e-commerce site?

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

### 4.45 How would you design a data platform for a mid-size company from scratch, and what would you build vs buy?

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

### 4.46 How would you design an enterprise ontology, and how do data platforms and AI agents use it?

An **ontology** is a formal model of a domain: its entity types, their attributes, the relationships between them and the rules that constrain them (see 8.88 for the definition and 20.46 for its role in retrieval). An **enterprise ontology** applies this to the business itself (customers, accounts, products, orders, suppliers, assets) so that every system, analyst and AI application shares one meaning for each concept.

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
- **GraphRAG and question answering:** the ontology defines which entities and relations to extract from documents and how to traverse them for multi-hop questions (see 20.15).
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

---

← [Chapter 3. Computer Science and Software Design](../part-1-foundations/03-computer-science-and-software-design.md) · [Table of contents](../README.md) · [Chapter 5. SQL, Databases and Data Modeling](05-sql-databases-and-data-modeling.md) →
