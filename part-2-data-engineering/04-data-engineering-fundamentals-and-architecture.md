# Chapter 4. Data Engineering Fundamentals and Architecture

_Part II — Data Engineering_

Warehouses, lakes and lakehouses, storage formats, cloud data platforms, data quality, security and governance. These questions test whether a candidate can design a data platform, not just operate one.

This chapter contains **35 questions**.

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

---

← [Chapter 3. Computer Science and Software Design](../part-1-foundations/03-computer-science-and-software-design.md) · [Table of contents](../README.md) · [Chapter 5. SQL, Databases and Data Modeling](05-sql-databases-and-data-modeling.md) →
