<!-- notion-id: 3d967418285480448e90e6349ebbdb9b -->
# Data engineering interview question

Apache Airflow Q&A plus a topic checklist and links to the OBenner data-engineering-interview-questions repo.

## Apache Airflow

### Q1. What is Airflow?

Apache Airflow is an open-source workflow management platform. It began in October 2014 at Airbnb as a solution for managing the company's increasingly complex workflows. Airbnb's creation of Airflow enabled them to programmatically author, schedule, and monitor their workflows via the built-in Airflow user interface. Airflow is a data transformation pipeline ETL (Extract, Transform, Load) workflow orchestration tool.

### Q2. What issues does Airflow resolve?

Crons are an old technique of task scheduling. Scalable Cron requires external assistance to log, track, and manage tasks. The Airflow UI is used to track and monitor the workflow's execution. Creating and maintaining a relationship between tasks in cron is a challenge, whereas it is as simple as writing Python code in Airflow. Cron jobs are not reproducible until they are configured externally. Airflow maintains an audit trail of all tasks completed.

### Q3. Explain how a workflow is designed in Airflow?

A directed acyclic graph (DAG) is used to design an Airflow workflow. That is to say, when creating a workflow, consider how it can be divided into tasks that can be completed independently. The tasks can then be combined into a graph to form a logical whole. The overall logic of your workflow is based on the shape of the graph. An Airflow DAG can have multiple branches, and you can choose which ones to follow and which to skip during workflow execution. Airflow Pipeline DAG Airflow could be completely stopped, and able to run workflows would then resume through restarting the last unfinished task. It is important to remember that airflow operators can be run more than once when designing airflow operators. Each task should be idempotent, or capable of being performed multiple times without causing unintended consequences.

### More Airflow questions (links)

- [Explain how workflow is designed in Airflow?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#Explain-how-workflow-is-designed-in-Airflow)
- [Explain Airflow Architecture and its components?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#Explain-Airflow-Architecture-and-its-components)
- [What are the types of Executors in Airflow?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-are-the-types-of-Executors-in-Airflow)
- [What are the pros and cons of SequentialExecutor?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-are-the-pros-and-cons-of-SequentialExecutor)
- [What are the pros and cons of LocalExecutor?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-are-the-pros-and-cons-of-LocalExecutor)
- [What are the pros and cons of CeleryExecutor?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-are-the-pros-and-cons-of-CeleryExecutor)
- [What are the pros and cons of KubernetesExecutor?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-are-the-pros-and-cons-of-KubernetesExecutor)
- [How to define a workflow in Airflow?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#How-to-define-a-workflow-in-Airflow)
- [How do you make the module available to airflow if you're using Docker Compose?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#How-do-you-make-the-module-available-to-airflow-if-you're-using-Docker-Compose)
- [How to schedule DAG in Airflow?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#How-to-schedule-DAG-in-Airflow)
- [What is XComs In Airflow?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-is-XComs-In-Airflow)
- [What is xcom\_pull in XCom Airflow?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-is-xcom_pull-in-XCom-Airflow)
- [What is Jinja templates?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#What-is-Jinja-templates)
- [How to use Airflow XComs in Jinja templates?](https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/airflow.md#How-to-use-Airflow-XComs-in-Jinja-templates)

## Other topics

These topic headings exist in Notion with no notes under them yet:

- Apache Avro
- Apache Cassandra
- Apache Flume
- Apache Hadoop
- Apache Hive
- Apache Impala
- Apache Kafka
- Apache NiFi
- Apache Spark
- Apache Flink
- Apache Hbase
- Amazon Web Services
- Google Cloud Platform
- BigQuery
- BigTable
- DWH Architectures
- Microsoft Azure
- Tableau
- Apache Superset
- Looker
- Kubernetes
- SQL
- Greenplum
- Redshift
- Data Structures
- dbt
- Apache Iceberg
- Change Data Capture (CDC)
- Data Modeling
- Data Quality
- Data Observability
- Data Governance
- Apache Hudi
- Cost Optimization
- Python for Data Engineering
- Data System Design

## Reference

- <https://github.com/OBenner/data-engineering-interview-questions/blob/master/content/full.md#DWH-Architectures>
