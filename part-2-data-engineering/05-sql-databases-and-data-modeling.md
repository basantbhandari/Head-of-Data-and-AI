# Chapter 5. SQL, Databases and Data Modeling

_Part II — Data Engineering_

Relational fundamentals, SQL, indexing, normalization, dimensional modeling and NoSQL trade-offs.

This chapter contains **49 questions**.

## Contents

- **5.1** [What are NoSQL databases? Mention different types of NoSQL databases.](#51-what-are-nosql-databases-mention-different-types-of-nosql-databases)
- **5.2** [Explain SQL injection.](#52-explain-sql-injection)
- **5.3** [Differentiate between clustered and non-clustered indexes?](#53-differentiate-between-clustered-and-non-clustered-indexes)
- **5.4** [What are some common problems with ORMs?](#54-what-are-some-common-problems-with-orms)
- **5.5** [What is the difference between MySQL and SQL?](#55-what-is-the-difference-between-mysql-and-sql)
- **5.6** [What are the various subsets in SQL?](#56-what-are-the-various-subsets-in-sql)
- **5.7** [What is RDBMS?](#57-what-is-rdbms)
- **5.8** [What is the "SELECT" statement?](#58-what-is-the-select-statement)
- **5.9** [What are the most common clauses used with a SQL SELECT query?](#59-what-are-the-most-common-clauses-used-with-a-sql-select-query)
- **5.10** [What is OLTP?](#510-what-is-oltp)
- **5.11** [What are the differences between OLAP and OLTP?](#511-what-are-the-differences-between-olap-and-oltp)
- **5.12** [What are SQL comments, and how do they work?](#512-what-are-sql-comments-and-how-do-they-work)
- **5.13** [What do you mean by Denormalization?](#513-what-do-you-mean-by-denormalization)
- **5.14** [What's an Index?](#514-whats-an-index)
- **5.15** [What is the DISTINCT statement? What is its use?](#515-what-is-the-distinct-statement-what-is-its-use)
- **5.16** [What is a "Trigger" in SQL?](#516-what-is-a-trigger-in-sql)
- **5.17** [What's the difference between a natural join and a cross join?](#517-whats-the-difference-between-a-natural-join-and-a-cross-join)
- **5.18** [What is a UNIQUE constraint?](#518-what-is-a-unique-constraint)
- **5.19** [What is a FOREIGN KEY in SQL?](#519-what-is-a-foreign-key-in-sql)
- **5.20** [How to display the current date in SQL?](#520-how-to-display-the-current-date-in-sql)
- **5.21** [Explain Star Schema vs. Snowflake Schema with pros/cons.](#521-explain-star-schema-vs-snowflake-schema-with-proscons)
- **5.22** [What are SQL window functions?](#522-what-are-sql-window-functions)
- **5.23** [What is the difference between RANK(), DENSE\_RANK(), and ROW\_NUMBER()?](#523-what-is-the-difference-between-rank-dense_rank-and-row_number)
- **5.24** [Explain OLTP vs. OLAP.](#524-explain-oltp-vs-olap)
- **5.25** [What are the types of Slowly Changing Dimensions (SCD)?](#525-what-are-the-types-of-slowly-changing-dimensions-scd)
- **5.26** [What is a "Factless Fact Table"?](#526-what-is-a-factless-fact-table)
- **5.27** [How do you optimize a slow SQL query?](#527-how-do-you-optimize-a-slow-sql-query)
- **5.28** [What is a Surrogate Key vs. a Natural Key?](#528-what-is-a-surrogate-key-vs-a-natural-key)
- **5.29** [What is a "Common Table Expression" (CTE)?](#529-what-is-a-common-table-expression-cte)
- **5.30** [Explain ACID properties in detail.](#530-explain-acid-properties-in-detail)
- **5.31** [What is an "Upsert"?](#531-what-is-an-upsert)
- **5.32** [What is a "Self-Join" and when would you use it?](#532-what-is-a-self-join-and-when-would-you-use-it)
- **5.33** [What is Data Normalization and its levels (1NF, 2NF, 3NF)?](#533-what-is-data-normalization-and-its-levels-1nf-2nf-3nf)
- **5.34** [What is a Clustered Index?](#534-what-is-a-clustered-index)
- **5.35** [What is the difference between UNION and UNION ALL?](#535-what-is-the-difference-between-union-and-union-all)
- **5.36** [Assuming you have a two-column table of users and their friends coupled with a two-column table of users and their liked pages. Write a SQL query to make recommendations using pages that your friends have liked when it should not recommend pages that you have liked.](#536-assuming-you-have-a-two-column-table-of-users-and-their-friends-coupled-with-a-two-column-table-of-users-and-their-liked-pages-write-a-sql-query-to-make-recommendations-using-pages-that-your-friends-have-liked-when-it-should-not-recommend-pages-that-you-have-liked)
- **5.37** [When can SQL indexing make a SELECT query slower?](#537-when-can-sql-indexing-make-a-select-query-slower)
- **5.38** [Compare Kimball, Inmon and Data Vault modeling approaches. When would you choose each?](#538-compare-kimball-inmon-and-data-vault-modeling-approaches-when-would-you-choose-each)
- **5.39** [What is the grain of a fact table, and what are transaction, periodic snapshot and accumulating snapshot fact tables?](#539-what-is-the-grain-of-a-fact-table-and-what-are-transaction-periodic-snapshot-and-accumulating-snapshot-fact-tables)
- **5.40** [What are conformed dimensions and the bus matrix, and why do they matter?](#540-what-are-conformed-dimensions-and-the-bus-matrix-and-why-do-they-matter)
- **5.41** [How do you implement an SCD Type 2 dimension with a MERGE statement?](#541-how-do-you-implement-an-scd-type-2-dimension-with-a-merge-statement)
- **5.42** [How do you solve a "gaps and islands" problem in SQL?](#542-how-do-you-solve-a-gaps-and-islands-problem-in-sql)
- **5.43** [How would you sessionize clickstream events in SQL with a 30-minute inactivity timeout?](#543-how-would-you-sessionize-clickstream-events-in-sql-with-a-30-minute-inactivity-timeout)
- **5.44** [How do you deduplicate rows and keep only the latest record per key?](#544-how-do-you-deduplicate-rows-and-keep-only-the-latest-record-per-key)
- **5.45** [What are transaction isolation levels and the anomalies each prevents? How does MVCC work?](#545-what-are-transaction-isolation-levels-and-the-anomalies-each-prevents-how-does-mvcc-work)
- **5.46** [How do you read a query execution plan, and how do nested loop, hash and merge joins differ?](#546-how-do-you-read-a-query-execution-plan-and-how-do-nested-loop-hash-and-merge-joins-differ)
- **5.47** [What are the common NULL pitfalls in SQL?](#547-what-are-the-common-null-pitfalls-in-sql)
- **5.48** [How do partitioning and clustering work in cloud data warehouses like BigQuery and Snowflake, and how do they enable pruning?](#548-how-do-partitioning-and-clustering-work-in-cloud-data-warehouses-like-bigquery-and-snowflake-and-how-do-they-enable-pruning)
- **5.49** [How would you write a cohort retention query?](#549-how-would-you-write-a-cohort-retention-query)

---

### 5.1 What are NoSQL databases? Mention different types of NoSQL databases.

- **NoSQL (Not Only SQL) databases** are databases designed to store and retrieve data using flexible, non-relational data models rather than the traditional relational table-and-row structure.
- They are commonly used for **large-scale, distributed, high-throughput applications** where flexible schemas and horizontal scalability are important.

#### Types of NoSQL databases

1. **Document databases**
   - Store data as documents, commonly JSON/BSON.
   - Examples: **MongoDB, CouchDB**
2. **Key-Value databases**
   - Store data as key-value pairs.
   - Examples: **Redis, Amazon DynamoDB**
3. **Wide-Column databases**
   - Store data in column families and are optimized for distributed workloads.
   - Examples: **Apache Cassandra, Google Bigtable**
4. **Graph databases**
   - Store entities as nodes and relationships as edges.
   - Examples: **Neo4j, Amazon Neptune**

**In short:**

**NoSQL = flexible, scalable databases designed for non-relational data models.**

### 5.2 Explain SQL injection.

- **SQL Injection (SQLi)** is a security vulnerability where an attacker manipulates an application's **SQL query through untrusted input**, potentially causing the database to perform unintended operations.
- **Example:** If an application directly concatenates user input into a SQL query, malicious input could alter the query's logic.
- **Potential impact:**
  - Unauthorized access to data
  - Modification or deletion of database records
  - Authentication bypass
  - Exposure of sensitive information
- **How to prevent SQL injection:**
  - Use **parameterized queries / prepared statements**
  - Use a trusted **ORM** correctly
  - Validate and constrain user input
  - Apply **least-privilege** database permissions
  - Avoid dynamically constructing SQL with raw user input

**In short:** SQL injection happens when **untrusted input is treated as SQL code**. The primary defense is to **separate SQL code from user-supplied data using parameterized queries**.

### 5.3 Differentiate between clustered and non-clustered indexes?

- **Clustered Index:**
  - Determines the **physical order of rows** in a table based on the index key.
  - A table can generally have **only one clustered index** because the data can have only one physical ordering.
  - Efficient for **range queries and sequential data retrieval**.
  - Example: An index on `employee_id` where table rows are stored in employee ID order.
- **Non-Clustered Index:**
  - Creates a **separate index structure** containing the indexed values and pointers/references to the actual table rows.
  - A table can have **multiple non-clustered indexes**.
  - Useful for quickly searching different columns.
  - Example: Separate indexes on `email`, `department`, and `last_name`.

**In short:**

**Clustered index = determines the physical order of table data.**

**Non-clustered index = separate lookup structure pointing to the table data.**

### 5.4 What are some common problems with ORMs?

- **Performance overhead:** ORMs can generate inefficient SQL compared with carefully optimized raw SQL.
- **N+1 query problem:** Loading related objects incorrectly can cause many unnecessary database queries.
- **Complex queries:** Advanced joins, aggregations, window functions, or database-specific features can be difficult to express efficiently through an ORM.
- **Less control over SQL:** Developers may have less visibility and control over the exact queries being executed.
- **Abstraction limitations:** ORM abstractions can hide important database behavior and make performance problems harder to diagnose.
- **Migration issues:** Schema migrations can become complicated, especially with large production databases.
- **Memory usage:** ORMs may load more objects into application memory than necessary.
- **Learning curve:** Developers need to understand both the ORM and the underlying database to use it effectively.

**In short:** ORMs improve developer productivity and maintainability, but they can introduce **performance overhead, inefficient queries, and reduced control over database operations**.

### 5.5 What is the difference between MySQL and SQL?

- **SQL (Structured Query Language):**
  - A **standard language** used to interact with relational databases.
  - Used for creating, querying, updating, and managing data.
  - Examples of SQL commands: `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`.
- **MySQL:**
  - A **relational database management system (RDBMS)** that uses SQL to manage and interact with databases.
  - It provides additional features such as storage engines, indexing, transactions, user management, and replication.

**In short:**

**SQL = language**

**MySQL = database system that uses SQL**

Other RDBMSs that use SQL include **PostgreSQL, Oracle Database, Microsoft SQL Server, and SQLite**.

### 5.6 What are the various subsets in SQL?

#### Main Subsets of SQL

- **DDL — Data Definition Language**
  - Used to define and modify database structure.
  - Commands: `CREATE`, `ALTER`, `DROP`, `TRUNCATE`
- **DML — Data Manipulation Language**
  - Used to insert, modify, and delete data.
  - Commands: `INSERT`, `UPDATE`, `DELETE`
- **DQL — Data Query Language**
  - Used to retrieve data.
  - Command: `SELECT`
- **DCL — Data Control Language**
  - Used to control database access and permissions.
  - Commands: `GRANT`, `REVOKE`
- **TCL — Transaction Control Language**
  - Used to manage database transactions.
  - Commands: `COMMIT`, `ROLLBACK`, `SAVEPOINT`

**In short:**

**DDL = structure \| DML = data changes \| DQL = data retrieval \| DCL = permissions \| TCL = transactions**

### 5.7 What is RDBMS?

- **RDBMS (Relational Database Management System)** is software used to **store, organize, manage, and retrieve data in relational databases**.
- Data is stored in **tables consisting of rows and columns**.
- Tables can be connected using **relationships**, typically through primary keys and foreign keys.
- RDBMSs usually use **SQL** to create, query, and manage data.
- They support important features such as **transactions, constraints, indexing, concurrency control, and data integrity**.

**Examples:** MySQL, PostgreSQL, Oracle Database, Microsoft SQL Server, and SQLite.

**In short:**

**RDBMS = software that manages structured data stored in related tables.**

### 5.8 What is the "SELECT" statement?

- **`SELECT`** is an SQL statement used to **retrieve data from one or more database tables**.
- It can be used with clauses such as `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`, and `JOIN`.
- It does **not normally modify** the data.

Example:

```sql
SELECT name, email
FROM users
WHERE age >= 18;
```

This retrieves the **name and email** of users whose age is 18 or above.

**In short:** `SELECT` is used to **query and retrieve data from a database**.

### 5.9 What are the most common clauses used with a SQL SELECT query?

#### Common SQL `SELECT` Clauses

- **`FROM`** — Specifies the table(s) from which to retrieve data.
- **`WHERE`** — Filters rows based on a condition.
- **`JOIN`** — Combines data from multiple tables.
- **`GROUP BY`** — Groups rows for aggregate calculations.
- **`HAVING`** — Filters grouped results.
- **`ORDER BY`** — Sorts the result.
- **`LIMIT`** — Restricts the number of rows returned.

Example:

```sql
SELECT department, COUNT(*) AS employee_count
FROM employees
WHERE salary > 50000
GROUP BY department
HAVING COUNT(*) > 5
ORDER BY employee_count DESC
LIMIT 10;
```

**In short:**

`SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT` is a common query structure.

### 5.10 What is OLTP?

- **OLTP (Online Transaction Processing)** is a type of database system designed to handle **frequent, real-time business transactions**.
- It focuses on **fast inserts, updates, deletes, and short queries**.
- OLTP systems typically support **many concurrent users and transactions**.
- They emphasize **data consistency, integrity, and transaction reliability**.
- Common examples include **banking systems, e-commerce orders, payment systems, and payroll systems**.

**In short:**

**OLTP = fast, reliable processing of day-to-day transactions in real time.**

### 5.11 What are the differences between OLAP and OLTP?

#### OLAP vs OLTP

| Feature | OLTP | OLAP |
| --- | --- | --- |
| Full form | Online Transaction Processing | Online Analytical Processing |
| Purpose | Handle **day-to-day transactions** | Perform **analysis and reporting** |
| Operations | `INSERT`, `UPDATE`, `DELETE` | Mostly complex `SELECT` queries |
| Data | Current, operational data | Historical and aggregated data |
| Queries | Simple and fast | Complex and resource-intensive |
| Users | Clerks, applications, customers | Analysts, managers, data scientists |
| Database design | Usually normalized | Often denormalized |
| Transactions | High volume, short transactions | Fewer, long-running analytical queries |
| Example | Banking, payments, order processing | Business intelligence, dashboards, sales analysis |

**In short:**

- **OLTP = process transactions**
- **OLAP = analyze data**

### 5.12 What are SQL comments, and how do they work?

- **SQL comments** are notes written inside SQL code that are **ignored by the database engine**. They are used to explain queries, document logic, or temporarily disable code.

#### Types of SQL Comments

- **Single-line comment:** Use `-`

```sql
-- Get all users
SELECT * FROM users;
```

- **Multi-line comment:** Use `/* ... */`

```sql
/*
  Retrieve active users
  from the users table
*/
SELECT * FROM users
WHERE active = 1;
```

**In short:** SQL comments improve **code readability and maintainability** and are not executed as SQL statements.

### 5.13 What do you mean by Denormalization?

**Denormalization** is the process of **intentionally adding redundant data to a database** to improve query performance and reduce the need for complex joins.

- **Purpose:** Improve read/query performance.
- **Redundancy:** Some data may be stored in multiple places.
- **Fewer joins:** Related information can be accessed more quickly.
- **Trade-off:** Requires more storage and can make updates more complex.
- Common in **OLAP/data warehouses** where fast analytical queries are important.

**Example:** Instead of joining `Customers` and `Orders` every time, an order table may also store the customer's name to make frequent queries faster.

**In short:** Denormalization = **adding controlled redundancy to improve read performance.**

### 5.14 What's an Index?

An **index** in a database is a data structure that **speeds up data retrieval** from a table, similar to an index in a book.

- **Faster queries:** Helps the database find rows without scanning the entire table.
- **Commonly created on:** Columns frequently used in `WHERE`, `JOIN`, and `ORDER BY`.
- **Trade-off:** Indexes require additional storage and can slow down `INSERT`, `UPDATE`, and `DELETE` operations.
- **Types:** Clustered and non-clustered indexes are common concepts; databases may also support B-tree, hash, and other index types.

**Example:** An index on `user_id` allows the database to find a specific user much faster than scanning every row.

**In short:** An index **improves database query performance by providing a faster way to locate data**.

### 5.15 What is the DISTINCT statement? What is its use?

**`DISTINCT`** is a SQL keyword used with `SELECT` to **remove duplicate values from the query result**.

- Returns **unique values**.
- Can be applied to one or multiple columns.
- Does **not modify the actual data** in the table.

Example:

```sql
SELECT DISTINCT city
FROM customers;
```

If multiple customers are from Kathmandu, the result will contain **Kathmandu only once**.

**In short:** `DISTINCT` is used to **retrieve unique, non-duplicate values** from a query result.

### 5.16 What is a "Trigger" in SQL?

A **Trigger** in SQL is a special database program that **automatically executes when a specific event occurs** on a table or database.

- **Events:** `INSERT`, `UPDATE`, or `DELETE`.
- **Automatic:** It runs automatically without being explicitly called.
- **Common uses:** Auditing, validation, maintaining related data, and enforcing business rules.
- **Types:** `BEFORE` and `AFTER` triggers are commonly supported, depending on the database.

**Example:** When a new employee is inserted, a trigger can automatically record the action in an audit table.

**In short:** A trigger is **automatically executed SQL logic that responds to database events**.

### 5.17 What's the difference between a natural join and a cross join?

- **Natural Join:** Automatically joins two tables based on columns with the **same name and compatible data types**. It returns rows where those common columns have matching values.
- **Cross Join:** Produces the **Cartesian product** of two tables, combining every row from the first table with every row from the second table. It does not require a join condition.

**Example:**

- Table A has 3 rows, and Table B has 4 rows.
- `CROSS JOIN` → **3 × 4 = 12 rows**.
- `NATURAL JOIN` → Only matching rows based on common column names.

**In short:**

**NATURAL JOIN = joins on matching common columns.**

**CROSS JOIN = combines every possible pair of rows.**

### 5.18 What is a UNIQUE constraint?

A **`UNIQUE` constraint** in SQL ensures that **all values in a column or combination of columns are unique**, preventing duplicate values.

- Prevents duplicate data.
- Can be applied to one or multiple columns.
- A table can have **multiple `UNIQUE` constraints**.
- `NULL` Handling depends on the database system.
- Commonly used for fields such as email addresses, usernames, or employee IDs.

Example:

```sql
CREATE TABLE users (
    id INT,
    email VARCHAR(255) UNIQUE
);
```

**In short:** `UNIQUE` ensures that **duplicate values are not allowed** in the specified column(s).

### 5.19 What is a FOREIGN KEY in SQL?

A **FOREIGN KEY** in SQL is a column or set of columns that **references a primary key or unique key in another table**. It is used to establish a relationship between tables and maintain **referential integrity**.

- Links two related tables.
- Prevents invalid references to non-existent records.
- A table can have multiple foreign keys.
- A foreign key can reference another table's `PRIMARY KEY` or suitable `UNIQUE` key.

Example:

```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
```

Here, `customer_id` in `orders` references `customer_id` in `customers`.

**In short:** A foreign key **links tables and ensures that related data remains valid and consistent**.

### 5.20 How to display the current date in SQL?

The syntax for displaying the **current date** depends on the SQL database:

- **MySQL:** `SELECT CURDATE();`
- **PostgreSQL:** `SELECT CURRENT_DATE;`
- **SQL Server:** `SELECT CAST(GETDATE() AS DATE);`
- **Oracle:** `SELECT SYSDATE FROM dual;`

**In short:** For MySQL, use **`SELECT CURDATE();`**.

### 5.21 Explain Star Schema vs. Snowflake Schema with pros/cons.

**Star Schema**

Structure:

- A central **fact table** (containing measurable, quantitative data — e.g., sales amount, quantity) connected directly to multiple **dimension tables** (e.g., customer, product, time, store)
- Dimension tables are **denormalized** — each one is flat, containing all descriptive attributes in a single table, even if there's redundancy

Pros:

- **Simple and intuitive** — easy to understand and navigate, resembles a star shape
- **Fast query performance** — fewer joins needed since dimensions aren't split into sub-tables
- **Easier for BI tools** — most reporting/BI tools are optimized for star schemas
- **Simplified SQL queries** — less complexity for end users writing reports

Cons:

- **Data redundancy** — denormalized dimensions can repeat data (e.g., city and country repeated for every customer in that city), increasing storage use
- **Data integrity risks** — updates need to be applied across repeated fields, which can lead to inconsistencies if not managed carefully

---

**Snowflake Schema**

Structure:

- Same central fact table, but dimension tables are **normalized** — split into multiple related sub-tables (e.g., a "Product" dimension might break into Product → Category → Department)
- Looks like a snowflake shape due to the branching sub-dimensions

Pros:

- **Reduced data redundancy** — normalization eliminates repeated data, saving storage space
- **Better data integrity** — easier to maintain consistency since data isn't duplicated
- **More granular structure** — useful when dimension hierarchies are complex

Cons:

- **More complex queries** — requires more joins across normalized tables, which can hurt performance
- **Harder to understand** — less intuitive for end users and report writers
- **Slower query performance** — especially with large datasets, due to the extra joins

---

**Quick comparison**

| Aspect | Star Schema | Snowflake Schema |
| --- | --- | --- |
| Structure | Denormalized dimensions | Normalized dimensions |
| Query complexity | Simple (fewer joins) | Complex (more joins) |
| Query speed | Faster | Slower (generally) |
| Storage | More redundancy, more space | Less redundancy, less space |
| Ease of use | Easier for BI/reporting | Harder to navigate |
| Best for | Simpler reporting, performance-critical systems | Storage optimization, complex hierarchies |

**When to use which:**

- **Star schema** — most common choice in modern data warehousing, especially when query speed and simplicity matter more than storage savings (storage is cheap today)
- **Snowflake schema** — better when data integrity and normalization matter more, or when dimension hierarchies are deep and reused across many fact tables

### 5.22 What are SQL window functions?

**SQL Window Functions** let you perform calculations across a set of rows that are related to the current row — without collapsing those rows into a single output (unlike `GROUP BY`, which aggregates and reduces the number of rows).

**Key idea:**

- They operate on a "window" (a defined subset of rows) using the `OVER()` clause
- Each row keeps its individual identity, but also gets access to values from other rows in its window (like running totals, rankings, or comparisons to previous/next rows)

**Basic syntax:**

```sql
function_name() OVER (
    PARTITION BY column_name  -- optional: divides rows into groups
    ORDER BY column_name      -- optional: defines order within the group
)
```

**Common window functions:**

- `ROW_NUMBER()` — assigns a unique sequential number to each row
- `RANK()` / `DENSE_RANK()` — assigns a rank, with or without gaps for ties
- `SUM()`, `AVG()`, `COUNT()` — used as window functions for running totals/averages
- `LAG()` / `LEAD()` — access previous/next row's value
- `NTILE(n)` — divides rows into n buckets

---

**Example:**

Say you have a sales table:

| employee | department | sales |
| --- | --- | --- |
| Alice | Sales | 500 |
| Bob | Sales | 700 |
| Carol | Marketing | 400 |
| Dave | Marketing | 600 |

You want to **rank employees by sales within their department**:

```sql
SELECT 
    employee,
    department,
    sales,
    RANK() OVER (PARTITION BY department ORDER BY sales DESC) AS sales_rank
FROM sales_table;
```

**Result:**

| employee | department | sales | sales\_rank |
| --- | --- | --- | --- |
| Bob | Sales | 700 | 1 |
| Alice | Sales | 500 | 2 |
| Dave | Marketing | 600 | 1 |
| Carol | Marketing | 400 | 2 |

Notice: unlike `GROUP BY`, all rows are preserved — you still see every employee, but now each row also carries their rank **within their department**.

---

**Why window functions matter:**

- They let you do running totals, moving averages, rankings, and row-to-row comparisons **without losing row-level detail**
- Very useful for reporting, analytics, and time-series data (e.g., calculating month-over-month growth using `LAG()`)

### 5.23 What is the difference between RANK(), DENSE\_RANK(), and ROW\_NUMBER()?

All three are **window ranking functions** used with `OVER(ORDER BY ...)`, but they differ in how they handle **ties** and **numbering**.

---

**`ROW_NUMBER()`**

- Assigns a **unique, sequential number** to every row, regardless of ties
- No two rows ever get the same number — even if their values are identical, order is broken arbitrarily (or by additional tiebreaker columns)

**`RANK()`**

- Assigns the **same rank to tied rows**
- **Skips** the next rank(s) after a tie (leaves gaps)

**`DENSE_RANK()`**

- Also assigns the **same rank to tied rows**
- **Does NOT skip** ranks after a tie — ranks stay consecutive

---

**Example:**

Sales data:

| employee | sales |
| --- | --- |
| Alice | 700 |
| Bob | 700 |
| Carol | 500 |
| Dave | 400 |

```sql
SELECT 
    employee,
    sales,
    ROW_NUMBER() OVER (ORDER BY sales DESC) AS row_num,
    RANK() OVER (ORDER BY sales DESC) AS rank_num,
    DENSE_RANK() OVER (ORDER BY sales DESC) AS dense_rank_num
FROM sales_table;
```

**Result:**

| employee | sales | row\_num | rank\_num | dense\_rank\_num |
| --- | --- | --- | --- | --- |
| Alice | 700 | 1 | 1 | 1 |
| Bob | 700 | 2 | 1 | 1 |
| Carol | 500 | 3 | 3 | 2 |
| Dave | 400 | 4 | 4 | 3 |

---

**Key differences to notice:**

- **`ROW_NUMBER()`** — Alice and Bob (tied at 700) get different numbers: 1 and 2
- **`RANK()`** — Alice and Bob both get rank 1, but Carol jumps to rank **3** (rank 2 is skipped)
- **`DENSE_RANK()`** — Alice and Bob both get rank 1, and Carol gets rank **2** (no gap)

---

**When to use which:**

- **`ROW_NUMBER()`** — when you need strictly unique row identifiers (e.g., pagination, picking "top 1 per group")
- **`RANK()`** — when ties should share a rank, and you want the ranking to reflect how many rows came before (e.g., competition-style ranking: "1st, 1st, 3rd")
- **`DENSE_RANK()`** — when ties should share a rank, but you want no gaps in the ranking sequence (e.g., grouping into rank tiers)

### 5.24 Explain OLTP vs. OLAP.

**OLTP (Online Transaction Processing)**

What it is:

- Systems designed to manage and process **real-time transactional data** — the day-to-day operations of a business
- Optimized for **fast, frequent, small read/write operations**

Characteristics:

- Handles a **high volume of short transactions** (inserts, updates, deletes)
- Data is typically **normalized** (to reduce redundancy and ensure fast, consistent writes)
- Prioritizes **data integrity** and **speed** for individual transactions
- Users: front-line staff, applications, customers (e.g., placing an order, withdrawing cash from an ATM)

Examples:

- Banking systems (processing withdrawals/deposits)
- E-commerce checkout systems
- Airline reservation systems
- Point-of-sale (POS) systems

---

**OLAP (Online Analytical Processing)**

What it is:

- Systems designed for **complex analysis and reporting** on large volumes of historical data
- Optimized for **read-heavy, complex queries** across large datasets

Characteristics:

- Handles **fewer, but much larger and more complex queries** (aggregations, trends, multi-dimensional analysis)
- Data is typically **denormalized** (e.g., star/snowflake schemas) for faster query performance
- Prioritizes **query speed** and **analytical flexibility** over transactional speed
- Users: analysts, executives, data scientists (e.g., generating quarterly sales reports)

Examples:

- Data warehouses (e.g., Snowflake, Amazon Redshift)
- Business intelligence dashboards
- Sales trend analysis
- Financial forecasting reports

---

**Quick comparison**

| Aspect | OLTP | OLAP |
| --- | --- | --- |
| Purpose | Manage daily transactions | Analyze historical data |
| Query type | Simple (CRUD operations) | Complex (aggregations, joins) |
| Data structure | Normalized | Denormalized (star/snowflake) |
| Speed priority | Fast writes | Fast reads/aggregations |
| Data volume per query | Small | Large |
| Users | Operational staff, apps | Analysts, decision-makers |
| Examples | Banking, e-commerce, POS | Data warehousing, BI dashboards |

---

**How they work together:**

- OLTP systems generate the **raw transactional data** (e.g., every sale, every transaction)
- This data is periodically extracted, transformed, and loaded (**ETL**) into an OLAP system (data warehouse)
- OLAP systems then let analysts run complex queries on that data **without slowing down or interfering with** the live OLTP system

**In short:** OLTP = "what's happening right now" (operations), OLAP = "what happened and why" (analysis).

### 5.25 What are the types of Slowly Changing Dimensions (SCD)?

**Slowly Changing Dimensions (SCD)** refer to strategies for handling changes in dimension table data over time in a data warehouse — since attributes like a customer's address, an employee's department, or a product's category can change, but you often need to track history for accurate reporting.

---

**Type 0 — Retain Original**

- The attribute **never changes** once loaded — it's fixed/immutable
- Any updates are ignored; the original value is always kept
- Use case: attributes that shouldn't ever change (e.g., date of birth, original signup date)

---

**Type 1 — Overwrite**

- The old value is **simply overwritten** with the new value
- **No history is kept** — only the current value exists
- Use case: correcting errors (e.g., fixing a misspelled name) where historical tracking isn't needed

Example:

| Before | After |
| --- | --- |
| Customer: John, City: Chicago | Customer: John, City: Boston |
| (Old "Chicago" value is gone) |  |

---

**Type 2 — Add New Row**

- A **new row is added** each time a change occurs, preserving full history
- Typically includes columns like `start_date`, `end_date`, and/or a `current_flag` to identify the active record
- Most common and widely used SCD type — enables full historical tracking

Example:

| CustomerID | Name | City | Start Date | End Date | Current Flag |
| --- | --- | --- | --- | --- | --- |
| 1 | John | Chicago | 2020-01-01 | 2023-05-01 | N |
| 1 | John | Boston | 2023-05-01 | NULL | Y |

---

**Type 3 — Add New Column**

- Instead of adding new rows, a **new column is added** to track the previous value (usually just one prior state)
- Limited history — typically only tracks the "current" and "previous" value, not full history

Example:

| CustomerID | Name | Current City | Previous City |
| --- | --- | --- | --- |
| 1 | John | Boston | Chicago |

---

**Type 4 — History Table**

- Keeps the **current value in the main dimension table**, but maintains a **separate historical table** that logs all changes
- Useful when the main table needs to stay small/fast, but full history still needs to be preserved elsewhere

Example:

- Main table: `Dim_Customer` (current values only)
- History table: `Dim_Customer_History` (logs every change with timestamps)

---

**Type 6 — Hybrid (1+2+3)**

- Combines Type 1, 2, and 3 — keeps a full history (like Type 2) **and** a "current value" column (like Type 1) **and** a "previous value" column (like Type 3)
- Provides maximum flexibility for reporting (both point-in-time and current-state analysis)
- More complex to implement and maintain

---

**Quick comparison**

| Type | History Kept? | Method | Complexity |
| --- | --- | --- | --- |
| 0 | No changes allowed | Fixed value | Very low |
| 1 | No | Overwrite | Low |
| 2 | Full history | New row per change | Medium-High |
| 3 | Limited (1 prior value) | New column | Low-Medium |
| 4 | Full (separate table) | History table | Medium |
| 6 | Full + current + previous | Hybrid of 1, 2, 3 | High |

**Most commonly used in practice:** **Type 1** (simple corrections) and **Type 2** (full historical tracking) — these cover the vast majority of real-world data warehousing needs.

### 5.26 What is a "Factless Fact Table"?

A **Factless Fact Table** is a fact table in a data warehouse that contains **no measurable, numeric facts** — only **foreign keys** linking to dimension tables. Instead of storing a metric (like sales amount or quantity), it simply records that an **event or relationship occurred**.

**Why it's useful:**

Normally, a fact table exists to store measurable data (e.g., `sales_amount`, `quantity_sold`). But sometimes what you need to track isn't a _number_ — it's the fact that something _happened_, or that two things are _related_. That's where factless fact tables come in.

---

**Two Main Types**

**1. Event Tracking**

- Records that an **event occurred** at a specific point involving certain dimensions
- No numeric measure needed — the row's existence _is_ the fact

Example: **Student Attendance**

| StudentID | ClassID | DateID |
| --- | --- | --- |
| 101 | CS101 | 2024-01-10 |
| 102 | CS101 | 2024-01-10 |

- There's no "amount" or "quantity" — just the fact that a student attended a class on a date
- To answer "How many students attended CS101 on Jan 10?" — you **count the rows**

---

**2. Coverage / Relationship Tracking**

- Records a **relationship or condition** that exists between dimensions, even if no event happened
- Often used to capture what _could_ happen or _didn't_ happen (important for "what's missing" analysis)

Example: **Promotions Coverage**

| ProductID | StoreID | PromotionID | DateID |
| --- | --- | --- | --- |
| 501 | 10 | P1 | 2024-01-01 |
| 502 | 10 | P1 | 2024-01-01 |

- This tracks which products were **eligible** for a promotion in a store, regardless of whether they were actually sold
- Useful for analysis like "Which promoted products did NOT sell?" — by comparing this table against actual sales fact tables

---

**Key characteristics:**

- Contains only **foreign keys** (no measures/metrics)
- The **grain** (level of detail) is defined by the combination of dimension keys
- Analysis is typically done by **counting rows** rather than summing/averaging a measure

---

**Common use cases:**

- Student class attendance
- Employee training completion
- Promotion/product eligibility tracking
- Insurance policy coverage (what's covered, not claims amounts)
- Website visit logs (page views without a "value" per view)

---

**Why it matters:**

Factless fact tables let you analyze **absence** and **coverage** — questions like "what didn't happen" or "what could have happened but didn't" — which are hard to answer using only fact tables that store numeric measures.

**In short:** it's a fact table that tracks the **occurrence of events or existence of relationships**, without any quantitative measure attached.

### 5.27 How do you optimize a slow SQL query?

Optimizing a slow SQL query is usually a step-by-step diagnostic process — you need to first understand _why_ it's slow before applying fixes. Here's a structured approach:

---

**1. Diagnose First — Use EXPLAIN / EXPLAIN ANALYZE**

- Run `EXPLAIN` (or `EXPLAIN ANALYZE` in Postgres/MySQL) to see the **query execution plan**
- Look for:
  - **Full table scans** (bad — means no index is being used)
  - **Nested loop joins** on large tables (can be slow)
  - **High cost estimates** or actual row counts far off from estimated ones
  - Steps consuming the most time/resources

---

**2. Indexing**

- **Add indexes** on columns used in `WHERE`, `JOIN`, `ORDER BY`, and `GROUP BY` clauses
- Use **composite indexes** when queries filter on multiple columns together
- Avoid **over-indexing** — too many indexes slow down writes (`INSERT`/`UPDATE`/`DELETE`)
- Check for **unused indexes** and remove them (they add overhead without benefit)
- Consider **covering indexes** (an index that contains all columns needed by the query, avoiding a lookup into the actual table)

---

**3. Rewrite the Query**

- **Avoid `SELECT *`** — only fetch the columns you actually need
- **Avoid functions on indexed columns** in `WHERE` clauses (e.g., `WHERE YEAR(date_col) = 2024` prevents index usage — rewrite as a range: `WHERE date_col >= '2024-01-01' AND date_col < '2025-01-01'`)
- **Replace subqueries with JOINs** where possible (subqueries, especially correlated ones, can be much slower)
- **Use `EXISTS` instead of `IN`** for large subquery result sets (often faster, especially with NULLs involved)
- **Avoid unnecessary `DISTINCT` or `ORDER BY`** if not needed — they add sorting overhead

---

**4. Optimize Joins**

- Ensure **join columns are indexed** on both sides
- **Join smaller result sets first** if the query planner isn't already doing this well
- Avoid **cross joins** or accidental Cartesian products from missing join conditions
- Consider **denormalization** for frequently joined, read-heavy tables (trade storage for speed)

---

**5. Filter Early, Filter Smart**

- Apply **`WHERE` filters before `JOIN`s** where possible so fewer rows are processed downstream
- Use **partition pruning** if the table is partitioned (e.g., by date) — make sure your `WHERE` clause aligns with the partition key
- Avoid filtering on **computed/derived columns** unless they're indexed or materialized

---

**6. Limit Data Volume**

- Use **`LIMIT`** when you don't need the full result set (e.g., pagination)
- **Aggregate at the source** if possible, rather than pulling raw rows and aggregating in the application layer
- Consider **archiving old data** or using **table partitioning** to reduce the working dataset size

---

**7. Database/Infrastructure-Level Fixes**

- **Update table statistics** (`ANALYZE TABLE` in MySQL, `ANALYZE` in Postgres) so the query planner has accurate data for choosing the best plan
- Check for **table bloat** or fragmentation and run maintenance (`VACUUM`, `OPTIMIZE TABLE`, etc.)
- Ensure adequate **hardware resources** (memory for caching, disk I/O speed)
- Consider **caching** frequent query results (e.g., Redis, materialized views) if data doesn't need to be real-time

---

**8. Materialized Views / Pre-Aggregation**

- For expensive, frequently-run aggregation queries, consider **materialized views** that pre-compute and store results, refreshed periodically
- Useful for dashboards/reports where slightly stale data is acceptable

---

**Quick Diagnostic Checklist**

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Full table scan | Missing index | Add index on filtered/joined columns |
| Slow joins | Missing index on join keys, bad join order | Index join columns, rewrite query |
| Slow aggregation | Large unfiltered dataset | Filter earlier, pre-aggregate, add indexes |
| Query plan looks fine but still slow | Stale statistics | Run `ANALYZE`/update stats |
| High CPU/memory usage | Poor query design, no caching | Rewrite query, add caching layer |

---

**In short:** Start by reading the execution plan to find the actual bottleneck, then apply targeted fixes — indexing, query rewriting, better joins, and reducing data volume — rather than guessing. Optimization should always be **measured before and after** the change to confirm it actually helped.

### 5.28 What is a Surrogate Key vs. a Natural Key?

**A Natural Key has a real-world business meaning (like an SSN or email). A Surrogate Key is a system-generated unique ID (like an auto-incrementing integer) that has no inherent business meaning, making it more stable for database changes.**

### 5.29 What is a "Common Table Expression" (CTE)?

A **Common Table Expression (CTE)** is a **temporary, named result set** defined within a SQL query using the `WITH` clause, which you can reference like a table within that query. It exists only for the duration of the query and doesn't persist afterward.

**Basic syntax:**

```sql
WITH cte_name AS (
    SELECT column1, column2
    FROM table_name
    WHERE condition
)
SELECT *
FROM cte_name
WHERE another_condition;
```

---

**Why use a CTE?**

- **Improves readability** — breaks complex queries into logical, named steps instead of deeply nested subqueries
- **Reusability within the query** — you can reference the same CTE multiple times in the main query without repeating the logic
- **Supports recursion** — CTEs can be recursive, which is essential for hierarchical data (e.g., org charts, category trees)
- **Easier debugging** — you can test/run the CTE portion on its own to verify intermediate results

---

**Example 1: Simple CTE**

Instead of a nested subquery:

```sql
SELECT * FROM (
    SELECT department, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
) dept_avg
WHERE avg_salary > 50000;
```

You can write it more clearly with a CTE:

```sql
WITH dept_avg AS (
    SELECT department, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
)
SELECT *
FROM dept_avg
WHERE avg_salary > 50000;
```

---

**Example 2: Multiple CTEs**

You can chain multiple CTEs together:

```sql
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 80000
),
dept_counts AS (
    SELECT department, COUNT(*) AS emp_count
    FROM high_earners
    GROUP BY department
)
SELECT * FROM dept_counts;
```

---

**Example 3: Recursive CTE** (for hierarchical data)

Used to traverse hierarchies like an org chart:

```sql
WITH RECURSIVE employee_hierarchy AS (
    -- Anchor member: top-level manager
    SELECT employee_id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive member: employees reporting to previous level
    SELECT e.employee_id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy;
```

This builds out the full reporting chain, level by level.

---

**CTE vs. Subquery vs. Temp Table**

| Aspect | CTE | Subquery | Temp Table |
| --- | --- | --- | --- |
| Readability | High (named, structured) | Lower (nested) | High |
| Reusability in query | Yes (can reference multiple times) | No (must repeat) | Yes |
| Persistence | Only during query execution | Only during query execution | Persists in session/transaction |
| Supports recursion | Yes | No | No (not directly) |
| Performance | Similar to subquery (not materialized in most DBs\*) | Similar to CTE | Can be indexed, may be faster for reuse across queries |

### 5.30 Explain ACID properties in detail.

**ACID ensures reliable transactions: Atomicity (all or nothing), Consistency (follows database rules, correct), Isolation (transactions don't interfere), and Durability (committed data survives crashes).**

### 5.31 What is an "Upsert"?

**A portmanteau of "Update" and "Insert." It checks if a record exists; if it does, it updates the record; if it doesn't, it inserts a new one. This is often handled by the `MERGE` statement in SQL.**

### 5.32 What is a "Self-Join" and when would you use it?

**A self-join is when a table is joined with itself. It is commonly used for hierarchical data, such as a table where an `employee_id` is linked to a `manager_id` in the same table.**

### 5.33 What is Data Normalization and its levels (1NF, 2NF, 3NF)?

**Normalization reduces redundancy. 1NF ensures atomic values. 2NF removes partial dependencies (non-key columns must depend on the whole key). 3NF removes transitive dependencies (non-key columns must not depend on other non-key columns).**

#### 1 NF

Every column/attribute need to have single value.

Each row should be unique either through single or multiple columns. Not mandatory to have primary key.

#### 2 NF

Must be in 1 NF

All non key attribute must be fully dependent on candidate key.

I.e. if the non key attribute is partially dependent on candidate key (subset of column forming candidate key ) then split them into separate tables

Every table should have primary key and the relationship between table are formed using foreign key

Candidate key: set of columns which uniquely identifies record. A table can have multiple candidate keys because their can multiple set of columns that can unique identifies record.

Non key columns/ attribute: the columns in the table that are not the part of candidate key or primary key.

Partial dependency: if your candidate key is the combination of two columns ( or multiple columns ), then every non key column ( that are not part of candidate key ) should be fully dependent on all the column of candidate key. If there is any non key columns that depend only on one of candidate key columns then this results is partially dependent.

#### 3 NF

Must be in 2NF

Avoid transitive dependency

Transitive dependency: let's you have table T that has 3 columns A, B, C. If column A is functionally dependent on column B and column B is functionally depended on column C. Then we can say That column A is functionally depended on column C.

---

The notes above define each normal form but have no worked examples and stop at 3NF. The rest of this answer fills those gaps: one running example taken from unnormalized data to 3NF, then BCNF, and a short note on when to denormalize.

##### Worked example: from unnormalized data to 3NF

Start with an order sheet:

| order_id | order_date | customer_id | customer_name | customer_city | products |
| --- | --- | --- | --- | --- | --- |
| 1001 | 2026-01-05 | C1 | Asha | Pune | P1 (Keyboard, qty 2), P2 (Mouse, qty 1) |
| 1002 | 2026-01-06 | C2 | Ravi | Delhi | P1 (Keyboard, qty 1) |

**1NF:** `products` holds several values in one cell, so 1NF is broken. Give each product its own row. The rows are then unique on `(order_id, product_id)`, so that pair is the candidate key:

| order_id | product_id | order_date | customer_id | customer_name | customer_city | product_name | qty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1001 | P1 | 2026-01-05 | C1 | Asha | Pune | Keyboard | 2 |
| 1001 | P2 | 2026-01-05 | C1 | Asha | Pune | Mouse | 1 |
| 1002 | P1 | 2026-01-06 | C2 | Ravi | Delhi | Keyboard | 1 |

**2NF:** only `qty` depends on the whole key `(order_id, product_id)`. Some columns depend on just part of it:
- `product_name` depends on `product_id` alone.
- `order_date`, `customer_id`, `customer_name` and `customer_city` depend on `order_id` alone.

These are partial dependencies, so split the table:
- `order_items(order_id, product_id, qty)`, key `(order_id, product_id)`
- `products(product_id, product_name)`, key `product_id`
- `orders(order_id, order_date, customer_id, customer_name, customer_city)`, key `order_id`

**3NF:** `orders` still has a transitive dependency: `order_id → customer_id → customer_name, customer_city`. A non-key column (`customer_name`) depends on another non-key column (`customer_id`), not directly on the key. If Asha moves city, every one of her orders has to be updated (an update anomaly). Move the customer columns into their own table:
- `orders(order_id, order_date, customer_id)`, with `customer_id` as a foreign key
- `customers(customer_id, customer_name, customer_city)`

Every non-key column now depends on "the key, the whole key, and nothing but the key."

**Note on direction:** the standard way to write a transitive dependency is `key → X → Y`. Y depends on X, and X depends on the key, so Y depends on the key only through X. 3NF is broken when X is a non-key column. Here it is `order_id → customer_id → customer_city`.

##### BCNF (Boyce–Codd Normal Form)

**Rule:** for every non-trivial functional dependency `X → Y`, X must be a superkey. 3NF has an exception: it allows `X → Y` when Y is part of some candidate key (a "prime" attribute). BCNF removes that exception.

**Example that is in 3NF but not BCNF:** `course_tutor(student, course, tutor)`. The rules are:
- Each tutor teaches exactly one course, so `tutor → course`.
- A student has one tutor per course.

The candidate keys are `(student, course)` and `(student, tutor)`. `tutor → course` holds, but `tutor` is not a superkey. 3NF still allows it because `course` is a prime attribute. The cost is that a tutor's course is repeated for every student, so changing it means updating many rows.

**Fix:** decompose into `tutor_course(tutor, course)` and `student_tutor(student, tutor)`.

**Tradeoff:** a BCNF decomposition is always lossless, but it can lose dependency preservation. Here, "a student has one tutor per course" can no longer be enforced with a single key constraint; it needs a trigger or application logic. That is why many OLTP designs stop at 3NF.

##### Beyond BCNF, and when to denormalize

- **4NF** removes multi-valued dependencies. For example, an `employee(skill, language)` table where skills and languages are independent of each other should become two tables.
- **5NF** removes join dependencies. These are rare in practice.
- **Production angle:** normalize OLTP schemas, usually to 3NF or BCNF, for write integrity. Analytical models (star schemas, wide Gold tables) are deliberately denormalized to reduce joins. Fact tables stay roughly in 3NF, while dimensions are flattened. See [Question 5.21](#521-explain-star-schema-vs-snowflake-schema-with-proscons) for the star-schema trade-offs.

### 5.34 What is a Clustered Index?

**A clustered index determines the physical order of data in the table. Because the data can only be sorted one way, you can only have one clustered index per table (usually the primary key).**

### 5.35 What is the difference between UNION and UNION ALL?

**`UNION` Combines the results of two queries and removes duplicates, which requires an expensive sorting operation. `UNION ALL` combines results but keeps all rows, making it much faster.**

### 5.36 Assuming you have a two-column table of users and their friends coupled with a two-column table of users and their liked pages. Write a SQL query to make recommendations using pages that your friends have liked when it should not recommend pages that you have liked.

Assume the tables are:

- `friends(user_id, friend_id)`
- `likes(user_id, page_id)`

A SQL query would be:

```sql
SELECT DISTINCT
    f.user_id,
    l.page_id
FROM friends f
JOIN likes l
    ON f.friend_id = l.user_id
WHERE NOT EXISTS (
    SELECT 1
    FROM likes my_likes
    WHERE my_likes.user_id = f.user_id
      AND my_likes.page_id = l.page_id
);
```

#### How it works

- Join `friends` with `likes` to find **pages liked by your friends**.
- `NOT EXISTS` removes pages that **you have already liked**.
- `DISTINCT` prevents duplicate recommendations when multiple friends like the same page.

### 5.37 When can SQL indexing make a SELECT query slower?

- **Low selectivity** – If a query returns many rows, a sequential scan may be faster.
  ```sql
  SELECT * FROM users WHERE gender = 'M';
  ```
- **Very small tables** – Full table scans can be cheaper than index lookups.
- **Poorly designed indexes** – Indexes that don't match the query pattern provide little benefit.
- **Non-sargable predicates** – Functions on indexed columns may prevent efficient index usage.
  ```sql
  WHERE LOWER(name) = 'john';
  ```
- **Too many indexes** – Increase storage usage and can negatively affect overall database performance.

#### Senior-level approach

- Don't assume **"Index = faster."**
- Use:
  ```sql
  EXPLAIN ANALYZE
  SELECT ...;
  ```
- Check whether the optimizer chooses **Sequential Scan, Index Scan, or Index Only Scan**.

### 5.38 Compare Kimball, Inmon and Data Vault modeling approaches. When would you choose each?

All three answer the same question: how to turn many operational sources into a trustworthy analytical layer. They differ in where the integration happens and who the model is optimized for. Kimball optimizes for business users and query speed. Inmon optimizes for one normalized, integrated enterprise truth. Data Vault optimizes for auditability, parallel loading and absorbing change.

**Kimball (bottom-up, dimensional)**

- Build star schemas one business process at a time (orders, shipments, payments), tied together by conformed dimensions on a bus matrix.
- The warehouse *is* the collection of dimensional marts. There is no separate normalized layer.
- Strength: fast time to value, intuitive for BI tools and analysts, good query performance.
- Weakness: hard to retrofit when sources or business rules change a lot. Without strong governance of conformed dimensions, you get inconsistent marts.

**Inmon (top-down, Corporate Information Factory)**

- Build a normalized (roughly 3NF), subject-oriented enterprise data warehouse first. Then derive departmental dimensional marts from it.
- Strength: one integrated, non-redundant model of the enterprise. It is a strong base for many downstream consumers.
- Weakness: long upfront modeling effort, slower delivery, and many joins for analysts unless marts exist.

**Data Vault 2.0 (hybrid, raw-then-business)**

- **Hubs** hold unique business keys. **Links** hold relationships between hubs. **Satellites** hold descriptive attributes with full history. Keys are usually hashes of business keys, so tables can load in parallel without a central sequence.
- Loading is insert-only, so every value that ever arrived is kept with its load timestamp and record source. A raw vault stores data as received. A business vault adds derived rules. Dimensional marts (often views) sit on top for consumption.
- Strength: auditability, resilience to new sources, easy automation (dbt packages and code generators), parallel loading.
- Weakness: many tables and joins, a steep learning curve, and you still need a dimensional layer for users. On a small team it is overkill.

| Aspect | Kimball | Inmon | Data Vault 2.0 |
| --- | --- | --- | --- |
| Core model | Star schemas | 3NF EDW plus marts | Hubs, links, satellites plus marts |
| Delivery | Incremental, fast | Slow upfront | Medium; automation helps |
| History | SCD types in dimensions | In EDW design | Native, insert-only |
| Handles source churn | Poorly | Moderately | Very well |
| End-user friendliness | High | Low (EDW), high (marts) | Low (vault), high (marts) |

**How to decide**

- **Kimball:** most analytics teams with a handful of sources, a strong BI focus and a need to ship quickly. This is the default for a modern dbt and cloud warehouse stack.
- **Inmon-style normalized core:** a large enterprise where many systems (not only BI) consume an integrated master model, and where the business has the patience for upfront design.
- **Data Vault:** regulated industries (banking, insurance, healthcare) needing full audit trails, dozens of sources that change often, or mergers bringing new systems regularly.

In practice, many mature platforms are hybrids. The lakehouse Bronze layer holds raw data, a Data Vault or normalized Silver layer handles integration, and Kimball stars in Gold serve consumption. The senior answer is to pick the lightest approach that meets the audit and change requirements, not the most elaborate one.

### 5.39 What is the grain of a fact table, and what are transaction, periodic snapshot and accumulating snapshot fact tables?

The **grain** is a precise business statement of what one row in a fact table represents, for example "one row per order line per order" or "one row per account per day". Declaring the grain comes before choosing dimensions or measures. Every dimension and fact must be true at that grain, and grains must never be mixed in one table.

**Why grain matters**

- It decides which dimensions are possible. A daily account snapshot cannot carry a `transaction_type` dimension.
- It decides which measures are additive. Mixing order-header totals with line amounts double-counts.
- Choose the **lowest atomic grain** the source supports. You can always aggregate up, but you can never drill below what you stored.

**The three fact table types**

| Type | Grain | Row lifecycle | Typical measures | Example |
| --- | --- | --- | --- | --- |
| Transaction | One row per event | Insert only | Fully additive (amount, quantity) | Each POS line item, each click |
| Periodic snapshot | One row per entity per period | Insert one row per period | Semi-additive balances and levels | Daily account balance, monthly inventory |
| Accumulating snapshot | One row per process instance | Inserted, then updated as milestones occur | Milestone dates, lag durations, counts | Order fulfillment: ordered, picked, shipped, delivered |

**Transaction fact tables**

- The most common type and the most granular. Rows appear only when something happens, so sparse activity means sparse rows.
- Answers "what happened" questions, but "what was the state on date X" needs a scan of all history.

**Periodic snapshot fact tables**

- A row exists for every entity in every period, *even when nothing happened*. This makes the table dense and predictable.
- Balances are **semi-additive**: you can sum across accounts but not across time. Across time you take the period-end value or an average.
- Example: `fact_account_daily(account_key, date_key, balance, txn_count)`.

**Accumulating snapshot fact tables**

- Models a pipeline with a defined start and end. It has several date foreign keys (role-playing date dimensions such as `order_date_key`, `ship_date_key`, `delivery_date_key`).
- Rows are **updated** as each milestone is reached, and lag facts like `days_to_ship` are filled in.
- Great for funnel and SLA analysis ("average days from approval to funding"). It is awkward when processes loop back or have no fixed milestones.

**Production considerations**

- Accumulating snapshots need `MERGE`/update logic. On lakehouse formats this means file rewrites, so partition carefully or rebuild from the transaction fact.
- Many teams store the transaction fact as the system of record and **derive** snapshots from it. This keeps one source of truth and makes reprocessing simple.
- A fourth pattern worth mentioning is the **factless fact table** (see 5.26), which records events or coverage with no numeric measure.

### 5.40 What are conformed dimensions and the bus matrix, and why do they matter?

A **conformed dimension** is one dimension (same keys, same attribute names, same values and definitions) shared by several fact tables, or a strict subset of one. The **bus matrix** is the planning grid that shows which business processes (rows) use which conformed dimensions (columns). Together they are what turns separate star schemas into one integrated warehouse instead of a set of data silos.

**Conformed dimensions**

- `dim_customer`, `dim_product` and `dim_date` are defined once and reused by `fact_orders`, `fact_returns` and `fact_support_tickets`.
- **Shrunken conformed dimensions** are allowed. A `dim_month` derived from `dim_date`, or a `dim_brand` rolled up from `dim_product`, conforms if its attributes and values match the base dimension exactly.
- **Conformed facts** are the companion idea: a measure such as `net_revenue` must have the same definition everywhere it appears, or it must be renamed.

**Why it matters: drill-across**

Because both facts share the same dimension values, you can query each fact separately at a common grain and join the results. This is called drilling across.

```sql
WITH sales AS (
  SELECT p.category, SUM(f.net_amount) AS revenue
  FROM fact_orders f JOIN dim_product p ON f.product_key = p.product_key
  GROUP BY p.category
),
returns AS (
  SELECT p.category, SUM(r.refund_amount) AS refunds
  FROM fact_returns r JOIN dim_product p ON r.product_key = p.product_key
  GROUP BY p.category
)
SELECT s.category, s.revenue, r.refunds, r.refunds / NULLIF(s.revenue, 0) AS return_rate
FROM sales s LEFT JOIN returns r USING (category);
```

Never join two fact tables directly row to row. Their grains differ and the join fans out. Aggregate each one to the shared dimension grain first.

**The bus matrix**

| Business process | Date | Customer | Product | Store | Promotion | Employee |
| --- | --- | --- | --- | --- | --- | --- |
| Retail sales | X | X | X | X | X | X |
| Returns | X | X | X | X |  | X |
| Inventory snapshot | X |  | X | X |  |  |
| Support tickets | X | X | X |  |  | X |

- It is the enterprise architecture blueprint *and* the delivery roadmap. Each row is a deliverable mart. The columns show which dimensions must be built once and governed.
- Columns used by many rows (Date, Customer, Product) are the highest-value conformance investments.

**What goes wrong without them**

- Marketing's "active customer" and Finance's "active customer" disagree, and executives stop trusting dashboards.
- Each team builds its own `dim_customer` with different keys, so cross-process analysis needs fragile mapping tables.
- In a data mesh, conformed dimensions become shared domain data products with clear ownership. The governance problem does not disappear just because ownership is distributed.

**Senior angle:** conformance is mainly an organizational problem. It needs a data steward per dimension, agreed definitions in a semantic layer or metrics catalog, and tests that fail when values drift.

### 5.41 How do you implement an SCD Type 2 dimension with a MERGE statement?

The standard trick is to `MERGE` a staged change set that contains **each changed key twice**: once to match and expire the current row, and once with a NULL merge key so that it falls into `WHEN NOT MATCHED` and inserts the new version. Brand-new keys appear once and are inserted. Unchanged keys are filtered out before the merge so they cost nothing.

**Target table**

```sql
CREATE TABLE dim_customer (
  customer_sk   BIGINT GENERATED ALWAYS AS IDENTITY,
  customer_id   VARCHAR NOT NULL,      -- natural key
  name          VARCHAR,
  city          VARCHAR,
  row_hash      VARCHAR,               -- hash of tracked attributes
  valid_from    TIMESTAMP NOT NULL,
  valid_to      TIMESTAMP NOT NULL DEFAULT '9999-12-31',
  is_current    BOOLEAN NOT NULL DEFAULT TRUE
);
```

**The MERGE (ANSI-style; works with minor changes in Snowflake, Databricks, BigQuery, SQL Server, PostgreSQL 15+)**

```sql
MERGE INTO dim_customer t
USING (
  -- 1) rows that will expire existing versions or insert brand-new keys
  SELECT s.customer_id AS merge_key, s.*
  FROM stg_customer s
  UNION ALL
  -- 2) a second copy of CHANGED rows with a NULL key, forcing an insert
  SELECT NULL AS merge_key, s.*
  FROM stg_customer s
  JOIN dim_customer d
    ON d.customer_id = s.customer_id AND d.is_current
  WHERE d.row_hash <> s.row_hash
) src
ON t.customer_id = src.merge_key AND t.is_current
WHEN MATCHED AND t.row_hash <> src.row_hash THEN
  UPDATE SET valid_to = src.change_ts, is_current = FALSE
WHEN NOT MATCHED THEN
  INSERT (customer_id, name, city, row_hash, valid_from, valid_to, is_current)
  VALUES (src.customer_id, src.name, src.city, src.row_hash,
          src.change_ts, '9999-12-31', TRUE);
```

**How each case is handled**

- **New key:** no current row matches, so it is inserted as current.
- **Changed key:** copy 1 matches and the current row is closed with `valid_to = change_ts`. Copy 2 has a NULL key, cannot match, and is inserted as the new current row.
- **Unchanged key:** copy 1 matches but the `row_hash` condition is false, so nothing happens. Copy 2 is never generated.

**Design choices that matter**

- **Hash the tracked columns** (for example `md5(concat_ws('|', coalesce(name,''), coalesce(city,'')))`). Coalesce NULLs first, or a change from NULL to a value is missed.
- Use a **high-date sentinel** (`9999-12-31`) instead of NULL for `valid_to`. Point-in-time joins then become a simple `BETWEEN` without NULL handling.
- Use **half-open intervals**: `valid_from <= ts AND ts < valid_to`. This avoids double matches at boundaries.
- **Deduplicate the staging set** to one row per key per load (see 5.44). Otherwise MERGE fails with "multiple source rows matched", or two versions become current.
- Mix SCD types per column: Type 1 columns (such as a corrected spelling) are updated in place across all versions, while Type 2 columns create versions.

**Late-arriving changes**

A change with an effective timestamp *older* than the current version breaks the simple pattern. Options:

- **Reject and reprocess:** detect `change_ts < current.valid_from` and route the row to a correction job.
- **Splice:** find the version whose interval contains the late timestamp, close it at that time, insert the late version with `valid_to` set to the next version's `valid_from`, and re-key any facts that fall in the affected window.
- **Rebuild:** in a lakehouse, it is often simpler and safer to recompute the dimension for the affected keys from the full change log with window functions, using `LEAD(change_ts)` to set `valid_to`. This is idempotent by construction.

dbt snapshots implement this pattern for you (`dbt_valid_from`, `dbt_valid_to`), and Databricks Lakeflow offers `AUTO CDC ... STORED AS SCD TYPE 2` with ordering by a sequence column. Interviewers still expect you to be able to write the MERGE by hand.

### 5.42 How do you solve a "gaps and islands" problem in SQL?

A gaps-and-islands problem asks you to find runs of consecutive values (islands) and breaks between them (gaps): consecutive login days, uninterrupted subscription periods, sensor outages. The classic solution subtracts a `ROW_NUMBER()` from the ordered value. Within a consecutive run, both increase by one each step, so the difference stays constant and becomes a group identifier.

**Example: longest streak of consecutive login days per user**

```sql
WITH days AS (               -- 1) one row per user per day (dedupe multiple logins)
  SELECT DISTINCT user_id, CAST(login_ts AS DATE) AS login_date
  FROM logins
),
grouped AS (                 -- 2) date minus row number is constant inside a streak
  SELECT user_id, login_date,
         login_date - CAST(ROW_NUMBER() OVER (
             PARTITION BY user_id ORDER BY login_date) AS INT) AS grp
  FROM days
),
islands AS (                 -- 3) collapse each island
  SELECT user_id, grp,
         MIN(login_date) AS streak_start,
         MAX(login_date) AS streak_end,
         COUNT(*)        AS streak_days
  FROM grouped
  GROUP BY user_id, grp
)
SELECT user_id, MAX(streak_days) AS longest_streak
FROM islands
GROUP BY user_id;
```

How it works for one user:

| login_date | row_number | login_date minus row_number |
| --- | --- | --- |
| Jan 1 | 1 | Dec 31 |
| Jan 2 | 2 | Dec 31 |
| Jan 3 | 3 | Dec 31 |
| Jan 7 | 4 | Jan 3 |
| Jan 8 | 5 | Jan 3 |

Two islands: Jan 1 to 3 (three days) and Jan 7 to 8 (two days).

**Alternative: LAG plus running sum (more flexible)**

```sql
SELECT user_id, login_date,
       SUM(CASE WHEN login_date = prev_date + 1 THEN 0 ELSE 1 END)
         OVER (PARTITION BY user_id ORDER BY login_date) AS island_id
FROM (
  SELECT user_id, login_date,
         LAG(login_date) OVER (PARTITION BY user_id ORDER BY login_date) AS prev_date
  FROM days
) t;
```

Each row that does not continue the previous one starts a new island, and the running sum numbers the islands. Use this form when the "consecutive" rule is more complex than plus one: a tolerance of up to two days, business days only, or timestamps with a threshold (which is exactly how sessionization works, see 5.43).

**Finding the gaps instead**

```sql
SELECT user_id, prev_date + 1 AS gap_start, login_date - 1 AS gap_end
FROM (SELECT user_id, login_date,
             LAG(login_date) OVER (PARTITION BY user_id ORDER BY login_date) AS prev_date
      FROM days) t
WHERE login_date > prev_date + 1;
```

**Pitfalls**

- **Duplicates break the ROW_NUMBER trick.** Two logins on the same day give the same date two different row numbers, which splits the island. Deduplicate first, or use `DENSE_RANK`.
- **Date arithmetic is dialect-specific**: `DATEADD(day, -rn, d)` in Snowflake and SQL Server, `DATE_SUB(d, INTERVAL rn DAY)` in BigQuery, plain `date - int` in PostgreSQL.
- **Time zones:** "a day" must be defined in the user's local zone, or streaks break at midnight UTC.
- **Performance:** both methods need one sort per partition. On billions of rows, partition and cluster by `user_id` and date, or compute streaks incrementally (carry the open streak forward each day) instead of rescanning history.

### 5.43 How would you sessionize clickstream events in SQL with a 30-minute inactivity timeout?

Sessionization is a gaps-and-islands problem on timestamps. Order each user's events by time, flag an event as a session start when the gap since the previous event is more than 30 minutes (or there is no previous event), and take a running sum of those flags to get a session number.

```sql
WITH ordered AS (
  SELECT user_id, event_ts, event_name, page_url,
         LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts, event_id) AS prev_ts
  FROM events
),
flagged AS (
  SELECT *,
         CASE WHEN prev_ts IS NULL
                OR event_ts > prev_ts + INTERVAL '30 minutes'
              THEN 1 ELSE 0 END AS is_new_session
  FROM ordered
),
numbered AS (
  SELECT *,
         SUM(is_new_session) OVER (
           PARTITION BY user_id ORDER BY event_ts, event_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_seq
  FROM flagged
)
SELECT user_id,
       user_id || '-' || CAST(session_seq AS VARCHAR) AS session_id,
       MIN(event_ts)                                  AS session_start,
       MAX(event_ts)                                  AS session_end,
       COUNT(*)                                       AS events,
       MIN_BY(page_url, event_ts)                     AS landing_page   -- Snowflake/Databricks; use ARRAY_AGG ... LIMIT 1 in BigQuery
FROM numbered
GROUP BY user_id, session_seq;
```

**Details that interviewers probe**

- **Tie-breaker in the ORDER BY.** Events with identical timestamps need a deterministic secondary key (`event_id`), or session numbers can change between runs.
- **Explicit frame.** `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` avoids the default `RANGE` frame, which treats ties as peers and gives every tied row the same running total.
- **Stable session IDs.** `user_id` plus a sequence number shifts if earlier events arrive late. A hash of `user_id` and `session_start` is more stable, but still changes if a late event lands before the start.
- **Extra session boundaries.** Many products also start a new session at midnight, when the marketing source (UTM campaign) changes, or after a maximum session length. Add these as further `OR` conditions in the flag.
- **Identity stitching.** Anonymous `device_id` events often need mapping to `user_id` after login *before* sessionizing. Otherwise one visit splits into two sessions.

**Incremental processing in production**

Re-sessionizing all history every day is expensive. The usual pattern:

1. Process only new events plus a **lookback window** (for example, the last two hours of already-processed events per active user). A session that was open at the last run can then continue or merge.
2. `MERGE` the recomputed sessions for affected users, replacing the sessions that overlap the window.
3. Accept that very late events (beyond the lookback) either open a separate session or trigger a targeted backfill. Decide which, and document it.

**Streaming alternative:** Spark Structured Streaming (`session_window`), Flink and Beam provide native session windows with a gap duration and watermarks. Use them when sessions must be available within minutes. Use the SQL version for batch analytics and as a reference implementation to reconcile against.

**Performance:** the query needs one shuffle and sort by `user_id` and time. Partition the events table by date and cluster by `user_id`, and filter to the date range plus lookback, so the warehouse prunes most of the data.

### 5.44 How do you deduplicate rows and keep only the latest record per key?

Rank the rows within each key by recency with `ROW_NUMBER()` and keep rank 1. In warehouses that support `QUALIFY` (Snowflake, BigQuery, Databricks, DuckDB, Teradata), this is a single clean statement. Elsewhere, wrap the window function in a subquery.

```sql
-- Snowflake / BigQuery / Databricks
SELECT *
FROM raw_customers
QUALIFY ROW_NUMBER() OVER (
          PARTITION BY customer_id
          ORDER BY updated_at DESC, _ingested_at DESC, _source_offset DESC
        ) = 1;

-- Portable (PostgreSQL, SQL Server, etc.)
SELECT *
FROM (
  SELECT r.*,
         ROW_NUMBER() OVER (PARTITION BY customer_id
                            ORDER BY updated_at DESC, _ingested_at DESC) AS rn
  FROM raw_customers r
) t
WHERE rn = 1;
```

**Tie-breakers are the senior detail**

- If two rows share the same `updated_at`, `ROW_NUMBER` picks one arbitrarily, and the choice can differ between runs. That makes the pipeline non-deterministic and hard to debug.
- Always add tie-breakers until the ordering is unique: ingestion time, CDC log sequence number (LSN), Kafka partition and offset, file name and row position.
- With CDC feeds, order by the **source commit sequence**, not the wall-clock timestamp. Clocks on different source nodes can disagree.
- Handle **deletes** explicitly: if the latest record is a delete or tombstone (`op = 'D'`), filter it out *after* ranking. If you filter first, an older version comes back to life.

**Choosing the right function**

| Need | Use |
| --- | --- |
| Exactly one row per key | `ROW_NUMBER()` |
| All rows tied for latest | `RANK()` |
| Latest value of one column only | `MAX_BY(col, updated_at)` / `ARG_MAX` / `ARRAY_AGG(... ORDER BY ... LIMIT 1)[OFFSET(0)]` |
| Exact duplicate rows (all columns equal) | `SELECT DISTINCT` or `GROUP BY` all columns |

**PostgreSQL shortcut:** `SELECT DISTINCT ON (customer_id) * FROM raw_customers ORDER BY customer_id, updated_at DESC;`. It is concise and fast with a matching index, but it is PostgreSQL-only.

**Deleting duplicates in place**

```sql
-- PostgreSQL: delete all but the newest row per key
DELETE FROM raw_customers r
USING (
  SELECT ctid, ROW_NUMBER() OVER (PARTITION BY customer_id
                                  ORDER BY updated_at DESC) AS rn
  FROM raw_customers
) d
WHERE r.ctid = d.ctid AND d.rn > 1;
```

In cloud warehouses, in-place deletes are expensive file rewrites. It is usually better to rebuild the table with `CREATE OR REPLACE TABLE ... AS SELECT ... QUALIFY`, or to deduplicate inside the incremental `MERGE` source.

**Production practices**

- Deduplicate as close to ingestion as possible (Bronze to Silver), and do it **before** any `MERGE`. Duplicate source keys make MERGE fail or produce nondeterministic updates.
- For incremental loads, deduplicate only the new batch and then merge against the target, instead of re-ranking the entire history every run.
- Add a uniqueness test (dbt `unique`, or a `COUNT(*) > 1` check) on the key in the output model, so a regression is caught rather than silently doubling revenue.
- Distinguish *technical* duplicates (replays, at-least-once delivery) from *business* duplicates (the same customer under two IDs). The second is an entity-resolution problem, not a `ROW_NUMBER` problem.

### 5.45 What are transaction isolation levels and the anomalies each prevents? How does MVCC work?

Isolation levels define how much one transaction can observe the effects of other concurrent transactions. Stronger levels prevent more anomalies but cost more in blocking, aborts and retries. MVCC (multi-version concurrency control) is how most modern databases provide isolation without readers and writers blocking each other: each transaction reads a consistent snapshot of row versions.

**The anomalies**

- **Dirty read:** reading another transaction's uncommitted write.
- **Non-repeatable read:** re-reading a row returns a different value because another transaction committed an update in between.
- **Phantom read:** re-running a range query returns new or missing rows because another transaction inserted or deleted matching rows.
- **Lost update:** two transactions read, modify and write the same row, and one overwrite silently discards the other.
- **Write skew:** two transactions read overlapping data, write *different* rows, and together violate an invariant. Classic example: two doctors each check that someone else is on call, then both go off call.

**ANSI levels versus what PostgreSQL actually does**

| Level | Dirty read | Non-repeatable | Phantom | Write skew | PostgreSQL behavior |
| --- | --- | --- | --- | --- | --- |
| Read Uncommitted | Possible (ANSI) | Possible | Possible | Possible | Treated as Read Committed |
| Read Committed | Prevented | Possible | Possible | Possible | **Default.** New snapshot per statement |
| Repeatable Read | Prevented | Prevented | Possible (ANSI) | Possible | Snapshot isolation. Also prevents phantoms; raises serialization errors on concurrent updates of the same row |
| Serializable | Prevented | Prevented | Prevented | Prevented | Serializable Snapshot Isolation (SSI). Detects dangerous read/write dependencies and aborts one transaction |

Other engines differ. MySQL InnoDB defaults to Repeatable Read and uses gap locks. SQL Server defaults to locking Read Committed unless snapshot isolation is enabled. Oracle's "Serializable" is snapshot isolation, so it allows write skew. Snowflake supports only Read Committed for table data. Always check the specific engine rather than trusting the ANSI table.

**How MVCC works (PostgreSQL model)**

- Every row version (tuple) carries `xmin` (the creating transaction ID) and `xmax` (the deleting or updating transaction ID).
- An `UPDATE` does not overwrite. It marks the old version's `xmax` and writes a new version.
- A transaction takes a **snapshot**: the set of transaction IDs that were committed when it started. A version is visible if its `xmin` is committed in the snapshot and its `xmax` is not.
- Readers never block writers and writers never block readers. Writers still block other writers on the same row.
- **Cost:** dead versions pile up and must be removed by `VACUUM`. Long-running transactions hold back the cleanup horizon, which causes table and index bloat. Transaction-ID wraparound must be prevented by freezing. Oracle and MySQL keep old versions in undo logs instead of in the table.

**Practical guidance**

- Read Committed is fine for most OLTP work, but **prevent lost updates explicitly**. Use atomic updates (`SET qty = qty - 1`), `SELECT ... FOR UPDATE`, or optimistic locking with a version column.
- Use Serializable for invariants that span rows (inventory, bookings, balances), and **always wrap the transaction in retry logic**. Serialization failures (SQLSTATE `40001`) are expected behavior, not bugs.
- Keep transactions short. Long transactions under MVCC cause bloat. Under Serializable they raise abort rates.
- In lakehouse table formats (Delta Lake, Iceberg), commits use optimistic concurrency on table snapshots. Concurrent writers to the same files conflict and must retry, which is the same idea at file granularity.

### 5.46 How do you read a query execution plan, and how do nested loop, hash and merge joins differ?

An execution plan is the tree of physical operators the optimizer chose for your SQL. Read it from the innermost or leaf nodes (scans) upward to the root, and compare the optimizer's **estimated** row counts with the **actual** ones. Large misestimates are the root cause of most bad plans, because the join algorithm, join order and memory grants all depend on those estimates.

**Reading a PostgreSQL plan**

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT c.name, SUM(o.amount)
FROM orders o JOIN customers c ON c.id = o.customer_id
WHERE o.order_date >= DATE '2026-01-01'
GROUP BY c.name;
```

```text
HashAggregate  (cost=..rows=500) (actual time=..rows=480 loops=1)
  ->  Hash Join  (rows=20000) (actual rows=1950000)       <- 100x misestimate
        Hash Cond: (o.customer_id = c.id)
        ->  Index Scan using orders_date_idx on orders o
              Index Cond: (order_date >= '2026-01-01')
              Buffers: shared hit=120 read=48000
        ->  Hash  ->  Seq Scan on customers c
```

**What to look for**

- **Estimated versus actual rows** at each node. A 10x or worse gap suggests stale statistics (`ANALYZE`), correlated predicates (extended statistics), or functions hiding a column from the optimizer.
- **Where the time goes.** Actual time is cumulative per node, and you must multiply by `loops` for inner sides of nested loops.
- **Buffers read versus hit**, and spills: `Sort Method: external merge Disk` or hash batches greater than 1 mean `work_mem` was exceeded.
- **Scan type:** Seq Scan, Index Scan, Index Only Scan (check heap fetches) or Bitmap Heap Scan, and whether filters are applied as index conditions or as post-scan filters.
- In cloud warehouses, use the query profile instead: partitions scanned versus total, bytes spilled to remote storage, exploding joins (output rows much larger than input), and skewed workers.

**The three join algorithms**

| Algorithm | How it works | Best when | Cost profile | Weakness |
| --- | --- | --- | --- | --- |
| Nested loop | For each outer row, look up matching inner rows (ideally via an index) | Small outer input; indexed inner side; `LIMIT` queries; non-equi joins | About outer rows times the cost of an inner lookup | Disastrous when the outer input is underestimated and is actually large |
| Hash join | Build a hash table on the smaller input, then probe it with each row of the larger | Large, unsorted inputs; equality joins | About linear in both inputs; needs memory for the build side | Equality joins only; spills to disk if the build side exceeds memory |
| Merge join | Sort both inputs on the join key (or read them pre-sorted), then walk them together | Inputs already sorted (indexes, clustered keys); very large inputs; also needs sorted output | Linear after sorting; sorting costs n log n | Expensive sorts if inputs are not already ordered |

**Distributed engines add a dimension:** Spark, Snowflake and BigQuery choose between a **broadcast** join (copy the small table to every worker) and a **shuffle/partitioned** hash join (redistribute both sides by key). Skewed keys cause a few tasks to run far longer than the rest. Fixes include salting, Spark AQE skew-join handling, or filtering out hot keys such as NULL.

**Typical fixes, in order**

1. Refresh statistics and fix the misestimate (rewrite non-sargable predicates, add extended statistics).
2. Add or adjust an index so the right access path exists.
3. Reduce data earlier (filter or aggregate before the join, prune partitions).
4. Only as a last resort, use hints or planner settings, and document why. These freeze a plan that may become wrong as data grows.

### 5.47 What are the common NULL pitfalls in SQL?

NULL means "unknown", not "empty" or zero. SQL uses three-valued logic (TRUE, FALSE, UNKNOWN), and `WHERE`, `JOIN ON` and `HAVING` keep only rows that evaluate to TRUE. Most NULL bugs come from forgetting that any comparison with NULL, including `NULL = NULL`, yields UNKNOWN.

**1. `NOT IN` with a NULL in the subquery returns nothing**

```sql
-- If ANY blocked_users.user_id is NULL, this returns zero rows
SELECT * FROM users
WHERE id NOT IN (SELECT user_id FROM blocked_users);
```

`x NOT IN (1, NULL)` expands to `x <> 1 AND x <> NULL`. The second term is UNKNOWN, so the whole predicate is never TRUE. Use `NOT EXISTS` instead, which behaves correctly and is usually planned as an efficient anti-join:

```sql
SELECT * FROM users u
WHERE NOT EXISTS (SELECT 1 FROM blocked_users b WHERE b.user_id = u.id);
```

**2. `COUNT(*)` versus `COUNT(col)` versus `COUNT(DISTINCT col)`**

- `COUNT(*)` counts rows. `COUNT(col)` counts non-NULL values. `COUNT(DISTINCT col)` ignores NULLs too.
- A "percentage of users with email" computed as `COUNT(email) / COUNT(*)` is correct. Swapping them silently changes the meaning.

**3. Aggregates ignore NULLs**

- `AVG(score)` averages only non-NULL rows. If NULL actually means zero, the average is inflated. Use `AVG(COALESCE(score, 0))` only when that is the true business meaning.
- `SUM` over zero rows (or all NULLs) returns NULL, not 0. Wrap it in `COALESCE(SUM(x), 0)` for reports.

**4. NULLs in joins**

- `a.key = b.key` never matches NULL keys, so rows with NULL keys vanish from inner joins and fail to match in outer joins.
- To treat NULLs as equal, use `IS NOT DISTINCT FROM` (PostgreSQL, Snowflake, Databricks, BigQuery), `<=>` (MySQL and Spark), or `EQUAL_NULL` (Snowflake).
- In distributed joins, a large number of NULL keys all hash to the same partition and cause **skew**. Filter them out, or salt them, before the join.

**5. Filters on the outer table of a LEFT JOIN**

```sql
-- Turns the LEFT JOIN into an INNER JOIN: unmatched rows have o.status = NULL
SELECT ... FROM customers c LEFT JOIN orders o ON o.customer_id = c.id
WHERE o.status = 'paid';
-- Correct: put the predicate in the ON clause
... LEFT JOIN orders o ON o.customer_id = c.id AND o.status = 'paid';
```

**6. Negations and inequality filters drop NULLs**

`WHERE country <> 'US'` excludes rows where `country` is NULL. Analysts expect them to be included. Write `WHERE country IS DISTINCT FROM 'US'` or `WHERE country <> 'US' OR country IS NULL`.

**7. Other traps**

- `CASE WHEN col = NULL` is never true. Use `IS NULL`.
- String concatenation with `||` returns NULL if any part is NULL (in PostgreSQL and the standard). Use `CONCAT` or `CONCAT_WS`, or `COALESCE` each part.
- `GROUP BY`, `DISTINCT` and `UNION` treat all NULLs as one group, which is inconsistent with `=`.
- Sorting: NULLs sort last ascending in PostgreSQL and Oracle, but first in SQL Server and MySQL. Be explicit with `NULLS FIRST` or `NULLS LAST`.
- `UNIQUE` constraints usually allow many NULLs (PostgreSQL 15+ offers `NULLS NOT DISTINCT`).
- Hash-based change detection: `md5(a || b)` becomes NULL when any column is NULL, so changes are missed. Coalesce first.

**Senior practice:** decide in the model what NULL means for each column (unknown, not applicable, or not yet happened). Use explicit "Unknown" members (for example key `-1`) in dimensions so fact foreign keys are never NULL, and add `not_null` tests on keys.

### 5.48 How do partitioning and clustering work in cloud data warehouses like BigQuery and Snowflake, and how do they enable pruning?

Both warehouses store tables as large immutable columnar blocks and keep metadata about each block, including min and max values per column. **Pruning** means using that metadata and the query's filters to skip blocks that cannot contain matching rows. Partitioning and clustering exist to make pruning effective by physically co-locating rows with similar values. Pruning cuts bytes scanned, which is both your latency and, on BigQuery on-demand pricing, your bill.

**BigQuery**

- **Partitioning** splits a table into segments by a `DATE`, `TIMESTAMP` or `DATETIME` column (hourly, daily, monthly or yearly), by ingestion time, or by an integer range. There is a per-table limit on the number of partitions, so very high-cardinality partition keys are not possible.
- **Clustering** sorts data within each partition (or within an unpartitioned table) by up to **four columns**. Order matters: filters on the first clustering column prune best. BigQuery re-clusters automatically in the background at no charge.
- Partition pruning is visible *before* the query runs (the dry-run byte estimate is accurate). Clustering pruning happens at run time, so the estimate shows an upper bound and the actual bytes billed are lower.
- `require_partition_filter = TRUE` rejects queries that do not filter on the partition column. It is a cheap guardrail against accidental full scans.

```sql
CREATE TABLE analytics.events (
  event_ts TIMESTAMP, user_id STRING, event_name STRING, country STRING
)
PARTITION BY DATE(event_ts)
CLUSTER BY user_id, event_name
OPTIONS (require_partition_filter = TRUE, partition_expiration_days = 400);
```

**Snowflake**

- There are no user-defined partitions. Every table is automatically divided into **micro-partitions** of roughly 50 to 500 MB uncompressed, stored in columnar format, with per-column min and max values, distinct counts and other metadata.
- Pruning works on any column whose values are naturally ordered by load order. Event tables loaded daily prune well on date with no extra work.
- When load order does not match filter patterns, define a **clustering key** (`ALTER TABLE events CLUSTER BY (event_date, account_id)`). **Automatic Clustering** then reorganizes micro-partitions in the background, billed as serverless credits.
- Measure with `SYSTEM$CLUSTERING_INFORMATION` (average clustering depth and overlap) and, more importantly, with the query profile's "partitions scanned versus partitions total".

| Aspect | BigQuery | Snowflake |
| --- | --- | --- |
| Coarse unit | Partition (user-defined column) | None; micro-partitions are automatic |
| Fine organization | Clustering, up to 4 columns, free automatic re-clustering | Optional clustering key, paid Automatic Clustering |
| Pruning metadata | Partition boundaries plus block-level min and max | Micro-partition min and max per column |
| Visible in | Bytes processed, execution details | Partitions scanned versus total in the query profile |

**What defeats pruning**

- Wrapping the filter column in a function or cast, such as `WHERE CAST(account_id AS STRING) = '42'` or `WHERE UPPER(country) = 'US'`. The engine can no longer compare the literal with the stored min and max values. Write the predicate directly against the raw column instead.
- Filtering via a join or a subquery value that is not known at compile time. Some engines can prune dynamically at run time, but not always.
- Clustering on a very high-cardinality column like a UUID (no value locality), or on a very low-cardinality column like a boolean (everything overlaps).
- Lots of small, frequent writes that fragment clustering faster than background maintenance can repair.

**How to choose keys:** partition (BigQuery) or lead the clustering key (Snowflake) with the column almost every query filters on, typically event date. Then add one to three columns used in selective filters or joins, ordered from lower to higher cardinality. Do not bother clustering small tables. Snowflake recommends clustering keys mainly for multi-terabyte tables, and on small BigQuery tables the savings are negligible. In Snowflake especially, the maintenance credits can outweigh the scan savings. Validate the choice with real query history, not intuition.

The same concepts apply to lakehouse formats: Iceberg hidden partitioning and sort orders, and Delta Lake liquid clustering, both prune files using column statistics in table metadata.

### 5.49 How would you write a cohort retention query?

A cohort retention query groups users by when they first did something (usually signup or first purchase month), then measures what share of each cohort is still active in each following period. The output is the familiar triangle: cohorts as rows, "months since start" as columns, and retention percentages in the cells.

**Monthly retention by signup cohort**

```sql
WITH cohorts AS (                         -- 1) one row per user: their cohort
  SELECT user_id, DATE_TRUNC('month', MIN(event_ts)) AS cohort_month
  FROM events
  WHERE event_name = 'signup'
  GROUP BY user_id
),
activity AS (                             -- 2) distinct active months per user
  SELECT DISTINCT user_id, DATE_TRUNC('month', event_ts) AS activity_month
  FROM events
  WHERE event_name IN ('session_start', 'purchase')   -- define "active" explicitly
),
cohort_activity AS (                      -- 3) month offset since cohort start
  SELECT c.cohort_month,
         DATEDIFF('month', c.cohort_month, a.activity_month) AS month_number,
         a.user_id
  FROM cohorts c
  JOIN activity a ON a.user_id = c.user_id
               AND a.activity_month >= c.cohort_month
),
cohort_size AS (
  SELECT cohort_month, COUNT(*) AS users FROM cohorts GROUP BY cohort_month
)
SELECT ca.cohort_month,
       cs.users                                          AS cohort_users,
       ca.month_number,
       COUNT(DISTINCT ca.user_id)                        AS active_users,
       ROUND(100.0 * COUNT(DISTINCT ca.user_id) / cs.users, 1) AS retention_pct
FROM cohort_activity ca
JOIN cohort_size cs USING (cohort_month)
GROUP BY ca.cohort_month, cs.users, ca.month_number
ORDER BY ca.cohort_month, ca.month_number;
```

`DATEDIFF('month', a, b)` is Snowflake and Databricks syntax. In BigQuery use `DATE_DIFF(b, a, MONTH)`. In PostgreSQL use `(EXTRACT(YEAR FROM age(b, a)) * 12 + EXTRACT(MONTH FROM age(b, a)))`, or compute month indexes as `year * 12 + month` and subtract.

Pivot the result in the BI tool, or with conditional aggregation (`MAX(CASE WHEN month_number = 1 THEN retention_pct END) AS m1`, and so on), to get the triangle.

**Definitions to agree on before writing SQL**

- **Cohort event:** signup, first purchase, or first use of a feature. Different events answer different questions.
- **"Active":** any event, or a meaningful one (purchase, core action)? Counting page views inflates retention.
- **Classic (bounded) versus rolling (unbounded) retention.** Classic counts users active *in* month N. Rolling counts users active in month N *or later*. Rolling is always at least as high, so label it clearly.
- **Period:** calendar months are easy to explain but uneven (a user who signs up on the 31st gets one day of "month 0"). Day-number offsets (day 1, 7, 30) from each user's own start date are more precise for product analytics.

**Pitfalls**

- **Incomplete recent cohorts.** The latest cohort has not had time to reach month 6. Show blanks, not zeros, and do not average incomplete cells into a "retention curve".
- **Missing month 0 or gaps.** Cohorts with zero activity in a month disappear from an inner join. Generate the full cohort-by-offset grid with a date spine and `LEFT JOIN` if you need explicit zeros.
- **Time zones and late events** shift users between months. Use a consistent business time zone.
- **Revenue retention** (net revenue retention) uses the same structure but sums revenue instead of counting distinct users. It can exceed 100% through expansion.

**At scale:** precompute a `user_activity_monthly` table (one row per user per active month) incrementally, partitioned by month. The retention query then becomes a cheap aggregation, and adding a new month only touches new data.

---

← [Chapter 4. Data Engineering Fundamentals and Architecture](04-data-engineering-fundamentals-and-architecture.md) · [Table of contents](../README.md) · [Chapter 6. Distributed Processing and Streaming](06-distributed-processing-and-streaming.md) →
