# Chapter 5. SQL, Databases and Data Modeling

_Part II — Data Engineering_

Relational fundamentals, SQL, indexing, normalization, dimensional modeling and NoSQL trade-offs.

This chapter contains **37 questions**.

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

---

← [Chapter 4. Data Engineering Fundamentals and Architecture](04-data-engineering-fundamentals-and-architecture.md) · [Table of contents](../README.md) · [Chapter 6. Distributed Processing and Streaming](06-distributed-processing-and-streaming.md) →
