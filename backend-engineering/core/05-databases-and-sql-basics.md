<!-- notion-id: 3bd67418285480739651cd80ce66c26a -->
# Backend Engineering: Databases and SQL Basics

NoSQL, SQL injection, indexes, ORMs, SQL subsets and clauses, OLTP/OLAP, constraints, joins and triggers.

### Q1. What are NoSQL databases? Mention different types of NoSQL databases.

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

### Q2. Explain SQL injection.

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

### Q3. Differentiate between clustered and non-clustered indexes?

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

### Q4. What are some common problems with ORMs?

- **Performance overhead:** ORMs can generate inefficient SQL compared with carefully optimized raw SQL.
- **N+1 query problem:** Loading related objects incorrectly can cause many unnecessary database queries.
- **Complex queries:** Advanced joins, aggregations, window functions, or database-specific features can be difficult to express efficiently through an ORM.
- **Less control over SQL:** Developers may have less visibility and control over the exact queries being executed.
- **Abstraction limitations:** ORM abstractions can hide important database behavior and make performance problems harder to diagnose.
- **Migration issues:** Schema migrations can become complicated, especially with large production databases.
- **Memory usage:** ORMs may load more objects into application memory than necessary.
- **Learning curve:** Developers need to understand both the ORM and the underlying database to use it effectively.

**In short:** ORMs improve developer productivity and maintainability, but they can introduce **performance overhead, inefficient queries, and reduced control over database operations**.

### Q5. What is the difference between MySQL and SQL?

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

### Q6. What are the various subsets in SQL?

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

### Q7. What is RDBMS?

- **RDBMS (Relational Database Management System)** is software used to **store, organize, manage, and retrieve data in relational databases**.
- Data is stored in **tables consisting of rows and columns**.
- Tables can be connected using **relationships**, typically through primary keys and foreign keys.
- RDBMSs usually use **SQL** to create, query, and manage data.
- They support important features such as **transactions, constraints, indexing, concurrency control, and data integrity**.

**Examples:** MySQL, PostgreSQL, Oracle Database, Microsoft SQL Server, and SQLite.

**In short:**

**RDBMS = software that manages structured data stored in related tables.**

### Q8. What is the "SELECT" statement?

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

### Q9. What are the most common clauses used with a SQL SELECT query?

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

### Q10. What is OLTP?

- **OLTP (Online Transaction Processing)** is a type of database system designed to handle **frequent, real-time business transactions**.
- It focuses on **fast inserts, updates, deletes, and short queries**.
- OLTP systems typically support **many concurrent users and transactions**.
- They emphasize **data consistency, integrity, and transaction reliability**.
- Common examples include **banking systems, e-commerce orders, payment systems, and payroll systems**.

**In short:**

**OLTP = fast, reliable processing of day-to-day transactions in real time.**

### Q11. What are the differences between OLAP and OLTP?

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

### Q12. What are SQL comments, and how do they work?

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

### Q13. What do you mean by Denormalization?

**Denormalization** is the process of **intentionally adding redundant data to a database** to improve query performance and reduce the need for complex joins.

- **Purpose:** Improve read/query performance.
- **Redundancy:** Some data may be stored in multiple places.
- **Fewer joins:** Related information can be accessed more quickly.
- **Trade-off:** Requires more storage and can make updates more complex.
- Common in **OLAP/data warehouses** where fast analytical queries are important.

**Example:** Instead of joining `Customers` and `Orders` every time, an order table may also store the customer's name to make frequent queries faster.

**In short:** Denormalization = **adding controlled redundancy to improve read performance.**

### Q14. What's an Index?

An **index** in a database is a data structure that **speeds up data retrieval** from a table, similar to an index in a book.

- **Faster queries:** Helps the database find rows without scanning the entire table.
- **Commonly created on:** Columns frequently used in `WHERE`, `JOIN`, and `ORDER BY`.
- **Trade-off:** Indexes require additional storage and can slow down `INSERT`, `UPDATE`, and `DELETE` operations.
- **Types:** Clustered and non-clustered indexes are common concepts; databases may also support B-tree, hash, and other index types.

**Example:** An index on `user_id` allows the database to find a specific user much faster than scanning every row.

**In short:** An index **improves database query performance by providing a faster way to locate data**.

### Q15. What is the DISTINCT statement? What is its use?

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

### Q16. What is a "Trigger" in SQL?

A **Trigger** in SQL is a special database program that **automatically executes when a specific event occurs** on a table or database.

- **Events:** `INSERT`, `UPDATE`, or `DELETE`.
- **Automatic:** It runs automatically without being explicitly called.
- **Common uses:** Auditing, validation, maintaining related data, and enforcing business rules.
- **Types:** `BEFORE` and `AFTER` triggers are commonly supported, depending on the database.

**Example:** When a new employee is inserted, a trigger can automatically record the action in an audit table.

**In short:** A trigger is **automatically executed SQL logic that responds to database events**.

### Q17. What's the difference between a natural join and a cross join?

- **Natural Join:** Automatically joins two tables based on columns with the **same name and compatible data types**. It returns rows where those common columns have matching values.
- **Cross Join:** Produces the **Cartesian product** of two tables, combining every row from the first table with every row from the second table. It does not require a join condition.

**Example:**

- Table A has 3 rows, and Table B has 4 rows.
- `CROSS JOIN` → **3 × 4 = 12 rows**.
- `NATURAL JOIN` → Only matching rows based on common column names.

**In short:**

**NATURAL JOIN = joins on matching common columns.**

**CROSS JOIN = combines every possible pair of rows.**

### Q18. What is a UNIQUE constraint?

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

### Q19. What is a FOREIGN KEY in SQL?

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

### Q20. How to display the current date in SQL?

The syntax for displaying the **current date** depends on the SQL database:

- **MySQL:** `SELECT CURDATE();`
- **PostgreSQL:** `SELECT CURRENT_DATE;`
- **SQL Server:** `SELECT CAST(GETDATE() AS DATE);`
- **Oracle:** `SELECT SYSDATE FROM dual;`

**In short:** For MySQL, use **`SELECT CURDATE();`**.
