<!-- notion-id: 3bd67418285480d4abe5df245608077a -->
# Normalization notes

My notes on database normalization (1NF, 2NF, 3NF and the key terms), from the "Data Engineering" Notion page, followed by additions drafted during sync.

## 1 NF

Every column/attribute need to have single value.

Each row should be unique either through single or multiple columns. Not mandatory to have primary key.

## 2 NF

Must be in 1 NF

All non key attribute must be fully dependent on candidate key.

I.e. if the non key attribute is partially dependent on candidate key (subset of column forming candidate key ) then split them into separate tables

Every table should have primary key and the relationship between table are formed using foreign key

Candidate key: set of columns which uniquely identifies record. A table can have multiple candidate keys because their can multiple set of columns that can unique identifies record.

Non key columns/ attribute: the columns in the table that are not the part of candidate key or primary key.

Partial dependency: if your candidate key is the combination of two columns ( or multiple columns ), then every non key column ( that are not part of candidate key ) should be fully dependent on all the column of candidate key. If there is any non key columns that depend only on one of candidate key columns then this results is partially dependent.

## 3 NF

Must be in 2NF

Avoid transitive dependency

Transitive dependency: let's you have table T that has 3 columns A, B, C. If column A is functionally dependent on column B and column B is functionally depended on column C. Then we can say That column A is functionally depended on column C.

---

> ✍️ _Addition drafted during sync._

The notes above define each normal form but have no worked examples and stop at 3NF. The rest of this file fills those gaps: one running example taken from unnormalized data to 3NF, then BCNF, and a short note on when to denormalize.

### Worked example: from unnormalized data to 3NF

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

### BCNF (Boyce–Codd Normal Form)

**Rule:** for every non-trivial functional dependency `X → Y`, X must be a superkey. 3NF has an exception: it allows `X → Y` when Y is part of some candidate key (a "prime" attribute). BCNF removes that exception.

**Example that is in 3NF but not BCNF:** `course_tutor(student, course, tutor)`. The rules are:
- Each tutor teaches exactly one course, so `tutor → course`.
- A student has one tutor per course.

The candidate keys are `(student, course)` and `(student, tutor)`. `tutor → course` holds, but `tutor` is not a superkey. 3NF still allows it because `course` is a prime attribute. The cost is that a tutor's course is repeated for every student, so changing it means updating many rows.

**Fix:** decompose into `tutor_course(tutor, course)` and `student_tutor(student, tutor)`.

**Tradeoff:** a BCNF decomposition is always lossless, but it can lose dependency preservation. Here, "a student has one tutor per course" can no longer be enforced with a single key constraint; it needs a trigger or application logic. That is why many OLTP designs stop at 3NF.

### Beyond BCNF, and when to denormalize

- **4NF** removes multi-valued dependencies. For example, an `employee(skill, language)` table where skills and languages are independent of each other should become two tables.
- **5NF** removes join dependencies. These are rare in practice.
- **Production angle:** normalize OLTP schemas, usually to 3NF or BCNF, for write integrity. Analytical models (star schemas, wide Gold tables) are deliberately denormalized to reduce joins. Fact tables stay roughly in 3NF, while dimensions are flattened. See Q11 and Q23 in [100+ Data Engineer Interview Questions](01-100-data-engineer-interview-questions.md).
