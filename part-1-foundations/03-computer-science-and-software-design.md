# Chapter 3. Computer Science and Software Design

_Part I — Foundations_

Data structures, memory, concurrency, object-oriented design, SOLID and design patterns: the software craftsmanship expected of any senior engineer.

This chapter contains **26 questions**.

## Contents

- **3.1** [What are the DRY and DIE principles?](#31-what-are-the-dry-and-die-principles)
- **3.2** [Differentiate between software architecture and software design.](#32-differentiate-between-software-architecture-and-software-design)
- **3.3** [Why is Microservices architecture used?](#33-why-is-microservices-architecture-used)
- **3.4** [What is object-oriented programming?](#34-what-is-object-oriented-programming)
- **3.5** [What is a constructor?](#35-what-is-a-constructor)
- **3.6** [What are the various types of constructors?](#36-what-are-the-various-types-of-constructors)
- **3.7** [What are the SOLID design principles?](#37-what-are-the-solid-design-principles)
- **3.8** [How does multiple inheritance differ between Python and Java?](#38-how-does-multiple-inheritance-differ-between-python-and-java)
- **3.9** [What is a design pattern?](#39-what-is-a-design-pattern)
- **3.10** [How does a HashMap function?](#310-how-does-a-hashmap-function)
- **3.11** [What's a thread?](#311-whats-a-thread)
- **3.12** [What's the difference between a thread and a process?](#312-whats-the-difference-between-a-thread-and-a-process)
- **3.13** [In terms of access, how are array and linked lists different?](#313-in-terms-of-access-how-are-array-and-linked-lists-different)
- **3.14** [What is an Array?](#314-what-is-an-array)
- **3.15** [What is a linked list?](#315-what-is-a-linked-list)
- **3.16** [What is Static Memory Allocation?](#316-what-is-static-memory-allocation)
- **3.17** [How would you define Dynamic Memory Allocation?](#317-how-would-you-define-dynamic-memory-allocation)
- **3.18** [Are elements stored consecutively in linked lists?](#318-are-elements-stored-consecutively-in-linked-lists)
- **3.19** [What is Big-O notation, and what are the time complexities of common operations on arrays, hash maps, heaps, balanced trees and sorting?](#319-what-is-big-o-notation-and-what-are-the-time-complexities-of-common-operations-on-arrays-hash-maps-heaps-balanced-trees-and-sorting)
- **3.20** [What is consistent hashing, and why do distributed data systems use it?](#320-what-is-consistent-hashing-and-why-do-distributed-data-systems-use-it)
- **3.21** [What is a Bloom filter, and where is it used in data systems?](#321-what-is-a-bloom-filter-and-where-is-it-used-in-data-systems)
- **3.22** [How do LSM trees and B-trees differ, and which workloads suit each?](#322-how-do-lsm-trees-and-b-trees-differ-and-which-workloads-suit-each)
- **3.23** [How would you design an LRU cache?](#323-how-would-you-design-an-lru-cache)
- **3.24** [What is the difference between concurrency and parallelism, and how do threads, processes and async I/O fit in?](#324-what-is-the-difference-between-concurrency-and-parallelism-and-how-do-threads-processes-and-async-io-fit-in)
- **3.25** [What are HyperLogLog and Count-Min Sketch, and when would you use approximate algorithms in data pipelines?](#325-what-are-hyperloglog-and-count-min-sketch-and-when-would-you-use-approximate-algorithms-in-data-pipelines)
- **3.26** [How do you find the top-K most frequent items in a very large dataset or stream?](#326-how-do-you-find-the-top-k-most-frequent-items-in-a-very-large-dataset-or-stream)

---

### 3.1 What are the DRY and DIE principles?

- **DRY — Don’t Repeat Yourself:** Avoid duplicating code or logic. Put reusable functionality in a single function, class, or module and reuse it.
  - **Goal:** Reduce duplication and make code easier to maintain.
- **DIE — Duplication Is Evil:** A software design principle emphasizing that duplicated code creates maintenance problems. When the same logic exists in multiple places, changes can become inconsistent and error-prone.
  - **Goal:** Identify and eliminate unnecessary duplication.

### 3.2 Differentiate between software architecture and software design.

- **Software Architecture:** Defines the **high-level structure** of a software system—major components, services, their relationships, data flow, and technology choices.
  - Focus: **“What are the major parts, and how do they interact?”**
  - Example: Choosing a **microservices architecture** with API Gateway, authentication service, database, message queue, and AI service.
- **Software Design:** Defines the **detailed implementation** of those components—classes, functions, interfaces, algorithms, database schemas, and design patterns.
  - Focus: **“How will each component be implemented?”**
  - Example: Designing the classes, API endpoints, validation logic, and database queries inside the AI service.

**In short:**

**Architecture = high-level system structure**

**Design = detailed implementation of that structure**

### 3.3 Why is Microservices architecture used?

- **Independent scaling:** Each service can be scaled independently based on its workload.
- **Loose coupling:** Services are separated, so changes in one service have less impact on others.
- **Independent deployment:** Teams can develop, test, and deploy services independently.
- **Technology flexibility:** Different services can use different technologies or programming languages when appropriate.
- **Fault isolation:** A failure in one service does not necessarily bring down the entire system.
- **Team scalability:** Different teams can own different services and work in parallel.
- **Better maintainability:** Smaller, focused services can be easier to understand and modify.

**In short:** Microservices are used to build systems that are **scalable, independently deployable, loosely coupled, and easier to maintain as the application and development team grow**.

#### Repeated later as: "Why would you choose Microservices architecture?"

**Microservices architecture** is chosen when an application benefits from being divided into **small, independent services**, each responsible for a specific business capability.

- **Independent scaling:** Scale only the services that need more resources.
- **Independent deployment:** Deploy or update one service without redeploying the entire application.
- **Fault isolation:** Failure in one service is less likely to bring down the whole system.
- **Team scalability:** Different teams can develop and maintain different services.
- **Technology flexibility:** Services can use different technologies when appropriate.
- **Maintainability:** Smaller, focused services are easier to understand and modify.
- **Faster development:** Teams can work on services in parallel.

**In short:** I choose microservices when the system needs **independent scaling, deployment, fault isolation, and team ownership**, especially as the application grows.

### 3.4 What is object-oriented programming?

- **Object-Oriented Programming (OOP)** is a programming paradigm that organizes software around **objects**, which contain **data (attributes)** and **behavior (methods)**.
- It helps make software **modular, reusable, maintainable, and easier to extend**.

#### Four main principles of OOP

1. **Encapsulation:** Bundling data and methods together and controlling access to internal state.
2. **Abstraction:** Hiding unnecessary implementation details and exposing only what is needed.
3. **Inheritance:** Allowing one class to reuse or extend the properties and behavior of another class.
4. **Polymorphism:** Allowing the same interface or method to behave differently for different objects.

**In short:** OOP models a system as **interacting objects**, making complex software easier to organize and maintain.

### 3.5 What is a constructor?

- A **constructor** is a special method in a class that is **automatically called when an object is created**.
- It is mainly used to **initialize the object's attributes**.
- In Python, the constructor-like method is **`__init__()`**.

Example:

```py
class User:
    def __init__(self, name):
        self.name = name

user = User("John")
print(user.name)
```

Here, `__init__()` initializes the `name` attribute when the `User` object is created.

**In short:** A constructor **initializes an object when it is created**.

### 3.6 What are the various types of constructors?

In Python/OOP, constructors can commonly be described as:

- **Default constructor:** A constructor that takes no arguments other than `self` and initializes an object with default values.
- **Parameterized constructor:** A constructor that accepts arguments to initialize an object with specific values.
- **Non-parameterized constructor:** A constructor that does not accept additional arguments.

Example:

```py
class User:
    def __init__(self, name="Unknown"):  # default/non-parameterized
        self.name = name

class Employee:
    def __init__(self, name, salary):    # parameterized
        self.name = name
        self.salary = salary
```

**In short:** The main distinction is whether the constructor **requires arguments to initialize the object**.

### 3.7 What are the SOLID design principles?

#### SOLID Design Principles

**SOLID** is a set of five principles for writing **maintainable, flexible, and scalable object-oriented code**.

- **S — Single Responsibility Principle (SRP)**
  - A class should have **one responsibility** and therefore one reason to change.
  - Example: Separate user authentication logic from email notification logic.
- **O — Open/Closed Principle (OCP)**
  - Software entities should be **open for extension but closed for modification**.
  - Add new behavior without changing existing, tested code.
- **L — Liskov Substitution Principle (LSP)**
  - A subclass should be **substitutable for its parent class** without breaking the expected behavior of the program.
- **I — Interface Segregation Principle (ISP)**
  - Clients should not be forced to depend on **methods they do not use**.
  - Prefer several small, focused interfaces over one large interface.
- **D — Dependency Inversion Principle (DIP)**
  - High-level modules should not depend directly on low-level implementations. Both should depend on **abstractions**.

**In short:**

**S = Single Responsibility**

**O = Open/Closed**

**L = Liskov Substitution**

**I = Interface Segregation**

**D = Dependency Inversion**

### 3.8 How does multiple inheritance differ between Python and Java?

- **Python:** Supports **multiple inheritance** directly. A class can inherit from multiple parent classes.

```py
class A:
    pass

class B:
    pass

class C(A, B):
    pass
```

- **Java:** Does **not** support multiple inheritance through classes. A class can extend only **one class**.

```java
class C extends A {   // Only one parent class
}
```

- However, Java supports multiple inheritance through **interfaces**:

```java
class C implements A, B {
}
```

#### Key Difference

| Feature | Python | Java |
| --- | --- | --- |
| Multiple class inheritance | Yes | No |
| Multiple interface inheritance | Yes | Yes |
| Method Resolution Order | Uses **MRO** | Uses class hierarchy + interface rules |
| Diamond problem | Handled using MRO | Avoided for classes |

**In short:**

**Python → multiple inheritance of classes is supported.**

**Java → multiple class inheritance is not supported; use interfaces instead.**

### 3.9 What is a design pattern?

- A **design pattern** is a **reusable, proven solution to a common software design problem**.
- It provides a general approach for structuring classes, objects, and their interactions.
- Design patterns are **not ready-made code**; they are guidelines that can be adapted to different situations.

#### Common Design Patterns

- **Creational:** Factory, Singleton, Builder
- **Structural:** Adapter, Decorator, Facade
- **Behavioral:** Observer, Strategy, Command

**Example:** The **Factory Pattern** can be used to create different types of objects without requiring the calling code to know the exact class being instantiated.

**In short:** A design pattern is a **standard, reusable approach for solving recurring software design problems**.

### 3.10 How does a HashMap function?

A **HashMap** stores data as **key-value pairs** and uses a **hash function** to efficiently locate values.

1. **Hash the key:** The key is passed through a hash function to generate a hash value.
2. **Find the bucket:** The hash is used to determine which bucket should store the key-value pair.
3. **Store the value:** The key and value are stored in that bucket.
4. **Retrieve data:** When searching for a key, its hash is calculated again to find the corresponding bucket.
5. **Handle collisions:** If multiple keys map to the same bucket, the HashMap uses a collision-resolution mechanism, such as a linked structure or tree structure depending on the implementation.
6. **Resize when necessary:** When the map becomes sufficiently full, it increases its capacity and redistributes the entries.

**Time complexity:**

- Average **insert:** O(1)
- Average **lookup:** O(1)
- Average **delete:** O(1)
- Worst case can be **O(n)** depending on collisions and implementation.

**In short:** A HashMap uses **hashing + buckets** to provide fast key-based insertion, lookup, and deletion.

### 3.11 What's a thread?

- A **thread** is the **smallest unit of execution** within a process.
- A process can contain **multiple threads**, and these threads share the process's memory and resources.
- Threads allow a program to perform **multiple tasks concurrently**, improving responsiveness and resource utilization.
- Threads are lighter-weight than processes because they share resources rather than having completely separate memory spaces.

**Example:** In a web server, different threads can handle multiple client requests concurrently.

**In short:**

**Process = running program**

**Thread = execution path within a process**

### 3.12 What's the difference between a thread and a process?

#### Thread vs Process

- **Process**
  - An independent **program in execution**.
  - Has its own memory space and resources.
  - More expensive to create and switch between.
  - Communication between processes requires mechanisms such as IPC.
  - A crash in one process generally does not directly crash another process.
- **Thread**
  - A **unit of execution within a process**.
  - Threads in the same process share memory and resources.
  - Faster and cheaper to create and switch between.
  - Communication is easier because threads share memory.
  - A serious error in one thread can affect the entire process.

| Feature | Process | Thread |
| --- | --- | --- |
| Memory | Separate | Shared within process |
| Creation | Relatively expensive | Relatively lightweight |
| Communication | More complex | Easier through shared memory |
| Isolation | High | Low |
| Execution | Independent | Within a process |

**In short:**

**Process = independent execution environment**

**Thread = execution unit within a process**

### 3.13 In terms of access, how are array and linked lists different?

- **Array:** Provides **direct/random access** to elements using an index, typically in constant time.
- **Linked List:** Provides **sequential access**; you must traverse nodes from the beginning or another known node.
- **Key difference:** Arrays are faster for accessing elements by index, while linked lists are slower because they require traversal.

### 3.14 What is an Array?

- An **Array** is a data structure that stores **multiple elements of the same type** in a **contiguous block of memory**.
- Each element is accessed using an **index**.
- In most programming languages, array indexing starts from **0**.
- Arrays provide **fast access to elements** because an element can be directly accessed using its index.
- The size of a traditional array is usually **fixed** when it is created.
- Common operations include:
  - **Access** → Get an element using its index.
  - **Update** → Change an element.
  - **Search** → Find an element.
  - **Insertion/Deletion** → Add or remove elements, which may require shifting other elements.

**Example:**

`[10, 20, 30, 40]`

- Index `0` → `10`
- Index `1` → `20`
- Index `2` → `30`
- Index `3` → `40`

### 3.15 What is a linked list?

- A **Linked List** is a data structure made up of a sequence of **nodes**, where each node contains:
  - **Data** → Stores the actual value.
  - **Pointer/Reference** → Stores the address/reference of the next node.
- Unlike arrays, linked-list nodes **do not need to be stored in contiguous memory locations**.
- The first node is called the **head**, and the last node usually points to **NULL**.
- Common types include:
  - **Singly Linked List** → Each node points to the next node.
  - **Doubly Linked List** → Each node points to both the previous and next nodes.
  - **Circular Linked List** → The last node points back to the first node.
- **Insertion and deletion** can be efficient because elements do not need to be shifted, provided you already have the relevant node/reference.
- **Accessing an element by position is slower** than in an array because you generally have to traverse the nodes sequentially.

**Example:**

`10 → 20 → 30 → 40 → NULL`

### 3.16 What is Static Memory Allocation?

#### Static Memory Allocation

- **Static Memory Allocation** is the process of allocating memory for variables **before the program starts running**.
- The **size of memory is fixed** and generally cannot be changed during program execution.
- Memory is usually allocated in the **stack or static memory area**, depending on the variable.
- It is generally **faster** because the memory size is known in advance.
- Example: Declaring an array with a fixed size, such as `int arr[10]`.

### 3.17 How would you define Dynamic Memory Allocation?

#### Dynamic Memory Allocation

- **Dynamic Memory Allocation** is the process of allocating memory **during program execution (runtime)**.
- The amount of memory can be **decided based on the program's needs**.
- Memory is typically allocated from the **heap**.
- It is useful when the required memory size is **not known in advance**.
- The allocated memory can generally be **released when it is no longer needed**.
- Example: Creating an array at runtime based on user input.

### 3.18 Are elements stored consecutively in linked lists?

#### Are Elements Stored Consecutively in Linked Lists?

- **No**, elements in a linked list are **not necessarily stored consecutively** in memory.
- Each element is stored in a **node**, and each node contains:
  - The **data**
  - A **pointer/reference** to the next node
- Nodes can be located at **different memory addresses**.
- The pointers/references connect the nodes together.
- This is different from an **array**, where elements are typically stored in consecutive memory locations.

### 3.19 What is Big-O notation, and what are the time complexities of common operations on arrays, hash maps, heaps, balanced trees and sorting?

Big-O notation gives an upper bound on how an algorithm's running time (or memory) grows as the input size $n$ grows, ignoring constant factors and lower-order terms. It answers "what happens when the data is 100x bigger?", which is exactly the question a data engineer faces when a job that ran fine on a sample is pointed at production.

**Key distinctions interviewers probe:**

- **Worst, average and amortized:** a hash map lookup is $O(1)$ on average but $O(n)$ in the worst case; appending to a dynamic array is $O(1)$ amortized because occasional resizes are spread over many cheap appends.
- **Big-O vs. Big-Theta vs. Big-Omega:** $O$ is an upper bound, $\Omega$ a lower bound, $\Theta$ a tight bound. In interviews "Big-O" usually means the tight worst case.
- **Space complexity matters as much as time** when data does not fit in memory.

| Structure / operation | Access / lookup | Insert | Delete | Notes |
| --- | --- | --- | --- | --- |
| Array (dynamic, e.g. Python list) | $O(1)$ by index | $O(1)$ amortized at end, $O(n)$ in middle | $O(n)$ (shifting) | Search for a value is $O(n)$ |
| Sorted array | $O(\log n)$ binary search | $O(n)$ | $O(n)$ | Great for read-heavy, static data |
| Hash map / set | $O(1)$ average, $O(n)$ worst | $O(1)$ average | $O(1)$ average | No ordering; range queries are $O(n)$ |
| Binary heap | $O(1)$ peek min/max | $O(\log n)$ | $O(\log n)$ pop | Building from $n$ items is $O(n)$ |
| Balanced BST (red-black, AVL) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | Ordered iteration and range queries |
| B-tree (database index) | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | High fan-out keeps depth at 3 to 4 levels |
| Comparison sort (merge, Timsort, heap sort) | not applicable | not applicable | not applicable | $O(n \log n)$; quicksort is $O(n^2)$ worst case |
| Counting / radix sort | not applicable | not applicable | not applicable | $O(n + k)$ for small integer key ranges |

**Why this matters for data work:**

- A Python loop doing `if x in some_list` inside another loop is $O(n \cdot m)$; converting the list to a set makes it $O(n + m)$. This single change regularly turns hours into seconds.
- Joins follow the same logic: a nested-loop join is $O(n \cdot m)$, a hash join is $O(n + m)$, a sort-merge join is $O(n \log n + m \log m)$. Spark and database optimizers pick among these for that reason.
- Sorting a large dataset is $O(n \log n)$ plus a shuffle; an `ORDER BY` on a billion rows is expensive, and a top-K with a heap ($O(n \log k)$) is often all you actually need.

**Production caveat:** Big-O hides constants and memory effects. A linear scan over a contiguous array can beat a tree lookup for small $n$ because of CPU caches, and in distributed systems network I/O and shuffles usually dominate CPU complexity. Use Big-O to rule out bad designs, then measure.

### 3.20 What is consistent hashing, and why do distributed data systems use it?

Consistent hashing is a way of assigning keys to nodes so that when a node joins or leaves, only a small fraction of keys move, roughly $K/N$ for $K$ keys and $N$ nodes, instead of nearly all of them.

**The problem it solves:** naive partitioning uses `hash(key) % N`. If $N$ changes from 10 to 11, almost every key maps to a different node, which means a cluster-wide data reshuffle, cold caches and a thundering herd on the database.

**How it works:**

1. Hash both nodes and keys onto the same circular space (a "ring"), for example 0 to $2^{64} - 1$.
2. A key belongs to the first node found walking clockwise from the key's position.
3. Adding a node takes over only the keys between it and its predecessor; removing a node hands its keys to its successor. Everything else stays put.

**Virtual nodes (vnodes):** with one position per physical node, the ring is lumpy and one node can own a far larger arc than another. Each physical node is therefore placed at many points (tens to hundreds). This evens out load, lets a joining node take small slices from many peers instead of a big slice from one, and lets bigger machines own more vnodes.

**Replication:** the key is stored on the owning node plus the next $R - 1$ distinct physical nodes clockwise, which is how Dynamo-style systems build their preference lists.

**Where you meet it:**

- **Amazon Dynamo and Apache Cassandra:** token ring with vnodes for partitioning and replication.
- **Distributed caches:** memcached and Redis client libraries use it so adding a cache node does not invalidate the whole cache.
- **Load balancers and CDNs:** sticky routing of a user or object to the same backend.
- **Stream processing and sharded services:** routing a key to a consistent worker.

**Trade-offs and alternatives:**

| Approach | Rebalance cost | Range queries | Typical users |
| --- | --- | --- | --- |
| `hash % N` | Nearly all keys move | No | Fixed-size clusters only |
| Consistent hashing | About $K/N$ keys move | No (keys scattered) | Cassandra, Dynamo, caches |
| Fixed number of hash slots | Move whole slots | No | Redis Cluster (16,384 slots), Kafka partitions |
| Range partitioning | Split or merge ranges | Yes | HBase, Bigtable, CockroachDB |
| Rendezvous (highest random weight) hashing | About $K/N$ keys move | No | Some load balancers, simpler with no ring |

**Failure modes to mention:** hot keys still overload a single node (consistent hashing balances keys, not traffic), so you add key salting or caching; and rebalancing still streams data, so throttle it to protect live traffic. Note that Kafka does not use consistent hashing for producers: the default partitioner is effectively `hash(key) % partitions`, which is why adding partitions breaks key ordering for existing keys.

### 3.21 What is a Bloom filter, and where is it used in data systems?

A Bloom filter is a compact, probabilistic set that answers "is this element possibly in the set?" It can return false positives but never false negatives: "no" means definitely absent, "yes" means probably present.

**How it works:**

- A bit array of $m$ bits, all zero, and $k$ independent hash functions.
- **Insert:** hash the item $k$ times and set those $k$ bits to 1.
- **Query:** hash the item $k$ times; if any of those bits is 0, the item was never inserted. If all are 1, it was probably inserted (or the bits were set by other items).
- Standard Bloom filters cannot delete items, because clearing a bit could remove other items. Counting Bloom filters and cuckoo filters support deletion.

**False positive rate.** After inserting $n$ items, the probability of a false positive is approximately

$$
p \approx \left(1 - e^{-kn/m}\right)^{k}
$$

The number of hash functions that minimises $p$ is

$$
k_{\text{opt}} = \frac{m}{n} \ln 2
$$

and the bits needed for a target false positive rate $p$ are

$$
m = -\frac{n \ln p}{(\ln 2)^2}
$$

**Rule of thumb:** about 9.6 bits per element (with about 7 hash functions) gives a 1% false positive rate; each additional ~4.8 bits per element cuts it by another 10x. A filter for 100 million keys at 1% is roughly 120 MB, versus several GB for the keys themselves.

**Where data engineers meet them:**

- **LSM-tree storage engines (RocksDB, Cassandra, HBase, LevelDB):** each SSTable has a Bloom filter so point lookups skip files that cannot contain the key. This is what keeps LSM reads tolerable.
- **Columnar formats and table formats:** Parquet supports column-level Bloom filters; Delta Lake, Iceberg and engines such as Spark and Trino use them to skip row groups or files for selective equality predicates on high-cardinality columns.
- **Join optimisation:** runtime filters (Spark's runtime Bloom filter join, Impala, Trino dynamic filtering) build a filter from the small side and push it to the scan of the large side.
- **Deduplication in streams:** checking whether an event ID has probably been seen before doing an expensive lookup.
- **Caches and crawlers:** avoiding lookups for keys known not to exist (cache penetration protection), skipping already-visited URLs.

**Pitfalls:**

- **Sizing for the wrong $n$:** if you insert far more items than planned, the false positive rate climbs sharply; size for peak growth or use scalable Bloom filters.
- **Useless for range predicates:** they only help equality and IN checks.
- **Low-cardinality columns:** min/max statistics or dictionary encoding already prune well, so a Bloom filter just adds metadata overhead.
- **False positives must be safe:** design so that a "maybe" triggers a real check, never a wrong answer.

### 3.22 How do LSM trees and B-trees differ, and which workloads suit each?

Both are storage engine structures for sorted key-value data. A B-tree updates fixed-size pages in place and is optimised for reads; a log-structured merge (LSM) tree buffers writes in memory and appends immutable sorted files, which makes it optimised for writes.

**B-tree (PostgreSQL, MySQL InnoDB, SQL Server, Oracle):**

- Data lives in pages (typically 4 to 16 KB) arranged in a balanced tree with high fan-out, so a billion keys fit in 3 to 4 levels.
- A write finds the leaf page and modifies it in place, protected by a write-ahead log for crash safety. Page splits happen when a page fills.
- Reads are a single predictable traversal; range scans walk adjacent leaves.

**LSM tree (RocksDB, LevelDB, Cassandra, ScyllaDB, HBase, and the storage layer of many streaming state stores):**

- Writes go to a write-ahead log and an in-memory sorted memtable.
- When the memtable fills, it is flushed to disk as an immutable sorted file (SSTable).
- Background compaction merges SSTables, discarding overwritten values and tombstones.
- A read checks the memtable, then SSTables from newest to oldest, using Bloom filters and indexes to skip files.

| Aspect | B-tree | LSM tree |
| --- | --- | --- |
| Write pattern | Random in-place page writes | Sequential appends, batched |
| Write throughput | Moderate | High |
| Write amplification | A small change rewrites a whole page | Rewritten repeatedly during compaction (often 10x or more with leveled compaction) |
| Point reads | One traversal, predictable | May touch several levels; Bloom filters help |
| Range scans | Efficient | Must merge across levels; slower |
| Space usage | Fragmentation in partly full pages | Better compression; stale versions until compaction |
| Latency predictability | Stable | Compaction can cause latency spikes and write stalls |
| Deletes | Immediate | Tombstones linger until compacted |

**Compaction strategies matter:** size-tiered compaction (Cassandra's traditional default) favours write throughput but needs more temporary disk space and reads may touch more files; leveled compaction (RocksDB's default) keeps reads and space tighter at the cost of more write amplification; time-window compaction suits TTL-based time-series data because whole files expire together.

**Which workload suits which:**

- **B-tree:** OLTP with mixed reads and writes, transactional semantics, frequent range queries and secondary indexes, read-heavy services, predictable latency requirements.
- **LSM tree:** write-heavy ingestion such as event logs, metrics, IoT telemetry, messaging, time series and large key-value stores; also embedded state for stream processors (Flink and Kafka Streams use RocksDB for large state).

**Production failure modes:** an LSM store that cannot keep up with compaction accumulates files, read latency climbs and the engine throttles or stalls writes; heavy deletes create tombstone build-up that slows scans (a classic Cassandra issue). B-trees suffer under random-key insert patterns (for example random UUID primary keys causing page splits and index bloat), which is why time-ordered keys such as UUIDv7 are preferred.

**How to answer the "which would you choose" question:** start from the read/write ratio, whether you need range scans, latency SLOs and the hardware. Then note that benchmarks on your own data beat rules of thumb, since a well-tuned engine of either kind often outperforms a poorly tuned one of the other.

### 3.23 How would you design an LRU cache?

An LRU (least recently used) cache holds a fixed number of entries and, when full, evicts the entry that has gone unused the longest. The standard design achieves $O(1)$ `get` and `put` by combining a hash map with a doubly linked list.

**Why two structures:**

- The **hash map** gives $O(1)$ lookup from key to node.
- The **doubly linked list** keeps entries in recency order: most recently used at the head, least recently used at the tail. Because each node has `prev` and `next` pointers, you can unlink any node in $O(1)$ once the map has found it.
- A plain list or array would make "move to front" $O(n)$; a heap ordered by timestamp would make it $O(\log n)$.

**Operations:**

- **get(key):** look up the node; if absent return a miss. If present, move it to the head and return its value.
- **put(key, value):** if the key exists, update the value and move it to the head. Otherwise create a node at the head; if capacity is exceeded, remove the tail node and delete its key from the map.

```python
class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key, self.value = key, value
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.map = {}
        self.head, self.tail = Node(), Node()   # sentinels
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key, value):
        node = self.map.get(key)
        if node:
            node.value = value
            self._unlink(node)
        else:
            if len(self.map) >= self.capacity:
                lru = self.tail.prev
                self._unlink(lru)
                del self.map[lru.key]
            node = Node(key, value)
            self.map[key] = node
        self._push_front(node)
```

Sentinel head and tail nodes remove all the empty-list edge cases. In real Python code you would use `collections.OrderedDict` (`move_to_end` and `popitem(last=False)` are both $O(1)$) or `functools.lru_cache` for memoising functions; interviewers usually want the hand-built version first.

**Senior-level extensions:**

- **Thread safety:** wrap operations in a lock, or shard the cache into N independently locked segments to reduce contention.
- **TTL expiry:** store an expiry timestamp per entry and evict lazily on read plus a periodic sweep.
- **Size-based capacity:** evict by total bytes rather than entry count when values vary in size (for example cached query results or embeddings).
- **Scan resistance:** a single large scan can flush every hot entry from a pure LRU. Alternatives such as LRU-K, 2Q, ARC or W-TinyLFU (used by Caffeine) keep frequently used items despite one-off scans. Redis uses an approximated LRU or LFU by sampling keys rather than maintaining a full list.
- **Distributed caching:** for multiple processes, use Redis or memcached with consistent hashing rather than per-process caches that diverge.

**Data and AI uses:** caching feature lookups for online inference, embedding or LLM response caches keyed by a prompt hash, metadata caches in query engines, and broadcast lookup tables in stream processing.

### 3.24 What is the difference between concurrency and parallelism, and how do threads, processes and async I/O fit in?

Concurrency is about structuring a program to deal with many tasks that are in progress at the same time, interleaving them; parallelism is about executing multiple tasks at literally the same instant on multiple CPU cores. Concurrency is a design property; parallelism is an execution property. You can have concurrency on one core (task switching) and parallelism without much concurrency design (a vectorised computation split across cores).

**The core question is what the workload waits on:**

- **I/O-bound:** time is spent waiting on networks, disks, databases or APIs (calling an LLM API, reading from S3, querying a database). Concurrency helps: while one task waits, another runs.
- **CPU-bound:** time is spent computing (parsing, compression, feature engineering, model inference on CPU). Only parallelism helps: more cores doing work at once.

**The three main tools:**

| Model | How it works | Best for | Costs and risks |
| --- | --- | --- | --- |
| Threads | OS-scheduled, share memory within a process, preemptive switching | Blocking I/O with libraries that are not async; CPU work in runtimes or native extensions that run threads truly in parallel | Race conditions, locks, deadlocks; each thread costs memory (often around 1 MB of stack reserved) |
| Processes | Separate memory spaces, communicate via IPC, pipes or shared memory | CPU-bound work that must use many cores; isolation from crashes | Higher startup cost; data must be serialised between processes |
| Async I/O (event loop, coroutines) | A single thread runs many coroutines; each yields at `await` while waiting on I/O | Very high numbers of concurrent network calls (thousands of HTTP or database requests) | One blocking call stalls everything; requires async-aware libraries; harder debugging |

**A practical Python example:** fetching 10,000 documents from an API and then embedding them.

```python
import asyncio, aiohttp
from concurrent.futures import ProcessPoolExecutor

async def fetch_all(urls, limit=100):
    sem = asyncio.Semaphore(limit)              # cap in-flight requests
    async with aiohttp.ClientSession() as session:
        async def fetch(u):
            async with sem, session.get(u) as r:
                return await r.text()
        return await asyncio.gather(*(fetch(u) for u in urls))

docs = asyncio.run(fetch_all(urls))             # I/O-bound: async concurrency
with ProcessPoolExecutor() as pool:             # CPU-bound: process parallelism
    chunks = list(pool.map(clean_and_tokenise, docs, chunksize=256))
```

**How it scales beyond one machine:** distributed engines like Spark, Dask and Ray add parallelism across machines, with tasks as the unit of parallel work; within each executor, threads handle multiple task slots. The same reasoning applies: is the bottleneck I/O, CPU, memory or network shuffle?

**Production failure modes:**

- **Unbounded concurrency:** firing 50,000 requests at once hits rate limits, exhausts connection pools or file descriptors, and triggers retries that make things worse. Always bound with a semaphore, pool size or queue, and add backoff.
- **Blocking the event loop:** calling a synchronous database driver or `time.sleep` inside async code freezes every coroutine; offload with `asyncio.to_thread` or a thread pool.
- **Shared mutable state:** threads mutating a shared dictionary or counter without locks cause intermittent, hard-to-reproduce bugs. Prefer immutable data, queues, or message passing.
- **Oversubscription:** running a process pool where each process also spawns multithreaded native libraries (NumPy BLAS, PyTorch) can create far more threads than cores; set thread counts explicitly.
- **Serialisation overhead:** shipping large objects to worker processes can cost more than the computation; pass file paths or use shared memory instead.

**In short:** pick async for many concurrent I/O waits, threads for blocking I/O or native code that releases the interpreter, processes (or a distributed engine) for CPU-bound work, and always bound how much runs at once.

### 3.25 What are HyperLogLog and Count-Min Sketch, and when would you use approximate algorithms in data pipelines?

Both are probabilistic sketches: tiny, fixed-size summaries of a stream that answer questions approximately with a known error bound. HyperLogLog (HLL) estimates **how many distinct items** there are; Count-Min Sketch (CMS) estimates **how often a given item** appeared.

**HyperLogLog (cardinality):**

- Each item is hashed. The position of the first 1-bit in the hash (the number of leading zeros) is a signal of how many distinct values have been seen: seeing a hash with 20 leading zeros is unlikely unless you have seen roughly a million distinct items.
- The hash is split into a bucket index and the rest; each of $m$ registers keeps the maximum leading-zero count for its bucket, and a bias-corrected harmonic mean combines them.
- Standard error is about

$$
\frac{1.04}{\sqrt{m}}
$$

  Redis uses $m = 16384$ registers, giving about 0.81% error in roughly 12 KB, whether the true count is a thousand or a billion.
- **Mergeable:** the union of two HLLs is the register-wise maximum. This is the killer feature: store a daily HLL per segment and compute weekly or monthly distinct users by merging, without rescanning raw data.
- **Available in:** `APPROX_COUNT_DISTINCT` in BigQuery, Snowflake, Redshift, Spark's `approx_count_distinct`, Trino's `approx_distinct`, Druid, Redis `PFADD` and `PFCOUNT`, and HLL sketch functions in BigQuery and Snowflake that persist mergeable sketches.

**Count-Min Sketch (frequency):**

- A 2D array of counters with depth $d$ (rows, one hash function each) and width $w$.
- **Update:** for each row, hash the item to a column and increment that counter.
- **Query:** take the minimum across the $d$ counters for the item. Collisions only add, so the estimate never undercounts; the minimum is the least-polluted counter.
- **Error guarantee:** with $w = \lceil e/\varepsilon \rceil$ and $d = \lceil \ln(1/\delta) \rceil$, the estimate $\hat{f}$ satisfies $f \le \hat{f} \le f + \varepsilon N$ with probability at least $1 - \delta$, where $N$ is the total count. Error is relative to the total stream size, so CMS is accurate for frequent items and noisy for rare ones.
- Also mergeable (add the arrays element-wise) when built with the same dimensions and hash seeds.

**When approximation is the right call:**

- Distinct counts over huge or high-cardinality data (unique users, devices, IPs) where an exact `COUNT(DISTINCT)` requires a massive shuffle or memory.
- Real-time dashboards and streaming aggregations where you cannot keep every key in state.
- Rollups that must be re-aggregated across arbitrary dimensions or time windows (store sketches, not counts, because distinct counts are not additive).
- Heavy-hitter detection, rate limiting, abuse and anomaly detection, and query planning statistics.

**When not to use it:**

- Billing, finance, compliance and anything audited; those need exact numbers.
- Small datasets where exact computation is cheap anyway.
- Rare-item frequency queries with CMS, where the additive error swamps the true count.

**How to frame it to stakeholders:** state the error in business terms ("daily active users is 4.21 million, plus or minus about 1%"), document which metrics are approximate, and keep exact computation available for reconciliation. Related sketches worth naming: Theta sketches (support intersections and differences), KLL and t-digest for approximate quantiles and percentiles, all found in Apache DataSketches.

### 3.26 How do you find the top-K most frequent items in a very large dataset or stream?

The right approach depends on whether the data fits in memory, whether it is bounded (batch) or unbounded (stream), and whether the answer must be exact. The classic answer is "count, then use a size-K min-heap", and the senior answer is how that scales when counting itself is the hard part.

**1. Fits in memory: hash map plus min-heap.** Count frequencies with a hash map in $O(n)$, then keep a min-heap of size $K$: push each (count, item), and pop when the heap exceeds $K$. Total cost is $O(n + u \log K)$ for $u$ distinct items, versus $O(u \log u)$ for sorting all counts.

```python
from collections import Counter
import heapq

counts = Counter(stream_of_items)                       # O(n)
top_k = heapq.nlargest(k, counts.items(), key=lambda kv: kv[1])
```

Bucket sort by frequency gives $O(n)$ if frequencies are bounded by $n$; quickselect gives average $O(u)$.

**2. Too big for one machine (batch): distributed aggregation.** The exact pattern is the same as a MapReduce word count:

- Map: emit (item, 1), with partial aggregation (combiners) per partition.
- Shuffle by item key so each item's counts land on one reducer, then sum.
- Each reducer computes its local top K; a final step merges the local top K lists into the global top K.

This is exact because each item's full count is on one partition. In SQL it is simply `GROUP BY item ORDER BY cnt DESC LIMIT K`, and engines like Spark push a per-partition top-K before the final sort, avoiding a full global sort. **Pitfall:** taking a local top K per input partition before aggregating counts by key is wrong; an item ranked 11th everywhere can be first globally.

**3. Unbounded stream or too many distinct items: approximate algorithms.**

| Algorithm | Idea | Memory | Guarantees |
| --- | --- | --- | --- |
| Misra-Gries | Keep $k$ counters; on a new item when full, decrement all | $O(k)$ | Any item with frequency above $N/(k+1)$ is retained; counts underestimate by at most $N/(k+1)$ |
| Space-Saving | Keep $k$ counters; on a new item when full, replace the minimum counter and inherit its count plus one | $O(k)$ | Counts overestimate by at most $N/k$; very accurate in practice on skewed data |
| Count-Min Sketch plus heap | Sketch for frequencies, a size-K heap of current candidates | $O(w \cdot d + K)$ | Never underestimates; error at most $\varepsilon N$ with high probability |
| Lossy Counting | Process the stream in windows and prune low counts | Grows slowly with stream length | Error bounded by $\varepsilon N$ |

Real-world data is usually Zipf-like, which is exactly where these work well: the heavy hitters stand out from the long tail.

**4. Time-windowed top K ("trending in the last hour"):** keep a sketch or counter per small window (for example one per minute), merge the last 60 for the answer, and drop expired windows. Flink and Kafka Streams implement this with keyed windowed aggregations; for very high cardinality, pre-aggregate per key then route aggregated counts to a single top-K operator. Use exponential decay instead of hard windows if older events should fade gradually.

**Production concerns:**

- **Hot keys and skew:** the most frequent items are by definition the skewed keys, so the shuffle stage can bottleneck on one partition. Salting or two-stage aggregation (partial counts per salted key, then combine) fixes it.
- **Exact verification:** a common hybrid is to find candidates approximately, then compute exact counts for those candidates only in a second pass.
- **Ties and stability:** define tie-breaking deterministically so dashboards do not flicker between runs.

---

← [Chapter 2. Python Engineering](02-python-engineering.md) · [Table of contents](../README.md) · [Chapter 4. Data Engineering Fundamentals and Architecture](../part-2-data-engineering/04-data-engineering-fundamentals-and-architecture.md) →
