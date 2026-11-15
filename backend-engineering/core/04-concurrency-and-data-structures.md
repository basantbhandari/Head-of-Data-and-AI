<!-- notion-id: 3bd67418285480739651cd80ce66c26a -->
# Backend Engineering: Concurrency and Data Structures

HashMaps, threads, and threads vs processes.

### Q1. How does a HashMap function?

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

### Q2. What's a thread?

- A **thread** is the **smallest unit of execution** within a process.
- A process can contain **multiple threads**, and these threads share the process's memory and resources.
- Threads allow a program to perform **multiple tasks concurrently**, improving responsiveness and resource utilization.
- Threads are lighter-weight than processes because they share resources rather than having completely separate memory spaces.

**Example:** In a web server, different threads can handle multiple client requests concurrently.

**In short:**

**Process = running program**

**Thread = execution path within a process**

### Q3. What's the difference between a thread and a process?

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
