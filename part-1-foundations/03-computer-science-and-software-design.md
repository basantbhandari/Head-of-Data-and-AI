# Chapter 3. Computer Science and Software Design

_Part I — Foundations_

Data structures, memory, concurrency, object-oriented design, SOLID and design patterns: the software craftsmanship expected of any senior engineer.

This chapter contains **18 questions**.

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

---

← [Chapter 2. Python Engineering](02-python-engineering.md) · [Table of contents](../README.md) · [Chapter 4. Data Engineering Fundamentals and Architecture](../part-2-data-engineering/04-data-engineering-fundamentals-and-architecture.md) →
