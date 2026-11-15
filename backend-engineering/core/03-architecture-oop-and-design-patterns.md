<!-- notion-id: 3bd67418285480739651cd80ce66c26a -->
# Backend Engineering: Architecture, OOP and Design Patterns

DRY/DIE, architecture vs design, microservices, OOP, constructors, SOLID, multiple inheritance and design patterns.

### Q1. What are the DRY and DIE principles?

- **DRY — Don’t Repeat Yourself:** Avoid duplicating code or logic. Put reusable functionality in a single function, class, or module and reuse it.
  - **Goal:** Reduce duplication and make code easier to maintain.
- **DIE — Duplication Is Evil:** A software design principle emphasizing that duplicated code creates maintenance problems. When the same logic exists in multiple places, changes can become inconsistent and error-prone.
  - **Goal:** Identify and eliminate unnecessary duplication.

### Q2. Differentiate between software architecture and software design.

- **Software Architecture:** Defines the **high-level structure** of a software system—major components, services, their relationships, data flow, and technology choices.
  - Focus: **“What are the major parts, and how do they interact?”**
  - Example: Choosing a **microservices architecture** with API Gateway, authentication service, database, message queue, and AI service.
- **Software Design:** Defines the **detailed implementation** of those components—classes, functions, interfaces, algorithms, database schemas, and design patterns.
  - Focus: **“How will each component be implemented?”**
  - Example: Designing the classes, API endpoints, validation logic, and database queries inside the AI service.

**In short:**

**Architecture = high-level system structure**

**Design = detailed implementation of that structure**

### Q3. Why is Microservices architecture used?

- **Independent scaling:** Each service can be scaled independently based on its workload.
- **Loose coupling:** Services are separated, so changes in one service have less impact on others.
- **Independent deployment:** Teams can develop, test, and deploy services independently.
- **Technology flexibility:** Different services can use different technologies or programming languages when appropriate.
- **Fault isolation:** A failure in one service does not necessarily bring down the entire system.
- **Team scalability:** Different teams can own different services and work in parallel.
- **Better maintainability:** Smaller, focused services can be easier to understand and modify.

**In short:** Microservices are used to build systems that are **scalable, independently deployable, loosely coupled, and easier to maintain as the application and development team grow**.

#### Repeated later in Notion as: "Why would you choose Microservices architecture?"

**Microservices architecture** is chosen when an application benefits from being divided into **small, independent services**, each responsible for a specific business capability.

- **Independent scaling:** Scale only the services that need more resources.
- **Independent deployment:** Deploy or update one service without redeploying the entire application.
- **Fault isolation:** Failure in one service is less likely to bring down the whole system.
- **Team scalability:** Different teams can develop and maintain different services.
- **Technology flexibility:** Services can use different technologies when appropriate.
- **Maintainability:** Smaller, focused services are easier to understand and modify.
- **Faster development:** Teams can work on services in parallel.

**In short:** I choose microservices when the system needs **independent scaling, deployment, fault isolation, and team ownership**, especially as the application grows.

### Q4. What is object-oriented programming?

- **Object-Oriented Programming (OOP)** is a programming paradigm that organizes software around **objects**, which contain **data (attributes)** and **behavior (methods)**.
- It helps make software **modular, reusable, maintainable, and easier to extend**.

#### Four main principles of OOP

1. **Encapsulation:** Bundling data and methods together and controlling access to internal state.
2. **Abstraction:** Hiding unnecessary implementation details and exposing only what is needed.
3. **Inheritance:** Allowing one class to reuse or extend the properties and behavior of another class.
4. **Polymorphism:** Allowing the same interface or method to behave differently for different objects.

**In short:** OOP models a system as **interacting objects**, making complex software easier to organize and maintain.

### Q5. What is a constructor?

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

### Q6. What are the various types of constructors?

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

### Q7. SOLID Design

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

### Q8. Multiple Inheritance: Python vs Java

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

### Q9. What is a design pattern?

- A **design pattern** is a **reusable, proven solution to a common software design problem**.
- It provides a general approach for structuring classes, objects, and their interactions.
- Design patterns are **not ready-made code**; they are guidelines that can be adapted to different situations.

#### Common Design Patterns

- **Creational:** Factory, Singleton, Builder
- **Structural:** Adapter, Decorator, Facade
- **Behavioral:** Observer, Strategy, Command

**Example:** The **Factory Pattern** can be used to create different types of objects without requiring the calling code to know the exact class being instantiated.

**In short:** A design pattern is a **standard, reusable approach for solving recurring software design problems**.
