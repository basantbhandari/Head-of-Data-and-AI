<!-- notion-id: 3bd67418285480739651cd80ce66c26a -->
# Backend Engineering: DevOps, CI/CD and Testing

Feature delivery workflow, CI/CD, containerization, and functional, acceptance, performance and database testing.

### Q1. What is the typical workflow to implement a new feature on the back-end?

The typical workflow I follow to implement a new back-end feature is:

1. **Understand the requirements** — clarify the business goal, expected behavior, edge cases, and acceptance criteria.
2. **Design the solution** — decide the API contract, data model, service architecture, and dependencies.
3. **Create a development branch** — keep the feature isolated from the main codebase.
4. **Implement the feature** — develop the API, business logic, database changes, validations, and error handling.
5. **Write tests** — add unit and integration tests and cover important edge cases.
6. **Run code quality checks** — perform linting, formatting, security checks, and review the implementation.
7. **Test locally** — verify the API and its integration with databases and other services.
8. **Create a pull request** — document the changes and get feedback through code review.
9. **Deploy through CI/CD** — merge and deploy to staging for further validation, followed by production after approval.
10. **Monitor after deployment** — check logs, metrics, errors, latency, and user impact to ensure the feature works reliably.

Overall, I follow a cycle of **requirements → design → implementation → testing → code review → deployment → monitoring**.

### Q2. Talk about continuous integration and continuous delivery.

- **Continuous Integration (CI):** The practice of frequently merging code changes into a shared repository. Each change automatically triggers **builds, automated tests, linting, and code-quality checks** to detect problems early.
- **Continuous Delivery (CD):** Extends CI by automatically preparing validated code for **production release**. The application is kept in a deployable state, and production deployment may require manual approval.

#### Typical CI/CD workflow

**Code → Git → Build → Test → Quality/Security Checks → Staging → Production**

- **CI:** Automates build and testing whenever code is pushed or a pull request is created.
- **Continuous Delivery:** Automatically deploys validated changes to staging and keeps them ready for production.
- **Continuous Deployment:** Goes one step further by automatically deploying validated changes directly to production.

**In short:**

**CI = continuously integrate and validate code.**

**Continuous Delivery = continuously prepare validated code for release.**

**Continuous Deployment = automatically release validated code to production.**

### Q3. Differentiate between acceptance and functional tests.

- **Acceptance Testing:** Verifies whether the **complete system meets the business requirements and is acceptable to the customer or end user**.
  - Focus: **“Does the system meet the user's/business needs?”**
  - Usually performed toward the end of development.
  - Example: A customer verifies that an e-commerce system supports the required checkout workflow.
- **Functional Testing:** Verifies whether **specific features and functions work according to their requirements**.
  - Focus: **“Does each feature work correctly?”**
  - Usually performed throughout development and testing.
  - Example: Testing whether the checkout API correctly calculates the total price and processes an order.

**In short:**

**Functional testing = verifies individual functions.**

**Acceptance testing = verifies the complete system against business/user requirements.**

#### Repeated later in Notion as: "What is the difference between Acceptance Test and Functional Test?"

- **Functional Testing:** Verifies that a **specific feature or function** works according to its requirements.
  - Example: Testing whether a login API correctly validates username and password.
  - Usually performed throughout development/testing.
- **Acceptance Testing:** Verifies that the **complete system meets business and user requirements** and is acceptable for release.
  - Example: Testing the entire login and registration workflow from the user's perspective.
  - Usually performed near the end of development.

**In short:**

**Functional test = Does the feature work correctly?**

**Acceptance test = Does the complete system meet the user's/business needs?**

### Q4. Mention some performance testing steps.

- **Define performance requirements:** Establish expected response time, throughput, concurrent users, and resource limits.
- **Identify critical scenarios:** Select important APIs, workflows, and operations to test.
- **Prepare the test environment:** Configure an environment similar to production and use representative test data.
- **Create test scenarios:** Define realistic workloads and user behavior.
- **Run baseline tests:** Measure normal performance under expected traffic.
- **Run load tests:** Gradually increase traffic to determine how the system performs under expected load.
- **Run stress tests:** Push the system beyond normal capacity to identify its breaking point.
- **Monitor metrics:** Track latency, throughput, CPU, memory, database performance, and error rates.
- **Analyze bottlenecks:** Identify slow queries, inefficient code, network issues, or resource constraints.
- **Optimize and retest:** Apply improvements and repeat the tests to verify performance gains.

**In short:**

**Define requirements → prepare environment → create workload → test → monitor → identify bottlenecks → optimize → retest.**

### Q5. What is Containerization?

**Containerization** is the process of packaging an application together with its **code, dependencies, libraries, and configuration** into a lightweight, isolated container so it can run consistently across different environments.

- **Consistency:** Runs the same way across development, testing, and production.
- **Isolation:** Applications and their dependencies are separated from each other.
- **Portability:** Containers can run on different machines and cloud platforms.
- **Scalability:** Containers can be easily replicated and scaled.
- **Fast startup:** Containers are generally faster and lighter than virtual machines.
- **Common tool:** **Docker** is widely used for containerization.

**Example:** Package a FastAPI application with Python dependencies into a Docker container and run the same container locally, in staging, and in production.

**In short:** Containerization = **packaging an application and its dependencies into an isolated, portable environment.**

### Q6. What is Database Black Box Testing?

**Database Black Box Testing** is a testing approach where the **database is tested from the user's or application's perspective without examining its internal implementation**.

- Focuses on **inputs and outputs**, not internal database code.
- Tests whether data is correctly **inserted, updated, retrieved, and deleted**.
- Verifies **data integrity and business rules**.
- Checks constraints such as `PRIMARY KEY`, `FOREIGN KEY`, `UNIQUE`, and `NOT NULL`.
- Tests database responses to valid and invalid inputs.
- Does not require knowledge of the database's internal implementation.

**Example:** Insert a customer through an application and verify that the correct customer data is stored and retrieved.

**In short:** Database black box testing verifies **database behavior and results without looking at its internal structure or implementation**.
