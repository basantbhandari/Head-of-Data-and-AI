# Backend Engineering: Web, APIs and HTTP

HTTP status codes, APIs and REST, web servers, web services and static vs dynamic sites.

### Q1. HTTP response status codes

#### HTTP Response Status Codes

HTTP status codes are **three-digit codes** a server returns to indicate the result of an HTTP request.

- **1xx – Informational:** Request received; processing continues.
- **2xx – Success:** Request was successfully processed.
- **3xx – Redirection:** Client needs to follow a redirect or take additional action.
- **4xx – Client Error:** The request is invalid, or the client is not authorized.
- **5xx – Server Error:** The server encountered an error while processing a valid request.

#### Common Status Codes

- **200 OK** — Request succeeded.
- **201 Created** — A new resource was successfully created.
- **204 No Content** — Request succeeded with no response body.
- **301 Moved Permanently** — Resource has permanently moved to another URL.
- **302 Found** — Temporary redirect.
- **400 Bad Request** — Invalid or malformed request.
- **401 Unauthorized** — Authentication is missing or invalid.
- **403 Forbidden** — Client is authenticated but not allowed to access the resource.
- **404 Not Found** — Requested resource does not exist.
- **409 Conflict** — Request conflicts with the current state of the resource.
- **429 Too Many Requests** — Client has exceeded the allowed request rate.
- **500 Internal Server Error** — Unexpected server-side error.
- **502 Bad Gateway** — Gateway/proxy received an invalid response from an upstream server.
- **503 Service Unavailable** — Server is temporarily unavailable or overloaded.
- **504 Gateway Timeout** — Gateway/proxy did not receive a timely response from an upstream server.

### Q2. Api and its types

An **API (Application Programming Interface)** is a defined way for two software systems to communicate. It specifies what requests can be made, what data should be supplied, and what responses come back.

Common **types of APIs** include:

- **REST API** — Uses HTTP methods such as `GET`, `POST`, `PUT`, `PATCH`, and `DELETE`. Data is commonly exchanged as JSON.
- **SOAP API** — Uses a stricter XML-based protocol with standardized messaging and contracts.
- **GraphQL API** — Lets clients request exactly the fields they need, usually through a single endpoint.
- **RPC APIs** — Treat communication like calling functions remotely. Examples include **gRPC** and JSON-RPC.
- **WebSocket API** — Maintains a persistent two-way connection, useful for live chats, games, notifications, and real-time market data.
- **Web APIs** — A broad category for APIs accessed over a network using HTTP/HTTPS; REST and GraphQL are examples.
- **Library/Framework APIs** — Functions, classes, and methods exposed by a programming library, such as a Python package.
- **Operating System APIs** — Allow programs to interact with OS features such as files, networking, processes, cameras, or notifications.

For example, a weather REST API might work conceptually as:

```txt
GET /weather?city=London
```

and return:

```txt
{
  "city":"London",
  "temperature":18,
  "condition":"Cloudy"
}
```

If by **“API and its types”** you mean this as a computer-science/exam topic, the simplest classification to remember is **REST, SOAP, GraphQL, RPC, and WebSocket APIs**.

### Q3. Define the purpose of the back-end.

#### Purpose of the Back-End

- The **back-end** is the server-side part of an application that handles **business logic, data processing, security, and communication with databases and external services**.
- It receives requests from the front-end, processes them, and sends appropriate responses.
- Main responsibilities include:
  - **Business logic** — implements application rules and workflows.
  - **Database management** — stores, retrieves, updates, and deletes data.
  - **API handling** — provides APIs for the front-end and other services.
  - **Authentication & authorization** — verifies users and controls access.
  - **Data processing** — validates, transforms, and processes information.
  - **Integration** — communicates with external APIs, services, queues, and cloud systems.
  - **Security & reliability** — protects data and handles errors, logging, and monitoring.

**In short:** The back-end is the **engine behind an application** that processes requests, manages data and logic, and provides services to the front-end.

### Q4. What is a web server?

A **web server** is a software application and/or computer system that **receives HTTP/HTTPS requests from clients and sends back responses**, such as web pages, files, or API data.

- **Client:** Browser, mobile app, or another service
- **Web server:** Receives and processes HTTP/HTTPS requests
- **Response:** HTML, JSON, images, files, etc.
- **Common web servers:** Nginx, Apache HTTP Server, Microsoft IIS
- **Example:** A browser requests `GET /users`  the web server receives the request and returns the appropriate response.

In a modern back-end system, a web server often sits in front of an **application server/API**, handling incoming traffic and forwarding requests to the appropriate application service.

### Q5. What's the difference between GET and POST?

- **GET:** Used to **retrieve data** from the server.
  - Parameters are usually sent in the **URL/query string**.
  - Should not modify server data.
  - Can be cached and bookmarked.
  - Example: `GET /users/123`
- **POST:** Used to **send data to the server**, typically to create a resource or trigger an operation.
  - Data is usually sent in the **request body**.
  - Not normally cached or bookmarked.
  - Example: `POST /users` with user information in the body.

**In short:**

**GET = retrieve data**

**POST = submit/send data**

### Q6. Mention some advantages of web services.

- **Interoperability:** Allows different applications, platforms, and programming languages to communicate with each other.
- **Reusability:** A web service can be reused by multiple applications and clients.
- **Loose coupling:** Services can operate independently, making systems easier to modify and maintain.
- **Scalability:** Individual services can be scaled based on demand.
- **Platform independence:** Clients can consume services regardless of their underlying operating system or technology.
- **Easy integration:** Makes it easier to integrate different systems and third-party applications.
- **Remote access:** Services can be accessed over a network using standard protocols such as HTTP/HTTPS.
- **Maintainability:** Changes to one service can often be made without changing the entire application.

**In short:** Web services provide **interoperability, reusability, flexibility, scalability, and easier system integration**.

### Q7. What does REST stand for?

**REST** stands for **Representational State Transfer**.

It is an **architectural style for designing web APIs** that use standard HTTP methods to interact with resources.

- **GET** → Retrieve a resource
- **POST** → Create a resource
- **PUT** → Replace/update a resource
- **PATCH** → Partially update a resource
- **DELETE** → Delete a resource

**In short:** REST is a set of principles for building **simple, scalable, stateless, and resource-oriented web APIs**.

### Q8. What is the difference between static and dynamic websites?

#### Static vs Dynamic Websites

- **Static Website**
  - Content is **pre-built and served as stored files**.
  - The content generally remains the same for all users unless the files are changed.
  - Usually faster, simpler, and easier to host.
  - Common technologies: **HTML, CSS, JavaScript**.
  - Example: A portfolio or company information website.
- **Dynamic Website**
  - Content is **generated or updated based on user requests, data, or application logic**.
  - Can provide personalized and interactive experiences.
  - Usually involves a **back-end application and database**.
  - Common technologies: **Python, Java, Node.js, PHP, databases**, etc.
  - Example: An e-commerce site showing products and prices from a database.

**In short:**

**Static = pre-built, mostly fixed content.**

**Dynamic = content generated or updated based on data and user interaction.**
