# Backend Engineering: Python Basics

Core Python interview questions: strings, files, serialization, exceptions, memory, namespaces, modules, pickling, access modifiers and more.

### Q1. How do you capitalize the first letter in a string in Python?

In Python, you can capitalize the first letter of a string using the **`capitalize()`** method:

```py
text = "hello world"
result = text.capitalize()

print(result)
```

Output:

```bash
Hello world
```

**Note:** `capitalize()` also converts the remaining characters to lowercase.

If you only want to uppercase the first character while keeping the rest unchanged:

```py
text = "hELLO world"
result = text[0].upper() + text[1:]
```

Output:

```bash
HELLO world
```

### Q2. How to convert a string to all lowercase in Python?

In Python, use the **`lower()`** method:

```py
text = "Hello WORLD"
result = text.lower()

print(result)
```

Output:

```bash
hello world
```

**Answer:** `string.lower()` converts all letters in a string to lowercase.

### Q3. How to delete files in Python?

In Python, you can delete a file using the **`os.remove()`** function:

```py
import os

os.remove("example.txt")
```

You can also use `pathlib`:

```py
from pathlib import Path

Path("example.txt").unlink()
```

**In short:** `os.remove()` or `Path.unlink()` can be used to delete files in Python.

### Q4. What is serialization?

- **Serialization** is the process of converting an object or data structure into a **format that can be stored or transmitted**, such as JSON, XML, or binary data.
- It allows complex data to be saved to a file, database, cache, or sent over a network.
- The reverse process is called **deserialization**, which converts the serialized data back into its original object or data structure.

**Example in Python:**

```py
import json

data = {"name": "John", "age": 25}

serialized = json.dumps(data)
print(serialized)
```

Output:

```txt
{"name": "John", "age": 25}
```

**In short:**

**Serialization = object/data → transferable or storable format**

**Deserialization = serialized format → object/data**

### Q5. What is exception handling?

- **Exception handling** is a mechanism for detecting and managing **runtime errors** so that a program can respond gracefully instead of unexpectedly terminating.
- In Python, exception handling is mainly done using **`try`, `except`, `else`, and `finally`**.

```py
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
finally:
    print("Execution completed")
```

- **`try`** → Contains code that may raise an exception.
- **`except`** → Handles the exception.
- **`else`** → Executes when no exception occurs.
- **`finally`** → Executes regardless of whether an exception occurs.

**In short:** Exception handling allows applications to **handle errors gracefully and continue or terminate in a controlled manner**.

### Q6. What language is Python?

- **Python is a high-level, interpreted, general-purpose programming language.**
- It is **dynamically typed**, meaning you don't need to explicitly declare variable types.
- It supports multiple programming paradigms, including:
  - **Object-oriented programming**
  - **Procedural programming**
  - **Functional programming**
- Python is known for its **simple syntax, readability, and extensive standard library**.
- It is widely used in **web development, data engineering, AI/ML, automation, scripting, and scientific computing**.

**In short:** Python is a **high-level, interpreted, dynamically typed, general-purpose programming language**.

### Q7. Is Python a case-sensitive language?

Yes, **Python is case-sensitive**.

- Variable names, function names, class names, and keywords are case-sensitive.
- `name`, `Name`, and `NAME` are treated as **different identifiers**.

Example:

```py
name = "Alice"
Name = "Bob"

print(name)  # Alice
print(Name)  # Bob
```

**In short:** Python treats uppercase and lowercase letters as different.

### Q8. What is PEP 8?

**PEP 8** is the official **Python style guide** that provides conventions for writing **clean, readable, and consistent Python code**.

- Defines naming conventions for variables, functions, classes, and constants.
- Recommends indentation and spacing practices.
- Provides guidelines for line length and code layout.
- Encourages consistent formatting and organization.
- Improves code readability and maintainability.

**In short:** PEP 8 is a set of **coding style and formatting guidelines for Python**.

### Q9. What are decorators in Python?

**Decorators in Python** are functions that **modify or extend the behavior of another function or class without changing its source code**.

- Use the `@decorator` syntax.
- Commonly used for logging, authentication, caching, validation, and timing.
- A decorator takes a function as input and returns a modified function.

Example:

```py
def log_call(func):
    def wrapper():
        print("Function called")
        return func()
    return wrapper

@log_call
def greet():
    print("Hello")

greet()
```

**In short:** A decorator **adds or changes the functionality of an existing function or class without modifying its original code**.

### Q10. What's the difference between .py and .pyc files?

- **`.py` file:** Contains the **Python source code** written by the developer. It is human-readable and editable.
- **`.pyc` file:** Contains **compiled Python bytecode** generated from the `.py` file. It is primarily used to speed up program startup.
- `.pyc` files are usually stored in the `__pycache__` directory.
- Python can automatically generate `.pyc` files when modules are imported.

**In short:** `.py` = source code, `.pyc` = compiled bytecode.

### Q11. How does Python manage memory?

**Python manages memory automatically** using several mechanisms:

- **Private heap:** Python stores objects in a private memory area called the Python heap.
- **Reference counting:** Objects keep track of how many references point to them. When the count reaches zero, the memory can be released.
- **Garbage collection:** Python's garbage collector detects and cleans up **reference cycles** that reference counting alone cannot handle.
- **Memory allocation:** Python has its own memory allocator to efficiently manage memory for objects.
- **Automatic management:** Developers generally do not need to manually allocate or free memory.

**In short:** Python uses **reference counting, garbage collection, and automatic memory allocation** to manage memory.

### Q12. What are namespaces in Python?

**Namespaces in Python** are containers that **map names (identifiers) to objects**. They help avoid naming conflicts and determine where Python looks for a variable, function, or class.

Common types:

- **Local namespace:** Names inside a function.
- **Global namespace:** Names defined at the module/file level.
- **Built-in namespace:** Names provided by Python, such as `print()`, `len()`, and `str`.
- **Enclosing namespace:** Names from an outer function when nested functions are used.

Python generally searches namespaces using the **LEGB rule**:

**Local → Enclosing → Global → Built-in**

**In short:** A namespace is a **mapping of names to objects that organizes identifiers and prevents naming conflicts**.

### Q13. What are Python packages?

**Python packages** are collections of related **Python modules** organized in a directory to provide reusable functionality.

- A **module** is typically a single `.py` file.
- A **package** is a collection of modules and subpackages.
- Packages help organize large projects and promote code reuse.
- They can be installed using tools such as `pip`.
- Examples: `numpy`, `pandas`, `requests`, and `scikit-learn`.

**In short:** A Python package is a **structured collection of reusable Python modules and code**.

### Q14. Specify the tools that help in finding bugs or performing static analysis in Python.

Common Python tools for **finding bugs and performing static analysis** include:

- **Pylint** – Finds coding errors, potential bugs, and style issues.
- **Flake8** – Checks code for style violations and common programming errors.
- **Ruff** – Fast linter and formatter that detects many common issues.
- **mypy** – Performs static type checking.
- **Pyright** – Fast static type checker for Python.
- **Bandit** – Identifies common security vulnerabilities in Python code.
- **Black** – Automatically formats Python code consistently.
- **isort** – Checks and organizes import statements.

**In short:** Common choices are **Ruff/Pylint/Flake8 for linting, mypy/Pyright for type checking, and Bandit for security analysis**.

### Q15. What are Python Modules? Name some Python modules that are most frequently used.

**Python modules** are files containing Python code—such as **functions, classes, and variables**—that can be imported and reused in other Python programs.

- A module is typically a single **`.py` file**.
- Modules help organize code and promote **reusability and maintainability**.
- You can import a module using the `import` statement.

**Commonly used Python modules:**

- `os` – Operating system and file operations
- `sys` – System-specific functionality
- `math` – Mathematical functions
- `datetime` – Date and time operations
- `json` – JSON encoding and decoding
- `re` – Regular expressions
- `random` – Random number generation
- `collections` – Specialized data structures
- `pathlib` – Object-oriented filesystem paths
- `logging` – Application logging

**In short:** A Python module is a **reusable `.py` file containing related Python code**.

### Q16. What are Python's local variables and global variables?

**Local variables** and **global variables** differ based on their **scope**, meaning where they can be accessed.

- **Local variable:** Defined inside a function and accessible only within that function.
- **Global variable:** Defined outside functions and generally accessible throughout the module.

Example:

```py
x = 10  # Global variable

def test():
    y = 20  # Local variable
    print(x)
    print(y)
```

- `x` → global variable
- `y` → local variable

**In short:**

**Local = accessible within a function.**

**Global = accessible throughout the module.**

### Q17. Is indentation necessary in Python?

Yes, **indentation is necessary in Python** because it is used to **define blocks of code** instead of curly braces `{}` like in some other languages.

- Indentation is required after statements such as `if`, `for`, `while`, `def`, and `class`.
- Incorrect indentation causes an **`IndentationError`** or **`TabError`**.
- Python convention is **4 spaces per indentation level**.
- Consistent indentation makes code readable and defines its structure.

Example:

```py
if age >= 18:
    print("Adult")
```

**In short:** Indentation is **syntactically significant in Python and defines the structure of code blocks**.

### Q18. What's the difference between Python arrays and lists?

- **Python List:** Built-in data structure that can store **different data types** and is highly flexible.
  ```py
  items = [1, "Python", 3.5]
  ```
- **Python Array:** Usually refers to the `array` module's `array` type, which stores elements of **the same data type** and can be more memory-efficient for homogeneous numeric data.
  ```py
  from array import array
  numbers = array('i', [1, 2, 3])
  ```

#### Key differences

| Feature | List | Array |
| --- | --- | --- |
| Data types | Can be mixed | Same type |
| Flexibility | High | More restricted |
| Memory | Generally higher | Generally lower for homogeneous data |
| Use case | General-purpose collections | Typed numeric data |

**In short:** **List = flexible, general-purpose collection; Array = typed, memory-efficient collection.**

### Q19. What is self in Python?

**`self` In Python** is a reference to the **current object (instance)** of a class.

- Used inside instance methods to access the object's **attributes and methods**.
- It must be the first parameter of an instance method by convention.
- It allows each object to maintain its own data.

Example:

```py
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(self.name)

person = Person("Alice")
person.greet()
```

Here, `self.name` refers to the `name` attribute of the **current `Person` object**.

**In short:** `self` refers to the **current instance of a class**.

### Q20. What are unpickling and pickling?

**Pickling** and **unpickling** are Python processes used to **serialize and deserialize objects**.

- **Pickling:** Converts a Python object into a **byte stream** so it can be stored or transmitted.
- **Unpickling:** Converts the byte stream back into the **original Python object**.
- Python provides the `pickle` module for this purpose.
- Commonly used to save Python objects such as models, dictionaries, and lists.

Example:

```py
import pickle

# Pickling
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Unpickling
with open("data.pkl", "rb") as f:
    data = pickle.load(f)
```

**In short:**

**Pickling = object → byte stream**

**Unpickling = byte stream → object**

**Security note:** Never unpickle data from an untrusted source, because loading a malicious pickle can execute arbitrary code.

### Q21. How do you do data abstraction in Python?

**Data abstraction in Python** means **hiding the internal implementation details and exposing only the essential functionality** to the user.

- Achieved mainly through **classes, methods, and abstract classes**.
- Python provides the `abc` module for formal abstraction.
- **Abstract Base Classes (ABCs)** can define methods that subclasses must implement.
- Users interact with the public interface without needing to know how it works internally.

Example:

```py
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass

class Dog(Animal):
    def sound(self):
        return "Bark"
```

Here, the user only needs to know that `sound()` provides an animal's sound; the implementation is hidden behind the interface.

**In short:** Data abstraction = **hide implementation details and expose only essential functionality**.

### Q22. What is the usage of the help() and the dir() functionality in Python?

- **`help()`** – Provides **documentation and help** about Python objects, modules, classes, functions, or keywords.
  ```py
  help(str)
  help(print)
  ```
- **`dir()`** – Returns a list of **attributes and methods** available for an object or module.
  ```py
  dir(str)
  dir(my_object)
  ```

**In short:**

**`help()` = learn how something works.**

**`dir()` = see what attributes and methods it has.**

### Q23. What is len() in Python?

`len()` is a **built-in Python function** used to return the **number of items or elements** in an object.

- Works with strings, lists, tuples, dictionaries, sets, and other sized objects.
- For a string, it returns the number of characters.
- For a list, it returns the number of elements.

Example:

```py
name = "Python"
numbers = [10, 20, 30]

print(len(name))     # 6
print(len(numbers))  # 3
```

**In short:** `len()` returns the **size or number of items in an object**.

### Q24. Access Modifiers in Python?

Python doesn't have *true* private/protected/public variables enforced by the language like Java or C++ does. Instead, it relies on **naming conventions** — it's more of a social agreement than a hard rule.

Here's how it works:

#### Public

python

```py
class Person:
    def __init__(self):
        self.name = "Alice"  # public
```

No underscore. Accessible from anywhere, no restrictions.

#### Protected (convention only)

python

```py
class Person:
    def __init__(self):
        self._age = 30  # protected (by convention)
```

Single underscore prefix. This is just a *hint* to other developers: "treat this as internal, don't touch it from outside the class." Python does nothing to actually stop you — you can still access `obj._age` freely.

#### Private (name mangling, not true privacy)

python

```py
class Person:
    def __init__(self):
        self.__salary = 50000  # "private"
```

Double underscore prefix triggers **name mangling** — Python internally renames it to `_ClassName__salary`. This makes accidental access harder, but it's not real privacy — you can still access it if you know the mangled name:

python

```py
p = Person()
print(p._Person__salary)  # works! prints 50000
```

#### Summary

| Convention | Syntax | Enforcement |
| --- | --- | --- |
| Public | `name` | None — fully open |
| Protected | `_name` | None — just a convention/hint |
| Private | `__name` | Name mangling only, not true privacy |

So the honest answer: **no**, Python doesn't have real access control like some other languages. It trusts developers to respect the conventions rather than locking things down with the compiler/interpreter. If you truly need to hide something, you'd need tricks (closures, `__slots__`, properties with logic) but nothing is unbreakable in Python — "we're all consenting adults here" is a common Python philosophy phrase for this.

### Q25. Is Python an interpreted and a compiled language?

Python is actually **both**, in a sense — it depends on how you define "compiled."

#### The short answer

Python is generally called an **interpreted language**, but under the hood it does go through a compilation step first. It's not purely one or the other.

#### What actually happens

1. **Compilation to bytecode**: When you run a `.py` file, Python first compiles your source code into an intermediate form called **bytecode** (`.pyc` files, stored in `__pycache__`). This is a lower-level, platform-independent representation — not machine code.
2. **Interpretation of bytecode**: That bytecode is then run by the **Python Virtual Machine (PVM)**, which interprets it line-by-line at runtime.

```txt
your_code.py  →  [compiler]  →  bytecode (.pyc)  →  [PVM interpreter]  →  execution
```

#### Why people call it "interpreted"

- It doesn't compile straight to native machine code (like C or Rust does with `gcc`).
- You don't get a standalone executable by default.
- Execution happens through an interpreter (CPython, by default) reading bytecode on the fly.
- There's no separate manual "build" step — you just run `python script.py` and it compiles + executes seamlessly, which *feels* interpreted.

#### Why "compiled" isn't wrong either

- That bytecode compilation step is real and mandatory.
- Tools like **Cython**, **Nuitka**, or **PyPy** (with JIT compilation) can compile Python closer to machine code for performance.

#### Comparison

| **Language** | **Process** |
| --- | --- |
| C | Source → machine code (compiled), runs directly on hardware |
| Python (CPython) | Source → bytecode (compiled) → interpreted by PVM |
| Java | Source → bytecode (compiled) → interpreted/JIT-compiled by JVM |

Interestingly, Python's model is actually very similar to Java's — both compile to bytecode and run on a virtual machine — yet Java is usually labeled "compiled" and Python "interpreted." That's mostly due to convention and how transparent/automatic the compilation step feels to the user, not a big technical difference.

**Bottom line**: Python is technically a **hybrid** — compiled to bytecode, then interpreted — but it's colloquially and commonly referred to as an "interpreted language" because that's the dominant, visible behavior from a user's perspective.
