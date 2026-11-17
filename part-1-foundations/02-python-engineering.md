# Chapter 2. Python Engineering

_Part I — Foundations_

Python language semantics, CPython internals, memory management, concurrency (GIL, free-threading, asyncio), packaging, typing, testing and performance for data workloads. Senior candidates are expected to explain what the interpreter is doing and to write maintainable, testable, fast Python.

This chapter contains **77 questions**.

## Contents

- **2.1** [How do you change the case of a string in Python (capitalise, lowercase, case-insensitive comparison)?](#21-how-do-you-change-the-case-of-a-string-in-python-capitalise-lowercase-case-insensitive-comparison)
- **2.2** [How do you delete files and directories safely in Python?](#22-how-do-you-delete-files-and-directories-safely-in-python)
- **2.3** [What is serialization, how does pickling work, and what are its risks?](#23-what-is-serialization-how-does-pickling-work-and-what-are-its-risks)
- **2.4** [How does exception handling work in Python, and how do you make it robust in a production ETL job?](#24-how-does-exception-handling-work-in-python-and-how-do-you-make-it-robust-in-a-production-etl-job)
- **2.5** [What kind of language is Python (typing, case sensitivity, indentation)?](#25-what-kind-of-language-is-python-typing-case-sensitivity-indentation)
- **2.6** [What are decorators, how do they work internally, and how are they used in production?](#26-what-are-decorators-how-do-they-work-internally-and-how-are-they-used-in-production)
- **2.7** [What are .pyc files, and how do bytecode caching and the compile-time optimiser improve performance?](#27-what-are-pyc-files-and-how-do-bytecode-caching-and-the-compile-time-optimiser-improve-performance)
- **2.8** [How does Python manage memory, and what does that mean in practice?](#28-how-does-python-manage-memory-and-what-does-that-mean-in-practice)
- **2.9** [What are namespaces and scopes in Python, and how does the LEGB rule work internally?](#29-what-are-namespaces-and-scopes-in-python-and-how-does-the-legb-rule-work-internally)
- **2.10** [What are Python modules and packages, and which standard-library modules matter most?](#210-what-are-python-modules-and-packages-and-which-standard-library-modules-matter-most)
- **2.11** [What is PEP 8, and which tools enforce code quality and find bugs through static analysis?](#211-what-is-pep-8-and-which-tools-enforce-code-quality-and-find-bugs-through-static-analysis)
- **2.12** [What is the difference between Python lists, `array.array` and NumPy arrays?](#212-what-is-the-difference-between-python-lists-arrayarray-and-numpy-arrays)
- **2.13** [What is self in Python?](#213-what-is-self-in-python)
- **2.14** [How do you do data abstraction in Python?](#214-how-do-you-do-data-abstraction-in-python)
- **2.15** [How do you introspect objects in Python with help(), dir() and related tools?](#215-how-do-you-introspect-objects-in-python-with-help-dir-and-related-tools)
- **2.16** [What does len() do, and how does it work under the hood?](#216-what-does-len-do-and-how-does-it-work-under-the-hood)
- **2.17** [How do access modifiers (public, protected, private) work in Python?](#217-how-do-access-modifiers-public-protected-private-work-in-python)
- **2.18** [Is Python an interpreted and a compiled language?](#218-is-python-an-interpreted-and-a-compiled-language)
- **2.19** [What is the GIL, why does CPython have it, and how does it affect data processing?](#219-what-is-the-gil-why-does-cpython-have-it-and-how-does-it-affect-data-processing)
- **2.20** [What's the difference between `is` and `==`? How does CPython implement identity checks?](#220-whats-the-difference-between-is-and--how-does-cpython-implement-identity-checks)
- **2.21** [How does reference counting work in CPython? What happens when an object's refcount hits zero?](#221-how-does-reference-counting-work-in-cpython-what-happens-when-an-objects-refcount-hits-zero)
- **2.22** [How does Python's garbage collector handle reference cycles?](#222-how-does-pythons-garbage-collector-handle-reference-cycles)
- **2.23** [What's the difference between `__new__` and `__init__`?](#223-whats-the-difference-between-__new__-and-__init__)
- **2.24** [Why is small integer caching (-5 to 256) used, and how does it affect `is` comparisons?](#224-why-is-small-integer-caching--5-to-256-used-and-how-does-it-affect-is-comparisons)
- **2.25** [What is object interning, how does string immutability enable it, and when would you use sys.intern()?](#225-what-is-object-interning-how-does-string-immutability-enable-it-and-when-would-you-use-sysintern)
- **2.26** [How does `sys.getrefcount()` work, and why is its result often higher than expected?](#226-how-does-sysgetrefcount-work-and-why-is-its-result-often-higher-than-expected)
- **2.27** [What's the difference between mutable and immutable objects at the C level?](#227-whats-the-difference-between-mutable-and-immutable-objects-at-the-c-level)
- **2.28** [What is `__slots__`, and how does it change an object's memory layout?](#228-what-is-__slots__-and-how-does-it-change-an-objects-memory-layout)
- **2.29** [What's the difference between a function's `__globals__` and `__locals__`?](#229-whats-the-difference-between-a-functions-__globals__-and-__locals__)
- **2.30** [How do closures capture variables — by reference or by value?](#230-how-do-closures-capture-variables--by-reference-or-by-value)
- **2.31** [What is a cell object, and how does it enable closures?](#231-what-is-a-cell-object-and-how-does-it-enable-closures)
- **2.32** [How does the `nonlocal` keyword work under the hood?](#232-how-does-the-nonlocal-keyword-work-under-the-hood)
- **2.33** [What is a code object (`__code__`), and what does it contain?](#233-what-is-a-code-object-__code__-and-what-does-it-contain)
- **2.34** [How does Python compile source code into bytecode? What role does `dis` play?](#234-how-does-python-compile-source-code-into-bytecode-what-role-does-dis-play)
- **2.35** [What's the difference between a function and a method at the descriptor level?](#235-whats-the-difference-between-a-function-and-a-method-at-the-descriptor-level)
- **2.36** [How does the descriptor protocol (`__get__`, `__set__`, `__delete__`) work, and what is the difference between data and non-data descriptors?](#236-how-does-the-descriptor-protocol-__get__-__set__-__delete__-work-and-what-is-the-difference-between-data-and-non-data-descriptors)
- **2.37** [How are default mutable arguments evaluated, and why does this cause bugs?](#237-how-are-default-mutable-arguments-evaluated-and-why-does-this-cause-bugs)
- **2.38** [What is `__call__`, and how does it make objects callable?](#238-what-is-__call__-and-how-does-it-make-objects-callable)
- **2.39** [How does the Method Resolution Order (MRO) work with multiple inheritance, and what is C3 linearisation?](#239-how-does-the-method-resolution-order-mro-work-with-multiple-inheritance-and-what-is-c3-linearisation)
- **2.40** [What is a metaclass, and how does `type` fit in?](#240-what-is-a-metaclass-and-how-does-type-fit-in)
- **2.41** [How does `super()` actually resolve which method to call?](#241-how-does-super-actually-resolve-which-method-to-call)
- **2.42** [What's stored in a class's `__dict__` vs an instance's `__dict__`?](#242-whats-stored-in-a-classs-__dict__-vs-an-instances-__dict__)
- **2.43** [What is the difference between `@staticmethod`, `@classmethod`, and instance methods internally?](#243-what-is-the-difference-between-staticmethod-classmethod-and-instance-methods-internally)
- **2.44** [What's the difference between an iterable and an iterator at the protocol level (`__iter__`, `__next__`)?](#244-whats-the-difference-between-an-iterable-and-an-iterator-at-the-protocol-level-__iter__-__next__)
- **2.45** [How do generators work, how do they keep state between `yield` calls, and why do they matter for large data?](#245-how-do-generators-work-how-do-they-keep-state-between-yield-calls-and-why-do-they-matter-for-large-data)
- **2.46** [What happens internally when you call `.send()` on a generator?](#246-what-happens-internally-when-you-call-send-on-a-generator)
- **2.47** [How do `async def` and `await` differ from generators under the hood?](#247-how-do-async-def-and-await-differ-from-generators-under-the-hood)
- **2.48** [What is the event loop, and how does it schedule coroutines?](#248-what-is-the-event-loop-and-how-does-it-schedule-coroutines)
- **2.49** [How is a Python `list` implemented internally (array vs linked list)?](#249-how-is-a-python-list-implemented-internally-array-vs-linked-list)
- **2.50** [How does a `dict` achieve O(1) average lookup? How are hash collisions handled?](#250-how-does-a-dict-achieve-o1-average-lookup-how-are-hash-collisions-handled)
- **2.51** [Why did dict ordering become guaranteed in Python 3.7+? What changed internally?](#251-why-did-dict-ordering-become-guaranteed-in-python-37-what-changed-internally)
- **2.52** [How is a `set` different from a `dict` at the implementation level?](#252-how-is-a-set-different-from-a-dict-at-the-implementation-level)
- **2.53** [What happens internally when you run `import module`, and what is the role of `sys.modules`?](#253-what-happens-internally-when-you-run-import-module-and-what-is-the-role-of-sysmodules)
- **2.54** [How do `__init__.py` and packages affect the import system?](#254-how-do-__init__py-and-packages-affect-the-import-system)
- **2.55** [What's the difference between absolute and relative imports at resolution time?](#255-whats-the-difference-between-absolute-and-relative-imports-at-resolution-time)
- **2.56** [How does exception propagation work internally (the traceback stack)?](#256-how-does-exception-propagation-work-internally-the-traceback-stack)
- **2.57** [What's the difference between `except Exception` and a bare `except:`?](#257-whats-the-difference-between-except-exception-and-a-bare-except)
- **2.58** [How does `finally` interact with return statements and exceptions in bytecode?](#258-how-does-finally-interact-with-return-statements-and-exceptions-in-bytecode)
- **2.59** [Why is CPython considered slower than compiled languages, and where's the overhead?](#259-why-is-cpython-considered-slower-than-compiled-languages-and-wheres-the-overhead)
- **2.60** [What is the difference between CPython, PyPy, and other implementations?](#260-what-is-the-difference-between-cpython-pypy-and-other-implementations)
- **2.61** [How does the GIL affect the choice between threads, processes and asyncio for a data workload?](#261-how-does-the-gil-affect-the-choice-between-threads-processes-and-asyncio-for-a-data-workload)
- **2.62** [What are the differences between a list and a tuple, and when do you use each in data engineering?](#262-what-are-the-differences-between-a-list-and-a-tuple-and-when-do-you-use-each-in-data-engineering)
- **2.63** [What is the difference between a shallow copy and a deep copy?](#263-what-is-the-difference-between-a-shallow-copy-and-a-deep-copy)
- **2.64** [What do `*args` and `**kwargs` do in function definitions and calls?](#264-what-do-args-and-kwargs-do-in-function-definitions-and-calls)
- **2.65** [What are comprehensions, and how do they compare with `map()`, `filter()` and lambda functions?](#265-what-are-comprehensions-and-how-do-they-compare-with-map-filter-and-lambda-functions)
- **2.66** [What is the difference between the `/` and `//` operators, and what are the edge cases?](#266-what-is-the-difference-between-the--and--operators-and-what-are-the-edge-cases)
- **2.67** [How does the `with` statement work, and how do you write context managers?](#267-how-does-the-with-statement-work-and-how-do-you-write-context-managers)
- **2.68** [How do you profile and optimise Python code that processes millions of rows?](#268-how-do-you-profile-and-optimise-python-code-that-processes-millions-of-rows)
- **2.69** [How do you keep a Python codebase maintainable in a large team?](#269-how-do-you-keep-a-python-codebase-maintainable-in-a-large-team)
- **2.70** [What is Monkey Patching and why is it generally discouraged?](#270-what-is-monkey-patching-and-why-is-it-generally-discouraged)
- **2.71** [What does maintainable, well-tested Python look like in an AI or agent codebase, given that LLM outputs are non-deterministic?](#271-what-does-maintainable-well-tested-python-look-like-in-an-ai-or-agent-codebase-given-that-llm-outputs-are-non-deterministic)
- **2.72** [What is free-threaded (no-GIL) CPython, what is its status, and should you adopt it?](#272-what-is-free-threaded-no-gil-cpython-what-is-its-status-and-should-you-adopt-it)
- **2.73** [How do you use asyncio correctly in production: tasks, TaskGroup, cancellation, timeouts and mixing sync with async code?](#273-how-do-you-use-asyncio-correctly-in-production-tasks-taskgroup-cancellation-timeouts-and-mixing-sync-with-async-code)
- **2.74** [How do you manage Python packaging, dependencies and environments reproducibly (pyproject.toml, uv, lock files)?](#274-how-do-you-manage-python-packaging-dependencies-and-environments-reproducibly-pyprojecttoml-uv-lock-files)
- **2.75** [How do you use type hints at scale: mypy or Pyright, Protocols, generics, TypedDict, and runtime validation with Pydantic?](#275-how-do-you-use-type-hints-at-scale-mypy-or-pyright-protocols-generics-typeddict-and-runtime-validation-with-pydantic)
- **2.76** [How do you find and fix memory growth or leaks in a long-running Python service or worker?](#276-how-do-you-find-and-fix-memory-growth-or-leaks-in-a-long-running-python-service-or-worker)
- **2.77** [What is your testing strategy for Python code: pytest fixtures, parametrisation, mocking and property-based testing with Hypothesis?](#277-what-is-your-testing-strategy-for-python-code-pytest-fixtures-parametrisation-mocking-and-property-based-testing-with-hypothesis)

---

### 2.1 How do you change the case of a string in Python (capitalise, lowercase, case-insensitive comparison)?

Use the built-in `str` methods: `capitalize()`, `title()`, `upper()`, `lower()` and, for case-insensitive comparison, `casefold()`. Strings are immutable, so every method returns a new string.

```python
text = "hELLO wORLD"

text.capitalize()          # "Hello world"  - first char upper, REST lowered
text.title()               # "Hello World"  - each word capitalised
text.upper()               # "HELLO WORLD"
text.lower()               # "hello world"
text[:1].upper() + text[1:]  # "HELLO wORLD" - only the first char changed, safe on ""
```

**Pitfalls interviewers look for:**

- **`capitalize()` lowers the rest.** `"iPhone".capitalize()` gives `"Iphone"`. Use slicing if you only want to touch the first character, and use `text[:1]` rather than `text[0]` so an empty string does not raise `IndexError`.
- **`title()` is naive.** `"they're o'neil".title()` gives `"They'Re O'Neil"`. For names and headings use `string.capwords()` or a proper library.
- **Case-insensitive comparison should use `casefold()`, not `lower()`.** `casefold()` applies full Unicode case folding, so `"Straße".casefold() == "STRASSE".casefold()` is `True`, while `lower()` leaves `"ß"` alone.
- **Unicode normalisation.** `"é"` can be one code point or `e` plus a combining accent. For robust matching, normalise first: `unicodedata.normalize("NFKC", s).casefold()`.
- **Locale-specific rules** (Turkish dotted and dotless i) are not handled by `str` methods; use ICU (`PyICU`) when it matters.

**In data work**, prefer the vectorised versions: `df["city"].str.lower()` in pandas or `pl.col("city").str.to_lowercase()` in Polars, rather than `.apply(lambda s: s.lower())`. Normalising case and whitespace on join keys before a merge is a classic fix for "missing" rows.

### 2.2 How do you delete files and directories safely in Python?

Use `pathlib` for single files (`Path.unlink`) and `shutil.rmtree` for directory trees, and handle the "already gone" case explicitly so the code is idempotent.

```python
from pathlib import Path
import shutil

Path("output/part-0001.parquet").unlink(missing_ok=True)   # file; no error if absent (3.8+)
Path("output/empty_dir").rmdir()                         # only works if the directory is empty
shutil.rmtree("output/tmp_run_42", ignore_errors=False)   # whole tree

import os
os.remove("example.txt")                                 # older equivalent of unlink()
```

**Production concerns:**

- **Idempotency.** Re-runs of a pipeline step should not fail because a file was already deleted; `missing_ok=True` or catching `FileNotFoundError` gives that.
- **Guard rails on recursive deletes.** Resolve the path and check it is inside the directory you expect before calling `rmtree`. A bug that turns `base / run_id` into `base / ""` can wipe the whole base directory.

```python
base = Path("/data/scratch").resolve()
target = (base / run_id).resolve()
if base not in target.parents:
    raise ValueError(f"refusing to delete {target}")
shutil.rmtree(target)
```

- **Windows quirks.** Open files cannot be deleted, and read-only files make `rmtree` fail; pass an `onexc` handler (3.12+, `onerror` before) that clears the read-only bit and retries.
- **Temporary files.** Prefer `tempfile.TemporaryDirectory()` as a context manager so cleanup happens even when an exception is raised.
- **Atomic replacement.** To "delete and rewrite" a file, write to a temporary file in the same directory and `os.replace()` it over the target; readers never see a half-written file.
- **Object stores.** On S3, GCS or ADLS there is no `unlink`: use the SDK (`boto3`), `fsspec` or a table format's own `DELETE`/`VACUUM`. Deleting files underneath Delta Lake or Iceberg tables directly corrupts the table metadata.

### 2.3 What is serialization, how does pickling work, and what are its risks?

**Serialisation** converts an in-memory object into bytes or text that can be stored or sent over a network; **deserialisation** reconstructs the object. In Python, `pickle` is the native, Python-only binary format (pickling = object to byte stream, unpickling = byte stream to object), while JSON, Avro, Protobuf, Parquet and Arrow are language-neutral alternatives.

```python
import json, pickle

data = {"name": "John", "age": 25}
text = json.dumps(data)              # '{"name": "John", "age": 25}'
assert json.loads(text) == data

with open("data.pkl", "wb") as f:    # pickling
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("data.pkl", "rb") as f:    # unpickling - only trusted files
    restored = pickle.load(f)
```

**How pickle works:** it walks the object graph and emits opcodes for a small stack machine. For custom classes it records the import path of the class and the state (`__dict__` or what `__reduce__`/`__getstate__` returns). On load it imports that class and calls whatever callable `__reduce__` named.

**Risks:**

- **Arbitrary code execution.** Because `__reduce__` can name any callable (for example `os.system`), loading an attacker's pickle runs their code. Never unpickle untrusted data. This includes model files: `torch.load` uses pickle, which is why PyTorch 2.6 made `weights_only=True` the default and why Hugging Face pushes `safetensors`.
- **Version coupling.** A pickle stores class paths, not schemas. Renaming a module or changing a class breaks old pickles, and pickles made with a newer library version may not load in an older one.
- **Python-only.** Other languages cannot read it, so it is a poor contract between services.
- **Not a long-term storage format** for data; there is no schema evolution.

**Choosing a format:**

| Need | Format |
| --- | --- |
| Human-readable config or API payloads | JSON (with a Pydantic or JSON Schema contract) |
| High-throughput service messages with schema evolution | Protobuf or Avro |
| Analytical tables | Parquet (columnar, compressed, typed) |
| Zero-copy between processes or languages | Apache Arrow IPC |
| Model weights | `safetensors`, ONNX |
| Short-lived Python-to-Python transfer (multiprocessing, caches you control) | pickle, `cloudpickle` for lambdas and closures |

**What interviewers look for:** you know pickle is how `multiprocessing`, Spark (PySpark UDFs) and Dask move Python objects between processes, you would not accept a pickled model from an unknown source, and you pick a schema'd, language-neutral format for anything that crosses a team or service boundary.

### 2.4 How does exception handling work in Python, and how do you make it robust in a production ETL job?

Python uses `try`/`except`/`else`/`finally`: `try` holds code that may fail, `except` handles specific exception types, `else` runs only when no exception occurred, and `finally` always runs, which makes it the place for cleanup.

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    result = None
else:
    print("no error")          # skipped here
finally:
    print("always runs")
```

**Production ETL pattern:**

```python
import logging
logger = logging.getLogger(__name__)

class DataQualityError(Exception):
    """Raised when a batch fails validation; not retryable."""

def run_batch(batch_id: str) -> None:
    try:
        raw = extract(batch_id)                       # may raise transient IO errors
        clean = transform(raw)
        if clean.empty:
            raise DataQualityError(f"batch {batch_id} produced no rows")
        load(clean, batch_id)                          # idempotent: overwrite partition
    except (ConnectionError, TimeoutError):
        logger.warning("transient failure, will retry", extra={"batch_id": batch_id})
        raise                                          # let the orchestrator retry
    except DataQualityError:
        logger.error("quarantining batch", extra={"batch_id": batch_id})
        quarantine(batch_id)
        raise
    else:
        logger.info("batch loaded", extra={"batch_id": batch_id, "rows": len(clean)})
```

**Principles:**

- **Catch specific exceptions.** Catch what you can actually handle; let everything else propagate (see 2.57 on `except Exception` versus bare `except`).
- **Classify errors as transient or permanent.** Retry transient ones (network, throttling) with exponential backoff and jitter (see 25.10); fail fast on permanent ones (bad schema, failed validation) and quarantine the data.
- **Never swallow exceptions silently.** `except Exception: pass` turns a loud failure into silent data loss. If you must continue, log with `logger.exception(...)` so the traceback is kept, and emit a metric.
- **Preserve the cause.** Use `raise PipelineError("load failed") from exc` so the traceback shows both errors.
- **Clean up with context managers** (`with` for connections, transactions, temporary files) rather than hand-written `finally` blocks (see 2.67).
- **Make steps idempotent** so a retry after a partial failure does not duplicate data (see 7.14).
- **Group concurrent failures.** With `asyncio.TaskGroup` several tasks can fail at once; Python 3.11+ raises an `ExceptionGroup`, handled with `except*`.
- **Fail the job visibly.** Exit non-zero so the orchestrator marks the task failed and alerts, instead of logging an error and returning success.

### 2.5 What kind of language is Python (typing, case sensitivity, indentation)?

Python is a **high-level, general-purpose, dynamically and strongly typed language** with automatic memory management, supporting procedural, object-oriented and functional styles. The reference implementation, CPython, compiles source to bytecode and interprets it (see 2.18).

**Key characteristics:**

| Property | What it means | Example |
| --- | --- | --- |
| **Dynamically typed** | Types belong to objects, not variables; checked at run time | `x = 1; x = "a"` is legal |
| **Strongly typed** | No implicit coercion between unrelated types | `"1" + 1` raises `TypeError` |
| **Optionally statically checked** | Type hints are ignored at run time but checked by mypy or Pyright (see 2.75) | `def f(x: int) -> str: ...` |
| **Case-sensitive** | `name`, `Name` and `NAME` are three different identifiers; keywords are lowercase (`True`, `None` are capitalised) | `True` is valid, `true` is a `NameError` |
| **Significant indentation** | Blocks are defined by indentation, not braces; inconsistent indentation raises `IndentationError` or `TabError` | PEP 8 recommends 4 spaces |
| **Everything is an object** | Functions, classes and modules are first-class objects | functions can be passed to other functions |
| **Batteries included** | Large standard library plus the PyPI ecosystem | `pathlib`, `json`, `asyncio` |

```python
def classify(age: int) -> str:
    if age >= 18:          # block defined by indentation
        return "Adult"
    return "Minor"

name, Name = "Alice", "Bob"   # two different variables
```

**Why it dominates data and AI:** readable syntax, and a "glue language" design in which the heavy lifting is done by C, C++, Rust or CUDA libraries (NumPy, PyTorch, Polars, Arrow) while Python orchestrates. The trade-off is that pure-Python loops are slow (see 2.59), so senior engineers keep hot paths inside those libraries.

### 2.6 What are decorators, how do they work internally, and how are they used in production?

A decorator is a callable that takes a function (or class) and returns a replacement. `@decorator` above `def f` is syntactic sugar for `f = decorator(f)`, executed once when the `def` statement runs; the name `f` is rebound to whatever the decorator returns.

**What is actually replaced:** the original function object is not modified. The module namespace entry `f` now points to the wrapper, and the wrapper holds the original in a closure cell (see 2.31).

```python
import functools, logging, time

def timed(func):
    @functools.wraps(func)                 # copy __name__, __doc__, __wrapped__, ...
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logging.info("%s took %.3fs", func.__qualname__, time.perf_counter() - start)
    return wrapper

@timed
def add(a, b):
    return a + b
# equivalent to: add = timed(add)
```

**Decorators with arguments** need a third level: a factory that returns the decorator.

```python
def retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError), base_delay=1.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts:
                        raise
                    time.sleep(base_delay * 2 ** (attempt - 1))
        return wrapper
    return decorator

@retry(max_attempts=5)
def fetch_page(url): ...
# equivalent to: fetch_page = retry(max_attempts=5)(fetch_page)
```

**Stacking:** `@A` above `@B` above `def f` means `f = A(B(f))`; the decorator nearest the function is applied first, and the outermost runs first at call time. Order matters, for example a cache outside a retry caches only successes.

**Production uses:**

- Logging, timing, tracing spans and metrics (OpenTelemetry instrumentation is often decorators).
- Retry with backoff (`tenacity` is the standard library), rate limiting, circuit breaking.
- Caching: `functools.lru_cache` and `functools.cache`.
- Registration: Airflow `@task`, Prefect `@flow`, FastAPI `@app.get`, pytest `@pytest.fixture` register the function in a framework registry.
- Validation and access control: `pydantic.validate_call`, auth checks on API handlers.

**Pitfalls:**

- **Forgetting `functools.wraps`** loses the name and docstring, breaks introspection, and confuses frameworks that read signatures (FastAPI, pytest).
- **Retrying non-idempotent operations** (a payment, an append) duplicates side effects.
- **Async functions** need an `async def wrapper` that awaits the call; a sync wrapper returns an un-awaited coroutine.
- **`lru_cache` on methods** keeps `self` alive for the life of the cache, a common memory leak.
- **Typing:** use `ParamSpec` so type checkers keep the wrapped function's signature.
- **Import-time side effects:** decorator code runs at import, so heavy work there slows startup.

### 2.7 What are .pyc files, and how do bytecode caching and the compile-time optimiser improve performance?

A `.py` file is source code; a `.pyc` file is the **compiled bytecode** of a module, cached in `__pycache__/module.cpython-312.pyc` (the tag names the interpreter version) so that the next import can skip parsing and compiling.

**How the cache works:**

- On `import`, CPython looks for a `.pyc` whose header matches. By default it checks the source file's modification time and size; with PEP 552 (3.7+) it can instead store a hash of the source, which is useful for reproducible builds.
- If valid, the bytecode is loaded directly; otherwise the source is recompiled and the cache rewritten (if the directory is writable).
- The script you run directly (`python main.py`) is compiled every time and not cached; only imported modules are.
- `python -m compileall` precompiles a tree, which is common in Docker images so containers do not compile on first start. `PYTHONDONTWRITEBYTECODE=1` disables writing caches.

**Compile-time optimisation** (historically the "peephole optimiser", now mostly done on the AST and control-flow graph):

- Constant folding: `60 * 60 * 24` becomes `86400` in the bytecode.
- Converting constant lists in membership tests to tuples or frozensets: `x in [1, 2, 3]` becomes a constant tuple lookup.
- Removing unreachable code and simplifying jumps.

**What each one does and does not speed up:**

| Mechanism | Speeds up | Does not speed up |
| --- | --- | --- |
| `.pyc` caching | Import and startup time | Run-time execution; the same bytecode runs either way |
| Compile-time optimisation | Execution slightly, by removing instructions | Anything dominated by I/O or library calls |
| Specialising adaptive interpreter (3.11+) | Hot bytecode is rewritten at run time into type-specialised instructions | Startup |

**Pitfalls:** `.pyc` files are not protection for source code (they decompile easily) and are not portable across minor versions. Stale caches are rarely an issue because of the timestamp check, but committing `__pycache__` to git is a mistake; add it to `.gitignore`.

### 2.8 How does Python manage memory, and what does that mean in practice?

CPython manages memory automatically with **reference counting** as the primary mechanism, a **cyclic garbage collector** for reference cycles, and its own **small-object allocator** on top of the system allocator. Developers do not free memory by hand, but they do need to understand when memory is released and why a process's memory often does not shrink.

**The layers:**

| Layer | What it does |
| --- | --- |
| **Reference counting** | Each object stores a count of references; when it reaches zero the object is freed immediately (see 2.21) |
| **Cyclic GC** (`gc` module) | Periodically finds groups of container objects that only reference each other and frees them (see 2.22) |
| **pymalloc** | Allocator for small objects (512 bytes or less): carves arenas into pools of fixed-size blocks, which is much faster than `malloc` for millions of tiny objects |
| **System allocator** | Large objects and buffers (NumPy arrays, big strings) go to `malloc` directly; the free-threaded build uses `mimalloc` |
| **Private heap** | All Python objects live in a heap managed by the interpreter, not by user code |

**Practical consequences:**

- **Objects are big.** An `int` is about 28 bytes and a small `dict` hundreds; a list of a million Python floats uses roughly 32 MB versus 8 MB in a NumPy `float64` array. This is why data work keeps values in columnar buffers (NumPy, Arrow, Polars) rather than Python objects.
- **Memory is not always returned to the OS.** pymalloc can only release an arena when every block in it is free, so fragmentation keeps RSS high after a peak. Freed memory is reused by Python but the process does not shrink. Long-running workers are often recycled (`max_requests` in Gunicorn, `max_tasks_per_child` in multiprocessing and Celery) for this reason.
- **Deterministic cleanup is a CPython detail.** Refcounting frees files and sockets promptly in CPython, but PyPy and the free-threaded build behave differently; always use `with` for resources.
- **GC pauses** come from the cyclic collector, not refcounting. Services that build many container objects can tune `gc.set_threshold`, or call `gc.freeze()` after startup in pre-fork servers so the collector does not touch (and copy-on-write) shared pages.
- **Measuring:** `sys.getsizeof` is shallow; use `tracemalloc`, `memray` or `pympler` for real numbers (see 2.76).

**Interview answer:** CPython frees most objects the instant their reference count hits zero, uses a generational cyclic collector for cycles, and uses pymalloc for small objects. The things that matter in production are per-object overhead, fragmentation keeping RSS high, and lingering references (caches, globals, closures) that keep objects alive.

### 2.9 What are namespaces and scopes in Python, and how does the LEGB rule work internally?

A **namespace** is a mapping from names to objects; a **scope** is the region of code where a namespace is directly accessible. Python resolves an unqualified name using **LEGB: Local, Enclosing, Global, Built-in**, and the compiler decides at compile time which of these each name belongs to.

| Scope | What lives there | Implementation |
| --- | --- | --- |
| **Local** | Parameters and names assigned inside the function | Fast local slots in the frame, accessed with `LOAD_FAST` by index (not a dict) |
| **Enclosing** | Locals of outer functions used by a nested function | Cell objects, accessed with `LOAD_DEREF` (see 2.31) |
| **Global** | Module-level names | The module's `__dict__` (`func.__globals__`), accessed with `LOAD_GLOBAL` |
| **Built-in** | `len`, `print`, `ValueError` | `builtins` module dict, checked after globals |

```python
x = 10                    # global

def outer():
    y = 20                # local to outer, enclosing for inner
    def inner():
        z = 30            # local
        return x + y + z + len([1])   # G, E, L, B
    return inner()
```

**Key rules and gotchas:**

- **Assignment makes a name local for the whole function.** The compiler sees the assignment and marks the name local, so reading it earlier raises `UnboundLocalError`:

```python
count = 0
def increment():
    count += 1        # UnboundLocalError: count is local here
```

- **`global` and `nonlocal`** change that decision: `global count` targets the module namespace, `nonlocal count` the nearest enclosing function (see 2.32).
- **Class bodies are not enclosing scopes** for methods; methods reach class attributes through `self` or the class name. Comprehensions have their own scope (inlined for speed in 3.12 by PEP 709, but semantics unchanged).
- **Local access is faster** than global or built-in lookup because it is an array index, which is why hot loops sometimes bind globals to locals.
- **Local versus global variables:** locals exist only during the call; globals live as long as the module. Mutable global state makes code hard to test and unsafe under threads; prefer passing values explicitly or using objects.

Inspecting namespaces: `globals()`, `locals()`, `vars(obj)`, `func.__code__.co_varnames` (locals), `co_freevars` (captured), `co_names` (globals and attributes); see 2.29.

### 2.10 What are Python modules and packages, and which standard-library modules matter most?

A **module** is a single importable unit, usually a `.py` file (or a compiled extension such as `.so`); a **package** is a directory of modules, normally with an `__init__.py`, that creates a dotted namespace such as `myapp.io.readers`. A **distribution** is what you install from PyPI (a wheel), which may contain several packages; the install name and import name can differ (`pip install scikit-learn`, `import sklearn`).

```text
myapp/
  pyproject.toml
  src/myapp/
    __init__.py        # package marker and public API
    io/
      __init__.py
      readers.py       # module: myapp.io.readers
    transforms.py
  tests/
```

**Why the distinction matters:**

- Modules are executed once per process and cached in `sys.modules` (see 2.53); top-level code runs at import, so keep it cheap and side-effect free.
- Packages define the public API: re-export stable names in `__init__.py` and treat underscore-prefixed modules as private.
- The `src/` layout prevents tests accidentally importing the working directory instead of the installed package.
- Namespace packages (no `__init__.py`, PEP 420) let several distributions share a top-level name; see 2.54.
- Packaging and dependency management are covered in 2.74.

**Standard-library modules a senior engineer uses constantly:**

| Area | Modules |
| --- | --- |
| Files and OS | `pathlib`, `os`, `shutil`, `tempfile`, `subprocess` |
| Data formats | `json`, `csv`, `pickle`, `sqlite3`, `gzip`, `zipfile` |
| Data structures and functional tools | `collections` (`defaultdict`, `Counter`, `deque`), `itertools`, `functools`, `heapq`, `bisect`, `dataclasses`, `enum` |
| Concurrency | `concurrent.futures`, `asyncio`, `threading`, `multiprocessing`, `queue` |
| Typing | `typing`, `abc` |
| Observability and debugging | `logging`, `traceback`, `cProfile`, `tracemalloc`, `pdb` |
| Time | `datetime`, `zoneinfo`, `time` |
| Other | `re`, `math`, `random`, `statistics`, `hashlib`, `uuid`, `contextlib`, `argparse` |

### 2.11 What is PEP 8, and which tools enforce code quality and find bugs through static analysis?

**PEP 8** is the official style guide for Python code: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants, import ordering, whitespace rules and a recommended line length (79, though most teams configure 88 to 120). Its real value is that nobody argues about style in code review, because a formatter applies it automatically.

**The modern toolchain:**

| Purpose | Tools | Notes |
| --- | --- | --- |
| Formatting | **Ruff format**, Black | Deterministic; run on save and in CI |
| Linting | **Ruff** (replaces Flake8, isort, pyupgrade and many Pylint rules), Pylint | Ruff is written in Rust and fast enough to run on every commit |
| Type checking | **mypy**, **Pyright** (and Pylance in VS Code), newer Rust checkers such as Astral's ty and Meta's Pyrefly | Catch `None` errors, wrong argument types, missing attributes |
| Security | **Bandit** (or Ruff's `S` rules), `pip-audit`, Semgrep | Hard-coded secrets, `pickle.load`, shell injection, vulnerable dependencies |
| Dead code and complexity | `vulture`, Ruff's complexity rules, `radon` | Find unused code and over-complex functions |
| Notebooks | `nbqa`, Ruff supports `.ipynb` | Notebooks drift fastest |

**How to roll it out on a team:**

1. Put all configuration in `pyproject.toml` so editors, pre-commit and CI use the same rules.
2. Run fast checks in `pre-commit` and the full set in CI; CI is the source of truth.
3. On a legacy codebase, adopt a baseline: enable rules gradually, use per-file ignores, and type-check new modules strictly (`strict = true` per package) rather than blocking everything at once.
4. Treat warnings as errors in CI so the count only goes down.

**What interviewers look for:** you automate style rather than review it, you know type checkers find a different class of bugs from linters, and you can introduce tooling to an existing codebase without stopping delivery.

### 2.12 What is the difference between Python lists, `array.array` and NumPy arrays?

A **list** is a general-purpose dynamic array of pointers to arbitrary Python objects; an **`array.array`** stores primitive values of one C type compactly; a **NumPy `ndarray`** stores typed values in a contiguous buffer and adds vectorised maths, multiple dimensions and broadcasting. For numeric data work you almost always want NumPy (or Arrow, Polars, pandas built on top of it).

```python
from array import array
import numpy as np

items = [1, "Python", 3.5]                  # list: mixed types, pointers to objects
ints = array("i", [1, 2, 3])                # array: C ints, 4 bytes each
vec = np.arange(1_000_000, dtype=np.float64)

vec * 2.0                                   # vectorised, runs in C
[x * 2.0 for x in vec.tolist()]             # Python loop, typically 50-100x slower
```

| Feature | `list` | `array.array` | NumPy `ndarray` |
| --- | --- | --- | --- |
| Element types | Any, mixed | One C numeric type | One dtype (numeric, bool, datetime, fixed-width string, object) |
| Storage | Pointers to boxed objects | Raw values, contiguous | Raw values, contiguous, strided, N-dimensional |
| Memory for 1M floats | about 32 MB (8-byte pointer plus 24-byte float object) | about 8 MB | about 8 MB |
| Arithmetic | None built in (`+` concatenates) | None | Vectorised element-wise maths, broadcasting, linear algebra |
| Resizing | Cheap amortised `append` | Cheap `append` | Fixed size; appending copies the whole array |
| Typical use | General collections, heterogeneous records | Compact binary I/O, rare today | Numerical and ML work |

**Pitfalls:** building a NumPy array by repeated `np.append` is quadratic (collect in a list, convert once); `dtype=object` arrays lose all the speed benefits; and integer overflow in fixed-width dtypes wraps silently, unlike Python's arbitrary-precision `int`.

### 2.13 What is self in Python?

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

### 2.14 How do you do data abstraction in Python?

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

### 2.15 How do you introspect objects in Python with help(), dir() and related tools?

`help()` shows the documentation of an object (its docstring, signature and methods) and `dir()` lists the attribute names reachable on it. Together with `type()`, `vars()` and the `inspect` module they let you explore unfamiliar libraries and debug at a REPL.

```python
import inspect, pandas as pd

help(pd.DataFrame.merge)          # docstring and signature, paged
dir(pd.DataFrame)                 # all attribute names, including inherited ones
[n for n in dir(obj) if not n.startswith("_")]   # public API only

type(obj), isinstance(obj, pd.DataFrame)
vars(obj)                         # the instance __dict__ (fails with __slots__)
inspect.signature(pd.read_csv)    # parameters and defaults
inspect.getsource(my_function)    # source code
inspect.getmodule(obj).__file__   # where something is defined
```

| Tool | Returns | Use |
| --- | --- | --- |
| `help(x)` | Formatted documentation | Learn how to call something |
| `dir(x)` | Sorted list of attribute names (customisable with `__dir__`) | Discover what exists |
| `vars(x)` | The object's `__dict__` | See instance state |
| `type(x)`, `isinstance` | Class, subclass check | Debug type problems |
| `inspect` | Signatures, source, members, frames | Tooling, frameworks, debugging |

**Notes:** `dir()` on a module or object with `__getattr__` may not list dynamically provided attributes. In notebooks and IPython, `obj?` and `obj??` give the same information as `help` and source. Frameworks such as FastAPI and pytest rely on `inspect.signature` to wire dependencies, which is why decorators must use `functools.wraps`.

### 2.16 What does len() do, and how does it work under the hood?

`len(obj)` returns the number of items in a container. It works by calling the object's `__len__` method (at C level the `sq_length` or `mp_length` slot), so any class can support it.

```python
class Batch:
    def __init__(self, rows):
        self._rows = rows
    def __len__(self):
        return len(self._rows)

len("Python")       # 6 characters (code points, not bytes)
len([10, 20, 30])   # 3
len({"a": 1})       # 1 key
len(Batch([1, 2]))  # 2
```

**Points worth knowing:**

- **O(1) for built-ins.** Lists, tuples, dicts, sets and strings store their size, so `len` does not iterate.
- **Characters versus bytes.** `len("café")` is 4, `len("café".encode("utf-8"))` is 5. This matters for API payload limits and database column widths.
- **Generators and iterators have no length.** `len(x for x in data)` raises `TypeError`; count with `sum(1 for _ in it)`, which consumes the iterator.
- **Truthiness uses `__len__`.** If a class has no `__bool__`, `if obj:` is false when `len(obj) == 0`. Beware: `if df:` on a pandas DataFrame raises `ValueError` because truthiness is ambiguous; use `df.empty`.
- **`__len__` must return a non-negative `int`**, and it is limited to `sys.maxsize`; lazy or infinite collections should not define it.
- **Prefer `len(x) == 0` or `not x`** for emptiness; both are idiomatic.

### 2.17 How do access modifiers (public, protected, private) work in Python?

Python has **no enforced access modifiers**. It uses naming conventions: no prefix means public, a single underscore means internal, and a double underscore triggers **name mangling**, which avoids accidental clashes in subclasses but is not security.

```python
class Account:
    def __init__(self):
        self.owner = "Alice"      # public API
        self._balance = 100       # internal: "please don't touch", not enforced
        self.__pin = 1234         # mangled to _Account__pin

a = Account()
a._balance                       # works
a.__pin                          # AttributeError
a._Account__pin                  # works: mangling is only a rename
```

| Convention | Syntax | Enforcement | Intended use |
| --- | --- | --- | --- |
| Public | `name` | None | Stable API |
| Internal ("protected") | `_name` | None; excluded from `from module import *`; linters warn on external access | Implementation details, may change without notice |
| Name-mangled ("private") | `__name` (no trailing double underscore) | Compiler rewrites to `_ClassName__name` inside the class body | Avoid name collisions in inheritance hierarchies |
| Dunder | `__name__` | None; reserved for the language | Never invent your own |

**How to control access properly:**

- Use `@property` to expose read-only attributes or validate on assignment.
- Use `__slots__` or frozen dataclasses (`@dataclass(frozen=True)`) to prevent arbitrary attribute creation or mutation.
- Define the public surface of a module with `__all__` and underscore-prefixed modules.
- Real secrets do not belong in Python objects that untrusted code can reach; there is no sandbox inside one process.

**Interview point:** the philosophy is "we are all consenting adults". Conventions communicate intent, name mangling prevents accidental overrides, and type checkers plus linters enforce the boundary in teams.

### 2.18 Is Python an interpreted and a compiled language?

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

### 2.19 What is the GIL, why does CPython have it, and how does it affect data processing?

The **Global Interpreter Lock** is a mutex in CPython that allows only one thread at a time to execute Python bytecode within a process. It exists because CPython's memory management, especially reference counting, is not thread-safe: without a lock, two threads incrementing and decrementing the same object's refcount could corrupt it and free live objects.

**Why it was kept for 30 years:**

- One global lock is simpler and faster for single-threaded code than fine-grained locks on every object.
- It made writing C extensions easy, which is a large part of why NumPy, SciPy and the data ecosystem exist.
- Earlier attempts to remove it (Larry Hastings' "Gilectomy") slowed single-threaded code too much.

**How it behaves:**

- A running thread releases the GIL periodically (every 5 ms by default, `sys.setswitchinterval`) so others can run.
- **Blocking I/O releases the GIL**: socket reads, file reads, `time.sleep`, database drivers waiting on the network.
- **C extensions can release it** around heavy computation: NumPy vectorised operations, many pandas internals, Polars, PyTorch kernels, compression libraries, `hashlib` on large inputs.

**Impact on data workloads:**

| Workload | Threads under the GIL | Better choice |
| --- | --- | --- |
| Downloading files, calling APIs, querying databases | Scale well (I/O releases the GIL) | Threads or `asyncio` |
| Pure-Python row-by-row transforms | No speed-up; can be slower from contention | Vectorise, multiprocessing, or Spark |
| NumPy, Polars, PyTorch heavy kernels | Can scale, because the libraries release the GIL | Threads are fine; Polars already parallelises internally |
| CPU-bound Python in a web service | Blocks other requests | Offload to a process pool or worker queue |

**Current status:** the GIL is still the default, but PEP 703 introduced a free-threaded build (experimental in 3.13, officially supported but opt-in since 3.14); see 2.72. Python 3.12+ also offers per-interpreter GILs, exposed in 3.14 through `concurrent.interpreters` and `InterpreterPoolExecutor`.

**Common misconception:** the GIL does not make your code thread-safe. It protects the interpreter's internals, but `counter += 1` is several bytecodes and can still lose updates between threads; use locks or queues for shared state.

### 2.20 What's the difference between `is` and `==`? How does CPython implement identity checks?

| Aspect | `is` | `==` |
| --- | --- | --- |
| **Purpose** | Checks **object identity** — whether two references point to the same object. | Checks **value/equality** — whether two objects are considered equal. |
| **CPython implementation** | Essentially compares the two **object pointers/addresses**. Conceptually: `left_pointer == right_pointer`. | Calls the objects' **rich-comparison mechanism**, typically `tp_richcompare`, which may invoke `__eq__`. |
| **Does it call `__eq__`?** | **No.** | **Yes, potentially**, depending on the object's type and comparison implementation. |
| **Example** | `a is b` → `True` only when `a` and `b` reference the same object. | `a == b` → `True` when their values are equal according to the type's equality rules. |
| **Different objects, same value** | `a is b` → `False` | `a == b` → `True` |
| **Integer example** | `100 is 100` may be `True` because CPython commonly reuses small integer objects. | `100 == 100` is `True` because their values are equal. |
| **String example** | Two equal strings may sometimes be the same object due to interning, but this is not something to rely on. | Compares their string contents. |
| **`None`** | `x is None` is the recommended check because `None` is a singleton object. | `x == None` invokes equality semantics and is generally not preferred. |
| **Performance** | Usually very cheap because it is essentially an identity/pointer comparison. | Can be more expensive because it may execute type-specific comparison logic. |
| **Interview answer** | **`is` checks whether two references point to the exact same object; CPython implements this as an identity/pointer comparison.** | **`==` checks whether two objects have equal values using Python's equality protocol, potentially calling `__eq__`.** |

### 2.21 How does reference counting work in CPython? What happens when an object's refcount hits zero?

| Aspect | Explanation |
| --- | --- |
| **Reference counting** | CPython tracks how many references point to an object. The count is stored in the object's runtime header, conceptually as `ob_refcnt`. |
| **When a reference is created** | CPython increments the object's reference count, conceptually `Py_INCREF(obj)`. |
| **When a reference is removed** | CPython decrements the count, conceptually `Py_DECREF(obj)`. |
| **Example** | `a = obj` creates another reference to `obj`; `del a` removes that reference. |
| **Refcount = 0** | When an object's reference count reaches **zero**, CPython knows that the object is no longer referenced and can normally deallocate it immediately. |
| **Deallocation** | CPython invokes the object's deallocation mechanism (`tp_dealloc`), which releases the object's resources and eventually frees its memory when appropriate. |
| **Nested references** | During deallocation, an object's references to other objects are released, potentially causing their reference counts to reach zero as well. |
| **Reference cycles** | Reference counting alone cannot collect cycles such as `A → B → A`, because their counts never reach zero. CPython's **cyclic garbage collector** handles such unreachable cycles. |
| **Main advantage** | Objects that become unreachable without participating in cycles are generally reclaimed immediately rather than waiting for a separate GC pass. |
| **Interview answer** | **CPython maintains a reference count for objects. References increment the count and releasing references decrements it. When the count reaches zero, CPython normally deallocates the object immediately. Cyclic garbage collection is additionally used for unreachable reference cycles that reference counting cannot detect.** |

### 2.22 How does Python's garbage collector handle reference cycles?

| Aspect | Explanation |
| --- | --- |
| **Problem** | Reference counting alone cannot reclaim objects involved in a cycle because each object still has at least one reference from another object in the cycle. |
| **Example** | `a → b` and `b → a`. Even if no external references to `a` or `b` remain, their reference counts are not zero. |
| **How Python handles it** | CPython uses a **cyclic garbage collector (GC)** in addition to reference counting to detect and collect unreachable reference cycles. |
| **Detection** | The GC tracks container objects that can contain references to other objects, such as `list`, `dict`, and custom objects. It determines which groups of objects are unreachable from the program's roots. |
| **Generational approach** | CPython's cyclic GC groups tracked objects by generations and focuses collection work on younger objects more frequently. Python 3.14.0 briefly shipped an incremental collector to cut pause times, but 3.14.5 reverted to the generational design after reports of higher memory use. |
| **Collection** | Once a cycle is determined to be unreachable, the GC can break the cycle and allow the objects to be deallocated. |
| **`__del__` consideration** | Objects with finalizers (`__del__`) require special handling. Modern CPython can safely finalize many cyclic objects, but finalizers can still complicate collection. |
| **Reference counting vs GC** | **Reference counting** handles most objects immediately when their count reaches zero; **cyclic GC** handles unreachable cycles that reference counting alone cannot reclaim. |
| **Interview answer** | **CPython primarily uses reference counting, but reference cycles keep their counts above zero. Its cyclic garbage collector detects unreachable groups of container objects and collects those cycles, preventing them from leaking indefinitely.** |

### 2.23 What's the difference between `__new__` and `__init__`?

| Aspect | `__new__` | `__init__` |
| --- | --- | --- |
| **Main role** | Creates and returns a new object. | Initializes an already-created object. |
| **Called** | First. | After `__new__`. |
| **First argument** | `cls` — the class. | `self` — the instance. |
| **Return value** | Must return an object, normally an instance of `cls`. | Must return `None`. |
| **Object creation** | Yes. Controls whether/how an instance is created. | No. The instance already exists. |
| **Typical use** | Immutable types, singleton patterns, custom allocation/creation. | Setting instance attributes and validating initialization data. |
| **Example** | `def __new__(cls, x): return super().__new__(cls)` | `def __init__(self, x): self.x = x` |
| **Execution flow** | `Class(...) → __new__() → __init__()` | Runs only if `__new__()` returns a suitable instance of the class. |
| **Key interview point** | **`__new__` creates the object.** | **`__init__` initializes the object.** |

- **`__new__()`** → Creates and returns a new instance of the class.
- **`__init__()`** → Initializes the instance after it has been created.
- `__new__()` runs **before** `__init__()`.
- Most classes only need `__init__()`; overriding `__new__()` is uncommon.

```py
class User:
    def __new__(cls, name):
        print("Creating object")
        return super().__new__(cls)

    def __init__(self, name):
        print("Initializing object")
        self.name = name
```

**Typical use of `__new__()`:** controlling object creation, implementing certain singleton patterns, or subclassing immutable types such as `tuple` or `str`.

### 2.24 Why is small integer caching (-5 to 256) used, and how does it affect `is` comparisons?

| Aspect | Explanation |
| --- | --- |
| **Small integer caching** | CPython pre-creates and reuses integer objects for commonly used small integers, traditionally **`-5` through `256`**. |
| **Why used** | Small integers occur extremely frequently in Python programs (`0`, `1`, `2`, loop counters, indexes, Boolean-related operations, etc.). Reusing them avoids repeatedly allocating and deallocating integer objects. |
| **Memory benefit** | One shared integer object can be referenced by many variables instead of creating a separate object each time. |
| **Performance benefit** | Reduces object allocation/deallocation overhead and can improve performance for common integer operations. |
| **Effect on `is`** | Two variables containing the same cached integer may reference the **same object**, so `is` can unexpectedly return `True`. |
| **Example** | `a = 100; b = 100; a is b` may return `True` because `100` is commonly cached. |
| **Outside the cached range** | `a = 1000; b = 1000; a is b` is **not guaranteed** to be `False` either. Compiler optimizations or constant reuse can sometimes make it `True`. |
| **Correct comparison** | Use **`==`** to compare integer values. Use **`is`** only when checking object identity, such as `x is None`. |
| **Important nuance** | The `-5` to `256` range is a **CPython implementation detail**, not a general Python language guarantee. |
| **Interview answer** | **CPython caches commonly used small integer objects to reduce allocation and memory overhead. As a result, two variables containing the same small integer may have the same identity, making `is` return `True`. However, `is` should never be used for numeric value comparison; use `==`.** |

### 2.25 What is object interning, how does string immutability enable it, and when would you use sys.intern()?

**Interning** means keeping one canonical object for each distinct immutable value and reusing it instead of creating duplicates. It is safe only because the objects are **immutable**: if strings could be changed in place, modifying one reference would silently change every other variable sharing it.

**What CPython interns automatically:**

- Identifier-like strings used in code: variable, function, attribute and keyword-argument names, and dictionary keys used for attribute lookup.
- Many short string literals that look like identifiers (letters, digits and underscores) at compile time.
- Small integers from -5 to 256 are a related cache (see 2.24).
- Strings built at run time (read from files, concatenated, decoded from JSON) are generally **not** interned.

**Why it helps:**

- **Memory:** a million rows with the same `"status"` value can share one object.
- **Speed:** dict lookups first compare by identity, so interned keys skip the character-by-character comparison. Strings also cache their hash, another benefit of immutability.

**Manual interning with `sys.intern()`:**

```python
import sys

a = sys.intern("customer_id")
b = sys.intern("".join(["customer_", "id"]))   # built at run time
assert a is b                                    # same object

# parsing millions of records with a small set of repeated keys or labels
rows = [{sys.intern(k): v for k, v in rec.items()} for rec in records]
```

Use it when a program holds **many copies of a small vocabulary** of strings: field names in parsed JSON, tokens, categorical labels, symbol tables. Do not intern high-cardinality values (IDs, free text): you gain nothing and grow the intern table.

**Rules:**

- Never use `is` to compare strings; use `==`. Interning is an implementation detail and `is` results vary between versions and contexts.
- In data work the same idea appears at a higher level and is usually the better tool: pandas `category` dtype, Polars `Categorical`, Arrow dictionary encoding and Parquet dictionary pages all store each distinct string once.

### 2.26 How does `sys.getrefcount()` work, and why is its result often higher than expected?

| Aspect | Explanation |
| --- | --- |
| **What it does** | `sys.getrefcount(obj)` returns the number of references currently held to `obj` in CPython. |
| **C-level basis** | CPython maintains a reference count in the object's `PyObject` header, conceptually `ob_refcnt`. |
| **Why the value is higher than expected** | Calling `sys.getrefcount(obj)` temporarily creates **another reference** to `obj` because the object is passed as an argument. |
| **Example** | If you believe `x` has 1 reference, `sys.getrefcount(x)` will commonly report **2**: one from `x` and one temporary reference used for the function call. |
| **Other references** | Additional references may come from containers, temporary expressions, interpreter internals, function arguments, debugger/REPL variables, etc. |
| **Typical example** | `x = []` → `sys.getrefcount(x)` commonly returns `2`, not `1`. |
| **Important caveat** | The exact count can vary depending on the Python implementation, execution context, and temporary references. `getrefcount()` is primarily useful for CPython diagnostics, not application logic. |
| **Interview answer** | **`sys.getrefcount()` reads CPython's reference count for an object. Its result is often one higher than expected because passing the object to `getrefcount()` itself temporarily adds a reference. Other temporary or hidden references can increase it further.** |

### 2.27 What's the difference between mutable and immutable objects at the C level?

| Aspect | Mutable Objects | Immutable Objects |
| --- | --- | --- |
| **Meaning** | Object's internal state can be changed after creation. | Object's state cannot be changed after creation. |
| **C-level representation** | C structure typically contains fields/buffers whose contents can be modified in place. | C structure represents a fixed value; operations that appear to modify it create/return another object. |
| **Memory address** | Usually remains the same while its contents change. | The existing object's value cannot change; a new object may be created for a different value. |
| **Example** | `list`, `dict`, `set`, `bytearray` | `int`, `float`, `str`, `tuple`, `bytes` |
| **Example internally** | `list.append()` modifies the list's underlying array of `PyObject*` references. | `int` arithmetic creates a new integer object rather than changing the existing integer's value. |
| **Reference behavior** | Multiple references see the same object's changed state. | Multiple references continue to see the same unchanged value. |
| **Hashing** | Generally not safely hashable because mutable state could change after insertion into a hash table. | Often hashable because the value remains stable, allowing a stable hash. |
| **C-level key point** | **Mutability is about whether the object's C-level state can be modified in place.** | **Immutability means the object's represented value is fixed after creation; changing the value requires another object.** |
| **Important nuance** | Immutability is a property of the Python object's semantics/type, not simply whether its C struct contains writable memory. | An immutable object can still have internal implementation details or cached data that change without changing its logical Python value. |

### 2.28 What is `__slots__`, and how does it change an object's memory layout?

| Aspect | Explanation |
| --- | --- |
| **`__slots__`** | A class attribute that explicitly declares which instance attributes are allowed, e.g. `class User: __slots__ = ("name", "age")`. |
| **Main purpose** | Reduces per-instance memory usage and can make attribute access more memory-efficient by avoiding the normal instance `__dict__`. |
| **Normal object layout** | A typical Python object has an object header plus an **instance `__dict__`**, where attributes such as `name` and `age` are stored in a dictionary. |
| **With `__slots__`** | The instance normally has **no `__dict__`**. Instead, CPython creates slot descriptors and stores each declared attribute in a fixed slot associated with the object. |
| **Example** | `class User: __slots__ = ("name", "age")` allows `u.name` and `u.age`, but `u.email = "x"` raises `AttributeError`. |
| **Memory effect** | Avoiding a separate per-instance dictionary can save significant memory when creating many objects with the same fixed attributes. |
| **Attribute lookup** | Slot attributes are accessed through descriptors rather than dictionary lookup. This can also improve attribute-access efficiency in some cases. |
| **Dynamic attributes** | Not allowed unless `"__dict__"` is explicitly included in `__slots__`. |
| **Weak references** | Objects normally cannot have weak references unless `"__weakref__"` is included when needed. |
| **Inheritance** | `__slots__` interacts with inheritance; subclasses can define additional slots, and a subclass may still have a `__dict__` depending on its definition and base classes. |
| **Interview answer** | **`__slots__` replaces the usual per-instance attribute dictionary with predefined storage slots. This gives objects a more fixed memory layout, reduces memory overhead, and restricts dynamically added attributes.** |

### 2.29 What's the difference between a function's `__globals__` and `__locals__`?

| Aspect | `__globals__` | `__locals__` |
| --- | --- | --- |
| **What it represents** | The **global namespace** associated with a function | There is **no standard `function.__locals__` attribute** |
| **Access** | `func.__globals__` | Use `locals()` inside the executing function or inspect a frame's `f_locals` |
| **Scope** | Module where the function was defined | Current function's local execution scope |
| **Lifetime** | Usually the module dictionary exists as long as the module is loaded | Local variables normally exist for the duration of the function's execution |
| **Contains** | Module-level variables, imported names, functions, classes, etc. | Function parameters and local variables |
| **Used by function** | Global name resolution (`LOAD_GLOBAL`) | Local name resolution (`LOAD_FAST` in typical optimized functions) |

#### Example

```py
x = 10

def foo(a):
    y = 20
    return x + a + y
```

| Expression | Result / meaning |
| --- | --- |
| `foo.__globals__` | Dictionary for the module where `foo` was defined |
| `foo.__globals__["x"]` | `10` |
| `foo.__code__.co_varnames` | Typically `('a', 'y')` |
| `locals()` inside `foo` | Contains `a` and `y` while executing |
| `foo.__code__.co_names` | Includes names such as `x` used by the code |

#### Important distinction

| Scope | Example | Where Python looks |
| --- | --- | --- |
| **Local** | `a`, `y` | Current function's local scope |
| **Enclosing** | Variable captured by nested function | Closure **cell objects** |
| **Global** | `x` | `foo.__globals__` |
| **Built-in** | `len`, `print` | Built-in namespace |

This is the **LEGB** lookup model:

| Order | Scope |
| --- | --- |
| **L** | Local |
| **E** | Enclosing |
| **G** | Global → `func.__globals__` |
| **B** | Built-ins |

### 2.30 How do closures capture variables — by reference or by value?

| Aspect | Closure behavior |
| --- | --- |
| **Capture mechanism** | Closures capture **variables through references to cell objects**, not by copying their values at function creation time. |
| **What is captured?** | The closure stores a reference to the **cell** containing the variable. |
| **Value changes?** | If the variable changes, the closure sees the updated value because it refers to the same cell. |
| **`nonlocal`** | Allows the closure to modify the value stored in that shared cell. |
| **Important distinction** | Python's closure behavior is **late binding**: the variable's value is looked up when the closure executes. |

#### Example

```py
def outer():
    x = 10

    def inner():
        return x

    x = 20
    return inner

f = outer()
f()  # 20
```

| Step | What happens |
| --- | --- |
| `x = 10` | Cell for `x` contains `10` |
| `inner()` created | `inner` captures the **cell**, not a copy of `10` |
| `x = 20` | Same cell now contains `20` |
| `f()` | `inner` reads the cell → `20` |

#### Classic late-binding example

```py
funcs = []

for i in range(3):
    funcs.append(lambda: i)
```

| Expression | Result |
| --- | --- |
| `funcs[0]()` | `2` |
| `funcs[1]()` | `2` |
| `funcs[2]()` | `2` |

All three closures refer to the **same `i` cell**, whose final value is `2`.

#### Capturing by value with a default argument

```py
funcs = []

for i in range(3):
    funcs.append(lambda i=i: i)
```

| Expression | Result |
| --- | --- |
| `funcs[0]()` | `0` |
| `funcs[1]()` | `1` |
| `funcs[2]()` | `2` |

Here, the default argument stores the current value separately.

### 2.31 What is a cell object, and how does it enable closures?

| Aspect | Cell object |
| --- | --- |
| **What is it?** | A small runtime object used by Python to hold a variable that must remain accessible to a nested function. |
| **Why needed?** | The outer function may finish, but the nested function can still need access to its local variable. |
| **Created when** | The compiler determines that an outer local variable is referenced by a nested function. |
| **Contains** | A reference to the actual value, not a copy of the value. |
| **Accessed by** | The inner function through its closure. |
| **Inspect it** | `func.__closure__` contains the closure cells. |
| **Variable names** | `func.__code__.co_freevars` contains the names associated with those cells. |

#### Example

```py
def outer():
    x = 10

    def inner():
        return x

    return inner

f = outer()
```

| Step | What happens |
| --- | --- |
| `x = 10` | `x` is a local variable of `outer`. |
| `inner` references `x` | Compiler determines `x` must survive for `inner`. |
| `x` becomes a **cell variable** | Instead of being only a normal local slot, it is stored in a cell. |
| `inner` becomes a closure | `inner` keeps a reference to that cell. |
| `outer()` finishes | `outer`'s normal execution ends, but the cell remains alive because `inner` references it. |
| `f()` | `inner` reads the value from the cell and returns `10`. |

#### Closure relationship

```txt
outer
 └── x
      ↓
   Cell(10)
      ↑
      │
   inner.__closure__
```

| Object | Example |
| --- | --- |
| `f.__code__.co_freevars` | `('x',)` |
| `f.__closure__` | Tuple containing the cell for `x` |
| `f.__closure__[0].cell_contents` | `10` |

#### With `nonlocal`

```py
def counter():
    count = 0

    def increment():
        nonlocal count
        count += 1
        return count

    return increment
```

| Operation | Cell behavior |
| --- | --- |
| `count = 0` | Cell contains `0` |
| `increment()` | Reads the cell |
| `count += 1` | Updates the value referenced by the cell |
| Next `increment()` | Reads the **same cell**, now containing `1` |
| Result | `1`, then `2`, then `3`, ... |

#### Why a cell is necessary

| Without a cell | With a cell |
| --- | --- |
| Inner function would lose access to the outer local after `outer()` returns | Inner function retains access |
| Variable would be tied only to the outer execution context | Variable lives as long as the closure needs it |
| Assignment in inner function creates a local by default | `nonlocal` allows modification of the shared cell |

### 2.32 How does the `nonlocal` keyword work under the hood?

| Aspect | `nonlocal` behavior |
| --- | --- |
| **Purpose** | Allows a nested function to **modify a variable in an enclosing function's scope**. |
| **Scope targeted** | The nearest enclosing **function scope**, excluding the current local scope and global scope. |
| **Without `nonlocal`** | Assignment such as `x = ...` inside the nested function creates a **new local variable**. |
| **With `nonlocal`** | Assignment updates the existing variable from an enclosing function scope. |
| **Under the hood** | Python implements this using **closure cells**. |
| **Storage** | The enclosing variable is stored in a **cell object** so both functions can access the same variable. |
| **Compiler role** | The compiler marks the variable as a **cell variable** in the outer function and a **free variable** in the inner function. |
| **Bytecode access** | The inner function accesses it using closure-related operations such as `LOAD_DEREF` / `STORE_DEREF`. |

#### Example

```py
def counter():
    count = 0

    def increment():
        nonlocal count
        count += 1
        return count

    return increment
```

| Stage | What happens |
| --- | --- |
| `count = 0` | `count` becomes a local variable of `counter`. |
| `nonlocal count` | Compiler knows `increment` refers to the enclosing `count`. |
| `counter()` | A **cell** is created to hold `count`. |
| `increment()` | The inner function captures that cell in its closure. |
| `count += 1` | The value in the shared cell is read and updated. |
| Multiple calls | All calls to `increment` access the **same cell**, so state persists. |

#### Closure structure

```txt
counter()
   │
   └── count → Cell(0)
                 ↑
                 │
          increment.__closure__
```

| Object | Relevant attribute |
| --- | --- |
| `increment.__code__.co_freevars` | Contains names captured from the enclosing scope, e.g. `('count',)` |
| `increment.__closure__` | Contains the corresponding closure cell |
| Cell | Holds the actual referenced value |

#### `nonlocal` vs `global`

| Keyword | Searches/updates |
| --- | --- |
| Normal assignment | Current local scope |
| `nonlocal x` | Nearest enclosing function scope |
| `global x` | Module/global scope |

### 2.33 What is a code object (`__code__`), and what does it contain?

| Aspect | `__code__` / Code Object |
| --- | --- |
| **What is it?** | An immutable object containing the **compiled executable representation** of a Python function/code block. |
| **Created when?** | When Python compiles the source code, such as during a `def` statement. |
| **Access** | `func.__code__` |
| **Contains bytecode?** | Yes — `co_code` contains the bytecode representation. |
| **Contains function object?** | No. The function object wraps/references the code object and also stores things like globals, defaults, and closure information. |
| **Mutable?** | Code objects are effectively immutable. |
| **Executed by** | CPython's interpreter executes the code represented by the code object. |

#### Important attributes

| Attribute | Contains |
| --- | --- |
| `co_code` | Compiled bytecode |
| `co_consts` | Constants used by the code |
| `co_names` | Names used for global/module/class attribute operations |
| `co_varnames` | Local variable and argument names |
| `co_freevars` | Names captured from enclosing scopes |
| `co_cellvars` | Local variables made available to nested functions |
| `co_argcount` | Number of positional parameters |
| `co_kwonlyargcount` | Number of keyword-only parameters |
| `co_filename` | Source filename associated with the code |
| `co_name` | Name associated with the code object |
| `co_firstlineno` | Starting source line |
| `co_exceptiontable` | Exception-handling metadata in modern CPython |
| `co_flags` | Flags describing properties of the code |

#### Function vs Code Object

```py
def add(a, b):
    return a + b
```

| Object | Role |
| --- | --- |
| `add` | **Function object** |
| `add.__code__` | **Code object** |
| `add.__code__.co_code` | Compiled **bytecode** |
| `add.__globals__` | Global namespace used by the function |
| `add.__defaults__` | Default argument values |

Conceptually:

```txt
Function object
├── __code__       → Code object
├── __globals__    → Global namespace
├── __defaults__   → Default arguments
├── __closure__    → Closure cells
└── other metadata
```

### 2.34 How does Python compile source code into bytecode? What role does `dis` play?

| Stage | What Python does |
| --- | --- |
| **1. Source code** | Python reads `.py` source code as text. |
| **2. Tokenization / parsing** | Source is tokenized and parsed according to Python's grammar. |
| **3. AST creation** | Python builds an **Abstract Syntax Tree (AST)** representing the program's structure. |
| **4. AST optimization** | CPython performs compiler-level optimizations such as constant folding where applicable. |
| **5. Bytecode compilation** | The compiler converts the AST into **bytecode instructions** for the Python virtual machine. |
| **6. Code object** | Bytecode and related metadata are stored in a **code object** (`types.CodeType`). |
| **7. Execution** | CPython's interpreter executes the code object's bytecode. |
| **8. `.pyc` caching** | For imported modules, CPython can cache compiled bytecode in `__pycache__/*.pyc` to avoid recompiling unchanged source on future imports. |

#### Simplified pipeline

| Input | → | Output |
| --- | --- | --- |
| `program.py` | Tokenize → Parse → AST → Compile | **Code object / bytecode** |
| Bytecode | → | CPython interpreter executes it |

#### Example

```py
def add(a, b):
    return a + b
```

| Object | What it contains |
| --- | --- |
| `add` | Function object |
| `add.__code__` | Code object |
| `add.__code__.co_code` | Compiled bytecode bytes |
| `add.__code__.co_consts` | Constants used by the code |
| `add.__code__.co_varnames` | Local/argument variable names |

#### What `dis` does

| `dis` feature | Purpose |
| --- | --- |
| `dis.dis(func)` | Disassembles a function's bytecode into human-readable instructions |
| `dis.dis(code)` | Displays bytecode for a code object |
| `dis.dis(module)` | Can disassemble code associated with a module |
| Main purpose | **Inspection/debugging/learning**, not compilation |

Example:

```py
import dis

def add(a, b):
    return a + b

dis.dis(add)
```

You may see instructions such as:

```txt
LOAD_FAST
LOAD_FAST
BINARY_OP
RETURN_VALUE
```

| Important distinction | Meaning |
| --- | --- |
| **Compiler** | Converts Python source/AST into bytecode |
| **Code object** | Stores the compiled representation |
| **CPython VM** | Executes the bytecode |
| **`dis`** | Shows/disassembles that bytecode for humans |
| **`.pyc`** | Persistent cached representation of compiled module bytecode |

### 2.35 What's the difference between a function and a method at the descriptor level?

| Aspect | **Function** | **Method** |
| --- | --- | --- |
| **Definition** | A function object stored anywhere, including a class | A function when accessed through an object/class and **bound by the descriptor protocol** |
| **Stored in class** | A normal function can be stored in `Class.__dict__` | Instance methods are normally functions stored in `Class.__dict__` |
| **Descriptor** | Python function objects implement `__get__()` | `__get__()` creates the bound method when accessed through an instance |
| **Access** | `func()` | `obj.method()` |
| **Automatic `self`** | No | Yes, for instance methods |
| **Binding** | No automatic binding when called directly | Function is bound to the instance |
| **Example** | `A.f(obj)` | `obj.f()` |
| **Internal equivalence** | `A.f(obj)` | `obj.f()` is roughly equivalent to `A.f.__get__(obj, A)()` |

#### What actually happens

```py
class A:
    def f(self, x):
        return x

a = A()
```

| Expression | Descriptor behavior |
| --- | --- |
| `A.__dict__["f"]` | Returns the **raw function object** |
| `A.f` | Function's `__get__(None, A)` → function itself |
| `a.f` | Function's `__get__(a, A)` → **bound method** |
| `a.f(10)` | Bound method supplies `a` as `self` |
| `A.f(a, 10)` | Calls the raw function explicitly with `a` |

#### Simplified descriptor model

```py
class Function:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return BoundMethod(self, instance)
```

| Access | Result |
| --- | --- |
| `A.f` | Original function |
| `a.f` | Bound method containing approximately `(function, a)` |
| `a.f()` | Calls the function with `a` automatically |

#### Key distinction

| Term | Meaning |
| --- | --- |
| **Function** | The underlying callable object. |
| **Method** | A function that has been **bound to an instance or class** through descriptor behavior. |
| **Important** | A function defined inside a class is still a **function when stored in `Class.__dict__`**. It becomes a bound method when accessed through an instance. |

### 2.36 How does the descriptor protocol (`__get__`, `__set__`, `__delete__`) work, and what is the difference between data and non-data descriptors?

A **descriptor** is an object stored as a **class attribute** whose type defines `__get__`, `__set__` or `__delete__`. When you access that attribute through an instance, Python calls those methods instead of returning the object itself. Descriptors are the mechanism behind methods, `property`, `classmethod`, `staticmethod`, `__slots__` and ORM fields (SQLAlchemy, Django).

```python
class Positive:
    """A reusable validated attribute."""
    def __set_name__(self, owner, name):          # 3.6+: learn the attribute name
        self.private = "_" + name

    def __get__(self, instance, owner):
        if instance is None:                       # accessed on the class
            return self
        return getattr(instance, self.private)

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError(f"{self.private[1:]} must be positive")
        setattr(instance, self.private, value)

class Order:
    quantity = Positive()
    price = Positive()

    def __init__(self, quantity, price):
        self.quantity = quantity                   # calls Positive.__set__
        self.price = price

o = Order(3, 9.99)
o.quantity                                         # Positive.__get__(o, Order)
```

| Operation | Conceptual call |
| --- | --- |
| `o.quantity` | `type(o).__dict__["quantity"].__get__(o, type(o))` |
| `o.quantity = 5` | `type(o).__dict__["quantity"].__set__(o, 5)` |
| `del o.quantity` | `type(o).__dict__["quantity"].__delete__(o)` |
| `Order.quantity` | `__get__(None, Order)` |

**Data versus non-data descriptors (the key senior point):**

- A **data descriptor** defines `__set__` or `__delete__` (for example `property`, slots). It **takes precedence over the instance `__dict__`**.
- A **non-data descriptor** defines only `__get__` (for example plain functions, `classmethod`). The **instance `__dict__` wins**, which is why you can shadow a method by assigning `obj.method = something`, and why `functools.cached_property` works: the first access stores the value in the instance dict and later accesses never call `__get__` again.

**Attribute lookup order for `obj.attr`** (implemented in `object.__getattribute__`):

1. Find `attr` on the type's MRO. If it is a data descriptor, call its `__get__`.
2. Otherwise, if `attr` is in `obj.__dict__`, return it.
3. Otherwise, if the type attribute is a non-data descriptor, call its `__get__`; if it is a plain value, return it.
4. Otherwise call `__getattr__` if defined, else raise `AttributeError`.

**When to write your own:** reusable validation or type coercion across many attributes, lazy loading, unit conversions. For one-off cases, `@property` is clearer. Note that descriptors must live on the class; putting one in an instance dict does nothing.

### 2.37 How are default mutable arguments evaluated, and why does this cause bugs?

| Aspect | Explanation |
| --- | --- |
| **When defaults are evaluated** | Function default arguments are evaluated **once, when the `def` statement executes**, not every time the function is called. |
| **Where stored** | Defaults are stored in the function object's `__defaults__` tuple. |
| **Mutable default** | A list, dict, set, or other mutable object created as a default is therefore **shared across calls**. |
| **Why this causes bugs** | If one call modifies the object, later calls see the modified object. |
| **Common example** | `def add_item(item, items=[]): items.append(item)` |
| **First call** | Creates the default list once and adds the item. |
| **Later calls** | Reuse the **same list**, rather than creating a new list. |

#### Example

```py
def add_item(item, items=[]):
    items.append(item)
    return items

add_item("A")  # ["A"]
add_item("B")  # ["A", "B"]
add_item("C")  # ["A", "B", "C"]
```

| Call | What happens |
| --- | --- |
| Function definition | `[]` is created once |
| `add_item("A")` | Shared list becomes `["A"]` |
| `add_item("B")` | Same list becomes `["A", "B"]` |
| `add_item("C")` | Same list becomes `["A", "B", "C"]` |

#### Why?

```py
def add_item(item, items=[]):
    ...
```

is conceptually similar to:

```py
_default_items = []

def add_item(item, items=_default_items):
    ...
```

The important point is that `[]` isn't recreated for every call.

#### Correct pattern

```py
def add_item(item, items=None):
    if items is None:
        items = []

    items.append(item)
    return items
```

| Pattern | Behavior |
| --- | --- |
| `items=[]` | One shared list across calls |
| `items=None` | New list created when needed |
| `items=[]` for intentional shared state | Valid, but should be deliberate |

### 2.38 What is `__call__`, and how does it make objects callable?

| Aspect | Explanation |
| --- | --- |
| **`__call__`** | A special method that defines what happens when an **object is called like a function**. |
| **Syntax** | `obj()` |
| **How Python handles it** | Python looks for the object's `__call__()` method and invokes it. |
| **Callable object** | An object whose type provides `__call__`. |
| **Check callability** | `callable(obj)` returns `True` when the object can be called. |
| **Common uses** | Function-like objects, decorators, classes used as factories, stateful callable objects, ML model objects. |

#### Example

```py
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)

double(5)        # 10
callable(double) # True
```

| Operation | Internally |
| --- | --- |
| `double(5)` | Python performs a call operation on `double` |
| `double`'s type | `Multiplier` |
| `Multiplier` provides | `__call__()` |
| Result | `Multiplier.__call__(double, 5)` |

#### Function objects are callable too

| Object | Why callable? |
| --- | --- |
| `function` | Its type supports the call protocol |
| `class` | Its metaclass (`type`) provides the mechanism for creating instances |
| `Multiplier()` instance | Its class defines `__call__()` |
| Generator object | **Not generally callable** |

#### `__call__` vs `__init__`

| Method | Purpose |
| --- | --- |
| `__new__()` | Creates the instance |
| `__init__()` | Initializes the instance |
| `__call__()` | Defines behavior when the **already-created object** is called |

```py
obj = Multiplier(2)  # creation/initialization

obj(5)               # calls __call__()
```

### 2.39 How does the Method Resolution Order (MRO) work with multiple inheritance, and what is C3 linearisation?

The **MRO** is the ordered list of classes Python searches when looking up an attribute on an object. With multiple inheritance Python builds it using **C3 linearisation**, which guarantees that a class appears before its parents, that the left-to-right order of bases is respected, and that a shared ancestor in a diamond is visited only once. Inspect it with `Class.__mro__` or `Class.mro()`.

```python
class A:
    def show(self): return "A"
class B(A):
    def show(self): return "B"
class C(A):
    def show(self): return "C"
class D(B, C):
    pass

D.__mro__     # (D, B, C, A, object)
D().show()    # "B": the first class in the MRO that defines show
```

**How C3 works:** the MRO of a class is the class itself followed by a merge of its parents' MROs and the list of parents. The merge repeatedly takes the first head of a list that does not appear in the tail of any other list.

```text
L[B] = B, A, object
L[C] = C, A, object
L[D] = D + merge(L[B], L[C], [B, C])
     = D, B, C, A, object     # A is postponed until both B and C are done
```

**Properties that matter:**

- **Local precedence order:** in `class D(B, C)`, `B` is always searched before `C`.
- **Monotonicity:** a class's MRO is consistent with its parents' MROs.
- **Inconsistent hierarchies are rejected** at class creation with `TypeError: Cannot create a consistent method resolution order`, for example `class X(A, B)` together with `class Y(B, A)` and `class Z(X, Y)`.

**Attribute lookup in practice:**

- `obj.attr` checks data descriptors on the MRO, then the instance `__dict__`, then the rest of the MRO (see 2.36).
- `super()` means "the next class in the MRO of `type(self)`", not "my parent". In the diamond above, `super()` inside `B` goes to `C`, not `A` (see 2.41). This is what makes **cooperative multiple inheritance** work: every class in the chain calls `super().method(**kwargs)` and passes through arguments it does not consume.

**Practical guidance:** use multiple inheritance mainly for small **mixins** (logging, serialisation) placed to the left of the main base class, keep mixins stateless, and prefer composition when classes carry their own state. Python differs from Java here, which has single class inheritance plus interfaces (see 3.8).

### 2.40 What is a metaclass, and how does `type` fit in?

| Aspect | Explanation |
| --- | --- |
| **Metaclass** | A **class whose instances are classes**. It controls how classes themselves are created and behave. |
| **Normal object** | `p = Person()` → `p` is an instance of `Person`. |
| **Class** | `Person` is itself an object, whose type is normally `type`. |
| **Metaclass** | `type(Person)` returns `type`, so `type` is the default metaclass for most Python classes. |
| **`type`** | `type` is both a built-in function for inspecting/creating objects **and the default metaclass**. |
| **Object hierarchy** | `object` is the base class of most objects; `type` is the metaclass that normally creates classes. |
| **Custom metaclass** | You can subclass `type` and use it as a class's `metaclass` to customize class creation. |

#### The basic relationship

| Expression | Meaning |
| --- | --- |
| `x = 10` | `x` is an instance of `int` |
| `type(x)` | `int` |
| `class Person: pass` | Creates the class object `Person` |
| `type(Person)` | `type` |
| `Person()` | Creates an instance of `Person` |
| `type(Person())` | `Person` |

Conceptually:

```txt
Person instance
      ↓
   Person
      ↓
     type
```

More precisely:

```py
type(42)       # int
type(int)      # type

type(Person)   # type
type(Person()) # Person
```

#### `type` can create classes

`type` has a three-argument form:

```py
Person = type(
    "Person",
    (),
    {"name": "Alice"}
)
```

| Argument | Meaning |
| --- | --- |
| `"Person"` | Class name |
| `()` | Base classes |
| `{"name": "Alice"}` | Class namespace |
| Result | A new class object |

This is conceptually similar to:

```py
class Person:
    name = "Alice"
```

#### Custom metaclass

```py
class Meta(type):
    def __new__(mcls, name, bases, namespace):
        print("Creating:", name)
        return super().__new__(mcls, name, bases, namespace)

class Person(metaclass=Meta):
    pass
```

| Stage | What happens |
| --- | --- |
| Define `Person` | Python needs to create the `Person` class object |
| Metaclass selected | `Meta` |
| `Meta.__new__()` | Controls creation of the class object |
| `Person` created | `Person` becomes an instance of `Meta` |
| `Person()` | `Meta` ultimately enables normal instance creation through the class |

**Key interview answer:** A **metaclass is the class of a class**. In normal Python code, classes are instances of `type`. `type` therefore acts as Python's **default metaclass** and can also dynamically create classes. Custom metaclasses subclass `type` when you need to customize **class creation or class-level behavior**.

### 2.41 How does `super()` actually resolve which method to call?

| Aspect | How `super()` works |
| --- | --- |
| **Core idea** | `super()` creates a **proxy object** that performs attribute lookup starting **after a specific class in the MRO**. |
| **Does it mean “parent”?** | Not exactly. It means **“continue lookup from the next class in the MRO.”** |
| **Zero-argument `super()`** | Inside a method, `super()` uses the current class (`__class__`) and the first argument (`self`/`cls`) to determine where to start. |
| **Lookup order** | It follows the class's **MRO (C3 linearization)**. |
| **Method call** | `super().foo()` finds `foo` in the next applicable class in the MRO and invokes it with the same instance/class context. |
| **Multiple inheritance** | This is why `super()` can correctly move through multiple parent classes rather than jumping directly to one parent. |

#### Example

```py
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")
        super().show()

class C(A):
    def show(self):
        print("C")
        super().show()

class D(B, C):
    def show(self):
        print("D")
        super().show()
```

| Step | MRO / lookup |
| --- | --- |
| `D().show()` | Finds `D.show()` |
| `super()` inside `D` | Start **after `D`** |
| Next class | `B` |
| `super().show()` in `D` | Calls `B.show()` |
| `super()` inside `B` | Start **after `B`** |
| Next class | `C` |
| `super().show()` in `B` | Calls `C.show()` |
| `super()` inside `C` | Start **after `C`** |
| Next class | `A` |
| `super().show()` in `C` | Calls `A.show()` |

So:

```txt
D → B → C → A → object
```

#### Conceptually

| Expression | Rough meaning |
| --- | --- |
| `super()` | `super(__class__, self)` |
| `super(B, self).show()` | Search for `show` **after `B`** in `type(self).__mro__` |
| `super().show()` | Use the current class and instance to perform that MRO-based lookup |

**Key interview answer:** `super()` does not simply call the parent class. It creates a proxy that searches for the requested attribute **after the current class in the instance's MRO**. This is what makes `super()` work correctly with multiple inheritance and cooperative method calls.

### 2.42 What's stored in a class's `__dict__` vs an instance's `__dict__`?

| Aspect | **Class `__dict__`** | **Instance `__dict__`** |
| --- | --- | --- |
| Belongs to | The **class object** | A particular **instance** |
| Stores | Class attributes, methods, descriptors, class variables | Instance-specific attributes |
| Created from | `class` definition | Usually when attributes are assigned to an instance |
| Shared? | Yes — class attributes are shared through the class | No — each instance has its own dictionary |
| Example | `Person.__dict__` | `person.__dict__` |
| Methods | Contains function objects such as `greet` | Normally does **not** contain methods |
| Instance data | Not normally stored here | Stored here |
| Lookup role | Used when instance lookup doesn't find an attribute in the instance dictionary, subject to descriptor rules | Checked during instance attribute lookup |

#### Example

```py
class Person:
    species = "human"

    def __init__(self, name):
        self.name = name

    def greet(self):
        return "Hello"

p = Person("Alice")
```

| Dictionary | Simplified contents |
| --- | --- |
| `Person.__dict__` | `species → "human"` |
|  | `__init__ → function` |
|  | `greet → function` |
|  | Other class metadata such as `__module__`, `__dict__`, `__weakref__` |
| `p.__dict__` | `name → "Alice"` |

#### Attribute lookup

| Expression | Typical lookup |
| --- | --- |
| `p.name` | Found in `p.__dict__` |
| `p.species` | Not in `p.__dict__` → found through `Person` |
| `p.greet` | Not in `p.__dict__` → found in `Person.__dict__`; function descriptor binds `p` as `self` |
| `Person.species` | Found in `Person.__dict__` |
| `Person.greet` | Found in `Person.__dict__`; accessed from class, so no instance is automatically bound |

**Key interview answer:** `Class.__dict__` stores attributes defined on the **class**, including methods and class variables, while `instance.__dict__` stores **per-instance state**. Instance attribute lookup checks the instance and then follows the class's lookup rules/MRO.

### 2.43 What is the difference between `@staticmethod`, `@classmethod`, and instance methods internally?

| Aspect | **Instance method** | **`@classmethod`** | **`@staticmethod`** |
| --- | --- | --- | --- |
| Definition | Normal method | Method decorated with `@classmethod` | Method decorated with `@staticmethod` |
| First parameter | `self` | `cls` | None automatically |
| Receives instance? | Yes | No | No |
| Receives class? | Indirectly through `self.__class__` | Yes, automatically | No |
| Access instance state | Yes | No, unless given an instance explicitly | No |
| Access class state | Yes | Yes | Only if explicitly given the class |
| Binding behavior | **Bound to instance** | **Bound to class** | **Not automatically bound** |
| Descriptor involved | Function's `__get__()` creates a bound method | `classmethod` descriptor | `staticmethod` descriptor |
| Typical use | Object-specific behavior | Alternative constructors / class-level behavior | Utility/helper related to the class |
| Example | `obj.method()` | `MyClass.method()` or `obj.method()` | `MyClass.method()` or `obj.method()` |

### 2.44 What's the difference between an iterable and an iterator at the protocol level (`__iter__`, `__next__`)?

An **iterable** is any object that can give you an iterator: it implements `__iter__()` (or the legacy `__getitem__` sequence protocol). An **iterator** is the object that actually produces values: it implements `__next__()`, which returns the next item or raises `StopIteration`, and `__iter__()` returning itself.

| Aspect | Iterable | Iterator |
| --- | --- | --- |
| Protocol | `__iter__()` | `__iter__()` returning `self`, and `__next__()` |
| Holds position? | No, represents the data | Yes, remembers where it is |
| Reusable? | Yes, each `iter()` gives a fresh iterator | No, once exhausted it stays exhausted |
| Examples | `list`, `tuple`, `str`, `dict`, `range`, `DataFrame` | `iter(list)`, file objects, generators, `map`, `zip` |

**What a `for` loop does:**

```python
numbers = [1, 2, 3]
it = iter(numbers)        # calls numbers.__iter__()
next(it)                  # 1
next(it)                  # 2
next(it)                  # 3
next(it)                  # raises StopIteration, which ends a for loop

# for x in numbers: body   is roughly:
it = iter(numbers)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    ...                   # body
```

**A custom iterable that can be iterated many times:**

```python
class Batches:
    def __init__(self, rows, size):
        self.rows, self.size = rows, size
    def __iter__(self):                       # a generator function returns a new iterator each time
        for i in range(0, len(self.rows), self.size):
            yield self.rows[i:i + self.size]
```

**Pitfalls:**

- **Single-pass iterators.** Passing a generator or file to code that loops twice (for example computing a mean, then a variance) silently gets nothing the second time. Materialise with `list()` or pass an iterable.
- **`map`, `filter` and `zip` return iterators** in Python 3, so they are lazy and single-use.
- **Do not raise `StopIteration` inside a generator** to end it; since PEP 479 it becomes `RuntimeError`. Use `return`.

### 2.45 How do generators work, how do they keep state between `yield` calls, and why do they matter for large data?

A **generator function** is a function containing `yield`. Calling it does not run the body; it returns a **generator object**, which is an iterator. Each `next()` runs the body until the next `yield`, hands the value out, and **suspends**, keeping its frame alive: local variables, the instruction pointer and any open `try` or `with` blocks. The next `next()` resumes from exactly that point.

```python
def read_records(path):
    with open(path) as f:              # stays open while the generator is suspended
        for line in f:
            yield parse(line)          # one record at a time

def only_valid(records):
    for r in records:
        if r.is_valid():
            yield r

def batched(items, n):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

for chunk in batched(only_valid(read_records("events.jsonl")), 10_000):
    load(chunk)                        # memory bounded by one chunk, whatever the file size
```

**Internals:** the generator object owns a frame (in CPython 3.11+ an interpreter frame embedded in the generator). `yield` saves the frame state and returns to the caller; `next()` or `.send(value)` restores it (see 2.46). When the body returns, `StopIteration` is raised, carrying the return value. `gen.close()` raises `GeneratorExit` inside the generator so its `finally` and `with` blocks run.

**Why they matter for big data:**

- **Constant memory:** a pipeline of generators streams one item at a time, so a 50 GB file can be processed with megabytes of memory.
- **Laziness:** nothing is computed until a consumer pulls, and consumers can stop early (`itertools.islice`, `next(...)`).
- **Composable pipelines:** each stage is small and testable; `itertools` (`chain`, `islice`, `groupby`, `batched` in 3.12+) provides building blocks.
- **Generator expressions** (`sum(x.amount for x in rows)`) give the same laziness inline, without building a list.
- **Framework support:** PyTorch `IterableDataset`, Keras data generators, chunked `pd.read_csv(..., chunksize=...)` and streaming LLM responses all use the same pattern.

**Pitfalls:**

- Generators are single-pass (see 2.44).
- Exceptions surface where the generator is consumed, not where it was created, which can make tracebacks confusing.
- A suspended generator holding a file or connection keeps it open until it is exhausted, closed or garbage-collected; close it explicitly or wrap with `contextlib.closing`.
- Per-item Python overhead is still high; for numeric work, stream **chunks** (Arrow record batches, DataFrame chunks) and vectorise inside each chunk rather than yielding single rows.
- `yield from sub()` delegates to a sub-generator, including `send` and `throw`.

### 2.46 What happens internally when you call `.send()` on a generator?

| Aspect | What happens when `generator.send(value)` is called |
| --- | --- |
| **Purpose** | `.send(value)` **resumes a suspended generator** and sends `value` into the generator at its current `yield` expression. |
| **Generator state** | The generator must already be **suspended at a `yield`**. |
| **Example** | A generator whose body is `x = yield "hello"` followed by `yield x` (see the code below the table) |
| **First call** | `g = gen()` creates the generator but does not execute it. `next(g)` starts execution and stops at `yield "hello"`. |
| **Calling `.send()`** | `g.send(42)` resumes execution. The suspended `yield "hello"` expression evaluates to `42`. |
| **Result** | Execution continues with `x = 42`, then reaches the next `yield`, whose value is returned by `.send()`. |
| **Conceptual flow** | `yield` → generator pauses → caller calls `.send(42)` → suspended `yield` becomes `42` → generator resumes |
| **Important detail** | The value passed to `.send()` does **not** replace the value produced by `yield`; it becomes the **result of the suspended `yield` expression** inside the generator. |
| **`send(None)`** | Equivalent to `next(g)` for a suspended generator. It resumes execution without injecting a meaningful value. |
| **Before first `yield`** | A non-`None` value cannot be sent before the generator has started. `g.send(42)` raises `TypeError`. Use `next(g)` or `g.send(None)` first. |
| **Generator finishes** | If execution reaches `return`, the generator raises `StopIteration`; its `return` value is available as `StopIteration.value`. |
| **Exception injection** | `.throw(exc)` is related but different: it resumes the generator by **raising an exception at the suspended `yield`** rather than sending a normal value. |
| **Under the hood** | CPython resumes the generator's suspended execution state/frame and injects the supplied value as the result of the suspended `yield`. Modern CPython uses generator/coroutine suspension semantics in its bytecode execution machinery. |
| **Why this is powerful** | `.send()` allows **two-way communication** between the caller and generator, rather than generators being limited to producing values. |
| **Interview answer** | **`generator.send(value)` resumes a suspended generator and makes the current `yield` expression evaluate to `value`. The generator then continues executing until it reaches another `yield`, returns, or raises an exception. `send(None)` is essentially the programmatic equivalent of `next()`.** |

```python
def gen():
    x = yield "hello"
    yield x

g = gen()
next(g)        # "hello" - runs to the first yield
g.send(42)     # 42 - the suspended yield evaluates to 42, then yield x returns it
```

### 2.47 How do `async def` and `await` differ from generators under the hood?

| Aspect | Generators | `async def` / Coroutines |
| --- | --- | --- |
| **Purpose** | Produce values lazily and pause/resume execution | Perform asynchronous operations and cooperate with an event loop |
| **Syntax** | `def` + `yield` | `async def` + `await` |
| **Object created** | Calling a generator function creates a **generator object** | Calling an `async def` function creates a **coroutine object** |
| **Execution starts** | When `next()` is called | When the coroutine is driven by an event loop/task |
| **Suspension point** | `yield` | `await` |
| **What suspension means** | Generator returns a value to its caller and saves its execution state | Coroutine suspends while waiting for an awaitable and gives control back to the event loop |
| **How it resumes** | Caller calls `next()` again | Event loop/task resumes the coroutine when the awaited operation is ready |
| **Control mechanism** | Mainly controlled by the **caller** | Mainly coordinated by the **event loop** |
| **Communication** | `yield` sends values outward; `.send()` can send values back into a generator | `await` waits for an awaitable and eventually receives its result |
| **Underlying mechanism** | Uses suspended frame/execution state and generator protocol | Uses coroutine protocol, suspended execution state, and the **awaitable/task protocol** |
| **Bytecode concept** | Generator suspension uses `yield`-related execution semantics | `await` uses coroutine/awaitable bytecode semantics such as `GET_AWAITABLE` and `SEND` in modern CPython |
| **Can it be iterated?** | Yes, through the iterator protocol | Not normally with `next()`; it is consumed through coroutine/asyncio machinery |
| **Can it be awaited?** | A normal generator is **not** an awaitable | A coroutine object is awaitable |
| **Scheduling** | No built-in event-loop scheduling | Tasks created from coroutines are scheduled by an event loop |
| **I/O use** | Can implement cooperative protocols, but generators themselves don't provide async I/O | Designed specifically for asynchronous I/O and concurrency |
| **Key relationship** | Both can **suspend and later resume execution** | Coroutines evolved conceptually from generator-based coroutine mechanisms, but modern `async`/`await` has a distinct coroutine/awaitable protocol |
| **Simple mental model** | `yield` → **"Give a value to my caller and pause me."** | `await` → **"Pause me until this asynchronous operation can continue."** |
| **Interview answer** | — | **Generators suspend at `yield` and are resumed by the iterator/caller. `async def` creates coroutine objects that suspend at `await` and are resumed through the coroutine/awaitable protocol, typically by an event loop. Both preserve execution state across suspension, but generators are primarily for lazy iteration, while coroutines are designed for asynchronous control flow.** |

### 2.48 What is the event loop, and how does it schedule coroutines?

| Aspect | Explanation |
| --- | --- |
| **Event loop** | The event loop is the core mechanism behind Python's `asyncio` that **runs and coordinates asynchronous tasks and coroutines**. |
| **Coroutine** | A function defined with `async def` that can pause at an `await` and later resume. |
| **Main idea** | Instead of blocking while waiting for I/O, a coroutine yields control at `await`, allowing the event loop to run another ready coroutine. |
| **Starting a coroutine** | Calling an `async def` function creates a **coroutine object**; it does not execute the function body immediately. |
| **Task** | `asyncio.create_task(coro())` wraps the coroutine in a **Task**, which the event loop can schedule and execute. |
| **Scheduling** | The event loop maintains queues/state for tasks that are ready to run and I/O operations that are waiting. |
| **Execution** | The loop selects a ready task and advances its coroutine until it **returns, raises an exception, or reaches an `await` that cannot complete immediately**. |
| **At `await`** | The coroutine suspends, allowing the event loop to execute other ready tasks instead of blocking the thread. |
| **I/O waiting** | For operations such as sockets/network I/O, the event loop monitors the underlying I/O and wakes the associated task when the operation becomes ready. |
| **Resuming** | When the awaited operation completes, the task becomes ready again and is eventually scheduled by the event loop. |
| **Concurrency** | Multiple coroutines can make progress **concurrently on one thread**, even though only one Python coroutine executes at a time on that thread. |
| **CPU-bound code** | A coroutine that performs long-running CPU work without yielding can **block the event loop**, preventing other tasks from running. |
| **Example** | `async def fetch(): data = await network_call() return data` |
| **Typical flow** | `coroutine` → `Task` → event loop runs it → reaches `await` → suspends → other task runs → I/O completes → task becomes ready → coroutine resumes |
| **Key distinction** | `asyncio` provides **concurrency**, not automatic CPU parallelism. For CPU-heavy work, processes or other appropriate mechanisms are typically needed. |
| **Interview answer** | **The event loop is the scheduler at the heart of `asyncio`. It runs ready coroutines, and when a coroutine reaches an `await` that requires waiting—such as network I/O—it suspends that coroutine and runs another ready task. When the awaited operation completes, the event loop schedules the suspended task to resume.** |

For production patterns (TaskGroup, timeouts, cancellation, mixing sync and async code and common bugs) see 2.73.

### 2.49 How is a Python `list` implemented internally (array vs linked list)?

| Aspect | Python `list` |
| --- | --- |
| **Implementation** | A Python `list` is implemented as a **dynamic array**, **not a linked list**. |
| **Internal storage** | It maintains a contiguous array of **pointers/references to Python objects**. |
| **Example** | `items = [10, "hello", 3.14]` → the list stores references to the three objects, not the objects themselves inline. |
| **Indexing** | `items[i]` can directly access the `i`-th pointer using an offset calculation. |
| **Index lookup** | **O(1)** average/worst case for normal indexing. |
| **Appending** | `append()` is **O(1) amortized** because CPython allocates extra capacity when growing the list. |
| **Why amortized O(1)?** | Most appends only place a new pointer into existing capacity. Occasionally, the array must be resized and pointers copied, which is expensive but infrequent. |
| **Insertion at beginning/middle** | **O(n)** because existing pointers must be shifted to make room. |
| **Deletion at beginning/middle** | **O(n)** because remaining pointers generally need to be shifted. |
| **Memory layout** | The pointer array is contiguous, giving good **cache locality** and efficient sequential iteration. |
| **Linked-list comparison** | A linked list would store nodes containing pointers to neighboring nodes. Python's built-in `list` does **not** use this structure. |
| **Random access** | Dynamic array: **O(1)** |
| **Middle insertion** | Dynamic array: **O(n)** |
| **Growth strategy** | CPython over-allocates capacity when a list grows, reducing how often it needs to resize. |
| **Type flexibility** | Because the array stores object references, one list can contain objects of different types. |
| **Key takeaway** | **Python `list` = dynamically resizable array of object references.** |
| **Interview answer** | **A Python `list` is implemented as a dynamic array, not a linked list. It stores contiguous references to Python objects, giving O(1) indexing and amortized O(1) append. Insertions or deletions near the beginning or middle are O(n) because elements must be shifted.** |

### 2.50 How does a `dict` achieve O(1) average lookup? How are hash collisions handled?

| Aspect | How `dict` works |
| --- | --- |
| **Underlying structure** | A Python `dict` is a **hash table** that maps keys to values. |
| **Lookup** | For `d[key]`, Python computes `hash(key)` and uses that hash to determine where to look in the table. |
| **Why average O(1)?** | The hash function spreads keys across the table, so Python can usually jump directly to the relevant location rather than scanning all entries. |
| **Typical lookup** | `hash(key)` → determine table position → inspect candidate entry → compare key → return value |
| **Collision** | A collision occurs when two different keys map to the same initial table position. |
| **Collision handling in CPython** | CPython uses **open addressing** rather than storing a linked list at each bucket. |
| **Open addressing** | If the initial position is occupied by another key, Python probes other positions according to its probing algorithm until it finds the correct key or an available slot. |
| **Key equality** | Hash equality alone is not enough. Python also checks **key equality** (`==`) to determine whether the candidate is actually the requested key. |
| **Why `__hash__` and `__eq__` matter** | Dictionary keys must be hashable. If two keys compare equal, they must have the same hash. |
| **Example collision** | `hash(A)` and `hash(B)` may lead to the same slot → Python probes → finds another slot for one key → later lookup follows the same probing strategy. |
| **Load factor** | As the table becomes crowded, collisions increase. CPython periodically **resizes** the table to maintain efficient lookup. |
| **Average complexity** | **O(1)** for lookup, insertion, and deletion under normal hash-distribution assumptions. |
| **Worst case** | **O(n)** if many keys collide or the hash function behaves badly. |
| **Security consideration** | Python uses randomized hashing for certain built-in types such as strings, helping make deliberate collision attacks harder. |
| **Why not O(1) guaranteed?** | Hash tables depend on a reasonably distributed hash function and controlled table occupancy; therefore O(1) is an **average-case** guarantee, not a strict worst-case guarantee. |
| **Interview answer** | **A `dict` achieves average O(1) lookup by hashing the key and using the resulting hash to locate a candidate table slot. If multiple keys map to the same slot, CPython handles collisions using open addressing and probing. It then checks key equality to identify the exact key. Resizing keeps the table sufficiently sparse, maintaining average O(1) performance.** |

### 2.51 Why did dict ordering become guaranteed in Python 3.7+? What changed internally?

| Aspect | Before Python 3.6 | Python 3.6 | Python 3.7+ |
| --- | --- | --- | --- |
| **Dict ordering** | Insertion order was **not guaranteed** | CPython's new implementation preserved insertion order as an implementation detail | Insertion order became a **language guarantee** |
| **Main internal change** | Older dict used a more traditional hash-table layout | CPython introduced a **compact dict** design | The compact dict became the basis for the guaranteed ordering behavior |
| **Key/value storage** | Hash table entries contained more information directly in the table | Separate **index table** and compact **entries array** | Same compact design |
| **Insertion behavior** | Table placement was primarily determined by hashing | Entries are appended to a compact array in insertion order; the hash table stores indices pointing to those entries | This naturally preserves insertion order |
| **Example** | `d = {"a": 1, "b": 2, "c": 3}` → order not guaranteed | CPython normally iterates `a → b → c` | `a → b → c` is guaranteed by Python's language specification |
| **Why it was faster/smaller** | More memory overhead | Compact representation reduced memory usage and improved cache locality | The same implementation remained efficient while providing predictable ordering |
| **Deletion** | More traditional table management | Deleted entries can leave special markers/holes while preserving the ordering structure | Iteration continues to follow the surviving entries' insertion order |
| **Reinsertion** | Behavior depended on implementation details | Reinserted keys generally appear according to the implementation's insertion behavior | If a key is deleted and then inserted again, it appears at the **end** |
| **Important distinction** | No ordering guarantee | Ordering was a **CPython implementation detail**, not a language guarantee | Ordering is a **language-level guarantee**, so Python implementations are expected to preserve it |
| **What actually changed?** | — | **The compact dict implementation made insertion ordering a natural consequence of the data layout.** | Python 3.7 formally standardized that behavior |
| **Interview answer** | — | — | **Python 3.6 introduced CPython's compact dict implementation, which naturally preserved insertion order while reducing memory usage. Python 3.7 then made that behavior a language guarantee. Internally, the hash table maps keys to positions in a compact entries array whose entries are maintained in insertion order.** |

### 2.52 How is a `set` different from a `dict` at the implementation level?

| Aspect | `set` | `dict` |
| --- | --- | --- |
| **Underlying structure** | Hash table optimized for **storing unique values** | Hash table optimized for **mapping keys to values** |
| **Stored data** | Only **keys/elements** | **Key-value pairs** |
| **Hashing** | Each element is hashed to determine its position | Each key is hashed to determine its position |
| **Lookup** | `x in my_set` → searches for `x` | `x in my_dict` → searches for key `x` |
| **Collision handling** | Hash collisions are handled internally by the hash-table implementation | Same principle: collisions between keys are handled internally |
| **Value storage** | No associated value is needed | Each key has an associated value |
| **Conceptually** | `{a, b, c}` | `{a: 1, b: 2, c: 3}` |
| **Implementation relationship in CPython** | A set uses a specialized hash-table implementation (`setobject.c`) | A dict uses CPython's specialized dictionary implementation (`dictobject.c`) |
| **Why not just use a dict?** | A set avoids storing unnecessary values and is optimized specifically for membership/uniqueness operations | A dict needs additional machinery/storage to maintain key → value associations |
| **Uniqueness** | Elements must be unique | Keys must be unique |
| **Average membership/lookup** | **O(1)** average | **O(1)** average |
| **Insertion** | Hash the element → locate slot → insert if not already present | Hash the key → locate slot → insert/update key-value entry |
| **Deletion** | Hash element → locate → remove | Hash key → locate → remove key-value entry |
| **Memory** | Generally stores less information because there is no value for each element | Stores keys plus their corresponding values |
| **Ordering** | Modern CPython sets should **not be relied upon for insertion order** | `dict` preserves **insertion order** as a language guarantee in modern Python |
| **Example** | `users = {"Alice", "Bob"}` | `users = {"Alice": 25, "Bob": 30}` |
| **Key implementation difference** | **Hash table of unique keys/elements** | **Hash table of unique keys associated with values** |
| **Interview answer** | **A `set` is essentially a hash table optimized for storing unique elements and performing fast membership tests. A `dict` is a hash table that stores unique keys together with associated values. Both use hashing and provide average O(1) lookup, but their storage and purpose differ.** |  |

### 2.53 What happens internally when you run `import module`, and what is the role of `sys.modules`?

`import module` asks the import system for a module object. It first checks the in-memory cache **`sys.modules`**; on a miss it finds, creates, caches and executes the module; finally it binds the name in the current namespace. Because of the cache, a module's top-level code normally runs **once per process**, however many times it is imported.

**Step by step:**

1. **Cache lookup.** If `"module"` is a key in `sys.modules` (a plain dict of fully qualified names to module objects), return that object. Importing `a.b.c` checks and imports `a`, then `a.b`, then `a.b.c`.
2. **Find.** Each finder in `sys.meta_path` is asked for a `ModuleSpec`. The path-based finder searches `sys.path` (script directory, `PYTHONPATH`, site-packages) for a package directory, a `.py` file, a compiled extension or a namespace package.
3. **Create.** The spec's loader creates an empty module object and sets `__name__`, `__file__`, `__spec__`, `__package__`.
4. **Cache before execution.** The module is inserted into `sys.modules` **before** its code runs. This is what lets circular imports terminate, and why they can see a partially initialised module.
5. **Execute.** The loader gets bytecode (from a valid `.pyc` or by compiling source, see 2.7) and runs it in the module's namespace: defining functions and classes, running top-level statements and nested imports.
6. **On failure**, the module is removed from `sys.modules` and the exception propagates.
7. **Bind.** `import module` binds `module`; `from module import name` binds `name` after fetching it from the module object.

```python
import sys, importlib
import json
json is sys.modules["json"]      # True: same object every time
importlib.reload(json)           # re-executes the code into the SAME module object
del sys.modules["mymod"]         # next import creates a fresh module object
```

**Production implications:**

- **Import-time side effects** (opening database connections, reading config, loading a model) run whenever anything imports the module, including tests and tooling. Keep top-level code cheap; initialise lazily.
- **Circular imports** show up as `ImportError: cannot import name ... (most likely due to a circular import)`. Fix by moving shared code into a third module, importing inside functions, or using `typing.TYPE_CHECKING` for type-only imports.
- **Startup time** of CLIs and serverless functions is often dominated by imports; measure with `python -X importtime`. Heavy libraries (pandas, torch) can be imported lazily inside the functions that need them.
- **`importlib.reload`** does not update references other modules already hold (`from mod import f` keeps the old `f`), so it is a debugging tool, not a deployment strategy.
- **Shadowing:** a local file named `json.py` or `random.py` shadows the standard library because the script directory comes first on `sys.path`.
- `sys.modules` is an in-memory cache that prevents re-execution; `.pyc` is an on-disk cache that prevents recompilation. They solve different problems.

### 2.54 How do `__init__.py` and packages affect the import system?

| Aspect | Explanation |
| --- | --- |
| **Package** | A package is a directory that groups related Python modules and subpackages under a common namespace. |
| **`__init__.py`** | Traditionally marks a directory as a **regular Python package** and is executed when that package is imported. |
| **Example structure** | `myapp/` → `__init__.py`, `utils.py`, `models.py` |
| **Importing package** | `import myapp` causes Python to locate the package and execute `myapp/__init__.py`. |
| **Importing submodule** | `import myapp.utils` first loads `myapp`, then locates and loads `utils` within that package. |
| **Initialization role** | `__init__.py` can initialize package-level state, expose selected names, or run package setup code. |
| **Namespace exposure** | `__init__.py` can make APIs available directly: `from myapp import User` instead of `from myapp.models import User`. |
| **Without `__init__.py`** | Modern Python can still recognize a directory as a **namespace package** (PEP 420), provided the import system can find the directory. |
| **Regular vs namespace package** | Regular package → typically has `__init__.py` and can execute initialization code. Namespace package → can span multiple directories and has no `__init__.py`. |
| **Import resolution** | Python's import machinery searches import locations such as `sys.path`, finds a package/module, creates a module object, and loads/executes it. |
| **`__package__`** | During package imports, Python sets `__package__`, which helps resolve **relative imports** such as `from .utils import helper`. |
| **Caching** | Once imported, the module/package is normally stored in `sys.modules`, so subsequent imports reuse the existing module object rather than executing it again. |
| **Key distinction** | `__init__.py` is **package initialization code**, while the package directory provides the namespace/hierarchy used by the import system. |
| **Interview answer** | **`__init__.py` traditionally makes a directory a regular Python package and runs when the package is imported. It can initialize package state and expose package-level APIs. Modern Python also supports namespace packages without `__init__.py`, which allow package portions to exist across multiple directories.** |

### 2.55 What's the difference between absolute and relative imports at resolution time?

| Aspect | Absolute Import | Relative Import |
| --- | --- | --- |
| **Meaning** | Specifies the **full package/module path** | Specifies a module **relative to the current package** |
| **Example** | `from myapp.utils import helper` | `from .utils import helper` |
| **Resolution starting point** | Starts from entries in `sys.path` (typically the top-level package) | Starts from the **current package's location** |
| **Dots (`.`)** | No leading dots | `.` = current package, `..` = parent package, etc. |
| **Dependency on package location** | Less dependent on where the importing module sits within the package | Depends on the importing module's package hierarchy |
| **`sys.path` role** | Python searches `sys.path` to locate the top-level package | Python first determines the current package, then resolves the relative path from it |
| **Readability** | Often clearer for larger projects | Convenient for closely related modules within the same package |
| **Moving modules** | Can require fewer changes if the package's top-level name remains stable | Can be easier when moving an entire package subtree |
| **Can be used from a standalone script?** | Generally yes, if the package is importable | Usually no; relative imports require a recognized package context |
| **Resolution example** | `myapp.services.api` → `from myapp.models import User` searches for `myapp` as a top-level package | `myapp.services.api` → `from ..models import User` resolves `models` relative to `myapp.services` |
| **Key difference** | **"Find this package from the import search path."** | **"Find this module relative to my current package."** |
| **Interview answer** | **Absolute imports resolve from the top-level import search path (`sys.path`), while relative imports resolve from the importing module's package using leading dots.** |  |

### 2.56 How does exception propagation work internally (the traceback stack)?

| Aspect | Explanation |
| --- | --- |
| **Exception raised** | When an exception occurs, CPython creates an exception object containing information such as its type and message. |
| **Current frame** | The interpreter checks the current function's execution frame to see whether an applicable `except` handler exists. |
| **No matching handler** | If no handler matches, the exception **propagates upward** to the caller's frame. |
| **Call stack** | This continues through the chain of function calls until a matching handler is found or the exception reaches the top level. |
| **Example flow** | `main()` → `foo()` → `bar()` → exception in `bar()` → no handler → `foo()` → no handler → `main()` → handler or program termination. |
| **Traceback** | CPython records the frames where the exception propagated. These frames form the traceback showing **where the exception occurred and how execution reached that point**. |
| **What a traceback contains** | Typically the exception type/message plus the **file, line number, function name, and call chain** for relevant frames. |
| **If a handler is found** | Python unwinds the intervening stack frames until it reaches the frame containing the matching `except` block, then transfers control there. |
| **Stack unwinding** | Local execution state in skipped frames is cleaned up as the exception propagates. `finally` blocks and relevant cleanup mechanisms are still executed during unwinding. |
| **`finally` interaction** | A `finally` block runs while the stack is being unwound, even when an exception is propagating through that frame. |
| **Re-raising** | `raise` inside an `except` handler continues propagation while preserving the exception context/traceback appropriately. |
| **Exception chaining** | `raise NewError(...) from original_error` explicitly records a relationship between the new and original exceptions, which Python displays in the traceback. |
| **Modern CPython** | In CPython 3.11+, exception handling uses the interpreter's exception state and exception-table-based control flow rather than the older `SETUP_EXCEPT`/`SETUP_FINALLY` bytecode approach. |
| **Key idea** | **An exception propagates by moving up the call stack looking for a matching handler. During this unwinding, cleanup code such as `finally` executes, while traceback information records the relevant execution frames.** |
| **Interview answer** | **When an exception is raised, CPython checks the current frame for a matching handler. If none exists, it unwinds the frame and propagates the exception to the caller. This continues up the call stack until a handler is found or the exception reaches the top level. The traceback records the call chain and location information that shows how the exception propagated.** |

### 2.57 What's the difference between `except Exception` and a bare `except:`?

| Aspect | `except Exception:` | Bare `except:` |
| --- | --- | --- |
| **What it catches** | Exceptions derived from `Exception` | **Almost everything**, including exceptions derived directly from `BaseException` |
| **Catches `ValueError`, `TypeError`, `KeyError`, etc.** | Yes | Yes |
| **Catches `KeyboardInterrupt`** | No | **Yes** |
| **Catches `SystemExit`** | No | **Yes** |
| **Catches `GeneratorExit`** | No | **Yes** |
| **Typical usage** | **Recommended** for normal application error handling | Rare; mainly for very specific cases where you intentionally need to catch all exceptions |
| **Risk** | Lower; allows system-control exceptions to propagate | Higher; can accidentally swallow interrupts, exits, or other important control-flow exceptions |
| **Example** | `except Exception:` followed by `handle_error()` | `except:` followed by `handle_everything()` |
| **Inheritance** | `Exception` is a subclass of `BaseException` | Catches the `BaseException` hierarchy |
| **Best practice** | Prefer `except Exception:` unless you have a specific reason otherwise | Avoid in most application code |
| **Interview answer** | **`except Exception` catches normal application exceptions, while bare `except` catches all `BaseException` subclasses, including `KeyboardInterrupt` and `SystemExit`. Therefore, `except Exception` is generally safer and preferred.** |  |

### 2.58 How does `finally` interact with return statements and exceptions in bytecode?

| Aspect | How `finally` behaves |
| --- | --- |
| **Basic purpose** | `finally` contains cleanup code that runs whether the `try` block **returns normally, returns early, or raises an exception**. |
| **`return` inside `try`** | Python does **not immediately return**. It first executes the `finally` block, then completes the return. |
| **Example** | `try: return 10` with `finally: print("cleanup")` prints `"cleanup"`, then returns `10` (see the code below the table). |
| **Bytecode concept** | The compiler arranges control flow so that paths leaving the `try` block—such as `return`, exception, or normal completion—pass through the `finally` code. |
| **Return value handling** | When `return 10` is encountered, the return value is preserved while `finally` executes. Afterward, the original return proceeds. |
| **Exception in `try`** | The exception is temporarily handled so the `finally` block can execute. After `finally`, the exception continues propagating unless handled. |
| **Exception in `finally`** | A new exception from `finally` replaces/interferes with the original exception. |
| **`return` in `finally`** | A `return` inside `finally` **overrides** a pending return or exception from `try`. |
| **Example** | `try: return 10` with `finally: return 20` returns `20`. |
| **Modern CPython** | Modern CPython (3.11+) uses specialized exception-handling bytecode and compiler-generated control flow rather than the older `SETUP_FINALLY` mechanism found in older versions. |
| **Key bytecode idea** | `finally` effectively creates **multiple control-flow paths**: normal execution, `return`, and exception paths all ensure the cleanup code runs before control leaves the `try`. |
| **Interview answer** | **A `finally` block is guaranteed to run when leaving a `try`, including during `return` and exception paths. In bytecode, CPython arranges these control-flow paths so they execute the `finally` code before completing the return or propagating the exception. A `return` or exception raised inside `finally` can override the original control flow.** |

```python
def f():
    try:
        return 10
    finally:
        print("cleanup")     # runs before the return completes; f() returns 10

def g():
    try:
        return 10
    finally:
        return 20            # overrides the pending return; g() returns 20 (SyntaxWarning in 3.14, PEP 765)
```

### 2.59 Why is CPython considered slower than compiled languages, and where's the overhead?

| Aspect | CPython | Compiled Languages (e.g., C/C++) |
| --- | --- | --- |
| **Execution** | Python source → bytecode → CPython interpreter | Source → native machine code |
| **Main overhead** | CPython must interpret/execute Python bytecode at runtime | CPU executes optimized machine instructions directly |
| **Dynamic typing** | Types are determined and checked at runtime | Types are generally known at compile time |
| **Operation cost** | `a + b` may involve type checks, object handling, and dynamic dispatch | Often becomes a small number of native CPU instructions |
| **Python objects** | Almost everything is a heap-allocated Python object with metadata and reference-counting/GC overhead | Primitive values can often be stored and manipulated directly |
| **Memory management** | Reference counting + cyclic garbage collection | More direct/manual or compiler-assisted memory management |
| **Function calls** | Python function calls have significant interpreter/object overhead | Native function calls are generally much cheaper |
| **Optimization** | CPython performs runtime optimizations, but traditionally does not compile all Python code directly to optimized native machine code | Compilers such as GCC/Clang can perform extensive ahead-of-time optimization |
| **Example** | `x += 1` involves Python object operations and interpreter/runtime machinery | An integer increment can be reduced to a simple CPU instruction |
| **Where overhead occurs** | **Bytecode dispatch, dynamic type checking, object allocation, reference counting, attribute lookup, function calls, and interpreter/runtime operations** | Much less runtime interpretation overhead |
| **Why slower?** | CPython prioritizes **dynamic flexibility, simplicity, and compatibility** over raw CPU execution speed | Compiled languages can specialize and optimize code before execution |
| **Important nuance** | CPython can be quite fast for I/O-bound applications, and native libraries such as NumPy can move heavy computation outside the Python interpreter | Raw CPU-bound loops generally favor native compiled code |
| **Interview answer** | **CPython is slower mainly because Python code is dynamically typed and executed through a runtime that performs bytecode dispatch, object management, type checks, and dynamic lookups. Compiled languages can translate code ahead of time into optimized machine instructions, eliminating much of this runtime overhead.** |  |

### 2.60 What is the difference between CPython, PyPy, and other implementations?

| Aspect | CPython | PyPy | Other Python Implementations |
| --- | --- | --- | --- |
| **Definition** | The **reference and most widely used Python implementation**, written mainly in C | A Python implementation with a **JIT (Just-In-Time) compiler**, largely written in RPython | Alternative implementations designed for specific platforms or goals |
| **Implementation language** | C | RPython/Python ecosystem | Varies |
| **Execution model** | Interprets bytecode with a specialising adaptive interpreter (3.11+); an experimental, opt-in JIT compiler has been included since 3.13 (PEP 744) | Uses a **JIT compiler** to compile frequently executed code into machine code at runtime | Depends on implementation |
| **Performance** | Excellent general-purpose performance and broad compatibility | Can be faster for **long-running, repetitive pure-Python workloads** due to JIT | Varies |
| **Startup time** | Generally good | Can have additional JIT warm-up overhead | Varies |
| **GIL** | Traditional CPython has a GIL; newer CPython versions also have optional **free-threaded builds** | Has its own threading/runtime behavior and is not simply equivalent to CPython's GIL model | Depends on implementation |
| **C extensions** | **Best compatibility** with CPython C extensions | Compatibility can differ; many CPython extensions require adaptation | Depends on implementation |
| **Memory management** | Reference counting + cyclic garbage collector | Garbage collection with a tracing GC approach | Depends on implementation |
| **`.pyc` bytecode** | Uses `.pyc` files for cached bytecode | Has different runtime/bytecode behavior and should not be assumed identical to CPython | Depends on implementation |
| **Best suited for** | Most applications, libraries, web development, data science, and production Python | Long-running applications with substantial pure-Python workloads | Specialized environments |
| **Examples** | CPython | PyPy | **Jython** (JVM), **IronPython** (.NET), **MicroPython** (microcontrollers), **GraalPy** (GraalVM) |
| **Key difference** | **CPython prioritizes compatibility and mature ecosystem support.** | **PyPy focuses heavily on JIT-based runtime performance.** | **Other implementations target different platforms, environments, or constraints.** |
| **Interview answer** | **CPython is the standard implementation with the strongest ecosystem compatibility. PyPy uses JIT compilation to speed up suitable long-running Python code. Other implementations, such as Jython, IronPython, MicroPython, and GraalPy, target different runtimes or hardware environments.** |  |  |

### 2.61 How does the GIL affect the choice between threads, processes and asyncio for a data workload?

Pick the concurrency model by **where the time goes**: waiting on I/O, or computing in Python. Under the standard GIL build, threads and `asyncio` overlap waiting, while only separate processes (or native code that releases the GIL) give parallel CPU execution.

| Aspect | Threads (`ThreadPoolExecutor`) | Processes (`ProcessPoolExecutor`, `multiprocessing`) | `asyncio` |
| --- | --- | --- | --- |
| GIL | Shared; one thread runs Python bytecode at a time | One GIL per process, so true parallelism | Single thread |
| Best for | Blocking I/O with sync libraries (boto3, requests, JDBC drivers), NumPy-heavy work that releases the GIL | CPU-bound pure-Python transforms, parsing, feature computation | Very high-concurrency network I/O with async libraries (httpx, aiohttp, asyncpg) |
| Memory | Shared address space | Separate; data copied or shared explicitly | Shared |
| Communication cost | Near zero | Arguments and results are pickled; large DataFrames are expensive to send | Near zero |
| Start-up cost | Low | High (process spawn, imports), amortise with pools | Lowest per task |
| Failure isolation | A crash in a C extension kills everything | A crashed worker can be detected and replaced | One blocking call stalls everything |
| Typical scale | Tens to hundreds of concurrent calls | One worker per core | Thousands of concurrent connections |

**Ingestion example:** downloading 10,000 files from S3 and then parsing them. Use threads or async for the downloads (I/O-bound, 32 to 128 concurrent requests, bounded by a semaphore or pool size to respect rate limits), then a process pool sized to the number of cores for CPU-heavy parsing, or better, a vectorised engine such as Polars or DuckDB that parallelises internally.

**Production details:**

- **Start method.** On Linux the default was `fork` (fast, but unsafe with threads and some libraries); Python 3.14 changed the Linux default to `forkserver`, and macOS and Windows use `spawn`. With `spawn`, worker code must be importable and guarded by `if __name__ == "__main__":`.
- **Minimise data crossing processes.** Send file paths or partition keys, not DataFrames; use `multiprocessing.shared_memory`, Arrow or memory-mapped files for large arrays.
- **Bound concurrency** to protect downstream systems; unbounded thread or task creation is a common way to take down a database.
- **Beyond one machine** use Spark, Dask or Ray rather than hand-rolled multiprocessing.
- **Free-threaded Python** (3.14+, opt-in) changes this table: threads can run Python code in parallel, giving process-level parallelism with shared memory, once your dependencies support it (see 2.72).

**Interview answer:** threads for blocking I/O, asyncio for massive I/O concurrency with async libraries, processes for CPU-bound Python, and native libraries that release the GIL wherever possible; and always measure, because the bottleneck is often not where you expect.

### 2.62 What are the differences between a list and a tuple, and when do you use each in data engineering?

A **list** is a mutable, growable sequence; a **tuple** is an immutable, fixed-size sequence. Conventionally lists hold **many items of the same kind** (a collection), while tuples hold **a fixed number of fields of different meaning** (a record).

| Feature | List | Tuple |
| --- | --- | --- |
| Mutability | Mutable: append, remove, sort in place | Immutable after creation |
| Hashable | No, cannot be a dict key or set member | Yes, if all elements are hashable |
| Memory | Over-allocates spare capacity for appends | Exact size, slightly smaller; no growth buffer |
| Creation speed | Built at run time | Constant tuples are created once at compile time and reused |
| Semantics | Homogeneous collection | Heterogeneous record, like a row |
| Thread safety | Shared mutation needs care | Safe to share, but contained objects may still be mutable |

```python
from typing import NamedTuple

partition_key = ("2024-06-01", "eu-west-1")          # tuple as composite dict key
row_counts = {partition_key: 18_204}

class Event(NamedTuple):                              # tuple with field names and types
    user_id: int
    event_type: str
    ts: float

batch: list[Event] = []                               # list: a growing collection of records
batch.append(Event(42, "click", 1717200000.0))
```

**Data engineering uses:**

- Tuples as **composite keys** (partition, date, customer) in dicts and sets, for deduplication and grouping.
- **DB-API rows** come back as tuples from `cursor.fetchall()`; `executemany` takes a list of tuples.
- `NamedTuple` or frozen dataclasses for small, immutable records passed between pipeline stages.
- Lists for buffers and batches that grow before a bulk write.

**Pitfall:** immutability is shallow. `t = ([1, 2], 3)` cannot be rebound, but `t[0].append(9)` works, and such a tuple is not hashable. And a single-element tuple needs a trailing comma: `(5,)`.

### 2.63 What is the difference between a shallow copy and a deep copy?

Assignment copies nothing; it creates another reference to the same object. A **shallow copy** creates a new outer container but fills it with references to the **same inner objects**. A **deep copy** recursively copies the container and everything it contains, so the result shares no mutable state with the original.

```python
import copy

config = {"source": "s3", "columns": ["id", "amount"]}

alias = config                     # same object
shallow = copy.copy(config)        # also: dict(config), config.copy(), lst[:]
deep = copy.deepcopy(config)

shallow["source"] = "gcs"          # only affects shallow (top-level rebinding)
shallow["columns"].append("ts")    # ALSO changes config["columns"]: shared inner list
deep["columns"].append("region")   # original untouched
```

| Feature | Assignment | Shallow copy | Deep copy |
| --- | --- | --- | --- |
| New outer object | No | Yes | Yes |
| Nested objects | Shared | Shared | Copied recursively |
| Cost | Free | Proportional to top-level size | Proportional to the whole object graph; can be very slow |
| How | `b = a` | `copy.copy`, `.copy()`, slicing, constructors | `copy.deepcopy` |

**Details worth knowing:**

- `deepcopy` keeps a memo dictionary, so shared references and cycles are preserved rather than copied twice or looping forever.
- Classes can customise copying with `__copy__` and `__deepcopy__`; objects such as locks, sockets and open files cannot be meaningfully deep-copied.
- **pandas:** `df.copy()` is deep by default for the data (`deep=True`), but Python objects inside `object` columns are not recursively copied. Slicing a DataFrame historically could return a view or a copy, causing `SettingWithCopyWarning`; pandas 3.0 makes Copy-on-Write the default, so derived DataFrames behave like copies and are copied lazily only when modified.
- **NumPy:** basic slicing returns a **view** sharing memory; use `.copy()` when you need independence.

**When to use which:** a classic bug is a mutable default config shared between pipeline runs or tests; deep-copy the template per run, or better, use immutable structures (frozen dataclasses, tuples) so copying is unnecessary. Avoid `deepcopy` in hot paths; it is often the hidden cost in slow code.

### 2.64 What do `*args` and `**kwargs` do in function definitions and calls?

In a definition, `*args` collects extra **positional** arguments into a tuple and `**kwargs` collects extra **keyword** arguments into a dict. In a call, `*` and `**` do the reverse: they **unpack** a sequence or a mapping into arguments. The names are convention; the stars are what matter.

```python
def process_data(source, *args, batch_size=100, **kwargs):
    print(source, args, batch_size, kwargs)

process_data("api", 1, 2, batch_size=500, retries=3)
# api (1, 2) 500 {'retries': 3}

params = {"batch_size": 1000, "retries": 5}
process_data("s3", *[7, 8], **params)       # unpacking at the call site
```

**The full parameter grammar** (worth knowing for API design):

```python
def f(pos_only, /, normal, *, kw_only, **rest): ...
```

- Parameters before `/` are positional-only (3.8+), so you can rename them without breaking callers.
- Parameters after `*` (or after `*args`) are keyword-only, which forces readable calls such as `load(df, overwrite=True)` instead of `load(df, True)`.

**Where they are used:**

- **Decorators and wrappers** that forward any signature: `return func(*args, **kwargs)` (see 2.6).
- **Cooperative inheritance:** `super().__init__(**kwargs)` passes through arguments a class does not consume.
- Forwarding options to an underlying library, for example `def read(path, **read_csv_kwargs): return pd.read_csv(path, **read_csv_kwargs)`.

**Pitfalls and good practice:**

- Over-using `**kwargs` hides the real interface: no autocompletion, no type checking, typos silently ignored. Prefer explicit keyword-only parameters in public APIs.
- To type them, annotate the element type (`*args: int`, `**kwargs: str`), use `Unpack[TypedDict]` for structured kwargs (3.12+), and `ParamSpec` for decorators that preserve signatures.
- Dict merging uses the same idea: `{**defaults, **overrides}` or `defaults | overrides` (3.9+).

### 2.65 What are comprehensions, and how do they compare with `map()`, `filter()` and lambda functions?

**Comprehensions** build lists, dicts, sets or lazy generators from an iterable with an optional filter in one expression. **`map()`** applies a function to every item and **`filter()`** keeps items for which a function returns true; both return lazy iterators in Python 3. A **lambda** is an anonymous single-expression function, often passed to them.

```python
numbers = [1, -2, 3, 4]

squares = [x * x for x in numbers if x > 0]            # list comprehension
lookup = {u.id: u for u in users}                      # dict comprehension
domains = {e.split("@")[1] for e in emails}            # set comprehension
total = sum(x * x for x in numbers)                    # generator expression: lazy, no list

squares_m = list(map(lambda x: x * x, filter(lambda x: x > 0, numbers)))
names = list(map(str.upper, raw_names))                # map with an existing function reads well

rows.sort(key=lambda r: (r.country, -r.revenue))      # lambda as a short key function
```

**Choosing between them:**

| Situation | Prefer |
| --- | --- |
| Transform and filter in one go | Comprehension: more readable than nested `map`/`filter`/`lambda` |
| Applying an existing named function | `map(func, items)` is fine and slightly faster than a lambda |
| Large or unbounded input | Generator expression or `map` (lazy) instead of a list |
| Sort keys, `max(key=...)`, small callbacks | Lambda or `operator.itemgetter` / `attrgetter` |
| Logic with statements, several steps or reuse | A named `def` with a docstring and tests |

**Performance:** comprehensions avoid repeated `append` lookups and are usually faster than an equivalent `for` loop; since 3.12 they are inlined (PEP 709). But they are still Python-level loops. On a DataFrame, a vectorised expression (`df["amount"] * 1.2`) beats both a comprehension and `df.apply(lambda ...)` by one to two orders of magnitude.

**Lambda limitations:** a single expression, no statements or annotations, an unhelpful `<lambda>` name in tracebacks, and they cannot be pickled by standard `pickle` (a problem for `multiprocessing`). PEP 8 says not to assign a lambda to a name; use `def`. Beware late binding in loops (see 2.30).

**Readability rule:** if a comprehension needs more than one `for` and one `if`, or does not fit on two lines, write a loop or a helper function.

### 2.66 What is the difference between the `/` and `//` operators, and what are the edge cases?

`/` is **true division** and always returns a `float` (for `int` operands); `//` is **floor division**, rounding towards negative infinity, and returns an `int` for integers. `%` is the matching remainder, so `a == (a // b) * b + a % b` always holds.

```python
10 / 4      # 2.5
10 // 4     # 2
10 % 4      # 2
divmod(10, 4)   # (2, 2)

-7 // 2     # -4  (floors, does NOT truncate towards zero)
-7 % 2      # 1   (result takes the sign of the divisor)
int(-7 / 2) # -3  (truncation, as in C, Java and SQL)

7.5 // 2    # 3.0 (float in, float out)
10 / 5      # 2.0 (always float, even when exact)
```

**Edge cases that cause real bugs:**

- **Negative numbers.** Python floors while C, Java, Spark SQL and most databases truncate, so the same bucketing logic gives different results in Python and SQL for negative values.
- **Large integers.** `/` converts to float, losing precision above 2 to the power 53: `(10**20 + 1) / 1` is not exact. Use `//` for integer arithmetic, and `decimal.Decimal` for money.
- **Division by zero** raises `ZeroDivisionError` for both; NumPy and pandas return `inf` or `NaN` with a warning instead.
- **Python 2 legacy:** in Python 2, `/` on two ints floored; old code ported carelessly can change results.
- **Floor division on floats** can surprise: `1 // 0.1` is `9.0` because 0.1 is slightly more than one tenth in binary.

**Typical uses of `//` and `%`:** pagination and batch numbering (`page = offset // page_size`), hash bucketing (`hash(key) % n_partitions`), and time bucketing (`epoch_seconds // 3600 * 3600`).

### 2.67 How does the `with` statement work, and how do you write context managers?

The `with` statement guarantees that setup and teardown code run around a block, even if the block raises. It works with any **context manager**: an object with `__enter__` (called on entry, its return value is bound by `as`) and `__exit__(exc_type, exc, tb)` (always called on exit; returning `True` suppresses the exception).

```python
with open("data.csv") as f:          # file closed even if parsing fails
    data = f.read()

# roughly what the interpreter does:
mgr = open("data.csv")
f = mgr.__enter__()
try:
    data = f.read()
except BaseException as e:
    if not mgr.__exit__(type(e), e, e.__traceback__):
        raise
else:
    mgr.__exit__(None, None, None)
```

**Writing one as a class** (when there is state or reuse):

```python
class Transaction:
    def __init__(self, conn):
        self.conn = conn
    def __enter__(self):
        return self.conn.cursor()
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        return False                     # do not swallow the exception
```

**Writing one with `contextlib`** (usually simpler):

```python
from contextlib import contextmanager
import time, logging

@contextmanager
def timed(step: str):
    start = time.perf_counter()
    try:
        yield                            # the body of the with block runs here
    finally:
        logging.info("%s took %.2fs", step, time.perf_counter() - start)

with timed("load"):
    load_partition()
```

**Useful `contextlib` tools:** `ExitStack` for a dynamic number of resources (open N files, roll all back on failure), `suppress(FileNotFoundError)`, `closing()` for objects with only `close()`, `nullcontext()` for optional managers, `chdir()` (3.11+), and `asynccontextmanager` for `async with`.

**Data engineering uses:** database connections and transactions, file and object-store handles, locks, temporary directories, Spark sessions, tracing spans, temporarily changing settings (`pd.option_context`, `torch.no_grad()`, `np.errstate`).

**Pitfalls:** a `@contextmanager` generator without `try`/`finally` around `yield` skips cleanup on exceptions; returning `True` from `__exit__` accidentally swallows errors; and a context manager on a connection object may commit a transaction without closing the connection (sqlite3 and psycopg2 behave this way), so read the library's semantics.

### 2.68 How do you profile and optimise Python code that processes millions of rows?

Measure first, then remove Python-level per-row work. Almost every speed-up in data code comes from moving the inner loop out of the interpreter into vectorised or compiled code, and from reading less data; micro-optimising Python syntax rarely matters.

**1. Profile to find the real bottleneck:**

| Tool | Use |
| --- | --- |
| `cProfile` plus `snakeviz` | Deterministic function-level profile of a script or test |
| `py-spy` | Sampling profiler; attach to a running production process with no code change and produce flame graphs |
| `scalene` | CPU, memory and GPU split between Python and native code per line |
| `line_profiler` | Line-by-line timing of one hot function |
| `memray`, `tracemalloc` | Memory allocations and peak usage |
| `timeit`, `pytest-benchmark` | Micro-benchmarks and regression tests for speed |

**2. Apply the optimisation ladder, cheapest first:**

1. **Do less I/O.** Read only needed columns and partitions (Parquet column pruning, predicate pushdown), filter early, and avoid CSV for large data.
2. **Better algorithms and data structures.** Replace list membership tests with sets, nested loops with joins or dict lookups.
3. **Vectorise.** NumPy and pandas operations run in C over whole columns. Replace `df.apply(f, axis=1)` and `iterrows()` with column expressions, `np.where`, `np.select`, `merge` and `groupby`.
4. **Use a faster engine.** **Polars** (Rust, multi-threaded, lazy query optimiser) or **DuckDB** are often 5 to 50 times faster than pandas for joins and group-bys and use less memory.
5. **Efficient dtypes.** `category` for low-cardinality strings, downcast numerics, Arrow-backed strings in pandas.
6. **Compile the remaining loop.** **Numba** (`@njit`) JIT-compiles numeric NumPy loops; **Cython** compiles typed Python-like code to C; **Rust via PyO3 and maturin** for custom logic with safe parallelism (this is how Polars, Pydantic v2 and tokenizers are built).
7. **Parallelise.** Process pools for CPU-bound Python, or Spark, Dask and Ray when data exceeds one machine (see 2.61).
8. **Stream in chunks** to bound memory (`pl.scan_parquet`, `pd.read_csv(chunksize=...)`, Arrow record batches); see 2.45.

```python
import numpy as np, pandas as pd, polars as pl

# slow: Python function called once per row
df["band"] = df.apply(lambda r: "high" if r["amount"] > 1000 else "low", axis=1)

# fast: vectorised
df["band"] = np.where(df["amount"] > 1000, "high", "low")

# Polars lazy query: optimiser pushes filters and projections into the scan
out = (pl.scan_parquet("events/*.parquet")
         .filter(pl.col("country") == "GB")
         .group_by("user_id")
         .agg(pl.col("amount").sum())
         .collect())
```

**Pitfalls:** optimising without a profile; benchmarking on tiny samples; Python UDFs in Spark or Polars that silently fall back to row-at-a-time execution (prefer built-in expressions, or vectorised pandas UDFs with Arrow); and over-engineering, since a two-times speed-up on a job that runs nightly in ten minutes is rarely worth extra complexity. State the target (latency, cost, memory) before optimising.

### 2.69 How do you keep a Python codebase maintainable in a large team?

Maintainability comes from **clear boundaries, enforced conventions and fast feedback**, so that many people can change the code safely. Style is the easy part and should be automated; the hard parts are module structure, interfaces and tests.

**Structure and design:**

- Package by domain (`ingestion`, `features`, `serving`), not by technical layer, with a small public API per package and private internals.
- Keep I/O at the edges: pure transformation functions in the middle are easy to test and reuse; readers and writers are thin adapters (ports and adapters).
- Make dependencies explicit (constructor arguments, dependency injection) rather than globals and import-time singletons.
- Use typed data contracts between components: dataclasses or Pydantic models instead of loose dicts, and schemas (Pandera, Pydantic) for DataFrames and events.
- Keep notebooks for exploration; move anything reused into the package with tests.

**Automated guard rails** (see 2.11):

- Ruff for formatting and linting, mypy or Pyright for types (strict on new code), Bandit and `pip-audit` for security, all configured in `pyproject.toml` and run in pre-commit and CI.
- Reproducible environments with a lock file (see 2.74).
- A test pyramid with fast unit tests on every commit (see 2.77).
- Import-boundary rules (for example `import-linter`) so layers do not become a tangle.

**Team practices:**

- Small pull requests, code review focused on design and correctness rather than style, and code owners for critical areas.
- Architecture decision records for choices such as "Polars over pandas" so newcomers know why.
- Docstrings on public functions, a README per package, and runnable examples.
- Deprecation policy for shared libraries: warn, keep the old API for a release, then remove; semantic versioning for internal packages.
- Track signals: CI duration, flaky tests, type coverage, time to first commit for new joiners.

**What interviewers look for at senior level:** you can describe how you introduced these practices to an existing messy codebase incrementally (baselines, ratchets, "boy scout rule"), and how you balanced them against delivery pressure rather than imposing everything at once.

### 2.70 What is Monkey Patching and why is it generally discouraged?

**Monkey patching** is the practice of **modifying or replacing classes, functions, or methods at runtime**, often in third-party libraries.

```py
class Service:
    def process(self):
        return "original"

def new_process(self):
    return "patched"

Service.process = new_process
```

#### Why is it discouraged?

- **Unpredictable behavior:** Runtime behavior differs from the original source code.
- **Hard to debug:** It can be unclear where a method was modified.
- **Maintenance issues:** Library updates may break the patch.
- **Hidden dependencies:** Code may silently depend on the modification.
- **Testing complexity:** Patches can affect unrelated parts of the application.

**Use case:** It can be appropriate for controlled testing or as a temporary workaround for a third-party bug, but explicit extension, dependency injection, or configuration is generally preferable.

### 2.71 What does maintainable, well-tested Python look like in an AI or agent codebase, given that LLM outputs are non-deterministic?

The answer is to **separate what is deterministic from what is not** and test each appropriately: ordinary unit tests for the code around the model, and statistical evaluation suites for the model's behaviour. Normal engineering discipline (types, linting, CI, reviews, versioning) still applies, plus versioning of prompts, models and evaluation datasets.

**Layered testing:**

| Layer | What it covers | How |
| --- | --- | --- |
| Unit tests | Prompt templating, output parsers, tool argument validation, retrieval filters, retries, routing logic | pytest with the LLM client mocked or faked; fully deterministic and fast |
| Contract tests | Structured outputs match schemas; tools receive valid arguments | Pydantic models, JSON Schema; test parsers against recorded real responses including malformed ones |
| Recorded integration tests | The full chain against real responses | Record and replay HTTP (VCR-style cassettes) so CI is cheap and stable |
| Evaluation suite | Quality: correctness, faithfulness, tool-choice accuracy, safety | Curated dataset, metrics with thresholds, LLM-as-judge with calibrated rubrics, run on prompt or model changes |
| Production monitoring | Drift and regressions in real traffic | Tracing, sampled online evaluation, user feedback (see Chapter 24) |

```python
from pydantic import BaseModel

class Extraction(BaseModel):
    invoice_id: str
    total: float

class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
    def complete(self, prompt: str) -> str:
        return self.reply

def test_extractor_ignores_text_around_json():
    llm = FakeLLM('Sure! Here it is: {"invoice_id": "A1", "total": 12.5} Hope that helps.')
    result = extract_invoice(llm, "...")
    assert result == Extraction(invoice_id="A1", total=12.5)
```

**Code practices that make this possible:**

- Inject the model client behind a small interface (a `Protocol`), so tests pass a fake and you can switch providers.
- Keep prompts as versioned artefacts, not strings scattered through code; log the prompt version, model name and parameters with every call.
- Validate every model output at the boundary (Pydantic) and handle failures explicitly: retry with the validation error, fall back, or escalate.
- Set temperature low and fix seeds where the provider supports it, but do not rely on determinism; assert on properties (valid JSON, cites a retrieved document, correct tool chosen) rather than exact strings.
- Treat evaluation results like test results: thresholds in CI, tracked over time, with a regression blocking a release.

**Pitfalls:** exact-match assertions on generated text (flaky), calling live models in unit tests (slow, costly, flaky), and evaluation sets that are too small to detect a regression or that leak into few-shot examples. See Chapter 24 for evaluation methodology and Chapter 21 for agent testing.

### 2.72 What is free-threaded (no-GIL) CPython, what is its status, and should you adopt it?

Free-threaded CPython is a separate build of the interpreter, introduced by **PEP 703**, in which the GIL is removed so that threads can execute Python bytecode **in parallel on multiple cores**. It shipped as **experimental in Python 3.13** (October 2024), became **officially supported but opt-in in Python 3.14** (PEP 779, "phase II"), and Python 3.15 continues that path with a stable ABI for free-threaded extensions. Whether it becomes the default build ("phase III") has not been decided; the standard build with the GIL remains the default.

**How it works without the GIL:**

- **Biased reference counting:** the owning thread updates an object's refcount cheaply without atomics; other threads use a separate atomic shared count.
- **Immortal objects** (PEP 683): `None`, small ints and other common objects never have their refcounts changed, avoiding contention on hot objects.
- **Per-object locks** on built-in containers (`list`, `dict`) so concurrent operations do not corrupt them.
- **`mimalloc`** replaces pymalloc as the thread-safe allocator, and the cyclic GC stops the world briefly while it runs.
- The specialising adaptive interpreter was made thread-safe in 3.14, which is a large part of why single-threaded overhead fell.

**Using it:**

```python
# installed as a separate executable, e.g. python3.14t (uv: uv python install 3.14t)
import sys, sysconfig
from concurrent.futures import ThreadPoolExecutor

print(sysconfig.get_config_var("Py_GIL_DISABLED"))   # 1 on a free-threaded build
print(sys._is_gil_enabled())                          # False if running without the GIL

def score(chunk):                                     # pure-Python CPU work
    return sum(expensive_feature(x) for x in chunk)

with ThreadPoolExecutor(max_workers=8) as pool:       # now scales across cores
    results = list(pool.map(score, chunks))
```

- Importing a C extension that has not declared itself free-thread-safe **re-enables the GIL** for the process, with a warning; `PYTHON_GIL=0` or `-X gil=0` forces it off.
- Wheels for free-threaded builds carry a separate ABI tag (`cp314t`).

**Trade-offs:**

| Benefit | Cost |
| --- | --- |
| Parallel CPU-bound Python with threads, without pickling data between processes | Single-threaded code is slower than the standard build (the 3.14 docs put it at roughly 5 to 10 percent) and memory use is higher |
| Shared memory: one copy of a model, cache or lookup table instead of one per process | Ecosystem support is still uneven; every C extension in your stack must be audited and rebuilt |
| Simpler than multiprocessing for many workloads | **Latent race conditions** in Python code that relied on the GIL's accidental serialisation now surface |

**Who benefits:** CPU-bound pure-Python work that is hard to vectorise (tokenisation, feature engineering on nested records, simulation), data loaders feeding GPUs (PyTorch has been experimenting with thread-based workers), and inference servers wanting several threads sharing one model in memory. Workloads already dominated by NumPy, Polars or PyTorch kernels gain little, because those already release the GIL.

**Adoption guidance for a team:**

1. Check dependencies against the community compatibility trackers and test suites on a `3.14t` CI job.
2. Benchmark your real workload on both builds; do not assume a speed-up.
3. Audit shared mutable state: module-level caches, lazily initialised singletons, and check-then-act patterns need locks. The GIL never made `counter += 1` atomic, but it hid many races in practice.
4. Pilot on an internal batch job before latency-sensitive services, and keep the standard build as the fallback.

**What interviewers look for:** you know the GIL's removal is real but opt-in, you can explain the technical changes (biased refcounting, immortal objects, per-object locks), and you treat adoption as a dependency-and-correctness migration rather than a free performance win. Subinterpreters (`concurrent.interpreters`, 3.14) are the other route to parallelism, with isolation instead of shared objects.

### 2.73 How do you use asyncio correctly in production: tasks, TaskGroup, cancellation, timeouts and mixing sync with async code?

`asyncio` gives **single-threaded concurrency for I/O-bound work**: coroutines suspend at `await`, and the event loop runs other ready tasks meanwhile (see 2.48). Using it well in production means **structured concurrency** (every task has an owner that waits for it and sees its errors), explicit **timeouts and cancellation**, **bounded concurrency**, and never **blocking the event loop**. Framework-specific advice for FastAPI is in 25.33.

**Structured concurrency with `TaskGroup` (3.11+):**

```python
import asyncio, httpx

async def fetch(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> bytes:
    async with sem:                                    # bound concurrency
        async with asyncio.timeout(10):                # per-request deadline (3.11+)
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

async def fetch_all(urls: list[str]) -> list[bytes]:
    sem = asyncio.Semaphore(50)
    async with httpx.AsyncClient() as client:
        async with asyncio.TaskGroup() as tg:          # waits for all; cancels siblings on failure
            tasks = [tg.create_task(fetch(client, sem, u)) for u in urls]
    return [t.result() for t in tasks]

asyncio.run(fetch_all(urls))
```

- If any task fails, `TaskGroup` **cancels the others** and raises an `ExceptionGroup`, handled with `except*`.
- `asyncio.gather` is the older API: by default the first exception propagates but the **other tasks keep running**, which leaks work; `return_exceptions=True` collects errors instead. Prefer `TaskGroup` for new code.

**Mixing sync and async:**

| Situation | Tool |
| --- | --- |
| Calling a blocking library (boto3, requests, a DB-API driver, file I/O) from async code | `await asyncio.to_thread(func, *args)` or `loop.run_in_executor(pool, func)` |
| CPU-heavy work from async code | `run_in_executor` with a `ProcessPoolExecutor` |
| Calling async code from sync code at the top level | `asyncio.run(main())` once per program |
| Calling async code from another thread while a loop is running | `asyncio.run_coroutine_threadsafe(coro, loop)` |
| Running async code in Jupyter (a loop is already running) | `await coro` directly in the cell, not `asyncio.run` |

**Common bugs:**

1. **Blocking the loop:** `time.sleep`, `requests.get`, a heavy pandas transform or synchronous DB call inside `async def` stalls every other task. Detect with `asyncio.run(main(), debug=True)` (logs slow callbacks) or `PYTHONASYNCIODEBUG=1`.
2. **Forgetting `await`:** calling `fetch()` without `await` creates a coroutine that never runs ("coroutine was never awaited" warning).
3. **Fire-and-forget tasks disappearing:** the loop holds only weak references to tasks, so a task from `create_task` with no stored reference can be garbage-collected mid-flight; keep references or use a `TaskGroup`.
4. **Swallowing cancellation:** catching `asyncio.CancelledError` (or a broad `except BaseException`) without re-raising breaks timeouts and shutdown.
5. **Unbounded fan-out:** creating 100,000 tasks at once overwhelms the target API and file descriptors; use a semaphore or a fixed pool of worker tasks reading from an `asyncio.Queue`.
6. **Sharing clients incorrectly:** create one `AsyncClient` or connection pool per process and reuse it; creating one per request is slow, and using one across event loops fails.
7. **Mixing loops and threads:** asyncio objects are not thread-safe; use `call_soon_threadsafe` to talk to a loop from another thread.

**Graceful shutdown:** on SIGTERM, stop accepting new work, cancel or drain in-flight tasks with a deadline, then close clients. `asyncio.Runner` and signal handlers via `loop.add_signal_handler` help. Python 3.14 adds `python -m asyncio ps PID` and `pstree` to inspect the tasks of a running process, useful for debugging hangs.

**When not to use asyncio:** CPU-bound work, codebases built on sync libraries with no async equivalent, or modest concurrency where a `ThreadPoolExecutor` is simpler. Async is viral: once the core is async, callers must be too.

### 2.74 How do you manage Python packaging, dependencies and environments reproducibly (pyproject.toml, uv, lock files)?

A reproducible Python project declares its metadata and **direct dependencies with ranges** in `pyproject.toml`, pins the **full resolved dependency tree with hashes** in a lock file, and builds environments and container images only from that lock file. In 2026 the common toolchain is **uv** (from Astral, written in Rust) for Python versions, virtual environments, resolution, locking and running, with Poetry, PDM, Hatch and pip-tools as alternatives, and conda or Pixi where non-Python native dependencies dominate.

**The pieces:**

| Piece | Role |
| --- | --- |
| `pyproject.toml` (PEP 518, 621) | Single file for project metadata, dependencies, optional extras, dependency groups (PEP 735) and tool configuration (Ruff, mypy, pytest) |
| Build backend (hatchling, setuptools, uv_build, maturin for Rust) | Turns source into a **wheel**; declared in `[build-system]` |
| Lock file (`uv.lock`, `poetry.lock`, or the standard `pylock.toml` from PEP 751) | Exact versions and hashes of every transitive dependency for every platform |
| Virtual environment (`.venv`) | Isolated interpreter and site-packages per project |
| Python version pin (`.python-version`, `requires-python`) | Same interpreter everywhere |

```text
[project]
name = "feature-pipeline"
version = "1.4.0"
requires-python = ">=3.12"
dependencies = [
    "polars>=1.20,<2",
    "pydantic>=2.7,<3",
    "boto3>=1.34",
]

[project.optional-dependencies]
gpu = ["torch>=2.5"]

[dependency-groups]
dev = ["pytest>=8", "hypothesis>=6", "ruff", "mypy"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```bash
uv init && uv add polars pydantic      # edit pyproject and update uv.lock
uv sync --locked                       # CI: install exactly the lock; fail if it is stale
uv run pytest                          # run inside the project environment
uv lock --upgrade-package boto3        # controlled upgrade of one dependency
```

**Libraries versus applications:**

- A **library** published for others declares **broad version ranges** and never pins exact versions, or it will conflict with other packages in the user's environment.
- An **application or pipeline** (a service, a DAG image, a model server) **locks everything** and deploys from the lock.

**Reproducible builds and deployments:**

- Install with hashes (`uv sync --locked`, `pip install --require-hashes`) so a compromised or re-uploaded package cannot slip in.
- Build Docker images in two stages: install locked dependencies in a layer cached on the lock file, then copy the source; use a slim base image and pin the base image digest.
- Pin the Python version and, for GPU stacks, the CUDA and driver versions alongside PyTorch wheels (the PyTorch index URL encodes the CUDA version).
- Publish internal packages to a private index (Artifactory, CodeArtifact) and configure the index explicitly to avoid **dependency confusion** attacks.
- Automate upgrades with Renovate or Dependabot and scan with `pip-audit`; an untouched lock file slowly accumulates vulnerabilities.

**Pitfalls:** `pip freeze > requirements.txt` captures one machine's environment, misses platform differences and mixes direct with transitive dependencies; installing into the system or base conda environment; notebooks with `!pip install` cells; and lock files that are generated on macOS but deployed to Linux without cross-platform resolution (uv and Poetry resolve universally to avoid this).

**What interviewers look for:** you can explain why "works on my machine" happens, the difference between declaring and locking dependencies, and how you would move a team from ad hoc `requirements.txt` files to locked, hash-verified builds without disrupting delivery.

### 2.75 How do you use type hints at scale: mypy or Pyright, Protocols, generics, TypedDict, and runtime validation with Pydantic?

Type hints are **ignored by the interpreter at run time** but checked statically by **mypy** or **Pyright**, catching whole classes of bugs (`None` handling, wrong argument types, misspelt attributes) before code runs and making large codebases navigable. Static types do not validate external data, so at system boundaries you add **runtime validation**, typically with **Pydantic**.

**Core tools beyond basic annotations:**

```python
from typing import Protocol, TypedDict, NotRequired, Literal, Callable
from dataclasses import dataclass

class Reader(Protocol):                       # structural typing: anything with read() fits
    def read(self, path: str) -> bytes: ...

class EventDict(TypedDict):                   # typed shape for dicts, e.g. parsed JSON
    user_id: int
    kind: Literal["click", "view"]
    referrer: NotRequired[str]

def first[T](items: list[T]) -> T | None:     # generic function, PEP 695 syntax (3.12+)
    return items[0] if items else None

@dataclass(frozen=True, slots=True)
class Partition:
    table: str
    date: str

def run(reader: Reader, on_done: Callable[[Partition], None]) -> None: ...
```

| Feature | Use |
| --- | --- |
| `X \| None` and narrowing | Forces handling of missing values; the checker narrows after `if x is None: return` |
| `Protocol` | Interfaces without inheritance; ideal for dependency injection and test fakes |
| Generics (`TypeVar`, `class Repo[T]`) | Reusable containers and functions that keep element types |
| `TypedDict` | Dict-shaped data (JSON, kwargs) when you cannot use a class |
| `Literal`, `Enum` | Restrict values to a fixed set |
| `ParamSpec`, `Concatenate` | Decorators that preserve signatures (see 2.6) |
| `@overload` | Functions whose return type depends on arguments |
| `Final`, `@final`, `ReadOnly` | Prevent reassignment and overriding |

**mypy versus Pyright:** mypy is the reference checker with a plugin system (Pydantic and Django plugins). Pyright is faster, powers VS Code's Pylance, and infers more aggressively. Newer Rust-based checkers (Astral's ty, Meta's Pyrefly) aim at very large monorepos. Pick one for CI and match the editor to it.

**Runtime validation at boundaries:**

```python
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class Event(BaseModel):
    user_id: int
    amount: float = Field(ge=0)
    kind: Literal["click", "view"]

try:
    event = Event.model_validate_json(raw_message)   # parses, coerces, validates
except ValidationError as e:
    send_to_dead_letter(raw_message, str(e))
```

Use Pydantic (or `msgspec`, `attrs` with validators) for API payloads, config (`pydantic-settings`), messages from queues and LLM structured outputs; use **plain dataclasses** internally where data is already trusted, because validation has a cost. For DataFrames, schema libraries such as **Pandera** validate columns and dtypes.

**Rolling it out on a large codebase:**

1. Start with `mypy --strict` (or Pyright strict) on new packages only; set a lenient global baseline.
2. Annotate public function signatures first; inference fills in much of the rest.
3. Use stub packages (`types-requests`, `pandas-stubs`) and `py.typed` markers for internal libraries.
4. Ratchet: forbid new `Any` and `type: ignore` without a reason code, and track type coverage.
5. Avoid over-engineering: precise types for core domain objects, `Any` at messy edges is acceptable if contained.

**Pitfalls:** believing hints validate data at run time (they do not); `Any` spreading silently through a codebase; typing pandas DataFrames, where static checkers know little about columns; and circular imports created by type-only imports (use `from __future__ import annotations` or `if TYPE_CHECKING:`).

### 2.76 How do you find and fix memory growth or leaks in a long-running Python service or worker?

In Python a "leak" is almost always **objects that are still referenced** (unbounded caches, growing globals, references held by closures, tasks or exception tracebacks), **memory held by native libraries**, or **allocator fragmentation** that keeps the process's resident memory (RSS) high even after objects are freed. The approach is to confirm the growth pattern, find which allocations grow, then decide whether it is a true leak or a high-water mark.

**1. Characterise the growth:**

- Plot RSS over time and against request count. Unbounded linear growth suggests a leak; growth that plateaus after warm-up suggests caches or fragmentation.
- Compare `tracemalloc` totals (Python allocations) with RSS. If Python allocations are flat while RSS grows, the problem is native (NumPy, PyTorch, Arrow, C extensions) or fragmentation.

**2. Find what is growing:**

```python
import tracemalloc, gc
from collections import Counter

tracemalloc.start(25)                        # keep 25 frames per allocation
before = tracemalloc.take_snapshot()
handle_many_requests()
after = tracemalloc.take_snapshot()
for stat in after.compare_to(before, "traceback")[:10]:
    print(stat)                               # allocation sites that grew most

gc.collect()
print(Counter(type(o).__name__ for o in gc.get_objects()).most_common(15))
```

| Tool | Strength |
| --- | --- |
| `tracemalloc` | Standard library; diff snapshots to find growing allocation sites |
| **memray** | Tracks Python and native allocations, flame graphs, can attach to a live process |
| `objgraph`, `gc.get_referrers` | Show what is keeping a leaked object alive |
| `pympler` | Object size summaries over time |
| `py-spy dump` | See what a stuck or bloated process is doing right now |

**3. Common causes and fixes:**

| Cause | Fix |
| --- | --- |
| Unbounded module-level dict or list used as a cache | `functools.lru_cache(maxsize=...)`, `cachetools.TTLCache`, or an external cache (Redis) |
| `lru_cache` on instance methods keeps every `self` alive | Cache on a module-level function keyed by IDs, or use `weakref` |
| Event handlers or callbacks registered and never removed | Unregister, or hold listeners with `weakref.WeakSet` / `WeakMethod` |
| Stored exceptions keep tracebacks, which keep every frame's locals (including large DataFrames) alive | Log and drop the exception; store `str(e)` not the object |
| asyncio tasks or queues that grow faster than they drain | Bound queues, apply backpressure, cancel abandoned tasks |
| ML frameworks: tensors kept in a list for logging hold GPU memory and the autograd graph | Log `loss.item()`, not `loss`; use `torch.no_grad()` for inference |
| pandas: chained intermediate DataFrames referenced in a long-lived scope | Restructure into functions so intermediates go out of scope; `del` large objects |
| Reference cycles with finalisers or large objects waiting for the cyclic GC | Break cycles, use `weakref`, or `gc.collect()` after large batch completions |
| Fragmentation after a peak (many small objects freed in a scattered pattern) | Process data in bounded chunks, use columnar buffers, or recycle the worker |

**4. Operational safeguards:**

- **Recycle workers** after N requests or tasks (`gunicorn --max-requests` with jitter, Celery `worker_max_tasks_per_child`, `maxtasksperchild` in multiprocessing); this caps the damage of slow leaks and fragmentation.
- Set container memory limits with alerts well below the OOM kill threshold, and export RSS and GC metrics.
- For pre-fork servers, call `gc.freeze()` after loading models in the parent so child processes do not touch shared pages and trigger copy-on-write duplication.
- Tune the allocator for native-heavy services (for example `jemalloc` via `LD_PRELOAD`, or `MALLOC_ARENA_MAX` for glibc with many threads).

**Version note:** Python 3.14.0 shipped an incremental cyclic garbage collector to reduce pause times, but it was reverted to the generational collector in 3.14.5 after reports of higher memory use in production. GC behaviour does change between versions, so load-test memory after interpreter upgrades.

**What interviewers look for:** a methodical approach (measure, diff, find referrers), knowing that RSS not shrinking is not necessarily a leak, and pragmatic mitigations while the root cause is found.

### 2.77 What is your testing strategy for Python code: pytest fixtures, parametrisation, mocking and property-based testing with Hypothesis?

A good strategy is a **pyramid**: many fast, deterministic unit tests on pure logic, fewer integration tests against real dependencies (databases, object stores in containers), and a handful of end-to-end tests, all run automatically in CI. **pytest** is the standard framework; **Hypothesis** adds property-based tests that find edge cases you did not think of. Data-pipeline specifics (data quality tests, dbt tests, CI/CD for DAGs) are covered in 7.22.

**Fixtures** provide setup with explicit dependencies and scopes:

```python
import pytest, polars as pl

@pytest.fixture
def orders() -> pl.DataFrame:                       # fresh for each test
    return pl.DataFrame({"id": [1, 2, 3], "amount": [10.0, -5.0, 0.0]})

@pytest.fixture(scope="session")
def warehouse():                                    # expensive: once per test run
    with start_postgres_container() as db:          # e.g. testcontainers
        yield db                                    # teardown after the yield

def test_drops_negative_amounts(orders):
    assert clean(orders)["amount"].min() >= 0
```

- Built-in fixtures: `tmp_path` for files, `monkeypatch` for environment variables and attributes, `caplog` for logs, `capsys` for output.
- Put shared fixtures in `conftest.py`; keep them small and composable rather than one giant setup.

**Parametrisation** covers many cases without copy-paste:

```python
@pytest.mark.parametrize("raw, expected", [
    ("2024-06-01", date(2024, 6, 1)),
    ("01/06/2024", date(2024, 6, 1)),
    ("", None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected
```

**Property-based testing with Hypothesis** generates hundreds of inputs, including nasty ones (empty strings, NaN, Unicode, extreme numbers), and **shrinks** any failure to a minimal example:

```python
from hypothesis import given, strategies as st

@given(st.lists(st.dictionaries(st.sampled_from(["id", "v"]), st.integers())))
def test_dedupe_is_idempotent(records):
    once = dedupe(records)
    assert dedupe(once) == once              # property: applying twice changes nothing
    assert len(once) <= len(records)         # property: never adds rows
```

Good properties for data code: round-trips (`decode(encode(x)) == x`), idempotence, invariants (row counts, sums preserved by a reshuffle), and comparison against a slow but obviously correct reference implementation. `hypothesis.extra.pandas` generates DataFrames.

**Mocking guidelines:**

- Mock at **system boundaries** (HTTP APIs, cloud SDKs, LLM clients, clocks), not your own internal functions; over-mocking tests the implementation, not the behaviour.
- Prefer **fakes** (an in-memory repository implementing the same `Protocol`) and dependency injection to `mock.patch` on import paths.
- Use `moto` for AWS, `respx` or `responses` for HTTP, `freezegun` or `time-machine` for time, and real databases in containers for SQL behaviour, since SQLite is not Postgres.
- Use `autospec=True` so mocks fail when the real signature changes.

**Keeping the suite healthy:**

- Fast feedback: unit tests in seconds, `pytest-xdist` for parallelism, markers (`@pytest.mark.integration`) to split CI stages.
- Coverage as a signal, not a target; mutation testing (`mutmut`) on critical code reveals weak assertions.
- Quarantine and fix flaky tests quickly; a flaky suite gets ignored.
- Golden-file or snapshot tests (`syrupy`) for complex outputs, reviewed on change.
- For ML code, test the deterministic parts (feature functions, shapes, no leakage between train and test splits, a tiny model can overfit a small batch) and evaluate model quality separately (see 2.71 and Chapter 24).

**What interviewers look for:** you test behaviour rather than implementation, know where mocks help and hurt, can explain what property-based testing finds that example tests miss, and treat test speed and reliability as part of developer productivity.

---

← [Chapter 1. Mathematics, Probability and Statistics](01-mathematics-probability-and-statistics.md) · [Table of contents](../README.md) · [Chapter 3. Computer Science and Software Design](03-computer-science-and-software-design.md) →
