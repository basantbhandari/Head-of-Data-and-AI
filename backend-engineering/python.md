# Python

Python interview Q&A: CPython internals (memory, objects, scoping, descriptors, import system, concurrency) and Python for data engineering.

## Reference links

- [https://datavidhya.com/blog/python-data-engineering-interview-questions/](https://datavidhya.com/blog/python-data-engineering-interview-questions/)
- [https://medium.com/@gulsaba.fiha/15-advanced-python-interview-questions-to-crack-that-mid-senior-role-with-code-chuckles-0522eaa154b6](https://medium.com/@gulsaba.fiha/15-advanced-python-interview-questions-to-crack-that-mid-senior-role-with-code-chuckles-0522eaa154b6)
- [https://github.com/nileshhadalgi016/150-python-interview-questions-and-answers](https://github.com/nileshhadalgi016/150-python-interview-questions-and-answers)

## CPython internals and language semantics

### Q1. GIL in Python. Why?

- The **Global Interpreter Lock (GIL)** is a mutex (lock) in the standard CPython interpreter that **ensures only one thread executes Python bytecode at a time**.
- Because **CPython’s memory management is not thread-safe**

<!-- merged from duplicate question: "What is the Global Interpreter Lock (GIL), and why does CPython have one?" -->

| Aspect | Explanation |
| --- | --- |
| **GIL** | The **Global Interpreter Lock (GIL)** is a lock in the traditional CPython execution model that allows only **one thread at a time to execute Python bytecode within a process**. |
| **Why CPython has it** | Historically, it greatly simplified CPython's memory management, especially **reference counting**, by preventing multiple threads from modifying object reference counts simultaneously. |
| **Threading impact** | Multiple threads can exist and run concurrently, but CPU-bound Python bytecode generally cannot execute in parallel across threads in a traditional GIL-enabled CPython process. |
| **I/O-bound tasks** | Threads can still be very useful because CPython can release the GIL while waiting for many I/O operations, allowing another thread to run Python code. |
| **CPU-bound tasks** | Threads generally do not provide true CPU parallelism for Python bytecode under the traditional GIL model. Multiprocessing or other approaches can provide parallelism. |
| **C extensions** | Native C/C++ extensions can release the GIL while performing work, allowing other Python threads to execute. |
| **Modern CPython nuance** | **Free-threaded CPython builds** became available experimentally in Python 3.13, allowing the GIL to be disabled. This is an alternative build/runtime mode rather than the historical default model. |
| **Interview answer** | **The GIL is a CPython lock that traditionally allows only one thread to execute Python bytecode at a time per process. It simplified CPython's memory management and reference-counting implementation, while still allowing useful concurrency for I/O-bound workloads.** |

### Q2. Access Modifier in python

| Aspect | Access Modifier in Python |
| --- | --- |
| **Meaning** | Controls or indicates how class attributes and methods should be accessed. |
| **Python approach** | Python does **not have strict access modifiers** like Java or C++ (`public`, `private`, `protected`). It mainly uses naming conventions and name mangling. |
| **Public** | `name` — accessible from anywhere. |
| **Protected (convention)** | `_name` — indicates “internal/protected” use; Python does not enforce it. |
| **Private (name mangling)** | `__name` — Python transforms it roughly to `_ClassName__name`, making accidental external access harder. |
| **Example** | `class User: __password = "..."` |
| **Name mangling** | `user.__password` normally fails, but `user._User__password` can access it. Therefore, it is **not true security/privacy**. |
| **Best practice** | Use `_name` for internal APIs and `__name` when you specifically want name-mangling to avoid subclass/name collisions. |
| **Interview answer** | **Python doesn't enforce public/protected/private access modifiers. It uses naming conventions (`name`, `_name`) and name mangling (`__name`) to communicate or restrict attribute access.** |

### Q3. What's the difference between `is` and `==`? How does CPython implement identity checks?

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

### Q4. How does reference counting work in CPython? What happens when an object's refcount hits zero?

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

### Q5. How does Python's garbage collector handle reference cycles?

| Aspect | Explanation |
| --- | --- |
| **Problem** | Reference counting alone cannot reclaim objects involved in a cycle because each object still has at least one reference from another object in the cycle. |
| **Example** | `a → b` and `b → a`. Even if no external references to `a` or `b` remain, their reference counts are not zero. |
| **How Python handles it** | CPython uses a **cyclic garbage collector (GC)** in addition to reference counting to detect and collect unreachable reference cycles. |
| **Detection** | The GC tracks container objects that can contain references to other objects, such as `list`, `dict`, and custom objects. It determines which groups of objects are unreachable from the program's roots. |
| **Generational approach** | CPython's cyclic GC groups tracked objects by generations and focuses collection work on younger objects more frequently. |
| **Collection** | Once a cycle is determined to be unreachable, the GC can break the cycle and allow the objects to be deallocated. |
| **`__del__` consideration** | Objects with finalizers (`__del__`) require special handling. Modern CPython can safely finalize many cyclic objects, but finalizers can still complicate collection. |
| **Reference counting vs GC** | **Reference counting** handles most objects immediately when their count reaches zero; **cyclic GC** handles unreachable cycles that reference counting alone cannot reclaim. |
| **Interview answer** | **CPython primarily uses reference counting, but reference cycles keep their counts above zero. Its cyclic garbage collector detects unreachable groups of container objects and collects those cycles, preventing them from leaking indefinitely.** |

### Q6. What's the difference between `__new__` and `__init__`?

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

<!-- merged from duplicate question: "What is the `__init__` method and how does it differ from `__new__`?" -->

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

### Q7. Why is small integer caching (-5 to 256) used, and how does it affect `is` comparisons?

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

### Q8. What's object interning, and how does it apply to strings?

| Aspect | Explanation |
| --- | --- |
| **Object interning** | A technique where Python/CPython stores a single canonical object for certain immutable values and reuses that object instead of creating duplicates. |
| **Strings** | CPython commonly interns certain strings, especially **identifier-like strings** such as variable, function, and attribute names. Some string literals may also be interned by the compiler. |
| **Example** | `a = "hello"` and `b = "hello"` may refer to the **same string object** because `"hello"` can be interned. |
| **`sys.intern()`** | You can explicitly request interning: `a = sys.intern("some_repeated_string")`. |
| **Memory benefit** | Repeated identical interned strings can share one object, reducing memory usage. |
| **Performance benefit** | When both strings are interned, identity checks can sometimes be used internally, potentially making comparisons faster. |
| **Why strings can be interned safely** | Strings are **immutable**, so sharing one object cannot cause one part of a program to modify it for another part. |
| **Important distinction** | **Interning is not the same as string equality.** Two equal strings do not necessarily have the same identity unless they happen to be interned/shared. |
| **`is` vs `==`** | Use `==` for string value comparison. **Do not use `is`** to compare string contents, even if interning sometimes makes `is` appear to work. |
| **Interview answer** | **Object interning reuses a single canonical object for identical immutable values. For strings, CPython interns many identifier-like strings and allows explicit interning with `sys.intern()`, reducing memory usage and potentially improving repeated comparisons.** |

### Q9. How does `sys.getrefcount()` work, and why is its result often higher than expected?

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

### Q10. What's the difference between mutable and immutable objects at the C level?

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

### Q11. What is `__slots__`, and how does it change an object's memory layout?

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

### Q12. How does Python implement variable scoping (LEGB rule) internally?

| Aspect | Internal implementation |
| --- | --- |
| **LEGB** | Python resolves an unqualified name in the order **Local → Enclosing → Global → Built-ins**. |
| **Local (L)** | Function-local variables are typically stored in **fast local slots** associated with the executing frame. |
| **Enclosing (E)** | Variables captured by nested functions are stored in **cell objects**; the inner function accesses them as free variables. |
| **Global (G)** | Global names are looked up in the function's `__globals__` dictionary, normally the module's namespace. |
| **Built-ins (B)** | If not found globally, Python checks the built-in namespace associated with the execution environment. |
| **Compiler's role** | Python's compiler determines whether a name is local, free, cell, or global and emits appropriate bytecode. |
| **Runtime's role** | The interpreter executes those bytecode operations to retrieve/store the name. |

### Q13. What's the difference between a function's `__globals__` and `__locals__`?

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

### Q14. How do closures capture variables — by reference or by value?

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

### Q15. What is a cell object, and how does it enable closures?

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

### Q16. How does the `nonlocal` keyword work under the hood?

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

### Q17. What is a code object (`__code__`), and what does it contain?

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

### Q18. How does Python compile source code into bytecode? What role does `dis` play?

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

### Q19. What's the difference between a function and a method at the descriptor level?

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

### Q20. How do decorators work internally — what's actually being replaced?

| Aspect | What happens internally |
| --- | --- |
| **Decorator** | A callable that receives a function/class and returns a replacement object, usually another function. |
| **`@decorator` syntax** | `@decorator` above `def f(): ...` is essentially syntactic sugar for `f = decorator(f)` after the function is created. |
| **What's replaced?** | The **name bound to the original function** is replaced with whatever the decorator returns. |
| **Original function** | Usually captured inside the wrapper through a closure. |
| **Function object** | The original function isn't necessarily modified; the name `f` simply points to the decorated result. |
| **Calling `f()` afterward** | Calls the replacement/wrapper object, which can call the original function. |

#### Example

```py
def logger(func):
    def wrapper(*args, **kwargs):
        print("Calling function")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    return a + b
```

Conceptually, Python transforms this into:

```py
def add(a, b):
    return a + b

add = logger(add)
```

| Before decoration | After decoration |
| --- | --- |
| `add` → original function | `add` → `wrapper` |
| `add.__name__` → `"add"` | `add.__name__` → `"wrapper"` unless `functools.wraps` is used |
| Calling `add()` → original function | Calling `add()` → `wrapper()` → original function |

#### With `functools.wraps`

```py
from functools import wraps

def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Calling function")
        return func(*args, **kwargs)
    return wrapper
```

| Without `wraps` | With `wraps` |
| --- | --- |
| Metadata can look like the wrapper | Copies important metadata from original function |
| `__name__` may become `"wrapper"` | `__name__` remains `"add"` |
| `__doc__` may be lost | Original docstring is preserved |

#### Multiple decorators

```py
@A
@B
def f():
    pass
```

is equivalent to:

```py
f = A(B(f))
```

| Order | Operation |
| --- | --- |
| 1 | Create original `f` |
| 2 | Apply `B` → `f = B(f)` |
| 3 | Apply `A` → `f = A(f)` |

### Q21. How does the descriptor protocol (`__get__`, `__set__`, `__delete__`) work?

| Aspect | Explanation |
| --- | --- |
| **Descriptor** | An object that defines how an attribute is accessed, assigned, or deleted through `__get__`, `__set__`, and/or `__delete__`. |
| **Where defined** | Usually placed as a **class attribute**. |
| **`__get__()`** | Controls what happens when the attribute is **read**. |
| **`__set__()`** | Controls what happens when the attribute is **assigned**. |
| **`__delete__()`** | Controls what happens when the attribute is **deleted**. |
| **Main purpose** | Customize attribute access without requiring the caller to explicitly invoke methods. |

#### Basic descriptor

```py
class Descriptor:
    def __get__(self, instance, owner):
        return "value"

    def __set__(self, instance, value):
        print("setting:", value)

    def __delete__(self, instance):
        print("deleting")
```

```py
class User:
    name = Descriptor()

u = User()

u.name = "Alice"   # __set__
print(u.name)      # __get__
del u.name         # __delete__
```

| Operation | Descriptor method | Conceptual call |
| --- | --- | --- |
| `u.name` | `__get__` | `User.__dict__["name"].__get__(u, User)` |
| `u.name = "Alice"` | `__set__` | `User.__dict__["name"].__set__(u, "Alice")` |
| `del u.name` | `__delete__` | `User.__dict__["name"].__delete__(u)` |
| `User.name` | `__get__` | `descriptor.__get__(None, User)` |

### Q22. How are default mutable arguments evaluated, and why does this cause bugs?

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

### Q23. What is `__call__`, and how does it make objects callable?

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

### Q24. How does the Method Resolution Order (MRO) work, and what algorithm does it use (C3 linearization)?

| Aspect | Explanation |
| --- | --- |
| **MRO** | Method Resolution Order is the ordered sequence Python uses to search for attributes/methods in a class hierarchy. |
| **Why needed** | It determines which implementation is selected when multiple inheritance creates several possible paths. |
| **Algorithm** | Python uses **C3 linearization** to construct the MRO. |
| **Main properties** | C3 preserves **local precedence order**, **monotonicity**, and avoids inconsistent inheritance orders. |
| **Inspection** | Use `Class.mro()` or `Class.__mro__`. |
| **`super()` relationship** | `super()` follows this MRO rather than simply choosing the direct parent. |

### Q25. What is a metaclass, and how does `type` fit in?

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

### Q26. How does `super()` actually resolve which method to call?

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

### Q27. What's stored in a class's `__dict__` vs an instance's `__dict__`?

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

### Q28. How does multiple inheritance affect attribute lookup?

| Aspect | Multiple inheritance behavior |
| --- | --- |
| **Basic mechanism** | Python searches attributes according to the class's **MRO (Method Resolution Order)**. |
| **MRO** | A linear ordering of the class and its parent classes that determines lookup order. |
| **Lookup start** | For `obj.attr`, Python first checks the object's class and then follows its MRO. |
| **Multiple parents** | Python combines the parent classes into a consistent MRO using the **C3 linearization** algorithm. |
| **First match** | The first class in the MRO containing the requested attribute generally provides it. |
| **`super()`** | Uses the **next class in the MRO**, not simply the direct parent. |
| **Diamond inheritance** | MRO ensures a shared ancestor is normally visited only once. |

#### Example

```py
class A:
    def show(self):
        return "A"

class B(A):
    def show(self):
        return "B"

class C(A):
    def show(self):
        return "C"

class D(B, C):
    pass
```

| Expression | Result |
| --- | --- |
| `D.mro()` | `[D, B, C, A, object]` |
| `D().show()` | `"B"` |
| Why? | `D` has no `show`, then Python finds it in `B` first |

#### `super()` with multiple inheritance

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

| Call | Lookup sequence |
| --- | --- |
| `D().show()` | `D → B → C → A → object` |
| `D.show()` | Finds `D.show` |
| `super()` inside `D` | Continues at `B` |
| `super()` inside `B` | Continues at `C`, **not directly to `A`** |
| `super()` inside `C` | Continues at `A` |

### Q29. What is the difference between `@staticmethod`, `@classmethod`, and instance methods internally?

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

### Q30. What's the difference between an iterable and an iterator at the protocol level (`__iter__`, `__next__`)?

| Aspect | **Iterable** | **Iterator** |
| --- | --- | --- |
| **Protocol** | Implements `__iter__()` | Implements `__iter__()` **and** `__next__()` |
| **Purpose** | Provides a way to obtain an iterator | Produces the next item |
| **`__iter__()`** | Returns an iterator | Usually returns `self` |
| **`__next__()`** | Not required | Required |
| **State** | Usually represents the collection/source | Maintains the current iteration position |
| **Can be restarted?** | Usually yes; each `iter()` can produce a new iterator | Usually no; once exhausted, it stays exhausted |
| **Used by `for`** | `for` calls `iter(iterable)` | Then repeatedly calls `next(iterator)` |
| **Example** | `list`, `tuple`, `str`, `dict` | `iter(list)`, generator objects |
| **Key relationship** | `iterable → iterator` | `iterator → next item` |

#### Protocol flow

| Operation | What Python does |
| --- | --- |
| `for x in obj:` | Calls `iter(obj)` |
| `iter(obj)` | Calls `obj.__iter__()` |
| `iterator = iter(obj)` | Gets an iterator |
| Each loop iteration | Calls `next(iterator)` |
| More items available | `__next__()` returns the next value |
| No items left | `__next__()` raises `StopIteration` |

#### Example

```txt
numbers= [1,2,3]it=iter(numbers)next(it)# 1next(it)# 2next(it)# 3next(it)# StopIteration
```

| Object | `__iter__()` | `__next__()` | Role |
| --- | --- | --- | --- |
| `numbers` (`list`) | Yes | No | Iterable |
| `it` (`list_iterator`) | Yes | Yes | Iterator |

**Interview answer:** An **iterable** implements `__iter__()` to provide an iterator, while an **iterator** implements both `__iter__()` and `__next__()` to produce values one at a time.

### Q31. How do generators maintain state between `yield` calls?

| Aspect | How generators maintain state |
| --- | --- |
| **Generator object** | Calling a generator function creates a **generator object** without executing the function body. |
| **Execution state** | The generator stores its current execution state, including where execution should resume. |
| **Local variables** | Local variables remain associated with the suspended generator, so their values survive across `yield` calls. |
| **`yield`** | When `yield` executes, the generator **pauses** and returns a value to the caller. |
| **`next()`** | Resumes execution from exactly where it previously paused. |
| **`.send(value)`** | Resumes execution and makes the suspended `yield` expression evaluate to `value`. |
| **Example** | `x = yield 10` → after `yield 10`, the generator pauses. `send(42)` resumes it and `x` becomes `42`. |
| **Internal idea** | CPython preserves the generator's **execution frame/interpreter state**, including the instruction position and local-variable state. |
| **After resumption** | Execution continues until another `yield`, `return`, or exception. |
| **Completion** | Once the generator returns or raises an unhandled exception, its state cannot be resumed again. |

#### Example

| Step | Generator state |
| --- | --- |
| `g = gen()` | Generator created; body hasn't run |
| `next(g)` | Runs until `yield`; state is saved |
| `g.send(42)` | Resumes from saved point; suspended `yield` becomes `42` |
| Next `yield` | State is saved again |
| `next(g)` | Resumes from the new saved point |

### Q32. What happens internally when you call `.send()` on a generator?

| Aspect | What happens when `generator.send(value)` is called |
| --- | --- |
| **Purpose** | `.send(value)` **resumes a suspended generator** and sends `value` into the generator at its current `yield` expression. |
| **Generator state** | The generator must already be **suspended at a `yield`**. |
| **Example** | `def gen(): x = yield "hello" yield x` |
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

### Q33. How do `async def` and `await` differ from generators under the hood?

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

### Q34. What is the event loop, and how does it schedule coroutines?

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

### Q35. How is a Python `list` implemented internally (array vs linked list)?

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

### Q36. How does a `dict` achieve O(1) average lookup? How are hash collisions handled?

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

### Q37. Why did dict ordering become guaranteed in Python 3.7+? What changed internally?

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

### Q38. How is a `set` different from a `dict` at the implementation level?

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

### Q39. How does string immutability enable optimizations like interning?

| Aspect | Explanation |
| --- | --- |
| **String immutability** | Once a Python string object is created, its characters **cannot be changed in place**. |
| **Why this enables interning** | Because an interned string can never be modified, Python can safely let multiple references point to the **same string object**. |
| **Example** | `a = sys.intern("customer_id")b = sys.intern("customer_id")a is b` → `True` |
| **What would happen with mutable strings?** | If strings could be modified, changing `a` could unexpectedly change `b` if both referenced the same object. Sharing would therefore be unsafe. |
| **Memory optimization** | Many references to the same interned string can share **one object**, reducing duplicate string objects. |
| **Comparison optimization** | Interned strings can sometimes be compared using **object identity** (`is`) after interning, avoiding character-by-character comparison. |
| **Hashing benefit** | Strings are immutable and hashable, so their hash can be safely cached/reused, making them efficient as dictionary keys and set elements. |
| **Dictionary example** | `{"customer_id": 123}` can efficiently use the immutable string `"customer_id"` as a key because its value and hash cannot change. |
| **Core principle** | **Immutability makes object sharing safe.** Python can safely reuse the same string object because nobody can modify its contents. |
| **Important distinction** | Immutability **enables** interning; it does not mean every Python string is automatically interned. CPython automatically interns some strings, while `sys.intern()` lets you request interning explicitly. |
| **Interview answer** | **String immutability makes interning safe because a shared string object cannot be modified by one reference and unexpectedly affect another. Therefore, Python can reuse a single string object for identical values, saving memory and potentially making comparisons faster.** |

### Q40. What happens internally when you run `import module`?

| Step | Internal process |
| --- | --- |
| **1. Execute import statement** | Python's import machinery is invoked for `module`. |
| **2. Check `sys.modules`** | Python first checks whether `"module"` already exists in the in-memory module cache. |
| **3. If cached** | If present, Python reuses the existing module object. **The module's top-level code is not executed again.** |
| **4. If not cached** | Python asks the import system to find the module. |
| **5. Search** | Finders search locations such as `sys.path` for `module.py`, a package, extension module, etc. |
| **6. Create `ModuleSpec`** | The finder returns a `ModuleSpec` describing the module and its loader. |
| **7. Create module object** | The loader creates the module object using the specification. |
| **8. Add to `sys.modules`** | The module is inserted into `sys.modules` **before its code executes**. This helps support circular imports. |
| **9. Execute module** | The loader executes the module's top-level code. Function/class definitions are created, variables initialized, and nested imports processed. |
| **10. Bytecode handling** | For a `.py` module, CPython can use an existing valid `.pyc` or compile the source to bytecode and potentially cache it in `__pycache__`. |
| **11. Complete import** | The initialized module object remains in `sys.modules`. |
| **12. Bind name** | With `import module`, the name `module` is bound in the current namespace to that module object. |
| **Simplified flow** | `import module` → `sys.modules` check → find → create → cache → execute → return/bind |
| **If import fails** | If execution raises an exception, the import machinery removes the unsuccessfully initialized module from `sys.modules` as appropriate, and the exception propagates. |
| **Important distinction** | `sys.modules` is an **in-memory cache that prevents repeated execution**; `.pyc` is an **on-disk bytecode cache that can avoid recompilation**. |
| **Interview answer** | **`import module` first checks `sys.modules`. If the module isn't cached, Python finds it through the import machinery, creates a module object, places it in `sys.modules`, and executes its code. Subsequent imports normally reuse that cached module without executing it again.** |

### Q41. What's the role of `sys.modules`and how does it prevent repeated execution on re-import?

| Aspect | Explanation |
| --- | --- |
| **What is `sys.modules`?** | A dictionary maintained by Python containing **modules that have already been imported** in the current process. |
| **Key** | The module's fully qualified name, e.g. `"math"` or `"myapp.utils"`. |
| **Value** | The actual module object loaded into memory. |
| **First import** | Python finds the module → creates a module object → places it in `sys.modules` → executes the module's code. |
| **Re-import** | Python checks `sys.modules` first. If the module is already present, it **reuses the existing module object** instead of loading and executing the source again. |
| **Example** | `import math` → loads and executes/initializes it. A later `import math` → finds `"math"` in `sys.modules` and reuses it. |
| **Why it matters** | Prevents expensive repeated module loading and, importantly, prevents module-level initialization code from normally running repeatedly. |
| **Object identity** | Multiple imports of the same module normally reference the **same module object**. |
| **Example** | `import mymoduleimport mymodulemymodule is sys.modules["mymodule"]` → `True` |
| **Package behavior** | Importing `myapp.utils` can place entries such as `"myapp"` and `"myapp.utils"` in `sys.modules`. |
| **Circular imports** | Python adds a module to `sys.modules` **before fully executing it**, which helps the import system handle circular-import situations, although partially initialized modules can cause problems. |
| **Removing an entry** | `del sys.modules["mymodule"]` removes the cache entry. A subsequent `import mymodule` can load and execute it again, potentially creating a new module object. |
| **Important distinction** | `sys.modules` is an **in-memory module cache**. `.pyc` is an **on-disk bytecode cache**. They solve different problems. |
| **Interview answer** | **`sys.modules` is Python's in-memory module cache. On the first import, the module is loaded and stored there. On subsequent imports, Python finds the existing module in `sys.modules` and reuses it, so the module's top-level code is normally not executed again.** |

### Q42. How do `__init__.py` and packages affect the import system?

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

### Q43. What's the difference between absolute and relative imports at resolution time?

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

### Q44. How does exception propagation work internally (the traceback stack)?

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

### Q45. What's the difference between `except Exception` and a bare `except:`?

| Aspect | `except Exception:` | Bare `except:` |
| --- | --- | --- |
| **What it catches** | Exceptions derived from `Exception` | **Almost everything**, including exceptions derived directly from `BaseException` |
| **Catches `ValueError`, `TypeError`, `KeyError`, etc.** | Yes | Yes |
| **Catches `KeyboardInterrupt`** | No | **Yes** |
| **Catches `SystemExit`** | No | **Yes** |
| **Catches `GeneratorExit`** | No | **Yes** |
| **Typical usage** | **Recommended** for normal application error handling | Rare; mainly for very specific cases where you intentionally need to catch all exceptions |
| **Risk** | Lower; allows system-control exceptions to propagate | Higher; can accidentally swallow interrupts, exits, or other important control-flow exceptions |
| **Example** | `try: ...except Exception: handle_error()` | `try: ...except: handle_everything()` |
| **Inheritance** | `Exception` is a subclass of `BaseException` | Catches the `BaseException` hierarchy |
| **Best practice** | Prefer `except Exception:` unless you have a specific reason otherwise | Avoid in most application code |
| **Interview answer** | **`except Exception` catches normal application exceptions, while bare `except` catches all `BaseException` subclasses, including `KeyboardInterrupt` and `SystemExit`. Therefore, `except Exception` is generally safer and preferred.** |  |

### Q46. How does `finally` interact with return statements and exceptions in bytecode?

| Aspect | How `finally` behaves |
| --- | --- |
| **Basic purpose** | `finally` contains cleanup code that runs whether the `try` block **returns normally, returns early, or raises an exception**. |
| **`return` inside `try`** | Python does **not immediately return**. It first executes the `finally` block, then completes the return. |
| **Example** | `def f(): try: return 10 finally: print("cleanup")`Result: prints `"cleanup"` and returns `10`. |
| **Bytecode concept** | The compiler arranges control flow so that paths leaving the `try` block—such as `return`, exception, or normal completion—pass through the `finally` code. |
| **Return value handling** | When `return 10` is encountered, the return value is preserved while `finally` executes. Afterward, the original return proceeds. |
| **Exception in `try`** | The exception is temporarily handled so the `finally` block can execute. After `finally`, the exception continues propagating unless handled. |
| **Exception in `finally`** | A new exception from `finally` replaces/interferes with the original exception. |
| **`return` in `finally`** | A `return` inside `finally` **overrides** a pending return or exception from `try`. |
| **Example** | `def f(): try: return 10 finally: return 20`Result: `20` |
| **Modern CPython** | Modern CPython (3.11+) uses specialized exception-handling bytecode and compiler-generated control flow rather than the older `SETUP_FINALLY` mechanism found in older versions. |
| **Key bytecode idea** | `finally` effectively creates **multiple control-flow paths**: normal execution, `return`, and exception paths all ensure the cleanup code runs before control leaves the `try`. |
| **Interview answer** | **A `finally` block is guaranteed to run when leaving a `try`, including during `return` and exception paths. In bytecode, CPython arranges these control-flow paths so they execute the `finally` code before completing the return or propagating the exception. A `return` or exception raised inside `finally` can override the original control flow.** |

### Q47. Why is CPython considered slower than compiled languages, and where's the overhead?

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

### Q48. What is the difference between CPython, PyPy, and other implementations?

| Aspect | CPython | PyPy | Other Python Implementations |
| --- | --- | --- | --- |
| **Definition** | The **reference and most widely used Python implementation**, written mainly in C | A Python implementation with a **JIT (Just-In-Time) compiler**, largely written in RPython | Alternative implementations designed for specific platforms or goals |
| **Implementation language** | C | RPython/Python ecosystem | Varies |
| **Execution model** | Primarily interprets Python bytecode using the CPython interpreter, with modern runtime optimizations | Uses a **JIT compiler** to compile frequently executed code into machine code at runtime | Depends on implementation |
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

### Q49. How does the GIL affect multithreading vs multiprocessing performance?

| Aspect | Multithreading | Multiprocessing |
| --- | --- | --- |
| **GIL impact** | In standard CPython, the GIL allows only **one thread at a time to execute Python bytecode** within a process | Each process has its **own Python interpreter and GIL**, so processes can execute Python bytecode in parallel |
| **CPU-bound tasks** | Usually **poor scaling** because threads compete for the GIL | **Good scaling** across multiple CPU cores |
| **I/O-bound tasks** | **Very effective** because the GIL is released while waiting for I/O | Effective, but often has more overhead than threads |
| **Parallel CPU execution** | Python bytecode generally cannot run simultaneously across threads in the same process | Can run simultaneously on multiple CPU cores |
| **Memory** | Threads share the same process memory | Processes have separate memory spaces |
| **Communication** | Easier and relatively cheap through shared memory | Requires IPC mechanisms such as queues, pipes, or shared memory |
| **Overhead** | Lower; threads are lightweight | Higher; creating and communicating between processes is more expensive |
| **Best use case** | Network requests, file I/O, database operations, API calls | CPU-intensive work such as numerical computation, image processing, and CPU-heavy data processing |
| **Example** | Downloading 100 files concurrently | Processing 100 CPU-intensive datasets across multiple cores |
| **Key takeaway** | **Threads are mainly useful for concurrency, especially I/O-bound work.** | **Processes are useful for true CPU parallelism in GIL-bound Python code.** |
| **Interview answer** | **The GIL limits multiple threads from executing Python bytecode simultaneously, so multithreading generally doesn't improve CPU-bound Python workloads. Multiprocessing uses separate processes, each with its own GIL, allowing CPU-bound work to run in parallel across cores.** |  |

### Q50. What is `sys.intern()`, and when would you use it manually?

| Aspect | Explanation |
| --- | --- |
| **What is `sys.intern()`?** | `sys.intern()` manually adds a string to Python's **interned string pool** and returns the canonical version of that string. |
| **Why use it?** | If many identical strings exist, interning allows them to **share the same string object**, potentially reducing memory usage and making equality checks faster in some cases. |
| **Example** | `a = sys.intern("hello")b = sys.intern("hello")a is b` → `True` |
| **Without interning** | Two dynamically created equal strings may be different objects: `a == b` can be `True` while `a is b` can be `False`. |
| **When would you use it manually?** | Mainly when your application creates **very large numbers of repeated strings**, such as identifiers, tokens, column names, or categorical labels. |
| **Typical use case** | A parser processing millions of records with repeated field names could intern those names so that identical names reuse the same object. |
| **Performance benefit** | Can reduce memory consumption and sometimes make comparisons faster because interned strings can be compared by identity. |
| **Important caution** | Don't intern strings indiscriminately. Interning many unique strings can increase memory usage because interned strings are kept available for reuse. |
| **Import** | `import sys` |
| **Example** | `name = sys.intern("customer_id")` |
| **Interview answer** | **`sys.intern()` stores a string in Python's interned-string pool and returns a canonical reference to it. I would use it manually when many repeated string values are created and memory usage or string-comparison performance matters.** |

### Q51. How does the peephole optimizer or bytecode caching (`.pyc` files) improve performance?

| Aspect | Bytecode Caching (`.pyc`) | Peephole / AST Optimization |
| --- | --- | --- |
| **Main purpose** | Avoid recompiling `.py` source into bytecode | Make generated bytecode more efficient |
| **Performance benefit** | Faster **module import/startup time** | Can improve **runtime execution speed** |
| **When it happens** | During import, across program runs | During compilation |
| **What it does** | Stores compiled bytecode on disk | Removes or simplifies unnecessary operations |
| **Example** | `.py` → `.pyc` → reuse bytecode | `2 + 3` → `5` at compile time |
| **Repeated execution** | Avoids parsing and compiling unchanged modules | Optimized bytecode executes with less work |
| **Does it improve execution speed?** | **No**, same bytecode is executed | **Yes, slightly**, by reducing unnecessary instructions |
| **Typical benefit** | Faster application startup/imports | Fewer instructions and computations |
| **Modern CPython** | Uses `__pycache__/*.pyc` | Optimization is largely handled during AST/bytecode compilation rather than a separate traditional peephole pass |
| **Key takeaway** | **Caches compilation results** | **Optimizes compilation results** |

## Python for data engineering

### Q52. What are the primary differences between a list and a tuple in the context of data engineering?

| Feature | List | Tuple |
| --- | --- | --- |
| **Mutability** | Mutable | Immutable |
| **Modification** | Can add, remove, or modify elements | Cannot be modified after creation |
| **Use case** | Dynamic data collections | Fixed records or data structures |
| **Memory** | Generally uses more memory | Generally more memory-efficient |
| **Dictionary key** | Cannot be used as a key | Can be used if all elements are hashable |
| **Data engineering** | Collecting and processing changing data | Representing fixed records and ensuring data integrity |

### Q53. How does Python handle memory management and garbage collection?

Python manages memory automatically using a **private heap**, **reference counting**, and **garbage collection**.

- **Reference Counting:** Tracks references to objects. When an object's reference count reaches zero, its memory can be released.
- **Garbage Collector:** Detects and removes **circular references** that reference counting cannot handle.
- **Generational GC:** Groups objects by age and focuses collection on younger objects more frequently.
- **Memory Allocator:** Python's memory manager efficiently allocates and deallocates memory for objects.
- **`gc` Module:** Provides control and inspection of Python's garbage collector.

### Q54. Explain the difference between a shallow copy and a deep copy.

| Feature | Shallow Copy | Deep Copy |
| --- | --- | --- |
| **Nested objects** | Shares references | Creates independent copies |
| **Nested modification** | Can affect the original | Does not affect the original |
| **Method** | `copy.copy()` | `copy.deepcopy()` |
| **Performance** | Faster | Slower |
| **Use case** | Simple/flat data structures | Complex nested data structures |
| **Data engineering** | Useful when nested data can be shared | Useful when complete data isolation is required |

### Q55. What are decorators, and how might a data engineer use them in production?

**Decorators** allow you to extend a function's behavior without modifying its original code.

In production data engineering, they are commonly used for:

- **Logging** — track ETL task execution.
- **Monitoring** — measure execution time and failures.
- **Retry logic** — automatically retry failed API/database operations.
- **Authentication/authorization** — validate access before data operations.
- **Error handling** — consistently handle exceptions across pipeline tasks.

#### Example: Retry Decorator for an API Call

```py
import time

def retry(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)
        return wrapper
    return decorator

@retry(max_retries=3)
def fetch_data():
    print("Fetching data from API...")
    # API request here
```

Here, `@retry` adds **automatic retry behavior** to `fetch_data()` without changing its original function logic.

### Q56. What is the purpose of `*args` and `**kwargs` in function definitions?

- **`args`** collects any number of **positional arguments** into a tuple.
- **`*kwargs`** collects any number of **keyword arguments** into a dictionary.
- They are useful for **flexible functions, decorators, and wrapper functions** where the number or names of arguments may vary.

```py
def process_data(*args, **kwargs):
    print(args)
    print(kwargs)

process_data(1, 2, source="API", batch_size=100)
```

Output:

```txt
(1, 2)
{'source': 'API', 'batch_size': 100}
```

### Q57. Explain the Global Interpreter Lock (GIL) and its impact on data processing.

The **GIL** is a lock in standard CPython that allows only one thread at a time to execute Python bytecode.

- **CPU-bound tasks:** Threads cannot fully utilize multiple CPU cores because of the GIL.
- **I/O-bound tasks:** Threading can still be effective because threads can run while waiting for I/O.
- **Data processing:** CPU-heavy transformations can use **multiprocessing**, native libraries that release the GIL, or distributed frameworks such as **PySpark**.
- **Purpose:** It helps protect CPython's internal memory/object management.

### Q58. What are list and dictionary comprehensions?

**Comprehensions** provide a concise way to create lists or dictionaries from existing iterables, often with filtering or transformation.

**List comprehension:**

```py
squares = [x * x for x in numbers if x > 0]
```

**Dictionary comprehension:**

```py
filtered = {k: v for k, v in data.items() if v > 0}
```

They can be more concise and often faster than equivalent explicit `for` loops for simple operations.

### Q59. How do you implement robust exception handling in an ETL script?

Use `try`, `except`, `else`, and `finally` to handle failures safely.

```py
try:
    data = extract_data()
    transform(data)
    load_data(data)

except DatabaseError as e:
    logger.error(f"Database error: {e}")
    raise

except FileNotFoundError as e:
    logger.error(f"File missing: {e}")
    raise

else:
    logger.info("ETL completed successfully")

finally:
    close_connections()
```

- **`try`** → ETL operations
- **`except`** → Handle specific errors
- **`else`** → Runs when no exception occurs
- **`finally`** → Always executes; useful for cleanup
- **Best practice:** Catch specific exceptions and log failures rather than using only generic `Exception`.

### Q60. What is the difference between the `/` and `//` operators?

| Operator | Meaning | Example | Result |
| --- | --- | --- | --- |
| `/` | True division | `10 / 4` | `2.5` |
| `//` | Floor division | `10 // 4` | `2` |

### Q61. What is a lambda function and when is it appropriate to use one?

A **lambda function** is a small, anonymous, one-line function used for simple operations.

```py
square = lambda x: x ** 2
print(square(5))  # 25
```

**Common uses:**

- `map()` — transform data
- `filter()` — filter data
- Pandas `.apply()` — apply simple transformations

### Q62. What are Python generators, and why are they critical for Big Data?

A **generator** produces values **one at a time** using `yield` instead of storing the entire sequence in memory.

```py
def read_records(records):
    for record in records:
        yield record
```

**Why important for Big Data:**

- **Memory efficient:** Processes data incrementally.
- **Lazy evaluation:** Generates values only when needed.
- **Suitable for large datasets:** Useful for processing large files, logs, and data streams.
- Helps reduce the risk of **Out-of-Memory (OOM)** errors.

### Q63. Explain the `with` statement and the concept of Context Managers.

A **context manager** manages the setup and cleanup of resources automatically. The `with` statement is used to work with context managers.

```py
with open("data.csv") as f:
    data = f.read()
```

Here, Python automatically **closes the file** after the block, even if an exception occurs.

**Data engineering uses:**

- File handling
- Database connections and transactions
- Locks
- Temporary resources
- Cloud/resource management

Custom context managers can be created using `__enter__()` and `__exit__()`.

Eg:

```py
class DatabaseConnection:
    def __enter__(self):
        print("Opening connection")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing connection")

with DatabaseConnection() as db:
    print("Using database")
```

#### How it works

- `__enter__()` → runs when entering the `with` block.
- `__exit__()` → runs automatically when leaving the block, even if an exception occurs.
- `return self` → makes the context manager available through `as db`.

### Q64. How do you optimize Python code for processing millions of rows?

Optimization focuses on **reducing Python-level loops and memory usage** while identifying bottlenecks.

- **Vectorization:** Use NumPy/Pandas vectorized operations instead of row-by-row loops.
- **Chunk processing:** Process large datasets in batches to avoid excessive memory usage.
- **Generators:** Use lazy evaluation for streaming or sequential processing.
- **Built-in functions:** Prefer optimized Python built-ins over custom loops where possible.
- **Efficient data types:** Use appropriate Pandas dtypes to reduce memory consumption.
- **`__slots__`:** Use `__slots__` for large numbers of class instances when appropriate.
- **Profiling:** Use `cProfile`, `line_profiler`, or similar tools to identify bottlenecks.
- **Parallel/distributed processing:** For very large datasets, use multiprocessing, PySpark, or other distributed frameworks.

**Example:** Instead of applying a Python function to every row, use Pandas vectorized operations or process the data in chunks.

### Q65. Compare multiprocessing and multithreading for a data ingestion task.

| Aspect | Multithreading | Multiprocessing |
| --- | --- | --- |
| Best for | **I/O-bound** tasks | **CPU-bound** tasks |
| Example | API calls, S3 downloads, database reads | Complex transformations, encryption, CPU-heavy processing |
| Execution | Multiple threads within one process | Separate processes |
| GIL | Limited for Python CPU-bound code | Bypasses GIL |
| Memory | Shared memory | Separate memory spaces |
| Data ingestion use | Parallel file/API downloads | Parallel CPU-intensive transformations |

### Q66. What is "Pickling" and what are the risks associated with it?

**Pickling** is the process of **serializing a Python object into a byte stream** so it can be stored or transmitted and later reconstructed.

- **Use cases:** Saving Python objects, ML model states, configurations, and cached data.
- **Main risk:** Never unpickle data from an **untrusted source**, because malicious pickle data can execute arbitrary code during deserialization.
- **Data exchange:** Prefer **JSON, Parquet, or other language-independent formats** when interoperability and security are important.

```py
import pickle

# Serialize
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Deserialize — only use trusted files
with open("data.pkl", "rb") as f:
    data = pickle.load(f)
```

### Q67. How do you ensure your Python code remains maintainable in a large team?

Maintainability comes from **consistent standards, clear code, automated testing, and good collaboration practices**.

- **PEP 8:** Follow consistent Python coding conventions.
- **Type hints:** Improve readability and make interfaces explicit.
- **Automated testing:** Use `pytest` for unit and integration tests.
- **Linting:** Use tools like `flake8` or Ruff to detect code issues.
- **Formatting:** Use **Black** for consistent code formatting.
- **Code reviews:** Review changes before merging to maintain quality.
- **Documentation:** Document complex logic, APIs, and important design decisions.
- **CI/CD:** Run tests, linting, and formatting checks automatically in the pipeline.

### Q68. What is Monkey Patching and why is it generally discouraged?

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

### Q69. Explain how `map()` and `filter()` work compared to comprehensions.

- **`map()`** applies a function to every item in an iterable.
- **`filter()`** keeps only items for which the function returns `True`.
- Both return **iterators** in Python 3, so they are lazy and memory-efficient.
- **Comprehensions** are often more readable and Pythonic for simple transformations and filtering.
- `map()` can be efficient when using **built-in functions**, such as `str.upper` or `abs`.

```py
numbers = [1, 2, 3, 4, 5]

# map()
squares = map(lambda x: x ** 2, numbers)

# filter()
even = filter(lambda x: x % 2 == 0, numbers)

# Comprehensions
squares = [x ** 2 for x in numbers]
even = [x for x in numbers if x % 2 == 0]
```

**In practice:** Use comprehensions for readability; use `map()`/`filter()` when they make the transformation clearer, especially with existing functions.
