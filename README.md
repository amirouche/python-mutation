# 🐛 mutation

Mutation testing tells you something coverage numbers can't: whether your tests would actually catch a bug. It works by introducing small deliberate changes into your code — flipping a `+` to a `-`, removing a condition — and checking whether your tests fail. If they don't, the mutation *survived*, and that's a gap worth knowing about.

`mutation` is built around three ideas:

**Fast.** Mutations run in parallel. Most tools write mutated code to disk and run one test at a time — `mutation` doesn't, so you get results in minutes rather than hours.

**Interactive.** `mutation replay` is a guided workflow, not a report. It walks you through each surviving mutation one by one: you inspect it, fix your tests, verify they're green, commit, and move on to the next. Less like a dashboard, more like an interactive rebase.

**Light.** A single Python file. No Rust compiler, no configuration ceremony. Results stored in a local `.mutation.db` SQLite file. Source code you can actually read and understand — which matters when you're trusting a tool to tell you the truth about your tests.

## Getting started

`mutation` runs your tests with pytest. The `-- PYTEST-COMMAND` option lets you pass any pytest arguments — specific paths, flags, plugins — giving you full control over how the test suite runs.

```
pip install mutation
mutation play tests.py --include=foobar/ex.py --include=foobar/__init__.py --exclude=tests.py
```

Then work through the results:

```
mutation replay
```

## Usage

```
mutation play [--verbose] [--exclude=<glob>]... [--only-deadcode-detection] [--include=<glob>]... [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- PYTEST-COMMAND ...]
mutation replay [--verbose] [--max-workers=<n>]
mutation list
mutation show MUTATION
mutation apply MUTATION
mutation (-h | --help)
mutation --version
```

`mutation` only mutates code with test coverage, so it works best when coverage is high.

`mutation` detects whether tests can run in parallel — making your test suite parallel-safe will significantly speed things up.

## Options

**`--include=<glob>` and `--exclude=<glob>`**

Glob patterns matched against relative file paths. Repeat the flag to supply multiple patterns.

```
# Mutate only specific modules, exclude both test files and migrations
mutation play tests.py --include=src/*.py --include=lib/*.py --exclude=tests.py --exclude=migrations/*.py
```

Default `--include` is `*.py` (all Python files). Default `--exclude` is `*test*` (any path whose relative path contains "test"). The patterns are applied before the coverage filter, so files with no coverage are always skipped regardless.

**`--sampling=<s>`**

Limit how many mutations are actually tested — useful for a quick sanity check before a full run.

- `--sampling=100` — test only the first 100 mutations (deterministic order)
- `--sampling=10%` — test a random 10% of all mutations (probability-based; set `--randomly-seed` for reproducibility)

Default: all mutations are tested.

**`--randomly-seed=<n>`**

Integer seed that controls three things at once: the order pytest-randomly uses to shuffle your tests, the random values injected by numeric mutations (`MutateNumber`), and which mutations are selected when using `--sampling=N%`. Setting a fixed seed makes any of these behaviors reproducible across runs.

Default: current Unix timestamp (a different seed each run).

```
mutation play tests.py --randomly-seed=12345 --sampling=20%
```

**`-- PYTEST-COMMAND`**

A full pytest invocation to run instead of the built-in default (`pytest --exitfirst --no-header --tb=no --quiet --assert=plain`). Useful when you need specific pytest flags, plugins, or a subset of tests.

`mutation` always appends `--mutation=<uid>` to whatever command you supply — this flag is how it injects each mutation in-process without touching files on disk. Because of this, the command **must** be a `pytest` invocation; other test runners are not supported. Coverage flags (`--cov`, etc.) are added automatically during the baseline run.

`-- PYTEST-COMMAND` and `<file-or-directory>` are mutually exclusive.

```
# Run only the unit tests, with verbose output
mutation play --include="src/*.py" -- pytest -x -v tests/unit/
```

## Mutations

<details><summary>StatementDrop</summary>

Replace a covered statement with `pass`, verifying that no statement is inert dead code.

```python
# before
x = compute()
validate(x)

# after
x = compute()
pass
```

</details>

<details><summary>DefinitionDrop</summary>

Remove a function or class definition entirely (only when others remain in the same body), surfacing unreferenced definitions.

```python
# before
def helper():
    return 42

def main():
    return helper()

# after
def main():
    return helper()
```

</details>

<details><summary>MutateNumber</summary>

Replace an integer or float literal with a random value in the same bit-range, verifying that the exact numeric value is tested.

```python
# before
TIMEOUT = 30

# after
TIMEOUT = 17
```

</details>

<details><summary>MutateString</summary>

Prepend a fixed prefix to a string or bytes literal, verifying that callers check the actual content.

```python
# before
label = "hello"

# after
label = "mutated string hello"
```

</details>

<details><summary>MutateKeyword</summary>

Rotate flow keywords (break/continue/pass), swap boolean constants (True/False/None), and flip boolean operators (and/or).

```python
# before
while True:
    if done:
        break

# after
while True:
    if done:
        continue
```

</details>

<details><summary>Comparison</summary>

Negate a comparison expression by wrapping it with `not (...)`, verifying that the direction of every comparison is tested.

```python
# before
if x > 0:
    process(x)

# after
if not (x > 0):
    process(x)
```

</details>

<details><summary>MutateOperator</summary>

Replace an arithmetic, bitwise, shift, or comparison operator with another in the same group, verifying the exact operator matters.

```python
# before
result = a + b

# after
result = a - b
```

</details>

<details><summary>MutateMatchCase</summary>

Remove one case branch at a time from a match statement (Python 3.10+ only), verifying that each branch is exercised by the test suite.

```python
# before
match command:
    case "quit":
        quit()
    case "go":
        go()

# after
match command:
    case "go":
        go()
```

</details>

<details><summary>MutateStringMethod</summary>

Swap directionally symmetric string methods (lower↔upper, lstrip↔rstrip, find↔rfind, ljust↔rjust, removeprefix↔removesuffix, partition↔rpartition, split↔rsplit), verifying that the direction matters.

```python
# before
name = text.lower()

# after
name = text.upper()
```

</details>

<details><summary>MutateCallArgs</summary>

Replace each positional call argument with `None`, and drop one argument at a time from multi-argument calls, verifying that every argument is actually used.

```python
# before
result = process(data, config)

# after
result = process(None, config)
```

</details>

<details><summary>ForceConditional</summary>

Force the test of an if/while/assert/ternary to always be `True` or always `False`, verifying that both branches are meaningfully exercised.

```python
# before
if is_valid(x):
    save(x)

# after
if True:
    save(x)
```

</details>

<details><summary>MutateExceptionHandler</summary>

Replace the specific exception type in an except clause with the generic `Exception`, verifying that the handler is tested for the right error kind.

```python
# before
try:
    connect()
except ConnectionError:
    retry()

# after
try:
    connect()
except Exception:
    retry()
```

</details>

<details><summary>ZeroIteration</summary>

Replace a for-loop's iterable with an empty list, forcing the body to never execute, verifying that callers handle empty-collection cases.

```python
# before
for item in items:
    process(item)

# after
for item in []:
    process(item)
```

</details>

<details><summary>RemoveDecorator</summary>

Remove one decorator at a time from a decorated function or class, verifying that each decorator's effect is covered by tests.

```python
# before
@login_required
def dashboard(request):
    return render(request)

# after
def dashboard(request):
    return render(request)
```

</details>

<details><summary>NegateCondition</summary>

Wrap a bare (non-comparison) condition with `not`, inserting the logical inverse of the test, verifying that the truthiness of the value actually matters.

```python
# before
if user.is_active:
    allow()

# after
if not user.is_active:
    allow()
```

</details>

<details><summary>MutateReturn</summary>

Replace a return value with a type-appropriate default (`None`, `0`, `False`, or `""`), verifying that callers check what the function returns.

```python
# before
def get_count():
    return len(items)

# after
def get_count():
    return 0
```

</details>

<details><summary>MutateLambda</summary>

Replace the body of a lambda with `None` (or `0` when the body is already `None`), verifying that the lambda's computation is actually used.

```python
# before
transform = lambda x: x * 2

# after
transform = lambda x: None
```

</details>

<details><summary>MutateAssignment</summary>

Replace the right-hand side of a plain assignment with `None`, verifying that the assigned value is not silently ignored.

```python
# before
result = compute()

# after
result = None
```

</details>

<details><summary>AugAssignToAssign</summary>

Convert an augmented assignment (`x += v`) to a plain assignment (`x = v`), dropping the accumulation, verifying that the update operator is tested.

```python
# before
total += amount

# after
total = amount
```

</details>

<details><summary>RemoveUnaryOp</summary>

Strip a unary operator (`not`, `-`, `~`) and leave only the operand, verifying that the operator's effect is covered by tests.

```python
# before
if not flag:
    skip()

# after
if flag:
    skip()
```

</details>

<details><summary>MutateIdentity</summary>

Swap `is` ↔ `is not` in identity comparisons, verifying that the expected identity relationship is directly tested.

```python
# before
if obj is None:
    init()

# after
if obj is not None:
    init()
```

</details>

<details><summary>MutateContainment</summary>

Swap `in` ↔ `not in` in membership tests, verifying that the expected membership relationship is directly tested.

```python
# before
if key in cache:
    return cache[key]

# after
if key not in cache:
    return cache[key]
```

</details>

<details><summary>BreakToReturn</summary>

Replace `break` with `return`, exiting the enclosing function instead of just the loop, verifying that the loop's exit path is tested.

```python
# before
for item in items:
    if item.done:
        break

# after
for item in items:
    if item.done:
        return
```

</details>

<details><summary>SwapArguments</summary>

Swap each pair of positional call arguments, verifying that argument order is tested.

```python
# before
result = process(source, dest)

# after
result = process(dest, source)
```

</details>

<details><summary>MutateSlice</summary>

Drop the lower or upper bound of a slice (`a[i:j]` → `a[:j]` or `a[i:]`) and negate the step (`a[::2]` → `a[::-2]`), verifying that slice boundary conditions and direction are tested.

```python
# before
chunk = data[start:end]

# after
chunk = data[:end]
```

</details>

<details><summary>MutateYield</summary>

Replace the value of a yield expression with `None`, verifying that the yielded value is actually used by callers.

```python
# before
def generate():
    yield compute()

# after
def generate():
    yield None
```

</details>

<details><summary>MutateDefaultArgument</summary>

Remove leading default argument values one at a time, making parameters required, verifying that callers always supply them explicitly.

```python
# before
def connect(host, port=8080, timeout=30):
    ...

# after
def connect(host, port, timeout=30):
    ...
```

</details>

<details><summary>MutateIterator</summary>

Wrap a for-loop's iterable in `reversed()`, verifying that iteration order assumptions are tested.

```python
# before
for item in queue:
    process(item)

# after
for item in reversed(queue):
    process(item)
```

</details>

<details><summary>MutateContextManager</summary>

Strip context managers from a `with` statement one at a time, keeping the body, verifying that each manager's effect is tested.

```python
# before
with lock:
    update_shared_state()

# after
update_shared_state()
```

</details>

## Status

Early stage. Things may break. Bug reports and questions welcome at amirouche.boubekki@gmail.com.
