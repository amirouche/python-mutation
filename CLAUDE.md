# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mutation` is a Python mutation testing tool. It introduces controlled mutations into source code and verifies that a test suite detects them, revealing gaps in test coverage quality.

## Environment Setup

Dependencies are managed with pip-tools. The `./venv` script creates a cached virtual environment:

```sh
./venv pip install -r requirements.txt
./venv pip install -r requirements.dev.txt
```

Or directly with pip (if already in a venv):

```sh
pip install -r requirements.txt
```

To regenerate the locked `requirements.txt` from `requirements.source.txt`:

```sh
make lock
```

## Common Commands

```sh
make check          # Run mutation tests on foobar/ example + bandit security scan
make check-only     # Run mutation tests only (no bandit)
make check-fast     # Run pytest with fail-fast (-x)
make check-coverage # Generate HTML coverage report
make lint           # Run pylama linter
make clean          # Remove untracked/ignored files (git clean -fX)
make wip            # Format with black+isort and commit as "wip"
```

**Running mutation tests directly:**

```sh
python3 mutation.py play tests.py --include="foobar/ex.py,foobar/__init__.py" --exclude="tests.py"
mutation replay      # Re-test previously failed mutations interactively
mutation list        # List stored mutation failures
mutation show MUTATION   # Display a specific mutation diff (syntax highlighted)
mutation apply MUTATION  # Apply a mutation to source files
```

**Running the test suite for mutation.py itself:**

```sh
pytest -x -vvv --capture=no mutation.py
```

## Architecture

Everything lives in a single file: **`mutation.py`** (1052 lines). It functions as both a standalone CLI tool and a pytest plugin.

### Mutation Classes

Mutations are implemented via a `Mutation` metaclass that registers all subclasses automatically:

- `StatementDrop` — removes statements
- `DefinitionDrop` — removes function/class definitions (deadcode detection)
- `MutateNumber` — randomizes numeric literals
- `MutateString` — mutates string values
- `MutateKeyword` — flips keywords (`continue`/`break`/`pass`, `True`/`False`/`None`, `and`/`or`)
- `MutateOperator` — swaps binary, bitwise, comparison, and assignment operators
- `Comparison` — inverts comparison expressions

### Core Pipeline (`play` command)

1. **`check_tests`** — runs the baseline test suite to confirm it passes; detects xdist parallel support
2. **`coverage_read`** — parses `.coverage` data to determine which lines are actually executed
3. **`iter_deltas`** — walks the AST via `parso`, applies `mutate()` per node, filters to covered lines via `interesting()`, yields unified diffs
4. **`mutation_create`** — parallelizes delta generation using a process pool; stores mutations in the SQLite database compressed with zstandard
5. **`mutation_pass`** — runs each mutation through the test suite via a thread pool; records survivors (undetected mutations)

### Storage

Mutations are persisted in `.mutation.db` (a SQLite database). Keys use `lexode` encoding; values are `zstandard`-compressed unified diffs indexed by ULID.

### Pytest-Only

`mutation` is fundamentally pytest-specific. Although the CLI accepts a custom `-- TEST-COMMAND`, it always appends `--mutation=<uid>` to whatever command is used. That flag is a pytest option registered by `mutation.py` itself acting as a pytest plugin (`pytest_configure` / `pytest_addoption` hooks). The plugin calls `install_module_loader` to patch the target module in-memory for that test run, without modifying files on disk. Any custom `TEST-COMMAND` must therefore still be a pytest invocation — swapping in a different test runner is not supported.

### Async Execution

`pool_for_each_par_map` drives the parallel mutation workflow using `asyncio` + `concurrent.futures` (process pool for mutation creation, thread pool for test execution) with `aiostream` for streaming results.

## Key Files

| File | Purpose |
|------|---------|
| `mutation.py` | Entire application: CLI, mutation engine, pytest plugin |
| `tests.py` | Example test suite (tests `foobar/ex.py`) used for self-testing |
| `foobar/ex.py` | Example module (`decrement_by_two`) mutated during self-tests |
| `requirements.source.txt` | Hand-maintained dependency list (input to pip-compile) |
| `requirements.txt` | pip-compiled locked dependencies (auto-generated, do not edit) |
| `requirements.dev.txt` | Dev-only tools: black, isort, bandit, tbvaccine |

## Known Issues / TODOs

- Mutations that produce syntax errors are not filtered out (requires Python 3.9+ `ast.unparse`)
- Removing docstrings can trigger errors in `mutation play`
- PyPy support is untested (sqlite3 is in the stdlib but other dependencies may not support PyPy)
- `rc.sh` contains an unresolved git merge conflict
