---
description: Apply mutation testing to an existing project's pytest test suite using mutation.py
allowed-tools: Bash(python3:*), Bash(pip:*), Bash(git clone:*)
---

Apply mutation testing to the project at: $ARGUMENTS

## Steps

### 1. Install mutation.py's dependencies into the target project's venv

From within the target project directory (with its venv active):

```sh
pip install aiostream docopt humanize loguru pygments \
    pytest-cov pytest-randomly pytest-xdist python-ulid \
    termcolor tqdm zstandard coverage
```

### 2. Verify the baseline test suite is green

```sh
pytest <test-file>
```

mutation.py will also check this automatically, but it's good to confirm first.

### 3. Run mutation testing

```sh
python3 /src/python-mutation/mutation.py play <test-file> \
    --include=<source-file-or-glob> \
    --max-workers=<N>
```

- `--include` — glob of files to mutate (e.g. `src/mylib.py` or `src/**/*.py`); omit to mutate all non-test `.py` files
- `--exclude` — defaults to `*test*`, so test files are already excluded
- `--max-workers` — parallelism (e.g. `--max-workers=8`); default is cpu_count-1
- Results are stored in `.mutation.db` in the current directory

### 4. Inspect results

```sh
python3 /src/python-mutation/mutation.py list           # surviving mutations
python3 /src/python-mutation/mutation.py show <ID>      # diff of a specific survivor
python3 /src/python-mutation/mutation.py replay         # interactively re-test survivors
```

## Gotchas

- The target project must use **pytest** (mutation.py is pytest-only)
- Run mutation.py from **inside** the target project directory so `.mutation.db` and coverage data land there
- If `play` errors with "Tests are not green", check that `pytest-xdist` can run the suite in parallel — some tests have ordering dependencies
- `mutation.py` acts as a pytest plugin; it patches source files in-memory, never on disk
