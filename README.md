# 🐛 mutation

Mutation testing tells you something coverage numbers can't: whether your tests would actually catch a bug. It works by introducing small deliberate changes into your code — flipping a `+` to a `-`, removing a condition — and checking whether your tests fail. If they don't, the mutation *survived*, and that's a gap worth knowing about.

`mutation` is built around three ideas:

**Fast.** Mutations run in parallel. Most tools write mutated code to disk and run one test at a time — `mutation` doesn't, so you get results in minutes rather than hours.

**Interactive.** `mutation replay` is a guided workflow, not a report. It walks you through each surviving mutation one by one: you inspect it, fix your tests, verify they're green, commit, and move on to the next. Less like a dashboard, more like an interactive rebase.

**Light.** A single Python file. No Rust compiler, no configuration ceremony. Results stored in a local `.mutation.db` SQLite file. Source code you can actually read and understand — which matters when you're trusting a tool to tell you the truth about your tests.

## Getting started

`mutation` runs your tests with pytest. The `-- TEST-COMMAND` option lets you pass any pytest arguments — specific paths, flags, plugins — giving you full control over how the test suite runs.

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
mutation play [--verbose] [--exclude=<glob>]... [--only-deadcode-detection] [--include=<glob>]... [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- TEST-COMMAND ...]
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

**`-- TEST-COMMAND`**

A full pytest invocation to run instead of the built-in default (`pytest --exitfirst --no-header --tb=no --quiet --assert=plain`). Useful when you need specific pytest flags, plugins, or a subset of tests.

`mutation` always appends `--mutation=<uid>` to whatever command you supply — this flag is how it injects each mutation in-process without touching files on disk. Because of this, the command **must** be a `pytest` invocation; other test runners are not supported. Coverage flags (`--cov`, etc.) are added automatically during the baseline run.

`-- TEST-COMMAND` and `<file-or-directory>` are mutually exclusive.

```
# Run only the unit tests, with verbose output
mutation play --include="src/*.py" -- pytest -x -v tests/unit/
```

## Status

Early stage. Things may break. Bug reports and questions welcome at amirouche.boubekki@gmail.com.
