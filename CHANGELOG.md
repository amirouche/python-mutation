# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.5.5] - 2026-03-07

### Packaging

- Fixed `build-backend` to use `setuptools.build_meta` instead of `setuptools.backends.legacy:build` (the latter requires a newer setuptools not available in CI)

## [0.5.4] - 2026-03-07

### Packaging

- Added `[build-system]` and `[tool.setuptools] py-modules = ["mutation"]` so `mutation.py` is included in the wheel

## [0.5.3] - 2026-03-07

### Bug fixes

- Fixed stale `__version__` tuple in `mutation.py` (was `(0, 4, 7)`, now tracks `pyproject.toml`)

## [0.5.2] - 2026-03-07

### Bug fixes

- Fixed `[project.scripts]` entry point module name (`mutation` → `mutationpy`)

## [0.5.1] - 2026-03-07

### Packaging

- Added `[project.scripts]` entrypoint so `pip install` exposes the `mutation` command

## [0.5.0] - 2026-03-07

### New mutations

- `AugAssignToAssign` — replace `x += y` with `x = y`
- `BreakToReturn` — replace `break` with `return`
- `ForceConditional` — replace `if`/`while`/`assert` condition with `True` or `False`
- `InjectException` — inject an exception raise inside a function body
- `MutateAssignment` — replace assignment right-hand side with `None`
- `MutateCallArgs` — replace call arguments with `None` or drop them
- `MutateContainment` — swap `in` / `not in`
- `MutateContextManager` — mutate context manager expressions
- `MutateDefaultArgument` — mutate default argument values
- `MutateExceptionHandler` — replace specific exception type with `Exception`
- `MutateFString` — mutate f-string expressions
- `MutateGlobal` — remove `global`/`nonlocal` declarations
- `MutateIdentity` — swap `is` / `is not`
- `MutateIterator` — mutate iterators, including injecting `random.shuffle()`
- `MutateLambda` — replace lambda body with `None`
- `MutateReturn` — replace return value with `None`, `0`, `False`, or `""`
- `MutateSlice` — mutate slice expressions
- `MutateStringMethod` — swap symmetric string methods (`lower`/`upper`, `lstrip`/`rstrip`, etc.)
- `MutateYield` — mutate `yield` expressions
- `NegateCondition` — insert `not` around bare `if`/`while`/`assert` conditions
- `RemoveDecorator` — drop one decorator at a time from functions or classes
- `RemoveUnaryOp` — strip `not` / `-` / `~` from unary expressions
- `SwapArguments` — swap adjacent function call arguments
- `ZeroIteration` — replace for-loop iterable with `[]`

### New CLI commands and options

- `mutation summary` — print a classification breakdown of stored results
- `mutation gc` — garbage-collect ignored mutations
- `--without-exception-injection` — skip all `InjectException` mutations
- `--only-deadcode-detection` — restrict run to dead-code detection mutations only
- `--include` and `--exclude` flags are now repeatable (one glob per flag)
- Renamed positional `TEST-COMMAND` to `PYTEST-COMMAND`

### Dependency reduction

Dropped 13 third-party runtime dependencies; `mutation.py` now requires only `coverage`
and `docopt` at runtime. Removed packages:

`aiostream`, `astunparse`, `humanize`, `lexode`, `loguru`, `lsm-db`, `parso`,
`pathlib3x`, `pygments`, `termcolor`, `tqdm`, `ulid`, `zstandard`

Replaced by stdlib equivalents: `ast` (parse/unparse), `sqlite3`, `logging`,
`pathlib`, `hashlib`, `zlib`, `json`, `subprocess`.

### Storage

- Replaced `lsm-db` key-value store with `sqlite3`; schema uses native columns and JSON
- Batch mutation inserts via `executemany` to reduce commit overhead
- Added `classification` column to `results` table for replay workflow

### Tooling

- Migrated from `poetry` to `uv`; replaced `black`/`isort`/`bandit` with `ruff`
- Added GitHub Actions CI (push to `main` only) and PyPI release workflow on version tags
- `mutation.py` self-registers as a pytest plugin via `pytest_configure`
- Self-re-exec under a systemd transient scope; worker count defaults to `nproc`
- Added `foobar/` example package with its own test suite and `make check-foobar` target
- Added `/mutation-test` slash command for applying the tool to external projects

### Bug fixes

- `--max-workers` no longer forced to 1 when xdist detection fails
- Docopt defaults no longer silently override `--sampling`, `--randomly-seed`, `--max-workers`
- Relative paths in `coverage_read` resolved correctly; `cov-fail-under` no longer blocks runs
- Package `__init__.py` handled correctly in `install_module_loader`
- `sys.path`-relative lookup used for module name in `install_module_loader`
- Unpicklable lambda replaced with named function for `--without-exception-injection`
- `get_event_loop()` replaced with `new_event_loop()` (asyncio deprecation)
- `pytest_addoption` double-registration error on repeated plugin load
- Syntax-error mutations filtered before being stored
- Deprecated `imp` module replaced
