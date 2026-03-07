# REDO

Projects to re-run after fixes:

- **whenever** — coverage shows `—` in summary; `coverage report` exits non-zero due to `--fail-under=100` in pytest.ini; fix: add `--fail-under=0` to `get_coverage()` in `_update_summaries` (already fixed in manual regen, needs script update)
- **confuse** — 0 mutations; likely no covered source files found; investigate coverage_read output
- **dateparser** — git clone failed (transient network error); just re-run
- **pydash** — claude fixed baseline (2228 tests pass) but retry failed rc=4 (no tests collected); likely `-p no:randomly` + `--ignore=tests/pytest_mypy_testing` in addopts conflicted with mutation.py's pytest invocation; investigate pyproject.toml addopts before re-run
