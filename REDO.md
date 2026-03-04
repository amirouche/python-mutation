# REDO

Projects to re-run after fixes:

- **whenever** — coverage shows `—` in summary; `coverage report` exits non-zero due to `--fail-under=100` in pytest.ini; fix: add `--fail-under=0` to `get_coverage()` in `_update_summaries` (already fixed in manual regen, needs script update)
- **confuse** — 0 mutations; likely no covered source files found; investigate coverage_read output
