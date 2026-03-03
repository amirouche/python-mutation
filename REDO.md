# REDO

Projects to re-run after fixes:

- **whenever** — `pyproject.toml` has `--cov-fail-under=100`; fixed in `mutation.py` by always passing `--cov-fail-under=0` to override project thresholds
- **chardet** — same `--cov-fail-under` issue; ran before the fix was deployed
- **anyio** — coverage.py recorded relative paths in `.coverage`; `coverage_read` only matched absolute paths so all 62 files were filtered → 0 mutations; fixed in `mutation.py` by resolving relative paths against root
- **blist** — C extension; `ez_setup.py` tries to download setuptools at build time → fails without it; fix: `pip install setuptools` first (added to script fixups); skipped this run due to hanging claude debug session
- **dateparser** — transient git clone failure (`could not read Username`); GitHub rate-limit or network hiccup; just re-run
