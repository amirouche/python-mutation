# mutation.py — Claude Code Integration Guide

## Overview

`mutation.py` is a mutation testing tool. It generates source mutations, runs pytest against each, and records which mutations *survived* (weren't caught by tests). Survivors indicate gaps in test coverage.

`mutation play` exits 1 when unreviewed survivors exist — this is the CI gate.

---

## Claude's Workflow

```bash
# 1. Run mutation testing (exits 1 if unreviewed survivors exist)
mutation play

# 2. Check overall state
mutation summary --json

# 3. List unreviewed survivors (sorted largest diff first)
mutation survivors --json

# 4. Inspect each survivor in detail
mutation inspect <uid> --json --context=15

# 5a. Write a test to kill the mutation, then verify:
pytest --mutation=<uid>
# pytest exits non-zero → mutation killed ✓ (no classify needed)

# 5b. Or classify without fixing:
mutation classify <uid> real_gap
mutation classify <uid> fragile
mutation classify <uid> equivalent --reason="both branches produce identical output"
mutation classify <uid> wont_fix
mutation classify <uid> todo
```

---

## Command Reference

### `mutation summary [--json]`

Shows overall mutation testing statistics.

**JSON output schema:**
```json
{
  "generated": 1000,
  "stale": 10,
  "tested": 990,
  "killed": 975,
  "survived": 15,
  "classifications": {
    "real_gap": 5,
    "fragile": 3,
    "equivalent": 4,
    "wont_fix": 2,
    "todo": 1,
    "unreviewed": 0
  },
  "ignored_files": 4
}
```

### `mutation survivors [--json]`

Lists unreviewed survivors (the replay queue). Sorted largest diff first.

**JSON output schema:**
```json
{
  "count": 3,
  "survivors": [
    {"uid": "abc123...", "path": "src/mymodule.py", "diff_size": 142},
    {"uid": "def456...", "path": "src/other.py",    "diff_size": 87}
  ]
}
```

### `mutation inspect <uid> [--json] [--context=<n>]`

Returns full details for one mutation. `--context=<n>` controls lines of source shown around the mutation (default: 10).

**JSON output schema:**
```json
{
  "uid": "abc123...",
  "path": "src/mymodule.py",
  "diff": "--- a/src/mymodule.py\n+++ b/src/mymodule.py\n@@ -42,7 +42,7 @@\n...",
  "line": 42,
  "source_context": {
    "start_line": 32,
    "end_line": 52,
    "lines": ["    def foo(self):\n", "        return x + 1\n"]
  },
  "classification": null,
  "status": "survived"
}
```

`classification` is `null` if unreviewed, otherwise one of: `real_gap`, `fragile`, `equivalent`, `wont_fix`, `todo`, `stale`.

### `mutation classify <uid> <category> [--reason=<text>] [--json]`

Non-interactively classify a mutation. Valid categories:

| Category     | Meaning |
|-------------|---------|
| `real_gap`   | Missing test — write one |
| `fragile`    | Test exists but is fragile/implementation-dependent |
| `equivalent` | Mutation is semantically equivalent — both variants produce identical output |
| `wont_fix`   | Known gap, not worth fixing |
| `todo`       | Known gap, fix later |

For `equivalent`, a `.diff` file is written to `.mutations.ignored/` and the mutation is permanently ignored in future runs.

---

## Classification Decision Guide

When inspecting a survivor, ask:

1. **Does the diff change observable behavior?**
   - No → `equivalent`
   - Yes → continue

2. **Should a test catch this?**
   - Yes, and it's a real code path → `real_gap` (write the test)
   - Yes, but it's an error-handling path that's hard to test → `wont_fix` or `todo`

3. **Is an existing test supposed to catch this but doesn't due to fragile assertions?**
   - Yes → `fragile`

4. **Not sure / will address later?**
   - `todo`

---

## Verifying a Test Kills a Mutation

```bash
pytest --mutation=<uid>
```

- Exit code **non-zero** → mutation killed ✓ (test caught it)
- Exit code **0** → mutation survived ✗ (test didn't catch it, or no test runs)

After writing a test and confirming it kills the mutation, no classify step is needed — the next `mutation play` run will record it as killed.

---

## CI Pipeline Pattern

```yaml
- name: Mutation testing gate
  run: mutation play        # exits 1 if unreviewed survivors > 0

- name: Upload mutation report
  run: mutation summary --json > mutation-report.json
  if: always()
```

---

## Database & Files

- `.mutation.db` — SQLite database (mutations, results, classifications)
- `.mutations.ignored/` — diff files for equivalent mutations (excluded from future runs)

Do not delete `.mutation.db` between runs unless you want to start fresh. Use `mutation gc` to clean up stale entries.
