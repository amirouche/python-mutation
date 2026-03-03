# Why Did This Mutation Survive?

You just ran `mutation replay` and a mutation came back green — meaning your test suite
**did not catch** the code change. This document helps you decide what to do about it.

---

## The Five Cases

### 1. Real gap — write a test

The mutation exposes a genuine hole in your test coverage. The code behaviour
it changes **is observable** and **should be tested**, but currently isn't.

**Action:** Write a test that fails when this mutation is applied, then commit it.
The mutation will disappear from the queue once the test catches it.

---

### 2. Fragile coverage — risk accepted

A test *does* exercise this code path, but only incidentally — the assertion
doesn't pin down the exact behaviour that the mutation changes. Adding a
targeted assertion would make the test brittle or overly implementation-aware.

**Action:** Mark it as fragile and move on. Keep it on your radar — if the
surrounding code changes, revisit whether coverage is still adequate.

---

### 3. Equivalent mutation — semantically invisible

The mutation produces code that is **behaviourally identical** to the original
under all reachable inputs. No test can ever catch it, because there is nothing
wrong to catch.

Classic examples:
- Swapping `x * 1` → `x * 2` when `x` is always `0` at that point
- Reordering independent dictionary key assignments
- Changing a constant that is never read

**Action:** Mark it as equivalent. The tool writes a `.diff` file to
`.mutations.ignored/` so this mutation is silently skipped on every future
`play` run without touching the database. Commit the ignore file so CI never
flags it again.

The ignore file is named by a SHA-256 hash of the diff text, so it remains
valid across refactors that don't touch the mutated lines.

---

### 4. Won't fix — consciously accepted gap

The gap is real, but fixing it is not worth the effort right now (or ever).
Maybe the code path is deprecated, the risk is negligible, or the test would be
too expensive to maintain.

**Action:** Mark it as "won't fix". It is permanently removed from the replay
queue and will not resurface in future runs.

---

### 5. Todo — real gap, not fixing now

Same as "Real gap", but you're acknowledging you won't fix it in this session.
It stays on the queue and will appear again in the next `replay`.

**Action:** Mark it as todo and come back to it later.

---

## Decision Guide

```
Does the mutation change observable behaviour?
│
├── No  → Equivalent (Case 3)
│
└── Yes → Is it tested anywhere, even loosely?
          │
          ├── No  → Real gap (Case 1) or Todo (Case 5)
          │
          └── Yes → Is the existing assertion precise enough?
                    │
                    ├── Yes (test should catch it but doesn't) → Real gap (Case 1)
                    │
                    └── No (only incidental coverage)
                              │
                              ├── Risk matters → Real gap (Case 1)
                              └── Risk low     → Fragile (Case 2) or Won't fix (Case 4)
```

---

## The `.mutations.ignored/` directory

Each file is a human-readable diff prefixed with a comment explaining why it
was classified as equivalent:

```
# Case 3: equivalent mutation — x is always 0 at this point
--- a/mymodule/calc.py
+++ b/mymodule/calc.py
@@ -10,7 +10,7 @@
-    return x * 1
+    return x * 2
```

These files are safe to commit. They are content-addressed (SHA-256 of the diff),
so moving or renaming source files does not break them — only editing the
mutated lines themselves would make the ignore file stale.

Run `mutation gc` to remove stale ignore files whose diffs no longer apply to
the current source.

---

## Quick reference

| Choice | Meaning | Resurfaces? |
|--------|---------|-------------|
| `1` Real gap | Write a test | Yes, until caught |
| `2` Fragile | Risk accepted | Yes, as reminder |
| `3` Equivalent | Writes ignore file | No (skipped at play time) |
| `4` Won't fix | Consciously accepted | No |
| `5` Todo | Fix later | Yes, next replay |
| `s` Skip | Undecided | Yes, later this session |
| `q` Quit | Save and exit | Yes, next replay |
