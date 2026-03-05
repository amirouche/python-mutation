#!/usr/bin/env python3
"""Mutation.

Usage:
  mutation play [--verbose] [--exclude=<glob>]... [--only-deadcode-detection] [--without-exception-injection] [--include=<glob>]... [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- PYTEST-COMMAND ...]
  mutation replay [--verbose] [--max-workers=<n>]
  mutation list
  mutation show MUTATION
  mutation apply MUTATION
  mutation summary
  mutation gc
  mutation (-h | --help)
  mutation --version

Options:
  --include=<glob>           Glob pattern for files to mutate, matched against relative paths.
                             Repeat the flag for multiple patterns [default: *.py]
  --exclude=<glob>           Glob pattern for files to skip. Repeat the flag for multiple
                             patterns [default: *test*]
  --sampling=<s>             Limit mutations tested: N tests the first N, N% tests a random
                             N% (e.g. "--sampling=100" or "--sampling=10%") (default: all)
  --randomly-seed=<n>        Integer seed controlling test order (pytest-randomly) and random
                             number mutations; also makes --sampling=N% reproducible
                             (default: current Unix timestamp)
  --only-deadcode-detection  Only apply dead-code detection mutations (StatementDrop,
                             DefinitionDrop).
  --without-exception-injection  Skip all InjectException mutations (useful when error-handling
                                 paths are intentionally untested or produce too much noise).
  --max-workers=<n>          Number of parallel workers (default: cpu_count - 1)
  --verbose                  Show more information.
  -h --help                  Show this screen.
  --version                  Show version.
"""
import ast
import asyncio
import fnmatch
import functools
import hashlib
import itertools
import json
import os
import random
import re
import shlex
import sqlite3
import subprocess
import sys
import time
import types
from concurrent import futures
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from difflib import unified_diff
from uuid import UUID

import pygments
import pygments.formatters
import pygments.lexers
import zstandard as zstd
from aiostream import pipe, stream
from coverage import Coverage
from docopt import docopt
from humanize import precisedelta
from loguru import logger as log
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
from ulid import ULID

__version__ = (0, 4, 7)


MINUTE = 60  # seconds
HOUR = 60 * MINUTE
DAY = 24 * HOUR
MONTH = 31 * DAY

CLASSIFICATION_REAL_GAP  = 1
CLASSIFICATION_FRAGILE   = 2
CLASSIFICATION_EQUIVALENT = 3
CLASSIFICATION_WONT_FIX  = 4
CLASSIFICATION_TODO      = 5


def humanize(seconds):
    if seconds < 1:
        precision = "seconds"
    elif seconds // DAY != 0:
        precision = "days"
    elif seconds // DAY != 0:
        precision = "hours"
    elif seconds // HOUR != 0:
        precision = "minutes"
    else:
        precision = "seconds"
    return precisedelta(timedelta(seconds=seconds), minimum_unit=precision)


MUTATION = "https://youtu.be/ihZEaj9ml4w?list=PLOSNaPJYYhrtliZqyEWDWL0oqeH0hOHnj"


log.remove()
if os.environ.get("DEBUG", False):
    log.add(
        sys.stdout,
        format="<level>{level}</level> {message}",
        level="TRACE",
        colorize=True,
        enqueue=True,
    )
else:
    log.add(
        sys.stdout,
        format="<level>{level}</level> {message}",
        level="INFO",
        enqueue=True,
    )


# The function patch was taken somewhere over the rainbow...
_hdr_pat = re.compile(r"^@@ -(\d+),?(\d+)? \+(\d+),?(\d+)? @@$")


def patch(diff, source):
    """Apply unified diff patch to string s to recover newer string.  If
    revert is True, treat s as the newer string, recover older string.

    """
    s = source.splitlines(True)
    p = diff.splitlines(True)
    t = ""
    i = sl = 0
    (midx, sign) = (1, "+")
    while i < len(p) and p[i].startswith(("---", "+++")):
        i += 1  # skip header lines
    while i < len(p):
        m = _hdr_pat.match(p[i])
        if not m:
            raise Exception("Cannot process diff")
        i += 1
        l = int(m.group(midx)) - 1 + (m.group(midx + 1) == "0")
        t += "".join(s[sl:l])
        sl = l
        while i < len(p) and p[i][0] != "@":
            if i + 1 < len(p) and p[i + 1][0] == "\\":
                line = p[i][:-1]
                i += 2
            else:
                line = p[i]
                i += 1
            if len(line) > 0:
                if line[0] == sign or line[0] == " ":
                    t += line[1:]
                sl += line[0] != sign
    t += "\n" + "".join(s[sl:])
    return t


def glob2predicate(patterns):
    def regex_join(regexes):
        """Combine a list of regexes into one that matches any of them."""
        return "|".join("(?:%s)" % r for r in regexes)

    regexes = (fnmatch.translate(pattern) for pattern in patterns)
    regex = re.compile(regex_join(regexes))

    def predicate(path):
        return regex.match(path) is not None

    return predicate


def ast_walk(tree):
    """Depth-first traversal of an AST, yielding every node."""
    yield tree
    for child in ast.iter_child_nodes(tree):
        yield from ast_walk(child)


def copy_tree_at(tree, index):
    """Deep-copy *tree* and return (copy, node_at_index_in_copy)."""
    tree_copy = deepcopy(tree)
    return tree_copy, list(ast_walk(tree_copy))[index]


def get_parent_field_idx(tree, node):
    """Return (parent, field_name, list_index_or_None) for *node* in *tree*."""
    for parent in ast_walk(tree):
        for field, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for i, child in enumerate(value):
                    if child is node:
                        return parent, field, i
            elif value is node:
                return parent, field, None
    return None, None, None


@contextmanager
def timeit():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


class Database:
    def __init__(self, path, timeout=300):
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=timeout)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS mutations "
            "(uid BLOB PRIMARY KEY, path TEXT, diff BLOB)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS results "
            "(uid BLOB PRIMARY KEY, status INTEGER)"
        )
        self._conn.commit()
        try:
            self._conn.execute("ALTER TABLE results ADD COLUMN classification INTEGER")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._conn.close()

    # --- config ---
    def get_config(self, key):
        row = self._conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def set_config(self, key, value):
        self._conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    # --- mutations ---
    def store_mutations(self, rows):
        """Insert multiple (uid, path, diff) rows in a single transaction."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO mutations (uid, path, diff) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get_mutation(self, uid):
        row = self._conn.execute(
            "SELECT path, diff FROM mutations WHERE uid = ?", (uid,)
        ).fetchone()
        if row is None:
            raise KeyError(uid)
        return row[0], row[1]  # path: str, diff: bytes

    def list_mutations(self):
        return self._conn.execute(
            "SELECT uid FROM mutations ORDER BY uid"
        ).fetchall()

    # --- results ---
    def set_result(self, uid, status):
        self._conn.execute(
            "INSERT OR REPLACE INTO results (uid, status) VALUES (?, ?)",
            (uid, status),
        )
        self._conn.commit()

    def del_result(self, uid):
        self._conn.execute("DELETE FROM results WHERE uid = ?", (uid,))
        self._conn.commit()

    def list_results(self, status=None):
        if status is not None:
            return self._conn.execute(
                "SELECT uid, status FROM results WHERE status = ? ORDER BY uid",
                (status,),
            ).fetchall()
        return self._conn.execute(
            "SELECT uid, status FROM results ORDER BY uid"
        ).fetchall()

    def count_results(self):
        return self._conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def count_mutations(self):
        return self._conn.execute("SELECT COUNT(*) FROM mutations").fetchone()[0]

    def set_classification(self, uid, cls):
        self._conn.execute(
            "UPDATE results SET classification = ? WHERE uid = ?", (cls, uid)
        )
        self._conn.commit()

    def list_results_for_replay(self):
        """Return uids in the replay queue: survived + not permanently dismissed."""
        return self._conn.execute(
            "SELECT uid FROM results "
            "WHERE status IN (0, 1) "
            "AND (classification IS NULL OR classification NOT IN (?, ?))",
            (CLASSIFICATION_EQUIVALENT, CLASSIFICATION_WONT_FIX),
        ).fetchall()

    def get_classification_counts(self):
        """Return dict mapping classification value (or None) -> count."""
        rows = self._conn.execute(
            "SELECT classification, COUNT(*) FROM results GROUP BY classification"
        ).fetchall()
        return {cls: count for cls, count in rows}


class Mutation(type):
    ALL = set()
    DEADCODE = set()

    deadcode_detection = False

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        obj = cls()
        type(cls).ALL.add(obj)
        if cls.deadcode_detection:
            type(cls).DEADCODE.add(obj)


class StatementDrop(metaclass=Mutation):
    """Replace a statement with pass, verifying that no covered statement is inert dead code."""

    deadcode_detection = True

    def predicate(self, node):
        return isinstance(node, ast.stmt) and not isinstance(
            node, (ast.Expr, ast.Pass, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None or idx is None:
            return
        replacement = ast.Pass(lineno=node_copy.lineno, col_offset=node_copy.col_offset)
        getattr(parent, field)[idx] = replacement
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class DefinitionDrop(metaclass=Mutation):
    """Remove a function or class definition entirely (only when others remain in the same body), surfacing unreferenced definitions."""

    deadcode_detection = True

    def predicate(self, node):
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None or idx is None:
            return
        body = getattr(parent, field)
        if len(body) <= 1:
            return
        body.pop(idx)
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


def chunks(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    iterable = iter(iterable)
    for chunk in tuple(itertools.islice(iterable, n)):
        yield chunk


class MutateNumber(metaclass=Mutation):
    """Replace an integer or float literal with a random value in the same bit-range, verifying that the exact numeric value is tested."""

    COUNT = 5

    def predicate(self, node):
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        )

    def mutate(self, node, index, tree):
        value = node.value

        if isinstance(value, int):
            def randomize(x):
                return random.randint(0, x)
        else:
            def randomize(x):
                return random.random() * x

        for size in range(8, 32):
            if value < 2 ** size:
                break

        count = 0
        while count != self.COUNT:
            count += 1
            new_value = randomize(2 ** size)
            if new_value == value:
                continue
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.value = new_value
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateString(metaclass=Mutation):
    """Prepend a fixed prefix to a string or bytes literal, verifying that callers check the actual content."""

    def predicate(self, node):
        return isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        if isinstance(node_copy.value, bytes):
            node_copy.value = b"coffeebad" + node_copy.value
        else:
            node_copy.value = "mutated string " + node_copy.value
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class MutateKeyword(metaclass=Mutation):
    """Rotate flow keywords (break/continue/pass), swap boolean constants (True/False/None), and flip boolean operators (and/or)."""

    FLOW_STMTS = (ast.Continue, ast.Break, ast.Pass)
    BOOL_OPS = (ast.And, ast.Or)

    def predicate(self, node):
        if isinstance(node, self.FLOW_STMTS):
            return True
        if isinstance(node, ast.Constant) and (
            node.value is True or node.value is False or node.value is None
        ):
            return True
        if isinstance(node, ast.BoolOp):
            return True
        return False

    def mutate(self, node, index, tree):
        if isinstance(node, self.FLOW_STMTS):
            for new_cls in self.FLOW_STMTS:
                if isinstance(node, new_cls):
                    continue
                tree_copy, node_copy = copy_tree_at(tree, index)
                parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
                if parent is None or idx is None:
                    continue
                getattr(parent, field)[idx] = new_cls(
                    lineno=node_copy.lineno, col_offset=node_copy.col_offset
                )
                ast.fix_missing_locations(tree_copy)
                yield tree_copy, node_copy

        elif isinstance(node, ast.Constant):
            if node.value is True:
                swaps = [False, None]
            elif node.value is False:
                swaps = [True, None]
            else:
                swaps = [True, False]
            for new_value in swaps:
                tree_copy, node_copy = copy_tree_at(tree, index)
                node_copy.value = new_value
                ast.fix_missing_locations(tree_copy)
                yield tree_copy, node_copy

        elif isinstance(node, ast.BoolOp):
            for new_op_cls in self.BOOL_OPS:
                if isinstance(node.op, new_op_cls):
                    continue
                tree_copy, node_copy = copy_tree_at(tree, index)
                node_copy.op = new_op_cls()
                ast.fix_missing_locations(tree_copy)
                yield tree_copy, node_copy


class Comparison(metaclass=Mutation):
    """Negate a comparison expression by wrapping it with not (...), verifying that the direction of every comparison is tested."""

    def predicate(self, node):
        return isinstance(node, ast.Compare)

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None:
            return
        not_node = ast.UnaryOp(
            op=ast.Not(),
            operand=node_copy,
            lineno=node_copy.lineno,
            col_offset=node_copy.col_offset,
        )
        if idx is not None:
            getattr(parent, field)[idx] = not_node
        else:
            setattr(parent, field, not_node)
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, not_node


class MutateOperator(metaclass=Mutation):
    """Replace an arithmetic, bitwise, shift, or comparison operator with another in the same group, verifying the exact operator matters."""

    BINARY_OPS = [
        ast.Add, ast.Sub, ast.Mod, ast.BitOr, ast.BitAnd,
        ast.FloorDiv, ast.Div, ast.Mult, ast.BitXor, ast.Pow, ast.MatMult,
    ]
    SHIFT_OPS = [ast.LShift, ast.RShift]
    CMP_OPS = [ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.GtE, ast.Gt]

    BINOP_GROUPS = [BINARY_OPS, SHIFT_OPS]

    def predicate(self, node):
        return isinstance(node, (ast.BinOp, ast.AugAssign, ast.Compare))

    def mutate(self, node, index, tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            for op_group in self.BINOP_GROUPS:
                if type(node.op) not in op_group:
                    continue
                for new_op_cls in op_group:
                    if new_op_cls is type(node.op):
                        continue
                    tree_copy, node_copy = copy_tree_at(tree, index)
                    node_copy.op = new_op_cls()
                    ast.fix_missing_locations(tree_copy)
                    yield tree_copy, node_copy

        elif isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) not in self.CMP_OPS:
                    continue
                for new_op_cls in self.CMP_OPS:
                    if new_op_cls is type(op):
                        continue
                    tree_copy, node_copy = copy_tree_at(tree, index)
                    node_copy.ops[i] = new_op_cls()
                    ast.fix_missing_locations(tree_copy)
                    yield tree_copy, node_copy


if hasattr(ast, "Match"):

    class MutateMatchCase(metaclass=Mutation):
        """Remove one case branch at a time from a match statement (Python 3.10+), verifying that each branch is exercised by the test suite."""

        def predicate(self, node):
            return isinstance(node, ast.Match) and len(node.cases) > 1

        def mutate(self, node, index, tree):
            for i in range(len(node.cases)):
                tree_copy, node_copy = copy_tree_at(tree, index)
                node_copy.cases.pop(i)
                ast.fix_missing_locations(tree_copy)
                yield tree_copy, node_copy


_STRING_METHOD_SWAPS = {
    "lower": ["upper"], "upper": ["lower"],
    "lstrip": ["rstrip", "removeprefix"], "rstrip": ["lstrip", "removesuffix"],
    "find": ["rfind"], "rfind": ["find"],
    "ljust": ["rjust"], "rjust": ["ljust"],
    "removeprefix": ["removesuffix"], "removesuffix": ["removeprefix"],
    "partition": ["rpartition"], "rpartition": ["partition"],
    "split": ["rsplit"], "rsplit": ["split"],
}


class MutateStringMethod(metaclass=Mutation):
    """Swap directionally symmetric string methods (lower↔upper, lstrip↔rstrip, lstrip↔removeprefix, rstrip↔removesuffix, find↔rfind, ljust↔rjust, removeprefix↔removesuffix, partition↔rpartition, split↔rsplit), verifying that the direction matters."""

    def predicate(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _STRING_METHOD_SWAPS
        )

    def mutate(self, node, index, tree):
        for target_attr in _STRING_METHOD_SWAPS[node.func.attr]:
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.func.attr = target_attr
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateCallArgs(metaclass=Mutation):
    """Replace each positional call argument with None, and drop one argument at a time from multi-argument calls, verifying that every argument is actually used."""

    def predicate(self, node):
        return isinstance(node, ast.Call) and len(node.args) > 0

    def mutate(self, node, index, tree):
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Constant) and arg.value is None:
                continue
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.args[i] = ast.Constant(
                value=None, lineno=arg.lineno, col_offset=arg.col_offset
            )
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy

        if len(node.args) > 1:
            for i in range(len(node.args)):
                tree_copy, node_copy = copy_tree_at(tree, index)
                node_copy.args.pop(i)
                ast.fix_missing_locations(tree_copy)
                yield tree_copy, node_copy


class ForceConditional(metaclass=Mutation):
    """Force the test of an if/while/assert/ternary to always be True or always False, verifying that both branches are meaningfully exercised."""

    def predicate(self, node):
        return isinstance(node, (ast.If, ast.While, ast.Assert, ast.IfExp))

    def mutate(self, node, index, tree):
        for value in (True, False):
            if isinstance(node.test, ast.Constant) and node.test.value is value:
                continue
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.test = ast.Constant(
                value=value, lineno=node_copy.test.lineno, col_offset=node_copy.test.col_offset
            )
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateExceptionHandler(metaclass=Mutation):
    """Replace the specific exception type in an except clause with the generic Exception, verifying that the handler is tested for the right error kind."""

    def predicate(self, node):
        return isinstance(node, ast.ExceptHandler) and node.type is not None

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        node_copy.type = ast.Name(
            id="Exception",
            ctx=ast.Load(),
            lineno=node_copy.type.lineno,
            col_offset=node_copy.type.col_offset,
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class ZeroIteration(metaclass=Mutation):
    """Replace a for-loop's iterable with an empty list, forcing the body to never execute, verifying that callers handle empty-collection cases."""

    def predicate(self, node):
        return isinstance(node, (ast.For, ast.AsyncFor))

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        node_copy.iter = ast.List(
            elts=[],
            ctx=ast.Load(),
            lineno=node_copy.iter.lineno,
            col_offset=node_copy.iter.col_offset,
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class RemoveDecorator(metaclass=Mutation):
    """Remove one decorator at a time from a decorated function or class, verifying that each decorator's effect is covered by tests."""

    def predicate(self, node):
        return (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and len(node.decorator_list) > 0
        )

    def mutate(self, node, index, tree):
        for i in range(len(node.decorator_list)):
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.decorator_list.pop(i)
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class NegateCondition(metaclass=Mutation):
    """Wrap a bare (non-comparison) condition with not, inserting the logical inverse of the test, verifying that the truthiness of the value actually matters."""

    def predicate(self, node):
        return isinstance(node, (ast.If, ast.While, ast.Assert, ast.IfExp)) and not isinstance(
            node.test, ast.Compare
        )

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        test = node_copy.test
        node_copy.test = ast.UnaryOp(
            op=ast.Not(),
            operand=test,
            lineno=test.lineno,
            col_offset=test.col_offset,
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class MutateReturn(metaclass=Mutation):
    """Replace a return value with a type-appropriate default (None, 0, False, or ""), verifying that callers check what the function returns."""

    DEFAULTS = [None, 0, False, ""]

    def predicate(self, node):
        return isinstance(node, ast.Return) and node.value is not None

    def mutate(self, node, index, tree):
        for default in self.DEFAULTS:
            if isinstance(node.value, ast.Constant) and node.value.value is default:
                continue
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.value = ast.Constant(
                value=default, lineno=node_copy.lineno, col_offset=node_copy.col_offset
            )
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateLambda(metaclass=Mutation):
    """Replace the body of a lambda with None (or 0 when the body is already None), verifying that the lambda's computation is actually used."""

    def predicate(self, node):
        return isinstance(node, ast.Lambda)

    def mutate(self, node, index, tree):
        new_value = 0 if (isinstance(node.body, ast.Constant) and node.body.value is None) else None
        tree_copy, node_copy = copy_tree_at(tree, index)
        node_copy.body = ast.Constant(
            value=new_value, lineno=node_copy.body.lineno, col_offset=node_copy.body.col_offset
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class MutateAssignment(metaclass=Mutation):
    """Replace the right-hand side of a plain assignment with None, verifying that the assigned value is not silently ignored."""

    def predicate(self, node):
        return isinstance(node, ast.Assign) and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        )

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        node_copy.value = ast.Constant(
            value=None, lineno=node_copy.lineno, col_offset=node_copy.col_offset
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class AugAssignToAssign(metaclass=Mutation):
    """Convert an augmented assignment (x += v) to a plain assignment (x = v), dropping the accumulation, verifying that the update operator is tested."""

    def predicate(self, node):
        return isinstance(node, ast.AugAssign)

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None or idx is None:
            return
        assign = ast.Assign(
            targets=[node_copy.target],
            value=node_copy.value,
            lineno=node_copy.lineno,
            col_offset=node_copy.col_offset,
        )
        getattr(parent, field)[idx] = assign
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class RemoveUnaryOp(metaclass=Mutation):
    """Strip a unary operator (not, -, ~) and leave only the operand, verifying that the operator's effect is covered by tests."""

    def predicate(self, node):
        return isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.Not, ast.USub, ast.Invert)
        )

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None:
            return
        operand = node_copy.operand
        if idx is not None:
            getattr(parent, field)[idx] = operand
        else:
            setattr(parent, field, operand)
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class MutateIdentity(metaclass=Mutation):
    """Swap is ↔ is not in identity comparisons, verifying that the expected identity relationship is directly tested."""

    def predicate(self, node):
        return isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
        )

    def mutate(self, node, index, tree):
        for i, op in enumerate(node.ops):
            if not isinstance(op, (ast.Is, ast.IsNot)):
                continue
            new_op = ast.IsNot() if isinstance(op, ast.Is) else ast.Is()
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.ops[i] = new_op
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateContainment(metaclass=Mutation):
    """Swap in ↔ not in in membership tests, verifying that the expected membership relationship is directly tested."""

    def predicate(self, node):
        return isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        )

    def mutate(self, node, index, tree):
        for i, op in enumerate(node.ops):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            new_op = ast.NotIn() if isinstance(op, ast.In) else ast.In()
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.ops[i] = new_op
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class BreakToReturn(metaclass=Mutation):
    """Replace break with return, exiting the enclosing function instead of just the loop, verifying that the loop's exit path is tested."""

    def predicate(self, node):
        return isinstance(node, ast.Break)

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None or idx is None:
            return
        getattr(parent, field)[idx] = ast.Return(
            value=None, lineno=node_copy.lineno, col_offset=node_copy.col_offset
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class SwapArguments(metaclass=Mutation):
    """Swap each pair of positional call arguments, verifying that argument order is tested."""

    def predicate(self, node):
        return isinstance(node, ast.Call) and len(node.args) >= 2

    def mutate(self, node, index, tree):
        for i in range(len(node.args)):
            for j in range(i + 1, len(node.args)):
                tree_copy, node_copy = copy_tree_at(tree, index)
                node_copy.args[i], node_copy.args[j] = node_copy.args[j], node_copy.args[i]
                ast.fix_missing_locations(tree_copy)
                yield tree_copy, node_copy


class MutateSlice(metaclass=Mutation):
    """Drop the lower or upper bound of a slice (a[i:j] → a[:j] or a[i:]) and negate the step (a[::2] → a[::-2]), verifying that slice boundary conditions and direction are tested."""

    def predicate(self, node):
        return isinstance(node, ast.Slice) and (
            node.lower is not None or node.upper is not None or node.step is not None
        )

    def mutate(self, node, index, tree):
        if node.lower is not None:
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.lower = None
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy
        if node.upper is not None:
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.upper = None
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy
        if node.step is not None:
            tree_copy, node_copy = copy_tree_at(tree, index)
            step = node_copy.step
            if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
                node_copy.step = step.operand
            else:
                node_copy.step = ast.UnaryOp(
                    op=ast.USub(),
                    operand=step,
                    lineno=step.lineno,
                    col_offset=step.col_offset,
                )
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateYield(metaclass=Mutation):
    """Replace the value of a yield expression with None, verifying that the yielded value is actually used by callers."""

    def predicate(self, node):
        return (
            isinstance(node, ast.Yield)
            and node.value is not None
            and not (isinstance(node.value, ast.Constant) and node.value.value is None)
        )

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        node_copy.value = ast.Constant(
            value=None, lineno=node_copy.lineno, col_offset=node_copy.col_offset
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class MutateDefaultArgument(metaclass=Mutation):
    """Remove leading default argument values one at a time, making parameters required, verifying that callers always supply them explicitly."""

    def predicate(self, node):
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and (
            len(node.args.defaults) > 0
            or any(d is not None for d in node.args.kw_defaults)
        )

    def mutate(self, node, index, tree):
        for i in range(len(node.args.defaults)):
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.args.defaults = node_copy.args.defaults[i + 1:]
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy
        for i, default in enumerate(node.args.kw_defaults):
            if default is None:
                continue
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.args.kw_defaults[i] = None
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateIterator(metaclass=Mutation):
    """Wrap a for-loop's iterable in reversed() or random.shuffle(), verifying that iteration order assumptions are tested."""

    def predicate(self, node):
        return isinstance(node, (ast.For, ast.AsyncFor)) and not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "reversed"
        )

    def mutate(self, node, index, tree):
        # Mutation 1: reversed(iterable)
        tree_copy, node_copy = copy_tree_at(tree, index)
        node_copy.iter = ast.Call(
            func=ast.Name(id="reversed", ctx=ast.Load()),
            args=[node_copy.iter],
            keywords=[],
            lineno=node_copy.iter.lineno,
            col_offset=node_copy.iter.col_offset,
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy

        # Mutation 2: (__import__('random').shuffle(_s := list(iterable)) or _s)
        # Shuffles the iterable in-place without adding an import statement.
        tree_copy, node_copy = copy_tree_at(tree, index)
        lineno = node_copy.iter.lineno
        col = node_copy.iter.col_offset
        seq_name = ast.Name(id="_mutation_seq_", ctx=ast.Store(), lineno=lineno, col_offset=col)
        list_call = ast.Call(
            func=ast.Name(id="list", ctx=ast.Load(), lineno=lineno, col_offset=col),
            args=[node_copy.iter],
            keywords=[],
            lineno=lineno,
            col_offset=col,
        )
        walrus = ast.NamedExpr(target=seq_name, value=list_call, lineno=lineno, col_offset=col)
        shuffle_call = ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="__import__", ctx=ast.Load(), lineno=lineno, col_offset=col),
                    args=[ast.Constant(value="random", lineno=lineno, col_offset=col)],
                    keywords=[],
                    lineno=lineno,
                    col_offset=col,
                ),
                attr="shuffle",
                ctx=ast.Load(),
                lineno=lineno,
                col_offset=col,
            ),
            args=[walrus],
            keywords=[],
            lineno=lineno,
            col_offset=col,
        )
        seq_load = ast.Name(id="_mutation_seq_", ctx=ast.Load(), lineno=lineno, col_offset=col)
        node_copy.iter = ast.BoolOp(
            op=ast.Or(),
            values=[shuffle_call, seq_load],
            lineno=lineno,
            col_offset=col,
        )
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class MutateContextManager(metaclass=Mutation):
    """Strip context managers from a with statement one at a time, keeping the body, verifying that each manager's effect is tested."""

    def predicate(self, node):
        return isinstance(node, (ast.With, ast.AsyncWith))

    def mutate(self, node, index, tree):
        for i in range(len(node.items)):
            tree_copy, node_copy = copy_tree_at(tree, index)
            if len(node_copy.items) == 1:
                parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
                if parent is None or idx is None:
                    continue
                getattr(parent, field)[idx:idx + 1] = node_copy.body
            else:
                node_copy.items.pop(i)
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateFString(metaclass=Mutation):
    """Replace each interpolated expression in an f-string with an empty string, verifying that callers check the formatted content rather than just the surrounding template."""

    def predicate(self, node):
        return isinstance(node, ast.JoinedStr) and any(
            isinstance(v, ast.FormattedValue) for v in node.values
        )

    def mutate(self, node, index, tree):
        for i, value in enumerate(node.values):
            if not isinstance(value, ast.FormattedValue):
                continue
            tree_copy, node_copy = copy_tree_at(tree, index)
            node_copy.values[i] = ast.Constant(
                value="", lineno=node_copy.lineno, col_offset=node_copy.col_offset
            )
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


class MutateGlobal(metaclass=Mutation):
    """Remove a global or nonlocal declaration entirely, causing assignments to bind a local variable instead, verifying that the scoping is exercised by tests."""

    deadcode_detection = True

    def predicate(self, node):
        return isinstance(node, (ast.Global, ast.Nonlocal))

    def mutate(self, node, index, tree):
        tree_copy, node_copy = copy_tree_at(tree, index)
        parent, field, idx = get_parent_field_idx(tree_copy, node_copy)
        if parent is None or idx is None:
            return
        body = getattr(parent, field)
        if len(body) <= 1:
            return
        body.pop(idx)
        ast.fix_missing_locations(tree_copy)
        yield tree_copy, node_copy


class InjectException(metaclass=Mutation):
    """Replace expressions with the exception they can raise, targeting error-handling paths that are commonly forgotten."""

    _DICT_HINTS = frozenset(["dict", "map", "table", "cache", "store", "config", "registry", "lookup"])
    _LIST_HINTS = frozenset(["list", "array", "arr", "seq", "items", "elements"])

    def predicate(self, node):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return True
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            return True
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("int", "float", "open", "next")):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            return True
        if isinstance(node, ast.Attribute):
            return True
        return False

    def _type_specs(self, node):
        """Return list of (exc_name, args_fn) where args_fn(node_copy) -> list of AST nodes."""
        if isinstance(node, (ast.For, ast.AsyncFor)):
            return [("StopIteration", lambda n: [])]
        if isinstance(node, ast.Subscript):
            return self._subscript_specs(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn in ("int", "float"):
                return [("ValueError", lambda n: n.args[:1] if n.args else [])]
            if fn == "open":
                return [("FileNotFoundError", lambda n: n.args[:1] if n.args else [])]
            if fn == "next":
                return [("StopIteration", lambda n: [])]
        if isinstance(node, ast.BinOp):
            return [("ZeroDivisionError", lambda n: [])]
        if isinstance(node, ast.Attribute):
            return [("AttributeError", lambda n: [ast.Constant(
                value=n.attr, lineno=n.lineno, col_offset=n.col_offset,
            )])]
        return []

    def _subscript_specs(self, node):
        """Heuristic: KeyError for dict-like, IndexError for list-like, both if ambiguous."""
        slice_ = node.slice
        value = node.value
        # String key → definitely KeyError
        if isinstance(slice_, ast.Constant) and isinstance(slice_.value, str):
            return [("KeyError", lambda n: [n.slice])]
        # Integer key → more likely IndexError, but could be dict
        if isinstance(slice_, ast.Constant) and isinstance(slice_.value, int):
            return [("IndexError", lambda n: [n.slice])]
        # Check variable name for hints
        name = None
        if isinstance(value, ast.Name):
            name = value.id.lower()
        elif isinstance(value, ast.Attribute):
            name = value.attr.lower()
        if name:
            if any(h in name for h in self._DICT_HINTS):
                return [("KeyError", lambda n: [n.slice])]
            if any(h in name for h in self._LIST_HINTS):
                return [("IndexError", lambda n: [n.slice])]
        # Ambiguous — generate both
        return [
            ("KeyError", lambda n: [n.slice]),
            ("IndexError", lambda n: [n.slice]),
        ]

    def _build_parent_map(self, tree):
        parent_map = {}
        for n in ast.walk(tree):
            for child in ast.iter_child_nodes(n):
                parent_map[id(child)] = n
        return parent_map

    def _find_enclosing_stmt(self, parent_map, node):
        """Return (stmt, parent, field, idx) for the innermost statement in a body list."""
        current = node
        while id(current) in parent_map:
            parent = parent_map[id(current)]
            if isinstance(current, ast.stmt):
                for field, value in ast.iter_fields(parent):
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            if item is current:
                                return current, parent, field, i
            current = parent
        return None, None, None, None

    def _is_guarded(self, parent_map, node, exc_name):
        """Return True if node is inside an except block or a try that handles exc_name."""
        current = node
        while id(current) in parent_map:
            parent = parent_map[id(current)]
            if isinstance(parent, ast.ExceptHandler):
                return True  # never inject inside except blocks
            if isinstance(parent, ast.Try) and any(current is s for s in parent.body):
                for handler in parent.handlers:
                    if handler.type is None:
                        return True  # bare except catches everything
                    names = []
                    if isinstance(handler.type, ast.Name):
                        names = [handler.type.id]
                    elif isinstance(handler.type, ast.Tuple):
                        names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
                    if exc_name in names or "Exception" in names or "BaseException" in names:
                        return True
            current = parent
        return False

    def _make_raise(self, exc_name, args, lineno, col_offset):
        if args:
            exc = ast.Call(
                func=ast.Name(id=exc_name, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                args=args,
                keywords=[],
                lineno=lineno,
                col_offset=col_offset,
            )
        else:
            exc = ast.Name(id=exc_name, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
        return ast.Raise(exc=exc, cause=None, lineno=lineno, col_offset=col_offset)

    def mutate(self, node, index, tree):
        specs = self._type_specs(node)
        for exc_name, args_fn in specs:
            tree_copy, node_copy = copy_tree_at(tree, index)
            parent_map = self._build_parent_map(tree_copy)
            if self._is_guarded(parent_map, node_copy, exc_name):
                continue
            lineno = getattr(node_copy, "lineno", 1)
            col_offset = getattr(node_copy, "col_offset", 0)
            stmt, parent, field, idx = self._find_enclosing_stmt(parent_map, node_copy)
            if stmt is None or idx is None:
                continue
            exc_args = args_fn(node_copy)
            raise_node = self._make_raise(exc_name, exc_args, lineno, col_offset)
            getattr(parent, field)[idx] = raise_node
            ast.fix_missing_locations(tree_copy)
            yield tree_copy, node_copy


def diff(source, target, filename=""):
    lines = unified_diff(
        source.split("\n"), target.split("\n"), filename, filename, lineterm=""
    )
    out = "\n".join(lines)
    return out


def mutate(node, index, tree, mutations):
    for mutation in mutations:
        if not mutation.predicate(node):
            continue
        yield from mutation.mutate(node, index, tree)


def interesting(node, coverage):
    return getattr(node, "lineno", None) in coverage


def iter_deltas(source, path, coverage, mutations):
    tree = ast.parse(source)
    canonical = ast.unparse(tree)
    ignored = 0
    invalid = 0
    for index, node in enumerate(ast_walk(tree)):
        for tree_copy, new_node in mutate(node, index, tree, mutations):
            if not interesting(new_node, coverage):
                ignored += 1
                continue
            target = ast.unparse(tree_copy)
            try:
                ast.parse(target)
            except SyntaxError:
                invalid += 1
                continue
            delta = diff(canonical, target, path)
            yield delta
    if ignored > 1:
        msg = "Ignored {} mutations from file at {}"
        msg += " because there is no associated coverage."
        log.trace(msg, ignored, path)
    if invalid > 0:
        msg = "Skipped {} invalid (syntax error) mutations from {}"
        log.trace(msg, invalid, path)


async def pool_for_each_par_map(loop, pool, f, p, iterator):
    zx = stream.iterate(iterator)
    zx = zx | pipe.map(lambda x: loop.run_in_executor(pool, p, x))
    async with zx.stream() as streamer:
        limit = pool._max_workers
        unfinished = []
        while True:
            tasks = []
            for i in range(limit):
                try:
                    task = await streamer.__anext__()
                except StopAsyncIteration:
                    limit = 0
                else:
                    tasks.append(task)
            tasks = tasks + list(unfinished)
            if not tasks:
                break
            finished, unfinished = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for finish in finished:
                out = finish.result()
                f(out)
            limit = pool._max_workers - len(unfinished)


def mutation_create(item):
    path, source, coverage, mutation_predicate = item

    if not coverage:
        msg = "Ignoring file {} because there is no associated coverage."
        log.trace(msg, path)
        return []

    log.trace("Mutating file: {}...", path)
    mutations = [m for m in Mutation.ALL if mutation_predicate(m)]
    deltas = iter_deltas(source, path, coverage, mutations)
    # return the compressed deltas to save some time in the
    # mainthread.
    out = [(path, zstd.compress(x.encode("utf8"))) for x in deltas]
    log.trace("There is {} mutations for the file `{}`", len(out), path)
    return out


def install_module_loader(uid):
    mutation_show(uid.hex)

    with Database(".mutation.db") as db:
        path, diff = db.get_mutation(uid.bytes)
    diff = zstd.decompress(diff).decode("utf8")

    with open(path) as f:
        source = f.read()

    patched = patch(diff, ast.unparse(ast.parse(source)))

    # Derive the importable module name by finding which sys.path entry
    # contains this file.  For src/ layouts (e.g. src/mypkg/__init__.py)
    # the editable install adds src/ to sys.path, so the module name is
    # "mypkg" not "src.mypkg".  We resolve every sys.path entry and pick
    # the one that gives the *shortest* (most specific) module name.
    path_obj = Path(path).resolve()
    if path_obj.name == "__init__.py":
        module_file = path_obj.parent   # package dir: strip __init__.py
    else:
        module_file = path_obj.with_suffix("")  # regular module: strip .py

    module_path = None
    for pythonpath in sys.path:
        base = Path(pythonpath).resolve() if pythonpath else Path(".").resolve()
        try:
            rel = module_file.relative_to(base)
        except ValueError:
            continue
        candidate = ".".join(rel.parts)
        if module_path is None or len(candidate) < len(module_path):
            module_path = candidate
    if module_path is None:
        raise Exception("sys.path oops!")

    patched_module = types.ModuleType(module_path)
    try:
        exec(patched, patched_module.__dict__)
    except Exception:
        exec("", patched_module.__dict__)

    sys.modules[module_path] = patched_module


def pytest_configure(config):
    mutation = config.getoption("mutation", default=None)
    if mutation is not None:
        uid = UUID(hex=mutation)
        install_module_loader(uid)


def pytest_addoption(parser, pluginmanager):
    try:
        parser.addoption("--mutation", dest="mutation", type=str)
    except ValueError:
        pass  # already registered (e.g. conftest.py + -p mutation both active)


def for_each_par_map(loop, pool, inc, proc, items):
    out = []
    for item in items:
        item = proc(item)
        item = inc(item)
        out.append(item)
    return out


def mutation_pass(args):  # TODO: rename
    command, uid, timeout = args
    # Check if this mutation was previously classified as equivalent
    with database_open(".", timeout=timeout) as db:
        _, diff_bytes = db.get_mutation(uid)
    diff_text = zstd.decompress(diff_bytes).decode("utf8")
    ignored_file = Path(".mutations.ignored") / "{}.diff".format(diff_hash(diff_text))
    if ignored_file.exists():
        log.debug("Skipping ignored mutation: {}", uid.hex())
        with database_open(".", timeout=timeout) as db:
            db.del_result(uid)
        return True
    command = command + ["--mutation={}".format(uid.hex())]
    log.debug("Running command: {}", ' '.join(command))
    out = run(command, timeout=timeout, silent=True)
    if out == 4:
        # pytest exit code 4 = "command line usage error": --mutation flag was
        # not recognised, which means mutation.py is not loaded as a pytest
        # plugin.  Treat this as a hard error so it doesn't silently look like
        # every mutation was caught.
        log.error(
            "pytest exited with code 4 (unrecognised arguments) for command: `{}`\n"
            "Hint: mutation.py is not loaded as a pytest plugin. "
            "Add `pytest_plugins = [\"mutation\"]` to your conftest.py.",
            " ".join(command),
        )
        sys.exit(1)
    if out == 0:
        msg = "no error with mutation: {} ({})"
        log.trace(msg, " ".join(command), out)
        with database_open(".", timeout=timeout) as db:
            db.set_result(uid, 0)
        return False
    else:
        # TODO: pass root path...
        with database_open(".", timeout=timeout) as db:
            db.del_result(uid)
        return True


PYTEST = "python3 -m pytest -p mutation --exitfirst --no-header --tb=no --quiet --assert=plain"
PYTEST = shlex.split(PYTEST)


def coverage_read(root):
    coverage = Coverage(".coverage")  # use pathlib
    coverage.load()
    data = coverage.get_data()
    filepaths = data.measured_files()
    out = dict()
    root = root.resolve()
    for filepath in filepaths:
        if filepath.startswith(str(root)):
            key = str(Path(filepath).relative_to(root))
        else:
            # coverage.py sometimes records relative paths; resolve against root
            resolved = (root / filepath).resolve()
            if not resolved.is_relative_to(root):
                continue
            key = str(resolved.relative_to(root))
        value = set(data.lines(filepath) or [])
        out[key] = value
    return out


def database_open(root, recreate=False, timeout=300):
    root = root if isinstance(root, Path) else Path(root)
    db = root / ".mutation.db"
    if recreate and db.exists():
        log.trace("Deleting existing database...")
        for file in root.glob(".mutation.db*"):
            file.unlink()

    if not recreate and not db.exists():
        log.error("No database, can not proceed!")
        sys.exit(1)

    return Database(str(db), timeout=timeout)


def run(command, timeout=None, silent=True, verbose=False):
    if timeout and timeout < 60:
        timeout = 60

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    devnull = subprocess.DEVNULL if (silent and not verbose) else None

    try:
        result = subprocess.run(
            command,
            env=env,
            timeout=timeout,
            stdout=devnull,
            stderr=devnull,
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        return 1


def sampling_setup(sampling, total):

    if sampling is None:
        sampling = "100%"

    if sampling.endswith("%"):
        # randomly choose percent mutations
        cutoff = float(sampling[:-1]) / 100

        def sampler(iterable):
            for item in iterable:
                value = random.random()
                if value < cutoff:
                    yield item

        total = int(total * cutoff)
    elif sampling.isdigit():
        # otherwise, it is the first COUNT mutations that are used.
        total = int(sampling)

        def sampler(iterable):
            remaining = total
            for item in iterable:
                yield item
                remaining -= 1
                if remaining == 0:
                    return

    else:
        msg = "Sampling passed via --sampling option must be a positive"
        msg += " integer or a percentage!"
        log.error(msg)
        sys.exit(2)

    if sampling:
        log.info("Taking into account sampling there is {} mutations.", total)

    return sampler, total


# TODO: the `command` is a hack, maybe there is a way to avoid the
# following code: `if command is not None.
def check_tests(root, seed, arguments, command=None):
    max_workers = int(arguments["--max-workers"] or (os.cpu_count() - 1) or 1)

    log.info("Let's check that the tests are green...")

    if arguments["<file-or-directory>"] and arguments["PYTEST-COMMAND"]:
        log.error("<file-or-directory> and PYTEST-COMMAND are exclusive!")
        sys.exit(1)

    if command is not None:
        command = list(command)
        if max_workers > 1:
            command.extend(
                [
                    # Use pytest-xdist to make sure it is possible to run the
                    # tests in parallel
                    "--numprocesses={}".format(max_workers),
                ]
            )
    else:
        if arguments["PYTEST-COMMAND"]:
            command = list(arguments["PYTEST-COMMAND"])
        else:
            command = list(PYTEST)
            command.extend(arguments["<file-or-directory>"])

        if max_workers > 1:
            command.append(
                # Use pytest-xdist to make sure it is possible to run
                # the tests in parallel
                "--numprocesses={}".format(max_workers)
            )

        command.extend(
            [
                # Setup coverage options to only mutate what is tested.
                "--cov=.",
                "--cov-branch",
                "--no-cov-on-fail",
                # Override any project-level --cov-fail-under; mutation.py only
                # cares whether tests pass, not whether coverage meets a threshold.
                "--cov-fail-under=0",
                # Pass random seed
                "--randomly-seed={}".format(seed),
            ]
        )

    verbose = arguments.get("--verbose", False)

    with timeit() as alpha:
        out = run(command, verbose=verbose)

    if out == 0:
        log.info("Tests are green 💚")
        alpha = alpha() * max_workers
    else:
        msg = "Tests are not green... return code is {}..."
        log.warning(msg, out)
        log.warning("I tried the following command: `{}`", " ".join(command))

        # Same command without parallelization
        if arguments["PYTEST-COMMAND"]:
            command = list(arguments["PYTEST-COMMAND"])
        else:
            command = list(PYTEST)
            command.extend(arguments["<file-or-directory>"])

        command += [
            # Setup coverage options to only mutate what is tested.
            "--cov=.",
            "--cov-branch",
            "--no-cov-on-fail",
            "--cov-fail-under=0",
            # Pass random seed
            "--randomly-seed={}".format(seed),
        ]

        with timeit() as alpha:
            out = run(command, verbose=verbose)

        if out != 0:
            msg = "Tests are definitly red! Return code is {}!!"
            log.error(msg, out)
            log.error("I tried the following command: `{}`", " ".join(command))
            sys.exit(2)

        # Otherwise, it is possible to run the tests but without
        # parallelization via xdist. Mutations can still be tested
        # concurrently (each as an independent serial pytest run).
        msg = "Tests do not pass with xdist; each mutation will run without --numprocesses"
        log.warning(msg)
        alpha = alpha()

    msg = "Approximate time required to run the tests once: {}..."
    log.info(msg, humanize(alpha))

    return alpha, max_workers


def mutation_only_deadcode(x):
    return getattr(x, "deadcode_detection", False)


def mutation_all(x):
    return True


def mutation_without_inject_exception(x):
    return not isinstance(x, InjectException)


async def play_create_mutations(loop, root, db, max_workers, arguments):
    # Go through all files, and produce mutations, take into account
    # include pattern, and exclude patterns.  Also, exclude what has
    # no coverage.
    include = arguments.get("--include") or ["*.py"]
    include = glob2predicate(include)

    exclude = arguments.get("--exclude") or ["*test*"]
    exclude = glob2predicate(exclude)

    filepaths = root.rglob("*.py")
    filepaths = (x for x in filepaths if include(str(x)) and not exclude(str(x)))

    # setup coverage support
    coverage = coverage_read(root)
    only_dead_code = arguments["--only-deadcode-detection"]
    without_inject = arguments.get("--without-exception-injection", False)
    if only_dead_code:
        mutation_predicate = mutation_only_deadcode
    elif without_inject:
        mutation_predicate = mutation_without_inject_exception
    else:
        mutation_predicate = mutation_all

    def make_item(filepath):
        with filepath.open() as f:
            content = f.read()

        out = (
            str(filepath),
            content,
            coverage.get(str(filepath), set()),
            mutation_predicate,
        )
        return out

    items = (make_item(x) for x in filepaths if coverage.get(str(x), set()))
    # Start with biggest files first, because that is those that will
    # take most time, that way, it will make most / best use of the
    # workers.
    items = sorted(items, key=lambda x: len(x[1]), reverse=True)

    # prepare to create mutations
    total = 0

    log.info("Crafting mutations from {} files...", len(items))
    with tqdm(total=len(items), desc="Files") as progress:

        def on_mutations_created(items):
            nonlocal total

            progress.update()
            total += len(items)
            # TODO: replace ULID with a content addressable hash.
            rows = [(ULID().to_uuid().bytes, str(path), delta) for path, delta in items]
            db.store_mutations(rows)

        with timeit() as delta:
            with futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                await pool_for_each_par_map(
                    loop, pool, on_mutations_created, mutation_create, items
                )

    log.info("It took {} to compute mutations...", humanize(delta()))
    log.info("The number of mutation is {}!", total)

    return total


async def play_mutations(loop, db, seed, alpha, total, max_workers, arguments):
    # prepare to run tests against mutations
    command = list(arguments["PYTEST-COMMAND"] or PYTEST)
    command.append("--randomly-seed={}".format(seed))
    command.extend(arguments["<file-or-directory>"])

    eta = humanize(alpha * total / max_workers)
    log.info("Worst-case estimate (if every mutation takes the full test suite): {}", eta)

    timeout = alpha * 2
    rows = db.list_mutations()
    uids = ((command, uid, timeout) for (uid,) in rows)

    # sampling
    sampling = arguments["--sampling"]
    make_sample, total = sampling_setup(sampling, total)
    uids = make_sample(uids)

    log.info("Testing mutations in progress...")

    with tqdm(total=total, desc="Mutations") as progress:

        def on_progress(_):
            progress.update(1)

        with timeit() as delta:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                await pool_for_each_par_map(
                    loop, pool, on_progress, mutation_pass, uids
                )

    errors = db.count_results()

    if errors > 0:
        msg = "It took {} to compute {} mutation failures!"
        log.error(msg, humanize(delta()), errors)
    else:
        msg = "Checking that the test suite is strong against mutations took:"
        msg += " {}... And it is a success 💚"
        log.info(msg, humanize(delta()))

    return errors


async def play(loop, arguments):
    root = Path(".")

    seed = arguments["--randomly-seed"] or int(time.time())
    log.info("Using random seed: {}".format(seed))
    random.seed(seed)

    alpha, max_workers = check_tests(root, seed, arguments)

    with database_open(root, recreate=True) as db:
        # store arguments used to execute command
        if arguments["PYTEST-COMMAND"]:
            command = list(arguments["PYTEST-COMMAND"])
        else:
            command = list(PYTEST)
            command += arguments["<file-or-directory>"]
        command = dict(
            command=command,
            seed=seed,
        )
        db.set_config("command", command)

        # GC stale ignore files before generating new mutations
        mutation_ignored_gc(root)
        # let's create mutations!
        count = await play_create_mutations(loop, root, db, max_workers, arguments)
        # Let's run tests against mutations!
        await play_mutations(loop, db, seed, alpha, count, max_workers, arguments)


def mutation_diff_size(db, uid):
    _, diff = db.get_mutation(uid)
    out = len(zstd.decompress(diff))
    return out


def diff_hash(diff_text):
    return hashlib.sha256(diff_text.encode()).hexdigest()


def write_ignored_file(root, diff_text, path, reason):
    ignored_dir = Path(root) / ".mutations.ignored"
    ignored_dir.mkdir(exist_ok=True)
    header = "# Case 3: equivalent mutation"
    if reason:
        header += " — " + reason
    content = header + "\n" + diff_text
    h = diff_hash(diff_text)
    ignored_file = ignored_dir / "{}.diff".format(h)
    ignored_file.write_text(content)
    return h


def replay_mutation(db, uid, alpha, seed, max_workers, command):
    log.info("* You can use Ctrl+C to exit at anytime, your progress is saved.")

    command = list(command)
    command.append("--randomly-seed={}".format(seed))
    max_workers = 1
    if max_workers > 1:
        command.append("--numprocesses={}".format(max_workers))
    timeout = alpha * 2

    while True:
        ok = mutation_pass((command, uid, timeout))
        if ok:
            # Mutation now caught — green
            return None

        # Mutation still survives — show diff and classification menu
        mutation_show(uid.hex())
        log.info("")
        log.info("  [r] Replay        — re-run this mutation against current test suite")
        log.info("  [1] Real gap      — I will write a test (keeps in queue)")
        log.info("  [2] Fragile       — risk accepted (keeps in queue, marked)")
        log.info("  [3] Equivalent    — semantically invisible (writes to .mutations.ignored/)")
        log.info("  [4] Won't fix     — known gap, consciously accepted (never resurfaces)")
        log.info("  [5] Todo          — real gap, not fixing now (resurfaces next replay)")
        log.info("  [s] Skip          — undecided, back of queue")
        log.info("  [q] Quit")
        choice = input("> ").strip().lower()

        if choice == "r":
            continue
        elif choice == "1":
            db.set_classification(uid, CLASSIFICATION_REAL_GAP)
            return None
        elif choice == "2":
            db.set_classification(uid, CLASSIFICATION_FRAGILE)
            return None
        elif choice == "3":
            reason = input("Optional one-line reason (or Enter to skip): ").strip()
            path, diff_bytes = db.get_mutation(uid)
            diff = zstd.decompress(diff_bytes).decode("utf8")
            write_ignored_file(".", diff, path, reason)
            db.set_classification(uid, CLASSIFICATION_EQUIVALENT)
            return None
        elif choice == "4":
            db.set_classification(uid, CLASSIFICATION_WONT_FIX)
            return None
        elif choice == "5":
            db.set_classification(uid, CLASSIFICATION_TODO)
            return None
        elif choice == "s":
            return "skip"   # caller appends uid to back of queue
        elif choice == "q":
            sys.exit(0)
        # else: invalid input — loop back to menu


def replay(arguments):
    root = Path(".")

    with database_open(root) as db:
        command = db.get_config("command")

    seed = command.pop("seed")
    random.seed(seed)
    command = command.pop("command")

    alpha, max_workers = check_tests(root, seed, arguments, command)

    with database_open(root) as db:
        while True:
            uids = [uid for (uid,) in db.list_results_for_replay()]
            uids = sorted(
                uids,
                key=functools.partial(mutation_diff_size, db),
                reverse=True,
            )
            if not uids:
                log.info("No mutation failures 👍")
                sys.exit(0)
            while uids:
                uid = uids.pop(0)
                result = replay_mutation(db, uid, alpha, seed, max_workers, command)
                if result == "skip":
                    uids.append(uid)


def mutation_list():
    with database_open(".") as db:
        uids = db.list_results()
        uids = sorted(uids, key=lambda x: mutation_diff_size(db, x[0]), reverse=True)
    if not uids:
        log.info("No mutation failures 👍")
        sys.exit(0)
    for (uid, status) in uids:
        log.info("{}\t{}".format(uid.hex(), "skipped" if status == 1 else ""))


def mutation_summary():
    root = Path(".")
    with database_open(root) as db:
        total_mutations = db.count_mutations()
        total_results = db.count_results()
        counts = db.get_classification_counts()

    killed = total_mutations - total_results

    unreviewed  = counts.get(None, 0) + counts.get(1, 0)  # None + old status=1 skip
    real_gaps   = counts.get(CLASSIFICATION_REAL_GAP, 0)
    fragile     = counts.get(CLASSIFICATION_FRAGILE, 0)
    equivalent  = counts.get(CLASSIFICATION_EQUIVALENT, 0)
    wont_fix    = counts.get(CLASSIFICATION_WONT_FIX, 0)
    todo        = counts.get(CLASSIFICATION_TODO, 0)

    survived = total_results
    tested   = total_mutations

    ignored_dir = root / ".mutations.ignored"
    ignored_files = len(list(ignored_dir.glob("*.diff"))) if ignored_dir.exists() else 0

    log.info("Mutations generated:  {:>6,}", total_mutations)
    log.info("Tested:               {:>6,}", tested)
    log.info("Killed:               {:>6,}", killed)
    log.info("Survived:             {:>6,}", survived)
    log.info("  — Real gaps:        {:>6,}", real_gaps)
    log.info("  — Fragile coverage: {:>6,}", fragile)
    log.info("  — Equivalent:       {:>6,}  (in .mutations.ignored/)", equivalent)
    log.info("  — Won't fix:        {:>6,}", wont_fix)
    log.info("  — Todo:             {:>6,}", todo)
    log.info("  — Unreviewed:       {:>6,}", unreviewed)
    log.info("Ignored:              {:>6,}", ignored_files)


def mutation_ignored_gc(root):
    root = Path(root)
    ignored_dir = root / ".mutations.ignored"
    if not ignored_dir.exists():
        return
    removed = 0
    for ignore_file in ignored_dir.glob("*.diff"):
        content = ignore_file.read_text()
        # Strip header comment lines to isolate the diff
        diff_lines = [l for l in content.splitlines(keepends=True) if not l.startswith("#")]
        diff_text = "".join(diff_lines).lstrip("\n")
        # Extract target path from "--- a/path/to/file.py"
        path = None
        for line in diff_lines:
            if line.startswith("--- "):
                path = line[4:].strip().removeprefix("a/")
                break
        if path is None or not (root / path).exists():
            ignore_file.unlink()
            log.info("GC: removed stale ignore file {} (source not found)", ignore_file.name)
            removed += 1
            continue
        try:
            source = (root / path).read_text()
            normalized = ast.unparse(ast.parse(source))
            patch(diff_text, normalized)
        except Exception:
            ignore_file.unlink()
            log.info("GC: removed stale ignore file {} (diff no longer applies)", ignore_file.name)
            removed += 1
    if removed:
        log.info("Removed {} stale .mutations.ignored/ file(s).", removed)


def mutation_show(uid):
    uid = UUID(hex=uid)
    log.info("mutation show {}", uid.hex)
    log.info("")
    with database_open(".") as db:
        path, diff = db.get_mutation(uid.bytes)
    diff = zstd.decompress(diff).decode("utf8")

    terminal256 = pygments.formatters.get_formatter_by_name("terminal256")
    python = pygments.lexers.get_lexer_by_name("python")

    for line in diff.split("\n"):
        if line.startswith("+++"):
            delta = colored("+++", "green", attrs=["bold"])
            highlighted = pygments.highlight(line[3:], python, terminal256)
            log.info(delta + highlighted.rstrip())
        elif line.startswith("---"):
            delta = colored("---", "red", attrs=["bold"])
            highlighted = pygments.highlight(line[3:], python, terminal256)
            log.info(delta + highlighted.rstrip())
        elif line.startswith("+"):
            delta = colored("+", "green", attrs=["bold"])
            highlighted = pygments.highlight(line[1:], python, terminal256)
            log.info(delta + highlighted.rstrip())
        elif line.startswith("-"):
            delta = colored("-", "red", attrs=["bold"])
            highlighted = pygments.highlight(line[1:], python, terminal256)
            log.info(delta + highlighted.rstrip())
        else:
            highlighted = pygments.highlight(line, python, terminal256)
            log.info(highlighted.rstrip())


def mutation_apply(uid):
    uid = UUID(hex=uid)
    with database_open(".") as db:
        path, diff = db.get_mutation(uid.bytes)
    diff = zstd.decompress(diff).decode("utf8")
    with open(path, "r") as f:
        source = f.read()
    patched = patch(diff, ast.unparse(ast.parse(source)))
    with open(path, "w") as f:
        f.write(patched)


def main():
    arguments = docopt(__doc__, version=__version__)

    if arguments.get("--verbose", False):
        log.remove()
        log.add(
            sys.stdout,
            format="<level>{level}</level> {message}",
            level="DEBUG",
            colorize=True,
            enqueue=True,
        )

    log.debug("Mutation at {}", MUTATION)

    log.trace(arguments)

    if arguments["replay"]:
        replay(arguments)
        sys.exit(0)

    if arguments.get("list", False):
        mutation_list()
        sys.exit(0)

    if arguments.get("show", False):
        mutation_show(arguments["MUTATION"])
        sys.exit(0)

    if arguments.get("apply", False):
        mutation_apply(arguments["MUTATION"])
        sys.exit(0)

    if arguments.get("summary", False):
        mutation_summary()
        sys.exit(0)

    if arguments.get("gc", False):
        mutation_ignored_gc(".")
        sys.exit(0)

    # Otherwise run play.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(play(loop, arguments))
    loop.close()


if __name__ == "__main__":
    main()
