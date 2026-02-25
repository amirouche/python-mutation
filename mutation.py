#!/usr/bin/env python3
"""Mutation.

Usage:
  mutation play [--verbose] [--exclude=<glob>]... [--only-deadcode-detection] [--include=<glob>]... [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- PYTEST-COMMAND ...]
  mutation replay [--verbose] [--max-workers=<n>]
  mutation list
  mutation show MUTATION
  mutation apply MUTATION
  mutation (-h | --help)
  mutation --version

Options:
  --include=<glob>           Glob pattern for files to mutate, matched against relative paths.
                             Repeat the flag for multiple patterns [default: *.py]
  --exclude=<glob>           Glob pattern for files to skip. Repeat the flag for multiple
                             patterns [default: *test*]
  --sampling=<s>             Limit mutations tested: N tests the first N, N% tests a random
                             N% (e.g. "--sampling=100" or "--sampling=10%") [default: all]
  --randomly-seed=<n>        Integer seed controlling test order (pytest-randomly) and random
                             number mutations; also makes --sampling=N% reproducible
                             [default: current Unix timestamp]
  --only-deadcode-detection  Only apply dead-code detection mutations (StatementDrop,
                             DefinitionDrop).
  --max-workers=<n>          Number of parallel workers [default: cpu_count - 1]
  --verbose                  Show more information.
  -h --help                  Show this screen.
  --version                  Show version.
"""
import ast
import asyncio
import fnmatch
import functools
import itertools
import json
import os
import random
import re
import shlex
import sqlite3
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
    def store_mutation(self, uid, path, diff):
        self._conn.execute(
            "INSERT OR REPLACE INTO mutations (uid, path, diff) VALUES (?, ?, ?)",
            (uid, path, diff),
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


class ForceConditional(metaclass=Mutation):
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

    components = path[:-3].split("/")

    while components:
        for pythonpath in sys.path:
            filepath = os.path.join(pythonpath, "/".join(components))
            filepath += ".py"
            ok = os.path.exists(filepath)
            if ok:
                module_path = ".".join(components)
                break
        else:
            components.pop()
            continue
        break
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
    parser.addoption("--mutation", dest="mutation", type=str)


def for_each_par_map(loop, pool, inc, proc, items):
    out = []
    for item in items:
        item = proc(item)
        item = inc(item)
        out.append(item)
    return out


def mutation_pass(args):  # TODO: rename
    command, uid, timeout = args
    command = command + ["--mutation={}".format(uid.hex())]
    log.debug("Running command: {}", ' '.join(command))
    out = run(command, timeout=timeout, silent=True)
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


PYTEST = "pytest --exitfirst --no-header --tb=no --quiet --assert=plain"
PYTEST = shlex.split(PYTEST)


def coverage_read(root):
    coverage = Coverage(".coverage")  # use pathlib
    coverage.load()
    data = coverage.get_data()
    filepaths = data.measured_files()
    out = dict()
    root = root.resolve()
    for filepath in filepaths:
        if not filepath.startswith(str(root)):
            continue
        key = str(Path(filepath).relative_to(root))
        value = set(data.lines(filepath))
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


def run(command, timeout=None, silent=True):
    if timeout and timeout < 60:
        timeout = 60

    if timeout:
        command.insert(0, "timeout {}".format(timeout))

    command.insert(0, "PYTHONDONTWRITEBYTECODE=1")

    if silent:
        command.append("> /dev/null 2>&1")

    return os.system(" ".join(command))


def sampling_setup(sampling, total):
    if sampling is None:
        return lambda x: x, total

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
    max_workers = arguments["--max-workers"] or (os.cpu_count() - 1) or 1
    max_workers = int(max_workers)

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
                # Pass random seed
                "--randomly-seed={}".format(seed),
            ]
        )

    with timeit() as alpha:
        out = run(command)

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
            # Pass random seed
            "--randomly-seed={}".format(seed),
        ]

        with timeit() as alpha:
            out = run(command)

        if out != 0:
            msg = "Tests are definitly red! Return code is {}!!"
            log.error(msg, out)
            log.error("I tried the following command: `{}`", " ".join(command))
            sys.exit(2)

        # Otherwise, it is possible to run the tests but without
        # parallelization.
        msg = "Setting max_workers=1 because tests do not pass in parallel"
        log.warning(msg)
        max_workers = 1
        alpha = alpha()

    msg = "Approximate time required to run the tests once: {}..."
    log.info(msg, humanize(alpha))

    return alpha, max_workers


def mutation_only_deadcode(x):
    return getattr(x, "deadcode_detection", False)


def mutation_all(x):
    return True


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
    if only_dead_code:
        mutation_predicate = mutation_only_deadcode
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
            for path, delta in items:
                # TODO: replace ULID with a content addressable hash.
                uid = ULID().to_uuid().bytes
                # delta is a compressed unified diff
                db.store_mutation(uid, str(path), delta)

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
    log.info("At most, it will take {} to run the mutations", eta)

    timeout = alpha * 2
    rows = db.list_mutations()
    uids = ((command, uid, timeout) for (uid,) in rows)

    # sampling
    sampling = arguments["--sampling"]
    make_sample, total = sampling_setup(sampling, total)
    uids = make_sample(uids)

    step = 10

    gamma = time.perf_counter()

    remaining = total

    log.info("Testing mutations in progress...")

    with tqdm(total=100) as progress:

        def on_progress(_):
            nonlocal remaining
            nonlocal step
            nonlocal gamma

            remaining -= 1

            if (remaining % step) == 0:

                percent = 100 - ((remaining / total) * 100)
                now = time.perf_counter()
                delta = now - gamma
                eta = (delta / step) * remaining

                progress.update(int(percent))
                progress.set_description("ETA {}".format(humanize(eta)))

                msg = "Mutation tests {:.2f}% done..."
                log.debug(msg, percent)
                log.debug("ETA {}...", humanize(eta))

                for speed in [10_000, 1_000, 100, 10, 1]:
                    if total // speed == 0:
                        continue
                    step = speed
                    break

                gamma = time.perf_counter()

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

        # let's create mutations!
        count = await play_create_mutations(loop, root, db, max_workers, arguments)
        # Let's run tests against mutations!
        await play_mutations(loop, db, seed, alpha, count, max_workers, arguments)


def mutation_diff_size(db, uid):
    _, diff = db.get_mutation(uid)
    out = len(zstd.decompress(diff))
    return out


def replay_mutation(db, uid, alpha, seed, max_workers, command):
    log.info("* You can use Ctrl+C to exit at anytime, you progress is saved.")

    command = list(command)
    command.append("--randomly-seed={}".format(seed))

    max_workers = 1
    if max_workers > 1:
        command.append("--numprocesses={}".format(max_workers))
    timeout = alpha * 2

    while True:
        ok = mutation_pass((command, uid, timeout))
        if not ok:
            mutation_show(uid.hex())
            msg = "* Type 'skip' to go to next mutation or enter to retry."
            log.info(msg)
            skip = input().startswith("s")
            if skip:
                db.set_result(uid, 1)
                return
            # Otherwise loop to re-test...
        else:
            db.del_result(uid)
            return


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
            uids = [uid for (uid, _) in db.list_results(status=0)]
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
                replay_mutation(db, uid, alpha, seed, max_workers, command)


def mutation_list():
    with database_open(".") as db:
        uids = db.list_results()
        uids = sorted(uids, key=lambda x: mutation_diff_size(db, x[0]), reverse=True)
    if not uids:
        log.info("No mutation failures 👍")
        sys.exit(0)
    for (uid, status) in uids:
        log.info("{}\t{}".format(uid.hex(), "skipped" if status == 1 else ""))


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

    # Otherwise run play.
    loop = asyncio.get_event_loop()
    loop.run_until_complete(play(loop, arguments))
    loop.close()


if __name__ == "__main__":
    main()
