"""Mutation.

Usage:
  mutation play [--verbose] [--exclude=<globs>] [--only-deadcode-detection] [--include=<globs>] [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [-- TEST-COMMAND ...]
  mutation show failed
  mutation show MUTATION
  mutation (-h | --help)
  mutation --version

Options:
  --verbose     Show more information.
  -h --help     Show this screen.
  --version     Show version.
"""
from ast import Constant
import operator
import asyncio
import concurrent.futures
import fnmatch
import itertools
import os
import random
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from difflib import unified_diff
from importlib.abc import SourceLoader
from importlib.machinery import FileFinder
from pathlib import Path
from uuid import UUID

import git
import lexode
import parso
import zstandard as zstd
from aiostream import pipe, stream
from coverage import Coverage
from docopt import docopt
from humanize import precisedelta
from loguru import logger as log
from lsm import LSM
from ulid import ULID
from astunparse import unparse


__version__ = (0, 1, 0)


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


PRONOTION = "https://youtu.be/ihZEaj9ml4w?list=PLOSNaPJYYhrtliZqyEWDWL0oqeH0hOHnj"


log.remove()
if os.environ.get("DEBUG", False):
    log.add(
        sys.stderr,
        format="<level>{level}</level> {message}",
        level="TRACE",
        colorize=True,
        enqueue=True,
    )
else:
    log.add(
        sys.stderr,
        format="<level>{level}</level> {message}",
        level="INFO",
        colorize=True,
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


def tree_iter(repository, tree):
    for blob in tree.blobs:
        yield blob
    for subtree in tree.trees:
        yield from tree_iter(repository, subtree)


def repository_iter_latest_files(repository):
    repository = (
        repository if isinstance(repository, git.Repo) else git.Repo(str(repository))
    )
    commit = repository.head.object
    yield from tree_iter(repository, commit.tree)


def glob2predicate(patterns):
    def regex_join(regexes):
        """Combine a list of regexes into one that matches any of them."""
        return "|".join("(?:%s)" % r for r in regexes)

    regexes = (fnmatch.translate(pattern) for pattern in patterns)
    regex = re.compile(regex_join(regexes))

    def predicate(path):
        return regex.match(path) is not None

    return predicate


def node_iter(node, level=1):
    yield node
    for child in node.children:
        level += 1
        if child.type == "endmarker":
            yield child
            continue
        if child.type == "number":
            yield child
            continue
        if child.type == "string":
            yield child
            continue
        if child.type == "newline":
            continue
        if child.type == "operator":
            yield child
            continue
        if child.type == "keyword":
            yield child
            continue
        if child.type == "name":
            yield child
            continue

        yield from node_iter(child, level)


def node_copy_tree(node, index):
    root = node.get_root_node()
    root = deepcopy(root)
    iterator = itertools.dropwhile(
        lambda x: x[0] != index, zip(itertools.count(0), node_iter(root))
    )
    index, node = next(iterator)
    return root, node


@contextmanager
def timeit():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


class Mutation(type):
    # TODO: understand the difference between class decorator and
    #       metaclass.
    #
    # TODO: document why a metaclass, and why not a class decorator.
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
        return "stmt" in node.type and node.type != "expr_stmt"

    def mutate(self, node, index):
        log.warning(node.type)
        root, new = node_copy_tree(node, index)
        index = new.parent.children.index(new)
        passi = parso.parse("pass").children[0]
        passi.prefix = new.get_first_leaf().prefix
        new.parent.children[index] = passi
        yield root, new


class DefinitionDrop(metaclass=Mutation):

    deadcode_detection = True

    def predicate(self, node):
        # There is also node.type = 'lambdadef' but lambadef are
        # always part of a assignation statement. So, that case is
        # handled in StatementDrop.
        return node.type in ("classdef", "funcdef")

    def mutate(self, node, index):
        root, new = node_copy_tree(node, index)
        new.parent.children.remove(new)
        yield root, new


def chunks(iterable, n):
    """Yield successive n-sized chunks from iterable.

    """
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


class MutateNumber(metaclass=Mutation):

    COUNT = 5

    def predicate(self, node):
        return node.type == "number"

    def mutate(self, node, index):
        value = eval(node.value)

        if isinstance(value, int):
            def randomize(x):
                return random.randint(0, x)
        else:
            def randomize(x):
                return random.random() * x

        for size in range(6):
            if value < 2 ** size:
                break

        for _ in range(type(self).COUNT):
            root, new = node_copy_tree(node, index)
            new.value = str(randomize(2 ** size))
            yield root, new


class MutateString(metaclass=Mutation):

    def predicate(self, node):
        # str or bytes.
        return node.type == "string"

    def mutate(self, node, index):
        root, new = node_copy_tree(node, index)
        value = eval(new.value)
        if isinstance(value, bytes):
            value = b'coffeebad' + value
        else:
            value = "mutated string " + value
        value = Constant(value=value, kind='')
        value = unparse(value).strip()
        new.value = value
        yield root, new


class MutateKeyword(metaclass=Mutation):

    KEYWORDS = set(["continue", "break", "pass"])
    SINGLETON = set(["True", "False", "None"])
    # Support xor operator ^
    BOOLEAN = set(["and", "or"])

    TARGETS = KEYWORDS | SINGLETON | BOOLEAN

    def predicate(self, node):
        return node.type == "keyword" and node.value in type(self).TARGETS

    def mutate(self, node, index):
        value = node.value
        targets = type(self).KEYWORDS if value in type(self).KEYWORDS else type(self).SINGLETON
        for target in targets:
            if target == value:
                continue
            root, new = node_copy_tree(node, index)
            new.value = target
            yield root, new


class Comparison(metaclass=Mutation):

    def predicate(self, node):
        return node == 'comparison'

    def mutate(self, node, index):
        root, new = node_copy_tree(node, index)
        not_test = parso.parse("not ({})".format(new.get_code()))
        index = new.parent.children.index(new)
        new.parent.children[index] = not_test
        return root, new


class MutateOperator(metaclass=Mutation):

    BINARY = ["+", "-", "%", "|", "&", "//", "/", "*", "^", "**", "@"]
    BITWISE = ["<<", ">>"]
    COMPARISON = ["<", "<=", "==", "!=", ">=", ">"]
    ASSIGNEMENT = ["="] + [x + "=" for x in BINARY + BITWISE]

    # TODO support OPERATORS_CONTAINS = ["in", "not in"]

    OPERATORS = [
        BINARY,
        BITWISE,
        BITWISE,
        COMPARISON,
        ASSIGNEMENT,
    ]

    def predicate(self, node):
        return node.type == "operator"

    def mutate(self, node, index):
        for operators in type(self).OPERATORS:
            if node.value not in operators:
                continue
            for new_operator in operators:
                if node.value == new_operator:
                    continue
                root, new = node_copy_tree(node, index)
                new.value = new_operator
                yield root, new


def diff(source, target, filename=None):
    lines = unified_diff(
        source.split("\n"), target.split("\n"), filename, filename, lineterm=""
    )
    out = "\n".join(lines)
    return out


def mutate(node, index, mutations):
    for mutation in mutations:
        if not mutation.predicate(node):
            continue
        yield from mutation.mutate(node, index)


def deltas_compute(source, path, coverage, predicate):
    ast = parso.parse(source)

    mutations = [m for m in Mutation.ALL if predicate(m)]

    for (index, node) in zip(itertools.count(0), node_iter(ast)):
        for root, new_node in mutate(node, index, mutations):
            if getattr(new_node, "line", False) and new_node.line not in coverage:
                msg = "Ignoring mutation because there is no coverage:"
                msg += " path={}, line={}"
                log.trace(msg, path, new_node.line)
                return
                continue
            target = root.get_code()
            delta = diff(source, target, path)
            if delta.isspace():
                log.warning('diff is empty!')
            else:
                yield delta


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


def proc(item):  # TODO: rename
    path, source, coverage, node_predicate = item
    log.trace("Mutating file: {}...", path)
    # return the compressed deltas to save some time in the
    # mainthread.
    deltas = deltas_compute(source, path, coverage, node_predicate)
    out = [(path, zstd.compress(x.encode("utf8"))) for x in deltas]
    log.trace("There is {} mutations for the file `{}`", len(out), path)
    return out


def install_module_loader(uid):
    db = LSM(".mutation.okvslite")

    path, diff = lexode.unpack(db[lexode.pack([1, uid])])
    diff = zstd.decompress(diff).decode("utf8")

    # TODO: replace with file from git
    with open(path) as f:
        source = f.read()

    patched = patch(diff, source)

    # TODO: replace with a type(...) call with toplevel functions.
    class MutationLoader(SourceLoader):

        __slots__ = ("fullname", "path")

        def __init__(self, fullname, path):
            self.fullname = fullname
            self.path = path

        def get_filename(self, fullname):
            return self.path

        def get_data(self, filepath):
            """exec_module is already defined for us, we just have to provide a way
            of getting the source code of the module"""
            if filepath.endswith(path):
                return patched
            # TODO: fetch files from git...
            with open(filepath) as fp:
                return fp.read()

    # insert the path hook before of other path hooks
    sys.path_hooks.insert(0, FileFinder.path_hook((MutationLoader, [".py"])))


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


def run(args):  # TODO: rename
    command, uid, timeout = args
    command = command + ["--mutation={}".format(uid.hex)]
    try:
        out = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            timeout=timeout,
        )
    except Exception as exc:
        log.error("Exception with `{}`, exception=`{}`", uid.hex, exc)
    else:
        if out.returncode == 0:
            msg = "no error with mutation: {}"
            log.error(msg, " ".join(command))
            mutation_show(uid.hex)


PYTEST = "pytest --exitfirst --no-header --tb=long --showlocals --quiet"
PYTEST = shlex.split(PYTEST)


def coverage_read(root):
    coverage = Coverage(".coverage")  # use pathlib
    coverage.load()
    data = coverage.get_data()
    files = data.measured_files()
    out = {str(Path(path).relative_to(root)): set(data.lines(path)) for path in files}
    return out


async def _main(loop, arguments):
    # TODO: Replay failed tests and remove them from failed if it is
    #       now ok...
    #
    # TODO: Always use git HEAD, and display a message as critical
    #       explaining what is happenning...
    #
    # TODO: mutation run -n ie. dryrun: display files taken into consideration
    #
    # TODO: use transactions to make tests as failed...
    #
    # TODO: mutation show all to display all failed tests with diff.
    #
    # TODO: pass plain foobar to pytest and capture output and store
    #       it when test is failed.

    max_workers = arguments["--max-workers"] or (os.cpu_count() - 1) or 1
    max_workers = int(max_workers)

    include = arguments.get("--include") or "*.py"
    include = include.split(",")
    include = glob2predicate(include)

    exclude = arguments.get("--exclude") or "*test*"
    exclude = exclude.split(",")
    exclude = glob2predicate(exclude)

    root = Path(".").resolve()

    try:
        git.Repo(str(root))
    except git.exc.InvalidGitRepositoryError:
        log.error("There is no git repository at {}", root)
        sys.exit(1)

    seed = arguments["--randomly-seed"] or int(time.time())
    log.info("Using random seed: {}".format(seed))
    random.seed(seed)

    log.info("Checking the tests are green...")
    #
    # TODO: use the coverage program instead with something along the
    # lines of:
    #
    #   coverage run --omit=tests.py --include=hoply*.py -m pytest tests.py
    #
    # To be able to pass --omit and --include.
    #
    command = arguments["TEST-COMMAND"] or PYTEST
    command = command + [
        # Use pytest-xdist to make sure it is possible to run the
        # tests in parallel
        "--numprocesses=2",
        # Petup coverage options to only mutate what is tested.
        "--cov=.",
        "--cov-branch",
        "--no-cov-on-fail",
        # Pass random seed
        "--randomly-seed={}".format(seed),
    ]

    with timeit() as alpha:
        out = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if out.returncode != 0:
        msg = "Tests are not green or something... return code is {}"
        log.warning(msg, out.returncode)
        log.warning("I tried the following command: `{}`", " ".join(command))

        command = arguments["TEST-COMMAND"] or PYTEST
        command = command + [
            # Force numprocesses to one
            "--numprocesses=1",
            # Setup coverage options to only mutate what is tested.
            "--cov=.",
            "--cov-branch",
            "--no-cov-on-fail",
            # Pass random seed
            "--randomly-seed={}".format(seed),
        ]

        # alpha is the approximate time in seconds to execute the test
        # suite once.  Multiply the difference with two because there
        # is two processus running the tests.
        with timeit() as alpha:
            out = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if out.returncode != 0:
            msg = "Tests are definitly red! Return code is {}"
            log.error(msg, out.returncode)
            log.error("I tried the following command: `{}`", " ".join(command))
            sys.exit(1)

        log.info("Overriding max_workers=1 because tests do not pass in parallel")
        max_workers = 1
        alpha = alpha()
    else:
        # TODO: do that in the first branch
        log.info("Tests are green!")
        alpha = alpha() * 2

    msg = "Time required to run the full test suite once: {}..."
    log.info(msg, humanize(alpha))

    coverage = coverage_read(root)

    only_dead_code = arguments["--only-deadcode-detection"]
    if only_dead_code:
        node_predicate = operator.attrgetter('deadcode_detection')
    else:
        node_predicate = operator.attrgetter('predicate')

    blobs = repository_iter_latest_files(str(root))
    blobs = (x for x in blobs if include(x.path) and not exclude(x.path))

    def make_item(blob):
        out = (
            blob.path,
            blob.data_stream.read().decode("utf8"),
            coverage.get(blob.path, set()),
            node_predicate,
        )
        return out

    items = (make_item(blob) for blob in blobs)

    db = root / ".mutation.okvslite"
    if db.exists():
        log.trace("Deleting existing database...")
        for file in root.glob(".mutation.okvslite*"):
            file.unlink()

    db = LSM(str(db))

    total = 0

    def increment(items):
        nonlocal total
        total += len(items)
        for path, delta in items:
            # TODO: replace ULID with a content addressable hash.
            uid = ULID().to_uuid()
            # delta is a compressed unified diff
            db[lexode.pack([1, uid])] = lexode.pack([path, delta])

    log.info("Mutation in progress...")
    with timeit() as delta:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            await pool_for_each_par_map(loop, pool, increment, proc, items)

    log.info("It took {} to compute mutations...", humanize(delta()))
    log.info("The number of mutation is {}!", total)

    log.info("Testing in progress...")
    command = arguments["TEST-COMMAND"] or PYTEST
    # fix random seed...
    command.append("--randomly-seed={}".format(seed))

    timeout = alpha * 2
    uids = ((command, lexode.unpack(key)[1], timeout) for (key, _) in db)

    # sampling
    sampling = arguments["--sampling"]

    if sampling and sampling.endswith("%"):
        # randomly choose percent mutations
        cutoff = int(sampling[:-1]) / 100

        def sampler(iterable):
            for item in iterable:
                value = random.random()
                if value < cutoff:
                    yield item

        total = int(total * cutoff)
        uids = sampler(uids)
    elif sampling and sampling.isdigit():
        # otherwise, it is the first COUNT mutations that are used.
        total = int(sampling)

        def sampler(iterable):
            remaining = total
            for item in iterable:
                yield item
                remaining -= 1
                if remaining == 0:
                    return

        uids = sampler(uids)
    elif sampling is not None:
        msg = "Sampling passed via --sampling option must be a positive"
        msg += " integer or a percentage!"
        log.error(msg)
        sys.exit(1)

    if sampling:
        log.info("Taking into account sampling there is {} mutations.", total)

    for speed in [10_000, 1_000, 100, 10, 1]:
        if total // speed == 0:
            continue
        step = speed
        break

    gamma = time.perf_counter()

    remaining = total

    def progress(_):
        nonlocal remaining
        remaining -= 1
        if (remaining % step) == 0 or (total - remaining == 10):
            percent = 100 - ((remaining / total) * 100)
            now = time.perf_counter()
            delta = now - gamma
            eta = (delta / (total - remaining)) * remaining
            msg = "Mutation tests {:.2f}% done..."
            log.debug(msg, percent)
            log.info("ETA {}...", humanize(eta))

    with timeit() as delta:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            await pool_for_each_par_map(loop, pool, progress, run, uids)

    msg = "Checking that the test suite is strong against mutations took:"
    msg += " {}..."
    log.info(msg, humanize(delta()))

    db.close()

    return None


def diff_highlight(diff):
    # adapted from diff-highlight
    import re

    from highlights.command import show_hunk, write

    new, old = [], []
    in_header = True

    for line in diff.split("\n"):
        if in_header:
            if re.match("^(@|commit \w+$)", line):
                in_header = False
        else:
            if not re.match("^(?:[ +\-@\\\\]|diff)", line):
                in_header = True

        if not in_header and line.startswith("+"):
            new.append(line)
        elif not in_header and line.startswith("-"):
            old.append(line)
        else:
            show_hunk(new, old)
            new, old = [], []
            # XXX: add a new line here
            write(line + "\n")

    show_hunk(new, old)  # flush last hunk


def mutation_show(uid):
    # TODO: pass the printer as an argument or at least make it
    # possible to pass sys.stderr.
    uid = UUID(hex=uid)
    with LSM(".mutation.okvslite") as db:
        path, diff = lexode.unpack(db[lexode.pack([1, uid])])
    diff = zstd.decompress(diff).decode("utf8")
    diff_highlight(diff)


def main():
    arguments = docopt(__doc__, version=__version__)

    if arguments.get("--verbose", False):
        log.remove()
        log.add(
            sys.stderr,
            format="<level>{level}</level> {message}",
            level="DEBUG",
            colorize=True,
            enqueue=True,
        )

    log.debug("Mutation at {}", PRONOTION)

    log.trace(arguments)

    if arguments.get("show", False):
        mutation_show(arguments["MUTATION"])
        sys.exit(0)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_main(loop, arguments))
    loop.close()


if __name__ == "__main__":
    main()
