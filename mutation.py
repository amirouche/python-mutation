"""Mutation.

Usage:
  mutation play [--verbose] [--exclude=<globs>] [--only-deadcode-detection] [--include=<globs>] [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- TEST-COMMAND ...]
  mutation replay
  mutation show failed
  mutation show MUTATION
  mutation (-h | --help)
  mutation --version

Options:
  --verbose     Show more information.
  -h --help     Show this screen.
  --version     Show version.
"""
from tqdm import tqdm
import functools
import asyncio
import fnmatch
import itertools
import os
import random
import re
import shlex
import subprocess
import sys
import time
from ast import Constant
from concurrent import futures
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
from astunparse import unparse
from coverage import Coverage
from docopt import docopt
from humanize import precisedelta
from loguru import logger as log
from lsm import LSM
from ulid import ULID


__version__ = (0, 3, 0)


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
        if not getattr(child, "children", False):
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
    NEWLINE = "a = 42\n"

    def predicate(self, node):
        return "stmt" in node.type and node.type != "expr_stmt"

    def mutate(self, node, index):
        root, new = node_copy_tree(node, index)
        index = new.parent.children.index(new)
        passi = parso.parse("pass").children[0]
        passi.prefix = new.get_first_leaf().prefix
        new.parent.children[index] = passi
        newline = parso.parse(type(self).NEWLINE).children[0].children[1]
        new.parent.children.insert(index + 1, newline)
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
    """Yield successive n-sized chunks from iterable."""
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
            value = b"coffeebad" + value
        else:
            value = "mutated string " + value
        value = Constant(value=value, kind="")
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
        targets = (
            type(self).KEYWORDS
            if value in type(self).KEYWORDS
            else type(self).SINGLETON
        )
        for target in targets:
            if target == value:
                continue
            root, new = node_copy_tree(node, index)
            new.value = target
            yield root, new


class Comparison(metaclass=Mutation):
    def predicate(self, node):
        return node == "comparison"

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


def diff(source, target, filename=""):
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


def interesting(new_node, coverage):
    if getattr(new_node, "line", False):
        return new_node.line in coverage
    return new_node.get_first_leaf().line in coverage


def deltas_compute(source, path, coverage, mutations):
    ast = parso.parse(source)
    ignored = 0
    for (index, node) in zip(itertools.count(0), node_iter(ast)):
        for root, new_node in mutate(node, index, mutations):
            if not interesting(new_node, coverage):
                ignored += 1
                continue
            target = root.get_code()
            delta = diff(source, target, path)
            yield delta
    if ignored > 1:
        msg = "Ignored {} mutations from file at {} because there is no associated coverage."
        log.trace(msg, ignored, path)


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
        log.trace("Ignoring file {} because there is no associated coverage.", path)
        return []

    log.trace("Mutating file: {}...", path)
    mutations = [m for m in Mutation.ALL if mutation_predicate(m)]
    deltas = deltas_compute(source, path, coverage, mutations)
    # return the compressed deltas to save some time in the
    # mainthread.
    out = [(path, zstd.compress(x.encode("utf8"))) for x in deltas]
    log.trace("There is {} mutations for the file `{}`", len(out), path)
    return out


def install_module_loader(uid):
    db = LSM(".mutation.okvslite")

    mutation_show(uid.hex)

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


def mutation_pass(args):  # TODO: rename
    command, uid, timeout = args
    command = command + ["--mutation={}".format(uid.hex)]
    out = run(command, timeout=timeout)

    if out == 0:
        msg = "no error with mutation: {}"
        log.error(msg, " ".join(command))
        with database_open(".") as db:
            db[lexode.pack([2, uid])] = b"\x42"
        mutation_show(uid.hex)
        return False
    else:
        # TODO: pass root path...
        with database_open(".") as db:
            del db[lexode.pack([2, uid])]
        return True


PYTEST = "pytest --exitfirst --no-header --tb=no --quiet"
PYTEST = shlex.split(PYTEST)


def coverage_read(root):
    coverage = Coverage(".coverage")  # use pathlib
    coverage.load()
    data = coverage.get_data()
    files = data.measured_files()
    out = {str(Path(path).relative_to(root)): set(data.lines(path)) for path in files}
    return out


def database_open(root, recreate=False):
    root = root if isinstance(root, Path) else Path(root)
    db = root / ".mutation.okvslite"
    if recreate and db.exists():
        log.trace("Deleting existing database...")
        for file in root.glob(".mutation.okvslite*"):
            file.unlink()

    if not recreate and not db.exists():
        log.error("No database, can not proceed!")
        sys.exit(1)

    db = LSM(str(db))

    return db


def git_open(root):
    try:
        repository = git.Repo(str(root))
    except git.exc.InvalidGitRepositoryError:
        log.error("There is no git repository at {}", root)
        sys.exit(2)
    else:
        return repository


def run(command, timeout=None):
    try:
        out = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=None
        )
    except Exception:
        return -1
    else:
        return out.returncode


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


def play_test_tests(root, seed, repository, arguments, command=None):
    max_workers = arguments["--max-workers"] or (os.cpu_count() - 1) or 1
    max_workers = int(max_workers)

    log.info("Let's check that the tests are green...")
    #
    # TODO: use the coverage program instead with something along the
    # lines of:
    #
    #   coverage run --omit=tests.py --include=hoply*.py -m pytest tests.py
    #
    # To be able to pass --omit and --include.
    #
    if arguments["<file-or-directory>"] and arguments["TEST-COMMAND"]:
        log.error("<file-or-directory> and TEST-COMMAND are exclusive!")
        sys.exit(1)

    if command is not None:
        if max_workers > 1:
            command.extend([
                # Use pytest-xdist to make sure it is possible to run the
                # tests in parallel
                "--numprocesses={}".format(max_workers),
            ])
    else:
        command = list(arguments["TEST-COMMAND"] or PYTEST)
        if max_workers > 1:
            command.append(
                # Use pytest-xdist to make sure it is possible to run
                # the tests in parallel
                "--numprocesses={}".format(max_workers)
            )
        command.extend([
            # Setup coverage options to only mutate what is tested.
            "--cov=.",
            "--cov-branch",
            "--no-cov-on-fail",
            # Pass random seed
            "--randomly-seed={}".format(seed),
        ])
        command.extend(arguments["<file-or-directory>"])

    with timeit() as alpha:
        out = run(command)

    if out == 0:
        log.info("Tests are green 💚")
        alpha = alpha() * max_workers
    else:
        msg = "Tests are not green or something... return code is {}..."
        log.warning(msg, out)
        log.warning("I tried the following command: `{}`", " ".join(command))

        command = list(arguments["TEST-COMMAND"] or PYTEST)
        command = command + [
            # Setup coverage options to only mutate what is tested.
            "--cov=.",
            "--cov-branch",
            "--no-cov-on-fail",
            # Pass random seed
            "--randomly-seed={}".format(seed),
        ]
        command += arguments["<file-or-directory>"]

        with timeit() as alpha:
            out = run(command)

        if out != 0:
            msg = "Tests are definitly red! Return code is {}!!"
            log.error(msg, out)
            log.error("I tried the following command: `{}`", " ".join(command))
            sys.exit(2)

        # Otherwise, it is possible to run the tests but without
        # parallelization.
        log.info("Overriding max_workers=1 because tests do not pass in parallel")
        max_workers = 1
        alpha = alpha()

    msg = "Time required to run the tests once: {}..."
    log.info(msg, humanize(alpha))

    return alpha, max_workers


def mutation_only_deadcode(x):
    return getattr(x, "deadcode_detection", False)


def mutation_all(x):
    return True


async def play_create_mutations(loop, root, db, repository, max_workers, arguments):
    # Go through all blobs in head, and produce mutations, take into
    # account include pattern, and exclude patterns.  Also, exclude
    # what has no coverage.
    include = arguments.get("--include") or "*.py"
    include = include.split(",")
    include = glob2predicate(include)

    exclude = arguments.get("--exclude") or "*test*"
    exclude = exclude.split(",")
    exclude = glob2predicate(exclude)

    blobs = repository_iter_latest_files(repository)
    blobs = (x for x in blobs if include(x.path) and not exclude(x.path))

    # setup coverage support
    coverage = coverage_read(root)
    only_dead_code = arguments["--only-deadcode-detection"]
    if only_dead_code:
        mutation_predicate = mutation_only_deadcode
    else:
        mutation_predicate = mutation_all

    def make_item(blob):
        out = (
            blob.path,
            blob.data_stream.read().decode("utf8"),
            coverage.get(blob.path, set()),
            mutation_predicate,
        )
        return out

    items = (make_item(blob) for blob in blobs if coverage.get(blob.path, set()))
    items = sorted(items, key=lambda x: len(x[1]), reverse=True)

    # prepare to create mutations
    total = 0

    log.info("Creating mutations from {} files...", len(items))
    with tqdm(total=len(items), desc="Files") as progress:

        def on_mutations_created(items):
            nonlocal total

            progress.update()
            total += len(items)
            for path, delta in items:
                # TODO: replace ULID with a content addressable hash.
                uid = ULID().to_uuid()
                # delta is a compressed unified diff
                db[lexode.pack([1, uid])] = lexode.pack([path, delta])

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
    command = list(arguments["TEST-COMMAND"] or PYTEST)
    command.append("--randomly-seed={}".format(seed))

    timeout = alpha * 2
    uids = db[lexode.pack([1]):lexode.pack([2])]
    uids = ((command, lexode.unpack(key)[1], timeout) for (key, _) in uids)

    # sampling
    sampling = arguments["--sampling"]
    sampler, total = sampling_setup(sampling, total)
    uids = sampler(uids)

    for speed in [10_000, 1_000, 100, 10, 1]:
        if total // speed == 0:
            continue
        step = speed
        break

    gamma = time.perf_counter()

    remaining = total

    log.info("Testing mutations in progress...")

    with tqdm(total=100) as progress:

        def on_progress(_):
            nonlocal remaining

            remaining -= 1

            if (remaining % step) == 0 or (total - remaining == 10):
                percent = 100 - ((remaining / total) * 100)
                now = time.perf_counter()
                delta = now - gamma
                eta = (delta / (total - remaining)) * remaining

                progress.update(int(percent))
                progress.set_description("ETA {}".format(humanize(eta)))

                msg = "Mutation tests {:.2f}% done..."
                log.debug(msg, percent)
                log.debug("ETA {}...", humanize(eta))

        with timeit() as delta:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                await pool_for_each_par_map(loop, pool, on_progress, mutation_pass, uids)

        errors = len(list(db[lexode.pack([2]):lexode.pack([3])]))

    if errors > 0:
        msg = "It took {} to compute {} mutation failures!"
        log.error(msg, humanize(delta()), errors)
    else:
        msg = "Checking that the test suite is strong against mutations took:"
        msg += " {}... And it is a success 💚"
        log.info(msg, humanize(delta()))

    return errors


async def play(loop, arguments):
    # TODO: Always use git HEAD, and display a message as critical
    #       explaining what is happenning... Not sure about that.

    root = Path(".").resolve()
    repository = git_open(root)

    seed = arguments["--randomly-seed"] or int(time.time())
    log.info("Using random seed: {}".format(seed))
    random.seed(seed)

    alpha, max_workers = play_test_tests(root, seed, repository, arguments)

    with database_open(root, recreate=True) as db:
        # store arguments used to execute command
        command = list(arguments["TEST-COMMAND"] or PYTEST)
        command += arguments["<file-or-directory>"]
        command = dict(
            command=command,
            seed=seed,
        )
        value = list(command.items())
        db[lexode.pack((0, 'command'))] = lexode.pack(value)

        # let's play!
        count = await play_create_mutations(loop, root, db, repository, max_workers, arguments)
        await play_mutations(
            loop, db, seed, alpha, count, max_workers, arguments
        )


def mutation_diff_size(db, uid):
    _, diff = lexode.unpack(db[lexode.pack([1, uid])])
    out = len(zstd.decompress(diff))
    return out


def replay_mutation(db, uid, alpha, seed, max_workers, arguments):
    print("* Use Ctrl+C to exit.")

    repository = git_open(".")

    command = list(arguments["TEST-COMMAND"] or PYTEST)
    command.append("--randomly-seed={}".format(seed))
    max_workers = 1
    if max_workers > 1:
        command.append("--numprocesses={}".format(max_workers))
    timeout = alpha * 2

    while True:
        ok = mutation_pass((command, uid, timeout))
        if not ok:
            msg = "* Type 'skip' to go to next mutation or just enter to retry."
            print(msg)
            retry = input("> ") == 'retry'
            if not retry:
                return
            # Otherwise loop to re-test...
        else:
            non_indexed = repository.index.diff(None)
            indexed = repository.index.diff("HEAD")
            if indexed or non_indexed:
                print("* They are uncommited changes, do you want to commit?")
                yes = input("> ").startswith("y")
                if not yes:
                    return

                for file in non_indexed:
                    repository.add(file)
                repository.index.commit("fixed mutation bug uid={}".format(uid.hex))
            return


def replay(arguments):
    root = Path(".").resolve()
    repository = git_open(root)

    with database_open(root) as db:
        command = db[lexode.pack((0, 'command'))]

    command = lexode.unpack(command)
    command = dict(command)
    seed = command.pop("seed")
    random.seed(seed)
    command = command.pop("command")

    alpha, max_workers = play_test_tests(root, seed, repository, arguments, command)

    with database_open(root) as db:
        while True:
            uids = (lexode.unpack(key)[1] for key, _ in db[lexode.pack([2]):])
            uids = sorted(
                uids,
                key=functools.partial(mutation_diff_size, db),
                reverse=True
            )
            if not uids:
                log.info("No mutation failures 👍")
                sys.exit(0)
            while uids:
                uid = uids.pop(0)
                replay_mutation(db, uid, alpha, max_workers)


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
    uid = UUID(hex=uid)
    with database_open(".") as db:
        path, diff = lexode.unpack(db[lexode.pack([1, uid])])
    diff = zstd.decompress(diff).decode("utf8")
    diff_highlight(diff)


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

    log.debug("Mutation at {}", PRONOTION)

    log.trace(arguments)

    if arguments["replay"]:
        replay(arguments)
        sys.exit(0)

    if arguments.get("show", False):
        mutation_show(arguments["MUTATION"])
        sys.exit(0)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(play(loop, arguments))
    loop.close()


if __name__ == "__main__":
    main()
