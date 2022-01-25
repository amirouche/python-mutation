"""Mutation.

Usage:
  mutation play [--verbose] [--exclude=<globs>] [--only-deadcode-detection] [--include=<globs>] [--sampling=<s>] [--randomly-seed=<n>] [--max-workers=<n>] [<file-or-directory> ...] [-- TEST-COMMAND ...]
  mutation replay [--verbose] [--max-workers=<n>]
  mutation list
  mutation show MUTATION
  mutation apply MUTATION
  mutation (-h | --help)
  mutation --version

Options:
  --verbose     Show more information.
  -h --help     Show this screen.
  --version     Show version.
"""
import asyncio
import fnmatch
import functools
import itertools
import os
import random
import re
import shlex
import sys
import time
from ast import Constant
from concurrent import futures
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
from difflib import unified_diff
from uuid import UUID

import lexode
import parso
import pygments
import pygments.formatters
import pygments.lexers
import zstandard as zstd
from aiostream import pipe, stream
from astunparse import unparse
from coverage import Coverage
from docopt import docopt
from humanize import precisedelta
from loguru import logger as log
from lsm import LSM
from pathlib3x import Path
from termcolor import colored
from tqdm import tqdm
from ulid import ULID

__version__ = (0, 4, 6)


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


def node_iter(node, level=1):
    yield node
    for child in node.children:
        if not getattr(child, "children", False):
            yield child
            continue

        yield from node_iter(child, level + 1)


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
    iterable = iter(iterable)
    for chunk in tuple(itertools.islice(iterable, n)):
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

        for size in range(8, 32):
            if value < 2 ** size:
                break

        count = 0
        while count != self.COUNT:
            count += 1
            root, new = node_copy_tree(node, index)
            new.value = str(randomize(2 ** size))
            if new.value == node.value:
                continue
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
        for targets in [self.KEYWORDS, self.SINGLETON, self.BOOLEAN]:
            if value in targets:
                break
        else:
            raise NotImplementedError

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


def iter_deltas(source, path, coverage, mutations):
    ast = parso.parse(source)
    ignored = 0
    for (index, (index, node)) in enumerate(zip(itertools.count(0), node_iter(ast))):
        for root, new_node in mutate(node, index, mutations):
            if not interesting(new_node, coverage):
                ignored += 1
                continue
            target = root.get_code()
            delta = diff(source, target, path)
            yield delta
    if ignored > 1:
        msg = "Ignored {} mutations from file at {}"
        msg += " because there is no associated coverage."
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
    db = LSM(".mutation.okvslite")

    mutation_show(uid.hex)

    path, diff = lexode.unpack(db[lexode.pack([1, uid.bytes])])
    diff = zstd.decompress(diff).decode("utf8")

    with open(path) as f:
        source = f.read()

    patched = patch(diff, source)

    import imp

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

    patched_module = imp.new_module(module_path)
    try:
        exec(patched, patched_module.__dict__)
    except Exception:
        # TODO: syntaxerror, do not produce those mutations
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
        with database_open(".") as db:
            db[lexode.pack([2, uid])] = b"\x00"
        return False
    else:
        # TODO: pass root path...
        with database_open(".") as db:
            del db[lexode.pack([2, uid])]
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

    if arguments["<file-or-directory>"] and arguments["TEST-COMMAND"]:
        log.error("<file-or-directory> and TEST-COMMAND are exclusive!")
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
        if arguments["TEST-COMMAND"]:
            command = list(arguments["TEST-COMMAND"])
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
        if arguments["TEST-COMMAND"]:
            command = list(arguments["TEST-COMMAND"])
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
    include = arguments.get("--include") or "*.py"
    include = include.split(",")
    include = glob2predicate(include)

    exclude = arguments.get("--exclude") or "*test*"
    exclude = exclude.split(",")
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
    command.extend(arguments["<file-or-directory>"])

    eta = humanize(alpha * total / max_workers)
    log.info("At most, it will take {} to run the mutations", eta)

    timeout = alpha * 2
    uids = db[lexode.pack([1]) : lexode.pack([2])]
    uids = ((command, lexode.unpack(key)[1], timeout) for (key, _) in uids)

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

        errors = len(list(db[lexode.pack([2]) : lexode.pack([3])]))

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
        if arguments["TEST-COMMAND"]:
            command = list(arguments["TEST-COMMAND"])
        else:
            command = list(PYTEST)
            command += arguments["<file-or-directory>"]
        command = dict(
            command=command,
            seed=seed,
        )
        value = list(command.items())
        db[lexode.pack((0, "command"))] = lexode.pack(value)

        # let's create mutations!
        count = await play_create_mutations(loop, root, db, max_workers, arguments)
        # Let's run tests against mutations!
        await play_mutations(loop, db, seed, alpha, count, max_workers, arguments)


def mutation_diff_size(db, uid):
    _, diff = lexode.unpack(db[lexode.pack([1, uid])])
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
                db[lexode.pack([2, uid])] = b"\x01"
                return
            # Otherwise loop to re-test...
        else:
            del db[lexode.pack([2, uid])]
            return


def replay(arguments):
    root = Path(".")

    with database_open(root) as db:
        command = db[lexode.pack((0, "command"))]

    command = lexode.unpack(command)
    command = dict(command)
    seed = command.pop("seed")
    random.seed(seed)
    command = command.pop("command")

    alpha, max_workers = check_tests(root, seed, arguments, command)

    with database_open(root) as db:
        while True:
            uids = (
                lexode.unpack(k)[1] for k, v in db[lexode.pack([2]) :] if v == b"\x00"
            )
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
        uids = ((lexode.unpack(k)[1], v) for k, v in db[lexode.pack([2]) :])
        uids = sorted(uids, key=lambda x: mutation_diff_size(db, x[0]), reverse=True)
    if not uids:
        log.info("No mutation failures 👍")
        sys.exit(0)
    for (uid, type) in uids:
        log.info("{}\t{}".format(uid.hex(), "skipped" if type == b"\x01" else ""))


def mutation_show(uid):
    uid = UUID(hex=uid)
    log.info("mutation show {}", uid.hex)
    log.info("")
    with database_open(".") as db:
        path, diff = lexode.unpack(db[lexode.pack([1, uid.bytes])])
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
    # TODO: add support for uuid inside lexode
    uid = UUID(hex=uid)
    with database_open(".") as db:
        path, diff = lexode.unpack(db[lexode.pack([1, uid])])
    diff = zstd.decompress(diff).decode("utf8")
    with open(path, "r") as f:
        source = f.read()
    patched = patch(diff, source)
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
