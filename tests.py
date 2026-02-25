import ast as stdlib_ast
import sys

from foobar.ex import decrement_by_two
from mutation import Mutation, iter_deltas
from mutation import patch as mutation_patch


def test_one():
    x = decrement_by_two(10)
    # Example sloppy assert
    assert x < 10


def test_two():
    x = decrement_by_two(44)
    assert x < 44


# -- regression tests for syntax-error mutations ------------------------------


def _full_coverage(source):
    """Return a coverage set spanning every line in source."""
    return set(range(1, source.count("\n") + 2))


def test_no_syntax_error_mutations_empty_class_body():
    """DefinitionDrop on the sole method of a class produces an empty class
    body, which is a SyntaxError.  iter_deltas must not yield such deltas."""
    source = "class Foo:\n    def bar(self):\n        pass\n"
    coverage = _full_coverage(source)

    bad = []
    for delta in iter_deltas(source, "test.py", coverage, list(Mutation.ALL)):
        mutated = mutation_patch(delta, source)
        try:
            stdlib_ast.parse(mutated)
        except SyntaxError:
            bad.append(delta)

    assert not bad, "iter_deltas yielded {:d} syntax-error mutation(s):\n{}".format(
        len(bad), "\n---\n".join(bad)
    )


def test_no_syntax_error_mutations_docstring():
    """Mutations on code with module- and function-level docstrings (the
    structure of foobar/ex.py) must not produce syntax-error mutations."""
    source = (
        '"""Module docstring."""\n'
        "\n"
        "def decrement_by_two(a):\n"
        '    """Function docstring."""\n'
        "    return a - 2\n"
    )
    coverage = _full_coverage(source)

    bad = []
    for delta in iter_deltas(source, "test.py", coverage, list(Mutation.ALL)):
        mutated = mutation_patch(delta, source)
        try:
            stdlib_ast.parse(mutated)
        except SyntaxError:
            bad.append(delta)

    assert not bad, "iter_deltas yielded {:d} syntax-error mutation(s):\n{}".format(
        len(bad), "\n---\n".join(bad)
    )
