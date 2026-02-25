import ast as stdlib_ast
import sys

from foobar.ex import decrement_by_two
from mutation import (
    AugAssignToAssign,
    BreakToReturn,
    MutateAssignment,
    MutateContainment,
    MutateIdentity,
    MutateLambda,
    MutateReturn,
    Mutation,
    NegateCondition,
    RemoveUnaryOp,
    iter_deltas,
)
from mutation import patch as mutation_patch


def test_one():
    x = decrement_by_two(10)
    # Example sloppy assert
    assert x < 10


def test_two():
    x = decrement_by_two(44)
    assert x < 44


def test_negate_condition():
    source = "def f(flag):\n    if flag:\n        return 1\n    return 0\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [NegateCondition()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("if not flag" in m for m in mutated)


def test_mutate_return():
    source = "def f(x):\n    return x + 1\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateReturn()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("return None" in m for m in mutated)
    assert any("return 0" in m for m in mutated)
    assert any("return False" in m for m in mutated)
    assert any("return ''" in m for m in mutated)


def test_mutate_lambda():
    source = "f = lambda x: x * 2\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateLambda()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("lambda x: None" in m for m in mutated)


def test_mutate_assignment():
    source = "def f():\n    result = compute()\n    return result\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateAssignment()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("result = None" in m for m in mutated)


def test_aug_assign_to_assign():
    source = "def f():\n    x = 0\n    x += 1\n    return x\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [AugAssignToAssign()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("+=" not in m and "x = 1" in m for m in mutated)


def test_remove_unary_op():
    source = "def f(x):\n    return not x\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [RemoveUnaryOp()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("not" not in m.split("def")[1] for m in mutated)


def test_mutate_identity():
    source = "def f(x):\n    return x is None\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateIdentity()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("is not" in m for m in mutated)


def test_mutate_containment():
    source = "def f(x, c):\n    return x in c\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateContainment()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("not in" in m for m in mutated)


def test_mutate_containment_not_in():
    source = "def f(x, c):\n    return x not in c\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateContainment()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("not in" not in m and "in" in m for m in mutated)


def test_break_to_return():
    source = "def f():\n    for x in range(10):\n        break\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [BreakToReturn()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("return" in m for m in mutated)
    assert all("break" not in m for m in mutated)


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
