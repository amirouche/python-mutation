import ast as stdlib_ast
import sys

from foobar.ex import decrement_by_two
import ast as _ast
from mutation import (
    AugAssignToAssign,
    BreakToReturn,
    ForceConditional,
    MutateAssignment,
    MutateCallArgs,
    MutateContainment,
    MutateExceptionHandler,
    MutateIdentity,
    MutateLambda,
    MutateReturn,
    MutateStringMethod,
    Mutation,
    NegateCondition,
    RemoveDecorator,
    RemoveUnaryOp,
    ZeroIteration,
    iter_deltas,
)
from mutation import patch as mutation_patch

if hasattr(_ast, "Match"):
    from mutation import MutateMatchCase


def test_one():
    x = decrement_by_two(10)
    # Example sloppy assert
    assert x < 10


def test_two():
    x = decrement_by_two(44)
    assert x < 44


def test_mutate_string_method():
    source = "def f(s):\n    return s.lower()\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateStringMethod()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("s.upper()" in m for m in mutated)


def test_mutate_call_args():
    source = "def f(a, b):\n    return g(a, b)\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateCallArgs()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("g(None, b)" in m for m in mutated)
    assert any("g(a, None)" in m for m in mutated)
    assert any("g(b)" in m for m in mutated)
    assert any("g(a)" in m for m in mutated)


def test_force_conditional():
    source = "def f(x):\n    if x > 0:\n        return 1\n    return 0\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [ForceConditional()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("if True" in m for m in mutated)
    assert any("if False" in m for m in mutated)


def test_mutate_exception_handler():
    source = "def f():\n    try:\n        pass\n    except ValueError:\n        pass\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateExceptionHandler()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("except Exception" in m for m in mutated)
    assert all("except ValueError" not in m for m in mutated)


def test_zero_iteration():
    source = "def f(items):\n    for x in items:\n        pass\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [ZeroIteration()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("for x in []" in m for m in mutated)


def test_remove_decorator():
    source = "def decorator(f): return f\n\n@decorator\ndef f(): pass\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [RemoveDecorator()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("@decorator" not in m for m in mutated)


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


if hasattr(_ast, "Match"):

    def test_mutate_match_case():
        source = (
            "def f(x):\n"
            "    match x:\n"
            "        case 1:\n"
            "            return 'one'\n"
            "        case 2:\n"
            "            return 'two'\n"
        )
        canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
        coverage = _full_coverage(source)
        deltas = list(iter_deltas(source, "test.py", coverage, [MutateMatchCase()]))
        assert deltas
        mutated = [mutation_patch(d, canonical) for d in deltas]
        assert any("case 1:" not in m for m in mutated)
        assert any("case 2:" not in m for m in mutated)
