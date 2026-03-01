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
    MutateContextManager,
    MutateDefaultArgument,
    MutateExceptionHandler,
    MutateFString,
    MutateGlobal,
    MutateIdentity,
    MutateIterator,
    MutateLambda,
    MutateReturn,
    MutateSlice,
    MutateStringMethod,
    MutateYield,
    Mutation,
    NegateCondition,
    RemoveDecorator,
    RemoveUnaryOp,
    SwapArguments,
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


def test_mutate_string_method_lstrip_multi_target():
    # lstrip should produce both rstrip and removeprefix mutations
    source = "def f(s):\n    return s.lstrip('x')\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateStringMethod()]))
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("rstrip" in m for m in mutated)
    assert any("removeprefix" in m for m in mutated)


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


def test_mutate_global():
    source = "x = 0\ndef f():\n    global x\n    x = 1\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateGlobal()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("global" not in m for m in mutated)


def test_mutate_nonlocal():
    source = "def outer():\n    x = 0\n    def inner():\n        nonlocal x\n        x = 1\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateGlobal()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("nonlocal" not in m for m in mutated)


def test_mutate_global_only_statement_skipped():
    # sole statement in body — removing it would produce empty body → skipped
    source = "x = 0\ndef f():\n    global x\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateGlobal()]))
    assert not deltas


def test_mutate_fstring():
    source = 'def f(name):\n    return f"hello {name}"\n'
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateFString()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("{name}" not in m for m in mutated)


def test_mutate_fstring_multiple():
    source = 'def f(a, b):\n    return f"{a} and {b}"\n'
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateFString()]))
    assert len(deltas) == 2
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("{a}" not in m and "{b}" in m for m in mutated)
    assert any("{b}" not in m and "{a}" in m for m in mutated)


def test_mutate_context_manager():
    source = "def f():\n    with lock:\n        do_stuff()\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateContextManager()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    # with stripped, body preserved
    assert any("with" not in m.split("def")[1] for m in mutated)
    assert any("do_stuff()" in m for m in mutated)


def test_mutate_context_manager_multi():
    source = "def f():\n    with a, b:\n        pass\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateContextManager()]))
    assert len(deltas) == 2
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("with a:" in m for m in mutated)
    assert any("with b:" in m for m in mutated)


def test_mutate_default_argument():
    source = "def f(x, y=1, z=2):\n    return x + y + z\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateDefaultArgument()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("def f(x, y, z=2):" in m for m in mutated)   # drop y's default
    assert any("def f(x, y, z):" in m for m in mutated)     # drop both defaults


def test_mutate_default_argument_kwonly():
    source = "def f(x, *, y=1):\n    return x + y\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateDefaultArgument()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("def f(x, *, y):" in m for m in mutated)


def test_mutate_iterator():
    source = "def f(items):\n    for x in items:\n        pass\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateIterator()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("reversed(items)" in m for m in mutated)
    assert any("shuffle" in m for m in mutated)
    assert any("_mutation_seq_" in m for m in mutated)


def test_mutate_iterator_no_double_wrap():
    source = "def f(items):\n    for x in reversed(items):\n        pass\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateIterator()]))
    assert not deltas


def test_swap_arguments():
    source = "def f(a, b):\n    return g(a, b)\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [SwapArguments()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("g(b, a)" in m for m in mutated)


def test_mutate_slice():
    source = "def f(a):\n    return a[1:3]\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateSlice()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("a[:3]" in m for m in mutated)
    assert any("a[1:]" in m for m in mutated)


def test_mutate_slice_step_negation():
    # positive step → negated
    source = "def f(a):\n    return a[::2]\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateSlice()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("a[::-2]" in m for m in mutated)


def test_mutate_slice_step_negation_reverse():
    # negative step (reversal idiom) → stripped to positive
    source = "def f(a):\n    return a[::-1]\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateSlice()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("a[::1]" in m for m in mutated)


def test_mutate_yield():
    source = "def f():\n    yield 42\n"
    canonical = stdlib_ast.unparse(stdlib_ast.parse(source))
    coverage = _full_coverage(source)
    deltas = list(iter_deltas(source, "test.py", coverage, [MutateYield()]))
    assert deltas
    mutated = [mutation_patch(d, canonical) for d in deltas]
    assert any("yield None" in m for m in mutated)


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
