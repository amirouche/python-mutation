"""docstring for testing false-positive mutation (module)"""

from foobar.ex import decrement_by_two


def test_decrement_by_two():
    """docstring for testing false-positive mutation (function)"""
    decrement_by_two(5)
    assert True
