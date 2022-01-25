import sys
from foobar.ex import decrement_by_two


def test_one():
    x = decrement_by_two(10)
    assert x < 10


def test_two():
    x = decrement_by_two(44)
    assert x < 44
