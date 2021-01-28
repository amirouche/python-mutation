import sys
from foobar.ex import decrement


def test_foobar():
    x = decrement(10)
    assert 7 < x < 9
