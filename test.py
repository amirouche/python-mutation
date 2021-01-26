import sys
from foobar.ex import decrement


def test_foobar():
    print(sys.meta_path)
    x = decrement(10)
    assert 7 < x < 9
