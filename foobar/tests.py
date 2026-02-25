from foobar import ex


def test_000():
    assert ex.decrement_by_two(42) == 40


def test_001():
    assert ex.decrement_by_two(5) == 3
