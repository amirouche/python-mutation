"""docstring for testing false-positive mutation (module)"""


def decrement_by_two(a):
    """docstring for testing false-positive mutation (function)"""
    abc = 101  # noqa: F841
    ijk = 123 / a  # noqa: F841
    return a - 2
