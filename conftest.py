import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mutation import pytest_addoption, pytest_configure  # noqa: E402, F401
