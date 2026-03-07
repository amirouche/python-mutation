.PHONY: init check check-with-coverage check-fail-fast check-survivors lock

init:
	uv sync

lock:
	uv lock

check:
	uv run python3 mutation.py play tests.py --include="foobar/ex.py,foobar/__init__.py" --exclude="tests.py"

check-with-coverage:
	uv run pytest --cov=foobar --cov-report=html tests.py

check-fail-fast:
	uv run pytest -x -vvv --capture=no tests.py

check-survivors:
	uv run python3 mutation.py play foobar/test.py foobar/tests.py --include="foobar/ex.py"
