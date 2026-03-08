.PHONY: init check check-with-coverage check-fail-fast check-survivors lock

init:
	uv sync

lock:
	uv lock

check:
	uv run pytest tests.py

check-with-coverage:
	uv run pytest --cov=foobar --cov-report=html tests.py

check-fail-fast:
	uv run pytest -x -vvv --capture=no tests.py

check-survivors:
	uv run python3 mutation.py play foobar/test.py --include="foobar/ex.py"

lint: ## Lint the code
	ruff check $(MAIN)

doc: ## Build the documentation
	cd doc && make html
	@echo "\033[95m\n\nBuild successful! View the docs homepage at doc/build/html/index.html.\n\032[0m"

clean: ## Clean up
	git clean -fX

todo: ## Things that should be done
	@grep -nR --color=always --before-context=2 --after-context=2 TODO $(MAIN)

xxx: ## Things that require attention
	@grep -nR --color=always --before-context=2 --after-context=2 XXX $(MAIN)

serve: ## Run the server
	uvicorn --lifespan on --log-level warning --reload $(MAIN):uvicorn

lock: ## Lock dependencies
	uv export --no-dev --no-hashes -o requirements.txt
	uv export --only-dev --no-hashes -o requirements.dev.txt

wip: ## clean up code, and commit wip
	ruff format $(MAIN)
	ruff check --fix $(MAIN)
	git add .
	git commit -m "wip"
