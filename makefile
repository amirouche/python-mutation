.PHONY: help doc

MAIN=$(shell basename $(PWD)).py

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

init: ## Prepare the host sytem for development
	pip install -r requirements.txt
	@echo "\033[95m\n\nYou may now run 'make check'.\n\033[0m"

check-only:
	python3 mutation.py play tests.py --include="foobar/ex.py,foobar/__init__.py" --exclude="tests.py" || exit 1

check: ## Run tests
	make check-only
	#pytest -vvv --exitfirst --capture=no $(MAIN)
	bandit --skip=B101 -r $(MAIN)

check-fast: ## Run tests, fail fast
	pytest -x -vvv --capture=no $(MAIN)

check-coverage: ## Code coverage
	pytest --quiet --cov-report=term --cov-report=html --cov=. $(MAIN)

lint: ## Lint the code
	pylama $(MAIN)

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
	pip-compile -o requirements.txt requirements.source.txt

wip: ## clean up code, and commit wip
	black $(MAIN)
	isort --profile black $(MAIN)
	git add .
	git commit -m "wip"
