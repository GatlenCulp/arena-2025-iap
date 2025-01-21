#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = arena-2025-iap
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
DOCS_PORT ?= 8000

#################################################################################
# UTILITIES                                                                     #
#################################################################################

_prep: ## Clean up .DS_Store files
	rm -f **/*/.DS_store

_welcome: ## Print a Welcome screen
	curl -s https://raw.githubusercontent.com/GatlenCulp/gatlens-opinionated-template/vscode-customization/welcome.txt

#################################################################################
# PACKAGE COMMANDS                                                              #
#################################################################################

.PHONY: create_environment
create_environment: ## Set up python interpreter environment
	uv venv
	@echo ">>> New virtualenv with uv created. Activate with:\nsource '.venv/bin/activate'"

.PHONY: install-homebrew
install-homebrew: ## Install homebrew on a mac or linux system
	/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	# TODO: Link bin to path.

.PHONY: install-prereqs
install-prereqs: ## Installs needed dependencies with
	brew install git
	brew install uv
	brew install unzip

.PHONY: install-submodule
install-submodule: ## Installs ARENA_3.0 submodule
	git submodule update --init

.PHONY: requirements
requirements: ## Install Python Dep
	
	uv sync
	



publish-all: format lint publish docs-publish ## Run format, lint, publish package and docs

#################################################################################
# CLEAN COMMANDS                                                                #
#################################################################################


.PHONY: clean
clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


.PHONY: lint ## Lint using ruff (use `make format` to do formatting)
lint:
	ruff check --config pyproject.toml arena


.PHONY: format ## Format source code with black
format:
	ruff --config pyproject.toml arena

#################################################################################
# DOCS COMMANDS                                                                 #
#################################################################################

# Switched to using uv
docs-serve: ## Serve documentation locally on port $(DOCS_PORT)
	cd docs && \
	mkdocs serve -a localhost:$(DOCS_PORT) || \
	echo "\n\nInstance found running on $(DOCS_PORT), try killing process and rerun."

# Makes sure docs can be served prior to actually deploying
docs-publish: ## Build and deploy documentation to GitHub Pages
	cd docs && \
	mkdocs build && \
	mkdocs gh-deploy --clean

#################################################################################
# DATA COMMANDS                                                                 #
#################################################################################

#################################################################################
# TEST COMMANDS                                                                 #
#################################################################################

test: _prep ## Run all tests
	pytest -vvv --durations=0

test-fastest: _prep ## Run tests with fail-fast option
	pytest -vvv -FFF

# Requires pytest-watcher (Continuous Testing for Fast Tests)
test-continuous: _prep ## Run tests in watch mode using pytest-watcher
	ptw . --now --runner pytest --config-file pyproject.toml -vvv -FFF

test-debug-last: ## Debug last failed test with pdb
	pytest --lf --pdb

_clean_manual_test:
	rm -rf manual_test

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

.PHONY: data ## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) arena/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "\033[38;5;39m   ____  ___ _____              "
	@echo "  / ___|/ _ \_   _|__ _ __ ___  "
	@echo " | |  _| | | || |/ _ \ '_ \` _ \ "
	@echo " | |_| | |_| || |  __/ | | | | |"
	@echo "  \____|\___/ |_|\___|_| |_| |_|\033[0m"
	@echo "\n\033[1m~ Available rules: ~\033[0m\n"
	@echo "For VSCode/Cursor, try: ⇧ ⌘ P, Tasks: Run Task\n"
	@grep -E '^[a-zA-Z][a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[38;5;222m%-30s\033[0m %s\n", $$1, $$2}'
