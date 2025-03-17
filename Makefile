.PHONY: lock env test precommit all

# Target: lock
# This target regenerates your dependency lock file.
# For example, if you're using conda-lock to pin every package exactly, this will update the lock file.
lock:
	@echo "Generating lock file..."
	conda-lock -f environment.yml

# Target: env
# This target creates or updates your conda environment based on your environment.yml.
env:
	@echo "Creating conda environment..."
	conda env create -f environment.yml 



# Target: precommit
# This target runs all pre-commit hooks on all files in your repo.
# It ensures code quality checks (formatting, linting, etc.) are executed.
precommit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# Target: test
# This target runs your test suite using pytest.
test:
	@echo "Running tests..."
# pytest

# Target: all
# This target runs the full workflow sequentially:
# 1. Generate the lock file.
# 2. Create or update the environment.
# 3. Run pre-commit hooks.
# 4. Run tests.
all: lock env precommit test
	@echo "Full workflow completed!"
