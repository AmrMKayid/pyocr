clean:
	find . -name "*.pyc" -type f -exec rm -rf {} +
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".mypy_cache" -type d -exec rm -rf {} +
	find . -name ".ruff_cache" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +
	find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +