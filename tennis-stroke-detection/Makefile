# Tennis Stroke Detection Pipeline
# Author: Noah Cladera

.PHONY: all install install-dev venv test format clean pre-cut post-cut

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
BLACK = black
ISORT = isort

# Default target
.DEFAULT_GOAL := help

# Help command
help:
	@echo "Tennis Stroke Detection Makefile"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make venv        - Create a virtual environment"
	@echo "  make install     - Install production dependencies"
	@echo "  make install-dev - Install development dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make test        - Run tests"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pre-cut     - Run pre-cut processing (download, pose extraction, normalization)"
	@echo "  make post-cut    - Run post-cut processing (feature engineering, clip generation)"
	@echo ""
	@echo "Clean:"
	@echo "  make clean       - Remove generated files"

# Setup
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install:
	@echo "Installing production dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# Quality Assurance
test:
	@echo "Running tests..."
	$(PYTEST) tests/ -v

format:
	@echo "Formatting code..."
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/

# Pipeline Commands
pre-cut:
	@echo "Running pre-cut processing..."
	$(PYTHON) src/data/all_in_one_precut.py

post-cut:
	@echo "Running post-cut processing..."
	$(PYTHON) src/data/all_in_one_postcut.py

# Cleanup
clean:
	@echo "Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +