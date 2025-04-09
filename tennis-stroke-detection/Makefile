# Tennis Stroke Detection Pipeline
# Author: Noah Cladera

.PHONY: all install install-dev venv test lint format docs clean data download extract normalize features train predict clip

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
BLACK = black
FLAKE8 = flake8
ISORT = isort
MYPY = mypy

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
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make docs        - Generate documentation"
	@echo ""
	@echo "Pipeline:"
	@echo "  make all         - Run full pipeline"
	@echo "  make download    - Download and process videos"
	@echo "  make extract     - Extract pose data"
	@echo "  make normalize   - Normalize pose data"
	@echo "  make features    - Generate features"
	@echo "  make train       - Train the model"
	@echo "  make predict     - Run inference"
	@echo "  make clip        - Generate video clips"
	@echo ""
	@echo "Data:"
	@echo "  make data        - Prepare data directories"
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

lint:
	@echo "Running linters..."
	$(FLAKE8) src/ tests/
	$(ISORT) --check-only src/ tests/
	$(MYPY) src/

format:
	@echo "Formatting code..."
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/

docs:
	@echo "Generating documentation..."
	cd docs && make html

# Data Setup
data:
	@echo "Setting up data directories..."
	mkdir -p data/raw
	mkdir -p data/interim
	mkdir -p data/processed
	mkdir -p data/external

# Pipeline Commands
download:
	@echo "Downloading and processing raw videos..."
	$(PYTHON) -m src.data.acquisition

extract:
	@echo "Extracting pose data from videos..."
	$(PYTHON) -m src.data.pose_extraction

normalize:
	@echo "Normalizing pose data..."
	$(PYTHON) -m src.data.normalization

features:
	@echo "Performing feature engineering..."
	$(PYTHON) -m src.data.feature_engineering

train:
	@echo "Training the detection model..."
	$(PYTHON) -m src.training.trainer

predict:
	@echo "Running inference..."
	$(PYTHON) -m src.inference.predictor

clip:
	@echo "Generating clips from detection results..."
	$(PYTHON) -m src.inference.clip_generator

# Full pipeline
all: data download extract normalize features train predict clip

# Cleanup
clean:
	@echo "Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	rm -rf docs/_build/