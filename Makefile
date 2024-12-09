# Python version
PYTHON := python3

# Virtual environment directories
VENV := venv
VENV_BIN := $(VENV)/bin

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo "make install    - Install all dependencies"
	@echo "make run       - Run the Flask application"
	@echo "make clean     - Remove virtual environment and cached files"

# Create virtual environment
$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt --no-cache-dir

# Install dependencies
.PHONY: install
install: $(VENV)/bin/activate

# Run the application
.PHONY: run
run: install
	$(VENV_BIN)/python app.py

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf static/results/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
