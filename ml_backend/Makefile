.PHONY: install train inference api clean test help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)DeepScan ML Backend - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          Install all dependencies"
	@echo "  make setup            Setup directories and configuration"
	@echo ""
	@echo "$(GREEN)Training:$(NC)"
	@echo "  make train            Train all models"
	@echo "  make train-cnn        Train CNN model only"
	@echo "  make train-rnn        Train RNN model only"
	@echo ""
	@echo "$(GREEN)Inference:$(NC)"
	@echo "  make inference        Run inference engine"
	@echo "  make api              Start Flask API server"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test             Run tests"
	@echo "  make test-models      Test all models"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make clean            Clean cache and temp files"
	@echo "  make demo             Run example demonstrations"
	@echo "  make docs             Generate documentation"

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

setup:
	@echo "$(BLUE)Setting up directories...$(NC)"
	mkdir -p trained_models uploads temp data
	@echo "$(GREEN)✓ Directories created$(NC)"

train:
	@echo "$(BLUE)Starting model training...$(NC)"
	python -m training.train
	@echo "$(GREEN)✓ Training complete$(NC)"

train-cnn:
	@echo "$(BLUE)Training CNN model...$(NC)"
	python -c "from training.train import *; import json; trainer = DeepfakeDetectionTrainer(); trainer.config['models_to_train'] = ['cnn']"

inference:
	@echo "$(BLUE)Starting inference engine...$(NC)"
	python -c "from inference.engine import get_inference_engine; engine = get_inference_engine(); print('Engine ready')"

api:
	@echo "$(BLUE)Starting Flask API server...$(NC)"
	python -m inference.api

test:
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-models:
	@echo "$(BLUE)Testing model loading...$(NC)"
	python -c "from models import *; print('✓ All models imported successfully')"

demo:
	@echo "$(BLUE)Running demonstrations...$(NC)"
	python examples.py
	@echo "$(GREEN)✓ Demonstrations complete$(NC)"

clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache build dist *.egg-info
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "README.md - Main documentation"
	@echo "SETUP.md - Setup guide"
	@echo "config.py - Configuration options"

lint:
	@echo "$(BLUE)Running linter...$(NC)"
	pylint models/ data/ training/ inference/ utils/ --disable=all --enable=E,F
	@echo "$(GREEN)✓ Linting complete$(NC)"

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	black models/ data/ training/ inference/ utils/ --line-length=100
	@echo "$(GREEN)✓ Formatting complete$(NC)"

requirements:
	@echo "$(BLUE)Updating requirements.txt...$(NC)"
	pip freeze > requirements.txt
	@echo "$(GREEN)✓ Requirements updated$(NC)"

.DEFAULT_GOAL := help
