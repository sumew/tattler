.PHONY: setup test-all test-vehicle test-lpr test-extraction clean help

# Default Python version
PYTHON := python3

help: ## Show this help message
	@echo "Available commands:"
	@echo "  setup           - Set up all virtual environments"
	@echo "  test-vehicle    - Test vehicle detection service"
	@echo "  test-lpr        - Test license plate recognition service"
	@echo "  test-integrated - Test integrated vehicle + license plate pipeline"
	@echo "  test-all        - Test all services"
	@echo "  clean           - Clean all virtual environments"
	@echo "  docker-build    - Build all Docker containers"
	@echo "  docker-up       - Start all services with Docker"
	@echo "  docker-down     - Stop all Docker services"
	@echo ""
	@echo "Use IMAGE=path/to/image.jpg with test commands for specific images"

setup: ## Set up all virtual environments for services
	@echo "Setting up virtual environments for all services..."
	@./scripts/setup_environments.sh

test-vehicle: ## Test vehicle detection service standalone (use IMAGE=path/to/image.jpg for specific image)
	@echo "Testing vehicle detection service..."
	@if [ -n "$(IMAGE)" ]; then \
		echo "Using specified image: $(IMAGE)"; \
		if [ -f "$(IMAGE)" ]; then \
			IMAGE_PATH="$$(pwd)/$(IMAGE)"; \
		else \
			IMAGE_PATH="$(IMAGE)"; \
		fi; \
		cd services_new/vehicle_detection && source venv/bin/activate && python test_standalone.py "$$IMAGE_PATH"; \
	else \
		echo "Using default image (first found)"; \
		cd services_new/vehicle_detection && source venv/bin/activate && python test_standalone.py; \
	fi

test-lpr: ## Test license plate recognition service standalone (use IMAGE=path/to/image.jpg for specific image)
	@echo "Testing license plate recognition service..."
	@if [ -n "$(IMAGE)" ]; then \
		echo "Using specified image: $(IMAGE)"; \
		if [ -f "$(IMAGE)" ]; then \
			IMAGE_PATH="$$(pwd)/$(IMAGE)"; \
		else \
			IMAGE_PATH="$(IMAGE)"; \
		fi; \
		cd services_new/license_plate_recognition && source venv/bin/activate && python test_standalone.py "$$IMAGE_PATH"; \
	else \
		echo "Using default image (first found)"; \
		cd services_new/license_plate_recognition && source venv/bin/activate && python test_standalone.py; \
	fi

test-extraction: ## Test frame extraction service standalone
	@echo "Testing frame extraction service..."
	@cd services_new/frame_extraction && source venv/bin/activate && python test_standalone.py

test-integrated: ## Test integrated vehicle detection + license plate recognition (use IMAGE=path/to/image.jpg for specific image)
	@echo "Testing integrated pipeline..."
	@if [ -n "$(IMAGE)" ]; then \
		echo "Using specified image: $(IMAGE)"; \
		if [ -f "$(IMAGE)" ]; then \
			IMAGE_PATH="$$(pwd)/$(IMAGE)"; \
		else \
			IMAGE_PATH="$(IMAGE)"; \
		fi; \
		cd services_new/license_plate_recognition && source venv/bin/activate && python test_integrated.py "$$IMAGE_PATH"; \
	else \
		echo "Using default image (first found)"; \
		cd services_new/license_plate_recognition && source venv/bin/activate && python test_integrated.py; \
	fi

test-all: test-vehicle test-lpr test-integrated ## Test all services standalone

clean: ## Clean all virtual environments
	@echo "Cleaning virtual environments..."
	@rm -rf services_new/vehicle_detection/venv
	@rm -rf services_new/license_plate_recognition/venv
	@rm -rf services_new/frame_extraction/venv
	@rm -rf services_new/persistence/venv
	@echo "Cleaned!"

run-pipeline: ## Run the full pipeline (requires VIDEO parameter)
	@if [ -z "$(VIDEO)" ]; then \
		echo "Usage: make run-pipeline VIDEO=path/to/video.mp4"; \
		exit 1; \
	fi
	@echo "Running pipeline with video: $(VIDEO)"
	@./scripts/run_pipeline.sh $(VIDEO)

# Development helpers
dev-install: ## Install development dependencies
	pip install -r requirements-dev.txt

docker-build: ## Build all Docker containers
	docker-compose build

docker-up: ## Start all services with Docker
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Show logs from all services
	docker-compose logs -f
