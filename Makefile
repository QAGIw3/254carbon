.PHONY: help setup infra init-db services web clean test lint

help:
	@echo "254Carbon Market Intelligence Platform"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup      - Initial project setup"
	@echo "  make infra      - Deploy infrastructure (K8s)"
	@echo "  make init-db    - Initialize databases"
	@echo "  make services   - Start all backend services"
	@echo "  make web        - Start web frontend"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make clean      - Clean up resources"

setup:
	@echo "Setting up 254Carbon platform..."
	pip install --upgrade pip
	cd platform/apps/gateway && pip install -r requirements.txt
	cd platform/apps/curve-service && pip install -r requirements.txt
	cd platform/apps/scenario-engine && pip install -r requirements.txt
	cd platform/apps/web-hub && npm install

infra:
	@echo "Deploying infrastructure services..."
	chmod +x platform/infra/helm/deploy-infrastructure.sh
	./platform/infra/helm/deploy-infrastructure.sh

init-db:
	@echo "Initializing databases..."
	chmod +x platform/data/schemas/init-databases.sh
	./platform/data/schemas/init-databases.sh

services:
	@echo "Starting backend services..."
	cd platform && docker-compose up -d gateway curve-service scenario-engine

web:
	@echo "Starting web frontend..."
	cd platform/apps/web-hub && npm run dev

test:
	@echo "Running tests..."
	pytest platform/apps/*/tests/ -v

lint:
	@echo "Running linters..."
	black platform/apps platform/data
	ruff check platform/apps platform/data
	cd platform/apps/web-hub && npm run lint

clean:
	@echo "Cleaning up..."
	cd platform && docker-compose down -v
	kubectl delete namespace market-intelligence market-intelligence-infra || true

