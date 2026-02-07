.PHONY: up down restart install test lint clean infra-up infra-down infra-logs app-up app-down app-build app-rebuild app-logs rebuild-api rebuild-bff rebuild-web bff-test web-test reset-db reset-minio reset-all

# Catch-all start command
up: app-up

# Catch-all stop command
down:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml down

# Catch-all restart command
restart: down up

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest --cov=python/src/synthetic_pipeline --cov-report=term-missing

lint:
	uv run ruff check python/src python/tests
	uv run ruff format --check python/src python/tests

lint-fix:
	uv run ruff check --fix python/src python/tests
	uv run ruff format python/src python/tests

clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Infra lifecycle (docker-compose.infra.yml)
infra-up:
	docker compose -f docker-compose.infra.yml up -d

infra-down:
	docker compose -f docker-compose.infra.yml down

infra-logs:
	docker compose -f docker-compose.infra.yml logs -f

# App lifecycle (requires infra; use -f infra -f app)
app-up:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d

app-down:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml down

app-build:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build

app-rebuild:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build --no-cache

app-logs:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml logs -f

# Rebuild and restart Training Server (API)
rebuild-training-server:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build training-server
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d training-server

# Rebuild and restart Inference Server (gRPC)
rebuild-inference-server:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build inference-server
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d inference-server

# Reset commands (destructive)
reset-db:
	docker compose -f docker-compose.infra.yml stop db
	docker rm -f synthetic-data-db 2>/dev/null || true
	docker volume rm labellag_postgres_data 2>/dev/null || true
	docker compose -f docker-compose.infra.yml up -d db

reset-minio:
	docker compose -f docker-compose.infra.yml stop minio
	docker rm -f synthetic-data-minio synthetic-data-create-buckets 2>/dev/null || true
	docker volume rm labellag_minio_data 2>/dev/null || true
	docker compose -f docker-compose.infra.yml up -d minio create-buckets

reset-all:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml down -v

# BFF (Backend for Frontend) targets
rebuild-bff:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build bff
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d bff

bff-test:
	cd typescript/bff && npm test

bff-dev:
	cd typescript/bff && npm run dev

# Web (React UI) targets
rebuild-web:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build web
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d web

web-test:
	cd typescript/ui && npm test

web-dev:
	cd typescript/ui && npm run dev
