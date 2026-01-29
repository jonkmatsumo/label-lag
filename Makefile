.PHONY: install test lint clean infra-up infra-down infra-logs app-up app-down app-build app-rebuild app-logs rebuild-api rebuild-bff rebuild-web bff-test web-test reset-db reset-minio reset-all verify

verify: test bff-test
	cd web && npm run build

install:
	uv sync --all-extras
	uv run pre-commit install

test:
	uv run pytest --cov=src/synthetic_pipeline --cov-report=term-missing

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

lint-fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

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
	docker compose -f docker-compose.app.yml down

app-build:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build

app-rebuild:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build --no-cache

app-logs:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml logs -f

# Rebuild and restart API only
rebuild-api:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build api
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d api

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
	docker compose -f docker-compose.infra.yml down -v
	docker compose -f docker-compose.app.yml down

# BFF (Backend for Frontend) targets
rebuild-bff:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build bff
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d bff

bff-test:
	cd bff && npm test

bff-dev:
	cd bff && npm run dev

# Web (React UI) targets
rebuild-web:
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml build web
	docker compose -f docker-compose.infra.yml -f docker-compose.app.yml up -d web

web-test:
	cd web && npm test

web-dev:
	cd web && npm run dev
