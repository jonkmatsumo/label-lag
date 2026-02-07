.PHONY: up down restart install test lint clean infra-up infra-down infra-logs app-up app-down app-build app-rebuild app-logs rebuild-api rebuild-bff rebuild-web bff-test web-test reset-db reset-minio reset-all proto-gen proto-gen-go proto-gen-python

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

# Proto generation
PROTO_DIR = proto
PYTHON_SRC_DIR = python/src

proto-gen: proto-gen-go proto-gen-python

proto-gen-go:
	@echo "Generating Go stubs..."
	# Inference service
	@mkdir -p go/inference/internal/grpc/inferencev1
	protoc -I $(PROTO_DIR) \
		--go_out=. --go_opt=module=github.com/jonkmatsumo/label-lag \
		--go-grpc_out=. --go-grpc_opt=module=github.com/jonkmatsumo/label-lag \
		$(PROTO_DIR)/inference/v1/inference.proto \
		$(PROTO_DIR)/inference/v1/signal.proto
	@mv go/inference/internal/grpc/inferencev1/inference/v1/*.go go/inference/internal/grpc/inferencev1/ 2>/dev/null || true
	@rm -rf go/inference/internal/grpc/inferencev1/inference 2>/dev/null || true

	# Gateway service
	@mkdir -p go/inference/internal/http/gatewayv1
	protoc -I $(PROTO_DIR) \
		--go_out=. --go_opt=module=github.com/jonkmatsumo/label-lag \
		--go-grpc_out=. --go-grpc_opt=module=github.com/jonkmatsumo/label-lag \
		$(PROTO_DIR)/inference/v1/gateway.proto
	@mv go/inference/internal/http/gatewayv1/inference/v1/*.go go/inference/internal/http/gatewayv1/ 2>/dev/null || true
	@rm -rf go/inference/internal/http/gatewayv1/inference 2>/dev/null || true

	# Analytics service
	@mkdir -p go/analytics/proto/crud/v1
	protoc -I $(PROTO_DIR) \
		--go_out=. --go_opt=module=github.com/jonkmatsumo/label-lag \
		--go-grpc_out=. --go-grpc_opt=module=github.com/jonkmatsumo/label-lag \
		$(PROTO_DIR)/analytics/v1/analytics.proto
	@mv go/analytics/proto/crud/v1/analytics/v1/*.go go/analytics/proto/crud/v1/ 2>/dev/null || true
	@rm -rf go/analytics/proto/crud/v1/analytics 2>/dev/null || true

	# Training service
	@mkdir -p go/training/proto/trainingv1
	protoc -I $(PROTO_DIR) \
		--go_out=. --go_opt=module=github.com/jonkmatsumo/label-lag \
		--go-grpc_out=. --go-grpc_opt=module=github.com/jonkmatsumo/label-lag \
		$(PROTO_DIR)/training/v1/training.proto
	@mv go/training/proto/trainingv1/training/v1/*.go go/training/proto/trainingv1/ 2>/dev/null || true
	@rm -rf go/training/proto/trainingv1/training 2>/dev/null || true

	# Forecast service
	@mkdir -p go/forecast/proto/forecastv1
	protoc -I $(PROTO_DIR) \
		--go_out=. --go_opt=module=github.com/jonkmatsumo/label-lag \
		--go-grpc_out=. --go-grpc_opt=module=github.com/jonkmatsumo/label-lag \
		$(PROTO_DIR)/forecast/v1/forecast.proto
	@mv go/forecast/proto/forecastv1/forecast/v1/*.go go/forecast/proto/forecastv1/ 2>/dev/null || true
	@rm -rf go/forecast/proto/forecastv1/forecast 2>/dev/null || true

proto-gen-python:
	@echo "Generating Python stubs..."
	# Inference Server stubs
	@mkdir -p $(PYTHON_SRC_DIR)/inference_server/proto
	uv run python -m grpc_tools.protoc -I $(PROTO_DIR) \
		--python_out=$(PYTHON_SRC_DIR)/inference_server/proto \
		--grpc_python_out=$(PYTHON_SRC_DIR)/inference_server/proto \
		$(PROTO_DIR)/inference/v1/*.proto
	# Training Server & Shared stubs
	# Generated into python/src so that imports like 'from analytics.v1 import ...' work
	uv run python -m grpc_tools.protoc -I $(PROTO_DIR) \
		--python_out=$(PYTHON_SRC_DIR) \
		--grpc_python_out=$(PYTHON_SRC_DIR) \
		$(PROTO_DIR)/training/v1/*.proto \
		$(PROTO_DIR)/analytics/v1/analytics.proto \
		$(PROTO_DIR)/inference/v1/gateway.proto
	@touch $(PYTHON_SRC_DIR)/training/v1/__init__.py
	@touch $(PYTHON_SRC_DIR)/analytics/__init__.py
	@touch $(PYTHON_SRC_DIR)/analytics/v1/__init__.py
	@touch $(PYTHON_SRC_DIR)/inference/__init__.py
	@touch $(PYTHON_SRC_DIR)/inference/v1/__init__.py
	# Forecast stubs
	@mkdir -p $(PYTHON_SRC_DIR)/forecast_server/proto
	uv run python -m grpc_tools.protoc -I $(PROTO_DIR) \
		--python_out=$(PYTHON_SRC_DIR)/forecast_server/proto \
		--grpc_python_out=$(PYTHON_SRC_DIR)/forecast_server/proto \
		$(PROTO_DIR)/forecast/v1/*.proto
	# Gateway stubs (legacy)
	@mkdir -p $(PYTHON_SRC_DIR)/gateway_grpc
	uv run python -m grpc_tools.protoc -I $(PROTO_DIR) \
		--python_out=$(PYTHON_SRC_DIR)/gateway_grpc \
		--grpc_python_out=$(PYTHON_SRC_DIR)/gateway_grpc \
		$(PROTO_DIR)/inference/v1/gateway.proto \
		$(PROTO_DIR)/analytics/v1/analytics.proto
	# Ensure __init__.py files
	@touch $(PYTHON_SRC_DIR)/inference_server/proto/inference/__init__.py
	@touch $(PYTHON_SRC_DIR)/inference_server/proto/inference/v1/__init__.py
	@touch $(PYTHON_SRC_DIR)/training_server/proto/training/__init__.py
	@touch $(PYTHON_SRC_DIR)/training_server/proto/training/v1/__init__.py
	@touch $(PYTHON_SRC_DIR)/forecast_server/proto/forecast/__init__.py
	@touch $(PYTHON_SRC_DIR)/forecast_server/proto/forecast/v1/__init__.py
	@touch $(PYTHON_SRC_DIR)/gateway_grpc/inference/__init__.py
	@touch $(PYTHON_SRC_DIR)/gateway_grpc/inference/v1/__init__.py
	@touch $(PYTHON_SRC_DIR)/gateway_grpc/analytics/__init__.py
	@touch $(PYTHON_SRC_DIR)/gateway_grpc/analytics/v1/__init__.py
