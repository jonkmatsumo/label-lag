FROM golang:1.25.5-alpine AS builder

WORKDIR /app

RUN apk add --no-cache build-base

# Copy all services go.mod to resolve dependencies
COPY go/analytics/go.mod go/analytics/
COPY go/forecast/go.mod go/forecast/
COPY go/training/go.mod go/training/
COPY go/orchestrator/go.mod go/orchestrator/go.sum ./go/orchestrator/

WORKDIR /app/go/orchestrator
RUN go mod download

# Copy source code for all services
COPY go/analytics /app/go/analytics
COPY go/forecast /app/go/forecast
COPY go/training /app/go/training
COPY go/orchestrator /app/go/orchestrator

WORKDIR /app/go/orchestrator
RUN go build -o orchestrator main.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/go/orchestrator/orchestrator .
COPY go/orchestrator/config /app/go/orchestrator/config
EXPOSE 8081
CMD ["./orchestrator"]
