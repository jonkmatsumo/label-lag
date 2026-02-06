FROM golang:1.25.5-alpine AS builder

WORKDIR /app

RUN apk add --no-cache build-base

# Copy both services go.mod to resolve dependencies
# Copy both services go.mod to resolve dependencies
COPY go/analytics-crud/go.mod go/analytics-crud/
COPY go/inference-gateway/go.mod go/inference-gateway/go.sum ./go/inference-gateway/

WORKDIR /app/go/inference-gateway
RUN go mod download

# Copy source code for both services
# Copy source code for both services
COPY go/analytics-crud /app/go/analytics-crud
COPY go/inference-gateway /app/go/inference-gateway

WORKDIR /app/go/inference-gateway
RUN go build -o inference-gateway main.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/go/inference-gateway/inference-gateway .
EXPOSE 8081
CMD ["./inference-gateway"]
