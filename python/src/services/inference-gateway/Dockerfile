FROM golang:1.25.5-alpine AS builder

WORKDIR /app

RUN apk add --no-cache build-base

# Copy both services go.mod to resolve dependencies
COPY src/services/analytics-crud/go.mod src/services/analytics-crud/
COPY src/services/inference-gateway/go.mod src/services/inference-gateway/go.sum ./src/services/inference-gateway/

WORKDIR /app/src/services/inference-gateway
RUN go mod download

# Copy source code for both services
COPY src/services/analytics-crud /app/src/services/analytics-crud
COPY src/services/inference-gateway /app/src/services/inference-gateway

WORKDIR /app/src/services/inference-gateway
RUN go build -o inference-gateway main.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/src/services/inference-gateway/inference-gateway .
EXPOSE 8081
CMD ["./inference-gateway"]
