FROM golang:1.25.5-alpine AS builder

WORKDIR /app

RUN apk add --no-cache build-base

# Copy both services go.mod to resolve dependencies
# Copy both services go.mod to resolve dependencies
# Copy both services go.mod to resolve dependencies
COPY go/analytics/go.mod go/analytics/
COPY go/inference/go.mod go/inference/go.sum ./go/inference/

WORKDIR /app/go/inference
RUN go mod download

# Copy source code for both services
# Copy source code for both services
COPY go/analytics /app/go/analytics
COPY go/inference /app/go/inference

WORKDIR /app/go/inference
RUN go build -o inference main.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/go/inference/inference .
EXPOSE 8081
CMD ["./inference"]
