FROM golang:1.25.5-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base

# Copy go mod and sum files
COPY go/analytics/go.mod go/analytics/go.sum ./go/analytics/
WORKDIR /app/go/analytics
RUN go mod download

# Copy source code
COPY go/analytics/ .

# Build the application
RUN go build -o analytics main.go

# Use a minimal runtime image
FROM alpine:latest

WORKDIR /app

# Copy the binary and config from the builder
COPY --from=builder /app/go/analytics/analytics .
COPY --from=builder /app/go/analytics/config ./config

# Expose the gRPC port
EXPOSE 50051

# Run the service
CMD ["./analytics"]
