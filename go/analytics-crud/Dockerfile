FROM golang:1.25.5-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base

# Copy go mod and sum files
COPY go/analytics-crud/go.mod go/analytics-crud/go.sum ./go/analytics-crud/
WORKDIR /app/go/analytics-crud
RUN go mod download

# Copy source code
COPY go/analytics-crud/ .

# Build the application
RUN go build -o analytics-crud main.go

# Use a minimal runtime image
FROM alpine:latest

WORKDIR /app

# Copy the binary from the builder
COPY --from=builder /app/go/analytics-crud/analytics-crud .

# Expose the gRPC port
EXPOSE 50051

# Run the service
CMD ["./analytics-crud"]
