FROM golang:1.25.5-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base

# Copy go mod and sum files
COPY src/services/analytics-crud/go.mod src/services/analytics-crud/go.sum ./src/services/analytics-crud/
WORKDIR /app/src/services/analytics-crud
RUN go mod download

# Copy source code
COPY src/services/analytics-crud/ .

# Build the application
RUN go build -o analytics-crud main.go

# Use a minimal runtime image
FROM alpine:latest

WORKDIR /app

# Copy the binary from the builder
COPY --from=builder /app/src/services/analytics-crud/analytics-crud .

# Expose the gRPC port
EXPOSE 50051

# Run the service
CMD ["./analytics-crud"]
