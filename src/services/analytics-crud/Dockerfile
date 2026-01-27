FROM golang:1.25.5-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache build-base

# Copy go mod and sum files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN go build -o analytics-crud main.go

# Use a minimal runtime image
FROM alpine:latest

WORKDIR /app

# Copy the binary from the builder
COPY --from=builder /app/analytics-crud .

# Expose the gRPC port
EXPOSE 50051

# Run the service
CMD ["./analytics-crud"]
