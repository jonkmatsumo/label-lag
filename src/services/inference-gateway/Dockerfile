FROM golang:1.25.5-alpine AS builder

WORKDIR /app

RUN apk add --no-cache build-base

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o inference-gateway main.go

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/inference-gateway .

EXPOSE 8081

CMD ["./inference-gateway"]
