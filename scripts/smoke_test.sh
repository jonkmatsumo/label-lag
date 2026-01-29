#!/bin/bash
set -e

# Basic smoke test for Label Lag stack

API_URL=${API_URL:-"http://localhost:8100"}
BFF_URL=${BFF_URL:-"http://localhost:3210"}

echo "Checking API health..."
curl -f "$API_URL/health"

echo "Checking BFF health..."
curl -f "$BFF_URL/health"

echo "Testing simple inference request (API)..."
curl -f -X POST "$API_URL/evaluate/signal" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "smoke_test_user",
    "amount": 100.0,
    "currency": "USD",
    "client_transaction_id": "smoke_txn_001"
  }'

echo "Smoke test passed!"

