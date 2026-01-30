#!/bin/bash
set -e

# Basic smoke test for Label Lag stack

API_URL=${API_URL:-"http://localhost:8100"}
BFF_URL=${BFF_URL:-"http://localhost:3210"}

echo "--- Phase 1: Authentication ---"
echo "Verifying unauthenticated request is blocked..."
curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/data/generate" -H "Content-Type: application/json" -d '{"num_users": 10, "fraud_rate": 0.1}' | grep 401

echo "Obtaining admin token from BFF..."
TOKEN=$(curl -s -X POST "$BFF_URL/bff/v1/auth/dev-login" -H "Content-Type: application/json" -d '{"role": "admin"}' | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')

if [ -z "$TOKEN" ]; then
  echo "Failed to obtain token"
  exit 1
fi

echo "--- Phase 2: Background Jobs ---"
echo "Triggering data generation job..."
JOB_RESP=$(curl -s -X POST "$BFF_URL/bff/v1/dataset/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"num_users": 10, "fraud_rate": 0.1}')

JOB_ID=$(echo $JOB_RESP | sed -n 's/.*"job_id":\([0-9]*\).*/\1/p')
echo "Job ID: $JOB_ID"

echo "Polling job status..."
for i in {1..30}; do
  STATUS_RESP=$(curl -s -X GET "$API_URL/jobs/$JOB_ID" -H "Authorization: Bearer $TOKEN")
  STATUS=$(echo $STATUS_RESP | sed -n 's/.*"status":"\([^"]*\)".*/\1/p')
  echo "Current status: $STATUS"
  if [ "$STATUS" == "completed" ]; then
    echo "Job completed successfully!"
    break
  fi
  if [ "$STATUS" == "failed" ]; then
    echo "Job failed!"
    echo $STATUS_RESP
    exit 1
  fi
  sleep 2
done

echo "--- Phase 3: Rule Persistence ---"
echo "Creating a draft rule..."
# Get a user_id for testing
USER_ID=$(curl -s -X GET "http://localhost:8100/inference/events?limit=1" -H "Authorization: Bearer $TOKEN" | sed -n 's/.*"user_id":"\([^"]*\)".*/\1/p')
if [ -z "$USER_ID" ]; then USER_ID="smoke_test_user"; fi

RULE_ID="smoke_test_rule_$(date +%s)"
curl -s -X POST "$API_URL/rules/draft" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"id\": \"$RULE_ID\",
    \"field\": \"amount\",
    \"op\": \">\",
    \"value\": 1000,
    \"action\": \"reject\",
    \"severity\": \"high\",
    \"reason\": \"Smoke test rejection\",
    \"actor\": \"smoke-test\"
  }" > /dev/null

echo "Restarting API container..."
docker compose restart api
sleep 10

echo "Verifying rule survived restart..."
RULE_CHECK=$(curl -s -X GET "$API_URL/rules/draft/$RULE_ID" -H "Authorization: Bearer $TOKEN")
if echo "$RULE_CHECK" | grep -q "$RULE_ID"; then
  echo "Rule survived restart!"
else
  echo "Rule LOST after restart!"
  echo "$RULE_CHECK"
  exit 1
fi

echo "--- Phase 4: Durable Inference Logging ---"
echo "Triggering inference..."
curl -s -X POST "$BFF_URL/bff/v1/evaluate/signal" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"user_id\": \"$USER_ID\",
    \"amount\": 50.0,
    \"currency\": \"USD\",
    \"client_transaction_id\": \"smoke_final_$(date +%s)\"
  }" > /dev/null

echo "Verifying event persisted in DB..."
curl -s -X GET "$API_URL/inference/events?limit=5" \
  -H "Authorization: Bearer $TOKEN" | grep "$USER_ID"

echo "All integration checks passed!"

