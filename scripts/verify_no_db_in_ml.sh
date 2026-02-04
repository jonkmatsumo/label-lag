#!/usr/bin/env bash
# verify_no_db_in_ml.sh - Ensures no direct database access in ML service paths

set -euo pipefail

PATHS=("src/api" "src/model" "src/monitor" "src/pipeline" "src/generator" "src/main.py")
FORBIDDEN=("sqlalchemy" "DatabaseSession" "GeneratedRecordDB" "EvaluationMetadataDB")

echo "Verifying no direct DB access in ML paths..."

EXIT_CODE=0

for path in "${PATHS[@]}"; do
  for pattern in "${FORBIDDEN[@]}"; do
    # Exclude proto files and pycache
    if grep -r "$pattern" "$path" --exclude="*pb2*" --exclude-dir="__pycache__" > /dev/null; then
      echo "❌ ERROR: Forbidden pattern '$pattern' found in $path"
      grep -r "$pattern" "$path" --exclude="*pb2*" --exclude-dir="__pycache__"
      EXIT_CODE=1
    fi
  done
done

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ SUCCESS: No direct DB access found in ML paths."
else
  echo "❌ FAILED: Direct DB access detected."
fi

exit $EXIT_CODE
