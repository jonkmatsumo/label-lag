# Analytics Envelope and Meta Contract

This document defines the shared analytics request envelope, migration behavior, validation guardrails, and response meta shape used by BFF, UI, and Go analytics handlers.

## Request Contract: `AnalyticsQueryEnvelope`

Fields:

- `start_time` (ISO timestamp string)
- `end_time` (ISO timestamp string)
- `granularity` (`hour` | `day`)

Covered BFF endpoints:

- `GET /bff/v1/kpis`
- `GET /bff/v1/volume`
- `GET /bff/v1/analytics/confusion-matrix`
- `GET /bff/v1/analytics/rules/:rule_id/impact`
- `GET /bff/v1/jobs/summary`
- `POST /bff/v1/analytics/transactions/search`

## Migration Behavior (Legacy + Envelope)

During migration, BFF accepts both:

- legacy time fields (`start_time`/`end_time`, or `start_date`/`end_date` for transaction search)
- `query` envelope (`query.start_time`, `query.end_time`, `query.granularity`)

Deterministic precedence:

1. If `query` is present, BFF resolves from `query`.
2. If `query` is absent, BFF resolves from legacy fields.
3. If both are provided and any overlapping value differs, BFF returns `400` with:
   - `error.code = "INVALID_RANGE"`
   - `error.message = "query and legacy time fields must match when both are provided"`

Granularity compatibility:

- `group_by` is still accepted for KPI compatibility.
- If both `group_by` and `granularity` are provided, values must match.

## Validation Rules

All covered endpoints use the shared analytics validator via `resolveAnalyticsQueryInput(...)` and `validateAnalyticsQuery(...)`.

Validation:

- ISO timestamp parsing for `start_time` and `end_time`
- strict ordering: `start_time < end_time`
- granularity enum validation: `hour` or `day`
- bounded query windows:
  - `day`: max 90 days
  - `hour`: max 14 days

Validation failures return `400` using the standard BFF error envelope (`error.code`, `error.message`).

## Response Contract: `meta`

Analytics responses expose:

```json
{
  "meta": {
    "truncated": true,
    "partial": true,
    "effective_limit": 500
  }
}
```

Field meanings:

- `meta.truncated`: response was capped or truncated by server-side limits
- `meta.partial`: response is incomplete for any reason (including truncation)
- `meta.effective_limit` (optional): server-applied limit when present

UI reads `meta.truncated` and `meta.partial` as the canonical source for truncation/partial states.

## Deprecation Plan

- Release N: support legacy fields + envelope; enforce mismatch rejection.
- Release N+1: keep compatibility, mark legacy fields as deprecated in release notes and client docs.
- Release N+2 (earliest): remove legacy time fields from BFF request handling and contract tests.

No endpoint additions or breaking changes are introduced in Release N.

## Guardrail

`typescript/bff/tests/analytics-query-guardrail.test.ts` enforces that every time-ranged analytics route is explicitly covered and uses `resolveAnalyticsQueryInput(...)`. Any new time-ranged analytics route must be added to that test and use the shared validator path.
