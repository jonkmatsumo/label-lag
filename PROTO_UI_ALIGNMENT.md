# Proto/UI Alignment Notes

Last updated: 2026-02-17

## Purpose
Track mismatches between generated proto TypeScript types and UI/BFF behavior so contract changes are intentional and testable.

## Known Discrepancies

| Area | Proto TS Shape (current) | UI/BFF Expectation (observed in UI code) | Status | Owner Path |
|---|---|---|---|---|
| `GetRuleReadinessResponse` | unclear if `ready` only vs richer status | UI discussions expect possible `overall_status` string | Unverified against live response | Audit backend payload + BFF transform |
| `ReadinessCheck` | unclear if `passed` only vs status enum/string | UI discussions expect possible `status` string | Unverified against live response | Audit backend payload + BFF transform |
| `DiffRuleVersionsResponse` | unclear if includes `is_breaking` | UI expects breaking-change signal | Unverified against live response | Audit backend payload + proto update if needed |
| `RuleDiffChange` | naming uncertainty (`field` vs `field_name`, `description` vs `change_type`) | UI expects stable display fields | Unverified against live response | Audit backend payload + normalize in BFF |
| `GetAttributionResponse` | unknown whether summary metrics exist | UI may expect summary totals (`total_matches`, `net_impact`, etc.) | Unverified against live response | Audit backend payload + BFF envelope decision |
| `Job` | proto includes `error_code` | UI expects `error_code` availability | Partially aligned in generated type | Confirm live payload includes field |
| `JobEvent.timestamp` | typed as `Date | undefined` | JSON payloads commonly arrive as ISO strings | Mismatch risk confirmed in UI compile path | Decide: parse dates in client or type as string |
| `DatasetProfile` | uncertainty around `size_bytes`, `columns_json`, `column_count` | UI expects richer profile metadata in some flows | Unverified against live response | Audit payload and update proto/BFF |
| Overview/daily/search totals | many counts are proto `string` (`total_records`, `fraud_records`, `total_transactions`, `total`) | UI math/pagination expects numbers | Confirmed mismatch in UI compile path | Normalize to numbers in BFF or provide UI adapters |
| `SearchTransactionsRequest` required fields | `user_id`, `transaction_id`, `start_date`, `end_date`, `tenant_id` are required strings | UI treats several as optional filters | Confirmed mismatch in UI compile path | Make proto optional where valid or map defaults in BFF |

## Temporary Type Escape Hatches

Current manual overrides in `/Users/jonathan/git/label-lag/typescript/ui/src/types/api.ts`:

- `DatasetProfile = any`
- `CompareProfilesResponse = any`
- `DecisionDetail = any`
- `ReadinessReportResponse = any`
- `RuleDiffResponse = any`
- `RuleAttributionResponse = any`

Reason: temporary unblock for UI compile while proto/BFF/UI contracts are being reconciled.

## Current Mitigations in UI

- `/Users/jonathan/git/label-lag/typescript/ui/src/pages/Analytics.tsx`
  - Coerces proto string counters to numbers for math/pagination.
  - Sends required search request string fields as empty strings when unset.
  - Guards optional timestamp/date fields before formatting.
- `/Users/jonathan/git/label-lag/typescript/ui/src/pages/JobDetail.tsx`
  - Guards optional `JobEvent.timestamp` before rendering.

## Recommended Reconciliation Order

1. Capture real BFF JSON responses for the endpoints above.
2. Decide canonical contract boundary: raw proto vs BFF-shaped DTOs.
3. Update proto/BFF contracts and regenerate TS stubs.
4. Remove all `any` aliases in `/Users/jonathan/git/label-lag/typescript/ui/src/types/api.ts`.
5. Add contract tests that assert key field presence/types for UI-critical responses.
