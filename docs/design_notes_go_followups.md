# Go Service Follow-up Design Notes

## Phase 1: Circuit Breaker Strategy
- **Failure Classification**:
  - **Failures**: `context.DeadlineExceeded`, `codes.Unavailable`, `codes.Internal` (connectivity issues).
  - **Successes**: `codes.OK`, `codes.NotFound`, `codes.InvalidArgument` (application-level errors).
- **Configuration**:
  - `FailureThreshold`: 5 consecutive failures.
  - `ResetTimeout`: 10 seconds.
  - **Storage**: In-memory `CircuitBreaker` struct attached to `AnalyticsClient`.

## Phase 2 & 3: Async Logging Strategy
- **Trace Propagation**:
  - Use `trace.Link` to connect the async span to the original request span.
  - Propagate `request_id` via context metadata.
- **Concurrency Bounding**:
  - **Buffer Size**: 100 events.
  - **Worker Pool**: 5 goroutines.
  - **Drop Policy**: If buffer is full, drop event and increment `inference_log_dropped_total`.
  - **Rationale**: Prevents unbounded goroutine growth during IO spikes. 5 workers sufficient for expected log volume.

## Phase 4: Error Mapping Strategy
- **Mapping**:
  - `sql.ErrNoRows` -> `codes.NotFound`
  - Validation failures -> `codes.InvalidArgument`
  - DB Connectivity -> `codes.Unavailable`
  - Timeout -> `codes.DeadlineExceeded`
  - Others -> `codes.Internal` (log internal details, return generic message).
- **Goal**: Provide actionable status codes for clients and alerts.
