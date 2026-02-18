/**
 * Contract tests: validate that fixture files in testdata/contracts/ match
 * the actual shapes produced by Go backend handlers.
 *
 * Ground truth was established by reading Go handler source:
 *   - protojson handlers (UseProtoNames: true) → snake_case, int64 as string
 *   - encoding/json on custom structs → JSON numbers for int64 fields
 *   - See docs/contracts/discrepancy-matrix.md for full analysis
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { join, dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixtureDir = join(__dirname, '..', 'testdata', 'contracts');

function loadFixture(relativePath: string): unknown {
  const fullPath = join(fixtureDir, relativePath);
  return JSON.parse(readFileSync(fullPath, 'utf-8'));
}

// ─── Readiness (protojson handler) ──────────────────────────────────────────

describe('Contract: GET /rules/:rule_id/readiness', () => {
  it('pass fixture has readiness enum status fields', () => {
    const payload = loadFixture('readiness/pass.json') as Record<string, unknown>;

    expect(typeof payload.rule_id).toBe('string');
    expect(typeof payload.ready).toBe('boolean');
    expect(payload.ready).toBe(true);
    expect(payload.overall_status).toBe('READINESS_STATUS_PASS');
    expect(Array.isArray(payload.checks)).toBe(true);

    const validReadinessStatuses = new Set([
      'READINESS_STATUS_UNSPECIFIED',
      'READINESS_STATUS_PASS',
      'READINESS_STATUS_WARN',
      'READINESS_STATUS_FAIL',
      'UNRECOGNIZED',
    ]);

    const checks = payload.checks as Array<Record<string, unknown>>;
    for (const check of checks) {
      expect(typeof check.name).toBe('string');
      expect(typeof check.passed).toBe('boolean');
      expect(typeof check.status).toBe('string');
      expect(validReadinessStatuses.has(check.status as string)).toBe(true);
      expect(typeof check.message).toBe('string');
    }
  });

  it('fail fixture has ready=false and failing overall status', () => {
    const payload = loadFixture('readiness/fail.json') as Record<string, unknown>;

    expect(payload.ready).toBe(false);
    expect(payload.overall_status).toBe('READINESS_STATUS_FAIL');
    const checks = payload.checks as Array<Record<string, unknown>>;
    const failedChecks = checks.filter(c => c.passed === false);
    expect(failedChecks.length).toBeGreaterThan(0);
    expect(failedChecks.every(c => c.status === 'READINESS_STATUS_FAIL')).toBe(true);
  });
});

// ─── Rule Diff (protojson handler) ───────────────────────────────────────────

describe('Contract: GET /rules/:rule_id/diff', () => {
  it('with-changes fixture includes stable breaking-change signal', () => {
    const payload = loadFixture('rule-diff/with-changes.json') as Record<string, unknown>;

    expect(typeof payload.rule_id).toBe('string');
    // Proto field: version_a / version_b (NOT version_a_id / version_b_id)
    expect(typeof payload.version_a).toBe('string');
    expect(typeof payload.version_b).toBe('string');
    expect(payload.version_a_id).toBeUndefined();
    expect(payload.version_b_id).toBeUndefined();
    expect(typeof payload.is_breaking).toBe('boolean');
    expect(payload.is_breaking).toBe(true);

    expect(Array.isArray(payload.changes)).toBe(true);
    const changes = payload.changes as Array<Record<string, unknown>>;
    expect(changes.length).toBeGreaterThan(0);

    for (const change of changes) {
      // Proto field names: field, old_value, new_value, description
      expect(typeof change.field).toBe('string');
      expect(typeof change.old_value).toBe('string');
      expect(typeof change.new_value).toBe('string');
      expect(typeof change.description).toBe('string');
      // Canonical renames happen in a later refactor phase.
      expect(change.field_name).toBeUndefined();
      expect(change.change_type).toBeUndefined();
    }
  });

  it('no-changes fixture has empty changes array and non-breaking status', () => {
    const payload = loadFixture('rule-diff/no-changes.json') as Record<string, unknown>;

    expect(Array.isArray(payload.changes)).toBe(true);
    expect((payload.changes as unknown[]).length).toBe(0);
    expect(payload.is_breaking).toBe(false);
  });
});

// ─── Attribution (protojson handler — camelCase mismatch risk) ───────────────

describe('Contract: GET /analytics/attribution', () => {
  it('items fixture has items array with daily attribution fields', () => {
    const payload = loadFixture('attribution/items.json') as Record<string, unknown>;

    expect(Array.isArray(payload.items)).toBe(true);
    const items = payload.items as Array<Record<string, unknown>>;
    expect(items.length).toBeGreaterThan(0);

    for (const item of items) {
      expect(typeof item.date).toBe('string');
      expect(typeof item.rule_id).toBe('string');
      // Proto int64 fields serialized as strings by protojson
      expect(typeof item.contribution_score).toBe('string');
      expect(typeof item.volume).toBe('string');
    }
  });

  it('empty fixture has empty items array', () => {
    const payload = loadFixture('attribution/empty.json') as Record<string, unknown>;

    expect(Array.isArray(payload.items)).toBe(true);
    expect((payload.items as unknown[]).length).toBe(0);
  });
});

// ─── Jobs (encoding/json on proto → camelCase mismatch risk) ─────────────────

describe('Contract: GET /jobs/:job_id', () => {
  it('running job fixture has required job fields', () => {
    const payload = loadFixture('jobs/job-running.json') as Record<string, unknown>;

    expect(payload.job).toBeDefined();
    const job = payload.job as Record<string, unknown>;

    expect(typeof job.job_id).toBe('string');
    expect(typeof job.job_type).toBe('string');
    expect(job.status).toBe('RUNNING');
    expect(typeof job.created_at).toBe('string');
    expect(typeof job.started_at).toBe('string');
    // In-progress job: ended_at is empty string (not null/undefined)
    expect(job.ended_at).toBe('');
    expect(job.error_code).toBe('');
    expect(job.error_message).toBe('');
  });

  it('failed job fixture has error fields populated', () => {
    const payload = loadFixture('jobs/job-failed.json') as Record<string, unknown>;

    const job = payload.job as Record<string, unknown>;
    expect(job.status).toBe('FAILED');
    expect(typeof job.error_code).toBe('string');
    expect((job.error_code as string).length).toBeGreaterThan(0);
    expect(typeof job.error_message).toBe('string');
    expect((job.error_message as string).length).toBeGreaterThan(0);
    expect(typeof job.ended_at).toBe('string');
    expect((job.ended_at as string).length).toBeGreaterThan(0);
  });
});

// ─── Overview Metrics (custom Go struct → JSON numbers, not strings) ──────────

describe('Contract: GET /analytics/overview', () => {
  it('metrics fixture has numeric fields (not strings despite proto int64)', () => {
    const payload = loadFixture('overview/metrics.json') as Record<string, unknown>;

    // Custom overviewMetricsResponse struct uses int64 → JSON numbers (not strings)
    // This differs from protojson behaviour where int64 would be strings.
    expect(typeof payload.total_records).toBe('number');
    expect(typeof payload.fraud_records).toBe('number');
    expect(typeof payload.unique_users).toBe('number');
    expect(typeof payload.fraud_rate).toBe('number');
    expect(typeof payload.total_amount).toBe('number');
    expect(typeof payload.fraud_amount).toBe('number');

    // Sanity: values are positive
    expect(payload.total_records as number).toBeGreaterThan(0);
  });
});

// ─── Transaction Search (encoding/json on proto → numbers, total as number) ──

describe('Contract: POST /analytics/transactions/search', () => {
  it('with-results fixture has transactions array and numeric total', () => {
    const payload = loadFixture('search/with-results.json') as Record<string, unknown>;

    expect(Array.isArray(payload.transactions)).toBe(true);
    // total is a number in the fixture (backend sends number via json.Encode on custom struct)
    expect(typeof payload.total).toBe('number');
    expect(payload.total).toBe((payload.transactions as unknown[]).length);
  });

  it('empty fixture has empty transactions array and zero total', () => {
    const payload = loadFixture('search/empty.json') as Record<string, unknown>;

    expect(Array.isArray(payload.transactions)).toBe(true);
    expect((payload.transactions as unknown[]).length).toBe(0);
    expect(payload.total).toBe(0);
  });
});
