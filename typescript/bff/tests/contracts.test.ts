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
import { parseInt64, timestampToIso } from '../src/utils/protojson.js';

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
      expect(typeof change.field_name).toBe('string');
      expect((change.field_name as string).trim().length).toBeGreaterThan(0);
      expect(typeof change.change_type).toBe('string');
      expect((change.change_type as string).trim().length).toBeGreaterThan(0);
      expect(typeof change.before_value).toBe('string');
      expect(typeof change.after_value).toBe('string');
      expect(typeof change.description).toBe('string');
      expect(change.field).toBeUndefined();
      expect(change.old_value).toBeUndefined();
      expect(change.new_value).toBeUndefined();
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

// ─── KPIs (protojson handler with normalization) ───────────────────────────

const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$/;

describe('Contract: GET /bff/v1/kpis', () => {
  it('hourly fixture matches normalized BFF output', () => {
    const raw = loadFixture('kpis/hourly.json') as any;
    const expected = loadFixture('kpis/hourly.bff.json');

    const normalized = {
      total_decisions: parseInt64(raw.total_decisions),
      total_alerts: parseInt64(raw.total_alerts),
      alert_rate: raw.alert_rate,
      avg_score: raw.avg_score,
      rules_fired_total: parseInt64(raw.rules_fired_total),
      buckets: raw.buckets?.map((b: any) => ({
        timestamp: timestampToIso(b.timestamp),
        decisions: parseInt64(b.decisions),
        alerts: parseInt64(b.alerts),
        rules_fired: parseInt64(b.rules_fired),
      })),
    };

    expect(normalized).toEqual(expected);
  });

  it('all top-level int64 fields are numbers after normalization', () => {
    const raw = loadFixture('kpis/hourly.json') as any;

    const total_decisions = parseInt64(raw.total_decisions);
    const total_alerts = parseInt64(raw.total_alerts);
    const rules_fired_total = parseInt64(raw.rules_fired_total);

    expect(typeof total_decisions).toBe('number');
    expect(typeof total_alerts).toBe('number');
    expect(typeof rules_fired_total).toBe('number');
  });

  it('all bucket int64 fields are numbers and timestamps are ISO strings', () => {
    const raw = loadFixture('kpis/hourly.json') as any;

    expect(Array.isArray(raw.buckets)).toBe(true);
    expect(raw.buckets.length).toBeGreaterThan(0);

    for (const b of raw.buckets) {
      const ts = timestampToIso(b.timestamp);
      const decisions = parseInt64(b.decisions);
      const alerts = parseInt64(b.alerts);
      const rules_fired = parseInt64(b.rules_fired);

      // Timestamps must be ISO 8601
      expect(typeof ts).toBe('string');
      expect(ISO_RE.test(ts!)).toBe(true);

      // All numeric fields must be JS numbers, not strings
      expect(typeof decisions).toBe('number');
      expect(typeof alerts).toBe('number');
      expect(typeof rules_fired).toBe('number');
    }
  });

  it('raw int64 strings are NOT numbers before normalization (ensures test catches regressions)', () => {
    const raw = loadFixture('kpis/hourly.json') as any;
    // The upstream fixture intentionally uses string-serialized int64
    expect(typeof raw.total_decisions).toBe('string');
    expect(typeof raw.buckets[0].decisions).toBe('string');
  });

  it('second bucket uses ISO string timestamp and still normalizes correctly', () => {
    const raw = loadFixture('kpis/hourly.json') as any;
    // Second bucket uses an ISO string (not a protojson timestamp object)
    const b = raw.buckets[1];
    expect(typeof b.timestamp).toBe('string');
    const ts = timestampToIso(b.timestamp);
    expect(typeof ts).toBe('string');
    expect(ISO_RE.test(ts!)).toBe(true);
    expect(typeof parseInt64(b.decisions)).toBe('number');
  });
});

// ─── Volume (protojson handler with normalization) ──────────────────────────

describe('Contract: GET /bff/v1/volume', () => {
  it('daily fixture matches normalized BFF output', () => {
    const raw = loadFixture('volume/daily.json') as any;
    const expected = loadFixture('volume/daily.bff.json');

    const normalized = {
      points: raw.points?.map((p: any) => ({
        timestamp: timestampToIso(p.timestamp),
        count: parseInt64(p.count),
        alerts: parseInt64(p.alerts),
      })),
    };

    expect(normalized).toEqual(expected);
  });

  it('all point int64 fields are numbers and timestamps are ISO strings', () => {
    const raw = loadFixture('volume/daily.json') as any;

    expect(Array.isArray(raw.points)).toBe(true);
    expect(raw.points.length).toBeGreaterThan(0);

    for (const p of raw.points) {
      const ts = timestampToIso(p.timestamp);
      const count = parseInt64(p.count);
      const alerts = parseInt64(p.alerts);

      expect(typeof ts).toBe('string');
      expect(ISO_RE.test(ts!)).toBe(true);
      expect(typeof count).toBe('number');
      expect(typeof alerts).toBe('number');
    }
  });

  it('raw point int64 fields are strings before normalization', () => {
    const raw = loadFixture('volume/daily.json') as any;
    expect(typeof raw.points[0].count).toBe('string');
    expect(typeof raw.points[0].alerts).toBe('string');
  });
});

// ─── Confusion Matrix (protojson handler with normalization) ─────────────────

describe('Contract: GET /bff/v1/analytics/confusion-matrix', () => {
  it('basic fixture matches normalized BFF output', () => {
    const raw = loadFixture('confusion-matrix/basic.json') as any;
    const expected = loadFixture('confusion-matrix/basic.bff.json');

    const normalized = {
      true_positives: parseInt64(raw.true_positives),
      false_positives: parseInt64(raw.false_positives),
      true_negatives: parseInt64(raw.true_negatives),
      false_negatives: parseInt64(raw.false_negatives),
      precision: raw.precision,
      recall: raw.recall,
      f1_score: raw.f1_score,
      insufficient_labels: raw.insufficient_labels,
    };

    expect(normalized).toEqual(expected);
  });

  it('all count int64 fields are numbers after normalization', () => {
    const raw = loadFixture('confusion-matrix/basic.json') as any;

    expect(typeof parseInt64(raw.true_positives)).toBe('number');
    expect(typeof parseInt64(raw.false_positives)).toBe('number');
    expect(typeof parseInt64(raw.true_negatives)).toBe('number');
    expect(typeof parseInt64(raw.false_negatives)).toBe('number');
  });

  it('raw count int64 fields are strings before normalization (regression guard)', () => {
    const raw = loadFixture('confusion-matrix/basic.json') as any;
    // Protojson serializes int64 as strings — test fails if upstream changes encoding
    expect(typeof raw.true_positives).toBe('string');
    expect(typeof raw.false_positives).toBe('string');
    expect(typeof raw.true_negatives).toBe('string');
    expect(typeof raw.false_negatives).toBe('string');
  });

  it('float metric fields are numbers in the raw fixture (no normalization needed)', () => {
    const raw = loadFixture('confusion-matrix/basic.json') as any;
    expect(typeof raw.precision).toBe('number');
    expect(typeof raw.recall).toBe('number');
    expect(typeof raw.f1_score).toBe('number');
  });

  it('insufficient-labels fixture preserves boolean flag and normalizes counts', () => {
    const raw = loadFixture('confusion-matrix/insufficient-labels.json') as any;
    const expected = loadFixture('confusion-matrix/insufficient-labels.bff.json');

    const normalized = {
      true_positives: parseInt64(raw.true_positives),
      false_positives: parseInt64(raw.false_positives),
      true_negatives: parseInt64(raw.true_negatives),
      false_negatives: parseInt64(raw.false_negatives),
      precision: raw.precision,
      recall: raw.recall,
      f1_score: raw.f1_score,
      insufficient_labels: raw.insufficient_labels,
    };

    expect(normalized).toEqual(expected);
    expect(normalized.insufficient_labels).toBe(true);
    expect(typeof normalized.true_positives).toBe('number');
  });
});

// ─── Rule Impact (protojson handler with normalization) ──────────────────────

describe('Contract: GET /bff/v1/analytics/rules/:rule_id/impact', () => {
  it('basic fixture matches normalized BFF output', () => {
    const raw = loadFixture('rule-impact/basic.json') as any;
    const expected = loadFixture('rule-impact/basic.bff.json');

    const normalized = {
      rule_id: raw.rule_id,
      total_triggers: parseInt64(raw.total_triggers),
      avg_score_delta: raw.avg_score_delta,
      daily_buckets: raw.daily_buckets?.map((b: any) => ({
        date: b.date,
        trigger_count: parseInt64(b.trigger_count),
        avg_score_delta: b.avg_score_delta,
        decisions_changed_count: parseInt64(b.decisions_changed_count),
      })),
    };

    expect(normalized).toEqual(expected);
  });

  it('all count int64 fields are numbers after normalization', () => {
    const raw = loadFixture('rule-impact/basic.json') as any;

    expect(typeof parseInt64(raw.total_triggers)).toBe('number');
    expect(typeof parseInt64(raw.daily_buckets[0].trigger_count)).toBe('number');
    expect(typeof parseInt64(raw.daily_buckets[0].decisions_changed_count)).toBe('number');
  });

  it('raw count int64 fields are strings before normalization (regression guard)', () => {
    const raw = loadFixture('rule-impact/basic.json') as any;
    expect(typeof raw.total_triggers).toBe('string');
    expect(typeof raw.daily_buckets[0].trigger_count).toBe('string');
  });
});

// ─── Jobs Summary (protojson handler with normalization) ───────────────────

describe('Contract: GET /bff/v1/jobs/summary', () => {
  it('basic fixture matches normalized BFF output', () => {
    const raw = loadFixture('jobs-summary/basic.json') as any;
    const expected = loadFixture('jobs-summary/basic.bff.json');

    const normalized = {
      summaries: raw.summaries?.map((s: any) => ({
        bucket_time: timestampToIso(s.bucket_time),
        total_jobs: parseInt64(s.total_jobs),
        completed_jobs: parseInt64(s.completed_jobs),
        failed_jobs: parseInt64(s.failed_jobs),
      })),
    };

    expect(normalized).toEqual(expected);
  });

  it('all bucket int64 fields are numbers and timestamps are ISO strings', () => {
    const raw = loadFixture('jobs-summary/basic.json') as any;

    expect(Array.isArray(raw.summaries)).toBe(true);
    expect(raw.summaries.length).toBeGreaterThan(0);

    for (const s of raw.summaries) {
      const ts = timestampToIso(s.bucket_time);
      const total = parseInt64(s.total_jobs);
      const completed = parseInt64(s.completed_jobs);
      const failed = parseInt64(s.failed_jobs);

      expect(typeof ts).toBe('string');
      expect(ISO_RE.test(ts!)).toBe(true);
      expect(typeof total).toBe('number');
      expect(typeof completed).toBe('number');
      expect(typeof failed).toBe('number');
    }
  });

  it('raw int64 strings are NOT numbers before normalization', () => {
    const raw = loadFixture('jobs-summary/basic.json') as any;
    expect(typeof raw.summaries[0].total_jobs).toBe('string');
  });
});
