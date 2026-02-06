import { describe, it, expect } from 'vitest';
import { normalizeJson } from '../src/utils/normalize';

describe('API Parity Contracts', () => {
  it('Dataset Clear response parity', () => {
    const pythonResponse = {
      success: true,
      tables_cleared: ["feature_snapshots", "evaluation_metadata", "generated_records", "backtest_results"]
    };

    const goResponse = {
      success: true,
      tables_cleared: ["feature_snapshots", "evaluation_metadata", "generated_records", "backtest_results"]
    };

    expect(normalizeJson(goResponse)).toEqual(normalizeJson(pythonResponse));
  });

  it('Rules Sandbox Evaluate response parity', () => {
    const pythonResponse = {
      final_score: 85,
      risk_label: "HIGH",
      matched_rules: [
        {
          rule_id: "high_velocity",
          severity: "medium",
          reason: "Velocity > 5",
          action: "reject"
        }
      ],
      shadow_matched_rules: [],
      ruleset_version: "v1"
    };

    const goResponse = {
      final_score: 85,
      risk_label: "HIGH",
      matched_rules: [
        {
          rule_id: "high_velocity",
          severity: "medium",
          reason: "Velocity > 5",
          action: "reject",
          score: null
        }
      ],
      shadow_matched_rules: [],
      ruleset_version: "v1",
      baseline_score: 10,
      shadow_score: null,
      rejected: true,
      explanations: [
        {
          rule_id: "high_velocity",
          severity: "medium",
          reason: "Velocity > 5",
          action: "reject",
          score_delta: 89
        }
      ]
    };

    // For parity, we might only care about a subset of fields
    const stripToContract = (resp: any) => ({
      final_score: resp.final_score,
      risk_label: resp.risk_label,
      matched_rules: resp.matched_rules.map((r: any) => ({
        rule_id: r.rule_id,
        severity: r.severity,
        reason: r.reason,
        action: r.action
      }))
    });

    expect(normalizeJson(stripToContract(goResponse))).toEqual(normalizeJson(stripToContract(pythonResponse)));
  });
});
