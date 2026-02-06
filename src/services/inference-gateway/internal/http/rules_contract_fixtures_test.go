package httpserver

const (
	// ListRulesResponseFixture represents the response from GET /rules
	ListRulesResponseFixture = `{
  "version": "live",
  "rules": [
    {
      "id": "high_velocity",
      "field": "velocity_24h",
      "op": ">",
      "value": 10,
      "action": "reject",
      "score": null,
      "severity": "high",
      "reason": "Velocity too high",
      "status": "active"
    }
  ]
}`

	// GetRuleResponseFixture represents the response from GET /rules/{id}
	GetRuleResponseFixture = `{
  "rule_id": "high_velocity",
  "field": "velocity_24h",
  "op": ">",
  "value": 10,
  "action": "reject",
  "score": null,
  "severity": "high",
  "reason": "Velocity too high",
  "status": "active",
  "created_at": "2024-01-01T00:00:00Z"
}`

	// ReadinessResponseFixture represents the response from GET /rules/{id}/readiness
	ReadinessResponseFixture = `{
  "rule_id": "high_velocity",
  "status": "ready",
  "checks": [
    {
      "name": "syntax_check",
      "passed": true,
      "message": "Rule syntax is valid"
    },
    {
        "name": "shadow_mode_check",
        "passed": true,
        "message": "Shadow mode evaluation passed"
    }
  ],
  "overall_ready": true,
  "generated_at": "2024-01-01T00:00:00Z"
}`

	// RuleVersionsListResponseFixture represents the response from GET /rules/{id}/versions
	RuleVersionsListResponseFixture = `{
  "rule_id": "high_velocity",
  "versions": [
    {
      "version": "v2",
      "created_at": "2024-01-02T00:00:00Z",
      "created_by": "user1",
      "status": "active",
      "change_description": "Updated threshold"
    },
    {
      "version": "v1",
      "created_at": "2024-01-01T00:00:00Z",
      "created_by": "user1",
      "status": "archived",
      "change_description": "Initial creation"
    }
  ],
  "total": 2
}`

	// RuleDiffResponseFixture represents the response from GET /rules/{id}/diff
	RuleDiffResponseFixture = `{
  "rule_id": "high_velocity",
  "version_a": "v1",
  "version_b": "v2",
  "diff_type": "modification",
  "changes": [
    {
      "field": "value",
      "old_value": 5,
      "new_value": 10,
      "description": "Value changed from 5 to 10"
    }
  ],
  "generated_at": "2024-01-02T00:00:00Z"
}`
)
