-- Migration: 001_create_rules_table
-- Description: Create rules table for PostgreSQL rule store backend
-- Required: Only when RULE_STORE_BACKEND=postgres
--
-- Note: This migration is optional. The PostgresRuleStore automatically
-- creates tables on first use via SQLAlchemy's create_all(). This SQL
-- is provided for manual database setup or migration tooling.

CREATE TABLE IF NOT EXISTS rules (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(100) NOT NULL UNIQUE,
    field VARCHAR(100) NOT NULL,
    op VARCHAR(20) NOT NULL,
    value TEXT NOT NULL,  -- JSON-encoded value
    action VARCHAR(50) NOT NULL,
    score INTEGER,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    reason TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS ix_rules_rule_id ON rules (rule_id);
CREATE INDEX IF NOT EXISTS ix_rules_status ON rules (status);
CREATE INDEX IF NOT EXISTS ix_rules_field ON rules (field);

-- Comments for documentation
COMMENT ON TABLE rules IS 'Rule storage for fraud detection rules (PostgreSQL backend)';
COMMENT ON COLUMN rules.rule_id IS 'Unique rule identifier (business key)';
COMMENT ON COLUMN rules.field IS 'Feature field the rule evaluates';
COMMENT ON COLUMN rules.op IS 'Comparison operator (>, >=, <, <=, ==, in, not_in)';
COMMENT ON COLUMN rules.value IS 'Threshold value (JSON-encoded for complex types)';
COMMENT ON COLUMN rules.action IS 'Rule action (override_score, clamp_min, clamp_max, reject)';
COMMENT ON COLUMN rules.score IS 'Score to apply when rule matches (for score actions)';
COMMENT ON COLUMN rules.severity IS 'Rule severity level (low, medium, high, critical)';
COMMENT ON COLUMN rules.reason IS 'Human-readable explanation for rule';
COMMENT ON COLUMN rules.status IS 'Rule lifecycle status (draft, pending_review, approved, active, shadow, disabled, archived)';
