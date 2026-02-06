-- Migration: 002_add_rule_versions
-- Description: Create rule_versions table for rule history tracking
-- Required: For Go Control Plane

CREATE TABLE IF NOT EXISTS rule_versions (
    version_id VARCHAR(100) NOT NULL PRIMARY KEY, -- e.g. "v1", "v2" or UUID
    rule_id VARCHAR(100) NOT NULL,
    rule_json TEXT NOT NULL, -- Full snapshot of the rule at this version
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    change_description TEXT,
    status VARCHAR(20) NOT NULL, -- snapshot status: active, archived
    is_active BOOLEAN DEFAULT FALSE,
    CONSTRAINT fk_rule_id FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_rule_versions_rule_id ON rule_versions (rule_id);
CREATE INDEX IF NOT EXISTS ix_rule_versions_created_at ON rule_versions (created_at);

-- Add active_version_id to rules table to track current head
ALTER TABLE rules ADD COLUMN IF NOT EXISTS active_version_id VARCHAR(100);
-- We don't enforce FK from rules to rule_versions to avoid circular dependency issues during insertion 
-- (though verifiable with deferred constraints, simpler to keep loose)
