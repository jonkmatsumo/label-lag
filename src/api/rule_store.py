"""Durable rule storage and management in Postgres."""

import hashlib
import json
import logging

from sqlalchemy import select

from api.rules import Rule, RuleSet, RuleStatus
from synthetic_pipeline.db.models import (
    PublishedRuleSetDB,
    RuleDB,
    RuleVersionDB,
)
from synthetic_pipeline.db.models import (
    RuleStatus as RuleStatusDB,
)
from synthetic_pipeline.db.session import DatabaseSession

logger = logging.getLogger(__name__)


class RuleStore:
    """Manages rules and rulesets in Postgres."""

    def __init__(self, db_session: DatabaseSession | None = None):
        self.db_session = db_session or DatabaseSession()

    def _compute_content_hash(self, rule: Rule) -> str:
        """Compute SHA-256 hash of rule content for deduping versions."""
        content = {
            "field": rule.field,
            "op": rule.op,
            "value": rule.value,
            "action": rule.action,
            "score": rule.score,
            "severity": rule.severity,
            "reason": rule.reason,
        }
        dump = json.dumps(content, sort_keys=True)
        return hashlib.sha256(dump.encode()).hexdigest()

    def save_rule(self, rule: Rule, actor: str | None = None) -> RuleVersionDB:
        """Save a rule and a new version if content changed."""
        content_hash = self._compute_content_hash(rule)

        with self.db_session.get_session() as session:
            # Ensure logical rule exists
            db_rule = session.get(RuleDB, rule.id)
            if not db_rule:
                db_rule = RuleDB(id=rule.id, status=RuleStatusDB(rule.status))
                session.add(db_rule)
            else:
                db_rule.status = RuleStatusDB(rule.status)

            # Check for existing version with same hash
            stmt = select(RuleVersionDB).where(
                RuleVersionDB.rule_id == rule.id,
                RuleVersionDB.content_hash == content_hash,
            )
            existing_version = session.execute(stmt).scalar_one_or_none()

            if existing_version:
                return existing_version

            # Create new version
            new_version = RuleVersionDB(
                rule_id=rule.id,
                field=rule.field,
                op=rule.op,
                # Wrap in dict for JSONB compatibility if needed
                value={"v": rule.value},
                action=rule.action,
                score=rule.score,
                severity=rule.severity,
                reason=rule.reason,
                content_hash=content_hash,
                created_by=actor,
            )
            session.add(new_version)
            session.flush()
            return new_version

    def publish_ruleset(
        self, version_name: str, actor: str, reason: str | None = None
    ) -> PublishedRuleSetDB:
        """Snapshot all active rules into a new published ruleset."""
        with self.db_session.get_session() as session:
            # Find latest versions of all ACTIVE rules
            # latest version per rule ID where rule is ACTIVE
            active_rules_stmt = select(RuleDB).where(
                RuleDB.status == RuleStatusDB.ACTIVE
            )
            active_rules = session.execute(active_rules_stmt).scalars().all()

            latest_versions = []
            for r in active_rules:
                # Get the latest version for this rule
                v_stmt = (
                    select(RuleVersionDB)
                    .where(RuleVersionDB.rule_id == r.id)
                    .order_by(RuleVersionDB.created_at.desc())
                    .limit(1)
                )
                v = session.execute(v_stmt).scalar_one_or_none()
                if v:
                    latest_versions.append(v)

            published_rs = PublishedRuleSetDB(
                version_name=version_name,
                published_by=actor,
                reason=reason,
                rule_versions=latest_versions,
            )
            session.add(published_rs)
            session.flush()
            return published_rs

    def get_latest_published_ruleset(self) -> RuleSet | None:
        """Load the most recent published ruleset from DB."""
        with self.db_session.get_session() as session:
            stmt = (
                select(PublishedRuleSetDB)
                .order_by(PublishedRuleSetDB.published_at.desc())
                .limit(1)
            )
            db_rs = session.execute(stmt).scalar_one_or_none()

            if not db_rs:
                return None

            rules = []
            for v in db_rs.rule_versions:
                val = (
                    v.value["v"]
                    if isinstance(v.value, dict) and "v" in v.value
                    else v.value
                )
                rules.append(
                    Rule(
                        id=v.rule_id,
                        field=v.field,
                        op=v.op,
                        value=val,
                        action=v.action,
                        score=v.score,
                        severity=v.severity,
                        reason=v.reason,
                        status=RuleStatus.ACTIVE.value,
                    )
                )

            return RuleSet(version=db_rs.version_name, rules=rules)

    def list_draft_rules(self) -> list[Rule]:
        """List all rules from DB (System of Record)."""
        with self.db_session.get_session() as session:
            stmt = select(RuleDB).where(RuleDB.status != RuleStatusDB.ARCHIVED)
            db_rules = session.execute(stmt).scalars().all()

            rules = []
            for r in db_rules:
                # Get latest version for content
                v_stmt = (
                    select(RuleVersionDB)
                    .where(RuleVersionDB.rule_id == r.id)
                    .order_by(RuleVersionDB.created_at.desc())
                    .limit(1)
                )
                v = session.execute(v_stmt).scalar_one_or_none()
                if v:
                    val = (
                        v.value["v"]
                        if isinstance(v.value, dict) and "v" in v.value
                        else v.value
                    )
                    rules.append(
                        Rule(
                            id=r.id,
                            field=v.field,
                            op=v.op,
                            value=val,
                            action=v.action,
                            score=v.score,
                            severity=v.severity,
                            reason=v.reason,
                            status=r.status.value,
                        )
                    )
            return rules

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a logical rule and its latest version content."""
        with self.db_session.get_session() as session:
            db_rule = session.get(RuleDB, rule_id)
            if not db_rule or db_rule.status == RuleStatusDB.ARCHIVED:
                return None

            v_stmt = (
                select(RuleVersionDB)
                .where(RuleVersionDB.rule_id == rule_id)
                .order_by(RuleVersionDB.created_at.desc())
                .limit(1)
            )
            v = session.execute(v_stmt).scalar_one_or_none()
            if not v:
                return None

            val = (
                v.value["v"]
                if isinstance(v.value, dict) and "v" in v.value
                else v.value
            )
            return Rule(
                id=db_rule.id,
                field=v.field,
                op=v.op,
                value=val,
                action=v.action,
                score=v.score,
                severity=v.severity,
                reason=v.reason,
                status=db_rule.status.value,
            )
