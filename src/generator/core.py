"""Stateful synthetic data generator with fraud scenario profiles.

This module provides a UserSimulator class that maintains persistent state
for generating realistic transaction sequences with proper temporal ordering
and fraud detection simulation.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from faker import Faker

if TYPE_CHECKING:
    from numpy.random import Generator as NPGenerator

from api.crud_client import get_crud_client
from synthetic_pipeline.models import (
    AccountSnapshot,
    BehaviorMetrics,
    ConnectionMetrics,
    EvaluationMetadata,
    GeneratedRecord,
    IdentityChangeInfo,
    TransactionEvaluation,
)


class FraudScenario(str, Enum):
    """Types of fraud scenarios for stateful generation."""

    BUST_OUT = "bust_out"
    SLEEPER_ATO = "sleeper_ato"
    LEGITIMATE = "legitimate"


@dataclass
class UserState:
    """Persistent state for a user across transactions."""

    user_id: str
    balance: Decimal = Decimal("5000.00")
    transaction_count: int = 0
    total_amount_spent: Decimal = Decimal("0.00")
    avg_transaction_amount: Decimal = Decimal("0.00")
    last_transaction_time: datetime | None = None
    last_connection_time: datetime | None = None
    connections_24h: int = 0
    connections_7d: int = 0
    days_since_last_activity: float = 0.0
    pii: tuple[str, str, str] | None = None  # (name, email, phone)

    def update_after_transaction(
        self, amount: Decimal, timestamp: datetime, connections: int = 0
    ) -> None:
        """Update state after a transaction."""
        self.transaction_count += 1
        self.total_amount_spent += amount
        self.balance -= amount

        # Update running average
        if self.transaction_count > 0:
            self.avg_transaction_amount = (
                self.total_amount_spent / self.transaction_count
            )

        # Track time gaps
        if self.last_transaction_time is not None:
            delta = timestamp - self.last_transaction_time
            self.days_since_last_activity = delta.total_seconds() / 86400

        self.last_transaction_time = timestamp

        # Track connections
        if connections > 0:
            self.connections_24h = connections
            self.last_connection_time = timestamp


@dataclass
class TransactionResult:
    """Result of generating a single transaction."""

    record: GeneratedRecord
    metadata: EvaluationMetadata
    is_fraud_event: bool = False


class LabelDelaySimulator:
    """Simulates the delay between fraud occurrence and detection."""

    def __init__(
        self,
        mean_days: float = 5.0,
        sigma: float = 0.8,
        rng: NPGenerator | None = None,
    ):
        self.mean_days = mean_days
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def calculate_confirmation_time(
        self,
        fraud_transaction_time: datetime,
        simulation_date: datetime | None = None,
    ) -> tuple[datetime | None, bool]:
        if simulation_date is None:
            simulation_date = datetime.now()

        mu = np.log(self.mean_days) - (self.sigma**2) / 2
        delay_days = float(self.rng.lognormal(mu, self.sigma))
        delay_days = max(1 / 24, min(delay_days, 60.0))
        fraud_confirmed_at = fraud_transaction_time + timedelta(days=delay_days)
        is_detected = fraud_confirmed_at <= simulation_date
        return fraud_confirmed_at, is_detected


class FraudProfile(ABC):
    """Abstract base class for fraud scenario profiles."""

    @abstractmethod
    def should_trigger_fraud(self, state: UserState) -> bool:
        pass

    @abstractmethod
    def get_pre_fraud_transaction_count(self, rng: NPGenerator) -> int:
        pass

    @abstractmethod
    def generate_fraud_transaction(
        self, simulator: UserSimulator, state: UserState
    ) -> GeneratedRecord:
        pass

    @abstractmethod
    def get_scenario_type(self) -> FraudScenario:
        pass


class BustOutProfile(FraudProfile):
    def __init__(
        self,
        min_transactions: int = 20,
        max_transactions: int = 50,
        spike_multiplier: float = 5.0,
    ):
        self.min_transactions = min_transactions
        self.max_transactions = max_transactions
        self.spike_multiplier = spike_multiplier

    def should_trigger_fraud(self, state: UserState) -> bool:
        return state.transaction_count >= self.min_transactions

    def get_pre_fraud_transaction_count(self, rng: NPGenerator) -> int:
        return int(rng.integers(self.min_transactions, self.max_transactions + 1))

    def generate_fraud_transaction(
        self, simulator: UserSimulator, state: UserState
    ) -> GeneratedRecord:
        if state.avg_transaction_amount > 0:
            spike_amount = state.avg_transaction_amount * Decimal(
                str(self.spike_multiplier + simulator.rng.uniform(0.5, 2.0))
            )
        else:
            spike_amount = Decimal(str(simulator.rng.uniform(2000, 5000)))
        spike_amount = Decimal(str(round(float(spike_amount), 2)))
        amount_to_avg = (
            float(spike_amount / state.avg_transaction_amount)
            if state.avg_transaction_amount > 0
            else self.spike_multiplier + 1.0
        )
        name, email, phone = state.pii or simulator._generate_pii()
        timestamp = simulator._next_timestamp(state)
        return GeneratedRecord(
            record_id=simulator._generate_record_id(),
            user_id=state.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=state.balance,
                balance_to_transaction_ratio=float(state.balance / spike_amount)
                if spike_amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(state.balance * Decimal("1.2")), 2))
                ),
                balance_volatility_z_score=float(simulator.rng.uniform(-1.5, -0.5)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=state.connections_24h,
                bank_connections_count_7d=state.connections_7d,
                bank_connections_avg_30d=float(simulator.rng.uniform(0.5, 2.0)),
            ),
            transaction=TransactionEvaluation(
                amount=spike_amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=int(simulator.rng.integers(50, 85)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=True,
            fraud_type=FraudScenario.BUST_OUT.value,
        )

    def get_scenario_type(self) -> FraudScenario:
        return FraudScenario.BUST_OUT


class SleeperProfile(FraudProfile):
    def __init__(
        self,
        dormant_days: int = 30,
        burst_connections: int = 3,
        high_value_multiplier: float = 3.0,
    ):
        self.dormant_days = dormant_days
        self.burst_connections = burst_connections
        self.high_value_multiplier = high_value_multiplier

    def should_trigger_fraud(self, state: UserState) -> bool:
        return state.days_since_last_activity >= self.dormant_days

    def get_pre_fraud_transaction_count(self, rng: NPGenerator) -> int:
        return int(rng.integers(3, 10))

    def generate_link_burst_event(
        self, simulator: UserSimulator, state: UserState
    ) -> GeneratedRecord:
        name, email, phone = state.pii or simulator._generate_pii()
        timestamp = simulator._next_timestamp(state)
        amount = Decimal(str(round(float(simulator.rng.uniform(10, 50)), 2)))
        return GeneratedRecord(
            record_id=simulator._generate_record_id(),
            user_id=state.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=state.balance,
                balance_to_transaction_ratio=float(state.balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(state.balance * Decimal("0.9")), 2))
                ),
                balance_volatility_z_score=float(simulator.rng.uniform(-0.5, 0.5)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=self.burst_connections
                + int(simulator.rng.integers(0, 3)),
                bank_connections_count_7d=self.burst_connections
                + int(simulator.rng.integers(2, 8)),
                bank_connections_avg_30d=float(simulator.rng.uniform(0.1, 0.5)),
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=float(simulator.rng.uniform(0.2, 0.8)),
                merchant_risk_score=int(simulator.rng.integers(20, 50)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=False,
            fraud_type=None,
        )

    def generate_fraud_transaction(
        self, simulator: UserSimulator, state: UserState
    ) -> GeneratedRecord:
        if state.avg_transaction_amount > 0:
            high_value = state.avg_transaction_amount * Decimal(
                str(self.high_value_multiplier + simulator.rng.uniform(0.5, 2.0))
            )
        else:
            high_value = Decimal(str(simulator.rng.uniform(1500, 4000)))
        high_value = Decimal(str(round(float(high_value), 2)))
        name, email, phone = state.pii or simulator._generate_pii()
        timestamp = simulator._next_timestamp(state, off_hours=True)
        change_time = timestamp - timedelta(hours=float(simulator.rng.uniform(1, 70)))
        return GeneratedRecord(
            record_id=simulator._generate_record_id(),
            user_id=state.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=True,
            account=AccountSnapshot(
                available_balance=state.balance,
                balance_to_transaction_ratio=float(state.balance / high_value)
                if high_value > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(state.balance * Decimal("0.8")), 2))
                ),
                balance_volatility_z_score=float(simulator.rng.uniform(-2.0, -0.5)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=self.burst_connections + 2,
                bank_connections_count_7d=self.burst_connections + 5,
                bank_connections_avg_30d=float(simulator.rng.uniform(0.1, 0.3)),
            ),
            transaction=TransactionEvaluation(
                amount=high_value,
                amount_to_avg_ratio=float(high_value / state.avg_transaction_amount)
                if state.avg_transaction_amount > 0
                else 5.0,
                merchant_risk_score=int(simulator.rng.integers(60, 95)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(
                email_changed_at=change_time,
                phone_changed_at=change_time if simulator.rng.random() > 0.5 else None,
            ),
            is_fraudulent=True,
            fraud_type=FraudScenario.SLEEPER_ATO.value,
        )

    def get_scenario_type(self) -> FraudScenario:
        return FraudScenario.SLEEPER_ATO


class UserSimulator:
    def __init__(
        self,
        user_id: str | None = None,
        initial_balance: Decimal = Decimal("5000.00"),
        fraud_profile: FraudProfile | None = None,
        seed: int | None = None,
        start_time: datetime | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)
        self.user_id = user_id or self._generate_user_id()
        self.state = UserState(
            user_id=self.user_id, balance=initial_balance, pii=self._generate_pii()
        )
        self.sequence_number = 0
        self.fraud_profile = fraud_profile
        self.label_delay = LabelDelaySimulator(rng=self.rng)
        self._current_time = start_time or (
            datetime.now() - timedelta(days=int(self.rng.integers(60, 180)))
        )
        self._fraud_triggered = False
        self._fraud_confirmed_at = None
        self._fraud_transaction_time = None
        self._burst_events_generated = 0
        self._target_pre_fraud_count = (
            fraud_profile.get_pre_fraud_transaction_count(self.rng)
            if fraud_profile
            else None
        )
        self._dormancy_triggered = False

    def _generate_record_id(self) -> str:
        return f"rec_{uuid.uuid4().hex[:12]}"

    def _generate_user_id(self) -> str:
        return f"user_{uuid.uuid4().hex[:12]}"

    def _generate_pii(self) -> tuple[str, str, str]:
        return self.faker.name(), self.faker.email(), self.faker.phone_number()

    def _next_timestamp(
        self, state: UserState, off_hours: bool = False, min_gap_hours: float = 0.5
    ) -> datetime:
        self._current_time += timedelta(
            hours=float(self.rng.uniform(min_gap_hours, 72))
        )
        if off_hours:
            hour = int(self.rng.choice(list(range(22, 24)) + list(range(0, 6))))
            self._current_time = self._current_time.replace(
                hour=hour,
                minute=int(self.rng.integers(0, 60)),
                second=int(self.rng.integers(0, 60)),
                microsecond=0,
            )
        else:
            self._current_time = self._current_time.replace(
                minute=int(self.rng.integers(0, 60)),
                second=int(self.rng.integers(0, 60)),
                microsecond=0,
            )
        return self._current_time

    def _sample_lognormal_amount(
        self, mean: float = 75.0, sigma: float = 1.0
    ) -> Decimal:
        mu = np.log(mean) - (sigma**2) / 2
        amount = max(0.01, min(float(self.rng.lognormal(mu, sigma)), 50000.0))
        return Decimal(str(round(amount, 2)))

    def _generate_legitimate_transaction(self) -> GeneratedRecord:
        name, email, phone = self.state.pii or self._generate_pii()
        timestamp = self._next_timestamp(self.state)
        amount = self._sample_lognormal_amount()
        amount_to_avg = (
            float(amount / self.state.avg_transaction_amount)
            if self.state.avg_transaction_amount > 0
            else 1.0
        )
        return GeneratedRecord(
            record_id=self._generate_record_id(),
            user_id=self.user_id,
            full_name=name,
            email=email,
            phone=phone,
            transaction_timestamp=timestamp,
            is_off_hours_txn=False,
            account=AccountSnapshot(
                available_balance=self.state.balance,
                balance_to_transaction_ratio=float(self.state.balance / amount)
                if amount > 0
                else 0.0,
            ),
            behavior=BehaviorMetrics(
                avg_available_balance_30d=Decimal(
                    str(round(float(self.state.balance * Decimal("1.1")), 2))
                ),
                balance_volatility_z_score=float(self.rng.uniform(-1.0, 1.0)),
            ),
            connection=ConnectionMetrics(
                bank_connections_count_24h=int(self.rng.integers(0, 3)),
                bank_connections_count_7d=int(self.rng.integers(0, 10)),
                bank_connections_avg_30d=float(self.rng.uniform(0.5, 2.0)),
            ),
            transaction=TransactionEvaluation(
                amount=amount,
                amount_to_avg_ratio=amount_to_avg,
                merchant_risk_score=int(self.rng.integers(5, 45)),
                is_returned=False,
            ),
            identity_changes=IdentityChangeInfo(),
            is_fraudulent=False,
            fraud_type=None,
        )

    def tick(self, simulation_date: datetime | None = None) -> TransactionResult:
        self.sequence_number += 1
        if simulation_date is None:
            simulation_date = datetime.now()
        is_fraud_event = False
        if self.fraud_profile and not self._fraud_triggered:
            if isinstance(self.fraud_profile, SleeperProfile):
                if (
                    not self._dormancy_triggered
                    and self._target_pre_fraud_count is not None
                    and self.state.transaction_count >= self._target_pre_fraud_count
                ):
                    dormancy_days = self.fraud_profile.dormant_days + float(
                        self.rng.uniform(1, 10)
                    )
                    self._current_time += timedelta(days=dormancy_days)
                    self.state.days_since_last_activity = dormancy_days
                    self._dormancy_triggered = True
                if self._dormancy_triggered:
                    if (
                        self._burst_events_generated
                        < self.fraud_profile.burst_connections
                    ):
                        record = self.fraud_profile.generate_link_burst_event(
                            self, self.state
                        )
                        self._burst_events_generated += 1
                    else:
                        record = self.fraud_profile.generate_fraud_transaction(
                            self, self.state
                        )
                        (
                            is_fraud_event,
                            self._fraud_triggered,
                            self._fraud_transaction_time,
                        ) = True, True, record.transaction_timestamp
                        self._fraud_confirmed_at, is_detected = (
                            self.label_delay.calculate_confirmation_time(
                                record.transaction_timestamp, simulation_date
                            )
                        )
                        if not is_detected:
                            record = record.model_copy(
                                update={"is_fraudulent": False, "fraud_type": None}
                            )
                else:
                    record = self._generate_legitimate_transaction()
            elif isinstance(self.fraud_profile, BustOutProfile):
                if (
                    self._target_pre_fraud_count is not None
                    and self.state.transaction_count >= self._target_pre_fraud_count
                ):
                    record = self.fraud_profile.generate_fraud_transaction(
                        self, self.state
                    )
                    (
                        is_fraud_event,
                        self._fraud_triggered,
                        self._fraud_transaction_time,
                    ) = True, True, record.transaction_timestamp
                    self._fraud_confirmed_at, is_detected = (
                        self.label_delay.calculate_confirmation_time(
                            record.transaction_timestamp, simulation_date
                        )
                    )
                    if not is_detected:
                        record = record.model_copy(
                            update={"is_fraudulent": False, "fraud_type": None}
                        )
                else:
                    record = self._generate_legitimate_transaction()
            else:
                record = self._generate_legitimate_transaction()
        else:
            record = self._generate_legitimate_transaction()
        self.state.update_after_transaction(
            record.transaction.amount,
            record.transaction_timestamp,
            record.connection.bank_connections_count_24h,
        )
        if self._fraud_confirmed_at is not None:
            delta = self._fraud_confirmed_at - record.transaction_timestamp
            days_to_fraud = delta.total_seconds() / 86400
            is_pre_fraud = days_to_fraud > 0
            is_train_eligible = (
                True
                if (is_fraud_event and isinstance(self.fraud_profile, BustOutProfile))
                else is_pre_fraud
            )
        else:
            days_to_fraud, is_pre_fraud, is_train_eligible = None, True, True
        return TransactionResult(
            record=record,
            metadata=EvaluationMetadata(
                user_id=self.user_id,
                record_id=record.record_id,
                sequence_number=self.sequence_number,
                fraud_confirmed_at=self._fraud_confirmed_at,
                is_pre_fraud=is_pre_fraud,
                days_to_fraud=days_to_fraud,
                is_train_eligible=is_train_eligible,
            ),
            is_fraud_event=is_fraud_event,
        )

    def generate_full_sequence(
        self,
        num_transactions: int | None = None,
        post_fraud_transactions: int = 3,
        simulation_date: datetime | None = None,
    ) -> tuple[list[GeneratedRecord], list[EvaluationMetadata]]:
        records, metadata = [], []
        if num_transactions is None:
            if self.fraud_profile and self._target_pre_fraud_count is not None:
                num_transactions = (
                    self._target_pre_fraud_count + 1 + post_fraud_transactions
                )
                if isinstance(self.fraud_profile, SleeperProfile):
                    num_transactions += self.fraud_profile.burst_connections
            else:
                num_transactions = int(self.rng.integers(5, 20))
        for _ in range(num_transactions):
            result = self.tick(simulation_date)
            records.append(result.record)
            metadata.append(result.metadata)
        return records, metadata


@dataclass
class GenerationConfig:
    num_users: int = 100
    fraud_rate: float = 0.1
    bust_out_ratio: float = 0.5
    sleeper_ratio: float = 0.5
    seed: int | None = None
    simulation_date: datetime | None = None


def generate_and_persist(config: GenerationConfig, **kwargs) -> tuple[int, int]:
    """Generate synthetic data and persist via Analytics service."""
    from google.protobuf.timestamp_pb2 import Timestamp

    from api.proto.proto.crud.v1 import analytics_pb2

    rng = np.random.default_rng(config.seed)
    num_fraud_users = int(config.num_users * config.fraud_rate)
    num_bust_out = int(num_fraud_users * config.bust_out_ratio)
    num_sleeper = num_fraud_users - num_bust_out
    num_legitimate = config.num_users - num_fraud_users

    all_records, all_metadata = [], []
    for i in range(num_legitimate):
        records, metadata = UserSimulator(
            seed=int(rng.integers(0, 2**31)) if config.seed else None
        ).generate_full_sequence(simulation_date=config.simulation_date)
        all_records.extend(records)
        all_metadata.extend(metadata)
    for i in range(num_bust_out):
        records, metadata = UserSimulator(
            fraud_profile=BustOutProfile(),
            seed=int(rng.integers(0, 2**31)) if config.seed else None,
        ).generate_full_sequence(simulation_date=config.simulation_date)
        all_records.extend(records)
        all_metadata.extend(metadata)
    for i in range(num_sleeper):
        records, metadata = UserSimulator(
            fraud_profile=SleeperProfile(),
            seed=int(rng.integers(0, 2**31)) if config.seed else None,
            start_time=datetime.now() - timedelta(days=int(rng.integers(90, 180))),
        ).generate_full_sequence(simulation_date=config.simulation_date)
        all_records.extend(records)
        all_metadata.extend(metadata)

    client = get_crud_client()
    proto_records = []
    for r in all_records:
        ts, ec_ts, pc_ts = Timestamp(), Timestamp(), Timestamp()
        ts.FromDatetime(r.transaction_timestamp)
        if r.identity_changes.email_changed_at:
            ec_ts.FromDatetime(r.identity_changes.email_changed_at)
        if r.identity_changes.phone_changed_at:
            pc_ts.FromDatetime(r.identity_changes.phone_changed_at)
        proto_records.append(
            analytics_pb2.GeneratedRecord(
                record_id=r.record_id,
                user_id=r.user_id,
                full_name=r.full_name,
                email=r.email,
                phone=r.phone,
                transaction_timestamp=ts,
                is_off_hours_txn=r.is_off_hours_txn,
                available_balance=float(r.account.available_balance),
                balance_to_transaction_ratio=float(
                    r.account.balance_to_transaction_ratio
                ),
                avg_available_balance_30d=float(r.behavior.avg_available_balance_30d),
                balance_volatility_z_score=float(r.behavior.balance_volatility_z_score),
                bank_connections_count_24h=r.connection.bank_connections_count_24h,
                bank_connections_count_7d=r.connection.bank_connections_count_7d,
                bank_connections_avg_30d=float(r.connection.bank_connections_avg_30d),
                amount=float(r.transaction.amount),
                amount_to_avg_ratio=float(r.transaction.amount_to_avg_ratio),
                merchant_risk_score=r.transaction.merchant_risk_score,
                is_returned=r.transaction.is_returned,
                email_changed_at=ec_ts if r.identity_changes.email_changed_at else None,
                phone_changed_at=pc_ts if r.identity_changes.phone_changed_at else None,
                is_fraudulent=r.is_fraudulent,
                fraud_type=r.fraud_type or "",
            )
        )

    proto_metadata = []
    for m in all_metadata:
        fc_ts = Timestamp()
        if m.fraud_confirmed_at:
            fc_ts.FromDatetime(m.fraud_confirmed_at)
        proto_metadata.append(
            analytics_pb2.EvaluationMetadata(
                user_id=m.user_id,
                record_id=m.record_id,
                sequence_number=m.sequence_number,
                fraud_confirmed_at=fc_ts if m.fraud_confirmed_at else None,
                is_pre_fraud=m.is_pre_fraud,
                days_to_fraud=m.days_to_fraud or 0,
                is_train_eligible=m.is_train_eligible,
            )
        )

    resp = client.store_generated_data(records=proto_records, metadata=proto_metadata)
    return int(resp.records_saved), len(proto_metadata)
