"""Synthetic financial institution user journey data generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    pass


# State definitions for financial institution journeys
ONBOARDING_STATES = [
    "lead",
    "application_started",
    "kyc_pending",
    "kyc_approved",
    "kyc_rejected",
    "account_opened",
]

ACTIVITY_STATES = [
    "dormant",
    "low_activity",
    "medium_activity",
    "high_activity",
]

PRODUCT_STATES = [
    "checking_only",
    "savings_added",
    "credit_card_applied",
    "credit_card_approved",
    "credit_card_rejected",
    "loan_applied",
    "loan_approved",
    "loan_rejected",
    "mortgage_inquiry",
    "mortgage_approved",
    "investment_opened",
]

RISK_STATES = [
    "good_standing",
    "missed_payment",
    "delinquent_30",
    "delinquent_60",
    "delinquent_90",
    "collections",
    "closed_voluntary",
    "closed_involuntary",
]

ENGAGEMENT_STATES = [
    "app_download",
    "online_banking_active",
    "mobile_deposit_used",
    "direct_deposit_setup",
    "autopay_enabled",
    "rewards_enrolled",
]

# All possible states
ALL_STATES = (
    ONBOARDING_STATES
    + ACTIVITY_STATES
    + PRODUCT_STATES
    + RISK_STATES
    + ENGAGEMENT_STATES
)


@dataclass
class TransitionConfig:
    """Configuration for state transition probabilities."""

    # Onboarding success rate
    kyc_approval_rate: float = 0.85
    account_open_rate: float = 0.90

    # Product adoption rates
    savings_adoption_rate: float = 0.40
    credit_card_apply_rate: float = 0.30
    credit_card_approval_rate: float = 0.75
    loan_apply_rate: float = 0.15
    loan_approval_rate: float = 0.60
    mortgage_inquiry_rate: float = 0.08
    mortgage_approval_rate: float = 0.50
    investment_adoption_rate: float = 0.12

    # Activity distribution
    dormant_rate: float = 0.15
    low_activity_rate: float = 0.30
    medium_activity_rate: float = 0.35
    high_activity_rate: float = 0.20

    # Risk rates
    missed_payment_rate: float = 0.08
    delinquency_progression_rate: float = 0.40
    recovery_rate: float = 0.60
    collections_rate: float = 0.15

    # Churn rates
    voluntary_churn_rate: float = 0.05
    involuntary_churn_rate: float = 0.02

    # Engagement adoption rates
    app_download_rate: float = 0.60
    online_banking_rate: float = 0.50
    mobile_deposit_rate: float = 0.25
    direct_deposit_rate: float = 0.40
    autopay_rate: float = 0.35
    rewards_rate: float = 0.20


@dataclass
class SeasonalConfig:
    """Configuration for seasonal effects."""

    # Tax season (March-April): higher loan applications
    tax_season_loan_boost: float = 1.5

    # Holiday season (November-December): higher credit card applications
    holiday_cc_boost: float = 1.3

    # New year (January): higher new account signups
    new_year_signup_boost: float = 1.4

    # Summer (June-August): slightly lower activity
    summer_activity_reduction: float = 0.9


@dataclass
class JourneyConfig:
    """Configuration for journey generation.

    Attributes:
        transitions: Configuration for state transition probabilities.
        seasonal: Configuration for seasonal effects.
        min_observations_per_year: Minimum number of observations per sequence.
        max_observations_per_year: Maximum number of observations per sequence.
        time_slots_per_day: Number of time slots per day.
            1 = daily (default), 4 = 6-hour intervals, 24 = hourly.
            Max observations = 365 * time_slots_per_day.
    """

    transitions: TransitionConfig = field(default_factory=TransitionConfig)
    seasonal: SeasonalConfig = field(default_factory=SeasonalConfig)
    min_observations_per_year: int = 12
    max_observations_per_year: int = 365
    time_slots_per_day: int = 1


def generate_financial_journeys(
    n_users: int = 100_000,
    year: int = 2025,
    seed: int | None = None,
    config: JourneyConfig | None = None,
) -> pl.DataFrame:
    """
    Generate synthetic financial institution user journey data.

    Creates realistic customer journey sequences for a financial institution,
    including onboarding, product adoption, account activity, risk events,
    and engagement milestones.

    Args:
        n_users: Number of users to generate (default: 100,000).
        year: Year for the journey data.
        seed: Random seed for reproducibility.
        config: Journey configuration parameters.

    Returns:
        Polars DataFrame with columns:
        - id: User identifier
        - time: Time slot (0 to 365*time_slots_per_day - 1)
        - date: Actual date (derived from time slot)
        - state: Current state
        - state_category: Category of the state

    Example:
        >>> journeys = generate_financial_journeys(n_users=1000, seed=42)
        >>> print(journeys.describe())
    """
    rng = np.random.default_rng(seed)
    config = config or JourneyConfig()

    all_records: list[dict[str, object]] = []

    # Generate journeys for each user
    for user_id in range(n_users):
        user_journey = _generate_single_journey(
            user_id=user_id,
            year=year,
            rng=rng,
            config=config,
        )
        all_records.extend(user_journey)

    # Create DataFrame
    df = pl.DataFrame(all_records)

    # Add date column (convert time slots back to days for date calculation)
    start_date = date(year, 1, 1)
    time_slots_per_day = config.time_slots_per_day
    df = df.with_columns(
        (
            pl.lit(start_date)
            + pl.duration(days=(pl.col("time") // time_slots_per_day))
        ).alias("date")
    )

    # Sort by user and time
    df = df.sort(["id", "time"])

    return df


def _generate_single_journey(
    user_id: int,
    year: int,
    rng: np.random.Generator,
    config: JourneyConfig,
) -> list[dict[str, object]]:
    """Generate a single user's journey."""
    records = []
    transitions = config.transitions
    seasonal = config.seasonal

    # Determine number of observations for this user
    n_obs = int(
        rng.integers(
            config.min_observations_per_year,
            config.max_observations_per_year + 1,
        )
    )

    # Generate observation time slots (sorted, unique)
    # Total available slots = 365 days * time_slots_per_day
    total_slots = 365 * config.time_slots_per_day
    max_obs = min(n_obs, total_slots)
    obs_slots_arr = rng.choice(total_slots, size=max_obs, replace=False)
    obs_slots = sorted(int(s) for s in obs_slots_arr)

    # For internal logic, we need the time_slots_per_day
    slots_per_day = config.time_slots_per_day

    # User state tracking
    onboarding_complete = False
    has_savings = False
    has_credit_card = False
    has_loan = False
    has_mortgage = False
    has_investment = False
    is_delinquent = False
    delinquency_level = 0
    is_churned = False

    # Onboarding start slot and day
    signup_slot = obs_slots[0]
    signup_day = signup_slot // slots_per_day

    for slot in obs_slots:
        if is_churned:
            break

        # Convert slot to day for seasonal and onboarding calculations
        day = slot // slots_per_day

        # Get seasonal multipliers
        month = (day // 30) + 1
        is_tax_season = month in [3, 4]
        is_holiday_season = month in [11, 12]
        is_summer = month in [6, 7, 8]

        # Determine state for this observation
        if not onboarding_complete:
            # Onboarding phase
            days_since_signup = day - signup_day

            if days_since_signup == 0:
                state = "lead"
            elif days_since_signup <= 2:
                state = "application_started"
            elif days_since_signup <= 5:
                state = "kyc_pending"
            elif days_since_signup <= 7:
                if rng.random() < transitions.kyc_approval_rate:
                    state = "kyc_approved"
                else:
                    state = "kyc_rejected"
                    is_churned = True
            else:
                if rng.random() < transitions.account_open_rate:
                    state = "account_opened"
                    onboarding_complete = True
                else:
                    state = "kyc_approved"  # Still processing

            category = "onboarding"

        elif is_delinquent:
            # Handle delinquency progression
            if rng.random() < transitions.recovery_rate:
                state = "good_standing"
                is_delinquent = False
                delinquency_level = 0
                category = "risk"
            elif delinquency_level == 1:
                if rng.random() < transitions.delinquency_progression_rate:
                    state = "delinquent_30"
                    delinquency_level = 2
                else:
                    state = "missed_payment"
                category = "risk"
            elif delinquency_level == 2:
                if rng.random() < transitions.delinquency_progression_rate:
                    state = "delinquent_60"
                    delinquency_level = 3
                else:
                    state = "delinquent_30"
                category = "risk"
            elif delinquency_level == 3:
                if rng.random() < transitions.delinquency_progression_rate:
                    state = "delinquent_90"
                    delinquency_level = 4
                else:
                    state = "delinquent_60"
                category = "risk"
            else:
                if rng.random() < transitions.collections_rate:
                    state = "collections"
                    if rng.random() < transitions.involuntary_churn_rate * 3:
                        state = "closed_involuntary"
                        is_churned = True
                else:
                    state = "delinquent_90"
                category = "risk"

        else:
            # Active account - determine state based on probabilities
            # Check for churn first
            if rng.random() < transitions.voluntary_churn_rate / 12:
                state = "closed_voluntary"
                is_churned = True
                category = "risk"

            # Check for delinquency
            elif rng.random() < transitions.missed_payment_rate / 12:
                state = "missed_payment"
                is_delinquent = True
                delinquency_level = 1
                category = "risk"

            # Product adoption (with seasonal effects)
            elif (
                not has_savings
                and rng.random() < transitions.savings_adoption_rate / 12
            ):
                state = "savings_added"
                has_savings = True
                category = "product"

            elif not has_credit_card:
                cc_rate = transitions.credit_card_apply_rate / 12
                if is_holiday_season:
                    cc_rate *= seasonal.holiday_cc_boost
                if rng.random() < cc_rate:
                    state = "credit_card_applied"
                    if rng.random() < transitions.credit_card_approval_rate:
                        state = "credit_card_approved"
                        has_credit_card = True
                    else:
                        state = "credit_card_rejected"
                    category = "product"
                else:
                    state, category = _get_activity_state(
                        rng, transitions, is_summer, seasonal
                    )

            elif not has_loan:
                loan_rate = transitions.loan_apply_rate / 12
                if is_tax_season:
                    loan_rate *= seasonal.tax_season_loan_boost
                if rng.random() < loan_rate:
                    state = "loan_applied"
                    if rng.random() < transitions.loan_approval_rate:
                        state = "loan_approved"
                        has_loan = True
                    else:
                        state = "loan_rejected"
                    category = "product"
                else:
                    state, category = _get_activity_state(
                        rng, transitions, is_summer, seasonal
                    )

            elif (
                not has_mortgage
                and rng.random() < transitions.mortgage_inquiry_rate / 24
            ):
                state = "mortgage_inquiry"
                if rng.random() < transitions.mortgage_approval_rate:
                    state = "mortgage_approved"
                    has_mortgage = True
                category = "product"

            elif (
                not has_investment
                and rng.random() < transitions.investment_adoption_rate / 12
            ):
                state = "investment_opened"
                has_investment = True
                category = "product"

            # Engagement events
            elif rng.random() < 0.1:
                state, category = _get_engagement_state(rng, transitions)

            else:
                state, category = _get_activity_state(
                    rng, transitions, is_summer, seasonal
                )

        records.append(
            {
                "id": user_id,
                "time": slot,
                "state": state,
                "state_category": category,
            }
        )

    return records


def _get_activity_state(
    rng: np.random.Generator,
    transitions: TransitionConfig,
    is_summer: bool,
    seasonal: SeasonalConfig,
) -> tuple[str, str]:
    """Determine activity state."""
    roll = rng.random()

    # Apply summer reduction
    activity_mult = seasonal.summer_activity_reduction if is_summer else 1.0

    if roll < transitions.dormant_rate:
        return "dormant", "activity"
    elif roll < transitions.dormant_rate + transitions.low_activity_rate:
        return "low_activity", "activity"
    elif roll < (
        transitions.dormant_rate
        + transitions.low_activity_rate
        + transitions.medium_activity_rate * activity_mult
    ):
        return "medium_activity", "activity"
    else:
        return "high_activity", "activity"


def _get_engagement_state(
    rng: np.random.Generator,
    transitions: TransitionConfig,
) -> tuple[str, str]:
    """Determine engagement state."""
    engagement_probs = [
        (transitions.app_download_rate, "app_download"),
        (transitions.online_banking_rate, "online_banking_active"),
        (transitions.mobile_deposit_rate, "mobile_deposit_used"),
        (transitions.direct_deposit_rate, "direct_deposit_setup"),
        (transitions.autopay_rate, "autopay_enabled"),
        (transitions.rewards_rate, "rewards_enrolled"),
    ]

    # Normalize probabilities
    total = sum(p for p, _ in engagement_probs)
    roll = rng.random() * total

    cumsum = 0.0
    for prob, state in engagement_probs:
        cumsum += prob
        if roll < cumsum:
            return state, "engagement"

    return "online_banking_active", "engagement"


def generate_simple_sequences(
    n_sequences: int = 1000,
    sequence_length: int = 10,
    n_states: int = 4,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Generate simple random sequences for testing.

    Args:
        n_sequences: Number of sequences to generate.
        sequence_length: Length of each sequence.
        n_states: Number of possible states (A, B, C, ...).
        seed: Random seed for reproducibility.

    Returns:
        Polars DataFrame with columns: id, time, state.
    """
    rng = np.random.default_rng(seed)

    # Generate state labels
    states = [chr(ord("A") + i) for i in range(n_states)]

    records = []
    for seq_id in range(n_sequences):
        for t in range(sequence_length):
            state = rng.choice(states)
            records.append(
                {
                    "id": seq_id,
                    "time": t,
                    "state": state,
                }
            )

    return pl.DataFrame(records)


def generate_markov_sequences(
    n_sequences: int = 1000,
    sequence_length: int = 20,
    transition_matrix: np.ndarray | None = None,
    states: list[str] | None = None,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Generate sequences following a Markov chain.

    Args:
        n_sequences: Number of sequences to generate.
        sequence_length: Length of each sequence.
        transition_matrix: Square transition probability matrix.
            If None, generates a random matrix.
        states: State labels. If None, uses A, B, C, etc.
        seed: Random seed for reproducibility.

    Returns:
        Polars DataFrame with columns: id, time, state.
    """
    rng = np.random.default_rng(seed)

    # Generate default transition matrix if not provided
    if transition_matrix is None:
        n_states = 4
        transition_matrix = rng.dirichlet(np.ones(n_states), size=n_states)
    else:
        n_states = int(transition_matrix.shape[0])

    # Generate state labels if not provided
    if states is None:
        states = [chr(ord("A") + i) for i in range(n_states)]

    records: list[dict[str, int | str]] = []
    for seq_id in range(n_sequences):
        # Random initial state
        current_state = int(rng.integers(n_states))

        for t in range(sequence_length):
            records.append(
                {
                    "id": seq_id,
                    "time": t,
                    "state": states[current_state],
                }
            )

            # Transition to next state
            current_state = int(
                rng.choice(
                    n_states,
                    p=transition_matrix[current_state],
                )
            )

    return pl.DataFrame(records)
