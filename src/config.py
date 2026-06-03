"""
Configuration and constants for the MTA simulator.

Defines:
- Channel universe and properties
- Simulation parameters
- Reproducibility seed
- Date ranges and seasonality
- Channel-specific behavior curves
"""

from dataclasses import dataclass
from typing import Dict, List

# ============================================================================
# Random Seed for Reproducibility
# ============================================================================
RANDOM_SEED = 42


# ============================================================================
# Channel Definition
# ============================================================================
@dataclass
class ChannelConfig:
    """Configuration for a single marketing channel."""
    name: str
    channel_type: str  # 'paid' or 'organic'
    avg_daily_spend: float  # Average daily spend in dollars
    spend_std_dev: float  # Standard deviation as fraction of mean
    adstock_half_life: float  # Half-life of adstock decay in days
    saturation_exponent: float  # Shape of diminishing returns (0 < exp < 1)
    baseline_contribution: float  # Baseline lift without spend
    interaction_effect: Dict[str, float]  # Multiplicative lift with other channels


CHANNELS: List[ChannelConfig] = [
    ChannelConfig(
        name="Paid Search",
        channel_type="paid",
        avg_daily_spend=5000.0,
        spend_std_dev=0.15,
        adstock_half_life=1.0,
        saturation_exponent=0.5,
        baseline_contribution=0.05,
        interaction_effect={"Paid Social": 1.1, "Display": 1.05},
    ),
    ChannelConfig(
        name="Paid Social",
        channel_type="paid",
        avg_daily_spend=3000.0,
        spend_std_dev=0.20,
        adstock_half_life=2.0,
        saturation_exponent=0.6,
        baseline_contribution=0.03,
        interaction_effect={"Paid Search": 1.08, "Display": 1.12},
    ),
    ChannelConfig(
        name="Display",
        channel_type="paid",
        avg_daily_spend=2000.0,
        spend_std_dev=0.25,
        adstock_half_life=3.0,
        saturation_exponent=0.7,
        baseline_contribution=0.02,
        interaction_effect={"Paid Search": 1.05, "Paid Social": 1.08},
    ),
    ChannelConfig(
        name="Video",
        channel_type="paid",
        avg_daily_spend=2500.0,
        spend_std_dev=0.18,
        adstock_half_life=2.5,
        saturation_exponent=0.65,
        baseline_contribution=0.02,
        interaction_effect={"Paid Social": 1.15, "Display": 1.10},
    ),
    ChannelConfig(
        name="Affiliate",
        channel_type="organic",
        avg_daily_spend=500.0,
        spend_std_dev=0.30,
        adstock_half_life=1.0,
        saturation_exponent=0.5,
        baseline_contribution=0.01,
        interaction_effect={"Paid Search": 1.05},
    ),
    ChannelConfig(
        name="Email",
        channel_type="organic",
        avg_daily_spend=200.0,
        spend_std_dev=0.35,
        adstock_half_life=1.0,
        saturation_exponent=0.4,
        baseline_contribution=0.02,
        interaction_effect={"Paid Search": 1.08},
    ),
]

# Create lookup dictionaries
CHANNEL_NAMES = [ch.name for ch in CHANNELS]
CHANNEL_BY_NAME = {ch.name: ch for ch in CHANNELS}


# ============================================================================
# Simulation Parameters
# ============================================================================
@dataclass
class SimulationConfig:
    """High-level simulation parameters."""
    start_date: str  # "2024-01-01"
    num_days: int  # 365
    num_users: int  # Number of active users to simulate
    base_conversion_rate: float  # 0.02 = 2%
    avg_order_value: float  # Average revenue per conversion
    order_value_std_dev: float  # Standard deviation of order value


SIMULATION = SimulationConfig(
    start_date="2024-01-01",
    num_days=365,
    num_users=10000,
    base_conversion_rate=0.02,
    avg_order_value=100.0,
    order_value_std_dev=25.0,
)


# ============================================================================
# Seasonality and Trend
# ============================================================================
# Seasonality as a multiplier on baseline daily conversion rate
# For example, higher in Q4 (holiday season), lower in summer
SEASONALITY_MULTIPLIER: Dict[int, float] = {
    1: 0.9,   # January: back to school, lower
    2: 0.85,  # February: slower
    3: 0.9,   # March: ramp up
    4: 0.95,  # April: spring
    5: 0.92,  # May: moderate
    6: 0.88,  # June: summer dip
    7: 0.85,  # July: summer dip
    8: 0.90,  # August: back to school
    9: 1.0,   # September: normal
    10: 1.05, # October: Halloween prep, ramp to holiday
    11: 1.4,  # November: Black Friday, Cyber Monday
    12: 1.6,  # December: holiday season
}

# Annual trend: gentle growth over the year
TREND_GROWTH_RATE = 0.02  # 2% annual growth


# ============================================================================
# Journey and Path Generation
# ============================================================================
@dataclass
class JourneyConfig:
    """Parameters controlling user journeys and touch paths."""
    min_touches: int  # Minimum touches before conversion
    max_touches: int  # Maximum touches before conversion
    avg_days_to_conversion: float  # Average length of journey in days
    p_visit_without_touch: float  # Prob user visits without being touched
    p_organic_touch: float  # Prob of organic channel in journey


JOURNEY = JourneyConfig(
    min_touches=1,
    max_touches=8,
    avg_days_to_conversion=14.0,
    p_visit_without_touch=0.1,
    p_organic_touch=0.2,
)


# ============================================================================
# Attribution Model Configuration
# ============================================================================
MARKOV_ORDER = 1  # For Markov models: 1 = bigram transitions


# ============================================================================
# Output Configuration
# ============================================================================
OUTPUT_DIR = "output"
CSV_OUTPUTS = [
    "daily_spend.csv",
    "journey_touches.csv",
    "journey_truth.csv",
    "attribution_results.csv",
    "model_comparison.csv",
    "channel_truth_summary.csv",
]
