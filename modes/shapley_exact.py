"""
Exact Shapley value attribution model.

Computes exact Shapley values by evaluating a channel's marginal contribution
across all possible coalitions (subsets of channels).

For each journey:
- Evaluate model with all channels
- For each channel: measure its marginal contribution in all possible coalitions
- Average contributions = exact Shapley value

Theoretically optimal but computationally expensive: O(2^N) coalitions where N = #channels.
With 6 channels: 64 coalitions per journey. For 6,459 journeys: feasible but slow.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple
from itertools import combinations

from config import CHANNEL_NAMES
from simulate_journeys import parse_touches
from latent_demand import (
    adstock_geometric,
    saturation_power_law,
    compute_interaction_multiplier,
)


def _evaluate_coalition(
    touches: list,
    included_channels: set,
) -> float:
    """
    Evaluate conversion probability for a specific coalition of channels.
    
    For a given set of channels, simulate the journey using only those channels
    and compute the resulting latent demand signal.
    
    Args:
        touches: List of (date, channel) tuples
        included_channels: Set of channel names to include in evaluation
        
    Returns:
        float: Latent demand (conversion probability) for this coalition
    """
    if not touches or not included_channels:
        return 0.0
    
    # Build demand signal for this coalition
    from config import CHANNEL_BY_NAME, SIMULATION
    
    touches_sorted = sorted(touches, key=lambda x: x[0])
    
    channel_impact = {}
    total_impact = 0.0
    
    for channel in CHANNEL_NAMES:
        if channel not in included_channels:
            continue
        
        channel_config = CHANNEL_BY_NAME[channel]
        
        # Build spend history for this channel
        spend_history = []
        for touch_date, touch_channel in touches_sorted:
            if touch_channel == channel:
                spend_history.append(1.0)
            else:
                spend_history.append(0.0)
        
        if not spend_history or sum(spend_history) == 0:
            channel_impact[channel] = 0.0
            continue
        
        # Apply adstock
        adstocked = adstock_geometric(
            np.array(spend_history),
            half_life=channel_config.adstock_half_life,
        )
        
        adstocked_final = adstocked[-1]
        
        # Apply saturation
        saturated = saturation_power_law(
            adstocked_final,
            exponent=channel_config.saturation_exponent,
        )
        
        channel_impact[channel] = saturated
        total_impact += saturated
    
    # Compute interaction multiplier
    from config import CHANNEL_BY_NAME
    config_interactions = {
        ch: CHANNEL_BY_NAME[ch].interaction_effect
        for ch in CHANNEL_NAMES
        if ch in included_channels
    }
    interaction_mult = compute_interaction_multiplier(
        channel_impact,
        config_interactions,
    )
    
    # Convert to probability
    baseline_demand = SIMULATION.base_conversion_rate
    demand_signal = baseline_demand * np.exp(total_impact / 1000.0) * interaction_mult
    demand_prob = np.clip(float(demand_signal), 0.0, 1.0)
    
    return demand_prob


def _compute_shapley_values(
    touches: list,
    revenue: float,
) -> Dict[str, float]:
    """
    Compute exact Shapley values for a single journey.
    
    Algorithm:
    1. For each channel C:
       - For each possible coalition S not containing C:
         - Compute value of S
         - Compute value of S ∪ {C}
         - Marginal contribution = value(S ∪ C) - value(S)
       - Shapley[C] = average marginal contribution
    2. Allocate revenue proportionally to Shapley values
    
    Args:
        touches: List of (date, channel) tuples from journey
        revenue: Total revenue for this journey
        
    Returns:
        Dict[channel -> allocated revenue]
    """
    if not touches:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    
    # Get unique channels in this journey
    journey_channels = set(ch for _, ch in touches)
    
    shapley_values = {}
    
    # For each channel in the journey
    for target_channel in journey_channels:
        other_channels = journey_channels - {target_channel}
        
        marginal_contributions = []
        
        # For each coalition of other channels (2^|other_channels| coalitions)
        for r in range(len(other_channels) + 1):
            for coalition in combinations(other_channels, r):
                coalition_set = set(coalition)
                
                # Value without target channel
                value_without = _evaluate_coalition(touches, coalition_set)
                
                # Value with target channel
                value_with = _evaluate_coalition(touches, coalition_set | {target_channel})
                
                # Marginal contribution
                marginal = value_with - value_without
                marginal_contributions.append(marginal)
        
        # Shapley value = average marginal contribution
        shapley_values[target_channel] = np.mean(marginal_contributions)
    
    # Channels not in journey get 0
    for ch in CHANNEL_NAMES:
        if ch not in shapley_values:
            shapley_values[ch] = 0.0
    
    # Normalize and allocate revenue
    total_value = sum(shapley_values.values())
    if total_value > 0:
        allocated = {ch: (v / total_value) * revenue for ch, v in shapley_values.items()}
    else:
        # Uniform allocation if no value
        uniform_share = revenue / len([ch for ch in journey_channels])
        allocated = {
            ch: uniform_share if ch in journey_channels else 0.0
            for ch in CHANNEL_NAMES
        }
    
    return allocated


def apply_shapley_exact(
    journeys_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply exact Shapley value attribution to journeys.
    
    WARNING: This is computationally expensive. For 6 channels and 6,459 journeys,
    expect ~30-60 seconds runtime. For larger datasets, use shapley_approx instead.
    
    Args:
        journeys_df: DataFrame from generate_journeys
        
    Returns:
        pd.DataFrame with columns:
            - journey_id
            - channel
            - attributed_revenue
            - attributed_conversions
    """
    rows = []
    
    for idx, journey in journeys_df.iterrows():
        if idx % 500 == 0:
            print(f"    Processing journey {idx}/{len(journeys_df)}...", end="\r", flush=True)
        
        journey_id = journey["journey_id"]
        revenue = journey["revenue"]
        
        # Parse touches
        touches = parse_touches(journey["touches_json"])
        
        if not touches:
            continue
        
        # Compute Shapley values
        allocated = _compute_shapley_values(touches, revenue)
        
        # Record results
        for channel, attr_revenue in allocated.items():
            conv_weight = 1.0 / len(set(ch for _, ch in touches)) if touches else 0.0
            rows.append({
                "journey_id": journey_id,
                "channel": channel,
                "attributed_revenue": attr_revenue,
                "attributed_conversions": conv_weight if attr_revenue > 0 else 0.0,
            })
    
    print("\n", end="")  # Clear progress line
    result_df = pd.DataFrame(rows)
    return result_df


if __name__ == "__main__":
    from simulate_spend import simulate_daily_spend
    from latent_demand import generate_daily_latent_demand
    from simulate_journeys import generate_journeys
    
    print("Generating test data...")
    daily_spend = simulate_daily_spend()
    daily_demand = generate_daily_latent_demand(daily_spend)
    journeys = generate_journeys(daily_spend, daily_demand)
    
    print("Applying exact Shapley attribution...")
    start = time.time()
    result = apply_shapley_exact(journeys)
    elapsed = time.time() - start
    
    print(f"Runtime: {elapsed:.2f}s")
    
    summary = (
        result[result["attributed_revenue"] > 0]
        .groupby("channel")
        .agg(
            total_revenue=("attributed_revenue", "sum"),
            num_conversions=("attributed_conversions", "sum"),
        )
        .reset_index()
    )
    summary["pct_of_total"] = 100.0 * summary["total_revenue"] / summary["total_revenue"].sum()
    
    print("\nExact Shapley attribution summary:")
    print(summary.to_string(index=False))
