"""
Markov removal-effect attribution model.

Uses a removal-effect approach within a Markov chain framework.
For each channel, measures its contribution by removing it from the journey
and computing how much the conversion probability decreases.

Algorithm:
1. For each journey, evaluate conversion probability with all channels
2. For each channel in the journey, remove it and re-evaluate
3. Channel contribution = (prob_full - prob_without) / sum_of_all_removals
4. Allocate revenue proportionally

Complexity: O(# unique channels) per journey → ~6-10 evaluations per journey
Expected runtime: ~1-2 seconds for 6,459 journeys

This is similar to the ground-truth removal-effect but applied to channel attribution
instead of measuring true contribution—tests whether Markov structure helps.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict

from config import CHANNEL_NAMES, RANDOM_SEED
from simulate_journeys import parse_touches
from latent_demand import (
    adstock_geometric,
    saturation_power_law,
    compute_interaction_multiplier,
)


def _evaluate_channel_subset(
    touches: list,
    included_channels: set,
) -> float:
    """
    Evaluate conversion probability for a subset of channels in a journey.
    
    Reuses latent demand logic: applies adstock, saturation, interactions.
    
    Args:
        touches: List of (date, channel) tuples
        included_channels: Set of channel names to include in evaluation
        
    Returns:
        float: Estimated conversion probability [0, 1]
    """
    if not touches or not included_channels:
        return 0.0
    
    from config import CHANNEL_BY_NAME, SIMULATION
    
    touches_sorted = sorted(touches, key=lambda x: x[0])
    
    channel_impact = {}
    total_impact = 0.0
    
    for channel in CHANNEL_NAMES:
        if channel not in included_channels:
            continue
        
        channel_config = CHANNEL_BY_NAME[channel]
        
        # Build spend history for this channel (1.0 if touched, 0.0 otherwise)
        spend_history = []
        for touch_date, touch_channel in touches_sorted:
            if touch_channel == channel:
                spend_history.append(1.0)
            else:
                spend_history.append(0.0)
        
        if not spend_history or sum(spend_history) == 0:
            channel_impact[channel] = 0.0
            continue
        
        # Apply adstock decay
        adstocked = adstock_geometric(
            np.array(spend_history),
            half_life=channel_config.adstock_half_life,
        )
        
        # Apply saturation
        saturated = saturation_power_law(
            adstocked[-1],
            exponent=channel_config.saturation_exponent,
        )
        
        channel_impact[channel] = saturated
        total_impact += saturated
    
    # Apply interaction effects
    config_interactions = {
        ch: CHANNEL_BY_NAME[ch].interaction_effect
        for ch in CHANNEL_NAMES
        if ch in included_channels
    }
    interaction_mult = compute_interaction_multiplier(
        channel_impact,
        config_interactions,
    )
    
    # Compute conversion probability
    baseline_demand = SIMULATION.base_conversion_rate
    demand_signal = baseline_demand * np.exp(total_impact / 1000.0) * interaction_mult
    demand_prob = np.clip(float(demand_signal), 0.0, 1.0)
    
    return demand_prob


def _compute_markov_removal_effect(
    touches: list,
    revenue: float,
) -> Dict[str, float]:
    """
    Compute Markov removal-effect attribution for a journey.
    
    Args:
        touches: List of (date, channel) tuples
        revenue: Total revenue for this journey
        
    Returns:
        Dict[channel -> allocated revenue]
    """
    if not touches:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    
    # Get unique channels in this journey
    journey_channels = set(ch for _, ch in touches)
    
    # Evaluate baseline: all channels present
    baseline_value = _evaluate_channel_subset(touches, journey_channels)
    
    # Evaluate removal effect for each channel
    removal_effects = {}
    total_removal_effect = 0.0
    
    for channel in journey_channels:
        # Remove this channel and re-evaluate
        channels_without = journey_channels - {channel}
        value_without = _evaluate_channel_subset(touches, channels_without)
        
        # Removal effect = how much does removing this channel hurt conversion?
        removal_effect = max(0.0, baseline_value - value_without)
        removal_effects[channel] = removal_effect
        total_removal_effect += removal_effect
    
    # Allocate revenue based on removal effects
    if total_removal_effect > 0:
        allocated = {
            ch: (removal_effects.get(ch, 0) / total_removal_effect) * revenue
            for ch in CHANNEL_NAMES
        }
    else:
        # If all channels have zero removal effect, allocate uniformly
        uniform_share = revenue / len(journey_channels)
        allocated = {
            ch: uniform_share if ch in journey_channels else 0.0
            for ch in CHANNEL_NAMES
        }
    
    return allocated


def apply_markov_rigorous(journeys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Markov removal-effect attribution to journeys.
    
    Fast and interpretable: tests whether Markov structure (via removal effect)
    provides better attribution than simpler methods.
    
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
        
        # Parse touches from JSON
        touches = parse_touches(journey["touches_json"])
        
        if not touches:
            continue
        
        # Compute Markov removal-effect attribution
        allocated = _compute_markov_removal_effect(touches, revenue)
        
        # Record results
        journey_channels = set(ch for _, ch in touches)
        for channel, attr_revenue in allocated.items():
            conv_weight = 1.0 / len(journey_channels) if journey_channels else 0.0
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
    
    print("Applying Markov rigorous attribution...")
    start = time.time()
    result = apply_markov_rigorous(journeys)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f}s")
    print(f"\nAttributed revenue by channel:")
    by_channel = result.groupby("channel")["attributed_revenue"].sum()
    for ch, rev in by_channel.items():
        print(f"  {ch:20s}: ${rev:>12,.2f}")
    print(f"  {'TOTAL':20s}: ${by_channel.sum():>12,.2f}")
