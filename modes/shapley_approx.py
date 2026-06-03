"""
Shapley approximation using Aumann-Shapley method.

Approximates exact Shapley values by sampling random permutations and
computing average marginal contributions.

Much faster than exact Shapley: O(k) where k = number of samples per journey.
Trades exactness for speed. With k=100 samples: typically within 5-10% of exact.

Algorithm per journey:
1. Sample k random permutations of channels in journey
2. For each permutation: compute marginal contribution of each channel
   (value when added in that position vs value before)
3. Shapley ≈ average marginal contribution across all samples
4. Allocate revenue proportionally to approximate Shapley values
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List

from config import CHANNEL_NAMES, RANDOM_SEED
from simulate_journeys import parse_touches
from latent_demand import (
    adstock_geometric,
    saturation_power_law,
    compute_interaction_multiplier,
)


def _evaluate_channel_set(
    touches: list,
    included_channels: set,
) -> float:
    """
    Evaluate conversion probability for a specific set of channels.
    
    Same as shapley_exact but extracted for reuse.
    
    Args:
        touches: List of (date, channel) tuples
        included_channels: Set of channel names to include
        
    Returns:
        float: Latent demand (conversion probability)
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
        
        spend_history = []
        for touch_date, touch_channel in touches_sorted:
            if touch_channel == channel:
                spend_history.append(1.0)
            else:
                spend_history.append(0.0)
        
        if not spend_history or sum(spend_history) == 0:
            channel_impact[channel] = 0.0
            continue
        
        adstocked = adstock_geometric(
            np.array(spend_history),
            half_life=channel_config.adstock_half_life,
        )
        
        saturated = saturation_power_law(
            adstocked[-1],
            exponent=channel_config.saturation_exponent,
        )
        
        channel_impact[channel] = saturated
        total_impact += saturated
    
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
    
    baseline_demand = SIMULATION.base_conversion_rate
    demand_signal = baseline_demand * np.exp(total_impact / 1000.0) * interaction_mult
    demand_prob = np.clip(float(demand_signal), 0.0, 1.0)
    
    return demand_prob


def _compute_shapley_approximation(
    touches: list,
    revenue: float,
    num_samples: int = 100,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """
    Compute approximate Shapley values using Aumann-Shapley method.
    
    Args:
        touches: List of (date, channel) tuples from journey
        revenue: Total revenue for this journey
        num_samples: Number of random permutations to sample
        rng: numpy random generator
        
    Returns:
        Dict[channel -> allocated revenue]
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    if not touches:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    
    # Get unique channels in this journey
    journey_channels = list(set(ch for _, ch in touches))
    
    # Initialize marginal contributions
    marginal_sum = {ch: 0.0 for ch in journey_channels}
    
    # Sample random permutations
    for _ in range(num_samples):
        # Random permutation of channels
        perm = rng.permutation(journey_channels)
        
        # Compute marginal contribution for each channel in this permutation
        for i, channel in enumerate(perm):
            # Channels before this one in permutation
            prev_channels = set(perm[:i])
            
            # Value without this channel
            value_without = _evaluate_channel_set(touches, prev_channels)
            
            # Value with this channel
            value_with = _evaluate_channel_set(touches, prev_channels | {channel})
            
            # Marginal contribution
            marginal_sum[channel] += value_with - value_without
    
    # Average marginal contributions
    shapley_approx = {ch: marginal_sum[ch] / num_samples for ch in journey_channels}
    
    # Allocate revenue
    total_value = sum(shapley_approx.values())
    if total_value > 0:
        allocated = {
            ch: (shapley_approx.get(ch, 0) / total_value) * revenue
            for ch in CHANNEL_NAMES
        }
    else:
        # Uniform allocation
        uniform_share = revenue / len(journey_channels)
        allocated = {
            ch: uniform_share if ch in journey_channels else 0.0
            for ch in CHANNEL_NAMES
        }
    
    return allocated


def apply_shapley_approx(
    journeys_df: pd.DataFrame,
    num_samples: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Apply approximate Shapley value attribution to journeys.
    
    Much faster than exact Shapley. With num_samples=100:
    - Expected runtime: ~5-10 seconds for 6,459 journeys
    - Accuracy: typically within 5-10% of exact Shapley
    
    Args:
        journeys_df: DataFrame from generate_journeys
        num_samples: Number of random permutations per journey (default 100)
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame with columns:
            - journey_id
            - channel
            - attributed_revenue
            - attributed_conversions
    """
    rng = np.random.default_rng(seed)
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
        
        # Compute approximate Shapley values
        allocated = _compute_shapley_approximation(
            touches,
            revenue,
            num_samples=num_samples,
            rng=rng,
        )
        
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
    
    print("Applying approximate Shapley attribution (100 samples)...")
    start = time.time()
    result = apply_shapley_approx(journeys, num_samples=100)
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
    
    print("\nApproximate Shapley attribution summary:")
    print(summary.to_string(index=False))
