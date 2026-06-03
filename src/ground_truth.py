"""
Ground-truth channel contribution calculation.

For each journey, calculates the TRUE channel contribution by applying
the exact same adstock/saturation/interaction logic used in demand generation.

This ground truth is the benchmark all attribution models will be evaluated against.

Key principle: A channel's true contribution is its marginal impact on conversion
probability, weighted by the revenue from that conversion.

Approach: Removal-effect
1. Calculate conversion probability WITH all channels
2. For each channel: calculate conversion probability WITHOUT that channel
3. The difference is that channel's true impact
4. Allocate revenue proportionally to channel impacts
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import json

from config import (
    CHANNEL_BY_NAME,
    CHANNEL_NAMES,
    SIMULATION,
    RANDOM_SEED,
)
from latent_demand import (
    adstock_geometric,
    saturation_power_law,
    compute_interaction_multiplier,
)
from simulate_journeys import parse_touches


def _calculate_channel_contributions(
    touches: List[Tuple[datetime, str]],
) -> Dict[str, float]:
    """
    Calculate removal-effect contribution for each channel in a journey.
    
    For each channel:
    1. Create a spend signal from the journey touches
    2. Calculate demand WITH that channel
    3. Calculate demand WITHOUT that channel
    4. Difference = channel's contribution
    
    Returns normalized contributions (sum to 1.0).
    
    Args:
        touches: List of (date, channel) tuples for this journey
        
    Returns:
        Dict[channel_name -> contribution (0 to 1)]
    """
    if not touches:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    
    # Create a sparse spend signal from touches
    # Map each touch to a spend amount (e.g., $1 per touch for simplicity)
    touch_spend_by_date = {}
    for touch_date, channel in touches:
        if touch_date not in touch_spend_by_date:
            touch_spend_by_date[touch_date] = {}
        if channel not in touch_spend_by_date[touch_date]:
            touch_spend_by_date[touch_date][channel] = 0
        touch_spend_by_date[touch_date][channel] += 1.0  # 1 unit of spend per touch
    
    # Sort by date
    sorted_dates = sorted(touch_spend_by_date.keys())
    
    # Calculate baseline demand (with all channels)
    baseline_demand = _calculate_demand_from_touches(
        sorted_dates,
        touch_spend_by_date,
        exclude_channel=None,
    )
    
    contributions = {}
    
    # For each channel, calculate impact of removing it
    for channel in CHANNEL_NAMES:
        demand_without = _calculate_demand_from_touches(
            sorted_dates,
            touch_spend_by_date,
            exclude_channel=channel,
        )
        
        # Channel's impact = difference in demand
        impact = max(0, baseline_demand - demand_without)
        contributions[channel] = impact
    
    # Normalize to 0-1
    total_impact = sum(contributions.values())
    if total_impact > 0:
        contributions = {ch: v / total_impact for ch, v in contributions.items()}
    else:
        # If no impact, uniform distribution
        contributions = {ch: 1.0 / len(CHANNEL_NAMES) for ch in CHANNEL_NAMES}
    
    return contributions


def _calculate_demand_from_touches(
    sorted_dates: List[datetime],
    touch_spend_by_date: Dict[datetime, Dict[str, float]],
    exclude_channel: str = None,
) -> float:
    """
    Calculate latent demand given a set of touches (possibly excluding one channel).
    
    Uses same logic as latent_demand.py:
    - Adstock with channel-specific half-lives
    - Saturation with channel-specific exponents
    - Pairwise interactions
    
    Args:
        sorted_dates: Sorted list of touch dates
        touch_spend_by_date: Dict[date -> Dict[channel -> spend_amount]]
        exclude_channel: Channel to exclude from calculation (if any)
        
    Returns:
        float: Demand signal (0 to 1 range typically)
    """
    channel_impact = {}
    total_impact = 0.0
    
    # For each channel, accumulate its adstocked+saturated contribution
    for channel in CHANNEL_NAMES:
        if channel == exclude_channel:
            continue
        
        channel_config = CHANNEL_BY_NAME[channel]
        
        # Build spend history for this channel across all touch dates
        spend_history = []
        for date in sorted_dates:
            spend = touch_spend_by_date.get(date, {}).get(channel, 0.0)
            spend_history.append(spend)
        
        if not spend_history or sum(spend_history) == 0:
            channel_impact[channel] = 0.0
            continue
        
        # Apply adstock
        adstocked = adstock_geometric(
            np.array(spend_history),
            half_life=channel_config.adstock_half_life,
        )
        
        # Get final adstocked value
        adstocked_final = adstocked[-1]
        
        # Apply saturation
        saturated = saturation_power_law(
            adstocked_final,
            exponent=channel_config.saturation_exponent,
        )
        
        channel_impact[channel] = saturated
        total_impact += saturated
    
    # Compute interaction multiplier
    config_interactions = {
        ch: CHANNEL_BY_NAME[ch].interaction_effect
        for ch in CHANNEL_NAMES
        if ch != exclude_channel
    }
    interaction_mult = compute_interaction_multiplier(
        channel_impact,
        config_interactions,
    )
    
    # Convert to demand probability
    baseline_demand = SIMULATION.base_conversion_rate
    demand_signal = baseline_demand * np.exp(total_impact / 1000.0) * interaction_mult
    demand_prob = np.clip(float(demand_signal), 0.0, 1.0)
    
    return demand_prob


def calculate_journey_truth(
    journey_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Calculate ground-truth channel contributions for all journeys.
    
    Args:
        journey_df: DataFrame from generate_journeys with columns:
            - journey_id
            - revenue
            - touches_json
            
    Returns:
        pd.DataFrame with columns:
            - journey_id
            - channel
            - true_contribution_pct: Channel's % of total revenue (0-100)
            - true_contribution_revenue: Channel's revenue allocation ($)
    """
    np.random.seed(seed)
    
    rows = []
    
    for _, journey in journey_df.iterrows():
        journey_id = journey["journey_id"]
        revenue = journey["revenue"]
        
        # Parse touches
        touches = parse_touches(journey["touches_json"])
        
        # Calculate channel contributions
        contributions = _calculate_channel_contributions(touches)
        
        # Allocate revenue proportionally
        for channel, contrib_pct in contributions.items():
            allocated_revenue = contrib_pct * revenue
            
            rows.append({
                "journey_id": journey_id,
                "channel": channel,
                "true_contribution_pct": contrib_pct * 100.0,
                "true_contribution_revenue": allocated_revenue,
            })
    
    truth_df = pd.DataFrame(rows)
    return truth_df


def aggregate_channel_truth(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ground-truth contributions by channel.
    
    Args:
        truth_df: DataFrame from calculate_journey_truth
        
    Returns:
        pd.DataFrame with columns:
            - channel
            - total_revenue: Total revenue attributed to this channel
            - num_journeys: Number of journeys this channel participated in
            - avg_contribution_pct: Average contribution % when channel participates
    """
    agg = (
        truth_df[truth_df["true_contribution_revenue"] > 0]
        .groupby("channel")
        .agg(
            total_revenue=("true_contribution_revenue", "sum"),
            num_journeys=("journey_id", "count"),
            avg_contribution_pct=("true_contribution_pct", "mean"),
        )
        .reset_index()
    )
    
    # Add total for reference
    agg["pct_of_total"] = 100.0 * agg["total_revenue"] / agg["total_revenue"].sum()
    
    return agg


if __name__ == "__main__":
    # Quick smoke test
    from simulate_spend import simulate_daily_spend
    from latent_demand import generate_daily_latent_demand
    from simulate_journeys import generate_journeys
    
    print("Generating spend, demand, and journeys...")
    daily_spend = simulate_daily_spend()
    daily_demand = generate_daily_latent_demand(daily_spend)
    journeys = generate_journeys(daily_spend, daily_demand)
    
    print("Computing ground-truth channel contributions...")
    truth = calculate_journey_truth(journeys)
    
    print(f"\nGenerated {len(truth)} journey-channel pairs")
    
    summary = aggregate_channel_truth(truth)
    print(f"\nGround-truth channel summary:")
    print(summary[["channel", "total_revenue", "num_journeys", "pct_of_total"]].to_string(index=False))
    
    print(f"\nTotal revenue: ${truth['true_contribution_revenue'].sum():,.2f}")
    print(f"Total journeys: {len(journeys)}")
    
    # Show sample journey's truth
    if len(journeys) > 0:
        sample_id = journeys.iloc[0]["journey_id"]
        sample_truth = truth[truth["journey_id"] == sample_id]
        print(f"\nSample journey {sample_id} truth allocation:")
        for _, row in sample_truth.iterrows():
            if row["true_contribution_revenue"] > 0:
                print(f"  {row['channel']:20s}: ${row['true_contribution_revenue']:>8.2f} ({row['true_contribution_pct']:>5.1f}%)")
