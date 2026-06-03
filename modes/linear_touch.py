"""
Linear multi-touch attribution model.

Credit equally distributed across all channels in the customer journey.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from simulate_journeys import parse_touches


def apply_linear_touch(
    journeys_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply linear (equal-weight) multi-touch attribution to journeys.
    
    Each channel in the journey receives equal credit, even if it appears
    multiple times.
    
    Args:
        journeys_df: DataFrame from generate_journeys with columns:
            - journey_id
            - revenue
            - touches_json
            
    Returns:
        pd.DataFrame with columns:
            - journey_id
            - channel
            - attributed_revenue: Revenue allocated to this channel
            - attributed_conversions: Conversion count for this channel
    """
    rows = []
    
    for _, journey in journeys_df.iterrows():
        journey_id = journey["journey_id"]
        revenue = journey["revenue"]
        
        # Parse touches
        touches = parse_touches(journey["touches_json"])
        
        if not touches:
            continue
        
        # Linear: divide revenue equally by number of touches
        credit_per_touch = revenue / len(touches)
        conv_per_touch = 1.0 / len(touches)
        
        # Track which channels we've already added (to avoid duplicating touches)
        channel_touches = {}
        for _, channel in touches:
            if channel not in channel_touches:
                channel_touches[channel] = 0
            channel_touches[channel] += 1
        
        # Add one row per unique channel in journey
        for channel, num_touches in channel_touches.items():
            rows.append({
                "journey_id": journey_id,
                "channel": channel,
                "attributed_revenue": credit_per_touch * num_touches,
                "attributed_conversions": conv_per_touch * num_touches,
            })
    
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
    
    print("Applying linear multi-touch attribution...")
    result = apply_linear_touch(journeys)
    
    summary = (
        result
        .groupby("channel")
        .agg(
            total_revenue=("attributed_revenue", "sum"),
            num_conversions=("attributed_conversions", "sum"),
        )
        .reset_index()
    )
    summary["pct_of_total"] = 100.0 * summary["total_revenue"] / summary["total_revenue"].sum()
    
    print("\nLinear multi-touch attribution summary:")
    print(summary.to_string(index=False))
