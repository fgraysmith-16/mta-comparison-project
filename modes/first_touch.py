"""
First-touch attribution model.

All credit allocated to the first channel in the customer journey.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

from simulate_journeys import parse_touches


def apply_first_touch(
    journeys_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply first-touch attribution to journeys.
    
    Args:
        journeys_df: DataFrame from generate_journeys with columns:
            - journey_id
            - revenue
            - touches_json
            
    Returns:
        pd.DataFrame with columns:
            - journey_id
            - channel
            - attributed_revenue: Revenue allocated to this channel (0 for non-first)
            - attributed_conversions: 1 if this channel is first-touch, else 0
    """
    rows = []
    
    for _, journey in journeys_df.iterrows():
        journey_id = journey["journey_id"]
        revenue = journey["revenue"]
        
        # Parse touches
        touches = parse_touches(journey["touches_json"])
        
        if not touches:
            continue
        
        # First touch channel
        first_channel = touches[0][1]
        
        # All channels get 0, except first which gets 100%
        for _, channel in touches:
            rows.append({
                "journey_id": journey_id,
                "channel": channel,
                "attributed_revenue": revenue if channel == first_channel else 0.0,
                "attributed_conversions": 1.0 if channel == first_channel else 0.0,
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
    
    print("Applying first-touch attribution...")
    result = apply_first_touch(journeys)
    
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
    
    print("\nFirst-touch attribution summary:")
    print(summary.to_string(index=False))
