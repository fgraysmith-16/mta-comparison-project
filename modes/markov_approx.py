"""
Markov approximation using first-order bigram transitions.

Approximates channel contribution by analyzing transitions in the journey.

Algorithm:
1. For each journey, extract all adjacent channel pairs (bigrams)
2. Count transitions: START→channel, channel→channel, channel→CONVERSION
3. Score each channel by how frequently it transitions to conversions
4. Additionally, give credit for transitioning to "strong" downstream channels
5. Allocate revenue proportionally to transition scores

Complexity: O(N) where N = journey length ≈ 1-8 evaluations per journey
Expected runtime: <1 second for all journeys (fastest model)

This tests whether simple transition analysis can approximate attribution
without expensive removal-effect or coalition enumeration.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict

from config import CHANNEL_NAMES, RANDOM_SEED
from simulate_journeys import parse_touches


def _compute_markov_approximation_bigram(
    touches: list,
    revenue: float,
) -> Dict[str, float]:
    """
    Compute approximate Markov attribution using first-order bigram transitions.
    
    Args:
        touches: List of (date, channel) tuples
        revenue: Total revenue for this journey
        
    Returns:
        Dict[channel -> allocated revenue]
    """
    if not touches:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    
    # Sort touches by date
    touches_sorted = sorted(touches, key=lambda x: x[0])
    
    # Extract channel sequence (dates don't matter for bigrams)
    channel_sequence = [ch for _, ch in touches_sorted]
    
    if not channel_sequence:
        return {ch: 0.0 for ch in CHANNEL_NAMES}
    
    # Count transitions
    transition_scores = defaultdict(float)
    
    # START → first channel (give credit for initiating)
    first_channel = channel_sequence[0]
    transition_scores[first_channel] += 0.5
    
    # Adjacent pairs (bigrams)
    for i in range(len(channel_sequence) - 1):
        current_ch = channel_sequence[i]
        next_ch = channel_sequence[i + 1]
        
        # Both channels get credit: current for transitioning to next,
        # and next for being transitioned to
        transition_scores[current_ch] += 0.25
        transition_scores[next_ch] += 0.25
    
    # Last channel → CONVERSION (give credit for closing)
    last_channel = channel_sequence[-1]
    transition_scores[last_channel] += 1.0
    
    # Normalize and allocate
    total_score = sum(transition_scores.values())
    
    if total_score > 0:
        allocated = {
            ch: (transition_scores.get(ch, 0) / total_score) * revenue
            for ch in CHANNEL_NAMES
        }
    else:
        # Uniform allocation (shouldn't happen)
        unique_channels = set(channel_sequence)
        uniform_share = revenue / len(unique_channels) if unique_channels else 0
        allocated = {
            ch: uniform_share if ch in unique_channels else 0.0
            for ch in CHANNEL_NAMES
        }
    
    return allocated


def apply_markov_approx(journeys_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply approximate Markov bigram attribution to journeys.
    
    Fastest model: uses only first-order transition analysis.
    Tests whether simple bigram structure can approximate channel contribution.
    
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
        
        # Compute Markov bigram approximation
        allocated = _compute_markov_approximation_bigram(touches, revenue)
        
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
    import time
    from simulate_spend import simulate_daily_spend
    from latent_demand import generate_daily_latent_demand
    from simulate_journeys import generate_journeys
    
    print("Generating test data...")
    daily_spend = simulate_daily_spend()
    daily_demand = generate_daily_latent_demand(daily_spend)
    journeys = generate_journeys(daily_spend, daily_demand)
    
    print("Applying Markov approximation bigram attribution...")
    start = time.time()
    result = apply_markov_approx(journeys)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed:.2f}s")
    print(f"\nAttributed revenue by channel:")
    by_channel = result.groupby("channel")["attributed_revenue"].sum()
    for ch, rev in by_channel.items():
        print(f"  {ch:20s}: ${rev:>12,.2f}")
    print(f"  {'TOTAL':20s}: ${by_channel.sum():>12,.2f}")
