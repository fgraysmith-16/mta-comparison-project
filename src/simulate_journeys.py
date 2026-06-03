"""
User journey simulation - multi-touch paths leading to conversion.

Generates realistic customer journeys with:
- Stochastic number of touches (1-8 typical)
- Channel mix reflecting spend and upper/lower funnel differences
- Temporal spread over multiple days
- Conversions sampled from daily latent demand

Ground truth is preserved: we track exact touch sequence so we can later
calculate true channel contribution using removal-effect logic.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

from config import (
    CHANNELS,
    CHANNEL_BY_NAME,
    CHANNEL_NAMES,
    SIMULATION,
    JOURNEY,
    RANDOM_SEED,
)


# ============================================================================
# Channel Classification for Realistic Path Generation
# ============================================================================

UPPER_FUNNEL_CHANNELS = {"Display", "Video"}
LOWER_FUNNEL_CHANNELS = {"Paid Search", "Email"}
MID_FUNNEL_CHANNELS = {"Paid Social"}
ORGANIC_CHANNELS = {"Affiliate"}

# When generating a path, we bias towards realistic sequences:
# - Upper funnel comes early
# - Lower funnel comes late
# - Organic can appear anytime


def _get_daily_spend_weights(daily_spend_df: pd.DataFrame, date: datetime) -> Dict[str, float]:
    """
    Get normalized spend weights for a specific date.
    
    These weights guide channel selection in journeys - channels with higher
    spend are more likely to appear in user touches.
    
    Args:
        daily_spend_df: DataFrame from simulate_daily_spend
        date: Target date
        
    Returns:
        Dict[channel_name -> probability weight (0 to 1)]
    """
    day_spend = daily_spend_df[daily_spend_df["date"] == date]
    
    if len(day_spend) == 0:
        # Fallback: uniform weights
        return {ch: 1.0 / len(CHANNEL_NAMES) for ch in CHANNEL_NAMES}
    
    spend_by_channel = dict(zip(day_spend["channel"], day_spend["spend"]))
    total = sum(spend_by_channel.values())
    
    if total == 0:
        return {ch: 1.0 / len(CHANNEL_NAMES) for ch in CHANNEL_NAMES}
    
    return {ch: spend_by_channel.get(ch, 0) / total for ch in CHANNEL_NAMES}


def _sample_channel(
    rng: np.random.Generator,
    weights: Dict[str, float],
    exclude_channels: set = None,
    position_in_journey: int = 0,
    max_positions: int = 5,
) -> str:
    """
    Sample a channel for a journey touch, respecting realistic path structure.
    
    Earlier positions more likely to be upper-funnel.
    Later positions more likely to be lower-funnel.
    
    Args:
        rng: numpy random generator
        weights: spend weights by channel
        exclude_channels: channels to exclude (avoid duplicates)
        position_in_journey: 0-indexed position in path
        max_positions: maximum number of touches expected
        
    Returns:
        str: Selected channel name
    """
    exclude = exclude_channels or set()
    
    # Modify weights based on position
    modified_weights = {}
    progress = position_in_journey / max_positions if max_positions > 0 else 0
    
    for ch in CHANNEL_NAMES:
        if ch in exclude:
            modified_weights[ch] = 0.0
        else:
            base_weight = weights.get(ch, 0)
            
            # Position bias
            if ch in UPPER_FUNNEL_CHANNELS and progress < 0.4:
                # Boost upper funnel early
                modified_weights[ch] = base_weight * 1.5
            elif ch in LOWER_FUNNEL_CHANNELS and progress > 0.5:
                # Boost lower funnel late
                modified_weights[ch] = base_weight * 1.3
            else:
                modified_weights[ch] = base_weight
    
    # Normalize and sample
    total = sum(modified_weights.values())
    if total == 0:
        return rng.choice(CHANNEL_NAMES)
    
    probs = [modified_weights.get(ch, 0) / total for ch in CHANNEL_NAMES]
    return rng.choice(CHANNEL_NAMES, p=probs)


def generate_journeys(
    daily_spend_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate user-level multi-touch journeys.
    
    For each day in the simulation, sample conversions based on latent demand,
    then generate a realistic multi-touch path leading to that conversion.
    
    Args:
        daily_spend_df: DataFrame from simulate_daily_spend
        daily_demand_df: DataFrame from generate_daily_latent_demand
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame with columns:
            - journey_id: unique identifier (int)
            - user_id: user identifier (int)
            - conversion_date: datetime
            - revenue: float (order value)
            - num_touches: int
            - touches_json: str (JSON serialized list of (date, channel) tuples)
    """
    rng = np.random.default_rng(seed)
    
    # Merge spend and demand
    merged = daily_demand_df.merge(
        daily_spend_df.pivot_table(
            index="date",
            columns="channel",
            values="spend",
            fill_value=0,
        ),
        left_on="date",
        right_index=True,
    )
    
    journeys = []
    journey_id = 0
    user_id = 0
    
    start_date = datetime.strptime(SIMULATION.start_date, "%Y-%m-%d")
    
    for _, row in merged.iterrows():
        date = row["date"]
        demand_prob = row["latent_demand"]
        
        # Sample how many conversions happen on this day
        num_conversions = rng.poisson(demand_prob * SIMULATION.num_users / 100)
        
        for _ in range(num_conversions):
            user_id += 1
            
            # Generate journey length
            journey_length = rng.integers(JOURNEY.min_touches, JOURNEY.max_touches + 1)
            
            # Generate touch dates (going backwards from conversion date)
            touch_dates = []
            current_date = date
            for touch_idx in range(journey_length):
                days_back = rng.geometric(0.3)  # Exponential spacing (more recent is more likely)
                touch_date = current_date - timedelta(days=days_back)
                
                # Don't go before simulation start
                if touch_date < start_date:
                    touch_date = start_date
                
                touch_dates.append(touch_date)
            
            # Sort by date (ascending)
            touch_dates.sort()
            
            # Sample channels for each touch
            touches = []
            used_channels = set()
            
            for touch_idx, touch_date in enumerate(touch_dates):
                # Get spend weights for this date
                weights = _get_daily_spend_weights(daily_spend_df, touch_date)
                
                # Sample channel (with position bias)
                channel = _sample_channel(
                    rng,
                    weights,
                    exclude_channels=None,  # Allow repeats
                    position_in_journey=touch_idx,
                    max_positions=journey_length,
                )
                touches.append((touch_date, channel))
            
            # Generate revenue from order value distribution
            revenue = rng.normal(
                SIMULATION.avg_order_value,
                SIMULATION.order_value_std_dev,
            )
            revenue = max(0, revenue)  # No negative revenue
            
            # Serialize touches as JSON string
            import json
            touches_json = json.dumps([(d.isoformat(), ch) for d, ch in touches])
            
            journeys.append({
                "journey_id": journey_id,
                "user_id": user_id,
                "conversion_date": date,
                "revenue": revenue,
                "num_touches": journey_length,
                "touches_json": touches_json,
            })
            
            journey_id += 1
    
    journey_df = pd.DataFrame(journeys)
    return journey_df


def parse_touches(touches_json: str) -> List[Tuple[datetime, str]]:
    """
    Parse JSON touches back to list of (date, channel) tuples.
    
    Args:
        touches_json: JSON string from journey DataFrame
        
    Returns:
        List of (datetime, channel) tuples
    """
    import json
    raw = json.loads(touches_json)
    return [(datetime.fromisoformat(d), ch) for d, ch in raw]


if __name__ == "__main__":
    # Quick smoke test
    from simulate_spend import simulate_daily_spend
    from latent_demand import generate_daily_latent_demand
    
    print("Generating spend and demand...")
    daily_spend = simulate_daily_spend()
    daily_demand = generate_daily_latent_demand(daily_spend)
    
    print("Generating journeys...")
    journeys = generate_journeys(daily_spend, daily_demand)
    
    print(f"\nGenerated {len(journeys)} journeys")
    print(f"  Median touches per journey: {journeys['num_touches'].median():.1f}")
    print(f"  Total revenue: ${journeys['revenue'].sum():,.2f}")
    print(f"  Average order value: ${journeys['revenue'].mean():,.2f}")
    
    print(f"\nJourney samples:")
    print(journeys[["journey_id", "user_id", "conversion_date", "revenue", "num_touches"]].head(10))
    
    # Show a sample journey
    if len(journeys) > 0:
        sample = journeys.iloc[0]
        touches = parse_touches(sample["touches_json"])
        print(f"\nSample journey {sample['journey_id']}:")
        for touch_date, channel in touches:
            print(f"  {touch_date.date()} → {channel}")
        print(f"  Converted: {sample['conversion_date'].date()}, Revenue: ${sample['revenue']:.2f}")
