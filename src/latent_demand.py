"""
Latent demand signal generation from channel spend.

Models realistic demand drivers:
- Adstock (decay/lag effects over time)
- Saturation (diminishing returns with high spend)
- Pairwise channel interactions
- Baseline demand

The latent demand is converted to daily conversion probability, which drives
user journey generation and conversions in the simulation.

Key assumption: Demand is modeled in log-space additively, then exponentiated
and scaled to a probability. Interactions are multiplicative.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime, timedelta

from config import (
    CHANNELS,
    CHANNEL_BY_NAME,
    SIMULATION,
    RANDOM_SEED,
)


def adstock_geometric(
    spend_series: np.ndarray,
    half_life: float,
    decay_rate: float = 0.5,
) -> np.ndarray:
    """
    Apply geometric adstock decay to a spend series.
    
    Geometric adstock formula:
        adstocked[t] = spend[t] + decay_rate * adstocked[t-1]
    
    Where decay_rate is typically 0.5 (half-life of 1 day means 50% carryover).
    
    Args:
        spend_series: 1D array of daily spend
        half_life: Half-life of decay in days (e.g., 1.0 for 1-day half-life)
        decay_rate: Decay rate; if not provided, computed from half_life
        
    Returns:
        np.ndarray: Adstocked spend series, same shape as input
    """
    # Compute decay rate from half_life
    # At half_life, remaining fraction = 0.5
    # 0.5 = decay_rate^half_life => decay_rate = 0.5^(1/half_life)
    decay_rate = 0.5 ** (1.0 / max(half_life, 0.1))
    
    adstocked = np.zeros_like(spend_series, dtype=float)
    adstocked[0] = spend_series[0]
    
    for t in range(1, len(spend_series)):
        adstocked[t] = spend_series[t] + decay_rate * adstocked[t - 1]
    
    return adstocked


def saturation_power_law(
    spend: np.ndarray,
    exponent: float = 0.5,
) -> np.ndarray:
    """
    Apply power-law saturation to spend.
    
    Saturation formula:
        saturated = spend ^ exponent
    
    Where 0 < exponent < 1 creates diminishing returns:
    - exponent = 1.0: no saturation (linear)
    - exponent = 0.5: moderate saturation (sqrt)
    - exponent = 0.3: strong saturation
    
    Args:
        spend: 1D array or scalar of spend amounts
        exponent: Saturation exponent (0 < exp <= 1)
        
    Returns:
        np.ndarray: Saturated spend, same shape as input
    """
    # Ensure exponent is in (0, 1]
    exponent = np.clip(exponent, 0.01, 1.0)
    # Prevent negative spend from power operation
    saturated = np.power(np.maximum(spend, 0), exponent)
    return saturated


def compute_interaction_multiplier(
    channel_adstocked_spend: Dict[str, float],
    config_interactions: Dict[str, Dict[str, float]],
) -> float:
    """
    Compute multiplicative interaction effect across all channels.
    
    For each channel present in the journey, look at its configured interactions
    with other channels that are also present. Apply multiplicative uplift.
    
    Example:
    - Paid Search spends $1000, Paid Social spends $500
    - Paid Search config has interaction {"Paid Social": 1.1}
    - Uplift = 1.1 (10% boost to Paid Search when Social is present)
    
    Args:
        channel_adstocked_spend: Dict[channel_name -> adstocked_spend]
        config_interactions: Dict[channel_name -> Dict[interacting_ch -> uplift]]
        
    Returns:
        float: Cumulative multiplicative interaction effect (>= 1.0)
    """
    multiplier = 1.0
    
    for channel, interactions in config_interactions.items():
        if channel not in channel_adstocked_spend:
            continue
        
        # Check which other channels this one interacts with
        for interacting_channel, interaction_lift in interactions.items():
            if interacting_channel in channel_adstocked_spend:
                # Both channels are active; apply the multiplicative lift
                # We apply it once per pair to avoid double-counting
                # (conservative: only apply if both spend > threshold)
                if channel_adstocked_spend[interacting_channel] > 10:
                    multiplier *= interaction_lift
    
    return multiplier


def generate_daily_latent_demand(
    daily_spend_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate daily latent demand (conversion probability) from channel spend.
    
    For each day:
    1. Adstock each channel's spend
    2. Apply saturation to adstocked spend
    3. Sum saturated spend across channels (log-space additivity)
    4. Compute pairwise interaction multiplier
    5. Convert to daily conversion probability
    
    Args:
        daily_spend_df: DataFrame from simulate_daily_spend with columns:
            [date, channel, spend]
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame with columns:
            - date: datetime
            - latent_demand: float (daily conversion probability, 0 to 1)
            - spend_impact_dict: dict (for inspection)
    """
    np.random.seed(seed)
    
    # Pivot spend to get matrix form: rows=date, cols=channel
    spend_pivot = daily_spend_df.pivot_table(
        index="date",
        columns="channel",
        values="spend",
        fill_value=0.0,
    )
    
    # Ensure all channels are present
    for ch in CHANNEL_BY_NAME.keys():
        if ch not in spend_pivot.columns:
            spend_pivot[ch] = 0.0
    
    # Sort by date and get channel order
    spend_pivot = spend_pivot.sort_index()
    
    rows = []
    
    for date_idx, (date, row) in enumerate(spend_pivot.iterrows()):
        # Dictionary to hold adstocked+saturated spend by channel
        channel_impact = {}
        total_impact = 0.0
        
        for channel_name, base_spend in row.items():
            channel_config = CHANNEL_BY_NAME[channel_name]
            
            # Get the full adstocked history for this channel up to this day
            channel_hist = spend_pivot[channel_name].iloc[:date_idx + 1].values
            
            # Apply adstock
            adstocked_hist = adstock_geometric(
                channel_hist,
                half_life=channel_config.adstock_half_life,
            )
            adstocked_today = adstocked_hist[-1]
            
            # Apply saturation
            saturated = saturation_power_law(
                adstocked_today,
                exponent=channel_config.saturation_exponent,
            )
            
            channel_impact[channel_name] = saturated
            # Accumulate impact in log space (then will exponentiate)
            total_impact += saturated
        
        # Compute interaction multiplier
        config_interactions = {
            ch: channel_config.interaction_effect
            for ch, channel_config in CHANNEL_BY_NAME.items()
        }
        interaction_mult = compute_interaction_multiplier(
            channel_impact,
            config_interactions,
        )
        
        # Convert total impact to probability
        # Formula: baseline + (total_impact * interaction) with sigmoid-like scaling
        # We scale by 1/1000 to keep the exponent reasonable
        baseline_demand = SIMULATION.base_conversion_rate
        
        # Combine baseline, total channel impact, and interaction
        # exp() for multiplicative effect, then clip to [0, 1]
        demand_signal = baseline_demand * np.exp(total_impact / 1000.0) * interaction_mult
        demand_prob = np.clip(demand_signal, 0.0, 1.0)
        
        rows.append({
            "date": date,
            "latent_demand": demand_prob,
            "total_impact": total_impact,
            "interaction_multiplier": interaction_mult,
        })
    
    demand_df = pd.DataFrame(rows)
    return demand_df


def compare_demand_with_and_without_interactions(
    daily_spend_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate demand both with and without interactions for comparison.
    
    Useful for validating that interactions have meaningful effect.
    
    Args:
        daily_spend_df: DataFrame from simulate_daily_spend
        
    Returns:
        pd.DataFrame with columns:
            - date
            - latent_demand_with_interactions
            - latent_demand_without_interactions
            - interaction_impact_pct
    """
    # TODO: Implement version that zeros out interaction effects
    pass


if __name__ == "__main__":
    # Quick smoke test
    from simulate_spend import simulate_daily_spend
    
    print("Generating spend...")
    daily_spend = simulate_daily_spend()
    
    print("Computing latent demand...")
    demand = generate_daily_latent_demand(daily_spend)
    
    print(f"\nGenerated {len(demand)} daily demand records")
    print(f"Demand range: {demand['latent_demand'].min():.4f} to {demand['latent_demand'].max():.4f}")
    print(f"Mean demand: {demand['latent_demand'].mean():.4f}")
    
    print("\nDaily demand (first 15 days):")
    print(demand[["date", "latent_demand", "total_impact", "interaction_multiplier"]].head(15))
    
    print("\nDaily demand (last 15 days):")
    print(demand[["date", "latent_demand", "total_impact", "interaction_multiplier"]].tail(15))
    
    # Check seasonality effect on demand
    demand["month"] = demand["date"].dt.month
    monthly_demand = demand.groupby("month")["latent_demand"].mean()
    print("\nAverage demand by month:")
    print(monthly_demand)
