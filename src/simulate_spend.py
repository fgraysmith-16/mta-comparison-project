"""
Daily channel spend simulation with realistic patterns.

Generates daily spend across channels with:
- Base spend from channel configs
- Day-of-week effects (weekday vs weekend)
- Seasonality (monthly patterns)
- Random noise
- Optional trends

Ground truth: Daily spend is known and logged, forming the basis for 
attribution evaluation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

from config import (
    CHANNELS,
    CHANNEL_NAMES,
    CHANNEL_BY_NAME,
    SIMULATION,
    SEASONALITY_MULTIPLIER,
    TREND_GROWTH_RATE,
    RANDOM_SEED,
)


def _get_seasonality_factor(date: datetime) -> float:
    """
    Get seasonality multiplier for a given date.
    
    Args:
        date: datetime object
        
    Returns:
        float: Multiplicative factor for baseline spend
    """
    month = date.month
    return SEASONALITY_MULTIPLIER.get(month, 1.0)


def _get_trend_factor(day_index: int, total_days: int) -> float:
    """
    Get trend growth factor for a given day.
    
    Applied as: 1 + (day_index / total_days) * TREND_GROWTH_RATE
    
    Args:
        day_index: Day number (0-indexed)
        total_days: Total number of simulation days
        
    Returns:
        float: Multiplicative growth factor
    """
    progress = day_index / max(total_days, 1)
    return 1.0 + (progress * TREND_GROWTH_RATE)


def _get_dayofweek_factor(date: datetime) -> float:
    """
    Get day-of-week adjustment factor.
    
    Weekdays (Mon-Fri) = normal, weekend (Sat-Sun) = 0.7 (lower spend).
    
    Args:
        date: datetime object
        
    Returns:
        float: Multiplicative factor
    """
    weekday = date.weekday()  # 0=Monday, 6=Sunday
    if weekday >= 5:  # Saturday or Sunday
        return 0.7
    return 1.0


def simulate_daily_spend(
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate realistic daily spend across all channels.
    
    Each channel gets its own time series with:
    - Base daily spend (from config)
    - Seasonality pattern
    - Trend growth
    - Day-of-week effects
    - Gaussian noise
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        pd.DataFrame with columns:
            - date: datetime
            - channel: str
            - spend: float (dollars)
    """
    np.random.seed(seed)
    
    start_date = datetime.strptime(SIMULATION.start_date, "%Y-%m-%d")
    dates = [start_date + timedelta(days=i) for i in range(SIMULATION.num_days)]
    
    rows = []
    
    for date_idx, date in enumerate(dates):
        seasonality = _get_seasonality_factor(date)
        trend = _get_trend_factor(date_idx, SIMULATION.num_days)
        dow_factor = _get_dayofweek_factor(date)
        
        for channel in CHANNELS:
            # Base spend with all modifiers
            base_spend = channel.avg_daily_spend
            spend = base_spend * seasonality * trend * dow_factor
            
            # Add Gaussian noise proportional to std_dev config
            noise = np.random.normal(0, spend * channel.spend_std_dev)
            spend = max(0, spend + noise)  # Never negative
            
            rows.append({
                "date": date,
                "channel": channel.name,
                "spend": spend,
            })
    
    df = pd.DataFrame(rows)
    return df


def aggregate_spend_by_channel(spend_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily spend by channel to get summary statistics.
    
    Args:
        spend_df: DataFrame from simulate_daily_spend()
        
    Returns:
        pd.DataFrame with columns:
            - channel: str
            - total_spend: float
            - avg_daily_spend: float
            - min_daily_spend: float
            - max_daily_spend: float
            - num_days: int
    """
    summary = (
        spend_df
        .groupby("channel")
        .agg(
            total_spend=("spend", "sum"),
            avg_daily_spend=("spend", "mean"),
            min_daily_spend=("spend", "min"),
            max_daily_spend=("spend", "max"),
            num_days=("spend", "count"),
        )
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    # Quick smoke test
    print("Generating daily spend simulation...")
    daily_spend = simulate_daily_spend()
    print(f"\nGenerated {len(daily_spend)} rows")
    print(f"Date range: {daily_spend['date'].min()} to {daily_spend['date'].max()}")
    print(f"Channels: {daily_spend['channel'].nunique()}")
    print(f"\nDaily spend head:\n{daily_spend.head(10)}")
    
    summary = aggregate_spend_by_channel(daily_spend)
    print(f"\nChannel spend summary:\n{summary}")
