"""
CSV export utilities for Tableau visualization.

Exports simulation results in clean, Tableau-friendly format.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime

from config import OUTPUT_DIR


def export_daily_spend(
    daily_spend_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export daily spend to CSV.
    
    Args:
        daily_spend_df: From simulate_daily_spend
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "daily_spend.csv")
    
    daily_spend_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


def export_journey_touches(
    journeys_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export journey touches to CSV (one row per journey).
    
    Args:
        journeys_df: From generate_journeys
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "journey_touches.csv")
    
    # Export with key columns
    export_df = journeys_df[[
        "journey_id",
        "user_id",
        "conversion_date",
        "revenue",
        "num_touches",
        "touches_json",
    ]].copy()
    
    # Rename for clarity
    export_df.columns = [
        "journey_id",
        "user_id",
        "conversion_date",
        "revenue",
        "num_touches",
        "touch_sequence",
    ]
    
    export_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


def export_journey_truth(
    journey_truth_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export ground-truth channel contributions (one row per journey-channel pair).
    
    Args:
        journey_truth_df: From calculate_journey_truth
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "journey_truth.csv")
    
    export_df = journey_truth_df[[
        "journey_id",
        "channel",
        "true_contribution_pct",
        "true_contribution_revenue",
    ]].copy()
    
    export_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


def export_channel_truth_summary(
    channel_truth_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export ground-truth summary by channel.
    
    Args:
        channel_truth_df: From aggregate_channel_truth
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "channel_truth_summary.csv")
    
    export_df = channel_truth_df[[
        "channel",
        "total_revenue",
        "num_journeys",
        "avg_contribution_pct",
        "pct_of_total",
    ]].copy()
    
    export_df.columns = [
        "channel",
        "truth_total_revenue",
        "num_journeys_participated",
        "avg_contribution_pct",
        "pct_of_total_revenue",
    ]
    
    export_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


def export_attribution_results(
    attribution_df: pd.DataFrame,
    model_name: str,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export attribution model results.
    
    Args:
        attribution_df: From attribution model (e.g., apply_first_touch)
        model_name: Name of model (for filename)
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    output_path = os.path.join(output_dir, f"attribution_{safe_name}.csv")
    
    attribution_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


def export_model_comparison(
    summary_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export model comparison summary.
    
    Args:
        summary_df: From compare_models (summary DataFrame)
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_comparison.csv")
    
    summary_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


def export_channel_comparison(
    channel_comp_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Export channel-level model comparison.
    
    Args:
        channel_comp_df: From compare_channel_attribution
        output_dir: Output directory path
        
    Returns:
        str: Path to exported file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "channel_comparison.csv")
    
    channel_comp_df.to_csv(output_path, index=False)
    print(f"  Exported: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Export module test - run from main.py")
