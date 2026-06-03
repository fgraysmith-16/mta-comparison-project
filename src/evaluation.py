"""
Model evaluation framework.

Compares attribution model outputs against ground truth and computes
error metrics, runtime comparisons, and rank correlations.

Key evaluation metrics:
- MAE/RMSE: Absolute distance from ground truth
- Rank correlation: How well model ranks channels correctly
- Runtime: Computational cost tradeoff
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Callable, Tuple
from scipy.stats import spearmanr

from ground_truth import aggregate_channel_truth


def evaluate_model(
    model_func: Callable,
    journeys_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    model_name: str,
) -> Dict:
    """
    Evaluate a single attribution model against ground truth.
    
    Args:
        model_func: Function that takes journeys_df and returns attribution result
        journeys_df: Journey data
        ground_truth_df: Ground truth aggregated by channel
        model_name: Name of model for reporting
        
    Returns:
        Dict with keys:
            - model_name
            - total_revenue: Total revenue attributed by model
            - runtime_seconds: Execution time
            - channel_results: Dict[channel -> attributed_revenue]
            - mae: Mean Absolute Error vs ground truth
            - rmse: Root Mean Squared Error vs ground truth
            - rank_correlation: Spearman rank correlation
            - rank_correlation_p_value
    """
    # Run model and measure time
    start_time = time.time()
    attribution_result = model_func(journeys_df)
    runtime = time.time() - start_time
    
    # Aggregate model results by channel
    model_agg = (
        attribution_result
        .groupby("channel")
        .agg(total_revenue=("attributed_revenue", "sum"))
        .reset_index()
    )
    model_agg = model_agg.set_index("channel")["total_revenue"].to_dict()
    
    # Ensure all channels present
    for ch in ground_truth_df["channel"]:
        if ch not in model_agg:
            model_agg[ch] = 0.0
    
    # Extract ground truth by channel
    truth_dict = dict(zip(ground_truth_df["channel"], ground_truth_df["total_revenue"]))
    for ch in model_agg.keys():
        if ch not in truth_dict:
            truth_dict[ch] = 0.0
    
    # Sort channels consistently
    channels = sorted(set(list(model_agg.keys()) + list(truth_dict.keys())))
    
    model_values = np.array([model_agg.get(ch, 0) for ch in channels])
    truth_values = np.array([truth_dict.get(ch, 0) for ch in channels])
    
    # Compute error metrics
    mae = np.mean(np.abs(model_values - truth_values))
    rmse = np.sqrt(np.mean((model_values - truth_values) ** 2))
    
    # Rank correlation (Spearman)
    if len(channels) > 1:
        rank_corr, p_value = spearmanr(model_values, truth_values)
    else:
        rank_corr = 1.0
        p_value = 0.0
    
    return {
        "model_name": model_name,
        "total_revenue": np.sum(model_values),
        "runtime_seconds": runtime,
        "channel_results": {ch: model_agg.get(ch, 0) for ch in channels},
        "mae": mae,
        "rmse": rmse,
        "rank_correlation": rank_corr,
        "rank_correlation_p_value": p_value,
    }


def compare_models(
    model_dict: Dict[str, Callable],
    journeys_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Compare multiple attribution models against ground truth.
    
    Args:
        model_dict: Dict[model_name -> model_func]
        journeys_df: Journey data
        ground_truth_df: Ground truth aggregated by channel
        
    Returns:
        Tuple of:
        - Summary DataFrame with error metrics and runtime
        - Full results Dict[model_name -> evaluation results]
    """
    results = {}
    
    for model_name, model_func in model_dict.items():
        print(f"  Evaluating {model_name}...", end=" ", flush=True)
        result = evaluate_model(model_func, journeys_df, ground_truth_df, model_name)
        results[model_name] = result
        print(f"({result['runtime_seconds']:.2f}s, MAE=${result['mae']:.2f})")
    
    # Create summary DataFrame
    summary_rows = []
    for model_name, result in results.items():
        summary_rows.append({
            "model_name": model_name,
            "total_revenue": result["total_revenue"],
            "runtime_seconds": result["runtime_seconds"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "rank_correlation": result["rank_correlation"],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, results


def create_benchmark_report(
    summary_df: pd.DataFrame,
    ground_truth_total_revenue: float,
) -> pd.DataFrame:
    """
    Create a comprehensive benchmark report showing accuracy vs complexity tradeoff.
    
    Args:
        summary_df: From compare_models() summary DataFrame
        ground_truth_total_revenue: Total revenue from ground truth
        
    Returns:
        pd.DataFrame with extended metrics:
            - model_name
            - runtime_seconds
            - relative_complexity: Runtime normalized to fastest model
            - mae
            - mae_pct_of_truth: MAE as % of ground truth revenue
            - mape: Mean Absolute Percentage Error
            - rank_correlation
            - accuracy_rating: Qualitative assessment
    """
    report = summary_df.copy()
    
    # Normalize runtime
    min_runtime = report["runtime_seconds"].min()
    report["relative_complexity"] = report["runtime_seconds"] / min_runtime
    
    # MAE as percentage of ground truth
    report["mae_pct_of_truth"] = 100.0 * report["mae"] / ground_truth_total_revenue
    
    # Mean Absolute Percentage Error (simplified)
    report["mape"] = report["mae_pct_of_truth"]
    
    # Accuracy rating based on MAE
    def rate_accuracy(mae_pct):
        if mae_pct < 5:
            return "Excellent"
        elif mae_pct < 15:
            return "Good"
        elif mae_pct < 30:
            return "Fair"
        else:
            return "Poor"
    
    report["accuracy_rating"] = report["mae_pct_of_truth"].apply(rate_accuracy)
    
    # Complexity rating based on runtime relative to linear model
    def rate_complexity(rel_complexity):
        if rel_complexity <= 1.0:
            return "Fast"
        elif rel_complexity < 5:
            return "Moderate"
        elif rel_complexity < 50:
            return "Expensive"
        else:
            return "Very Expensive"
    
    report["complexity_rating"] = report["relative_complexity"].apply(rate_complexity)
    
    return report


def compare_channel_attribution(
    results: Dict[str, Dict],
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare channel-level attribution across models.
    
    Args:
        results: From compare_models() full results
        ground_truth_df: Ground truth aggregated by channel
        
    Returns:
        pd.DataFrame with channel comparison:
            - channel
            - truth_revenue
            - model1_revenue
            - model1_error
            - model2_revenue
            - model2_error
            - ...
    """
    truth_dict = dict(zip(ground_truth_df["channel"], ground_truth_df["total_revenue"]))
    
    rows = []
    for channel in sorted(truth_dict.keys()):
        row = {
            "channel": channel,
            "truth_revenue": truth_dict.get(channel, 0),
        }
        
        for model_name, result in sorted(results.items()):
            model_rev = result["channel_results"].get(channel, 0)
            error = model_rev - row["truth_revenue"]
            error_pct = 100.0 * error / max(row["truth_revenue"], 1)
            
            row[f"{model_name}_revenue"] = model_rev
            row[f"{model_name}_error"] = error
            row[f"{model_name}_error_pct"] = error_pct
        
        rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from simulate_spend import simulate_daily_spend
    from latent_demand import generate_daily_latent_demand
    from simulate_journeys import generate_journeys
    from ground_truth import calculate_journey_truth
    from models.first_touch import apply_first_touch
    from models.last_touch import apply_last_touch
    from models.linear_touch import apply_linear_touch
    
    print("Generating test data...")
    daily_spend = simulate_daily_spend()
    daily_demand = generate_daily_latent_demand(daily_spend)
    journeys = generate_journeys(daily_spend, daily_demand)
    journey_truth = calculate_journey_truth(journeys)
    ground_truth = aggregate_channel_truth(journey_truth)
    
    print("\nEvaluating models...")
    models = {
        "First Touch": apply_first_touch,
        "Last Touch": apply_last_touch,
        "Linear": apply_linear_touch,
    }
    
    summary, results = compare_models(models, journeys, ground_truth)
    
    print("\nModel comparison summary:")
    print(summary.to_string(index=False))
    
    print("\nChannel-level comparison:")
    channel_comp = compare_channel_attribution(results, ground_truth)
    print(channel_comp.to_string(index=False))
