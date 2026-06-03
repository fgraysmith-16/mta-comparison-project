"""
Main entry point for the MTA simulator.

This module orchestrates the full simulation pipeline:
1. Generate realistic daily channel spend
2. Compute latent demand signal (conversion probability)
3. Simulate user journeys with multi-touch interactions
4. Calculate ground-truth channel contribution per journey
5. Fit and apply attribution models
6. Evaluate model accuracy versus ground truth
7. Export results to CSV for Tableau
"""

import sys
from pathlib import Path
import os

# Add src to path so we can import config, simulate_spend, etc.
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import OUTPUT_DIR, SIMULATION
from simulate_spend import simulate_daily_spend, aggregate_spend_by_channel
from latent_demand import generate_daily_latent_demand
from simulate_journeys import generate_journeys
from ground_truth import calculate_journey_truth, aggregate_channel_truth
from models.first_touch import apply_first_touch
from models.last_touch import apply_last_touch
from models.linear_touch import apply_linear_touch
from models.shapley_exact import apply_shapley_exact
from models.shapley_approx import apply_shapley_approx
from models.markov_rigorous import apply_markov_rigorous
from models.markov_approx import apply_markov_approx
from evaluation import compare_models, compare_channel_attribution, create_benchmark_report
from export import (
    export_daily_spend,
    export_journey_touches,
    export_journey_truth,
    export_channel_truth_summary,
    export_attribution_results,
    export_model_comparison,
    export_channel_comparison,
)


def main():
    """Run the full MTA simulation pipeline."""
    print("=" * 70)
    print("MULTI-TOUCH ATTRIBUTION SIMULATOR")
    print("=" * 70)
    
    # =========================================================================
    # STAGE 1: Spend Simulation
    # =========================================================================
    print("\n[1] Generating realistic daily channel spend...")
    daily_spend = simulate_daily_spend()
    print(f"    Generated {len(daily_spend)} spend records")
    
    spend_summary = aggregate_spend_by_channel(daily_spend)
    print(f"\n    Spend by channel (total):")
    for _, row in spend_summary.iterrows():
        print(f"      {row['channel']:20s}: ${row['total_spend']:>12,.2f}")
    
    # =========================================================================
    # STAGE 2: Latent Demand Signal
    # =========================================================================
    print("\n[2] Computing latent demand (conversion probability)...")
    daily_demand = generate_daily_latent_demand(daily_spend)
    print(f"    Generated {len(daily_demand)} demand records")
    print(f"    Demand range: {daily_demand['latent_demand'].min():.4f} to {daily_demand['latent_demand'].max():.4f}")
    print(f"    Mean demand: {daily_demand['latent_demand'].mean():.4f}")
    
    # Show seasonality effect
    demand_by_month = daily_demand.copy()
    demand_by_month["month"] = demand_by_month["date"].dt.month
    monthly_avg = demand_by_month.groupby("month")["latent_demand"].mean()
    print(f"\n    Seasonality (Q4 holiday lift):")
    print(f"      October: {monthly_avg[10]:.4f}")
    print(f"      November: {monthly_avg[11]:.4f}")
    print(f"      December: {monthly_avg[12]:.4f}")
    
    # =========================================================================
    # STAGE 3: Journey Simulation
    # =========================================================================
    print("\n[3] Generating user journeys...")
    journeys = generate_journeys(daily_spend, daily_demand)
    print(f"    Generated {len(journeys)} journeys")
    print(f"    Median touches per journey: {journeys['num_touches'].median():.1f}")
    print(f"    Total revenue: ${journeys['revenue'].sum():,.2f}")
    
    # =========================================================================
    # STAGE 4: Ground Truth Attribution
    # =========================================================================
    print("\n[4] Computing ground-truth channel contribution...")
    journey_truth = calculate_journey_truth(journeys)
    channel_truth = aggregate_channel_truth(journey_truth)
    print(f"    Computed truth for {len(journey_truth)} journey-channel pairs")
    print(f"\n    Ground-truth revenue by channel:")
    for _, row in channel_truth.iterrows():
        print(f"      {row['channel']:20s}: ${row['total_revenue']:>12,.2f} ({row['pct_of_total']:>5.1f}%)")
    
    # =========================================================================
    # STAGE 5: Attribution Models
    # =========================================================================
    print("\n[5] Running attribution models...")
    
    models = {
        "First Touch": apply_first_touch,
        "Last Touch": apply_last_touch,
        "Linear": apply_linear_touch,
        "Shapley Approx": apply_shapley_approx,
        "Shapley Exact": apply_shapley_exact,
        "Markov Rigorous": apply_markov_rigorous,
        "Markov Approx": apply_markov_approx,
    }
    
    attribution_results = {}
    for model_name, model_func in models.items():
        print(f"    Applying {model_name}...", end=" ", flush=True)
        result = model_func(journeys)
        attribution_results[model_name] = result
        summary = result.groupby("channel")["attributed_revenue"].sum()
        print(f"Total: ${summary.sum():,.2f}")
    
    # =========================================================================
    # STAGE 6: Evaluation
    # =========================================================================
    print("\n[6] Evaluating model accuracy...")
    model_summary, eval_results = compare_models(models, journeys, channel_truth)
    
    # Create detailed benchmark report
    truth_total = channel_truth["total_revenue"].sum()
    benchmark_report = create_benchmark_report(model_summary, truth_total)
    
    print(f"\n    Model comparison (runtime vs accuracy vs complexity):")
    print(f"    {'Model':<20} {'Runtime (s)':<12} {'Complexity':<12} {'MAE ($)':<12} {'Error %':<10} {'Accuracy':<12}")
    print(f"    {'-'*80}")
    for _, row in benchmark_report.iterrows():
        print(f"    {row['model_name']:<20} {row['runtime_seconds']:<12.3f} {row['relative_complexity']:<12.1f}x ${row['mae']:<11,.0f} {row['mae_pct_of_truth']:<9.1f}% {row['accuracy_rating']:<12}")
    
    channel_comp = compare_channel_attribution(eval_results, channel_truth)
    
    # =========================================================================
    # STAGE 7: Export
    # =========================================================================
    print("\n[7] Exporting results to CSV...")
    
    export_daily_spend(daily_spend, OUTPUT_DIR)
    export_journey_touches(journeys, OUTPUT_DIR)
    export_journey_truth(journey_truth, OUTPUT_DIR)
    export_channel_truth_summary(channel_truth, OUTPUT_DIR)
    
    for model_name, attr_result in attribution_results.items():
        export_attribution_results(attr_result, model_name, OUTPUT_DIR)
    
    export_model_comparison(model_summary, OUTPUT_DIR)
    export_channel_comparison(channel_comp, OUTPUT_DIR)
    
    # Export benchmark report
    benchmark_report.to_csv(
        os.path.join(OUTPUT_DIR, "benchmark_report.csv"),
        index=False,
    )
    print(f"  Exported: {os.path.join(OUTPUT_DIR, 'benchmark_report.csv')}")
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
