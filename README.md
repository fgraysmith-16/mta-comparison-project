# Multi-Touch Attribution (MTA) Simulator

A production-quality synthetic data generator and attribution evaluation framework demonstrating how different multi-touch attribution methods produce different results and the tradeoffs between computational complexity and accuracy.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation
python main.py
```

## Project Structure

```
mta-simulator/
├── src/
│   ├── config.py              # Channel definitions, simulation parameters
│   ├── simulate_spend.py      # Daily channel spend generation
│   ├── simulate_journeys.py   # User journey and touch path generation
│   ├── ground_truth.py        # Ground-truth channel contribution calculation
│   ├── evaluation.py          # Model comparison vs ground truth
│   ├── export.py              # CSV output for Tableau
│   └── models/
│       ├── first_touch.py     # First-touch attribution
│       ├── last_touch.py      # Last-touch attribution
│       ├── linear_touch.py    # Linear multi-touch
│       ├── shapley_exact.py   # Exact Shapley value
│       ├── shapley_approx.py  # Approximated Shapley
│       ├── markov_rigorous.py # Rigorous removal-effect Markov
│       └── markov_approx.py   # First-order Markov approximation
├── main.py                    # Simulation orchestration
├── requirements.txt
├── .gitignore
└── README.md
```

## Key Features

- **Realistic simulation** with seasonality, adstock, saturation curves, and channel interactions
- **Ground-truth tracking** at the journey level for accurate model evaluation
- **Multiple attribution models** from simple to sophisticated
- **Comprehensive evaluation** showing where simpler models diverge from ground truth
- **CSV outputs** ready for downstream analysis in Tableau

## Simulation Components

### Daily Spend (config.py + simulate_spend.py)
- 6 channels: Paid Search, Paid Social, Display, Video, Affiliate, Email
- Daily spend generation with:
  - Base spend per channel
  - Monthly seasonality (e.g., Q4 holiday lift)
  - Trend growth (2% annual)
  - Day-of-week effects (weekends lower)
  - Gaussian noise

### User Journeys (simulate_journeys.py - TODO)
- User-level multi-touch paths
- Realistic touch lengths (1-8 touches typical)
- Channel mix reflecting spend weights
- Conversion events with revenue

### Ground Truth (ground_truth.py - TODO)
- For each journey: true channel contribution by touch
- Support for saturation and interactions
- Channel-level aggregated truth

### Attribution Models (models/*.py - TODO)
1. **First Touch** - credit first channel
2. **Last Touch** - credit last channel
3. **Linear** - equal credit to all touches
4. **Shapley Exact** - game-theoretic fair value (expensive)
5. **Shapley Approx** - Aumann-Shapley approximation
6. **Markov Rigorous** - removal-effect approach
7. **Markov Approx** - first-order bigram transitions

## Design Principles

- **Clarity over cleverness** — readable, explicit code
- **Modularity** — each component testable in isolation
- **Reproducibility** — fixed random seed for consistency
- **Realism** — design simulation so simple models are plausibly wrong
- **Truth preservation** — always store ground truth for evaluation

## Next Steps

1. Implement journey simulation (random user generation, touch paths)
2. Implement ground truth engine (channel contribution per journey)
3. Implement attribution models (starting with simple ones)
4. Add evaluation framework
5. Export to CSV and validate in Tableau
