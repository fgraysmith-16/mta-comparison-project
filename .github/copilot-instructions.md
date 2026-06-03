Project-wide instructions
Purpose
This repository builds a synthetic multi-touch attribution simulator for demonstrating how different attribution methods produce different results.
Tech standards
Use Python
Prefer pandas and numpy for data manipulation
Use scikit-learn where appropriate
Use networkx or equivalent if helpful for Markov implementations
Keep dependencies minimal
Architecture
Separate simulation, truth generation, attribution models, evaluation, and CSV export into different modules
Keep business logic out of notebooks
Make randomness reproducible with seed controls
Add clear function boundaries
Coding style
Prefer readable, explicit code over clever abstractions
Use type hints where reasonable
Add docstrings for public functions
Keep module responsibilities narrow
Avoid giant files
Output expectations
Every major stage should be runnable from a main entry point
CSV outputs should be Tableau-friendly
Use consistent column names and snake_case
Modeling expectations
Include realistic lag/adstock
Include nonlinear saturation
Include channel interactions
Preserve ground truth for evaluation
Make it easy to compare runtime vs attribution accuracy
Collaboration behavior
When implementing features:
propose a short plan
implement incrementally
explain assumptions
identify risks or simplifications
