# Optimization & Machine Learning Portfolio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)
![Focus](https://img.shields.io/badge/Focus-Optimization%20%26%20ML-orange)

This repository hosts a comprehensive portfolio project demonstrating advanced mathematical optimization techniques and predictive modeling. It addresses two distinct computational challenges: **Non-Linear Regression Analysis** for demand forecasting and **Constrained Convex Optimization** for resource allocation.

## 👤 Author
**Aryan Malik** *Computer Science & Engineering*

---

## 📂 Repository Structure

The project is modularized into two independent solvers, each with its own source code, datasets, and output pipelines.

```text
Optimization-Portfolio/
│
├── Question1Report.pdf          # Detailed Analysis Report (Regression)
├── Question2Report.pdf          # Detailed Analysis Report (Optimization)
├── requirements.txt             # Project dependencies
├── README.md                    # Documentation
│
├── main_proj_code/              # [MODULE 1] Regression Analysis Source
│   ├── __init__.py
│   ├── model_definitions.py     # Architectures: Linear, Polynomial, Interaction
│   ├── preprocess.py            # Data cleaning & One-Hot Encoding pipeline
│   ├── train_and_evaluate.py    # Training & Cross-validation logic
│   ├── plot_results.py          # Visualization scripts (Scatter plots)
│   ├── run_all.py               # Main execution entry point
│   └── data/
│       └── train.xlsx           # Bike-sharing historical dataset
│
├── constrained_lagrangian_code/ # [MODULE 2] Constrained Optimization Source
│   ├── __init__.py
│   ├── problem_definition.py    # Cost functions & Linear Constraints
│   ├── lagrangian_relaxation.py # Analytical primal variable derivation
│   ├── dual_gradient_ascent.py  # Numerical iterative solver
│   ├── plots.py                 # Convergence visualization
│   └── run.py                   # Main execution entry point
│
└── results/                     # Generated Artifacts (Created upon execution)
    ├── best_model.txt           # Q1 Performance Metrics (MSE, R2)
    ├── metrics_table.xlsx       # Q1 Comparative Analysis
    ├── solution.txt             # Q2 Optimization Results (Lambda, Cost)
    ├── plots/                   # Q1 Prediction Visualizations
    └── dual_convergence.png     # Q2 Convergence Plot
```
## ⚙️ Setup & Installation
Clone the repository:
git clone [https://github.com/AryanOg0209/Optimization-Project.git](https://github.com/AryanOg0209/Optimization-Project.git)
cd Optimization-Project



