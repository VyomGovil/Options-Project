import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from src.visualization import (
    plot_call_put_vs_stock,
    plot_price_vs_volatility,
    plot_greeks_vs_stock,
    plot_monte_carlo_convergence,
    plot_implied_volatility_skew,
    plot_price_vs_time_to_maturity,
    plot_price_vs_strike
)


S = 100
K = 100
R = 0.05
T = 1
SIGMA = 0.20

PLOTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "plots",
)

os.makedirs(PLOTS_DIR, exist_ok=True)


s_values = np.linspace(60, 140, 100)
sigma_values = np.linspace(0.05, 0.60, 100)
t_values = np.linspace(0.05, 2.0, 20)
k_values = np.linspace(70, 130, 20)

plot_price_vs_time_to_maturity(
    t_values,
    s=100,
    k=100,
    r=0.05,
    sigma=0.20,
    option_type="call",
    filename="plots/price_vs_time_to_maturity.png"
)

plot_price_vs_strike(
    k_values,
    s=100,
    r=0.05,
    t=1,
    sigma=0.20,
    option_type="call",
    filename="plots/price_vs_strike.png"
)

plot_price_vs_volatility(
    sigma_values,
    S,
    K,
    R,
    T,
    "call",
    os.path.join(PLOTS_DIR, "price_vs_volatility.png"),
)


plot_greeks_vs_stock(
    s_values,
    K,
    R,
    T,
    SIGMA,
    "call",
    os.path.join(PLOTS_DIR, "greeks_vs_stock.png"),
)


simulation_counts = [
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
]

plot_monte_carlo_convergence(
    S,
    K,
    R,
    T,
    SIGMA,
    "call",
    simulation_counts,
    seed=42,
    filename=os.path.join(
        PLOTS_DIR,
        "monte_carlo_convergence.png",
    ),
)

plot_call_put_vs_stock(
    s_values,
    K,
    R,
    T,
    SIGMA,
    os.path.join(
        PLOTS_DIR,
        "call_put_vs_stock.png",
    ),
)

strikes = np.linspace(70, 130, 13)

plot_implied_volatility_skew(
    strikes,
    S,
    R,
    T,
    SIGMA,
    "put",
    os.path.join(
        PLOTS_DIR,
        "implied_volatility_skew.png",
    ),
)

print("Plots generated successfully.")