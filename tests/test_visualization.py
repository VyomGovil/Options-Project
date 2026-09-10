import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.visualization import (
    plot_price_vs_stock,
    plot_price_vs_volatility,
    plot_monte_carlo_convergence,
)


def test_plot_price_vs_stock():
    s_values = np.array([90, 100, 110])

    plot_price_vs_stock(
        s_values,
        100,
        0.05,
        1,
        0.20,
        "call",
    )


def test_plot_price_vs_volatility():
    sigma_values = np.array([0.10, 0.20, 0.30])

    plot_price_vs_volatility(
        sigma_values,
        100,
        100,
        0.05,
        1,
        "call",
    )

def test_plot_monte_carlo_convergence():
    plot_monte_carlo_convergence(
        100,
        100,
        0.05,
        1,
        0.20,
        "call",
        [1_000, 10_000],
        seed=42,
    )