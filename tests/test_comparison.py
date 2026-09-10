import pytest

from src.comparison import compare_prices


S = 100
K = 100
R = 0.05
T = 1
SIGMA = 0.20


def test_call_comparison():
    result = compare_prices(
        S, K, R, T, SIGMA,
        "call",
        n_simulations=100_000,
        seed=42,
    )

    bs_price = result["black_scholes"]
    mc_price = result["monte_carlo"]
    standard_error = result["standard_error"]

    assert standard_error > 0
    assert abs(mc_price - bs_price) <= 1.96 * standard_error


def test_put_comparison():
    result = compare_prices(
        S, K, R, T, SIGMA,
        "put",
        n_simulations=100_000,
        seed=42,
    )

    bs_price = result["black_scholes"]
    mc_price = result["monte_carlo"]
    standard_error = result["standard_error"]

    assert standard_error > 0
    assert abs(mc_price - bs_price) <= 1.96 * standard_error


def test_confidence_interval_contains_monte_carlo_price():
    result = compare_prices(
        S, K, R, T, SIGMA,
        "call",
        n_simulations=100_000,
        seed=42,
    )

    lower, upper = result["confidence_interval"]

    assert lower <= result["monte_carlo"] <= upper


def test_difference_is_correct():
    result = compare_prices(
        S, K, R, T, SIGMA,
        "call",
        n_simulations=10_000,
        seed=42,
    )

    assert result["difference"] == pytest.approx(
        result["monte_carlo"] - result["black_scholes"]
    )


def test_invalid_option_type():
    with pytest.raises(ValueError):
        compare_prices(
            S, K, R, T, SIGMA,
            "invalid",
            n_simulations=10_000,
            seed=42,
        )