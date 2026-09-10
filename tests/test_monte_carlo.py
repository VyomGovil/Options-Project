import pytest

from src.monte_carlo import monte_carlo_price
from src.black_scholes import black_scholes


S = 100
K = 100
R = 0.05
T = 1
SIGMA = 0.20


def test_monte_carlo_call():
    call, _ = black_scholes(S, K, R, T, SIGMA)

    mc_price, standard_error = monte_carlo_price(
        S, K, R, T, SIGMA, "call",
        n_simulations=100_000,
        seed=42,
    )

    assert mc_price == pytest.approx(call, abs=0.10)
    assert standard_error > 0


def test_monte_carlo_put():
    _, put = black_scholes(S, K, R, T, SIGMA)

    mc_price, standard_error = monte_carlo_price(
        S, K, R, T, SIGMA, "put",
        n_simulations=100_000,
        seed=42,
    )

    assert mc_price == pytest.approx(put, abs=0.10)
    assert standard_error > 0


def test_monte_carlo_call_and_put_positive():
    call, _ = monte_carlo_price(
        S, K, R, T, SIGMA, "call",
        n_simulations=10_000,
        seed=42,
    )

    put, _ = monte_carlo_price(
        S, K, R, T, SIGMA, "put",
        n_simulations=10_000,
        seed=42,
    )

    assert call > 0
    assert put > 0


def test_invalid_option_type():
    with pytest.raises(ValueError):
        monte_carlo_price(
            S, K, R, T, SIGMA, "invalid",
            n_simulations=10_000,
            seed=42,
        )