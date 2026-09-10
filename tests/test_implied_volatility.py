import pytest

from src.black_scholes import black_scholes
from src.implied_volatility import implied_volatility


S = 100
K = 100
R = 0.05
T = 1
SIGMA = 0.20


def test_call_implied_volatility():
    call, _ = black_scholes(S, K, R, T, SIGMA)

    result = implied_volatility(
        S, K, R, T, call, "call"
    )

    assert result == pytest.approx(SIGMA, abs=1e-6)


def test_put_implied_volatility():
    _, put = black_scholes(S, K, R, T, SIGMA)

    result = implied_volatility(
        S, K, R, T, put, "put"
    )

    assert result == pytest.approx(SIGMA, abs=1e-6)


def test_implied_volatility_recovers_different_volatility():
    sigma = 0.35

    call, _ = black_scholes(S, K, R, T, sigma)

    result = implied_volatility(
        S, K, R, T, call, "call"
    )

    assert result == pytest.approx(sigma, abs=1e-6)


def test_invalid_option_type():
    with pytest.raises(ValueError):
        implied_volatility(
            S, K, R, T, 10, "invalid"
        )