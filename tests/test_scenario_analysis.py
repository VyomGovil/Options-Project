import numpy as np
import pytest

from src.scenario_analysis import (
    price_scenarios,
    volatility_scenarios,
    greek_scenarios,
    time_to_maturity_scenarios,
    strike_scenarios
)


S = 100
K = 100
R = 0.05
T = 1
SIGMA = 0.20


def test_price_scenarios():
    s_values = np.array([90, 100, 110])

    prices = price_scenarios(
        s_values,
        K,
        R,
        T,
        SIGMA,
        "call",
    )

    assert len(prices) == len(s_values)
    assert np.all(prices > 0)
    assert prices[0] < prices[1] < prices[2]


def test_volatility_scenarios():
    sigma_values = np.array([0.10, 0.20, 0.30])

    prices = volatility_scenarios(
        sigma_values,
        S,
        K,
        R,
        T,
        "call",
    )

    assert len(prices) == len(sigma_values)
    assert np.all(prices > 0)
    assert prices[0] < prices[1] < prices[2]


def test_greek_scenarios():
    s_values = np.array([90, 100, 110])

    results = greek_scenarios(
        s_values,
        K,
        R,
        T,
        SIGMA,
        "call",
    )

    assert set(results.keys()) == {
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    }

    for values in results.values():
        assert len(values) == len(s_values)
        assert np.all(np.isfinite(values))


def test_invalid_option_type():
    with pytest.raises(ValueError):
        price_scenarios(
            np.array([90, 100, 110]),
            K,
            R,
            T,
            SIGMA,
            "invalid",
        )
        
def test_time_to_maturity_scenarios():
    t_values = np.array([0.25, 0.5, 1.0])
    prices = time_to_maturity_scenarios(
        t_values, 100, 100, 0.05, 0.20, "call"
    )

    assert len(prices) == 3
    assert np.all(prices > 0)
    assert prices[0] < prices[-1]


def test_strike_scenarios():
    k_values = np.array([80, 100, 120])
    prices = strike_scenarios(
        k_values, 100, 0.05, 1.0, 0.20, "call"
    )

    assert len(prices) == 3
    assert np.all(prices > 0)
    assert prices[0] > prices[-1]