import numpy as np
import pytest

from src.black_scholes import black_scholes
from src.greeks import delta, gamma, vega, theta, rho
from src.implied_volatility import implied_volatility
from src.monte_carlo import monte_carlo_price


def test_call_arbitrage_bounds():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.20

    call, _ = black_scholes(s, k, r, t, sigma)

    lower_bound = max(s - k * np.exp(-r * t), 0)

    assert call >= lower_bound
    assert call <= s


def test_put_arbitrage_bounds():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.20

    _, put = black_scholes(s, k, r, t, sigma)

    lower_bound = max(k * np.exp(-r * t) - s, 0)

    assert put >= lower_bound
    assert put <= k * np.exp(-r * t)


def test_put_call_parity():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.20

    call, put = black_scholes(s, k, r, t, sigma)

    expected_difference = s - k * np.exp(-r * t)

    assert call - put == pytest.approx(
        expected_difference,
        abs=1e-6
    )


def test_call_matches_known_reference():
    call, _ = black_scholes(
        100,
        100,
        0.05,
        1,
        0.20
    )

    assert call == pytest.approx(10.4506, abs=0.01)


def test_put_matches_known_reference():
    _, put = black_scholes(
        100,
        100,
        0.05,
        1,
        0.20
    )

    assert put == pytest.approx(5.5735, abs=0.01)


def test_call_increases_with_volatility():
    low_vol, _ = black_scholes(
        100,
        100,
        0.05,
        1,
        0.10
    )

    high_vol, _ = black_scholes(
        100,
        100,
        0.05,
        1,
        0.30
    )

    assert high_vol > low_vol


def test_call_increases_with_time():
    short_call, _ = black_scholes(
        100,
        100,
        0.05,
        0.5,
        0.20
    )

    long_call, _ = black_scholes(
        100,
        100,
        0.05,
        2.0,
        0.20
    )

    assert long_call > short_call


def test_put_increases_with_time():
    _, short_put = black_scholes(
        100,
        100,
        0.05,
        0.5,
        0.20
    )

    _, long_put = black_scholes(
        100,
        100,
        0.05,
        2.0,
        0.20
    )

    assert long_put > short_put


def test_call_delta_bounds():
    for s in [50, 80, 100, 120, 150]:
        value = delta(
            s,
            100,
            0.05,
            1,
            0.20,
            "call"
        )

        assert 0 <= value <= 1


def test_put_delta_bounds():
    for s in [50, 80, 100, 120, 150]:
        value = delta(
            s,
            100,
            0.05,
            1,
            0.20,
            "put"
        )

        assert -1 <= value <= 0


def test_delta_changes_in_expected_direction():
    low_delta = delta(
        90,
        100,
        0.05,
        1,
        0.20,
        "call"
    )

    high_delta = delta(
        110,
        100,
        0.05,
        1,
        0.20,
        "call"
    )

    assert high_delta > low_delta


def test_gamma_is_positive_across_scenarios():
    for s in [80, 90, 100, 110, 120]:
        assert gamma(
            s,
            100,
            0.05,
            1,
            0.20
        ) > 0


def test_vega_is_positive_across_scenarios():
    for s in [80, 90, 100, 110, 120]:
        assert vega(
            s,
            100,
            0.05,
            1,
            0.20
        ) > 0


def test_implied_volatility_round_trip():
    for sigma in [0.10, 0.20, 0.30, 0.40]:
        call, _ = black_scholes(
            100,
            100,
            0.05,
            1,
            sigma
        )

        recovered_sigma = implied_volatility(
            100,
            100,
            0.05,
            1,
            call,
            "call"
        )

        assert recovered_sigma == pytest.approx(
            sigma,
            abs=1e-6
        )


def test_monte_carlo_matches_black_scholes():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.20

    bs_call, _ = black_scholes(
        s,
        k,
        r,
        t,
        sigma
    )

    mc_call, standard_error = monte_carlo_price(
        s,
        k,
        r,
        t,
        sigma,
        "call",
        n_simulations=100_000,
        seed=42
    )

    lower = mc_call - 1.96 * standard_error
    upper = mc_call + 1.96 * standard_error

    assert lower <= bs_call <= upper