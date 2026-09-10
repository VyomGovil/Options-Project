import pytest

from src.greeks import delta, gamma, vega, theta, rho


S = 100
K = 100
R = 0.05
T = 1
SIGMA = 0.20


# -------------------------
# Known-value tests
# -------------------------

def test_call_delta():
    assert delta(S, K, R, T, SIGMA, "call") == pytest.approx(0.6368, abs=1e-4)


def test_put_delta():
    assert delta(S, K, R, T, SIGMA, "put") == pytest.approx(-0.3632, abs=1e-4)


def test_gamma():
    assert gamma(S, K, R, T, SIGMA) == pytest.approx(0.0188, abs=1e-4)


def test_vega():
    assert vega(S, K, R, T, SIGMA) == pytest.approx(37.5240, abs=1e-4)


def test_call_theta():
    assert theta(S, K, R, T, SIGMA, "call") == pytest.approx(-6.4140, abs=1e-4)


def test_put_theta():
    assert theta(S, K, R, T, SIGMA, "put") == pytest.approx(-1.6579, abs=1e-4)


def test_call_rho():
    assert rho(S, K, R, T, SIGMA, "call") == pytest.approx(53.2325, abs=1e-4)


def test_put_rho():
    assert rho(S, K, R, T, SIGMA, "put") == pytest.approx(-41.8905, abs=1e-4)


# -------------------------
# Greek relationships
# -------------------------

def test_delta_put_call_relationship():
    call_delta = delta(S, K, R, T, SIGMA, "call")
    put_delta = delta(S, K, R, T, SIGMA, "put")

    assert call_delta - put_delta == pytest.approx(1.0)


def test_gamma_is_positive():
    assert gamma(S, K, R, T, SIGMA) > 0


def test_vega_is_positive():
    assert vega(S, K, R, T, SIGMA) > 0


def test_call_delta_range():
    value = delta(S, K, R, T, SIGMA, "call")

    assert 0 < value < 1


def test_put_delta_range():
    value = delta(S, K, R, T, SIGMA, "put")

    assert -1 < value < 0


def test_call_rho_positive():
    assert rho(S, K, R, T, SIGMA, "call") > 0


def test_put_rho_negative():
    assert rho(S, K, R, T, SIGMA, "put") < 0


# -------------------------
# Input validation
# -------------------------

def test_invalid_option_type_delta():
    with pytest.raises(ValueError):
        delta(S, K, R, T, SIGMA, "invalid")


def test_invalid_option_type_theta():
    with pytest.raises(ValueError):
        theta(S, K, R, T, SIGMA, "invalid")


def test_invalid_option_type_rho():
    with pytest.raises(ValueError):
        rho(S, K, R, T, SIGMA, "invalid")