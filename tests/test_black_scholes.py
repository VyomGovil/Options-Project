from src.black_scholes import black_scholes
import numpy as np
def test_black_scholes_known_value():
    call, put = black_scholes(
        100,
        100,
        0.05,
        1,
        0.20
    )

    assert abs(call - 10.4506) < 1e-4
    assert abs(put - 5.5735) < 1e-4
    
def test_put_call_parity():
    s = 100
    k = 100
    r = 0.05
    t = 1
    sigma = 0.20

    call, put = black_scholes(s, k, r, t, sigma)

    lhs = call - put
    rhs = s - k * np.exp(-r * t)

    assert abs(lhs - rhs) < 1e-10
    
def test_call_increases_with_stock_price():
    call_low, _ = black_scholes(90, 100, 0.05, 1, 0.20)
    call_high, _ = black_scholes(110, 100, 0.05, 1, 0.20)

    assert call_high > call_low


def test_put_increases_with_strike_price():
    _, put_low = black_scholes(100, 90, 0.05, 1, 0.20)
    _, put_high = black_scholes(100, 110, 0.05, 1, 0.20)

    assert put_high > put_low