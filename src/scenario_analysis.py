import numpy as np

from src.black_scholes import black_scholes
from src.greeks import delta, gamma, vega, theta, rho


def price_scenarios(
    s_values,
    k,
    r,
    t,
    sigma,
    option_type,
):
    prices = []

    for s in s_values:
        call, put = black_scholes(s, k, r, t, sigma)

        if option_type == "call":
            prices.append(call)
        elif option_type == "put":
            prices.append(put)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    return np.array(prices)


def volatility_scenarios(
    sigma_values,
    s,
    k,
    r,
    t,
    option_type,
):
    prices = []

    for sigma in sigma_values:
        call, put = black_scholes(s, k, r, t, sigma)

        if option_type == "call":
            prices.append(call)
        elif option_type == "put":
            prices.append(put)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    return np.array(prices)


def greek_scenarios(
    s_values,
    k,
    r,
    t,
    sigma,
    option_type,
):
    results = {
        "delta": [],
        "gamma": [],
        "vega": [],
        "theta": [],
        "rho": [],
    }

    for s in s_values:
        results["delta"].append(
            delta(s, k, r, t, sigma, option_type)
        )
        results["gamma"].append(
            gamma(s, k, r, t, sigma)
        )
        results["vega"].append(
            vega(s, k, r, t, sigma)
        )
        results["theta"].append(
            theta(s, k, r, t, sigma, option_type)
        )
        results["rho"].append(
            rho(s, k, r, t, sigma, option_type)
        )

    for key in results:
        results[key] = np.array(results[key])

    return results