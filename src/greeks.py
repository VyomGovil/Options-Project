import numpy as np
from scipy.stats import norm


def _d1(s, k, r, t, sigma):
    return (
        np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def delta(s, k, r, t, sigma, option_type):
    d1 = _d1(s, k, r, t, sigma)

    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(s, k, r, t, sigma):
    d1 = _d1(s, k, r, t, sigma)

    return norm.pdf(d1) / (s * sigma * np.sqrt(t))


def vega(s, k, r, t, sigma):
    d1 = _d1(s, k, r, t, sigma)

    return s * norm.pdf(d1) * np.sqrt(t)


def theta(s, k, r, t, sigma, option_type):
    d1 = _d1(s, k, r, t, sigma)
    d2 = d1 - sigma * np.sqrt(t)

    first_term = -(s * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))

    if option_type == "call":
        return first_term - r * k * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        return first_term + r * k * np.exp(-r * t) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(s, k, r, t, sigma, option_type):
    d2 = _d1(s, k, r, t, sigma) - sigma * np.sqrt(t)

    if option_type == "call":
        return k * t * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        return -k * t * np.exp(-r * t) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")