import numpy as np


def monte_carlo_price(
    s,
    k,
    r,
    t,
    sigma,
    option_type,
    n_simulations=100_000,
    seed=None,
):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    rng = np.random.default_rng(seed)

    z = rng.standard_normal(n_simulations)

    st = s * np.exp(
        (r - 0.5 * sigma**2) * t
        + sigma * np.sqrt(t) * z
    )

    if option_type == "call":
        payoff = np.maximum(st - k, 0)
    else:
        payoff = np.maximum(k - st, 0)

    discounted_payoff = np.exp(-r * t) * payoff

    price = np.mean(discounted_payoff)

    standard_error = (
        np.std(discounted_payoff, ddof=1)
        / np.sqrt(n_simulations)
    )

    return price, standard_error