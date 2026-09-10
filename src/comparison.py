import numpy as np

from src.black_scholes import black_scholes
from src.monte_carlo import monte_carlo_price


def compare_prices(
    s,
    k,
    r,
    t,
    sigma,
    option_type,
    n_simulations=100_000,
    seed=None,
    ):
    call, put = black_scholes(s, k, r, t, sigma)

    if option_type == "call":
        bs_price = call
    elif option_type == "put":
        bs_price = put
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    mc_price, standard_error = monte_carlo_price(
        s,
        k,
        r,
        t,
        sigma,
        option_type,
        n_simulations,
        seed,
    )

    confidence_interval = (
        mc_price - 1.96 * standard_error,
        mc_price + 1.96 * standard_error,
    )

    return {
        "black_scholes": bs_price,
        "monte_carlo": mc_price,
        "standard_error": standard_error,
        "confidence_interval": confidence_interval,
        "difference": mc_price - bs_price,
    }
    
def monte_carlo_convergence(
    s,
    k,
    r,
    t,
    sigma,
    option_type,
    simulation_counts,
    seed=None,
    ):
    results = []

    for n_simulations in simulation_counts:
        mc_price, standard_error = monte_carlo_price(
            s,
            k,
            r,
            t,
            sigma,
            option_type,
            n_simulations,
            seed,
        )

        results.append({
            "n_simulations": n_simulations,
            "monte_carlo_price": mc_price,
            "standard_error": standard_error,
        })

    return results