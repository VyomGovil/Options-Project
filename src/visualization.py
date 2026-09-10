import matplotlib.pyplot as plt

from src.scenario_analysis import price_scenarios, volatility_scenarios
from src.comparison import monte_carlo_convergence
from src.black_scholes import black_scholes

def plot_price_vs_stock(
    s_values,
    k,
    r,
    t,
    sigma,
    option_type="call",
):
    prices = price_scenarios(
        s_values,
        k,
        r,
        t,
        sigma,
        option_type,
    )

    plt.figure()
    plt.plot(s_values, prices)
    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.title(f"{option_type.capitalize()} Price vs Stock Price")
    plt.grid(True)
    plt.show()


def plot_price_vs_volatility(
    sigma_values,
    s,
    k,
    r,
    t,
    option_type="call",
):
    prices = volatility_scenarios(
        sigma_values,
        s,
        k,
        r,
        t,
        option_type,
    )

    plt.figure()
    plt.plot(sigma_values, prices)
    plt.xlabel("Volatility")
    plt.ylabel("Option Price")
    plt.title(f"{option_type.capitalize()} Price vs Volatility")
    plt.grid(True)
    plt.show()


def plot_monte_carlo_convergence(
    s,
    k,
    r,
    t,
    sigma,
    option_type,
    simulation_counts,
    seed=None,
):
    results = monte_carlo_convergence(
        s,
        k,
        r,
        t,
        sigma,
        option_type,
        simulation_counts,
        seed,
    )

    call, put = black_scholes(s, k, r, t, sigma)

    if option_type == "call":
        bs_price = call
    else:
        bs_price = put

    simulations = [
        result["n_simulations"]
        for result in results
    ]

    mc_prices = [
        result["monte_carlo_price"]
        for result in results
    ]

    standard_errors = [
        result["standard_error"]
        for result in results
    ]

    lower = [
        price - 1.96 * error
        for price, error in zip(mc_prices, standard_errors)
    ]

    upper = [
        price + 1.96 * error
        for price, error in zip(mc_prices, standard_errors)
    ]

    plt.figure()
    plt.plot(simulations, mc_prices, marker="o")
    plt.axhline(bs_price, linestyle="--")
    plt.fill_between(simulations, lower, upper, alpha=0.2)

    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price")
    plt.title("Monte Carlo Convergence")
    plt.xscale("log")
    plt.grid(True)
    plt.show()