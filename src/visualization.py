import matplotlib.pyplot as plt

from src.scenario_analysis import (
    price_scenarios,
    volatility_scenarios,
    greek_scenarios,
)
from src.comparison import monte_carlo_convergence
from src.black_scholes import black_scholes
from src.implied_volatility import implied_volatility

def plot_price_vs_stock(
    s_values,
    k,
    r,
    t,
    sigma,
    option_type="call",
    filename=None,
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

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()


def plot_price_vs_volatility(
    sigma_values,
    s,
    k,
    r,
    t,
    option_type="call",
    filename=None,
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

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()


def plot_greeks_vs_stock(
    s_values,
    k,
    r,
    t,
    sigma,
    option_type="call",
    filename=None,
):
    results = greek_scenarios(
        s_values,
        k,
        r,
        t,
        sigma,
        option_type,
    )

    fig, axes = plt.subplots(5, 1, figsize=(8, 12))

    for ax, (name, values) in zip(axes, results.items()):
        ax.plot(s_values, values)
        ax.set_ylabel(name.capitalize())
        ax.grid(True)

    axes[-1].set_xlabel("Stock Price")

    fig.suptitle(
        f"{option_type.capitalize()} Greeks vs Stock Price"
    )

    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_monte_carlo_convergence(
    s,
    k,
    r,
    t,
    sigma,
    option_type,
    simulation_counts,
    seed=None,
    filename=None,
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

    call, put = black_scholes(
        s,
        k,
        r,
        t,
        sigma,
    )

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
        for price, error in zip(
            mc_prices,
            standard_errors,
        )
    ]

    upper = [
        price + 1.96 * error
        for price, error in zip(
            mc_prices,
            standard_errors,
        )
    ]

    plt.figure()
    plt.plot(
        simulations,
        mc_prices,
        marker="o",
        label="Monte Carlo",
    )
    plt.axhline(
        bs_price,
        linestyle="--",
        label="Black-Scholes",
    )
    plt.fill_between(
        simulations,
        lower,
        upper,
        alpha=0.2,
        label="95% CI",
    )

    plt.xlabel("Number of Simulations")
    plt.ylabel("Option Price")
    plt.title("Monte Carlo Convergence")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()
    
def plot_call_put_vs_stock(
    s_values,
    k,
    r,
    t,
    sigma,
    filename=None,
):
    call_prices = []
    put_prices = []

    for s in s_values:
        call, put = black_scholes(
            s,
            k,
            r,
            t,
            sigma,
        )

        call_prices.append(call)
        put_prices.append(put)

    plt.figure()

    plt.plot(
        s_values,
        call_prices,
        label="Call",
    )

    plt.plot(
        s_values,
        put_prices,
        label="Put",
    )

    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.title("Call and Put Prices vs Stock Price")
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()

def plot_implied_volatility_skew(
    strikes,
    s,
    r,
    t,
    sigma,
    option_type="put",
    filename=None,
):
    market_prices = []

    for strike in strikes:
        call, put = black_scholes(
            s,
            strike,
            r,
            t,
            sigma,
        )

        if option_type == "call":
            price = call
        elif option_type == "put":
            price = put
        else:
            raise ValueError(
                "option_type must be 'call' or 'put'"
            )

        # Synthetic skew:
        # increase implied volatility for lower strikes
        skew_adjustment = (
            0.10 * max((s - strike) / s, 0)
        )

        synthetic_sigma = sigma + skew_adjustment

        call, put = black_scholes(
            s,
            strike,
            r,
            t,
            synthetic_sigma,
        )

        if option_type == "call":
            price = call
        else:
            price = put

        market_prices.append(price)

    implied_vols = []

    for strike, market_price in zip(
        strikes,
        market_prices,
    ):
        iv = implied_volatility(
            s,
            strike,
            r,
            t,
            market_price,
            option_type,
        )

        implied_vols.append(iv)

    plt.figure()

    plt.plot(
        strikes,
        implied_vols,
        marker="o",
    )

    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.title(
        "Synthetic Implied Volatility Skew"
    )
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()