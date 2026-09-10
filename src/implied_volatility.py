from scipy.optimize import brentq

from src.black_scholes import black_scholes


def implied_volatility(s, k, r, t, market_price, option_type):
    def objective(sigma):
        call, put = black_scholes(s, k, r, t, sigma)

        if option_type == "call":
            model_price = call
        elif option_type == "put":
            model_price = put
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return model_price - market_price

    return brentq(objective, 1e-6, 5.0)