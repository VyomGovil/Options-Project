import numpy as np
from scipy.stats import norm

#k = strike price
#s = current stock price
#t = time to expiry
#r = rist free rate
#sigma = volitality of stock
def black_scholes(s, k, r, t, sigma):
    d2 = (np.log(s/k) + (r - 0.5 * sigma**2)*t) / (sigma * np.sqrt(t))
    d1 = d2 + sigma * np.sqrt(t)
    
    call = s * norm.cdf(d1) - k * np.exp(-r*t) * norm.cdf(d2)
    put = k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
    return call, put


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.20

call, put = black_scholes(S, K, r, T, sigma)

print("Call:", call)
print("Put:", put)