from math import floor
from scipy import integrate
import numpy as np

def lorentz_deriv(t0, X, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorenz system."""
    (x, y, z) = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
# end

def lorenz_equ(x0=[-20, 10, -.5], begin=0, end=60, timesteps=60000, train_per=.66, clip=0):
    """Use solve_ivp to produce a solution to the lorenz equations"""
    t = np.linspace(begin,end,timesteps)
    clipped_start = floor(timesteps * clip / (end - begin))
    n_train = floor(train_per * timesteps * (end - clip) / (end - begin))
    train_t = t[clipped_start:n_train]
    test_t = t[n_train:]
    u = integrate.solve_ivp(lorentz_deriv, (begin,end), x0, dense_output=True).sol
    return train_t, test_t, u
