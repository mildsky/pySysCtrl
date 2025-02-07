import numpy as np

# euler integration
# x_{k+1} = x_k + x'_k * dt

# runge-kutta 4th order
# k1 = f(x_k, u_k)
# k2 = f(x_k + k1 * dt / 2, u_k)
# k3 = f(x_k + k2 * dt / 2, u_k)
# k4 = f(x_k + k3 * dt, u_k)
# x_{k+1} = x_k + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
