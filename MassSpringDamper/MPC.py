import numpy as np
from MassSpringDamper import MassSpringDamper
import matplotlib.pyplot as plt
import nlopt
import random
import jax
import jax.numpy as jnp
import autograd
import time

def ModelPredictiveControl(m=1, k=1, b=1, draw=True):
    dt = 0.01
    t = np.arange(0, 10, dt)
    x = np.zeros_like(t)
    msd = MassSpringDamper(m, k, b, 0)
    pos_desired = 10
    def model(p, v, u, dt):
        a = (u - b * v - k * p) / m
        v += a * dt
        p += v * dt
        return p, v
    def objective(u, grad):
        current_state = [msd.position, msd.velocity]
        horizon = len(u)
        Q = 1000
        R = 0.01
        cost = 0
        for i in range(horizon):
            current_state = model(current_state[0], current_state[1], u[i], dt)
            cost +=  Q * (pos_desired - current_state[0]) ** 2 + R * u[i] ** 2
        if grad.size > 0:
            # for i in range(horizon):
            #     grad[i] = 2 * horizon * u[i] # not precise
            ## numerical gradient
            epsilon = 1e-6
            for j in range(horizon):
                u_eps = np.array(u)
                u_eps[j] += epsilon
                current_state_eps = [msd.position, msd.velocity]
                cost_eps = 0
                for k in range(horizon):
                    current_state_eps = model(current_state_eps[0], current_state_eps[1], u_eps[k], dt)
                    cost_eps += Q * (pos_desired - current_state_eps[0]) ** 2 + R * u_eps[k] ** 2
                grad[j] = (cost_eps - cost) / epsilon
        return cost
    # MPC
    for i in range(len(t)):
        # objective function
        horizon = 16
        opt = nlopt.opt(nlopt.LD_MMA, horizon)
        opt.set_min_objective(objective)
        opt.set_xtol_rel(1e-4)
        opt.set_lower_bounds([-1000] * horizon)
        opt.set_upper_bounds([1000] * horizon)
        u_init = [random.uniform(-1, 1) for _ in range(horizon)]  # Initial guess for control inputs
        u = opt.optimize(u_init)
        force = u[0]
        x[i] = msd.step(force, dt)
    if draw:
        plt.plot(t, x)
        plt.plot(t, np.ones_like(t) * pos_desired, 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Model Predictive Control')
        plt.show()
    return x

## autograd version
def autogradMPC(m=1, k=1, b=1, draw=True):
    dt = 0.01
    t = np.arange(0, 10, dt)
    x = np.zeros_like(t)
    msd = MassSpringDamper(m, k, b, 0)
    pos_desired = 10
    def model(p, v, u, dt):
        a = (u - b * v - k * p) / m
        v += a * dt
        p += v * dt
        return p, v
    def objective(u):
        current_state = [msd.position, msd.velocity]
        horizon = len(u)
        Q = 1000
        R = 0.01
        cost = 0
        for i in range(horizon):
            current_state = model(current_state[0], current_state[1], u[i], dt)
            cost +=  Q * (pos_desired - current_state[0]) ** 2 + R * u[i] ** 2
        return cost
    def autograd_objective(u, grad):
        grad = autograd.grad(objective)(u)
        return objective(u)
    # MPC
    for i in range(len(t)):
        # objective function
        horizon = 16
        opt = nlopt.opt(nlopt.LD_MMA, horizon)
        opt.set_min_objective(autograd_objective)
        opt.set_xtol_rel(1e-4)
        opt.set_lower_bounds([-1000] * horizon)
        opt.set_upper_bounds([1000] * horizon)
        u_init = [random.uniform(-1, 1) for _ in range(horizon)]  # Initial guess for control inputs
        u = opt.optimize(u_init)
        force = u[0]
        x[i] = msd.step(force, dt)
    if draw:
        plt.plot(t, x)
        plt.plot(t, np.ones_like(t) * pos_desired, 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Model Predictive Control')
        plt.show()
    return x

## JAX version
def JaxMPC(m=1, k=1, b=1, draw=True):
    dt = 0.01
    t = np.arange(0, 10, dt)
    x = np.zeros_like(t)
    msd = MassSpringDamper(m, k, b, 0)
    pos_desired = 10
    def model(p, v, u, dt):
        a = (u - b * v - k * p) / m
        v += a * dt
        p += v * dt
        return p, v
    def objective(u):
        current_state = [msd.position, msd.velocity]
        horizon = len(u)
        Q = 1000
        R = 0.001
        cost = 0
        for i in range(horizon):
            current_state = model(current_state[0], current_state[1], u[i], dt)
            cost +=  Q * (pos_desired - current_state[0]) ** 2 + R * u[i] ** 2
        return cost
    compiled_grad = jax.jit(jax.grad(objective))
    # MPC
    for i in range(len(t)):
        # objective function
        def nlopt_objective(u, g):
            cost = objective(u)
            if g.size > 0:
                g[:] = np.array(compiled_grad(u))
            return cost
        horizon = 16
        # optimizer settings
        opt = nlopt.opt(nlopt.LD_MMA, horizon)
        opt.set_min_objective(nlopt_objective)  # Using JAX gradient
        opt.set_xtol_rel(1e-4)
        opt.set_lower_bounds([-100] * horizon)
        opt.set_upper_bounds([100] * horizon)
        u = opt.optimize(np.array([random.uniform(-1, 1) for _ in range(horizon)]))
        force = u[0]
        x[i] = msd.step(force, dt)
    if draw:
        plt.plot(t, x)
        plt.plot(t, np.ones_like(t) * pos_desired, 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Model Predictive Control')
        plt.show()
    return x

# tested on M2 MacBook Air
# MPC time:         7.536072254180908
# autograd time:    5.152461051940918
# jax time:         1.274223804473877
if __name__ == "__main__":
    t = time.time()
    ModelPredictiveControl(1,1,0, draw=False)
    print('Elapsed time: ', time.time() - t)
    t = time.time()
    autogradMPC(1,1,0, draw=False)
    print('Elapsed time: ', time.time() - t)
    t = time.time()
    JaxMPC(1,1,0, draw=False)
    print('Elapsed time: ', time.time() - t)
