import numpy as np
from MassSpringDamper import MassSpringDamper
import matplotlib.pyplot as plt

def PIDControl(m=1, k=1, b=1, kp=1, kd=1, ki=1, draw=True):
    dt = 0.01
    t = np.arange(0, 10, dt)
    x = np.zeros_like(t)
    msd = MassSpringDamper(m, k, b, 0)
    pos_desired = 10
    pos_error_prev = 0
    pos_error_sum = 0
    for i in range(len(t)):
        pos_error = pos_desired - msd.position
        pos_error_sum += pos_error
        pos_error_diff = pos_error - pos_error_prev
        force = kp * pos_error + ki * pos_error_sum * dt + kd * pos_error_diff / dt
        pos_error_prev = pos_error
        x[i] = msd.step(force, dt)
    if draw:
        plt.plot(t, x)
        plt.plot(t, np.ones_like(t) * pos_desired, 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('PID Control')
        plt.show()
    return x

def PIDTrackingControl(m=1, k=1, b=1, kp=1, kd=1, ki=1, draw=True):
    dt = 0.01
    t = np.arange(0, 20, dt)
    x = np.zeros_like(t)
    msd = MassSpringDamper(m, k, b, 0)
    pos_desired = np.sin(t) * 10
    pos_error_prev = 0
    pos_error_sum = 0
    for i in range(len(t)):
        pos_error = pos_desired[i] - msd.position
        pos_error_sum += pos_error
        pos_error_diff = pos_error - pos_error_prev
        force = kp * pos_error + ki * pos_error_sum * dt + kd * pos_error_diff / dt
        pos_error_prev = pos_error
        x[i] = msd.step(force, dt)
    if draw:
        plt.plot(t, x)
        plt.plot(t, np.ones_like(t) * pos_desired, 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('PID Control')
        plt.show()
    return x

if __name__ == "__main__":
    PIDControl(1, 1, 0, 4, 3, 3)
    PIDTrackingControl(1, 1, 0, 4, 3, 3)