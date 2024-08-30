import numpy as np
import matplotlib.pyplot as plt

class MassSpringDamper:
    mass = 1
    spring = 1
    damper = 1
    def __init__(self, m=1, k=1, b=1, p=0, v=0):
        self.mass = m
        self.spring = k
        self.damper = b
        self.position = p
        self.velocity = v
        self.acceleration = 0
    def step(self, force, dt):
        self.acceleration = (force - self.damper * self.velocity - self.spring * self.position) / self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        return self.position

def noControl(m=1, k=1, b=1, draw=True):
    dt = 0.01
    t = np.arange(0, 10, dt)
    x = np.zeros_like(t)
    msd = MassSpringDamper(m, k, b, 10)
    for i in range(len(t)):
        x[i] = msd.step(0, dt)
    if draw:
        plt.plot(t, x)
        plt.plot(t, np.zeros_like(t), 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('No Control')
        plt.show()
    return x

if __name__ == "__main__":
    noControl(1,2,2)
