import numpy as np

class Quadrotor:
    def __init__(self) -> None:
        self.mass = 1
        self.gravity = 9.81
        self.inertia = np.eye(3)/1000