import numpy as np
from math import sin, cos, pi

# state is 3D vector of angular velocity measurement
# desired is also 3D vector of angular velocity
# return 3D vector of torque
class RateController:
    def __init__(self, Kp=1, Kd=0.001, Ki=0.2):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.error = 0
        self.error_dot = 0
        self.error_sum = 0
    def control(self, error, dt):
        self.error_sum += error * dt
        self.error_dot = (error - self.error) / dt
        self.error = error
        # print(f"p term: {self.Kp * error}")
        # print(f"d term: {self.Kd * self.error_dot}")
        # print(f"i term: {self.Ki * self.error_sum}")
        return self.Kp * error + self.Kd * self.error_dot + self.Ki * self.error_sum