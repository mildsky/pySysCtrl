import numpy as np
from math import sin, cos, pi, tanh, atan2
import nlopt

def euler2Quaternion(r, p, y):
    return np.array([cos(r/2)*cos(p/2)*cos(y/2) + sin(r/2)*sin(p/2)*sin(y/2),
                    sin(r/2)*cos(p/2)*cos(y/2) - cos(r/2)*sin(p/2)*sin(y/2),
                    cos(r/2)*sin(p/2)*cos(y/2) + sin(r/2)*cos(p/2)*sin(y/2),
                    cos(r/2)*cos(p/2)*sin(y/2) - sin(r/2)*sin(p/2)*cos(y/2)])


# error is 3D vector of position
# return z force and desired attitude
class PositionController:
    def __init__(self, Kp=1, Kd=0.1, Ki=0.1):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.error = 0
        self.error_dot = 0
        self.error_sum = 0
    def control(self, error, dt):
        self.error_sum += error * dt
        self.error_sum = np.clip(self.error_sum, -3, 3)
        self.error_dot = (error - self.error) / dt
        self.error = error
        # print(f"p term: {self.Kp * error}")
        # print(f"d term: {self.Kd * self.error_dot}")
        # print(f"i term: {self.Ki * self.error_sum}")
        cmdAcc = self.Kp * error + self.Kd * self.error_dot + self.Ki * self.error_sum
        # cmdAcc = np.tanh(cmdAcc)
        cmdAcc[2] = cmdAcc[2] + 9.81
        desiredRoll = atan2(cmdAcc[1], cmdAcc[2])
        desiredPitch = atan2(-cmdAcc[0], cmdAcc[2])
        # desiredRoll = cmdAcc[1] * pi/2
        # desiredPitch = -cmdAcc[0] * pi/2
        desiredYaw = 0
        attiDesired = euler2Quaternion(desiredRoll, desiredPitch, desiredYaw)
        return cmdAcc[2], attiDesired

class ModelPredictivePositionController:
    def __init__(self, model, Q, R, N, dt):
        '''
        model: function take state and control input, return next state \\
        Q: state cost matrix \\
        R: control cost matrix \\
        N: prediction horizon \\
        dt: time step \\
        '''
        self.model = model
        self.Q = Q
        self.R = R
        self.N = N
        self.dt = dt
    def control(self, state, hint=None):
        '''
        state: current state \\
        hint: initial guess of control input \\
        return Fz, Mx, My, Mz
        '''
        u = np.zeros((4,self.N))
        if hint:
            u = hint
        x = np.copy(state)
        cost = 0
        def objective(u, grad):
            x = x
            horizon = len(u)
            cost = 0
            for i in range(horizon):
                x = self.model(x, u[i])
                cost +=  x.T @ self.Q @ x + u[i].T @ self.R @ u[i]
            return cost
        for i in range(self.N):
            cost += x.T @ self.K @ x + u.T @ self.Q @ u
            x = self.model(x, u)