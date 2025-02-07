from math import pi, sin, cos
import pybullet as p
import pybullet_data
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from RateControl import RateController
from AttitudeControl import AttitudeController
from PositionControl import PositionController

## pybullet keyboard shortcuts
# g: toggle debug gui
# s: toggle shade
# w: toggle wireframe
class BulletQuadrotor:
    def __init__(self, position=[0, 0, 0.1], orientation=[0, 0, 0, 1], headless=False):
        self.physicsClient = p.connect(p.DIRECT) if headless else p.connect(p.GUI)
        p.setTimeStep(1./500.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        startPos = position
        startOrientation = orientation
        self.robotID = p.loadURDF("./Quadrotor/quadrotor.urdf", startPos, startOrientation)
        self.state = p.getLinkState(self.robotID, 0)
        self.twist = p.getBaseVelocity(self.robotID)
        self.d = 0.07
        self.cf = 10
        self.ct = 0.06
        self.dynamics = np.matrix([
            [self.cf, self.cf, self.cf, self.cf],
            [-self.d*self.cf, self.d*self.cf, self.d*self.cf, -self.d*self.cf],
            [-self.d*self.cf, self.d*self.cf, -self.d*self.cf, self.d*self.cf],
            [-self.ct, -self.ct, self.ct, self.ct]
        ])
    def step(self, throttle):
        wrench = self.dynamics @ np.matrix(throttle).transpose()
        zForce = wrench[0,0]
        torque = wrench[1:4,0]
        p.applyExternalForce(self.robotID, -1, [0, 0, zForce], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(self.robotID, -1, torque, p.LINK_FRAME)
        # p.applyExternalForce(self.robotID, 0, [0, 0, self.cf*throttle[0]], [0, 0, 0], p.LINK_FRAME)
        # p.applyExternalForce(self.robotID, 1, [0, 0, self.cf*throttle[1]], [0, 0, 0], p.LINK_FRAME)
        # p.applyExternalForce(self.robotID, 2, [0, 0, self.cf*throttle[2]], [0, 0, 0], p.LINK_FRAME)
        # p.applyExternalForce(self.robotID, 3, [0, 0, self.cf*throttle[3]], [0, 0, 0], p.LINK_FRAME)
        # p.applyExternalTorque(self.robotID, 0, [0, 0, -self.ct*throttle[0]], p.LINK_FRAME)
        # p.applyExternalTorque(self.robotID, 1, [0, 0, -self.ct*throttle[1]], p.LINK_FRAME)
        # p.applyExternalTorque(self.robotID, 2, [0, 0, self.ct*throttle[2]], p.LINK_FRAME)
        # p.applyExternalTorque(self.robotID, 3, [0, 0, self.ct*throttle[3]], p.LINK_FRAME)
        # p.applyExternalTorque(self.robotID, -1, 0.4*np.random.randn(3), p.LINK_FRAME)
        p.stepSimulation()
    def getPosition(self):
        '''return 3D vector of position'''
        return np.array(p.getBasePositionAndOrientation(self.robotID)[0])
    def getOrientation(self):
        '''return quaternion in wxyz format'''
        orient = p.getBasePositionAndOrientation(self.robotID)[1]
        return np.array([orient[3], orient[0], orient[1], orient[2]])
    def getLinearVelocityBodyFrame(self):
        quaternion = p.getBasePositionAndOrientation(self.robotID)[1]
        linear_velocity_world = np.array(p.getBaseVelocity(self.robotID)[0])
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        linear_velocity_body = np.dot(rotation_matrix.T, linear_velocity_world)
        return linear_velocity_body
    def getAngularVelocityBodyFrame(self):
        quaternion = p.getBasePositionAndOrientation(self.robotID)[1]
        angular_velocity_world = np.array(p.getBaseVelocity(self.robotID)[1])
        rotation_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        angular_velocity_body = np.dot(rotation_matrix.T, angular_velocity_world)
        return angular_velocity_body
    def compensateInput(self, forceThrottle, torqueThrottle):
        return forceThrottle + torqueThrottle - np.max(torqueThrottle) * np.ones_like(torqueThrottle)
    def controlAllocation(self, wrench):
        throttle = np.linalg.pinv(self.dynamics) @ wrench
        throttle = np.clip(throttle, -1, 1)
        return throttle

def euler2Quaternion(r, p, y):
    return np.array([cos(r/2)*cos(p/2)*cos(y/2) + sin(r/2)*sin(p/2)*sin(y/2),
                    sin(r/2)*cos(p/2)*cos(y/2) - cos(r/2)*sin(p/2)*sin(y/2),
                    cos(r/2)*sin(p/2)*cos(y/2) + sin(r/2)*cos(p/2)*sin(y/2),
                    cos(r/2)*cos(p/2)*sin(y/2) - sin(r/2)*sin(p/2)*cos(y/2)])

def quaternion2Euler(q):
    return np.array([
        np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2)),
        np.arcsin(2*(q[0]*q[2] - q[3]*q[1])),
        np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    ])

def quaternionConjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternionError(q1, q2):
    q2 = quaternionConjugate(q2)
    return np.array([
        q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3],
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]
    ])

if __name__ == "__main__":
    robot = BulletQuadrotor(headless=False)
    time.sleep(1)
    dt = 1./500.
    tmax = 5
    t = np.arange(0, tmax, dt)
    x = np.matrix(np.zeros((3, len(t))))
    rateCtrlr = RateController(Kp=3, Kd=0.001, Ki=0.01)
    attiCtrlr = AttitudeController(Kp=3, Kd=0.001, Ki=0)
    posiCtrlr = PositionController(Kp=15, Kd=6, Ki=3)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    trajectory = []
    atti_traj = []
    for i in range(int(tmax/dt)):
        posiDesired = np.array([sin(i*dt), cos(i*dt), 1])
        trajectory.append(posiDesired)
        # position controller
        posiMeas = robot.getPosition()
        x[:,i] = np.reshape(posiMeas, (3,1))
        posiError = posiDesired - posiMeas
        zforce, attiDesired = posiCtrlr.control(posiError, dt)
        attiDesired = euler2Quaternion(0, -pi/6, 0)
        # attitude controller
        attiMeas = robot.getOrientation()
        atti_traj.append(quaternion2Euler(attiMeas))
        attiError = quaternionError(attiDesired, attiMeas)
        eulerAttiError = quaternion2Euler(attiError)
        rateDesired = attiCtrlr.control(eulerAttiError, dt)
        # rate controller
        rateMeas = robot.getAngularVelocityBodyFrame()
        rateError = rateDesired - rateMeas
        torque = rateCtrlr.control(rateError, dt)
        throttle = robot.controlAllocation(np.array([zforce, torque[0], torque[1], torque[2]]))
        robot.step(throttle)
        time.sleep(dt)
    p.disconnect()
    attiDesired = euler2Quaternion(0, pi/6, 0)
    # plt.plot(t, attiDesired[1]*np.ones_like(t), 'k--')
    pitch_traj = np.array(atti_traj)[:,1]
    plt.plot(t, pitch_traj, 'k')
    # plt.plot(t, x[0,:].transpose(), 'r')
    # plt.plot(t, x[1,:].transpose(), 'g')
    # plt.plot(t, x[2,:].transpose(), 'b')
    # plt.plot(t, np.array(trajectory)[:,0], 'r--')
    # plt.plot(t, np.array(trajectory)[:,1], 'g--')
    # plt.plot(t, np.array(trajectory)[:,2], 'b--')
    # plt.plot(t, np.ones_like(t) * posiDesired[0], 'r--')
    # plt.plot(t, np.ones_like(t) * posiDesired[1], 'g--')
    # plt.plot(t, np.ones_like(t) * posiDesired[2], 'b--')
    plt.xlabel('Time (s)')
    # plt.ylabel('Position (m)')
    plt.ylabel('Pitch (rad)')
    plt.show()
