import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import xml.dom.minidom

np.set_printoptions(precision=3, suppress=True)

class Rotor:
    def __init__(self, position, orientation, cf=1, ct=0.1, range=[1, -1]):
        self.position = position        # 3D vector, position of rotor
        self.orientation = orientation  # 3D vector, normal of rotor plane
        self.cf = cf
        self.ct = ct
        self.range = range

class Copter:
    def __init__(self):
        self.rotors = []
    def getAllocationMatrix(self):
        '''
        allocation matrix is conversion from rotor throttle to body wrench
        body wrench = allocation matrix * rotor throttle
        therefore, size of allocation matrix is 6 by n_rotors
        '''
        columns = []
        for rotor in self.rotors:
            forceVec = rotor.orientation * rotor.cf
            torqueVec = np.cross(rotor.position, forceVec) + rotor.orientation * rotor.ct
            column = np.hstack([forceVec, torqueVec])
            columns.append(column)
        allocation_matrix = np.vstack(columns)
        return allocation_matrix.T
    def visualizeWrenchSpace(self):
        allocation_matrix = self.getAllocationMatrix()
        n_rotors = allocation_matrix.shape[1]
        n_points = 1000
        throttle_samples = []
        for _ in range(n_points):
            throttle_samples.append(2 * np.random.beta(0.1, 0.1, n_rotors) - 1)
        throttle_samples = np.array(throttle_samples)
        wrench_samples = np.dot(allocation_matrix, throttle_samples.T)
        # visualize force space
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(wrench_samples[0, :], wrench_samples[1, :], wrench_samples[2, :])
        # ax.plot([-1, 1], [0, 0], [0, 0], color='r', linewidth=1, label="X-axis")
        # ax.plot([0, 0], [-1, 1], [0, 0], color='g', linewidth=1, label="Y-axis")
        # ax.plot([0, 0], [0, 0], [-1, 1], color='b', linewidth=1, label="Z-axis")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # visualize torque space
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(wrench_samples[3, :], wrench_samples[4, :], wrench_samples[5, :])
        # ax.plot([-1, 1], [0, 0], [0, 0], color='r', linewidth=1, label="X-axis")
        # ax.plot([0, 0], [-1, 1], [0, 0], color='g', linewidth=1, label="Y-axis")
        # ax.plot([0, 0], [0, 0], [-1, 1], color='b', linewidth=1, label="Z-axis")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    def mjcf(self, asset: ET.ElementTree = None):
        _root = ET.Element('mujoco')
        _compiler = ET.SubElement(_root, 'compiler')
        _compiler.attrib["angle"] = "radian"
        _option = ET.SubElement(_root, 'option')
        _option.attrib["timestep"] = f"{1/1000}"
        _asset = ET.SubElement(_root, 'asset')
        if asset is not None:
            _asset.append(asset)
        _contact = ET.SubElement(_root, 'contact')
        _worldbody = ET.SubElement(_root, 'worldbody')
        _actuator = ET.SubElement(_root, 'actuator')
        _sensor = ET.SubElement(_root, 'sensor')
        flatxml = ET.tostring(_root, encoding="unicode")
        dom = xml.dom.minidom.parseString(flatxml)
        pretty_xml = dom.toprettyxml(indent="    ")
        return pretty_xml

class TiltCopter(Copter):
    '''
    TiltCopter is a subclass of Copter
    TiltCopter can tilt its rotors
    each rotors now have two additional parameters: tilt axes and tilt ranges
    '''

def quadrotor():
    copter = Copter()
    copter.rotors.append(Rotor(np.array([1, 1, 0]), np.array([0, 0, 1]), ct=0.1))
    copter.rotors.append(Rotor(np.array([-1, 1, 0]), np.array([0, 0, 1]), ct=-0.1))
    copter.rotors.append(Rotor(np.array([-1, -1, 0]), np.array([0, 0, 1]), ct=0.1))
    copter.rotors.append(Rotor(np.array([1, -1, 0]), np.array([0, 0, 1]), ct=-0.1))
    # allocation_matrix = copter.getAllocationMatrix()
    # print(allocation_matrix)
    # print(np.linalg.pinv(allocation_matrix))
    # copter.visualizeWrenchSpace()
    color_asset = ET.Element('asset')
    ET.SubElement(color_asset, 'texture', attrib={'name': 'grid', 'type': '2d', 'builtin': 'checker', 'width': '512', 'height': '512', 'rgb1': '.1 .2 .3', 'rgb2': '.2 .3 .4'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'grid', 'texture': 'grid', 'texrepeat': '1 1', 'texuniform': 'true', 'reflectance': '0'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'skyblue', 'rgba': '0.64 0.835 0.97 1'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'gray', 'rgba': '0.63 0.61 0.615 1'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'pink', 'rgba': '0.89 0.61 0.61 1'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'blue', 'rgba': '0.18 0.33 0.92 1'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'green', 'rgba': '0.7 0.84 0.725 0.8'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'purple', 'rgba': '0.83 0.78 0.93 0.8'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'orange', 'rgba': '0.95 0.65 0.31 1'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'black', 'rgba': '0 0 0 1'})
    ET.SubElement(color_asset, 'material', attrib={'name': 'white', 'rgba': '1 1 1 1'})
    print(copter.mjcf(asset=color_asset))

def omnicopter():
    copter = Copter()
    raise NotImplementedError

if __name__ == "__main__":
    quadrotor()
