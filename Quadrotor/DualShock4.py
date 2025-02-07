#!/usr/bin/env python
import multiprocessing
import pygame
import os
import time

# pygame 2.6.0
# (SDL 2.28.4, Python 3.11.4)
# DualShock4 wrapper without display window and with multiprocessing
class DualShock4:
    recv = {}
    def __init__(self):
        self.__pipe = multiprocessing.Pipe()
        self.__process = multiprocessing.Process(target=self.runner, args=(self.__pipe[1],))
    def open(self):
        self.__process.start()
    def close(self):
        self.__pipe[0].send(-1)
        self.__process.join()
    def runner(self, conn):
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("Waiting for joystick...")
            while True:
                found = False
                for event in pygame.event.get():
                    if event.type == pygame.JOYDEVICEADDED:
                        print("Joystick found")
                        found = True
                if found:
                    break
        print("Joystick count:"+str(pygame.joystick.get_count()))
        joystick_number = 0
        for x in range(pygame.joystick.get_count()):
            print(f"Joystick {x}: {pygame.joystick.Joystick(x).get_name()}")
            if pygame.joystick.Joystick(x).get_name() == "Wireless Controller":
                joystick_number = x
        joystick = pygame.joystick.Joystick(joystick_number)
        joystick.init()
        self.buttons = [False] * joystick.get_numbuttons()
        self.axis = [0] * joystick.get_numaxes()
        self.hat = [(0,0)] * joystick.get_numhats()
        while True:
            for event in pygame.event.get():
                # read button click
                if event.type == pygame.JOYBUTTONDOWN:
                    self.buttons[event.button] = True
                elif event.type == pygame.JOYBUTTONUP:
                    self.buttons[event.button] = False
                # read axis
                if event.type == pygame.JOYAXISMOTION:
                    self.axis[event.axis] = event.value
                if event.type == pygame.JOYHATMOTION:
                    self.hat[event.hat] = event.value
            conn.send({"button": self.buttons, "axis": self.axis, "hat": self.hat, })
            if conn.poll():
                recv = conn.recv()
                if recv == -1:
                    return -1
            time.sleep(0.1)
    def read(self):
        recv = self.__pipe[0].recv()
        while self.__pipe[0].poll():
            recv = self.__pipe[0].recv()
        self.recv = recv
        return recv
    def getCross(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][0]
    def getCircle(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][1]
    def getSquare(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][2]
    def getTriangle(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][3]
    def getShare(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][4]
    def getPS(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][5]
    def getOptions(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][6]
    def getLStickIn(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][7]
    def getRStickIn(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][8]
    def getL1(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][9]
    def getR1(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][10]
    def getDPadUp(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][11]
    def getDPadDown(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][12]
    def getDPadLeft(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][13]
    def getDPadRight(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][14]
    def getTouchPad(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["button"][15]
    def getAxisLStickX(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["axis"][0]
    def getAxisLStickY(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["axis"][1]
    def getAxisRStickX(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["axis"][2]
    def getAxisRStickY(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["axis"][3]
    def getAxisL2(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["axis"][4]
    def getAxisR2(self, msg=None):
        if msg is None:
            msg = self.recv
        return msg["axis"][5]

if __name__ == "__main__":
    print("DS4 module test")
    ds4 = DualShock4()
    ds4.open()
    for i in range(1000):
        recv = ds4.read()
        print("i:", i, recv)
        print("Up:", ds4.getDPadUp(recv))
        print("Dn:", ds4.getDPadDown(recv))
        print("LeftAxisX", ds4.getAxisLStickX(recv))
        print("LeftAxisY", ds4.getAxisLStickY(recv))
        print("RightAxisX", ds4.getAxisRStickX(recv))
        print("RightAxisY", ds4.getAxisRStickY(recv))
        time.sleep(0.1)
    ds4.close()