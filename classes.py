import numpy as np
import math
import matplotlib.pyplot as plt
from rk1 import rk


class EMField:
    def __init__(self, conf):
        self.k = np.array([1, 0, 0])
        self.mode = conf

    def cross(self):
        E = np.array([1, 0, 0])
        H = np.array([0, 1, 0])
        return [E, H]

    def wave(self, x, t, phi):
        k = self.k
        E1 = np.cos(np.dot(k, x) - t)
        E2 = np.cos(np.dot(k, x) - t + phi)
        E3 = 0
        E = np.array([E1, E2, E3])
        H = np.cross(E, k)
        return [E, H]

    def impulse(self, x, t, t0, phi):
        E = (self.wave(x, t, phi)[0]) * math.exp(-t ** 2 / (2 * t0 ** 2))
        H = (self.wave(x, t, phi)[1]) * math.exp(-t ** 2 / (2 * t0 ** 2))
        return [E, H]

    def intencity(self, x, t, phi, t0):
        if self.mode == 'cross':
            E = self.cross()[0]
            H = self.cross()[1]
        if self.mode == 'wave':
            E = self.wave(x, t, phi)[0]
            H = self.wave(x, t, phi)[1]
        if self.mode == 'impulse':
            E = self.impulse(x, t, t0, phi)[0]
            H = self.impulse(x, t, t0, phi)[1]

        I = (np.dot(E, E) + np.dot(H, H)) / (4 * math.pi)
        return I

    def drawIntensity(self, t, data):
        y = np.linspace(-10, 10, 100)
        intens = np.vectorize(lambda x, z: self.intencity(np.array([x, 0, z]), t, data))
        xi, zi = np.meshgrid(y, y)
        int = intens(xi, zi)
        fig, axs = plt.subplots()
        e = axs.pcolor(int)
        fig.tight_layout()
        fig.colorbar(e, ax=axs)
        plt.show()

class Particle:
    def __init__(self, field, intial_conditions, time_interval, mode, data):
        self.field = field
        self.intial_momentum = intial_conditions[1]
        self.intial_coordinate = intial_conditions[0]
        self.time_interval = time_interval
        self.data = data
        self.par = mode

    def trajectory(self):
        par = self.par

        if  par=='cross':
            Er = lambda t,x: self.field.cross()[0]
            Hr = lambda t,x: self.field.cross()[1]
        if  par=='wave':
            delta = self.data[0]
            Er = lambda t,x: self.field.wave(x,t,delta)[0]
            Hr = lambda t,x: self.field.wave(x,t,delta)[1]
        if par == 'laser':
            delta = self.data[0]
            t0 = self.data[1]
            Er = lambda t,x: self.field.laser(x,t,t0,delta)[0]
            Hr = lambda t,x: self.field.laser(x,t,t0,delta)[1]


        def fct(arg, t):
                p = np.hsplit(arg, 2)[0]
                x = np.hsplit(arg, 2)[1]
                E = Er(t, x)
                H = Hr(t, x)
                fp = -E-np.cross(p,H)/math.sqrt(1+ np.dot(p,p))
                fx = p/math.sqrt(1+ np.dot(p,p))

                return np.hstack((fp, fx))

        t1 = self.time_interval
        y1 = np.hstack((self.intial_momentum,self.intial_coordinate))
        k = rk(y1, t1)
        a = k.solve(fct)

        return [a[0],[a[4],a[5],a[6]],[a[1],a[2],a[3]]]

class Drawer:
    def __init__(self, targ, xarg):
        self.targ = targ
        self.xarg = xarg

    def drawTime(self):
        fig, ax = plt.subplots()
        leg = ['t', ['x', 'y', 'z']]
        for i in range(len(self.xarg)):
            plt.plot(self.targ, self.xarg[i], label = leg[1][i])
        ax.set_xlabel(leg[0])
        plt.grid()
        plt.legend()
        plt.show()

    def draw3d(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        z = self.xarg[2]
        y = self.xarg[1]
        x = self.xarg[0]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot3D(x, y, z, 'red')
        plt.show()

    def drawPlanes(self):
        z = self.xarg[2]
        y = self.xarg[1]
        x = self.xarg[0]
        plt.subplot(2, 2, 1)
        plt.plot(x, y)
        plt.title("xy", loc = 'left', y = 0.85)
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.plot(x, z)
        plt.title("xz", loc = 'left', y = 0.85)
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.plot(y, z)
        plt.title("yz", loc = 'left', y = 0.85)
        plt.grid()
        plt.show()
