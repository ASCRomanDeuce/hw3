#!/usr/bin/env python
from classes import Drawer, Particle, EMField
import numpy as np
import math

mode = 'wave'

t = np.array([0., 100], dtype='d')
init = np.array([[0, 0, 0], [0.1, 0.1, 0.1]], dtype='d')

a = EMField(mode)
p = Particle(a, init, t, mode, [math.pi/2, 24])
g = p.trajectory()
p = Drawer(g[0], g[1])
p.drawPlanes()
