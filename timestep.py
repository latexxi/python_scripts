# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 09:37:03 2025

@author: lauri
"""

import numpy as np
import matplotlib.pyplot as plt

g1 = 1.0
g2 = 1.0
c1 = 1.0
c2 = 1.0

g0 = 1.0

G = np.array( [[g1, -g1, 0.0], [-g1, g1+g2, -g2], [0.0, -g2, g1 + g0]] )
C = np.array( [[c1, -c1, 0.0], [-c1, c1+c2, -c2], [0.0, -c2, c1]] )

x0 = np.array( [0.0, 0.0, 0.0] )
f = np.array( [1.0, 0.0, -1.0] )

dt = 0.002
A = C + dt * G

# Gx + Cx' = f
# IE: C (xn - xp)/dt + G xn = fn
# (C + dt * G) xn = dt * fn + C * xp

def tstep(xp):
    rhs = dt * f + C @ xp
    xn = np.linalg.solve(A, rhs)
    return xn

nt = 5000
xvec = np.zeros((nt, 3))
for i in range(1, nt):
    xvec[i] = tstep(xvec[i-1])
    
plt.plot(xvec)
plt.grid()