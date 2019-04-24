import matplotlib.pyplot as plt
import scipy.special as spl
import numpy as np
from sympy import *


def jinc(x):
    return spl.j1(x) / x

# mm
# aperture
r = 1
# wavelength
Lambda_ = 0.000532
# focal length
f = 1.5

# formula
x = np.linspace(-0.02,0.02,2000)
y = np.pi*(r**2)*jinc(2*np.pi*r*abs(x)/(Lambda_ * f))
# normalize
# y = (y-min(y)) / (max(y)-min(y))
t = (max(y)-min(y))/2.0
for i in range(len(x)):
    if (y[i-1] < t) and (y[i] >= t):
        l = y[i]
        left = x[i]
        print(left*1000000,'nm')
    if (y[i-1] > t) and (y[i] <= t):
        rr = y[i]
        right = x[i]
        print(right*1000000,'nm')
print(right*1000000-left*1000000,'nm')
plt.figure(1)
plt.plot(1000000*x,y)
plt.xlabel('x (nm)')
plt.ylabel('y')
plt.xlim(-2000,2000)
# plt.ylim(-0.1,1.2)
plt.show()