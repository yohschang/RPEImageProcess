import matplotlib.pyplot as plt
import scipy.special as spl
import numpy as np
from sympy import *


def jinc(x):
    return spl.j1(x) / x
def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

# mm
# aperture
r = 0.5
# wavelength
Lambda_ = 0.000532
# focal length
f = 2

# formula
x = np.linspace(-0.02,0.02,268)
y = np.pi*(r**2)*jinc(2*np.pi*r*abs(x)/(Lambda_ * f))
# normalize


# y = (y-min(y)) / (max(y)-min(y))
t = (max(y)-min(y))/2.0
t = 0
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
x = 1000*x

# y = normalize(y)

# output PSF
path_psf = r"/home/bt/文件/bosi_optics/DPM_verify/std_PSF_1um.txt"
with open(path_psf, "wt") as f:
    for i in range(len(x)):
        f.write(str(x[i])+"\t"+str(y[i])+"\n")


plt.figure(1)
plt.plot(x,y)
plt.xlabel('x (um)')
plt.ylabel('y')
plt.xlim(-10,10)
# plt.ylim(-0.1,1.2)
plt.show()