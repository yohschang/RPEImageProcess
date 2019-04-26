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
# r = 1.23
# wavelength
Lambda_ = 0.000532
# focal length
# f = 1.5
# NA
NA = 0.5


# formula
x = np.linspace(-0.01579, 0.01579, 4001)  # total 31.57 um ; 4001 points
# y = np.pi*(r**2)*jinc(2*np.pi*r*abs(x)/(Lambda_ * f))
b = 2*np.pi/Lambda_*NA*abs(x)
y = 2*jinc(b)
x = 1000*x  # mm to um

t = 0
rlist, llist = [], []
for i in range(len(x)):
    if (y[i] <= 0.0001) and ((x[i] >= -0.8) and (x[i] <= 0.8)):
        if x[i]<0:
            rlist.append(x[i])
        else:
            llist.append(x[i])
print("null width:", round(llist[int(len(llist)/2)] - rlist[int(len(rlist)/2)], 3), "um")
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
plt.xlim(-2.5,2.5)
# plt.ylim(-0.1,1.2)
plt.show()