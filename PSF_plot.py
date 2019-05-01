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


def formula_psf(Lambda_, NA, points):

    # formula (mm)
    x_psf = np.linspace(-0.00472, 0.00472, 1200)  # total 31.57 um ; 4001 points # 2.36*2 um 600points
    # y = np.pi*(r**2)*jinc(2*np.pi*r*abs(x)/(Lambda_ * f))
    b = 2*np.pi/Lambda_*NA*abs(x_psf)
    y_psf = 2*jinc(b)
    x_psf = 1000*x_psf  # mm to um

    # print null width
    rlist, llist = [], []
    for i in range(len(x_psf)):
        if (y_psf[i] <= 0.0001) and ((x_psf[i] >= -1) and (x_psf[i] <= 1)):
            if x_psf[i] < 0:
                rlist.append(x_psf[i])
            else:
                llist.append(x_psf[i])
    print("null width:", round(llist[int(len(llist)/2)] - rlist[int(len(rlist)/2)], 3), "um")

    return x_psf, y_psf


x, y = formula_psf(0.000532, 0.35, 1200)


# output PSF
path_psf = r"/home/bt/文件/bosi_optics/DPM_verify/std_PSF_1um.txt"
with open(path_psf, "wt") as f:
    for i in range(len(x)):
        f.write(str(x[i])+"\t"+str(y[i])+"\n")

plt.figure(1)
plt.plot(x, y)
plt.xlabel('x (um)')
plt.ylabel('y')
plt.xlim(-5, 5)
plt.title("PSF from formula")
# plt.ylim(-0.1,1.2)
plt.show()