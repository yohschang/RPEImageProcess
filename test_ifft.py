import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from sympy import *
import scipy.special as spl
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
x = np.linspace(-0.02,0.02,200)
y = np.pi*(r**2)*jinc(2*np.pi*r*abs(x)/(Lambda_ * f))
#############################################
plt.figure(1, dpi=250)
plt.plot(1000*x, y)
plt.show()

# set
n = 1.598
n0 = 1.566
beadradius_at_ccd = 4.8*46.5/5.5  # 42.27 pixel
x_ideal = np.arange(-164, 163)
h = []
for i in x_ideal:
    if (abs(i) < beadradius_at_ccd):
        h.append(2*np.sqrt(beadradius_at_ccd**2-i**2))
    else:
        h.append(0)
h = np.array(h)
# h (pixel)
y_ideal = 2*np.pi/532*(h*5.5/46.5*1000)*(n-n0)

#############################################
yf = fft(y)
yf1 = abs(fft(y))/len(x)   #歸一化處理
yf2 = yf1[range(int(len(x)/2))] #由於對稱性，只取一半區間
xf = np.arange(len(y))  # 頻率
xf1 = xf
xf2 = xf[range(int(len(x)/2))] #取一半區間

plt.figure(2, dpi=250)
plt.plot(xf2, yf2)
plt.show()

y_i = ifft(yf)

plt.figure(3, dpi=250)
plt.plot(x, y_i)
plt.show()
