import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

#xprofile_DPMBead
path = r"E:\DPM\20190420_mag\test.txt"
# os.chdir(path)

x = []
y = []
with open(path, 'r') as f:
    row = f.readlines()
    for i in row:
        x.append(i.split("\t")[0])
        y.append(i.split("\t")[1].strip("\n"))
x = np.array(x).astype(float)
y = np.array(y).astype(float)

# centralize
x = x-126
# moving average
y = np.convolve(y, np.ones((18,))/18, mode='same')
for i in range(len(x)):
    if (abs(x[i])>50):
        y[i] = 0

n = 1.598
n0 = 1.566
beadradius_at_ccd = 4.8*46.5/5.5  # 42.27 pixel

# set
x_ideal = np.arange(x[0], x[len(x)-1]+1)
h = []
for i in x_ideal:
    if (abs(i) < beadradius_at_ccd):
        h.append(2*np.sqrt(beadradius_at_ccd**2-i**2))
    else:
        h.append(0)
h = np.array(h)
# h (pixel)
y_ideal = 2*np.pi/532*(h*5.5/46.5*1000)*(n-n0)

#plot
plt.figure(1, dpi=300)
plt.plot(x, y, label="measurement")
plt.plot(x_ideal, y_ideal, label="ideal")
plt.legend()
plt.title("raw profile of bead")
plt.xlabel("x(pixel)")
plt.ylabel("phase")
plt.show()

# measured FFT
yf = fft(y)    # 取絕對值
yf1 = abs(fft(y))/len(x)   #歸一化處理
yf2 = yf1[range(int(len(x)/2))] #由於對稱性，只取一半區間
xf = np.arange(len(y))  # 頻率
xf1 = xf
xf2 = xf[range(int(len(x)/2))] #取一半區間
# ideal FFT
yf_ideal = fft(y_ideal)   # 取絕對值
yf1_ideal = abs(fft(y_ideal))/len(x_ideal)   #歸一化處理
yf2_ideal = yf1_ideal[range(int(len(x_ideal)/2))] #由於對稱性，只取一半區間
xf_ideal = np.arange(len(y_ideal))  # 頻率
xf1_ideal = xf_ideal
xf2_ideal = xf1_ideal[range(int(len(x_ideal)/2))] #取一半區間

plt.figure(2, dpi=250)
plt.plot(xf2, yf2, label="measurement")
plt.plot(xf2_ideal, yf2_ideal, label="ideal")
plt.legend()
plt.title("FFT")
plt.xlabel("freq")
plt.ylabel("au")
plt.show()

over = yf/yf_ideal
y_i = ifft(over)
y_ii = normalize(y_i)

plt.figure(3, dpi=250)
plt.plot(x, y_ii, label="PSF")
plt.legend()
plt.title("PSF")
plt.xlabel("x(pixel)")
plt.ylabel("au")
plt.xlim(-5,5)
plt.show()