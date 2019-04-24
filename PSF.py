import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

path = r"/home/bt/文件/bosi_optics/DPM_verify/bead_xprofile.txt"
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
x = x-139
# moving average
moving_window = 1  # no moving window
y = np.convolve(y, np.ones((moving_window,))/moving_window, mode='same')
# background 0
# for i in range(len(x)):
#     if (abs(x[i])>47):
#         y[i] = 0

# hight adjust
buf = []
for i, j in zip(x, y):
    if (i<-75) or (i>75):
        buf.append(j)
base = np.mean(buf)
y = y - base

n = 1.598
n0 = 1.566
beadradius_at_ccd = 4.8*46.5/5.5  # 42.27 pixel

# compute ideal bead
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

# over in F domain
over = yf/yf_ideal
# inverse FT
y_i = ifft(over)
# normalize
y_ii = normalize(y_i)

buf = []
for i,j in zip(x,y_ii):
    if (i>-20) and (i<0):
        if (abs(j-0.5)<=0.2):
            um = i * 5.5 /46.5
            buf.append(um)
            print(um)

#plot
plt.figure(1, dpi=300)
plt.plot(x, y, label="measurement")
plt.plot(x_ideal, y_ideal, label="ideal")
plt.legend()
plt.title("raw profile of bead")
plt.xlabel("x(pixel)")
plt.ylabel("phase")
plt.show()

plt.figure(2, dpi=250)
plt.plot(xf2, yf2, label="measurement")
plt.plot(xf2_ideal, yf2_ideal, label="ideal")
plt.legend()
plt.title("FFT")
plt.xlabel("freq")
plt.ylabel("au")
plt.show()

plt.figure(3, dpi=250)
plt.plot(x*5.5/46.5, y_ii, label="PSF")
plt.legend()
plt.title("PSF")
plt.xlabel("x(um)")
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.ylabel("au")
plt.xlim(-5,5)
plt.show()