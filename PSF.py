import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftn, ifftn
from scipy import interpolate


def deconvolve(star, psf):
    star_fft = fftshift(fftn(star))
    psf_fft = fftshift(fftn(psf))
    return fftshift(ifftn(ifftshift(star_fft/psf_fft)))


def convolve(star, psf):
    star_fft = fftshift(fftn(star))
    psf_fft = fftshift(fftn(psf))
    return fftshift(ifftn(ifftshift(star_fft*psf_fft)))


def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d


def read_profile(path):
    x = []
    y = []
    with open(path, 'r') as f:
        row = f.readlines()
        for i in row:
            x.append(i.split("\t")[0])
            y.append(i.split("\t")[1].strip("\n"))
    x = np.array(x).astype(float)
    y = np.array(y).astype(float)
    return x, y


def gofft(x,y):
    # measured FFT
    yf = fft(y)    # 取絕對值
    yf1 = abs(fft(y))/len(x)   #歸一化處理
    yf2 = yf1[range(int(len(x)/2))] #由於對稱性，只取一半區間
    xf = np.arange(len(y))  # 頻率
    xf1 = xf
    xf2 = xf[range(int(len(x)/2))] #取一半區間
    return xf, yf, xf2, yf2


def runnung_mean(y,window):
    y = np.convolve(y, np.ones((window,)) / window, mode='same')
    return y


def ideal_bead(beadradius, length_of_profile):
    n = 1.598
    n0 = 1.566
    beadradius_at_ccd = beadradius  # 42.27 pixel
    # compute ideal bead
    half = length_of_profile/2
    x_ideal = np.arange(-half, half, length_of_profile/4001)  # 31.5727 um ; 4001 points
    h = []
    for i in x_ideal:
        if abs(i) < beadradius_at_ccd:
            h.append(2*np.sqrt(beadradius_at_ccd**2-i**2))
        else:
            h.append(0)
    h = np.array(h)
    # h (um)
    y_ideal = 2*np.pi/532*(h*1000)*(n-n0)
    return x_ideal, y_ideal


def interplo(x, y, number):
    f = interpolate.interp1d(x, y)
    xnew = np.arange(x[0], x[len(x)-1], (x[len(x)-1]-x[0])/number)
    ynew = f(xnew)  # use interpolation function returned by `interp1d`
    return xnew, ynew


def from_idealpsf_to_computepsf(y_sin, y_ideal):
    # y_sin -> 300
    # y_ideal -> 4000

    # convulotion
    y_sinconv = np.convolve(y_sin, y_ideal, mode='valid')  # 3701

    # zeros padding
    y_sinconv_f = np.zeros(150)
    y_sinconv_f = np.concatenate((y_sinconv_f, y_sinconv))
    y_sinconv_f = np.concatenate((y_sinconv_f, np.zeros(149)))

    # fft
    yf_sinconv = fft(y_sinconv_f)
    yf_ideal = fft(y_ideal)

    # plot F domain
    # xr, yr, xf_plot1, yf_plot1 = gofft(x_ideal, y_ideal)
    # xr, yr, xf_plot2, yf_plot2 = gofft(x_ideal, y_sinconv_f)

    # resume sin
    y_testsin = ifft(np.divide(yf_sinconv, yf_ideal))

    # take the head and tail
    final = np.zeros(300)
    for i in range(150, 300):
        final[i] = y_testsin[i - 150]
    for i in range(150):
        final[i] = y_testsin[3850 + i]
    return final, y_sinconv_f

####################################################################

# loading data
path = r"/home/bt/文件/bosi_optics/DPM_verify/bead_xprofile.txt"
x, y = read_profile(path)
x, y = interplo(x, y, 4001)
# y = runnung_mean(y,300)
length_of_profile = x[4000] - x[0]

path_psf = r"/home/bt/文件/bosi_optics/DPM_verify/std_PSF_1um.txt"
x_psf, y_psf = read_profile(path_psf)
# x_psf, y_psf = interplo(x_psf, y_psf, 4001)

path_sin = r"/home/bt/文件/bosi_optics/DPM_verify/sinwave.txt"
x_sin, y_sin = read_profile(path_sin)


# centralize
x = x-138
# pixel to um
toum = 5.5/46.5  # 0.11827
x = x*toum
# moving average
# y = runnung_mean(y, 100)

# hight adjust
buf = []
for i, j in zip(x, y):
    if i < -5.5:
        buf.append(j)
base = np.mean(buf)
y = y - base

# # background 0
# for i in range(len(x)):
#     if (x[i] <= -55) or (x[i] >= 55):
#         y[i] = 0


# 4.78
# for k in np.arange(4.75,4.85,0.01):
k = 4.78
x_ideal, y_ideal = ideal_bead(k, length_of_profile)

# # measured FFT
# xf, yf, xf2, yf2 = gofft(x, y)

# # ideal FFT
# xf_ideal, yf_ideal, xf2_ideal, yf2_ideal = gofft(x_ideal, y_ideal)

# fft
# yf_conv = fft(y)
# yf_ideal = fft(y_ideal)

# plot F domain
# xr, yr, xf_plot1, yf_plot1 = gofft(x_ideal, y_ideal)
# xr, yr, xf_plot2, yf_plot2 = gofft(x_ideal, y_sinconv_f)

# resume sin
# y_resumepsf = ifft(np.divide(yf_conv, yf_ideal))

for i in x_psf:
    if not i:
        print("here")
        
print("y:",y.shape, type(y[0]))
print("y:",y_ideal.shape, type(y_ideal[0]))
print("y:",y_psf.shape, type(y_psf[0]))
y_resumepsf = deconvolve(y, y_ideal)
y_resumebead = deconvolve(y, y_psf)
y_sinconv_f = convolve(y_ideal, y_psf)

# take the head and tail
# final = np.zeros(300)
# for i in range(150, 300):
#     final[i] = y_resumepsf[i - 150]
# for i in range(150):
#     final[i] = y_resumepsf[3850 + i]

#plot
# plt.figure(dpi=300)
# plt.plot(x, y, label="measurement")
# plt.plot(x_ideal, y_ideal, label="ideal")
# plt.legend()
# plt.title("raw profile of bead")
# plt.xlabel("x(um)")
# plt.xlim(-15, 15)
# plt.ylabel("phase")
# plt.show()

plt.figure(dpi=300)
plt.plot(x, y, label="measurement")
plt.plot(x_ideal, 3.7*normalize(y_sinconv_f), label="ideal measurement")
plt.legend()
plt.title("raw profile of bead")
plt.xlabel("x(um)")
# plt.xlim(-15, 15)
plt.ylabel("phase")
plt.show()
# plt.figure(2, dpi=250)
# plt.plot(xf2, yf2, label="measurement")
# plt.plot(xf2_ideal, yf2_ideal, label="ideal")
# plt.legend()
# plt.title("FFT")
# plt.xlabel("freq")
# plt.ylabel("au")
# plt.show()

plt.figure(dpi=250)
plt.plot(x, y_resumepsf, label="PSF")
plt.legend()
plt.title("PSF")
plt.xlabel("x(um)")
# plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.xlim(-5,5)
plt.ylabel("au")
plt.show()

plt.figure(dpi=250)
plt.plot(x, y_resumebead, label="y_resumebead")
plt.legend()
plt.title("y_resumebead")
plt.xlabel("x(um)")
# plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.ylabel("au")
plt.show()

####################################################
## forward simulation

# um
# x_sample = x_ideal
# y_sample = y_ideal
#
# y_sinconv_f = convolve(y_ideal, y_psf)
# final_psf = deconvolve(y_sinconv_f, y_ideal)
# # final_psf, y_sinconv_f = from_idealpsf_to_computepsf(y_psf, y_ideal)
#
# plt.figure(4, dpi=500)
# plt.plot(x_sample, normalize(y_sample), label="ideal bead",linewidth=2)
# plt.title("ideal bead")
# plt.xlabel("x(um)")
# plt.ylabel("au")
# plt.xlim(-10,10)
# plt.show()
#
# plt.figure(11, dpi=500)
# plt.plot(x_sample, normalize(y_sinconv_f), label="ideal measure",linewidth=2)
# plt.title("ideal measurement")
# plt.xlabel("x(um)")
# plt.ylabel("au")
# plt.xlim(-10,10)
# plt.show()
#
# plt.figure(5, dpi=250)
# plt.plot(x_psf, runnung_mean(final_psf, 1), label="test psf")
# plt.legend()
# plt.title("test psf")
# plt.xlabel("x(um)")
# plt.ylabel("au")
# # plt.xticks(np.arange(min(x), max(x)+1, 1))
# plt.xlim(-2.5, 2.5)
# plt.show()

####################################################
# test sin wave
# final = from_idealpsf_to_computepsf(y_sin, y_ideal)

# plt.figure(6, dpi=250)
# plt.plot(x_sin, y_sin, label="sin (3/period)")
# plt.title("sin (3/period)")
# plt.xlabel("x")
# plt.ylabel("au")
# plt.show()
#
# plt.figure(8, dpi=250)
# plt.plot(x_ideal, y_sinconv_f, label="sin")
# plt.title("conv result")
# plt.xlabel("x")
# plt.ylabel("au")
# plt.show()
#
# plt.figure(7, dpi=250)
# plt.plot(x_ideal, y_ideal, label="ideal bead")
# plt.legend()
# plt.title("ideal bead")
# plt.xlabel("x")
# plt.ylabel("au")
# plt.show()

# plt.figure(9, dpi=250)
# plt.plot(xf_plot1, yf_plot1, label="ideal")
# plt.plot(xf_plot2, yf_plot2, label="afterconv")
# plt.legend()
# plt.title("F domain")
# plt.xlabel("x")
# plt.xlim(0,50)
# plt.ylabel("au")
# plt.show()

# plt.figure(8, dpi=250)
# plt.plot(x_sin, final, label="sin")
# plt.title("sin?")
# plt.xlabel("x")
# plt.ylabel("au")
# # plt.xticks(np.arange(min(x), max(x)+1, 1))
# plt.show()
