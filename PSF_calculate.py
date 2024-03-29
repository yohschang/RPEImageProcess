import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftn, ifftn
from scipy import interpolate
import scipy.special as spl


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
    beadradius_at_ccd = beadradius
    # compute ideal bead
    half = length_of_profile/2
    x_ideal = np.arange(-half, half, length_of_profile/3999)  # 31.5727 um ; 4001 points
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


def interplo(t_x, t_y, number):
    f = interpolate.interp1d(t_x, t_y)
    xnew = np.arange(t_x[0], t_x[len(t_x)-1], (t_x[len(t_x)-1]-t_x[0])/number)
    ynew = f(xnew)  # use interpolation function returned by `interp1d`
    return xnew, ynew


def jinc(x):
    return spl.j1(x) / x


def formula_psf(Lamb, NA, leng):
    # setting
    points = 1200
    t_length = (leng/4000) * points / 1000 / 2  # 31.57 um / 4000 points

    # formula (mm)
    x_psf_f = np.linspace(-t_length, t_length, points)  # 2.36 um 300 points

    b = 2 * np.pi / Lamb * NA * abs(x_psf_f)
    y_psf_f = 2 * jinc(b)
    x_psf_f = 1000 * x_psf_f  # mm to um

    # print null width
    rlist, llist = [], []
    for i in range(len(x_psf_f)):
        if (y_psf_f[i] <= 0.0001) and ((x_psf_f[i] >= -1) and (x_psf_f[i] <= 1)):
            if x_psf_f[i] < 0:
                rlist.append(x_psf_f[i])
            else:
                llist.append(x_psf_f[i])
    print("null width:", round(llist[int(len(llist) / 2)] - rlist[int(len(rlist) / 2)], 3), "um")

    return x_psf_f, y_psf_f, np.sum(y_psf_f)


def from_idealpsf_to_computepsf(y_sin, y_ideal, area):
    # print("y_sin:", y_sin.shape)
    # print("y_ideal:", y_ideal.shape)

    # convolution
    y_sinconv = np.convolve(y_sin, y_ideal, mode='valid')/area  # 3402

    # zeros padding
    padding_length = y_ideal.shape[0] - y_sinconv.shape[0]
    y_sinconv_f = np.zeros(round(padding_length/2))
    y_sinconv_f = np.concatenate((y_sinconv_f, y_sinconv))
    y_sinconv_f = np.concatenate((y_sinconv_f, np.zeros(int(padding_length/2))))
    print("y_sinconv:", y_sinconv.shape)
    print("y_ideal:", y_ideal.shape)
    # fft
    yf_sinconv = fft(y_sinconv_f)
    yf_ideal = fft(y_ideal)

    # plot F domain
    # xr, yr, xf_plot1, yf_plot1 = gofft(x_ideal, y_ideal)
    # xr, yr, xf_plot2, yf_plot2 = gofft(x_ideal, y_sinconv_f)

    # resume sin
    y_testsin = ifft(np.divide(yf_sinconv, yf_ideal))

    # take the head and tail
    middle = int(y_sin.shape[0] / 2)
    total = int(y_sin.shape[0])
    alltotal = int(y_ideal.shape[0])
    final = np.zeros(total)
    for i in range(middle, total):
        final[i] = y_testsin[i - middle]
    for i in range(middle):
        final[i] = y_testsin[alltotal-middle + i]

    return final, y_sinconv_f

####################################################################


# loading measurement data
path = r"C:\Users\BT\PycharmProjects\untitled1\xprofile_DPMBead.txt"
x, y = read_profile(path)
x, y = interplo(x, y, 4000)
# centralize
x = x-138
# pixel to um
toum = 5.5/46.5  # 0.11827
x = x*toum
length = x[3999] - x[0]  # 31.57 um

# formula_psf
x_psf, y_psf, area_under_psf = formula_psf(0.000532, 0.5, length)

# hight adjust
buf = []
for i, j in zip(x, y):
    if i < -5.5:
        buf.append(j)
base = np.mean(buf)
y = y - base

# 4.78
# for k in np.arange(4.75,4.85,0.01):
k = 5
x_ideal, y_ideal = ideal_bead(k, length)

y_final, rr = from_idealpsf_to_computepsf(y_psf, y_ideal, area_under_psf)
y_final = y_final / max(y_final)
# y_resumepsf = deconvolve(y, y_ideal)
# y_resumebead = deconvolve(y, y_psf)
# y_sinconv_f = convolve(y_ideal, y_psf)

####################################################################
# plot
x_test = [1,2,3]

fig, axs = plt.subplots(1, 3, figsize=[15, 6])
axs[0].plot(x_psf, y_psf)
axs[0].set_title("PSF from formula")
axs[0].set_xlabel('x (um)')
axs[0].set_ylabel('a.u.')
axs[0].set_xlim(-5, 5)

# axs[1].plot(x, y, label="measurement")
axs[1].plot(x_ideal, y_ideal, label="ideal")
axs[1].plot(x_ideal, rr, label="psf conv ideal")
axs[1].legend()
axs[1].set_title("profile of bead")
axs[1].set_xlabel('x (um)')
axs[1].set_ylabel('phase')
axs[1].set_xlim(-15, 15)

axs[2].plot(x_psf, y_final, label="PSF")
axs[2].set_title("retrieved PSF")
axs[2].set_xlabel('x (um)')
axs[2].set_ylabel('a.u.')
axs[2].set_xlim(-5, 5)

plt.tight_layout()
plt.show()

# plt.figure(dpi=200)
# plt.plot(x_psf, y_psf)
# plt.xlabel('x (um)')
# plt.ylabel('y')
# plt.xlim(-5, 5)
# plt.title("PSF from formula")
# plt.show()
#
# plt.figure(dpi=300)
# plt.plot(x, y, label="measurement")
# plt.plot(x_ideal, y_ideal, label="ideal")
# plt.plot(x_ideal, rr, label="psf conv ideal")
# plt.legend()
# plt.title("raw profile of bead")
# plt.xlabel("x(um)")
# plt.xlim(-15, 15)
# plt.ylabel("phase")
# plt.show()
#
# plt.figure(dpi=250)
# plt.plot(x_psf, y_final, label="PSF")
# plt.legend()
# plt.title("retrieved PSF")
# plt.xlabel("x(um)")
# # plt.xticks(np.arange(min(x), max(x)+1, 1))
# plt.xlim(-5,5)
# plt.ylabel("au")
# plt.show()



