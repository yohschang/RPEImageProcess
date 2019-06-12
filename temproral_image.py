from btimage import BT_image
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from welford_algorithm import OnlineVariance
###########################################################################
# Averaging 500 interferogram to one BG

# path = "E:\\DPM\\20190524\\BG\\"
# list_ = glob.glob(path + "*.bmp")
#
# average_img = np.zeros((3072, 3072))
# for i in tqdm.trange(len(list_)):
#     img_tem = BT_image(list_[i])
#     img_tem.open_image()
#     average_img += img_tem.img
#
# average_img /= len(list_)
#
# plt.figure()
# plt.title("Interferogram Averaging 5 images")
# plt.imshow(average_img, plt.cm.gray)
# plt.show()
#
# cv2.imwrite(path + "bg.bmp", average_img)

###########################################################################
# temporal noise
path = 'E:\\DPM\\20190521\\time\\finish\\'
list_ = glob.glob(path + "Buffer*.bmp.phimap")

x, y = 3072//2 - 1, 3072//2 - 1
r = 3072//2


# # spatial noise
# circle_spatial_noise = []
# circle_temporal_noise = []
# buffer = []
# for i in tqdm.trange(500):
#     im = BT_image(list_[i])
#     im.open_raw_image()
#     im.crop_img2circle(x, y, r)
#     circle_spatial_noise.append(np.std(im.flat_board))
#     for j in range(7411887):
#         if i == 0:
#             buffer.append(OnlineVariance(ddof=0))
#         buffer[j].include(im.flat_board[j])
#
# for k in range(7411887):
#     circle_temporal_noise.append(buffer[k].std)


# plot temporal noise
# circle = np.load("E:\\DPM\\20190529\\circle_temporal.npy")
# plt.figure()
# plt.hist(circle, bins=800)
# plt.title("temporal noise per pixel")
# plt.xlabel("rad")
# plt.ylabel("number of pixel")
# plt.xlim(-0.1, 0.3)
# plt.text(0.2, 200000, "mean: 0.026\nSD:  0.075")
# plt.show()

# reconstruction temporal noise
path = "E:\\DPM\\20190529\\circle_temporal.npy"
tem_noise = np.load(path)
k = 0
ze = np.zeros((3072, 3072))
for i in tqdm.trange(3072):
    for j in range(3072):
        if (i - 3072//2)**2 + (j - 3072//2)**2 <= (3072//2)**2:
            ze[i, j] = tem_noise[k]
            k += 1

plt.figure(dpi=300)
plt.imshow(ze, cmap='jet', vmax=0.08, vmin=0.0)
cb = plt.colorbar()
cb.ax.set_title('rad')
plt.title("Temporal noise in FOV (500 frames)")
plt.show()

# path = "E:\\DPM\\20190521_8position\\time\\1\\BG\\bg.bmp.phimap"
# im = BT_image(path)
# im.open_raw_image()
# # im.plot_it(im.img)
# k = 0
# ze = np.zeros((3072, 3072))
# for i in tqdm.trange(3072):
#     for j in range(3072):
#         if (i - 3072//2)**2 + (j - 3072//2)**2 <= (3072//2)**2:
#             ze[i, j] = im.img[i, j]
#             k += 1
#
# plt.figure(dpi=300)
# plt.imshow(ze, cmap="jet",  vmax=-0.1, vmin=0.1)
# cb = plt.colorbar()
# cb.ax.set_title('rad')
#
# plt.title("Spatial noise in FOV (1 frame)")
# plt.show()

