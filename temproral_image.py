from btimage import BT_image
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

###########################################################################
# Averaging 500 interferogram to one BG

# path = "E:\\DPM\\20190521\\time\\1\\SP\\"
# list_ = glob.glob(path + "*.bmp")
#
# average_img = np.zeros((3072, 3072))
# for i in tqdm.trange(len(list_)):
#     img_tem = BT_image(list_[i])
#     average_img += img_tem.img
#
# average_img /= len(list_)
#
# plt.figure()
# plt.title("Interferogram Averaging 500 images")
# plt.imshow(average_img, plt.cm.gray)
# plt.show()

# cv2.imwrite(path + "bg.bmp", average_img)

###########################################################################
# temporal noise
path = 'E:\\DPM\\20190521\\time\\finish\\'
list_ = glob.glob(path + "Buffer*.bmp.phimap")

x, y = 3072//2 - 1, 3072//2 - 1
r = 3072//2

circle_mean = []
for i in tqdm.trange(len(list_)):
    im = BT_image(list_[i])
    im.open_raw_image()
    im.crop_img2circle(x, y, r)
    circle_mean.append(np.mean(im.flat_board))

x_ax = np.arange(500)
plt.figure()
plt.plot(x_ax, circle_mean)
plt.title("temporal noise")
plt.xlabel("500 frame")
plt.ylabel("phase")
plt.show()

plt.figure()
plt.hist(circle_mean, bins=100)
plt.title("temporal noise distribution")
plt.xlabel("phase")
plt.ylabel("count")
plt.show()

print(np.std(circle_mean))

