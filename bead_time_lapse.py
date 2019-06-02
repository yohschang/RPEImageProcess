import tqdm
from btimage import BT_image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2


# # temporal noise
# path = 'E:\\DPM\\20190601\\blank\\3\\SP\\'
# list_ = glob.glob(path + "*.bmp")
#
# average = np.zeros((3072, 3072))
# for i in tqdm.trange(len(list_)):
#     im = BT_image(list_[i])
#     im.open_image()
#     average += im.img
# average /= len(list_)
# plt.figure()
# plt.imshow(average)
# plt.show()
#
# cv2.imwrite(path + "bg3.bmp", average)


path = 'E:\\DPM\\20190601\\blank\\3\\phimap\\'
list_ = glob.glob(path + "*.bmp.phimap")

# circle_mean = []
# for i in tqdm.trange(len(list_)):
#     im = BT_image(list_[i])
#     im.open_raw_image()
#
#     # speed up
#     im.img = im.img[484:484+2102, 484:484+2102]
#
#     im.crop_img2circle(2102/2, 2102/2, 2100/2)
#     # im.crop(1536, 1536)
#     # im.find_centroid()
#     # im.crop_img2circle_after_crop_it_to_tiny_square(im.centroid_y, im.centroid_x)
#     # im.plot_it(im.board)
#     # im.plot_it(im.img)
#     # im.write_image(path+"crop_image\\", im.img * 255.0/(im.img.max()- im.img.min()))
#     # im.write_image(path + "circle_image\\", im.board * 255.0/(im.board.max()- im.board.min()))
#     circle_mean.append(np.std(im.flat_board))




#
#     im.crop_img2circle_after_crop_it_to_tiny_square(im.centroid_y, im.centroid_x)
#     im.plot_it(im.board)
#     circle_mean.append(np.mean(im.flat_board))
#
#
#
# x_ax = np.arange(len(list_))
# plt.figure()
# plt.plot(x_ax, circle_mean)
# plt.title("temporal phase")
# plt.xlabel("40 frame")
# plt.ylabel("phase")
# plt.show()
#
# plt.figure()
# plt.hist(circle_mean, bins=100)
# plt.title("temporal noise distribution")
# plt.xlabel("phase")
# plt.ylabel("count")
# plt.show()

t1 = np.load('E:\\DPM\\20190601\\blank\\1\\circle_mean.npy')
t2 = np.load('E:\\DPM\\20190601\\blank\\2\\circle_mean.npy')
t3 = np.load('E:\\DPM\\20190601\\blank\\3\\circle_mean.npy')
t4 = np.load("E:\\DPM\\20190529\\circle_spatial.npy")
list_accumulate = []
list_accumulate.extend(t2)
list_accumulate.extend(t3)

plt.figure()
plt.hist(list_accumulate, bins=60)
plt.title("spatial noise distribution")
plt.xlabel("rad")
plt.ylabel("count")
plt.text(0.07, 140, s="mean: 0.034\nSD:  0.005")
plt.xlim(0, 0.1)
plt.show()

