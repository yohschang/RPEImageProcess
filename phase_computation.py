import numpy as np
# import cv2
from matplotlib import pyplot as plt
from btimage import BT_image, PhaseCombo
import glob
import tqdm

p = "E:\\DPM\\20190602_check_python\\"
implement = PhaseCombo(p)
# implement.combo(shift1=-1, shift2=1.2)
# implement.npy2png()
implement.combo(target=2)

# circel_mean = []
# # read from npy file
# path_look = "E:\\DPM\\20190602_check_python\\phase_npy\\*.npy"
# list_look = glob.glob(path_look)
# for i in tqdm.trange(len(list_look)):
#     im_test = BT_image(list_look[i])
#     im_test.opennpy()
#     im_test.plot_it(im_test.img)
#     # im_test.crop(1650, 1550)
#     # im_test.plot_it(im_test.img)
#     # im_test.find_centroid()
#     # im_test.crop_img2circle_after_crop_it_to_tiny_square(im_test.centroid_x, im_test.centroid_y)
#     # im_test.write_image("E:\\DPM\\20190527\\phase_png\\", im_test.img*255/(5-(-1)))
#     # im_test.plot_it(im_test.board)
#     # im_test.plot_it(im_test.threshold)
#
#     # circel_mean.append(np.mean(im_test.flat_board))

# plt.figure()
# plt.plot(np.arange(40), circel_mean)
# plt.xlabel("frame")
# plt.ylabel("phase")
# plt.ylim(1.4, 1.9)
# plt.show()
