import numpy as np
# import cv2
from matplotlib import pyplot as plt
from btimage import BT_image, PhaseCombo, round_all_the_entries_ndarray
import glob
import tqdm
import pandas as pd
from sklearn.metrics import r2_score

p = "E:\\DPM\\20190603\\"
implement = PhaseCombo(p)
# implement.combo(shift1=-1, shift2=1.2)
# implement.npy2png()
implement.combo(target=1)

circel_mean = []

# # read from npy file
# path_look = "E:\\DPM\\20190603\\phase_npy\\*.npy"
# list_look = glob.glob(path_look)
# for i in tqdm.trange(len(list_look)):
#     im_test = BT_image(list_look[i])
#     im_test.opennpy()
#     im_test.plot_it(im_test.img)
#     # im_test.crop(1650, 1550)
#     # im_test.plot_it(im_test.img)
#     # im_test.find_centroid()
#     # im_test.crop_img2circle_after_crop_it_to_tiny_square(im_test.centroid_x, im_test.centroid_y)
#     # im_test.write_image("E:\\DPM\\20190601\\analysis\\", im_test.img*255/(5-(-1)))
#     # im_test.plot_it(im_test.board)
#     # im_test.plot_it(im_test.threshold)
#
#     circel_mean.append(im_test.flat_board)

# matrix = round_all_the_entries_ndarray(np.corrcoef(circel_mean), 3)
# # dataframe visualization
# df = pd.DataFrame(data=matrix)
# df.insert(0, " ", col)
# m_list = []
# b_list = []
# r2_list = []
# for i in tqdm.trange(len(circel_mean)):
#     (m, b) = np.polyfit(circel_mean[i], circel_mean[0], 1)
#     coef = r2_score(circel_mean[i], circel_mean[0])
#     sort_x = np.sort(circel_mean[i])
#     yp = np.polyval([m, b], sort_x)
#     plt.figure()
#     plt.scatter(circel_mean[i], circel_mean[0], s=2)
#     plt.title("Comparison of phase per pixel between {} and {}".format(i, 0))
#     plt.plot(sort_x, yp, 'r',
#              label="y={}*x + {}.\nR square={}".format(round(m, 2), round(b, 2), round(coef, 2)))
#     plt.legend()
#     plt.xlabel("phase (rad)")
#     plt.ylabel("phase (rad)")
#     plt.savefig("E:\\DPM\\20190601\\analysis\\{}_{}.png".format(i, 0))
#     plt.show()
#     m_list.append(m)
#     b_list.append(b)
#     r2_list.append(coef)
#
# print("m mean: ", np.mean(m_list), "\nm sd: ", np.std(m_list))
# print("b mean: ", np.mean(b_list), "\nb sd: ", np.std(b_list))
# print("r2 mean: ", np.mean(r2_list), "\nr2 sd: ", np.std(r2_list))

# plt.figure()
# plt.plot(np.arange(40), circel_mean)
# plt.xlabel("frame")
# plt.ylabel("phase")
# plt.ylim(1.4, 1.9)
# plt.show()
