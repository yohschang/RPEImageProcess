import os
import cv2
import numpy as np
from btimage import check_file_exist
from btimage import BT_image, CellLabelOneImage, PrevNowCombo, TimeLapseCombo
import glob
from matplotlib import pyplot as plt


root_path = "E:\\DPM\\20190708\\Bead\\1\\SP\\time-lapse\\"

####################################################################################
# current_target = 4
# after = CellLabelOneImage(root_path, target=current_target).run(adjust=True, plot_mode=False, load_old=True, save_water=True)
# plt.close()
# plt.figure()
# plt.title(str(current_target) + "label img")
# plt.imshow(after, cmap='jet')
# plt.colorbar()
# plt.show()

####################################################################################

# output = PrevNowCombo(root_path).combo(now_target=5, save=True)

####################################################################################
# test
path = "E:\\DPM\\20190708\\Bead\\1\\SP\\time-lapse\\afterwater\\1_afterwater.npy"
# check_file_exist(path, "npy file")
# test_afterwater = np.load(path)
# plt.figure()
# plt.imshow(test_afterwater, cmap="jet")
# plt.show()
#
# label_list = []
# for label in range(90):
#     cur_label_num = len(test_afterwater[test_afterwater == label])
#     if 4000 <= cur_label_num <= 1000000:
#         # remove too small area and BG area
#         label_list.append(label)
#         print("label:", label, "has:", cur_label_num, "pixel")


# kernel = np.ones((3, 3), np.uint8)
# e = cv2.erode(test_afterwater, kernel)



from btimage import BT_image


# path = "E:\\DPM\\20190708\\Bead\\1\\SP\\time-lapse\\phase_npy\\3_phase.npy"
# im = BT_image(path)
# im.opennpy()
# im.img += 0.8
# plt.figure()
# plt.title("Histogram of phase image")
# plt.hist(im.img.flatten(), bins=900)
# plt.ylabel("number")
# plt.xlabel("phase (rad)")
# plt.xlim(-2, 5)
# plt.show()

x = np.arange(0, 5, 0.01)
y = 1 / (1 + np.exp((30 * (0.15 - x/5))))


plt.figure()
plt.title("Sigmoid Correction (cutoff: 0.15, gain: 30)")
plt.xlabel("phase (rad)")
plt.plot(x, y)
plt.show()


