import os
import cv2
import numpy as np
from btimage import check_file_exist
from btimage import BT_image, CellLabelOneImage, PrevNowCombo, TimeLapseCombo, Fov, WorkFlow
import glob
from matplotlib import pyplot as plt


root_path = "E:\\DPM\\20190708_time_lapse_succ\\Bead\\1\\SP\\time-lapse\\"

####################################################################################
# t = TimeLapseCombo(root_path=root_path)
# t.read(1, 36)
# t.combo(target=26, save=True, m_factor=0.5, strategy="cheat")

####################################################################################
# f = Fov(root_path, 1, 36)
# f.run()


###################################################################################
# label and match

# current_target = 30

# after = CellLabelOneImage(root_path, target=current_target).run(adjust=True, plot_mode=False, load="old", save_water=True)
# output = PrevNowCombo(root_path).combo(now_target=current_target, save=True)

# plt.close()
# plt.figure()
# plt.title(str(current_target) + "label img")
# plt.imshow(after, cmap='jet')
# plt.colorbar()
# plt.show()
# ####################################################################################



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

