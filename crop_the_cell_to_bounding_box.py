from matplotlib import pyplot as plt
import btimage
import numpy as np
import cv2
from skimage.exposure import adjust_sigmoid

root_path = "E:\\DPM\\20190708_time_lapse_succ\\Bead\\1\\SP\\time-lapse\\"
env = btimage.WorkFlow(root_path)

save_img_path = "C:\\Users\\BT\\Desktop\\kaggle\\RPE_crop_image\\"
save_mask_path = "C:\\Users\\BT\\Desktop\\kaggle\\RPE_crop_mask\\"


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 5)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


phase_im_path = env.phase_npy_path + str(1) + "_phase.npy"
label_im_path = env.afterwater_path + str(1) + "_afterwater.npy"
phase_im = np.load(phase_im_path)
label_im = np.load(label_im_path)
label_im = label_im.astype(np.uint8)
phase_im = phase_im.astype(np.uint8)
phase_im = 255 * (phase_im +0.2) / (3+0.2)
# plt.figure()
# plt.imshow(label_im, cmap="jet")
# plt.show()

for label in range(5, 85):
    if len(label_im[label_im == label]) > 0:
        print(label)
        label_cur_im = label_im.copy()
        phase_cur_im = phase_im.copy()
        label_cur_im[label_cur_im != label] = 0
        label_cur_im[label_cur_im == label] = 255
        x, y, w, h = cv2.boundingRect(label_cur_im)
        crop_img_label = label_cur_im[y:y + h, x:x + w]
        crop_img_phase = phase_cur_im[y:y + h, x:x + w]

        max_v = crop_img_phase.max()
        min_v = crop_img_phase.min()



        # plt.figure()
        # plt.imshow(crop_img_phase, cmap="jet")
        # plt.show()
        # plt.figure()
        # plt.imshow(crop_img_label, cmap="jet")
        # plt.show()

        cv2.imwrite(save_img_path + str(1) + "_" + str(label) + "_cell.png", crop_img_phase)
        cv2.imwrite(save_mask_path + str(1) + "_" + str(label) + "_label_cell.png", crop_img_label)



