from btimage import BT_image
import cv2
from matplotlib import pyplot as plt
import numpy as np

path_f = "E:\\DPM\\20190619\\20190612_RPE_5hr_random4_f.tif"
path_b = "E:\\DPM\\20190619\\20190612_RPE_5hr_random4.tif"
im_f = BT_image(path_f)
im_f.open_image(color="rgb")

# image cleaning
# green_channel = cv2.erode(im_f.img[:, :, 1], (10, 10))
green_channel = im_f.img[:, :, 1]
red_channel = np.zeros((im_f.img.shape[0], im_f.img.shape[1]))
blue_channel = np.zeros((im_f.img.shape[0], im_f.img.shape[1]))
im_f.img[:, :, 0] = red_channel
im_f.img[:, :, 1] = green_channel
im_f.img[:, :, 2] = blue_channel

plt.figure(figsize=(10, 10))
plt.imshow(im_f.img)
plt.title("RPE_5hr_f_image", fontsize=20)
plt.savefig("E:\\DPM\\20190612\\finsh\\process\\20190612_90min_f_image.png")
plt.show()

im_b = BT_image(path_b)
im_b.open_image(color="rgb")
plt.figure(figsize=(10, 10))
plt.imshow(im_b.img)
plt.title("RPE_5hr_pc_image", fontsize=20)
plt.savefig("E:\\DPM\\20190612\\finsh\\process\\20190612_90min_pc_image.png")
plt.show()

# overlay
dst = cv2.addWeighted(im_f.img, 1, im_b.img, 1, 0)

plt.figure(figsize=(10, 10))
plt.imshow(dst)
plt.title("RPE_5hr_overlay_image", fontsize=20)
plt.savefig("E:\\DPM\\20190612\\finsh\\process\\20190612_90min_overlay_image.png")
plt.show()

