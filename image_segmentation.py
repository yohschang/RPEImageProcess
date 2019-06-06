from btimage import BT_image
import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt


path = "E:\\DPM\\20190603_RPE\\phase_npy\\img_2019_06_03_17_46_38_phase.npy"

im = BT_image(path)
im.opennpy()
# im.plot_it(im.img)
image_rescale = (im.img + 0.5) * 255 / (3.8 + 0.5)
t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
image = np.uint8(image_rescale)
kernel = np.ones((8,8), np.uint8)
image = cv2.dilate(image, kernel)
image = cv2.erode(image, kernel)

image = cv2.erode(image, kernel)
image = cv2.dilate(image, kernel)

result = cv2.Canny(image, 50, 130)

plt.figure()
plt.imshow(result, cmap="gray", vmax=255, vmin=0)
plt.colorbar()
plt.show()
