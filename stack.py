import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np

pa = "E:\\DPM\\20190722\\*.bmp"
image_list = glob.glob(pa)

sp = cv2.imread(image_list[0])
sp = cv2.cvtColor(sp, cv2.COLOR_BGR2GRAY)
bg = cv2.imread(image_list[1])
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(cv2.medianBlur(sp-bg, 5), cmap="gray")
plt.savefig("E:\\DPM\\20190722\\sp_bg.png")
plt.show()

pa = "E:\\DPM\\20190722\\many_bead\\*.bmp"
image_list = glob.glob(pa)
# # stack
# for pat in image_list:
#     img = cv2.imread(pat)
#     print(img.shape)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.medianBlur(img, 17)
#     plt.figure(dpi=200, figsize=(10, 10))
#     plt.title("1 image")
#     plt.imshow(img, cmap="gray")
#     plt.axis("off")
#     plt.show()
#     black = img.copy()
#     break
#
# for pat in image_list:
#     img = cv2.imread(pat)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.medianBlur(img, 17)
#     black += img
#
# plt.figure(dpi=200, figsize=(10, 10))
# plt.title("sum 53 image")
# plt.imshow(cv2.medianBlur(black, 17), cmap="gray")
# plt.axis("off")
# plt.show()


