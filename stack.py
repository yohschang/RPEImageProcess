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


# # stack
# for pat in image_list:
#     img = cv2.imread(pat)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.medianBlur(img, 5)
#     plt.figure()
#     plt.imshow(img, cmap="gray")
#     plt.show()
#     black = img.copy()
#     break

# for pat in image_list:
#     img = cv2.imread(pat)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.medianBlur(img, 5)
#     black += img
#
# plt.figure()
# plt.imshow(cv2.medianBlur(black, 17), cmap="gray")
# plt.show()


