import cv2
import numpy as np
from btimage import WorkFlow, BT_image
from matplotlib import pyplot as plt


root_path = r"E:\DPM\20190708_time_lapse_succ\Bead\1\SP\time-lapse" + "\\"
env = WorkFlow(root_path)

im1 = BT_image(env.phase_npy_path + "1_phase.npy")
im1.opennpy()
im1.phase2int8()
im2 = BT_image(env.phase_npy_path + "2_phase.npy")
im2.opennpy()
im2.phase2int8()

sift = cv2.xfeatures2d.SIFT_create()
keypoint = sift.detect(im1.img, None)
img = cv2.drawKeypoints(im1.img, keypoint, im1.img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(img)
plt.show()

