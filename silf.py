import cv2
import numpy as np
from btimage import WorkFlow, BT_image
from matplotlib import pyplot as plt


root_path = r"E:\DPM\20190708_time_lapse_succ\Bead\1\SP\time-lapse" + "\\"
env = WorkFlow(root_path)
######################
im1 = BT_image(env.phase_npy_path + "1_phase.npy")
im1.opennpy()
im1.phase2int8()
im2 = BT_image(env.phase_npy_path + "2_phase.npy")
im2.opennpy()
im2.phase2int8()

mask1 = BT_image(env.afterwater_path + "1_afterwater.npy")
mask1.opennpy()
plt.figure()
plt.imshow(mask1.img, cmap='jet')
plt.colorbar()
plt.show()

mask2 = BT_image(env.afterwater_path + "2_afterwater.npy")
mask2.opennpy()

#######################

cell1 = im1.img.copy()
cell1[mask1.img != 66] = 0
cell2 = im2.img.copy()

fig, axe = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axe[0].imshow(cell1, cmap="gray")
axe[0].set_title("prev")
axe[0].axis("off")
axe[1].imshow(cell2, cmap="gray")
axe[1].set_title("now")
axe[1].axis("off")
fig.show()


# feature extraction
sift_whole = cv2.xfeatures2d.SIFT_create()
keypoint1, descriptor1 = sift_whole.detectAndCompute(cell1, None)
keypoint2, descriptor2 = sift_whole.detectAndCompute(cell2, None)
# img1 = cv2.drawKeypoints(cell1, keypoint1, cell1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img2 = cv2.drawKeypoints(cell1, keypoint1, cell1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# feature matching
matcher = cv2.DescriptorMatcher_create("BruteForce")
rawMatches = matcher.knnMatch(descriptor1, descriptor2, 2)
print(len(rawMatches))


# choose the good matching result
matches = []
table = []
for m in rawMatches:
    if len(m) == 2 and m[0].distance < m[1].distance * 0.9:
        table.append((m[0].trainIdx, m[0].queryIdx))
        matches.append(m[0])
print(len(matches))


# plot the result
matches = np.expand_dims(matches, 1)
img3 = cv2.drawMatchesKnn(cell1, keypoint1, cell2, keypoint2, matches[:], None, flags=2)
img4 = cv2.drawMatchesKnn(mask1.img*3, keypoint1, mask1.img*3, keypoint2, matches[:], None, flags=2)
plt.figure(dpi=150, figsize=(10, 6))
plt.imshow(img3)
plt.axis("off")
plt.show()
plt.figure(dpi=150, figsize=(10, 6))
plt.imshow(img4)
plt.axis("off")
plt.show()

statistic = []
for (trainIdx, queryIdx) in table:
    ptA = (int(keypoint1[queryIdx].pt[0]), int(keypoint1[queryIdx].pt[1]))
    if mask1.img[ptA] != -1 and mask1.img[ptA] != 1:
        print(mask1.img[ptA])
        statistic.append(mask1.img[ptA])


from collections import Counter
dict_label = Counter(statistic)

# 46 on cell
# 85 - 46 = 39 cell no keypoint


