from btimage import BT_image, CellLabelOneImage
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from skimage.transform import rescale
import watershed



def show(image, name="set", detail=False, colorbar=True):
    if detail:
        plt.figure(dpi=200, figsize=(10, 10))
    else:
        plt.figure()
    plt.title(name)
    plt.imshow(image, cmap="gray")
    if colorbar:
        plt.colorbar()
    plt.show()


def show_hist(image, bin, img_title):
    plt.figure()
    plt.title(img_title)
    plt.hist(image.flatten(), bins=bin)
    plt.show()


def phase2uint8(image):
    image[image >= 4] = 0
    image[image <= -0.5] = -0.5
    max_value = image.max()
    min_value = image.min()
    image_rescale = (image - min_value) * 255 / (max_value - min_value)
    t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
    t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
    image = np.uint8(np.round(image_rescale))
    return image


def adaptive_threshold(image):
    array_image = image.flatten()
    # plt.figure()
    n, b, patches = plt.hist(array_image, bins=200)
    # plt.title("Histogram of phase image")
    # plt.xlabel("gray value")
    # plt.ylabel("number of pixel")
    # plt.show()

    # Adaptive threshold
    n = n[4:]
    bin_max = np.where(n == n.max())[0][0]
    print("bin_max", bin_max)
    max_value = b[bin_max]
    threshold = 0.7 * np.sum(array_image) / len(array_image[array_image > max_value])
    print("Adaptive threshold is:", threshold)
    # thresholding
    ret, thresh_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh_img, threshold


path = "E:\\DPM\\20190614_RPE2\\phase_npy\\img_2019_06_14_19_56_33_phase.npy"


after = CellLabelOneImage(path).run()
plt.figure()
plt.imshow(after, cmap='jet')
plt.show()


# im = BT_image(path)
# im.opennpy()
#
# gray = phase2uint8(im.img)
# show(gray, "gray")
#
# # image smoothing
# gray = cv2.GaussianBlur(gray, (7, 7), sigmaX=1)
# show(gray, "gaussian filter")
#
# # # image sharpening --- Sigmoid Correction
# image_contrast = np.uint8(gray.copy())
# image_contrast = enhance_contrast(image_contrast, disk(5))
# show(image_contrast, "image_contrast")
# show_hist(image_contrast, 200, "before")
# img1 = adjust_sigmoid(image_contrast, cutoff=0.08, gain=18)
# show(img1, "Sigmoid Correction")
# show_hist(img1, 200, "after")
#
#
# gain = 18
# cutoff = 0.08
# x = np.arange(0, 1, 0.01)
# y = 1/(1 + np.exp((gain*(cutoff - x))))
# plt.figure()
# plt.title("Sigmoid Correction (cutoff: 0.08, gain: 18)")
# plt.plot(x, y)
# plt.show()
#
#
# # manually thresholding
# image_binary = img1.copy()
# image_binary, th = adaptive_threshold(image_binary)
# show(image_binary, "image_binary")
#
# kernel = np.ones((3, 3), np.uint8)
# image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel, iterations=4)
# kernel = np.ones((30, 30), np.uint8)
# image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel, iterations=1)
# show(image_binary, "image_binary_morphological_oper")

# # sure bg
# sure_bg = image_binary.copy()
# kernel = np.ones((5, 5), np.uint8)
# sure_bg = np.uint8(cv2.dilate(sure_bg, kernel, iterations=8))
# show(sure_bg, "sure_bg")

# # distance transform
# dist_transform = cv2.distanceTransform(np.uint8(image_binary), 1, 5)
# show(dist_transform, 'dist_transform')
# dist_binary = dist_transform.copy()
# dist_binary[dist_binary < 20] = 0
# show(dist_binary, "dist_binary")

# # rgb img
# rgb = cv2.cvtColor(np.uint8(dist_binary), cv2.COLOR_GRAY2BGR)

# # find local maximum
# marker = np.zeros((3072, 3072), np.uint8)
# local_maxi = peak_local_max(dist_binary, indices=False, footprint=np.ones((220, 220)))
# marker[local_maxi == True] = 255
# kernel = np.ones((5, 5), np.uint8)
# marker = np.uint8(cv2.dilate(marker, kernel, iterations=8))

# # label local maximum
# ret, markers1 = cv2.connectedComponents(marker)
# markers1[sure_bg == 0] = 1
# markers1 = np.int32(markers1)

# plt.figure()
# plt.title("markers1")
# plt.imshow(markers1, cmap='jet')
# plt.colorbar()
# plt.show()
#
# # watershed
# pre_markers = markers1.copy()
# after_water = cv2.watershed(rgb, markers1)
#
# plt.figure()
# plt.title("after_water")
# plt.imshow(after_water, cmap='jet')
# plt.colorbar()
# plt.show()

# # overlay
# gray = np.uint8(gray)
# after_water = np.uint8(after_water)
# plt.figure(figsize=(10, 10))
# plt.imshow(after_water, cmap='jet', alpha=0.5, vmax=45, vmin=0)
# plt.imshow(gray, cmap='gray', alpha=0.5)
# plt.show()


# # maunally adjust
# gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# dist_binary = cv2.cvtColor(np.uint8(dist_binary), cv2.COLOR_GRAY2BGR)
# print(watershed.__doc__)
# r = watershed.App(dist_binary, pre_markers, gray)
# r.run()

