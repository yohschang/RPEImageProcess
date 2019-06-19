from btimage import BT_image
import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import watershed


def show(image, name="set", detail=False):
    if detail:
        plt.figure(dpi=200, figsize=(10, 10))
    else:
        plt.figure()
    plt.title(name)
    plt.imshow(image, cmap="gray")
    plt.colorbar()
    plt.show()


path = "E:\\DPM\\20190614_RPE2\\phase_npy\\img_2019_06_14_19_56_33_phase.npy"

im = BT_image(path)
im.opennpy()
plt.figure()
plt.title("original")
plt.imshow(im.img, cmap="gray", vmax=2, vmin=-0.5)
plt.colorbar()
plt.show()
max_value = 2
min_value = -0.5
image_rescale = (im.img - min_value) * 255 / (max_value - min_value)
t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
image = np.uint8(image_rescale)
image = cv2.GaussianBlur(image, (5, 5), 0)
rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# show(rgb)
# plt.figure()
# plt.hist(image_rescale.flatten(), bins= 100)
# # plt.colorbar()
# plt.show()

# show(image_rescale)

# ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(image, 95, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
show(thresh, "threshold")

# noise remove
kernel = np.ones((20, 20), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
show(opening, "opening")

# fg
# kernel = np.ones((8, 8), np.uint8)
# dist_transform = cv2.distanceTransform(opening, 1, 5)
# show(dist_transform, 'dist_transform')
# ret, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
# loc_max = peak_local_max(dist_transform, min_distance=2, indices=False, threshold_abs=40)
sure_fg = np.uint8(opening)
sure_fg[sure_fg == 255] = 255
sure_fg[sure_fg != 255] = 0
sure_fg = cv2.erode(sure_fg, np.ones((3, 3), np.uint8), iterations=5)
show(sure_fg, 'sure_fg', detail=False)
######################################################################

ret, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=6)
# show(thresh, "threshold")

# noise remove
kernel = np.ones((20, 20), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# show(opening, "opening")
# sure bg
kernel = np.ones((20, 20), np.uint8)
sure_bg = np.uint8(cv2.dilate(opening, kernel, iterations=8))
show(sure_bg, 'sure_bg')


# watershed fg: 2: bg: 1 un: 0
ret, markers1 = cv2.connectedComponents(sure_fg)
markers = np.int32(markers1)
markers[markers != 0] += 1
markers[markers != 0] *= 2
markers[sure_bg == 0] = 1
show(markers, "markers")
pre_marker = markers.copy()
show(pre_marker, "pre_marker")
markers2 = cv2.watershed(rgb, markers)
show(markers2, "markers2")
# show(markers, "markers_after")

# # marker the boundary
# rgb[markers2 == -1] = [255, 0, 0]
# marker the boundary
# rgb[markers2 == 0] = [255, 0, 0]
# show(rgb, "rgb")
# rgb[markers2 == 1] = [0, 0, 0]
# show(rgb, "rgb without bg")

# maunally adjust
markers = np.zeros((3072, 3072), np.int32)
print(watershed.__doc__)
r = watershed.App(rgb, pre_marker)
r.run()
a = r.overlay
b = r.markers.copy()
m = r.m



# Canny
# rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# result = cv2.Canny(rgb, 65, 100)
# plt.figure(dpi=300, figsize=(10, 10))
# plt.imshow(result, cmap="gray")
# # plt.colorbar()
# plt.show()

# im.plot_it(im.img)
# image_rescale = (im.img + 0.5) * 255 / (3.8 + 0.5)
# t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
# t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
# image = np.uint8(image_rescale)
# kernel = np.ones((8,8), np.uint8)
# image = cv2.GaussianBlur(image, (5, 5), 0)
#
# result = cv2.Canny(image, 10, 100, 3)
# # result_lap = cv2.Laplacian(image, cv2.CV_64F)
# # result_lap = np.uint8(np.absolute(result_lap))
# plt.figure()
# plt.imshow(image, cmap="gray", vmax=255, vmin=0)
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.imshow(result, cmap="gray")
# plt.colorbar()
# plt.show()
