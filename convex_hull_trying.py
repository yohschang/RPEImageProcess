from btimage import BT_image
import cv2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import watershed
from math import hypot, atan2, degrees
from skimage.segmentation import clear_border
from skimage.measure import label

def correct_angle(angle):
    if 90 <= angle < 270:
        angle -= 180
    elif 270 <= angle < 360:
        angle -= 360
    elif -270 < angle <= -90:
        angle += 180
    elif -360 < angle <= -270:
        angle += 360
    return angle


def two_line_angle(point1_1, point1_2, point2_1, point2_2, t=25):
    x1, y1 = point1_1
    x2, y2 = point1_2
    line1deg = degrees(atan2(y2-y1, x2-x1))
    line1deg = correct_angle(line1deg)
    # print("line1 deg:", line1deg)
    x3, y3 = point2_1
    x4, y4 = point2_2
    line2deg = degrees(atan2(y4-y3, x4-x3))
    line2deg = correct_angle(line2deg)
    # print("line2 deg:", line2deg)
    difference = abs(line2deg - line1deg)
    print(difference)
    if difference <= t:
        return False
    else:
        return True


def debug(obj_list, contours):
    # plot debug
    plot_board = np.zeros((3072, 3072, 3), np.uint8)
    plot_board = cv2.drawContours(plot_board, contours, -1, (255, 255, 255), 5, 3)
    for obj in obj_list:
        if obj.size_too_small:
            plot_board[obj.only_this_cr_img == 255] = (255, 0, 0)
        if obj.too_small:
            plot_board[obj.only_this_cr_img == 255] = (0, 255, 0)
        if obj.internal_convex_region:
            plot_board[obj.only_this_cr_img == 255] = (0, 0, 255)
    plt.figure()
    plt.title("r--> size; g--> d; b--> internal")
    plt.imshow(plot_board)
    plt.show()


def clean_cr(list_of_obj):
    new = []
    trash_can = []
    for obj in list_of_obj:
        if not obj.delete:
            new.append(obj)
        else:
            print("del ", obj.num)
    return new


def connect_to_neighbor(internal, normal):
    min_d = 1000000
    internal_p = None
    normal_p = None
    for i in internal.inner_boundary:
        for j in normal:
            for k in j.inner_boundary:
                sy, sx = i
                y, x = k
                if j.splitting_point:
                    ssy, ssx = j.splitting_point
                    splitting_region = hypot(ssx - x, ssy - y)
                    if splitting_region >= 40:
                        dist = hypot(sx - x, sy - y)
                        if dist < min_d:
                            min_d = dist
                            internal_p = (sx, sy)
                            normal_p = (x, y)
                else:
                    dist = hypot(sx - x, sy - y)
                    if dist < min_d:
                        min_d = dist
                        internal_p = (sx, sy)
                        normal_p = (x, y)

    return internal_p, normal_p, min_d


class ConvexRegion(object):
    def __init__(self, father_num, only_this_cr_img, num, cr_contours):
        self.father_num = father_num
        self.only_this_cr_img = only_this_cr_img
        self.num = num
        self.cr_contours = cr_contours
        self.weight = 0
        self.splitting_point = None
        self.internal_convex_region = False
        self.inner_boundary = []
        self.outer_boundary = []
        self.too_small = False
        self.size_too_small = False
        self.delete = False

    def calculate_weight(self):
        self.weight = np.count_nonzero(self.only_this_cr_img)

    def plot(self):
        plt.figure()
        plt.title("Father"+str(self.father_num)+"Kid"+str(self.num))
        plt.imshow(self.only_this_cr_img, cmap='jet')
        plt.show()

    def assert_outboundary(self):
        if not self.outer_boundary:
            self.internal_convex_region = True

    def assert_size(self):
        if self.weight < 600:
            self.size_too_small = True

    def find_splitting_point(self, convexhull_shell):
        target_img = self.only_this_cr_img.copy()
        # inner boundary --> 50
        target_img = cv2.drawContours(target_img, self.cr_contours, -1, 50, 2, 3)
        # outer boundary --> 100
        target_img[(target_img == 50) & (convexhull_shell == 255)] = 100

        # find the splitting point
        # how to get inner outer boundary coordinate?
        for j in range(len(self.cr_contours)):
            x, y = self.cr_contours[j][0][0], self.cr_contours[j][0][1]
            if target_img[y, x] == 50:
                self.inner_boundary.append([y, x])
            elif target_img[y, x] == 100:
                self.outer_boundary.append([y, x])
        self.assert_outboundary()

        if not self.internal_convex_region:
            p1 = self.outer_boundary[0]
            p2 = self.outer_boundary[-1]
            p2_p1_vector = np.subtract(p2, p1)

            d = []
            for p3 in self.inner_boundary:
                p1_p3_vector = np.subtract(p1, p3)
                cur_d = np.linalg.norm(np.cross(p2_p1_vector, p1_p3_vector)) / np.linalg.norm(p2_p1_vector)
                d.append(cur_d)
            if np.max(d) <= 60:
                self.too_small = True
            pointy, pointx = self.inner_boundary[int(np.argmax(d))]
            self.splitting_point = (pointx, pointy)
            # print(self.splitting_point)
            # plt.figure()
            # plt.title("num"+str(self.num))
            # plt.imshow(target_img, cmap='jet')
            # plt.scatter(pointx, pointy, s=20, c='g')
            # plt.show()


def up2cut(image, obj1, obj2, color):
    image = cv2.line(image, obj1.splitting_point, obj2.splitting_point, color, thickness=5, lineType=8)
    return image


def cut(image, point1, point2, color):
    image = cv2.line(image, point1, point2, color, thickness=5, lineType=8)
    return image


def find_max_weight(list_of_obj):
    max_cr_weight = 0
    record_index = 0
    max_cr = None
    for i, cr_obj in enumerate(list_of_obj):
        if cr_obj.weight > max_cr_weight:
            max_cr_weight = cr_obj.weight
            max_cr = cr_obj
            record_index = i
    list_of_obj.pop(record_index)
    return max_cr, list_of_obj


def calculate_centroid(image):
    m = cv2.moments(image)
    # calculate x,y coordinate of center
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])
    return cX, cY


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


def phase2uint8(image):
    max_value = 2
    min_value = -0.5
    image_rescale = (image - min_value) * 255 / (max_value - min_value)
    t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
    t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
    image = np.uint8(image_rescale)
    return image


def adaptive_threshold(image):
    array_image = image.flatten()
    # plt.figure()
    n, b, patches = plt.hist(array_image, bins=100)
    # plt.title("Histogram of phase image")
    # plt.xlabel("gray value")
    # plt.ylabel("number of pixel")
    # plt.show()

    # Adaptive threshold
    n = n[1:]
    bin_max = np.where(n == n.max())[0][0]
    print("bin_max", bin_max)
    max_value = b[bin_max]
    threshold = np.sum(array_image) / len(array_image[array_image > max_value])
    print("Adaptive threshold is:", threshold)
    # thresholding
    ret, thresh_img = cv2.threshold(image, 0.9*threshold, 255, cv2.THRESH_BINARY)

    # plt.figure()
    # plt.title("After Adaptive threshold")
    # plt.imshow(thresh_img, cmap='gray')
    # plt.show()

    return thresh_img, threshold
##################################  Start  ################################


path = "E:\\DPM\\20190614_RPE2\\phase_npy\\img_2019_06_14_19_56_33_phase.npy"

im = BT_image(path)
im.opennpy()

gray = phase2uint8(im.img)
show(gray, "gray")

# adaptive thresholding
img, t = adaptive_threshold(np.array(gray))
show(img, "img")

# image sharpening - power law transformation
img1 = np.power(gray/float(np.max(gray)), 0.4)


# img1 = img1 * 255
# image = np.uint8(img1)
# image = cv2.bilateralFilter(image, 9, 75, 75)
# th = 125
# image_binary = image.copy()
# image_binary[image_binary >= th] = 255
# image_binary[image_binary < th] = 0
# image[image_binary != 255] = 0
#
#
# kernel = np.ones((3, 3), np.uint8)
# image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel, iterations=4)
#
# cleared = label(image_binary)
# show(image_binary, "image_binary")
# show(cleared, "cleared")

# la = cv2.Laplacian(image, -1)
#
# kernel = np.ones((3, 3), np.uint8)
# la = cv2.morphologyEx(la, cv2.MORPH_CLOSE, kernel, iterations=2)


# ##################################  foreground  ###########################
raw = gray.copy()
fore_back_marker = np.zeros((gray.shape[0], gray.shape[1]))

# foreground is 1; background is 0
fore_back_marker[img == 255] = 1

# opening
kernel = np.ones((5, 5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
show(img, "opening img")
cleared = label(img)
show(cleared, "cleared")

##################################  segmentation  ###########################
# find contour
img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
# img_contours = cv2.drawContours(rgb, contours, -1, (0, 0, 255), 10)
# plt.figure()
# plt.imshow(img_contours)
# plt.show()

# find convex hull
hull = []
length_contours = len(contours)
for i in range(length_contours):
    # remove convex hull too simple
    if len(contours[i]) >= 150:
        hull.append(cv2.convexHull(contours[i], False))

# fill convex hull
img_convex_hull = img.copy()
for i in tqdm.trange(len(hull)):
    img_convex_hull = cv2.fillConvexPoly(img_convex_hull, hull[i], 255)
show(img_convex_hull, "img_convex_hull")

# label convex hull marker
magic_list = []
ch_marker = np.zeros((3072, 3072))
for i in tqdm.trange(len(hull)):
    ch_marker = cv2.fillConvexPoly(ch_marker, hull[i], i*5+10)
    magic_list.append(i*5+10)
show(ch_marker, "ch_marker")

# label convex hull shell marker
ch_shell_marker = cv2.drawContours(np.zeros((3072, 3072)), hull, -1, 255, 5, 3)
show(ch_shell_marker, "ch_shell_marker")
# # draw contours and hull points
# plt.figure(dpi=200, figsize=(10, 10))
# color = (255, 0, 0) # blue - color for convex hull
# for i in tqdm.trange(len(hull)):
#     # draw ith convex hull object
#     b = cv2.drawContours(rgb, hull, i, color, 10, 3)
#     plt.imshow(b)
# plt.show()

# convex region
cr = img_convex_hull - img
show(cr, "substrate")

# crop it
######################## for loop for each convex hull ##################
# for i in magic_list:
ch_i = 160
print(ch_i)

# choose target region
cr_crop = cr.copy()
cr_crop[ch_marker != ch_i] = 0
mass = np.count_nonzero(cr_crop)
print("mass", mass)
# if mass <= 5000:
#     continue
show(cr_crop, "cr_crop" + str(ch_i))


# opening depend on "mass"
if mass >= 8000:
    kernel = np.ones((3, 3), np.uint8)
elif 8000 < mass <= 30000:
    kernel = np.ones((8, 8), np.uint8)
elif mass > 30000:
    kernel = np.ones((8, 8), np.uint8)


cr_crop = cv2.morphologyEx(cr_crop, cv2.MORPH_OPEN, kernel, iterations=7)
cr_crop = cv2.morphologyEx(cr_crop, cv2.MORPH_CLOSE, kernel, iterations=7)
show(cr_crop, "cr_crop_opening" + str(ch_i))

#-----------------------------------------------------------------
flag = 1
while flag == 1:

    flag = 0
    plot_list = []
    # how many convex region
    cr_crop, cr_contours, cr_hierarchy = cv2.findContours(cr_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("How many convex region: ", len(cr_contours))

    # current label
    cr_crop_marker = cr_crop.copy()
    ret, cr_crop_marker = cv2.connectedComponents(cr_crop_marker)
    ######################## for loop for each convex region ##################

    # create cr object list TODO: create
    convex_region_obj = []
    for_case1_connection = []
    previous = 0
    for i, current in enumerate(cr_contours):
        cur_num = cr_crop_marker[current[0][0][1], current[0][0][0]]
        print(cur_num)
        if cur_num == previous:
            print("repeat", cur_num)
            continue
        # highlight one region
        cr_crop_current = cr_crop.copy()
        cr_crop_current[cr_crop_marker != cur_num] = 0
        # show(cr_crop_current, "contour"+str(cur_num))
        cr1 = ConvexRegion(ch_i, cr_crop_current, cur_num, cr_contours[i])
        cr1.calculate_weight()
        cr1.find_splitting_point(ch_shell_marker)

        convex_region_obj.append(cr1)
        for_case1_connection.append(cr1)
        plot_list.append(cr1)

        previous = cur_num
    print("len of convex_region_obj", len(convex_region_obj))
    black = np.zeros((3072, 3072), np.uint8)


    for i, obj in enumerate(convex_region_obj):
        obj.assert_size()
        if obj.size_too_small:
            print("small:", obj.num)
            obj.delete = True

        if obj.internal_convex_region:
            internal_cr = obj
            print("internal region:", obj.num)
            # obj.plot()
            internal_point, normal_point, distance = connect_to_neighbor(internal_cr, convex_region_obj)
            black = cv2.line(black, internal_point, normal_point, 255)
            black = cv2.drawContours(black, cr_contours, -1, 255, 5, 3)
            cr_crop = cv2.line(cr_crop, internal_point, normal_point, 255)
            obj.delete = True
            flag = 1

        if obj.too_small:
            print("d small:", obj.num)
            obj.delete = True

    convex_region_obj = clean_cr(convex_region_obj)


show(cr_crop, "crop")

#-----------------------------------------------------------------

print("How many meaningful convex region: ", len(convex_region_obj))
print("num  weight  too_small  too small d  internal")
for obj in convex_region_obj:
    print(obj.num, "\t", obj.weight, "\t\t", obj.size_too_small, "\t\t", obj.too_small, "\t\t", obj.internal_convex_region)
print()

#####
debug(plot_list, cr_contours)

convex_region_obj_backup = convex_region_obj.copy()


# cutting algorithm TODO: cutting
while convex_region_obj:

    if len(convex_region_obj) > 1:
        max_cr_current, convex_region_obj = find_max_weight(convex_region_obj)

        sx, sy = max_cr_current.splitting_point
        p1 = max_cr_current.outer_boundary[0]
        p2 = max_cr_current.outer_boundary[-1]
        short_dist = 100000
        short_dist_index = 0
        # plt.figure()
        # plt.imshow(np.zeros((3072, 3072), np.uint8))
        for i, rest in enumerate(convex_region_obj):
            x, y = rest.splitting_point
            dist = hypot(sx - x, sy - y)
            # plt.scatter(sx, sy, s=20, c='r')
            # plt.scatter(x, y, s=20, c='g')
            if two_line_angle(p1, p2, (sx, sy), (x, y)):
                if dist < short_dist:
                    short_dist = dist
                    short_dist_index = i
        # plt.show()
        img = up2cut(img, max_cr_current, convex_region_obj[short_dist_index], 0)
        black = up2cut(black, max_cr_current, convex_region_obj[short_dist_index], 255)
        print("two region cut", max_cr_current.num, "and ", convex_region_obj[short_dist_index].num)
        convex_region_obj.pop(short_dist_index)
        print("How many meaningful convex region: ", len(convex_region_obj))

    elif len(convex_region_obj) == 1:
        max_cr_current = convex_region_obj[0]
        if max_cr_current.weight < 1500:
            break
        else:
            print("here:", len(for_case1_connection))
            # others convex region
            buffer = []
            for i, obj in enumerate(for_case1_connection):
                if obj.num != max_cr_current.num:
                    buffer.append(obj)

            print("here2:", len(buffer))
            min_dist = 100000
            min_point = (0, 0)
            sx, sy = max_cr_current.splitting_point
            p1 = max_cr_current.outer_boundary[0]
            p2 = max_cr_current.outer_boundary[-1]

            for i, rest in enumerate(buffer):
                for inner in rest.inner_boundary:
                    y, x = inner
                    dist = hypot(sx - x, sy - y)
                    if two_line_angle(p1, p2, (sy, sx), (y, x), t=60):
                        if dist < min_dist:
                            min_dist = dist
                            min_point = (x, y)

            # plt.figure()
            # plt.imshow(np.zeros((3072, 3072), np.uint8))
            # plt.scatter(min_point[0], min_point[1], s=20, c='w')
            # plt.show()
            img = cut(img, (sx, sy), min_point, 0)
            black = cv2.line(black, (sx, sy), min_point, 255, thickness=5, lineType=8)
            print("one region cut", max_cr_current.num, "and inner boundary", min_point)
            break
    else:
        break

plt.figure()
plt.title("After cut algorithm")
plt.imshow(black, cmap="gray")
plt.colorbar()
plt.show()

# # opening & closing
# kernel = np.ones((5, 5), np.uint8)
# cr = cv2.morphologyEx(cr, cv2.MORPH_OPEN, kernel, iterations=2)
# show(cr, "convex region")



