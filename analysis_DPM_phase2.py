import cv2
import glob
import numpy as np
import pandas as pd
from btimage import BT_image
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from btimage import round_all_the_entries_ndarray


def stack(path, file_name):
    buffer = np.zeros((3072, 3072))
    img_number = len(glob.glob(path + file_name))
    for i in range(30):
        img = cv2.imread(glob.glob(path + file_name)[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        buffer += gray
    cv2.imwrite("E:\\DPM\\20190510\\540bandpass" + "\\stack.bmp", buffer)
    return

###################################################################


path = "E:\\DPM\\20190521_8position\\8position\\"
list_ = glob.glob(path + "*.phimap")

# manually find the center of bead
center_list = [[1543, 2822],
               [265, 2792],
               [2814, 2848],
               [1579, 1548],
               [301, 1506],
               [2862, 1564],
               [1618, 262],
               [326, 226],
               [2901, 285]]

im_square = []
im_circle = []
im_test = []

im_tem = BT_image(list_[3])
im_tem.open_raw_image()
im_tem.crop(center_list[3][0], center_list[3][1])
# im_tem.plot_it(im_tem.img)
im_tem.find_centroid()
im_tem.crop_img2circle_after_crop_it_to_tiny_square(im_tem.centroid_x, im_tem.centroid_y)
image_template = im_tem.board[38:130, 38:130]
image_template = image_template.astype(np.float32)
im_tem.plot_it(image_template)
im_test.append(image_template.flatten())

fig, axes = plt.subplots(4, 2, figsize=(6, 12))
a, b = 0, 0
i = 0

for a, b, i in zip([0,1,2,3,0,1,2,3], [0,0,0,0,1,1,1,1], [0,1,2,4,5,6,7,8]):
    img_tar = BT_image(list_[i])
    img_tar.open_raw_image()
    img_tar.crop(center_list[i][0], center_list[i][1])
    axes[a, b].imshow(img_tar.img, cmap='gray')
    axes[a, b].set_title(img_tar.name.strip("phimap"), fontsize=15)
    axes[a, b].axis('off')
fig.tight_layout()
fig.show()

for i in tqdm.trange(len(list_)):
    if i != -1:
        img_tar = BT_image(list_[i])
        img_tar.open_raw_image()
        img_tar.crop(center_list[i][0], center_list[i][1])
        # img_tar.plot_it(img_tar.img)
        result = cv2.matchTemplate(img_tar.img, image_template, cv2.TM_CCORR)
        # plt.figure()
        # plt.imshow(result, cmap='gray')
        # plt.title("correlation coefficients map")
        # plt.show()
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(i, "th :", max_val, max_loc)
        # img_tem.write_image(path)
        im_square.append(img_tar.img.flatten())

        # circle
        # img_tem.find_centroid()
        # img_tem.plot_it(img_tem.threshold)
        # img_tem.crop_img2circle_after_crop_it_to_tiny_square(img_tem.centroid_x, img_tem.centroid_y)
        # img_tem.plot_it(img_tem.board)
        # im_circle.append(img_tem.flat_board)
        row_shift = max_loc[0] - 38
        col_shift = max_loc[1] - 38
        print("row_shift:", row_shift)
        print("col_shift:", col_shift)
        img_regis = BT_image(list_[i])
        img_regis.open_raw_image()
        b = 92//2
        startx = center_list[i][0] + row_shift - b
        starty = center_list[i][1] + col_shift - b
        img_regis.img = img_regis.img[starty: starty+92, startx: startx+92]
        # img_regis.plot_it(img_regis.img)
        im_circle.append(img_regis.img.flatten())
        if i == 3:
            im_test.append(img_regis.img.flatten())





# # Input is an array
# matrix_square = np.corrcoef(im_square)
matrix_circle = np.corrcoef(im_circle)
matrix_circle = round_all_the_entries_ndarray(matrix_circle, 3)
col = ["bottom", "bottom left", "bottom right",
       "center", "left", "right",
       "top", "top left", "top right"]
#
# dataframe visualization
df = pd.DataFrame(data=matrix_circle, columns=col)
df.insert(0, " ", col)

# matrix_circle_test = np.corrcoef(im_test)
# df_test = pd.DataFrame(data=matrix_circle_test)

# calculate linearity
m_list = []
b_list = []
r2_list = []
# for i in tqdm.trange(len(col)):
i = 3

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
a, b = 0, 0
#
# for a, b, i in zip([0,0,0,0,1,1,1,1], [0,1,2,3,0,1,2,3], [0,1,2,4,5,6,7,8]):
#     axes[a, b].imshow(img_tar.img, cmap='gray')
#     axes[a, b].set_title(img_tar.name.strip("phimap"), fontsize=15)
#     axes[a, b].axis('off')

for p, k, j in zip([0,0,0,0,1,1,1,1], [0,1,2,3,0,1,2,3], [0,1,2,4,5,6,7,8]):
    (m, b) = np.polyfit(im_circle[3], im_circle[j], 1)
    sort_x = np.sort(im_circle[3])

    yp = np.polyval([m, b], sort_x)
    coef = r2_score(im_circle[3], im_circle[j])
    # plt.figure()
    # plt.scatter(im_circle[3], im_circle[j], s=2)
    # plt.title(" {} and {}".format(col[3], col[j]))
    # plt.plot(sort_x, yp, 'r', label="y={}*x + {}.\nR square={}".format(round(m,2), round(b,2), round(coef,2)))
    # plt.legend()
    # plt.xlabel("phase (rad)")
    # plt.ylabel("phase (rad)")
    axes[p, k].plot(sort_x, yp, 'r', label="y={}*x+{}.\nR square={}".format(round(m,2), round(b,2), round(coef,2)))
    axes[p, k].set_title(" {} and {}".format(col[3], col[j]), fontsize=10)
    axes[p, k].scatter(im_circle[3], im_circle[j], s=1)
    axes[p, k].legend(fontsize=8)
    if k == 0:
        axes[p, k].set_ylabel("phase (rad)")
    if p == 1:
        axes[p, k].set_xlabel("phase (rad)")
    # plt.savefig("E:\\DPM\\20190609\\{}_{}.png".format(i, j))
    plt.show()
    m_list.append(m)
    b_list.append(b)
    r2_list.append(coef)

fig.tight_layout()
fig.show()

print("m mean: ", np.mean(m_list), "\nm sd: ", np.std(m_list))
print("b mean: ", np.mean(b_list), "\nb sd: ", np.std(b_list))
print("r2 mean: ", np.mean(r2_list), "\nr2 sd: ", np.std(r2_list))


# img_demo = BT_image(path + "center2_save.bmp")
# plt.figure()
# plt.imshow(img_demo.img, plt.cm.gray)
# for point in center_list:
#     plt.scatter(point[0], point[1], s=10, c='b')
# plt.show()
