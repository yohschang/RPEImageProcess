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


path = "E:\\DPM\\20190521\\8position\\"
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
for i in tqdm.trange(len(list_)):
    img_tem = BT_image(list_[i])
    img_tem.open_raw_image()

    img_tem.crop(center_list[i][0], center_list[i][1])
    # img_tem.plot_it(img_tem.img)
    # img_tem.normalize_after_crop()

    # img_tem.write_image(path)
    im_square.append(img_tem.img.flatten())

    # circle
    img_tem.find_centroid()
    img_tem.plot_it(img_tem.threshold)
    img_tem.crop_img2circle_after_crop_it_to_tiny_square(img_tem.centroid_x, img_tem.centroid_y)
    img_tem.plot_it(img_tem.board)
    im_circle.append(img_tem.flat_board)

# # Input is an array
# matrix_square = np.corrcoef(im_square)
matrix_circle = np.corrcoef(im_circle)
matrix_circle = round_all_the_entries_ndarray(matrix_circle, 3)
col = ["bottom", "bottom left", "bottom right",
       "center", "left", "right",
       "top", "top left", "top right"]

# dataframe visualization
df = pd.DataFrame(data=matrix_circle, columns=col)
df.insert(0, " ", col)


m_list = []
b_list = []
r2_list = []
for i in tqdm.trange(len(col)):
    for j in range(len(col)):
        if i > j:
            (m, b) = np.polyfit(im_circle[i], im_circle[j], 1)
            sort_x = np.sort(im_circle[i])

            yp = np.polyval([m, b], sort_x)
            coef = r2_score(im_circle[i], im_circle[j])
            plt.figure()
            plt.scatter(im_circle[i], im_circle[j], s=2)
            plt.title("Comparison of phase per pixel between {} and {}".format(col[i], col[j]))
            plt.plot(sort_x, yp, 'r', label="y={}*x + {}.\nR square={}".format(round(m,2), round(b,2), round(coef,2)))
            plt.legend()
            plt.xlabel("phase (rad)")
            plt.ylabel("phase (rad)")
            plt.savefig("E:\\DPM\\20190529\\{}_{}.png".format(i, j))
            plt.show()
            m_list.append(m)
            b_list.append(b)
            r2_list.append(coef)

print("m mean: ", np.mean(m_list), "\nm sd: ", np.std(m_list))
print("b mean: ", np.mean(b_list), "\nb sd: ", np.std(b_list))
print("r2 mean: ", np.mean(r2_list), "\nr2 sd: ", np.std(r2_list))


# img_demo = BT_image(path + "center2_save.bmp")
# plt.figure()
# plt.imshow(img_demo.img, plt.cm.gray)
# for point in center_list:
#     plt.scatter(point[0], point[1], s=10, c='b')
# plt.show()
