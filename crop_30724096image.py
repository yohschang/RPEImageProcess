import cv2
import glob
import numpy as np
import pandas as pd
from btimage import BT_image
from btimage import round_all_the_entries_ndarray
import tqdm

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
list_ = glob.glob(path + "*_save.bmp")

# manually find the center of bead
center_list = [[265, 2792],
               [2814, 2848],
               [1543, 2822],
               [1579, 1548],
               [301, 1507],
               [2862, 1564],
               [326, 226],
               [2901, 285],
               [1618, 262]]

im_square = []
im_circle = []
for i in tqdm.trange(len(list_)):
    img_tem = BT_image(list_[i])
    img_tem.open_image()
    img_tem.crop(center_list[i][0], center_list[i][1])
    img_tem.normalize_after_crop()
    # img_tem.plot_it()
    # img_tem.write_image(path)
    im_square.append(img_tem.img.flatten())

    # circle
    img_tem.crop_img2circle_after_crop_it_to_tiny_square()
    img_tem.plot_board()
    im_circle.append(img_tem.flat_board)

# Input is an array
matrix_square = np.corrcoef(im_square)
matrix_circle = np.corrcoef(im_circle)
matrix_circle = round_all_the_entries_ndarray(matrix_circle, 3)
col = ["bottom left", "bottom right", "bottom",
       "center", "left", "right",
       "top left", "top right", "top"]

# dataframe visualization
df = pd.DataFrame(data=matrix_circle, columns=col)
df.insert(0, " ", col)



# img_demo = BT_image(path + "center2_save.bmp")
# plt.figure()
# plt.imshow(img_demo.img, plt.cm.gray)
# for point in center_list:
#     plt.scatter(point[0], point[1], s=10, c='b')
# plt.show()
