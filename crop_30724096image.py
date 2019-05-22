import cv2
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class BT_image(object):
    """docstring"""
    def __init__(self, path):
        my_file = Path(path)
        if not my_file.exists():
            raise OSError("Cannot find image!")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img
        self.name = path.split("\\")[-1]
        self.board = None
        self.flat_board = None

    def beadcenter2croprange(self, x, y):
        return x - 84, y - 84

    def crop(self, x_center, y_center):
        w = 168
        h = 168
        x, y = self.beadcenter2croprange(x_center, y_center)
        self.img = self.img[y:y+h, x:x+w]

    def crop_img2circle(self):
        """choose the area of bead and append to a list"""
        radius = 50  # pixel
        self.board = np.zeros((self.img.shape[0], self.img.shape[0]))
        self.flat_board = []
        centerx, centery = self.img.shape[0]//2 - 1, self.img.shape[0]//2 - 1
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[0]):
                if (i - centerx)**2 + (j - centery)**2 <= 2500:
                    self.board[i, j] = self.img[i, j]
                    self.flat_board.append(self.img[i, j])

    def write_image(self, path):
        cv2.imwrite(path + self.name.split(".")[0] + "_circle.bmp", self.img)

    def plot_it(self):
        plt.figure()
        plt.title(self.name.split(".")[0])
        plt.imshow(self.img, plt.cm.gray)
        plt.show()

    def plot_board(self):
        plt.figure()
        plt.title(self.name.split(".")[0])
        plt.imshow(self.board, plt.cm.gray)
        plt.show()

    def normalize_after_crop(self):
        background = round(float(np.mean(self.img[:20, :20])), 2)
        self.img = self.img - background


def round_all_the_entries_ndarray(matrix, decimal):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = round(matrix[i, j], decimal)
    return matrix


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
for i in range(len(list_)):
    img_tem = BT_image(list_[i])
    img_tem.crop(center_list[i][0], center_list[i][1])
    img_tem.normalize_after_crop()
    # img_tem.plot_it()
    # img_tem.write_image(path)
    im_square.append(img_tem.img.flatten())

    # circle
    img_tem.crop_img2circle()
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



# def read_image(path, file_name):
#     my_file = Path(path+file_name)
#     if not my_file.exists():
#         raise OSError("Cannot find image!")
#     img = cv2.imread(path+file_name)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img