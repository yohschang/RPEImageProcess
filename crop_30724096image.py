import cv2
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt


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

    def Beadcenter2croprange(self, x, y):
        return x - 84, y - 84

    def crop(self, x_center, y_center):
        w = 168
        h = 168
        x, y = self.Beadcenter2croprange(x_center, y_center)
        self.img = self.img[y:y+h, x:x+w]

    def write_image(self, path, file_name):
        cv2.imwrite(path + file_name + ".bmp", self.img)

    def plot_it(self):
        plt.figure()
        plt.title(self.name.split(".")[0])
        plt.imshow(self.img, plt.cm.gray)
        plt.show()


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
list = glob.glob(path + "*_save.bmp")

# manually find the center of bead
center_list = [[265, 2792],
               [2814, 2848],
               [1543, 2820],
               [1579, 1548],
               [301, 1507],
               [2859, 1566],
               [323, 229],
               [2900, 285],
               [1618, 262]]

im = []
for i in range(len(list)):
    img_tem = BT_image(list[i])
    img_tem.crop(center_list[i][0], center_list[i][1])
    img_tem.plot_it()
    im.append(img_tem.img.flatten())

matrix = np.corrcoef(im)



# img1 = BT_image(list[0])
# img2 = BT_image(list[2])
# img1.crop(265, 2792)
# img2.crop(1543, 2820)
# img1.plot_it()
# img2.plot_it()



# def read_image(path, file_name):
#     my_file = Path(path+file_name)
#     if not my_file.exists():
#         raise OSError("Cannot find image!")
#     img = cv2.imread(path+file_name)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return img