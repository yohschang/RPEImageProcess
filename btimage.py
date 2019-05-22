import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def round_all_the_entries_ndarray(matrix, decimal):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = round(matrix[i, j], decimal)
    return matrix


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
                if (i - centerx)**2 + (j - centery)**2 <= 3136:
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

