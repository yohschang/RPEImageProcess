import cv2
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt

def crop(path, file_name, x, y, w, h):
    my_file = Path(path+file_name)
    if not my_file.exists():
        raise OSError("Cannot find image!")
    img = cv2.imread(path+file_name)
    if (img.shape[0] != 3072) or (img.shape[1] != 4096):
        print("img size is not 3072*4096!")
        return
    crop_img = img[y:y+h, x:x+w]
    cv2.imwrite(path + file_name.split(".")[0] + "crop.bmp", crop_img)
    return

def stack(path, file_name):
    buffer = np.zeros((3072, 4096))
    for i in range(len(glob.glob(path + file_name))):
        img = cv2.imread(glob.glob(path + file_name)[i])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        buffer += gray
    cv2.imwrite("E:\\DPM\\20190502_10 um_pinhole" + "\\stack.bmp", buffer)
    return
###################################################################
path = "E:\\DPM\\20190502*"
file_name = "\\img_2019_05*.bmp"
x = 1440
y = 1880
w = 1024
h = 1024

# stack
stack(path, file_name)

# crop
# crop(path, file_name, x, y, w, h)



# plt.imshow(buffer)
