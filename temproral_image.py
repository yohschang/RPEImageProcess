from btimage import BT_image
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "E:\\DPM\\20190521\\time\\1\\SP\\"
list_ = glob.glob(path + "*.bmp")

average_img = np.zeros((3072, 3072))
for i in tqdm.trange(len(list_)):
    img_tem = BT_image(list_[i])
    average_img += img_tem.img

average_img /= len(list_)

plt.figure()
plt.title("Interferogram Averaging 500 images")
plt.imshow(average_img, plt.cm.gray)
plt.show()

cv2.imwrite(path + "bg.bmp", average_img)

