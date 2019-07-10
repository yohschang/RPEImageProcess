import os
import numpy as np
import cv2
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage, PrevNowMatching, PrevNowCombo
import glob
from matplotlib import pyplot as plt

# input
# 1. after watershed label
# 2. phase maps
root_path = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\"
output = PrevNowCombo(root_path).combo(now_target=6, save=False)








