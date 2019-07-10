import os
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage
import glob
import numpy as np
from matplotlib import pyplot as plt

path = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\"
change_file_path = "E:\\DPM\\20190708\\Bead\\2\\SP\\time-lapse\\"

# for i in range(1, 11):
#     dir_num = str(i)
#     new = str(i+30)
#     check_file_exist(change_file_path+dir_num, dir_num)
#     os.rename(change_file_path+dir_num, change_file_path+new)
