import os
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage
import glob
from matplotlib import pyplot as plt


path = "E:\\DPM\\20190708\\SP\\time-lapse\\"

t = TimeLapseCombo(path)
t.read(1, 5)
t.combo(target=4, m_factor=0.3, save=False)



