import os
import numpy as np
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage
import glob
from matplotlib import pyplot as plt


path = "E:\\DPM\\20190708\\Bead\\1\\SP\\time-lapse\\"

t = TimeLapseCombo(path)
t.read(1, 40)
t.combo(target=-1, m_factor=0.3, save=False, strategy="cheat")





