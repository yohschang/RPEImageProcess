import os
import numpy as np
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage, WorkFlow
import glob
from matplotlib import pyplot as plt


root_path = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\"

after = CellLabelOneImage(root_path, target=5).run(adjust=True, plot_mode=False, load_old=True, save_water=True)
plt.close()
plt.figure()
plt.title(str(5) + "label img")
plt.imshow(after, cmap='jet')
plt.colorbar()
plt.show()



