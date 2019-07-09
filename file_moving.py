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

# t = TimeLapseCombo(path)
# t.read(1, 41)
# t.combo(target=24, m_factor=0.3, save=True)

# path_modify = 'E:\\DPM\\20190708\\Bead\\1\\SP\\time-lapse\\phase_npy\\'
path_modify = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\"

for i in range(5, 6):
    path = path_modify + str(i+1) + "_phase.npy"
    print(path)
    im = BT_image(path)
    im.opennpy()
    plt.figure()
    plt.title(str(i+1)+" phase image")
    plt.imshow(im.img, cmap="jet", vmax=3.5, vmin=-0.2)
    plt.colorbar()
    plt.savefig(path_modify + "pic\\" + str(i+1) + ".tif", format="png")
    plt.show()

    # labeling
    marker_file = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\pic\\" + str(i+1) + "_marker.npy"
    # marker_file = None
    after = CellLabelOneImage(path, path_modify + "pic\\", i+1).run(adjust=True, plot_mode=False, marker_file=marker_file)
    np.save("E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\" + str(i+1) + "_afterwater.npy", after)
    plt.close()
    plt.figure()
    plt.title(str(i+1)+"label img")
    plt.imshow(after, cmap='jet')
    plt.colorbar()
    plt.show()

