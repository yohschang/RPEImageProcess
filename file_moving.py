import os
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage
import glob
from matplotlib import pyplot as plt

path = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\"
#
# for i in range(1, 12):
#     dir_num = str(i)
#     new = str(i+29)
#     check_file_exist(path+dir_num, dir_num)
#     os.rename(path+dir_num, path+new)

# t = TimeLapseCombo(path)
# t.read(1, 41)
# t.combo(target=24, m_factor=0.3, save=True)

path_modify = 'E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\'
for i in range(6, 7):
    path = path_modify + str(i+1) + "_phase.npy"
    print(path)
    im = BT_image(path)
    im.opennpy()
    plt.figure()
    plt.title(str(i+1)+" phase image")
    plt.imshow(im.img, cmap="jet", vmax=3.5, vmin=0)
    plt.colorbar()
    # plt.savefig(path_modify + "pic\\" + str(i+1) + ".tif", format="png")
    plt.show()

    # labeling
    marker_file = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\pic\\7_marker.npy"
    after = CellLabelOneImage(path, path_modify + "pic\\", i+1).run(adjust=True, plot_mode=False, marker_file=marker_file)

    plt.close()
    plt.figure()
    plt.title(str(i+1)+"label img")
    plt.imshow(after, cmap='jet')
    plt.colorbar()
    plt.show()

