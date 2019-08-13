import os
import cv2
import numpy as np
from btimage import check_file_exist
from btimage import BT_image, CellLabelOneImage, PrevNowCombo, TimeLapseCombo, Fov, WorkFlow, AnalysisCellFeature
import glob
from matplotlib import pyplot as plt


root_path = "E:\\DPM\\20190708_time_lapse_succ\\Bead\\1\\SP\\time-lapse\\"

####################################################################################


# def test():
#     t = TimeLapseCombo(root_path=root_path)
#     t.read(1, 36)
#     t.combo(target=26, save=True, strategy="cheat")
#
# test()
####################################################################################
# f = Fov(root_path, 1, 36)
# f.run()


###################################################################################
# label and match

# current_target = 30

# after = CellLabelOneImage(root_path, target=current_target).run(adjust=True, plot_mode=False, load="old", save_water=True)
# output = PrevNowCombo(root_path).combo(now_target=current_target, save=True)

# plt.close()
# plt.figure()
# plt.title(str(current_target) + "label img")
# plt.imshow(after, cmap='jet')
# plt.colorbar()
# plt.show()
# ####################################################################################
# analysis

# ana = AnalysisCellFeature(root_path)


####################################################################################
acf = AnalysisCellFeature(root_path)
# acf.image_by_image(dbsave=True)
acf.check_last_id()




