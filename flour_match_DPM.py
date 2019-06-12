from btimage import BT_image, MatchFlourPhase
import numpy as np

import cv2

path = "E:\\DPM\\20190610\\phase_npy\\img_2019_06_10_14_46_19_phase.npy"
path_fluor = "E:\\DPM\\20190610\\00000000_00000000942E4B60_1958742671.png"

mf = MatchFlourPhase(path, path_fluor).match(340, 555)

np.save("E:\\DPM\\20190610\\phase_npy\\", mf)
