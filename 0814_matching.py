from btimage import MatchFlourPhase
from matplotlib import pyplot as plt
import cv2

flour_path = r"E:\DPM\20190814_pointgray_test\power11_exposure1s_bead.png"
phase_path = r"E:\DPM\20190814_pointgray_test\viework_bead.png"

mf = MatchFlourPhase(phase_path, flour_path)
output = mf.match(0, 100)

