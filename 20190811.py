from btimage import PhaseRetrieval, TimeLapseCombo, BT_image
from matplotlib import pyplot as plt

# sp = "‪E:\\DPM\\20190811\\img_2019_08_11_13_51_36.bmp"
# bg = "E:\\DPM\\20190811\\img_2019_08_11_13_52_18.bmp"
root_path = "E:\\DPM\\20190811\\"
t = TimeLapseCombo(root_path=root_path)
t.read(1, 1)
t.combo(target=0, save=True, m_factor=0, strategy="cheat")


pa = "E:\\DPM\\20190811\\phase_npy\\0_phase.npy"
im = BT_image(pa)
im.opennpy()
a = im.img

