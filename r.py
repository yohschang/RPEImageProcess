import numpy as np
from btimage import BT_image
from matplotlib import pyplot as plt

path = "E:\\DPM\\20190614_RPE2\\phase_npy\\img_2019_06_14_18_27_58_phase.npy"

im = BT_image(path)
im.opennpy()

plt.figure(dpi=200, figsize=(10, 10))
plt.imshow(im.img-0.1, cmap='jet', vmax=3.5, vmin=-1)
plt.show()


