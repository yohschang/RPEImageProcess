import imageio
from btimage import BT_image
import glob
from matplotlib import pyplot as plt
import numpy as np


def image_average_shift(path, save=False):
    """when image fluctuate, shift to mean value."""
    list_ = glob.glob(path)
    list_.pop(0)

    i = 0
    image_mean = []
    for filename in list_:
        im = BT_image(filename)
        im.opennpy()
        image_mean.append(np.mean(im.img.flatten()))
        i += 1
    all_image_mean = np.mean(image_mean)

    i = 0
    for filename in list_:
        im = BT_image(filename)
        im.opennpy()
        plt.figure(dpi=200, figsize=(10, 10))
        plt.imshow(im.img - (image_mean[i] - all_image_mean), cmap='jet', vmax=2.5, vmin=-0.4)
        if save:
            plt.savefig("E:\\DPM\\20190614_RPE2\\{}.png".format(i))
        plt.show()
        print(i)
        i += 1


path = "E:\\DPM\\20190614_RPE2\\phase_npy\\*.npy"
image_average_shift(path)


# # png to gif
# path = "E:\\DPM\\20190614_RPE2\\*.png"
# list_ = glob.glob(path)
# images = []
# for filename in list_:
#     im = BT_image(filename)
#     im.open_image()
#     images.append(im.img)
# imageio.mimsave('E:\\DPM\\20190614_RPE2\\movie.gif', images)