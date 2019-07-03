"""
class:
    BT_image
    PhaseRetrieval
    PhaseCombo
    TimeLapseCombo
    MatchFlourPhase
    CellLabelOneImage

function:
    check_file_exist
    check_img_size

"""


import cv2
from os import path, makedirs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast
from skimage.exposure import adjust_sigmoid
import watershed
from random import randint
import glob
import tqdm

# green colorbar
cdict1 = {'red': ((0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

          'green': ((0.0, 0.0, 0.0),
                    (0.15, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),

          'blue': ((0.0, 0.0, 0.0),
                   (0.15, 0.0, 0.0),
                   (1.0, 0.5, 0.5))
          }
green = LinearSegmentedColormap('green', cdict1)


def round_all_the_entries_ndarray(matrix, decimal):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = round(matrix[i, j], decimal)
    return matrix


def check_file_exist(this_path, text):
    my_file = Path(this_path)
    if not my_file.exists():
        raise OSError("Cannot find " + text + "!")


def check_img_size(image):
    try:
        if image.shape[0] != 3072:
            raise AssertionError("Image size is not 3072!")
    except:
        raise TypeError("This file is not ndarray!")


class BT_image(object):
    """docstring"""
    def __init__(self, path):
        check_file_exist(path, "image")
        self.path = path
        self.img = None
        self.name = path.split("\\")[-1]
        self.board = None
        self.flat_board = None
        self.centroid_x = 0
        self.centroid_y = 0
        self.threshold = None
        self.f_domain = None
        self.crop_f_domain = None
        self.raw_f_domain = None
        self.crop_raw_f_domain = None
        self.iff = None
        self.test = None

    def open_image(self, color="g"):
        img = cv2.imread(self.path)
        if color == "g":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img

    def opennpy(self):
        self.img = np.load(self.path)

    def open_raw_image(self):
        fd = open(self.path, 'rb')
        rows = 3072
        cols = 3072
        f = np.fromfile(fd, dtype=np.float32, count=rows * cols)
        im_real = f.reshape((rows, cols))
        fd.close()
        self.img = im_real

    def scaling_image(self, x, y):
        self.img = cv2.resize(self.img, None, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)

    def beadcenter2croprange(self, x, y):
        return x - 84, y - 84

    def crop(self, x_center, y_center):
        w = 168
        h = 168
        x, y = self.beadcenter2croprange(x_center, y_center)
        self.img = self.img[y:y+h, x:x+w]

    def crop_img2circle_after_crop_it_to_tiny_square(self, centerx, centery):
        """choose the area of bead and append to a list"""
        radius = 48  # pixel
        self.board = np.zeros((self.img.shape[0], self.img.shape[0]))
        self.flat_board = []

        # for i in range(self.img.shape[0]):
        #     for j in range(self.img.shape[0]):
        #         if (i - centerx)**2 + (j - centery)**2 <= radius**2:
        #             self.board[i, j] = self.img[i, j]
        #             self.flat_board.append(self.img[i, j])
        self.board = self.img

    def crop_img2circle(self, centerx, centery, radius):
        self.flat_board = []
        # self.board = np.zeros((self.img.shape[0], self.img.shape[1]))
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if (i - centerx)**2 + (j - centery)**2 <= radius**2:
                    self.flat_board.append(self.img[i, j])
                    # self.board[i, j] = self.img[i, j]

    def write_image(self, path, image):
        cv2.imwrite(path + self.name.split(".")[0] + ".png", image)

    def plot_it(self, image):
        plt.figure()
        plt.title(self.name.split(".")[0])
        plt.imshow(image, plt.cm.gray, vmax=4, vmin=-0.5)
        # plt.scatter(self.centroid_y, self.centroid_x)
        plt.colorbar()
        plt.show()

    def normalize_after_crop(self):
        background = round(float(np.mean(self.img[:20, :20])), 2)
        self.img = self.img - background

    def find_centroid(self):
        # determine threshold
        thres = np.mean(self.img) + 0.7
        # threshold image
        ret, self.threshold = cv2.threshold(self.img, thres, 0, cv2.THRESH_TOZERO)
        # centroid
        moments = cv2.moments(self.threshold)
        if moments['m00'] != 0:
            self.centroid_y = int(moments['m10'] / moments['m00'])
            self.centroid_x = int(moments['m01'] / moments['m00'])
        else:
            print("Cannot find centroid!")

    def twodfft(self):
        # step 1
        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # step 2
        dft_shift = np.fft.fftshift(dft)
        # step 3
        self.raw_f_domain = dft_shift

        # visualize
        self.f_domain = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    def twodifft(self, image):
        # step 6
        f_ishift = np.fft.ifftshift(image)
        # step 7
        img_back = cv2.idft(f_ishift)  # complex ndarray [:,:,0]--> real
        # step 8
        self.iff = np.arctan2(img_back[:, :, 1], img_back[:, :, 0])
        self.test = img_back
        img_back = cv2.magnitude(img_back[:, :, 0],img_back[:, :, 1])
        return img_back

    def crop_first_order(self, sx, sy, size):

        x_start = 1536 - size//2
        y_start = 0
        width = size
        height = size

        # find the approximate area to crop
        tem_crop = 20 * np.log(self.f_domain[0:0 + 768, (1536 - 384):(1536 - 384) + 768])
        max_y, max_x = np.unravel_index(np.argmax(tem_crop), tem_crop.shape)
        x_final = x_start + max_x + sx
        y_final = y_start + max_y + sy
        crop_f_domain_test = 20 * np.log(self.f_domain[y_final-width//2:y_final+width//2, x_final-height//2:x_final+height//2])
        self.crop_f_domain = self.f_domain[y_final-width//2:y_final+width//2, x_final-height//2:x_final+height//2]

        # step 4
        self.crop_raw_f_domain = self.raw_f_domain[y_final-width//2:y_final+width//2, x_final-height//2:x_final+height//2]
        return self.crop_f_domain, tem_crop, x_final, y_final, crop_f_domain_test


class PhaseRetrieval(object):

    def __init__(self, pathsp, pathbg):
        self.name = pathsp.split("\\")[-1].replace(".bmp", "")
        self.path = pathsp.replace(self.name, "").replace(".bmp", "")
        self.sp = BT_image(pathsp)
        self.bg = BT_image(pathbg)
        self.wrapped_sp = None
        self.wrapped_bg = None
        self.unwarpped_sp = None
        self.unwarpped_bg = None
        self.final_sp = None
        self.final_bg = None
        self.final = None

    def phase_retrieval(self, m_factor, strategy="try", bg=(0, 0)):
        # open img
        self.sp.open_image()
        self.bg.open_image()
        check_img_size(self.sp.img)
        check_img_size(self.bg.img)

        # FFT
        self.sp.twodfft()
        self.bg.twodfft()

        # ----------------------------------------------------------------
        x, y = 0, 0
        if strategy == "try":
            x, y = self.try_the_position(self.sp)

        # crop real or virtual image
        self.sp.crop_first_order(x, y, 768)
        self.bg.crop_first_order(bg[0], bg[1], 768)
        print(x, y)

        # iFFT
        self.sp.twodifft(self.sp.crop_raw_f_domain)
        self.bg.twodifft(self.bg.crop_raw_f_domain)
        self.wrapped_sp = self.sp.iff
        self.wrapped_bg = self.bg.iff

        # unwapping
        self.unwarpped_sp = unwrap_phase(self.wrapped_sp)
        self.unwarpped_bg = unwrap_phase(self.wrapped_bg)

        # ----------------------------------------------------------------

        # shift
        sp_mean = np.mean(self.unwarpped_sp)
        bg_mean = np.mean(self.unwarpped_bg)
        self.unwarpped_sp += np.pi * self.shift(sp_mean)
        self.unwarpped_bg += np.pi * self.shift(bg_mean)

        # resize
        self.final_sp = self.resize_image(self.unwarpped_sp, 3072)
        self.final_bg = self.resize_image(self.unwarpped_bg, 3072)

        # subtract
        self.final = self.final_sp - self.final_bg

        # m_factor
        diff = m_factor - np.mean(self.final)
        self.final = self.final + diff

    def try_the_position(self, bt_obj):
        min_sd = 10000
        mini = 100
        minj = 100
        for i in np.arange(-2, 3, 1):
            for j in np.arange(-8, 3, 1):
                bt_obj.crop_first_order(i, j, 768)
                bt_obj.twodifft(bt_obj.crop_raw_f_domain)
                unwrap_ = unwrap_phase(bt_obj.iff)
                buffer_sd = np.std(unwrap_)
                if buffer_sd < min_sd:
                    mini, minj, min_sd = i, j, buffer_sd
        return mini, minj

    def shift(self, sp_mean):
        interval_list = [x - np.pi/2 for x in np.arange(-6 * np.pi, 7 * np.pi, np.pi)]
        i = 0
        for i, interval in enumerate(interval_list):
            if sp_mean < interval:
                break
        shift_pi = 7 - i
        # print(sp_mean, 'so', shift_pi)
        return shift_pi

    def resize_image(self, image, size):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

    def plot_fdomain(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25, 10))
        axes[0].imshow(self.sp.f_domain, cmap='gray')
        axes[0].set_title("sp f_domain ")

        axes[1].imshow(self.bg.f_domain, cmap='gray')
        axes[1].set_title("bg f_domain ")

        fig.subplots_adjust(right=1)
        plt.show()

    def plot_sp_bg(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25, 10))
        im = axes[0].imshow(self.unwarpped_sp, cmap='gray', vmin=-20, vmax=20)
        axes[0].set_title("sp")

        im = axes[1].imshow(self.unwarpped_bg, cmap='gray', vmin=-20, vmax=20)
        axes[1].set_title("bg")

        fig.subplots_adjust(right=1)
        cbar_ax = fig.add_axes([0.47, 0.1, 0.02, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    def plot_final(self, center=False):
        """beware of Memory"""
        plt.figure(dpi=200, figsize=(10, 10))
        if center:
            plt.imshow(self.final[1500:1700, 1500:1700], vmin=-1, vmax=3.5)
        else:
            plt.imshow(self.final, cmap='jet', vmin=-0.5, vmax=3)
        plt.colorbar()
        plt.title("sp - bg")
        plt.show()

    def plot_hist(self):
        plt.figure()
        plt.hist(self.final.flatten(), bins=100)
        plt.xlim(-5, 5)
        plt.show()

    def write_final(self, dir_npy):
        np.save(dir_npy + self.name + "_phase.npy", self.final)

    def write_to_png(self, dir_png):
        rescale_img = self.final * 255/(5-(-1))
        cv2.imwrite(dir_png + self.name + "_phase.png", rescale_img)


class PhaseCombo(object):
    """Using PhaseRetrieval, ShiftPi"""

    def __init__(self, root_path):
        check_file_exist(root_path + "SP\\", "SP dir")
        check_file_exist(root_path + "BG\\", "BG dir")
        self.root = root_path
        self.pathsp_list = glob.glob(root_path + "SP\\*.bmp")
        for i in self.pathsp_list:
            print(i)
        self.pathbg_list = glob.glob(root_path + "BG\\*.bmp")

        self.only_one_bg = True
        self.check_input()

    def check_input(self):
        if not self.pathsp_list:
            raise FileNotFoundError("no SP in this directory")
        if not self.pathbg_list:
            raise FileNotFoundError("no BG in this directory")
        if not self.only_one_bg:
            if len(self.pathsp_list) != len(self.pathbg_list):
                raise AssertionError("SP length and BG length do not match!")

    def combo(self, target=-1, save=False, m_factor=0):
        # output
        output_dir = self.root + "phase_npy//"
        if not path.exists(output_dir):
            makedirs(output_dir)

        # combo
        if target == -1:
            for i in tqdm.trange(len(self.pathsp_list)):
                pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[0])
                try:
                    pr.phase_retrieval(m_factor)
                    pr.plot_final(center=False)
                    # pr.plot_hist()
                    if save:
                        pr.write_final(output_dir)
                except TypeError as e:
                    print(i, "th cannot be retrieved ", e)

            # # shift pi
            # print("Shift pi...")
            # spp = ShiftPi(output_dir)
            # spp.check_mean()
            # spp.shift(shift1, shift2)
            # print("Finish!")

        else:
            # only for checking
            for i in tqdm.trange(len(self.pathsp_list)):
                if i == target:
                    pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[0])
                    pr.phase_retrieval(m_factor)
                    pr.plot_final(center=False)
                    pr.plot_hist()
                    pr.plot_sp_bg()
                    # pr.plot_fdomain()
                    if save:
                        pr.write_final(output_dir)

    def npy2png(self):
        output_dir = self.root + "phase_npy//"
        dir_list = glob.glob(output_dir + "*.npy")
        for i in tqdm.trange(len(dir_list)):
            im_test = BT_image(dir_list[i])
            im_test.opennpy()
            im_test.write_image(output_dir, im_test.img * 255/(3.5-(-1)))


class TimeLapseCombo(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.pathsp_list = []
        self.pathbg_list = []
        self.cur_num = 1

    def read(self, start, end):
        for i in range(start, end+1):
            path_cur = self.root_path + str(i) + "\\"
            check_file_exist(path_cur, "i")
            found_file = glob.glob(path_cur + "*.bmp")
            if len(found_file) != 2:
                raise FileExistsError("SP or BG lost")
            print("SP:", found_file[0])
            print("BG:", found_file[1])
            self.pathsp_list.append(found_file[0])
            self.pathbg_list.append(found_file[1])

    def combo(self, target=-1, save=False, m_factor=0):
        # create output file
        output_dir = self.root_path + "phase_npy//"
        if not path.exists(output_dir):
            makedirs(output_dir)

        # combo
        if target == -1:
            for i in tqdm.trange(len(self.pathsp_list)):
                pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[6])
                try:
                    pr.phase_retrieval(m_factor, bg=(-1, -5))
                    print(np.std(pr.final))
                    if np.std(pr.final) > 1.2:
                        pr.phase_retrieval(m_factor, bg=(-1, -4))
                    pr.plot_final(center=False)
                    # pr.plot_hist()
                    if save:
                        np.save(output_dir + str(self.cur_num) + "_phase.npy", pr.final)
                        self.cur_num += 1
                except TypeError as e:
                    print(i, "th cannot be retrieved ", e)

        else:
            # only for checking
            for i in tqdm.trange(len(self.pathsp_list)):
                if i == target:
                    pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[6])
                    pr.phase_retrieval(m_factor, bg=(-1, -5))
                    if np.std(pr.final) > 1:
                        pr.phase_retrieval(m_factor, bg=(-1, -5))
                    pr.plot_final(center=False)
                    pr.plot_hist()
                    pr.plot_sp_bg()
                    print(np.std(pr.final))
                    # pr.plot_fdomain()
                    if save:
                        np.save(output_dir + str(25) + "_phase.npy", pr.final)
                        # pr.write_final(output_dir)


class MatchFlourPhase(object):
    def __init__(self, path_phasemap, path_fluor):
        self.path_phasemap = path_phasemap
        self.path_fluor = path_fluor
        self.completed = None

    def match(self, shift_x, shift_y):
        im = BT_image(self.path_phasemap)
        im.opennpy()
        im_f = BT_image(self.path_fluor)
        im_f.open_image()
        im_f.img = cv2.flip(im_f.img, -1)

        # two image ratio
        m_obj = 27.778
        m = 46.5
        view_pixel = 5.5
        photon_pixel = 8
        ratio = (m / view_pixel) / (m_obj / photon_pixel)
        new_size = int(im_f.img.shape[0] * ratio)
        im_f.img = cv2.resize(im_f.img, (new_size, new_size), interpolation=cv2.INTER_CUBIC)

        # Photonfocus crop to 3072 * 3072
        b = im_f.img.shape[0]
        start = b // 2 - 3072 // 2
        end = b // 2 + 3072 // 2
        im_f.img = im_f.img[start - shift_y: end - shift_y, start - shift_x: end - shift_x]

        # subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(25, 10))
        im0 = axes[0].imshow(im.img, cmap='gray', vmax=14, vmin=-0.5)
        axes[0].set_title("Phase image", fontsize=30)

        im1 = axes[1].imshow(im_f.img, cmap=green)
        axes[1].set_title("Fluorescent image", fontsize=30)

        fig.subplots_adjust(right=1)
        cbar_ax0 = fig.add_axes([0.47, 0.1, 0.02, 0.8])
        cbar_ax1 = fig.add_axes([0.93, 0.1, 0.02, 0.8])
        fig.colorbar(im0, cax=cbar_ax0)
        cbar_ax0.set_title('rad')
        fig.colorbar(im1, cax=cbar_ax1)
        cbar_ax1.set_title('a.u.')
        plt.show()

        self.completed = im_f.img

    def save(self, pathandname):
        np.save(pathandname, self.completed)


class CellLabelOneImage(object):
    def __init__(self, path):
        im = BT_image(path)
        im.opennpy()
        self.img = im.img
        check_img_size(self.img)
        self.img_origin = None
        self.sure_bg = None
        self.sure_fg = None
        self.pre_marker = None
        self.distance_img = None
        self.after_water = None

    def run(self, adjust=False):
        self.phase2uint8()
        self.smoothing()
        self.sharpening()
        self.adaptive_threshold()
        self.morphology_operator()
        self.prepare_bg()
        self.distance_trans()
        self.find_local_max()
        self.watershed_algorithm()
        if adjust:
            self.watershed_manually()

        return self.after_water

    def phase2uint8(self):
        self.img[self.img >= 4] = 0
        self.img[self.img <= -0.5] = -0.5
        max_value = self.img.max()
        min_value = self.img.min()
        image_rescale = (self.img - min_value) * 255 / (max_value - min_value)
        t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
        t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
        self.img = np.uint8(np.round(image_rescale))
        # show
        return self.img

    def smoothing(self):
        self.img = cv2.GaussianBlur(self.img, (7, 7), sigmaX=1)
        # show

    def sharpening(self):
        # self.img = enhance_contrast(self.img, disk(5))
        self.img = adjust_sigmoid(self.img, cutoff=0.08, gain=18)
        self.img_origin = self.img.copy()
        # show

        # gain = 18
        # cutoff = 0.08
        # x = np.arange(0, 1, 0.01)
        # y = 1/(1 + np.exp((gain*(cutoff - x))))
        # plt.figure()
        # plt.title("Sigmoid Correction (cutoff: 0.08, gain: 18)")
        # plt.plot(x, y)
        # plt.show()

    def adaptive_threshold(self):
        array_image = self.img.flatten()
        # plt.figure()
        n, b, patches = plt.hist(array_image, bins=200)
        # plt.title("Histogram of phase image")
        # plt.xlabel("gray value")
        # plt.ylabel("number of pixel")
        # plt.show()

        # Adaptive threshold
        n = n[4:]
        bin_max = np.where(n == n.max())[0][0]
        print("bin_max", bin_max)
        max_value = b[bin_max]
        threshold = 0.7 * np.sum(array_image) / len(array_image[array_image > max_value])
        print("Adaptive threshold is:", threshold)
        # thresholding
        ret, self.img = cv2.threshold(self.img, 155, 255, cv2.THRESH_BINARY)

    def morphology_operator(self):
        kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=4)
        kernel = np.ones((30, 30), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=1)

    def prepare_bg(self):
        self.sure_bg = self.img.copy()
        kernel = np.ones((5, 5), np.uint8)
        self.sure_bg = np.uint8(cv2.dilate(self.sure_bg, kernel, iterations=8))

    def distance_trans(self):
        self.img = cv2.distanceTransform(self.img, 1, 5)
        self.img[self.img < 20] = 0
        self.distance_img = self.img.copy()
        # show

    def find_local_max(self):
        marker = np.zeros((3072, 3072), np.uint8)
        local_maxi = peak_local_max(self.img, indices=False, footprint=np.ones((220, 220)))
        marker[local_maxi == True] = 255
        kernel = np.ones((5, 5), np.uint8)
        marker = np.uint8(cv2.dilate(marker, kernel, iterations=5))

        ret, markers1 = cv2.connectedComponents(marker)
        markers1[self.sure_bg == 0] = 1
        self.pre_marker = np.int32(markers1)

    def watershed_algorithm(self):
        rgb = cv2.cvtColor(np.uint8(self.distance_img), cv2.COLOR_GRAY2BGR)
        self.after_water = self.pre_marker.copy()
        cv2.watershed(rgb, self.after_water)

    def watershed_manually(self):
        self.img_origin = cv2.cvtColor(self.img_origin, cv2.COLOR_GRAY2BGR)
        self.distance_img = cv2.cvtColor(np.uint8(self.distance_img), cv2.COLOR_GRAY2BGR)
        print(watershed.__doc__)
        r = watershed.App(self.distance_img, self.pre_marker, self.img_origin)
        r.run()






