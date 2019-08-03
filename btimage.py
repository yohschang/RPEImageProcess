"""
Purpose: BT thesis
author: BT
Date: 20190703

class:
    WorkFlow
    BT_image
    PhaseRetrieval
    PhaseCombo
    TimeLapseCombo
    MatchFlourPhase
    CellLabelOneImage
    App
    Sketcher
    PrevNowMatching

function:
    check_file_exist
    check_img_size

"""

import cv2
from os import makedirs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.restoration import unwrap_phase
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast
from skimage.exposure import adjust_sigmoid
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

_jet_data ={'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                     (1, 0.5, 0.5)),
            'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                     (0.91,0,0), (1, 0, 0)),
            'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                     (1, 0, 0))}

cmap_data = {'jet': _jet_data}


def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)


jet_color = make_cmap('jet')


def round_all_the_entries_ndarray(matrix, decimal):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = round(matrix[i, j], decimal)
    return matrix


def check_file_exist(this_path, text):
    my_file = Path(this_path)
    if not my_file.exists():
        raise OSError("Cannot find " + str(text) + "!")


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


class WorkFlow(object):
    def __init__(self, root):
        self.root = root
        check_file_exist(self.root, "root directory")
        self.phase_npy_path = root + "phase_npy\\"
        self.pic_path = root + "pic\\"
        self.marker_path = root + "marker\\"
        self.afterwater_path = root + "afterwater\\"
        self.analysis_path = root + "analysis\\"
        self.fluor_path = root + "fluor\\"

        # create dir
        self.create_dir(self.phase_npy_path)
        self.create_dir(self.pic_path)
        self.create_dir(self.marker_path)
        self.create_dir(self.afterwater_path)
        self.create_dir(self.analysis_path)
        self.create_dir(self.fluor_path)

        # prepare
        self.kaggle_img_path = "C:\\Users\\BT\\Desktop\\kaggle\\RPE_crop_image\\"
        self.kaggle_mask_path = "C:\\Users\\BT\\Desktop\\kaggle\\RPE_crop_mask\\"

    def create_dir(self, path):
        my_file = Path(path)
        if not my_file.exists():
            makedirs(path)


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

    def phase_retrieval(self, m_factor, strategy="try", sp=(0, 0)):
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
        elif strategy == "cheat":
            x, y = sp[0], sp[1]

        # crop real or virtual image
        self.sp.crop_first_order(x, y, 768)
        self.bg.crop_first_order(0, 0, 768)
        print(x, y)

        # iFFT
        self.sp.twodifft(self.sp.crop_raw_f_domain)
        self.bg.twodifft(self.bg.crop_raw_f_domain)
        self.wrapped_sp = self.sp.iff
        self.wrapped_bg = self.bg.iff
        # self.plot_fdomain()

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
            for j in np.arange(-2, 3, 1):
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

    def plot_final(self, center=False, num=0):
        """beware of Memory"""
        plt.figure(dpi=200, figsize=(10, 10))
        if center:
            plt.imshow(self.final[1500:1700, 1500:1700], vmin=-1, vmax=3.5)
        else:
            plt.imshow(self.final, cmap='jet', vmin=-0.5, vmax=3)
        plt.colorbar()
        plt.title("sp - bg"+str(num))
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


class TimeLapseCombo(WorkFlow):
    def __init__(self, root_path):
        super().__init__(root_path)
        self.pathsp_list = []
        self.pathbg_list = []
        self.cur_num = 1

    def read(self, start, end):
        for i in range(start, end+1):

            # read one BG at the root dir
            found_bg = glob.glob(self.root + "*.bmp")
            if len(found_bg) < 1:
                raise FileExistsError("BG lost")
            print("BG:", found_bg[0])
            self.pathbg_list.append(found_bg[0])

            # read many SP
            path_cur = self.root + str(i) + "\\"
            check_file_exist(path_cur, "i")
            found_file = glob.glob(path_cur + "*.bmp")
            if len(found_file) != 1:
                raise FileExistsError("SP lost or too many SP")
            print("SP:", found_file[0])
            self.pathsp_list.append(found_file[0])

    def combo(self, target=-1, save=False, m_factor=0, strategy="try"):
        # combo
        if target == -1:
            for i, m in zip(range(len(self.pathsp_list)), np.arange(0.3, 0, -0.3/40)):
                pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[0])
                try:
                    pr.phase_retrieval(m, sp=(0, 0), strategy=strategy)
                    print(str(i), " SD:", np.std(pr.final))
                    if np.std(pr.final) > 1.51:
                        pr.phase_retrieval(m, sp=(0, 4), strategy=strategy)
                    pr.plot_final(center=False, num=i)
                    # pr.plot_hist()
                    if save:
                        np.save(self.phase_npy_path + str(self.cur_num) + "_phase.npy", pr.final)
                        self.cur_num += 1
                except TypeError as e:
                    print(i, "th cannot be retrieved ", e)

        else:
            # only for checking
            for i in tqdm.trange(len(self.pathsp_list)):
                if i == target:
                    pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[0])
                    pr.phase_retrieval(m_factor, sp=(0, 0))
                    if np.std(pr.final) > 1:
                        pr.phase_retrieval(m_factor, sp=(0, 0), strategy=strategy)
                    pr.plot_final(center=False, num=i)
                    pr.plot_hist()
                    pr.plot_sp_bg()
                    print(np.std(pr.final))
                    # pr.plot_fdomain()
                    if save:
                        np.save(self.phase_npy_path + str(i) + "_phase.npy", pr.final)
                        print(self.phase_npy_path + str(i) + "_phase.npy")
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


class CellLabelOneImage(WorkFlow):
    """Instance recognition"""

    def __init__(self, root, target=-1):
        super().__init__(root)

        # open image
        self.img = np.load(self.phase_npy_path + str(target) + "_phase.npy")
        check_img_size(self.img)

        self.target = target
        self.img_origin = None
        self.sure_bg = None
        self.sure_fg = None
        self.pre_marker = None
        self.distance_img = None
        self.after_water = None
        self.plot_mode = False

    def run(self, adjust=False, plot_mode=False, load_old=False, save_water=False):
        self.plot_mode = plot_mode
        self.phase2uint8()
        self.smoothing()
        self.sharpening(0.15, 30)
        self.adaptive_threshold()
        self.morphology_operator()
        self.prepare_bg()
        self.distance_trans()
        self.find_local_max()
        self.watershed_algorithm()

        if adjust:
            if load_old:
                marker_file = self.marker_path + str(self.target) + "_marker.npy"
                check_file_exist(marker_file, str(self.target) + "_marker.npy")
            else:
                marker_file = None
            self.watershed_manually(marker_file)

        if save_water:
            np.save(self.afterwater_path + str(self.target) + "_afterwater.npy", self.after_water)
            print("saving  ", self.afterwater_path + str(self.target) + "_afterwater.npy")
        return self.after_water

    def plot_gray(self, image, title_str):
        plt.figure()
        plt.title(title_str)
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.show()

    def phase2uint8(self):
        plt.figure()
        plt.title(str(self.target) + "original image")
        plt.imshow(self.img, cmap='jet', vmax=3.5, vmin=-0.2)
        plt.colorbar()
        plt.show()
        self.img[self.img >= 4] = 0
        self.img[self.img <= -0.5] = -0.5
        max_value = self.img.max()
        min_value = self.img.min()
        image_rescale = (self.img - min_value) * 255 / (max_value - min_value)
        t, image_rescale = cv2.threshold(image_rescale, 255, 255, cv2.THRESH_TRUNC)
        t, image_rescale = cv2.threshold(image_rescale, 0, 0, cv2.THRESH_TOZERO)
        self.img = np.uint8(np.round(image_rescale))
        if self.plot_mode:
            self.plot_gray(self.img, "original image")
        return self.img

    def smoothing(self):
        self.img = cv2.GaussianBlur(self.img, (7, 7), sigmaX=1)
        # show

    def sharpening(self, cutoff_value, gain_value):
        if self.plot_mode:
            plt.figure()
            plt.hist(self.img.flatten(), bins=200)
            plt.show()
        # self.img = enhance_contrast(self.img, disk(5))
        self.img = adjust_sigmoid(self.img, cutoff=cutoff_value, gain=gain_value)
        self.img_origin = self.img.copy()
        # show

        x = np.arange(0, 1, 0.01)
        y = 1/(1 + np.exp((gain_value*(cutoff_value - x))))
        if self.plot_mode:
            plt.figure()
            plt.title("Sigmoid Correction (cutoff: 0.08, gain: 18)")
            plt.plot(x, y)
            plt.show()

    def adaptive_threshold(self):
        array_image = self.img.flatten()
        # plt.figure()
        n, b, patches = plt.hist(array_image, bins=200)
        # plt.title("Histogram of phase image")
        # plt.xlabel("gray value")
        # plt.ylabel("number of pixel")
        # plt.show()

        # Adaptive threshold
        b = b[:-1]
        n[b < 70] = 0
        n[b > 220] = 0
        bin_max = np.argmax(n)
        print("bin_max", bin_max)
        max_value = b[bin_max]
        print(max_value)
        threshold = 0.5 * np.sum(array_image) / len(array_image[array_image > max_value])
        print("Adaptive threshold is:", threshold)
        # thresholding
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 1)
        # ret, self.img = cv2.threshold(self.img, threshold, 255, cv2.THRESH_BINARY)
        if self.plot_mode:
            self.plot_gray(self.img, "binary image")

    def morphology_operator(self):
        kernel = np.ones((3, 3), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=4)
        kernel = np.ones((30, 30), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.plot_mode:
            self.plot_gray(self.img, "morphology image")

    def prepare_bg(self):
        self.sure_bg = self.img.copy()
        # kernel = np.ones((5, 5), np.uint8)
        # self.sure_bg = np.uint8(cv2.dilate(self.sure_bg, kernel, iterations=5))

    def distance_trans(self):
        """
                    force watershed algorithm flow to the value distance = 0 but not sure bg,
                    so distance map += 1
                    distance map where the location is sure bg -= 1
                """
        self.img = cv2.distanceTransform(self.img, 1, 5) + 1
        self.img[self.sure_bg == 0] -= 1
        # plt.figure()
        # plt.hist(self.img.flatten(), bins=100)
        # plt.show()
        # self.img = np.power(self.img/float(np.max(self.img)), 0.6) * 255
        # remove too small region
        # self.img[self.img < 50] = 0

        self.distance_img = self.img.copy()
        if self.plot_mode:
            # self.plot_gray(self.img, "dist image")
            plt.figure()
            plt.imshow(self.img, cmap='jet')
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.hist(self.img.flatten(), bins=100)
            plt.show()

    def find_local_max(self):
        marker = np.zeros((3072, 3072), np.uint8)

        # 220 is the size of RPE
        local_maxi = peak_local_max(self.img, indices=False, footprint=np.ones((220, 220)), threshold_abs=20)
        marker[local_maxi == True] = 255
        kernel = np.ones((5, 5), np.uint8)
        marker = np.uint8(cv2.dilate(marker, kernel, iterations=15))

        ret, markers1 = cv2.connectedComponents(marker)
        markers1[self.sure_bg == 0] = 1
        self.pre_marker = np.int32(markers1)
        if self.plot_mode:
            self.plot_gray(self.pre_marker, "local max image")

    def watershed_algorithm(self):
        rgb = cv2.cvtColor(np.uint8(self.distance_img), cv2.COLOR_GRAY2BGR)
        self.after_water = self.pre_marker.copy()
        cv2.watershed(rgb, self.after_water)
        if self.plot_mode:
            self.plot_gray(self.after_water, "watershed image")
        ###########################################################
        # if no manually adjust, self.after_water is final output #
        ###########################################################

    def watershed_manually(self, marker_file=None):
        """Implement App"""
        self.img_origin = cv2.cvtColor(self.img_origin, cv2.COLOR_GRAY2BGR)
        self.distance_img = cv2.cvtColor(np.uint8(self.distance_img), cv2.COLOR_GRAY2BGR)
        print(App.__doc__)

        if marker_file:
            try:
                self.pre_marker = np.load(marker_file)
            except:
                raise FileExistsError("cannot open marker file")

        r = App(self.distance_img, self.pre_marker, self.img_origin, save_path=self.marker_path, cur_img_num=self.target)
        r.run()
        self.after_water = r.m


class Sketcher(object):
    def __init__(self, windowname, dests, colors_func, eraser):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.eraser = eraser
        self.show()
        self.mouse_track = None
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.dests[0])
        cv2.resizeWindow(self.windowname, 640, 640)

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)

        # the track of mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt

        # if event == cv2.EVENT_RBUTTONDOWN:
        #     print("right click")
        #     cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        #     cv2.imshow(self.windowname, self.raw)
        #     cv2.resizeWindow(self.windowname, 640, 640)

        # draw a line
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            # print(self.colors_func())
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, self.eraser)
            self.dirty = True
            self.prev_pt = pt
            self.mouse_track = pt
            self.show()
        else:
            self.prev_pt = None
            self.mouse_track = pt


class App(object):
    """
            Watershed segmentation
            =========

            This program demonstrates the watershed segmentation algorithm
            in OpenCV: watershed().

            Usage
            -----
            watershed.py [image filename]

            Keys
            ----
              1-7   - switch marker color
              SPACE - update segmentation
              r     - reset
              a     - toggle autoupdate
              ESC   - exit

        """
    def __init__(self, fn, existed_marker, show_img, save_path, cur_img_num):
        # input parameter
        self.img = fn
        self.markers = existed_marker
        self.show_img = show_img
        self.save_path = save_path
        self.cur_img_num = cur_img_num

        # create parameter
        self.markers_vis = self.show_img.copy()
        self.cur_marker = 1
        self.colors = jet_color
        self.overlay = None
        self.m = None

        # marker pen diameter
        diameter = 20
        self.auto_update = False

        # canvas
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors, diameter)

    def get_colors(self):
        # print(list(map(int, self.colors[self.cur_marker])))
        return list(map(int, self.colors[self.cur_marker*3])), int(self.cur_marker)

    def watershed(self):

        # because watershed will change m
        self.m = self.markers.copy()

        # watershed algorithm
        cv2.watershed(self.img, self.m)

        # transfer marker to color but remove negative marker
        self.overlay = self.colors[np.maximum(self.m, 0)*2]

        vis = cv2.addWeighted(self.img, 0.5, self.overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        cv2.namedWindow('watershed', cv2.WINDOW_NORMAL)
        cv2.imshow('watershed', vis)
        cv2.resizeWindow("watershed", 640, 640)
        # plt.close()
        # plt.figure()
        # plt.imshow(self.markers, cmap='jet')
        # plt.show()

    def run(self):

        # init marker
        decision = 1
        while True:
            ch = 0xFF & cv2.waitKey(50)

            # Esc
            if ch == 27:
                break

            if ch == ord("0"):
                self.cur_marker = 0
                print('marker: ', self.cur_marker)

            if ch in [ord('l'), ord('L')]:
                self.cur_marker = self.cur_marker + 1
                print('marker: ', self.cur_marker)

            if ch in [ord('t'), ord('T')]:
                self.markers[self.markers == self.cur_marker] = 0
                print('reset: ', self.cur_marker, " in the image")

            # update watershed
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False

            # # automatic update
            # if ch in [ord('a'), ord('A')]:
            #     self.auto_update = not self.auto_update
            #     print('auto_update if', ['off', 'on'][self.auto_update])

            if ch in [ord('a'), ord('A')]:
                for label in range(85):
                    if len(self.markers[self.markers == label]) == 0:
                        print(label, " is available")

            if ch in [ord('q'), ord('Q')]:
                for label in range(2, 85):
                    black = np.zeros((3072, 3072), dtype=np.uint8)
                    black[self.m == label] = 255
                    ret, b = cv2.connectedComponents(black)
                    if ret > 2:
                        print("label ", label, "is too many region!!")
                print("Q: finish checking!")

            # reset
            if ch in [ord('r'), ord('R')]:
                # self.markers[:] = 0
                self.markers_vis[:] = self.show_img
                self.sketch.show()

            # save
            if ch in [ord('s'), ord('S')]:
                self.watershed()
                np.save(self.save_path + str(self.cur_img_num) + "_marker.npy", self.markers)
                print("save marker to ", self.save_path + str(self.cur_img_num) + "_marker.npy")

            # catch the marker
            if ch in [ord('c'), ord('C')]:
                print("track the mouse:", self.sketch.mouse_track)
                self.cur_marker = self.markers[self.sketch.mouse_track[1], self.sketch.mouse_track[0]]
                print("marker:", self.cur_marker)
        cv2.destroyAllWindows()


class PrevNowMatching(object):
    """ creare the list of linkage"""
    def __init__(self, prev, now):
        self.prev_label_map = prev
        self.now_label_map = now
        self.prev_list = []
        self.now_list = []
        self.output = None

        # lost map
        self.lost_map = np.zeros((3072, 3072))

        # plot input
        self.show(self.prev_label_map, "prev_label_map")
        self.show(self.now_label_map, "now_label_map")

    def run(self):
        self.check_prev_label()
        self.check_now_label()
        self.first_round_matching()
        self.second_round_matching()
        return self.output

    def show(self, image, text):
        plt.figure()
        plt.imshow(image, cmap='jet', vmax=90, vmin=0)
        plt.title(text)
        plt.show()

    def check_prev_label(self):
        for label in range(90):
            cur_label_num = len(self.prev_label_map[self.prev_label_map == label])
            if 4000 <= cur_label_num <= 1000000:
                # remove too small area and BG area
                self.prev_list.append(label)
                # print("label:", label, "has:", cur_label_num, "pixel")

    def check_now_label(self):
        for label in range(90):
            cur_label_num = len(self.now_label_map[self.now_label_map == label])
            if 4000 <= cur_label_num <= 1000000:
                # remove too small area and BG area
                self.now_list.append(label)
                # print("label:", label, "has:", cur_label_num, "pixel")

    def first_round_matching(self):
        self.output = self.now_label_map.copy()
        iterative_label = self.prev_list.copy()
        # find prev label
        for i in range(len(iterative_label)):
            # choose iterative label in box
            label = iterative_label[i]


            # find corresponding label in now
            black = np.zeros((3072, 3072))
            black[self.prev_label_map == label] = 255
            x, y = self.centroid(black)
            corresponded_label = self.now_label_map[y, x]

            # black[self.now_label_map == corresponded_label] = 100
            # plt.figure()
            # plt.imshow(black, cmap='gray', vmax=255, vmin=0)
            # plt.scatter(x, y, s=20, c="g")
            # plt.show()

            # print("prev label:", label, "match --> now label: ", corresponded_label)

            if corresponded_label != 1:
                # registering corresponding label into new_now_map
                self.output[self.now_label_map == corresponded_label] = label
                # pop corresponded_label
                try:
                    self.prev_list.remove(label)
                except:
                    pass
                try:
                    self.now_list.remove(corresponded_label)
                except:
                    pass

            elif corresponded_label == 1:
                print("prev label:", label, "match BG label !!!!!")
                print()

    def second_round_matching(self):
        if self.prev_list and self.now_list:
            for disappear in self.prev_list:
                for appear in self.now_list:
                    black = np.zeros((3072, 3072))
                    if len(black[(self.prev_label_map == disappear) & (self.now_label_map == appear)]) != 0:
                        # find overlap
                        print("Round 2 : prev label:", disappear, "match --> now label: ", appear)
                        self.output[self.now_label_map == appear] = disappear
        # appear
        print("appear: ", self.now_list)
        if self.now_list:
            for i in self.now_list:
                self.lost_map[self.now_label_map == i] = 200
                # if appear, then cancel their label
                self.output[self.now_label_map == i] = 1

        # disappear
        print("disappear: ", self.prev_list)
        if self.prev_list:
            for i in self.prev_list:
                self.lost_map[self.prev_label_map == i] = 100

        print("finish second round!")
        self.show(self.output, "new_now_map")
        plt.figure()
        plt.imshow(self.lost_map, cmap='jet')
        plt.figtext(0.83, 0.5, "g: disappear\nr: appear", transform=plt.gcf().transFigure)
        plt.title("lost_map")
        plt.show()

    def centroid(self, binary_image):
        """ find centroid"""
        moments = cv2.moments(binary_image)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
        else:
            centroid_y, centroid_x = 0, 0
            print("Cannot find centroid!")
        return centroid_x, centroid_y


class PrevNowCombo(WorkFlow):
    def __init__(self, root):
        super().__init__(root)
        self.prev = None
        self.now = None

    def combo(self, now_target=-1, save=False):
        self.prev = np.load(self.afterwater_path + str(now_target-1) + "_afterwater.npy")
        self.now = np.load(self.afterwater_path + str(now_target) + "_afterwater.npy")
        output = PrevNowMatching(self.prev, self.now).run()
        if save:
            np.save(self.afterwater_path + str(now_target) + "_afterwater.npy", output)


###########################################################################################



class AnalysisCellFeature(WorkFlow):
    def __init__(self, root):
        super().__init__(root)

        # find 40 frames in afterwater_path
        # self.afterwater_path

        # find phase image in phase npy
        # self.phase_npy_path

        # check the length of phase and label is 40

        self.phase_img_list = None
        self.label_img_list = None

    def one_by_one(self):
        pass
        # for each label of the first frame

        # calculate phase mean

        # calculate area

        # calculate circularity

        # calculate distribution

        # create cell object

    def connect_to_db(self):
        # use db to store the feature, crop phase image
        pass

    def plot_the_graph(self):
        # take the data and plot it in different plot
        pass


class Cell(object):
    def __init__(self):
        """RPE cell"""
        # basic attribute
        self.id = -1
        self.disappear = False
        self.disappear_frame = -1

        # label
        self.full_attendance = False

        # list of feature
        self.cell_area = []
        self.circularity = []
        self.cell_phase_mean = []


class Fov(WorkFlow):
    def __init__(self, root, start, end):
        super().__init__(root)
        self.file_list = [self.phase_npy_path + str(p) + "_phase.npy" for p in range(start, end+1)]
        self.pic_save = [self.pic_path + str(p) + ".png" for p in range(start, end+1)]
        self.cur_num = [str(p) for p in range(start, end+1)]
        self.check_file()

    def check_file(self):
        for p in self.file_list:
            check_file_exist(p, p)

    def run(self):
        center = (3072//2, 3072//2)
        radius = 3072//2
        for i, im_p, pic_p in zip(self.cur_num, self.file_list, self.pic_save):
            print(im_p)
            img = np.load(im_p)
            black = np.zeros((3072, 3072), dtype=np.uint8)
            black = cv2.circle(black, center, radius, 1, thickness=-1)
            img[black == 0] = 0
            np.save(im_p, img)
            plt.figure(i)
            plt.title(i + "phase image")
            plt.imshow(img, cmap='jet', vmax=3, vmin=-0.2)
            plt.axis("off")
            plt.colorbar()
            plt.savefig(pic_p)
            plt.close(i)

