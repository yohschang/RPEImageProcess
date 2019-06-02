import cv2
from os import path, makedirs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
from random import randint
import glob
import tqdm


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

    def open_image(self):
        img = cv2.imread(self.path)
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
        w = 300
        h = 300
        x, y = self.beadcenter2croprange(x_center, y_center)
        self.img = self.img[y:y+h, x:x+w]

    def crop_img2circle_after_crop_it_to_tiny_square(self, centerx, centery):
        """choose the area of bead and append to a list"""
        radius = 48  # pixel
        self.board = np.zeros((self.img.shape[0], self.img.shape[0]))
        self.flat_board = []

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[0]):
                if (i - centerx)**2 + (j - centery)**2 <= radius**2:
                    self.board[i, j] = self.img[i, j]
                    self.flat_board.append(self.img[i, j])

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
        print(thres)
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
        self.iff = np.arctan(np.divide(img_back[:, :, 1], img_back[:, :, 0]))
        self.test = img_back
        img_back = cv2.magnitude(img_back[:, :, 0],img_back[:, :, 1])
        return img_back

    def crop_first_order(self):
        x_start = 1536 - 384
        y_start = 0
        width = 768
        height = 768

        # find the approximate area to crop
        tem_crop = 20 * np.log(self.f_domain[y_start:y_start + 768, x_start:x_start + 768])
        max_y, max_x = np.unravel_index(np.argmax(tem_crop), tem_crop.shape)
        x_final = x_start + max_x
        y_final = y_start + max_y
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
        self.final = None

    def phase_retrieval(self):
        # open img
        self.sp.open_image()
        self.bg.open_image()

        # FFT
        self.sp.twodfft()
        self.bg.twodfft()

        # crop real or virtual image
        self.sp.crop_first_order()
        self.bg.crop_first_order()

        # iFFT
        self.sp.twodifft(self.sp.crop_raw_f_domain)
        self.bg.twodifft(self.bg.crop_raw_f_domain)
        self.wrapped_sp = 2 * self.sp.iff
        self.wrapped_bg = 2 * self.bg.iff

        # unwapping
        self.unwarpped_sp = unwrap_phase(self.wrapped_sp)/2
        self.unwarpped_bg = unwrap_phase(self.wrapped_bg)/2

        # subtract
        self.final = self.unwarpped_sp - self.unwarpped_bg

        # resize
        self.final = self.resize_image(self.final, 3072)

    def resize_image(self, image, size):
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

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
            plt.imshow(self.final[1500:1700, 1500:1700], cmap='gray', vmin=-1, vmax=5)
        else:
            plt.imshow(self.final, cmap='gray', vmin=-1, vmax=5)
        plt.colorbar()
        plt.title("sp - bg")
        plt.show()

    def plot_hist(self):
        plt.figure()
        plt.hist(self.final.flatten(), bins=1000)
        plt.show()

    def write_final(self, dir_npy):
        np.save(dir_npy + self.name + "_phase.npy", self.final)

    def write_to_png(self, dir_png):
        rescale_img = self.final * 255/(5-(-1))
        cv2.imwrite(dir_png + self.name + "_phase.png", rescale_img)


class ShiftPi(object):
    def __init__(self, path):
        self.path = path
        self.list_ = None
        self.mean_list = []

    def check_mean(self):

        # generate point
        point = []
        img_boundary_small = 3072 / 4
        img_boundary_big = 3 * 3072 / 4
        for k in range(1000):
            point.append([randint(img_boundary_small, img_boundary_big), randint(img_boundary_small, img_boundary_big)])

        # read npy
        self.path = self.path + "*.npy"
        self.list_ = glob.glob(self.path)

        for i in tqdm.trange(len(self.list_)):
            img = np.load(self.list_[i])

            buffer = []
            for p, q in point:
                buffer.append(img[p, q])

            self.mean_list.append(buffer)

        self.mean_list = np.array(self.mean_list)
        self.mean_list = np.mean(self.mean_list, axis=1)
        plt.figure()
        plt.hist(self.mean_list, bins=3)
        plt.title("before shift")
        plt.xlabel("phase")
        plt.ylabel("count")
        plt.show()

    def shift(self, t1, t2):
        for i, img in enumerate(self.mean_list):
            if img < t1:
                img_small = np.load(self.list_[i])
                np.save(self.list_[i], img_small + np.pi)
                print(i, 'th + pi')
            elif img > t2:
                img_small = np.load(self.list_[i])
                np.save(self.list_[i], img_small - np.pi)
                print(i, 'th - pi')


class PhaseCombo(object):
    """Using PhaseRetrieval, ShiftPi"""

    def __init__(self, root_path):
        check_file_exist(root_path + "SP\\", "SP dir")
        check_file_exist(root_path + "BG\\", "BG dir")
        self.root = root_path
        self.pathsp_list = glob.glob(root_path + "SP\\*.bmp")
        self.pathbg_list = glob.glob(root_path + "BG\\*.bmp")
        self.check_input()

    def check_input(self):
        if not self.pathsp_list:
            raise FileNotFoundError("no SP in this directory")
        if not self.pathbg_list:
            raise FileNotFoundError("no BG in this directory")
        if len(self.pathsp_list) != len(self.pathbg_list):
            raise AssertionError("SP length and BG length do not match!")

    def combo(self, target=-1, shift1=-1, shift2=1):
        # output
        output_dir = self.root + "phase_npy//"
        if not path.exists(output_dir):
            makedirs(output_dir)

        # combo
        if target == -1:
            for i in tqdm.trange(len(self.pathsp_list)):
                pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[i])
                pr.phase_retrieval()
                pr.write_final(output_dir)

            # shift pi
            print("Shift pi...")
            spp = ShiftPi(output_dir)
            spp.check_mean()
            spp.shift(shift1, shift2)
            print("Finish!")

        else:
            # only for checking
            for i in tqdm.trange(len(self.pathsp_list)):
                if i == target:
                    pr = PhaseRetrieval(self.pathsp_list[i], self.pathbg_list[i])
                    pr.phase_retrieval()
                    pr.plot_final(center=False)
                    pr.plot_sp_bg()
                    pr.plot_fdomain()



    def npy2png(self):
        output_dir = self.root + "phase_npy//"
        dir_list = glob.glob(output_dir + "*.npy")
        for i in tqdm.trange(len(dir_list)):
            im_test = BT_image(dir_list[i])
            im_test.opennpy()
            im_test.write_image(output_dir, im_test.img * 255/(5-(-1)))
