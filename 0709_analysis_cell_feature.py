import os
import numpy as np
import cv2
from btimage import check_file_exist
from btimage import TimeLapseCombo
from btimage import BT_image, CellLabelOneImage
import glob
from matplotlib import pyplot as plt

# input
# 1. after watershed label
# 2. phase maps


class PrevNowMatching(object):
    """ creare the list of linkage"""
    def __init__(self, prev, now):
        # self.prev_phase_map = None
        self.prev_label_map = prev
        # self.now_phase_map = None
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

    def show(self, image, text):
        plt.figure()
        plt.imshow(image, cmap='jet', vmax=80, vmin=0)
        plt.title(text)
        plt.show()

    def check_prev_label(self):
        for label in range(81):
            cur_label_num = len(self.prev_label_map[self.prev_label_map == label])
            if 10000 <= cur_label_num <= 1000000:
                # remove too small area and BG area
                self.prev_list.append(label)
                # print("label:", label, "has:", cur_label_num, "pixel")

    def check_now_label(self):
        for label in range(81):
            cur_label_num = len(self.now_label_map[self.now_label_map == label])
            if 10000 <= cur_label_num <= 1000000:
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
            print("prev label:", label, "match --> now label: ", corresponded_label)

            if corresponded_label != 1:
                # registering corresponding label into new_now_map
                self.output[self.now_label_map == corresponded_label] = label
                # pop corresponded_label
                self.prev_list.remove(label)
                self.now_list.remove(corresponded_label)

            elif corresponded_label == 1:
                print("prev label:", label, "match BG label !!!!!")
                print()
                self.lost_map[self.prev_label_map == label] = 100

        if self.now_list:
            for i in self.now_list:
                self.lost_map[self.now_label_map == i] = 200

        self.show(self.output, "new_now_map")
        plt.figure()
        plt.imshow(self.lost_map, cmap='jet')
        plt.figtext(0.83, 0.5, "g: disappear\nr: appear", transform=plt.gcf().transFigure)
        plt.title("lost_map")
        plt.show()

    def second_round_matching(self):
        print("disappear: ", self.prev_list)
        print("appear: ", self.now_list)
        if self.prev_list and self.now_list:
            for disappear in self.prev_list:
                for appear in self.now_list:
                    black = np.zeros((3072, 3072))
                    if len(black[(self.prev_label_map == disappear) & (self.now_label_map == appear)]) != 0:
                        # find overlap
                        print("Round 2 : prev label:", disappear, "match --> now label: ", appear)
                        self.output[self.now_label_map == appear] = disappear
        print("finish second round!")

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


after_water_path = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\5_afterwater.npy"
prev_label = np.load(after_water_path)
after_water_path = "E:\\DPM\\20190701\\1\\SP\\time-lapse\\phase_npy_finish\\6_afterwater.npy"
now_label = np.load(after_water_path)

PrevNowMatching(prev_label, now_label).run()


#######################################################
class Cell(object):
    def __init__(self):
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





