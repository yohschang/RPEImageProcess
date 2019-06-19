#!/usr/bin/env python

'''
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

'''

import numpy as np
import cv2
from common import Sketcher
from btimage import check_img_size
from matplotlib import pyplot as plt


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


# make color
# palette data from matplotlib/_cm.py
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }
jet_color = make_cmap('jet')


class App(object):
    def __init__(self, fn, existed_marker):
        # self.img = cv2.imread(fn)
        self.img = fn
        check_img_size(self.img)
        h, w = self.img.shape[:2]

        self.markers = existed_marker
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = jet_color
        self.overlay = None
        self.m = None

        # marker pen diameter
        self.eraser = 20
        self.auto_update = False
        self.sketch = Sketcher('img', [self.markers_vis, self.markers], self.get_colors, self.eraser)

    def get_colors(self):
        # print(list(map(int, self.colors[self.cur_marker])))
        return list(map(int, self.colors[self.cur_marker*2])), int(self.cur_marker)

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
        plt.figure()
        plt.imshow(self.markers, cmap='jet')
        plt.show()

    def run(self):

        # init marker
        decision = 1
        while True:
            ch = 0xFF & cv2.waitKey(50)

            # Esc
            if ch == 27:
                break

            # # current marker
            # if ch in [ord('o'), ord('O')]:
            #     decision = input("Please decide a marker:")
            #     self.cur_marker = decision
            #     print('marker: ', self.cur_marker)
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker)

            if ch == ord("0"):
                self.cur_marker = 0
                print('marker: ', self.cur_marker)

            if ch in [ord('q'), ord('Q')]:
                self.eraser += 1
                print("dot: ", self.eraser)
            if ch in [ord('w'), ord('W')]:
                self.eraser -= 1
                print("dot: ", self.eraser)

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

            # automatic update
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])

            # reset
            if ch in [ord('r'), ord('R')]:
                # self.markers[:] = 0
                self.markers_vis[:] = self.img
                self.sketch.show()

            # save
            if ch in [ord('s'), ord('S')]:
                self.watershed()
                np.save("E:\\DPM\\20190614_RPE2\\marker.npy", self.markers)

            # catch the marker
            if ch in [ord('c'), ord('C')]:
                print("track the mouse:", self.sketch.mouse_track)
                self.cur_marker = self.markers[self.sketch.mouse_track[1], self.sketch.mouse_track[0]]
                print("marker:", self.cur_marker)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'C:\\Users\\BT\\PycharmProjects\\untitled1\\img_2019_06_14_18_12_10_phase.png'
    print(__doc__)
    pre_markers = np.zeros((3072, 3072), np.int32)
    App(fn, pre_markers).run()




