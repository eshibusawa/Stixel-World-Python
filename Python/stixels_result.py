# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import numpy as np
import cv2

from colormap import disparity_to_color

class Section:
    def __init__(self, vB, vT, typeInt, disparity):
        self.vB = int(vB)
        self.vT = int(vT)
        self.disparity = float(disparity)
        t = int(typeInt)
        if t == 0:
            self.type = 'Ground'
        elif t == 1:
            self.type = 'Object'
        elif t == 2:
            self.type = 'Sky'
        else:
            self.type = None

class StixelsResult:
    def __init__(self, param, camera_param, d_type, d_disparity):
        self.max_disparity = camera_param.max_dis
        self.xs = list()
        self.stixels = list()
        self.fB = camera_param.focal * camera_param.baseline
        for ts, ds, k in zip(d_type.get(), d_disparity.get(), range(d_disparity.shape[0])):
            x = k * param.column_step + param.width_margin
            self.xs.append((x, x + param.column_step))
            section_row = list()
            for t, d in zip(ts, ds):
                if t[2] == -1:
                    break
                s = Section(int(t[0]), int(t[1]), int(t[2]), float(d))
                section_row.append(s)
            self.stixels.append(section_row)

    def draw(self, image):
        image_drawn = np.copy(image)
        h = image.shape[0]
        for xs, st in zip(self.xs, self.stixels):
            for s in st:
                if s.type == 'Object':
                    c = disparity_to_color(s.disparity / self.max_disparity)
                    cv2.rectangle(image_drawn, (xs[0], h - s.vT + 1), (xs[1], h - s.vB), c, thickness=-1)
                    cv2.rectangle(image_drawn, (xs[0], h - s.vT + 1), (xs[1], h - s.vB), (0, 0, 0), thickness=1)

        ret = cv2.addWeighted(image, 0.5, image_drawn, 0.5, 0)
        return ret
