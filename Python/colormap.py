# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import math

def disparity_to_color(disp_val):
    h_scale = 6
    h = 0.6 * (1. - 2 * disp_val)
    s = 1.
    v = 1.

    sector_data = ((1, 3, 0), (1, 0, 2), (3, 0, 1), (0, 2, 1), (0, 1, 3), (2, 1, 0))
    h *= h_scale
    if h < 0:
        while h < 0:
            h += 6
    else:
        while (h >= 0):
            h -= 6

    sector = math.floor(h)
    h -= sector
    if sector >= 6:
        sector = 0
        h = 0

    tab = (v, v * (1. - s), v * (1. - s * h), v * (1. - s * (1. - h)))
    b = 255 * tab[sector_data[sector][0]]
    g = 255 * tab[sector_data[sector][1]]
    r = 255 * tab[sector_data[sector][2]]

    return b, g, r
