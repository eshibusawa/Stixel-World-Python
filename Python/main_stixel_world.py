# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import os

import add_environment
import numpy as np
import cupy as cp
import cv2

from stixels_params import camera_params
from stixels_params import stixels_params
from stixels import Stixels
from stixels_result import StixelsResult
from road_estimation import RoadEstimation

def demo():
    dn = os.path.dirname(os.path.dirname(__file__))
    dn_output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'imgs')

    fpfn_disparity = os.path.join(dn, 'data', 'disparities', 'ap_000_29-02-2016_09-00-09_000002.png')
    disparity_raw = cv2.imread(fpfn_disparity, cv2.IMREAD_UNCHANGED)

    if disparity_raw.dtype == np.uint8:
        disparity_cpu = disparity_raw.astype(np.float32)
    else:
        disparity_cpu = disparity_raw / np.float32(256)
    disparity = cp.array(disparity_cpu, dtype=cp.float32)

    fpfn_image = fpfn_disparity.replace('disparities', 'left')
    left_image_cpu = cv2.imread(fpfn_image)

    cam_params = camera_params(disparity)
    st_params = stixels_params()

    stixels = Stixels(st_params, cam_params)
    stixels.setup_module()
    stixels.initialize()

    road_estimation = RoadEstimation(cam_params)
    road_estimation.setup_module()

    # (1) road estimation
    camera_tilt, camera_height, horizon_point, alpha_ground = road_estimation.compute(disparity)
    stixels.update_ground_model(disparity.shape[0], horizon_point, alpha_ground, camera_height, camera_tilt)

    # (2) stixel computation
    # (2.1) smooth disparity
    stixels.horizontal_smoothing_and_transpose(disparity)

    # (2.2) precompute LUT
    stixels.compute_object_LUT()

    # (2.3) do computation
    stixels.compute_stixels()
    sr = StixelsResult(stixels.param, stixels.camera_param, stixels.section_type, stixels.section_disparity)
    sr_image = sr.draw(left_image_cpu)
    if not os.path.exists(dn_output):
        os.mkdir(dn_output)
    fpfn_img = os.path.join(dn_output, 'stixels_result.png')
    cv2.imwrite(fpfn_img, sr_image)

demo()