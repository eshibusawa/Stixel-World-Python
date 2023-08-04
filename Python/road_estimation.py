# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import math
import os

import numpy as np
import cupy as cp

import cv2

class RoadEstimation:
    def __init__(self, camera_param):
        self.camera_param = camera_param
        self.gpu_module_nvcc = None
        self.histogram_threshold = 0.5
        self.max_camera_tilt = math.radians(self.camera_param.max_camera_tilt)
        self.min_camera_tilt = math.radians(self.camera_param.min_camera_tilt)

    def compile_module_nvcc(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'road_estimation_cub.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        self.histogram_threads = 512
        assert(self.camera_param.cols % self.histogram_threads == 0)
        histogram_items = self.camera_param.cols // self.histogram_threads
        cuda_source = cuda_source.replace('ROAD_ESTIMATION_HISTOGRAM_THREADS', str(self.histogram_threads))
        cuda_source = cuda_source.replace('ROAD_ESTIMATION_HISTOGRAM_ITEMS', str(histogram_items))
        cuda_source = cuda_source.replace('ROAD_ESTIMATION_HEIGHT_DISPARITY', str(self.camera_param.rows))
        cuda_source = cuda_source.replace('ROAD_ESTIMATION_WIDTH_DISPARITY', str(self.camera_param.cols))
        cuda_source = cuda_source.replace('ROAD_ESTIMATION_MAX_DISPARITY', str(self.camera_param.max_dis))
        cuda_source = cuda_source.replace('ROAD_ESTIMATION_HISTOGRAM_THRESHOLD', self.strf(self.histogram_threshold))

        self.gpu_module_nvcc = cp.RawModule(code=cuda_source, backend='nvcc')
        self.gpu_module_nvcc.compile()

    def setup_module(self):
        if self.gpu_module_nvcc is None:
            self.compile_module_nvcc()

    @staticmethod
    def strf(val):
        eps = 1E-7
        if np.abs(val) < eps:
            return str(0)
        return str(val) + 'f'

    def compute(self, disparity):
        assert disparity.shape[0] == self.camera_param.rows
        assert disparity.shape[1] == self.camera_param.cols

        d = cp.zeros((self.camera_param.rows, self.camera_param.max_dis), dtype=cp.int32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module_nvcc.get_function('computeHistogram')
        sz_block = self.histogram_threads, 1
        sz_grid = d.shape[0], 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(d, disparity)
        )
        cp.cuda.runtime.deviceSynchronize()
        self.vd_histogram = d
        histogram_max = cp.max(self.vd_histogram[:,1:]).get()

        d = cp.empty(self.vd_histogram.shape, dtype=cp.uint8)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module_nvcc.get_function('thresholdingHistogram')
        sz_block = 1024, 1
        sz_grid = math.ceil(d.size/sz_block[0]), 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(d, self.vd_histogram, cp.float32(histogram_max))
        )
        cp.cuda.runtime.deviceSynchronize()
        self.vd_binary = d

        vd_binary_cpu = self.vd_binary.get()
        lines = cv2.HoughLines(vd_binary_cpu, 1.0, np.pi/180, 25)
        for l in lines:
            rho, theta = l[0,0], l[0,1]
            if np.abs(theta) > 1E-7:
                horizon_point = rho / math.sin(theta)
                camera_tilt = -math.atan((self.camera_param.camera_center_y - horizon_point) / (self.camera_param.focal))
                bottom = vd_binary_cpu.shape[0] - 1
                alpha_ground = -(rho - bottom*math.sin(theta))/math.cos(theta)/(horizon_point - bottom)
                camera_height = self.camera_param.baseline * math.cos(camera_tilt) / alpha_ground
                if (camera_tilt >= self.min_camera_tilt) and (camera_tilt <= self.max_camera_tilt):
                    horizon_point = math.ceil(horizon_point)
                    return camera_tilt, camera_height, horizon_point, alpha_ground
        return None
