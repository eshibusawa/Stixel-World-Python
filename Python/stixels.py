# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import math
import os

import numpy as np
import cupy as cp

from stixels_params import observation

class Stixels:
    def __init__(self, param, camera_param):
        self.param = param
        self.camera_param = camera_param
        self.gpu_module = None
        self.gpu_module_nvcc = None

    def compile_module(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'stixels.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        self.max_sections = 50
        self.max_log_probability = 10000.0
        o = observation()
        self.sky_model = self.compute_sky_model()
        self.compute_probability_and_log()
        cuda_source = cuda_source.replace('STIXELS_FOCAL_LENGTH', self.strf(self.camera_param.focal))
        cuda_source = cuda_source.replace('STIXELS_BASELINE', self.strf(self.camera_param.baseline))
        cuda_source = cuda_source.replace('STIXELS_MAX_DISPARITY', str(self.camera_param.max_dis))
        cuda_source = cuda_source.replace('STIXELS_SIGMA_CAMERA_HEIGHT', self.strf(o.sigma_camera_height))
        cuda_source = cuda_source.replace('STIXELS_SIGMA_CAMERA_TILT', self.strf(np.deg2rad(o.sigma_camera_tilt))) # deg -> rad
        cuda_source = cuda_source.replace('STIXELS_SIGMA_DISPARITY_GROUND', self.strf(self.param.sigma_disparity_ground))
        cuda_source = cuda_source.replace('STIXELS_SIGMA_DISPARITY_OBJECT', self.strf(self.param.sigma_disparity_object))
        cuda_source = cuda_source.replace('STIXELS_P_OUT', self.strf(self.param.pout))
        cuda_source = cuda_source.replace('STIXELS_STEP_SIZE', str(self.param.column_step))
        cuda_source = cuda_source.replace('STIXELS_WIDTH_MARGIN', str(self.param.width_margin))
        cuda_source = cuda_source.replace('STIXELS_RANGE_OBJECTS_Z', self.strf(self.param.range_objects_z))
        cuda_source = cuda_source.replace('STIXELS_P_UNIFORM', self.strf(self.p_uniform))
        cuda_source = cuda_source.replace('STIXELS_NO_P_N_EXISTS_OBJECT_LOG', self.strf(self.no_p_n_exists_object_log))
        cuda_source = cuda_source.replace('STIXELS_MATH_SQRT_2PI', self.strf(float(np.sqrt(2 * np.pi))))
        cuda_source = cuda_source.replace('STIXELS_MATH_SQRT_2', self.strf(float(np.sqrt(2.0))))

        self.gpu_module = cp.RawModule(code=cuda_source)
        self.gpu_module.compile()

    def compile_module_nvcc(self):
        dn = os.path.dirname(__file__)
        fnl = list()
        fnl.append(os.path.join(dn, 'stixels_cub.cu'))

        cuda_source = None
        for fpfn in fnl:
            with open(fpfn, 'r') as f:
                cs = f.read()
            if cuda_source is None:
                cuda_source = cs
            else:
                cuda_source += cs

        height_raw = self.camera_param.rows
        height_lut, width_lut = self.get_object_LUT_size()
        assert width_lut <= 1024 #
        thread_blocks = width_lut
        # items_per_thread = int(math.ceil(width_lut / thread_blocks))
        cuda_source = cuda_source.replace('CUDA_WARP_SIZE', str(32))
        cuda_source = cuda_source.replace('STIXELS_WIDTH_DISPARITY', str(height_raw))
        cuda_source = cuda_source.replace('STIXELS_MAX_DISPARITY', str(self.camera_param.max_dis))
        cuda_source = cuda_source.replace('STIXELS_WIDTH_LUT', str(width_lut))
        cuda_source = cuda_source.replace('STIXELS_HEIGHT_LUT', str(height_lut))
        cuda_source = cuda_source.replace('STIXELS_THREAD_BLOCKS', str(thread_blocks))
        # cuda_source = cuda_source.replace('STIXELS_THREAD_ITEMS', str(items_per_thread))
        cuda_source = cuda_source.replace('STIXELS_NORMALIZATION_SKY', self.strf(self.sky_model[0]))
        cuda_source = cuda_source.replace('STIXELS_INV_SIGMA2_SKY', self.strf(self.sky_model[1]))
        cuda_source = cuda_source.replace('STIXELS_P_UNIFORM_SKY', self.strf(self.sky_model[2]))
        cuda_source = cuda_source.replace('STIXELS_NO_P_EXISTS_GIVEN_SKY_LOG', self.strf(self.no_p_n_exists_given_sky_log))
        cuda_source = cuda_source.replace('STIXELS_MAX_LOGPROB', self.strf(self.max_log_probability))
        cuda_source = cuda_source.replace('STIXELS_P_UNIFORM', self.strf(self.p_uniform))
        cuda_source = cuda_source.replace('STIXELS_NO_P_EXISTS_GIVEN_GROUND_LOG', self.strf(self.no_p_n_exists_given_ground_log))
        cuda_source = cuda_source.replace('STIXELS_PRIOR_COST_GROUND_BOTTOM', self.strf(self.prior_cost_ground_bottom))
        cuda_source = cuda_source.replace('STIXELS_PRIOR_COST_OBJECT_BOTTOM0', self.strf(self.prior_cost_object_bottom0))
        cuda_source = cuda_source.replace('STIXELS_PRIOR_COST_OBJECT_BOTTOM1', self.strf(self.prior_cost_object_bottom1))
        cuda_source = cuda_source.replace('STIXELS_MATH_LOG_2', self.strf(np.log(2.)))
        cuda_source = cuda_source.replace('STIXELS_MATH_LOG_3E-1', self.strf(np.log(.3)))
        cuda_source = cuda_source.replace('STIXELS_MATH_LOG_7E-1', self.strf(np.log(.7)))
        cuda_source = cuda_source.replace('STIXELS_PRIOR_COST_OBJECT_EPSILON', self.strf(self.param.epsilon))
        cuda_source = cuda_source.replace('STIXELS_P_GRAVITY', self.strf(self.param.pgrav))
        cuda_source = cuda_source.replace('STIXELS_P_BELOW_GROUND', self.strf(self.param.pblg))
        cuda_source = cuda_source.replace('STIXELS_P_ORDER', self.strf(self.param.pord))
        cuda_source = cuda_source.replace('STIXELS_MAX_SECTIONS', str(self.max_sections))
        cuda_source = cuda_source.replace('STIXELS_MINIMUM_OBJECT_DISPARITY', self.strf(self.param.minimum_object_disparity))

        self.gpu_module_nvcc = cp.RawModule(code=cuda_source, backend='nvcc')
        self.gpu_module_nvcc.compile()

    def setup_module(self):
        if self.gpu_module is None:
            self.compile_module()
        if self.gpu_module_nvcc is None:
            self.compile_module_nvcc()

    @staticmethod
    def strf(val):
        eps = 1E-7
        if np.abs(val) < eps:
            return str(0)
        return str(val) + 'f'

    def get_smoothed_disparity_size(self):
        height = (self.camera_param.cols - self.param.width_margin) // self.param.column_step
        width = self.camera_param.rows
        return height, width

    def get_object_LUT_size(self):
        height, width = self.get_smoothed_disparity_size()
        # width + 1== height-raw + 1 is ceiled to warp unit (32)
        width_wu = 32 * int(math.ceil((width + 1)/ 32))
        return height, width_wu

    def initialize(self):
        self.setup_module()
        self.object_disparity_range = self.compute_object_disparity_range()
        self.object_cost_LUT = self.compute_object_cost_LUT()

    def compute_probability_and_log(self):
        eps = 1E-7
        log_inf = 10000 # +inf
        def log_error_check(p):
            if p < eps:
                ret = log_inf
            else:
                ret = np.log(p)
            return float(ret)

        self.p_uniform = np.log(self.camera_param.max_dis) - log_error_check(self.param.pout)

        self.p_n_exists_object = (self.param.pobject_given_nexist * self.param.pnexist_dis / self.param.pobject)
        self.no_p_n_exists_object_log = -log_error_check(1 - self.p_n_exists_object)

        p_n_exists_given_sky = (self.param.psky_given_nexist * self.param.pnexist_dis)/self.param.psky
        self.no_p_n_exists_given_sky_log = -log_error_check(1 - p_n_exists_given_sky)

        p_n_exists_given_ground = (self.param.pground_given_nexist * self.param.pnexist_dis)/self.param.pground
        self.no_p_n_exists_given_ground_log = -log_error_check(1 - p_n_exists_given_ground)

        self.prior_cost_ground_bottom = np.log(2.) + np.log(self.camera_param.rows)
        self.prior_cost_object_bottom1 = np.log(self.camera_param.rows * self.camera_param.max_dis)
        self.prior_cost_object_bottom0 = np.log(2.) + self.prior_cost_object_bottom1

    def compute_object_disparity_range(self):
        d = cp.empty((self.camera_param.max_dis), dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('computeObjectDisparityRange')
        sz_block = 1024, 1
        sz_grid = math.ceil(d.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_object_model(self):
        d = cp.empty((self.camera_param.max_dis, 2), dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('computeObjectModel')
        sz_block = 1024, 1
        sz_grid = math.ceil(d.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_object_cost_LUT(self):
        object_model = self.compute_object_model()

        d = cp.empty((self.camera_param.max_dis, self.camera_param.max_dis), dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('computeObjectCostLUT')
        sz_block = 32, 32
        sz_grid = math.ceil(d.shape[0] / sz_block[0]), math.ceil(d.shape[1] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d,
                object_model
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_object_LUT(self):
        object_model = self.compute_object_model()

        d = cp.empty((self.camera_param.max_dis, self.camera_param.max_dis), dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('computeObjectCostLUT')
        sz_block = 32, 32
        sz_grid = math.ceil(d.shape[0] / sz_block[0]), math.ceil(d.shape[1] / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d,
                object_model
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        return d

    def compute_sky_model(self):
        a_range = (math.erf(self.camera_param.max_dis / (self.param.sigma_sky * np.sqrt(2.))) - math.erf(0.))/2.
        normalization_sky = np.log(a_range * (self.param.sigma_sky * np.sqrt(2*np.pi)) / (1 - self.param.pout_sky))
        inverse_sigma2_sky = 1 / (2 * self.param.sigma_sky * self.param.sigma_sky)
        uniform_sky = np.log(self.camera_param.max_dis) - np.log(self.param.pout_sky)
        return normalization_sky, inverse_sigma2_sky, uniform_sky

    def update_ground_model(self, sz, v_horizontal, gound_alpha, camera_height, cameara_tilt):
        v_horizontal = self.camera_param.rows - v_horizontal - 1
        d = cp.empty((sz, 3), dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module.get_function('updateGroundModel')
        sz_block = 1024, 1
        sz_grid = math.ceil(d.shape[0] / sz_block[0]), 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, cp.int32(v_horizontal), cp.float32(gound_alpha), cp.float32(camera_height), cp.float32(cameara_tilt), d.shape[0]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.v_horizontal = v_horizontal
        self.ground_model = d
        return d

    def horizontal_smoothing_and_transpose(self, disparity):
        assert disparity.flags.c_contiguous
        assert disparity.dtype == cp.float32
        assert disparity.shape[0] == self.camera_param.rows
        assert disparity.shape[1] == self.camera_param.cols
        # compute smoothed column size
        sz = self.get_smoothed_disparity_size()
        d = cp.empty(sz, dtype=cp.float32)
        assert d.flags.c_contiguous
        if self.param.median_step:
            gpu_func = self.gpu_module.get_function('horizontalMedianAndTranspose')
        else:
            gpu_func = self.gpu_module.get_function('horizontalMeanAndTranspose')

        assert disparity.shape[0] == d.shape[1]
        sz_block = 1024, 1
        sz_grid = math.ceil(d.shape[0] * d.shape[1] / sz_block[0]), 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, disparity, disparity.shape[0], disparity.shape[1], d.shape[0]
            )
        )
        cp.cuda.runtime.deviceSynchronize()
        self.disparity = d
        return d

    def compute_object_LUT(self):
        sz = self.get_object_LUT_size()
        sz_LUT = sz[0], self.camera_param.max_dis, sz[1]
        d = cp.empty(sz_LUT, dtype=cp.float32)
        assert d.flags.c_contiguous
        gpu_func = self.gpu_module_nvcc.get_function('computeObjectLUT')
        sz_block = 32, 32
        sz_grid = d.shape[0], math.ceil(self.camera_param.max_dis / sz_block[1])
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(
                d, self.disparity, self.object_cost_LUT)
        )
        cp.cuda.runtime.deviceSynchronize()
        self.object_LUT = d
        return d

    def compute_stixels(self):
        sz = self.get_object_LUT_size()
        d_type = cp.empty((sz[0], self.max_sections, 3), dtype=cp.int16)
        d_disparity = cp.empty((sz[0], self.max_sections), dtype=cp.float32)
        assert d_type.flags.c_contiguous
        assert d_disparity.flags.c_contiguous
        gpu_func = self.gpu_module_nvcc.get_function('computeStixels')
        sz_block = sz[1], 1
        sz_grid = sz[0], 1
        gpu_func(
            block=sz_block, grid=sz_grid,
            args=(d_type, d_disparity,
                self.disparity, self.ground_model, self.object_LUT, self.object_disparity_range,
                cp.int32(self.v_horizontal))
        )
        cp.cuda.runtime.deviceSynchronize()
        self.section_type = d_type
        self.section_disparity = d_disparity

    @staticmethod
    def get_stixels(d_type, d_disparity):
        def int_to_type(t):
            if t == 0:
                ret = 'Ground'
            elif t == 1:
                ret = 'Object'
            elif t == 2:
                ret = 'Sky'
            else:
                ret = None
            return ret

        ret = list()
        for ts, ds in zip(d_type, d_disparity):
            section_row = list()
            for t, d in zip(ts, ds):
                if t[2] == -1:
                    break
                s = dict()
                s['type'] = int_to_type(int(t[2]))
                s['vT'] = int(t[1])
                s['vB'] = int(t[0])
                s['disparity'] = float(d)
                section_row.append(s)
            ret.append(section_row)
        return ret
