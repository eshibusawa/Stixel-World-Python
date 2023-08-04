# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

class camera_params:
    def __init__(self, img = None):
        if img is None:
            self.rows = 768
            self.cols = 1024
        else:
            self.rows = img.shape[0]
            self.cols = img.shape[1]
        self.focal = 704.7082
        self.baseline = 0.8
        self.camera_center_y = 384.0
        self.max_dis = 128
        self.max_camera_tilt = 50 # define in deg
        self.min_camera_tilt = -50 # define in deg

class observation:
    def __init__(self):
        self.camera_tilt = 0.
        self.camera_height = 0.
        self.horizon_point = 0.
        self.alhpa_ground = 0.
        self.sigma_camera_tilt = 0.05 # define in deg
        self.sigma_camera_height = 0.05

class stixels_params:
    def __init__(self):
        self.sigma_disparity_object = 1.0
        self.sigma_disparity_ground = 2.0
        self.sigma_sky = 0.1

        self.pout = 0.15
        self.pout_sky = 0.4
        self.pord = 0.2
        self.pgrav = 0.1
        self.pblg = 0.04

        self.pground_given_nexist = 0.36
        self.pobject_given_nexist = 0.28
        self.psky_given_nexist = 0.36

        self.pnexist_dis = 0.0
        self.pground = 1.0/3.0
        self.pobject = 1.0/3.0
        self.psky = 1.0/3.0

        self.column_step = 5
        self.width_margin = 0

        self.median_step = False
        self.epsilon = 3.0
        self.range_objects_z = 10.20
        self.minimum_object_disparity = 1.0 # stixels with disparity below the minimum are considered sky
