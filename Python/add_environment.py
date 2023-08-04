# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import os
if os.environ.get('NVCC') is None:
    os.environ['NVCC'] = '/usr/local/cuda/bin/nvcc'
import cupy as cp
