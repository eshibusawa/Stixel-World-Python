# This file is part of Stixel-World-Python
# Copyleft 2023, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
# This file is licensed under the GPL-3.0 license.

import cupy as cp

def upload_constant(module, arr, key, dtype=cp.float32):
    arr_ptr = module.get_global(key)
    arr_gpu = cp.ndarray(arr.shape, dtype, arr_ptr)
    arr_gpu[:] = cp.array(arr, dtype=dtype)

def download_constant(module, key, shape, dtype=cp.float32):
    arr_ptr = module.get_global(key)
    arr_gpu = cp.ndarray(shape, dtype, arr_ptr)
    return arr_gpu.get()

def get_array_from_ptr(module, ptr, shape, dtype):
    mem = cp.cuda.UnownedMemory(ptr, 0, module)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    arr = cp.ndarray(shape=shape, dtype=dtype, memptr=memptr)
    return arr
