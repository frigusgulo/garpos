import ctypes
import os
import numpy as np
from typing import Tuple

LIB_DIR = os.path.join(os.path.dirname(__file__), "f90lib")
LIB_RAYTRACE = os.path.join("libraytrace.so")


class Raytracer:
    def __init__(self, lib_raytrace: str = LIB_RAYTRACE, libdir: str = LIB_DIR):

        assert os.path.exists(lib_raytrace), f"Library {lib_raytrace} not found"
        assert os.path.exists(libdir), f"Library directory {libdir} not found"

        self.lib_raytrace = lib_raytrace
        self.libdir = libdir

        self.f90 = np.ctypeslib.load_library(lib_raytrace, libdir)
        self.f90.raytrace_.argtypes = [
            ctypes.POINTER(ctypes.c_int32),  # n
            ctypes.POINTER(ctypes.c_int32),  # nlyr
            np.ctypeslib.ndpointer(dtype=np.float64),  # l_depth
            np.ctypeslib.ndpointer(dtype=np.float64),  # l_speed
            np.ctypeslib.ndpointer(dtype=np.float64),  # dist
            np.ctypeslib.ndpointer(dtype=np.float64),  # yd
            np.ctypeslib.ndpointer(dtype=np.float64),  # ys
            np.ctypeslib.ndpointer(dtype=np.float64),  # dsv
            np.ctypeslib.ndpointer(dtype=np.float64),  # ctm (output)
            np.ctypeslib.ndpointer(dtype=np.float64),  # cag (output)
        ]
        self.f90.raytrace_.restype = ctypes.c_void_p

    def raytrace(
        self,
        l_depth: np.ndarray,
        l_speed: np.ndarray,
        n_dat,
        dst,
        yd,
        ys,
        dsv,
    
      
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_l = len(l_depth)
        n_n = ctypes.byref(ctypes.c_int32(n_dat * 2))
        n_l = ctypes.byref(ctypes.c_int32(n_l))
        ctm = np.zeros_like(dst)
        cag = np.zeros_like(dst)

        self.f90.raytrace_(n_n, n_l, l_depth, l_speed, dst, yd, ys, dsv, ctm, cag)

        cal_time = np.array(ctm)
        cal_angle = np.array(cag)

        cal_a0 = 180.0 - (cal_angle[:n_dat] + cal_angle[n_dat:]) / 2.0 * 180.0 / np.pi

        cal_tt = cal_time[:n_dat] + cal_time[n_dat:]

        return cal_tt, cal_a0
