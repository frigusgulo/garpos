import ctypes
import os
import numpy as np
from typing import Tuple
from pandas import DataFrame
import logging
import sys

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


    def calc_traveltime(self,
        shotdat:DataFrame,
        mp:np.ndarray,
        nMT:int,
        svp:DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        
        # station pos
        sta0_e = mp[shotdat['mtid']+0] + mp[nMT*3+0]
        sta0_n = mp[shotdat['mtid']+1] + mp[nMT*3+1]
        sta0_u = mp[shotdat['mtid']+2] + mp[nMT*3+2]
        
        e0 = shotdat.ant_e0.values + shotdat.ple0.values
        n0 = shotdat.ant_n0.values + shotdat.pln0.values
        u0 = shotdat.ant_u0.values + shotdat.plu0.values
        e1 = shotdat.ant_e1.values + shotdat.ple1.values
        n1 = shotdat.ant_n1.values + shotdat.pln1.values
        u1 = shotdat.ant_u1.values + shotdat.plu1.values
        
        dist0 = ((e0 - sta0_e)**2. + (n0 - sta0_n)**2.)**0.5
        dist1 = ((e1 - sta0_e)**2. + (n1 - sta0_n)**2.)**0.5
        
        dst = np.append(dist0, dist1)
        yd  = np.append(sta0_u, sta0_u)
        ys  = np.append(u0, u1)
        dsv = np.zeros(len(dst))
        
        # sv layer
        l_depth = svp.depth.values
        l_speed = svp.speed.values

        if np.isnan(yd).any():
            response = "nan in yd"
            logging.error(response)
            raise ValueError(response)

       
        if np.isnan(ys).any():
            response = "nan in ys"
            logging.error(response)
            raise ValueError(response)
       

        if min(yd) < -l_depth[-1]:
            response = "yd is deeper than layer"
            logging.error(response)
            raise ValueError(response)

     
        if max(ys) > -l_depth[0]:
            l_depth = np.append(-40.,l_depth)
            l_speed = np.append(l_speed[0],l_speed)
            if len(ys[ys > -l_depth[0]]) > 50:
                response = "many of ys are shallower than layer"
                logging.error(response)
                raise ValueError(response)
   
            if max(ys) > -l_depth[0]:
                response = "ys is shallower than layer"
                logging.error(response)
                raise ValueError(response)
            
        ndat = len(shotdat.index)

   
        return self.raytrace(l_depth, l_speed, ndat, dst, yd, ys, dsv)