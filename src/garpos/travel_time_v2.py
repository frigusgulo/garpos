# External Imports
import numpy as np
from pandas import DataFrame
import sys
# Internal Imports
from ...src_2.ray_tracer import Raytracer
from ..schemas.obs_data import ObservationData,SoundVelocityProfile
# TODO need a way to index model params by shot_data.mtid
# TODO what are shot_data.pl* values?

def calc_traveltime(
        shot_data:DataFrame[ObservationData],
        model_params:np.ndarray,
        nMT:int,
        svp:DataFrame[SoundVelocityProfile]):

    rayTracer = Raytracer()


    # station pos
    sta0_e =model_params[shot_data['mtid']+0] +model_params[nMT*3+0]
    sta0_n =model_params[shot_data['mtid']+1] +model_params[nMT*3+1]
    sta0_u =model_params[shot_data['mtid']+2] +model_params[nMT*3+2]


    e0 = shot_data.ant_e0.values + shot_data.ple0.values
    n0 = shot_data.ant_n0.values + shot_data.pln0.values
    u0 = shot_data.ant_u0.values + shot_data.plu0.values
    e1 = shot_data.ant_e1.values + shot_data.ple1.values
    n1 = shot_data.ant_n1.values + shot_data.pln1.values
    u1 = shot_data.ant_u1.values + shot_data.plu1.values

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
        print(yd[np.isnan(yd)])
        print("nan in yd")
        sys.exit(1)
    if np.isnan(ys).any():
        print(ys[np.isnan(ys)])
        print("nan in ys")
        sys.exit(1)

    if min(yd) < -l_depth[-1]:
        print(min(yd) , -l_depth[-1])
        print("yd is deeper than layer")
        print(model_params[0:15] , model_params[nMT*3+2])
        sys.exit(1)
    if max(ys) > -l_depth[0]:
        l_depth = np.append(-40.,l_depth)
        l_speed = np.append(l_speed[0],l_speed)
        if len(ys[ys > -l_depth[0]]) > 50:
            print(ys[ys > -l_depth[0]] , -l_depth[0])
            print("many of ys are shallower than layer")
            print(l_depth)
            sys.exit(1)
        if max(ys) > -l_depth[0]:
            print(max(ys) , -l_depth[0])
            print(ys[ys > -l_depth[0]] , -l_depth[0])
            print("ys is shallower than layer")
            print(l_depth)
            sys.exit(1)
	

    calTT,calA0 = rayTracer.raytrace(
        l_depth = l_depth,
        l_speed = l_speed,
        n_dat = shot_data.shape[0],
        yd = yd,
        ys = ys,
        dst = dst,
        dsv = dsv)
    
    return calTT,calA0