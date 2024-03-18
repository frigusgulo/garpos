import os
import sys
import math
import configparser
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, linalg, identity,block_diag
from sksparse.cholmod import cholesky
import pandas as pd
from typing import List, Tuple, Dict

# garpos module
from ..schemas.hyp_params import InversionParams
from ..schemas.obs_data import Site, ShotData,SoundVelocityProfile
from ..schemas.module_io import GaussianModelParameters
from .setup_model_v2 import init_position, make_knots, derivative2, data_correlation
from .forward import calc_forward, calc_gamma, jacobian_pos
from .output import outresults


def MPestimate(
        site_data:Site,
        shot_data:ShotData,
        sound_velocity_data: pd.DataFrame[SoundVelocityProfile],
        inversion_params:InversionParams):

    positional_model_params: GaussianModelParameters = init_position(site_data, inversion_params)

    knots_ctrlpts: Tuple[List[np.ndarray], List[int]] = make_knots(shot_data, inversion_params)

    knots, num_ctrl_points = knots_ctrlpts

    # set pointers for model paramter vector
    model_parameter_pointer: np.ndarray = np.cumsum(np.array([positional_model_params.num_params] + num_ctrl_points))

    H: lil_matrix = derivative2(model_parameter_pointer,knots,inversion_params)

    # TODO implement lines 187+ from mp_estimation.py

    # Set priori covariance matrix for model parameters
    # [[GMP,*],[* , H]]

    pmp_mean,cov_inv = positional_model_params.get_mean(),positional_model_params.get_cov_inv()
    # Set priori covariance matrix for model parameters

    Di = block_diag((cov_inv,H)).toarray()

    rank_Di = np.linalg.matrix_rank(Di,tol=1.0e-8)
    eigv_Di = np.linalg.eigvals(Di)[0]
    eigv_Di = eigv_Di[np.where(np.abs(eigv_Di.real) > 1.0e-8/(inversion_params.log_lambda[0]*10))].real

    if rank_Di != len(eigv_Di):
        print(np.linalg.matrix_rank(Di,tol=1.0e-8),len(eigv_Di))
        print("Di is not full rank")
        sys.exit(1)
    
    log_det_Di = np.log(eigv_Di).sum()

    # Set initial params for gradient gamma

    ##################################
    # Set log(TT/T0) and initial values
    ##################################

    # calc average depth*2 (charateristic depth)
    # Transponder depth mean + site position delta mean
    transponder_depth_mean = []
    for t in positional_model_params.transponder_positions.keys():
        transponder_depth_mean.append(
            t.mean[2]
        )
    L0 = np.array(transponder_depth_mean) + positional_model_params.site_position_delta.mean[2]
    L0 = abs(L0.mean())*2.0


    sound_vel_speed:np.ndarray = sound_velocity_data.speed
    sound_vel_depth:np.ndarray = sound_velocity_data.depth
    
    delta_speed = sound_vel_speed[1:] - sound_vel_speed[:-1]
    delta_depth = sound_vel_depth[1:] - sound_vel_depth[:-1]
    # line 233
    avg_vel: np.ndarray = (delta_speed * delta_depth) / 2

    V0 = avg_vel.sum() / (sound_vel_depth[-1] - sound_vel_depth[0])

    # Caclulate characteristic time

    T0 = L0/V0

    # Implement calc_forward and define "shots" schema, and other inputs



