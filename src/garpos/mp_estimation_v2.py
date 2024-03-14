import os
import sys
import math
import configparser
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, linalg, identity
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

    model_parameter_pointer: np.ndarray = np.cumsum(np.array([positional_model_params.num_params] + num_ctrl_points))

    H: lil_matrix = derivative2(model_parameter_pointer,knots,inversion_params)

    # TODO implement lines 187+ from mp_estimation.py