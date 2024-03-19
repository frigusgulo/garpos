import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix


import sys
import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import csc_matrix, lil_matrix, linalg, block_diag
from sksparse.cholmod import cholesky
import pandas as pd

from ..schemas.obs_data import (
    Site,
    ATDOffset,
    Transponder,
    PositionENU,
    DataFrame,
    ObservationData,
    SoundVelocityProfile,
)
from ..schemas.hyp_params import InversionParams, InversionType
from ..schemas.module_io import GaussianModelParameters, Normal
from .travel_time_v2 import calc_traveltime
from .coordinate_trans_v2 import corr_attitude

def calc_forward(
        shot_data: DataFrame[ObservationData],
        sound_velocity_profile: DataFrame[SoundVelocityProfile], 
        T0:float,
        model_params:np.ndarray,
        n_transponders:int,
        inversion_params:InversionParams) -> DataFrame[ObservationData]:
    
    rejection_criteria: float = inversion_params.rejectcriteria

    calc_ATD = np.vectorize(corr_attitude)


    pl0 =model_params[(n_transponders+1)*3+0]
    pl1 =model_params[(n_transponders+1)*3+1]
    pl2 =model_params[(n_transponders+1)*3+2]
    hd0 =shot_data.head0.values
    hd1 =shot_data.head1.values
    rl0 =shot_data.roll0.values
    rl1 =shot_data.roll1.values
    pc0 =shot_data.pitch0.values
    pc1 =shot_data.pitch1.values
    ple0, pln0, plu0 = calc_ATD(pl0, pl1, pl2, hd0, rl0, pc0)
    ple1, pln1, plu1 = calc_ATD(pl0, pl1, pl2, hd1, rl1, pc1)
    shot_data['ple0'] = ple0
    shot_data['pln0'] = pln0
    shot_data['plu0'] = plu0
    shot_data['ple1'] = ple1
    shot_data['pln1'] = pln1
    shot_data['plu1'] = plu1

    cTT,cTO = calc_traveltime(shot_data, model_params, n_transponders, inversion_params, sound_velocity_profile)
