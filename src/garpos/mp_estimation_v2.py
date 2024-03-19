import os
import sys
import math
import configparser
import numpy as np
from scipy.sparse import (
    csr_matrix,
    csc_matrix,
    lil_matrix,
    linalg,
    identity,
    block_diag,
)
from sksparse.cholmod import cholesky
import pandas as pd
from typing import List, Tuple, Dict

# garpos module
from ..schemas.hyp_params import InversionParams, InversionType
from ..schemas.obs_data import Site, ShotData, SoundVelocityProfile
from ..schemas.module_io import GaussianModelParameters
from .setup_model_v2 import init_position, make_knots, derivative2, data_correlation
from .forward_v2 import calc_forward, calc_gamma, jacobian_pos
from .output import outresults


def MPestimate(site_data: Site, inversion_params: InversionParams):

    init_position_output: Tuple[List[float], np.ndarray, List[int], Dict[str, int]] = (
        init_position(site_data, inversion_params)
    )

    model_params_mean, priori_cov_inv, slvidx0, transponder_idx = init_position_output

    knots: List[np.ndarray] = make_knots(site_data.shot_data, inversion_params)

    num_ctrl_points: List[int] = [
        max([0, len(kn) - inversion_params.spline_degree - 1]) for kn in knots
    ]

    # set pointers for model paramter vector

    model_parameter_pointer: np.ndarray = np.cumsum(
        np.array([len(model_params_mean)] + num_ctrl_points)
    )

    # set full model parameter vector

    model_parameter = np.zeros(model_parameter_pointer[-1])
    model_parameter[: model_parameter_pointer[0]] = model_params_mean

    slvidx = np.append(
        slvidx0,
        np.arange(model_parameter_pointer[0], model_parameter_pointer[-1], dtype=int),
    ).astype(int)

    H: lil_matrix = derivative2(model_parameter_pointer, knots, inversion_params)

    Di = block_diag((priori_cov_inv, H)).toarray()

    rank_Di = np.linalg.matrix_rank(Di, tol=1.0e-8)
    eigv_Di = np.linalg.eigvals(Di)[0]
    eigv_Di = eigv_Di[
        np.where(np.abs(eigv_Di.real) > 1.0e-8 / (inversion_params.log_lambda[0] * 10))
    ].real

    if rank_Di != len(eigv_Di):
        print(np.linalg.matrix_rank(Di, tol=1.0e-8), len(eigv_Di))
        print("Di is not full rank")
        sys.exit(1)

    log_det_Di = np.log(eigv_Di).sum()

    center_mean = [model_parameter[idx : idx + 3] for idx in transponder_idx.values()]
    center_mean = np.array(center_mean).mean(axis=0)
    number_transponders = len(site_data.transponders)

    site_data.shot_data["sta0_e"] = (
        model_parameter[site_data.shot_data["mtid"] + 0]
        + model_parameter[number_transponders * 3 + 0]
    )  # transponder position + station center position
    site_data.shot_data["sta0_n"] = (
        model_parameter[site_data.shot_data["mtid"] + 1]
        + model_parameter[number_transponders * 3 + 1]
    )
    site_data.shot_data["sta0_u"] = (
        model_parameter[site_data.shot_data["mtid"] + 2]
        + model_parameter[number_transponders * 3 + 2]
    )
    site_data.shot_data["mtde"] = (
        site_data.shot_data["sta0_e"].values - center_mean[0]
    )  # station center position - mean transponder position
    site_data.shot_data["mtdn"] = site_data.shot_data["sta0_n"].values - center_mean[1]
    site_data.shot_data["de0"] = (
        site_data.shot_data["ant_e0"].values
        - site_data.shot_data["ant_e0"].values.mean()
    )  # Normalized antennta positions
    site_data.shot_data["dn0"] = (
        site_data.shot_data["ant_n0"].values
        - site_data.shot_data["ant_n0"].values.mean()
    )
    site_data.shot_data["de1"] = (
        site_data.shot_data["ant_e1"].values
        - site_data.shot_data["ant_e1"].values.mean()
    )
    site_data.shot_data["dn1"] = (
        site_data.shot_data["ant_n1"].values
        - site_data.shot_data["ant_n1"].values.mean()
    )
    site_data.shot_data["iniflag"] = site_data.shot_data["flag"].copy()

    ##################################
    # Set log(TT/T0) and initial values
    ##################################

    # calc average depth*2 (charateristic depth)
    # Transponder depth mean + site position delta mean

    L0 = np.array(
        [
            (model_parameter[i * 3 + 2] + model_parameter[number_transponders * 3 + 2])
            for i in range(number_transponders)
        ]
    )
    L0 = abs(L0.mean()) * 2.0

    sound_vel_speed: np.ndarray = site_data.sound_speed_data.speed.values()
    sound_vel_depth: np.ndarray = site_data.sound_speed_data.depth.values()

    delta_speed = sound_vel_speed[1:] - sound_vel_speed[:-1]
    delta_depth = sound_vel_depth[1:] - sound_vel_depth[:-1]
    # line 233
    avg_vel: np.ndarray = (delta_speed * delta_depth) / 2

    V0 = avg_vel.sum() / (sound_vel_depth[-1] - sound_vel_depth[0])

    # Caclulate characteristic time

    T0 = L0 / V0

    # Implement calc_forward and define "shots" schema, and other inputs
    """
        # Initial parameters for gradient gamma
    shots['sta0_e'] = mp[shots['mtid']+0] + mp[len(MTs)*3+0]  # transponder position + station center position
    shots['sta0_n'] = mp[shots['mtid']+1] + mp[len(MTs)*3+1]
    shots['sta0_u'] = mp[shots['mtid']+2] + mp[len(MTs)*3+2]
    shots['mtde'] = (shots['sta0_e'].values - cnt[0])  # station center position - mean transponder position
    shots['mtdn'] = (shots['sta0_n'].values - cnt[1])
    shots['de0'] = shots['ant_e0'].values - shots['ant_e0'].values.mean()  # Normalized antennta positions
    shots['dn0'] = shots['ant_n0'].values - shots['ant_n0'].values.mean()
    shots['de1'] = shots['ant_e1'].values - shots['ant_e1'].values.mean()
    shots['dn1'] = shots['ant_n1'].values - shots['ant_n1'].values.mean()
    shots['iniflag'] = shots['flag'].copy()

    shots["logTT"] = np.log(shots.TT.values/T0)


  if invtyp != 0:
        shots["gamma"] = 0.

    """

    site_data.shot_data["logTT"] = np.log(
        site_data.shot_data.transmission_time.values / T0
    )

    if inversion_params.invtyp != InversionType.gammas:
        site_data.shot_data["gamma"] = 0.0

    site_data.shot_data = calc_forward(
        shot_data=site_data.shot_data,
        sound_velocity_profile=site_data.sound_speed_data,
        T0=T0,
        model_params=model_parameter,
        n_transponders=number_transponders,
        inversion_params=inversion_params,
    )

    icorrE = inversion_params.rejectcriteria < 0.1 and inversion_params.mu_t > 1.0e-3
    if not icorrE:
        inversion_params.mu_t = 0.0

    tmp = (
        site_data.shot_data[~site_data.shot_data["flag"]].reset_index(drop=True).copy()
    )
    ndata = len(tmp.index)
    scale = inversion_params.traveltimescale / T0

    TT0 = tmp.travel_times.values / T0

    if icorrE:
        E_factor = data_correlation(tmp, TT0, inversion_params)
        logdetEi = -E_factor.logdet()
    else:
        Ei = csc_matrix(np.diag(TT0**2.0)) / scale**2.0
        logdetEi = (np.log(TT0**2.0)).sum()
    # TODO implement lines 390 + in mp_estimation.py