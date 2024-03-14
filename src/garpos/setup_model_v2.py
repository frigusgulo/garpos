import sys
import numpy as np
from typing import List,Tuple,Dict
from scipy.sparse import csc_matrix, lil_matrix, linalg,block_diag
from sksparse.cholmod import cholesky
import pandas as pd

from ..schemas.obs_data import Site,PositionENU,ATDOffset,Transponder,ShotData
from ..schemas.hyp_params import HyperParams, InversionParams


def init_position(site_data: Site, hyper_params: HyperParams) -> Tuple[List[float], np.ndarray, List[int], Dict[str, int]]:
    """
    Calculate Jacobian matrix for positions.

    Args:
        site_data (Site): The site data containing information about transponders, site position, and offsets.
        hyper_params (HyperParams): The hyperparameters for the inversion.

    Returns:
        Tuple[List[float], np.ndarray, List[int], Dict[str, int]]: A tuple containing:
            - mean_positions (List[float]): The mean positions of the transponders, site position delta, and offset.
            - cov_pos_inv (np.ndarray): The inverse of the initial covariance matrix.
            - solve_idx (List[int]): The indices of the model parameters to be solved.
            - transponder_covmat_positions (Dict[str, int]): A dictionary mapping transponder IDs to their covariance matrix positions.
    """

    atd_offset: ATDOffset = site_data.atd_offset
    atd_offset_cov: np.ndarray = atd_offset.get_covariance()
    atd_offset_position: List[float] = atd_offset.get_offset()

    transponders: List[Transponder] = site_data.transponders
    site_position_delta: np.ndarray = site_data.delta_center_position.get_position()
    if hyper_params.inversiontype.gammas:
        site_position_delta += np.asarray(hyper_params.positionalOffset)

    site_position_cov: np.ndarray = site_data.center_enu.get_covariance()

    # 1. Create 1d array for station positions, and seperate array for station position uncertainty

    # compute a prirori covariance for site data
    # transponder position covariance: T_n
    # site center position covariance: C
    # antenna transducer offset covariance: ATD 
    # cov_pos_init = [T_1, T_2, ... T_n-1, C, ATD]
    mean_positions : List[float] = []
    transponder_covmat_positions : dict[str,int] = {}
    transponder_position_covs : List[np.ndarray] = []

    for idx,t in enumerate(transponders):
        transponder_covmat_positions[t.id] = idx*3
        transponder_position_covs.append(t.position_enu.get_covariance())
        mean_positions.extend(t.position_enu.get_position().tolist())
    

    mean_positions.extend(site_position_delta.tolist())
    mean_positions.extend(atd_offset_position)
    # if position_uncertainty.sum() > 0.002:
    #     # Dont know what this is actually checking, from line 70 of setup_model.py
    #     raise ValueError("Error: ape for each station must be 0 in rigid-array mode!")
    
    
    # Perform funcs from lines 77-88. Thresholds the diagonal of the atd_offset_cov matrix
    for i in range(atd_offset_cov.shape[0]):
        if atd_offset_cov[i,i] > 1.e-8:
            atd_offset_cov[i,i] = 3.0


    cov_pos_init:np.ndarray = block_diag((transponder_position_covs,site_position_cov,atd_offset_cov)).toarray()

    # Get the indices of the model parameters to be solved by checking the diagonal of the covariance matrix
    solve_idx = np.where(np.diag(cov_pos_init) > 1.e-14)[0].tolist()

    cov_pos_inv = np.linalg.inv(cov_pos_init)

    return mean_positions,cov_pos_inv,solve_idx,transponder_covmat_positions


def make_knots(shot_data: ShotData, inv_params: InversionParams) -> List[np.ndarray]:
    """
    Create the B-spline knots for correction value "gamma".

    Args:
        shotdat (pd.DataFrame): GNSS-A shot dataset.
        spdeg (int): Spline degree (=3).
        knotintervals (List[int]): Approximate knot intervals.

    Returns:
        List[np.ndarray]: B-spline knots for each component in "gamma".
    """

    n_sets = len(shot_data.sets)
    
    trans_times_first = np.min([shot_obs.trans_time[0] for shot_data in list(shot_data.sets) for shot_obs in shot_data.values()])
    trans_times_last = np.max([shot_obs.trans_time[-1] for shot_data in list(shot_data.sets) for shot_obs in shot_data.values()])

    observation_duration = trans_times_last - trans_times_first

    # knotintervals = [knotint0, knotint1, knotint1, knotint2, knotint2]
    # Create the B-spline knots for correction value "gamma".
    knot_intervals = [inv_params.knotint0,inv_params.knotint1,inv_params.knotint1,inv_params.knotint2,inv_params.knotint2]
    n_knots = [ int(observation_duration / interval) for interval in knot_intervals]
    knots = [np.linspace(trans_times_first,trans_times_last,knot+1) for knot in n_knots]

    for k, _ in enumerate(knots):
        if n_knots[k] == 0:
            knots[k] = np.array([])
        rm_knot = np.array([])

        for i in range(n_sets-1):
            isetkn = np.where( (knots[k] > trans_times_last) & (knots[k]<trans_times_first[i+1]) )[0]
            if len(isetkn) > 2*(inv_params.spline_degree+2):
                rmknot = np.append(rm_knot, isetkn[isetkn[inv_params.spline_degree+1:-inv_params.spline_degree-1]])

        rm_knot = rm_knot.astype(int)
        if len(rmknot) > 0:
            knots[k] = np.delete(knots[k], rm_knot)
        
        dkn = (observation_duration) / float(n_knots[k])

        add_kn_first = np.array([trans_times_first - dkn*(n+1) for n in reversed(range(inv_params.spline_degree))])
        add_kn_last = np.array([trans_times_last + dkn*(n+1) for n in range(inv_params.spline_degree)])

        knots[k] = np.append(add_kn_first,knots[k])
        knots[k] = np.append(knots[k],add_kn_last)

    
    return knots