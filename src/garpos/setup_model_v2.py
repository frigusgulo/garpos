import sys
import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse import csc_matrix, lil_matrix, linalg, block_diag
from sksparse.cholmod import cholesky
import pandas as pd

from ..schemas.obs_data import Site,ATDOffset,Transponder,PositionENU,DataFrame,ObservationData,SoundVelocityProfile
from ..schemas.hyp_params import InversionParams,InversionType
from ..schemas.module_io import GaussianModelParameters,Normal


def init_position(
    site_data: Site, inversion_params: InversionParams
) -> GaussianModelParameters:
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

    transponder_idx = {}
    model_params_mean = np.array([])
    model_params_var = []
    for t_idx,transponder in enumerate(site_data.transponders):

        transponder_idx[transponder.id] = t_idx * 3
        transponder_position_mean: List[float] = transponder.position_enu.get_position()
        transponder_position_cov: np.ndarray = transponder.position_enu.get_covariance()

        # creating 2d array for station position
        model_params_mean = np.append(model_params_mean, transponder_position_mean)
        # creating 2d array for station psotion uncertainty
        model_params_var.append(transponder_position_cov)

    # Get site position delta data
    site_position_delta: PositionENU = site_data.delta_center_position
    site_position_delta_mean: List[float] = site_position_delta.get_position()
    site_position_delta_cov: np.ndarray = site_position_delta.get_covariance()

    model_params_mean = np.append(model_params_mean, site_position_delta_mean)
    model_params_var.append(site_position_delta_cov)

    # Get Antenna-Transducer-Offset (ATD) data
    atd_offset: ATDOffset = site_data.atd_offset
    atd_offset_position_mean: List[float] = atd_offset.get_offset()
    atd_offset_cov: np.ndarray = atd_offset.get_covariance()

    ## Perform funcs from lines 77-88. Thresholds the diagonal of the atd_offset_cov matrix
    for i in range(atd_offset_cov.shape[0]):
        if atd_offset_cov[i, i] > 1.0e-8:
            atd_offset_cov[i, i] = 3.0

    model_params_mean = np.append(model_params_mean, atd_offset_position_mean)
    model_params_var.append(atd_offset_cov)

    # Set priori covariance matrix for model parameters
    priori_cov:np.ndarray = block_diag(model_params_var).toarray()

    slvidx0 = np.where(np.diag(priori_cov) > 1.0e-14)[0]

    priori_cov_inv = np.linalg.inv(priori_cov)
    return model_params_mean, priori_cov_inv, slvidx0, transponder_idx


def make_knots(shot_data: DataFrame[ObservationData], inv_params: InversionParams) -> List[np.ndarray]:
    """
    Create the B-spline knots for correction value "gamma".

    Args:
        shotdat (pd.DataFrame[ObservationData]): GNSS-A shot dataset.
        inv_params (InversionParams): The inversion parameters.

    Returns:
        List[np.ndarray]: B-spline knots for each component in "gamma".
    """

    sets: List[str] = shot_data['set'].unique()

    transmission_times_first_set = np.array([shot_data.loc[shot_data.set==s, "transmission_time"].min() for s in sets]).min()

    reception_times_last_set = np.array([shot_data.loc[shot_data.set==s, "reception_time"].max() for s in sets]).max()
    

    transmission_times_first: float = shot_data.transmission_time.values.min()
    reception_times_last:float  = shot_data.reception_time.values.max()

    
    observation_duration: float = transmission_times_first - reception_times_last

    # knotintervals = [knotint0, knotint1, knotint1, knotint2, knotint2]
    # Create the B-spline knots for correction value "gamma".
    knot_intervals = [
        inv_params.knotint0,
        inv_params.knotint1,
        inv_params.knotint1,
        inv_params.knotint2,
        inv_params.knotint2,
    ]
    n_knots = [int(observation_duration / interval) for interval in knot_intervals]

    knots: List[np.ndarray] = [
        np.linspace(transmission_times_first, reception_times_last, knot + 1) for knot in n_knots
    ]

    for k, _ in enumerate(knots):
        if n_knots[k] == 0:
            knots[k] = np.array([])
        rm_knot = np.array([])

        for i in range(len(sets) - 1):
            isetkn = np.where(
                (knots[k] > reception_times_last_set) & (knots[k] < transmission_times_first_set[i + 1])
            )[0]
            if len(isetkn) > 2 * (inv_params.spline_degree + 2):
                rmknot = np.append(
                    rm_knot,
                    isetkn[
                        isetkn[
                            inv_params.spline_degree + 1 : -inv_params.spline_degree - 1
                        ]
                    ],
                )

        rm_knot = rm_knot.astype(int)
        if len(rmknot) > 0:
            knots[k] = np.delete(knots[k], rm_knot)

        dkn = (observation_duration) / float(n_knots[k])

        add_kn_first = np.array(
            [
                transmission_times_first - dkn * (n + 1)
                for n in reversed(range(inv_params.spline_degree))
            ]
        )
        add_kn_last = np.array(
            [transmission_times_first + dkn * (n + 1) for n in range(inv_params.spline_degree)]
        )

        knots[k] = np.append(add_kn_first, knots[k])
        knots[k] = np.append(knots[k], add_kn_last)

    return knots


def derivative2(
    model_param_pointer:np.ndarray, knots: List[np.ndarray], inv_params: InversionParams
) -> lil_matrix:
    # Calculate the matrix for 2nd derivative of the B-spline basis

    lambda_grad = inv_params.log_gradlambda * 10
    lambda_0 = inv_params.log_lambda[0] * 10
    lambdas = [lambda_0] + [lambda_0 * lambda_grad] * 4

    p = inv_params.spline_degree
    diff = lil_matrix((model_param_pointer[5], model_param_pointer[5]))

    for k in range(len(lambdas)):
        kn = knots[k]
        if len(kn) == 0:
            continue

        delta = lil_matrix((model_param_pointer[k + 1] - model_param_pointer[k] - 2, model_param_pointer[k + 1] - model_param_pointer[k]))
        w = lil_matrix((model_param_pointer[k + 1] - model_param_pointer[k] - 2, model_param_pointer[k + 1] - model_param_pointer[k] - 2))

        for j in range(model_param_pointer[k + 1] - model_param_pointer[k] - 2):
            dkn0 = (kn[j + p + 1] - kn[j + p]) / 3600.0
            dkn1 = (kn[j + p + 2] - kn[j + p + 1]) / 3600.0

            delta[j, j] = 1.0 / dkn0
            delta[j, j + 1] = -1.0 / dkn0 - 1.0 / dkn1
            delta[j, j + 2] = 1.0 / dkn1

            if j >= 1:
                w[j, j - 1] = dkn0 / 6.0
                w[j - 1, j] = dkn0 / 6.0
            w[j, j] = (dkn0 + dkn1) / 3.0
        delta = delta.tocsr()
        w = w.tocsr()

        dk = (delta.T @ w) @ delta
        diff[model_param_pointer[k] : model_param_pointer[k + 1], model_param_pointer[k] : model_param_pointer[k + 1]] = dk / lambdas[k]

    H = diff[model_param_pointer[0] :, model_param_pointer[0] :]

    return H


def data_correlation(
    shot_data: DataFrame[ObservationData],TT0: np.ndarray, inv_params: InversionParams
) -> np.ndarray:
    """
    Calculate the data correlation matrix.

    Args:
        shot_data (ShotData): The shot data.
        inv_params (InversionParams): The inversion parameters.

    Returns:
        np.ndarray: The data correlation matrix.
    """
    n_data = shot_data.shape[0]
    transmission_times = shot_data.transmission_time.values
    mtids = shot_data.mtid.values
    negative_delta_transmission_times: List = shot_data[
        (shot_data.transmission_times.diff(1) == 0.0) & (shot_data.mtid.diff(1) == 0.0)
    ]

    if len(negative_delta_transmission_times) > 0:
        print("transmission times have negative delta")
        print("error in data_var_base; see setup_model.py")
        sys.exit(1)

    E = lil_matrix((n_data, n_data))
    for i, (iMT,iST) in enumerate(zip(mtids,transmission_times)):
        idx = shot_data[ ( abs(transmission_times - iST) < inv_params.mu_t * 4.)].index
        dshot = np.abs(iST - transmission_times[idx]) / inv_params.mu_t
        dcorr = np.exp(-dshot) * (inv_params.mu_mt + (1.0 - inv_params.mu_mt) * (iMT == mtids[idx]))
        E[i, idx] = dcorr / TT0[i] / TT0[idx]

    E = E.tocsc()

    # cholesky decomposition
    E_factor = cholesky(E, ordering_method="natural")

    return E_factor

