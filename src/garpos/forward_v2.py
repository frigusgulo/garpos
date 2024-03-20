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
    T0: float,
    model_params: np.ndarray,
    n_transponders: int,
    inversion_params: InversionParams,
) -> DataFrame[ObservationData]:

    rejection_criteria: float = inversion_params.rejectcriteria

    calc_ATD = np.vectorize(corr_attitude)

    pl0 = model_params[(n_transponders + 1) * 3 + 0]
    pl1 = model_params[(n_transponders + 1) * 3 + 1]
    pl2 = model_params[(n_transponders + 1) * 3 + 2]
    hd0 = shot_data.head0.values
    hd1 = shot_data.head1.values
    rl0 = shot_data.roll0.values
    rl1 = shot_data.roll1.values
    pc0 = shot_data.pitch0.values
    pc1 = shot_data.pitch1.values
    ple0, pln0, plu0 = calc_ATD(pl0, pl1, pl2, hd0, rl0, pc0)
    ple1, pln1, plu1 = calc_ATD(pl0, pl1, pl2, hd1, rl1, pc1)
    shot_data["ple0"] = ple0
    shot_data["pln0"] = pln0
    shot_data["plu0"] = plu0
    shot_data["ple1"] = ple1
    shot_data["pln1"] = pln1
    shot_data["plu1"] = plu1

    cTT, cTO = calc_traveltime(
        shot_data,
        model_params,
        n_transponders,
        inversion_params,
        sound_velocity_profile,
    )

    logTTc = np.log(cTT / T0) - shot_data.gamma.values
    ResiTT = shot_data.logTT.values - logTTc

    shot_data["calcTT"] = cTT
    shot_data["TakeOff"] = cTO
    shot_data["logTTc"] = logTTc
    shot_data["ResiTT"] = ResiTT
    # approximation log(1 + x) ~ x
    shot_data["ResiTTreal"] = ResiTT * shot_data.TT.values

    if inversion_params.rejectcriteria > 0.1:
        aveRTT = shot_data[~shot_data["flag"]].ResiTT.values.mean()
        sigRTT = shot_data[~shot_data["flag"]].ResiTT.values.std()
        th0 = aveRTT + inversion_params.rejectcriteria * sigRTT
        th1 = aveRTT - inversion_params.rejectcriteria * sigRTT
        shot_data["flag"] = (
            (shot_data["ResiTT"] > th0)
            | (shot_data["ResiTT"] < th1)
            | shot_data["iniflag"]
        )
        aveRTT1 = shot_data[~shot_data["flag"]].ResiTT.mean()
        sigRTT1 = shot_data[~shot_data["flag"]].ResiTT.std()

    return shot_data


def calc_gamma(
    mp: np.ndarray,
    shotdat: DataFrame[ObservationData],
    imp0: np.ndarray,
    knots: List[np.ndarray],
    inversion_params: InversionParams,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Calculate correction value "gamma" in the observation eqs.

    Parameters
    ----------
    mp : ndarray
        complete model parameter vector.
    shotdat : DataFrame
        GNSS-A shot dataset.
    imp0 : ndarray (len=5)
        Indices where the type of model parameters change.
    p : int
        spline degree (=3).
    knots : list of ndarray (len=5)
        B-spline knots for each component in "gamma".

    Returns
    -------
    gamma : ndarray
        Values of "gamma". Note that scale facter is not applied.
    a : 2-d list of ndarray
        [a0[<alpha>], a1[<alpha>]] :: a[<alpha>] at transmit/received time.
        <alpha> is corresponding to <0>, <1E>, <1N>, <2E>, <2N>.
    """
    a0 = []
    a1 = []
    for k, kn in enumerate(knots):
        if len(kn) == 0:
            a0.append(0.0)
            a1.append(0.0)
            continue
        ct = mp[imp0[k] : imp0[k + 1]]
        bs = BSpline(kn, ct, inversion_params.spdeg, extrapolate=False)
        a0.append(bs(shotdat.ST.values))
        a1.append(bs(shotdat.RT.values))

    ls = 1000.0  # m/s/m to m/s/km order for gradient
    de0 = shotdat.de0.values
    de1 = shotdat.de1.values
    dn0 = shotdat.dn0.values
    dn1 = shotdat.dn1.values
    mte = shotdat.mtde.values
    mtn = shotdat.mtdn.values
    gamma0_0 =  a0[0]
    gamma0_1 = (a0[1] * de0 + a0[2] * dn0) / ls
    gamma0_2 = (a0[3] * mte + a0[4] * mtn) / ls
    gamma1_0 =  a1[0]
    gamma1_1 = (a1[1] * de1 + a1[2] * dn1) / ls
    gamma1_2 = (a1[3] * mte + a1[4] * mtn) / ls
    gamma0 = gamma0_0 + gamma0_1 + gamma0_2
    gamma1 = gamma1_0 + gamma1_1 + gamma1_2
    gamma = (gamma0 + gamma1)/2.
    a = [a0, a1]
    return gamma, a

def jacobian_pos(
        shotdat: DataFrame[ObservationData],
        svp: DataFrame[SoundVelocityProfile],
        inversion_params:InversionParams,
        mp: np.ndarray,
        mtidx: Dict[str,int],
        slvidx0: List[int],
        T0: float
        ) -> lil_matrix:

    """
    Calculate Jacobian matrix for positions.

    Parameters
    ----------
    icfg : configparser
            Config file for inversion conditions.
    mp : ndarray
            complete model parameter vector.
    slvidx0 : list
            Indices of model parameters to be solved.
    shotdat : DataFrame
            GNSS-A shot dataset.
    mtidx : dictionary
            Indices of mp for each MT.
    svp : DataFrame
            Sound speed profile.
    T0 : float
            Typical travel time.

    Returns
    -------
    jcbpos : lil_matrix
            Jacobian matrix for positions.
    """

    deltap = float(inversion_params.deltap)
    deltab = float(inversion_params.deltab)

    ndata = shotdat.index.size

    MTs = mtidx.keys()
    nMT = len(MTs)
    nmppos = len(slvidx0)

    jcbpos  = lil_matrix( (nmppos, ndata) )
    imp = 0

    gamma = shotdat.gamma.values
    logTTc = shotdat.logTTc.values
    ##################################
    ### Calc Jacobian for Position ###
    ##################################
    # for eastward
    mpj = mp.copy()
    mpj[nMT*3 + 0] += deltap
    cTTj, cTOj = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
    logTTcj = np.log( cTTj/T0 ) - gamma
    shotdat['jacob0'] = (logTTcj - logTTc) / deltap
    # for northward
    mpj = mp.copy()
    mpj[nMT*3 + 1] += deltap
    cTTj, cTOj = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
    logTTcj = np.log( cTTj/T0 ) - gamma
    shotdat['jacob1'] = (logTTcj - logTTc) / deltap
    # for upward
    mpj = mp.copy()
    mpj[nMT*3 + 2] += deltap
    cTTj, cTOj = calc_traveltime(shotdat, mpj, nMT, icfg, svp)
    logTTcj = np.log( cTTj/T0 ) - gamma
    shotdat['jacob2'] = (logTTcj - logTTc) / deltap

    ### Jacobian for each MT ###
    for mt in MTs:
        for j in range(3):
            idx = mtidx[mt] + j
            if not (idx in  slvidx0):
                continue
            jccode = "jacob%1d" % j
            shotdat['hit'] = shotdat[jccode] * (shotdat['MT'] == mt)
            jcbpos[imp,:] = np.array([shotdat.hit.values])
            imp += 1

    ### Jacobian for Center Pos ###
    for j in range(3):
        idx = nMT*3 + j
        if not (idx in  slvidx0):
            continue
        jccode = "jacob%1d" % j
        jcbpos[imp,:] = shotdat[jccode].values
        imp += 1

    ####################################
    ### Calc Jacobian for ATD offset ###
    ####################################
    for j in range(3): # j = 0:rightward, 1:forward, 2:upward
        idx = nMT*3 + 3 + j
        if not (idx in  slvidx0):
            continue
        # calc Jacobian
        mpj = mp.copy()
        mpj[(nMT+1)*3 + j] += deltap
        tmpj = shotdat.copy()

        # calc ATD offset
        calATD = np.vectorize(corr_attitude)
        pl0 = mpj[(nMT+1)*3 + 0]
        pl1 = mpj[(nMT+1)*3 + 1]
        pl2 = mpj[(nMT+1)*3 + 2]
        hd0 = shotdat.head0.values
        hd1 = shotdat.head1.values
        rl0 = shotdat.roll0.values
        rl1 = shotdat.roll1.values
        pc0 = shotdat.pitch0.values
        pc1 = shotdat.pitch1.values
        ple0, pln0, plu0 = calATD(pl0, pl1, pl2, hd0, rl0, pc0)
        ple1, pln1, plu1 = calATD(pl0, pl1, pl2, hd1, rl1, pc1)
        tmpj['ple0'] = ple0
        tmpj['pln0'] = pln0
        tmpj['plu0'] = plu0
        tmpj['ple1'] = ple1
        tmpj['pln1'] = pln1
        tmpj['plu1'] = plu1

        cTTj, cTOj = calc_traveltime(shot_data=shotdat,
                                     model_params=mp,
                                     nMT=nMT,
                                     svp=svp)
        logTTcj = np.log( cTTj/T0 ) - gamma
        jcbpos[imp,:] = (logTTcj - logTTc) / deltap
        imp += 1

    return jcbpos
