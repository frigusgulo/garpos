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
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime
# garpos module
from schemas.hyp_params import InversionParams, InversionType
from schemas.obs_data import (
    Site,
    ATDOffset,
    Transponder,
    PositionENU,
    SoundVelocityProfile,
    ObservationData,
    DataFrame,
)
from schemas.module_io import GaussianModelParameters
from setup_model_v2_2 import init_position, make_knots, derivative2, data_correlation
from forward_v2_2 import calc_forward, calc_gamma, jacobian_pos

from ray_tracer import Raytracer

# from .forward_v2 import calc_forward, calc_gamma, jacobian_pos
# from .output import outresults
# Configure logging

from output_v2_2 import output_results

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def MPestimate_v2(
    observation_data: DataFrame[ObservationData],
    sound_speed_data: DataFrame[SoundVelocityProfile],
    atd_offset: ATDOffset,
    transponders: List[Transponder],
    delta_center_position: PositionENU,
    stations: List[str],
    site_name: str,
    campaign:str,
    date_utc:datetime,
    date_jday:float,
    ref_frame:str,
    lamb0: float,
    lgrad: float,
    mu_t: float,
    mu_m: float,
    invtype: InversionType,
    knots: List[float],
    rsig: float,
    deltab: float,
    deltap: float,
    scale: float,
    maxloop: int,
    ConvCriteria: float,
    spdeg: int,
    denu: Optional[np.ndarray] = [0,0,0,0,0,0],
    **kwargs,
) -> Tuple:
    """
    Estimate model parameters using the Maximum Posteriori (MP) method.
    """
    ray_tracer = Raytracer()

    if lamb0 <= 0.0:
        response = "Lambda must be > 0"
        logger.error(response)
        sys.exit(1)

    if invtype == InversionType.positions:
        if denu is None:
            response = "Positional offset is required for InversionType.positions"
            logger.error(response)
            sys.exit(1)
        else:
            knots = [0, 0, 0]

    knotint0 = knots[0]
    knotint1 = knots[1]
    knotint2 = knots[2]

    if knotint0 + knotint1 + knotint2 <= 1.0e-4:
        invtype = InversionType.positions

    shots = observation_data
    site = site_name
    svp = sound_speed_data
    MTs = stations
    nMT = len(MTs)

    chkMT = rsig > 0.1

    ############################
    ### Set Model Parameters ###
    ############################

    mode = invtype.value

    station_dpos = {
        trns.id: trns.position_enu.get_position() + trns.position_enu.get_std_dev()
        for trns in transponders
    }
    atd_offset_params: List[float] = atd_offset.get_offset() + atd_offset.get_std_dev()
    station_array_dcnt: List[float] = (
        delta_center_position.get_position() + delta_center_position.get_std_dev()
    )
    # mp, Dipos, slvidx0, mtidx
    mppos, Dipos, slvidx0, mtidx = init_position(
        station_dpos=station_dpos,
        array_dcnt=station_array_dcnt,
        atd_offset=atd_offset,
        denu=denu,
        MTs=MTs,
    )

    if invtype.value == 1:
        Dipos = lil_matrix((0, 0))
        slvidx0 = np.array([])

    nmppos = len(slvidx0)
    cnt = np.array([mppos[imt * 3 : imt * 3 + 3] for imt in range(nMT)])
    cnt = np.mean(cnt, axis=0)

    shots["mtid"] = [mtidx[mt] for mt in shots["MT"]]

    ### Set Model Parameters for gamma ###
    knotintervals = [knotint0, knotint1, knotint1, knotint2, knotint2]
    glambda = lamb0 * lgrad
    lambdas = [lamb0] + [lamb0 * lgrad] * 4

    knots: List[np.ndarray] = make_knots(shots, spdeg, knotintervals)

    ncps = [max([0, len(kn) - spdeg - 1]) for kn in knots]

    # NCPS [LIST] - number of control points for each component of gamma

    # set pointers for model parameter vector
    imp0 = np.cumsum(np.array([len(mppos)] + ncps))

    # IMP0 - indices of model parameters for each component of gamma

    # set full model parameter vector
    mp = np.zeros(imp0[-1])
    mp[: imp0[0]] = mppos
    """
    >>> imp0
    array([ 21,  98, 175, 252, 329, 406])

    >>> ncps
    [77, 77, 77, 77, 77]

    >>> mppos
    array([-4.700500e+01,  4.086450e+02, -1.345044e+03,  4.866430e+02,
        4.812800e+01, -1.354312e+03, -2.635800e+01, -5.061430e+02,
       -1.335817e+03, -5.381190e+02, -2.274800e+01, -1.330488e+03,
        0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,
        0.000000e+00,  0.000000e+00,  1.554700e+00, -1.269000e+00,
        2.372950e+01])

    >>> len(mppos)
    21
    """
    slvidx = np.append(slvidx0, np.arange(imp0[0], imp0[-1], dtype=int))
    slvidx = slvidx.astype(int)

    H = derivative2(imp0, spdeg, knots, lambdas)

    mp0 = mp[slvidx]
    mp1 = mp0.copy()
    nmp = len(mp0)

    ### Set a priori covariance for model parameters ###
    print(nmp)
    Di = lil_matrix((nmp, nmp))
    Di[:nmppos, :nmppos] = Dipos
    Di[nmppos:, nmppos:] = H
    Di = Di.tocsc()

    rankDi = np.linalg.matrix_rank(Di.toarray(), tol=1.0e-8)
    eigvDi = np.linalg.eigh(Di.toarray())[0]
    eigvDi = eigvDi[np.where(np.abs(eigvDi.real) > 1.0e-8 / lamb0)].real
    # print(rankDi, len(eigvDi))
    if rankDi != len(eigvDi):
        # print(eigvDi)
        response = f"Error in calculating eigen value of Di !!! rankDi = {rankDi}, len(eigvDi) = {len(eigvDi)}"
        logger.error(response)
        raise ValueError(response)

    logdetDi = np.log(eigvDi).sum()

    # Initial parameters for gradient gamma
    shots["sta0_e"] = (
        mp[shots["mtid"] + 0] + mp[len(MTs) * 3 + 0]
    )  # transponder position + station center position
    shots["sta0_n"] = mp[shots["mtid"] + 1] + mp[len(MTs) * 3 + 1]
    shots["sta0_u"] = mp[shots["mtid"] + 2] + mp[len(MTs) * 3 + 2]
    shots["mtde"] = (
        shots["sta0_e"].values - cnt[0]
    )  # station center position - mean transponder position
    shots["mtdn"] = shots["sta0_n"].values - cnt[1]
    shots["de0"] = (
        shots["ant_e0"].values - shots["ant_e0"].values.mean()
    )  # Normalized antennta positions
    shots["dn0"] = shots["ant_n0"].values - shots["ant_n0"].values.mean()
    shots["de1"] = shots["ant_e1"].values - shots["ant_e1"].values.mean()
    shots["dn1"] = shots["ant_n1"].values - shots["ant_n1"].values.mean()
    shots["iniflag"] = shots["flag"].copy()

    L0 = np.array([(mp[i * 3 + 2] + mp[nMT * 3 + 2]) for i in range(nMT)]).mean()
    L0 = abs(L0 * 2.0)

    # calc depth-averaged sound speed (characteristic length/time)
    vl = svp.speed.values
    dl = svp.depth.values
    avevlyr = [(vl[i + 1] + vl[i]) * (dl[i + 1] - dl[i]) / 2.0 for i in svp.index[:-1]]
    V0 = np.array(avevlyr).sum() / (dl[-1] - dl[0])

    # calc characteristic time
    T0 = L0 / V0
    shots["logTT"] = np.log(shots.TT.values / T0)

    if invtype.value != 0:
        shots["gamma"] = 0.0

    shots = calc_forward(
        ray_tracer=ray_tracer, shots=shots, svp=svp, T0=T0, mp=mp, nMT=nMT, rsig=rsig
    )

    # Set data covariance
    icorrE = rsig < 0.1 and mu_t > 1.0e-3
    if not icorrE:
        mu_t = 0.0
    tmp = shots[~shots["flag"]].reset_index(drop=True).copy()
    ndata = len(tmp.index)
    scale = scale / T0

    TT0 = tmp.TT.values / T0
    if icorrE:
        E_factor = data_correlation(tmp, TT0, mu_t, mu_m)
        logdetEi = -E_factor.logdet()
    else:
        Ei = csc_matrix(np.diag(TT0**2.0)) / scale**2.0
        logdetEi = (np.log(TT0**2.0)).sum()

    #############################
    ### loop for Least Square ###
    #############################
    comment = ""
    iconv = 0
    loop_results = []
    for iloop in range(maxloop):

        # tmp contains unrejected data
        tmp = shots[~shots["flag"]].reset_index(drop=True).copy()
        ndata = len(tmp.index)

        ############################
        ### Calc Jacobian matrix ###
        ############################

        # Set array for Jacobian matrix
        if rsig > 0.1 or iloop == 0:
            jcb = lil_matrix((nmp, ndata))

        # Calc Jacobian for gamma
        if invtype.value != 0 and (rsig > 0.1 or iloop == 0):
            mpj = np.zeros(imp0[5])
            imp = nmppos

            for impsv in range(imp0[0], imp0[-1]):
                mpj[impsv] = 1.0
                gamma, a = calc_gamma(mpj, tmp, imp0, spdeg, knots)

                jcb[imp, :] = -gamma * scale
                imp += 1
                mpj[impsv] = 0.0

        # Calc Jacobian for position
        if invtype.value != 1:
            jcb0 = jacobian_pos(
                slvidx0=slvidx0,
                shotdat=tmp,
                ray_tracer=ray_tracer,
                deltab=deltab,
                deltap=deltap,
                mp=mp,
                mtidx=mtidx,
                svp=svp,
                T0=T0,
            )
            jcb[:nmppos, :] = jcb0[:nmppos, :]

        jcb = jcb.tocsc()

        ############################
        ### CALC model parameter ###
        ############################
        alpha = 1.0  # fixed
        if icorrE:
            LiAk = E_factor.solve_L(jcb.T.tocsc(), use_LDLt_decomposition=False)
            AktEiAk = LiAk.T @ LiAk / scale**2.0
            rk = jcb @ E_factor(tmp.ResiTT.values) / scale**2.0 + Di @ (mp0 - mp1)
        else:
            AktEi = jcb @ Ei
            AktEiAk = AktEi @ jcb.T
            rk = AktEi @ tmp.ResiTT.values + Di @ (mp0 - mp1)

        Cki = AktEiAk + Di
        Cki_factor = cholesky(Cki.tocsc(), ordering_method="natural")
        Ckrk = Cki_factor(rk)
        dmp = alpha * Ckrk

        dxmax = max(abs(dmp[:]))
        if invtype.value == 1 and rsig <= 0.1:
            dposmax = 0.0  # no loop needed in invtyp = 1
        elif invtype.value == 1 and rsig > 0.1:
            dposmax = ConvCriteria / 200.0
        else:
            dposmax = max(abs(dmp[:nmppos]))
            if dxmax > 10.0:
                alpha = 10.0 / dxmax
                dmp = alpha * dmp
                dxmax = max(abs(dmp[:]))

        mp1 += dmp  # update mp1 (=x(k+1))
        for j in range(len(mp1)):
            mp[slvidx[j]] = mp1[j]

        ####################
        ### CALC Forward ###
        ####################
        if invtype.value != 0:
            gamma, a = calc_gamma(mp, shots, imp0, spdeg, knots)
            shots["gamma"] = gamma * scale
            av = np.array(a) * scale * V0
        else:
            av = 0.0  # dummy
        shots["dV"] = shots.gamma * V0

        shots = calc_forward(
            ray_tracer=ray_tracer,
            shots=shots,
            svp=svp,
            T0=T0,
            mp=mp,
            nMT=nMT,
            rsig=rsig,
        )

        # for mis-response in MT number (e.g., TU sites) verification
        if chkMT and iconv >= 1:
            print("Check MT number for shots named 'M00'")
            comment += "Check MT number for shots named 'M00'\n"
            rsigm0 = 1.0
            aveRTT = shots[~shots["flag"]].ResiTT.mean()
            sigRTT = shots[~shots["flag"]].ResiTT.std()
            th0 = aveRTT + rsigm0 * sigRTT
            th1 = aveRTT - rsigm0 * sigRTT
            shots.loc[(shots.m0flag), ["flag"]] = (shots["ResiTT"] > th0) | (
                shots["ResiTT"] < th1
            )
            aveRTT1 = shots[~shots["flag"]].ResiTT.mean()
            sigRTT1 = shots[~shots["flag"]].ResiTT.std()

        tmp = shots[~shots["flag"]].reset_index(drop=True).copy()
        ndata = len(tmp.index)

        TT0 = tmp.TT.values / T0
        if rsig > 0.1:
            if icorrE:
                E_factor = data_correlation(tmp, TT0, mu_t, mu_m)
                logdetEi = -E_factor.logdet()
            else:
                Ei = csc_matrix(np.diag(TT0**2.0)) / scale**2.0
                logdetEi = (np.log(TT0**2.0)).sum()

        rttadp = tmp.ResiTT.values

        if icorrE:
            misfit = rttadp @ E_factor(rttadp) / scale**2.0
        else:
            rttvec = csr_matrix(np.array([rttadp]))
            misfit = ((rttvec @ Ei) @ rttvec.T)[0, 0]

        # Calc Model-parameters' RMSs
        rms = lambda d: np.sqrt((d**2.0).sum() / d.size)
        mprms = rms(dmp)
        rkrms = rms(rk)
        datarms = rms(tmp.ResiTTreal.values)

        aved = np.array([(mp[i * 3 + 2] + mp[nMT * 3 + 2]) for i in range(nMT)]).mean()
        reject = shots[shots["flag"]].index.size
        ratio = 100.0 - float(reject) / float(len(shots.index)) * 100.0

        ##################
        ### Check Conv ###
        ##################
        loopres = {
            "mode": mode,
            "loop": iloop + 1,
            "rms": datarms * 1000.0,
            "used_shot": ratio,
            "reject": reject,
            "dxmax": dxmax,
            "aved": aved,
        }
        loop_results.append(loopres)

        if (
            dxmax < ConvCriteria / 100.0 or dposmax < ConvCriteria / 1000.0
        ) and not chkMT:
            break
        elif dxmax < ConvCriteria:
            iconv += 1
            if iconv == 2:
                break
        else:
            iconv = 0

    #######################
    # calc ABIC and sigma #
    #######################
    dof = float(ndata + rankDi - nmp)
    S = misfit + ((mp0 - mp1) @ Di) @ (mp0 - mp1)

    logdetCki = Cki_factor.logdet()

    abic = dof * math.log(S) - logdetEi - logdetDi + logdetCki
    sigobs = (S / dof) ** 0.5 * scale

    Ck = Cki_factor(identity(nmp).tocsc())
    C = S / dof * Ck.toarray()
    rmsmisfit = (misfit / ndata) ** 0.5 * sigobs

    abic_results = {
        "ABIC": abic, # akaike bayesian information criterion
        "rmsmisfit": rmsmisfit * 1000.0,
        "sigobs": sigobs,
        "conv_criteria": rsig,
        "mu_t": mu_t, # Correlation length (in sec.)
        "mu_m": mu_m, # Ratio of correlation between the different transponders.
        "lamb0": lamb0, 
        "lgrad": lgrad,
        "deltab": deltab,
        "deltap": deltap,
        "scale": scale, # time scale factor
        "maxloop": maxloop,
    }

    site_data_results = output_results(
        site_name=site_name,
        campaign=campaign,
        date_utc=date_utc,
        date_jday=date_jday,
        ref_frame=ref_frame,
        latitude=0.0,
        longitude=0.0,
        height=0.0,
        imp0=imp0,
        slvidx0=slvidx0,
        C=C,
        mp=mp,
        MTs=MTs,
        shots=shots,
        svp=svp,
        mtidx=mtidx,
    )

    return (site_data_results, abic_results, loop_results)
