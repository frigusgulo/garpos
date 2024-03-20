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


def MPestimate(site_data: Site, inversion_params: InversionParams, suf: str):

    init_position_output: Tuple[List[float], np.ndarray, List[int], Dict[str, int]] = (
        init_position(site_data, inversion_params)
    )

    chkMT = False

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

    model_parameter_init = model_parameter.copy()

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

    #############################
    ### loop for Least Square ###
    #############################
    comment = ""
    iconv = 0
    for iloop in range(inversion_params.maxloop):

        tmp = (
            site_data.shot_data[~site_data.shot_data["flag"]]
            .reset_index(drop=True)
            .copy()
        )
        ndata = len(tmp.index)

        ############################
        ### Calc Jacobian matrix ###
        ############################

        # Set array for Jacobian matrix
        if inversion_params.rejectcriteria > 0.1 or iloop == 0:
            jcb = lil_matrix((slvidx.shape[0], ndata))

        # Calc Jacobian for gamma
        if inversion_params.inversiontype != InversionType.positions and (
            inversion_params.rejectcriteria > 0.1 or iloop == 0
        ):
            mpj = np.zeros(model_parameter_pointer[5])
            imp = len(slvidx0)

            for impsv in range(model_parameter_pointer[0], model_parameter_pointer[-1]):
                mpj[impsv] = 1.0
                gamma, a = calc_gamma(
                    mpj,
                    tmp,
                    model_parameter_pointer,
                    inversion_params.spline_degree,
                    knots,
                )

                jcb[imp, :] = -gamma * scale
                imp += 1
                mpj[impsv] = 0.0

        if inversion_params.inversiontype != InversionType.gammas:
            jcb0 = jacobian_pos(
                shotdat=site_data.shot_data,
                svp=site_data.sound_speed_data,
                inversion_params=inversion_params,
                mp=model_parameter,
                mtidx=transponder_idx,
                T0=T0,
                slvidx0=slvidx,
            )

            jcb[: len(slvidx0), :] = jcb0[: len(slvidx0), :]

        jcb = jcb.tocsc()

        ############################
        ### CALC model parameter ###
        ############################
        alpha = 1.0  # fixed
        if icorrE:
            LiAk = E_factor.solve_L(jcb.T.tocsc(), use_LDLt_decomposition=False)
            AktEiAk = LiAk.T @ LiAk / scale**2.0
            rk = jcb @ E_factor(tmp.ResiTT.values) / scale**2.0 + Di @ (
                model_parameter_init - model_parameter
            )  # need to fix at some point
        else:
            AktEi = jcb @ Ei
            AktEiAk = AktEi @ jcb.T
            rk = AktEi @ tmp.ResiTT.values + Di @ (
                model_parameter_init - model_parameter
            )

        Cki = AktEiAk + Di
        Cki_factor = cholesky(Cki.tocsc(), ordering_method="natural")
        Ckrk = Cki_factor(rk)
        dmp = alpha * Ckrk

        dxmax = max(abs(dmp[:]))
        if (
            inversion_params.inversiontype == InversionParams.gammas
            and inversion_params.rejectcriteria <= 0.1
        ):
            dposmax = 0.0  # no loop needed in invtyp = 1
        elif (
            inversion_params.inversiontype == InversionParams.gammas
            and inversion_params.rejectcriteria > 0.1
        ):
            dposmax = inversion_params.inversion_params.convcriteria / 200.0
        else:
            dposmax = max(abs(dmp[: len(slvidx0)]))
            if dxmax > 10.0:
                alpha = 10.0 / dxmax
                dmp = alpha * dmp
                dxmax = max(abs(dmp[:]))

        model_parameter += dmp  # update model_parameter (=x(k+1))
        for j in range(len(model_parameter)):
            model_parameter[slvidx[j]] = model_parameter[j]

        ####################
        ### CALC Forward ###
        ####################
        if inversion_params.inversiontype != InversionType.positions:
            gamma, a = calc_gamma(
                model_parameter,
                site_data.shot_data,
                model_parameter_pointer,
                inversion_params.spdeg,
                knots,
            )
            site_data.shot_data["gamma"] = gamma * scale
            av = np.array(a) * scale * V0
        else:
            av = 0.0  # dummy
        site_data.shot_data["dV"] = site_data.shot_data.gamma.values * V0

        site_data.shot_data = calc_forward(
            shot_data=site_data.shot_data,
            sound_velocity_profile=site_data.sound_speed_data,
            T0=T0,
            model_params=model_parameter,
            n_transponders=number_transponders,
            inversion_params=inversion_params,
        )

        # calc_forward(site_data.shot_data, mp, number_transponders, icfg, svp, T0)

        # for mis-response in MT number (e.g., TU sites) verification
        # if chkMT and iconv >= 1:
        #     print("Check MT number for shots named 'M00'")
        #     comment += "Check MT number for shots named 'M00'\n"
        #     rsigm0 = 1.0
        #     aveRTT = shots[~shots["flag"]].ResiTT.mean()
        #     sigRTT = shots[~shots["flag"]].ResiTT.std()
        #     th0 = aveRTT + rsigm0 * sigRTT
        #     th1 = aveRTT - rsigm0 * sigRTT
        #     shots.loc[(shots.m0flag), ["flag"]] = (shots["ResiTT"] > th0) | (
        #         shots["ResiTT"] < th1
        #     )
        #     aveRTT1 = shots[~shots["flag"]].ResiTT.mean()
        #     sigRTT1 = shots[~shots["flag"]].ResiTT.std()

        tmp = (
            site_data.shot_data[~site_data.shot_data["flag"]]
            .reset_index(drop=True)
            .copy()
        )
        ndata = len(tmp.index)

        TT0 = tmp.TT.values / T0
        if inversion_params.rejectcriteria > 0.1:
            if icorrE:
                E_factor = data_correlation(
                    tmp, TT0, inversion_params.mu_t, inversion_params.mu_mt
                )
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

        aved = np.array(
            [
                (
                    model_parameter[i * 3 + 2]
                    + model_parameter[number_transponders * 3 + 2]
                )
                for i in range(number_transponders)
            ]
        ).mean()
        reject = site_data.shot_data[site_data.shot_data["flag"]].index.size
        ratio = 100.0 - float(reject) / float(len(site_data.shot_data.index)) * 100.0

        ##################number_transponders
        ### Check Conv ###
        ##################

        mode = "Inversion-type %1d" % inversion_params.inversiontype

        loopres = "%s Loop %2d-%2d, " % (mode, 1, iloop + 1)
        loopres += "RMS(TT) = %10.6f ms, " % (datarms * 1000.0)
        loopres += "used_shot = %5.1f%%, reject = %4d, " % (ratio, reject)
        loopres += "Max(dX) = %10.4f, Hgt = %10.3f" % (dxmax, aved)
        print(loopres)
        comment += "#" + loopres + "\n"

        if (
            dxmax < inversion_params.convcriteria / 100.0
            or dposmax < inversion_params.convcriteria / 1000.0
        ) and not chkMT:
            break
        elif dxmax < inversion_params.convcriteria:
            iconv += 1
            if iconv == 2:
                break
        else:
            iconv = 0

        #######################
        # calc ABIC and sigma #
        #######################
        dof = float(ndata + rank_Di - model_parameter.shape[0])
        S = misfit + ((model_parameter_init - model_parameter) @ Di) @ (
            model_parameter_init - model_parameter
        )

        logdetCki = Cki_factor.logdet()

        abic = dof * math.log(S) - logdetEi - log_det_Di + logdetCki
        sigobs = (S / dof) ** 0.5 * scale

        Ck = Cki_factor(identity(slvidx.shape[0]).tocsc())
        C = S / dof * Ck.toarray()
        rmsmisfit = (misfit / ndata) ** 0.5 * sigobs

        finalres = " ABIC = %18.6f " % abic
        finalres += " misfit = % 6.3f " % (rmsmisfit * 1000.0)
        finalres += suf
        print(finalres)
        comment += "# " + finalres + "\n"

        comment += "# lambda_0^2 = %12.8f\n" % inversion_params.log_lambda * 10
        comment += "# lambda_g^2 = %12.8f\n" % (
            inversion_params.log_lambda * 10 * inversion_params.log_gradlambda * 10
        )
        comment += "# mu_t = %12.8f sec.\n" % inversion_params.mu_t
        comment += "# mu_MT = %5.4f\n" % inversion_params.mu_mt


        # TODO implement outresults


        
        #####################
        # Write Result data #
        #####################

        # resf, dcpos = outresults(
        #     odir, suf, cfg, invtyp, imp0, slvidx0, C, mp, shots, comment, MTs, mtidx, av
        # )

        # return [resf, datarms, abic, dcpos]
