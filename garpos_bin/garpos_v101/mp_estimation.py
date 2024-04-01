"""
Created:
	07/01/2020 by S. Watanabe
Modified:
	02/01/2021 by S. Watanabe
		to fix unknown MT number flag "M00" for mis-response seen in TU's sites.
	01/07/2022 by S. Watanabe
		to set B-spline's knots by time interval (also need to change "Setup.ini" file)
		to use cholesky decomposition for calc. inverse
	03/30/2022 by S. Watanabe and Y. Nakamura
		to adjust the threshold for rank calculation
"""
import os
import sys
import math
import configparser
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, linalg, identity
from sksparse.cholmod import cholesky
import pandas as pd
import pdb
from typing import List
# garpos module
from .setup_model import init_position, make_knots, derivative2, data_correlation
from .forward import calc_forward, calc_gamma, jacobian_pos
from .output import outresults


def MPestimate(cfgf, icfgf, odir, suf, lamb0, lgrad, mu_t, mu_m, denu):
    """
	Run the model parameter estimation. (under given hyperparameter)

	Parameters
	----------
	cfgf : string
		Path to the site-parameter file.
	icfgf : stirng
		Path to the analysis-setting file.
	odir : string
		Directory to store the results.
	suf : string
		Suffix to be added for result files.
	lamb0 : float
		Hyperparameter (Lambda_0)^2.
		Controls the smoothness for gamma.
	lgrad : float
		Hyperparameter (Lambda_g/Lambda_0)^2.
		Controls the smoothness for gradient-components of gamma.
	mu_t : float
		Hyperparameter (mu_t).
		Controls the correlation length for acoustic data.
	mu_m : float
		Hyperparameter (mu_MT).
		Controls the inter-transponder data correlation.
	denu : ndarray (len=3)
		Positional offset (only applicable in case invtyp = 1).

	Returns
	-------
	resf : string
		Result site-parameter file name (min-ABIC model).
	datarms : float
		RMS for "real" travel time (NOTE: not in log).
	abic : float
		ABIC value.
	dcpos : ndarray
		Positional difference of array center and its variances.
	"""

    spdeg = 3
    np.set_printoptions(threshold=np.inf)

    if lamb0 <= 0.:
        print("Lambda must be > 0")
        sys.exit(1)

    ################################
    ### Set Inversion Parameters ###
    ################################
    icfg = configparser.ConfigParser()
    icfg.read(icfgf, 'UTF-8')
    invtyp = int(icfg.get("Inv-parameter","inversiontype"))
    knotint0 = float(icfg.get("Inv-parameter","knotint0"))*60.
    knotint1 = float(icfg.get("Inv-parameter","knotint1"))*60.
    knotint2 = float(icfg.get("Inv-parameter","knotint2"))*60.
    rsig = float(icfg.get("Inv-parameter","RejectCriteria"))
    scale = float(icfg.get("Inv-parameter","traveltimescale"))
    maxloop = int(icfg.get("Inv-parameter","maxloop"))
    ConvCriteria = float(icfg.get("Inv-parameter","ConvCriteria"))

    if invtyp == 0:
        knotint0 = 0.
        knotint1 = 0.
        knotint2 = 0.
    if knotint0+knotint1+knotint2 <= 1.e-4:
        invtyp = 0

    #############################
    ### Set Config Parameters ###
    #############################
    cfg = configparser.ConfigParser()
    cfg.read(cfgf, 'UTF-8')

    ### Read obs file ###
    obsfile  = cfg.get("Data-file", "datacsv")
    """
	>> obsfile
	'/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/obsdata/SAGA/SAGA.1903.kaiyo_k4-obs.csv'
	"""
    shots = pd.read_csv(obsfile, comment='#', index_col=0)

    # check NaN in shotdata
    shots = shots[~shots.isnull().any(axis=1)].reset_index(drop=True)
    # check TT > 0 in shotdata
    shots = shots[~(shots.TT <= 0.)].reset_index(drop=True)

    ### Sound speed profile ###
    svpf = cfg.get("Obs-parameter", "SoundSpeed")
    svp = pd.read_csv(svpf, comment='#')
    site = cfg.get("Obs-parameter", "Site_name")

    ### IDs of existing transponder ###
    """
	>> site
	'SAGA'
    
	>> svp.columns
	'/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/obsdata/SAGA/SAGA.1903.kaiyo_k4-obs.csv'

	>> MTs
	['M11', 'M12', 'M13', 'M14']
	"""
    MTs = cfg.get("Site-parameter", "Stations").split()
    MTs = [ str(mt) for mt in MTs ]
    nMT = len(MTs)
    M00 = shots[ shots.MT == "M00" ].reset_index(drop=True)
    shots = shots[ shots.MT.isin(MTs) ].reset_index(drop=True)

    """
		>>shots.columns
		Index(['SET', 'LN', 'MT', 'TT', 'ResiTT', 'TakeOff', 'gamma', 'flag', 'ST',
			'ant_e0', 'ant_n0', 'ant_u0', 'head0', 'pitch0', 'roll0', 'RT',
			'ant_e1', 'ant_n1', 'ant_u1', 'head1', 'pitch1', 'roll1'],
			dtype='object')
	"""

    # for mis-response in MT number (e.g., TU sites) verification
    shots["m0flag"] = False
    M00["m0flag"] = True
    M00["flag"] = True
    for mt in MTs:
        addshots = M00.copy()
        addshots['MT'] = mt
        shots = pd.concat([shots,addshots])
    shots = shots.reset_index()
    chkMT = rsig > 0.1 and len(M00) >= 1

    ############################
    ### Set Model Parameters ###
    ############################
    mode = "Inversion-type %1d" % invtyp

    # >> denu
    # array([0., 0., 0.])

    # >> MTs
    # ['M11', 'M12', 'M13', 'M14']

    mppos, Dipos, slvidx0, mtidx = init_position(cfg, denu, MTs)

    # MPPOS - model parameter position
    # Dipos - a priori covariance for model parameters
    # slvidx0 - indices of model parameters to be solved
    # mtidx - indices of mp for each transponder
    """
	>> mppos
	array([-4.700500e+01,  4.086450e+02, -1.345044e+03, # station position
	      4.866430e+02,4.812800e+01, -1.354312e+03, #station position
		 -2.635800e+01, -5.061430e+02, -1.335817e+03, # station position
		 -5.381190e+02, -2.274800e+01, -1.330488e+03, # station position
		  0.000000e+00,  0.000000e+00,  0.000000e+00, # center position
		  1.554700e+00,-1.269000e+00,  2.372950e+01]) # ATD offset

	>> Dipos
	<12x12 sparse matrix of type '<class 'numpy.float64'>'
	with 12 stored elements in Compressed Sparse Column format>

	>> slvidx0
	array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

	>> mtidx
	{'M11': 0, 'M12': 3, 'M13': 6, 'M14': 9}

	"""
    # in case where positions should not be solved
    if invtyp == 1:
        Dipos = lil_matrix((0, 0))
        slvidx0 = np.array([])

    nmppos = len(slvidx0)

    # nmppos - number of position model parameters to be solved

    cnt = np.array([ mppos[imt*3:imt*3+3] for imt in range(nMT)])

    # cnt - array of transponder positions
    cnt = np.mean(cnt, axis=0)
    """
	>> cnt
	array([  -31.20975,   -18.0295 , -1341.41525])

	>> mppos

	array([-4.700500e+01,  4.086450e+02, -1.345044e+03,  4.866430e+02,
			4.812800e+01, -1.354312e+03, -2.635800e+01, -5.061430e+02,
		-1.335817e+03, -5.381190e+02, -2.274800e+01, -1.330488e+03,
			0.000000e+00,  0.000000e+00,  0.000000e+00,  1.554700e+00,
		-1.269000e+00,  2.372950e+01])
		

		
	"""
    # cnt - mean position of transponders (i.e. station center?)

    # MT index for model parameter
    shots['mtid'] = [ mtidx[mt] for mt in shots['MT'] ]

    ### Set Model Parameters for gamma ###
    knotintervals = [knotint0, knotint1, knotint1, knotint2, knotint2]
    glambda = lamb0 * lgrad
    lambdas = [lamb0] +[lamb0 * lgrad]*4

    """
	>> shots.columns
	Index(['index', 'SET', 'LN', 'MT', 'TT', 'ResiTT', 'TakeOff', 'gamma', 'flag',
		'ST', 'ant_e0', 'ant_n0', 'ant_u0', 'head0', 'pitch0', 'roll0', 'RT',
		'ant_e1', 'ant_n1', 'ant_u1', 'head1', 'pitch1', 'roll1', 'm0flag',
		'mtid'],
		dtype='object')
		
	>> spdeg
	3

	>> knotintervals
	[300.0, 300.0, 300.0, 300.0, 300.0]
	"""
    knots: List[np.ndarray] = make_knots(shots, spdeg, knotintervals)

    """
	>> knots
	[array([29162.14853703, 29465.56406635, 29768.97959568, 30072.395125  ,
		30375.8...2, 53131.97535365,
		53435.39088297]), array([29162.14853703, 29465.56406635, 29768.97959568, 30072.395125  ,
		30375.8...2, 53131.97535365,
		53435.39088297]), array([29162.14853703, 29465.56406635, 29768.97959568, 30072.395125  ,
		30375.8...2, 53131.97535365,
		53435.39088297]), array([29162.14853703, 29465.56406635, 29768.97959568, 30072.395125  ,
		30375.8...2, 53131.97535365,
		53435.39088297]), array([29162.14853703, 29465.56406635, 29768.97959568, 30072.395125  ,
		30375.8...2, 53131.97535365,
		53435.39088297])]

	"""
    ncps = [ max([0, len(kn)-spdeg-1]) for kn in knots]

    # NCPS [LIST] - number of control points for each component of gamma

    # set pointers for model parameter vector
    imp0 = np.cumsum(np.array([len(mppos)] + ncps))

    # IMP0 - indices of model parameters for each component of gamma

    # set full model parameter vector
    mp = np.zeros(imp0[-1])
    mp[:imp0[0]] = mppos

    slvidx = np.append(slvidx0, np.arange(imp0[0],imp0[-1],dtype=int))
    slvidx = slvidx.astype(int)
    """
	>> ncps
	[77, 77, 77, 77, 77]

	>> imp0
	array([ 18,  95, 172, 249, 326, 403])
    
    >> slvidx
    array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  18,
        19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
        45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
        58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
        71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
        97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,....,402
	
        
	"""
    H = derivative2(imp0, spdeg, knots, lambdas)

    ### set model parameter to be estimated ###
    # slvidx is used to filter out ATD offset and Center Position
    mp0 = mp[slvidx]
    mp1 = mp0.copy()
    nmp = len(mp0)

    ### Set a priori covariance for model parameters ###
    print(nmp)
    Di = lil_matrix( (nmp, nmp) )
    Di[:nmppos,:nmppos] = Dipos
    Di[nmppos:,nmppos:] = H
    Di = Di.tocsc()

    rankDi = np.linalg.matrix_rank(Di.toarray(), tol=1.e-8)
    eigvDi = np.linalg.eigh(Di.toarray())[0]
    eigvDi = eigvDi[ np.where(np.abs(eigvDi.real) > 1.e-8/lamb0)].real
    # print(rankDi, len(eigvDi))
    if rankDi != len(eigvDi):
        # print(eigvDi)
        print(np.linalg.matrix_rank(Di), len(eigvDi))
        print("Error in calculating eigen value of Di !!!")
        sys.exit(1)
    logdetDi = np.log(eigvDi).sum()

    # Initial parameters for gradient gamma
    shots['sta0_e'] = mp[shots['mtid']+0] + mp[len(MTs)*3+0] # transponder position + station center position
    shots['sta0_n'] = mp[shots['mtid']+1] + mp[len(MTs)*3+1]
    shots['sta0_u'] = mp[shots['mtid']+2] + mp[len(MTs)*3+2]
    shots['mtde'] = (shots['sta0_e'].values - cnt[0]) # station center position - mean transponder position
    shots['mtdn'] = (shots['sta0_n'].values - cnt[1])
    shots['de0'] = shots['ant_e0'].values - shots['ant_e0'].values.mean() # Normalized antennta positions
    shots['dn0'] = shots['ant_n0'].values - shots['ant_n0'].values.mean()
    shots['de1'] = shots['ant_e1'].values - shots['ant_e1'].values.mean()
    shots['dn1'] = shots['ant_n1'].values - shots['ant_n1'].values.mean()
    shots['iniflag'] = shots['flag'].copy()

    #####################################
    # Set log(TT/T0) and initial values #
    #####################################
    # calc average depth*2 (characteristic length)
    L0 = np.array([(mp[i*3+2] + mp[nMT*3+2]) for i in range(nMT)]).mean()
    L0 = abs(L0 * 2.)

    # calc depth-averaged sound speed (characteristic length/time)
    vl = svp.speed.values
    dl = svp.depth.values
    avevlyr = [ (vl[i+1]+vl[i])*(dl[i+1]-dl[i])/2. for i in svp.index[:-1]]
    V0 = np.array(avevlyr).sum()/(dl[-1]-dl[0])

    # calc characteristic time
    T0 = L0 / V0
    shots["logTT"] = np.log(shots.TT.values/T0)

    ######################
    ## data correlation ##
    ######################
    # Calc initial ResiTT
    if invtyp != 0:
        shots["gamma"] = 0.

	# >>> shots
   
	#       index  SET   LN   MT        TT  ResiTT  TakeOff  gamma  ...       mtde      mtdn       de0          dn0        de1          dn1  iniflag     logTT
	# 0         0  S01  L01  M11  2.289306     0.0      0.0    0.0  ...  -15.79525  426.6745   1.76721  1482.570718   2.981088  1471.462113    False  0.239196
	# 1         1  S01  L01  M13  3.126690     0.0      0.0    0.0  ...    4.85175 -488.1135   7.53716  1422.313778   7.339668  1411.223463    False  0.550922
	# 2         2  S01  L01  M14  2.702555     0.0      0.0    0.0  ... -506.90925   -4.7185   7.36635  1419.303338   7.306488  1409.409173    False  0.405145
	# 3         3  S01  L01  M14  2.681070     0.0      0.0    0.0  ... -506.90925   -4.7185   6.36498  1396.816408   5.724938  1387.906783    False  0.397163
	# 4         4  S01  L01  M11  2.218846     0.0      0.0    0.0  ...  -15.79525  426.6745   6.04311  1394.158908   5.643268  1386.536913    False  0.207934

    shots = calc_forward(shots, mp, nMT, icfg, svp, T0)

    # Set data covariance
    icorrE = rsig < 0.1 and mu_t > 1.e-3
    if not icorrE:
        mu_t = 0.
    tmp = shots[~shots['flag']].reset_index(drop=True).copy()
    ndata = len(tmp.index)
    scale = scale/T0

    TT0 = tmp.TT.values / T0
    if icorrE:
        E_factor = data_correlation(tmp, TT0, mu_t, mu_m)
        logdetEi = -E_factor.logdet()
    else:
        Ei = csc_matrix( np.diag(TT0**2.) )/scale**2.
        logdetEi = (np.log(TT0**2.)).sum()

    #############################
    ### loop for Least Square ###
    #############################
    comment = ""
    iconv = 0
    for iloop in range(maxloop):

        # tmp contains unrejected data
        tmp = shots[~shots['flag']].reset_index(drop=True).copy()
        ndata = len(tmp.index)

        ############################
        ### Calc Jacobian matrix ###
        ############################

        # Set array for Jacobian matrix
        if rsig > 0.1 or iloop == 0:
            jcb = lil_matrix( (nmp, ndata) )

        # Calc Jacobian for gamma
        if invtyp != 0 and (rsig > 0.1 or iloop == 0):
            mpj = np.zeros(imp0[5])
            imp = nmppos

            for impsv in range(imp0[0],imp0[-1]):
                mpj[impsv] = 1.
                gamma, a = calc_gamma(mpj, tmp, imp0, spdeg, knots)

                jcb[imp,:] = -gamma*scale
                imp += 1
                mpj[impsv] = 0.

        # Calc Jacobian for position
        if invtyp != 1:
            jcb0 = jacobian_pos(icfg, mp, slvidx0, tmp, mtidx, svp, T0)
            jcb[:nmppos, :] = jcb0[:nmppos, :]

        jcb = jcb.tocsc()

        ############################
        ### CALC model parameter ###
        ############################
        alpha = 1.0 # fixed
        if icorrE:
            LiAk = E_factor.solve_L(jcb.T.tocsc(), use_LDLt_decomposition=False)
            AktEiAk = LiAk.T @ LiAk / scale**2.
            rk = jcb @ E_factor(tmp.ResiTT.values) / scale**2. + Di @ (mp0-mp1)
        else:
            AktEi = jcb @ Ei
            AktEiAk = AktEi @ jcb.T
            rk  = AktEi @ tmp.ResiTT.values + Di @ (mp0-mp1)

        Cki = AktEiAk + Di
        Cki_factor = cholesky(Cki.tocsc(), ordering_method="natural")
        Ckrk = Cki_factor(rk)
        dmp  = alpha * Ckrk

        dxmax = max(abs(dmp[:]))
        if invtyp == 1 and rsig <= 0.1:
            dposmax = 0. # no loop needed in invtyp = 1
        elif invtyp == 1 and rsig > 0.1:
            dposmax = ConvCriteria/200.
        else:
            dposmax = max(abs(dmp[:nmppos]))
            if dxmax > 10.:
                alpha = 10./dxmax
                dmp = alpha * dmp
                dxmax = max(abs(dmp[:]))

        mp1 += dmp # update mp1 (=x(k+1))
        for j in range(len(mp1)):
            mp[slvidx[j]] = mp1[j]

        ####################
        ### CALC Forward ###
        ####################
        if invtyp != 0:
            gamma, a  = calc_gamma(mp, shots, imp0, spdeg, knots)
            shots["gamma"] = gamma * scale
            av = np.array(a) * scale * V0
        else:
            av = 0. # dummy
        shots['dV'] = shots.gamma * V0

        shots = calc_forward(shots, mp, nMT, icfg, svp, T0)

        # for mis-response in MT number (e.g., TU sites) verification
        if chkMT and iconv >= 1:
            print("Check MT number for shots named 'M00'")
            comment += "Check MT number for shots named 'M00'\n"
            rsigm0 = 1.0
            aveRTT = shots[~shots['flag']].ResiTT.mean()
            sigRTT = shots[~shots['flag']].ResiTT.std()
            th0 = aveRTT + rsigm0 * sigRTT
            th1 = aveRTT - rsigm0 * sigRTT
            shots.loc[ (shots.m0flag), ['flag']] = ((shots['ResiTT'] > th0) | (shots['ResiTT'] < th1))
            aveRTT1 = shots[~shots['flag']].ResiTT.mean()
            sigRTT1 = shots[~shots['flag']].ResiTT.std()

        tmp = shots[~shots['flag']].reset_index(drop=True).copy()
        ndata = len(tmp.index)

        TT0 = tmp.TT.values / T0
        if rsig > 0.1:
            if icorrE:
                E_factor = data_correlation(tmp, TT0, mu_t, mu_m)
                logdetEi = -E_factor.logdet()
            else:
                Ei = csc_matrix( np.diag(TT0**2.) )/scale**2.
                logdetEi = (np.log(TT0**2.)).sum()

        rttadp = tmp.ResiTT.values

        if icorrE:
            misfit = rttadp @ E_factor(rttadp) / scale**2.
        else:
            rttvec = csr_matrix( np.array([rttadp]) )
            misfit = ((rttvec @ Ei) @ rttvec.T)[0,0]

        # Calc Model-parameters' RMSs
        rms = lambda d: np.sqrt((d ** 2.).sum() / d.size)
        mprms   = rms( dmp )
        rkrms   = rms( rk  )
        datarms = rms(tmp.ResiTTreal.values)

        aved = np.array([(mp[i*3+2] + mp[nMT*3+2]) for i in range(nMT)]).mean()
        reject = shots[shots['flag']].index.size
        ratio  = 100. - float(reject)/float(len(shots.index))*100.

        ##################
        ### Check Conv ###
        ##################
        loopres  = "%s Loop %2d-%2d, " % (mode, 1, iloop+1)
        loopres += "RMS(TT) = %10.6f ms, " % (datarms*1000.)
        loopres += "used_shot = %5.1f%%, reject = %4d, " % (ratio, reject)
        loopres += "Max(dX) = %10.4f, Hgt = %10.3f" % (dxmax, aved)
        print(loopres)
        comment += "#"+loopres+"\n"

        if (dxmax < ConvCriteria/100. or dposmax < ConvCriteria/1000.) and not chkMT:
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
    S = misfit + ( (mp0-mp1) @ Di ) @ (mp0-mp1)

    logdetCki = Cki_factor.logdet()

    abic = dof * math.log(S) - logdetEi - logdetDi + logdetCki
    sigobs  = (S/dof)**0.5 * scale

    Ck = Cki_factor(identity(nmp).tocsc())
    C = S/dof * Ck.toarray()
    rmsmisfit = (misfit/ndata) **0.5 * sigobs

    finalres  = " ABIC = %18.6f " % abic
    finalres += " misfit = % 6.3f " % (rmsmisfit*1000.)
    finalres += suf
    print(finalres)
    comment += "# " + finalres + "\n"

    comment += "# lambda_0^2 = %12.8f\n" % lamb0
    comment += "# lambda_g^2 = %12.8f\n" % (lamb0 * lgrad)
    comment += "# mu_t = %12.8f sec.\n" % mu_t
    comment += "# mu_MT = %5.4f\n" % mu_m

    #####################
    # Write Result data #
    #####################

    resf, dcpos = outresults(odir, suf, cfg, invtyp, imp0, slvidx0,
								C, mp, shots, comment, MTs, mtidx, av)

    return [resf, datarms, abic, dcpos]
