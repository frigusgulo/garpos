import os
import numpy as np
from typing import List,Optional
from pandas import DataFrame
from pydantic import BaseModel
from datetime import datetime
from .schemas import InversionType,SoundVelocityProfile,ObservationData

class Point(BaseModel):
    value: float
    sigma: Optional[float] = 0.0


class PositionENU(BaseModel):
    east: Point
    north: Point
    up: Point
    cov_nu: Optional[float] = 0.0
    cov_ue: Optional[float] = 0.0
    cov_en: Optional[float] = 0.0

    def get_position(self) -> List[float]:
        return [self.east.value, self.north.value, self.up.value]

    def get_std_dev(self) -> List[float]:
        return [self.east.sigma, self.north.sigma, self.up.sigma]

    def get_covariance(self) -> np.ndarray:
        cov_mat = np.diag([self.east.sigma**2, self.north.sigma**2, self.up.sigma**2])
        cov_mat[0, 1] = cov_mat[1, 0] = self.cov_en**2
        cov_mat[0, 2] = cov_mat[2, 0] = self.cov_ue**2
        cov_mat[1, 2] = cov_mat[2, 1] = self.cov_nu**2
        return cov_mat


class PositionLLH(BaseModel):
    latitude: Point
    longitude: Point
    height: Point


class ATDOffset(BaseModel):
    forward: Point
    rightward: Point
    downward: Point
    cov_rd: Optional[float] = 0.0
    cov_df: Optional[float] = 0.0
    cov_fr: Optional[float] = 0.0

    def get_offset(self) -> List[float]:
        return [self.forward.value, self.rightward.value, self.downward.value]

    def get_std_dev(self) -> List[float]:
        return [self.forward.sigma, self.rightward.sigma, self.downward.sigma]

    def get_covariance(self) -> np.ndarray:
        cov_mat = np.diag(
            [self.forward.sigma**2, self.rightward.sigma**2, self.downward.sigma**2]
        )
        cov_mat[0, 1] = cov_mat[1, 0] = self.cov_fr**2
        cov_mat[0, 2] = cov_mat[2, 0] = self.cov_df**2
        cov_mat[1, 2] = cov_mat[2, 1] = self.cov_rd**2
        return cov_mat


class Transponder(BaseModel):
    id: str
    position_enu: PositionENU
    # position_array_enu: PositionENU


class SiteData(BaseModel):
    center_enu: PositionENU
    center_llh: PositionLLH
    transponders: List[Transponder]


class ModelParameters(BaseModel):
    delta_center_position: PositionENU
    atd_offset: ATDOffset
    transponder_delta_position: Optional[List[PositionENU]] = None


class Site(BaseModel):
    name: str
    campaign: Optional[str] = None
    date_utc: datetime
    date_mjd: float
    ref_frame: str = "ITRF2014"
    site_data: SiteData
    atd_offset: ATDOffset
    delta_center_position: PositionENU

    shot_data: Optional[DataFrame[ObservationData]]
    sound_speed_data: Optional[DataFrame[SoundVelocityProfile]]


def output_results(
        site_name:str, 
        campaign:str, 
        date_utc:str, 
        date_jday:str, 
        ref_frame:str, 
        latitude:float, 
        longitude:float, 
        height:float,
        imp0: np.ndarray,
        slvidx0: np.ndarray,
        C: np.ndarray,
        mp: np.ndarray,
        MTs: List[str],
        shots:DataFrame,
        svp:DataFrame,
        mtidx:dict,
        ) -> Site:

    ##################
    # Write CFG Data #
    ##################
    ii = [1, 2, 0]
    jj = [2, 0, 1]
    imp = 0
    MTpos = []

    C0pos = np.zeros((imp0[0], imp0[0]))
    for i, ipos in enumerate(slvidx0):
        for j, jpos in enumerate(slvidx0):
            C0pos[ipos, jpos] = C[i, j]
    for mt in MTs:
        lmt = [mt]
        poscov = C0pos[imp : imp + 3, imp : imp + 3]
        for k in range(3):
            idx = mtidx[mt] + k
            lmt.append(mp[idx])
        lmt = lmt + [poscov[i][i] ** 0.5 for i in range(3)]
        lmt = lmt + [poscov[i][j] for i, j in zip(ii, jj)]
        MTpos.append(lmt)
        imp += 3

    poscov = C0pos[imp : imp + 3, imp : imp + 3]
    dcpos = [mp[len(MTs) * 3 + 0], mp[len(MTs) * 3 + 1], mp[len(MTs) * 3 + 2]]
    dcpos = dcpos + [poscov[i][i] ** 0.5 for i in range(3)]
    dcpos = dcpos + [poscov[i][j] for i, j in zip(ii, jj)]
    imp += 3

    poscov = C0pos[imp : imp + 3, imp : imp + 3]
    pbias = [mp[len(MTs) * 3 + 3], mp[len(MTs) * 3 + 4], mp[len(MTs) * 3 + 5]]
    pbias = pbias + [poscov[i][i] ** 0.5 for i in range(3)]
    pbias = pbias + [poscov[i][j] for i, j in zip(ii, jj)]
    imp += 3

    nn = float(len(MTpos))
    pe = np.array([tp[1] / nn for tp in MTpos]).sum() + dcpos[0]
    pn = np.array([tp[2] / nn for tp in MTpos]).sum() + dcpos[1]
    pu = np.array([tp[3] / nn for tp in MTpos]).sum() + dcpos[2]

    site_center_enu = PositionENU(
        east=Point(value=pe, sigma=0.0),
        north=Point(value=pn, sigma=0.0),
        up=Point(value=pu, sigma=0.0),
    )
    site_center_llh = PositionLLH(
        latitude=Point(value=latitude, sigma=0.0),
        longitude=Point(value=longitude, sigma=0.0),
        height=Point(value=height, sigma=0.0),
    )

    transponder_data = []
    for tp in MTpos:
        transponder = Transponder(
            id=tp[0],
            position_enu=PositionENU(
                east=Point(value=tp[1], sigma=tp[4]),
                north=Point(value=tp[2], sigma=tp[5]),
                up=Point(value=tp[3], sigma=tp[6]),
                cov_en=tp[7],
                cov_ue=tp[8],
                cov_nu=tp[9],
            ),
        )
        transponder_data.append(transponder)

    delta_center_position = PositionENU(
        east=Point(value=dcpos[0], sigma=dcpos[3]),
        north=Point(value=dcpos[1], sigma=dcpos[4]),
        up=Point(value=dcpos[2], sigma=dcpos[5]),
        cov_en=dcpos[6],
        cov_ue=dcpos[7],
        cov_nu=dcpos[8],
    )
	
    atd_offset = ATDOffset(
        forward=Point(value=pbias[0], sigma=pbias[3]),
        rightward=Point(value=pbias[1], sigma=pbias[4]),
        downward=Point(value=pbias[2], sigma=pbias[5]),
        cov_fr=pbias[6],
        cov_df=pbias[7],
        cov_rd=pbias[8],
    )

    site_data = SiteData(
        center_enu=site_center_enu,
        center_llh=site_center_llh,
        transponders=transponder_data,
    )

    site = Site(
        name=site_name,
        campaign=campaign,
        date_utc=date_utc,
        date_mjd=date_jday,
        ref_frame=ref_frame,
        site_data=site_data,
        atd_offset=atd_offset,
        delta_center_position=delta_center_position,
        shot_data=shots,
        sound_speed_data=svp,
    )