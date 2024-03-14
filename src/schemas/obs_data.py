"""
Author: Franklyn Dunbar
Date: 2024-03-12
Email: franklyn.dunbar@earthscope.org
"""
from typing import List,Optional
from pydantic import BaseModel,Field,model_validator,ValidationError
import numpy as np
import pandera as pa
from pandera.typing import Series,Index,DataFrame


class Point(BaseModel):
    value: float
    sigma: float

class PositionENU(BaseModel):
    east: Point
    north: Point
    up: Point
    cov_nu: Optional[float] = 0.0
    cov_ue: Optional[float] = 0.0
    cov_en: Optional[float] = 0.0

    def get_position(self) -> List[float]:
        return [self.east.value,self.north.value,self.up.value]
    
    def get_covariance(self) -> np.ndarray:
        cov_mat = np.diag([self.east.sigma**2,self.north.sigma**2,self.up.sigma**2])
        cov_mat[0,1] = cov_mat[1,0] = self.cov_en**2
        cov_mat[0,2] = cov_mat[2,0] = self.cov_ue**2
        cov_mat[1,2] = cov_mat[2,1] = self.cov_nu**2
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
        return [self.forward.value,self.rightward.value,self.downward.value]
    
    def get_covariance(self) -> np.ndarray:
        cov_mat = np.diag([self.forward.sigma**2,self.rightward.sigma**2,self.downward.sigma**2])
        cov_mat[0,1] = cov_mat[1,0] = self.cov_fr**2
        cov_mat[0,2] = cov_mat[2,0] = self.cov_df**2
        cov_mat[1,2] = cov_mat[2,1] = self.cov_rd**2
        return cov_mat
    
class Transponder(BaseModel):
    id: str
    position_enu: PositionENU

class Site(BaseModel):
    name: str
    transponders: List[Transponder]
    delta_center_position: PositionENU
    atd_offset: ATDOffset
    center_enu: PositionENU


class AntennaPosition(BaseModel):
    east: float
    north: float
    up: float
    heading: float
    pitch: float
    roll: float

class ShotObservation(BaseModel):
    set: str # Names of subset in each observation (typically S01, S02,...) |
    line: str #Names of survey lines in each observation (typically L01, L02,...)
    station: Transponder.id # Names of stations in each observation (typically M01, M02,...)
    travel_time: float # Observed travel time between transmission and reception of the signal [s]
    transmission_time: float # Time of transmission of the signal in MJD [s]
    reception_time: float # Time of reception of the signal in MJD [s]
    antenna0: AntennaPosition
    antenna1: AntennaPosition



class SoundVelocityProfile(pa.DataFrameModel):

    depth: Series[float] = pa.Field(
        ge=0, le=10000, description="Depth of the speed [m]"
    )
    speed: Series[float] = pa.Field(
        ge=0, le=5000, description="Spee of sound [m/s]"
    )

class ObservationData(pa.DataFrameModel):
    """Observation data file schema

    Example data:

    ,SET,LN,MT,TT,ResiTT,TakeOff,gamma,flag,ST,ant_e0,ant_n0,ant_u0,head0,pitch0,roll0,RT,ant_e1,ant_n1,ant_u1,head1,pitch1,roll1
    0,S01,L01,M11,2.289306,0.0,0.0,0.0,False,30072.395125,-27.85291,1473.14423,14.73469,176.47,0.59,-1.39,30075.74594,-26.70998,1462.01803,14.32703,177.07,-0.5,-1.1
    1,S01,L01,M13,3.12669,0.0,0.0,0.0,False,30092.395725,-22.08296,1412.88729,14.59827,188.24,0.41,-2.13,30096.58392,-22.3514,1401.77938,14.65401,190.61,-0.1,-2.14
    2,S01,L01,M14,2.702555,0.0,0.0,0.0,False,30093.48579,-22.25377,1409.87685,14.67772,188.93,0.15,-1.7,30097.24985,-22.38458,1399.96509,14.55534,190.82,-0.39,-2.21
    3,S01,L01,M14,2.68107,0.0,0.0,0.0,False,30102.396135,-23.25514,1387.38992,14.75355,192.39,0.1,-1.79,30106.13871,-23.96613,1378.4627,14.58135,192.92,0.21,-1.7
    4,S01,L01,M11,2.218846,0.0,0.0,0.0,False,30103.4862,-23.57701,1384.73242,14.65861,192.62,-0.14,-1.5,30106.766555,-24.0478,1377.09283,14.68464,193.04,0.59,-1.81
    """
    SET: Series[str] = pa.Field(
        description="Set name"
    )
    LN: Series[str] = pa.Field(
        description="Line name"
    )
    MT: Series[Transponder.id] = pa.Field(
        description="Station name"
    )
    TT: Series[float] = pa.Field(
        description="Travel time [sec]"
    )
    ResiTT: Series[float] = pa.Field()

    TakeOff: Series[float] = pa.Field()

    gamma: Series[float] = pa.Field()

    flag: Series[bool] = pa.Field()

    ST: Series[float] = pa.Field()

    ant_e0: Series[float] = pa.Field(
        description="Antenna position in east direction (ENU coords) at the time of the first measurement [m]")

    ant_n0: Series[float] = pa.Field(description="Antenna position in north direction (ENU coords) at the time of the first measurement [m]")
    
    ant_u0: Series[float] = pa.Field(description="Antenna position in up direction (ENU coords) at the time of the first measurement [m]")

    head0: Series[float] = pa.Field(description="Antenna heading at the time of the first measurement [deg]")

    pitch0: Series[float] = pa.Field(description="Antenna pitch at the time of the first measurement [deg]")

    roll0: Series[float] = pa.Field(description="Antenna roll at the time of the first measurement [deg]")

    RT: Series[float] = pa.Field()

    ant_e1: Series[float] = pa.Field(description="Antenna position in east direction (ENU coords) at the time of the second measurement [m]")

    ant_n1: Series[float] = pa.Field(description="Antenna position in north direction (ENU coords) at the time of the second measurement [m]")

    ant_u1: Series[float] = pa.Field(description="Antenna position in up direction (ENU coords) at the time of the second measurement [m]")

    head1: Series[float] = pa.Field(description="Antenna heading at the time of the second measurement [deg]")

    pitch1: Series[float] = pa.Field(description="Antenna pitch at the time of the second measurement [deg]")

    roll1: Series[float] = pa.Field(description="Antenna roll at the time of the second measurement [deg]")



# TODO define the schema for a single line of the observation data file
