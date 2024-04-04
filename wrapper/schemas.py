"""
Author: Franklyn Dunbar
Date: 2024-03-12
Email: franklyn.dunbar@earthscope.org
"""
import os
from typing import List, Optional, Dict,Union
from pydantic import BaseModel, Field, model_validator, ValidationError,root_validator
from enum import Enum
import numpy as np
from datetime import datetime
import pandera as pa
import pandas as pd
from pandas.errors import ParserError
from pandera.typing import Series, Index, DataFrame
from configparser import ConfigParser

class HyperParams(BaseModel):
    log_lambda: List[float] = Field(
        default=[-2, -1], description="Smoothness paramter for backgroun perturbation"
    )
    log_gradlambda: float = Field(
        default=-1, description="Smoothness paramter for spatial gradient"
    )
    mu_t: List[float] = Field(
        default=[0.0, 1.0],
        description="Correlation length of data for transmit time [minute]",
    )
    mu_mt: float = Field(
        default=0.5,
        description="Data correlation coefficient b/w the different transponders",
    )


class InversionType(Enum):
    positions = 0  # solve only positions
    gammas = 1  # solve only gammas (sound speed variation)
    both = 2  # solve both positions and gammas


class InversionParams(BaseModel):
    spline_degree: int = Field(default=3)
    log_lambda: List[float] = Field(
        default=[-2, -1], description="Smoothness paramter for backgroun perturbation"
    )
    log_gradlambda: float = Field(
        default=-1, description="Smoothness paramter for spatial gradient"
    )
    mu_t: List[float] = Field(
        default=[0.0, 1.0],
        description="Correlation length of data for transmit time [minute]",
    )
    mu_mt: float = Field(
        default=0.5,
        description="Data correlation coefficient b/w the different transponders",
    )

    knotint0: int = Field(
        default=5,
        description="Typical Knot interval (in min.) for gamma's component (a0, a1, a2)",
    )
    knotint1: int = Field(
        default=5,
        description="Typical Knot interval (in min.) for gamma's component (a0, a1, a2)",
    )
    knotint2: int = Field(
        default=5,
        description="Typical Knot interval (in min.) for gamma's component (a0, a1, a2)",
    )
    rejectcriteria: float = Field(
        default=0.0, description="Criteria for the rejection of data (+/- rsig * Sigma)"
    )
    inversiontype: InversionType = Field(
        default=InversionType.both, description="Inversion type"
    )
    positionalOffset: Optional[List[float]] = Field(
        default=[0.0, 0.0, 0.0], description="Positional offset for the inversion"
    )
    traveltimescale: float = Field(
        default=1.0e-4,
        description="Typical measurement error for travel time (= 1.e-4 sec is recommended in 10 kHz carrier)",
    )
    maxloop: int = Field(default=50, description="Maximum loop for iteration")
    convcriteria: float = Field(
        default=5.0e-3, description="Convergence criteria for model parameters"
    )
    deltap: float = Field(
        default=1.0e-6, description="Infinitesimal values to make Jacobian matrix"
    )
    deltab: float = Field(
        default=1.0e-6, description="Infinitesimal values to make Jacobian matrix"
    )

    @root_validator(pre=True)
    def validate(cls, values):
        match values["inversiontype"]:
            case InversionType.gammas:
                if any(values["positionalOffset"] <= 0):
                    raise Warning(
                        "positionalOffset is required for InversionType.positions"
                    )
            case [InversionType.positions, InversionType.both]:
                if any(values["positionalOffset"] > 0):
                    values["positionalOffset"] = [0.0, 0.0, 0.0]
                    raise Warning(
                        "positionalOffset is not required for InversionType.gammas"
                    )

        return values


class SoundVelocityProfile(pa.DataFrameModel):

    depth: Series[float] = pa.Field(
        ge=0, le=10000, description="Depth of the speed [m]"
    )
    speed: Series[float] = pa.Field(ge=0, le=3800, description="Spee of sound [m/s]")


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

    set: Series[str] = pa.Field(description="Set name", alias="SET")
    line: Series[str] = pa.Field(description="Line name", alias="LN")
    transponder_id: Series[str] = pa.Field(description="Station name", alias="MT")
    travel_time: Series[float] = pa.Field(description="Travel time [sec]", alias="TT")

    transmission_time: Series[float] = pa.Field(
        description="Time of transmission of the acoustic signal in MJD [s]", alias="ST"
    )

    reception_time: Series[float] = pa.Field(
        description="Time of reception of the acoustic signal in MJD [s]", alias="RT"
    )

    ant_e0: Series[float] = pa.Field(
        description="Antenna position in east direction (ENU coords) at the time of the first measurement [m]"
    )

    ant_n0: Series[float] = pa.Field(
        description="Antenna position in north direction (ENU coords) at the time of the first measurement [m]"
    )

    ant_u0: Series[float] = pa.Field(
        description="Antenna position in up direction (ENU coords) at the time of the first measurement [m]"
    )

    head0: Series[float] = pa.Field(
        description="Antenna heading at the time of the first measurement [deg]"
    )

    pitch0: Series[float] = pa.Field(
        description="Antenna pitch at the time of the first measurement [deg]"
    )

    roll0: Series[float] = pa.Field(
        description="Antenna roll at the time of the first measurement [deg]"
    )

    ant_e1: Series[float] = pa.Field(
        description="Antenna position in east direction (ENU coords) at the time of the second measurement [m]"
    )

    ant_n1: Series[float] = pa.Field(
        description="Antenna position in north direction (ENU coords) at the time of the second measurement [m]"
    )

    ant_u1: Series[float] = pa.Field(
        description="Antenna position in up direction (ENU coords) at the time of the second measurement [m]"
    )

    head1: Series[float] = pa.Field(
        description="Antenna heading at the time of the second measurement [deg]"
    )

    pitch1: Series[float] = pa.Field(
        description="Antenna pitch at the time of the second measurement [deg]"
    )

    roll1: Series[float] = pa.Field(
        description="Antenna roll at the time of the second measurement [deg]"
    )


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
    latitude: float
    longitude: float
    height: float


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


class ModelResults(ObservationData):
    # These fields are populated after the model run
    ResiTT: Optional[Series[float]] = pa.Field()

    TakeOff: Optional[Series[float]] = pa.Field()

    gamma: Optional[Series[float]] = pa.Field()

    flag: Optional[Series[bool]] = pa.Field()


class InversionLoop(BaseModel):
    iteration:int
    rms_tt:float # ms
    used_shot_percentage:float
    reject:int
    max_dx:float
    hgt:float
    inv_type:InversionType

""" 
class to process inversion results from the datfile:

[Obs-parameter] 
 Site_name   = SAGA
 Campaign    = 1903.kaiyo_k4
 Date(UTC)   = 2019-03-15
 Date(jday)  = 2019-074
 Ref.Frame   = ITRF2014
 SoundSpeed  = /Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/obsdata/SAGA/SAGA.1903.kaiyo_k4-svp.csv

[Data-file]
 datacsv     = /tmp/garpos/results/lambda/SAGA.1903.kaiyo_k4test_L-02.0_T1.0-obs.csv
 N_shot      =  3614
 used_shot   =  3614

[Site-parameter]
 Latitude0   =  34.96166667
 Longitude0  = 139.26333333
 Height0     =  43.00
 Stations    = M11 M12 M13 M14
# Array_cent :   'cntpos_E'  'cntpos_N'  'cntpos_U'
 Center_ENU  =    -31.1393    -17.8851  -1341.4609

[Model-parameter]
# MT_Pos     :   'stapos_E'  'stapos_N'  'stapos_U'   'sigma_E'   'sigma_N'   'sigma_U'   'cov_NU'    'cov_UE'    'cov_EN'
 M11_dPos    =    -46.9306    408.7659  -1345.0836      0.0420      0.0414      0.0409   1.667e-04   1.872e-05   8.809e-06
 M12_dPos    =    486.6786     48.2907  -1354.3655      0.0416      0.0439      0.0428   3.408e-05   1.034e-04  -5.557e-06
 M13_dPos    =    -26.2878   -505.9762  -1335.8696      0.0423      0.0414      0.0428  -1.272e-04  -1.774e-05  -1.438e-05
 M14_dPos    =   -538.0173    -22.6206  -1330.5250      0.0412      0.0430      0.0417  -4.587e-05  -1.144e-04  -1.675e-05
 dCentPos    =      0.0000      0.0000      0.0000      0.0000      0.0000      0.0000   0.000e+00   0.000e+00   0.000e+00
# ANT_to_TD  :    'forward' 'rightward'  'downward'   'sigma_F'   'sigma_R'   'sigma_D'   'cov_RD'    'cov_DF'    'cov_FR'
 ATDoffset   =      1.5547     -1.2690     23.7295      0.0000      0.0000      0.0000   0.000e+00   0.000e+00   0.000e+00

#Inversion-type 2 Loop  1- 1, RMS(TT) =   0.087357 ms, used_shot = 100.0%, reject =    0, Max(dX) =     7.7929, Hgt =  -1341.461
#Inversion-type 2 Loop  1- 2, RMS(TT) =   0.087357 ms, used_shot = 100.0%, reject =    0, Max(dX) =     0.0008, Hgt =  -1341.461
#Inversion-type 2 Loop  1- 3, RMS(TT) =   0.087357 ms, used_shot = 100.0%, reject =    0, Max(dX) =     0.0007, Hgt =  -1341.461
#  ABIC =       26201.906874  misfit =  0.092 test_L-02.0_T1.0
# lambda_0^2 =   0.01000000
# lambda_g^2 =   0.00100000
# mu_t =  60.00000000 sec.
# mu_MT = 0.5000



"""
class InversionResults(BaseModel):
    ABIC: float
    misfit: float
    inv_type: InversionType
    lambda_0_squared: float
    grad_lambda_squared: float
    mu_t: float # [s]
    mu_mt: float
    delta_center_position: List[float]
    loop_data: List[InversionLoop]

    @classmethod
    def from_dat_file(cls, file_path: str) -> "InversionResults":
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Extract data from the file
            loop_data = []
            for line in lines:
                if line.startswith("#Inversion-type"):
                    parsed_line = line.split()
                    iteration = int(parsed_line[4].replace(",", ""))
                    inv_type = InversionType(int(parsed_line[1]))
                    rms_tt = float(parsed_line[7])
                    used_shot_percentage = float(parsed_line[11].replace("%,", ""))
                    reject = int(parsed_line[14].replace(",", ""))
                    max_dx = float(parsed_line[17].replace(",", ""))
                    hgt = float(parsed_line[20])
                    loop_data.append(
                        InversionLoop(
                            iteration=iteration, 
                            rms_tt=rms_tt, 
                            used_shot_percentage=used_shot_percentage, 
                            reject=reject, 
                            max_dx=max_dx, 
                            hgt=hgt, 
                            inv_type=inv_type))
                if line.startswith("dcentpos"):
                    parsed_line = line.split()
                    delta_center_position = [float(x) for x in parsed_line[2]]
                if line.startswith("#  ABIC"):
                    parsed_line = line.split()
                    ABIC = float(parsed_line[3])
                    misfit = float(parsed_line[6])
                if line.startswith("# lambda_0^2"):
                    parsed_line = line.split()
                    lambda_0_squared = float(parsed_line[3])
                if line.startswith("# lambda_g^2"):
                    parsed_line = line.split()
                    grad_lambda_squared = float(parsed_line[3])
                if line.startswith("# mu_t"):
                    parsed_line = line.split()
                    mu_t = float(parsed_line[3])
                if line.startswith("# mu_MT"):
                    parsed_line = line.split()
                    mu_mt = float(parsed_line[3])
            return cls(
                delta_center_position=delta_center_position,
                ABIC=ABIC, 
                misfit=misfit, 
                inv_type=inv_type, 
                lambda_0_squared=lambda_0_squared, 
                grad_lambda_squared=grad_lambda_squared, 
                mu_t=mu_t, mu_mt=mu_mt, loop_data=loop_data)


class Observation(BaseModel):
    campaign: str
    date_utc: datetime
    date_mjd: str
    ref_frame: str = "ITRF2014"
    shot_data: Union[DataFrame[ObservationData],DataFrame[ModelResults]]
    sound_speed_data: DataFrame[SoundVelocityProfile]


class Site(BaseModel):
    name: str
    atd_offset: ATDOffset
    center_enu: PositionENU
    center_llh: PositionLLH
    transponders: List[Transponder]
    delta_center_position: PositionENU

class GarposFixed(BaseModel):
    lib_directory: str
    lib_raytrace: str
    inversion_params: InversionParams

    def to_dat_file(self,dir_path:str,file_path:str) -> None:
        fixed_str = f"""[HyperParameters]
# Hyperparameters
#  When setting multiple values, ABIC-minimum HP will be searched.
#  The delimiter for multiple HP must be "space".

# Smoothness parameter for background perturbation (in log10 scale)
Log_Lambda0 = {" ".join([str(x) for x in self.inversion_params.log_lambda])}

# Smoothness parameter for spatial gradient ( = Lambda0 * gradLambda )
Log_gradLambda = {self.inversion_params.log_gradlambda}

# Correlation length of data for transmit time (in min.)
mu_t = {" ".join([str(x) for x in self.inversion_params.mu_t])}

# Data correlation coefficient b/w the different transponders.
mu_mt = {self.inversion_params.mu_mt}

[Inv-parameter]
# The path for RayTrace lib.
lib_directory = {self.lib_directory}
lib_raytrace = {self.lib_raytrace}

# Typical Knot interval (in min.) for gamma's component (a0, a1, a2).
#  Note ;; shorter numbers recommended, but consider the computational resources.
knotint0 = {self.inversion_params.knotint0}
knotint1 = {self.inversion_params.knotint1}
knotint2 = {self.inversion_params.knotint2}

# Criteria for the rejection of data (+/- rsig * Sigma).
# if = 0, no data will be rejected during the process.
RejectCriteria = {self.inversion_params.rejectcriteria}

# Inversion type
#  0: solve only positions
#  1: solve only gammas (sound speed variation)
#  2: solve both positions and gammas
inversiontype = {self.inversion_params.inversiontype.value}

# Typical measurement error for travel time.
# (= 1.e-4 sec is recommended in 10 kHz carrier)
traveltimescale = {self.inversion_params.traveltimescale}

# Maximum loop for iteration.
maxloop = {self.inversion_params.maxloop}

# Convergence criteria for model parameters.
ConvCriteria = {self.inversion_params.convcriteria}

# Infinitesimal values to make Jacobian matrix.
deltap = {self.inversion_params.deltap}
deltab = {self.inversion_params.deltab}"""

        with open(os.path.join(dir_path,os.path.basename(file_path)), "w") as f:
            f.write(fixed_str)
        
    @classmethod
    def from_dat_file(cls, file_path: str) -> "GarposFixed":
        config = ConfigParser()
        config.read(file_path)

        # Extract data from config
        hyperparameters = config["HyperParameters"]
        inv_parameters = config["Inv-parameter"]

        # Populate InversionParams
        inversion_params = InversionParams(
            log_lambda=[float(x) for x in hyperparameters["Log_Lambda0"].split()],
            log_gradlambda=float(hyperparameters["Log_gradLambda"]),
            mu_t=[float(x) for x in hyperparameters["mu_t"].split()],
            mu_mt=float(hyperparameters["mu_mt"]),
            knotint0=int(inv_parameters["knotint0"]),
            knotint1=int(inv_parameters["knotint1"]),
            knotint2=int(inv_parameters["knotint2"]),
            rejectcriteria=float(inv_parameters["RejectCriteria"]),
            inversiontype=InversionType(int(inv_parameters["inversiontype"])),
            traveltimescale=float(inv_parameters["traveltimescale"]),
            maxloop=int(inv_parameters["maxloop"]),
            convcriteria=float(inv_parameters["ConvCriteria"]),
            deltap=float(inv_parameters["deltap"]),
            deltab=float(inv_parameters["deltab"]),
        )

        # Populate GarposFixed
        return cls(
            lib_directory=inv_parameters["lib_directory"],
            lib_raytrace=inv_parameters["lib_raytrace"],
            inversion_params=inversion_params,
        )

class GarposInput(BaseModel):
    observation: Observation
    site: Site
    shot_data_file: Optional[str] = None
    sound_speed_file: Optional[str] = None

   
    def to_dat_file(self, dir_path:str, file_path: str) -> None:
        if not self.shot_data_file:
            self.shot_data_file = os.path.join(dir_path, f"{self.site.name}_shot_data.csv")
        if not self.sound_speed_file:
            self.sound_speed_file = os.path.join(dir_path, f"{self.site.name}_sound_speed.csv")

        center_enu = self.site.center_enu.get_position()
        delta_center_position = self.site.delta_center_position.get_position() + self.site.delta_center_position.get_std_dev()
        delta_center_position += [0.0, 0.0, 0.0]
        atd_offset = self.site.atd_offset.get_offset() + self.site.atd_offset.get_std_dev() + [0.0, 0.0, 0.0]
        obs_str = f"""
[Obs-parameter]
    Site_name   = {self.site.name}
    Campaign    = {self.observation.campaign}
    Date(UTC)   = {self.observation.date_utc.strftime('%Y-%m-%d')}
    Date(jday)  = {self.observation.date_mjd}
    Ref.Frame   = {self.observation.ref_frame}
    SoundSpeed  = {self.sound_speed_file}

[Data-file]
    datacsv     = {self.shot_data_file}
    N_shot      = {len(self.observation.shot_data.index)}
    used_shot   = {0}

[Site-parameter]
    Latitude0   = {self.site.center_llh.latitude}
    Longitude0  = {self.site.center_llh.longitude}
    Height0     = {self.site.center_llh.height}
    Stations    = {' '.join([transponder.id for transponder in self.site.transponders])}
    Center_ENU  = {center_enu[0]} {center_enu[1]} {center_enu[2]}

[Model-parameter]
    dCentPos    = {" ".join(map(str, delta_center_position))}
    ATDoffset   = {" ".join(map(str, atd_offset))}"""

        # Add the transponder data to the string
        for transponder in self.site.transponders:
            position = transponder.position_enu.get_position() + transponder.position_enu.get_std_dev() + [0.0, 0.0, 0.0]
            obs_str += f"""
    {transponder.id}_dPos    = {" ".join(map(str, position))}"""

        with open(os.path.join(dir_path,os.path.basename(file_path)), "w") as f:
            f.write(obs_str)
  
    @classmethod
    def from_dat_file(cls, file_path: str) -> "GarposInput":
        config = ConfigParser()
        config.read(file_path)

        # Extract data from config
        observation_section = config["Obs-parameter"]
        site_section = config["Site-parameter"]
        model_section = config["Model-parameter"]
        data_section = config["Data-file"]
        # populate transponders
        transponder_list = []
        for key in model_section.keys():
            (
                east_value,
                north_value,
                up_value,
                east_sigma,
                north_sigma,
                up_sigma,
                cov_en,
                cov_ue,
                cov_nu,
            ) = [float(x) for x in model_section[key].split()]
            position = PositionENU(
                east=Point(value=east_value, sigma=east_sigma),
                north=Point(value=north_value, sigma=north_sigma),
                up=Point(value=up_value, sigma=up_sigma),
                cov_en=cov_en,
                cov_ue=cov_ue,
                cov_nu=cov_nu,
            )
            if "dpos" in key:
                transponder_id = key.split("_")[0].upper()
                transponder = Transponder(id=transponder_id, position_enu=position)
                transponder_list.append(transponder)
            if "dcentpos" in key:
                delta_center_position = position
            if "atdoffset" in key:
                atd_offset = ATDOffset(forward=position.east, rightward=position.north, downward=position.up, cov_fr=cov_en, cov_df=cov_ue, cov_rd=cov_nu)
        # Populate Site
        site = Site(
            name=observation_section["site_name"],
            atd_offset=atd_offset,
            center_enu=PositionENU(
                east=Point(value=float(site_section["center_enu"].split()[0])),
                north=Point(value=float(site_section["center_enu"].split()[1])),
                up=Point(value=float(site_section["center_enu"].split()[2])),
            ),
            center_llh=PositionLLH(
                latitude=float(site_section["latitude0"]),
                longitude=float(site_section["longitude0"]),
                height=float(site_section["height0"]),
            ),
            transponders=transponder_list,
            delta_center_position=delta_center_position,  # We'll handle this later
        )

        # Now handle shot_data_file and sound_speed_file
        shot_data_file = data_section["datacsv"]
        sound_speed_file = observation_section["soundspeed"]

        try:
            df = pd.read_csv(shot_data_file, index_col=0)
        except ParserError:
            df = pd.read_csv(shot_data_file, index_col=0, skiprows=1)
        shot_data_results = ModelResults(df)
        sound_speed_results = SoundVelocityProfile(pd.read_csv(sound_speed_file))
        # Populate Observation
        observation = Observation(
            campaign=observation_section["campaign"],
            date_utc=datetime.strptime(observation_section["date(utc)"], "%Y-%m-%d"),
            date_mjd=observation_section["date(jday)"],
            ref_frame=observation_section["ref.frame"],
            shot_data=shot_data_results,
            sound_speed_data=sound_speed_results,
        )

        # Instantiate GarposInput
        return cls(
            observation=observation,
            site=site,
            shot_data_file=shot_data_file,
            sound_speed_file=sound_speed_file,
        )


if __name__ == "__main__":

    datfile = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/output/lambda/SAGA.1903.kaiyo_k4test_L-02.0_T0.0-res.dat"
    garpos_input = GarposInput.from_dat_file(datfile)

    temp_file = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/output/temp.dat"

    garpos_input.to_dat_file(temp_file)

    garpos_input_2 = GarposInput.from_dat_file(temp_file)

    fixed_file = (
        "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/Settings-fix.ini"
    )
    garpos_fixed = GarposFixed.from_fixed_format(fixed_file)

    temp_fixed_file = "/Users/franklyndunbar/Project/SeaFloorGeodesy/garpos/sample/output/temp_fixed.ini"
    garpos_fixed.to_fixed_format(temp_fixed_file)

    garpos_fixed_2 = GarposFixed.from_fixed_format(temp_fixed_file)
    print("Done")
