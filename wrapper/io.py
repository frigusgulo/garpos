from .schemas import GarposInput, InversionType, SoundVelocityProfile, ObservationData,GarposFixed


def to_file_format(garpos_input: GarposInput) -> str:
    # Convert the observation and site data to the desired format
    center_enu = garpos_input.site.center_enu.get_position()
    delta_center_position = garpos_input.site.delta_center_position.get_position() + garpos_input.site.delta_center_position.get_std_dev()
    delta_center_position += [0.0, 0.0, 0.0]
    atd_offset = garpos_input.site.atd_offset.get_offset() + garpos_input.site.atd_offset.get_std_dev() + [0.0, 0.0, 0.0]
    obs_str = f"""[Obs-parameter]
 Site_name   = {garpos_input.site.name}
 Campaign    = {garpos_input.observation.campaign}
 Date(UTC)   = {garpos_input.observation.date_utc.strftime('%Y-%m-%d')}
 Date(jday)  = {garpos_input.observation.date_mjd}
 Ref.Frame   = {garpos_input.observation.ref_frame}
 SoundSpeed  = {garpos_input.sound_speed_file}

[Data-file]
 datacsv     = {garpos_input.shot_data_file}
 N_shot      = {len(garpos_input.observation.shot_data.index)}
 used_shot   = {0}

[Site-parameter]
 Latitude0   = {garpos_input.site.center_llh.latitude}
 Longitude0  = {garpos_input.site.center_llh.longitude}
 Height0     = {garpos_input.site.center_llh.height}
 Stations    = {' '.join([transponder.id for transponder in garpos_input.site.transponders])}
 Center_ENU  = {center_enu[0]} {center_enu[1]} {center_enu[2]}

[Model-parameter]
 dCentPos    = {" ".join(map(str, delta_center_position))}
 ATDoffset   = {" ".join(map(str, atd_offset))}"""

    # Add the transponder data to the string
    for transponder in garpos_input.site.transponders:
        position = transponder.position_enu.get_position() + transponder.position_enu.get_std_dev() + [0.0, 0.0, 0.0]
        obs_str += f"""
 {transponder.id}_dPos = {" ".join(map(str, position))}"""

    return obs_str


def to_fixed_format(garpos_fixed: GarposFixed) -> str:
    inversion_params = garpos_fixed.inversion_params
    fixed_str = f"""[HyperParameters]
# Hyperparameters
#  When setting multiple values, ABIC-minimum HP will be searched.
#  The delimiter for multiple HP must be "space".

# Smoothness parameter for background perturbation (in log10 scale)
Log_Lambda0 = {" ".join([x for x in inversion_params.log_lambda])}

# Smoothness parameter for spatial gradient ( = Lambda0 * gradLambda )
Log_gradLambda = {inversion_params.log_gradlambda}

# Correlation length of data for transmit time (in min.)
mu_t = {" ".join([x for x in inversion_params.mu_t])}

# Data correlation coefficient b/w the different transponders.
mu_mt = {inversion_params.mu_mt}

[Inv-parameter]
# The path for RayTrace lib.
lib_directory = {garpos_fixed.lib_directory}
lib_raytrace = {garpos_fixed.lib_raytrace}

# Typical Knot interval (in min.) for gamma's component (a0, a1, a2).
#  Note ;; shorter numbers recommended, but consider the computational resources.
knotint0 = {inversion_params.knotint0}
knotint1 = {inversion_params.knotint1}
knotint2 = {inversion_params.knotint2}

# Criteria for the rejection of data (+/- rsig * Sigma).
# if = 0, no data will be rejected during the process.
RejectCriteria = {inversion_params.rejectcriteria}

# Inversion type
#  0: solve only positions
#  1: solve only gammas (sound speed variation)
#  2: solve both positions and gammas
inversiontype = {inversion_params.inversiontype.value}

# Typical measurement error for travel time.
# (= 1.e-4 sec is recommended in 10 kHz carrier)
traveltimescale = {inversion_params.traveltimescale}

# Maximum loop for iteration.
maxloop = {inversion_params.maxloop}

# Convergence criteria for model parameters.
ConvCriteria = {inversion_params.conv_criteria}

# Infinitesimal values to make Jacobian matrix.
deltap = {inversion_params.deltap}
deltab = {inversion_params.deltab}"""

    return fixed_str
