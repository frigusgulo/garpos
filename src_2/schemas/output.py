
from pydantic import BaseModel
from typing import List


class TPEstimate(BaseModel):
    id: str
    x_enu:float # Predicted Positions
    y_enu:float
    z_enu:float
    dx_enu:float # Change from previous position
    dy_enu:float
    dz_enu:float
    rmse:float # GARPOS errors/ uncertainty

class StationPrediction(BaseModel):
    transponder_predictions: List[TPEstimate]
    site_lat:float
    site_lon:float
    site_elev:float
    site_enu_north:float
    site_enu_east:float
    site_enu_up:float



