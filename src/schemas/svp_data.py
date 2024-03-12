"""
Author: Franklyn Dunbar
Date: 2024-03-12
Email: franklyn.dunbar@earthscope.org
"""
from typing import List
from pydantic import BaseModel,Field,model_validator,ValidationError
from enum import Enum

import pandera as pa
from pandera.typing import Series,Index,DataFrame


class SoundVelocityProfile(pa.DataFrameModel):

    depth: Series[float] = pa.Field(
        ge=0, le=10000, description="Depth of the speed [m]"
    )
    speed: Series[float] = pa.Field(
        ge=0, le=5000, description="Spee of sound [m/s]"
    )