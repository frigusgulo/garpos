"""
Author: Franklyn Dunbar
Date: 2024-03-12
Email: franklyn.dunbar@earthscope.org
"""
from typing import List
from pydantic import BaseModel,Field,model_validator,ValidationError

class Point(BaseModel):
    value: float
    sigma: float

class PositionENU(BaseModel):
    east: Point
    north: Point
    up: Point
    cov_nu: float
    cov_ue: float
    cov_en: float

class PositionLLH(BaseModel):
    latitude: Point
    longitude: Point
    height: Point

class ATDOffset(BaseModel):
    forward: Point
    rightward: Point
    downward: Point
    cov_rd: float
    cov_df: float
    cov_fr: float

class Station(BaseModel):
    name: str
    position_enu: PositionENU

class Site(BaseModel):
    name: str
    stations: List[Station]
    delta_center_position: PositionENU
    atd_offset: ATDOffset
    center_enu: PositionENU
