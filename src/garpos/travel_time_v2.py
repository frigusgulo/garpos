# External Imports
import numpy as np
from pandas import DataFrame

# Internal Imports
from .ray_tracer import Raytracer
from ..schemas.obs_data import ShotData
# TODO need a way to index model params by shotdat.mtid
# TODO what are shotdat.pl* values?

def calc_traveltime(shot_data:ShotData,mp:np.ndarray,nMT:int,svp:DataFrame):

    rayTracer = Raytracer()

