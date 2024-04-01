"""
Author: Franklyn Dunbar
Date: 2024-03-12
Email: franklyn.dunbar@earthscope.org
"""
from typing import List,Optional
from pydantic import BaseModel,Field,model_validator,ValidationError,field_validator,root_validator
from enum import Enum

class HyperParams(BaseModel):
    log_lambda: List[float] = Field(default=[-2,-1],description="Smoothness paramter for backgroun perturbation")
    log_gradlambda: float = Field(default=-1,description="Smoothness paramter for spatial gradient")
    mu_t: List[float] = Field(default=[0.0,1.0],description="Correlation length of data for transmit time [minute]")
    mu_mt: float = Field(default=0.5,description="Data correlation coefficient b/w the different transponders")
  
class InversionType(Enum):
    positions = 0 # solve only positions
    gammas = 1 # solve only gammas (sound speed variation)
    both = 2 # solve both positions and gammas

class InversionParams(BaseModel):
    spline_degree:int = Field(default=3)
    log_lambda: List[float] = Field(default=[-2,-1],description="Smoothness paramter for backgroun perturbation")
    log_gradlambda: float = Field(default=-1,description="Smoothness paramter for spatial gradient")
    mu_t: List[float] = Field(default=[0.0,1.0],description="Correlation length of data for transmit time [minute]")
    mu_mt: float = Field(default=0.5,description="Data correlation coefficient b/w the different transponders")
  
    knotint0: int = Field(default=5,description="Typical Knot interval (in min.) for gamma's component (a0, a1, a2)")
    knotint1: int = Field(default=5,description="Typical Knot interval (in min.) for gamma's component (a0, a1, a2)")
    knotint2: int = Field(default=5,description="Typical Knot interval (in min.) for gamma's component (a0, a1, a2)")
    rejectcriteria: float = Field(default=0.0,description="Criteria for the rejection of data (+/- rsig * Sigma)")
    inversiontype: InversionType = Field(default=InversionType.both,description="Inversion type")
    positionalOffset: Optional[List[float]] = Field(default=[0.0,0.0,0.0],description="Positional offset for the inversion")
    traveltimescale: float = Field(
        default=1.e-4,description="Typical measurement error for travel time (= 1.e-4 sec is recommended in 10 kHz carrier)")
    maxloop: int = Field(default=50,description="Maximum loop for iteration")
    convcriteria: float = Field(default=5.e-3,description="Convergence criteria for model parameters")
    deltap: float = Field(default=1.0e-6,description="Infinitesimal values to make Jacobian matrix")
    deltab: float = Field(default=1.0e-6,description="Infinitesimal values to make Jacobian matrix")
    

    @root_validator(pre=True)
    def validate(cls,values):
        match values['inversiontype']:
            case InversionType.gammas:
                if any(values['positionalOffset'] <= 0):
                    raise Warning("positionalOffset is required for InversionType.positions")
            case [InversionType.positions , InversionType.both]:
                if any(values['positionalOffset'] > 0):
                    values['positionalOffset'] =[0.0,0.0,0.0]
                    raise Warning("positionalOffset is not required for InversionType.gammas")

        return values