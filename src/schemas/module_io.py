import numpy as np
from dataclasses import dataclass
from typing import List,Dict,Union


@dataclass
class Normal:
    mean: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray = None
    solve:bool = False

    def __init__(self,mean: Union[np.ndarray,List],cov: np.ndarray):
        if isinstance(mean,list):
            mean = np.array(mean)
        self.mean = mean
        self.cov = cov
        if any(np.diag(cov) > 1.0e-14):
            self.solve = True
        self.cov_inv = np.linalg.inv(cov)


@dataclass
class GaussianModelParameters:
    """
    Class for model parameter vector.
    """
    site_position_delta: Normal

    atd_offset_position: Normal

    transponder_positions: Dict[str,Normal]


    @property
    def num_params(self):
        """
        Get the number of parameters in the model.

        Returns:
            int: The number of parameters in the model.
        """
        # Get the number of transponder position vectors
        num_params = len(self.transponder_positions.keys())*3 

        # Add the number of site position delta parameters
        num_params += 2*3
        return num_params