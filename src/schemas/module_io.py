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
        # num_params += 2*3
        return num_params
    
    def get_mean(self):
        """
        Get the model parameter vector.

        [TP_0,TP_1,...]
        [transponser_positions,site_position_delta,atd_offset_position]
        Returns:
            np.ndarray: The model parameter vector.
        """
        model_parameters = np.array([])
        for transponder in self.transponder_positions.values():
            model_parameters = np.append(model_parameters,transponder.mean)
        # model_parameters = np.append(model_parameters,self.site_position_delta.mean)
        # model_parameters = np.append(model_parameters,self.atd_offset_position.mean)
        return model_parameters
    
    def get_cov_inv(self):
        """
        Get the inverse of the model parameter covariance matrix.

        [TP,SPD,ATD]
        Returns:
            np.ndarray: The inverse of the model parameter covariance matrix.
        """
        cov_inv = np.zeros((self.num_params,self.num_params))
        for i,transponder in enumerate(self.transponder_positions.values()):
            cov_inv[i*3:i*3+3,i*3:i*3+3] = transponder.cov_inv
        # cov_inv[-6:-3,-6:-3] = self.site_position_delta.cov_inv
        # cov_inv[-3:,-3:] = self.atd_offset_position.cov_inv
        return cov_inv
    

    def get_transponder_center_mean(self):
        # Get the mean position for the transponders
        positions:np.ndarray = np.stack([t.mean for t in self.transponder_positions.values()],axis=0).mean(axis=0)
        return positions