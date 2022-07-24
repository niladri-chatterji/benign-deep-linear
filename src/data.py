import torch
from typing import Any, Optional

def GaussianIID(dimension: int,
                num_samples: int,
                variance: Optional[float] = None,
                k: Optional[int] = None, 
                epsilon: Optional[float] = None):
    '''
    Generate Gaussian IID n datapoints with n independent samples.
    The top k components have variance 1 by default
    The bottom d-k component have variance epsilon
    '''

    data = torch.randn([dimension, num_samples])

    if variance is not None:
        data = (variance**(0.5)) * data
    if k is not None and epsilon is not None:
        data[k:,:] = (epsilon**(0.5))*data[k:,:]

    return data