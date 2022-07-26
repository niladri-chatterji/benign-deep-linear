import torch
from typing import Any, Optional

def GaussianIID(dimension: int,
                num_samples: int,
                output_dimension: int,
                variance: Optional[float] = None,
                k: Optional[int] = None, 
                epsilon: Optional[float] = None,
                noise_variance: Optional[float] = 1.0,
                theta: Optional[torch.tensor] = None):
    '''
    Generate Gaussian IID n datapoints with n independent samples.
    The top k components have variance 1 by default
    The bottom d-k component have variance epsilon
    '''

    X = torch.randn(num_samples, dimension)

    if variance is not None:
        X = (variance**(0.5)) * X
    if k is not None and epsilon is not None:
        X[:,k:] = (epsilon**(0.5))*X[:,k:]

    # Generate the responses
    if theta is None:
        theta = torch.randn(dimension, output_dimension)
        theta = theta/torch.norm(theta)  # Sample a theta uniformly over the sphere with unit norm
    
    noise = (noise_variance**(0.5))*torch.randn(num_samples,output_dimension)
    
    Y = torch.matmul(X, theta) + noise

    return X, Y, theta

# TODO: Modify the code below to implement a two-layer neural network
def PlantedTwoLayerIID(dimension: int,
                num_samples: int,
                variance: Optional[float] = None,
                width: Optional[float] = None,
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