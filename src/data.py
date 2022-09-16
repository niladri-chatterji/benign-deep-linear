from typing import Any, Optional

import torch
from torch import nn

from .model import ReLUNet

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


def PlantedTwoLayerIID(dimension: int,
                num_samples: int,
                output_dimension: int,
                variance: Optional[float] = None,
                width: Optional[float] = None,
                k: Optional[int] = None, 
                epsilon: Optional[float] = None,
                noise_variance: Optional[float] = 1.0,
                net: Optional[nn.Module] = None,
                device: Optional[str] = 'cuda'):
    '''
    Generate Gaussian IID n datapoints with n independent samples.
    The top k components have variance 1 by default
    The bottom d-k component have variance epsilon
    '''

    # Generate the X's

    X = torch.randn(num_samples, dimension)

    if variance is not None:
        X = (variance**(0.5)) * X

    if k is not None and epsilon is not None:
            X[:,k:] = (epsilon**(0.5))*X[:,k:]
    
    X = X.to(device)

    # Generate the responses
    # If a net is not provided, then initialize a new network
    if net is None:
        if width is not None:
            net = ReLUNet(num_hidden_layers=1, 
                         input_size = dimension, 
                         hidden_size = width, 
                         output_size = output_dimension).to(device)
        else:
            net = ReLUNet(num_hidden_layers=1, 
                         input_size = dimension, 
                         hidden_size = 10, 
                         output_size = output_dimension).to(device)

    
    noise = (noise_variance**(0.5))*torch.randn(num_samples,output_dimension)
    noise = noise.to(device)   

    Y = net(X) 

    assert Y.size()[0] == num_samples and Y.size()[1] == output_dimension

    Y = Y + noise

    return X, Y, net
