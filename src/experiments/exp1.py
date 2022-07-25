from ..data import GaussianIID
from ..model import LinearNet, ReLUNet
import json
import torch
import numpy as np
from ..utils import TrainNet


def exp1(config):
        '''
        Experiment that compares the norm different between Theta and Theta_OLS. Sweep over different dimensions
        '''
        # Parameters for the data
        num_runs = config.exp1.num_runs
        start_dimension = config.exp1.start_dimension
        end_dimension = config.exp1.end_dimension
        variance = config.exp1.variance
        output_dimension = config.exp1.output_dimension

        dimensions_range = list(np.linspace(start_dimension, 
                                        end_dimension, 
                                        num = config.num_sweep, 
                                        endpoint=True))

        # Parameters for the model
        num_hidden_layers = config.exp1.num_hidden_layers

        # Parameters for the optimizer
        lr = config.optim.lr
        target_loss = config.optim.target_loss
        max_epochs = config.optim.max_epochs

        for dimension in dimensions_range:
                num_samples = config.exp1.num_samples
                k = config.exp1.k
                epsilon = config.exp1.epsilon

                # The details of the model
                input_size = dimension
                output_size = output_dimension
                hidden_size = 3*(input_size+output_size)   # Set the hidden size to be the 3 times the input dim + ouput dim
                first_layer_std = config.exp1.first_layer_std
                last_layer_std = config.exp1.last_layer_std


                for _ in range(num_runs):
                
                        # Generate the data
                        X, Y = GaussianIID(dimension,
                                           num_samples,
                                           variance,
                                           k,
                                           epsilon,
                                           output_dimension)

                        # Initialize the Model
                        net = LinearNet(num_hidden_layers, 
                                        input_size, 
                                        hidden_size,
                                        output_size,
                                        first_layer_std,
                                        last_layer_std,)

                        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

                        net = TrainNet(net, X.t(), Y, optimizer, target_loss, max_epochs)


                        weights = net.weights()

                        OLS = torch.matmul(torch.matmul(X.t(),torch.inverse(torch.matmul(X,X.t()))),Y)

                        norm = torch.norm(weights-OLS)
                        print('Norm is {}'.format(norm))
                        return norm




