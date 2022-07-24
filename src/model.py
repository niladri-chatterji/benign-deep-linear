import torch
from torch import nn
from typing import Any, Optional

class LinearNet(nn.Module):
    '''
    Deep Linear Networks. Setting num_layers = -1 gives a standard linear model
    '''
    def __init__(self, 
                num_layers: int , 
                input_size: int , 
                hidden_size: int ,
                output_size: int,
                first_layer_std: Optional[float] = None,
                last_layer_std:  Optional[float] = None):

        self.num_layers = num_layers
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.first_layer_std = first_layer_std
        self.last_layer_std = last_layer_std

        super().__init__()

        if num_layers == -1:
            # Specifies a standard linear model
            self.layers = nn.Sequential(nn.Linear(input_size,output_size, bias = False))
            # Initialize to zero
            with torch.no_grad():
                self.layers[0].weight.data = torch.zeros(self.layers[0].weight.size())
        
        else:
            layers_list = []
            
            layers_list.append(nn.Linear(input_size,hidden_size, bias = False))  #first layer

            for _ in range(num_layers):
                layers_list.append(nn.Linear(hidden_size,hidden_size, bias = False))  # hidden layers
            
            layers_list.append(nn.Linear(hidden_size,output_size, bias = False))  # final layer
            
            self.layers = nn.Sequential(*layers_list)

            # Initialize the network
            if self.first_layer_std is not None or self.last_layer_std is not None:
                assert self.first_layer_std is not None and self.last_layer_std is not None
                self.initialize()

    def initialize(self):
        '''
        Initializes such that the first layer weights are drawn from N(0, first_layer_std)
        and the last layer weights are drawn from N(0, last_layer_std).
        The rest of the weights are initialized to the identity matrix.         
        '''
        assert self.num_layers != -1
        
        with torch.no_grad():
            self.layers[0].weight.data = self.first_layer_std*torch.randn(self.layers[0].weight.size())
        
            self.layers[-1].weight.data = self.last_layer_std*torch.randn(self.layers[0].weight.size())

            # Initialize hidden layers to identity
            for i in range(self.num_layers):
                self.layers[i+1].weight.data = torch.eye(self.layers[i+1].weight.size()[0])

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

    def weights(self):
        '''
        Returns the effective weight matrix with shape [input_size, output_size]
        '''
        weight_matrix = None 
        with torch.no_grad():
            
            if self.num_layers == -1:
                weight_matrix = self.layers[0].weight.data
            
            else:
                for i in range(self.num_layers+2):
                    if weight_matrix is None:
                        weight_matrix = self.layers[0].weight.data
                    else:
                        weight_matrix = torch.matmul( self.layers[i].weight.data,weight_matrix)
        
        return weight_matrix.t()
