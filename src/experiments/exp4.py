import logging
import os
import pickle

import torch
import torch.optim as optim
import numpy as np

from ..utils import TrainNet, loss
from ..data import GaussianIID
from ..model import LinearNet, ReLUNet


def exp4(args, config):
        '''
        Experiment that compares the norm different between Theta and Theta_OLS and calculates the excess risk. Sweep over different init scales beta while dimension is kept fixed
        '''
        logging.info('Starting the experiment')
        logging.info('Loading Parameters')
        # Parameters for the data
        num_runs = config.exp.num_runs
        dimension = int(config.exp.dimension)
        variance = config.exp.variance
        output_dimension = config.exp.output_dimension

        # Parameters for the model
        num_hidden_layers = config.exp.num_hidden_layers
        start_beta = config.exp.start_last_layer_std
        end_beta = config.exp.end_last_layer_std

        beta_range = list(np.linspace(start_beta,
                                        end_beta,
                                        num = config.exp.num_sweep,
                                        endpoint=True))

        # Parameters for the optimizer
        lr = config.optim.lr
        target_loss = config.optim.target_loss
        max_epochs = config.optim.max_epochs

        # Device
        device = config.device
        output_file = os.path.join(args.log_path, "result.pickle")

        logging.info('Parameters Loaded')
        logging.info('Starting a sweep over the beta')
        
        results = {}
        
        for idx, last_layer_std in enumerate(beta_range):
                logging.info('Current beta is {}'.format(last_layer_std))
                logging.info('Dimension is {}'.format(dimension))
                results[idx] = {}
                results[idx]["beta"] = last_layer_std
                results[idx]["dimension"] = dimension
                results[idx]["distance"] = []
                results[idx]["risk"] = []
                results[idx]["risk_ols"] = []
                results[idx]["training_loss"] = []

                

                num_samples = config.exp.num_samples
                k = config.exp.k
                epsilon = config.exp.epsilon

                # The details of the model
                input_size = dimension
                output_size = output_dimension
                hidden_size = 10*(dimension+output_dimension)   # Set the hidden size to be the 10 times the input dim + ouput dim
                first_layer_std = config.exp.first_layer_std


                for run_idx in range(num_runs):
                        logging.info('Run {}'.format(run_idx))
                
                        # Generate the data
                        X, Y, theta = GaussianIID(dimension = dimension,
                                           num_samples = num_samples,
                                           output_dimension = output_dimension,
                                           variance = variance,
                                           k = k, 
                                           epsilon= epsilon,
                                           )
                        X, Y = X.to(device), Y.to(device)

                        # Initialize the Model
                        net = LinearNet(num_hidden_layers = num_hidden_layers, 
                                        input_size = input_size, 
                                        hidden_size = hidden_size,
                                        output_size = output_size,
                                        first_layer_std = first_layer_std,
                                        last_layer_std = last_layer_std,).to(device)

                        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.0)
                        logging.info("Training Starting")
                        net = TrainNet(net, X, Y, optimizer, target_loss, max_epochs)
                        logging.info("Training Complete")
                        weights = net.weights()

                        OLS = torch.matmul(torch.matmul(X.t(),torch.inverse(torch.matmul(X,X.t()))),Y)

                        distance = (torch.norm(weights-OLS)**2).item()



                        # Calculate the excess risk
                        X_test, Y_test, _ = GaussianIID(dimension = dimension,
                                           num_samples = 10*dimension,
                                           output_dimension = output_dimension,
                                           variance = variance,
                                           k = k, 
                                           epsilon= epsilon,
                                           noise_variance=0.0,
                                           theta = theta)
                        
                        X_test, Y_test = X_test.to(device), Y_test.to(device)
                        risk = loss(X_test, Y_test, weights)
                        risk_ols = loss(X_test, Y_test, OLS)
                        training_loss = loss(X, Y, weights)
                        




                        logging.info('Distance between net weights and OLS: {}'.format(distance))
                        logging.info('Risk of Net: {}'.format(risk))
                        logging.info('Risk of OLS: {}'.format(risk_ols))
                        logging.info('Training loss of Net: {}'.format(training_loss))
                        logging.info('Training loss of OLS: {}'.format(loss(X,Y,OLS)))
                        

                        results[idx]["distance"].append(distance)
                        results[idx]["risk"].append(risk)
                        results[idx]["risk_ols"].append(risk_ols)
                        results[idx]["training_loss"].append(training_loss)

        

        with open(output_file, "wb") as f:
                pickle.dump(results, f)

        return None