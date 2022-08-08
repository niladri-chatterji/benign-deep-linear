import logging
import os
import pickle
import pdb

import torch
import torch.optim as optim
import numpy as np

from ..utils import TrainNet, loss, loss_net
from ..data import PlantedTwoLayerIID
from ..model import ReLUNet


def exp3(args, config):
        '''
        Experiment that compares the norm different between Theta and Theta_OLS and calculates the excess risk. Sweep over different init scales alpha while dimension is kept fixed
        '''
        logging.info('Starting the experiment')
        logging.info('Loading Parameters')
        # Parameters for the data
        num_runs = config.exp.num_runs
        dimension = config.exp.dimension
        variance = config.exp.variance
        output_dimension = config.exp.output_dimension
        width_teacher = config.exp.width_teacher
        width_student = config.exp.width_student

        # Parameters for the model
        num_hidden_layers = config.exp.num_hidden_layers
        start_alpha = config.exp.start_first_layer_std
        end_alpha = config.exp.end_first_layer_std
        width_student = config.exp.width_student

        alpha_range = list(np.linspace(start_alpha,
                                        end_alpha,
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
        logging.info('Starting a sweep over the alpha')
        
        results = {}
        
        for idx, first_layer_std in enumerate(alpha_range):
                logging.info('Current alpha is {}'.format(first_layer_std))
                logging.info('Dimension is {}'.format(dimension))
                results[idx] = {}
                results[idx]["alpha"] = first_layer_std
                results[idx]["dimension"] = dimension
                results[idx]["distance"] = []
                results[idx]["risk"] = []
                results[idx]["risk_teacher"] = []
                results[idx]["training_loss"] = []

                

                num_samples = config.exp.num_samples
                k = config.exp.k
                epsilon = config.exp.epsilon

                # The details of the model
                input_size = dimension
                output_size = output_dimension
                hidden_size = width_student
                last_layer_std = config.exp.last_layer_std


                for run_idx in range(num_runs):
                        logging.info('Run {}'.format(run_idx))
                
                        # Generate the data
                        X, Y, net_teacher = PlantedTwoLayerIID(dimension = input_size,
                                                        num_samples = num_samples,
                                                        output_dimension = output_size,
                                                        variance = variance,
                                                        width = width_teacher,
                                                        k = k, 
                                                        epsilon = epsilon,
                                                        device = device)
                        

                        # Initialize the Model
                        net = ReLUNet(num_hidden_layers = num_hidden_layers, 
                                        input_size = input_size, 
                                        hidden_size = hidden_size,
                                        output_size = output_size,
                                        first_layer_std = first_layer_std,
                                        last_layer_std = last_layer_std,).to(device)

                        pdb.set_trace()

                        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.0)
                        logging.info("Training Starting")
                        net = TrainNet(net, X, Y, optimizer, target_loss, max_epochs)
                        logging.info("Training Complete")

                        # Calculate the excess risk
                        X_test, Y_test, _ = PlantedTwoLayerIID(dimension = input_size,
                                                        num_samples = num_samples,
                                                        output_dimension = output_size,
                                                        variance = variance,
                                                        width = width_teacher,
                                                        k = k, 
                                                        epsilon = epsilon,
                                                        noise_variance = 0.0,
                                                        net = net_teacher,
                                                        device = device)
                        
                        X_test, Y_test = X_test.to(device), Y_test.to(device)

                    
                        risk = loss_net(X_test, Y_test, net)
                        risk_teacher = loss_net(X_test, Y_test, net_teacher)
                        training_loss = loss_net(X, Y, net)
                        
                        pdb.set_trace()
                        #logging.info('Distance between net weights and OLS: {}'.format(distance))
                        logging.info('Risk of Net: {}'.format(risk))
                        logging.info('Risk of Teacher: {}'.format(risk_teacher))
                        logging.info('Training loss of Net: {}'.format(training_loss))
                        logging.info('Training loss of Teacher: {}'.format(loss_net(X,Y,net_teacher)))
                        

                        #results[idx]["distance"].append(distance)
                        results[idx]["risk"].append(risk)
                        results[idx]["risk_teacher"].append(risk_teacher)
                        results[idx]["training_loss"].append(training_loss)

        

        with open(output_file, "wb") as f:
                pickle.dump(results, f)

        return None