import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np


import src.experiments as exp


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="./results", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--exp1",
        type=str,
        default=False,
        help="Run Experiment 1 that calculates the norm of theta_OLS - theta and excess risk while sweeping over different dimensions",
    )
    parser.add_argument(
        "--exp2",
        type=str,
        default=False,
        help="Run Experiment 2 that calculates the norm of theta_OLS - theta and excess risk while sweeping over different init scales alpha",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=False,
        help="override the value of alpha (first_layer_std) in the config file",
    )

    parser.add_argument(
        "--dimension",
        type=float,
        default=False,
        help="override the value of dimension (input_size) in the config file",
    )
    
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # If alpha or dimension are provided in the argument then override the default in the config file
    if args.alpha is not None:
        new_config.exp.first_layer_std = args.alpha
    if args.dimension is not None:
        new_config.exp.dimenion = args.dimension

    # Create logging path and save config fgile
    if os.path.exists(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    with open(os.path.join(args.log_path, "config.yml"), "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))

    try:
        # Run the right experiments
        if args.exp1:
            exp.exp1.exp1(args, config)
        elif args.exp2:
            exp.exp2.exp2(args, config)
        else:
            logging.info('No experiment provided')
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())