import argparse
import torch
import numpy as np
import random
import os, time

import yaml

STARTED_DATESTRING = time.strftime("%m-%dT%H-%M-%S",
                                   time.localtime())

def set_seet_every_where(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(args):
    """ Read configuration from file
    Returns:
        dict: configuration
    """
    with open(args.config, 'r') as stream:
        try:
            config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    for arg in args.__dict__:
        if config.get(arg, None) is not None and getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)

    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"])

    for name in ["train", "dev", "test", "entity", 
                "entity_frequency", "word", "entity_type", "relation"]:
        if config["{}_path".format(name)] is not None:
            assert os.path.exists(config["{}_path".format(name)]), \
                "{}_path invalid!\n> {}".format(name, config["{}_path".format(name)])

    config["mode"] = args.mode
    
    if config["ignored_features"] is None:
        config["ignored_features"] = []
    else:
        config["ignored_features"] = config["ignored_features"].split(',')

    return config


def parse_args(a=None):

    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str)
    parser.add_argument("--mode", type=str, help="train or eval", default='train')
    parser.add_argument("--model_path", type=str, help="model path to save/load")

    # Number of relations
    parser.add_argument("--n_rels", type=int, help="number of relations", default=10)

    # Training hyper params
    parser.add_argument("--k_samples", type=int, default=5)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--weight_decay", type=float, help="L2 regularizer", default=0)
    parser.add_argument("--freq_scale", type=float, default=0.75)
    parser.add_argument("--ent_embdim", type=int, default=10)

    # Loss coefficient
    parser.add_argument("--loss_coef_alpha", type=float, default=0.01)
    parser.add_argument("--loss_coef_beta", type=float, default=0.02)

    # Training setting
    parser.add_argument("--batchsize", type=int, help="batchsize", default=100)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=10)
    parser.add_argument("--patience", type=int, default=10,
                        help="number of max not-increasing-performance evaluations")

    # Model hyper params
    parser.add_argument("--ignored_features", type=str, default=None)
    if a is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(a)

    if args.seed is not None:
        set_seet_every_where(args.seed)

    assert os.path.isfile(args.config), "Config path is invalid!"
    return load_config(args)
