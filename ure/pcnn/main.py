import os
import pickle
import random
from collections import Counter
import ure.scorer as scorer

import torch
import torch.optim as optim

import numpy as np

import ure.utils as utils
from ure.dataset import TSVDataset
from ure.hyperparams import parse_args
from ure.train_eval import load_vocabularies, train
from ure.pcnn.encoder import Encoder
from ure.rel_dist import RelDist
from ure.vocabulary import Vocabulary


if __name__ == '__main__':
    config = parse_args()
    print(config)
    vocas = load_vocabularies(config)

    # load dataset
    datadirs = {
        "train": config["train_path"],
        "dev": config["dev_path"],
        "test": config["test_path"]
    }

    k_samples = config["k_samples"]
    max_len = config["max_len"]
    freq_scale = config["freq_scale"]

    dataset = TSVDataset(
        datadirs, vocas=vocas,
        k_samples=k_samples, max_len=max_len,
        mask_entity_type=True)
    dataset.load(_format='txt')

    # create model
    print('create model')
    n_rels = config["n_rels"]
    print('N relations = {}'.format(n_rels))
    if config["mode"] in ['train', 'tune']:
        model = RelDist(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_filters': config["n_filters"],
            'n_words': vocas['word'].size(),
            'word_embdim': config["word_embdim"],
            'encoder_class': Encoder
        })
        model = utils.cuda(model)
        model.summary()
        train(model, dataset, config)
    else:
        model = Encoder(config={
            'n_rels': n_rels,
            'n_words': vocas['word'].size(),
            'word_embdim': config["word_embdim"],
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_filters': config["n_filters"]
        })
        model = utils.cuda(model)
        model.summary()
        model.eval()
        test(model, dataset, config)
