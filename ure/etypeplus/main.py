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
from ure.train_eval import load_vocabularies, train, test
from ure.etypeplus.encoder import Encoder
from ure.rel_dist import RelDist
from ure.vocabulary import Vocabulary


def eval_etype(data):
    gold = []
    pred = []
    for item in data:
        pred.append(item["etype_pair"])
        gold.append(item["rel"])
        
    p, r, f1 = scorer.bcubed_score(gold, pred)
    print('b3: p={:.5f} r={:.5f} f1={:.5f}'.format(p, r, f1))
    homo, comp, v_m = scorer.v_measure(gold, pred)
    print('V-measure: hom.={:.5f} com.={:.5f} vm.={:.5f}'.format(homo, comp, v_m))
    ari = scorer.adjusted_rand_score(gold, pred)
    print('ARI={:.5f}'.format(ari))
    return f1



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
        mask_entity_type=False)
    dataset.load(_format='txt')

    # create model
    print('create model')
    n_rels = config["n_rels"]
    print('N relations = {}'.format(n_rels))
    if config["mode"] == "etype":
        f1 = eval_etype(dataset.test)
    elif config["mode"] in ['train', 'tune']:
        model = RelDist(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_etype_with_subjobj': vocas['etype_with_subjobj'].size(),
            'encoder_class': Encoder
        })
        model = utils.cuda(model)
        model.summary()
        train(model, dataset, config)
    else:
        model = Encoder(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'n_etype_with_subjobj': vocas['etype_with_subjobj'].size(),
            'n_filters': config["n_filters"]
        })
        model = utils.cuda(model)
        model.summary()
        model.eval()
        test(model, dataset, config)
