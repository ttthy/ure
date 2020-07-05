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
from ure.vocabulary import Vocabulary


def load_vocabularies(config):
    freq_scale = config["freq_scale"]

    # load word embeddings and vocabular
    print('load words and entities')
    voca_entity, _ = Vocabulary.load(
        config["entity_path"], normalization=False, add_pad_unk=False)
    print('Vocab entity size', voca_entity.size())
    ent_freq = utils.get_frequency(
        config["entity_frequency_path"], voca_entity, power=freq_scale)

    voca_word, _ = Vocabulary.load(config["word_path"])
    voca_etype, _ = Vocabulary.load(
        config["entity_type_path"], 
        normalization=False, add_pad_unk=False)
    print('Vocab entity type size', voca_etype.size())
    voca_etype_pair = utils.get_etype_pair(config["entity_type_path"])
    print('Vocab entity type pair size', voca_etype_pair.size())
    
    voca_etype_with_subjobj = utils.get_etype_with_subjobj(
        config["entity_type_path"])
    print('Vocab entity type with subjobj size', voca_etype_with_subjobj.size())

    voca_relation, _ = Vocabulary.load(
        config["relation_path"], normalization=False, add_pad_unk=False)
    print('Vocab relation', voca_relation.size())

    vocas = {
            'entity': voca_entity,
            'ent_freq': ent_freq,
            'relation': voca_relation,
            'word': voca_word,
            'etype': voca_etype,
            'etype_pair': voca_etype_pair,
            'etype_with_subjobj': voca_etype_with_subjobj
    }

    return vocas


# for training
def train(model, dataset, config):
    params = [v for v in model.parameters() if v.requires_grad]

    optimizer = optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=pow(0.5, 0.25))

    data = dataset.train

    dev_best = -1
    not_inc_count = 0

    evaluate(
        model, dataset=dataset, batchsize=config["batchsize"], data=dataset.dev)

    for e in range(config["n_epochs"]):
        if not_inc_count >= config["patience"]:
            print('over patience %d, stop' % config["patience"])
            break

        print('------------------------- epoch %d --------------------------' % (e))
        random.shuffle(data)
        model.train()
        start = end = 0
        total_loss = 0

        while True:
            if start >= len(data):
                print('\n%.6f\t\t\t\t\t\t' % (total_loss / len(data)))
                break

            end = min(start + config["batchsize"], len(data))
            _input = dataset.get_minibatch(data, start, end)
            _input['loss_coef'] = {
                'alpha': config["loss_coef_alpha"],
                'beta': config["loss_coef_beta"]
            }

            optimizer.zero_grad()
            loss, loss_details = model(_input)
            loss.backward()
            optimizer.step()
            loss = loss.data.cpu().item()
            total_loss += loss * (end - start)
            print("{:d}\tloss={:.6f}  {:s}   total_loss={:.6f}\t\t\t\t".format(
                end, loss,
                ' '.join(['{}={:.5f}'.format(k, v.data.cpu().item())
                          for k, v in loss_details.items()]),
                total_loss),
                end='\r' if random.random() < 0.9995 else '\n')

            start = end

        scheduler.step()
        print('-'*30, 'dev')
        s = evaluate(model, dataset=dataset, batchsize=config["batchsize"], data=dataset.dev)
        if s > dev_best:
            dev_best = s
            not_inc_count = 0
            print('new highest score', s)
            torch.save(
                model.state_dict(), 
                os.path.join(config["model_path"], 'best_dev.pth'))
        else:
            not_inc_count += 1
            print('previous best', dev_best)
            print('not increase count', not_inc_count)

    return dev_best


def evaluate(model, dataset, batchsize=100, data=None, printing=True):
    start = end = 0

    model.eval()
    gold = []
    pred = []

    while True:
        if start >= len(data):
            break

        end = min(start + batchsize, len(data))
        _input = dataset.get_minibatch(data, start, end)
        predictions = model.predict_relation(_input)
        if start < 1 and printing:
            print(predictions[:3])
        predictions = torch.argmax(predictions, dim=1)

        pred.extend(predictions.data.cpu().tolist())
        gold.extend(_input['rel'].data.cpu().tolist())

        start = end

    if printing:
        print('pred', Counter(pred))
        print('First 20 predictions', pred[:20])
        print('batch-rel', gold[:20])
    p, r, f1 = scorer.bcubed_score(gold, pred)
    print('b3: p={:.5f} r={:.5f} f1={:.5f}'.format(p, r, f1))
    homo, comp, v_m = scorer.v_measure(gold, pred)
    print('V-measure: hom.={:.5f} com.={:.5f} vm.={:.5f}'.format(homo, comp, v_m))
    ari = scorer.adjusted_rand_score(gold, pred)
    print('ARI={:.5f}'.format(ari))
    return f1


def test(model, dataset, config):
    model_dict = model.state_dict()
    load_dict = torch.load(os.path.join(config["model_path"], 'best_dev.pth'))
    pretrained_dict = {k: load_dict['encoder.{}'.format(k)] for k, v in model_dict.items() if 'encoder.{}'.format(k) in load_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    s = evaluate(model, dataset=dataset, batchsize=config["batchsize"], data=dataset.test, printing=False)
    return s
