import os
import pickle
import random
from collections import Counter
import ure.scorer as scorer

import torch
import torch.optim as optim

import numpy as np

import ure.utils as utils
from ure.feature.dataset import TSVDataset
from ure.feature.hyperparams import parse_args
from ure.feature.encoder import Encoder
from ure.rel_dist import RelDist
from ure.vocabulary import Vocabulary


# for training
def train(model, dataset, config):

    # create optimizer
    _params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    weight_decay_params = ['encoder.feature_emb.weight',
                           'link_predictor.sel_pre.weight', 'link_predictor.rescal.weight'
                        ]
    params = [
         {
             'params': [
                 v
                 for k, v in _params
                 if k in weight_decay_params
             ],
             'weight_decay': config["weight_decay"]
         },
         {
             'params': [
                 v
                 for k, v in _params
                 if k not in weight_decay_params
             ],
             'weight_decay': 0.0
         }
    ]

    optimizer = optim.Adagrad(params, lr=config["lr"])

    data = dataset.train

    dev_best = -1
    not_inc_count = 0

    evaluate(
        model, batchsize=config["batchsize"], data=dataset.dev)

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


def evaluate(model, dataset=dataset, batchsize=100, data=None, printing=True):
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


def linkpredictor_evaluate(data=None):
    start = end = 0
    if data is None:
        data = dataset.dev

    model.eval()
    ranks = 0
    total = 0

    while True:
        if start >= len(data):
            break

        end = min(start + config["batchsize"], len(data))
        _input = dataset.get_minibatch(data, start, end)
        _input['loss_coef'] = {'alpha': config["loss_coef_alpha"], 'beta': config["loss_coef_beta"]}
        model(_input)  # psi: [2*B, k]
        psi = model.psi
        #print(psi.shape)
        B2, k = psi.shape
        B = int(B2 / 2)
        sorted_psi = torch.sort(psi, dim=1)[0].view(2, B, -1).cpu().data.numpy() # [2 * B, n_rels]
        psi = psi.cpu().view(2, B, -1).data.numpy()

        for _t, _sorted, _org in zip(_input['tail_ent'].tolist(), sorted_psi[0], psi[0]):
            _t = 0
            r = k - np.searchsorted(_sorted, _org[_t])
            ranks += r
            total += 1

        for _h, _sorted, _org in zip(_input['head_ent'].tolist(), sorted_psi[1], psi[1]):
            _h = 0
            r = k - np.searchsorted(_sorted, _org[_h])
            ranks += r
            total += 1


        start = end

    s = ranks / total
    print('avg rank', s)
    return s


def get_data(config):
    datadirs = {
        "train": config["train_path"],
        "dev": config["dev_path"],
        "test": config["test_path"]
    }

    k_samples = config["k_samples"]
    max_len = config["max_len"]
    freq_scale = config["freq_scale"]
    ignored_features=config["ignored_features"]

    # load word embeddings and vocabulary
    print('load words and entities')
    voca_entity, _ = Vocabulary.load(
        config["entity_path"], normalization=False, add_pad_unk=False)
    print('Vocab entity size', voca_entity.size())
    ent_freq = utils.get_frequency(
        config["entity_frequency_path"], voca_entity, power=freq_scale)

    voca_feature, _ = Vocabulary.load(
        config["feature_path"], normalization=False, add_pad_unk=False)

    feature_ids_to_keep = utils.load_feature_ids_to_keep(
        feature_dict_path=config["feature_path"], ignored_features=ignored_features)
    print("feature_ids_to_keep", type(feature_ids_to_keep), len(feature_ids_to_keep))

    voca_relation, _ = Vocabulary.load(
        config["relation_path"], normalization=False, add_pad_unk=False)
    print('Vocab relation', voca_relation.size())

    vocas = {
            'entity': voca_entity,
            'ent_freq': ent_freq,
            'relation': voca_relation,
            'feature': voca_feature
    }
    # load dataset
    dataset = TSVDataset(
        datadirs, vocas=vocas,
        k_samples=k_samples,
        feature_ids_to_keep=feature_ids_to_keep)
    dataset.load(_format='txt')

    return vocas, dataset


if __name__ == '__main__':
    config = parse_args()
    print(config)
    vocas, dataset = get_data(config)
    config["weight_decay"] = config["weight_decay"] * (float(config["batchsize"]) / len(dataset.train))

    # create model
    print('create model')
    n_rels = config["n_rels"]
    print('N relations = {}'.format(n_rels))
    if config["mode"] in ['train', 'tune']:
        model = RelDist(config={
            'n_rels': n_rels,
            'n_ents': vocas['entity'].size(),
            'ent_embdim': config["ent_embdim"],
            'feature_dim': vocas['feature'].size(),
            'encoder_class': Encoder
        })
        model = utils.cuda(model)
        model.summary()
        train(model, dataset, config)
    else:
        model = Encoder(config={
            'n_rels': n_rels,
            'feature_dim': vocas['feature'].size()
        })
        model = utils.cuda(model)
        model.summary()
        model.eval()
        test(model, dataset, config)
