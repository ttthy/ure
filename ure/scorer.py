from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
import torch
import numpy as np


def bcubed_correctness(gold, pred, na_id=-1):
    # remove NA
    gp = [(x,y) for x, y in zip(gold, pred) if x != na_id]
    gold = [x for x,_ in gp]
    pred = [y for _,y in gp]

    # compute 'correctness'
    l = len(pred)
    assert(len(gold) == l)
    gold = torch.IntTensor(gold)
    pred = torch.IntTensor(pred)
    gc = ((gold.unsqueeze(0) - gold.unsqueeze(1)) == 0).int()
    pc = ((pred.unsqueeze(0) - pred.unsqueeze(1)) == 0).int()
    c = gc * pc
    return c, gc, pc


def bcubed_precision(c, gc, pc):
    pcsum = pc.sum(1)
    total = torch.where(pcsum > 0, pcsum.float(), torch.ones(pcsum.shape))
    return ((c.sum(1).float() / total).sum() / gc.shape[0]).item()


def bcubed_recall(c, gc, pc):
    gcsum = gc.sum(1)
    total = torch.where(gcsum > 0, gcsum.float(), torch.ones(gcsum.shape))
    return ((c.sum(1).float() / total).sum() / pc.shape[0]).item()


def bcubed_score(gold, pred, na_id=-1):
    c, gc, pc = bcubed_correctness(gold, pred, na_id)
    prec = bcubed_precision(c, gc, pc)
    rec = bcubed_recall(c, gc, pc)
    return prec, rec, 2 * (prec * rec) / (prec + rec)


def v_measure(gold, pred):
    homo = homogeneity_score(gold, pred)
    comp = completeness_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    return homo, comp, v_m


def check_with_bcubed_lib(gold, pred):
    import bcubed
    ldict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(gold)])
    cdict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(pred)])

    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)

    print('P={} R={} F1={}'.format(precision, recall, fscore))


if __name__ == '__main__':
    gold = [0, 0, 0, 0, 0, 1, 1, 2, 1, 3, 4, 1, 1, 1]
    pred = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

    print(bcubed_score(gold, pred, na_id = 1000), 'should be 0.69')

    check_with_bcubed_lib(gold, pred)
    homo = homogeneity_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    ari = adjusted_rand_score(gold, pred)
    print(homo, v_m, ari)

