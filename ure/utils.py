from ure.vocabulary import Vocabulary
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import numbers
import math
import sys, io


############################## removing stopwords #######################

STOPWORDS = set(
    ['all', 'whoever', 'anyway', 'four', 'mill', 'find', 'seemed', 'whose', 're', 'herself', 'enough', 'should', 'to',
     'only', 'under', 'herein', 'do', 'his', 'meanwhile', 'very', 'de', 'myself', 'cannot', 'every', 'yourselves',
     'him', 'is', 'did', 'these', 'she', 'where', 'ten', 'thin', 'namely', 'besides', 'are', 'further', 'best', 'even',
     'what', 'please', 'couldnt', 'behind', 'above', 'between', 'new', 'neither', 'ever', 'can', 'we', 'full', 'never',
     'however', 'here', 'others', 'alone', 'along', 'fifteen', 'both', 'last', 'many', 'whereafter', 'wherever',
     'against', 'etc', 'amount', 'whole', 'otherwise', 'among', 'via', 'co', 'afterwards', 'seems', 'whatever', 'hers',
     'moreover', 'throughout', 'yourself', 'from', 'would', 'two', 'been', 'next', 'whom', 'much', 'dont', 'therefore',
     'themselves', 'thru', 'until', 'empty', 'more', 'fire', 'am', 'hereby', 'else', 'everywhere', 'known', 'former',
     'those', 'must', 'me', 'none', 'this', 'will', 'while', 'anywhere', 'nine', 'three', 'theirs', 'my', 'at',
     'almost', 'sincere', 'thus', 'it', 'cant', 'itself', 'something', 'in', 'ie', 'if', 'end', 'perhaps', 'six', 's',
     'same', 'wherein', 'beside', 'how', 'several', 'whereas', 'see', 'may', 'after', 'upon', 'hereupon', 'such', 'a',
     'off', 'whereby', 'third', 'nevertheless', 'well', 'st', 'rather', 'without', 'so', 'the', 'con', 'yours', 'just',
     'less', 'being', 'indeed', 'over', 'years', 'front', 'already', 'through', 'during', 'fify', 'still', 'its',
     'before', 'thence', 'somewhere', 'had', 'except', 'ours', 'has', 'might', 'into', 'then', 'them', 'someone',
     'around', 'thereby', 'five', 'they', 'not', 'using', 'now', 'nor', 'hereafter', 'always', 'whither', 'either',
     'each', 'found', 'side', 'therein', 'twelve', 'because', 'often', 'doing', 'eg', 'some', 'back', 'year', 'our',
     'beyond', 'ourselves', 'out', 'for', 'bottom', 'since', 'forty', 'per', 'everything', 'does', 'thereupon', 'be',
     'whereupon', 'nowhere', 'although', 'sixty', 'anyhow', 'by', 'on', 'about', 'anything', 'of', 'could', 'whence',
     'due', 'ltd', 'hence', 'or', 'first', 'own', 'seeming', 'formerly', 'thereafter', 'within', 'one', 'down',
     'everyone', 'another', 'thick', 'your', 'i', 'her', 'eleven', 'twenty', 'top', 'there', 'system', 'least', 't',
     'anyone', 'their', 'too', 'hundred', 'was', 'himself', 'elsewhere', 'mostly', 'that', 'nobody', 'amongst',
     'somehow', 'part', 'with', 'than', 'he', 'whether', 'up', 'us', 'whenever', 'below', 'un', 'were', 'toward', 'and',
     'sometimes', 'few', 'beforehand', 'mine', 'an', 'as', 'sometime', 'amoungst', 'have', 'seem', 'any', 'fill',
     'again', 'hasnt', 'no', 'latter', 'when', 'detail', 'also', 'other', 'which', 'latterly', 'you', 'towards',
     'though', 'who', 'most', 'eight', 'but', 'nothing', 'why', 'don', 'noone', 'later', 'together', 'serious', 'inc',
     'having', 'once', '\'', '\'\''])


def is_important_word(s):
    try:
        if len(s) <= 1 or s.lower() in STOPWORDS:
            return False
        float(s)
        return False
    except:
        return True


def is_stopword(s):
    return s.lower() in STOPWORDS


################################ Similarity #########################

class Sim:
    @staticmethod
    def apply(N, M, batch=False, method='l2'):
        if method == 'l2':
            return -Sim.l2(N, M, batch)
        else:
            assert(False)

    @staticmethod
    def l2(N, M, batch=False):
        if batch:
            dist = (N - M).pow(2).sum(dim=1).sqrt()
        else:
            dist = (N.unsqueeze(dim=1) - M.unsqueeze(dim=0)).pow(2).sum(dim=2).sqrt()
        return dist


############################### coloring ###########################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def tokgreen(s):
    return bcolors.OKGREEN + s + bcolors.ENDC


def tfail(s):
    return bcolors.FAIL + s + bcolors.ENDC


def tokblue(s):
    return bcolors.OKBLUE + s + bcolors.ENDC


############################ process list of lists ###################

def load_voca_embs(voca_path, embs_path, normalization=True, add_pad_unk=True, lower=False, digit_0=False):
    voca, added = Vocabulary.load(voca_path, normalization=normalization, add_pad_unk=add_pad_unk, lower=lower, digit_0=digit_0)
    embs = np.load(embs_path)

    print('org emb shape', embs.shape)

    # check if sizes are matched
    assert((voca.size() - embs.shape[0]) <= 2)
    for w in added:
        if w == Vocabulary.pad_token:
            pad_emb = np.zeros([1, embs.shape[1]])
            embs = np.append(embs, pad_emb, axis=0)
        elif w == Vocabulary.unk_token:
            unk_emb = np.random.uniform(-1, 1, (1, embs.shape[1]))  # np.mean(embs, axis=0, keepdims=True)
            embs = np.append(embs, unk_emb, axis=0)

    print('new emb shape', embs.shape)
    return voca, embs


def make_equal_len(lists, fill_in=0, to_right=True):
    lens = [len(l) for l in lists]
    max_len = max(1, max(lens))
    if to_right:
        if fill_in is None:
            eq_lists = [l + [l[-1].copy() if isinstance(l[-1], list) else l[-1]] * (max_len - len(l)) for l in lists]
        else:
            eq_lists = [l + [fill_in] * (max_len - len(l)) for l in lists]
        mask = [[1.] * l + [0.] * (max_len - l) for l in lens]
    else:
        if fill_in is None:
            eq_lists = [[l[0].copy() if isinstance(l[0], list) else l[0]] * (max_len - len(l)) + l for l in lists]
        else:
            eq_lists = [[fill_in] * (max_len - len(l)) + l for l in lists]
        mask = [[0.] * (max_len - l) + [1.] * l for l in lens]

    return eq_lists, mask

################################## utils for pytorch ############################

def embedding_bag_3D(embs, ids, mode='sum'):
    """
    embs = bachsize x n x dim
    ids = batchsize x m x k
    for i in batch:
        output[i] = embedding_bag(ids[i], embs[i], mode)  # k x dim
    """
    batchsize, n, dim = embs.shape
    assert(batchsize == ids.shape[0])

    ids_flat = ids + Variable(torch.linspace(0, batchsize-1, steps=batchsize).long() * n).view(batchsize, 1, 1).cuda()
    ids_flat = ids_flat.view(batchsize * ids.shape[1], -1)
    embs_flat = embs.view(batchsize * n, dim)
    output_flat = nn.functional.embedding_bag(ids_flat, embs_flat, mode=mode)
    output = output_flat.view(batchsize, ids.shape[1], dim)
    return output


def embedding_3D(embs, ids, mode='sum'):
    """
    embs = bachsize x n x dim
    ids = batchsize x k
    for i in batch:
        output[i] = embedding(ids[i], embs[i])  # k x dim
    """
    batchsize, n, dim = embs.shape
    assert(batchsize == ids.shape[0])

    ids_flat = ids + Variable(torch.linspace(0, batchsize-1, steps=batchsize).long() * n).view(batchsize, 1).cuda()
    ids_flat = ids_flat.view(batchsize * ids.shape[1])
    embs_flat = embs.view(batchsize * n, dim)
    output_flat = nn.functional.embedding(ids_flat, embs_flat)
    output = output_flat.view(batchsize, ids.shape[1], dim)
    return output

def cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def to_long_tensor(x):
    return Variable(cuda(torch.LongTensor(x)), requires_grad=False)


def to_float_tensor(x):
    return Variable(cuda(torch.FloatTensor(x)), requires_grad=False)


class BaseModel(nn.Module):

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        params_memory = sum([sys.getsizeof(p) for p in model_parameters])
        print("""
            ---------------- summary ----------------
            Named modules: {}
            Trainable parameters: {}, {:.5f} MB, {}
            Model: {}
        """.format(
            [k for k, v in self.named_parameters()],
            params, params*32*1.25e-7, params_memory,
            self
        ))


def get_frequency(path, voca, power=0.75):
    freq = np.zeros(voca.size())
    with open(path, 'r') as f:
        for line in f:
            w, f = line.strip().split('\t')
            w = w.strip()
            w_id = voca.get_id(w)
            f = int(f)
            freq[w_id] = f ** power
        freq = freq / np.sum(freq)
    return freq



def get_etype_pair(path):
    etypes = []
    with open(path, 'r') as f:
        for line in f:
            etypes.append(line.strip())
    etype_pairs = []
    for h in etypes:
        for t in etypes:
            etype_pairs.append((h, t))
    voca_etype_pair = Vocabulary.load_from_list(
        etype_pairs, normalization=False, add_pad_unk=False)
    return voca_etype_pair


def get_etype_with_subjobj(path):
    etypes = []
    with open(path, 'r') as f:
        for line in f:
            etypes.append(line.strip())
    etype_with_subjobj = []
    for h in etypes:
        etype_with_subjobj.append('{}-SUBJ'.format(h))
        etype_with_subjobj.append('{}-OBJ'.format(h))
    voca_etype_with_subjobj = Vocabulary.load_from_list(
        etype_with_subjobj, normalization=False, add_pad_unk=False)
    return voca_etype_with_subjobj


def load_feature_ids_to_keep(feature_dict_path, ignored_features):
    # self.feattype_id2type = [
    #     "trigger#", 
    #     "entityTypes#",
    #     "arg1_lower#", 
    #     "arg2_lower#",
    #     "bow_clean#", 
    #     "entity1Type#", 
    #     "entity2Type#",
    #     "lexicalPattern#", 
    #     "posPatternPath#"
    # ]
    # self.feattype_type2id = dict(
    #     [(x, i) for i, x in enumerate(self.feattype_id2type)])
    feature_ids_to_keep = []
    f = io.open(feature_dict_path, "r", encoding='utf-8', errors='ignore')
    for line in f:
        token = line.strip()
        if any(token.startswith(prefix) for prefix in ignored_features):
            feature_ids_to_keep.append(False)
        else:
            feature_ids_to_keep.append(True)
    f.close()
    return feature_ids_to_keep

