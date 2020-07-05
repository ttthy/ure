import torch
import torch.nn as nn
import ure.utils as utils
from ure.dataset import MAX_POS
import math
import sys
import random


class Encoder(utils.BaseModel):
    def __init__(self, config):
        super().__init__()

        # word and position embeddings
        self.word_embs = nn.Embedding(config['n_words'], config['word_embdim'])
        self.word_embs.weight.data.uniform_(-0.001, 0.001)

        input_dim = config['word_embdim']

        self.convs = nn.ModuleDict([
            ['conv_{}'.format(p),
                nn.Conv1d(input_dim, config['n_filters'], kernel_size=3, padding=1)]
            for p in ['l', 'c', 'r']
        ])
        self.sen_dim = config['n_filters']
        self.classifier = nn.Linear(self.sen_dim, config['n_rels'])

    def encode_sentence(self, _input):
        B = _input['tokens'].shape[0]
        output = {}
        for p in ['l', 'c', 'r']:
            inp = self.word_embs(_input['tokens_{}'.format(p)])
            # filtering out padding
            mask = _input['pcnn_mask_'+p].unsqueeze(dim=2)
            inp = inp * mask
            inp = inp.permute(0, 2, 1)
            # n_sents x len x n_filter
            conved = self.convs['conv_{}'.format(p)](inp).permute(0, 2, 1)
            conved = conved * mask - (1 - mask) * 1e10
            # max pooling
            pooled = torch.tanh(torch.max(conved, 1)[0])
            output[p] = pooled
        
        sent_embs = (output['l'] + output['c'] + output['r']) / 3

        return sent_embs

    def forward(self, _input):
        sent_embs = self.encode_sentence(_input)
        logits = self.classifier(sent_embs)
        return logits

    def predict_relation(self, _input):
        logits = self.forward(_input)
        return nn.functional.softmax(logits, dim=1)

