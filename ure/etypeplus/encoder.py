import torch
import torch.nn as nn
from ure.dataset import MAX_POS
import ure.utils as utils
import math
import sys
import random


class Encoder(utils.BaseModel):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.n_rels = config['n_rels']
        # entity type embeddings
        self.etype_embs = nn.Embedding(config['n_etype_with_subjobj'], config['n_rels'])
        self.bias = nn.Parameter(torch.Tensor(config['n_rels']))
        self.init()

    def init(self):
        self.etype_embs.weight.data.uniform_(-0.001, 0.001)
        self.bias.data.fill_(0.0)

    def forward(self, _input):
        # [B] -> [B, D]
        head = self.etype_embs(_input['head_etype'])
        tail = self.etype_embs(_input['tail_etype'])
        # [B, 2D]
        # output = torch.cat([head, tail], dim=1)
        # This is equivalent to FFNN: [head,tail]W + b
        logits = head + tail + self.bias
        return logits

    def predict_relation(self, _input):
        logits = self.forward(_input)
        return nn.functional.softmax(logits, dim=1)
