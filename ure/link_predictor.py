import torch
import torch.nn as nn
import ure.utils as utils
import math


class LinkPredictor(utils.BaseModel):

    def __init__(self, config):
        super().__init__()
        self.ent_emb = nn.Embedding(config['n_ents'], config['ent_embdim'])
        self.ent_argument_bias = nn.Embedding(config['n_ents'], 1)
        self.ent_embdim = config['ent_embdim']
        self.n_rels = config['n_rels']

        self.rescal = nn.Bilinear(
            config['ent_embdim'], config['ent_embdim'], config['n_rels'], bias=False)

        self.sel_pre = nn.Linear(
            2*config['ent_embdim'], config['n_rels'], bias=False)

        self.init()

    def init(self):
        self.ent_emb.weight.data.uniform_(-0.01, 0.01)
        self.ent_argument_bias.weight.data.fill_(0.0)
        self.rescal.weight.data.normal_(0, math.sqrt(0.1))
        self.sel_pre.weight.data.normal_(0, math.sqrt(0.1))

    def forward(self, _input):
        # [2Bk] -> [2Bk, D]
        head_emb = self.ent_emb(_input['head_ent'])
        tail_emb = self.ent_emb(_input['tail_ent'])

        # [2Bk, D] bilinear [2Bk, D] -> [2Bk, n_rels]
        rescal = self.rescal(head_emb, tail_emb)
        # [2Bk, 2*D] -> [2Bk, n_rels]
        selectional_preferences = self.sel_pre(torch.cat([head_emb, tail_emb], dim=1))

        # [2Bk, n_rels]
        psi = rescal + selectional_preferences
        head_bias = self.ent_argument_bias(_input['head_ent'])
        tail_bias = self.ent_argument_bias(_input['tail_ent'])

        return psi, head_bias, tail_bias
