import torch
import torch.nn as nn
import ure.utils as utils


class Encoder(utils.BaseModel):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.feature_emb = nn.EmbeddingBag(
            config['feature_dim'], config['n_rels'], mode='sum')
        self.feature_bias = nn.Parameter(torch.Tensor(config['n_rels']))
        self.init()

    def init(self):
        self.feature_emb.weight.data.uniform_(-0.001, 0.001)
        self.feature_bias.data.fill_(0.0)

    def forward(self, _input):
        logits = self.feature_emb(
            _input['features'], offsets=_input['feature_offsets'])
        logits += self.feature_bias
        return logits

    def predict_relation(self, _input):
        logits = self.forward(_input)
        return nn.functional.softmax(logits, dim=1)
