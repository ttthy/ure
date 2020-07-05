import io
import json
import random
from ure.vocabulary import Vocabulary, UNK_TOKEN, PAD_TOKEN
import ure.utils as utils
import torch
from torch.autograd import Variable
from ure.alias_sampling import alias_setup, alias_draw
from collections import defaultdict


class Dataset:
    def __init__(self, data_path, vocas, k_samples=5, feature_ids_to_keep=[]):
        self.vocas = vocas
        self.data_path = data_path
        self.k_samples = k_samples
        self.feature_dim = self.vocas['feature'].size()

        # Require to provide entity frequency
        # if 'ent_freq' in vocas:
        J, q = alias_setup(vocas['ent_freq'])
        self.alias_params = {'J': J, 'q': q}

        self.feature_ids_to_keep = feature_ids_to_keep

    def read_from_file(self, path, _format='txt'):
        raise NotImplementedError("Need to implement")

    def load(self, _format='txt'):
        try:
            print('load train set')
            self.train = self.read_from_file(
                self.data_path['train'], _format=_format)
        except Exception as e:
            print(e)
            print('no train set to load')
            pass
        try:
            print('load dev set')
            self.dev = self.read_from_file(
                self.data_path['dev'], _format=_format)
        except Exception as e:
            print(e)
            print('no dev set to load')
            pass
        try:
            print('load test set')
            print('test data', self.data_path['test'])
            self.test = self.read_from_file(
                self.data_path['test'], _format=_format)
        except Exception as e:
            print(e)
            print('no test set to load')
            pass

    def get_minibatch(self, data, start, end):
        batch = data[start:end]

        _input = defaultdict(list)

        feat_offsets = []
        feat_features = []
        for item_id, item in enumerate(batch):
            _input['head_ent'].append(item['head_ent'])
            _input['tail_ent'].append(item['tail_ent'])
            _input['rel'].append(item['rel'])
            feat_offsets.append(len(feat_features))
            feat_features.extend(item['features'])

        _input['features'] = utils.to_long_tensor(feat_features)
        _input['feature_offsets'] = utils.to_long_tensor(feat_offsets)            
        
        # Entity mentions
        _input['head_ent'] = utils.to_long_tensor(_input['head_ent'])
        _input['tail_ent'] = utils.to_long_tensor(_input['tail_ent'])

        # Relations
        _input['rel'] = utils.to_long_tensor(_input['rel'])

        # sampling entities
        for name in ['head', 'tail']:
            _input['{}_ent_samples'.format(name)] = utils.to_long_tensor(
                    [alias_draw(self.alias_params['J'], self.alias_params['q'])
                        for _ in range(self.k_samples * (end - start))]
            )
            _input['{}_ent_samples'.format(name)] = _input['{}_ent_samples'.format(name)].view(end-start, -1)

        return _input


class TSVDataset(Dataset):
    def __init__(self, data_path, vocas, k_samples=5, feature_ids_to_keep=[]):
        super(TSVDataset, self).__init__(
            data_path=data_path, vocas=vocas, k_samples=k_samples, 
            feature_ids_to_keep=feature_ids_to_keep)
        self.data_path = data_path

    def read_txt(self, path):
        print('Reading data from path', path)
        with open(path, 'r') as f:
            data = []
            npasses = 0

            for (i, line) in enumerate(f):
                try:
                    relation, head_ent, tail_ent, features = line.rstrip().split('\t')
                except ValueError:
                    print('Line', i, '>>>>', line)
                    continue
                features = [int(x) for x in features.split(' ')]
                
                item = {
                    'head_ent': head_ent.strip(),
                    'tail_ent': tail_ent.strip(),
                    'rel': relation.strip(),
                    'features': features
                    }
                if len(features) > 0:
                    data.append(item)
                    npasses += 1
                if int(i+1) % 1e2:
                    print(i, npasses, end='\r')
        print('load %d raw items' % len(data))
        return data

    def read_from_file(self, path, _format='txt'):
        data = []

        if _format == 'txt':
            raw_data = self.read_txt(path)
        else:
            raise NotImplementedError

        count = 0
        for _i, item in enumerate(raw_data):
            relation = item['rel']
            head_ent, tail_ent = item['head_ent'], item['tail_ent']

            head_ent_id = self.vocas['entity'].get_id(head_ent)
            if head_ent_id == self.vocas['entity'].unk_id:
                head_ent_id = _i * 2
            tail_ent_id = self.vocas['entity'].get_id(tail_ent)
            if tail_ent_id == self.vocas['entity'].unk_id:
                tail_ent_id = _i * 2 + 1

            try:
                rel_id = self.vocas['relation'].get_id(relation)
            except:
                rel_id = 0

            features = item['features']
            if len(self.feature_ids_to_keep) > 0:
                features = [x for x in features 
                            if self.feature_ids_to_keep[x]]
            
            data.append({
                'head_ent': head_ent_id,
                'tail_ent': tail_ent_id,
                'rel': rel_id,
                'features': features})
            count += 1

            if count % 1000 == 0:
                print('load %d items' % count, end='\r')
        print('Total raw lines:', _i)
        print('load', len(data), 'triples')

        return data
