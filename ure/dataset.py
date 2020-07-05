import io
import json
import random
from ure.vocabulary import Vocabulary, UNK_TOKEN, PAD_TOKEN
import ure.utils as utils
import torch
from torch.autograd import Variable
from ure.alias_sampling import alias_setup, alias_draw
from collections import defaultdict

MAX_POS = 100

class Dataset:
    def __init__(self, data_path, vocas, k_samples=5, max_len=MAX_POS, mask_entity_type=False):
        self.vocas = vocas
        self.data_path = data_path
        self.k_samples = k_samples
        self.mask_entity_type = mask_entity_type
        self.max_len = max_len

        # Require to provide entity frequency
        # if 'ent_freq' in vocas:
        J, q = alias_setup(vocas['ent_freq'])
        self.alias_params = {'J': J, 'q': q}

    def read_from_file(self, path, _format='txt', max_len=MAX_POS):
        raise NotImplementedError("Need to implement")

    def load(self, _format='txt'):
        try:
            print('load train set')
            self.train = self.read_from_file(
                self.data_path['train'], _format=_format, max_len=self.max_len)
        except Exception as e:
            print(e)
            print('no train set to load')
            pass
        
        try:
            print('load dev set')
            self.dev = self.read_from_file(
                self.data_path['dev'], _format=_format, max_len=self.max_len)
        except Exception as e:
            print(e)
            print('no dev set to load')
            pass

        try:
            print('load test set')
            print('test data', self.data_path['test'])
            self.test = self.read_from_file(
                self.data_path['test'], _format=_format, max_len=self.max_len)
        except Exception as e:
            print(e)
            print('no test set to load')
            pass

    def get_minibatch(self, data, start, end):
        batch = data[start:end]

        _input = defaultdict(list)

        for _i, item in enumerate(batch):
            _input['sentence'].append(item['sentence'])
            _input['tokens_str'].append(item['tokens_str'])
            _input['head_ent'].append(item['head_ent'])
            _input['tail_ent'].append(item['tail_ent'])
            _input['head_etype'].append(item['head_etype'])
            _input['tail_etype'].append(item['tail_etype'])
            _input['rel'].append(item['rel'])
            _input['tokens'].append(item['tokens'])
            _input['length'].append(len(item['tokens']))

            for p in ['l', 'c', 'r']:
                _input['tokens_{}'.format(p)].append(item['tokens_{}'.format(p)])

            _input['etype_pair'].append(item['etype_pair'])

        _input['tokens'], _input['masks'] = utils.make_equal_len(
            _input['tokens'], fill_in=self.vocas['word'].pad_id)
        _input['tokens'] = utils.to_long_tensor(_input['tokens'])
        _input['masks'] = utils.to_float_tensor(_input['masks'])
        
        for p in ['l', 'c', 'r']:
            _input['tokens_{}'.format(p)], _input['pcnn_mask_{}'.format(p)] = utils.make_equal_len(
                _input['tokens_{}'.format(p)], fill_in=self.vocas['word'].pad_id)
            _input['tokens_{}'.format(p)] = utils.to_long_tensor(_input['tokens_{}'.format(p)])
            _input['pcnn_mask_{}'.format(p)] = utils.to_float_tensor(_input['pcnn_mask_{}'.format(p)])

        # Entity types
        _input['head_etype'] = utils.to_long_tensor(_input['head_etype'])
        _input['tail_etype'] = utils.to_long_tensor(_input['tail_etype'])
        
        # Sentence/Input length
        _input['length'] = utils.to_long_tensor(_input['length'])

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
    def __init__(self, data_path, vocas, k_samples=5, max_len=MAX_POS, mask_entity_type=False):
        super(TSVDataset, self).__init__(
            data_path=data_path, vocas=vocas, k_samples=k_samples, 
            max_len=max_len, mask_entity_type=mask_entity_type)
        self.data_path = data_path

    def read_txt(self, path):
        print('Reading data from path', path)
        with open(path, 'rb') as f:
            data = []
            npasses = 0

            for (i, line) in enumerate(f):
                line = line.decode(errors='replace')
                try:
                    try:
                        deppath, head_ent, tail_ent, enttypes, trigger, fname, sentence, postags, rel = line.split('\t')
                        hp = sentence.find(head_ent)
                        ss = sentence[:hp].count(' ')
                        se = ss + head_ent.count(' ') + 1
                        tp = sentence.find(tail_ent)
                        os = sentence[:tp].count(' ')
                        oe = os + tail_ent.count(' ') + 1
                        if head_ent in tail_ent and os <= ss and oe >= se:
                            sub_sent = sentence[hp+1:]
                            ss = sub_sent[:sub_sent.find(head_ent)].count(' ') + ss
                            se = ss + head_ent.count(' ') + 1
                        elif tail_ent in head_ent and ss <= os and se >= oe:
                            sub_sent = sentence[tp+1:]
                            os = sub_sent[:sub_sent.find(tail_ent)].count(' ') + os
                            oe = os + tail_ent.count(' ') + 1
                        assert(ss >= 0 and se >= 0 and os >= 0 and oe >= 0)
                        s_start, s_end = ss, se
                        o_start, o_end = os, oe
                    except Exception as e:
                        deppath, head_ent, tail_ent, enttypes, trigger, fname, sentence, postags, rel, s_pos, o_pos = line.split(
                            '\t')
                        s_start, s_end = s_pos.split('-')
                        s_start, s_end = int(s_start), int(s_end)
                        o_start, o_end = o_pos.split('-')
                        o_start, o_end = int(o_start), int(o_end)

                    head_etype, tail_etype = enttypes.split('-')
                    rel = rel.strip()
                    rel = 'NA' if rel == '' else rel
                    item = {
                        'sentence': sentence.strip(),
                        'postags': postags.strip(),
                        'head_ent': head_ent.strip(),
                        'head_etype': head_etype.strip(),
                        'tail_ent': tail_ent.strip(),
                        'tail_etype': tail_etype.strip(),
                        'rel': rel,
                        'subj_offset': (s_start, s_end),
                        'obj_offset': (o_start, o_end)
                    }
                    data.append(item)
                    if int(i+1) % 1e2:
                        print(i, npasses, end='\r')
                except Exception as e:
                    print(e)
                    npasses += 1
        return data

    def read_from_file(self, path, _format='txt', max_len=MAX_POS):
        data = []

        if _format == 'txt':
            raw_data = self.read_txt(path)
        else:
            raise NotImplementedError

        count = 0
        for _i, item in enumerate(raw_data):
            relation = item['rel']
            head_ent, tail_ent = item['head_ent'], item['tail_ent']
            sentence, (ss, se), (os, oe) = item['sentence'], item['subj_offset'], item['obj_offset']

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

            tokens = sentence.split(' ')
            head_etype = item['head_etype'] if self.vocas['etype'].get_id(
                item['head_etype']) != self.vocas['etype'].unk_id else 'MISC'
            tail_etype = item['tail_etype'] if self.vocas['etype'].get_id(
                item['tail_etype']) != self.vocas['etype'].unk_id else 'MISC'
            if self.mask_entity_type:
                head_etype_subj = PAD_TOKEN
                tail_etype_obj = PAD_TOKEN
            else:
                head_etype_subj = head_etype + '-SUBJ'
                tail_etype_obj = tail_etype + '-OBJ'

            if len(set(range(ss, se)).intersection(range(os, oe))) != 0:
                continue
            
            if ss < os:
                tokens = tokens[:ss] + [head_etype_subj] + \
                    tokens[se:os] + [tail_etype_obj] + \
                    tokens[oe:]
                se, os = (ss + 1, ss + 1 + os - se)
                oe = os + 1
                tokens_l = tokens[:ss]
                tokens_c = tokens[se:os]
                tokens_r = tokens[oe:]
            else:
                tokens = tokens[:os] + \
                    [tail_etype_obj] + \
                    tokens[oe:ss] + \
                    [head_etype_subj] + \
                    tokens[se:]
                oe, ss = (os + 1, os + 1 + ss - oe)
                se = ss + 1
                tokens_l = tokens[:os]
                tokens_c = tokens[oe:ss]
                tokens_r = tokens[se:]

            n_tokens = len(tokens)

            if n_tokens <= max_len:
                # get positions w.r.t to head/tail
                pos_wrt_head = [max(-MAX_POS, i - ss) for i in range(0, ss)
                                ] + [0]*(se-ss) + [min(MAX_POS, i - se + 1) for i in range(se, n_tokens)]
                pos_wrt_tail = [max(-MAX_POS, i - os) for i in range(0, os)
                                ] + [0]*(oe-os) + [min(MAX_POS, i - oe + 1) for i in range(oe, n_tokens)]
                if len(pos_wrt_head) == n_tokens and len(pos_wrt_tail) == n_tokens:
                    data.append({
                        'sentence': sentence,
                        'tokens_str': tokens,
                        'head_ent': head_ent_id,
                        'tail_ent': tail_ent_id,
                        'head_etype': self.vocas['etype_with_subjobj'].get_id(head_etype_subj),
                        'tail_etype': self.vocas['etype_with_subjobj'].get_id(tail_etype_obj),
                        'rel': rel_id,
                        'tokens': [self.vocas['word'].get_id(t) for t in tokens],
                        'tokens_l': [self.vocas['word'].get_id(t) for t in tokens_l],
                        'tokens_c': [self.vocas['word'].get_id(t) for t in tokens_c],
                        'tokens_r': [self.vocas['word'].get_id(t) for t in tokens_r],
                        'etype_pair': self.vocas['etype_pair'].get_id((head_etype, tail_etype))
                    })
                    count += 1
                else:
                    print(item)
                    print(tokens, '\n', pos_wrt_head, head_ent, tail_ent,
                          n_tokens, len(pos_wrt_head), len(pos_wrt_tail))
                    break

            if count % 1000 == 0:
                print('load %d items' % count, end='\r')
        print('Total raw lines:', _i)
        print('load', len(data), 'triples')

        return data
