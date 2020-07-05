import re
import io

UNK_TOKEN = "#UNK#"
PAD_TOKEN = "#PAD#"

BRACKETS = {"-LCB-": "{", "-LRB-": "(", "-LSB-": "[", "-RCB-": "}", "-RRB-": ")", "-RSB-": "]"}

class Vocabulary:
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN
    lower = False
    digit_0 = False

    def __init__(self, normalization=True, add_pad_unk=True, lower=False, digit_0=False):
        self.word2id = {}
        self.id2word = []
        self.counts = []
        self.unk_id = -1
        self.pad_id = -1
        self.normalization = normalization
        self.add_pad_unk = add_pad_unk
        self.lower = lower
        self.digit_0 = digit_0

    def normalize(self, token):
        if token in [Vocabulary.unk_token, Vocabulary.pad_token, "<s>", "</s>", "#head#", "#tail#"]:
            return token
        elif token in BRACKETS:
            token = BRACKETS[token]
        else:
            if self.digit_0:
                token = re.sub("[0-9]", "0", token)

        if self.lower:
            return token.lower()
        else:
            return token

    @staticmethod
    def load(path, normalization=True, add_pad_unk=True, lower=False, digit_0=False):
        voca = Vocabulary(normalization, add_pad_unk, lower, digit_0)
        added = voca.load_from_file(path)
        return voca, added

    def load_from_file(self, path):
        self.word2id = {}
        self.id2word = []
        self.counts = []
        self.probs = None

        f = io.open(path, "r", encoding='utf-8', errors='ignore')
        for line in f:
            token = line.strip()
            if self.normalization:
                token = self.normalize(token)
            self.id2word.append(token)
            self.word2id[token] = len(self.id2word) - 1

            self.counts.append(1)

        f.close()
        added = self._add_pad_unk()
        return added

    @staticmethod
    def load_from_list(l_items, normalization=True, add_pad_unk=True, lower=False, digit_0=False):
        voca = Vocabulary(normalization, add_pad_unk, lower, digit_0)
        if l_items is not None:
            voca._load_from_list(l_items)
        voca._add_pad_unk()
        return voca

    def _add_pad_unk(self):
        added = []
        if self.add_pad_unk:
            if Vocabulary.unk_token not in self.word2id:
                self.id2word.append(Vocabulary.unk_token)
                self.word2id[Vocabulary.unk_token] = len(self.id2word) - 1
                self.counts.append(1)
                added.append(Vocabulary.unk_token)

            if Vocabulary.pad_token not in self.word2id:
                self.id2word.append(Vocabulary.pad_token)
                self.word2id[Vocabulary.pad_token] = len(self.id2word) - 1
                self.counts.append(1)
                added.append(Vocabulary.pad_token)

            self.pad_id = self.get_id(self.pad_token)
            self.unk_id = self.get_id(self.unk_token)
        return added
    
    def _load_from_list(self, l_items):
        assert len(set(l_items)) == len(l_items), 'Not unique list of items'
        n_start = len(self.id2word)
        for i, item in enumerate(l_items):
            if item not in self.id2word:
                self.id2word.append(item)
                self.word2id[item] = i+n_start

    def size(self):
        return len(self.id2word)

    def get_id(self, token):
        if self.normalization:
            tok = self.normalize(token)
        else:
            tok = token
        return self.word2id.get(tok, self.unk_id)
