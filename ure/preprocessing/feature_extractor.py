# from https://github.com/diegma/relation-autoencoder/definitions/OieFeatures.py

import nltk
import re, string
import pickle

parsing = 0
entities = 1
trig = 2
sentence = 3
pos = 4
docPath = 5
#  ======= Relation features =======
stopwords_list = nltk.corpus.stopwords.words('english')
_digits = re.compile('\d')

def bow_clean(info, arg1, arg2):
    bow = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    result = []
    tmp = []
    for word in bow:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    for word in tmp:
        if word not in stopwords_list and not _digits.search(word) and not word[0].isupper():
            result.append(word)
    return result

def trigger(info, arg1, arg2):
    return info[trig].replace('TRIGGER:', '')

def entityTypes(info, arg1, arg2):
    return info[entities]

def entity1Type(info, arg1, arg2):
    return info[entities].split('-')[0]

def entity2Type(info, arg1, arg2):
    return info[entities].split('-')[1]

def arg1(info, arg1, arg2):
    return arg1

def arg1_lower(info, arg1, arg2):
    return arg1.lower()

def arg2(info, arg1, arg2):
    return arg2

def arg2_lower(info, arg1, arg2):
    return arg2.lower()

def lexicalPattern(info, arg1, arg2):
    # Dependency parsing
    # return info[parsing]
    p = info[parsing].replace('->', ' ').replace('<-', ' ').split()
    result = []
    for num, x in enumerate(p):
        if num % 2 != 0:
            result.append(x)
    return '_'.join(result)

def posPatternPath(info, arg1, arg2):
    words = info[sentence].split()
    postags = info[pos].split()
    assert len(postags) == len(words), 'error'
    a = []
    for w in range(len(words)):
        a.append((words[w], postags[w]))
    # a = info[4].split()
    if a:
        # print arg1, words
        # print [a.index(item) for item in a if item[0] == arg1.split()[-1]],'aaaaaaa'
        beginList = [a.index(item) for item in a if item[0] == arg1.split()[-1]]
        # print beginList
        endList = [a.index(item) for item in a if item[0] == arg2.split()[0]]
        # print endList
        if len(beginList) > 0 and len(endList) > 0:
            # posPattern = [item[1] for item in a if beginList[0] > a.index(item) > endList[0]]
            posPattern = []
            for num, item in enumerate(a):
                if beginList[0] < num < endList[0]:
                    posPattern.append(item[1])
            # print posPattern
            return '_'.join(posPattern)
        else:
            return ''
    else:
        return ''


def getBasicCleanFeatures():
    features = [trigger, entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern,
                posPatternPath]
    return features


class FeatureLexicon:

    def __init__(self):
        self.nextId = 0
        self.id2Str = {}
        self.str2Id = {}
        self.id2freq = {}
        self.nextIdPruned = 0
        self.id2StrPruned = {}
        self.str2IdPruned = {}

    def getOrAdd(self, s):
        if s not in self.str2Id:
            self.id2Str[self.nextId] = s
            self.str2Id[s] = self.nextId
            self.id2freq[self.nextId] = 1
            self.nextId += 1
        else:
            self.id2freq[self.str2Id[s]] += 1
        return self.str2Id[s]


    def getOrAddPruned(self, s):
        if s not in self.str2IdPruned:
            self.id2StrPruned[self.nextIdPruned] = s
            self.str2IdPruned[s] = self.nextIdPruned
            self.nextIdPruned += 1
        return self.str2IdPruned[s]

    def getId(self, s):
        if s not in self.str2Id:
            return None
        return self.str2Id[s]

    def getStr(self, idx):
        if idx not in self.id2Str:
            return None
        else:
            return self.id2Str[idx]

    def getStrPruned(self, idx):
        if idx not in self.id2StrPruned:
            return None
        else:
            return self.id2StrPruned[idx]

    def getFreq(self, idx):
        if idx not in self.id2freq:
            return None
        return self.id2freq[idx]


    def getDimensionality(self):
        return self.nextIdPruned
        # return self.nextId

    def from_file(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                feat = line.strip()
                self.getOrAddPruned(feat)
                self.getOrAdd(feat)
        return self


def getFeatures(lexicon, featureExs, info, arg1=None, arg2=None, expand=False):
    feats = []
    for f in featureExs:
        res = f(info, arg1, arg2)
        if res is not None:
            if type(res) == list:
                for el in res:
                    featStrId = f.__name__ + "#" + el
                    if expand:
                        feats.append(lexicon.getOrAdd(featStrId))
                    else:
                        featId = lexicon.getId(featStrId)
                        if featId is not None:
                            feats.append(featId)
            else:
                featStrId = f.__name__ + "#" + res
                if expand:
                    feats.append(lexicon.getOrAdd(featStrId))
                else:
                    featId = lexicon.getId(featStrId)
                    if featId is not None:
                        feats.append(featId)

    return feats


def getFeaturesThreshold(lexicon, featureExs, info, arg1=None, arg2=None, 
                        expand=False, threshold=0):
    feats = []
    for f in featureExs:
        res = f(info, arg1, arg2)
        if res is not None:
            if type(res) == list:
                for el in res:
                    featStrId = f.__name__ + "#" + el
                    if expand:
                        if lexicon.id2freq[lexicon.getId(featStrId)] > threshold:
                            feats.append(lexicon.getOrAddPruned(featStrId))
                    else:
                        featId = lexicon.getId(featStrId)
                        if featId is not None:
                            if lexicon.id2freq[featId] > threshold:
                                feats.append(lexicon.getOrAddPruned(featStrId))
            else:
                featStrId = f.__name__ + "#" + res
                if expand:
                    if lexicon.id2freq[lexicon.getId(featStrId)] > threshold:
                        feats.append(lexicon.getOrAddPruned(featStrId))
                else:
                    featId = lexicon.getId(featStrId)
                    if featId is not None:
                        if lexicon.id2freq[featId] > threshold:
                            feats.append(lexicon.getOrAddPruned(featStrId))
    return feats


def loadExamples(fileName):
    count = 0
    with open(fileName, 'rb') as fp:
        relationExamples = []
        for line_id, line in enumerate(fp):
            line = line.decode(errors='replace')
            if len(line) == 0 or len(line.split()) == 0:
                raise IOError

            else:
                fields = line.split('\t')
                if len(fields) != 9:
                    print("a problem with the file format at line" + str(line_id) + "(# fields is wrong) len is " + str(len(fields)) + "instead of 9")
                    continue
                # this will be 10
                relationExamples.append([str(count)] + fields)
                count += 1

    return relationExamples


def loadTACRED(fileName):
    count = 0
    with open(fileName, 'rb') as fp:
        relationExamples = []
        for line_id, line in enumerate(fp):
            line = line.decode(errors='replace')
            if len(line) == 0 or len(line.split()) == 0:
                raise IOError

            else:
                fields = line.split('\t')
                if len(fields) != 11:
                    print(
                        "a problem with the file format at line" + str(line_id) + "(# fields is wrong) len is " +
                        str(len(fields)) + "instead of 9")
                    continue
                # this will be 10
                relationExamples.append([str(count)] + fields)
                count += 1

    return relationExamples


def get_lexicon(args):
    input_file, lexicon_file, output_file = args.input_file, args.lexicon_file, args.output_file

    relationExamples = loadExamples(input_file)
    relationLexicon = FeatureLexicon()
    featureExtrs = getBasicCleanFeatures()
    for reIdx, re in enumerate(relationExamples):
        print (reIdx, end='\r')
        getFeatures(
            lexicon=relationLexicon, featureExs=featureExtrs, 
            info=[re[1], re[4], re[5], re[7], re[8], re[6]],
            arg1=re[2], arg2=re[3], expand=True)

    with open(output_file, 'w') as f:
        for reIdx, re in enumerate(relationExamples):
            print (reIdx, end='\r')
            feats = getFeaturesThreshold(
                lexicon=relationLexicon, featureExs=featureExtrs, 
                info=[re[1], re[4], re[5], re[7], re[8], re[6]],
                arg1=re[2], arg2=re[3], expand=True, threshold=args.threshold)
        f.write('{}\n'.format(' '.join(map(lambda x: str(x), feats))))

    with open(lexicon_file, 'w') as f:
        for k, v in relationLexicon.id2StrPruned.items():
            f.write('{}\n'.format(v))


def get_features(args):
    input_file, lexicon_file, output_file = args.input_file, args.lexicon_file, args.output_file

    relationLexicon = FeatureLexicon()
    relationLexicon.from_file(lexicon_file)
    featureExtrs = getBasicCleanFeatures()

    # relationExamples = loadExamples(input_file)
    relationExamples = loadTACRED(input_file)
    print (len(relationLexicon.id2StrPruned), len(relationLexicon.id2Str), len(relationLexicon.id2freq))
    with open(output_file, 'w') as f:
        for reIdx, re in enumerate(relationExamples):
            print (reIdx, end='\r')
            feats = getFeaturesThreshold(
                lexicon=relationLexicon, featureExs=featureExtrs, 
                info=[re[1], re[4], re[5], re[7], re[8], re[6]],
                arg1=re[2], arg2=re[3], expand=False, threshold=0)
            rel = re[9].strip()
            f.write('{}\t{}\t{}\t{}\n'.format(
                rel if rel != '' else 'no_relation', re[2], re[3],
                ' '.join(map(lambda x: str(x), feats))))


if __name__ == "__main__":
    import argparse
    import random
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-l', '--lexicon_file', required=True, help="generate by using argument --generate_lexicon")
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-t', '--threshold', default=0)
    parser.add_argument('--generate_lexicon', action="store_true")

    args = parser.parse_args()
    if args.generate_lexicon:
        get_lexicon(args)
    else:
        get_features(args)
