# -*- coding: utf-8 -*-
from zhon import hanzi
import string
import os
import re
from collections import Counter
import codecs
import pickle

regex_punctuations = re.compile('[%s0-9]' %(hanzi.punctuation + string.punctuation))
endofsentencepunctuation = "|".join(hanzi.punctuation[14]+hanzi.punctuation[-10:-5] + hanzi.punctuation[-5:]) +"|[.?!...;]|"
#...|period|period|!|?

def loadstopwords(stopwordfiles):
    #load stop words for both english and chinese, in the chinese list there are also some punctuations
    stopwords = set([])
    
    for stopwordfile in stopwordfiles:
        with codecs.open(stopwordfile, "r", "utf8") as fin:
            for aline in fin:
                aline = aline.strip()
                if aline.startswith("#") or len(aline) == 0:
                    continue
                stopwords.add(aline)
    return stopwords

def ischinese(s):
    return u'\u4e00' <= s <= u'\u9fff'

def process_singleunit(text, stopwords=set([])):
    text = regex_punctuations.sub('', text).strip()
    if len(text) > 0 and text not in stopwords and ischinese(text) :
        return text
    else:
        return None
        
def isallletters(s):
    return sum([u'\u0041' <= c <= u'\u005a' for c in s]) + sum([u'\u0061' <= c <= u'\u007a' for c in s]) == len(s)        
    
def process_sentence(sent, stopwords=set([])):
    characters = [item for item in sent if process_singleunit(item)]
    return characters
    
def process_text(text, stopwords=set([])):
    for sent in re.split(endofsentencepunctuation, text):
        yield process_sentence(sent, stopwords)
        

class CharacterDictionary(object):
    """
    A dictionary of characters (instead of words), generated from a given iterable, with optional preprocess step
    """
    def __init__(self, iterable=None, preprocess=None, startfrom=1, unknownC="O"):
        self.preprocess = preprocess
        if preprocess is None:
            self.preprocess = lambda s:s
        self.counter = Counter()
        self.startfrom = startfrom
        self.unknownC = unknownC
        if iterable:
            self._init_from_iterable(iterable)
    
    def _init_from_iterable(self, iterable):
        for oneline in iterable:
            oneline = self.preprocess(oneline)
            self.counter.update(oneline) # oneline must be itself also an iterable
        self.token2id = {item[0] : i + self.startfrom for i, item in enumerate(sorted(self.counter.iteritems(), key=lambda s:s[1], reverse=True))} #tokenid starts from 1
        self.id2token = {i : item for item, i in self.token2id.iteritems()}
        self.dictsize = len(self.token2id)
    
    def transform(self, iterable):
        """
        transform list of strings to list of integer list, characters not in the dictionary will be mapped to the number dictsize + 1
        """
        newiterable = [[self.token2id.get(item, self.dictsize + 1) for item in oneline] for oneline in iterable]
        return newiterable
        
    def inversetransform(self, iterable, returnlist=True, sep=""):
        newiterable = [[self.id2token.get(item, self.unknownC) for item in oneline] for oneline in iterable]
        if not returnlist:
            newiterable = [sep.join(item) for item in newiterable]
        return newiterable
    
    def save(self, outfile):
        self.preprocess = self.preprocess.func_name
        with open(outfile, "wb") as fout:
            pickle.dump(self.__dict__, fout)

    def load(self, outfile):
        with open(outfile, 'r') as fin:
            self.__dict__ = pickle.loads(fin.read())
        try:
            self.preprocess = eval(self.preprocess)
        except:
            self.preprocess = lambda s:s
