# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, help='path to train dataset')
opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(opt.data)
model = word2vec.Word2Vec(sentences, size=200, min_count=0, window=15)
model.save("./wiki.model")
