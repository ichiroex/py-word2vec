from gensim.models import word2vec
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('word', type=str, help='query word to calculate similar words')
opt = parser.parse_args()

model = word2vec.Word2Vec.load("./wiki.model")
results = model.wv.most_similar(positive=[opt.word])
for result in results:
    print(result)
