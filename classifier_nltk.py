import re, math, collections, itertools
from pymongo import Connection
import nltk,nltk.classify.util, nltk.metrics
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import bigrams
from sklearn.externals import joblib

server="mongodb://mongo:Student01@kahana.mongohq.com:10012/app27538292"

db='app27538292'
def bigramReturner (tweetString):
    # tweetString = tweetString.lower()

    bigramFeatureVector = []
    for item in nltk.bigrams(tweetString.split()):
        bigramFeatureVector.append(' '.join(item))

    return bigramFeatureVector

def prepare(server, database):

    conn = Connection(server)
    db=conn[database]
    result= db.tweet.find()

    data=[]
    for tuples in result:
        body=' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tuples['text']).split())

        body_bigram=bigramReturner(str(body))
        body_unigram=word_tokenize(str(body))
        body_bigram.extend((body_unigram))
        label= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tuples['classification']).split())
        data.append([str(body_bigram),str(label)])

    return data
train_data = prepare(server, db)
posCutoff = int(math.floor(len(train_data)*3/4))

train_sample=train_data[:posCutoff]
test_sample=train_data[posCutoff:]

vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in train_sample]))

feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in train_sample]

featurized_test_set =  [{i:(i in word_tokenize(feature.lower())) for i in vocabulary} for feature,tag in test_sample]

joblib.dump(feature_set, 'scikit_pickle_train_bigram.pkl')
joblib.dump(featurized_test_set, 'scikit_pickle_test_bigram.pkl')

