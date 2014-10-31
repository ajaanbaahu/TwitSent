#!/usr/bin/env python

from pymongo import Connection
import scipy
import nltk,nltk.classify.util, nltk.metrics
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain
from nltk.classify import NaiveBayesClassifier
from nltk.classify import maxent
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import bigrams
from sklearn.externals import joblib
import math,pickle



tag_data=[]
f=open('data_result.txt')
for line in f:
	tag_data.append(line.strip('\n'))
posCutoff = int(math.floor(len(tag_data)*3/4))
test_tags=tag_data[posCutoff:]
train_tags=tag_data[:posCutoff]

#print len(test_tags), len(train_tags)


#clf_f=('/home/ajaanbaahu/Documents/Projects/TwitSent/scikit_pickle_classifier.pkl')

#clf=joblib.load(clf_f)
#train_feat=joblib.load("/home/ajaanbaahu/Documents/Projects/TwitSent/scikit_pickle_train.pkl")

#test_feat=joblib.load("/home/ajaanbaahu/Documents/Projects/TwitSent/scikit_pickle_test.pkl")

train_feat_uni=joblib.load("/home/ajaanbaahu/Documents/Projects/TwitSent/scikit_pickle_train_bigram.pkl")

test_feat_uni=joblib.load("/home/ajaanbaahu/Documents/Projects/TwitSent/scikit_pickle_test_bigram.pkl")

#classifier = NaiveBayesClassifier.train(train_feat_uni)
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(train_feat_uni, 'GIS', trace=3, \
encoding=None, labels=None, sparse=True, gaussian_prior_sigma=.3, max_iter = 15)

# f = open('my_classifier.pickle', 'wb')
# pickle.dump(MaxEntClassifier, f)
# f.close()

# #joblib.dump(classifier, 'scikit_pickle_classifier.pkl')
# #classifier=joblib.load("/home/ajaanbaahu/Documents/Projects/TwitSent/scikit_pickle_classifier.pkl")
result=[]
for i in test_feat_uni:
 	result.append(MaxEntClassifier.classify(i))

print len([i for i,j in zip(result,test_tags) if i==j])