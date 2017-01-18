#File Name: opinion_mining.py

import nltk
from nltk.corpus import pros_cons, stopwords
import pickle
import random
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
from nltk.stem import WordNetLemmatizer

class VoteClassifier(ClassifierI):
	def __init__(self,*classifiers):
		self._classifiers = classifiers

	def classify(self,features):
		votes=[]
		for eachCfr in self._classifiers:
			v=eachCfr.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self,features):
		votes=[]
		for eachCfr in self._classifiers:
			v=eachCfr.classify(features)
			votes.append(v)

		Conf = (votes.count(mode(votes))) / len(votes)
		return Conf

open_opiniondict = open("pickle/opiniondict.pickle","rb")
opinion_dict = pickle.load(open_opiniondict)
open_opiniondict.close()

open_find_words = open("pickle/find_words.pickle","rb")
find_words = pickle.load(open_find_words)
open_find_words.close()

features = list(find_words.keys())[:3000]

def find_features(wordlist):
	wordset = word_tokenize(wordlist)
	featureDict = {}
	for w in features:
		featureDict[w] = (w in wordset)
	return featureDict

open_polarity = open("pickle/polarity.pickle","rb")
polarity = pickle.load(open_polarity)
open_polarity.close()

random.shuffle(polarity)

training = polarity[:2000]
testing = polarity[2000:]

open_NBClass = open("pickle/NBClass.pickle","rb")
classifier = pickle.load(open_NBClass)
open_NBClass.close()

#############################

open_MNBClass = open("pickle/MNBClass.pickle","rb")
MNB_classifier = pickle.load(open_MNBClass)
open_MNBClass.close()

# ########################

open_BNBClass = open("pickle/BNBClass.pickle","rb")
BNB_classifier = pickle.load(open_BNBClass)
open_BNBClass.close()

# ##########################

open_LRClass = open("pickle/LRClass.pickle","rb")
LR_classifier = pickle.load(open_LRClass)
open_LRClass.close()

# ###########################

open_SGDCClass = open("pickle/SGDCClass.pickle","rb")
SGDC_classifier = pickle.load(open_SGDCClass)
open_SGDCClass.close()

# ###########################

open_LRSVCClass = open("pickle/LRSVCClass.pickle","rb")
LinerSVC_classifier = pickle.load(open_LRSVCClass)
open_LRSVCClass.close()

# ###########################

open_NUSVClass = open("pickle/NUSVClass.pickle","rb")
NuSVC_classifier = pickle.load(open_NUSVClass)
open_NUSVClass.close()

# ###########################

voted_cfr = VoteClassifier(classifier, 
							MNB_classifier, 
							BNB_classifier,
							LR_classifier,
							SGDC_classifier,
							LinerSVC_classifier,
							NuSVC_classifier
							)

def opinion(text):
	feat = find_features(text)
	result = voted_cfr.classify(feat)
	return result, (voted_cfr.confidence(feat) * 100)