#File Name: opinion_mining_intermediate.py

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


# Tuple with a list of all words and its category(i.e. pro/con)
# Will be used as a dictionary
# opinion_dict = [ (" ".join(sent),category) for category in pros_cons.categories() for fileid in pros_cons.fileids(category) for sent in pros_cons.sents(fileid)]

#pickling:
# save_opiniondict = open("pickle/opiniondict.pickle","wb")
# pickle.dump(opinion_dict,save_opiniondict)
# save_opiniondict.close()

open_opiniondict = open("pickle/opiniondict.pickle","rb")
opinion_dict = pickle.load(open_opiniondict)
open_opiniondict.close()


# find_words = []
# #  j is adject, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]

# #to remove stopwords - meaningless words (and, a etc) not useful to nltk		
# stop_words=set(stopwords.words("english"))	

# # *Stemming - Obtaining the underlying word so that multiple sentences meaning the same are identified while processing.
# # *Lemmatizing - similar to Stemming but end result will be a real, meaningful word. Can customize the type of output as well. (Adjective etc)
# lemmatizer = WordNetLemmatizer()

# for eachOp in opinion_dict[:1500]:
# 	words = word_tokenize(eachOp[0])
# 	pos = nltk.pos_tag(words)
# 	for w in pos:
# 		if w[1][0] in allowed_word_types and w[0] not in stop_words:
# 			find_words.append(lemmatizer.lemmatize(w[0]).lower())

# find_words = nltk.FreqDist(find_words)

#pickling:
# save_find_words = open("pickle/find_words.pickle","wb")
# pickle.dump(find_words,save_find_words)
# save_find_words.close()

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

# polarity = [(find_features(op),category) for (op,category) in opinion_dict] 

# save_polarity = open("pickle/polarity.pickle","wb")
# pickle.dump(polarity,save_polarity)
# save_polarity.close()

open_polarity = open("pickle/polarity.pickle","rb")
polarity = pickle.load(open_polarity)
open_polarity.close()

random.shuffle(polarity)

training = polarity[:2000]
testing = polarity[2000:]


# classifier = nltk.NaiveBayesClassifier.train(training)

#pickling:
# save_NBClass = open("pickle/NBClass.pickle","wb")
# pickle.dump(classifier,save_NBClass)
# save_NBClass.close()

open_NBClass = open("pickle/NBClass.pickle","rb")
classifier = pickle.load(open_NBClass)
open_NBClass.close()

classifier.show_most_informative_features(25)
print("Naive Bayes Algorithm accuracy percentage: ",nltk.classify.accuracy(classifier,testing)*100)


#############################

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training)

#pickling:
# save_MNBClass = open("pickle/MNBClass.pickle","wb")
# pickle.dump(MNB_classifier,save_MNBClass)
# save_MNBClass.close()

open_MNBClass = open("pickle/MNBClass.pickle","rb")
MNB_classifier = pickle.load(open_MNBClass)
open_MNBClass.close()

print( "Multinomial NB Accuracy Percentage: ",nltk.classify.accuracy(MNB_classifier,testing)*100)

# ########################

# BNB_classifier = SklearnClassifier(BernoulliNB())
# BNB_classifier.train(training)

# #pickling:
# save_BNBClass = open("pickle/BNBClass.pickle","wb")
# pickle.dump(BNB_classifier,save_BNBClass)
# save_BNBClass.close()

open_BNBClass = open("pickle/BNBClass.pickle","rb")
BNB_classifier = pickle.load(open_BNBClass)
open_BNBClass.close()

print( "Bernaulli NB Accuracy Percentage: ",nltk.classify.accuracy(BNB_classifier,testing)*100)

# ##########################

# LR_classifier = SklearnClassifier(LogisticRegression())
# LR_classifier.train(training)

# save_LRClass = open("pickle/LRClass.pickle","wb")
# pickle.dump(LR_classifier,save_LRClass)
# save_LRClass.close()

open_LRClass = open("pickle/LRClass.pickle","rb")
LR_classifier = pickle.load(open_LRClass)
open_LRClass.close()

print( "Linear Regression Algorithm Accuracy Percentage: ",nltk.classify.accuracy(LR_classifier,testing)*100)

# ###########################

# SGDC_classifier = SklearnClassifier(SGDClassifier())
# SGDC_classifier.train(training)

# save_SGDCClass = open("pickle/SGDCClass.pickle","wb")
# pickle.dump(SGDC_classifier,save_SGDCClass)
# save_SGDCClass.close()

open_SGDCClass = open("pickle/SGDCClass.pickle","rb")
SGDC_classifier = pickle.load(open_SGDCClass)
open_SGDCClass.close()

print( "SGDC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(SGDC_classifier,testing)*100)

# ###########################

# LinerSVC_classifier = SklearnClassifier(LinearSVC())
# LinerSVC_classifier.train(training)

# save_LRSVCClass = open("pickle/LRSVCClass.pickle","wb")
# pickle.dump(LinerSVC_classifier,save_LRSVCClass)
# save_LRSVCClass.close()

open_LRSVCClass = open("pickle/LRSVCClass.pickle","rb")
LinerSVC_classifier = pickle.load(open_LRSVCClass)
open_LRSVCClass.close()

print( "Linear SVC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(LinerSVC_classifier,testing)*100)


# ###########################

# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training)

# pickling
# save_NUSVClass = open("pickle/NUSVClass.pickle","wb")
# pickle.dump(NuSVC_classifier,save_NUSVClass)
# save_NUSVClass.close()

open_NUSVClass = open("pickle/NUSVClass.pickle","rb")
NuSVC_classifier = pickle.load(open_NUSVClass)
open_NUSVClass.close()

print( "Numeric SVC Algorithm Accuracy Percentage: ",nltk.classify.accuracy(NuSVC_classifier,testing)*100)

# ###########################

voted_cfr = VoteClassifier(classifier, 
							MNB_classifier, 
							BNB_classifier,
							LR_classifier,
							SGDC_classifier,
							LinerSVC_classifier,
							NuSVC_classifier
							)

print("Classifiers Accuracy Percentage: " ,(nltk.classify.accuracy(voted_cfr,testing))*100)
print("Classification: ", voted_cfr.classify(testing[0][0]), "Confidence: ", voted_cfr.confidence(testing[0][0]))

