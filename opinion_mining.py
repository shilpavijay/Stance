#File Name: opinion_mining.py

import nltk
from nltk.corpus import pros_cons
import pickle
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.tokenize import word_tokenize

# Tuple with a list of all words and its category(i.e. pro/con)
# Will be used as a dictionary
# opinion_dict = [ (list(pros_cons.words(fileid)),category) for category in pros_cons.categories() for fileid in pros_cons.fileids(category)]



opinion_dict = []
find_words = []
#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J","R","V"]

for category in pros_cons.categories():
	for fileid in pros_cons.fileids(category):
		for each in pros_cons.words(fileid):
			if category == 'Çons':
				opinion_dict.append((each,'con'))
			elif category == 'Pros':
				opinion_dict.append((each,'pro'))
			words = word_tokenize(each)
			pos = nltk.pos_tag(words)
			for w in pos:
				if w[1][0] in allowed_word_types:
					find_words.append(w[0].lower())

print(opinion_dict)

# find_words = nltk.FreqDist(find_words)

# features = list(find_words.keys())[:3000]

# def find_features(wordlist):
# 	wordset = set(wordlist)
# 	featureDict = {}
# 	for w in features:
# 		featureDict[w] = (w in wordset)
# 	return featureDict

# polarity = [(find_features(op),category) for (op,category) in opinion_dict] 

# random.shuffle(polarity)

# training = polarity[:1900]
# testing = polarity[1900:]

# classifier = nltk.NaiveBayesClassifier.train(training)

# print("Naive Bayes Algorithm accuracy percentage: ",nltk.classify.accuracy(classifier,testing)*100)

# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training)
# classifier.show_most_informative_features(25)
# print("Multinomial NB Algorithm accuracy percentage: ",nltk.classify.accuracy(MNB_classifier,testing)*100)