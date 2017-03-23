import nltk
from nltk.corpus import pros_cons, stopwords

opinion_dict = [ (" ".join(sent),category) for category in pros_cons.categories() for fileid in pros_cons.fileids(category) for sent in pros_cons.sents(fileid)]
print(opinion_dict)