import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords, brown
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from gensim import corpora,models, similarities
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex

print("Logistic Regression & 3 Layer Multi Layer Neural Network on Brown Corpus category 'Mystery'")

documents = [(list(brown.words(fileid)), category)
             for category in brown.categories()
             for fileid in brown.fileids(category)]

all_words = []

for w in brown.words(categories='mystery'):
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:2000]

stop_words = stopwords.words('english')
docbrown1 = [w for w in documents if w not in stop_words]

model = Word2Vec(common_texts, size=20, min_count=1)

def find_features(document):
   words = set(document)
   features = {}
   for w in word_features:
      features[w] = (w in words)
   return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:250]
# print("feature set length = ", len(featuresets))

testing_set = featuresets[100:]

LogisticRegression_classifier = SklearnClassifier(LogisticRegression(solver='lbfgs',max_iter=500,multi_class='auto'))
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)), "%")

MLP_classifier = SklearnClassifier(MLPClassifier(hidden_layer_sizes=(3),solver='lbfgs',max_iter=500))
MLP_classifier.train(training_set)
print("Multi-layerPerceptron accuracy:", (nltk.classify.accuracy(MLP_classifier,testing_set)), "%")
