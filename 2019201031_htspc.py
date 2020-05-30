import pandas as pd
import numpy as np
import re
import nltk
import preprocessor as p
import statistics as st
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from gensim.parsing.preprocessing import remove_stopwords


def remove_extra_spaces(text):
    return ' '.join(text.split())

def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

def remove_urls(text):
    url_pattern = re.compile(r'http\S+')
    return url_pattern.sub('', text)

def remove_emoji(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def remove_mentions(text):
    mention_pattern = re.compile(r'@\S+')
    return mention_pattern.sub('',text)

def remove_digits(text):
    digit_pattern = re.compile(r'\d+')
    return digit_pattern.sub('',text)

def remove_hash(text):
    return text.replace('#','')

def remove_single_chars(text):
    local = []
    splitted = text.split()
    for word in splitted:
        if(len(word) is not 1):
            local.append(word)
    return " ".join(local)

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def stemming(text):
    splitted = text.split()
    stemming = PorterStemmer()
    return " ".join([stemming.stem(word) for word in splitted])


data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(remove_urls)
data['text'] = data['text'].apply(remove_emoji)
data['text'] = data['text'].apply(remove_mentions)
data['text'] = data['text'].apply(remove_digits)
data['text'] = data['text'].apply(remove_hash)
data['text'] = data['text'].apply(remove_extra_spaces)
data['text'] = data['text'].apply(remove_stopwords)
data['text'] = data['text'].apply(remove_single_chars)
data['text'] = data['text'].str.replace('[^\w\s]','')
data['text'] = data['text'].apply(lemmatize_words)
data['text'] = data['text'].apply(stemming)


test['text'] = test['text'].str.lower()
test['text'] = test['text'].apply(remove_urls)
test['text'] = test['text'].apply(remove_emoji)
test['text'] = test['text'].apply(remove_mentions)
test['text'] = test['text'].apply(remove_digits)
test['text'] = test['text'].apply(remove_hash)
test['text'] = test['text'].apply(remove_extra_spaces)
test['text'] = test['text'].apply(remove_stopwords)
test['text'] = test['text'].apply(remove_single_chars)
test['text'] = test['text'].str.replace('[^\w\s]','')
test['text'] = test['text'].apply(lemmatize_words)
test['text'] = test['text'].apply(stemming)

x_train =  np.asarray(data['text'])
y_train = np.asarray(data['labels'])

x_train_data = x_train
y_train_data = y_train

x_train_val =  np.asarray(test['text'])

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(x_train)
xtrain_tfidf =  tfidf_vect.transform(x_train_data)
xvalid_tfidf =  tfidf_vect.transform(x_train_val)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return predictions


predict_nb = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_svm = train_model(svm.SVC(kernel="rbf"),xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_msvm = train_model(svm.SVC(kernel="linear",C=0.1),xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_logistic = train_model(LogisticRegression(), xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_adaboost = train_model(AdaBoostClassifier(),xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_randomforest = train_model( RandomForestClassifier(),xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_mlp = train_model(MLPClassifier(alpha=1, max_iter=1000),xtrain_tfidf,y_train_data,xvalid_tfidf)
predict_decision = train_model( DecisionTreeClassifier(max_depth=5),xtrain_tfidf,y_train_data,xvalid_tfidf)

glob_list = []

for i in range(len(predict_svm)):
    list_loc = []
    list_loc.append(predict_nb[i])
    list_loc.append(predict_svm[i])
    list_loc.append(predict_logistic[i])
    list_loc.append(predict_adaboost[i])
    list_loc.append(predict_randomforest[i])
    list_loc.append(predict_decision[i])
    list_loc.append(predict_mlp[i])
    glob_list.append(st.mode(list_loc))

glob_list = np.asarray(glob_list)

np.savetxt("submission.csv",glob_list,fmt="%d",header ="labels",comments='', delimiter=",")
