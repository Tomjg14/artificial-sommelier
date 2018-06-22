# data loading
from pandas.io.json import json_normalize
import json
import pandas as pd
import os, random
from os import listdir

# data inspection
from collections import Counter

# preprocessing
import string
import nltk
from string import punctuation
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import dok_matrix
import scipy.sparse as sparse
from gensim import corpora, models
import gensim

# Word embeddings
from gensim.models import Word2Vec


# clustering
from sklearn import cluster
from sklearn import metrics

# computations
import numpy as np
import math
from numpy import asarray
from numpy import zeros
from keras.utils import np_utils

# visualization
import matplotlib
from matplotlib import pyplot as plt
from pprint import pprint
import time
from datetime import datetime

# data saving
import pickle

# progress
from tqdm import tqdm_notebook as tqdm

# classification
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.svm import SVC

# Load Data

file_path = "data/winemag-data-130k-v2.json"

with open(file_path) as f:
    data = json.load(f)
    
dataset = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

# Load Pickle Files

tokens = pickle.load(open("pickle_files/tokens.p","rb"))

pos_tags = pickle.load(open("pickle_files/pos_tags.p","rb"))

dataset['tokens'] = tokens
dataset['pos_tags'] = pos_tags

lda_model = pickle.load(open("pickle_files/lda_model.p","rb"))

glove_labels = pickle.load(open("pickle_files/glove_labels.p","rb"))

# Compute Content Words

def isContentWord(pos_tag):
    content_tags = ["JJ", "JJR", "JJS", "NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    if pos_tag in content_tags:
        return True
    else:
        return False

content_words = []
for i, token_list in enumerate(tokens):
    pos_tag_list = pos_tags[i]
    for j, word in enumerate(token_list):
        pos_tag = pos_tag_list[j]
        if isContentWord(pos_tag):
            content_words.append(word)

content_counts = Counter(content_words)

filtered_content_words = []
for word in tqdm(content_words):
    if content_counts[word] > 2:
        if not word in filtered_content_words:
            filtered_content_words.append(word)

CONTENT_COUNT = len(np.unique(filtered_content_words))

content_word_dict = {}
for i, word in enumerate(filtered_content_words):
    content_word_dict[word] = i

content_tokens = []
for token_list in tqdm(tokens):
    filtered_tokens = []
    for token in token_list:
        if token in content_word_dict:
            filtered_tokens.append(token)
    content_tokens.append(filtered_tokens)

dataset['content_tokens'] = content_tokens

# Compute Labels

scores = dataset['points'].tolist()

def getCategory(scores):
    category_string = []
    category_int = []
    for score in scores:
        score = int(score)
        if score < 80:
            category_string.append("unacceptable")
            category_int.append(0)
        elif score >= 80 and score <= 82:
            category_string.append("acceptable")
            category_int.append(1)
        elif score >= 83 and score <= 86:
            category_string.append("good")
            category_int.append(2)
        elif score >= 87 and score <= 89:
            category_string.append("very good")
            category_int.append(3)
        elif score >= 90 and score <= 93:
            category_string.append("excellent")
            category_int.append(4)
        elif score >= 94 and score <= 97:
            category_string.append("superb")
            category_int.append(5)
        elif score >= 98 and score <= 100:
            category_string.append("classic")
            category_int.append(6)
    return category_string, category_int

categories, labels = getCategory(scores)

dataset['category'] = categories
dataset['labels'] = labels

# Compute BoW Corpus

def createCorpus(tokens):
    corpus = []
    for token_list in tqdm(tokens):
        content_tokens = []
        for token in token_list:
            if token not in content_counts:
                continue
            else:
                content_tokens.append(token)
        doc = " ".join(content_tokens)
        corpus.append(doc)
    return corpus

corpus = createCorpus(tokens)

bag_of_words_vectorizer = CountVectorizer(min_df=2)
bow_feature_vector = bag_of_words_vectorizer.fit_transform(corpus)

# Filter Data

varieties = dataset['variety'].tolist()
wine_count = Counter(varieties)
nr_reviews = len(varieties)
threshold = 200

filtered_keys = []
for key, item in wine_count.items():
    if item < threshold:
        filtered_keys.append(key)

idx = 0
indices = []
for v in varieties:
    if v in filtered_keys:
        indices.append(idx)
    idx += 1

dataset_filtered = dataset.drop(dataset.index[indices]).copy()
dataset_filtered = dataset_filtered.reset_index()

print("New Total reviews: %s"%(len(dataset_filtered)))

# Split Data

wines = np.unique(dataset_filtered['variety'].tolist())

wine_indices = {}
for wine in wines:
    indices = dataset_filtered.index[dataset_filtered['variety'] == wine].tolist()
    wine_indices[wine] = indices

train_indices = {}
test_indices = {}
for wine in wines:
    indices = wine_indices[wine]
    nr_indices = len(indices)
    train_indices[wine] = indices[:round(nr_indices*0.8)]
    test_indices[wine] = indices[round(nr_indices*0.8):]
    
    nr_train_indices = len(indices[:round(nr_indices*0.8)])
    nr_test_indices = len(indices[round(nr_indices*0.8):])

tr_indices = []
for _, indices in train_indices.items():
    tr_indices = tr_indices + indices

t_indices = []
for _, indices in test_indices.items():
    t_indices = t_indices + indices

trainset = dataset_filtered.iloc[tr_indices,:].copy()
testset = dataset_filtered.iloc[t_indices,:].copy()

# Split SVM

def classify(train_features,train_labels,test_features):
    clf = SVC(kernel='rbf', C=5, gamma=0.02, verbose=True)
    clf.fit(train_features, train_labels)
    print("\ndone fitting classifier\n")
    return clf.predict(test_features)

def evaluate(y_true,y_pred):
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score

def main(train_features,train_data,test_features,test_data):
    train_labels = train_data['labels'].tolist()

    test_labels = test_data['labels'].tolist()
        
    y_pred = classify(train_features,train_labels,test_features)
        
    recall, precision, f1_score = evaluate(test_labels, y_pred)
    
    print("recall: %s"%(recall))
    print("precision: %s"%(precision))
    print("f1 score: %s"%(f1_score))
    
    return recall, precision, f1_score

# Obtain BoW Features

train_tokens = trainset['tokens'].tolist()
train_corpus = createCorpus(train_tokens)
train_bow_features = bag_of_words_vectorizer.transform(train_corpus)

test_tokens = testset['tokens'].tolist()
test_corpus = createCorpus(test_tokens)
test_bow_features = bag_of_words_vectorizer.transform(test_corpus)

# Obtain LDA Features

def createCorpusLDA(tokens):
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token_list) for token_list in tokens]
    return (dictionary,corpus)

train_lda_dictionary, train_lda_corpus = createCorpusLDA(train_tokens)

train_lda_features = dok_matrix((len(train_lda_corpus),100))

for i in tqdm(range(len(train_lda_corpus))):
    topic_distribution = lda_model[train_lda_corpus[i]]
    for (topic_nr,prob) in topic_distribution:
        train_lda_features[i, topic_nr] = prob

train_lda_features_csr = train_lda_features.tocsr()

test_lda_dictionary, test_lda_corpus = createCorpusLDA(test_tokens)

test_lda_features = dok_matrix((len(test_lda_corpus),100))

for i in tqdm(range(len(test_lda_corpus))):
    topic_distribution = lda_model[test_lda_corpus[i]]
    for (topic_nr,prob) in topic_distribution:
        test_lda_features[i, topic_nr] = prob

test_lda_features_csr = test_lda_features.tocsr()

# Obtain Glove Features

NR_TRAIN_REVIEWS = len(train_tokens)

NR_TEST_REVIEWS = len(test_tokens)

train_glove_features = dok_matrix((NR_TRAIN_REVIEWS,100))

test_glove_features = dok_matrix((NR_TEST_REVIEWS,100))

content_tokens = dataset['content_tokens'].tolist()

for i in tqdm(range(NR_TRAIN_REVIEWS)):
    content_token_list = content_tokens[i]
    for content_token in content_token_list:
        content_index = content_word_dict[content_token]
        cluster_id = glove_labels[content_index]
        train_glove_features[i, cluster_id] = 1

for i in tqdm(range(NR_TEST_REVIEWS)):
    content_token_list = content_tokens[i]
    for content_token in content_token_list:
        content_index = content_word_dict[content_token]
        cluster_id = glove_labels[content_index]
        test_glove_features[i, cluster_id] = 1

train_glove_features_csr = train_glove_features.tocsr()

test_glove_features_csr = test_glove_features.tocsr()

# Combine Features

train_combined_features = sparse.hstack((train_bow_features,train_lda_features_csr,train_glove_features_csr))

test_combined_features = sparse.hstack((test_bow_features,test_lda_features_csr,test_glove_features_csr))

# Run SVM

start = datetime.now()
print(start)
recall, precision, f1_score = main(train_combined_features,trainset,test_combined_features,testset)
end = datetime.now()
print(end)

# Save Results

with open("output/bow_lda_glove_experiment.txt","w") as outfile:
    outfile.write("recall: %s\n"%(recall))
    outfile.write("precision: %s\n"%(precision))
    outfile.write("f1_score: %s\n"%(f1_score))
