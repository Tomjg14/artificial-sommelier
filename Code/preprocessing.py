# data loading\n
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
from scipy.sparse import dok_matrix

# computations
import numpy as np
import math\n
from numpy import asarray
from numpy import zeros
from keras.utils import np_utils

# visualization
import matplotlib
from matplotlib import pyplot as plt
from pprint import pprint
import time

# data saving
import pickle

# progress
from tqdm import tqdm_notebook as tqdm

file_path = "data/winemag-data-130k-v2.json"

with open(file_path) as f:
    data = json.load(f)
    dataset = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

descriptions = dataset['description'].tolist()

def retrievePOS(descriptions):
    tags = []
    for description in tqdm(descriptions):
        text = nltk.word_tokenize(description)
        pos_tags = nltk.pos_tag(text)
        pos_dict = {}
        for (word,pos) in pos_tags:
            pos_dict[word.lower()] = pos
            tags.append(pos_dict)
    return tags

tags = retrievePOS(descriptions)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

text = nltk.word_tokenize(descriptions[2])
text = [t for t in text if len(t) > 1 and not hasNumbers(t)]
pos = nltk.pos_tag(text)
tokens = []
pos_tags = []
stop_words = set(stopwords.words('english'))
for (word,pos_tag) in pos:
    if not word.lower() in stop_words:
        tokens.append(word.lower())
        pos_tags.append(pos_tag)

def getTokens(descriptions):
    translator = str.maketrans('', '', string.punctuation)
    tokens = []
    for description in tqdm(descriptions):
        # remove punctuation
        description_without_punctuation = description.translate(translator)
        # make lowercase
        description_lower = description_without_punctuation.lower()
        # tokenize
        description_tokens = nltk.word_tokenize(description_lower)
        # remove remaining tokens that are not alphabetic
        description_tokens = [word for word in description_tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        description_tokens = [w for w in description_tokens if not w in stop_words]
        #filter out short tokens
        description_tokens = [word for word in description_tokens if len(word) > 1]
        tokens.append(description_tokens)
    return tokens

def getTokens2(descriptions):
    dataset_tokens = []
    dataset_pos_tags = []
    for description in tqdm(descriptions):
        # tokenize
        description_tokens = nltk.word_tokenize(description)
        # remove tokens with length of 1 and digits
        description_tokens = [t for t in description_tokens if len(t) > 1 and not hasNumbers(t)]
        # compute part-of-speech tags
        pos = nltk.pos_tag(description_tokens)
        tokens = []
        pos_tags = []
        stop_words = set(stopwords.words('english'))
        for (word,pos_tag) in pos:
            if not word.lower() in stop_words:
                tokens.append(word.lower())
                pos_tags.append(pos_tag)
            dataset_tokens.append(tokens)
            dataset_pos_tags.append(pos_tags)
        return dataset_tokens, dataset_pos_tags

tokens = getTokens(descriptions)  

tokens, pos_tags = getTokens2(descriptions)

pickle.dump(tokens,open("tokens.p","wb"))
pickle.dump(pos_tags,open("pos_tags.p","wb"))

dataset['tokens'] = tokens
dataset['pos_tags'] = pos_tags

