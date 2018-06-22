# Artificial Sommelier

This is a project for the course Cognitive Computational Modeling of Language and Web Interaction from the Radboud University (Nijmegen). The goal of this project was to reproduce the work by Hendrickx et al.<sup>[1](#hendrikx-footnote)</sup> but give it our own twist. We were mostly interested in trying to predict the score for a specific wine given an expert's review. 

This page will explain our way of working. We will go into more detail on the precise preprocessing steps we took, the way we split the data and how we defined our feature vectors. Then finally, we will show what classifier we trained, with what settings and how we evaluated its performance.

We worked with Python and Jupyter. This is why this repository contains code in both formats. However, all our work was done in Jupyter notebooks. The code in the .py files is simply copied from these notebooks and put in these files for convenience. There is no guarantee that the .py files will work. 

This blog has the following structure:
* [The Data](#the-data)
* [Preprocessing](#preprocessing)
* [Compute Content Words](#compute-content-words)
* [Compute Labels](#compute-labels)
* [Compute BoW Corpus](#compute-bow-corpus)
* [Data Filtering](#data-filtering)
* [Data Splitting](#data-splitting)
* [Define SVM](#define-svm)
* [Obtain BoW Features](#obtain-bow-features)
* [Obtain LDA Features](#obtain-lda-features)
* [Obtain W2V Features](#obtain-w2v-features)
* [Obtain GLOVE Features](#obtain-glove-features)
* [Results](#results)

<a name="hendrikx-footnote">1</a>: [Hendrickx et al.](http://www.aclweb.org/anthology/P16-2050)

## The Data

The data we will be working with dataset we collected from Kaggle. Kaggle is a platform that facilitates machine learning related competitions and enables users to share datasets. One of these datasets was the [wine review](https://www.kaggle.com/zynicide/wine-reviews) dataset. This dataset contains ~130k different wine reviews written by wine experts. The wine reviews were originally posted on [winemag.com](https://www.winemag.com/?s=&drink_type=wine). 

The dataset contains attributes like the wine variety, country, price, description, and points. The work by Hendrikx et al. focused on the attributes color, grape variety, countries and price. Therefore, we will be looking at the points per wine. To be more precise: we will be trying to classify what points/score belongs to a specific description of a wine. 

The descriptions, as said, are written by wine experts and are thus filled with very descriptive terms. Here a short example:

_Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity._

Our first guess is that the usage of specific adjectives already tells a lot about the amount of points a wine might get. Therefore, we will be working with different Natural Language Processing methods to be able to classify these different reviews.

## Preprocessing

Lets start with how we loaded and preprocessed the data before we move on to the actual classifying. 

The data was provided in a .json format, which makes it very easy to load the data by making use of pandas DataFrames:

```python
from pandas.io.json import json_normalize
import json
import pandas as pd

file_path = "data/winemag-data-130k-v2.json"

with open(file_path) as f:
    data = json.load(f)
    dataset = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

descriptions = dataset['description'].tolist()
```

Next we start by computing the Part-of-Speech tags per word in the dataset. This is needed as we want to reproduce the work by Hendrikx et al. as closely as possible and in their work they filter on content words (nouns, verbs, adjectives). Therefore, before we split the data or remove any terms from the reviews, we first need to perform pos-tagging. Luckily there exist libraries for this:

```python
import nltk
from tqdm import tqdm_notebook as tqdm

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
```

Now we can start with the actual preprocessing:

```python
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def getTokens(descriptions):
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

tokens, pos_tags = getTokens(descriptions)

dataset['tokens'] = tokens
dataset['pos_tags'] = pos_tags
```

First we tokenize the reviews to obtain a list of words and then we remove words of length 1 or that contain digits. After this cleaning we start with the part of speech tagging. Finally, we remove stop words and make every word lowercase.

To skip this step in the future, we make use of pickle to save variables.

```python
pickle.dump(tokens,open("tokens.p","wb"))
pickle.dump(pos_tags,open("pos_tags.p","wb"))
```

This were all the preprocessing steps we performed on the actual reviews. Next sections will explain how we filter each review on content words, how we turn the continuous points into one of six labels and how we compute the Bag-of-word Corpus for the entire dataset.

## Compute Content Words

## Compute Labels

## Compute BoW Corpus

## Data Filtering

## Data Splitting

## Define SVM

## Obtain BoW Features

## Obtain LDA Features

## Obtain W2V Features

## Obtain GLOVE Features

## Results
