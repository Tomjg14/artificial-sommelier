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
* [Perform Grid Search](#perform-grid-search)
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
from nltk.corpus import stopwords

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
import pickle

pickle.dump(tokens,open("tokens.p","wb"))
pickle.dump(pos_tags,open("pos_tags.p","wb"))
```

This were all the preprocessing steps we performed on the actual reviews. Next sections will explain how we filter each review on content words, how we turn the continuous points into one of six labels and how we compute the Bag-of-word Corpus for the entire dataset.

## Compute Content Words

Hendrikx et al. defined content words as either nouns, verbs or adjectives. So it seems as if we should filter on those three labels. However, when we performed the pos-tagging we worked with the Standford NLTK library which makes use of way more labels then just these three. Therefore we worked with the following code:

```python
def isContentWord(pos_tag):
    content_tags = ["JJ", "JJR", "JJS", "NN", "NNP", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    if pos_tag in content_tags:
        return True
    else:
        return False
```

By making use of this custom function we can determine if a certain word has one of many content-word labels. This way only content words are kept.

```python
content_words = []
for i, token_list in enumerate(tokens):
    pos_tag_list = pos_tags[i]
    for j, word in enumerate(token_list):
        pos_tag = pos_tag_list[j]
        if isContentWord(pos_tag):
            content_words.append(word)
```

We then use a Counter to count how many times each content word occurred in the dataset.

```python
content_counts = Counter(content_words)
```

Next we filter on content words that have more than 2 occurrences in the dataset. This is different from Hendrikx et al. as they filtered on more than 1 occurrences. We decided on more than 2 to reduce the total amount of content words, something that was necessary to decrease the size of our feature vectors and to fasten training time.

```python
filtered_content_words = []
for word in tqdm(content_words):
    if content_counts[word] > 2:
        if not word in filtered_content_words:
            filtered_content_words.append(word)
```

```python
CONTENT_COUNT = len(np.unique(filtered_content_words))
```

Here we initialize a dictionary where each content word is a key and its value is an unique index that will help to create the Bag-of-words feature vector.

```python
content_word_dict = {}
for i, word in enumerate(filtered_content_words):
    content_word_dict[word] = i
```

Finally, go over all reviews and only keep the content words.

```python
content_tokens = []
for token_list in tqdm(tokens):
    filtered_tokens = []
    for token in token_list:
        if token in content_word_dict:
            filtered_tokens.append(token)
    content_tokens.append(filtered_tokens)
```

Each new review (only containing content words) are then appended to the entire dataset.

```python
dataset['content_tokens'] = content_tokens
```

## Compute Labels

As we will try to classify the label of a description instead of the actual score, we first need to convert these scores into categories. Luckily, winemag.com had their own labels. 

![winemag labels][labels]

There are 7 labels, but winemag.com does not publish reviews with a score below 80.

```python
scores = dataset['points'].tolist()
```

```python
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
```

```python
categories, labels = getCategory(scores)

dataset['category'] = categories
dataset['labels'] = labels
```

[labels]: https://github.com/Tomjg14/artificial-sommelier/blob/master/images/labels.JPG 'winemag logos'

## Compute BoW Corpus

The final step we need to perform before we filter and split the dataset into training and test is to compute the Bag-of-Word corpus.

```python
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
```

```python
corpus = createCorpus(tokens)
bag_of_words_vectorizer = CountVectorizer(min_df=2)
bow_feature_vector = bag_of_words_vectorizer.fit_transform(corpus)
```

We first create a string of each review by joining the individual content words. Then we store all these strings inside a list which we then fit to a CountVectorizer. Important thing to note is that we use fit_transform on the entire corpus. When we split the data into training and test we will use only fit(). As the vectorizer is already prepared with the entire dataset. 

## Data Filtering

Like Hendrikx et al. we will be filtering out certain reviews. We filter based on the number of reviews per wine. The threshold is set to 200. 

```python
varieties = dataset['variety'].tolist()
wine_count = Counter(varieties)
nr_reviews = len(varieties)
threshold = 200
```

```python
filtered_keys = []
for key, item in wine_count.items():
    if item < threshold:
        filtered_keys.append(key)
```

```python
idx = 0
indices = []
for v in varieties:
    if v in filtered_keys:
        indices.append(idx)
    idx += 1
```

```python
dataset_filtered = dataset.drop(dataset.index[indices]).copy()
dataset_filtered = dataset_filtered.reset_index()

print("New Total reviews: %s"%(len(dataset_filtered)))
```

After filtering, we are left with 118263 reviews.

## Data Splitting

Next we split the data into a training and test set. We decided to turn 80% into training and 20% into test and we try to balance both sets on the different wines. 

```python
wines = np.unique(dataset_filtered['variety'].tolist())
```

We create a dictionary with the different wines as keys and there indices in the dataset as values.

```python
wine_indices = {}
for wine in wines:
    indices = dataset_filtered.index[dataset_filtered['variety'] == wine].tolist()
    wine_indices[wine] = indices
```

Then we use this dictionary to decide which dataset indices will go to the trainingset and which will go to the testset.

```python
train_indices = {}
test_indices = {}
for wine in wines:
    indices = wine_indices[wine]
    nr_indices = len(indices)
    train_indices[wine] = indices[:round(nr_indices*0.8)]
    test_indices[wine] = indices[round(nr_indices*0.8):]
    
    nr_train_indices = len(indices[:round(nr_indices*0.8)])
    nr_test_indices = len(indices[round(nr_indices*0.8):])
```

Finally, as we worked per wine we need to put all training and test indices into one big list. This list can then we used to copy a subset of the original dataset.

```python
tr_indices = []
for _, indices in train_indices.items():
    tr_indices = tr_indices + indices
```

```python
t_indices = []
for _, indices in test_indices.items():
    t_indices = t_indices + indices
```

```python
trainset = dataset_filtered.iloc[tr_indices,:].copy()
testset = dataset_filtered.iloc[t_indices,:].copy()
```

## Define SVM

We will be working with a Support Vector Machine to classify the different reviews. This section provides the different methods used and the parameter settings we used for our SVM. These settings were obtained by performing grid search, which we will explain in more detail in the following section. Just like Hendrikx et al. we made use of the RBF kernel instead of a linear one.

![rbf_kernel][rbf]

```python
def classify(train_features,train_labels,test_features):
    clf = SVC(kernel='rbf', C=5, gamma=0.02, verbose=True)
    clf.fit(train_features, train_labels)
    print("\ndone fitting classifier\n")
    return clf.predict(test_features)
```

```python
def evaluate(y_true,y_pred):
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score
```

```python
def main(train_features,train_data,test_features,test_data):
    train_labels = train_data['labels'].tolist()

    test_labels = test_data['labels'].tolist()
        
    y_pred = classify(train_features,train_labels,test_features)
        
    recall, precision, f1_score = evaluate(test_labels, y_pred)
    
    print("recall: %s"%(recall))
    print("precision: %s"%(precision))
    print("f1 score: %s"%(f1_score))
    
    return recall, precision, f1_score
```

In order to evaluate the performance of our SVM we are using f1-score. F1-score takes both recall and precision into consideration.

[rbf]: https://github.com/Tomjg14/artificial-sommelier/blob/master/images/svm_rbf.JPG 'svm with rbf kernel'

## Perform Grid Search

To figure out the optimal parameter settings for our SVM we decided to perform Grid Search. This is a method were the user specifies different parameter settings and where the classifier is tested for the amount of possible parameter combinations to figure out the most optimal setting.

```python
random_sample = trainset.sample(5000)
```

```python
random_tokens = random_sample['tokens'].tolist()
random_corpus = createCorpus(random_tokens)
```

```python
random_features = bag_of_words_vectorizer.transform(random_corpus)
random_labels = random_sample['labels'].tolist()
```

As we did not have that much time left until the deadline, we decided on the following parameters to reduce running time:

```python
parameters = {'kernel':['rbf'], 'C': np.arange(1,40,2), 'gamma': np.linspace(0.0, 0.2, 11)}
```

```python
svc = SVC()
clf = GridSearchCV(svc, parameters, verbose=10, n_jobs=4)
clf.fit(random_features,random_labels)
params = clf.best_params_
```

The best parameter settings were: {'kernel': 'rbf', 'gamma': 0.02, 'C': 5}.

## Obtain BoW Features

## Obtain LDA Features

## Obtain W2V Features

## Obtain GLOVE Features

## Results
