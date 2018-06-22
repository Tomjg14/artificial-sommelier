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
