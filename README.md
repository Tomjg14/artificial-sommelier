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

<a name="hendrikx-footnote">1</a>: http://www.aclweb.org/anthology/P16-2050
