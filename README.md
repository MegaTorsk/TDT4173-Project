# TDT4173-Project

### Overview

This repository contains the code for the final project in the NTNU course TDT4173 Machine Learning. The code for data preprocessing and training the model can be found in the main branch. The finalized machine learning model reads a comment and tries to predict which social media website the comment was written on. Currently, the model is able to predict comments written on Reddit, Hacker News and YouTube. The following data sets were used for training:

* [May 2015 Reddit Comments](https://www.kaggle.com/reddit/reddit-comments-may-2015)
* [Hacker News Corpus](https://www.kaggle.com/hacker-news/hacker-news-corpus)
* [Trending YouTube Video Statistics and Comments](https://www.kaggle.com/datasnaek/youtube)

### Requirements

* numpy >= 1.16.5
* TensorFlow >= 2.3.0
* Keras >= 2.4.0
* TensorFlowJS >= 2.7.0
* scikit-learn >= 0.23.2
* keras-tuner >= 1.0.1
* nltk >= 3.5
* emoji >= 0.6.0

### How to run

1. Install Python dependencies using `pip install -r requirements.txt`
1. Download the data sets into a directory.
1. Update the directory that the `preprocessing.ipynb` file uses when reading in data sets, and when storing processed data sets, and run the notebook in order.
1. Change the data directories in `logistic_regression.py` and `model.py` to the processed data files.
1. Run `logistic_regression.py` for testing logistic regression.
1. Run `model.py` to perform both hyperparameter optimization and training the final model.

### Application website

We also created a [website](https://commentclassifier.herokuapp.com/) with an implementation of the trained model. It is hosted on Heroku, so be aware that it might take some time loading the first time. This GitHub repository contains two additional branches: website and server, which hosts the front-end and back-end code for the website. Details regarding the code for the website can be found in those branches.
