# TDT4173-Project

## Overview

This repository contains the code for training the model used for the project in the NTNU course TDT4173 Machine Learning. The code can be found in the main branch. The goal of the model is to read a comment, and predict if it was written on Reddit, Hacker News or YouTube. The model was trained on the following data sets:

* [May 2015 Reddit Comments](https://www.kaggle.com/reddit/reddit-comments-may-2015)
* [Hacker News Corpus](https://www.kaggle.com/hacker-news/hacker-news-corpus)
* [Trending YouTube Video Statistics and Comments](https://www.kaggle.com/datasnaek/youtube)

## Reguirements

* numpy >= 1.18.5
* TensorFlow >= 2.3.0
* Keras >= 2.4.0
* TensorFlowJS >= 2.7.0
* scikit-learn >= 0.23.2

## Website

We also created a website with an application using the model. There are also two branches, website and server, which contains the front-end and back-end code for the website with our application using the model. The website can be found [here](https://commentclassifier.herokuapp.com/). It is hosted on Heroku, so it might use some time loading the first time it is opened.

The website was created using Node.js with Express.js for the back-end, together with React and Bootstrap for the front-end. We also used TensorFlow.js to use the already trained model on the back-end.
