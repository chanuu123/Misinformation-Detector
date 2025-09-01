# Misinformation detector

 A machine learning project designed to detect whether a news article is **real** or **fake** using **Natural Language Processing (NLP)** techniques.

## Overview

This project leverages text preprocessing, TF-IDF vectorization, and classification models to analyze news content and predict its authenticity. The goal is to help identify misinformation in digital media.

## Dataset

The dataset consists of news articles with associated labels (real or fake).
You can download the dataset from https://www.kaggle.com/code/therealsampat/fake-news-detection

## Installation & Requirements

* Install the required dependencies: 
                                             "pip install pandas nltk scikit-learn"
* Also, download the necessary NLTK resources: 
                                             "import nltk
                                             nltk.download('stopwords')"

## Preprocessing Steps

The text data undergoes several NLP transformations:

* Remove special and non-alphabetic characters
* Convert text to lowercase
* Tokenization
* Stopword removal
* Stemming using Porter Stemmer

## Model Pipeline

1) Feature Extraction: TF-IDF (Term Frequency - Inverse Document Frequency)
2) Model: Logistic Regression
3) Evaluation Metrics: Accuracy & F1-Score

## Technologies used

* **Python**
* **NLTK** for text preprocessing
* **Scikit-learn** for ML models and evaluation
* **Pandas/Numpy** for data handling

## Results

The trained models demonstrates reliable performance in distinguishing between reala and fake news showcasing the potential of NLP + ML in combating misinformation. 



