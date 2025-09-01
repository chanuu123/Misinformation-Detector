# Misinformation detector

 A machine learning project designed to detect whether a news article is **real** or **fake** using **Natural Language Processing (NLP)** techniques.

## Overview

This project leverages text preprocessing, TF-IDF vectorization, and classification models to analyze news content and predict its authenticity. The goal is to help identify misinformation in digital media.

## Dataset

The dataset consists of news articles with associated labels (real or fake).
You can download the dataset from https://www.kaggle.com/competitions/fake-news/data?select=train.csv&utm_source=chatgpt.com

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

## How to Run

* Train the model on the dataset
* Evaluate performance on training and test sets
* Predict authenticity for new news text inputs

## Technologies used

* **Python**
* **NLTK** for text preprocessing
* **Scikit-learn** for ML models and evaluation
* **Pandas/Numpy** for data handling

## Results

The trained models demonstrates reliable performance in distinguishing between reala and fake news showcasing the potential of NLP + ML in combating misinformation. 

## Future Enhancements

* Incorporate social context features for better prediction
* Use larger and more diverse datasets to reduce bias
* Experiment with advanced models (e.g., BERT, LSTM)

## License
This project is open-source and available under the MIT License.
* Collecting a larger and more diverse dataset to train the model, potentially reducing biases and improving generalization.

