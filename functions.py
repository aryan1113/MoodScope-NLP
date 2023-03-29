def libraries_basic():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    return()
    
def libraries_clean():
    import nltk 
    import string
    import warnings
    from nltk.corpus import stopwords
    return()

'''
Remove all instances of amp
Removing weird characters
Converting all tweets to lowercase
Removing Punctuation
Removing URL's
'''

def clean_words():
    df['SentimentText'] = df['SentimentText'].str.replace('amp', '', case=False)

    df['SentimentText'] = df['SentimentText'].str.replace('®|©|¯|ª|¿|¾|¨|à|¸|£|ˆ|‡|•|‰|ž|«|”|¢|—|µ|¡|›|¥|‚|–|ð|Ÿ|™|á|º|·|ã|¹|»|±|³|€|¬|‹|¤|§|°|ì|š|í|†|ë|¦|„|¼|´|²|½|', '', case=False)    
    df['SentimentText']=df['SentimentText'].str.lower()
    df['SentimentText']=df['SentimentText'].str.translate(str.maketrans('', '', string.punctuation))
    df['SentimentText'] = df['SentimentText'].str.replace('http\S+|www.\S+', '', case=False)
    return()

"""
The word "amp" is seen in all wordclouds, this is an issue as amp is not a stopword and it's presence may end up feeding wrong information to the model.
The amp-twitter component allows you to embed a Tweet or Moment.
HTML encoding has not been converted to text, and ended up in text field, so we remove "amp" from all tweets.
"""


def libraries_neural():
    import tensorflow as tf

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from sklearn.feature_extraction.text import CountVectorizer
    from keras.preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences

    from keras.models import Sequential
    from keras.layers import Dense, Embedding, LSTM
    from keras.utils.np_utils import to_categorical

    from tensorflow.keras.layers import Conv1D, Bidirectional, Input, Dropout
    from tensorflow.keras.layers import SpatialDropout1D
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    from tensorflow.keras.optimizers import Adam

    from sklearn.metrics import classification_report

    import pickle

def predict_tweet_sentiment(score):return "Positive" if score>0.5 else "Negative"