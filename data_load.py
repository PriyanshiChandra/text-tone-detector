import pandas as pd
import numpy as np
import matplotlib as plt
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords

from sklearn import *
import pickle

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clear_tweets(text):
    text=re.sub('<[^>]*>','',text)
    emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower()) +' '.join(emojis).replace('-','')

    return text  

def tokenize_tweets(tweet):
    return tweet.split()

def stopword_removal(token_tweet):
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))
    filtered_list=[]
    for word in token_tweet:
            if word.casefold() not in stop_words:
                filtered_list.append(word)
    return filtered_list


def stemming_tweets(tweet):
    stemmed_words = [stemmer.stem(word) for word in tweet]
    joined_str=""
    for word in stemmed_words:
        joined_str=joined_str+" "+word
    return joined_str

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def prediction(custom):
    
    with open('saved_steps.pk1','rb') as file:
        data=pickle.load(file)
    model_loaded=data['classifier']
    
    custom=clear_tweets(custom)
    custom=tokenize_tweets(custom)
    custom=stopword_removal(custom)
    custom=stemming_tweets(custom)
    custom=tokenize_tweets(custom)
    return (model_loaded.classify(dict([token, True] for token in custom)))
