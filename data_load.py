import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib as plt
import re
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

stop=stopwords.words('english')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clear_tweets(text):
    text=re.sub('<[^>]*>','',text)
    emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower()) +' '.join(emojis).replace('-','')

    return text   


def func_2(tweets):
    token=word_tokenize(tweets)
    filtered_list = []
    for word in token:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    stemmed_words = [stemmer.stem(word) for word in filtered_list]
    return stemmed_words


def predict(tweet,test):
    data=pd.DataFrame(tweet,columns=['tweet'])
#     data.head()
    text=data['tweet'].apply(clear_tweets)
    text=text.apply(func_2)

    arr=[]
    for tweet in text:
        joined_str=""
        for word in tweet:
            joined_str=joined_str+" "+word
        arr.append(joined_str)

    text=pd.DataFrame(arr,columns=['tweet'])
    # test.shape

    test=test.append(text,ignore_index=True)
    # test.tail(10)
    vec= TfidfVectorizer(max_features=10000)
    # from sklearn.linear_model import LogisticRegressionCV
    
    xval=vectoriser_loaded.fit_transform(test.tweet)

    yval=model_loaded.predict(xval)

    final=pd.DataFrame(test['tweet'],columns=['tweet'])
    final['label']=yval
    return final.label.tail(1).values[0]
#     final.head()


def data_preprocess():
    new_list=[]
    arr=[]
    i=0
    for tweet in df_train['tweet']:
        new_list=func_2(tweet)
        joined_str=""
        for word in new_list:
            joined_str=joined_str+" "+word
        arr.append(joined_str)

    df_train['new_tweet']=arr

def load_data():
    test=pd.read_csv("test.csv")
    df_test=pd.read_csv("test.csv")
    test=df_test['tweet']
    test=test.apply(clear_tweets)
    test=test.apply(func_2)

    arr=[]

    for tweet in test:

        joined_str=""
        for word in tweet:
            joined_str=joined_str+" "+word
        arr.append(joined_str)

    test=pd.DataFrame(arr,columns=['tweet'])
    return test



test=load_data()


with open('saved_steps.pk1','rb') as file:
    data=pickle.load(file)
model_loaded=data['model']
vectoriser_loaded=data['vectoriser']

def redirect(tweet):
    new_tweet=[tweet]
    i=predict(new_tweet,test)
    
    return i


