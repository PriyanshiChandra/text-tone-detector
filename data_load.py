import streamlit as st
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
from datetime import date
from nltk import NaiveBayesClassifier
from sklearn import *
import pickle

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()




yelp=pd.read_csv("training-models/yelp.csv")
amazon=pd.read_csv("training-models/amazon.csv")
new_dataset=amazon.append(yelp)
imdb=pd.read_csv("training-models/imdb.csv")

def convert_int(text):
    return int(text)

imdb['label']=imdb['label'].apply(convert_int)
new_dataset=new_dataset.append(imdb)


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

def addition_to_database(option,text,prev_tone):
    
    if option == "Yes":
        if prev_tone == "Positive":
            label_new=1
        else:
            label_new=0
    else:
        if prev_tone == "Positive":
            label_new=0
        else:
            label_new=1
        
    new_dataset.loc[len(new_dataset.index)] = [(new_dataset.tail(1).id.values[0]+1), label_new, text]
    st.write(new_dataset.tail(1)['tweet'].values[0])
    today = str(date.today())
    # model_training(today)

def model_training(k):
    new_dataset['cleared_tweets']=new_dataset['tweet'].apply(clear_tweets)

    new_dataset['tokenized_tweets']=new_dataset['cleared_tweets'].apply(tokenize_tweets)

    new_dataset['stopwords_removed']=new_dataset['tokenized_tweets'].apply(stopword_removal)

    new_dataset['stemmed_tweets']=new_dataset['stopwords_removed'].apply(stemming_tweets)
    pos_tweet_list=list(new_dataset['stemmed_tweets'][new_dataset['label']==1].apply(tokenize_tweets))
    neg_tweet_list=list(new_dataset['stemmed_tweets'][new_dataset['label']==0].apply(tokenize_tweets))

    pos_tokens = get_tweets_for_model(pos_tweet_list)
    neg_tokens = get_tweets_for_model(neg_tweet_list)
    positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in pos_tokens]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in neg_tokens]

    dataset = positive_dataset + negative_dataset
    
    
    new_classifier = NaiveBayesClassifier.train(dataset)
    import pickle
    data={"classifier":new_classifier}
    name_of_file = 'saved_steps_'+k+'.pk1'
    with open (name_of_file,'wb') as file:
        pickle.dump(data,file)

