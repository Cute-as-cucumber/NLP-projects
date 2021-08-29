#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 17:54:11 2021

@author: mm
"""

#Twitter sentiment analysis

#import the libraries and datasets needed
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tweets_df = pd.read_csv('/Users/mm/Desktop/NLP/twitter sentiment analysis/twitter.csv')

#checking if miss anything
tweets_df.shape
tweets_df.info()

#drop the column that we won't need
tweets_df = tweets_df.drop(['id'], axis = 1)

#axis = 1 indictes that we're gonna drop the whole column. Without it, error
tweets_df.info()
tweets_df.describe()

#visualize/check/see tweets
tweets_df['tweet']

#exploring the datasets
sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues')
#plain white shows that there is no null element
tweets_df.hist(bins = 30, figsize = (13, 5), color = 'r')
#or use seaborn
sns.countplot(tweets_df['label'])

#check the length of each tweet
#create a new column, and let it be length of each tweet
tweets_df['length'] = tweets_df['tweet'].apply(len)
tweets_df['length']
tweets_df['length'].plot( bins = 100, kind = 'hist')

#Find the shortest tweet
tweets_df.describe()
tweets_df[tweets_df['length'] == 11.0]['tweet'].iloc[0] #????

#divide the tweets into two dataframes
positive = tweets_df[tweets_df['label'] == 0]
positive
negative = tweets_df[tweets_df['label'] == 1]
negative

#tweets_df.head()

#Plotting word cloud
#Concatenate all the tweets into a single string
sentences = tweets_df['tweet'].tolist()
sentences
tweets_df.shape
len(sentences)#check if we got all the tweets in the list
sentences_one_string = ' '.join(sentences)

from wordcloud import WordCloud
wc = WordCloud().generate(sentences_one_string)
plt.figure(figsize = (20, 20)) #figsize: size of the figure image
plt.imshow(wc)

#plot the positive wordcloud
negative_tweets = negative['tweet'].tolist()
negative_one_string = ' '.join(negative_tweets)
nc = WordCloud().generate(negative_one_string)
plt.imshow(nc)

#perform data cleaning
#1 remove punctuations from text

import string
string.punctuation #import punctuations

#Just testing removing punc
#Test_sentence = 'Dany will know what Jon will eat :) She said: "I know everything. And you?"'
#Test_punc_rm = [word for word in Test_sentence if word not in string.punctuation]
#Test_punc_rm = ''.join(Test_punc_rm)
#Test_punc_rm

#remove stopwords
from nltk.corpus import stopwords
stopwords.words('english')

#Testing cleaning
#Test_cleaned = [word for word in Test_punc_rm.split() if word.lower() not in stopwords.words('english')]
#Test_cleaned

#tokenization
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is Queen Dany', 'This is King Jon', 'Aery is King and Queen baby', 'This is Aery']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)
#Oftern we use fit_transform() on the training data and transform() on the test data
print(vectorizer.get_feature_names())
print(X.toarray())

#create the pipeline to remove punc and stw; perform vectorization

def message_cleaning(message):
    '''Remove the punctuation and stopwords from the text'''
    rm_punc = [word for word in message if word not in string.punctuation]
    rm_punc_joint = ''.join(rm_punc)
    rm_punc_stw = [word for word in rm_punc_joint.split() if word.lower() not in stopwords.words('english')]
    return rm_punc_stw

#message_cleaning(Test_sentence)


#Check if runs well
tweets_df_cleaned = tweets_df['tweet'].apply(message_cleaning)
tweets_df_cleaned.describe
print(tweets_df_cleaned[5])
print(tweets_df['tweet'][5])#compare the two, see if something's missing

#vectorization(slightly different from how it was used above)

from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(analyzer = message_cleaning)
#This means, we let it call message_cleaning to PREPROCESS the data
tweets_countvectorized = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8').fit_transform(tweets_df['tweet']).toarray()
tweets_countvectorized.shape

X = tweets_countvectorized
y = tweets_df['label']

#Train a naïve bayes classifier model
X.shape
y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#split the data into training set and test set according or not according to a certain ratio

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
NB_classifier

#Assess trained model performance

from sklearn.metrics import classification_report, confusion_matrix
y_test_predicted = NB_classifier.predict(X_test)#predict the test results, compare it with the actual data to access performance

#plot the confusion matrix
cm = confusion_matrix(y_test, y_test_predicted)
sns.heatmap(cm, annot = True) #'annot = True' to show numbers

print(classification_report(y_test, y_test_predicted))









