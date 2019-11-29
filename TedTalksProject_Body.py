import numpy as np
import pandas as pd
import math
import pickle
import re
import string
import unicodedata
from collections import Counter
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from TedTalksFunctions import *

#################################
np.random.seed(0)

df1 = pd.read_csv("data/TEDonly_speakers_final.csv")
df2 = pd.read_csv("data/TEDplus_speakers_final.csv")
# concatenate the two main datasets
df = pd.concat([df1, df2])
# no duplicates found, but adding as part of standard pre-processing
df.drop_duplicates(subset ="Talk_ID", inplace = True)
# drop column with old csvs' row numbers
df.drop(['Unnamed: 0'], axis=1, inplace=True)
# since Talk ID's are unique, set as index
df.set_index('Talk_ID', inplace=True)

# create new column with character-length of each speech
df['speech_length'] = df['text'].apply(len)
# create new column to flag whether speech is over or under 1M views
df['reached_threshold'] = np.where(df['views']>=1250000, 1, 0)
# create new column to flag whether tags column contains (any of my) preferred list of tags
# social change, society, global issues, humanity, community, activism, future, health
df['prefers'] = np.where(
    df['tags'].str.contains('global')|df['tags'].str.contains('soci')
    |df['tags'].str.contains('nity') |df['tags'].str.contains('activism')
    |df['tags'].str.contains('future')|df['tags'].str.contains('health'), 1, 0)
# concatenate text from several columns to include it in main transcript column ('text')
cols = ['headline', 'tags', 'description', 'text']
df['text'] = df[cols].apply(lambda x: ' '.join(x), axis = 1)
# drop any rows in which 'tags' column contains these words
to_drop = ['performance', 'music', 'magic']
nomusic_df = df[~df['tags'].str.contains('|'.join(to_drop))]
# Missed one: Delete row with index/Talk_ID of 1464
nomusic_df = nomusic_df.drop(1464)
'''
potential to-do: truncate more non-lengthy speeches (find cut-off)
pd.set_option('max_colwidth',2000)
nomusic_df.sort_values(by=['speech_length']).text
# nomusic_df.sort_values(by=['speech_length']).speech_length

# Explode list of tags (skipped for now)
pre_expl_df = nomusic_df.reset_index(drop=True)
# (1) Create a new dataframe from the series with Talk_ID as the index
exploding_df = pd.DataFrame(pre_expl_df.tags.str.split(',').tolist(), index=pre_expl_df.Talk_ID).stack()
# (2) Now discard secondary index. To do this, make Talk_ID
# as a column (it can't be an index since the values will be duplicate)
exploded_df = exploding_df.reset_index([0, 'Talk_ID'])
# (3) The final step is to set the column names as we want them
exploded_df.columns = ['Talk_ID', 'tag']
'''
df = nomusic_df
# calling function clean_text_round1
round1 = lambda x: clean_text_round1(x)
df.text = pd.DataFrame(df.text.apply(round1))
# print(df.text[df.index == 2113])  # test it out
# calling function clean_text_round2
round2 = lambda x: clean_text_round2(x)
df.text = pd.DataFrame(df.text.apply(round2))
data_clean = df
data_clean['text'] = data_clean['text'].apply(lambda x: lemmadata(x))
pd.set_option('max_colwidth',2000)
# print(df.text[df.index == 2113])  # test it out
# print(data_clean.shape, data_clean.columns)
# We are going to create a document-term matrix using sklearn's CountVectorizer, and exclude common English stop words
# from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.text)
# print(data_cv.shape)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index

data = data_dtm.transpose()
# Find the top 30 words said in each speech
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))
# print(top_dict)

# Print the top 15 words
# for speech_num, top_words in top_dict.items():
#     print(speech_num)
#     print(', '.join([word for word, count in top_words[0:14]]))
#    print('---')

# Look at the most common top words --> add them to the stop word list
# Let's first pull out the top 30 words for each speech
words = []
for speech_num in data.columns:
    top = [word for (word, count) in top_dict[speech_num]]
    for t in top:
        words.append(t)
# print(words)
# Aggregate list (words) and identify the most common words along with how many times they occur
# print(Counter(words).most_common())

# stop words
j_stop_words = ['I', 'aa', 'aaa', 'aaaa', 'aaaaa', 'aaaaaah', 'aaaaaahaaaah',
                'aaaah', 'aaaahhh', 'aaah', 'aag', 'aah', 'aak', 'aakash',
                'aaleh', 'aarhus', 'aaron', 'aaronson', 'aaronsons', 'aarp',
                'aat', 'aatcagggaccc', 'ab', 'ababa', 'abacha', 'aback', 'abaco',
                'actually', 'applause', 'chris', 'come', 'did', 'different', 'dont',
                'ea', 'going', 'gonna', 'got', 'great', 'ha', 'honor', 'i', 'im',
                'is', 'just', 'kind', 'know', 'laughter', 'life', 'like', 'little',
                'lot', 'make', 'people', 'really', 'right', 'said', 'say', 'stage',
                'thank', 'thats', 'thing', 'think', 'time', 'truly', 'u', 'uaa', 'wa',
                'want', 'way', 'work', 'world', 'yeah', 'youre', 'zora',
                'zoroastrian', 'zosia', 'zq', 'zuccotti', 'zuckerberg', 'zuckerbergs',
                'zuckerman', 'zullinger', 'zune', 'zurich', 'zuzana', 'zweig',
                'zworkykins', 'zworykin', 'zygmunt', 'zygomatic', 'zygote', 'zywiecwa']
j_stop_words = sorted(list(set(j_stop_words)))
# If more than 400 of the speeches have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count >= 400]
add_stop_words.extend(j_stop_words)
# Update document-term matrix with the new list of stop words
# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.text)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index
# Find the number of unique words that each speech uses
# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
unique_list = []
for speech in data.columns:
    uniques = data[speech].to_numpy().nonzero()[0].size
    unique_list.append(uniques)

# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(data.columns, unique_list)), columns=['speech', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')

# To re-align shape (filtering out stopwords)
for sw in add_stop_words:
    if sw in data_dtm.columns:
        data_dtm.drop([sw], axis=1, inplace=True)
print(data_dtm.shape)

# Create quick lambda functions to find the polarity and subjectivity of each routine
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity
data_clean['polarity'] = data_clean['text'].apply(pol)
data_clean['subjectivity'] = data_clean['text'].apply(sub)
# Let's create a list to hold all of the pieces of text
list_pieces = []
for t in data_clean.text:
    split = split_text(t)
    list_pieces.append(split)
# print(list_pieces[:5])

# Calculate the polarity for each piece of text
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)
# plt.plot(polarity_transcript[0])
# plt.show()
######################
# Similarity Model
data_clean.reset_index(inplace=True, drop=True)
length = len(data_clean)
simple_data = data_clean[['public_url', 'headline', 'description', 'event', 'duration', 'published',
                         'speaker_1','speaker1_occupation', 'speaker1_introduction', 'speaker1_profile',
                         'speech_length', 'reached_threshold', 'prefers', 'polarity','subjectivity',
                          'text']]

# simple_data = simple_data.sample(frac=1).reset_index(drop=True) #shuffle
x = simple_data['text']
y = simple_data['prefers']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lemmatized = [lemmadata(t) for t in x_train]
tfidf = TfidfVectorizer()
response = tfidf.fit_transform(lemmatized)
tfidf_df = pd.DataFrame(response.toarray(), columns=tfidf.get_feature_names())

relevant = []
for word in tfidf_df.columns:
    if tfidf_df[word].mean() > 0.0001:
        relevant.append(tfidf_df[word])

relevant_df = pd.DataFrame(relevant).transpose()

pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(relevant_df.columns, open("relevantwords.pkl", "wb"))

mnb = MultinomialNB()
mnb.fit(relevant_df, y_train)
pickle.dump(simple_data, open("ff_simple.pkl", "wb"))
pickle.dump(mnb, open("mnbb.pkl", "wb"))
