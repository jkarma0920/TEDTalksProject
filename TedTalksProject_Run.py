import pickle
import time

import gensim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from selenium import webdriver
from TedTalksFunctions import *
########################################################

def search_function(sims, search_words):
    query_doc = [w.lower() for w in word_tokenize(search_words)]
#    print(query_doc)  # remove later
    query_doc_bow = dictionary.doc2bow(query_doc)
#    print(query_doc_bow) # remove later
    query_doc_tf_idf = tf_idf[query_doc_bow]
#    print(query_doc_tf_idf) # remove later
    ans = sims[query_doc_tf_idf]
    ind = np.argmax(ans)
    print('INDEX = ', ind)
    return ind


# def preprocess(data):
#     lemmatized = [lemmadata(speech) for speech in data]
#     tfidf = pickle.load(open("tfidf.pkl", "rb"))
#     transformed = tfidf.transform(lemmatized)
#     tfidf_df = pd.DataFrame(transformed.toarray(), columns=tfidf.get_feature_names())
#     relevant = pickle.load(open("relevantwords.pkl", "rb"))
#     testset = [tfidf_df[word] for word in relevant if word in tfidf_df.columns]
#     return pd.DataFrame(testset).transpose()


def classify_text(text, mnb):
    listtext = [text]
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    lemmatized = [lemmadata(speech) for speech in listtext]
    transformed = tfidf.transform(lemmatized)
    tfidf_df = pd.DataFrame(transformed.toarray(), columns=tfidf.get_feature_names())
    relevant = pickle.load(open("relevantwords.pkl", "rb"))
    testset = [tfidf_df[word] for word in relevant if word in tfidf_df.columns]
    processed = pd.DataFrame(testset).transpose()
    return mnb.predict(processed)


simple_data = pd.read_pickle('simplified_data.pkl')
sims = pd.read_pickle('sims.pkl')
polarity_transcript = pd.read_pickle('polarity_transcript.pkl')
mnb = pd.read_pickle('mnbb.pkl')

# test out with whole texts from df
sample_text = simple_data.text.iloc[2]
print(sample_text)
num = ''
num2 = ''
choice = ''
plt.rcParams['figure.figsize'] = [10, 6]  # for polarity transcript's display
while num != '1' and num != '2':
    num = input('Enter\n1 for TEDTalks Search\n2 for TEDTalks Classifier: ')
    if num == '1':
        while num2 != '1' and num2 != '2':
            num2 = input('Enter\n1 to Search by Description\n2 to Search by Speech Transcript: ')
            if num2 == '1':
                choice = 'description'
            elif num2 == '2':
                choice = 'text'
            #search_words = input('Enter Search Term(s): ')
            search_words = sample_text # test queries
            idx = search_function(simple_data[choice], search_words)
            #pd.set_option('max_colwidth', 500)

            print(pd.DataFrame(simple_data.iloc[idx].T))  # print result.transposed
            driver = webdriver.Chrome()
            driver.get(simple_data['public_url'].iloc[idx])
            play = driver.find_element_by_xpath('//*[@id="plyr-play"]')
            time.sleep(3)
            play.click()
            # Show the plot for one speech (to be built on top of...)
            plt.plot(polarity_transcript[idx])
            #plt.title(simple_data.headline[ind])
            plt.show()
    elif num == '2':
        search_words = input('Enter Search Term(s): ')
        result = classify_text(search_words, mnb)
        if result == [1]:
            print("One of Jay's favorite topics!")
        else:
            print("Not one of Jay's favorite topics")