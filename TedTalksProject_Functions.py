import numpy as np
import re
import string
import unicodedata
import math

import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import pickle
from sklearn.naive_bayes import MultinomialNB



# Apply a first round of text cleaning techniques
def clean_text_round1(text):
    '''Make text lowercase, remove text in parentheses, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    # inserted a contractions expander
    text = contractions.fix(text)
    text = re.sub('\[.*?\]', '', text)
    # further text standardizers
    text = re.sub('@\S+', '', text)
    text = re.sub("@", "at", text)
    text = re.sub('(Chris Anderson:)', '', text)
    # text = re.sub('\b(\(applause\)|\(laughter\))\b', '', text)
    text = re.sub('(\(applause\))', '', text)
    text = re.sub('(\(laughter\))', '', text)
    # remove extra whitespace
    text = re.sub(' +', ' ', text)
    # remove extra newlines
    text = re.sub(r'[\r|\n|\r\n]+', ' ', text)

    # remove everything w/ parentheses
    #    text = re.sub('\(.*?\)', '', text)
    #    text = re.sub('\([^)]*\)', '', text)
    #    text = re.sub(r'\([^()]*\)', '', text)

    # to remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', str(text))
    text = re.sub('\w*\d\w*', '', text)
    return text


# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”—…]', '', text)
    text = re.sub('\n', '', text)
    return text


# Lemmatizing Function
def lemmadata(doc):
    wordnet_lemmatizer = WordNetLemmatizer()
    english = set(nltk.corpus.words.words())
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    raw_tokens = nltk.regexp_tokenize(doc, pattern)
    tokens = [i.lower() for i in raw_tokens]
    stop_words = set(stopwords.words('english'))
    listed = [w for w in tokens if not w in stop_words]
    lemmatized = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in listed]
    words = list(filter(lambda w: w in english, lemmatized))
    return " ".join(words)


def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''
    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)

    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece] + size])
    return split_list


# TTP.py functions
def search_function(text_list, search_words):
    length = len(text_list)
    text_list = [text_list.iloc[i] for i in range(length)]

    def nltk_tokenize(text):
        return [w.lower() for w in word_tokenize(text)]

    gen_docs = []
    for i in range(length):
        gen_docs.append(nltk_tokenize(text_list[i]))
    dictionary = gensim.corpora.Dictionary(gen_docs)

    corpus = []
    for i in range(length):
        corpus.append(dictionary.doc2bow(gen_docs[i]))
    tf_idf = gensim.models.TfidfModel(corpus)
    #    print(tf_idf) # remove later
    sims = gensim.similarities.Similarity('', tf_idf[corpus],
                                          num_features=len(dictionary))
    #    print(sims) # remove later
    # TRY OUT QUERIES
    query_doc = [w.lower() for w in word_tokenize(search_words)]
    #    print(query_doc)  # for testing
    query_doc_bow = dictionary.doc2bow(query_doc)
    #    print(query_doc_bow) # for testing
    query_doc_tf_idf = tf_idf[query_doc_bow]
    #    print(query_doc_tf_idf) # for testing
    ans = sims[query_doc_tf_idf]
    ind = np.argmax(ans)
    print('INDEX = ', ind)
    return ind


def preprocess(data):
    lemmatized = [lemmadata(speech) for speech in data]
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    transformed = tfidf.transform(lemmatized)
    tfidf_df = pd.DataFrame(transformed.toarray(), columns=tfidf.get_feature_names())
    relevant = pickle.load(open("relevantwords.pkl", "rb"))
    testset = [tfidf_df[word] for word in relevant if word in tfidf_df.columns]
    return pd.DataFrame(testset).transpose()


def lemmadata(doc):
    wordnet_lemmatizer = WordNetLemmatizer()
    english = set(nltk.corpus.words.words())
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    raw_tokens = nltk.regexp_tokenize(doc, pattern)
    tokens = [i.lower() for i in raw_tokens]
    stop_words = set(stopwords.words('english'))
    listed = [w for w in tokens if not w in stop_words]
    lemmatized = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in listed]
    lemmatized = list(filter(lambda w: w != 'lb', lemmatized))
    words = list(filter(lambda w: w in english, lemmatized))
    return " ".join(words)


def classify_text(text, mnb):
    listtext = [text]
    processed = preprocess(listtext)
    return mnb.predict(processed)
