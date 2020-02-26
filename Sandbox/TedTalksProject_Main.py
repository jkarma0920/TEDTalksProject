import numpy as np
import pandas as pd
import math
import pickle
import string
import unicodedata
from collections import Counter

import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import streamlit as st

#################################
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

# Apply a first round of text cleaning techniques
def clean_text_round1(text):
    '''Make text lowercase, remove text in parentheses, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    #inserted a contractions expander
    text = contractions.fix(text)
    text = re.sub('\[.*?\]', '', text)
    # further text standardizers, taken from Eman's code
    text = re.sub('@\S+', '', text)
    text = re.sub("@", "at", text)
    text = re.sub('(Chris Anderson:)', '', text)
    # text = re.sub('\b(\(applause\)|\(laughter\))\b', '', text)
    text = re.sub('(\(applause\))', '', text)
    text = re.sub('(\(laughter\))', '', text)

    # remove extra whitespace
    text = re.sub(' +', ' ', text)
    # remove extra newlines
    text = re.sub(r'[\r|\n|\r\n]+', ' ',text)

#     text = text.apply(lambda x: word_lemmatizer(x))

    #Jay's attempts to remove everything w/ parentheses
#    text = re.sub('\(.*?\)', '', text)
#    text = re.sub('\([^)]*\)', '', text)
#    text = re.sub(r'\([^()]*\)', '', text)
    # from DJ's code
    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    remove_accented_chars(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))
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

###################################################
np.random.seed(0)

# Retrieve File (Un-pickle)
filename = 'raw1'
infile = open(filename,'rb')
df = pickle.load(infile)
infile.close()

st.title('TEDTalks Project Dataframe')

# create new column with character-length of each speech
df['speech_length'] = df['text'].apply(len)
# create new column to flag whether speech is over or under 1M views
df['reached_threshold'] = np.where(df['views']>=1250000, 1, 0)
# create new column to flag whether tags column contains (any of my) preferred list of tags
# social change, society, global issues, humanity, community, future
df['prefers'] = np.where(df['tags'].str.contains('global')|df['tags'].str.contains('soci')|df['tags'].str.contains('nity')
                         |df['tags'].str.contains('activism')|df['tags'].str.contains('future')|df['tags'].str.contains('health'), 1, 0)
# concatenate text from several columns to include it in speech text
cols = ['headline', 'tags', 'description', 'text']
df['text'] = df[cols].apply(lambda x: ' '.join(x), axis = 1)

# drop any rows in which 'tags' column contains these words
to_drop = ['performance', 'music', 'magic']
nomusic_df = df[~df['tags'].str.contains('|'.join(to_drop))]
# Missed one: Delete row with index/Talk_ID of 1464
nomusic_df = nomusic_df.drop(1464)
# Check tags- Before
# print(nomusic_df.prefers.value_counts())

pre_expl_df = nomusic_df.reset_index()
# (1) We start with creating a new dataframe from the series with Talk_Id as the index
exploding_df = pd.DataFrame(pre_expl_df.tags.str.split(',').tolist(), index=pre_expl_df.Talk_ID).stack()
# (2) We now want to get rid of the secondary index. To do this, we will make EmployeeId
# as a column (it can't be an index since the values will be duplicate)
exploded_df = exploding_df.reset_index([0, 'Talk_ID'])
# (3) The final step is to set the column names as we want them
exploded_df.columns = ['Talk_ID', 'tag']

# merge original dataframe with this new one
df = pd.merge(df, exploded_df, on='Talk_ID')

# Create quick lambda functions to find the polarity and subjectivity of each routine
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity
df['polarity'] = df['text'].apply(pol)
df['subjectivity'] = df['text'].apply(sub)



# Check tags- After
### ALL TAGS ###
#st.write(df['tag'].value_counts(ascending=True))
#print(df['tag'].value_counts())
# stbar = pd.DataFrame()
# for tag in df.tag.unique():
#     stbar['tag'] = tag
#     stbar['count'] = df.tag.count()
# st.bar_chart(stbar)

# option = st.selectbox(
#     'Which tag?',
#      st_df['tag'].unique())
# st.write('You selected: ', option)

# compress df to remove duplicate rows (created from exploding 'tags'), leaving only unique Talk_ID
de_exploded_df = df.drop_duplicates('Talk_ID')
df = de_exploded_df
st_df = df[['headline', 'description', 'event', 'duration', 'published',
                         'speaker_1','speaker1_occupation', 'speaker1_introduction', 'speaker1_profile',
                         'polarity','subjectivity', 'speech_length','prefers','tags']]

st.write(pd.DataFrame(st_df))

# Let's take a look at the updated text
df.text = df.text.apply(replace_contractions)
round1 = lambda x: clean_text_round1(x)
df.text = pd.DataFrame(df.text.apply(round1))
round2 = lambda x: clean_text_round2(x)
df.text = pd.DataFrame(df.text.apply(round2))
data_clean = df

# LEMMATIZE
# wordnet_lemmatizer = WordNetLemmatizer()  moved to functions (for lemmadata)
english = set(nltk.corpus.words.words())
data_clean['text'] = data_clean['text'].apply(lambda x: lemmadata(x))

# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm = data_dtm.transpose()
# Find the top 30 words said in each speech
top_dict = {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

# Let's first pull out the top 30 words for each speech
words = []
for speech_num in data_dtm.columns:
    top = [word for (word, count) in top_dict[speech_num]]
    for t in top:
        words.append(t)
# Let's aggregate this list and identify the most common words along with how many routines they occur in
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
# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.text)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Leave Pre-processing, Enter Sentiment Analysis
# To re-align shape (filtering out stopwords)
for sw in add_stop_words:
    if sw in data_dtm.columns:
        data_dtm.drop([sw], axis=1, inplace=True)

# Let's create a list to hold all of the pieces of text
list_pieces = []
for t in data_clean.text:
    split = split_text(t)
    list_pieces.append(split)
# Each transcript has been split into 10 pieces of text- calculate the polarity for each
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)


# Save & Store Files (Pickle)
filename = "final_clean.pkl"
outfile = open(filename, 'wb')
pickle.dump(data_clean, outfile)
outfile.close()
filename = "polarity_transcript.pkl"
outfile = open(filename, 'wb')
pickle.dump(polarity_transcript, outfile)
outfile.close()