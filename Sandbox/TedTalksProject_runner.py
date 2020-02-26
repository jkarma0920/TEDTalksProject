import pickle

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

########################################################

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


########################################################

data_clean = pd.read_pickle('final_clean.pkl')

# Similarity Model
data_clean.reset_index(inplace=True, drop=True)
length = len(data_clean)
simple_data = data_clean[['public_url', 'headline', 'description', 'event', 'duration', 'published',
                         'speaker_1','speaker1_occupation', 'speaker1_introduction', 'speaker1_profile',
                         'speech_length', 'reached_threshold', 'prefers', 'polarity','subjectivity',
                          'text']]
# moved from classifer()
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
print(type(relevant[-1]), relevant[-1]) # testing out
relevant_df = pd.DataFrame(relevant).transpose()

pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(relevant_df.columns, open("relevantwords.pkl", "wb"))

mnb = MultinomialNB()
mnb.fit(relevant_df, y_train)
pickle.dump(simple_data, open("ff_simple.pkl", "wb"))
pickle.dump(mnb, open("mnbb.pkl", "wb"))
print('done')