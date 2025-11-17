# Spam News Detection
import pandas as pd
import numpy as np

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score







trueNews = pd.read_csv('True.csv')
fakeNews = pd.read_csv("gossipcop_fake.csv")

trueNews['label']=0
fakeNews['label']=1

dataset1 = trueNews[['title', 'label']]
dataset2 = fakeNews[['title', 'label']]

dataset = pd.concat([dataset1, dataset2])

# To check The Null Values
# print(dataset.isnull().sum())

# print(dataset['label'].value_counts())

# To shuffle the false and true news data
dataset = dataset.sample(frac=1)

ps = WordNetLemmatizer()
stopwords= stopwords.words('english')
nltk.download('wordnet')
nltk.download('stopwords')


def clean_row(row):

    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)     # re is used for the regex expression

    token = row.split()


    news = [ps.lemmatize(word) for word in token if not word in stopwords]
    cleanned_news = ' '.join(news)

    return cleanned_news

# print(dataset['title'])

# All cleanned data will be diplayed
dataset['title'] = dataset['title'].apply(lambda X : clean_row(X))
# print(dataset.shape)  #5000




# Converting the data into Vector form means numbers
vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1,2))


X = dataset.iloc[:50000, 0]    #50000 is unique numbers created for all the words
Y = dataset.iloc[:50000, 1]

train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size= 0.2, random_state= 0)

vec_train_data = vectorizer.fit_transform(train_data)
vec_train_data = vec_train_data.toarray()

vec_test_data = vectorizer.fit_transform(test_data)
vec_test_data = vec_test_data.toarray()


training_data = pd.DataFrame(vec_train_data, columns=vectorizer.get_feature_names_out())
testing_data = pd.DataFrame(vec_test_data, columns=vectorizer.get_feature_names_out())


clf = MultinomialNB()

clf.fit(training_data, train_label)
y_pred = clf.predict(testing_data)

# print(accuracy_score(test_label, y_pred))

y_pred_train = clf.predict(training_data)
# print(accuracy_score(train_label, y_pred_train))


txt = input("Enter News: ")
news = clean_row(str(txt))
pred = clf.predict(vectorizer.transform([news]).toarrya())

if pred == 0:
    print("News is correct!")
else:
    print("News is Fake!")