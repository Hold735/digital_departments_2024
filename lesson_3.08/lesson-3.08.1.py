"""
Конспект занятия № 3.08 часть 1
"Основы кибербезопасности - спам"
"""

import pandas as pd

data = pd.read_csv("./lesson_3.08/data/spam.csv")
print(data)

print(data.columns)

data.info()

data["Spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)

print(data.head())

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
X = vect.fit_transform(data["Message"])

w = vect.get_feature_names_out()

print(X[:, 1000])

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = Pipeline([("vect", CountVectorizer()), ("NB", MultinomialNB())])
model2 = Pipeline([("vect", CountVectorizer()), ("NB", GaussianNB())])
X_train, X_test, y_train, y_test = train_test_split(
    data["Message"], 
    data["Spam"], 
    test_size=0.3
)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))

msg = [
    "Hello, World!",
    "Free subscription",
    "Call me now!",
    "Win the lottery"
]

print(model.predict(msg))
