"""
Конспект занятия № 3.08 часть 2
"Основы кибербезопасности - фишинг"
"""

import pandas as pd

data = pd.read_csv("./lesson_3.08/data/phishing.csv")
print(data.head())
data.info()

X = data.drop(columns="class")
print(X.head())

y = pd.DataFrame(data["class"])

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3
)

dt = DecisionTreeClassifier()
model = dt.fit(X_train, y_train)
dt_predict = model.predict(X_test)
print(accuracy_score(y_test, dt_predict))
