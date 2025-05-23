"""
Домашнее задание № 3.07
"Нейронные сети"
по курсу Цифровых кафедр "Python: от основ до машинного обучения"

Выполнил: студент группы № 5040103/30401
Курчуков Максим
"""

import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

def prepare_data() -> None:
    global df, X, y, X_train, X_test, y_train, y_test

    iris = load_iris(as_frame=True)
    df = iris.frame[iris.frame["target"] != 2]
    df["species"] = df["target"].map({0: "setosa", 1: "versicolor"})

    X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


def task_1() -> None:
    """
    Задание 1.
    Обучение с учителем (классификация).
    Выбрать ДВА ЛЮБЫХ СОРТА и для них реализовать метод опорных векторов.
    """
    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    df["pred"] = clf.predict(X)

    fig = px.scatter_3d(df, 
                        x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)", 
                        color="pred", symbol="species",
                        title="SVM классификация: Setosa vs Versicolor")
    fig.show()


def task_2() -> None:
    """
    Задание 2.
    Обучение с учителем (классификация).
    Выбрать ДВА ЛЮБЫХ СОРТА и для них реализовать метод главных компонент.
    """
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    df["pred"] = clf.predict(X)

    fig = px.scatter_3d(df,
                        x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
                        color="pred", symbol="species",
                        title="Logistic Regression: Setosa vs Versicolor")
    fig.show()


def task_3() -> None:
    """
    Задание 3.
    Обучение без учителя (классификация).
    Выбрать ДВА ЛЮБЫХ СОРТА и для них реализовать метод k средних.
    """
    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    fig = px.scatter_3d(df,
                        x="sepal length (cm)", y="sepal width (cm)", z="petal length (cm)",
                        color="cluster", symbol="species",
                        title="KMeans кластеризация: Setosa vs Versicolor")
    fig.show()


def main() -> None:
    prepare_data()
    task_1()
    task_2()
    task_3()


if __name__ == "__main__":
    main()
