"""
Домашнее задание № 3.05
"SVM"
по курсу Цифровых кафедр "Python: от основ до машинного обучения"

Выполнил: студент группы № 5040103/30401
Курчуков Максим
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

def task_1() -> None:
    """
    ДЗ: убрать из данных iris часть точек (на которых обучаемся) и убедиться, что на предсказание влияют только опорные вектора.
    Пункт 1. Исследуем влияние изменения данных (не опорных) на решение.
    """
    # Загрузка данных
    iris = sns.load_dataset("iris")

    # Подготовка данных: выделяем группы образцов setosa и versicolor и признаки sepal_length, petal_length и species
    data = iris[["sepal_length", "petal_length", "species"]]
    data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

    # Подготовка отображений
    plt.figure(figsize=(20, 10))

    # 1. Полный набор данных
    plt.subplot(1, 2, 1)
    plt.suptitle("П1. Влияние изменения неопорных данных")
    plt.title("Полный набор данных")
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")

    # 1.1. Отображение исходных данных 
    data_df_setosa = data_df[data_df["species"] == "setosa"]
    data_df_versicolor = data_df[data_df["species"] == "versicolor"]
    plt.scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
    plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

    # 1.2. Разделение данных на признаки и таргет
    X = data_df[["sepal_length", "petal_length"]]
    y = data_df[["species"]]

    # 1.3. Создание и обучение модели
    model = SVC(
        kernel="linear",
        C=1000            # используем достаточно жесткую границу, чтобы сократить количество опорных векторов
    )
    model.fit(X, y)
    
    # 1.4. Отображение опорных векторов
    plt.scatter(
        model.support_vectors_[:, 0], 
        model.support_vectors_[:, 1], 
        s=400, 
        facecolor="none", 
        edgecolors="black"
    )

    # 1.5. Построение решения
    x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
    x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)
    X1_p, X2_p = np.meshgrid(x1_p, x2_p)
    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
    )

    y_p = model.predict(X_p)
    
    # кодировка таргетов к численным данным
    y_p_int = np.array([1 if i == "setosa" else 2 for i in y_p])

    plt.contourf(X1_p, X2_p, y_p_int.reshape(X1_p.shape), alpha=0.2, levels=2, cmap="rainbow", zorder=1)

    # 2. Урезанный набор данных (с сохранением опорных векторов)
    plt.subplot(1, 2, 2)
    plt.title("Урезанный набор данных\n(с сохранением опорных векторов)")
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")

    # Разделение данных на опорные и "неопорные"
    mask = data_df.apply(lambda row: (row["sepal_length"], row["petal_length"]) in model.support_vectors_, axis=1)      # маска выделения опорных векторов из данных
    data_sv = data_df[mask].copy()          # применяем маску
    data_nsv = data_df[~mask].copy()        # применяем антимаску

    # Удаляем часть данных из неопорных точек
    data_nsv_cropp = data_nsv.sample(frac=0.5)      # удаление 50% данных

    # Восстанавливаем структуру данных
    data_df_cropp = pd.concat([data_sv, data_nsv_cropp], ignore_index=True).sort_index().reset_index(drop=True)

    # 2.1. Отображение исходных данных 
    data_df_cropp_setosa = data_df_cropp[data_df_cropp["species"] == "setosa"]
    data_df_cropp_versicolor = data_df_cropp[data_df_cropp["species"] == "versicolor"]
    plt.scatter(data_df_cropp_setosa["sepal_length"], data_df_cropp_setosa["petal_length"])
    plt.scatter(data_df_cropp_versicolor["sepal_length"], data_df_cropp_versicolor["petal_length"])

    # 2.2. Разделение данных на признаки и таргет
    X = data_df_cropp[["sepal_length", "petal_length"]]
    y = data_df_cropp[["species"]]

    # 2.3. Создание и обучение модели
    model = SVC(
        kernel="linear",
        C=1000            # используем достаточно жесткую границу, чтобы сократить количество опорных векторов
    )
    model.fit(X, y)
    
    # 2.4. Отображение опорных векторов
    plt.scatter(
        model.support_vectors_[:, 0], 
        model.support_vectors_[:, 1], 
        s=400, 
        facecolor="none", 
        edgecolors="black"
    )

    # 2.5. Построение решения
    x1_p = np.linspace(min(data_df_cropp["sepal_length"]), max(data_df_cropp["sepal_length"]), 100)
    x2_p = np.linspace(min(data_df_cropp["petal_length"]), max(data_df_cropp["petal_length"]), 100)
    X1_p, X2_p = np.meshgrid(x1_p, x2_p)
    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
    )

    y_p = model.predict(X_p)
    
    # кодировка таргетов к численным данным
    y_p_int = np.array([1 if i == "setosa" else 2 for i in y_p])

    plt.contourf(X1_p, X2_p, y_p_int.reshape(X1_p.shape), alpha=0.2, levels=2, cmap="rainbow", zorder=1)
    
    plt.savefig("lesson_3.05/res/task_1.png")
    plt.close


def task_2() -> None:
    """
    ДЗ: убрать из данных iris часть точек (на которых обучаемся) и убедиться, что на предсказание влияют только опорные вектора.
    Пункт 2. Исследуем влияние изменения данных (опорных) на решение
    """
    # Загрузка данных
    iris = sns.load_dataset("iris")

    # Подготовка данных: выделяем группы образцов setosa и versicolor и признаки sepal_length, petal_length и species
    data = iris[["sepal_length", "petal_length", "species"]]
    data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

    # Подготовка отображений
    plt.figure(figsize=(20, 10))

    # 1. Полный набор данных
    plt.subplot(1, 2, 1)
    plt.suptitle("П2. Влияние изменения опорных данных")
    plt.title("Полный набор данных")
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")

    # 1.1. Отображение исходных данных 
    data_df_setosa = data_df[data_df["species"] == "setosa"]
    data_df_versicolor = data_df[data_df["species"] == "versicolor"]
    plt.scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
    plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

    # 1.2. Разделение данных на признаки и таргет
    X = data_df[["sepal_length", "petal_length"]]
    y = data_df[["species"]]

    # 1.3. Создание и обучение модели
    model = SVC(
        kernel="linear",
        C=1000            # используем достаточно жесткую границу, чтобы сократить количество опорных векторов
    )
    model.fit(X, y)
    
    # 1.4. Отображение опорных векторов
    plt.scatter(
        model.support_vectors_[:, 0], 
        model.support_vectors_[:, 1], 
        s=400, 
        facecolor="none", 
        edgecolors="black"
    )

    # 1.5. Построение решения
    x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
    x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)
    X1_p, X2_p = np.meshgrid(x1_p, x2_p)
    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
    )

    y_p = model.predict(X_p)
    
    # кодировка таргетов к численным данным
    y_p_int = np.array([1 if i == "setosa" else 2 for i in y_p])

    plt.contourf(X1_p, X2_p, y_p_int.reshape(X1_p.shape), alpha=0.2, levels=2, cmap="rainbow", zorder=1)

    # 2. Урезанный набор данных (изменение опорных векторов)
    plt.subplot(1, 2, 2)
    plt.title("Урезанный набор данных\n(удаление опорных векторов)")
    plt.xlabel("Sepal length")
    plt.ylabel("Petal length")

    # Разделение данных на опорные и "неопорные"
    mask = data_df.apply(lambda row: (row["sepal_length"], row["petal_length"]) in model.support_vectors_, axis=1)      # маска выделения опорных векторов из данных
    data_sv = data_df[mask].copy()          # применяем маску
    data_nsv = data_df[~mask].copy()        # применяем антимаску

    # Восстанавливаем структуру данных без опорных точек
    data_df_cropp = data_nsv.reset_index(drop=True)

    # 2.1. Отображение исходных данных 
    data_df_cropp_setosa = data_df_cropp[data_df_cropp["species"] == "setosa"]
    data_df_cropp_versicolor = data_df_cropp[data_df_cropp["species"] == "versicolor"]
    plt.scatter(data_df_cropp_setosa["sepal_length"], data_df_cropp_setosa["petal_length"])
    plt.scatter(data_df_cropp_versicolor["sepal_length"], data_df_cropp_versicolor["petal_length"])

    # 2.2. Разделение данных на признаки и таргет
    X = data_df_cropp[["sepal_length", "petal_length"]]
    y = data_df_cropp[["species"]]

    # 2.3. Создание и обучение модели
    model = SVC(
        kernel="linear",
        C=1000            # используем достаточно жесткую границу, чтобы сократить количество опорных векторов
    )
    model.fit(X, y)
    
    # 2.4. Отображение опорных векторов
    plt.scatter(
        model.support_vectors_[:, 0], 
        model.support_vectors_[:, 1], 
        s=400, 
        facecolor="none", 
        edgecolors="black"
    )

    # 2.5. Построение решения
    x1_p = np.linspace(min(data_df_cropp["sepal_length"]), max(data_df_cropp["sepal_length"]), 100)
    x2_p = np.linspace(min(data_df_cropp["petal_length"]), max(data_df_cropp["petal_length"]), 100)
    X1_p, X2_p = np.meshgrid(x1_p, x2_p)
    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
    )

    y_p = model.predict(X_p)
    
    # кодировка таргетов к численным данным
    y_p_int = np.array([1 if i == "setosa" else 2 for i in y_p])

    plt.contourf(X1_p, X2_p, y_p_int.reshape(X1_p.shape), alpha=0.2, levels=2, cmap="rainbow", zorder=1)
    
    plt.savefig("lesson_3.05/res/task_2.png")
    plt.close


def main():
    task_1()
    task_2()


if __name__ == "__main__":
    main()
