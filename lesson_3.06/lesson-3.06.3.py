"""
Конспект занятия № 3.06 часть 3
"Метод главных компонент"
"""

# PCA - principal component analysis - алгоритм обучения без учителя
"""
PCA часто используют для понижения размерности.
Задача машинного обучения без учителя состоит в выяснении зависимости между признаками
В Методе Главных Компонент (МГК) выполняется качественная оценка этой зависимости путем поиска главных осей координат
и их дальнейшего использования для описания наборов данных.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

iris = sns.load_dataset("iris")

# sns.pairplot(iris, hue="species")
# plt.show()

data = iris[["petal_width", "petal_length", "species"]]

data_v = data[data["species"] == "versicolor"]

X = data_v["petal_width"]
Y = data_v["petal_length"]

plt.scatter(X, Y)

data_v.drop(columns=["species"], inplace=True)

# Найдем главные компоненты
P = PCA(
    n_components=2,     # две оси
)
P.fit(data_v)

print(P.components_)                # Компоненты
print(P.explained_variance_)        # "Важность"
print(P.mean_)

plt.scatter(P.mean_[0], P.mean_[1])

# Первая компонента
plt.plot(
    [
        P.mean_[0],
        P.mean_[0] + P.components_[0][0] * np.sqrt(P.explained_variance_[0])
    ],
    [
        P.mean_[1],
        P.mean_[1] + P.components_[0][1] * np.sqrt(P.explained_variance_[0]),
    ]
)
# Вторая компонента
plt.plot(
    [
        P.mean_[0],
        P.mean_[0] + P.components_[1][0] * np.sqrt(P.explained_variance_[1])
    ],
    [
        P.mean_[1],
        P.mean_[1] + P.components_[1][1] * np.sqrt(P.explained_variance_[1]),
    ]
)

# Для снижения размерности нужно убрать данные со второстепенной оси - уменьшаем количество осей

P1 = PCA(
    n_components=1,     # две оси
)
P1.fit(data_v)
X_p = P1.transform(data_v)

print(data_v.shape)
print(X_p.shape)

X_p_new = P1.inverse_transform(X_p)
print(X_p_new.shape)

plt.scatter(X_p_new[:, 0], X_p_new[:, 1], alpha=0.6)

plt.show()

"""
Достоинства:
- простота интерпретации
- эффективность в работе с многомерными данными

Недостатки:
- аномальные значения в данных оказывают сильное влияние на результат
"""
