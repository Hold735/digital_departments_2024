"""
Конспект занятия № 3.03
"Обучение регрессионных моделей"
"""
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt

data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28],
    ]
)

# Градиентный спуск - пакетный градиентный спуск. Для работы используются все доступные обучающие данные
"""
Стохастический градиентный спуск - на каждой итерации обучаемся только по одной выборке из данных
- сокращение числа вычислений
- вносим смещение - боремся с переобучением (засчет случайности)
Мини-пакетный градиентный спуск - на каждой итерации используется несколько выборок
"""
x = data[:, 0]
y = data[:, 1]

n = len(x)

w1 = 0.0
w0 = 0.0

L = 0.001

iterations = 100_000

for i in range(iterations):
    D_w0 = 2 * sum(-y[i] + w0 + w1 * x[i] for i in range(n))       # сумма по всем элементам!
    D_w1 = 2 * sum(x[i] * (-y[i] + w0 + w1 * x[i]) for i in range(n))
    w1 -= L * D_w1
    w0 -= L * D_w0

print(w1, w0)

w1 = 0.0
w0 = 0.0

# размер выборки
sample_size = 1

for i in range(iterations):
    idx = np.random.choice(n, sample_size, replace=False)
    D_w0 = 2 * sum(-y[idx] + w0 + w1 * x[idx])       # сумма по части элементов!
    D_w1 = 2 * sum(x[idx] * (-y[idx] + w0 + w1 * x[idx]))
    w1 -= L * D_w1
    w0 -= L * D_w0

print(w1, w0)

# Как оценить насколько сильно "промахиваются" прогнозы при использовании линейной регрессии
data_df = pd.DataFrame(data)
print(data_df.corr(method="pearson"))

data_df[1] = data_df[1].values[::-1]
print(data_df.corr(method="pearson"))

# Коэффициент корреляции помогает понять есть ли связь между двумя переменными

# Обучающие и тестовые выборки
"""
Основной метод борьбы с переобучением заключается в том, что набор данных делится на обучающую и тестовую выборки
Во всех видах машинного обучения с учителем это встречается

Обычная пропорция 2/3 - на обучение, 1/3 - на тест (4/5 к 1/5, 9/10 к 1/10 и т.д.)
"""

X = data_df.values[:, : -1]
Y = data_df.values[:, -1]

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
print(X_train)
print(Y_train)

print(X_test)
print(Y_test)

kfold = KFold(n_splits=3, random_state=1, shuffle=True)     # 3х кратная перекрестная валидация

model = LinearRegression()
# model.fit(X_train, Y_train)

# r = model.score(X_test, Y_test)     # коэффициент детерминации
# print(r)

results = cross_val_score(model, X, Y, cv=kfold)
print(results)      # средние квадратические ошибки
print(results.mean(), results.std())

"""
Метрики показывают насколько единообразно ведет себя модель на разных выборках
Возможно использование поэлементной перекрестной валидации - мало данных если
Если данные сильно разбросаны, то можно сделать случайную валидацию
"""

# Валидационная выборка - для сравнения различных моделей или конфигураций

# Многомерная линейная регрессия

data_df = pd.read_csv("./lesson_3.03/data/multiple_independent_variable_linear.csv")

print(data_df.head())

X = data_df.values[:,:-1]
Y = data_df.values[:, -1]

model = LinearRegression().fit(X, Y)

print(model.coef_, model.intercept_)

x1 = X[:, 0]
x2 = X[:, 1]
y = Y

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(x1, x2, y)

x1_ = np.linspace(min(x1), max(x1), 100)
x2_ = np.linspace(min(x2), max(x2), 100)

X1_, X2_ = np.meshgrid(x1_, x2_)

Y_ = model.intercept_ + model.coef_[0] * X1_ + model.coef_[1] * X2_

ax.plot_surface(X1_, X2_, Y_, cmap="Greys", alpha=0.2)

plt.show()
