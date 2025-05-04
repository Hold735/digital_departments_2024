"""
Конспект занятия № 3.05 часть 2
"Деревья решений и случайные леса"
"""

# Деревья решений и случайные леса
"""
СЛ - непараметрический алгоритм.
Это пример ансамблевого метода, основанного на агрегации результатов множества простых моделей.
В реализациях дерева принятия решений в машинной обучении, вопросы обычно ведут к разделению
данных по осям, т.е. каждый узел разбивает данные на две группы по одному из признаков
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

iris = sns.load_dataset("iris")

species_int = []
for r in iris.values:
    match r[4]:
        case "setosa":
            species_int.append(1)
        case "versicolor":
            species_int.append(2)
        case "virginica":
            species_int.append(3)

species_int_df = pd.DataFrame(species_int)

data = iris[["sepal_length", "petal_length"]]
data["species"] = species_int_df
data_df = data[(data["species"] == 1) | (data["species"] == 2)]

X = data_df[["sepal_length", "petal_length"]]
y = data_df[["species"]]

data_df_setosa = data_df[data_df["species"] == 1]
data_df_versicolor = data_df[data_df["species"] == 2]

plt.scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

model = DecisionTreeClassifier(
    max_depth=3     # максимальная глубина дерева
)
model.fit(X, y)

# Строим решение
# набор значений
x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

# построение сетки
X1_p, X2_p = np.meshgrid(x1_p, x2_p)

# состыковка данных
X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
)
# применение модели на данных
y_p = model.predict(X_p)

# отображаем результаты заливкой по контуру
plt.contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap="rainbow", zorder=1)

plt.close()

#-------------------------------------------------------------------------------------------------------
# Как влияет глубина дерева:
data_df = data[(data["species"] == 2) | (data["species"] == 3)]

X = data_df[["sepal_length", "petal_length"]]
y = data_df[["species"]]

data_df_virginica = data_df[data_df["species"] == 3]
data_df_versicolor = data_df[data_df["species"] == 2]



max_depth_value = [[1, 2, 3, 4], [5, 6, 7, 8]]

fig, ax = plt.subplots(2, 4, sharex="col", sharey="row")

for i in range(2):
    for j in range(4):
        ax[i, j].scatter(data_df_virginica["sepal_length"], data_df_virginica["petal_length"])
        ax[i, j].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])
        
        model = DecisionTreeClassifier(
            max_depth=max_depth_value[i][j]     # максимальная глубина дерева
        )
        model.fit(X, y)


        x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
        x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

        # построение сетки
        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        # состыковка данных
        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=["sepal_length", "petal_length"]
        )
        
        y_p = model.predict(X_p)

        ax[i, j].contourf(X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.2, levels=2, cmap="rainbow", zorder=1)
        ax[i, j].set_title(f"Max Depth = {max_depth_value[i][j]}")

plt.show()
