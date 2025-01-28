"""
Конспект занятия № 2.03
"Введение в Pandas"
"""

import numpy as np
import pandas as pd

"""
Pandas - расширение NumPy, где строки и столбцы индексируются метками а не только числовыми значениями
Бывают типы данных Series, Dataframe, Index
"""

# Series

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(type(data))

print(data.values)
print(type(data.values))

print(data.index)
print(type(data.index))


data = pd.Series([0.25,0.5,0.75,0.1])
print(data[0])
print(data[1:3])

data = pd.Series([0.25,0.5,0.75,0.1], index=["a","b","c","d"])
print(data)
print(data["a"])
print(data["b":"d"])

print(type(data.index))

data = pd.Series([0.25,0.5,0.75,0.1], index=[1,10,7,"d"])
print(data)
print(data[1])
print(data[10:"d"])

population_dict = {
    "city1": 1001,
    "city2": 1002,
    "city3": 1003,
    "city4": 1004,
    "city5": 1005,
}

population = pd.Series(population_dict)
print(population)

print(population["city4"])
print(population["city4":"city5"])

"""
Для создания Series можно использовать
- списки python или массивы numpy
- скалярные значения
- словари
"""

# Q1. Привести различные способы создания объектов типа Series 


# DataFrame - двумерный массив с явно определенными индексами. Последовательность "согласованных" объектов Series

population_dict = {
    "city1": 1001,
    "city2": 1002,
    "city3": 1003,
    "city4": 1004,
    "city5": 1005,
}

area_dict = {
    "city1": 9991,
    "city2": 9992,
    "city3": 9993,
    "city4": 9994,
    "city5": 9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

print(population)
print(area)

states = pd.DataFrame({
    "population1": population,
    "area1": area
})

print(states)

print(states.values)
print(states.index)
print(states.columns)

print(type(states.values))
print(type(states.index))
print(type(states.columns))

print(states["area1"])

"""
DataFrame. Способы создания
- через объекты Series
- списки словарей
- словари объектов Series
- двумерный массив numpy
- структурированный массив numpy
"""

# Q2. Привести различные способы создания объектов типа DataFrame

# Index - способ организации ссылки на данные объектов Series и DataFrame. Index - неизменяем и упорядочек, является мультимножеством (могут быть повторяющиеся значения)

ind = pd.Index([2,3,5,7,11])
print(ind[1])
print(ind[::2])

try:
    ind[1] = 5
except TypeError as e:
    print("!ERROR! TypeError:", e)

# Index - следует соглашениям объекта set (python)

indA = pd.Index([1,2,3,4,5])
indB = pd.Index([2,3,4,5,6])

print(indA.intersection(indB))

# Выборка данных из Series
data = pd.Series([0.25,0.5,0.75,1.0], index=["a", "b", "c", "d"])
print("a" in data)
print("z" in data)

print(data.keys())

print(list(data.items()))

data["a"] = 100
data["z"] = 1000

print(data)

# как одномерный массив

data = pd.Series([0.25,0.5,0.75,1.0], index=["a", "b", "c", "d"])

print(data["a":"c"])
print(data[0:2])
print(data[(data > 0.5) & (data < 1)])
print(data[["a", "d"]])

# атрибуты-индексаторы (против коллизий)
data = pd.Series([0.25,0.5,0.75,1.0], index=[1, 3, 10, 15])
print(data[1])
print(data.loc[1])
print(data.iloc[1])

# Выборка данных из DataFrame
population_dict = {
    "city1": 1001,
    "city2": 1002,
    "city3": 1003,
    "city4": 1004,
    "city5": 1005,
}

area_dict = {
    "city1": 9991,
    "city2": 9992,
    "city3": 9993,
    "city4": 9994,
    "city5": 9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

data = pd.DataFrame({
    "population1": population,
    "area1": area,
    "population": population
})

print(data)
print(data["area1"])
print(data.area1)

print(data.population1 is data["population1"])
print(data.population is data["population"])

data["new"] = data["area1"]
data["new1"] = data["area1"] / data["population1"]
print(data)

# двумерный numpy массив
population_dict = {
    "city1": 1001,
    "city2": 1002,
    "city3": 1003,
    "city4": 1004,
    "city5": 1005,
}

area_dict = {
    "city1": 9991,
    "city2": 9992,
    "city3": 9993,
    "city4": 9994,
    "city5": 9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

data = pd.DataFrame({
    "population1": population,
    "area1": area,
})

print(data)
print(data.values)
print(data.T)

print(data["area1"])

print(data.values[0:3])

# атрибуты-индексаторы (против коллизий обращений)
data = pd.DataFrame({
    "population1": population,
    "area1": area,
    "population": population
})
print(data.iloc[:3, 1:2])
print(data.loc[:"city3", "area1":"population"])

print(data.loc[data["population"] > 1002, ["area1", "population"]])

data.iloc[0, 2] = 999999

print(data)

rng = np.random.default_rng()
s = pd.Series(rng.integers(0,10,4))

print(s)
print(np.exp(s))

population_dict = {
    "city1": 1001,
    "city2": 1002,
    "city3": 1003,
    "city41": 1004,
    "city51": 1005,
}

area_dict = {
    "city1": 9991,
    "city2": 9992,
    "city3": 9993,
    "city42": 9994,
    "city52": 9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

data = pd.DataFrame({
    "population1": population,
    "area1": area,
})
print(data)

# NaN = not a number

# Q3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1

dfA = pd.DataFrame(rng.integers(0, 10, (2,2)), columns=["a", "b"])
dfB = pd.DataFrame(rng.integers(0, 10, (3,3)), columns=["a", "b", "c"])

print(dfA)
print(dfB)
print(dfA + dfB)

rng = np.random.default_rng(1)

A = rng.integers(0, 10, (3,4))
print(A)
print(A[0])
print(A - A[0])

df = pd.DataFrame(A, columns=["a", "b", "c", "d"])
print(df)
print(df.iloc[0])
print(df - df.iloc[0])

print(df.iloc[0, ::2])
print(df - df.iloc[0, ::2])

# Q4. Переписать примр с транслированием для DataFrame так чтобы вычитание происходило не по строкам а по столбцам

# NA-значения: NaN, null, -99999

"""
Pandas. Два способа хранения отсутствующих значений:
Индикаторы Nan, None
null

None - объект (накладные расходы). Не работает с sum min
"""

vall = np.array([1,None,2,3])
try:
    print(vall.sum())
except TypeError as e:
    print("!ERROR! TypeError:", e)

vall = np.array([1,np.nan,2,3])
print(vall.sum())
print(np.sum(vall))
print(np.nansum(vall))

x = pd.Series(range(10), dtype=int)
print(x)
x[0] = None
x[1] = np.nan
print(x)

x1 = pd.Series(['a', 'b', 'c'])
print(x1)
x1[0] = None
x1[1] = np.nan
print(x1)

#
x2 = pd.Series([1,2,3, np.nan, None, pd.NA])
print(x2)

x3 = pd.Series([1,2,3, np.nan, None, pd.NA], dtype="Int32")
print(x3)

print(x3.isnull())
print(x3[x3.isnull()])
print(x3[x3.notnull()])

print(x3.dropna())

df = pd.DataFrame(
    [
        [1,2,3,np.nan,None,pd.NA],
        [1,2,3,4,5,6],
        [1,np.nan,3,4,np.nan,6]
    ]
)
print(df)
print(df.dropna())
print(df.dropna(axis=0))
print(df.dropna(axis=1))

"""
how
-all - все значения NA
-any - хотя бы одно значение
- thresh = x, остается если присутствует минимум x непустых значений
"""
print(df.dropna(how="all"))
print(df.dropna(how="any"))
print(df.dropna(thresh=2))

# Q5. На примере обектов DataFrame продумонстрируйте использование методов ffill() и bfill()
