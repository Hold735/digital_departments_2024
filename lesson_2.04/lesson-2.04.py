"""
Конспект занятия № 2.04
"Индексы в Pandas"
"""

import numpy as np
import pandas as pd

# Если размерность данных > 2, то используют иерархическую индексацию
# (мультииндекс). В один индекс включается несколько уровней.

index = [
    ("city_1", 2010),
    ("city_1", 2020),
    ("city_2", 2010),
    ("city_2", 2020),
    ("city_3", 2010),
    ("city_3", 2020),
]

population = [101, 201, 102, 202, 103, 203]

pop = pd.Series(population, index=index)
print(pop)
print(pop[[i for i in pop.index if i[1] == 2020]])

# MultiIndex
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
print(pop)

print(pop[:, 2020])

pop_df = pop.unstack()
print(pop_df)

print(pop_df.stack())

index = [
    ("city_1", 2010, 1),
    ("city_1", 2010, 2),
    ("city_1", 2020, 1),
    ("city_1", 2020, 2),
    ("city_2", 2010, 1),
    ("city_2", 2010, 2),
    ("city_2", 2020, 1),
    ("city_2", 2020, 2),
    ("city_3", 2010, 1),
    ("city_3", 2010, 2),
    ("city_3", 2020, 1),
    ("city_3", 2020, 2),
]

population = [
    101,
    1010,
    201,
    2010,
    102,
    1020,
    202,
    2020,
    103,
    1030,
    203,
    2030,
]

index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
print(pop)

print(pop[:, 2010])
print(pop[:, :, 2])

pop_df = pop.unstack()
print(pop_df)

pop_df = pd.DataFrame(
    {
        "total": pop,
        "something": list(range(10, 22))
    }
)
print(pop_df)

print(pop_df["something"])

pop_df_1 = pop_df.loc["city_1", "something"]
print(pop_df_1)

# Q1. Разобрать как использовать мультииндексные клчи в данном примере

"""
Как можно создавать мультииндексы?
- список массивов, задающих значение индекса на каждом уровне
- список кортежей, задающих значение индекса в каждой точке
- декартово произведение обычных индексов
- описание внутреннего представления: levels, codes
"""

i1 = pd.MultiIndex.from_arrays([
        ["a", "a", "b", "b"],
        [1, 2, 1, 2]
    ]
)
print(i1)

i2 = pd.MultiIndex.from_tuples(
    [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]
)
print(i2)

i3 = pd.MultiIndex.from_product(
    [
        ["a", "b"],
        [1, 2]
    ]
)
print(i3)

i4 = pd.MultiIndex(
    levels=[
        ["a", "b"],
        [1, 2]
    ],
    codes=[
        [0, 0, 1, 1],       # a a b b
        [0, 1, 0, 1]        # 1 2 1 2
    ]
)
print(i4)

# Уровням модно задавать названия
data = {
    ("city_1", 2010): 100,
    ("city_1", 2020): 200,
    ("city_2", 2010): 1001,
    ("city_2", 2020): 1002,
}
s = pd.Series(data)
print(s)

s.index.names = ["city", "year"]
print(s)

index = pd.MultiIndex.from_product(
    [
        ["city_1", "city_2"],
        [2010, 2020]
    ],
    names=["city", "year"]
)
print(index)

columns = pd.MultiIndex.from_product(
    [
        ["person_1", "person_2", "person_3"],
        ["job_1", "job_2"]
    ],
    names=["worker", "job"]
)
print(columns)

rng = np.random.default_rng(1)

data = rng.random((4, 6))
print(data)

data_df = pd.DataFrame(data=data, columns=columns, index=index)
print(data_df)

"""
Q2. Из получившихся данных выбрать данные по
- 2020 году (для всех столбцов)
- job_1 (для всех строк)
- для city_1 и job_2
"""

# Индксация и срезы (по мультииндексам)

data = {
    ("city_1", 2010): 100,
    ("city_1", 2020): 200,
    ("city_2", 2010): 1001,
    ("city_2", 2020): 1002,
    ("city_3", 2010): 3002,
    ("city_3", 2020): 4002,

}
s = pd.Series(data)
s.index.names = ["city", "year"]
print(s)
print(s["city_1", 2010])
print(s["city_1"])

print(s.loc["city_1":"city_2"])
print(s[:, 2010])

print(s[s > 2000])

print(s[["city_1", "city_3"]])

"""
Q3. Взять за основу ДФ со следующей структурой и
выполнить запрос на получение следующих данных:
- все данные по person_1 и person_3
- все данные по первому городу и первым двум person-ам (с использоваем срезов)
Приведите пример (самостоятельно) с использоваем pd.IndexSlice.

index = pd.MultiIndex.from_product(
    [
        ["city_1", "city_2"],
        [2010, 2020]
    ],
    names=["city", "year"]
)

columns = pd.MultiIndex.from_product(
    [
        ["person_1", "person_2", "person_3"],
        ["job_1", "job_2"]
    ],
    names=["worker", "job"]
)
"""

# Перегруппировка мультииндексов

index = pd.MultiIndex.from_product(
    [
        ["a", "c", "b"],
        [1, 2]
    ]
)
data = pd.Series(rng.random(6), index=index)
data.index.names = ["char", "int"]

print(data)
try:
    print(data["a":"b"])
except pd.errors.UnsortedIndexError as e:
    print("!ERROR! pandas.errors.UnsortedIndexError:", e)

data = data.sort_index()
print(data)
print(data["a":"b"])

index = [
    ("city_1", 2010, 1),
    ("city_1", 2010, 2),
    ("city_1", 2020, 1),
    ("city_1", 2020, 2),
    ("city_2", 2010, 1),
    ("city_2", 2010, 2),
    ("city_2", 2020, 1),
    ("city_2", 2020, 2),
    ("city_3", 2010, 1),
    ("city_3", 2010, 2),
    ("city_3", 2020, 1),
    ("city_3", 2020, 2),
]

population = [
    101,
    1010,
    201,
    2010,
    102,
    1020,
    202,
    2020,
    103,
    1030,
    203,
    2030,
]

pop = pd.Series(population, index=index)
print(pop)

i = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(i)
print(pop)

print(pop.unstack())
print(pop.unstack(level=0))
print(pop.unstack(level=1))
print(pop.unstack(level=2))

# NumPy Конкатенация

x = [[1, 2, 3]]
y = [[4, 5, 6]]
z = [[7, 8, 9]]

print(np.concatenate([x, y, z]))
print(np.concatenate([x, y, z], axis=0))
print(np.concatenate([x, y, z], axis=1))

# Pandas - concat
ser1 = pd.Series(["a", "b", "c"], index=[1, 2, 3])
ser2 = pd.Series(["d", "e", "f"], index=[1, 2, 6])

print(pd.concat([ser1, ser2], verify_integrity=False))
print(pd.concat([ser1, ser2], ignore_index=True))
print(pd.concat([ser1, ser2], keys=["x", "y"]))

ser1 = pd.Series(["a", "b", "c"], index=[1, 2, 3])
ser2 = pd.Series(["b", "c", "f"], index=[4, 5, 6])

print(pd.concat([ser1, ser2], join="inner"))
print(pd.concat([ser1, ser2], join="outer"))

# Q4. Привести пример исползования inner и outer джойнов для Series
# на данных предыдущего примера
