"""
Конспект занятия № 2.02
"Массивы NumPy"
"""
import numpy as np
import matplotlib.pyplot as plt


# Суммирование значнений

rng = np.random.default_rng(1)
s = rng.random(50)

print(s)
print(sum(s))
print(np.sum(s))

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))

print(a.min())
print(a.min(0))
print(a.min(1))

print(np.nanmin(a))
print(np.nanmin(a, 0))
print(np.nanmin(a, 1))

# Транслирование (broadcasting)
# набор правил, которые позволяют осуществлять бинарные операции с массивами разных форм

a = np.array([0, 1, 2])
b = np.array([5, 5, 5])

print(a + b)
print(a + 5)    # Транслируется в [5, 5, 5], подстраивается под размер массива

a = np.array([[0, 1, 2], [3, 4, 5]])
print(a + 5)

a = np.array([0, 1, 2])
b = np.array([[0], [1], [2]])

"""
Правила:
1. Если размерности массивов отличаются, то форма массива с меньшей размерностью дополняется 1 с левой стороны.
2. Если формы массивов не совпадают в каком-то измерении, то если у массива форма равна 1, то он "растягивается" до соответствия формы второго массива.
3. Если в каком-либо измерении размеры отличаются и ни один из них не равен 1, то генерируется ошибка
"""

a = np.array([0, 1, 2])
b = np.array([5])

print(a.ndim, a.shape)
print(b.ndim, b.shape)

# a         (2, 3)
# b (1,) -> (1, 1) -> (2, 3)

a = np.ones((2, 3))
b = np.arange(3)

print(a)
print(b)

print(a.ndim, a.shape)
print(b.ndim, b.shape)

# (2, 3)        (2, 3)      (2, 3)
# (3,)      ->  (1, 3)  ->  (2, 3)

c = a + b
print(c, c.shape)

a = np.arange(3).reshape((3, 1))
b = np.arange(3)

print(a)
print(b)

print(a.ndim, a.shape)
print(b.ndim, b.shape)

# (3,1)         (3, 1)  ->  (3, 3)
# (3,)      ->  (1, 3)  ->  (3, 3)

c = a + b
print(c, c.shape)

"""
[000]   [012]
[111] + [012]
[222]   [012]
"""

a = np.ones((3, 2))
b = np.arange(3)

# 2 (3, 2)          (3, 2)      (3, 2)
# 1 (3,)        ->  (1, 3)  ->  (3, 3)

try:
    c = a + b
except ValueError as e:
    print('!ERROR! ValueError:', e)

## Q1. Что надо изменить в последнем примере, чтобы он заработал без ошибок?

X = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1]
])
Xmean = X.mean(0)
print(Xmean)

Xcenter = X - Xmean
print(Xcenter)

Xmean1 = X.mean(1)
print(Xmean1)

Xmean1 = Xmean1[:, np.newaxis]

Xcenter1 = X - Xmean1
print(Xcenter1)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 3 + np.cos(20 + y * x) * np.sin(y)
print(z.shape)

# plt.imshow(z)
# plt.colorbar()
# plt.show()


x = np.array([1, 2, 3, 4, 5])
y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(x < 3)
print(np.less(x, 3))

print(np.sum(x < 4))
print(np.sum(y < 4, axis=0))
print(np.sum(y < 4, axis=1))
print(np.sum(y < 4))

# & | ^ ~

## Q2. Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9

# Маски: булевы массивы
x = np.array([1, 2, 3, 4, 5])
y = x < 3
print(x[x < 3])

# Векторизация индекса

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
index = [1, 5, 7]
print(x[index])

index = [[1, 5, 7], [2, 4, 8]]
print(x[index])

"""
Форма результата отражает форму массива индексов, а не форму исходного массива
"""

x = np.arange(12).reshape((3, 4))

print(x)
print(x[2])
print(x[2, [2, 0, 1]])
print(x[1:, [2, 0, 1]])

x = np.arange(10)
i = np.array([2, 1, 8, 4])
print(x)
x[i] = 999
print(x)


# Сортировка массивов

x = [4, 5, 4, 2, 2, 6, 7, 8, 6, 5, 4, 32, 3]
print(sorted(x))
print(np.sort(x))

y = np.array([5, 5, 3, 45, 5, 46, 3, 255, 3, 3, 3, 5, 6, 7, 3, 8, 8])
y.sort()
print(y)

# Структурирование массива
data = np.zeros(3, dtype={
    'names': (
        'name', 'age'
    ),
    'formats': (
        'U10', 'i4'
    )
})

print(data.dtype)
name = ['a', 'b', 'c']
age = [1, 2, 3]

data['name'] = name
data['age'] = age

print(data)

print(data['age'] > 2)
print(data[data['age'] > 2]['name'])

# Массивы записей
data_rec = data.view(np.recarray)
print(data_rec)
print(data_rec[0])
print(data_rec[-1].name)
