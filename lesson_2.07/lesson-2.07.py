"""
Конспект занятия № 2.07
"Форматирование графиков"
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime


fig, ax = plt.subplots(2, 3, sharex="col", sharey="row")      # "склеивание" оси x по колонкам и оси y по строкам

for i in range(2):
    for j in range(3):
        ax[i,j].text(0.5,0.5, str((i, j)), fontsize=16, ha="center")
plt.close()

# Сетка

grid = plt.GridSpec(2, 3)

plt.subplot(grid[:2,0])
plt.subplot(grid[0,1:])
plt.subplot(grid[1,1])
plt.subplot(grid[1,2])

plt.close()

mean = [0,0]
cov = [[1,1],[1,2]]
rng = np.random.default_rng(1)
x,y = rng.multivariate_normal(mean, cov, 3000).T

fig = plt.figure()
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], sharey=main_ax, xticklabels=[])
x_hist = fig.add_subplot(grid[-1, 1:], sharex=main_ax, yticklabels=[])

main_ax.plot(x, y, "ok", markersize=3, alpha=0.2)
y_hist.hist(y, 40, orientation="horizontal", color="grey", histtype="stepfilled")
x_hist.hist(x, 40, orientation="vertical", color="grey", histtype="step")

plt.close()

"""
Поясняющие надписи
"""
births = pd.read_csv("lesson_2.06/data/births-1969.csv")
births.index = pd.to_datetime(10000 * births.year + 100*births.month + births.day, format="%Y%m%d")

print(births.head())

births_by_date = births.pivot_table("births", [births.index.month, births.index.day])
print(births_by_date.head())

births_by_date.index = [
    datetime(1969, month, day) for month, day in births_by_date.index
]
print(births_by_date.head())

fig, ax = plt.subplots()

births_by_date.plot(ax=ax)  # Отображение графиков из библиотеки pandas

style = dict(size=10, color="gray")

# Текст на графиках
ax.text("1969-01-01", 5500, "Новый год", **style)
ax.text("1969-09-01", 4500, "День знаний", ha="right")

# Подписи на осях
ax.set(title="Рождаемость в 1969 году", ylabel="Число рождений")

# Настройка меток
# formatter - как именно
# locator - что именно
ax.xaxis.set_major_formatter(plt.NullFormatter())       # в каком виде мажорные
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter("%h"))       # в каком виде минорные
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))       # тики идут каждые 15 дней (работа с датами)

plt.close()

plt.show()
