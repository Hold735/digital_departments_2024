"""
Конспект занятия № 2.06
"Визуализация данных в Matplotlib"
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Гистограмма
rng = np.random.default_rng(1)
data = rng.normal(size=1000)

plt.hist(
    data,
    bins=30,
    density=True,
    alpha=0.5,
    histtype="stepfilled",
    edgecolor="red"
)
plt.close()

x1 = rng.normal(0,0.8,1000)
x2 = rng.normal(-2,1,1000)
x3 = rng.normal(3,2,1000)

args = dict(
    alpha=0.3,
    bins=40
)

plt.hist(x1, **args)
plt.hist(x2, **args)
plt.hist(x3, **args)

plt.close()

print(np.histogram(x1, bins=1))
print(np.histogram(x1, bins=2))
print(np.histogram(x1, bins=40))

# Двумерные гистограммы

mean = [0,0]
cov = [[1,1,], [1,2]]

x,y = rng.multivariate_normal(mean, cov, 10000).T

plt.hist2d(x,y, bins=30)
cb = plt.colorbar()
cb.set_label("point in interval")
plt.close()

print(np.histogram2d(x,y,bins=1))
print(np.histogram2d(x,y,bins=10))

plt.hexbin(x,y,gridsize=30)     # гистограмма из шестиугольников
cb = plt.colorbar()
cb.set_label("point in interval")
plt.close()

# Легенда

x = np.linspace(0,10,1000)

fig, ax = plt.subplots()

ax.plot(x, np.sin(x), label="Синус")
ax.plot(x, np.cos(x), label="Косинус")
ax.plot(x, np.cos(x) + 2)

ax.legend(
    frameon=True,
    fancybox=True,
    shadow=True
)
plt.close()

y = np.sin(x[:,np.newaxis] + np.pi * np.arange(0,2,0.5))
lines = plt.plot(x,y)
plt.legend(
    lines, 
    ["1", "второй", "third", "4-ый"],
    loc="lower center"
)

plt.legend(lines[:2], ["1", "2"])

plt.plot(x,y)
plt.close()

cities = pd.read_csv("lesson_2.06/data/california_cities.csv")

lat, lon, pop, area = cities["latd"], cities["longd"], cities["population_total"], cities["area_total_km2"]

plt.scatter(lon, lat, c=np.log10(pop), s=area)
plt.xlabel("Широта")
plt.ylabel("Долгота")
plt.colorbar()
plt.clim(3, 7)      # ограничение колорбара

# построим легенду по площади кружков

plt.scatter([], [], c="k", alpha=0.5, s=100, label="100 $km^2$")    # создадим моки (пустые данные) с лейблами для легенды
plt.scatter([], [], c="k", alpha=0.5, s=300, label="300 $km^2$")
plt.scatter([], [], c="k", alpha=0.5, s=500, label="500 $km^2$")
plt.legend(labelspacing=3, frameon=False)

plt.close()

# несколько легенд на один график

fig, ax = plt.subplots()
lines = []
styles = ["-", "--", "-.", ":"]
x = np.linspace(0, 10, 1000)
for i in range(4):
    lines += ax.plot(
        x,
        np.sin(x - i + np.pi / 2),
        styles[i]
    )
ax.axis("equal")

ax.legend(lines[:2], ["line_1", "line_2"], loc="upper right")

# создадим новый слой для дополнительной легенды
leg = mpl.legend.Legend(
    ax,         # расположение на осях
    lines[1:],      # данные
    ["line 2", "line 3", "line 4"],         # наименования
    loc="lower left"            # положение на осях
)

ax.add_artist(leg)      # добавляем объект на оси
plt.close()

# Шкалы

x = np.linspace(0, 10, 1000)
y = np.sin(x) * np.cos(x[:, np.newaxis])

"""
Карты цветов:
- последовательные (градиентные)
- дивергентные (два цвета)
- качественные (смешиваются без четкого порядка)
"""
# 1
plt.imshow(y, cmap="Blues")
plt.colorbar()
# 2
plt.imshow(y, cmap="RdBu")
plt.colorbar()
# 3
plt.imshow(y, cmap="jet")
plt.colorbar()

plt.close()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(y, cmap="viridis")       # непрерывный cmap
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(y, cmap=plt.cm.get_cmap("viridis", 6))       # дискретный cmap (на 6 частей)
plt.colorbar()
plt.clim(-0.25, 0.25)

plt.close()

# Сабграфики

ax1 = plt.axes()

ax2 = plt.axes([0.4, 0.3, 0.2, 0.1])        # [нижний угол, левый угол, ширина, высота] в долях от "холста"

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.close()


fig = plt.figure()

ax1 = fig.add_axes([0.1,0.5,0.8,0.4])
ax2 = fig.add_axes([0.1,0.1,0.8,0.4])

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.close()

# Простые сетки

fig = plt.figure()

fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1,7):
    ax = fig.add_subplot(2,3,i)
    ax.plot(np.sin(x + np.pi / 4 * i))

plt.show()
