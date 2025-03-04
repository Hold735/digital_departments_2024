"""
Домашнее задание № 2.05
"Графики"
по курсу Цифровых кафедр "Python: от основ до машинного обучения"

Выполнил: студент группы № 5040103/30401
Курчуков Максим
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def task1() -> None:
    """
    Задание 1
    """
    x = [2, 5, 10, 15, 20]
    y1 = [1, 7, 4, 5, 11]
    y2 = [4, 3, 1, 8, 12]

    fig, ax = plt.subplots()

    ax.plot(x, y1, "or-", label="line 1")
    ax.plot(x, y2, "og-.", label="line 1")
    ax.legend()
    fig.savefig("lesson_2.08/res/task_1.png")
    plt.close()


def task2() -> None:
    """
    Задание 2
    """
    x = np.arange(1, 6, 1)
    y1 = [1, 7, 6, 3, 4]
    y2 = [9, 4, 2, 4, 9]
    y3 = [-7, -4, 2, -4, -7]

    grid = plt.GridSpec(2, 4)
    fig = plt.figure(figsize=(10, 5))

    top = fig.add_subplot(grid[0, :])
    bottom_left = fig.add_subplot(grid[1, :2])
    bottom_right = fig.add_subplot(grid[1, 2:])

    top.plot(x, y1)
    bottom_left.plot(x, y2)
    bottom_right.plot(x, y3)
    fig.savefig("lesson_2.08/res/task_2.png")
    plt.close()


def task3() -> None:
    """
    Задание 3
    """
    x = np.linspace(-5, 5, 15)
    y = x ** 2

    # Положение минимума
    xmin = 0
    ymin = xmin ** 2

    fig, ax = plt.subplots()

    ax.plot(x, y)

    ax.annotate(
        "min", 
        xy=(xmin, ymin),                # точка аннотации
        xytext=(xmin, ymin + 10),       # точка подписи
        arrowprops=dict(                # добавление стрелки
            facecolor="green",          # цвет заливки стрелки
        )
    )
    fig.savefig("lesson_2.08/res/task_3.png")
    plt.close()


def task4() -> None:
    """
    Задание 4
    """
    data = np.random.randint(11, size=(7, 7))
    fig = plt.figure(figsize=(8, 5))

    plt.pcolor(
        data, 
        cmap="viridis",
    )
    plt.colorbar(
        location="right",       # расположение колорбара
        shrink=0.5,             # доля высоты холста
        aspect=5,               # ширина колорбара
        anchor=(0, 0)           # отправная точка построения колорбара
    )
    plt.savefig("lesson_2.08/res/task_4.png")
    plt.close()


def task5() -> None:
    """
    Задание 5
    """
    x = np.linspace(0, 5, 1000)
    y = np.cos(np.pi * x)

    fig, ax = plt.subplots()

    ax.plot(x, y, "r")
    ax.fill_between(x, y, color="blue", alpha=0.7)

    fig.savefig("lesson_2.08/res/task_5.png")
    plt.close()


def task6() -> None:
    """
    Задание 6
    """
    x = np.linspace(0, 5, 1000)
    y = np.cos(np.pi * x)

    mask = np.where(y < -0.5)       # маска для отсекания графика
    x[mask] = np.nan
    y[mask] = np.nan

    fig, ax = plt.subplots()

    ax.plot(x, y, linewidth=2)
    ax.set_ylim(-1, 1)

    fig.savefig("lesson_2.08/res/task_6.png")
    plt.close()


def task7() -> None:
    """
    Задание 7
    """
    x = np.arange(7)
    y = x

    fig, ax = plt.subplots(1, 3)
    fig.set_figwidth(15)
    fig.set_figheight(4)

    drawstyles = ['steps-pre', 'steps-post', 'steps-mid']
    for i in range(3):
        ax[i].plot(x, y, "go-", drawstyle=drawstyles[i])
        ax[i].grid()
    
    fig.savefig("lesson_2.08/res/task_7.png")
    plt.close()


def task8() -> None:
    """
    Задание 8
    """
    x = np.linspace(0, 10, 10)
    y = lambda a, b, c: a * x ** 2 + b * x + c      # функция построения параболы по параметрам

    # Сформируем параболы для отображения
    y1 = y(-0.2, 2, 0)
    y2 = y(-0.6, 6, 0)
    y3 = y(-0.6, 8, 0)

    fig, ax = plt.subplots()

    ax.plot(x, y3, color="green")
    ax.plot(x, y2, color="orange")
    ax.plot(x, y1, color="blue")

    ax.fill_between(x, y3, color="green")
    ax.fill_between(x, y2, color="orange")
    ax.fill_between(x, y1, color="blue")

    # Сформируем пустые графики для корректного отображения легенды
    ax.plot([],[], color="blue", linewidth=5, label="y1")
    ax.plot([],[], color="orange", linewidth=5, label="y2")
    ax.plot([],[], color="green", linewidth=5, label="y3")

    ax.set_ylim(0)

    ax.legend(loc="upper left")     # положение легенды на графике
    fig.savefig("lesson_2.08/res/task_8.png")
    plt.close()


def task9() -> None:
    """
    Задание 9
    """
    data = pd.DataFrame({
        "Ford": [60, 0],
        "Toyota": [40, 0],
        "BMV": [120, 0.2],
        "AUDI": [50, 0],
        "Jaguar": [85, 0],
    }, index=["count", "explode"]).T

    fig, ax = plt.subplots()
    
    ax = plt.pie(
        data["count"], 
        labels=data.index, 
        explode=data["explode"],        # величина отклонения от центра
    )
    
    fig.savefig("lesson_2.08/res/task_9.png")
    plt.close()


def task10() -> None:
    """
    Задание 10
    """
    data = {
        "Ford": 60,
        "Toyota": 40,
        "BMV": 120,
        "AUDI": 50,
        "Jaguar": 85,
    }

    fig, ax = plt.subplots()

    ax.pie(
        data.values(),
        labels=data.keys(),
        wedgeprops=dict(width=0.5)      # настройки отображения (ширина в половину диаграммы)
    )
    fig.savefig("lesson_2.08/res/task_10.png")
    plt.close()


def main() -> None:
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()
    task8()
    task9()
    task10()


if __name__ == "__main__":
    main()
