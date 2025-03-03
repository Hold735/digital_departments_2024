"""
Домашнее задание № 2.05
"Графики"
по курсу Цифровых кафедр "Python: от основ до машинного обучения"

Выполнил: студент группы № 5040103/30401
Курчуков Максим
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def task1() -> None:
    """
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
    """
    x = np.linspace(-5, 5, 15)
    y = x ** 2

    fig, ax = plt.subplots()

    ax.plot(x, y)

    xmin = 0
    ymin = xmin ** 2
    ax.annotate(
        "min", 
        xy=(xmin, ymin), 
        xytext=(xmin, ymin + 10),
        arrowprops=dict(
            facecolor="green",
        )
    )
    fig.savefig("lesson_2.08/res/task_3.png")
    plt.close()


def task4() -> None:
    """
    """
    data = np.random.randint(0, 11, (8, 8))

    plt.imshow(
        data, 
        cmap="viridis",
    )
    plt.colorbar(
        location="right",
        shrink=0.5,
        aspect=5
    )
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.savefig("lesson_2.08/res/task_4.png")
    plt.close()


def task5() -> None:
    """
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 5, 100)
    y = np.cos(np.pi * x)

    ax.plot(x, y, "r")
    ax.fill_between(x, y, where=(y > 0), color="blue", alpha=0.5)
    ax.fill_between(x, y, where=(y < 0), color="blue", alpha=0.5)

    fig.savefig("lesson_2.08/res/task_5.png")
    plt.close()


def task6() -> None:
    """
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 5, 100)
    y = np.cos(np.pi * x)

    ax.plot(x, y)

    fig.savefig("lesson_2.08/res/task_5.png")
    plt.close()
    pass


def task7() -> None:
    """
    """
    pass


def task8() -> None:
    """
    """
    pass


def task9() -> None:
    """
    """
    pass


def task10() -> None:
    """
    """
    pass


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
