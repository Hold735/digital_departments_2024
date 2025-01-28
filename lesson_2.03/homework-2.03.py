"""
Домашнее задание № 2.03
"Введение в Pandas"
по курсу Цифровых кафедр "Python: от основ до машинного обучения" 

Выполнил: студент группы № 5040103/30401
Курчуков Максим
"""
import numpy as np
import pandas as pd


def task_1() -> None:
    """
    1. Привести различные способы создания объектов типа Series
    Для создания Series можно использовать
     - списки Python или массивы NumPy
     - скалярные значение
     - словари
    """
    data_list = [1, 2, 3, 4, 5]
    s1 = pd.Series(data_list)

    data_int = 5
    s2 = pd.Series(data_int)

    data_dict = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    s3 = pd.Series(data_dict)

    data_array = np.array([1, 2, 3, 4, 5], dtype=int)
    s4 = pd.Series(data_array)

    print(f"1) Series из списка Python:\n{s1}")
    print(f"1) Series из скалярного значения:\n{s2}")
    print(f"1) Series из словаря:\n{s3}")
    print(f"1) Series из массива NumPy:\n{s4}")


def task_2() -> None:
    """
    2. Привести различные способы создания объектов типа DataFrame
    DataFrame. Способы создания
     - через объекты Series
     - списки словарей
     - словари объектов Series
     - двумерный массив NumPy
     - структурированный массив Numpy
    """
    data_series1 = pd.Series([1, 2, 3, 4, 5])
    data_series2 = pd.Series([6, 7, 8, 9, 10])
    df1 = pd.DataFrame([data_series1, data_series2])

    data_list_of_dict = [
        {
            "a": 1,
            "b": 2
        },
        {
            "a": 3,
            "b": 4
        }
    ]
    df2 = pd.DataFrame(data_list_of_dict)

    data_dict_of_series = {
        "a": data_series1,
        "b": data_series2
    }
    df3 = pd.DataFrame(data_dict_of_series)

    data_array_numpy = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    df4 = pd.DataFrame(data_array_numpy, columns=["A", "B", "C", "D", "E"])

    data_array_numpy_struct = np.array(
        [(20, "Den"), (19, "Bob")],
        dtype=[
            ("age", "i4"),
            ("name", "U10")
        ]
    )
    df5 = pd.DataFrame(data_array_numpy_struct)

    print(f"2) DataFrame из Serieses:\n{df1}")
    print(f"2) DataFrame из списка словарей:\n{df2}")
    print(f"2) DataFrame из словаря объектов Serieses:\n{df3}")
    print(f"2) DataFrame из двумерного массива NumPy:\n{df4}")
    print(f"2) DataFrame из структурированного массива NumPy:\n{df5}")


def task_3() -> None:
    """
    3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1
    """
    s1 = pd.Series([1, 2], index=["a", "b"])
    s2 = pd.Series([3, 4], index=["b", "c"])
    result = s1.add(s2, fill_value=1)
    print(f"3) Объединенные Series с заменой NaN на 1:\n{result}")


def task_4() -> None:
    """
    4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
    """
    rng = np.random.default_rng(1)
    A = rng.integers(0, 10, (3,4))
    df = pd.DataFrame(A, columns=["a", "b", "c", "d"])
    print(f"4) Оригинальный DataFrame:\n{df}")
    print(f"4) DataFrame с вычитанием по строкам:\n{df - df.iloc[0]}")

    df_2 = df.sub(df.iloc[:, 0], axis=0)
    print(f"4) DataFrame с вычитанием по столбцам:\n{df_2}")


def task_5() -> None:
    """
    5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()
    """
    data = {
        "A": [1, 2, np.nan, 4, 5],
        "B": [1, 2, 3, np.nan, 5],
        "C": [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    print(f"5) Оригинальный DataFrame:\n{df}")
    df_fill = df.ffill()
    print(f"5) DataFrame с заполненными NaN вперед:\n{df_fill}")
    df_fill = df.bfill()
    print(f"5) DataFrame с заполненными NaN назад:\n{df_fill}")


def main() -> None:
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()


if __name__ == "__main__":
    main()
