"""
Домашнее задание № 2.04
"Индексы в Pandas"
по курсу Цифровых кафедр "Python: от основ до машинного обучения"

Выполнил: студент группы № 5040103/30401
Курчуков Максим
"""

import numpy as np
import pandas as pd


def task_1() -> None:
    """
    1. Разобраться как использовать мультииндексные ключи в данном примере:
    >>> index = [
    ...     ('city_1', 2010),
    ...     ('city_1', 2020),
    ...     ('city_2', 2010),
    ...     ('city_2', 2020),
    ...     ('city_3', 2010),
    ...     ('city_3', 2020),
    ... ]
    >>>
    >>> population = [
    ...     101,
    ...     201,
    ...     102,
    ...     202,
    ...     103,
    ...     203,
    ... ]
    >>> pop = pd.Series(population, index = index)
    >>> pop_df = pd.DataFrame(
    ...     {
    ...         'total': pop,
    ...         'something': [
    ...             10,
    ...             11,
    ...             12,
    ...             13,
    ...             14,
    ...             15,
    ...         ]
    ...     }
    ... )
    >>>
    >>> ## ???
    >>> pop_df_1 = pop_df.loc???['city_1', 'something']
    >>> pop_df_1 = pop_df.loc???[['city_1', 'city_3'], ['total', 'something']]
    >>> pop_df_1 = pop_df.loc???[['city_1', 'city_3'], 'something']
    """

    index = [
        ('city_1', 2010),
        ('city_1', 2020),
        ('city_2', 2010),
        ('city_2', 2020),
        ('city_3', 2010),
        ('city_3', 2020),
    ]

    population = [
        101,
        201,
        102,
        202,
        103,
        203,
    ]
    pop = pd.Series(population, index=pd.MultiIndex.from_tuples(index))
    pop_df = pd.DataFrame(
        {
            "total": pop,
            "something": [
                10,
                11,
                12,
                13,
                14,
                15,
            ],
        }
    )
    
    pop_df_1 = pop_df.loc['city_1', 'something']
    pop_df_2 = pop_df.loc[['city_1', 'city_3'], ['total', 'something']]
    pop_df_3 = pop_df.loc[['city_1', 'city_3'], 'something']
    print(f"1) Исходный Датафрейм:\n{pop_df}")
    print(f"1) Выбор всех значений 'something' для 'city_1':\n{pop_df_1}")
    print(f"1) Выбор 'total' и 'something' для городов 'city_1' и 'city_3':\n{pop_df_2}")
    print(f"1) Выбор только 'something' для городов 'city_1' и 'city_3':\n{pop_df_3}")


def task_2() -> None:
    """
    2. Из получившихся данных выбрать данные по:
    - 2020 году (для всех столбцов)
    - job_1 (для всех строк)
    - для city_1 и job_2
    """

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
    rng = np.random.default_rng(1)
    data = rng.random((4, 6))
    data_df = pd.DataFrame(data=data, columns=columns, index=index)
    print(f"2) Исходный Датафрейм:\n{data_df}")

    data_2020 = data_df.loc[(slice(None), 2020), :]
    print(f"2) Данные по 2020 году (все столбцы):\n{data_2020}")

    data_job_1 = data_df.loc[:, (slice(None), "job_1")]
    print(f"2) Данные по job_1 (все строки):\n{data_job_1}")

    data_city_1_job_2 = data_df.loc["city_1", (slice(None), "job_2")]
    print(f"2) Данные по city_1 и job_2:\n{data_city_1_job_2}")


def task_3() -> None:
    """
    3. Взять за основу DataFrame со следующей структурой:
    >>> index = pd.MultiIndex.from_product(
    ...     [
    ...         ['city_1', 'city_2'],
    ...         [2010, 2020]
    ...     ],
    ...     names=['city', 'year']
    ... )
    >>> columns = pd.MultiIndex.from_product(
    ...     [
    ...         ['person_1', 'person_2', 'person_3'],
    ...         ['job_1', 'job_2']
    ...     ],
    ...     names=['worker', 'job']
    ... )

    Выполнить запрос на получение следующих данных:
        - все данные по person_1 и person_3
        - все данные по первому городу и первым двум person-ам
          (с использование срезов)

    Приведите пример (самостоятельно) с использованием `pd.IndexSlice`
    """

    index = pd.MultiIndex.from_product(
        [
            ['city_1', 'city_2'],
            [2010, 2020]
        ],
        names=['city', 'year']
    )
    columns = pd.MultiIndex.from_product(
        [
            ['person_1', 'person_2', 'person_3'],
            ['job_1', 'job_2']
        ],
        names=['worker', 'job']
    )

    rng = np.random.default_rng(1)
    data = rng.random((4, 6))
    data_df = pd.DataFrame(data=data, columns=columns, index=index)
    print(f"3) Исходный Датафрейм:\n{data_df}")

    idx = pd.IndexSlice

    data_persons = data_df.loc[:, idx[["person_1", "person_3"], :]]
    print(f"3) Данные по person_1 и person_3:\n{data_persons}")

    data_city1_persons = data_df.loc["city_1", idx["person_1":"person_2", :]]
    print(f"3) Данные по первому городу и первым двум person-ам:\n{data_city1_persons}")


def task_4() -> None:
    """
    4. Привести пример использования inner и outer джойнов для Series
    (данные примера скорее всего нужно изменить):

    >>> ser1 = pd.Series(['a', 'b', 'c'], index=[1,2,3])
    >>> ser2 = pd.Series(['b', 'c', 'f'], index=[4,5,6])
    >>>
    >>> print (pd.concat([ser1, ser2], join='outer'))
    >>> print (pd.concat([ser1, ser2], join='inner'))
    """

    ser1 = pd.Series(["a", "b", "c"], index=[1, 2, 3])
    ser2 = pd.Series(["d", "e", "f"], index=[3, 4, 5])

    outer_join = pd.concat(
        [ser1, ser2],
        join="outer",
        keys=["head1", "head2"],
        axis=1
    )
    print(f"4) Outer join (все значения):\n{outer_join}")

    inner_join = pd.concat(
        [ser1, ser2],
        join="inner",
        keys=["head1", "head2"],
        axis=1
    )
    print(f"4) Inner join (только общие индексы):\n{inner_join}")


def main() -> None:
    task_1()
    task_2()
    task_3()
    task_4()


if __name__ == "__main__":
    main()
