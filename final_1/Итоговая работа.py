"""
Контрольная работа.
Раздел 1. Углубленное изучение языка Python. 
Выполнил: студент Курчуков Максим.
"""
# Импорт библиотеки для тестирования результатов контрольной работы
try:
    import pytest
    has_pytest = True
except ModuleNotFoundError:
    has_pytest = False


def reverse_string(s):
    """
    ## Разворот строки
    Напишите функцию, которая принимает строку в качестве входного 
    параметра и возвращает новую строку, в которой символы исходной 
    строки расположены в обратном порядке.
    """
    return s[::-1]


def draw(n):
    """
    ## Ромб
    Напишите функцию, которая выводит на экран ромб, составленный 
    из символов звёздочек `*`. Размер ромба определяется введённым 
    пользователем нечётным числом n, которое задаёт ширину (и высоту) 
    ромба в его самой широкой части.
    """
    result = ''
    for i in range(1, n + 1, 2):
        result += ' ' * ((n - i) // 2) + '*' * i + '\n'
    for i in range(n - 2, 0, -2):
        result += ' ' * ((n - i) // 2) + '*' * i + '\n'
    return result


def gcd(a, b):
    """
    ## НОД
    Напишите функцию, которая вычисляет наибольший 
    общий делитель (НОД) двух целых чисел.
    """
    while b:
        a, b = b, a % b
    return a


def convert_to_decimal(number_str, base):
    """
    ## Система счисления
    Напишите функцию, которая принимает строковое представление 
    числа в произвольной системе (макс. 36) счисления и его основание, 
    и возвращает это число в десятичной системе счисления.
    """
    value = 0
    for char in number_str:
        digit = int(char, 36)
        value = value * base + digit
    return value


def is_palindrome(s):
    """
    ## Палиндром
    Напишите функцию, которая проверяет, 
    является ли заданная строка палиндромом.
    """
    return s == s[::-1]


def count_greater_than_kth(arr, k):
    """
    ## k порядковая статистика
    Напишите функцию, которая принимает массив чисел и 
    целое число `k`, и вычисляет количество элементов в массиве, 
    которые больше, чем элемент, находящийся на позиции `k` 
    в упорядоченном по возрастанию массиве 
    (т.е. больше, чем `k`-я порядковая статистика).
    """
    arr.sort()
    threshold = arr[k - 1]
    return sum(x > threshold for x in arr)


def count_unique_substrings(text, k):
    """
    ## Уникальные подстроки
    Напишите функцию, которая принимает строку и целое число k, 
    и подсчитывает количество уникальных подстрок длины k в этом тексте.
    """
    substrings = set()
    for i in range(len(text) - k + 1):
        substrings.add(text[i:i + k])
    return len(substrings)


def minimum(n):
    """
    ## Минимум
    Напишите функцию, которая для заданного целого числа N 
    находит такие целые положительные числа `a, b, c`, 
    что произведение `a * b * c = N`, и 
    сумма `a + b + c` минимальна.
    """
    a, b, c = 1, 1, n
    for i in range(1, int(n ** (1 / 3)) + 1):
        for j in range(i, int((n // i) ** 0.5) + 1):
            if n % (i * j) == 0:
                k = n // (i * j)
                if i + j + k < a + b + c:
                    a, b, c = i, j, k
    return a, b, c


def determinant(matrix):
    """
    ## Определитель
    Напишите функцию, которая вычисляет 
    определитель заданной квадратной матрицы.
    """
    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(len(matrix)):
        minor = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += (-1) ** col * matrix[0][col] * determinant(minor)
    return det


def is_valid_sequence(s):
    """
    ## Скобочная последовательность
    Напишите функцию, которая проверяет правильность скобочной 
    последовательности в заданной строке. 
    Последовательность считается правильной, 
    если все открывающиеся скобки корректно закрываются 
    соответствующими закрывающими скобками в правильном порядке.
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack


def test_reverse_string():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("Привет") == "тевирП"
    assert reverse_string("12345") == "54321"


def test_draw():
    assert draw(7) == "   *\n  ***\n *****\n*******\n *****\n  ***\n   *\n"


def test_gcd():
    assert gcd(48, 18) == 6
    assert gcd(100, 25) == 25
    assert gcd(17, 13) == 1


def test_convert_to_decimal():
    assert convert_to_decimal("1010", 2) == 10
    assert convert_to_decimal("1A", 16) == 26
    assert convert_to_decimal("123", 8) == 83
    assert convert_to_decimal("Z", 36) == 35


def test_is_palindrome():
    assert is_palindrome("мадам")
    assert is_palindrome("топот")
    assert not is_palindrome("привет")


def test_count_greater_than_kth():
    arr = [5, 3, 8, 6, 2]
    k = 3
    result = count_greater_than_kth(arr, k)

    assert result == 2


def test_count_unique_substrings():
    text = "abcabc"
    k = 3
    result = count_unique_substrings(text, k)

    assert result == 3


def test_minimum():
    assert minimum(12) == (2, 2, 3)
    assert minimum(27) == (3, 3, 3)
    assert minimum(7) == (1, 1, 7)


def test_determinant():
    matrix = [
        [1, 2],
        [3, 4]
    ]
    result = determinant(matrix)

    assert result == -2


def test_is_valid_sequence():
    s = "({[]})"
    result = is_valid_sequence(s)

    assert result


def all_test():
    test_reverse_string()
    test_draw()
    test_gcd()
    test_convert_to_decimal()
    test_is_palindrome()
    test_count_greater_than_kth()
    test_count_unique_substrings()
    test_minimum()
    test_determinant()
    test_is_valid_sequence()


if __name__ == "__main__":
    if has_pytest:        
        pytest.main([__file__])
    else:
        all_test()
