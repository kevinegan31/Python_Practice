"""Code Signal practice"""


def solution(year):
    if (year % 100) == 0:
        return (year) // 100
    else:
        return (year) // 100 + 1


year = 1701
solution(year)

# Check Palindrome code
def solution(inputString):
    rev_string = inputString[::-1]
    return inputString == rev_string


inputString = "abac"
rev_string = inputString[::-1]
rev_string
solution(inputString)

# Adjacent Elements Product
# Given an array of integers,
# find the pair of adjacent elements that has
# the largest product and return that product.
inputArray = [3, 6, -2, -5, 7, 3]


def solution(inputArray):
    for i in inputArray:
        if i == 0:
            return 0
        else:
            return max(
                inputArray[i] * inputArray[i + 1] for i in range(len(inputArray) - 1)
            )


solution(inputArray)

# Shape Area
# Below we will define an n-interesting polygon.
# Your task is to find the area of a polygon for a given n.
def solution(n):
    return n * n + (n - 1) * (n - 1)


# Ratiorg Statues
statues = [6, 2, 3, 8]


def solution(statues):
    statues.sort()
    return sum([statues[i + 1] - statues[i] - 1 for i in range(len(statues) - 1)])


solution(statues)
i = 0
statues[i + 1] - statues[i] - 1
statues = [6, 2, 3, 8]
statues.sort()
for i in range(len(statues) - 1):
    print(statues[i + 1] - statues[i] - 1)


# Make Array Consecutive 2
# Given a sequence of integers as an array,
# determine whether it is possible to obtain a strictly
# increasing sequence by removing no more than one element
# from the array.
def solution(sequence):
    for i in range(len(sequence) - 1):
        if sequence[i] <= sequence[i + 1] or sequence[i] <= sequence[i - 1]:
            return False
    return True


# Almost Increasing Sequence
def increasingSequence(sequence):
    for i in range(len(sequence) - 1):
        if not sequence[i] < sequence[i + 1]:
            return False
    return True


def almostIncreasingSequence(sequence):
    i = 0
    while i < len(sequence) - 1:
        if not sequence[i] < sequence[i + 1]:
            if increasingSequence(
                sequence[:i] + sequence[i + 1 :]
            ) or increasingSequence(sequence[: i + 1] + sequence[i + 2 :]):
                return True
            else:
                return False
        i += 1
    return True


sequence = [1, 3, 2]
almostIncreasingSequence(sequence)

# Matrix Elements Sum
def solution(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                for k in range(i, len(matrix)):
                    matrix[k][j] = 0
    return sum([sum(i) for i in matrix])


matrix = [[0, 1, 1, 2], [0, 5, 0, 0], [2, 0, 3, 3]]
solution(matrix)

# All Longest Strings
# Given an array of strings,
# return another array containing all of its longest strings.
def solution(inputArray):
    max_len = max([len(i) for i in inputArray])
    return [i for i in inputArray if len(i) == max_len]


inputArray = ["aba", "aa", "ad", "vcd", "aba"]
solution(inputArray)

# Common Character Count
# Given two strings, find the number of common characters between them.
def solution(s1, s2):
    return sum([min(s1.count(i), s2.count(i)) for i in set(s1) & set(s2)])


s1 = "aabcc"
s2 = "adcaa"
solution(s1, s2)

# Is Lucky
# Ticket numbers usually consist of an even number of digits.
# A ticket number is considered lucky if the sum of the first
# half of the digits is equal to the sum of the second half.
def solution(n):
    for i in range(len(str(n)) // 2):
        if sum([int(i) for i in str(n)[: len(str(n)) // 2]]) == sum(
            [int(i) for i in str(n)[len(str(n)) // 2 :]]
        ):
            return True
    return False


n = 1230
solution(n)
n = 239017
solution(n)

# Sort By Height
# Some people are standing in a row in a park.
# There are trees between them which cannot be moved.
# Your task is to rearrange the people by their heights in a non-descending order
# without moving the trees.
# People can be very tall!
def solution(a):
    b = [i for i in a if i != -1]
    b.sort()
    for i in range(len(a)):
        if a[i] != -1:
            a[i] = b.pop(0)
    return a


a = [-1, 150, 190, 170, -1, -1, 160, 180]
solution(a)

# Reverse In Parentheses
# Write a function that reverses characters
# in (possibly nested) parentheses in the input string.
# Input strings will always be well-formed with matching ()s.
def solution(inputString):
    while "(" in inputString:
        i = inputString.rfind("(")
        j = inputString.find(")", i)
        inputString = (
            inputString[:i] + inputString[i + 1 : j][::-1] + inputString[j + 1 :]
        )
    return inputString


inputString = "foo(bar)baz(blim)"
solution(inputString)


# Alternating Sums
# Several people are standing in a row and
# need to be divided into two teams.
# The first person goes into team 1, the second goes into team 2,
# the third goes into team 1 again, the fourth into team 2, and so on.
# You are given an array of positive integers -
# the weights of the people.
# Return an array of two integers,
# where the first element is the total weight of team 1,
# and the second element is the total weight of team 2 after the division is complete.
def solution(a):
    return [sum(a[::2]), sum(a[1::2])]


a = [50, 60, 60, 45, 70]
solution(a)

# Add Border
# Given a rectangular matrix of characters,
# add a border of asterisks(*) to it.
def solution(picture):
    ast = "*" * (len(picture[0]) + 2)
    picture = [ast] + [f"*{i}*" for i in picture] + [ast]
    return picture


picture = ["abc", "ded"]
solution(picture)

# Are Similar
# Two arrays are called similar if one can be obtained
# from another by swapping at most one pair of elements
# in one of the arrays.
# Given two arrays a and b, check whether they are similar.
def solution(a, b):
    return sorted(a) == sorted(b) and sum([a[i] != b[i] for i in range(len(a))]) <= 2


def solution(a, b):
    j = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            j += 1
    if sorted(a) == sorted(b) and j <= 2:
        return True
    else:
        return False


a = [1, 2, 3]
b = [2, 1, 3]
solution(a, b)
