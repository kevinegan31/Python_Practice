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
