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
