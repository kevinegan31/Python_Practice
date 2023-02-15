'''Hackerrank Python Problems'''

# print all possible combinations of x, y, z, n]
x = 1
y = 1
z = 2
n = 3
print([[i,j,k] for i in range(0,x+1) for j in range(0, y+1) for k in range(0, z+1) if i+j+k !=n])

# Given the participants' score sheet for your University Sports Day,
# you are required to find the runner-up score.
# You are given  scores. Store them in a list and find the score of the runner-up.
n = 5
arr = [2, 3, 6, 6, 5]
list(set(arr))[-2]
