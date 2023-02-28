'''100 pandas exercises'''
# 1. Import pandas and check version
import numpy as np
import pandas as pd
print(pd.__version__)
#print(pd.show_versions(as_json=True))

# 2. Create a pandas series from each of the items
# below: a list, numpy and a dictionary
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))

list_series = pd.Series(mylist)
arr_series = pd.Series(myarr)
dict_series = pd.Series(mydict)

print(dict_series.head())

# 3. Convert the series ser into a dataframe
# with its index as another column on the
# dataframe.
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
ser = pd.Series(mydict)
ser
ser_df = ser.to_frame().reset_index()
ser_df

# 4. Combine ser1 and ser2 to form a dataframe.
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))

df = pd.concat([ser1, ser2], axis=1)
print(df)

# 5. Give a name to the series ser
# calling it ‘alphabets’.
ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser.name = 'alphabets'
ser

# 6. From ser1 remove items present in ser2.
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

# Removes values that are in ser2 based on True/False
res = ser1[~np.isin(ser1, ser2)]
res

# 7. Get all items of ser1 and ser2
# not common to both.
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])

pd.concat([ser1[~np.isin(ser1, ser2)], ser2[~np.isin(ser2, ser1)]])


# 8. Compute the minimum, 25th percentile,
# median, 75th, and maximum of ser.
np.random.RandomState(100)
ser = pd.Series(np.random.normal(10, 5, 25))
ser.describe()[['min', '25%', '50%', '75%', 'max']]
np.percentile(ser, q=[0, 25, 50, 75, 100])

# 9. Calculate the frequency counts of each unique value ser.
ser = pd.Series(np.take(list('abcdefgh'),
                        np.random.randint(8, size=30)))
ser.value_counts()

# 10. From ser, keep the top 2 most frequent items
# as it is and replace everything else as ‘Other’.
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))
ser
ser.value_counts()
ser[~ser.isin(ser.value_counts().index[:2])] = 'Other'
ser

# 11. Bin the series ser into 10 equal deciles and
# replace the values with the bin name.
ser = pd.Series(np.random.random(20))
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
labels = [1,2,3,4,5,6,7,8,9,10]
df['binned'] = pd.cut(ser, bins=bins, labels=labels)
print (df)

# 12. Reshape the series ser
# into a dataframe with 7 rows and 5 columns
ser = pd.Series(np.random.randint(1, 10, 35))
ser.shape

ser_df = pd.DataFrame(ser.values.reshape((7,5)))
print(ser_df)
ser_df['new_col'] = range(0,7)
ser_df
type(ser_df)

# 13. Find the positions of numbers
# that are multiples of 3 from ser.
ser = pd.Series(np.random.randint(1, 10, 7))
print(ser)
np.where(ser % 3 == 0)

# 14. From ser, extract the items at positions in list pos.
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]
ser.iloc[pos]
ser[pos]
ser.take(pos)

# 15. Stack ser1 and ser2 vertically and
# horizontally (to form a dataframe).
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))
# horizontal -- by columns
pd.concat([ser1, ser2], axis = 1)
# vertical -- by row
pd.concat([ser1, ser2], axis = 0)

# 16. Get the positions of items of ser2 in ser1 as a list.
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])

[np.where(i == ser1)[0].tolist()[0] for i in ser2]

# 17. Compute the mean squared error of truth and pred series.
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)

np.mean((truth - pred)**2)

# 18. Change the first character of
# each word to upper case in each word of ser.
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.str.title()

# 19. How to calculate the number of characters
# in each word in a series?
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
ser.str.len()

# 20. Difference of differences
# between the consequtive numbers of ser.
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
ser.diff().tolist()
ser.diff().diff().tolist()

# 21. How to convert a series of date-strings to a timeseries?
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
pd.to_datetime(ser)

# 22. Get the day of month, week number,
# day of year and day of week from ser.
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
ser_df = pd.to_datetime(ser)
date_month = ser_df.dt.month.tolist()
print(date_month)
week_number = ser_df.dt.isocalendar().iloc[:,1].tolist()
print(week_number)
day_number = ser_df.dt.dayofyear.tolist()
print(day_number)
day_of_week = ser_df.dt.strftime('%A').tolist()
print(day_of_week)

# 23. Change ser to dates that start
# with 4th of the respective months.
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])
ser_dates = pd.to_datetime(ser)
ser_dates[0]
ser_dates = ser_dates.dt.to_period('M')
from dateutil.parser import parse
ser.map(lambda x: parse('04 ' + x))

# 24. From ser, extract words that contain atleast 2 vowels.
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
ser.str.lower().str.count(r'[aeiou]')
ser.iloc[np.where(ser.str.lower().str.count(r'[aeiou]')>=2) ].tolist()

# 25. Extract the valid emails from the series emails.
# The regex pattern for valid emails is provided as reference.
import re
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
pattern
for email in emails:
    if re.search(pattern, email):
        print(email)
        
# 26. Compute the mean of weights of each fruit.
fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())
fruit_df = pd.concat([fruit, weights], axis=1)
fruit_df.columns = ["fruit", "weights"]
fruit_df
fruit_df.groupby(fruit).mean().round(1)

# 27. Compute the euclidean distance between series (points) p and q,
# without using a packaged formula.
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

np.sqrt(np.sum((p-q)**2))

# 28. Get the positions of peaks
# (values surrounded by smaller values on both sides) in ser.
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
dd = np.diff(np.sign(np.diff(ser)))
peak_locs = np.where(dd == -2)[0] + 1
peak_locs

# Replace the spaces in my_str with the least frequent character.
my_str = 'dbc deb abed gade'
from collections import Counter
min(Counter(my_str).values())
my_str.replace(' ', min(Counter(my_str)))


def fizzbuzz(n):
    for n_int in range(n+1):
        if n_int % 3 == 0 and n_int % 5 == 0:
            print("fizzbuzz")
            continue
        elif n_int % 3 == 0:
            print("fizz")
            continue
        elif n_int % 5 == 0:
            print("buzz")
            continue
        else:
            print(n_int)

def fizzBuzz(n):
    # Write your code here
    for n_int in range(1, n+1):
        if n_int % 3 == 0 and n_int % 5 == 0:
            print('FizzBuzz')
        elif n_int % 3 == 0:
            print('Fizz')
        elif n_int % 5 == 0:
            print('Buzz')
        else:
            print(str(n_int))
# print "\n".join(fizzBuzz(n) for n in range(1,n))
fizzbuzz(15)

n = 50
fizzbuzz(50)

def Fizzbuzz(n):
    for n_int in range(n+1):
        if n_int % 3 == 0 & n_int % 5 == 0:
            print("fizzbuzz")
            continue
        elif n_int % 3 == 0:
            print("fizz")
            continue
        elif n_int % 5 == 0:
            print("buzz")
            continue
        else:
            print(n_int)
            
Fizzbuzz(15)

import pandas as pd

transactions = {"transaction_id" : [1, 2, 3, 4, 5], "product_id" : [101, 102, 103, 104, 105], "amount" : [3, 5, 8, 3, 2]}

products = {"product_id" : [101, 102, 103, 104, 105], "price" : [20.00, 21.00, 15.00, 16.00, 52.00]}

df_transactions = pd.DataFrame(transactions)

df_products = pd.DataFrame(products)

df = pd.merge(df_transactions, df_products, on='product_id')
df['total_value'] = df['amount'] * df['price']
df.loc[df['total_value']> 100]

# Write a function complete_address
# to create a single dataframe with complete addresses
# in the format of street, city, state, zip code.
import pandas as pd

addresses = {"address": ["4860 Sunset Boulevard, San Francisco, 94105", "3055 Paradise Lane, Salt Lake City, 84103", "682 Main Street, Detroit, 48204", "9001 Cascade Road, Kansas City, 64102", "5853 Leon Street, Tampa, 33605"]}

cities = {"city": ["Salt Lake City", "Kansas City", "Detroit", "Tampa", "San Francisco"],
          "state": ["Utah", "Missouri", "Michigan", "Florida", "California"]}

df_addresses = pd.DataFrame(addresses)
df_cities = pd.DataFrame(cities)
df_cities
df_addresses[['street', 'city', 'zipcode']] = df_addresses["address"].apply(lambda x: pd.Series(str(x).split(", ")))
df_addresses = df_addresses.merge(df_cities, on='city')
df_addresses['address'] = df_addresses[['street', 'city', 'state', 'zipcode']].agg(', '.join, axis=1)
df_addresses.drop(columns=['street', 'city', 'state', 'zipcode'])

# You’re given a dataframe df_rain containing rainfall data.
# The dataframe has two columns: day of the week and rainfall in inches.

# Write a function median_rainfall to find the
# median amount of rainfall for the days on which it rained. 
import pandas as pd

rainfall = {"Day" : ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "Inches" : [0, 1.2, 0, 0.8, 1]}

df_rain = pd.DataFrame(rainfall)
df_rain.select_dtypes(include='Inches') != 0

df_rain[df_rain['Inches'] !=0].median(numeric_only=True)[0]

# Write a function named grades_colors to select only the rows
# where the student’s favorite color is green or red and their grade is above 90.
import pandas as pd

students = {"name" : ["Tim Voss", "Nicole Johnson", "Elsa Williams", "John James", "Catherine Jones"],
            "age" : [19, 20, 21, 20, 23],
            "favorite_color" : ["red", "yellow", "green", "blue", "green"],
            "grade" : [91, 95, 82, 75, 93]}

students_df = pd.DataFrame(students)
students_df[((students_df['favorite_color'] == 'red') | (students_df['favorite_color'] == 'green')) & (students_df['grade'] >90)]


df = pd.Series([-34, 40, -89, 5, -26])
mu0 = 1
df.mean()
df.sem()
(df.mean() - mu0) / df.sem()


def fizzbuzz(n):
    for FizzBuzz in range(n+1):
        if FizzBuzz % 5 == 0 and FizzBuzz % 3 == 0:
            print("FizzBuzz")
            continue
        elif FizzBuzz % 3 == 0:
            print("Fizz")
            continue
        elif FizzBuzz % 5 == 0:
            print("Buzz")
            continue
        else:
            print(FizzBuzz)
fizzbuzz(15)


# 29. How to replace missing spaces in a string with the least frequent character?
my_str = 'dbc deb abed gade'
ser = pd.Series(list(my_str))
freq = ser.value_counts()
freq
least_freq = freq.dropna().index[-1]
least_freq
''.join(ser.replace(' ', least_freq))

# 30. How to create a TimeSeries starting ‘2000-01-01’ and
# 10 weekends (saturdays) after that having random numbers as values?




# Write a Pandas program to select the specified columns and rows from a given DataFrame.
# Select 'name' and 'score' columns in rows 1, 3, 5, 6 from the following data frame.
import pandas as pd
import numpy as np

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df.iloc[[1,3,5,6],[0,1]]
df[['name', 'score']].iloc[[1,3,5,6]]

# Write a Pandas program to select the rows
# where the number of attempts in the examination is greater than 2.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts' : [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df[df['attempts'] > 2]

# Write a Pandas program to count the number of rows and columns of a DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df.shape[0] # rows
df.shape[1] # columns

# Write a Pandas program to select the rows where the score is missing
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df[df['score'].isnull()]

# Write a Pandas program to select the rows
# the score is between 15 and 20 (inclusive).

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df[(df['score'] >= 15) & (df['score'] <= 20)]
df[df['score'].between(15,20)]

# Write a Pandas program to select the rows
# where number of attempts in the examination
# is less than 2 and score greater than 15.

exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df[(df['score'] > 15) & (df['attempts'] < 2)]

# Write a Pandas program to change the score in row 'd' to 11.5.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df_new = df[['name', 'score', 'attempts']]
df_new

df
df.loc['d','score'] = 11.5
df

# Write a Pandas program to calculate the
# sum of the examination attempts by the students
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df['attempts'].sum()

# Write a Pandas program to calculate the
# mean score for each different student in DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df.score.mean()

# Write a Pandas program to append a new row 'k'
# to data frame with given values for each column.
# Now delete the new row and return the original DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
new_data = ['Kevin', 19, 1, 'yes']
df.loc['k'] = new_data
df
df.drop('k')

# Write a Pandas program to sort the DataFrame
# first by 'name' in descending order,
# then by 'score' in ascending order. 
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df.sort_values(by=['name', 'score'], ascending=[False, True])

# Write a Pandas program to replace the 'qualify' column
# contains the values 'yes' and 'no' with True and False.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df.replace({'qualify':{'yes':True, 'no':False}})

# Write a Pandas program to change the name
# 'James' to 'Suresh' in name column of the DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df['name'] = df['name'].replace('James', 'Suresh')
df

# Write a Pandas program to delete the 'attempts' column
# from the DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data, index=labels)
df.drop(columns='attempts')

# Write a Pandas program to insert a
# new column in existing DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
color = ['Red','Blue','Orange','Red','White','White','Blue','Green','Green','Red']
df['color'] = color
df

# Write a Pandas program to iterate over rows in a DataFrame.
exam_data = [{'name':'Anastasia', 'score':12.5}, {'name':'Dima','score':9}, {'name':'Katherine','score':16.5}]
df = pd.DataFrame(exam_data)
df

for index, rows in df.iterrows():
    print(rows['name'], rows['score'])

import pandas as pd
series = pd.Series([28, 9, 82, 38, 44, 55, 23, 7, 64, 59, 5, 76, 12, 89, 50, 25, 33, 45, 93, 60, 72, 21, 89, 86, 26])
series[0]
series.index[0]
series.index[-1]
round(len(series) / 2)
series.values[-1]
type(series.values[round(len(series) / 2)])
series.median().astype('int64')

values = list(range(0,26))
pd.Series(values).str.isalpha
values.str.isalpha


series = pd.Series([28, 9, 82, 38, 44, 55, 23, 7, 64, 59, 5, 76, 12, 89, 50, 25, 33, 45, 93, 60, 72, 21, 89, 86, 26])
series.index = range(1,len(series)+1)
series

populations = {
    'Cairo': 11001000,
    'Johannesburg': 3670000,
    'Dakar': 2863000,
    'Casablanca': 3284000,
    'Lagos': 10578000
}
series_pop = pd.Series(populations)
series_pop['Luanda'] = 4772000
series_pop.drop('Luanda')


series = pd.Series({'a':6, 'b':2, 'c':9, 'd':1, 'e':4, 'f':8, 'g':3, 'h':5, 'i':7})
series.loc[['d', 'e', 'f','g']]

series = pd.Series({'a':6, 'b':2, 'c':9, 'd':1, 'e':4, 'f':8, 'g':3, 'h':5, 'i':7})

list(zip(series.index, series))
