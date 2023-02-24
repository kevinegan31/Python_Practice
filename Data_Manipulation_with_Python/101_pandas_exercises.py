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
# horizontal
pd.concat([ser1, ser2], axis = 1)
# vertical
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
my_str.replace(' ', Counter(my_str).min())
