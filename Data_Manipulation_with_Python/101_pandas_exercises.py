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