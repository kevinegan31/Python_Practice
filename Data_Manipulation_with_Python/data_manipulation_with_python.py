
'''Data Manipulation in Python Practice'''

#Write a Pandas program to get the powers of an array values element-wise.
#Note: First array elements raised to powers from second array
import numpy as np
import pandas as pd

list1 = np.arange(6)
list2 = np.arange(6)[::-1]
df = pd.DataFrame([list1, list2]).T
print(df)
print(np.power(list1,list2))

# Write a Pandas program to create and display a DataFrame
# from a specified dictionary data which has the index labels.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
print(df)
# Sorted dataframe
df.reindex(sorted(df.columns), axis=1)

# Write a Pandas program to display
# a summary of the basic information about a specified DataFrame and its data.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("Summary of the basic information about this dataframe:")
print(df.info())

# Write a Pandas program to get the first 3 rows of a given DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("First three rows of the data frame:")
df.head(n=3)

# Write a Pandas program to select the 'name' and 'score' columns from the following DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("Select specific columns:")
print(df[['name', 'score']])

# Write a Pandas program to select the specified columns and rows from a given data frame
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("Select specific columns and rows:")
print(df[['name', 'score']].iloc[[1,3,5,6]])

# Write a Pandas program to select the rows
# where the number of attempts in the examination is greater than 2.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("Rows where the number of attempts in the examination is greater than 2:")
print(df[df["attempts"] > 2])

# Write a Pandas program to count the number of rows and columns of a DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
print("Total number of rows is ", df.shape[0])
print("Total number of columns is ", df.shape[1])

# Write a Pandas program to select the rows where the score is missing, i.e. is NaN.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
print("Rows where score is missing:")
print(df[df['score'].isnull()])

# Write a Pandas program to select the rows the score is between 15 and 20 (inclusive)
print("Rows between 15 and 20 inclusive:")
print(df[(df["score"] >= 15) & (df["score"] <= 20)])
# or
print(df[df["score"].between(15,20)])

# Write a Pandas program to select the rows where number of attempts
# in the examination is less than 2 and score greater than 15.
print("Rows where score greater than 15 and attempts less than 2:")
print(df[(df["score"] > 15) & (df["attempts"] < 2)])

# Write a Pandas program to change the score in row 'd' to 11.5.
df.loc["d", "score"] = 11.5
print(df.loc["d"])

# Write a Pandas program to calculate the sum of the examination attempts by the students.
print("Sum of examination attempts:")
print(df["attempts"].sum())

# Write a Pandas program to calculate the mean score for each different student in DataFrame.
print("Mean of examination scores:")
print(df["score"].mean())

# Write a Pandas program to append a new row 'k' to data frame with given values for each column.
# Now delete the new row and return the original DataFrame.
# Values for each column will be:
new_data = {'name' : 'Suresh', 'score': 15.5, 'attempts': 1, 'qualify': 'yes', 'label': 'k'}
new_data['attempts']
print(df)
df.loc['k'] = [new_data['name'], new_data['score'], new_data['attempts'],  new_data['qualify']]
df
df = df.drop('k')
df


# Write a Pandas program to sort the DataFrame first by 'name' in descending order,
# then by 'score' in ascending order.
print(df.sort_values(by=["name", "score"], ascending=[False, True]))

#  Write a Pandas program to replace the 'qualify' column
# contains the values 'yes' and 'no' with True and False.
#df.replace({"qualify": {'yes':True, 'no':False}})
# Or
#df['qualify'] = df['qualify'].map({'yes': True, 'no': False})
# or
df['qualify'].replace(['yes', 'no'], ['True', 'False'])

# Write a Pandas program to change the name 'James' to 'Suresh' in name column of the DataFrame. 
df["name"] = df["name"].replace("James", "Suresh")

df

# Write a Pandas program to delete the 'attempts' column from the DataFrame.
df = df.drop("attempts", axis = 1)
# or
df.pop("attempts")
df

# Write a Pandas program to insert a new column in existing DataFrame.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)
color = ['Red','Blue','Orange','Red','White','White','Blue','Green','Green','Red']
df["color"] = color
print(df)

# Write a Pandas program to iterate over rows in a DataFrame.
exam_data = [{'name':'Anastasia', 'score':12.5}, {'name':'Dima','score':9}, {'name':'Katherine','score':16.5}]
df = pd.DataFrame(exam_data)
for index, row in df.iterrows():
    print(row['name'], row['score'])

# Write a Pandas program to get list from DataFrame column headers.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)
print(df.columns.values)

# Write a Pandas program to rename columns of a given DataFrame.
d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
df = pd.DataFrame(data=d)

df.columns = ['Column1', 'Column2', 'Column3']
df.columns.values
df = df.rename(columns={'col1': 'Column1', 'col2': 'Column2', 'col3': 'Column3'})
df

# Write a Pandas program to select rows from a given DataFrame based on values in some columns.
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(data=d)
print("Original DataFrame")
print(df)
print("rows for col1 value == 4")
df[df["col1"] == 4]
df.loc[df["col1"] == 4]

# Write a Pandas program to change the order of a DataFrame columns.
print(df[["col3", "col2", "col1"]])

# Write a Pandas program to add one row in an existing DataFrame.
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(data=d)
print("Original DataFrame")
print(df)
new_data = {'col1' : 10, 'col2': 11, 'col3': 12}
df.append(new_data, ignore_index=True)
df

# Write a Pandas program to write a DataFrame to CSV file using tab separator.
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(data=d)
df.to_csv('new_file.csv', sep= '\t', index=False)

# Write a Pandas program to count city wise
# number of people from a given of data set (city, name of the person).
df1 = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']})
df1.groupby('city').size().reset_index(name="Number of people")

# Write a Pandas program to delete DataFrame row(s) based on given column value.
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(data=d)
print("Original DataFrame")
print(df)
new_df = df[df.col2 != 5]
print(new_df)

# Write a Pandas program to select a row of series/dataframe by given integer index.
print(df.loc[[2]])

# Write a Pandas program to replace all the NaN values with Zero's in a column of a dataframe.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
df.fillna(0)

# Write a Pandas program to convert index in a column of the given dataframe.
df.reset_index(level=0, inplace=True)
df
print("\nHiding index:")
print(df.to_string(index=False))

# Write a Pandas program to set a given value for particular cell in  DataFrame using index value.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)

df.set_value(8, 'score', 10.5)
df.at[8, 'score'] = 10.5
df

# Write a Pandas program to count the NaN values in one or more columns in DataFrame.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
df.isna().sum().sum()
df.isna().values.sum()

# Write a Pandas program to drop a list of rows from a specified DataFrame.
d = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}
df = pd.DataFrame(d)
df
df.drop([2,4])
df.drop(df.index[[2,4]])

# Write a Pandas program to reset index in a given DataFrame.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
df = df.drop(df.index[[0,1]])
df.reset_index()

# Write a Pandas program to divide a DataFrame in a given ratio.
df = pd.DataFrame(np.random.randn(10, 2))
part_70 = df.sample(frac=0.7, random_state=10)
print(part_70)
part_30 = df.drop(part_70.index)
print(part_30)

# Write a Pandas program to combining two series into a DataFrame.
s1 = pd.Series(['100', '200', 'python', '300.12', '400'])
s2 = pd.Series(['10', '20', 'php', '30.12', '40'])

df = pd.concat([s1, s2], axis = 1)
print(df)

# Write a Pandas program to shuffle a given DataFrame rows.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
print("Original DataFrame:")
print(df)

df_shuffle = df.sample(frac = 1)
print("Shuffled DataFrame:")
print(df_shuffle)

#  Write a Pandas program to convert DataFrame column type from string to datetime.
s = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'])
print("String Date:")
print(s)
df = pd.to_datetime(s)
df = pd.DataFrame(df)
print(df)

# Write a Pandas program to rename a specific column name in a given DataFrame.
d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
df = pd.DataFrame(data=d)
df.rename(columns = {"col2":"Column2"})

# Write a Pandas program to get a list of a specified column of a DataFrame.
d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}
df = pd.DataFrame(data=d)
print("Original DataFrame")
print(df)
col2_list = df["col2"].tolist()
print("Col2 of the DataFrame to list:")
print(col2_list)

# Write a Pandas program to create a DataFrame from a Numpy array
# and specify the index column and column headers.
dtype = [('Column1','int32'), ('Column2','float32'), ('Column3','float32')]
values = np.zeros(15, dtype=dtype)
index = ['Index'+str(i) for i in range(1, len(values)+1)]
# index = [i for i in range(1, len(values)+1)]
df = pd.DataFrame(values, index=index)
print(df)

# Write a Pandas program to find the row
# for where the value of a given column is maximum.
d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
df = pd.DataFrame(data=d)
print("Row where col1 has maximum value:")
df["col1"].argmax()
print("Row where col2 has maximum value:")
df["col2"].argmax()
print("Row where col3 has maximum value:")
df["col3"].argmax()

# Write a Pandas program to check whether a given column
# is present in a DataFrame or not.
d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
df = pd.DataFrame(data=d)
col_check = ["col4", "col1"]
for col in col_check:
        if col in df.columns:
                print(col + " is present in DataFrame.")
        else:
                print(col + " is not present in DataFrame.")

# Write a Pandas program to get the specified row value of a given DataFrame.
print("value of row 1")
for col in df.columns:
        print(col + " " + str( df[col].loc[0]))
print("value of row 4")
for col in df.columns:
        print(col + " " + str( df[col].loc[3]))

# Write a Pandas program to get the datatypes of columns of a DataFrame.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
print(df.dtypes)

# Write a Pandas program to append data to an empty DataFrame.
df = pd.DataFrame()
data = pd.DataFrame({"col1": range(3),"col2": range(3)})
df = pd.concat([df, data])
print("After appending some data:")
print(df)

# Write a Pandas program to sort a given DataFrame by two or more columns.
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
df.sort_values(by=["attempts", "name"], ascending=[True, False])

# Write a Pandas program to convert the datatype of a given column (floats to ints).
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df = pd.DataFrame(exam_data)
df.dtypes
# Fills na with the mean value of scores then converts to int
df["score"] = df["score"].fillna(df["score"].mean()).astype(int)
df.dtypes

# Write a Pandas program to remove infinite values from a given DataFrame.
df = pd.DataFrame([1000, 2000, 3000, -4000, np.inf, -np.inf])
print("Original DataFrame:")
print(df)
df = df.replace([np.Inf, -np.Inf], np.nan)
print(df)

# Write a Pandas program to insert a given column at a specific column index in a DataFrame.
d = {'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
df = pd.DataFrame(data=d)
col1 = [1, 2, 3, 4, 7]
str(col1)
idx = 0
df.insert(idx, column = 'col1', value = col1)
df

# Write a Pandas program to convert a given list of lists into a Dataframe.
my_lists = [['col1', 'col2'], [2, 4], [1, 3]]
headers = my_lists.pop(0)
df = pd.DataFrame(my_lists, columns = headers)
print(df)

# Write a Pandas program to group by the first column
# and get second column as lists in rows.
df = pd.DataFrame( {'col1':['C1','C1','C2','C2','C2','C3','C2'], 'col2':[1,2,3,3,4,6,5]})
print("Original DataFrame")
print(df)
df.groupby('col1')['col2'].apply(list)

# Write a Pandas program to get column index from column name of a given DataFrame.
d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
df = pd.DataFrame(data=d)
df.columns.get_loc("col2")

# Write a Pandas program to count number of columns of a DataFrame.
d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
df = pd.DataFrame(data=d)
df.columns.value_counts().sum()
len(df.columns)
df.shape[1]

# Write a Pandas program to select all columns,
# except one given column in a DataFrame.
d = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}
df = pd.DataFrame(data=d)
df.loc[:, df.columns != "col3"]

# Write a Pandas program to get first n records of a DataFrame.
df.head(3)

# Write a Pandas program to get last n records of a DataFrame.
df.tail(3)

# Write a Pandas program to get topmost n records within each group of a DataFrame.
df.nlargest(3, 'col1')
df.nlargest(3, 'col2')
df.nlargest(3, 'col3')

# Write a Pandas program to remove first n rows of a given DataFrame.
df[-2:]

# Write a Pandas program to remove last n rows of a given DataFrame.
df[:-3]

# Write a Pandas program to add a prefix or suffix to all columns of a given DataFrame.
df = pd.DataFrame({'W':[68,75,86,80,66],'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});
print("Original DataFrame")
print(df)
df.add_prefix("A_")

df.add_suffix("_1")

# Write a Pandas program to reverse order (rows, columns) of a given DataFrame.
# Columns
df[df.columns[::-1]]
# Rows
df.loc[::-1]

# Write a Pandas program to select columns by data type of a given DataFrame.
df = pd.DataFrame({
    'name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'date_of_birth': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'age': [18.5, 21.2, 22.5, 22, 23]
})

print("Original DataFrame")
print(df)
print("\nSelect numerical columns")
print(df.select_dtypes(include = "number"))

print("\nSelect string columns")
print(df.select_dtypes(include = "object"))

# Write a Pandas program to split a given DataFrame into two random subsets.
df = pd.DataFrame({
    'name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'date_of_birth': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'age': ['18', '21', '22', '22', '23']
})
df1 = df.sample(frac = 0.6)
print(df1)
df2 = df.drop(df1.index)
print(df2)

# Write a Pandas program to rename all columns with the same pattern of a given DataFrame.
df = pd.DataFrame({
    'Name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'Date_Of_Birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'Age': [18.5, 21.2, 22.5, 22, 23]
})
df.columns = df.columns.str.lower().str.rstrip()
print(df.columns)

# Write a Pandas program to merge datasets and check uniqueness.
df = pd.DataFrame({
    'Name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton'],
    'Date_Of_Birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'Age': [18.5, 21.2, 22.5, 22, 23]
})
print("Original DataFrame:")
print(df)
df1 = df.copy(deep = True)
df = df.drop([0, 1])
df1 = df1.drop([2])
print("\nNew DataFrames:")
print(df)
print(df1)
print('\n"one_to_one”: check if merge keys are unique in both left and right datasets:"')
df_one_to_one = pd.merge(df, df1, validate = "one_to_one")
print(df_one_to_one)
print('\n"one_to_many” or “1:m”: check if merge keys are unique in left dataset:')
df_one_to_many = pd.merge(df, df1, validate = "one_to_many")
print(df_one_to_many)
print('“many_to_one” or “m:1”: check if merge keys are unique in right dataset:')
df_many_to_one = pd.merge(df, df1, validate = "many_to_one")
print(df_many_to_one)

# Write a Pandas program to convert continuous values
# of a column in a given DataFrame to categorical.
df = pd.DataFrame({
    'name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Syed Wharton', 'Kierra Gentry'],
      'age': [18, 22, 85, 50, 80, 5]
})
print("Original DataFrame:")
print(df)
df.dtypes
df["age_groups"] = pd.cut(df["age"], bins = [0,18,65,99], labels = ["kids", "adult", "elderly"])
print(df["age_groups"])

# Write a Pandas program to display memory usage of a given DataFrame
# and every column of the DataFrame.
df.info(memory_usage="deep")
df.memory_usage(deep = True)

# Write a Pandas program to combine many given series to create a DataFrame.
sr1 = pd.Series(['php', 'python', 'java', 'c#', 'c++'])
sr2 = pd.Series([1, 2, 3, 4, 5])
ser_df = pd.DataFrame(sr1, sr2).reset_index()
print(ser_df)
ser_df = pd.concat([sr1, sr2], axis = 1)
print(ser_df)
ser_df = pd.DataFrame({"col1":sr1, "col2":sr2})
print(ser_df.head(5))

# Write a Pandas program to use a local variable within a query.
df = pd.DataFrame({'W':[68,75,86,80,66],'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});
print("Original DataFrame")
print(df)
max_w = df["W"].max()
# can reference variables with the @ symbol
print(df.query("W < @max_w"))

# Write a Pandas program to clean object column
# with mixed data of a given DataFrame using regular expression.
d = {"agent": ["a001", "a002", "a003", "a003", "a004"], "purchase":[4500.00, 7500.00, "$3000.25", "$1250.35", "9000.00"]}
df = pd.DataFrame(d)
df["purchase"].apply(type)
df["purchase"] = df["purchase"].replace("[$,]", "", regex = True).astype("float")
df["purchase"].apply(type)
df

# Write a Pandas program to get the numeric representation of an array
# by identifying distinct values of a given column of a dataframe.
df = pd.DataFrame({
    'Name': ['Alberto Franco','Gino Mcneill','Ryan Parkes', 'Eesha Hinton', 'Gino Mcneill'],
    'Date_Of_Birth ': ['17/05/2002','16/02/1999','25/09/1998','11/05/2002','15/09/1997'],
    'Age': [18.5, 21.2, 22.5, 22, 23]
})
print("Original DataFrame:")
print(df)
label1, unique1 = pd.factorize(df["Name"])
print(label1)
print(unique1)

# Write a Pandas program to replace the current value in a dataframe column based
# on last largest value.
# If the current value is less than last largest value replaces the value with 0.
df1=pd.DataFrame({'rnum':[23, 21, 27, 22, 34, 33, 34, 31, 25, 22, 34, 19, 31, 32, 19]})
print("Original DataFrame:")
print(df1)
df1_max = df1.max()[0]
df1['rnum']=df1.rnum.where(df1.rnum.eq(df1.rnum.cummax()),0)

# Write a Pandas program to check for inequality of two given DataFrames.
df1 = pd.DataFrame({'W':[68,75,86,80,None],'X':[78,85,None,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});
df2 = pd.DataFrame({'W':[78,75,86,80,None],'X':[78,85,96,80,76], 'Y':[84,84,89,83,86],'Z':[86,97,96,72,83]});

df1.ne(df2)

# Write a Pandas program to get lowest n records within each group of a given DataFrame.
d = {'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]}
df = pd.DataFrame(data=d)
df.nsmallest(3, 'col1')

