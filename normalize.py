import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

columns = ['indicies', 'Region', 'Office', 'Revenue', 'Region_normalized', 'Office_normalized', 'Revenue_normalized']
# columns = ['indicies', 'Region_normalized', 'Office_normalized', 'Revenue_normalized']
df = pd.read_csv('datasets/segment_training_v5.csv')

# add new column to the dataframe
#
df.insert(3, "Region_normalized", 0.0)
df.insert(4, "Office_normalized", 0.0)
df.insert(5, "Revenue_normalized", 0.0)
#
#print(df.columns)
#print(type('Revenue_normalized'))
#
# copy Revenue to Revenue_normalized, Office to Office_normalized and Region to Region_normalized
#
print("Creating column to be normalized")
for i in df.index:
    df.at[i, 'Region_normalized'] = df.at[i, 'Region']
    df.at[i, 'Office_normalized'] = df.at[i, 'Office']
    df.at[i, 'Revenue_normalized'] = df.at[i, 'Revenue']
#
# Let x represent the seven first column
#
x = df[df.columns[:6]]
print(x.head(5))
#
# y is the eight's column, label for each row in the dataset (segment)
#
y = df.Segment
print("y: ", y)
#
# Print first and last five rows in the dataset to be able to se the new columns
#
#print(x.head(5))
#
# Split data into train and test
#
x_train, x_test, y_train, y_test = train_test_split(
                                    x, y,
                                    train_size=0.7,
                                    random_state=90)
print(x_train.head(5))
#
# Select numerical columns which needs to be normalized
#
train_norm = x_train[x_train.columns[3:]]
test_norm = x_test[x_test.columns[3:]]
#
# Normalize Training Data 
#
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
#
# Converting numpy array to dataframe and update x_train
#
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)
#
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
#
# Show result
#
print(x_train.head(5))
print(x_test.head(5))