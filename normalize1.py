import pandas as pd
from pandas import read_csv
from sklearn import preprocessing

# from sklearn.model_selection import train_test_split
# import tensorflow as tf

# data = [10.0, 11.0, 656000.0]
columns = ['region', 'office', 'revenue', 'segment', 'revenue_normalized']

series = read_csv('datasets/segment_training_v5.csv', header=0, index_col=0)

#series = series.pop('Segment')
values = series.values
print(values)
print(series.head())
values = values.reshape((len(values), 3))
print(type(values))
scaler = preprocessing.StandardScaler()
scaler = scaler.fit(values)
normalized = scaler.transform(values)
print(type(normalized))
# np.sort(normalized,axis=0, kind='quicksort')
for i in range(len(normalized)):
    print("X: ", i,normalized[i])
# print(normalized)
inverse = scaler.inverse_transform(normalized)
print(type(inverse))
#
for i in range(len(inverse)):
    #print("{0:6.1f}".format(inverse[i]))
    print("Y: {} - {}".format(i, inverse[i]))
    #print("Y: {:3.0f} - {:6.2f}".format(i, inverse[i]))
#
print("Antal: ", len(values))
print("Inverse: ", len(inverse))
print("Normalized: ", len(normalized))
#print(inverse)
#training_norm_col = pd.DataFrame(x) 
#x.update(training_norm_col)
#print("Values: {0:3.1f} - {1:3.1f} - {2:10.1f} - {3:10.1f}".format(inverse[0], inverse[1], inverse[2], inverse[3]))
#print("Values: {0:d}".format(inverse[0]))

#
# Converting numpy array to dataframe and update x_train
#
# x_train_norm = std_scale.transform(data)
# training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
# x_train.update(training_norm_col)