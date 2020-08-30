from __future__ import print_function
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")

print("Cuda cores availablity: ", torch.cuda.is_available())
print(device)
x = torch.rand(5, 3)
print(x)

#Read the data from csv file
df = pd.read_csv('datasets/segment_training_v5.csv')
print(df.shape)
print(df.describe())
# Select predictors
#x = df[df.columns[:3]]
#Target variable 
# y = df.Segment
# print("x: {}", x)
# print("y: {}", y)
target_column = ['Segment']
predictors = list(set(list(df.columns))-set(target_column))

print("label column: ", target_column)
print("prediction columns: ", predictors)

x = df[predictors].values
y = df[target_column].values
#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
print(x_train.shape); print(x_test.shape)