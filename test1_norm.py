from sklearn import preprocessing
import tensorflow as tf
import pandas as pd
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.python.data import Dataset
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

df = pd.read_csv('datasets/segment_training_v5.csv')
# set revenue as predictor
x = df[df.columns[:3]]
y = df.Segment
print("x: {}", x)
print("y: {}", y)
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
#Select numerical columns which needs to be normalized
train_norm = x_train[x_train.columns[0:3]]
test_norm = x_test[x_test.columns[0:3]]
# Normalize Training Data 
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
#Converting numpy array to dataframe
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns) 
x_train.update(training_norm_col)
print (x_train.head())
# Normalize Testing Data by using mean and SD of training set
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
print (x_train_norm)

#Build neural network model with normalized data
model = tensorflow.keras.Sequential([
 tensorflow.keras.layers.Dense(64, activation=tf.nn.relu,                  
 input_shape=(x_train_norm.shape[1],)),
 tensorflow.keras.layers.Dense(64, activation=tf.nn.relu),
 tensorflow.keras.layers.Dense(8, activation='softmax')
 ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history2 = model.fit(
 x_train_norm, y_train,
 epochs=26, batch_size = 60,
 validation_data = (x_test_norm, y_test))