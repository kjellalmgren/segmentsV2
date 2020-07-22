#'''Using covertype dataset from kaggle to predict forest cover type'''
#Import pandas, tensorflow and keras
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data import Dataset
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

#Read the data from csv file
df = pd.read_csv('covtype.csv')
#Select predictors
x = df[df.columns[:54]]
#Target variable 
y = df.Cover_Type
#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
#'''As y variable is multi class categorical variable, hence using softmax as activation function and sparse-categorical cross entropy as loss function.'''
model = tensorflow.keras.Sequential([
 tensorflow.keras.layers.Dense(64, activation=tf.nn.relu,                  
 input_shape=(x_train.shape[1],)),
 tensorflow.keras.layers.Dense(64, activation=tf.nn.relu),
 tensorflow.keras.layers.Dense(8, activation=  'softmax')
 ])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history1 = model.fit(
 x_train, y_train,
 epochs= 26, batch_size = 60,
 validation_data = (x_test, y_test))