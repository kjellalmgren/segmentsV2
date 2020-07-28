# #################################################
# train_segment_v7.py (normalized dataset)
# author: Kjell Osse almgren
# date: 2020-07-25
# version: 0.5.0
# #################################################
#
from sklearn import preprocessing
import pandas as pd
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from tensorflow.python.data import Dataset
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from datetime import datetime

logdir = "saved_model/segment_model_v7/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.2
  if epoch > 10:
    learning_rate = 0.02
  if epoch > 20:
    learning_rate = 0.01
  if epoch > 50:
    learning_rate = 0.005

  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate
#
# here we go
#
df = pd.read_csv('datasets/segment_training_v5.csv')
# set revenue as predictor
x = df[df.columns[:3]]
y = df.Segment
print("x: {}", x)
print("y: {}", y)
x_train, x_test, y_train, y_test = train_test_split(x, y,
                              train_size=0.75,
                              random_state=90)
#
# Select numerical columns which needs to be normalized
#
train_norm = x_train[x_train.columns[0:3]]
test_norm = x_test[x_test.columns[0:3]]
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
print("- x_train -----------")
print(x_train.head())
print("---------------------")
#
# Normalize Testing Data by using mean and SD of training set and update x_test
#
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns) 
x_test.update(testing_norm_col)
print("- x_test ------------")
print(x_test_norm)
print("- x_train_updated ---")
print(x_train.head())
print("---------------------")
#
#Build neural network model with normalized data
#
model = tensorflow.keras.Sequential([
 tensorflow.keras.layers.Dense(64,
                            activation=tf.nn.relu,                  
                            input_shape=(x_train.shape[1],)),
                            tensorflow.keras.layers.Dense(64,
                            activation=tf.nn.relu),
                            tensorflow.keras.layers.Dense(8,
                            activation='softmax')
 ])
#
# lr_callback, learning rate, not used in thsi project
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
# tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="saved_model/segment_model_v7/tensorboard",
                                                    write_graph=True,
                                                    embeddings_freq=5,
                                                    histogram_freq=5,
                                                    embeddings_layer_names=None,
                                                    embeddings_metadata=None)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="saved_model/segment_model_v7/weights",
                                                 save_weights_only=True,
                                                 verbose=1)
#
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#
history2 = model.fit(
                x_train, # input
                y_train, # output
                epochs=26,
                batch_size=60,
                verbose=1,  # Supress chatty output; use Tensorboard instead
                validation_data=(x_test, y_test),
                callbacks=[tensorboard_callback, cp_callback]) # comment out lr_callback
#
print(history2.history)
loss, acc = model.evaluate(x_train, y_train, 
                            batch_size=60,
                            verbose=2,
                            callbacks=[tensorboard_callback, cp_callback])
#
print("loss: {}", loss)
print("accuracy: {:5.2f}%".format(100*acc))
#
# saved_format=[tf.h5]
tf.saved_model.SaveOptions(save_debug_info=False, namespace_whitelist=None, function_aliases=None)
model.save("saved_model/segment_model_v7",
                    save_format=tf,
                    overwrite=True,
                    include_optimizer=True)
model.summary()