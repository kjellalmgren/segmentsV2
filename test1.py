#'''Using covertype dataset from kaggle to predict forest cover type'''
#Import pandas, tensorflow and keras
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.data import Dataset
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers


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
#Read the data from csv file
df = pd.read_csv('datasets/segment_training_v5.csv')
#Select predictors
x = df[df.columns[:3]]
#Target variable 
y = df.segment
print("x: {}", x)
print("y: {}", y)
#Split data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
#'''As y variable is multi class categorical variable, hence using softmax as activation function and sparse-categorical cross entropy as loss function.'''
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
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="saved_model/segment_model_v61/tensorboard",
                                                    write_graph=True,
                                                    embeddings_freq=5,
                                                    histogram_freq=5,
                                                    embeddings_layer_names=None,
                                                    embeddings_metadata=None)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="saved_model/segment_model_v61/weights",
                                                 save_weights_only=True,
                                                 verbose=1)
#
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#
history1 = model.fit(
                x_train, # input
                y_train, # output
                epochs=26,
                batch_size=60,
                verbose=1,  # Supress chatty output; use Tensorboard instead
                validation_data=(x_test, y_test),
                callbacks=[tensorboard_callback, cp_callback]) # comment out lr_callback
#
print(history1.history)
loss, acc = model.evaluate(x_train, y_train, 
                            batch_size=60,
                            verbose=2,
                            callbacks=[tensorboard_callback, cp_callback])
#
print("loss: {}", loss)
print("accuracy: {:5.2f}%".format(100*acc))
#
#model.evaluate(x_train, y_train)
# saved_format=[tf.h5]
tf.saved_model.SaveOptions(save_debug_info=False, namespace_whitelist=None, function_aliases=None)
model.save("saved_model/segment_model_v61",
                    save_format=tf,
                    overwrite=True,
                    include_optimizer=True)
model.summary()