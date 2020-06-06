from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import datetime

# Lets define some constants to help us later on
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
#
print("Tensorflow version: {}".format(tf.version.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))
#
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

def input_fn1(features, labels, training=True, batch_size=256):
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()
        return dataset.batch(batch_size)
#
with tf.device("/device:gpu:0"):
    print("Entering GPU-device for computation...")
    #
    # Get training and evaluation dataset from tensorlake at localhost, if not available or
    # cached it will be found under path /.keras/datasets/
    #
    # Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
    #
    # force to handle float64
    # tf.keras.backend.set_floatx('float64')
    train_path = tf.keras.utils.get_file(
        "iris_training_v1.csv", "http://localhost:8443/iris_training_v1")
    test_path = tf.keras.utils.get_file(
        "iris_evaluation_v1.csv", "http://localhost:8443/iris_evaluation_v1")
    #
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    #
    print("-- train.dtypes --------------------------------------------------------")
    print(train.dtypes)
    print("-- train.head() --------------------------------------------------------")
    print(train.head()) # the species column is now present
    train_y = train.pop('Species')
    test_y = test.pop('Species')
    print("-- train.head() --------------------------------------------------------")
    print(train.head()) # the species column is now gone
    print("-- labels test_y.head() ------------------------------------------------")
    print(test_y.head())
    print("--Removed labels -------------------------------------------------------")
    #
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    print("-- features_columnes ---------------------------------------------------")
    print(my_feature_columns)
    #
    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 30 and 10 nodes respectively.
        hidden_units=[32, 32],
        # The model must choose between 3 classes.
        n_classes=3,
        model_dir="saved_model/iris_model")

    train_result = classifier.train(
        input_fn=lambda: input_fn1(train, train_y, training=True), 
        steps=10000)
    #   
    # results = train_result.get_variable_names()
    #for result in train_result.get_variable_names():
    #  print(train_result.get_variable_value(result)
    #
    # We include a lambda to avoid creating an inner function previously
    #
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn1(test, test_y, training=False))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))