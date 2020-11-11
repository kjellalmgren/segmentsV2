from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import datetime

# Define all column in the dataset
#
CSV_COLUMN_NAMES = ['Region', 'Office', 'Revenue', 'Segment']
# Target column to predict
LABELS = ['mini', 'micro', 'mellan', 'stor']

#
# train the model
#
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    # Inner function, this will be returned
    def input_function():
        # create tf.data.Dataset object with data and its labrl
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(10) # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs) # split data set into batches of 32 and repeat numer of epochs
        return ds # return batch of the dataset
    return input_function # return a function object of use
#
#
print("Tensorflow version: {}".format(tf.version.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

def input_fn1(features, labels, training=True, batch_size=32):
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        #tf.keras.backend.set_floatx('float64')
        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(10000).repeat()
        return dataset.batch(batch_size)
#
# with tf.device("/device:GPU:0"):
with tf.device("/device:GPU:0"):
    print("Using nvidia 2070 super, 2560 Cuda GPU cores")
    train_path = tf.keras.utils.get_file(
        "segment_training_v4.csv", "http://localhost:8443/segment_training_v4")
    test_path = tf.keras.utils.get_file(
        "segment_evaluation_v4.csv", "http://localhost:8443/segment_evaluation_v4")
    #dftrain = pd.read_csv('http://localhost:8443/segment_training_v4')
    #dfeval = pd.read_csv('http://localhost:8443/segment_evaluation_v4')

    dftrain = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    dfeval = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    print("-- dftrain.dtypes-------------------------------------------------------")
    print(dftrain.dtypes)
    print("------------------------------------------------------------------------")
    #
    print("--dftrain.head() -------------------------------------------------------")
    print(dftrain.head())
    print("--dfeval.head() --------------------------------------------------------")
    print(dfeval.head())
    #
    # SEGMENTS = dftrain["Segment"].unique()
    #
    # Drop target column that should be used for prediction
    y_train = dftrain.pop('Segment')
    y_eval = dfeval.pop('Segment')
    #print(SEGMENTS)
    print("-- dftrain.head() ------------------------------------------------------")
    print(dftrain.head())
    print("-- dfeval.head() -------------------------------------------------------")
    print(dfeval.head())
    print("-y_train format --------------------------------------------------------")
    print("{}".format(y_train))
    print("------------------------------------------------------------------------")
    #
    #CATEGORICAL_COLUMNS = ['Region', 'Office']
    #NUMERIC_COLUMNS = ['Revenue']
    #
    #feature_columns = []
    # gets a list of all unique values from feature columns
    #for feature_name in CATEGORICAL_COLUMNS:
    #    vocabulary = dftrain[feature_name].unique()
    #    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    #
    #for feature_name in NUMERIC_COLUMNS:
    #    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    #
    # Feature columns describe how to use the input.
    my_feature_columns = []
    # ================ Comment out =======================
    for key in dftrain.keys():
        #if key in 'Revenue': {
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        #}
    # ================ Comment out =======================
    print("-my_feature_columns ------------------------------------------------------")
    print(my_feature_columns)
    print("------------------------------------------------------------------------")
    #
    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 30 and 10 nodes respectively.
        hidden_units=[300, 50],
        optimizer='Adagrad',
        activation_fn=tf.nn.relu,
        dropout=None,
        # The model must choose between 4 classes. (0-3)
        n_classes=4,
        model_dir="saved_model/segment_model")

    train_result = classifier.train(
        input_fn=lambda: input_fn1(dftrain, y_train, training=True), 
        steps=300000)
    #   
    # results = train_result.get_variable_names()
    #for result in train_result.get_variable_names():
    #  print(train_result.get_variable_value(result)
    #
    # We include a lambda to avoid creating an inner function previously
    #
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn1(dfeval, y_eval, training=False))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
