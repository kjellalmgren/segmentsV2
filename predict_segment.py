from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

import datetime

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
COLUMN_NAMES = ['Region', 'Office', 'Revenue']
SPECIES = ['mini', 'micro', 'mellan', 'stor']

with tf.device("/device:cpu:0"):

    expected = ['mini', 'micro', 'mellan', 'stor']
    class_names = ['mini', 'micro', 'mellan', 'stor']

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in COLUMN_NAMES:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    print(my_feature_columns)

    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    # This is just to load saved model during training
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 30 and 10 nodes respectively.
        hidden_units=[100, 50],
        optimizer='Adagrad',
        activation_fn=tf.nn.relu,
        dropout=None,
        # The model must choose between 3 classes.
        n_classes=4,
        model_dir="saved_model/segment_model")
    #

    # ###################################################################
    def input_fn(features, batch_size=256):
        # Convert the inputs to a Dataset without labels.
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    features = ['Region', 'Office', 'Revenue']
    class_names = ['mini', 'micro', 'mellan', 'stor']
    #predict = {}

    # Manuel console input
    #print("Please type numeric values as prompted.")
    #for feature in features:
    #    valid = True
    #    while valid: 
    #        val = input(feature + ": ")
    #        if not val.isdigit(): valid = False
    #    predict[feature] = [float(val)]

    predict_x = {
        'Region': [10.0, 10.0, 10.0, 10.0, 10.0],
        'Office': [100.0, 100.0, 100.0, 100.0, 100.0],
        'Revenue': [1005.0, 3500.0, 5210.0, 8948.0, 1114.0],
    }

    #     
    predictions = classifier.predict(input_fn=lambda: input_fn(predict_x))
    print("-- Prediction segment ---------------------")
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability))
#