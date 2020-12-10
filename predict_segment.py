from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import datetime

print("Tensorflow version: {}".format(tf.version.VERSION))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
COLUMN_NAMES = ['region', 'office', 'revenue']
SPECIES = ['mini', 'micro', 'mellan', 'stor']

with tf.device("/device:gpu:0"):

    expected = ['mini', 'micro', 'mellan', 'stor']
    class_names = ['mini', 'micro', 'mellan', 'stor']

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in COLUMN_NAMES:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    #
    print(my_feature_columns)
    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    # This is just to load saved model during training
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 30 and 10 nodes respectively.
        hidden_units=[300, 50],
        optimizer='Adagrad',
        activation_fn=tf.nn.relu,
        dropout=None,
        # The model must choose between 4 classes. 0-3
        n_classes=4,
        model_dir="saved_model/segment_model")
    #
    # ###################################################################
    def input_fn(features, batch_size=32):
        # Convert the inputs to a Dataset without labels.
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    features = ['region', 'office', 'revenue']
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
    #
    predict_x = {
        'region': [10, 10, 10, 10, 10, 10],
        'office': [100, 100, 100, 100, 100, 100],
        'revenue': [291950.0, 705000.0, 1450010.0, 1980948.0, 410114.0, 1999999.0],
    }
    #     
    predictions = classifier.predict(input_fn=lambda: input_fn(predict_x))
    print("-- Prediction segment ---------------------")
    for pred_dict in predictions:
        # print(pred_dict)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        #region = pred_dict['region'][class_id]
        #office = pred_dict['office'][class_id]
        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability))
#