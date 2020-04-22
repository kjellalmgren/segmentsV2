from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

import datetime

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

with tf.device("/device:cpu:0"):

    expected = ['Setosa', 'Versicolor', 'Virginica']
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    

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
        hidden_units=[32, 32],
        # The model must choose between 3 classes.
        n_classes=3,
        model_dir="saved_model/iris_model")
    #

    # ###################################################################
    def input_fn(features, batch_size=256):
        # Convert the inputs to a Dataset without labels.
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    class_names = ['Setosa', 'Versicolor', 'Virginica']
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
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    #     
    predictions = classifier.predict(input_fn=lambda: input_fn(predict_x))
    # predictions = classifier.predict(predict_x3)
    print("--Prediction -----------------------")
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability))

#
#
