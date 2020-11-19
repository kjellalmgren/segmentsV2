# -*- coding: utf-8 -*-
# predict_preprocessing_layers.ipynb

"""

This tutorial demonstrates how to classify structured data (e.g. tabular data in a CSV). You will use [Keras](https://www.tensorflow.org/guide/keras) to define the model,
and [preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) as a bridge to map from columns in a CSV to features used to train the model.
This tutorial contains complete code to:

* Load a CSV file using [Pandas](https://pandas.pydata.org/).
* Build an input pipeline to batch and shuffle the rows using [tf.data](https://www.tensorflow.org/guide/datasets).
* Map from columns in the CSV to features used to train the model using Keras Preprocessing layers.
* Build, train, and evaluate a model using Keras.

Note: This tutorial is similar to [Classify structured data with feature columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns).
This version uses new experimental Keras [Preprocessing Layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing) instead of `tf.feature_column`.
Keras Preprocessing Layers are more intuitive, and can be easily included inside your model to simplify deployment.

## The Dataset

You will use a simplified version of the PetFinder [dataset](https://www.kaggle.com/c/petfinder-adoption-prediction).
There are several thousand rows in the CSV. Each row describes a pet, and each column describes an attribute. You will use this information to predict if the pet will be adopted.

Following is a description of this dataset. Notice there are both numeric and categorical columns. There is a free text column which you will not use in this tutorial.


Column | Description| Feature Type | Data Type
------------|--------------------|----------------------|-----------------
Region | Region identifier (10, 20, 30, 40) | Numerical | integer
Office | Office identifier (100, 200, 300, 400) | Numerical | integer
Revenue | Customer revenue | Numerical | float64
Segment | belong to segment | Classification | string

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#import numpy as np
#import pandas as pd
import tensorflow as tf

# Define all column in the dataset
#
#CSV_COLUMN_NAMES = ['region', 'office', 'revenue', 'segment']
# Target column to predict
LABELS = ['mini', 'micro', 'mellan', 'stor']

print("Predicition using nvidia 2070 super GPU, 2560 Cuda cores...")

reloaded_model = tf.keras.models.load_model('saved_model/my_segment_classifier')
#loss, evaluate = reloaded_model.evaluate()
#print("Loss, evaluation for model: {}".format(loss, evaluate))

"""To get a prediction for a new sample, you can simply call `model.predict()`. There are just two things you need to do:
1.   Wrap scalars into a list so as to have a batch dimension (models only process batches of data, not single samples)
2.   Call `convert_to_tensor` on each feature
"""
#
predict_x = {
        'region': [10],
        'office': [100],
        'revenue': [1678000.0],
}
predict_x1 = {
        'region': [20],
        'office': [200],
        'revenue': [102000.0],
}
#predict_x = {
#        'region': [10, 10],
#        'office': [100, 100],
#        'revenue': [395000.0, 1750000],
#}

# ------------------------------------------------------------------------------------
print("------------------------------------------------------------------------------")
input_dict = {name: tf.convert_to_tensor([value]) for name, value in predict_x.items()}
print('-------------------------------------')
predictions = reloaded_model.predict(input_dict)
print("------------------------------------------------------------------------------")
print(input_dict)
probabilities = tf.nn.sigmoid(predictions[0])
print("------------------------------------------------------------------------------")
#
def input_fn(features, batch_size=32):
  # Convert the inputs to a Dataset without labels.
  return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
#
print('----------------------------------------')
print(probabilities)
print('----------------------------------------')
i=0
for pred_dict in probabilities:
  probability = pred_dict
  print('Prediction is "{}" ({:.1f}%)'.format(
    LABELS[i], 100 * probability))
  i += 1
#
#print(probabilities)
#predictions = reloaded_model.predict(input_fn(predict_x))
# print("This particular pet had a %.1f percent probability " "of getting adopted." % (100 * prob))

#i=0
#for pred_dict in predictions:
  #print(pred_dict)
  #probability = pred_dict
  #print('Prediction is "{}" ({:.1f}%)'.format(
  #  SPECIES[i], 100 * probability))
  #i += 1
