from tensorflow.keras import models
from tensorflow.keras import layers
from datetime import datetime
import tensorflow as tf
import numpy as np

expected = ['mini', 'micro', 'mellan', 'stor']
class_names = ['mini', 'micro', 'mellan', 'stor']

#
# Load model
#
model = models.load_model("saved_model/segment_model_v6")
#
#
#
predict_x = {
       'Region': [10.0, 10.0, 10.0, 10.0, 10.0],
       'Office': [100.0, 100.0, 100.0, 100.0, 100.0],
       'Revenue': [2195.0, 3500.0, 5210.0, 8948.0, 41114.0],
}

np.testing.assert_allclose(model.predict(predict_x), model.predict(predict_x))
#predictions = model.predict(predict_x)
#print(predictions)

model.summary()