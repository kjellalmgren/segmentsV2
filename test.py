import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

print("Tensorflow version: {}".format(tf.version.VERSION))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

if tf.test.is_built_with_gpu_support(): 
    print("GPU support is present")
    if tf.test.is_built_with_cuda(): 
        print("Nvidia CUDA support is present")
    if tf.test.is_built_with_rocm(): 
        print("ROCM AMD support is present")

tf.debugging.set_log_device_placement(True)
# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)

with tf.device('/CPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)

#

