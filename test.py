import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import datetime

print("Tensorflow version: {}".format(tf.version.VERSION))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

tf.executing_eagerly()

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
    start_cpu = datetime.datetime.now()
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    end_cpu = datetime.datetime.now()
    total_cpu_time = end_cpu - start_cpu
    print(c)

with tf.device('/GPU:0'):
    start_gpu = datetime.datetime.now()
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    end_gpu = datetime.datetime.now()
    total_gpu_time = end_gpu -start_gpu
    print(c)
#
print("------")
print("total CPU-time: ", total_cpu_time)
print("total GPU-time: ", total_gpu_time)

