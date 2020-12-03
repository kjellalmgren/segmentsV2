
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import torch
import torchvision
import tensorflow as tf
import tensorflow_estimator as te
import tensorflow.keras as k
import pandas as pd

print("Tensorflow version: {}".format(tf.__version__))
print("Tensorflow estimator file: {}".format(te.__file__))
print("Tensorflow keras: {}".format(tf.keras.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("Pandas version: {}".format(pd.__version__))
print("Python version: {}".format(sys.version))
print("Torch: {}".format(torch.__version__))
print("Torchvision: {}".format(torchvision.__version__))
print("----------")
print(tf.config.list_physical_devices())
print("**********")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("==================")
print("GPU available: ", tf.test.is_gpu_available())
print("==================")
#print("Tensorflow build with cuda: ", tf.test.is_built_with_cuda())
print("==================")
#print(tf.device("/device:GPU:0"))

with tf.device("/device:CPU:0"):
    print("Using CPU")
with tf.device("/device:GPU:0"):
    print("Using GPU:0")
with tf.device("/device:GPU:1"):
    print("Using GPU:1")
if tf.test.is_built_with_cuda() == True:
    print("Tensorflow is build Using CUDA")
    print("Nvidia CUDA support is present")
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)