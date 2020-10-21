import tensorflow as tf
import tensorflow_estimator as te

print(tf.__version__)
print(te.__file__)
print("Eager execution: {}".format(tf.executing_eagerly()))
print("----------")
print(tf.config.list_physical_devices())
print("**********")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("==================")
print("GPU available: ", tf.test.is_gpu_available())
print("==================")
#print("Tensorflow build with cuda: ", tf.test.is_built_with_cuda())
print("==================")
#print(tf.device("/device:GPU:0"))

with tf.device("/device:CPU:0"):
    print("Using CPU")
with tf.device("/device:GPU:0"):
    print("Using GPU")
if tf.test.is_built_with_cuda() == True:
    print("Using CUDA")