import tensorflow as tf
import tensorflow_estimator as te

print(tf.__version__)
print(te.__file__)
print("Eager execution: {}".format(tf.executing_eagerly()))
