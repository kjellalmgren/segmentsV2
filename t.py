# find out if we have duplicate versions of tensorboard
# Tensorboard install with tf-nightly-gpu (guthub.com/tensorflow/tensorflow)
#
import pkg_resources

for entry_point in pkg_resources.iter_entry_points('tensorboard_plugins'):
    print(entry_point.dist)