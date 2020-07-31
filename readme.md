# readme

    https://machinelearningmastery.com/clustering-algorithms-with-python/
    https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029#:~:text=The%20goal%20of%20normalization%20is,when%20features%20have%20different%20ranges.

    https://www.tensorflow.org/tutorials/keras/save_and_load

## Matplotlib for python

    $ which matplotlib
    $ sudo apt-get install python3-matplotlib

## Uninstall tensorboard

    $ pip3 show tensordboard
    $ pip3 uninstall tensorboard

## Tensorboard

    $ sudo pip3 install tensorboard
    # show tensorboard, which tensorboard 
    # /home/kjell
    $ mkdir logdir
    # startup tensorboard on localhost:6006
    # NOT WORKING ***WORK-OUT WHY ***
    $ tensorboard --logdir /logdir
    $ tensorboard --logdir /Documents/tensorrep/src/segmentsV2/saved_model/segment_model_v6
    # working
    $ tensorboard --logdir .

## Profile plugin for tensorboard

    $ sudo pip list
    $ sudo pip install -U tensorboard-plugin-profile

## Tensorflow uninstall

    # Check for both pip and pip3 (probably wrong to install thru pip, should by pip3)
    # Also check Home/.local/lib/python3.8/site-packages
    $ pip show tensorflow
    $ pip uninstall tensorflow

## Tensorflow upgrade

    $ sudo pip show tensorflow
    $ sudo pip install --user --upgrade tensorflow
    $ sudo pip install --user tensorflow.....whl
    