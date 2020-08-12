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

## Training data

|#ofRegions|#ofOffices|#Min|#Max|#of values (v5)
|:-----:|:-----:|-----:|:-----:|-------:
|4|8|100 000|10 000 000|20 000
<br>

|Region|Office|#Revenue min|Revenue max|#of entries
|:-----|:-----|-----:|:-----:|-------:
10 SYD|11 Malmö|100 000|12 000 000|20 000
10 SYD|12 Göteborg|100 000|20 000 000|20 000
20 ÖST|21 Stockholm|100 000|12 000 000|20 000
20 ÖST|22 Uppsala|100 000|12 000 000|20 000
30 VÄST|31 Karlstad|100 000|15 000 000|20 000
30 VÄST|32 Örebro|100 000|15 000 000|20 000
40 NORR|41 Luleå|100 000|8 000 000|20 000
40 NORR|42 Kiruna|100 000|5 000 000|20 000

Distribution for each region/office is:

    distributions := []float64{0.25, 0.50, 0.75, 1.0}

    
## segemnt_training_v5.csv

|Region|Office|Revenue|Segment
|:-----:|:-----:|-----:|:-----:
|10|11|645000.0|3
|10|12|450000.0|2
|20|21|350000.0|2
|20|22|245000.0|1


|Column|type|s-type
|:-----|:-----|:-----
|Region|Integer|Categorical column
|Office|Integer|Categorical column
|Revenue|float64|Feature column
|Segment|Integer|Target (label)
    