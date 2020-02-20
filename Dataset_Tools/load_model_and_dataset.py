# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:42:39 2020

@author: Tobias
"""

import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
if (not tf.executing_eagerly()): 
    tf.compat.v1.enable_eager_execution(
        config=None,
        device_policy=None,
        execution_mode=None
    )
#from tensorflow.keras.callbacks import
import numpy as np
import pickle

# LOAD train_x, train_y

#TRAINING DATA
pickle_in = open("training_set_X.pickle", "rb")
train_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("training_set_y.pickle", "rb")
train_y = pickle.load(pickle_in)

#TEST DATA
pickle_in = open("test_set_X.pickle", "rb")
test_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("test_set_y.pickle", "rb")
test_y = pickle.load(pickle_in)

#Normalize data -- scale the data
train_x = train_x/255.0
test_x = test_x/255.0

train_y = np.array(train_y)
test_y = np.array(test_y)

from model_training_script import train_model
train_model('SIMPLE_CNN_1/my_test_model.hdf5', train_x, train_y)
