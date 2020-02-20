# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:24:17 2020

@author: Tobias
"""

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Add
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from glob import glob

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import cv2

#TRAINING DATA
pickle_in = open("C:/location/training_set_X.pickle", "rb")
train_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("C:/location/training_set_y.pickle", "rb")
train_y = pickle.load(pickle_in)

#TEST DATA
pickle_in = open("C:/location/test_set_X.pickle", "rb")
test_x = pickle.load(pickle_in)
#Answers to test data
pickle_in = open("C:/location/test_set_y.pickle", "rb")
test_y = pickle.load(pickle_in)

#Normalize data -- scale the data
train_x = train_x/255.0
test_x = test_x/255.0
train_y = np.array(train_y)
test_y = np.array(test_y)

def conv_block(X, filters, kernel_size, strides):
    shortcut = X
    if strides == 1:
        while (kernel_size > 1): # replacing the kernel_size with 3x3 kernels to cover the same area while saving in computational cost
            X = Conv2D(filters, kernel_size=3, strides=1, padding = 'same')(X)
            kernel_size = ((kernel_size - 3)/1) + 1
    else:
        X = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding = 'same')(X)
    #shortcut = AveragePooling2D(pool_size=kernel_size, strides = 1)
    X = Add()([shortcut, X])
    X = common_layers(X)
    return X
    
def common_layers(X):
    X = BatchNormalization()(X)
    X = tf.keras.layers.LeakyReLU()(X)
    return X
    
def conv_layer(X, filters, kernel_size, strides):
    shortcut = X
    if strides == 1:
        while (kernel_size > 1): # replacing the kernel_size with 3x3 kernels to cover the same area while saving in computational cost
            X = Conv2D(filters, kernel_size=3, strides=1, padding = 'same')(X)
            kernel_size = ((kernel_size - 3)/1) + 1
    else:
        X = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding = 'same')(X)
    X = Add()([shortcut, X])
    return X

def maxPool_layer(X, pool_size, strides):
    X = MaxPooling2D(pool_size = pool_size, strides = strides, padding = "valid")(X)
    return X

def inception_block(X, filters):
    #split_1 = int(filters/4)
    split_2 = int(filters/2)
    l1 = Conv2D(split_2, 1, 1, padding='same')(X)
    l1 = conv_layer(l1, split_2, 5, 1)
    l2 = Conv2D(split_2, 1, 1, padding='same')(X)
    l2 = conv_layer(l2, split_2, 3, 1)
    X = tf.keras.layers.concatenate([l1,l2])
    return X

def residual_block(X, filters):
    shortcut = X
    X = conv_block(X, filters, 5, 1)
    X = Add()([shortcut, X])
    X = common_layers(X)
    return X

def res_model():
    inputs = tf.keras.Input(shape=(128,128,3))
    X = Conv2D(24, 1, 1, padding='same')(inputs)
    X = maxPool_layer(X, 3, 3)
    X = Dropout(0.2, name="dout_initial")(X)
    #X = residual_block(X, 24)
    X = inception_block(X, 24)
    X = maxPool_layer(X, 3, 3)
    X = Dropout(0.2, name="dout_final")(X)
    X = common_layers(X)
    X = Flatten()(X)
    X = Dense(5, name="dense_out")(X)
    outputs = Activation("softmax", name="softmax")(X)
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name ="inception_resnet_simple")
    model.compile(loss="sparse_categorical_crossentropy", optimizaer="adam", metrics=['accuracy'])
    return model

res_model = res_model()
#SAVE THE MODEL
res_model.save("C:/location/" + res_model.name + ".hdf5")
#DISPLAY MODEL STRUCTURE
tf.keras.utils.plot_model(res_model, 'my_first_model_with_shape_info.png', show_shapes=True)

#PREPARE FOR TRAINING
#checkpoint_loss = ModelCheckpoint('C:/location/' + res_model.name + '_{epoch:02d}-{loss:.2f}_best_model.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
#checkpoint_val_loss = ModelCheckpoint('C:/location/' + res_model.name + '_{epoch:02d}-{loss:.2f}_best_val_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
#TRAIN THE MODEL AND STORE THE RESULTS
#history = res_model.fit(train_x, train_y, batch_size=16, validation_split=0.10, epochs = 10, callbacks=[checkpoint_loss, checkpoint_val_loss])

#TRAIN THE MODEL USING SCRIPT
from model_training_script import train_model
history = train_model('C:/location/' + res_model.name + '.hdf5', train_x, train_y, test_x, test_y)

#SHOW TRAINING RESULTS
def show(history):
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    acc = history['accuracy']
    loss = history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, loss, 'c', label='Training loss')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.legend()
    plt.ylim(top=1.5, bottom=0.0)
    plt.figure()
    plt.show()

show(history.history)

#MODEL EVALUATION
res_model.load_weights('C:/location/inception_resnet_simple_10-0.13_best_val_model.hdf5')
res_model.evaluate(test_x, test_y, batch_size=16, verbose=2)

#VALIDATION OF RESULTS
#Heatmap
#res_model.summary() #- run this to see layer names
from grad_cam import grad_cam
IMAGE = 'C:/location/img.jpg'

layer_name = "conv2d_5"
alpha=0.9
grad_cam(res_model, layer_name, IMAGE, alpha)
#Print with alpha 0.5 and alpha 1.0
def printHMvariations(layer, img):
    IMG = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    alpha=1.0
    for i in range(2):
        grad_cam(res_model, layer, img, alpha)
        alpha -= 0.5
    IMG = cv2.imread(IMAGE, cv2.IMREAD_UNCHANGED)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    plt.imshow(IMG)
    plt.show()

printHMvariations(layer_name, IMAGE)

#Used for displaying a single image
def displayImage(img):
    IMG = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    plt.imshow(IMG)
    plt.show()
    
displayImage(IMAGE)
