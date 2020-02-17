# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:25:33 2020

@author: Tobias
"""

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Add
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from glob import glob

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import cv2

import pydot
import os



#TRAINING DATA
pickle_in = open("C:/loc/to/your/dataset/training_set_X.pickle", "rb")
train_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("C:/loc/to/your/dataset/training_set_y.pickle", "rb")
train_y = pickle.load(pickle_in)

#TEST DATA
pickle_in = open("C:/loc/to/your/dataset/test_set_X.pickle", "rb")
test_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("C:/loc/to/your/dataset/Boats/test_set_y.pickle", "rb")
test_y = pickle.load(pickle_in)

#Normalize data -- scale the data
train_x = train_x/255.0
test_x = test_x/255.0
train_y = np.array(train_y)
test_y = np.array(test_y)


def conv_block(X, filters, kernel_size, strides, lname):
        X = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding = 'same', name=lname)(X)
        X = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001)(X)
        X = Activation("relu")(X)
        X = pooling_block(X, 2, 2)
        return X

def res_block_A(X, filters):
    shortcut_X = X
    shortcut_X = MaxPooling2D(2,2,padding="valid")(shortcut_X)
    L0 = conv_block(X, 32, 5, 1, "conv_5")
    L1 = conv_block(X, 32, 3, 1, "conv_3")
    L2 = conv_block(X, 64, 1, 1, "conv_1")
    X = tf.keras.layers.concatenate([L0,L1,L2], axis=3)
    #X = conv_block(X, 128, 3, 1)
    X = Add()([X, shortcut_X])
    X = Activation("relu")(X)
    return X

def pooling_block(X, pool_size, strides):
    X = MaxPooling2D(pool_size = pool_size, strides = strides, padding = "valid")(X)
    return X

def Residual_Model():
    input_dims=(128,128,3)
    inputs = tf.keras.Input(shape=(128,128,3))
    X = Conv2D(128, kernel_size=5, strides=3, padding = 'valid', input_shape=(input_dims), name='conv_0')(inputs)
    #X = BatchNormalization(axis=-1, momentum=0.87, epsilon=0.001)(X)
    X = Activation("relu")(X)
    X = pooling_block(X, 3, 3)
    X = res_block_A(X, 128)
    X = conv_block(X, 64, 3, 3, "conv_final")
    #X = pooling_block(X, 3, 3)
    X = Flatten()(X)
    X = Dense(64)(X)
    X = Dropout(0.4)(X)
    outputs = Dense(5)(X)
    outputs = Activation("softmax")(X)
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name='Simple_Residual_Model')
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

res_model = Residual_Model()
#res_model.summary()
res_model.save("C:/save/model/at/this/location/" + res_model.name + ".hdf5")
#DISPLAY MODEL STRUCTURE
tf.keras.utils.plot_model(res_model, 'my_first_model_with_shape_info.png', show_shapes=True)
#res_model.metrics_names
#PREPARE TRAINING
#checkpoint_loss = ModelCheckpoint("C:/location/to/save/"+ res_model.name + "_{epoch:02d}-{loss:.2f}_best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
#checkpoint_val_loss = ModelCheckpoint("C:/location/to/save/"+ res_model.name + "_{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
#TRAIN THE MODEL AND STORE THE RESULTS
#history = res_model.fit(train_x, train_y, batch_size=16, validation_split=0.10, epochs = 10, callbacks=[checkpoint_loss, checkpoint_val_loss, reduce_lr])

#USE FOR LATER
from model_training_script import train_model
model = train_model('C:/location/of/your/model/' + res_model.name + '.hdf5', train_x, train_y)

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
res_model.load_weights('C:/location/of/your/model/model.hdf5')
res_model.evaluate(test_x, test_y, batch_size=32, verbose=2)

#VALIDATION OF RESULTS
#Heatmap
from grad_cam import grad_cam
IMAGE = 'C:/location/of/your/image/image.jpg'
layer_name = "conv_final"
alpha=1
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

def displayImage(img):
    IMG = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    plt.imshow(IMG)
    plt.show()
    
displayImage(IMAGE)
