# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:24:17 2020

@author: Tobias
"""

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, Add
import tensorflow as tf

def common_layers(X):
    X = BatchNormalization()(X)
    X = tf.keras.layers.LeakyReLU()(X)
    return X

def maxPool_layer(X, pool_size, strides):
    X = MaxPooling2D(pool_size = pool_size, strides = strides, padding = "valid")(X)
    return X

def inception_residual_block(X, filters, sc = False):
    l1 = Conv2D(32, (1,1), 1, padding = 'same')(X)
    l1 = common_layers(l1)
    
    l2 = Conv2D(16, (1,1), 1, padding = 'same')(X)
    l2 = common_layers(l2)
    l2 = Conv2D(32, (5,5), 1, padding = 'same')(l2)
    l2 = common_layers(l2)

    l3 = Conv2D(32, (1,1), 1, padding = 'same')(X)
    l3 = common_layers(l3)
    l3 = Conv2D(64, (3,3), 1, padding = 'same')(l3)
    l3 = common_layers(l3)
    
    X = tf.keras.layers.concatenate([l1,l2,l3])
    return X

def init_model(in_dim, classes):
    inputs = tf.keras.Input(shape=in_dim)
    X = inception_residual_block(inputs, 128)
    X = maxPool_layer(X, 3, 3)
    X = Conv2D(128,(3,3),1)(X)
    shortcut = X
    X = common_layers(X)
    X = inception_residual_block(X, 128, True)
    X = Add()([shortcut, X])
    X = common_layers(X)
    X = maxPool_layer(X, 3, 3)
    X = Conv2D(128,(3,3),1)(X)
    shortcut = X
    X = common_layers(X)
    X = inception_residual_block(X, 128, True)
    X = Add()([shortcut, X])
    X = common_layers(X)
    X = AveragePooling2D((3,3), 2)(X)
    X = Flatten()(X)
    X = Dense(classes, name="dense_out")(X)
    outputs = Activation("softmax", name="softmax")(X)
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name ="mini_inception_resnet")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

model = init_model((128,128,3),5)
model.summary()

tf.keras.utils.plot_model(model, model.name + '_architecture.jpg', show_shapes=True) # SET TO CORRECT PATH
