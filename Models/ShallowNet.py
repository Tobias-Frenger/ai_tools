# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:32:49 2020

@author: combitech
"""
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.datasets import cifar100, cifar10

import tensorflow as tf
import matplotlib.pyplot as plt
lb = LabelBinarizer()
#(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
#y_train = keras.utils.to_categorical(y_train, 10)
#y_test = keras.utils.to_categorical(y_test, 10)

def init_shallownet(in_dim, classes):
    input_layer = Input(shape = in_dim, dtype='float32', name='in')  
    X = layers.Conv2D(32, (3,3), padding='same', name='conv_1')(input_layer)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)
    d1 = Activation('softmax')(X)
    return Model(inputs = input_layer, outputs = d1, name="model")



opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
#opt = SGD(lr=0.01, decay=0.01/40, momentum = 0.9)
shallowNet = init_shallownet((32,32, 3), 10)
shallowNet.summary()

shallowNet.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint_val_loss = ModelCheckpoint("C:/Users/Tobias/CNN/ShallowNet/" + shallowNet.name + '_{epoch:02d}-{loss:.2f}_best_val_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

history = shallowNet.fit(x_train, y_train, batch_size=64, epochs=10000, validation_split=0.15, callbacks=[checkpoint_val_loss, early_stopping])
shallowNet.evaluate(x_test, y_test, batch_size=64, verbose=2)

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
    plt.ylim(1.5)
    plt.xlim(0.0)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.figure()
    plt.show()

show(shallowNet.history.history)
