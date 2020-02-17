# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:16:49 2020

@author: Tobias
"""

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
#from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#import matplotlib.pyplot as plt

class callback_history(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
    def on_batch_end(self, batch, logs={}):
        #plt.clf()
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        acc = self.acc
        loss = self.loss
        val_loss = self.val_loss
        val_acc = self.val_acc
        batch = range(len(self.acc))
        plt.plot(batch, acc, 'g', label='Training acc')
        plt.plot(batch, loss, 'r', label='Training loss')
        plt.plot(batch, val_acc, 'b', label='Validation acc')
        plt.plot(batch, val_loss, 'c', label='Validation loss')
        plt.legend()
        plt.show()
        
        plt.savefig('testfigure.png')
        plt.pause(0.0001)
        #img = plt.imread('testfigure.png')
        #plt.imshow(img)
        plt.clf()

def train_model(model_location, train_x, train_y):
    model = load_model(model_location)
    IMG_SIZE = 128
    train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    train_y = np.array(train_y)
    model_name = model.name
    checkpoint_loss = ModelCheckpoint(str(model_name) + "/saved_models/" + model_name + "{epoch:02d}-{loss:.2f}_best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
    checkpoint_val_loss = ModelCheckpoint(str(model_name) + "/saved_models/" + model_name + "{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
    history = callback_history()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=64, validation_split=0.1, epochs = 100, callbacks=[checkpoint_loss, checkpoint_val_loss, reduce_lr, history])
