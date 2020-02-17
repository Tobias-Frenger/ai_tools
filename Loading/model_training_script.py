# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:16:49 2020

@author: Tobias
"""

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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
    plt.ylim(top=1.5, bottom=0.0)
    plt.legend()
    plt.figure()
    plt.show()

class callback_history(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.b_acc = []
        self.b_loss = []
        self.e_val_acc = []
        self.e_val_loss = []
        self.e_acc = []
        self.e_val_acc = []
        
    def on_batch_end(self, batch, logs={}):
        #plt.clf()
        self.b_acc.append(logs.get('accuracy'))
        self.b_loss.append(logs.get('loss'))
        acc = self.b_acc
        loss = self.b_loss
        batch = range(len(self.acc))
        plt.plot(batch, acc, 'g', label='Training acc')
        plt.plot(batch, loss, 'r', label='Training loss')
        plt.ylim(top=1.5, bottom=0.0)
        plt.legend()
        plt.show()
        
        plt.savefig('batch_figure.png')
        plt.pause(0.0001)
        plt.clf()
        
    def on_epoch_end(self, epoch, logs={}):
        self.e_acc.append(logs.get('accuracy'))
        self.e_val_acc.append(logs.get('val_accuracy'))
        self.e_loss.append(logs.get('loss'))
        self.e_val_loss.append(logs.get('val_loss'))
        acc = self.e_acc
        loss = self.e_loss
        val_loss = self.e_val_loss
        val_acc = self.e_val_acc
        batch = range(len(self.acc))
        plt.plot(batch, acc, 'g', label='Training acc')
        plt.plot(batch, loss, 'r', label='Training loss')
        plt.plot(batch, val_acc, 'b', label='Validation acc')
        plt.plot(batch, val_loss, 'c', label='Validation loss')
        plt.ylim(top=1.5, bottom=0.0)
        plt.legend()
        plt.show()
        
        plt.savefig('epoch_figure.png')
        plt.pause(0.0001)
        plt.clf()

def train_model(model_location, train_x, train_y):
    model = load_model(model_location)
    IMG_SIZE = 128
    train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    train_y = np.array(train_y)
    model_name = model.name
    checkpoint_loss = ModelCheckpoint('C:/Users/Tobias/CNN/Thesis_Models/' + str(model_name) + "/saved_models/" + model_name + "_{epoch:02d}-{loss:.2f}_best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
    checkpoint_val_loss = ModelCheckpoint('C:/Users/Tobias/CNN/Thesis_Models/' + str(model_name) + "/saved_models/" + model_name + "_{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)
    history = callback_history()
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    history = model.fit(train_x, train_y, batch_size=64, validation_split=0.15, epochs = 100, callbacks=[checkpoint_loss, checkpoint_val_loss, reduce_lr, history])
    show(history.history)
    return model
