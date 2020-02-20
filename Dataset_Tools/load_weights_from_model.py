# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:25:57 2020

@author: johan
"""
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


#use this. does only work with 1 dense layer
def set_weights_from_model(source_model, destination_model):
    source_model = remove_all_dense(source_model)   
    last_layer = destination_model.layers[-1]
    destination_model = remove_all_dense(destination_model)
    i = 0
    for layer in source_model.layers:
        destination_model.layers[i].set_weights(layer.get_weights())
        i += 1
    #destination_model.set_weights(source_model.get_weights())
    return Model(inputs = destination_model.layers[0].output, outputs = last_layer.output)

    
def remove_last_layer(model):
    return Model(inputs = model.layers[0].output, outputs = model.layers[-2].output) #deletes the last layer.

def remove_all_dense(model):
    new_last_layer_index = -1
    for i in reversed(range(len(model.layers))):    
        if isinstance(model.layers[i], Dense):
            new_last_layer_index -= 1        
    return Model(inputs = model.layers[0].output, outputs = model.layers[new_last_layer_index].output) #deletes the last layer.

#do not work######################################
def change_dimensions(model, in_dim, out_dim):
    new_model = remove_all_dense(model)
    input_layer = Input(shape = in_dim, dtype='float32', name='input_layer')
    x = input_layer
    for layer in new_model.layers[1:]:
        y = Model(inputs = layer.input, outputs = layer.output)
        x = y(x)
    print(x)   
    dense = Dense(out_dim, activation='softmax')(x) #x från for loopen?
    print(dense)
    new_model = Model(inputs = input_layer, outputs = dense, name="model")
    #print(model)
    #new_model = set_weights_from_model(model, new_model)
    return new_model
####################################
