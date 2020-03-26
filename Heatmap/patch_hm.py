# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:27:14 2020

@author: Tobias
"""
#### OCCULSION SENSITIVITY ####
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# Create function to apply a grey patch on an image
# Figure out what parts of an image which is important for the correct prediction
def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    patched_image = np.array(image, copy=True)
    patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = 127.5

    return patched_image

# Load image
IMAGE_PATH = 'C:/Users/Tobias/CNN/Dataset/Training/Johan/johan_reading_19.jpg'
img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(128, 128))
img = tf.keras.preprocessing.image.img_to_array(img)

# Instantiate model
model = load_model("C:/Users/Tobias/CNN/Thesis_Models/Simple_Residual_Model/saved_models/s1/Simple_Residual_Model97-0.25_best_val_model.hdf5")
test = np.argmax(model.predict(np.array([img])))
print(test)
CLASS_INDEX = 3  # index of class
PATCH_SIZE = 16

sensitivity_map = np.zeros((img.shape[0], img.shape[1]))

# Iterate the patch over the image
for top_left_x in range(0, img.shape[0], PATCH_SIZE):
    for top_left_y in range(0, img.shape[1], PATCH_SIZE):
        patched_image = apply_grey_patch(img, top_left_x, top_left_y, PATCH_SIZE)
        
        predicted_classes = model.predict(np.array([patched_image]))[0]
       # print(predicted_classes)
        confidence = predicted_classes[CLASS_INDEX]
        #print(confidence)
        # Save confidence for this specific patched image in map
        sensitivity_map[
            top_left_y:top_left_y + PATCH_SIZE,
            top_left_x:top_left_x + PATCH_SIZE,
        ] = confidence

plt.imshow(sensitivity_map)
#######################################################################################################
model.summary()

import numpy as np
import tensorflow as tf
import cv2
from tensorflow import reshape
# Layer name to inspect
layer_name = 'conv2d_42'

epochs = 100
step_size = 1.
filter_index = 0

# Create a connection between the input and the target layer
model = load_model("C:/Users/Tobias/CNN/Thesis_Models/res_model/saved_models/s1/res_model_10-0.37_best_model.hdf5")
#model = load_model("C:/Users/Tobias/CNN/Thesis_Models/inception_resnet_simple/saved_models/V30_/V30_inception_resnet_simple_01-0.03_best_val_model.hdf5")
#model = load_model("C:/Users/Tobias/CNN/Thesis_Models/Simple_Residual_Model/saved_models/s1/Simple_Residual_Model97-0.25_best_val_model.hdf5")
submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])

# Initiate random noise
input_img_data = np.random.random((1, 128, 128, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128.

# Cast random noise from np.float64 to tf.float32 Variable
input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

# Iterate gradient ascents
for _ in range(epochs):
    with tf.GradientTape() as tape:
        outputs = submodel(input_img_data)
        loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
    grads = tape.gradient(loss_value, input_img_data)
    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    input_img_data.assign_add(normalized_grads * step_size)


img_tensor = input_img_data/255.

my_im = np.squeeze(img_tensor, axis=None)
plt.imshow(my_im)
img_resized = cv2.resize(my_im, (128,128), interpolation = cv2.INTER_LINEAR)
plt.imshow(img_resized)
plt.show()

#####################################################
import cv2
import numpy as np
import tensorflow as tf
IMAGE_PATH = 'C:/Users/Tobias/CNN/Images/Hallplats/hallplats_12.jpg'
#IMAGE_PATH = 'C:/Users/Tobias/CNN/Dataset/Training/Johan/johan_reading_17.jpg'
LAYER_NAME = 'leaky_re_lu_4'
#CAT_CLASS_INDEX = 3

img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(128, 128))
img = tf.keras.preprocessing.image.img_to_array(img)

# Load initial model
model = load_model("C:/Users/Tobias/CNN/Thesis_Models/inception_resnet_simple/saved_models/V30_/V30_inception_resnet_simple_01-0.03_best_val_model.hdf5")

# Create a graph that outputs target convolution and output
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

# Get the score for target class
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(np.array([img]))
    loss = predictions[:, np.argmax(predictions[0])]

# Extract filters and gradients
output = conv_outputs[0]
grads = tape.gradient(loss, conv_outputs)[0]

# Average gradients spatially
weights = tf.reduce_mean(grads, axis=(0, 1))

# Build a ponderated map of filters according to gradients importance
cam = np.ones(output.shape[0:2], dtype=np.float32)

for index, w in enumerate(weights):
    cam += w * output[:, :, index]

# Heatmap visualization
cam = cv2.resize(cam.numpy(), (128, 128))
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())

cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 1, cam, 1, 0)
plt.imshow(output_image)
