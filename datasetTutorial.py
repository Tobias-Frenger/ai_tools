#######################################
#   @Original_author: sentdex: https://youtu.be/j-3vuBynnOE
#   @Edited_by: Tobias
#######################################

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "C:/path/to/your/Images"
CATEGORIES = ["A", "B", "C", "D", "E"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) # path to img class folder
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

IMG_SIZE = 128
#plt.imshow(new_arr)
#plt.show()

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to img class folder
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_arr = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_arr, class_num])
            except Exception as e:
                pass
            
create_training_data()
print(len(training_data))

#RANDOMIZE TRAINING DATA
import random
random.shuffle(training_data)

#PRINT 20 SAMPLE TO SEE THE RESULT
for sample in training_data[:20]:
    print(sample[1])
    
X = []
y = []

#X GETS THE MATRIX VALUES, y GETS THE LABELS
for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

import pickle
# SAVE train_x, train_y
pickle_out = open("C:/Users/Tobias/CNN/training_set_X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("C:/Users/Tobias/CNN/training_set_y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# LOAD train_x, train_y
pickle_in = open("C:/Users/Tobias/CNN/training_set_X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("C:/Users/Tobias/CNN/training_set_y.pickle", "rb")
y = pickle.load(pickle_in)
y[0]
