# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 01:23:54 2020

@author: Tobias
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# SETTING UP TRAINING AND TEST SETS #############################################################
DATADIR = "C:/Users/You/Your/Path/To/Dataset"
CATEGORIES = ["Class1", "Class2", "Class3"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

IMG_SIZE = 128
#plt.imshow(new_arr)
#plt.show()

training_data = []
test_data = []
def create_training_and_test_data():
    count = 1
    mod = 5
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                new_arr = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                if (np.equal(count % 5, 0)):
                    test_data.append([new_arr, class_num])
                else:
                    training_data.append([new_arr, class_num])
                count += 1
            except Exception as e:
                pass
            
create_training_and_test_data()
print(len(training_data))
print(len(test_data))

#RANDOMIZE TRAINING DATA
import random
random.shuffle(training_data)
random.shuffle(test_data)
#PRINT 20 SAMPLE TO SEE THE RESULT
for sample in training_data[:20]:
    print(sample[1])
    
train_x = []
train_y = []
#PRINT 20 SAMPLE TO SEE THE RESULT
for sample in test_data[:20]:
    print(sample[1])
    
test_x = []
test_y = []

#X GETS THE MATRIX VALUES, y GETS THE LABELS
for features, label in training_data:
    train_x.append(features)
    train_y.append(label)

for features, label in test_data:
    test_x.append(features)
    test_y.append(label)
    
train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# SAVING TRAIN AND TEST DATASETS #############################################################
import pickle
# SAVE train_x, train_y
pickle_out = open("C:/Users/You/Your/Save/Location/training_set_X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open("C:/Users/You/Your/Save/Location/training_set_y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
# Save test_x, test_y
pickle_out = open("C:/Users/You/Your/Save/Location/test_set_X.pickle", "wb")
pickle.dump(test_x, pickle_out)
pickle_out.close()
pickle_out = open("C:/Users/You/Your/Save/Location/test_set_y.pickle", "wb")
pickle.dump(test_y, pickle_out)
pickle_out.close()

# LOADING EXAMPLE #############################################################################
# LOAD train_x, train_y
pickle_in = open("C:/Users/You/Your/Save/Location/training_set_X.pickle", "rb")
X = pickle.load(pickle_in)
X[0]
pickle_in = open("C:/Users/You/Your/Save/Location/training_set_y.pickle", "rb")
y = pickle.load(pickle_in)
y[0]
