# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 03:59:39 2019

@author: Bananin
"""

import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
os.chdir("C:/Users/Bananin/Documents/eeg/")

imgs_folder = "ecosystem_data/images/"
all_images = os.listdir(imgs_folder)
# todas las imagenes deberian tener el mismo tamano, pero por si acaso...
img = cv2.imread(imgs_folder+all_images[0],0)
img_dims = img.shape

# organizamos cubos de entreno y prueba
train_size = int(len(all_images)*0.8)
X_train = np.zeros((train_size, img_dims[0], img_dims[1]), "uint8")
for i in range(train_size):
    X_train[i] = cv2.imread(imgs_folder+all_images[i],0)
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = np.zeros((len(all_images)-train_size, img_dims[0], img_dims[1]), "uint8")
for i in range(train_size, len(all_images)):
    X_test[i-train_size] = cv2.imread(imgs_folder+all_images[i],0)
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

# respuestas aleatorias
y_train = np.random.binomial(1,0.5,X_train.shape[0])
y_test = np.random.binomial(1,0.5,X_test.shape[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(img.shape[0],img.shape[1],1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(2, activation="softmax"))
#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)


    
cv2.imshow('image', X_test[5])
cv2.waitKey(0)