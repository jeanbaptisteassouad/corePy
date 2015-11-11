import numpy as np
import pickle

import cv2
import projection as proj

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

with open('classes.pickle', 'rb') as f:
    classes = pickle.load(f)

targets = np.zeros((len(classes),1))

training = np.zeros((len(classes),997))
nb_examples, data = training.shape

for i in range(0,2):
    print("Working on page : "+str(int(classes[i][0]))+", class : "+str(classes[i][1]))
    img = cv2.imread("hpc_low/-"+str(int(classes[i][0]))+".pgm")
    training[:][i] = proj.projection(img)
    targets[i] = classes[i][1]

x_test = proj.projection(cv2.imread("hpc_low/-"+str(int(classes[25][0]))+".pgm"))
y_test = targets[25]

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape
model.add(Dense(64, input_dim=997, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(training, classes[:][1], nb_epoch=20, batch_size=16)
score = model.evaluate(x_test, y_test, batch_size=16)