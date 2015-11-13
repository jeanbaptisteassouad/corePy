import cv2
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

with open('classes.pickle', 'rb') as f:
    classes = pickle.load(f)
with open('data.pickle', 'rb') as f:
    datas = pickle.load(f)

if datas is not None:
    dim = len(datas[0])
else:
    dim = -1

training = datas[0:20]
target = classes[0:20]

x_test = datas[20:30]
test_target = classes[20:30]

print("Running MLP >>>")

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape
model.add(Dense(64, input_dim=dim, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(training, target, nb_epoch=400, batch_size=200)
score = model.evaluate(x_test, test_target, batch_size=200)

print("score : "+str(score))
print(model.predict_classes(x_test))