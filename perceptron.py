# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
import numpy as np

# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(64, input_dim=20, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(2, init='uniform'))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)

X_all = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
print X_all
cross = 2
tot = len(X_all)
pas = tot/cross


p = np.array([[1,2],[3,4]])
p = np.append(p, [[5,6]], 0)
p = np.append(p, [[7],[8],[9]],1)
print p
# for x in range(0, cross):
#     X_train = np.zeros((1, len(X_all[0])))
#     X_test = np.zeros((1, len(X_all[0])))
#     for i in range(0, pas*x):
#         X_train = np.append(X_train, X_all[i], 0)
#     for i in range(pas*x, pas*(x+1)):
#         X_test = np.append(X_test, X_all[i], 0)
#     for i in range(pas*(x+1), tot):
#         X_train = np.append(X_train, X_all[i], 0)

#     print X_train
#     print X_test

#     pass

# X_all =
#     [   [3, ..., 6]
#         [2, ..., 23]
#         ...
#         [123, ..., 2]
#     ]

# X_train =
#     [
#     ]

# X_train =
#     [   [3, ..., 6]
#     ]

# X_train =
#     [   [3, ..., 6]
#         [2, ..., 23]
#     ]


# model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
# score = model.evaluate(X_test, y_test, batch_size=16)
