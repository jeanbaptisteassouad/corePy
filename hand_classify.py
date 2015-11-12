import cv2
import numpy as np
from os import listdir
# serialize our data
import pickle
# regex
import re

# put data in dataset/ directory
# naming convention : x.pgm, with x integer

dataset_directory = "dataset/"
# sort by file number
files = sorted(listdir(dataset_directory),key=lambda x: (int(re.sub('\D','',x)),x))

classes = np.zeros((len(files),2))

for i in range(0,len(files)):
    im = cv2.imread(dataset_directory+files[i])
    cv2.imshow('im',im)
    cv2.waitKey(500)
    isTable = input('enter 1 if image contains a table : ')
    # handle user input
    # empty input = no table
    if (isTable == ''):
        isTable = 0
    # above 1 = table
    elif int(isTable) >= 1:
        isTable = 1
    # 0 or negative = no table
    else:
        isTable = 0
    # add to classes Numpy vector
    classes[i][isTable] = 1

print(classes)

with open('classes.pickle', 'wb') as f:
    pickle.dump(classes, f, protocol=2)

