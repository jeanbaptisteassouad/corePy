import numpy as np
import pickle

with open('classes.pickle', 'rb') as f:
    classes = pickle.load(f)

print(classes)