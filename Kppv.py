import pickle
from Predictor import Predictor
import numpy as np
import scipy.spatial.distance as scipyDistance

class Kppv(Predictor):
    """docstring for Kppv"""
    def __init__(self):
        super(Kppv, self).__init__()
        self.k = 4

    def train(self, datas, classes):
        new_datas = np.resize( self.datas, (len(self.datas)+len(datas),500) )
        new_datas[0:len(self.datas)] = self.datas
        new_datas[len(self.datas):len(self.datas)+len(datas)] = datas
        self.datas = new_datas

        new_classes = np.resize( self.classes, (len(self.classes)+len(classes),2) )
        new_classes[0:len(self.classes)] = self.classes
        new_classes[len(self.classes):len(self.classes)+len(classes)] = classes
        self.classes = new_classes

        self.is_train_once = True


    def test(self, data):
        pass

    def predict(self, data):
        if self.is_train_once :
            score = []
            for x in range(0,len(self.datas)):
                score.append( [scipyDistance.euclidean( data, self.datas[x] ),x] )
                pass
            score.sort()
            indice = []
            for x in range(0,self.k):
                indice.append( score[x][1] )
                pass
            tot = 0
            for x in range(0,len(indice)):
                tot += self.classes[indice[x]][1]
            tot /= len(indice)
            if tot > 0.5:
                return (0,1)
            else:
                return (1,0)
        else:
            return (1,0)
        pass

    def serialize():
        pass

    def deserialize():
        pass