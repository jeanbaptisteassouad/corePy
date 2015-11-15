import cv2
import numpy as np
from os import listdir
# serialize our data
import pickle
# regexp
import re
# multiprocessing lib
import multiprocessing as mp

import projection as proj


# Dataset for table recognition
# to create data.pickle : d = DataSet(file_dir="dataset/")
# to load data.pickle and classes.pickle : d = DataSet(data_pickle="data.pickle", classes_pickle="classes.pickle")
class DataSet(object):
    # constructor
    def __init__(self, data_pickle=None, classes_pickle=None, file_dir=None):
        if file_dir is None and data_pickle is not None and classes_pickle is not None:
            with open(data_pickle, 'rb') as f:
                self.datas = pickle.load(f)
            with open(classes_pickle, 'rb') as f:
                self.classes = pickle.load(f)
        elif data_pickle is None and classes_pickle is not None and file_dir is not None:
            # load classes
            print("loading " + classes_pickle + " file >>> self.classes")
            with open(classes_pickle, 'rb') as f:
                self.classes = pickle.load(f)
            # nb_examples and data vector length
            self.nb_examples = len(self.classes)
            self.length = 0
            # Create dataset
            print("creating input data >>> " + file_dir + " directory")
            self.file_dir = file_dir
            self.datas = self.create_input()
            # dump data in data.pickle binary file p
            with open('data.pickle', 'wb') as f:
                pickle.dump(self.datas, f, protocol=2)
        else:
            print("arg error >>> set either file_dir and classes_pickle or data_pickle and classes_pickle")
            exit()

    def create_input(self):
        files = sorted(listdir(self.file_dir), key=lambda x: (int(re.sub('\D', '', x)), x))
        # inputs and output queue for multiprocessing purposes
        inputs = mp.JoinableQueue()
        output = mp.Queue()
        # add files to queue
        for i in range(len(files)):
            inputs.put((i, files[i]))
        # instantiate 4 processes
        processes = [mp.Process(target=self.worker, args=(x, inputs, output)) for x in range(4)]
        # run processes
        for process in processes:
            process.start()
        # join queue
        inputs.join()
        # get results
        res = [output.get() for process in processes]
        # create a sorted numpy array with res
        if res is not None:
            self.length = len(res[0][0][2])
            data = np.zeros((self.nb_examples,self.length))
            for i in range(len(res)):
                for j in range(len(res[i])):
                    data[:][res[i][j][0]] = res[i][j][2]
        #for i in range(len(res)):
        #    for j in range(len(res[i])):
        #        print("worker" + str(i) + ", index : " + str(res[i][j][0]) + ", img path : '" + str(
        #            res[i][j][1]) + "', size : " + str(len(res[i][j][2])))
        return data

    def worker(self, index, input, output):
        data = []
        while True:
            if input.empty():
                break
            args = input.get()
            data.append(self.algorithm(args[0], args[1]))
            input.task_done()
        output.put(data)

    def algorithm(self, pos, img_path):
        # Projection transform
        img = cv2.imread(self.file_dir + img_path)
        ans = proj.projectionHist(img)
        return (pos, img_path, ans)


dataset_directory = "dataset/"
input = DataSet(classes_pickle="classes.pickle",file_dir="dataset/")
