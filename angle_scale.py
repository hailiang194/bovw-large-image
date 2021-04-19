import numpy as np 
import h5py

class AngleScaleWritter(object):
    def __init__(self, path):
        self.__h5_file = h5py.File(path, "w")

    def insert(self, image, values):
        self.__h5_file.create_dataset(image, data=values)

    def close(self):
        try:
            self.__h5_file.close()
        except:
            return

    def __del__(self):
        self.close()

class AngleScaleReader(object):
    def __init__(self, path):
        self.__h5_file = h5py.File(path, "r")

    def keys(self):
        return self.__h5_file.keys()

    def get(self, key):
        return self.__h5_file.get(key)

    def close(self):
        try:
            self.__h5_file.close()
        except:
            return

    def __del__(self):
        self.close()
