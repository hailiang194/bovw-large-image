import h5py
import config
from scipy.cluster.vq import vq
import pickle
from features_indexer import FeatureIndexerReader
import numpy as np

class WordImagesWriter(object):
    def __init__(self, index_reader, codebook):
        self.__index_reader = index_reader
        self.__codebook = codebook    

    def save(self, output_path):
        output = h5py.File(output_path, 'w')

        for image in self.__index_reader.keys():
            desc = np.array(self.__index_reader.get(image))
            code, _ = vq(desc, self.__codebook)
            output.create_dataset(image, data=code)  
        output.close()

class WordImagesReader(object):
    def __init__(self, path):
        self.__h5_file = h5py.File(path, "r")
            
    def keys(self):
        return self.__h5_file.keys()

    def get(self, image):
        return self.__h5_file.get(image)

    def close(self):
        try: 
            self.__h5_file.close()
        except:
            return

    def __del__(self):
        self.close()

if __name__ == "__main__":
    config_value = config.get_config()
    
    #get codebook
    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file) 

    reader = FeatureIndexerReader(config_value['index']) 
    writer = WordImagesWriter(reader, codebook)
    writer.save(config_value['word-path'])
