import config
import cv2
import numpy as np
import pickle
from features_indexer import FeatureIndexerReader
from scipy.cluster.vq import vq
import progressbar
import sys

if __name__ == "__main__":
    config_value = config.get_config()

    reader = FeatureIndexerReader(config_value['index'])
    images = list(reader.keys())
    
    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file)
    
    idf = np.zeros((codebook.shape[0]))
    tf = np.zeros((len(images), codebook.shape[0]))
    for index, image in progressbar.progressbar(enumerate(images), 0, len(images)):
        desc = np.array(reader.get(image))
        words, distances = vq(desc, codebook)
        for word in words:
            tf[index, word] = tf[index, word] + 1
        
        tf[index, :] = tf[index, :] / words.shape[0]
        exist_words = np.unique(words)
        idf[exist_words] = idf[exist_words] + 1

    idf = np.log(len(images) / idf)

    print("[Process]Saving tf")
    with open(config_value['tf-path'], 'wb') as tf_file:
        pickle.dump(tf, tf_file)

    with open(config_value['idf-path'], 'wb') as idf_file:
        pickle.dump(idf, idf_file) 
