import config
import json
import pickle
from cv2 import cv2
import numpy as np 
from features_indexer import FeatureIndexerReader
from angle_scale import AngleScaleReader
# from signature_indexer import SignatureIndexerReader
from word_getter import WordImagesReader
from geometrical_consistency import GeometricalConsistency
from scipy.cluster.vq import vq
import search
import time
import progressbar
import he_search

if __name__ == "__main__":
    config_value = config.get_config()
    
    desc_reader = FeatureIndexerReader(config_value['index'])
    angle_scale_reader = AngleScaleReader(config_value['angle-scale-path'])
    
    images = list(desc_reader.keys())
    
    tf = None
    with open(config_value['tf-path'], 'rb') as tf_file:
        tf = pickle.load(tf_file)

    idf = None
    with open(config_value['idf-path'], 'rb') as idf_file:
        idf = pickle.load(idf_file)


    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file)

    relevant = None
    with open(config_value['relevant-path'], 'r') as relevant_file:
        relevant = json.load(relevant_file)
    
    # signature_codebook = None
    # with open(config_value['median-path'], 'rb') as signature_codebook_file:
    #     signature_codebook = np.array(pickle.load(signature_codebook_file))
    

    # signature_reader = SignatureIndexerReader(config_value['signature-path'])
    word_reader = WordImagesReader(config_value['word-path'])

    #reset evaluation file
    evaluation_file = open(config_value['evaluation-export'], 'w')
    evaluation_file.close()

    for image in progressbar.progressbar(images, 0, len(images)):
        start_time = time.perf_counter()
        query_image = cv2.imread(config_value['dataset-path'] + image)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)

        tf_idf_images, matching_images = search.search(query_image, config_value['detector'], codebook, tf, idf, desc_reader, angle_scale_reader, 100, 10, config_value['delta-angle']) 
        # tf_idf_images, matching_images = he_search.search(query_image, config_value['detector'], codebook, signature_codebook, signature_reader, desc_reader, word_reader, tf, idf)
        tf_idf_accurency = len(set(tf_idf_images).intersection(set(relevant[image])))

        matching_values = set(relevant[image]).intersection(set(matching_images))
        accurency = len(matching_values)
        with open(config_value['evaluation-export'], 'a') as evaluation_file:
            print("{},{},{},{}".format(image, tf_idf_accurency, accurency, time.perf_counter() - start_time), file=evaluation_file)
