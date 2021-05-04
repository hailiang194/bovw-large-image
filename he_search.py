import config
import cv2
import numpy as np
from features_indexer import FeatureIndexerReader
import pickle
import json
from scipy.cluster.vq import vq
import sys
import time

def score_by_he(query_desc, images, codebook, signature,tf, idf, hamming_threshold, top=20):
    query_code, _ = vq(query_desc, codebook) 
    
    scores = np.zeros((len(images)))
    for code_index in range(query_code.shape[0]):
        query_bits = 0
        for bit_index in range(median.shape[1]):
            query_bits = query_bits | int(query_desc[code_index, bit_index] > median[query_code[code_index], bit_index])
            query_bits = query_bits << 1
        
        same_code_images = signature[str(query_code[code_index])]
        for db_image in same_code_images.keys():
            image_index = images.index(db_image)
            for bits in same_code_images[db_image]:
                distance = bin(query_bits ^ bits).count('1')
                if distance < hamming_threshold:
                    scores[image_index] += 1#(idf[query_code[code_index]]) ** 2 #* np.exp(-(distance / 26.0) ** 2)
    
    # print(scores)
    # scores = scores / np.linalg.norm(tf, axis=1)
    images_scores = dict(zip(images, scores.tolist()))
    images_scores = list(sorted(images_scores.items(), key=lambda item: item[1]))
    print(images_scores)
    return [image for (image, score) in images_scores[:-top:-1]], [score for (image, score) in images_scores[:-top:-1]]

if __name__ == "__main__":
    config_value = config.get_config()
    query_image = cv2.imread(sys.argv[1])
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)

    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file)

    signature = None
    with open(config_value['signature-path'], 'r') as signature_file:
        signature = json.load(signature_file)
    

    median = None
    with open(config_value['median-path'], 'rb') as median_file:
        median = pickle.load(median_file)

    idf = None
    with open(config_value['idf-path'], 'rb') as idf_file:
        idf = pickle.load(idf_file)

    query_kp, query_desc = config_value['detector'].detectAndCompute(query_image, None)

    tf = None
    with open(config_value['tf-path'], 'rb') as tf_file:
        tf = pickle.load(tf_file)

    index_reader = FeatureIndexerReader(config_value['index'])
    images = list(index_reader.keys())
    
    print("Searching...")
    start_time = time.perf_counter()
    mapping_images, scores = score_by_he(query_desc, images, codebook, signature,tf, idf, 24)
    print(time.perf_counter() - start_time)
    print(mapping_images)
