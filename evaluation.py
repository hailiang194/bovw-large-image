import config
import json
import pickle
from cv2 import cv2
import numpy as np 
from features_indexer import FeatureIndexerReader
from angle_scale import AngleScaleReader
from geometrical_consistency import GeometricalConsistency
from scipy.cluster.vq import vq
import search
import time
import progressbar

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
    
    #reset evaluation file
    evaluation_file = open(config_value['evaluation-export'], 'w')
    evaluation_file.close()

    for image in progressbar.progressbar(images, 0, len(images)):
        start_time = time.perf_counter()
        query_image = cv2.imread(config_value['dataset-path'] + image)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)

        keypoints, descriptors = config_value['detector'].detectAndCompute(query_image, None)
        # if len(keypoints) <= 1:
        #     query_image = cv2.equalizeHist(query_image)
        #     keypoints, descriptors = config_value['detector'].detectAndCompute(query_image, None)

        #get tf of query image
        query_code, _ = vq(descriptors, codebook) 
        query_tf = np.zeros((int(config_value['num-cluster'])))

        uniq_code, counts = np.unique(query_code, return_counts=True) 
        for code, count in zip(uniq_code.tolist(), counts.tolist()):
            query_tf[code] = count
        
        #tf_idf matching
        query_tf = query_tf / query_code.shape[0]
        tf_idf_images, tf_idf_score = search.score_by_tf_idf(query_tf, images, tf, idf)

        #geometrical consistency
        query_angle_scale = np.array([[keypoint.angle, keypoint.octave] for keypoint in keypoints])
        matching_images, matching_scores = search.geometrical_consistency_remark(query_angle_scale, descriptors, tf_idf_images, angle_scale_reader, desc_reader)
        matching_values = set(relevant[image]).intersection(set(matching_images))
        accurency = len(matching_values)
        with open(config_value['evaluation-export'], 'a') as evaluation_file:
            print("{},{},{}".format(image, accurency, time.perf_counter() - start_time), file=evaluation_file)
