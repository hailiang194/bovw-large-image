import config
from angle_scale import AngleScaleReader
from features_indexer import FeatureIndexerReader
import numpy as np 
import sys
from cv2 import cv2
from scipy.cluster.vq import vq
import pickle

MAX_ANGLE = 360
MIN_ANGLE = 0

class GeometricalConsistency(object):
    def __init__(self, angle_scale, descriptors , distance_ratio=0.7, delta_angle=1):
         
        self.__angle_scale = angle_scale
        self.__descriptors = descriptors
        self.__distance_ratio = distance_ratio
        self.__delta_angle = delta_angle


    def score(self, desc, angle_scale):
        total_angle = int((MAX_ANGLE - MIN_ANGLE) / self.__delta_angle)
        
        bf = cv2.BFMatcher_create()
        matches = bf.knnMatch(desc, self.__descriptors, 2)
        angles = np.zeros((total_angle))
        HIGHEST_QUERY_SCALE = np.amax(self.__angle_scale[:, 1])
        HIGHEST_SCALE = np.amax(angle_scale[:, 1])
        MAX_SCALE_POS = np.log(np.maximum(HIGHEST_QUERY_SCALE, HIGHEST_SCALE).astype(np.float64)) + 1
        scales = np.zeros((int(MAX_SCALE_POS)))
        for m, n in matches:
            if m.distance < self.__distance_ratio * n.distance:
                angle_difference = np.abs(angle_scale[m.queryIdx, 0] - self.__angle_scale[m.trainIdx, 0])
                angle_pos = int(angle_difference / self.__delta_angle) 
                angles[angle_pos] += 1
                scale_diff = np.abs(np.log(angle_scale[m.queryIdx, 1]) - np.log(self.__angle_scale[m.trainIdx, 1]))
                scales[int(scale_diff)] += 1

        return np.minimum(np.amax(scales), np.amax(angles))

if __name__ == "__main__":
    config_value = config.get_config()

    #get keypoints and descriptors of query image 
    query_image = cv2.imread(sys.argv[1])
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)
    
    
    #get codes of each descriptors of query image
    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file) 


    angle_scale_reader = AngleScaleReader(config_value['angle-scale-path'])
    images = list(angle_scale_reader.keys())
    
    desc_reader = FeatureIndexerReader(config_value['index'])
    
    keypoints, descriptors = config_value['detector'].detectAndCompute(query_image, None)

    query_angle_scale = np.array([[keypoint.angle, keypoint.octave] for keypoint in keypoints])

    consistency = GeometricalConsistency(angle_scale, descriptors, 0.7, config_value['delta-angle'])
    final_scores = np.zeros((len(images)))
    #get scores
    for i, image in enumerate(images):
        angle_scale = np.array(angle_scale_reader.get(image))
        desc = np.array(desc_reader.get(image)) 
        print(image, consistency.score(desc, angle_scale))

