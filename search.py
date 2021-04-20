import config
from angle_scale import AngleScaleReader
from geometrical_consistency import GeometricalConsistency
from features_indexer import FeatureIndexerReader
from scipy.cluster.vq import vq
import numpy as np
from cv2 import cv2
import sys
import pickle
import imutils

def score_by_tf_idf(query_tf, image_labels, tf, idf=None, top=50):
    scores = np.zeros((tf.shape[0]))

    cosine_simularity = lambda x, y: (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    for index in range(tf.shape[0]):
        scores[index] = cosine_simularity(query_tf * idf, tf[index, :] * idf)

    image_score = dict(zip(image_labels, scores.tolist()))
    image_score = list(sorted(image_score.items(), key=lambda item: item[1]))    
    return [image for (image, score) in image_score[:-top:-1]], [score for (image, score) in image_score[:-top:-1]]

def geometrical_consistency_remark(query_angle_scale, descriptors, image_labels, angle_scale_reader, desc_reader, top=10, distance_ratio=0.7, delta_angle=1):
    scores = np.zeros((len(image_labels)))
    for index, image in enumerate(image_labels):
        desc = np.array(desc_reader.get(image))
        angle_scale = np.array(angle_scale_reader.get(image))
        consistency = GeometricalConsistency(query_angle_scale, descriptors)
        scores[index] = consistency.score(desc, angle_scale) 

    image_scores = dict(zip(image_labels, scores.tolist()))
    image_scores = list(sorted(image_scores.items(), key=lambda item: item[1]))
    return [image for (image, score) in image_scores[:-top:-1]], [score for (image, score) in image_scores[:-top:-1]]

if __name__ == "__main__":
    config_value = config.get_config()
    query_image = cv2.imread(sys.argv[1])
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)
    cv2.imshow("Temp", query_image)
    cv2.waitKey(0)
    quit()
    # get image labels
    angle_scale_reader = AngleScaleReader(config_value['angle-scale-path']) 
    image_labels = list(angle_scale_reader.keys())

    #detect keypoints and descriptor of query image
    keypoints, descriptors = config_value['detector'].detectAndCompute(query_image, None)
    print(len(keypoints))
    #get all tf
    tf = None
    with open(config_value['tf-path'], 'rb') as tf_file:
        tf = pickle.load(tf_file)
    
    #get idf
    idf = None
    with open(config_value['idf-path'], 'rb') as idf_file:
        idf = pickle.load(idf_file)

    # get tf of query image
    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file)

    #get tf of query image
    query_code, _ = vq(descriptors, codebook)
    query_tf = np.zeros((int(config_value['num-cluster'])))

    values, counts = np.unique(query_code, return_counts=True)

    for value, count in zip(values.tolist(), counts.tolist()):
        query_tf[value] = count

    query_tf = query_tf / query_code.shape[0]

    tf_idf_labels, tf_idf_score = score_by_tf_idf(query_tf, image_labels, tf, idf)
    print(tf_idf_labels)
    #get angle and scale of query_image
    query_angle_scale = np.array([[keypoint.angle, keypoint.octave] for keypoint in keypoints])
    matching_images, matching_scores = geometrical_consistency_remark(query_angle_scale, descriptors, tf_idf_labels, angle_scale_reader, FeatureIndexerReader(config_value['index']))
    print(matching_images)
    print(matching_scores)
