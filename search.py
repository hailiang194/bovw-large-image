import config
from angle_scale import AngleScaleReader
from geometrical_consistency import GeometricalConsistency
from features_indexer import FeatureIndexerReader
from scipy.cluster.vq import vq
import numpy as np
from cv2 import cv2
import sys
import pickle
import json
import imutils
from functools import cmp_to_key

def compare_final_score(first, second):
    if first[1] == second[1]:
        return second[2] - second[2]

    return second[1] - first[1]

def score_by_tf_idf(query_tf, image_labels, tf, idf=None, index_reader = None, top=100):
    scores = np.zeros((tf.shape[0]))

    cosine_simularity = lambda x, y: (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # determine_score = lambda x, y: np.linalg.norm(x / np.linalg.norm(x) - y / np.linalg.norm(y))
    for index in range(tf.shape[0]):
        num = index_reader.get(image_labels[index]).shape[0] if not image_labels is None else idf
        scores[index] = cosine_simularity(query_tf * num , tf[index, :] * num)
        # scores[index] = determine_score(query_tf * idf, tf[index, :] * idf)

    image_score = dict(zip(image_labels, scores.tolist()))
    image_score = list(sorted(image_score.items(), key=lambda item: item[1]))    
    return [image for (image, score) in image_score[:-top:-1]], [score for (image, score) in image_score[:-top:-1]]

def geometrical_consistency_remark(query_angle_scale, descriptors, image_labels, angle_scale_reader, desc_reader, image_pre_score=None, top=10, distance_ratio=0.7, delta_angle=1):
    scores = np.zeros((len(image_labels)))
    for index, image in enumerate(image_labels):
        desc = np.array(desc_reader.get(image))
        angle_scale = np.array(angle_scale_reader.get(image))
        consistency = GeometricalConsistency(query_angle_scale, descriptors, distance_ratio=distance_ratio, delta_angle=delta_angle)
        scores[index] = consistency.score(desc, angle_scale) 
    
    if image_pre_score is None: 
        image_scores = dict(zip(image_labels, scores.tolist()))
        image_scores = list(sorted(image_scores.items(), key=lambda item: item[1]))
        return [image for (image, score) in image_scores[:-top:-1]], [score for (image, score) in image_scores[:-top:-1]], None
    else:
        image_scores = list(zip(image_labels, scores.tolist(), image_pre_score))
        image_scores.sort(key=cmp_to_key(compare_final_score))
        return [image for (image, score, pre_score) in image_scores[:top]], [score for (image, score, pre_score) in image_scores[:top]], [pre_score for (image, score, pre_score) in image_scores[:top]]

def search(query_image, detector, codebook, tf, idf, desc_reader, angle_scale_reader, tf_idf_score_top=100, angle_scale_top=10, delta_angle=30, debug=False):
    #get keypoints and descriptors
    keypoints, descriptors = detector.detectAndCompute(query_image, None)
    if debug: print("Keypoints detector: {} keypoint(s) and shape of descriptors: {}".format(len(keypoints), descriptors.shape))

    #get tf of query image
    query_code, _ = vq(descriptors, codebook)
    query_tf = np.zeros((codebook.shape[0]))

    values, counts = np.unique(query_code, return_counts=True)

    for value, count in zip(values.tolist(), counts.tolist()):
        query_tf[value] = count

    query_tf = query_tf / 1 #query_code.shape[0]
    if debug: print("Term frequence of query image: {}".format(query_tf))
    
    image_labels = list(desc_reader.keys())
    if debug: print("Total database image(s): {}".format(len(image_labels)))

    tf_idf_labels, tf_idf_score = score_by_tf_idf(query_tf, image_labels, tf, idf, desc_reader, tf_idf_score_top)
    # if debug: print("\ntf-idf:\n{}".format(list(zip(tf_idf_labels, tf_idf_score))))
    # query_angle_scale = np.array([[keypoint.angle, keypoint.octave] for keypoint in keypoints])
    # matching_images, matching_scores, pre_score = geometrical_consistency_remark(query_angle_scale, descriptors, tf_idf_labels, angle_scale_reader, desc_reader, image_pre_score=tf_idf_score, top=angle_scale_top, delta_angle=delta_angle)
    # if debug: print("\nweak geometrical consistency:\n {}".format(list(zip(matching_images, matching_scores, pre_score))))
    
    return tf_idf_labels, tf_idf_labels[:10]

if __name__ == "__main__":
    config_value = config.get_config()
    query_image = cv2.imread(sys.argv[1])
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)
    # cv2.imshow("Temp", query_image)
    # cv2.waitKey(0)
    # quit()
    # get image labels
    # angle_scale_reader = AngleScaleReader(config_value['angle-scale-path']) 
    # image_labels = list(angle_scale_reader.keys())

    #detect keypoints and descriptor of query image
    # keypoints, descriptors = config_value['detector'].detectAndCompute(query_image, None)
    # print("keypoints = ", len(keypoints))
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
    # query_code, _ = vq(descriptors, codebook)
    # query_tf = np.zeros((int(config_value['num-cluster'])))

    # values, counts = np.unique(query_code, return_counts=True)

    # for value, count in zip(values.tolist(), counts.tolist()):
    #     query_tf[value] = count

    # query_tf = query_tf / query_code.shape[0]
    
    relevant = None
    with open(config_value['relevant-path'], 'r') as relevant_file:
        relevant = json.load(relevant_file)
    
    tf_idf, wgc = search(query_image, config_value['detector'], codebook, tf, idf, FeatureIndexerReader(config_value['index']), AngleScaleReader(config_value['angle-scale-path']), delta_angle=config_value['delta-angle'], tf_idf_score_top=int(sys.argv[2]), angle_scale_top=int(sys.argv[3]), debug=True)
    print(len(set(tf_idf).intersection(relevant[sys.argv[1].split('/')[-1]])), len(set(tf_idf).intersection(relevant[sys.argv[1].split('/')[-1]])))
