import pickle
from features_indexer import FeatureIndexerReader
from scipy.cluster.vq import vq
import numpy as np 
import config

if __name__ == "__main__":
    config_value = config.get_config()

    reader = FeatureIndexerReader(config_value['index'])
    original_cluster_center = None

    with open(config_value["original-cluster-path"], 'rb') as cluster_file:
        original_cluster_center = pickle.load(cluster_file)

    pca_matrix = None
    with open(config_value['pca-matrix-path'], 'rb') as pca_file:
        pca_matrix = pickle.load(pca_file)

    images = reader.keys()

    for image in images:
        desc = np.array(reader.get(image))
        code, dist = vq(desc, original_cluster_center)
        # nearest_centroid = original_cluster_center[code]
        projection = desc @ pca_matrix.T
        print(projection.shape)
        break
    reader.close()
