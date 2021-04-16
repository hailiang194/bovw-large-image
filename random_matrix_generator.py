import numpy as np 
import pickle
import config
from sklearn.preprocessing import StandardScaler

def get_pca_matrix(X, n_features):
    scaled = StandardScaler().fit_transform(X)
    feature = scaled.T
    covariance = np.cov(feature)

    _, vector = np.linalg.eig(covariance)
    return vector[:n_features] 

if __name__ == "__main__":
    config_value = config.get_config() 
    original_cluster = None
    with open(config_value['original-cluster-path'], 'rb') as cluster_file:
        original_cluster = pickle.load(cluster_file)
    
    pca_matrix = get_pca_matrix(original_cluster, int(config_value['binary-dimension']))
    with open(config_value['pca-matrix-path'], 'wb') as pca_file:
        pickle.dump(pca_matrix, pca_file, protocol=pickle.HIGHEST_PROTOCOL)
