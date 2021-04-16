from features_indexer import FeatureIndexerReader
import config
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import progressbar
import pickle

if __name__ == "__main__":
    config_value = config.get_config()
    reader = FeatureIndexerReader(config_value['index'])
    kmeans = MiniBatchKMeans(n_clusters=int(config_value['num-cluster']),
                             random_state=0, 
                             batch_size=int(config_value['batch-size']),
                             max_iter=int(config_value['max-iterator']))
    cluster_desc = []
    print("[Process]Getting cluster descriptor")
    keys = list(reader.keys())
    keys = keys[0: len(keys) // 4]
    for image in progressbar.progressbar(keys, 0, len(keys)):
        desc = np.array(reader.get(image))
        choise_len = int(desc.shape[0] * float(config_value['percentage']))
        cluster_desc.extend(desc[np.random.choice(desc.shape[0], choise_len), :].tolist())
    reader.close()
    print("[Process]Clustering")
    kmeans.fit(np.array(cluster_desc))
    print("[Process]Saving cluster center")
    with open(config_value['original-cluster-path'], 'wb') as cluster_center_file:
        pickle.dump(kmeans.cluster_centers_, cluster_center_file)
    print("Complete")
