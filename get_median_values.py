from features_indexer import FeatureIndexerReader
import pickle
import config
import numpy as np
from scipy.cluster.vq import vq
from statistics import median
from progressbar import progressbar

if __name__ == "__main__":
    config_value = config.get_config()
    reader = FeatureIndexerReader(config_value['index'])

    codebook = None
    with open(config_value['original-cluster-path'], 'rb') as codebook_file:
        codebook = pickle.load(codebook_file)

    print(codebook.shape)
    desc_cell_set = [[set() for _ in range(codebook.shape[1])] for _ in range(codebook.shape[0])]

    images = list(reader.keys())
    for image in progressbar(images, 0, len(images)):
        desc = np.array(reader.get(image))
        code, _ = vq(desc, codebook)
        for code_value in range(codebook.shape[0]):
            code_desc = desc[np.where(code==code_value)]
            if code_desc.shape[0] > 0:
                for i in range(codebook.shape[1]):
                    values = set(code_desc[:, i].astype(int).tolist())
                    desc_cell_set[code_value][i] = desc_cell_set[int(code_value)][i].union(values)
    for r_i in range(len(desc_cell_set)):
        for c_i in range(len(desc_cell_set[r_i])):
            desc_cell_set[r_i][c_i] = median(desc_cell_set[r_i][c_i]) if len(desc_cell_set[r_i][c_i]) > 0 else 0
    with open(config_value['median-path'], 'wb') as median_file:
        pickle.dump(np.array(desc_cell_set), median_file)
