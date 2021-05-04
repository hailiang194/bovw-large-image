import numpy as np
from features_indexer import FeatureIndexerReader
import config
from scipy.cluster.vq import vq
import pickle
import json
import progressbar

def get_binary_signature(image, desc, codebook, median, node_list):
    code, _ = vq(desc, codebook)
    
    for code_index in range(code.shape[0]):
        bits = 0
        for bit_index in range(median.shape[1]):
            bits = bits | int(desc[code_index, bit_index] > median[code[code_index], bit_index])
            bits = bits << 1
    
        node = node_list[code[code_index]]
        if image in node.keys():
            node[image].append(bits)
        else:
            node[image] = [bits]

if __name__ == "__main__":
    config_value = config.get_config()
    
    reader = FeatureIndexerReader(config_value['index'])

    images = list(reader.keys())

    codebook = None
    with open(config_value['original-cluster-path'],'rb') as codebook_file:
        codebook = pickle.load(codebook_file)

    median = None
    with open(config_value['median-path'], 'rb') as median_file:
        median = pickle.load(median_file)


    node_list = {a:dict() for a in range(codebook.shape[0])}

    for image in progressbar.progressbar(images, 0, len(images)):
        desc = np.array(reader.get(image))
        get_binary_signature(image, desc, codebook, median, node_list)
   
    with open(config_value['signature-path'], 'w') as signature_file:
        json.dump(node_list, signature_file)
