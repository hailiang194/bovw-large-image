from features_indexer import FeatureIndexerReader
import numpy as np

reader = FeatureIndexerReader('./index.h5')
print(reader.get("ukbench00925.jpg"))
# for image in reader.keys():
#     desc = np.array(reader.get(image))
#     if desc.shape[0] <= 1:
#         print(image)
