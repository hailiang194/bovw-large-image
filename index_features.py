import json
import config
from imutils import paths
import cv2
from features_indexer import FeatureIndexerWriter
import progressbar

if __name__ == "__main__":
    config_value = config.get_config()
    index_writer = FeatureIndexerWriter(config_value['index'])
    image_paths = list(paths.list_images(config_value['dataset-path']))
    for path in progressbar.progressbar(image_paths, 0, len(image_paths)):
        image_file_name = path.split('/')[-1]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        kp, desc = config_value['detector'].detectAndCompute(image, None)
        index_writer.add_image_descriptors(image_file_name, desc)
    print("Completed")
