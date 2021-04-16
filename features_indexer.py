import h5py

class FeatureIndexerWriter(object):
    def __init__(self, path):
        self.__h5_file = h5py.File(path, "w")
            
    def add_image_descriptors(self, image_path, descriptors):
        self.__h5_file.create_dataset(image_path, data=descriptors)

    def close(self):
        try:
            self.__h5_file.close()
        except:
            return

    def __del__(self):
        self.close()


class FeatureIndexerReader(object):
    def __init__(self, path):
        self.__h5_file = h5py.File(path, "r")
            
    def keys(self):
        return self.__h5_file.keys()

    def get(self, image):
        return self.__h5_file.get(image)

    def close(self):
        try: 
            self.__h5_file.close()
        except:
            return

    def __del__(self):
        self.close()

