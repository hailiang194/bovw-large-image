import json
import cv2

DEFAULT_CONFIG_FILE = "./config.json"

DETECTOR = {
    "SIFT": cv2.SIFT_create,
    "ORB": cv2.ORB_create
}

def get_detector(detector_name):
    return DETECTOR[detector_name]()

def get_config(config_path=DEFAULT_CONFIG_FILE):
    config = None
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    config['detector'] = get_detector(config['detector'])
    return config
