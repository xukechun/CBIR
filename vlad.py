# -*- coding: utf-8 -*-
# This script generates vlad dictionary
# Created on June 15th, 2020

from VLAD_lib import *
import pickle
from Search_images import FEATURE_METHOD

IMAGE_PATH = 'ukbench500'
DISCR_PATH = 'describe_surf.pickle'
VD_PATH = 'visdict_surf.pickle'
VLAD_PATH = 'vlad_dict_surf.pickle'
# MinBatchKMeans parameter
CLUSTER_NUM = 100

class Vlad:
    def __init__(self):
        self.k = CLUSTER_NUM

    def generate_describer(self):
        descriptors = getDescriptors(IMAGE_PATH, FEATURE_METHOD)
        with open(DISCR_PATH, 'wb') as f:
            pickle.dump(descriptors, f)

    def generate_visual_dict(self):
        # computing the visual dictionary
        with open(DISCR_PATH, 'rb') as f:
            descriptors = pickle.load(f)
        visualDictionary = kMeansDictionary(descriptors, self.k)
        with open(VD_PATH, 'wb') as f:
            pickle.dump(visualDictionary, f)
        print("The visual dictionary  is saved in " + VD_PATH)

    def generate_vlad_dict(self):
        # estimating VLAD descriptors for the whole dataset
        with open(VD_PATH, 'rb') as f:
            visualDictionary = pickle.load(f)
        # computing the VLAD descriptors
        V, idImages = getVLADDescriptors(IMAGE_PATH, FEATURE_METHOD, visualDictionary)
        # output
        with open(VLAD_PATH, 'wb') as f:
            pickle.dump([idImages, V, IMAGE_PATH], f)
        print("The VLAD descriptors are saved in " + VLAD_PATH)


if __name__ == '__main__':
    vlad = Vlad()
    vlad.generate_describer()
    vlad.generate_visual_dict()
    vlad.generate_vlad_dict()


