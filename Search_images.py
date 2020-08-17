# -*- coding: utf-8 -*-
# This script complement image retrieval with three methods: BOF、VLAD and BOF_SVM
# Created on June 14th, 2020

import pickle
from localdescriptors import sift
from imagesearch import imagesearch
from PCV.geometry import homography
from tools.imtools import get_imlist
from PIL import Image
import matplotlib.pyplot as plt
from VLAD_lib import *
from bof_svm import *
from bof_vgg import *
import argparse
import glob

# parser
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
                help="Index of a query image(BOF and VLAD within 500, BOF_SVM within 300 while BOF_VGG within 50)")
ap.add_argument("-r", "--retrieve", required=True,
                help="number of images to retrieve")
ap.add_argument("-m", "--method", required=True,
                help="method to use(0:BOF,1:VLAD,2:BOF_SVM,3:BOF_VGG)")

args = vars(ap.parse_args())

# args
q_ind = int(args["query"])
nbr_results = int(args["retrieve"])
METHOD_ID = int(args["method"])

IMAGE_PATH = 'ukbench500/'

BOF = 0
VLAD = 1
BOF_SVM = 2
BOF_VGG = 3
METHOD = ['BOF', 'VLAD', 'BOF_SVM', 'BOF_VGG']
FEATURE_METHOD = [sift_process, surf_process]
VD_PATH = ['visdict_sift.pickle', 'visdict_surf.pickle']
VLAD_PATH = ['vlad_dict_sift.pickle', 'vlad_dict_surf.pickle']
feature_ind = 0
FEATURE_METHOD = FEATURE_METHOD[feature_ind]
VD_PATH = VD_PATH[feature_ind]
VLAD_PATH = VLAD_PATH[feature_ind]
Vocabulary_path_list = ['vocabulary_bof.pkl', 'vlad_dict_sift.pkl', 'vocabulary_bof_svm.pkl']
Database_path = 'index_bof.db'


class Search_images:
    def __init__(self):
        self.image_path = IMAGE_PATH
        self.vocabulary_path = []
        self.method = METHOD_ID
        self.bof_rearrange = 0
        # load image list and vocabulary
        self.imlist = get_imlist(self.image_path)
        self.image_num = len(self.imlist)
        # load feature list
        self.featlist = [self.imlist[i][:-3] + 'sift' for i in range(self.image_num)]

    def image_searcher(self):
        # load_vocabularies
        with open('vocabulary_bof.pkl', 'rb') as f:
            voc = pickle.load(f)
        src = imagesearch.Searcher(Database_path, voc)
        return src
    
    def load_query_feature(self):
        # load image features for query image
        q_locs, q_descr = sift.read_features_from_file(self.featlist[q_ind])
        q_locs = np.array(q_locs)
        fp = homography.make_homog(q_locs[:, :2].T)
        return q_descr, fp

    def plot_results(self, res, match_scores):
        # show the top six match images
        plt.figure()
        plt.suptitle('Search Results with ' + METHOD[self.method])
        nbr_results = len(res)
        imname = get_filename(IMAGE_PATH, q_ind)
        ax = plt.subplot(3, 3, 1)
        plt.imshow(np.array(Image.open(imname)))
        plt.xticks([])
        plt.yticks([])
        ax.set_title('queried image', fontsize=10)
        for i in range(nbr_results):
            imname = get_filename(IMAGE_PATH, res[i])
            ax = plt.subplot(3, 3, i + 4)
            plt.imshow(np.array(Image.open(imname)))
            plt.xticks([])
            plt.yticks([])
            ax.set_title(str(match_scores[i])[0:5], fontsize=10)
        plt.show()

    def plot_svm_results(self, path, class_name):
        # show the corresponding class images
        train_path = 'svm_train/'
        class_path = train_path + class_name
        imgs_name = glob.glob(class_path + "/*.jpg")
        imgs_name = imgs_name[:6]
        plt.figure()
        plt.suptitle('Search Results with ' + METHOD[self.method])
        nbr_results = len(imgs_name)
        ax = plt.subplot(3, 3, 1)
        plt.imshow(np.array(Image.open(path)))
        plt.xticks([])
        plt.yticks([])
        ax.set_title('queried image', fontsize=10)
        for i in range(nbr_results):
            ax = plt.subplot(3, 3, i + 4)
            plt.imshow(np.array(Image.open(imgs_name[i])))
            plt.xticks([])
            plt.yticks([])
            if(i==1):
                ax.set_title('predicted class is ' + str(class_name), fontsize=10)
        plt.show()

    def bof_image_retrieval(self):
        # load vocabulary and query feature
        src = self.image_searcher()
        q_descr, fp = self.load_query_feature()
        # RANSAC model for homography fitting
        model = homography.RansacModel()
        rank = {}
        # query
        match_scores = [w[0] for w in src.query(self.imlist[q_ind])[:nbr_results]]
        res_reg = [w[1] for w in src.query(self.imlist[q_ind])[:nbr_results]]
        print('top matches:', res_reg)
        self.plot_results(res_reg[:6], match_scores[:6])
        if self.bof_rearrange:
            # load image features for result
            for ndx in res_reg[1:]:
                locs, descr = sift.read_features_from_file(self.featlist[ndx])
                # get matches
                matches = sift.match(q_descr, descr)
                ind = matches.nonzero()[0]
                ind2 = matches[ind]
                locs = np.array(locs)
                tp = homography.make_homog(locs[:, :2].T)
                # compute homography, count inliers.
                try:
                    H, inliers = homography.H_from_ransac(fp[:, ind], tp[:, ind2], model, match_theshold=4)
                except:
                    inliers = []
                # store inlier count
                rank[ndx] = len(inliers)
                # sort dictionary to get the most inliers first
                sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
                res_geom = [res_reg[0]] + [s[0] for s in sorted_rank]
            # print('top matches (homography):', res_geom)
            # show results
            self.plot_results(res_geom[:6], match_scores[:6])

    def vlad_image_retrieval(self):
        path = get_filename(IMAGE_PATH, q_ind)
        with open(VLAD_PATH, 'rb') as f:
            vladDescriptors = pickle.load(f)
        # load the visual dictionary
        with open(VD_PATH, 'rb') as f:
            visualDictionary = pickle.load(f)
        v_set = vladDescriptors[1]
        # print(len(v_set))
        # computing descriptors
        dist, ind = query(path, nbr_results, visualDictionary, v_set, FEATURE_METHOD)
        # print(dist)
        print('top matches:', ind)
        self.plot_results(ind[:6], dist[:6])

    def bof_svm_image_retrieval(self):
        test_path = 'svm_test/'
        testing_names = os.listdir(test_path)
        image_paths = []
        for testing_name in testing_names:
            dir = os.path.join(test_path, testing_name)
            class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
            image_paths += class_path
        image_path = image_paths[q_ind]
        # predict class
        prediction = predict_image(image_path)
        print("image: %s, classes : %s" % (image_path, prediction))
        # show results
        self.plot_svm_results(image_path, prediction[0])

    def bof_vgg_image_retrieval(self):
        search_engine = create_SearchEngine()
        demo_im, demo_bb = catch_BoundingBox(q_ind)
        retrieve_object(search_engine, demo_im, demo_bb, nbr_results)


    def search_similar_images(self):
        if self.method == BOF:
            self.bof_image_retrieval()
        elif self.method == VLAD:
            self.vlad_image_retrieval()
        elif self.method == BOF_SVM:
            self.bof_svm_image_retrieval()
        else:
            self.bof_vgg_image_retrieval()


if __name__ == '__main__':
    cbir = Search_images()
    cbir.search_similar_images()
