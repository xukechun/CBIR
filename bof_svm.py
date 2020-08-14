# -*- coding: utf-8 -*-
# This script generates bof_svm dictionary
# Created on June 16th, 2020

import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import os
import cv2 as cv
import numpy as np
import joblib
from scipy.cluster.vq import *

TRAIN_PATH = 'svm_train'
SVM_PATH = 'vocabulary_bof_svm.pkl'


def svm_train():
    train_path = TRAIN_PATH
    training_names = os.listdir(train_path)

    # image_paths and the corresponding class label in image_paths
    image_paths = []
    image_classes = []
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
        image_paths += class_path
        image_classes += [class_id] * len(class_path)
        class_id += 1

    # sift process
    sift = cv2.xfeatures2d.SIFT_create()
    des_list = []
    for image_path in image_paths:
        im = cv2.imread(image_path)
        if im is not None:
            im = cv2.resize(im, (300, 300))
        kps = sift.detect(im)
        kps, des = sift.compute(im, kps)
        des_list.append((image_path, des))
        # print("image file path : ", image_path)

    # get descriptors
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # K-Means
    k = 100
    voc, variance = kmeans(descriptors, k, 1)

    # generate feature histogram
    im_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # train the Linear SVM
    clf = LinearSVC(max_iter=2000)
    clf.fit(im_features, np.array(image_classes))

    # save model
    print("training and save model...")
    joblib.dump((clf, training_names, stdSlr, k, voc), SVM_PATH, compress=3)


def predict_image(image_path):
    # Load the classifier, class names, scaler, number of clusters and vocabulary
    clf, classes_names, stdSlr, k, voc = joblib.load(SVM_PATH)
    # sift process
    sift = cv.xfeatures2d.SIFT_create()
    des_list = []
    im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    im = cv.resize(im, (300, 300))
    kps = sift.detect(im)
    kps, des = sift.compute(im, kps)
    des_list.append((image_path, des))

    # get descriptors
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[0:]:
        descriptors = np.vstack((descriptors, descriptor))

    test_features = np.zeros((1, k), "float32")
    words, distance = vq(des_list[0][1], voc)
    for w in words:
        test_features[0][w] += 1

    # Tf-Idf vectorization
    nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # scale the features
    test_features = stdSlr.transform(test_features)

    # predicted class
    predictions = [classes_names[i] for i in clf.predict(test_features)]
    return predictions
