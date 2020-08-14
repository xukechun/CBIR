# -*- coding: utf-8 -*-
# This script contains several function used in vlad
# Created on June 15th, 2020

import numpy as np
import itertools
from sklearn.cluster import MiniBatchKMeans
import glob
import cv2
import heapq


def sift_process(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


def surf_process(image):
    surf = cv2.xfeatures2d.SURF_create()
    # it is better to have this value between 300 and 500
    surf.setHessianThreshold(400)
    kp, des = surf.detectAndCompute(image, None)
    return kp, des


def get_filename(path, ind):
    ima_name = glob.glob(path + "/*.jpg")
    return ima_name[ind]


def getDescriptors(path, functionHandleDescriptor):
    descriptors=list()
    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
        im = cv2.imread(imagePath)
        kp, des = functionHandleDescriptor(im)
        descriptors.append(des)
        print(len(kp))
    # flatten list
    descriptors = list(itertools.chain.from_iterable(descriptors))
    # list to array
    descriptors = np.asarray(descriptors)

    return descriptors


def kMeansDictionary(training, k):
    # K-means
    # est = KMeans(n_clusters=k, init='k-means++', tol=0.1, verbose=1).fit(training)
    # use MiniBatchKMeans to speed up
    est = MiniBatchKMeans(init='k-means++', n_clusters=1000, max_iter=1000, batch_size=100,
                    n_init=10, max_no_improvement=10, verbose=1).fit(training)
    return est


def getVLADDescriptors(path, functionHandleDescriptor, visualDictionary):
    descriptors = list()
    idImage = list()
    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
        im = cv2.imread(imagePath)
        kp, des = functionHandleDescriptor(im)
        v = VLAD(des, visualDictionary)
        descriptors.append(v)
        idImage.append(imagePath)
    # list to array
    descriptors = np.asarray(descriptors)
    return descriptors, idImage


def VLAD(X, visualDictionary):
    # compute vlad
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    k = visualDictionary.n_clusters
   
    m, d = X.shape
    V = np.zeros([k, d])

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels == i) > 0:
            # add the diferences
            V[i] = np.sum(X[predictedLabels == i, :]-centers[i], axis=0)

    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization
    V = V/np.sqrt(np.dot(V, V))
    return V


def query(image, k, visualDictionary, v_set, FEATURE_METHOD):
    # read image
    im = cv2.imread(image)
    # compute descriptors
    kp, descriptor = FEATURE_METHOD(im)
    # compute VLAD
    v = VLAD(descriptor, visualDictionary)
    # find the k most relevant images
    match_score = []
    dist = []
    for i in range(len(v_set)):
        match_score.append(np.sqrt(np.sum((v - v_set[i]) ** 2)))
    # search the nearest k images
    ind = map(match_score.index, heapq.nsmallest(k, match_score))
    ind = list(ind)
    for i in range(len(ind)):
        dist.append(match_score[ind[i]])
    return dist, ind





