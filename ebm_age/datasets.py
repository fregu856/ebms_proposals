# camera-ready

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import scipy.stats

import pickle

import cv2

# ################################################################################
# # run this once to generate the training data:
# ################################################################################
# import h5py
# hf = h5py.File("/root/ebms_proposals/ebm_age/UTKFace_64x64.h5", 'r')
# labels = hf['labels'][:]
# labels = labels.astype(np.float32)
# images = hf['images'][:]
# hf.close()
#
# print (images.shape)
# print (labels.shape)
#
# inds_filtered = []
# for i in range(labels.shape[0]):
#     if (labels[i] >= 1) and (labels[i] <= 60):
#         inds_filtered.append(i)
#
# labels = labels[inds_filtered]
# images = images[inds_filtered]
# print (images.shape)
# print (labels.shape)
# num_examples = labels.shape[0]
# print (num_examples)
#
# inds = list(range(num_examples))
#
# np.random.shuffle(inds)
# np.random.shuffle(inds)
# np.random.shuffle(inds)
# np.random.shuffle(inds)
#
# inds_train = inds[0:int(0.8*num_examples)]
# inds_test = inds[int(0.8*num_examples):]
#
# labels_train = labels[inds_train]
# images_train = images[inds_train]
# labels_test = labels[inds_test]
# images_test = images[inds_test]
# print (labels_train.shape)
# print (images_train.shape)
# print (labels_test.shape)
# print (images_test.shape)
#
# with open("/root/ebms_proposals/ebm_age/labels_train.pkl", "wb") as file:
#     pickle.dump(labels_train, file)
# with open("/root/ebms_proposals/ebm_age/images_train.pkl", "wb") as file:
#     pickle.dump(images_train, file)
#
# with open("/root/ebms_proposals/ebm_age/labels_test.pkl", "wb") as file:
#     pickle.dump(labels_test, file)
# with open("/root/ebms_proposals/ebm_age/images_test.pkl", "wb") as file:
#     pickle.dump(images_test, file)
# ################################################################################

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/ebms_proposals/ebm_age/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/ebms_proposals/ebm_age/images_train.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        print (np.min(self.labels))
        print (np.max(self.labels))

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (3, 64, 64))
        img = np.transpose(img, (1, 2, 0)) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 200, 200))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/ebms_proposals/ebm_age/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/ebms_proposals/ebm_age/images_test.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        print (np.min(self.labels))
        print (np.max(self.labels))

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (3, 64, 64))
        img = np.transpose(img, (1, 2, 0)) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 200, 200))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

# _ = DatasetTrain()
# _ = DatasetTest()
