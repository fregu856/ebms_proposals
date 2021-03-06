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
# import os
#
# img_filenames = os.listdir("/root/ebms_proposals/mdn_cell/Cell200")
# print (img_filenames)
# print (len(img_filenames))
#
# img = cv2.imread("/root/ebms_proposals/mdn_cell/Cell200/" + img_filenames[0], -1)
# print (img)
# print (img.shape)
# print (img.dtype)
#
# num_examples = len(img_filenames)
# print (num_examples)
#
# np.random.shuffle(img_filenames)
# np.random.shuffle(img_filenames)
# np.random.shuffle(img_filenames)
# np.random.shuffle(img_filenames)
#
# num_imgs_train = 10000
# num_imgs_test = 10000
# img_filenames_train = img_filenames[0:num_imgs_train]
# img_filenames_test = img_filenames[num_imgs_train:(num_imgs_train+num_imgs_test)]
# print (len(img_filenames_train))
# print (len(img_filenames_test))
#
# images_train = np.zeros((num_imgs_train, img.shape[0], img.shape[1]), dtype=img.dtype)
# images_test = np.zeros((num_imgs_test, img.shape[0], img.shape[1]), dtype=img.dtype)
# print (images_train.shape)
# print (images_train.dtype)
# print (images_test.shape)
# print (images_test.dtype)
#
# labels_train = []
# for (i, img_filename) in enumerate(img_filenames_train):
#     img = cv2.imread("/root/ebms_proposals/mdn_cell/Cell200/" + img_filename, -1)
#     images_train[i] = img
#
#     label = float(img_filename.split("_")[1].split(".0.")[0])
#     labels_train.append(label)
# labels_train = np.array(labels_train).astype(np.float32)
#
# labels_test = []
# for (i, img_filename) in enumerate(img_filenames_test):
#     img = cv2.imread("/root/ebms_proposals/mdn_cell/Cell200/" + img_filename, -1)
#     images_test[i] = img
#
#     label = float(img_filename.split("_")[1].split(".0.")[0])
#     labels_test.append(label)
# labels_test = np.array(labels_test).astype(np.float32)
#
# print (labels_train.shape)
# print (images_train.shape)
# print (labels_test.shape)
# print (images_test.shape)
#
# with open("/root/ebms_proposals/mdn_cell/labels_train.pkl", "wb") as file:
#     pickle.dump(labels_train, file)
# with open("/root/ebms_proposals/mdn_cell/images_train.pkl", "wb") as file:
#     pickle.dump(images_train, file)
#
# with open("/root/ebms_proposals/mdn_cell/labels_test.pkl", "wb") as file:
#     pickle.dump(labels_test, file)
# with open("/root/ebms_proposals/mdn_cell/images_test.pkl", "wb") as file:
#     pickle.dump(images_test, file)
# ################################################################################

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/ebms_proposals/mdn_cell/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/ebms_proposals/mdn_cell/images_train.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64))
        img = np.expand_dims(img, axis=2) # (shape: (64, 64, 1))
        img = img*np.ones((img.shape[0], img.shape[1], 3), dtype=img.dtype) # (shape: (64, 64, 3))

        # cv2.imwrite("/root/ebms_proposals/mdn_cell/%d_%f.png" % (index, angle), img)

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/ebms_proposals/mdn_cell/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/ebms_proposals/mdn_cell/images_test.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64))
        img = np.expand_dims(img, axis=2) # (shape: (64, 64, 1))
        img = img*np.ones((img.shape[0], img.shape[1], 3), dtype=img.dtype) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

# dataset_train = DatasetTrain()
# for i in range(10):
    # _, _ = dataset_train.__getitem__(i)
# _ = DatasetTest()
