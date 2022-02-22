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
import random

class DatasetTrainAug(torch.utils.data.Dataset):
    def __init__(self):
        biwi_file_path = "/root/data/BIWI/BIWI_train.npz"

        biwi_file = np.load(biwi_file_path)
        self.imgs = biwi_file["image"] # (shape: (10613, 64, 64, 3))
        self.poses = biwi_file["pose"] # (shape: (10613, 3)) (Yaw, Pitch, Roll)

        print (self.poses.shape)
        print (self.imgs.shape)

        self.crop_size = 64

        self.num_examples = self.imgs.shape[0]

        print ("DatasetTrainAug - number of images: %d" % self.num_examples)
        print ("DatasetTrainAug - max Yaw: %g" % np.max(self.poses[:, 0]))
        print ("DatasetTrainAug - min Yaw: %g" % np.min(self.poses[:, 0]))
        print ("DatasetTrainAug - max Pitch: %g" % np.max(self.poses[:, 1]))
        print ("DatasetTrainAug - min Pitch: %g" % np.min(self.poses[:, 1]))
        print ("DatasetTrainAug - max Roll: %g" % np.max(self.poses[:, 2]))
        print ("DatasetTrainAug - min Roll: %g" % np.min(self.poses[:, 2]))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        img = self.imgs[index] # (shape: (64, 64, 3))
        pose = self.poses[index] # (3, ) (Yaw, Pitch, Roll)

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

       # flip img along the vertical axis with 0.5 probability:
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)
            pose[0] = -1.0*pose[0]
            pose[2] = -1.0*pose[2]

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (flip)
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # scale the size of the image with factor in [0.7, 1.4]:
        f_scale = 0.7 + random.randint(0, 8)/10.0
        img = cv2.resize(img, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (f_scale)
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # pad the image if needed:
        img_h, img_w, _ = img.shape
        pad_h = max(self.crop_size - img_h, 0)
        pad_w = max(self.crop_size - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # print ("#####")
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        # select a random (64, 64) crop:
        img_h, img_w, _ = img.shape
        h_off = random.randint(0, img_h - self.crop_size)
        w_off = random.randint(0, img_w - self.crop_size)
        img = img[h_off:(h_off+self.crop_size), w_off:(w_off+self.crop_size)] # (shape: (crop_size, crop_size, 3))

        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization START
        # print (pose)
        # print (img.shape)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # # # # # # # # # # # # # # # # # # # # # # # # debug visualization END

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        pose = pose.astype(np.float32)

        return (img, pose)

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        biwi_file_path = "/root/data/BIWI/BIWI_test.npz"

        biwi_file = np.load(biwi_file_path)
        self.imgs = biwi_file["image"] # (shape: (5065, 64, 64, 3))
        self.poses = biwi_file["pose"] # (shape: (5065, 3)) (Yaw, Pitch, Roll)

        print (self.poses.shape)
        print (self.imgs.shape)

        self.num_examples = self.imgs.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)
        print ("DatasetTest - max Yaw: %g" % np.max(self.poses[:, 0]))
        print ("DatasetTest - min Yaw: %g" % np.min(self.poses[:, 0]))
        print ("DatasetTest - max Pitch: %g" % np.max(self.poses[:, 1]))
        print ("DatasetTest - min Pitch: %g" % np.min(self.poses[:, 1]))
        print ("DatasetTest - max Roll: %g" % np.max(self.poses[:, 2]))
        print ("DatasetTest - min Roll: %g" % np.min(self.poses[:, 2]))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        img = self.imgs[index] # (shape: (64, 64, 3))
        pose = self.poses[index] # (3, ) (Yaw, Pitch, Roll)

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        pose = pose.astype(np.float32)

        return (img, pose)

# _ = DatasetTrainAug()
# _ = DatasetTest()
