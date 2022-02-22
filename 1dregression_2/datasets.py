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

# ################################################################################
# # run this once to generate the training data:
# ################################################################################
# lst = []
#
# np.random.seed(41)
#
# size = 1000
#
# points = np.random.beta(0.5,1,8*size//10)*5+0.5
#
# np.random.shuffle(points)
# lst += points.tolist()
# zones = [[len(lst),'Asymmetric']]
#
# points = 3*np.cos(np.linspace(0,5,num=size))-2
# points = points+np.random.normal(scale=np.abs(points)/4,size=size)
# lst += points.tolist()
# zones += [[len(lst),'Symmetric']]
#
# lst += [np.random.uniform(low=i,high=j)
#         for i,j in zip(np.linspace(-2,-4.5,num=size//2),
#                        np.linspace(-0.5,9.,num=size//2))]
#
# zones += [[len(lst),'Uniform']]
#
# points = np.r_[8+np.random.uniform(size=size//2)*0.5,
#                1+np.random.uniform(size=size//2)*3.,
#              -4.5+np.random.uniform(size=-(-size//2))*1.5]
#
# np.random.shuffle(points)
#
# lst += points.tolist()
# zones += [[len(lst),'Multimodal']]
#
# y_train_synthetic = np.array(lst).reshape(-1,1)
# x_train_synthetic = np.arange(y_train_synthetic.shape[0]).reshape(-1,1)
# x_train_synthetic = x_train_synthetic/x_train_synthetic.max()
#
# disord = np.arange(y_train_synthetic.shape[0])
# np.random.shuffle(disord)
#
# x_train_synthetic = x_train_synthetic[disord]
# y_train_synthetic = y_train_synthetic[disord]
#
# # Train = 45%, Validation = 5%, Test = 50%
#
# x_test_synthetic = x_train_synthetic[:x_train_synthetic.shape[0]//2]
# y_test_synthetic = y_train_synthetic[:x_train_synthetic.shape[0]//2]
# y_train_synthetic = y_train_synthetic[x_train_synthetic.shape[0]//2:]
# x_train_synthetic = x_train_synthetic[x_train_synthetic.shape[0]//2:]
#
# x_valid_synthetic = x_train_synthetic[:x_train_synthetic.shape[0]//10]
# y_valid_synthetic = y_train_synthetic[:x_train_synthetic.shape[0]//10]
# y_train_synthetic = y_train_synthetic[x_train_synthetic.shape[0]//10:]
# x_train_synthetic = x_train_synthetic[x_train_synthetic.shape[0]//10:]
#
# plt.figure(figsize=(15,7))
#
# plt.plot(x_valid_synthetic,y_valid_synthetic,'o',label='validation points')
# plt.plot(x_train_synthetic,y_train_synthetic,'o',label='training points',alpha=0.2)
# plt.plot(x_test_synthetic,y_test_synthetic,'o',label='testing points',alpha=0.2)
# for i in range(len(zones)):
#     if i!= len(zones)-1:
#         plt.axvline(x=zones[i][0]/len(lst),linestyle='--',c='grey')
#     if i==0:
#         plt.text(x=(zones[i][0])/(2*len(lst)),y=y_train_synthetic.min()-0.5,
#                  s=zones[i][1], horizontalalignment='center', fontsize=20, color='grey')
#     else:
#         plt.text(x=(zones[i-1][0]+zones[i][0])/(2*len(lst)),y=y_train_synthetic.min()-0.5,
#                  s=zones[i][1], horizontalalignment='center', fontsize=20, color='grey')
#
# plt.legend(loc="lower left", bbox_to_anchor=(0.,0.1))
# plt.savefig("/root/ebms_proposals/1dregression_2/data.png")
#
# print(x_train_synthetic.shape)
# print(x_valid_synthetic.shape)
# print(x_test_synthetic.shape)
#
# with open("/root/ebms_proposals/1dregression_2/x_train.pkl", "wb") as file:
#     pickle.dump(x_train_synthetic, file)
# with open("/root/ebms_proposals/1dregression_2/y_train.pkl", "wb") as file:
#     pickle.dump(y_train_synthetic, file)
#
# with open("/root/ebms_proposals/1dregression_2/x_val.pkl", "wb") as file:
#     pickle.dump(x_valid_synthetic, file)
# with open("/root/ebms_proposals/1dregression_2/y_val.pkl", "wb") as file:
#     pickle.dump(y_valid_synthetic, file)
#
# with open("/root/ebms_proposals/1dregression_2/x_test.pkl", "wb") as file:
#     pickle.dump(x_test_synthetic, file)
# with open("/root/ebms_proposals/1dregression_2/y_test.pkl", "wb") as file:
#     pickle.dump(y_test_synthetic, file)
# ################################################################################

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        with open("/root/ebms_proposals/1dregression_2/x_train.pkl", "rb") as file: # (needed for python3)
            x = pickle.load(file)
        x = x.astype(np.float32)

        with open("/root/ebms_proposals/1dregression_2/y_train.pkl", "rb") as file: # (needed for python3)
            y = pickle.load(file)
        y = y.astype(np.float32)

        plt.figure(1)
        plt.plot(x, y, "k.")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.ylim([-10.0, 10.0])
        plt.savefig("/root/ebms_proposals/1dregression_2/training_data.png")
        plt.close(1)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i][0]
            example["y"] = y[i][0]
            self.examples.append(example)

        self.num_examples = len(self.examples)
        print(self.num_examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

class ToyDatasetEval(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        x = np.linspace(0.0, 1.0, 150000, dtype=np.float32)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]

        return (x)

    def __len__(self):
        return self.num_examples

class ToyDatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        with open("/root/ebms_proposals/1dregression_2/x_test.pkl", "rb") as file: # (needed for python3)
            x = pickle.load(file)
        x = x.astype(np.float32)

        with open("/root/ebms_proposals/1dregression_2/y_test.pkl", "rb") as file: # (needed for python3)
            y = pickle.load(file)
        y = y.astype(np.float32)

        plt.figure(1)
        plt.plot(x, y, "k.")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.ylim([-10.0, 10.0])
        plt.savefig("/root/ebms_proposals/1dregression_2/testing_data.png")
        plt.close(1)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i][0]
            example["y"] = y[i][0]
            self.examples.append(example)

        self.num_examples = len(self.examples)
        print(self.num_examples)

    def __getitem__(self, index):
        example = self.examples[index]

        x = example["x"]
        y = example["y"]

        return (x, y)

    def __len__(self):
        return self.num_examples

#_ = ToyDataset()
#_ = ToyDatasetTest()
