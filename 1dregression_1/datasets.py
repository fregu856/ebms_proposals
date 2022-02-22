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
# x = np.random.uniform(low=-3.0, high=3.0, size=(2000, ))
# x = x.astype(np.float32)
#
# y = []
# for x_value in x:
#     if x_value < 0:
#         component = np.random.randint(low=1, high=6) # (1, 2, 3, 4, 5 with 0.5 prob)
#
#         if component in [1, 2, 3, 4]:
#             mu_value = np.sin(x_value)
#             sigma_value = 0.15*(1.0/(1 + 1))
#         elif component == 5:
#             mu_value = -np.sin(x_value)
#             sigma_value = 0.15*(1.0/(1 + 1))
#
#         y_value = np.random.normal(mu_value, sigma_value)
#     else:
#         y_value = np.random.lognormal(0.0, 0.25) - 1.0
#
#     y.append(y_value)
# y = np.array(y, dtype=np.float32)
#
# with open("/root/ebms_proposals/1dregression_1/x.pkl", "wb") as file:
#     pickle.dump(x, file)
# with open("/root/ebms_proposals/1dregression_1/y.pkl", "wb") as file:
#     pickle.dump(y, file)
#
#
#
# num_samples = 2048
# x = np.linspace(-3.0, 3.0, num_samples, dtype=np.float32)
# y_samples = np.linspace(-3.0, 3.0, num_samples) # (shape: (num_samples, ))
# x_values_2_scores = {}
# for x_value in x:
#     if x_value < 0:
#         scores = 0.8*scipy.stats.norm.pdf(y_samples, np.sin(x_value), 0.15*(1.0/(1 + 1))) + 0.2*scipy.stats.norm.pdf(y_samples, -np.sin(x_value), 0.15*(1.0/(1 + 1)))
#     else:
#         scores = scipy.stats.lognorm.pdf(y_samples+1.0, 0.25)
#
#     x_values_2_scores[x_value] = scores
#
# with open("/root/ebms_proposals/1dregression_1/gt_x_values_2_scores.pkl", "wb") as file:
#     pickle.dump(x_values_2_scores, file)
# ################################################################################

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        # ########################################################################
        # x = np.linspace(-3.0, 3.0, 150000, dtype=np.float32)
        # y = []
        # for x_value in x:
        #     if x_value < 0:
        #         component = np.random.randint(low=1, high=6) # (1, 2, 3, 4, 5 with 0.5 prob)
        #
        #         if component in [1, 2, 3, 4]:
        #             mu_value = np.sin(x_value)
        #             sigma_value = 0.15*(1.0/(1 + 1))
        #         elif component == 5:
        #             mu_value = -np.sin(x_value)
        #             sigma_value = 0.15*(1.0/(1 + 1))
        #
        #         y_value = np.random.normal(mu_value, sigma_value)
        #     else:
        #         y_value = np.random.lognormal(0.0, 0.25) - 1.0
        #
        #     y.append(y_value)
        # y = np.array(y, dtype=np.float32)
        #
        # plt.figure(1)
        # plt.plot(x, y, "k.", alpha=0.01, markeredgewidth=0.00001)
        # plt.ylabel("y")
        # plt.xlabel("x")
        # plt.ylim([-1.5, 1.5])
        # plt.savefig("/root/ebms_proposals/1dregression_1/ground_truth.png")
        # plt.close(1)
        # ########################################################################

        with open("/root/ebms_proposals/1dregression_1/x.pkl", "rb") as file: # (needed for python3)
            x = pickle.load(file)

        with open("/root/ebms_proposals/1dregression_1/y.pkl", "rb") as file: # (needed for python3)
            y = pickle.load(file)

        plt.figure(1)
        plt.plot(x, y, "k.")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.ylim([-1.5, 1.5])
        plt.savefig("/root/ebms_proposals/1dregression_1/training_data.png")
        plt.close(1)

        for i in range(x.shape[0]):
            example = {}
            example["x"] = x[i]
            example["y"] = y[i]
            self.examples.append(example)

        self.num_examples = len(self.examples)

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

        x = np.linspace(-3.0, 3.0, 150000, dtype=np.float32)

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

class ToyDatasetEvalKL(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = []

        x = np.linspace(-3.0, 3.0, 2048, dtype=np.float32)

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

# _ = ToyDataset()
