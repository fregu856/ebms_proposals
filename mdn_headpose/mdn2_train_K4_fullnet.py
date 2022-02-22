from datasets import DatasetTrainAug # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from mdn_model_K4 import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributions

import math
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# NOTE! change this to not overwrite all log data when you train the model:
model_id = "mdn2_train_K4_fullnet"

num_epochs = 75
batch_size = 32
learning_rate = 0.001

train_dataset = DatasetTrainAug()

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

num_models = 20
for i in range(num_models):
    network = ToyNet(model_id + "_%d" % i, project_dir="/root/project5/headpose").cuda()

    K = network.noise_net.K
    print (K)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    epoch_losses_train = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("model: %d/%d  |  epoch: %d/%d" % (i+1, num_models, epoch+1, num_epochs))

        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        for step, (xs, ys) in enumerate(train_loader):
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size, 3))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            if epoch < 20:
                ####################################################################
                # make sure we do NOT train the resnet feature extractor:
                ####################################################################
                x_features = x_features.detach()
                ####################################################################
            means, log_sigma2s, weights = network.noise_net(x_features)
            # (means has shape: (batch_size, 3K))
            # (log_sigma2s has shape: (batch_size, 3K))
            # (weights has shape: (batch_size, K))
            sigmas = torch.exp(log_sigma2s/2.0) # (shape: (batch_size, 3K))
            means = means.view(-1, 3, K) # (shape: (batch_size, 3, K))
            sigmas = sigmas.view(-1, 3, K) # (shape: (batch_size, 3, K))

            q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
            # q_ys_K = torch.exp(q_distr.log_prob(ys.unsqueeze(2)).sum(1)) # (shape: (batch_size, K)
            # q_ys = torch.sum(weights*q_ys_K, dim=1) # (shape: (batch_size))

            log_q_ys_K = q_distr.log_prob(ys.unsqueeze(2)).sum(1) # (shape: (batch_size, K)
            log_q_ys = torch.logsumexp(torch.log(weights) + log_q_ys_K, dim=1) # (shape: (batch_size))

            ########################################################################
            # compute loss:
            ########################################################################
            # q_ys = F.relu(q_ys - 1.0e-6) + 1.0e-6

            # loss = torch.mean(-torch.log(q_ys))
            loss = torch.mean(-log_q_ys)

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

            # print ("model: %d/%d  |  epoch: %d/%d  |  step: %d/%d  |  loss: %g" % (i, num_models-1, epoch+1, num_epochs, step+1, num_train_batches, loss_value))

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        plt.close(1)

    # save the model weights to disk:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
