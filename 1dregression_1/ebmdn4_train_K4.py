# camera-ready

from datasets import ToyDataset # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from ebmdn_model_K4 import ToyNet

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
model_id = "ebmdn4_train_K4"

num_epochs = 75
batch_size = 32
learning_rate = 0.001

num_samples = 1024
print (num_samples)

train_dataset = ToyDataset()

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

num_models = 20
for i in range(num_models):
    network = ToyNet(model_id + "_%d" % i, project_dir="/root/ebms_proposals/1dregression_1").cuda()

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
            xs = xs.cuda().unsqueeze(1) # (shape: (batch_size, 1))
            ys = ys.cuda().unsqueeze(1) # (shape: (batch_size, 1))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            means, log_sigma2s, weights = network.noise_net(x_features.detach()) # (all have shape: (batch_size, K))
            sigmas = torch.exp(log_sigma2s/2.0) # (shape: (batch_size, K))

            q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
            q_ys_K = torch.exp(q_distr.log_prob(torch.transpose(ys, 1, 0).unsqueeze(2))) # (shape: (1, batch_size, K))
            q_ys = torch.sum(weights.unsqueeze(0)*q_ys_K, dim=2) # (shape: (1, batch_size))
            q_ys = q_ys.squeeze(0) # (shape: (batch_size))

            y_samples_K = q_distr.sample(sample_shape=torch.Size([num_samples])) # (shape: (num_samples, batch_size, K))
            inds = torch.multinomial(weights, num_samples=num_samples, replacement=True).unsqueeze(2) # (shape: (batch_size, num_samples, 1))
            inds = torch.transpose(inds, 1, 0) # (shape: (num_samples, batch_size, 1))
            y_samples = y_samples_K.gather(2, inds).squeeze(2) # (shape: (num_samples, batch_size))
            y_samples = y_samples.detach()
            q_y_samples_K = torch.exp(q_distr.log_prob(y_samples.unsqueeze(2))) # (shape: (num_samples, batch_size, K))
            q_y_samples = torch.sum(weights.unsqueeze(0)*q_y_samples_K, dim=2) # (shape: (num_samples, batch_size))
            y_samples = torch.transpose(y_samples, 1, 0) # (shape: (batch_size, num_samples))
            q_y_samples = torch.transpose(q_y_samples, 1, 0) # (shape: (batch_size, num_samples))

            scores_gt = network.predictor_net(x_features, ys) # (shape: (batch_size, 1))
            scores_gt = scores_gt.squeeze(1) # (shape: (batch_size))

            scores_samples = network.predictor_net(x_features, y_samples) # (shape: (batch_size, num_samples))

            ########################################################################
            # compute loss:
            ########################################################################
            f_samples = scores_samples
            p_N_samples = q_y_samples.detach()
            f_0 = scores_gt
            p_N_0 = q_ys.detach()
            exp_vals_0 = f_0-torch.log(p_N_0 + 0.0)
            exp_vals_samples = f_samples-torch.log(p_N_samples + 0.0)
            exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1)
            loss_ebm_nce = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))

            log_Z = torch.logsumexp(scores_samples.detach() - torch.log(q_y_samples), dim=1) - math.log(num_samples) # (shape: (batch_size))
            loss_mdn_kl = torch.mean(log_Z)

            loss_mdn_nll = torch.mean(-torch.log(q_ys))

            loss = loss_ebm_nce + loss_mdn_kl

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

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
