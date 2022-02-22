from datasets import DatasetTrainAug # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from ebmdn_model_K16 import ToyNet

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
model_id = "ebmdn3_2_train_K16_fullnet4"

num_epochs = 75
batch_size = 32
learning_rate = 0.001

num_samples = 1024
print (num_samples)

train_dataset = DatasetTrainAug()

num_train_batches = int(len(train_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

num_models = 20
for i in range(num_models):
    if i < 19:
        continue

    mdn_network = ToyNet(model_id + "_%d" % i, project_dir="/root/project5/headpose").cuda()
    ebm_network = ToyNet(model_id + "_%d" % i, project_dir="/root/project5/headpose").cuda()

    K = mdn_network.noise_net.K
    print (K)

    mdn_optimizer = torch.optim.Adam(mdn_network.parameters(), lr=learning_rate)
    ebm_optimizer = torch.optim.Adam(ebm_network.parameters(), lr=learning_rate)

    epoch_losses_train = []
    epoch_losses_mdn_nll_train = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("model: %d/%d  |  epoch: %d/%d" % (i+1, num_models, epoch+1, num_epochs))

        mdn_network.train() # (set in training mode, this affects BatchNorm and dropout)
        ebm_network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        batch_losses_mdn_nll = []
        for step, (xs, ys) in enumerate(train_loader):
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size, 3))

            mdn_x_features = mdn_network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            ebm_x_features = ebm_network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            if epoch < 20:
                ####################################################################
                # make sure we do NOT train the resnet feature extractor:
                ####################################################################
                mdn_x_features = mdn_x_features.detach()
                ebm_x_features = ebm_x_features.detach()
                ####################################################################
            means, log_sigma2s, weights = mdn_network.noise_net(mdn_x_features)
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

            y_samples_K = q_distr.sample(sample_shape=torch.Size([num_samples])) # (shape: (num_samples, batch_size, 3, K))
            inds = torch.multinomial(weights, num_samples=num_samples, replacement=True).unsqueeze(2).unsqueeze(2) # (shape: (batch_size, num_samples, 1, 1))
            inds = inds.expand(-1, -1, 3, 1) # (shape: (batch_size, num_samples, 3, 1))
            inds = torch.transpose(inds, 1, 0) # (shape: (num_samples, batch_size, 3, 1))
            y_samples = y_samples_K.gather(3, inds).squeeze(3) # (shape: (num_samples, batch_size, 3))
            y_samples = y_samples.detach()
            # q_y_samples_K = torch.exp(q_distr.log_prob(y_samples.unsqueeze(3)).sum(2)) # (shape: (num_samples, batch_size, K))
            # q_y_samples = torch.sum(weights.unsqueeze(0)*q_y_samples_K, dim=2) # (shape: (num_samples, batch_size))
            # y_samples = torch.transpose(y_samples, 1, 0) # (shape: (batch_size, num_samples, 3))
            # q_y_samples = torch.transpose(q_y_samples, 1, 0) # (shape: (batch_size, num_samples))
            log_q_y_samples_K = q_distr.log_prob(y_samples.unsqueeze(3)).sum(2) # (shape: (num_samples, batch_size, K))
            log_q_y_samples = torch.logsumexp(torch.log(weights.unsqueeze(0)) + log_q_y_samples_K, dim=2) # (shape: (num_samples, batch_size))
            y_samples = torch.transpose(y_samples, 1, 0) # (shape: (batch_size, num_samples, 3))
            log_q_y_samples = torch.transpose(log_q_y_samples, 1, 0) # (shape: (batch_size, num_samples))

            scores_gt = ebm_network.predictor_net(ebm_x_features, ys.unsqueeze(1)) # (shape: (batch_size, 1))
            scores_gt = scores_gt.squeeze(1) # (shape: (batch_size))

            scores_samples = ebm_network.predictor_net(ebm_x_features, y_samples) # (shape: (batch_size, num_samples))

            ########################################################################
            # compute loss:
            ########################################################################
            # q_ys = F.relu(q_ys - 1.0e-6) + 1.0e-6

            # f_samples = scores_samples
            # p_N_samples = q_y_samples.detach()
            # f_0 = scores_gt
            # p_N_0 = q_ys.detach()
            # exp_vals_0 = f_0-torch.log(p_N_0 + 0.0)
            # exp_vals_samples = f_samples-torch.log(p_N_samples + 0.0)
            # exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1)
            # loss_ebm_nce = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))
            f_samples = scores_samples
            log_p_N_samples = log_q_y_samples.detach()
            f_0 = scores_gt
            log_p_N_0 = log_q_ys.detach()
            exp_vals_0 = f_0-log_p_N_0
            exp_vals_samples = f_samples-log_p_N_samples
            exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1)
            loss_ebm_nce = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))

            # log_Z = torch.logsumexp(scores_samples.detach() - torch.log(q_y_samples), dim=1) - math.log(num_samples) # (shape: (batch_size))
            log_Z = torch.logsumexp(scores_samples.detach() - log_q_y_samples, dim=1) - math.log(num_samples) # (shape: (batch_size))
            loss_mdn_kl = torch.mean(log_Z)

            # loss_mdn_nll = torch.mean(-torch.log(q_ys))
            loss_mdn_nll = torch.mean(-log_q_ys)

            ebm_loss = loss_ebm_nce

            # (without this fix, I often get "RuntimeError: CUDA error: device-side assert triggered" at the very beginning)
            if epoch < 1:
                mdn_loss = loss_mdn_nll
            else:
                mdn_loss = 0.5*loss_mdn_kl + 0.5*loss_mdn_nll

            loss_value = ebm_loss.data.cpu().numpy()
            batch_losses.append(loss_value)

            loss_mdn_nll_value = loss_mdn_nll.data.cpu().numpy()
            batch_losses_mdn_nll.append(loss_mdn_nll_value)

            ########################################################################
            # optimization step:
            ########################################################################
            ebm_optimizer.zero_grad() # (reset gradients)
            ebm_loss.backward() # (compute gradients)
            ebm_optimizer.step() # (perform optimization step)

            mdn_optimizer.zero_grad() # (reset gradients)
            mdn_loss.backward() # (compute gradients)
            mdn_optimizer.step() # (perform optimization step)

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)
        with open("%s/epoch_losses_train.pkl" % mdn_network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("ebm train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("ebm train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % mdn_network.model_dir)
        plt.close(1)

        epoch_loss = np.mean(batch_losses_mdn_nll)
        epoch_losses_mdn_nll_train.append(epoch_loss)
        with open("%s/epoch_losses_mdn_nll_train.pkl" % mdn_network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_mdn_nll_train, file)
        print ("train loss_mdn_nll: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_mdn_nll_train, "k^")
        plt.plot(epoch_losses_mdn_nll_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss_mdn_nll per epoch")
        plt.savefig("%s/epoch_losses_mdn_nll_train.png" % mdn_network.model_dir)
        plt.close(1)

    # save the model weights to disk:
    mdn_checkpoint_path = mdn_network.checkpoints_dir + "/mdn_model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(mdn_network.state_dict(), mdn_checkpoint_path)
    ebm_checkpoint_path = ebm_network.checkpoints_dir + "/ebm_model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(ebm_network.state_dict(), ebm_checkpoint_path)
