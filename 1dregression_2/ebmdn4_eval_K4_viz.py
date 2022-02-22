from datasets import ToyDatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from ebmdn_model_K4 import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 32

model_id = "ebmdn4_train_K4"
M = 20

network = ToyNet(model_id, project_dir="/root/project5/umal_1dregression").cuda()

K = network.noise_net.K
print (K)

epoch = 75

num_samples = 4096
num_plot_samples = 1

val_dataset = ToyDatasetEval()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

for model_i in range(M):
    network.load_state_dict(torch.load("/root/project5/umal_1dregression/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))

    x_values = []
    means_values = []
    sigmas_values = []
    weights_values = []
    y_values_noise = []
    x_values_2_scores = {}
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (x) in enumerate(val_loader):
        with torch.no_grad():
            if (step % 1000) == 0:
                print (step)

            x = x.cuda().unsqueeze(1) # (shape: (batch_size, 1))

            x_features = network.feature_net(x)
            means, log_sigma2s, weights = network.noise_net(x_features) # (all have shape: (batch_size, K))
            sigmas = torch.exp(log_sigma2s/2.0) # (shape: (batch_size, K))

            q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
            y_samples_K = q_distr.sample(sample_shape=torch.Size([1])) # (shape: (1, batch_size, K))
            inds = torch.multinomial(weights, num_samples=1, replacement=True).unsqueeze(2) # (shape: (batch_size, 1, 1))
            inds = torch.transpose(inds, 1, 0) # (shape: (1, batch_size, 1))
            y_samples = y_samples_K.gather(2, inds).squeeze(2) # (shape: (1, batch_size))
            y_samples = y_samples.squeeze(0) # (shape: (batch_size))
            y_values_noise.extend(y_samples.cpu().tolist())

            x_values.extend(x.squeeze(1).cpu().tolist())

            y_samples = np.linspace(-12.0, 12.0, num_samples) # (shape: (num_samples, ))
            y_samples = y_samples.astype(np.float32)
            y_samples = torch.from_numpy(y_samples).cuda() # (shape: (num_samples))

            scores = network.predictor_net(x_features, y_samples.expand(x.shape[0], -1))

            for i, x_val in enumerate(x):
                means_value = means[i, :].data.cpu().numpy() # (shape: (K, ))
                sigmas_value = sigmas[i, :].data.cpu().numpy() # (shape: (K, ))
                weights_value = weights[i, :].data.cpu().numpy() # (shape: (K, ))
                means_values.append(means_value)
                sigmas_values.append(sigmas_value)
                weights_values.append(weights_value)

                x_values_2_scores[x_val.item()] = scores[i,:].cpu().numpy()

    means_values = np.asarray(means_values) # (shape: (num_val_examples, K))
    sigmas_values = np.asarray(sigmas_values) # (shape: (num_val_examples, K))
    weights_values = np.asarray(weights_values) # (shape: (num_val_examples, K))
    print (means_values.shape)
    print (sigmas_values.shape)
    print (weights_values.shape)

    plt.figure(1)
    plt.plot(x_values, y_values_noise, "k.", alpha=0.01, markeredgewidth=0.00001)
    plt.xlabel("x")
    plt.ylim([-10.0, 10.0])
    plt.savefig("%s/%d_pred_noise_sample_dens_epoch_%d.png" % (network.model_dir, model_i, epoch+1))
    plt.close(1)

    if K <= 10:
        plt.figure(1)
        for j in range(K):
            mean_values = means_values[:, j] # (shape: (num_val_examples, ))
            sigma_values = sigmas_values[:, j] # (shape: (num_val_examples, ))

            plt.plot(x_values, mean_values, color="C%d" % j)
            plt.fill_between(x_values, mean_values - sigma_values, mean_values + sigma_values, color="C%d" % j, alpha=0.25)
        plt.xlabel("x")
        plt.ylim([-10.0, 10.0])
        plt.savefig("%s/%d_pred_noise_dens_epoch_%d.png" % (network.model_dir, model_i, epoch+1))
        plt.close(1)

        plt.figure(1)
        for j in range(K):
            weight_values = weights_values[:, j] # (shape: (num_val_examples, ))

            plt.plot(x_values, weight_values, color="C%d" % j)
        plt.xlabel("x")
        plt.ylim([-0.05, 1.05])
        plt.savefig("%s/%d_pred_noise_dens_weights_epoch_%d.png" % (network.model_dir, model_i, epoch+1))
        plt.close(1)

    y_values = []
    most_y = []
    for step, x_value in enumerate(x_values):
        if (step % 1000) == 0:
            print (step)
        scores = np.exp(x_values_2_scores[x_value].flatten()) # (shape: (num_samples, ))
        if np.sum(scores) > 1e-40:
            prob = scores/np.sum(scores) # (shape: (num_samples, ))
        else:
            scores = np.ones((num_samples, ))
            prob = scores/np.sum(scores)

        max_score_ind = np.argmax(scores)
        max_score_y = np.linspace(-12.0, 12.0, num_samples)[max_score_ind]
        most_y.append(max_score_y)

        y_plot_samples = np.random.choice(np.linspace(-12.0, 12.0, num_samples), size=(num_plot_samples, ), p=prob) # (shape: (num_plot_samples, ))
        y_values.append(y_plot_samples[0])

    plt.figure(1)
    plt.plot(x_values, y_values, "k.", alpha=0.01, markeredgewidth=0.00001)
    plt.xlabel("x")
    plt.ylim([-10.0, 10.0])
    plt.savefig("%s/%d_pred_dens_epoch_%d.png" % (network.model_dir, model_i, epoch+1))
    plt.close(1)
