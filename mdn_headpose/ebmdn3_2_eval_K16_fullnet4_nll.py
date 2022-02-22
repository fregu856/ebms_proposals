from datasets import DatasetTest # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from ebmdn_model_K16 import ToyNet

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

model_id = "ebmdn3_2_train_K16_fullnet4"
M = 20

network = ToyNet(model_id, project_dir="/root/ebms_proposals/mdn_headpose").cuda()

K = network.noise_net.K
print (K)

epoch = 75

test_dataset = DatasetTest()

num_test_batches = int(len(test_dataset)/batch_size)
print ("num_test_batches:", num_test_batches)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mnlls = []
for model_i in range(M):
    network.load_state_dict(torch.load("/root/ebms_proposals/mdn_headpose/training_logs/model_%s_%d/checkpoints/mdn_model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))

    nll_values = []
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size, 3))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
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

            # nlls = -torch.log(q_ys) # (shape: (batch_size))
            nlls = -log_q_ys

            nlls = nlls.data.cpu().numpy() # (shape: (batch_size, ))
            nll_values += list(nlls)

    mnll = np.mean(nll_values)
    mnlls.append(mnll)
    print ("mnll: %g" % mnll)

print (mnlls)
print ("mnll: %g +/- %g" % (np.mean(np.array(mnlls)), np.std(np.array(mnlls))))
mnlls.sort()
print (mnlls[0:5])
print ("mnll top 5: %g +/- %g" % (np.mean(np.array(mnlls[0:5])), np.std(np.array(mnlls[0:5]))))
print (mnlls[0:10])
print ("mnll top 10: %g +/- %g" % (np.mean(np.array(mnlls[0:10])), np.std(np.array(mnlls[0:10]))))
