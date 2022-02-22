from datasets import ToyDatasetTest # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from ebmdn_model_K1 import ToyNet

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

model_id = "ebmdn4_train_K1"
M = 20

network = ToyNet(model_id, project_dir="/root/ebms_proposals/1dregression_2").cuda()

K = network.noise_net.K
print (K)

epoch = 75

num_samples = 8192
epsilon = 1.0e-30

test_dataset = ToyDatasetTest()

num_test_batches = int(len(test_dataset)/batch_size)
print ("num_test_batches:", num_test_batches)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mnlls = []
for model_i in range(M):
    network.load_state_dict(torch.load("/root/ebms_proposals/1dregression_2/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))

    nll_values = []
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda().unsqueeze(1) # (shape: (batch_size, 1))
            ys = ys.cuda().unsqueeze(1) # (shape: (batch_size, 1))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))

            y_samples = np.linspace(-12.5, 12.5, num_samples) # (shape: (num_samples, ))
            y_samples = y_samples.astype(np.float32)
            y_samples = torch.from_numpy(y_samples).cuda()

            scores = network.predictor_net(x_features, y_samples.expand(xs.shape[0], -1)) # (shape: (batch_size, num_samples))

            scores_gt = network.predictor_net(x_features, ys) # (shape: (batch_size, 1))
            scores_gt = scores_gt.squeeze(1) # (shape: (batch_size))

            p_ys = torch.exp(scores_gt)/(25*torch.mean(torch.exp(scores), dim=1)) # (shape: (batch_size))

            nlls = -torch.log(p_ys) # (shape: (batch_size))

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

print ("####")
mnlls = list(np.array(mnlls)[~np.isnan(mnlls)])
print (mnlls)
print ("mnll: %g +/- %g" % (np.mean(np.array(mnlls)), np.std(np.array(mnlls))))
mnlls.sort()
print (mnlls[0:5])
print ("mnll top 5: %g +/- %g" % (np.mean(np.array(mnlls[0:5])), np.std(np.array(mnlls[0:5]))))
print (mnlls[0:10])
print ("mnll top 10: %g +/- %g" % (np.mean(np.array(mnlls[0:10])), np.std(np.array(mnlls[0:10]))))
