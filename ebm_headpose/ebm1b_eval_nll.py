from datasets import DatasetTest # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from ebmdn_model_K4 import ToyNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 32

model_id = "ebm1b_train"
M = 20

network = ToyNet(model_id, project_dir="/root/project5/ebm_headpose").cuda()

epoch = 75

num_samples = 30

test_dataset = DatasetTest()

num_test_batches = int(len(test_dataset)/batch_size)
print ("num_test_batches:", num_test_batches)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

x_ = np.linspace(-80.0, 80.0, num_samples)
y_ = np.linspace(-80.0, 80.0, num_samples)
z_ = np.linspace(-80.0, 80.0, num_samples)
x, y, z = np.meshgrid(x_, y_, z_)
x = x.astype(np.float32)
y = y.astype(np.float32)
z = z.astype(np.float32)
x = torch.from_numpy(x).cuda()
y = torch.from_numpy(y).cuda()
z = torch.from_numpy(z).cuda()
y_samples = torch.cat([x.unsqueeze(3), y.unsqueeze(3), z.unsqueeze(3)], dim=3) # (shape: (num_samples, num_samples, num_samples, 3))
print (y_samples.size())
y_samples = y_samples.view(-1, 3) # (shape: (num_samples^3, 3))
print (y_samples.size())

mnlls = []
for model_i in range(M):
    network.load_state_dict(torch.load("/root/project5/ebm_headpose/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))

    nll_values = []
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size, 3))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))

            scores = network.predictor_net(x_features, y_samples.expand(xs.shape[0], -1, 3)) # (shape: (batch_size, num_samples^3, 3))

            scores_gt = network.predictor_net(x_features, ys.unsqueeze(1)) # (shape: (batch_size, 1))
            scores_gt = scores_gt.squeeze(1) # (shape: (batch_size))

            # p_ys = torch.exp(scores_gt)/(200*torch.mean(torch.exp(scores), dim=1)) # (shape: (batch_size))
            log_p_ys = scores_gt - torch.logsumexp(scores, dim=1) - math.log(160**3) + math.log(num_samples**3) # (shape: (batch_size))
            p_ys = torch.exp(log_p_ys)

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
