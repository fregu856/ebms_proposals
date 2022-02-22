import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import os

class ToyNoiseNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.K = 8

        self.fc1_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, 3*self.K)

        self.fc1_sigma = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, 3*self.K)

        self.fc1_weight = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_weight = nn.Linear(hidden_dim, self.K)

    def forward(self, x_feature):
        # (x_feature has shape: (batch_size, hidden_dim))

        means = F.relu(self.fc1_mean(x_feature))  # (shape: (batch_size, hidden_dim))
        means = self.fc2_mean(means)  # (shape: batch_size, 3K))

        log_sigma2s = F.relu(self.fc1_sigma(x_feature))  # (shape: (batch_size, hidden_dim))
        log_sigma2s = self.fc2_sigma(log_sigma2s)  # (shape: batch_size, 3K))

        weight_logits = F.relu(self.fc1_weight(x_feature))  # (shape: (batch_size, hidden_dim))
        weight_logits = self.fc2_weight(weight_logits)  # (shape: batch_size, K))
        weights = torch.softmax(weight_logits, dim=1) # (shape: batch_size, K))

        return means, log_sigma2s, weights


class ToyFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        resnet18 = models.resnet18(pretrained=True)
        # remove fully connected layer:
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # (x has shape (batch_size, 3, img_size, img_size))

        x_feature = self.resnet18(x) # (shape: (batch_size, 512, img_size/32, img_size/32))
        x_feature = self.avg_pool(x_feature) # (shape: (batch_size, 512, 1, 1))
        x_feature = x_feature.squeeze(2).squeeze(2) # (shape: (batch_size, 512))

        return x_feature


class ToyNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(ToyNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        hidden_dim = 512

        self.feature_net = ToyFeatureNet()
        self.noise_net = ToyNoiseNet(hidden_dim)

    def forward(self, x, y):
        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.noise_net(x_feature)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
