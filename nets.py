import torch.nn
import torch.nn as nn
import torch.nn.functional as F

import devices


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2_size_out(conv2_size_out(conv2_size_out(w)))
        convh = conv2_size_out(conv2_size_out(conv2_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(devices.cuda_otherwise_cpu)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def simple_DQN():
    # return nn.Sequential(
    #     nn.Linear(6, 256),
    #     nn.BatchNorm1d(256),
    #     nn.ReLU(),


    #     nn.Linear(256, 64),
    #     nn.BatchNorm1d(64),
    #     nn.ReLU(),

    #     nn.Linear(64, 3),
    # )
    
    # Kodama, N., Harada, T., & Miyazaki, K. (2019). Deep Reinforcement Learning with Dual Targeting Algorithm. 2019 International Joint Conference on Neural Networks
    return nn.Sequential(
        nn.Linear(6, 128),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Linear(64, 3),
    )


