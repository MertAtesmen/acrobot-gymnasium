import random
import math
import torch
import torch.nn as nn
import torch.optim as optim

import devices


class Model:

    def __init__(self, policy_net, target_net):
        assert type(policy_net) == type(target_net)
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.steps_done = 0

        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.eps = self.eps_start

    def __decay_eps(self):
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

    def save(self, file):
        torch.save(self.target_net.state_dict(), file)

    def load(self, file):
        self.target_net.load_state_dict(torch.load(file))

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        self.policy_net.train()

    def eval(self):
        self.policy_net.eval()

    def infer(self, state_batch, uses_target=True):
        if uses_target:
            return self.target_net(state_batch)
        else:
            return self.policy_net(state_batch)

    def reset(self):
        self.eps = 0
        self.steps_done = 0

    def select_action(self, state, train=True):
        if train:
            sample = random.random()
            self.__decay_eps()

            if sample > self.eps:
                with torch.no_grad():
                    output = self.policy_net(state)
                    return output.max(1)[1].view(1, 1)
            else:
                return torch.tensor([[
                    random.randrange(self.policy_net[-1].out_features)]],
                    device=devices.cuda_otherwise_cpu,
                    dtype=torch.long)
        else:
            with torch.no_grad():
                output = self.target_net(state)
                return output.max(1)[1].view(1, 1)



