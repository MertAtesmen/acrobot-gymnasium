from collections import deque, namedtuple
import random
import math
import torch

import devices

BATCH_SIZE = 128
GAMMA = 0.999
TARGET_UPDATE = 10


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))




