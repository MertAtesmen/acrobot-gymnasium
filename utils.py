import torch


def tensorize_state(state):
    return torch.tensor([state], dtype=torch.float)
