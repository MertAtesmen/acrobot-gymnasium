import gym
import torch.nn as nn
import torch.optim as optim

import devices
import nets
from models import Model
from trainings import Training

if __name__ == '__main__':
    env_name = 'Acrobot-v1'
    env = gym.make(env_name)
    model = Model(policy_net=nets.simple_DQN().to(devices.cuda_otherwise_cpu), target_net=nets.simple_DQN().to(devices.cuda_otherwise_cpu))
    optimizer = optim.RMSprop(model.policy_net.parameters())
    criterion = nn.SmoothL1Loss()

    training = Training(
        env=env,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=250,
        batch_size=128,
        gamma=0.999,
        model_file='state.pt'
    )

    training.train()
