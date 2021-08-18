from itertools import count

import numpy as np
import torch

import devices
import utils
from models import Model
from rl import ReplayMemory, Transition


class Training:

    def __init__(self, env, model: Model, optimizer, criterion, n_epochs, batch_size, gamma, model_file):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_file = model_file

        self.n_epochs = n_epochs

        self.memory = ReplayMemory(10000)

        self.target_update_frequency = 10

        self.batch_size = batch_size
        self.gamma = gamma

    def __optimize(self):
        if len(self.memory) < self.batch_size:
            return np.inf

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=devices.cuda_otherwise_cpu, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state).to(devices.cuda_otherwise_cpu)
        action_batch = torch.cat(batch.action).to(devices.cuda_otherwise_cpu)
        reward_batch = torch.cat(batch.reward).to(devices.cuda_otherwise_cpu)

        state_action_values = self.model.infer(state_batch, uses_target=False).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=devices.cuda_otherwise_cpu)
        next_state_values[non_final_mask] = self.model.infer(non_final_next_states, uses_target=True).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def train(self):
        for i_epoch in range(self.n_epochs):
            print(f'------ epoch {i_epoch}', end='')
            losses = []

            state = self.env.reset()
            state = utils.tensorize_state(state).to(devices.cuda_otherwise_cpu)
            for t in count():
                self.env.render()

                self.model.eval()
                action = self.model.select_action(state)

                next_state, reward, done, _ = self.env.step(action.item())

                next_state = utils.tensorize_state(next_state).to(devices.cuda_otherwise_cpu)
                reward = torch.tensor([reward], device=devices.cuda_otherwise_cpu)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.model.train()

                loss = self.__optimize()
                losses.append(loss)

                if done:
                    break

            epoch_loss = np.mean(losses)
            print(f' - loss : {epoch_loss}', end='')

            if i_epoch % self.target_update_frequency == 0:
                self.model.update_target_net()
                self.model.save(self.model_file)
                print(f' *** save ***', end='')
            print()











