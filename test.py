import time
from itertools import count

import gymnasium as gym

import devices
import nets
import utils
from models import Model

if __name__ == '__main__':
    env_name = 'Acrobot-v1'

    model = Model(policy_net=nets.simple_DQN().to(devices.cuda_otherwise_cpu),
                  target_net=nets.simple_DQN().to(devices.cuda_otherwise_cpu))
    model.load('state.pt')
    
    # env = gym.make(env_name)
    # render_mode = 'human' so we can see the gameplay
    env = gym.make(env_name, render_mode = 'human')
    # GYMNASIUM CHANGE
    # env.reset()

    # state = env.reset()
    state, _ = env.reset()
    state = utils.tensorize_state(state).to(devices.cuda_otherwise_cpu)

    for t in count():
        # env.render()
        action = model.select_action(state, train=False)
        # GYMNASIUM CHANGE
        # state, reward, done, _ = env.step(action.item())
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        state = utils.tensorize_state(state).to(devices.cuda_otherwise_cpu)
        if done:
            break
    env.close()



