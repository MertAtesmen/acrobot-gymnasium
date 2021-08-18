import time
from itertools import count

import gym

import devices
import nets
import utils
from models import Model

if __name__ == '__main__':
    env_name = 'Acrobot-v1'

    model = Model(policy_net=nets.simple_DQN().to(devices.cuda_otherwise_cpu),
                  target_net=nets.simple_DQN().to(devices.cuda_otherwise_cpu))
    model.load('state.pt')

    env = gym.make(env_name)
    env.reset()

    state = env.reset()
    state = utils.tensorize_state(state).to(devices.cuda_otherwise_cpu)

    for t in count():
        env.render()
        time.sleep(1 / 12)
        action = model.select_action(state, train=False)
        state, reward, done, _ = env.step(action.item())
        state = utils.tensorize_state(state).to(devices.cuda_otherwise_cpu)
        if done:
            break
    env.close()



