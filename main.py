import gym
import time

ENV_NAME = 'Acrobot-v1'

env = gym.make(ENV_NAME)
print('--------- START --------')

observation = env.reset()
for i in range(1000):
    print(f'State {i} : {observation}')
    env.render()
    time.sleep(1 / 11)

    action = 0
    print(f'Action {i} : {action}')

    observation, reward, done, info = env.step(action)
    print(f'Reward {i+1} : {reward}')
    if done:
        print('--------- END --------')
        break
    else:
        print('=========')
