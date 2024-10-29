import gymnasium as gym
import time

ENV_NAME = 'Acrobot-v1'

env = gym.make(ENV_NAME)
print('--------- START --------')

# GYMNASIUM CHANGE 
# observation = env.reset()
observation, _ = env.reset()
for i in range(1000):
    print(f'State {i} : {observation}')
    env.render()
    time.sleep(1 / 11)

    action = env.action_space.sample()
    print(f'Action {i} : {action}')
    # GYMNASIUM CHANGE 
    # observation, reward, done, info = env.step(action)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f'Reward {i+1} : {reward}')
    if done:
        print('--------- END --------')
        break
    else:
        print('=========')
