import numpy as np
import gym
import whotgame.envs

env = gym.make('WhotGame-v0')
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 100:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 71))
    reward_sum += reward
    if done:
        random_episodes += 1
        env.render()
        print("\nReward for this episode was: {}\n".format(reward_sum))
        reward_sum = 0
        env.reset()
