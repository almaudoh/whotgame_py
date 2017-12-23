import gym
import whotgame.envs
import numpy as np

from whotgame.agents.agent import SimpleAgent
from whotgame.gamestate import GameState

env = gym.make('WhotGame-v0')

# Determine the outcome of our action

game_state = env.reset()
agent = SimpleAgent()
agent.hand = game_state.hand
reward_sum = 0
while True:
    env.render()
    agent.hand = game_state.hand
    move = agent.get_move(game_state)
    game_state, reward, done, _ = env.step(move)

    reward_sum += reward
    if done:
        env.render()
        break

print("\nTotal score: {}".format(reward_sum))
print(GameState(env.env, env.env.opponents[0].get_name()))
print(GameState(env.env, env.env.opponents[1].get_name()))
