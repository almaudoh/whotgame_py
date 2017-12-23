from cnnestimator import *
from replaybuffer import ReplayBuffer
import numpy as np
import collections
import itertools
import gym
import whotgame.envs.whot


# Data structure to hold episode statistics.
EpisodeStats = collections.namedtuple("Stats",["episode_lengths", "episode_rewards"])
Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

TENSOR_BOARD_LOG_PATH = './logs/tensorboard'

# Data structure to hold learned data.


def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0, replay_size=1000):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        replay_size: Replay memory size before training and discarding the data

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    replay_memory = {}

    # Continuous running loop.
    while True:

        # Run some episodes for gathering training data.
        for i_episode in range(num_episodes):
            # Reset the environment in readiness for the next episode
            state = env.reset()
            episode = []

            # One step in the environment
            for t in itertools.count():

                # Take a step
                action_probs = estimator_policy.predict(state)
                action = np.random.choice(np.arange(len(action_probs)))  # , p=action_probs)
                next_state, reward, done, _ = env.step(action)

                # Keep track of the transition
                episode.append(Transition(state=state, action=action, reward=reward, next_state=next_state, done=done))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                if i_episode > num_episodes - 20:
                    # Print out which step we're on, useful for debugging.
                    # print("\n\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))
                    # env.render()
                    pass

                if done:
                    # Calculate the accumulated rewards for all states.
                    episode_reward, count = reward, 0
                    for transition in reversed(episode):
                        discounted_reward = (discount_factor ** count) * episode_reward
                        if transition.state not in replay_memory.keys():
                            replay_memory[transition.state] = {}

                        if transition.action in replay_memory[transition.state].keys():
                            replay_memory[transition.state][transition.action] += discounted_reward + (transition.reward if count > 0 else 0)
                        else:
                            replay_memory[transition.state][transition.action] = discounted_reward + (transition.reward if count > 0 else 0)
                        count += 1
                    break

                state = next_state

            if len(replay_memory) > replay_size:
                # Train with replay memory, then discard

                for state in replay_memory.keys():
                    # Update the value estimator
                    estimator_value.update(state, max(replay_memory[state]))

                for state in replay_memory.keys():
                    # Calculate TD Target
                    value = estimator_value.predict(state)
                    for action in replay_memory[state].keys():
                        # Update the policy estimator
                        # using the td error as our advantage estimate
                        td_target = replay_memory[state][action]
                        td_error = td_target - estimator_value.predict(state)
                        loss = estimator_policy.update(state, td_error, action)

                        print("Value: {}; Target: {}; Error: {}".format(value, td_target, td_error))
                        print("Loss: {}".format(loss))

                replay_memory = {}

    return stats


tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(logpath=TENSOR_BOARD_LOG_PATH)
value_estimator = ValueEstimator(logpath=TENSOR_BOARD_LOG_PATH)
env = gym.make('WhotGame-Single-v0')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = actor_critic(env, policy_estimator, value_estimator, num_episodes=3000, discount_factor=0.9)

# plotting.plot_episode_stats(stats, smoothing_window=10)
