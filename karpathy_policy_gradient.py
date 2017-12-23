import random
import tensorflow as tf
from cnnestimator import *
import numpy as np
import collections
import itertools
from whotgame import card
from whotgame.gamerule import GameRule
from whotgame.gamestate import GameStates


# Data structure to hold episode statistics.
EpisodeStats = collections.namedtuple("Stats",["episode_lengths", "episode_rewards"])
Transition = collections.namedtuple("Transition", ["state", "suggested", "action", "reward", "next_state", "done"])

# Data structure to hold learned data.
gamerule = GameRule()


def best_move(states):
    """
    Returns the best move for the current state
    :param state:
    :return:
    """
    valid_moves = []
    for state in states:
        valid_move = [0] + gamerule.filter_valid_moves(state.hand, state)
        valid_moves.append(np.random.choice(valid_move))
    return np.array(valid_moves)


def filter_illegal(action_probs, state):
    """
    Filters illegal moves from the action probabilities by setting them to zero and re-normalizing the
    sum to 1.0
    :param action_probabilities:
    :return:
    """

    moves = gamerule.filter_valid_moves(state.hand, state)
    valid_filter = [1 if i in moves else 0 for i in range(0, 72)]
    # Market is always valid
    valid_filter[0] = 1
    action_probs *= valid_filter
    action_probs /= np.sum(action_probs)
    return action_probs


def policy_gradient(env, policy_estimator, num_episodes, discount_factor=1.0, replay_size=1000, filter_invalid=0.50,
                    session=None, summary_writer=None, save_path=None):
    """
    Karpathy Algorithm for optimizes the policy function approximator using policy gradient.
    The

    Args:
        env: OpenAI environment.
        policy_estimator: Policy function to be optimized
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        replay_size: Replay memory size before training and discarding the data
        filter_invalid: Proportion of times invalid moves should be filtered out from the probability distribution
        session: Tensorflow session to be saved
        save_path: The path to the filesystem for saving network weights and other learned parameters
        summary_writer: A tensorboard summary writer

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    replay_memory = {}
    valid_suggestions = []
    longest_episode, games_won, games_lost, illegal_moves = 0, 0, 0, 0

    # with tf.name_scope('Summaries'):
    #     valid_input = tf.placeholder(tf.int64, [None], name='valid_input')
    #     valid_suggestions_op = tf.summary.scalar('valid_suggestions', tf.reduce_mean(valid_input))

    # Run episodes for gathering training data based on success / fail of episodes.
    for i_episode in range(num_episodes):
        # Reset the environment in readiness for the next episode
        state = env.reset()
        episode = []

        # One step in the environment
        for t in itertools.count():

            # Take a step
            action_probs = policy_estimator.predict(state)
            suggested = np.argmax(action_probs)
            if filter_invalid > random.uniform(0, 1):
                # Once in a while filter invalid moves to improve our chances of learning the good moves
                # Otherwise we would waste too much time exploring invalid regions of the action space.
                action_probs = filter_illegal(action_probs, state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Track valid move suggestions - excluding market
            if suggested > 0:
                is_valid = gamerule.is_valid_move(suggested, state)
                valid_suggestions.append(int(is_valid))
            # estimator_policy.add_summary_variable('valid_suggestions', np.average(valid_suggestions))

            # Keep track of the transition
            episode.append(Transition(state=state, suggested=suggested, action=action, reward=reward, next_state=next_state, done=done))
            state = next_state

            # Update statistics
            # stats.episode_rewards[i_episode] += reward
            # stats.episode_lengths[i_episode] = t
            longest_episode = max(t, longest_episode)

            if done:
                # Print episode header
                # print("Episode: {}".format(i_episode))
                # env.render()

                # Calculate the accumulated rewards for all states.
                discounted_reward, count = 0, 0
                for transition in reversed(episode):
                    if transition.state not in replay_memory.keys():
                        replay_memory[transition.state] = {}

                    if transition.action in replay_memory[transition.state].keys():
                        replay_memory[transition.state][transition.action] += discounted_reward + transition.reward
                    else:
                        replay_memory[transition.state][transition.action] = discounted_reward + transition.reward

                    # The last reward is always the entire episode reward.
                    if transition.reward == -1000:
                        # -1000 means illegal move, so we don't want to penalize prior moves. Also, we
                        # don't really know if we could have won or lost if not for the illegal move.
                        discounted_reward = 0
                    else:
                        discounted_reward = discount_factor * (discounted_reward + transition.reward)

                    count += 1

                    # Print moves played for this episode.
                    # if transition.state.current_player == 'agent':
                    # print("- State: {};\n\tSuggested Move: {}; Played Move: {}".format(
                    #     transition.state, card.from_cardpoint(transition.suggested),
                    #     card.from_cardpoint(transition.action)))

                # Statistics for games won and lost
                if episode[-1].reward == 1:
                    games_won += 1
                elif episode[-1].reward == -1:
                    games_lost += 1
                elif episode[-1].reward == -1000:
                    illegal_moves += 1

                break

        if len(replay_memory) > replay_size:
            # Initialize collections for storing game states, actions and rewards
            states, advantages, actions = GameStates(), [], []
            for state in replay_memory.keys():
                # Save the state, actions and rewards for later bulk training.
                for action in replay_memory[state].keys():
                    advantage = replay_memory[state][action]
                    states.append(state)
                    advantages.append(advantage)
                    actions.append(action)

            # Bulk train
            # Update the policy estimator using the calculated
            # rewards as our advantage estimate
            loss = policy_estimator.update(states, advantages, actions)
            accuracy = policy_estimator.learning_accuracy(states, best_move(states))

            if summary_writer is not None:
                # External summaries
                # valid_suggestions_op.eval(session=session, feed_dict={valid_input: valid_suggestions})
                # Track summaries
                policy_estimator.write_summaries(summary_writer)

            # Print some statistics before training the network.
            print("Won: {}; Lost: {}; Illegal: {}; Win ratio: {}%".format(games_won, games_lost, illegal_moves, games_won * 100 / (games_lost + illegal_moves)))
            print("Loss: {}; Accuracy: {}; Longest streak: {}".format(loss, accuracy, longest_episode))
            print("Valid suggestions: {}%\n".format(np.average(valid_suggestions) * 100))

            # Reset the variables for counting.
            replay_memory = {}
            longest_episode, games_won, games_lost, illegal_moves = 0, 0, 0, 0
            valid_suggestions = []

            # Save the network so far.
            if save_path is not None:
                if session is None: session = tf.get_default_session()
                saver = tf.train.Saver()
                saver.save(session, save_path)

    return stats
