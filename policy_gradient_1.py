from tensorflow.python.framework.errors_impl import NotFoundError
from cnnestimator import *
import gym
import karpathy_policy_gradient as kpg
import whotgame.envs.whot

TENSOR_BOARD_LOG_PATH = './logs/tensorboard'
SAVE_PATH = 'checkpoint/karpathy_learner'

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(logpath=TENSOR_BOARD_LOG_PATH)
# Load weights if it has been previously saved:
env = gym.make('WhotGame-Single-v0')

with tf.Session() as sess:
    saver = tf.train.Saver()
    try:
        saver.restore(sess, SAVE_PATH)
    except NotFoundError:
        sess.run(tf.global_variables_initializer())

    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = kpg.policy_gradient(env, policy_estimator, num_episodes=300000, discount_factor=1., filter_invalid=.99, session=sess, save_path=SAVE_PATH)

# plotting.plot_episode_stats(stats, smoothing_window=10)
sess.close()
