from tensorflow.python.framework.errors_impl import NotFoundError
from cnnestimator import *
import karpathy_policy_gradient as kpg
import gym
import whotgame.envs.whot
import policy_trainer as pt

SUMMARY_LOG_PATH = './logs/gradient2'
SAVE_PATH = 'checkpoint/trained_learner'

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator(learning_rate=.1, l2_regularization=1.)
# Load weights if it has been previously saved:
env = gym.make('WhotGame-Single-v0')

with tf.Session() as sess:
    saver = tf.train.Saver()
    try:
        saver.restore(sess, SAVE_PATH)
    except NotFoundError:
        sess.run(tf.global_variables_initializer())

    # Tensorflow summary writer.
    summary_writer = tf.summary.FileWriter(SUMMARY_LOG_PATH, sess.graph)

    # Pre-train the model using stored game information.
    pt.train_policy(policy_estimator, 'saved_moves.txt', summary_writer=summary_writer, save_path=SAVE_PATH, session=sess)

    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~300 seemed to work well for me.
    stats = kpg.policy_gradient(env, policy_estimator, num_episodes=300000, discount_factor=.999, filter_invalid=.99,
                                session=sess, save_path=SAVE_PATH, summary_writer=summary_writer)

# plotting.plot_episode_stats(stats, smoothing_window=10)
sess.close()