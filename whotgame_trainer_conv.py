import tensorflow as tf
import numpy as np
from tensorflow.python.framework.errors_impl import NotFoundError

import card
import gamestate as gs
import gamedata as gd
import math


def get_state(top, mode, hand):
    state = gs.GameState()
    state.hand = gd.cards(hand)
    state.top_card = top
    state.market_mode = mode
    return state


def top_matches(probability, number):
    # Find the highest probabilities.
    highest = np.argsort(-probability, axis=1)[0][0:number]
    return [card.from_cardpoint(cardpoint) for cardpoint in highest]


dataset = gd.GameDataSet('saved_moves.txt')

# shuffle the data
shuffle_indices = np.random.permutation(np.arange(len(dataset.y_data)))
dev_sample_percentage = .1
dev_sample_index = -1 * int(dev_sample_percentage * float(len(dataset.y_data)))
x_train, x_val = dataset.x_data[shuffle_indices[:dev_sample_index]], dataset.x_data[shuffle_indices[dev_sample_index:]]
y_train, y_val = dataset.y_data[shuffle_indices[:dev_sample_index]], dataset.y_data[shuffle_indices[dev_sample_index:]]

# Constants defining our neural network
input_size = 100
output_classes = 72
conv_layers_params = [[256, 5], [256, 1], [256, 1], [256, 1], ]
fully_connected_layers_params = [1024, 1024]

tf.reset_default_graph()

# Input data layer
with tf.name_scope('Input_Layer'):
    input_x = tf.placeholder(tf.float32, shape=[None, input_size], name='input_x')
    input_y = tf.placeholder(tf.float32, shape=[None, output_classes], name='input_y')

# Reshape and expand the dimension of input_x into conv input dimensions.
with tf.name_scope('Reshape_Input'):
    x = tf.reshape(input_x, [-1, 5, 20, 1])

with tf.name_scope('Convolution_Layer'):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 256], 0., 0.2), dtype='float32', name='W')
    b1 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
    conv = tf.nn.conv2d(x, W1, [1,1,1,1], 'VALID', name='Conv1')
    x = tf.nn.bias_add(conv, b1)

    W2 = tf.Variable(tf.truncated_normal([1, 5, 256, 256], 0., 0.2), dtype='float32', name='W')
    b2 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
    conv = tf.nn.conv2d(x, W2, [1,1,1,1], 'VALID', name='Conv1')
    x = tf.nn.bias_add(conv, b2)

    W3 = tf.Variable(tf.truncated_normal([1, 5, 256, 256], 0., 0.2), dtype='float32', name='W')
    b3 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
    conv = tf.nn.conv2d(x, W3, [1,1,1,1], 'VALID', name='Conv1')
    x = tf.nn.bias_add(conv, b3)

    W4 = tf.Variable(tf.truncated_normal([1, 5, 256, 256], 0., 0.2), dtype='float32', name='W')
    b4 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
    conv = tf.nn.conv2d(x, W4, [1,1,1,1], 'VALID', name='Conv1')
    x = tf.nn.bias_add(conv, b4)

# flatten conv output for fully connected layers
with tf.name_scope('Reshape_Layer'):
    vec_dim = x.get_shape()[1].value * x.get_shape()[2].value * x.get_shape()[3].value
    x = tf.reshape(x, [-1, vec_dim])

# fully connected layers input/output size
fc_input_size = [vec_dim] + list(fully_connected_layers_params)

# dropout keep probability
dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

# create fully connected + dropout layers
for i, output_size in enumerate(fully_connected_layers_params):
    with tf.name_scope('Linear_Layer'):
        stdv = 1 / math.sqrt(fc_input_size[i])
        W = tf.Variable(tf.truncated_normal(shape=[fc_input_size[i], output_size], mean=0, stddev=stdv),
                        dtype='float32', name='W')
        b = tf.Variable(tf.constant(shape=[output_size], value=0.1, dtype='float32'), dtype='float32', name='b')

        x = tf.nn.xw_plus_b(x, W, b)

    with tf.name_scope('Dropout_Layer'):
        x = tf.nn.dropout(x, dropout_keep_prob)

with tf.name_scope('Output_Layer'):
    stdv = 1 / math.sqrt(fc_input_size[-1])
    W = tf.Variable(tf.truncated_normal(shape=[fc_input_size[-1], output_classes], mean=0, stddev=stdv), dtype='float32', name='W')
    b = tf.Variable(tf.constant(shape=[output_classes], value=0.1, dtype='float32'), dtype='float32', name='b')

    p_y_given_x = tf.nn.xw_plus_b(x, W, b, name='scores')
    predictions = tf.argmax(p_y_given_x, 1)

with tf.name_scope('Loss_Layer'):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=p_y_given_x, labels=input_y)
    loss = tf.reduce_mean(losses)

with tf.name_scope('Accuracy_Layer'):
    correct_predictions = tf.nn.in_top_k(p_y_given_x, tf.argmax(input_y, 1), 1)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

with tf.name_scope('Summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.histogram('predictions', p_y_given_x)
    tf.summary.scalar('accuracy', accuracy)

# Creating an optimizer for the model.
# We use a moment-SGD solver with exponentially decaying learning rate.
global_step = tf.Variable(0, trainable=False)
base_rate = 1e-2 #base learning rate value
decay_step = 580 #every how many step do we decay the learning rate
decay_rate = 0.98 #learning rate decay multiplier
learning_rate = tf.train.exponential_decay(base_rate, global_step, decay_step, decay_rate, staircase=True)
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
grads_and_vars = optimizer.compute_gradients(loss)
max_gradient_norm = 5
grads_and_vars = [(tf.clip_by_norm(grad, max_gradient_norm), var) for grad, var in grads_and_vars]
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    # Writer for tensorboard charts
    writer = tf.summary.FileWriter('logs/trainer', sess.graph)

    save_path = 'checkpoint/move_predicter_model'
    saver = tf.train.Saver()
    try:
        # Load existing model if it exists...
        saver.restore(sess, save_path)
    except NotFoundError:
        # ... or randomly initialize the model weights if not
        sess.run(tf.global_variables_initializer())

    # Run the training for 10 epochs with batch size 128
    num_epochs = 1000
    batch_size = 128

    print('training...')

    # create train and test data iterators
    tr = gd.DataIterator(x_train, y_train)

    step, total_accuracy, total_loss = 0, 0, 0
    tr_losses, te_losses = [], []
    current_epoch = 0

    # iterate for 10 epochs
    while current_epoch < num_epochs:

        step += 1

        # get next batch of training data
        x_batch, y_batch = tr.next_batch(batch_size)
        feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 0.5}

        # run a single training iteration
        _, global_step_, loss_, accuracy_, summary = sess.run([train_op, global_step, loss, accuracy, summary_op],
                                                              feed_dict=feed_dict)

        if global_step_ % 10 == 0:
            print('iter {} accuracy {} loss {}'.format(global_step_, accuracy_, loss_))
            writer.add_summary(summary, global_step_)

            # Save the weights
            saver.save(sess, save_path)

        # accumulate training accuracy and loss
        total_accuracy += accuracy_
        total_loss += loss_

        # print training progress and run model validation every epoch
        if tr.epochs > current_epoch:

            current_epoch += 1
            tr_losses.append((total_accuracy / step, total_loss / step))
            step, total_accuracy, total_loss = 0, 0, 0

            # eval test set
            te = gd.DataIterator(x_val, y_val, False)
            te_epoch = te.epochs
            while te.epochs == te_epoch:
                step += 1
                x_batch, y_batch = te.next_batch(batch_size)
                feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0}
                loss_, accuracy_ = sess.run([loss, accuracy], feed_dict=feed_dict)
                total_accuracy += accuracy_
                total_loss += loss_

            te_losses.append((total_accuracy / step, total_loss / step))
            step, total_accuracy, total_loss = 0, 0, 0
            print(
                'After epoch {0} learning rate: {1} (Accuracy, Loss) - tr: ({2:.4f}, {3:.4f}) - te: ({4:.4f}, {5:.4f})'.format(
                    current_epoch,
                    learning_rate.eval(), tr_losses[-1][0], tr_losses[-1][1], te_losses[-1][0], te_losses[-1][1]))


# Load the model from the session
with sess.as_default():
    # testing wth a state
    x_state, y_state = gd.normalize("Square 3,Square 10,,Normal,Cross 1,Square 3,Triangle 4,Star 8")
    tr = gd.DataIterator(np.array([x_state]), np.array([y_state]))
    x_test, y_test = tr.next_batch(1)
    probability, predicted = sess.run([p_y_given_x, predictions], feed_dict={input_x: x_test, input_y: y_test, dropout_keep_prob: 1.})
    matches = top_matches(probability, 5)
    probs = sorted(probability[0], reverse=True)[0:5]
    print("")
    print("{} ({})".format(matches[0], probs[0]))
    print("{} ({})".format(matches[1], probs[1]))
    print("{} ({})".format(matches[2], probs[2]))
    print("{} ({})".format(matches[3], probs[3]))
    print("{} ({})".format(matches[4], probs[4]))


# Save the model and close the session
saver.save(sess, save_path)
sess.close()
