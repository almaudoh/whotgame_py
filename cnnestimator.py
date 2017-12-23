import tensorflow as tf
import math


class CnnEstimator():
    """
        Generic function approximation class using CNNs.
    """
    # Constants defining our neural network
    input_size = 100
    output_classes = 72
    # conv_layers_params = [[256, 5], [256, 1], [256, 1], [256, 1], ]
    # fully_connected_layers_params = [1024, 1024]

    def __init__(self):
        # Initialize variables
        self.global_step = 0
        self.summary = None

        # Input data layer
        with tf.name_scope('Input_Layer'):
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.input_size], name='inputs')
            self.action = tf.placeholder(tf.int64, shape=[None], name="action")
            self.target = tf.placeholder(tf.float32, shape=[None], name="target")
            self.train_inputs = tf.placeholder(tf.float32, shape=[None, self.output_classes], name="train_inputs")

        # Reshape and expand the dimension of inputs into convolution layer input dimensions.
        with tf.name_scope('Reshape_Input'):
            x = tf.reshape(self.inputs, [-1, 5, 20, 1])

        # Convolution layer
        with tf.name_scope('Convolution_Layer'):
            W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 256], 0., 0.2), dtype='float32', name='W')
            b1 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
            conv = tf.nn.conv2d(x, W1, [1, 1, 1, 1], 'VALID', name='Conv1')
            x = tf.nn.bias_add(conv, b1)

            W2 = tf.Variable(tf.truncated_normal([1, 5, 256, 256], 0., 0.2), dtype='float32', name='W')
            b2 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
            conv = tf.nn.conv2d(x, W2, [1, 1, 1, 1], 'VALID', name='Conv1')
            x = tf.nn.bias_add(conv, b2)

            W3 = tf.Variable(tf.truncated_normal([1, 5, 256, 256], 0., 0.2), dtype='float32', name='W')
            b3 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
            conv = tf.nn.conv2d(x, W3, [1, 1, 1, 1], 'VALID', name='Conv1')
            x = tf.nn.bias_add(conv, b3)

            W4 = tf.Variable(tf.truncated_normal([1, 5, 256, 256], 0., 0.2), dtype='float32', name='W')
            b4 = tf.Variable(tf.constant(1., 'float32', [256], 'b'))
            conv = tf.nn.conv2d(x, W4, [1, 1, 1, 1], 'VALID', name='Conv1')
            x = tf.nn.bias_add(conv, b4)

        # flatten conv output for fully connected layers
        with tf.name_scope('Reshape_Layer'):
            vec_dim = x.get_shape()[1].value * x.get_shape()[2].value * x.get_shape()[3].value
            x = tf.reshape(x, [-1, vec_dim])

        # fully connected layers input/output size
        fc_input_size = [vec_dim, 1024, 1024]

        # dropout keep probability
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # create fully connected + dropout layers
        for i, output_size in enumerate([1024, 1024]):
            with tf.name_scope('Linear_Layer'):
                stdv = 1 / math.sqrt(fc_input_size[i])
                W = tf.Variable(tf.truncated_normal(shape=[fc_input_size[i], output_size], mean=0, stddev=stdv),
                                dtype='float32', name='W')
                b = tf.Variable(tf.constant(shape=[output_size], value=0.1, dtype='float32'), dtype='float32', name='b')

                x = tf.nn.relu(tf.nn.xw_plus_b(x, W, b))

            with tf.name_scope('Dropout_Layer'):
                x = tf.nn.dropout(x, self.dropout_keep_prob)

        with tf.name_scope('Output_Layer'):
            stdv = 1 / math.sqrt(fc_input_size[-1])
            W = tf.Variable(tf.truncated_normal(shape=[fc_input_size[-1], self.output_classes], mean=0, stddev=stdv),
                            dtype='float32', name='W')
            b = tf.Variable(tf.constant(shape=[self.output_classes], value=0.1, dtype='float32'), dtype='float32', name='b')
            self.output = tf.nn.xw_plus_b(x, W, b, name='output')


class PolicyEstimator(CnnEstimator):
    """
        Policy function approximator.
    """

    def __init__(self, learning_rate=0.01, l2_regularization=0.1):
        with tf.variable_scope('policy_estimator'):
            CnnEstimator.__init__(self)
            with tf.name_scope('Output_Layer'):
                self.probabilities = tf.squeeze(tf.nn.softmax(tf.nn.l2_normalize(self.output, dim=1), dim=1))
                indices = tf.reshape(tf.range(0, tf.size(self.action, out_type=tf.int64)), shape=[-1, 1])
                actions = tf.reshape(self.action, [-1, 1])
                actions = tf.concat([indices, actions], axis=1)
                self.probability = tf.gather_nd(self.probabilities, actions)
                # self.probability = self.probabilities[:, self.action]

            with tf.name_scope('Loss_Layer'):
                # targets = tf.multiply(tf.one_hot(self.action, self.output_classes), tf.expand_dims(self.target, 1))
                # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.probabilities, labels=targets)
                # self.loss = tf.reduce_mean(losses)
                self.loss = tf.reduce_mean(tf.log(self.probability) * self.target)

                # Regularization
                loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
                self.loss += loss_l2 * l2_regularization

                # Loss for supervised training.
                self.loss_supervised = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.train_inputs))
                self.loss_supervised += loss_l2 * l2_regularization

            with tf.name_scope('Accuracy_Layer'):
                # Operation comparing prediction with true label
                correct_predictions = tf.equal(tf.argmax(self.probabilities, 1), self.action)
                self.learn_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

                # Accuracy for supervised training
                correct_predictions = tf.nn.in_top_k(self.output, tf.argmax(self.train_inputs, 1), 1)
                self.train_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            with tf.name_scope('Training_Layer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                self.learn_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
                self.train_op = self.optimizer.minimize(self.loss_supervised, global_step=tf.contrib.framework.get_global_step())

            with tf.name_scope('Summaries'):
                self.summaries = {
                    'loss': tf.summary.scalar('loss', self.loss),
                    'loss_supervised': tf.summary.scalar('loss_supervised', self.loss_supervised),
                    'probabilities': tf.summary.histogram('probabilities', self.probabilities),
                    'learn_accuracy': tf.summary.scalar('learn_accuracy', self.learn_accuracy),
                    'train_accuracy': tf.summary.scalar('train_accuracy', self.train_accuracy),
                }

    def learn_summaries(self):
        return tf.summary.merge([self.summaries[key] for key in ['probabilities', 'loss', 'learn_accuracy']])

    def train_summaries(self):
        return tf.summary.merge([self.summaries[key] for key in ['probabilities', 'loss_supervised', 'train_accuracy']])

    def predict(self, state):
        sess = tf.get_default_session()
        probs = sess.run(self.probabilities, feed_dict={self.inputs: state.normalize(), self.dropout_keep_prob: 1.})
        return probs

    def update(self, state, target, action):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: state.normalize(), self.action: action, self.target: target, self.dropout_keep_prob: 0.5}
        _, loss, self.summary, self.global_step = sess.run([self.learn_op, self.loss, self.learn_summaries(),
                                                            tf.contrib.framework.get_global_step()], feed_dict=feed_dict)
        return loss

    def learning_loss(self, state, action, target):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: state.normalize(), self.action: action, self.target: target, self.dropout_keep_prob: 1.}
        return sess.run(self.loss, feed_dict=feed_dict)

    def learning_accuracy(self, state, action):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: state.normalize(), self.action: action, self.dropout_keep_prob: 1.}
        return sess.run(self.learn_accuracy, feed_dict=feed_dict)

    def train(self, inputs, labels):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: inputs, self.train_inputs: labels, self.dropout_keep_prob: 0.5}
        _, loss, self.summary, self.global_step = sess.run([self.train_op, self.loss_supervised, self.train_summaries(),
                                                            tf.contrib.framework.get_global_step()], feed_dict=feed_dict)
        return loss

    def training_loss(self, inputs, labels):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: inputs, self.train_inputs: labels, self.dropout_keep_prob: 1.}
        return sess.run(self.loss_supervised, feed_dict=feed_dict)

    def training_accuracy(self, inputs, labels):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: inputs, self.train_inputs: labels, self.dropout_keep_prob: 1.}
        return sess.run(self.train_accuracy, feed_dict=feed_dict)

    def write_summaries(self, writer):
        writer.add_summary(self.summary, global_step=self.global_step)


class ValueEstimator(CnnEstimator):
    """
        Value function approximator.
    """

    def __init__(self, learning_rate=0.01):
        with tf.variable_scope('value_estimator'):
            CnnEstimator.__init__(self)
            with tf.name_scope('Output_Layer'):
                W = tf.Variable(tf.truncated_normal(shape=[self.output_classes, 1], mean=0, stddev=0.5), dtype='float32', name='W')
                b = tf.Variable(tf.constant(shape=[1], value=0.1, dtype='float32'), dtype='float32', name='b')
                self.value_estimate = tf.squeeze(tf.nn.xw_plus_b(tf.sigmoid(self.output), W, b), name='value_estimate')

            with tf.name_scope('Loss_Layer'):
                self.loss = tf.squared_difference(self.value_estimate, self.target)

            with tf.name_scope('Training_Layer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state):
        sess = tf.get_default_session()
        return sess.run(self.value_estimate, {self.inputs: state.normalize(), self.dropout_keep_prob: 1.})

    def update(self, state, target):
        sess = tf.get_default_session()
        feed_dict = {self.inputs: state.normalize(), self.target: target, self.dropout_keep_prob: 0.5}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
